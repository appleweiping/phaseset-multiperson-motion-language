"""Server-only software contracts for the frozen base latency session."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from phaseset_core import base_cohort_latency as latency
from phaseset_core import base_cohort_validation as scoring
from phaseset_core import training
from phaseset_core.experiments import base_run_ids


UUID = "GPU-11111111-2222-3333-4444-555555555555"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _snapshot(ordinal: int, *, uuid: str = UUID) -> latency.SharedCudaObservation:
    return latency.SharedCudaObservation(
        captured_unix_ns=1_800_000_000_000_000_000 + ordinal,
        device_uuid=uuid,
        memory_total_mib=49_140,
        memory_free_mib=12_000,
        memory_used_mib=36_000,
        gpu_utilization_percent=(ordinal * 17) % 101,
        memory_utilization_percent=(ordinal * 11) % 101,
        compute_mode="Default",
        driver_version="580.95.05",
    )


def _parameter_census(system_id: str) -> scoring.BaseParameterCensus:
    return scoring.BaseParameterCensus(
        system_id=system_id,
        rows=(scoring.BaseParameterRow("weight", (1,), "torch.float32", 1),),
        total=1,
        loaded_state_sha256=_sha(f"loaded/{system_id}"),
    )


def _analytic_inputs():
    rows = []
    scores = []
    for index, run_id in enumerate(base_run_ids()):
        _prefix, _role, seed_text, system_id = run_id.split("/")
        raw = f"selected-{index}".encode("ascii")
        selected = SimpleNamespace(
            name="checkpoint-000000000001-000001-validation.pt",
            sha256=hashlib.sha256(raw).hexdigest(),
            raw=raw,
            epoch=1,
            global_step=1,
            state_digest=_sha(f"state/{index}"),
        )
        rows.append(
            SimpleNamespace(
                run_id=run_id,
                attempt_id=f"attempt-{index}",
                system_id=system_id,
                seed=int(seed_text),
                terminal_sha256=_sha(f"terminal/{index}"),
                latest_checkpoint_sha256=_sha(f"latest/{index}"),
                latest_checkpoint_name="checkpoint-000000000030-000030-validation.pt",
                selected_checkpoint=selected,
                failed_predecessors=(),
            )
        )
        scores.append(
            SimpleNamespace(
                run_id=run_id,
                attempt_id=f"attempt-{index}",
                system_id=system_id,
                seed=int(seed_text),
                terminal_sha256=_sha(f"terminal/{index}"),
                selected_checkpoint_sha256=selected.sha256,
                selected_checkpoint_state_digest=selected.state_digest,
                environment_sha256=_sha("environment"),
                parameter_census=_parameter_census(system_id),
            )
        )
    cohort = SimpleNamespace(
        rows=tuple(rows),
        plan_sha256=_sha("plan"),
        matrix_sha256=_sha("matrix"),
        training_config_sha256=_sha("host-config"),
        source_tree_sha256=_sha("source-tree"),
        train_manifest_sha256=_sha("train"),
        val_manifest_sha256=_sha("val"),
    )
    scored = SimpleNamespace(
        observations=tuple(scores),
        sha256=_sha("scored-cohort"),
    )
    admission = SimpleNamespace(
        device="cuda",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32_768,
        checkpoint_every_updates=1,
    )
    return cohort, admission, scored


class _FakeSystem(nn.Module):
    def __init__(self, system_id: str) -> None:
        super().__init__()
        self.system_id = system_id
        self.weight = nn.Parameter(torch.ones((1,), dtype=torch.float32))


class _FakeRuntime:
    def __init__(self, *, fail_encode_at: int | None = None) -> None:
        self.snapshot_count = 0
        self.clock = 10_000
        self.encode_count = 0
        self.place_count = 0
        self.release_count = 0
        self.fail_encode_at = fail_encode_at

    def snapshot(self) -> latency.SharedCudaObservation:
        result = _snapshot(self.snapshot_count)
        self.snapshot_count += 1
        return result

    def initialize(self, *, device, precision_mode, admission_observation):
        assert str(device) == "cuda"
        assert admission_observation.device_uuid == UUID
        return latency.BaseLatencyRuntimeIdentity(
            device="cuda:0",
            device_uuid=UUID,
            device_name="registered-fixture-device",
            torch_version="2.12.0+cu126",
            cuda_runtime_version="12.6",
            cudnn_version=9_000,
            driver_version="580.95.05",
            nvidia_total_memory_mib=49_140,
            compute_mode="Default",
            torch_total_memory_bytes=48_453 * 1024**2,
            initial_allocated_bytes=0,
            initial_reserved_bytes=0,
            environment_sha256=_sha("environment"),
            precision_mode=precision_mode,
        )

    def synchronize(self, _device) -> None:
        return None

    def perf_counter_ns(self) -> int:
        self.clock += 100
        return self.clock

    def reset_peak_memory(self, _device) -> None:
        return None

    def peak_memory(self, _device) -> tuple[int, int]:
        return 256 * 1024**2, 512 * 1024**2

    def place(self, _system, _device) -> None:
        self.place_count += 1

    def encode(self, _system, groups):
        self.encode_count += 1
        assert groups.skeletons.shape == (1, 32, 200, 22, 3)
        if self.fail_encode_at == self.encode_count:
            raise RuntimeError("injected visit failure")
        return object()

    def validate_output(self, output, _device) -> None:
        assert type(output) is object

    def release(self, _system, _device) -> None:
        self.release_count += 1


def _patch_analytic_session(monkeypatch: pytest.MonkeyPatch):
    cohort, admission, scored = _analytic_inputs()
    monkeypatch.setattr(
        latency,
        "_checked_inputs",
        lambda *_args: (cohort, admission, scored),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def load(row, score, _admission, *, runtime_device):
        assert runtime_device == torch.device("cuda:0")
        config = training.TrainingConfig(
            stage="base",
            seed=row.seed,
            device="cuda",
            synthetic_contract=False,
        )
        return (
            _FakeSystem(row.system_id),
            config,
            training.PrecisionDecision("FP32", False, "FP32_REQUESTED"),
            score.parameter_census,
        )

    monkeypatch.setattr(latency, "_load_selected_system", load)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return cohort, admission, scored


def test_unindexed_cuda_device_matches_scorer_and_resolved_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    latency._checked_scorer_runtime_device(
        config_device="cuda",
        score_device="cuda:0",
        runtime_device=torch.device("cuda:0"),
    )

    with pytest.raises(latency.BaseCohortLatencyError, match="resolved physical"):
        latency._checked_scorer_runtime_device(
            config_device="cuda",
            score_device="cuda",
            runtime_device=torch.device("cuda:0"),
        )
    with pytest.raises(latency.BaseCohortLatencyError, match="resolved physical"):
        latency._checked_scorer_runtime_device(
            config_device="cuda",
            score_device="cuda:0",
            runtime_device=torch.device("cuda:1"),
        )


def test_cyclic_schedule_balances_every_row_and_ordinal() -> None:
    schedule = latency.cyclic_visit_schedule()
    rows = base_run_ids()

    assert len(schedule) == 9
    assert all(len(round_rows) == 9 and set(round_rows) == set(rows) for round_rows in schedule)
    for row in rows:
        assert sorted(round_rows.index(row) for round_rows in schedule) == list(range(9))
    assert schedule[0] == rows
    assert schedule[1] == rows[1:] + rows[:1]


def test_frozen_workload_uses_literal_formula_and_complete_masks() -> None:
    groups, receipt = latency.build_frozen_latency_workload()

    assert groups.skeletons.shape == (1, 32, 200, 22, 3)
    assert groups.skeletons.dtype == np.float32
    assert groups.actor_mask.all() and groups.frame_mask.all() and groups.track_mask.all()
    source_commitments = latency._source_actor_commitments()
    for actor, frame, joint, coordinate in ((0, 0, 0, 0), (31, 199, 21, 2), (13, 79, 8, 1)):
        raw = ((actor * 131 + frame * 17 + joint * 7 + coordinate * 3) % 2048) - 1024
        expected = np.float32(raw) * np.float32(2**-10)
        prepared_actor = receipt.actor_commitments.index(source_commitments[actor])
        assert groups.skeletons[0, prepared_actor, frame, joint, coordinate] == expected
    expected_actor0 = hashlib.sha256(latency._ACTOR_DOMAIN + (0).to_bytes(4, "big")).digest()
    assert expected_actor0 in receipt.actor_commitments
    assert tuple((name, shape, dtype) for name, _digest, shape, dtype in receipt.array_rows) == (
        ("skeletons", (1, 32, 200, 22, 3), "<f4"),
        ("actor_mask", (1, 32), "|b1"),
        ("frame_mask", (1, 200), "|b1"),
        ("track_mask", (1, 32, 200, 22), "|b1"),
    )
    assert receipt == latency._workload_receipt(groups)
    bad_rows = list(receipt.array_rows)
    name, digest, _shape, dtype = bad_rows[0]
    bad_rows[0] = (name, digest, (1, 31, 200, 22, 3), dtype)
    with pytest.raises(latency.BaseCohortLatencyError, match="array census"):
        replace(receipt, array_rows=tuple(bad_rows))


def test_row_statistics_are_exact_integer_median_mad_and_nearest_rank() -> None:
    samples = tuple(range(1, 100))
    row = latency.BaseLatencyRowObservation(
        run_id=base_run_ids()[0],
        system_id="B0",
        seed=1729,
        selected_checkpoint_sha256=_sha("selected"),
        selected_checkpoint_state_digest=_sha("state"),
        parameter_census_sha256=_sha("parameters"),
        parameter_count=123,
        runtime_sha256=_sha("runtime"),
        workload_sha256=_sha("workload"),
        samples_ns=samples,
        median_ns=50,
        median_absolute_deviation_ns=25,
        minimum_ns=1,
        maximum_ns=99,
        p05_nearest_rank_ns=5,
        p95_nearest_rank_ns=95,
    )

    assert row.frozen_runtime_latency_ns == 50
    with pytest.raises(latency.BaseCohortLatencyError, match="statistics"):
        replace(row, p95_nearest_rank_ns=94)


def test_fake_runtime_executes_exact_visits_but_cannot_emit_formal_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, admission, scored = _patch_analytic_session(monkeypatch)
    runtime = _FakeRuntime()

    result = latency.run_base_cohort_latency_session(
        cohort,
        admission=admission,
        scored_cohort=scored,
        registered_cuda_uuid=UUID,
        _runtime=runtime,
    )

    assert result.complete is True
    assert result.actual_runtime is False
    assert result.formal_samples_complete is False
    assert result.status == latency.FAKE_COMPLETE_STATUS
    assert len(result.visits) == 81
    assert all(visit.warmups_completed == 5 for visit in result.visits)
    assert all(len(visit.samples_ns) == 11 for visit in result.visits)
    assert tuple(row.run_id for row in result.rows) == base_run_ids()
    assert all(len(row.samples_ns) == 99 and row.median_ns == 100 for row in result.rows)
    assert runtime.encode_count == 81 * (5 + 11)
    assert runtime.place_count == runtime.release_count == 81
    with pytest.raises(latency.BaseCohortLatencyError, match="status"):
        replace(result, actual_runtime=True, formal_samples_complete=True)


def test_failed_visit_retains_completed_warmups_and_timed_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, admission, scored = _patch_analytic_session(monkeypatch)
    runtime = _FakeRuntime(fail_encode_at=9)

    with pytest.raises(latency.BaseCohortLatencyHold) as caught:
        latency.run_base_cohort_latency_session(
            cohort,
            admission=admission,
            scored_cohort=scored,
            registered_cuda_uuid=UUID,
            _runtime=runtime,
        )

    receipt = caught.value.receipt
    assert receipt.complete is False
    assert receipt.rows == ()
    assert receipt.formal_samples_complete is False
    assert len(receipt.visits) == 1
    failed = receipt.visits[0]
    assert failed.completed is False
    assert failed.warmups_completed == 5
    assert failed.samples_ns == (100, 100, 100)
    assert failed.peak_allocated_bytes == 256 * 1024**2
    assert runtime.release_count == 1


def test_runtime_drift_after_visit_is_a_whole_session_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, admission, scored = _patch_analytic_session(monkeypatch)
    runtime = _FakeRuntime()
    original_snapshot = runtime.snapshot

    def drifting_snapshot():
        observed = original_snapshot()
        if runtime.snapshot_count == 4:
            return replace(observed, device_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        return observed

    runtime.snapshot = drifting_snapshot
    with pytest.raises(latency.BaseCohortLatencyHold) as caught:
        latency.run_base_cohort_latency_session(
            cohort,
            admission=admission,
            scored_cohort=scored,
            registered_cuda_uuid=UUID,
            _runtime=runtime,
        )
    receipt = caught.value.receipt
    assert receipt.failure_code == "RUNTIME_DRIFT"
    assert len(receipt.visits) == 1
    assert receipt.visits[0].after is not None
    assert receipt.visits[0].after.device_uuid != UUID


@pytest.mark.parametrize(
    ("changed", "failure_code"),
    (
        ({"memory_free_mib": 8191}, "RESOURCE_LIMIT"),
        (
            {"device_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
            "RUNTIME_DRIFT",
        ),
    ),
)
def test_admission_failure_retains_zero_visit_observation_as_whole_session_hold(
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
    failure_code: str,
) -> None:
    cohort, admission, scored = _patch_analytic_session(monkeypatch)
    runtime = _FakeRuntime()
    runtime.snapshot = lambda: replace(_snapshot(0), **changed)

    with pytest.raises(latency.BaseCohortLatencyHold) as caught:
        latency.run_base_cohort_latency_session(
            cohort,
            admission=admission,
            scored_cohort=scored,
            registered_cuda_uuid=UUID,
            _runtime=runtime,
        )

    receipt = caught.value.receipt
    assert receipt.failure_code == failure_code
    assert receipt.complete is False
    assert receipt.rows == ()
    assert receipt.visits == ()
    assert receipt.admission_observation == replace(_snapshot(0), **changed)
    assert receipt.post_context_observation is None
    assert receipt.runtime_identity is None


def test_public_entry_has_no_submitted_latency_or_sample_parameter() -> None:
    names = tuple(latency.run_base_cohort_latency_session.__annotations__)
    assert "latency" not in names
    assert "samples" not in names
    assert "parameter_count" not in names
