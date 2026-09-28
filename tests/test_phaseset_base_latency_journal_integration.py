"""Server-only integration contracts for latency observation journaling."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from phaseset_core import base_cohort_latency as latency
from phaseset_core import base_cohort_validation as scoring
from phaseset_core import latency_progress_journal as journal
from phaseset_core import training
from phaseset_core.experiments import base_run_ids


pytestmark = pytest.mark.skipif(os.name != "posix", reason="journal is POSIX-only")
UUID = "GPU-11111111-2222-3333-4444-555555555555"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _snapshot(ordinal: int, *, free_mib: int = 12_000) -> latency.SharedCudaObservation:
    return latency.SharedCudaObservation(
        captured_unix_ns=1_800_000_000_000_000_000 + ordinal,
        device_uuid=UUID,
        memory_total_mib=49_140,
        memory_free_mib=free_mib,
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
    scored = SimpleNamespace(observations=tuple(scores), sha256=_sha("scored-cohort"))
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
    def __init__(self, *, fail_encode_at: int | None = None, free_mib: int = 12_000) -> None:
        self.snapshot_count = 0
        self.clock = 10_000
        self.encode_count = 0
        self.timer_open = False
        self.fail_encode_at = fail_encode_at
        self.free_mib = free_mib

    def snapshot(self):
        result = _snapshot(self.snapshot_count, free_mib=self.free_mib)
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
        self.timer_open = not self.timer_open
        self.clock += 100
        return self.clock

    def reset_peak_memory(self, _device) -> None:
        return None

    def peak_memory(self, _device):
        return 256 * 1024**2, 512 * 1024**2

    def place(self, _system, _device) -> None:
        return None

    def encode(self, _system, _groups):
        self.encode_count += 1
        if self.fail_encode_at == self.encode_count:
            raise RuntimeError("injected visit failure")
        return object()

    def validate_output(self, output, _device) -> None:
        assert type(output) is object

    def release(self, _system, _device) -> None:
        return None


class _ObservedWriter:
    def __init__(self, writer, runtime: _FakeRuntime) -> None:
        self.writer = writer
        self.runtime = runtime
        self.calls: list[str] = []

    def _call(self, name: str, **payload):
        if name == "timed_sample_completed":
            assert self.runtime.timer_open is False
        self.calls.append(name)
        return getattr(self.writer, name)(**payload)

    def session_start(self, **payload):
        return self._call("session_start", **payload)

    def context_ready(self, **payload):
        return self._call("context_ready", **payload)

    def visit_started(self, **payload):
        return self._call("visit_started", **payload)

    def warmup_completed(self, **payload):
        return self._call("warmup_completed", **payload)

    def timed_sample_completed(self, **payload):
        return self._call("timed_sample_completed", **payload)

    def visit_ended(self, **payload):
        return self._call("visit_ended", **payload)

    def terminal(self, **payload):
        return self._call("terminal", **payload)


class _FailAfterFirstSample(_ObservedWriter):
    def timed_sample_completed(self, **payload):
        super().timed_sample_completed(**payload)
        raise journal.LatencyJournalWriteError("injected post-commit write failure")


class _FailAfterCompletedTerminal(_ObservedWriter):
    def __init__(self, writer, runtime: _FakeRuntime) -> None:
        super().__init__(writer, runtime)
        self.committed_session_sha256: str | None = None

    def terminal(self, **payload):
        super().terminal(**payload)
        if payload["outcome"] == "COMPLETED":
            self.committed_session_sha256 = payload["latency_session_sha256"]
            raise journal.LatencyJournalWriteError("injected error after completed terminal commit")


def _patch_session(monkeypatch: pytest.MonkeyPatch):
    cohort, admission, scored = _analytic_inputs()
    monkeypatch.setattr(latency, "_checked_inputs", lambda *_args: (cohort, admission, scored))
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


def _bindings(cohort, scored) -> journal.LatencyJournalBindings:
    return journal.LatencyJournalBindings(
        resolved_cohort_sha256=latency._resolved_cohort_sha256(cohort),
        scored_cohort_sha256=scored.sha256,
        source_tree_sha256=cohort.source_tree_sha256,
        latency_protocol_sha256=_sha("frozen-latency-protocol"),
        wrapper_sha256=_sha("outer-timeout-wrapper"),
        registered_cuda_uuid=UUID,
    )


def _payload(record) -> dict[str, object]:
    return json.loads(record.raw.decode("ascii"))["payload"]


def test_fake_entry_writes_exact_complete_prefix_after_every_timer_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cohort, admission, scored = _patch_session(monkeypatch)
    runtime = _FakeRuntime()
    bindings = _bindings(cohort, scored)
    with journal.create_latency_progress_journal(
        tmp_path,
        session_id="complete-session",
        bindings=bindings,
    ) as writer:
        observer = _ObservedWriter(writer, runtime)
        result = latency.run_base_cohort_latency_session(
            cohort,
            admission=admission,
            scored_cohort=scored,
            registered_cuda_uuid=UUID,
            observer=observer,
            _runtime=runtime,
        )
    read = journal.read_latency_progress_journal(
        tmp_path / "complete-session",
        expected_bindings=bindings,
    )

    assert read.status == journal.COMPLETED_STATUS
    assert read.formal_latency is False
    assert len(read.records) == 1461
    expected_events = ["SESSION_START", "CONTEXT_READY"]
    for _round_rows in latency.cyclic_visit_schedule():
        for _run_id in _round_rows:
            expected_events.extend(
                ["VISIT_START"]
                + ["WARMUP_COMPLETED"] * latency.WARMUPS_PER_VISIT
                + ["TIMED_SAMPLE_COMPLETED"] * latency.TIMED_PER_VISIT
                + ["VISIT_END"]
            )
    expected_events.append("TERMINAL")
    assert tuple(record.event_type for record in read.records) == tuple(expected_events)
    callback_name = {
        "SESSION_START": "session_start",
        "CONTEXT_READY": "context_ready",
        "VISIT_START": "visit_started",
        "WARMUP_COMPLETED": "warmup_completed",
        "TIMED_SAMPLE_COMPLETED": "timed_sample_completed",
        "VISIT_END": "visit_ended",
        "TERMINAL": "terminal",
    }
    assert tuple(observer.calls) == tuple(callback_name[name] for name in expected_events)
    assert _payload(read.records[-1]) == {
        "failure_code": None,
        "latency_session_sha256": result.sha256,
        "outcome": "COMPLETED",
    }
    assert observer.calls.count("timed_sample_completed") == 81 * 11
    assert runtime.timer_open is False
    assert result.actual_runtime is False
    assert result.formal_samples_complete is False


def test_handled_visit_hold_writes_matching_visit_and_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cohort, admission, scored = _patch_session(monkeypatch)
    runtime = _FakeRuntime(fail_encode_at=3)
    bindings = _bindings(cohort, scored)
    with journal.create_latency_progress_journal(
        tmp_path,
        session_id="held-visit",
        bindings=bindings,
    ) as writer:
        observer = _ObservedWriter(writer, runtime)
        with pytest.raises(latency.BaseCohortLatencyHold) as caught:
            latency.run_base_cohort_latency_session(
                cohort,
                admission=admission,
                scored_cohort=scored,
                registered_cuda_uuid=UUID,
                observer=observer,
                _runtime=runtime,
            )
    read = journal.read_latency_progress_journal(
        tmp_path / "held-visit",
        expected_bindings=bindings,
    )

    assert read.status == journal.HELD_STATUS
    assert tuple(record.event_type for record in read.records[-2:]) == (
        "VISIT_END",
        "TERMINAL",
    )
    visit_payload, terminal_payload = map(_payload, read.records[-2:])
    assert visit_payload["failure_code"] == terminal_payload["failure_code"]
    assert terminal_payload["failure_code"] == caught.value.failure_code
    assert terminal_payload["latency_session_sha256"] == caught.value.receipt.sha256
    assert caught.value.receipt.visits[0].warmups_completed == 2


def test_admission_hold_has_session_start_and_bound_terminal_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cohort, admission, scored = _patch_session(monkeypatch)
    runtime = _FakeRuntime(free_mib=8191)
    bindings = _bindings(cohort, scored)
    with journal.create_latency_progress_journal(
        tmp_path,
        session_id="admission-hold",
        bindings=bindings,
    ) as writer:
        observer = _ObservedWriter(writer, runtime)
        with pytest.raises(latency.BaseCohortLatencyHold) as caught:
            latency.run_base_cohort_latency_session(
                cohort,
                admission=admission,
                scored_cohort=scored,
                registered_cuda_uuid=UUID,
                observer=observer,
                _runtime=runtime,
            )
    read = journal.read_latency_progress_journal(
        tmp_path / "admission-hold",
        expected_bindings=bindings,
    )

    assert tuple(record.event_type for record in read.records) == (
        "SESSION_START",
        "TERMINAL",
    )
    assert read.status == journal.HELD_STATUS
    assert _payload(read.records[-1])["latency_session_sha256"] == caught.value.receipt.sha256
    assert caught.value.receipt.visits == ()


def test_observer_failure_holds_session_without_fabricating_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cohort, admission, scored = _patch_session(monkeypatch)
    runtime = _FakeRuntime()
    bindings = _bindings(cohort, scored)
    with journal.create_latency_progress_journal(
        tmp_path,
        session_id="observer-failure",
        bindings=bindings,
    ) as writer:
        observer = _FailAfterFirstSample(writer, runtime)
        with pytest.raises(latency.BaseCohortLatencyHold) as caught:
            latency.run_base_cohort_latency_session(
                cohort,
                admission=admission,
                scored_cohort=scored,
                registered_cuda_uuid=UUID,
                observer=observer,
                _runtime=runtime,
            )
    read = journal.read_latency_progress_journal(
        tmp_path / "observer-failure",
        expected_bindings=bindings,
    )

    assert read.status == journal.INCOMPLETE_STATUS
    assert read.records[-1].event_type == "TIMED_SAMPLE_COMPLETED"
    assert all(record.event_type not in {"VISIT_END", "TERMINAL"} for record in read.records)
    assert caught.value.receipt.complete is False
    assert caught.value.receipt.rows == ()
    assert caught.value.receipt.visits[0].samples_ns == (100,)


def test_postcommit_completed_terminal_stays_nonformal_when_api_holds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cohort, admission, scored = _patch_session(monkeypatch)
    runtime = _FakeRuntime()
    bindings = _bindings(cohort, scored)
    with journal.create_latency_progress_journal(
        tmp_path,
        session_id="postcommit-completed-terminal",
        bindings=bindings,
    ) as writer:
        observer = _FailAfterCompletedTerminal(writer, runtime)
        with pytest.raises(latency.BaseCohortLatencyHold) as caught:
            latency.run_base_cohort_latency_session(
                cohort,
                admission=admission,
                scored_cohort=scored,
                registered_cuda_uuid=UUID,
                observer=observer,
                _runtime=runtime,
            )
    read = journal.read_latency_progress_journal(
        tmp_path / "postcommit-completed-terminal",
        expected_bindings=bindings,
    )

    terminal_payload = _payload(read.records[-1])
    assert read.status == journal.COMPLETED_STATUS
    assert read.formal_latency is False
    assert terminal_payload["outcome"] == "COMPLETED"
    assert terminal_payload["latency_session_sha256"] == observer.committed_session_sha256
    assert terminal_payload["latency_session_sha256"] != caught.value.receipt.sha256
    assert caught.value.receipt.complete is False
    assert caught.value.receipt.rows == ()
    assert len(caught.value.receipt.visits) == 81
