from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
from fractions import Fraction
import hashlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phaseset_core import base_qualification_host as controller
from phaseset_core import base_cohort_qualification as assembly
from phaseset_core import cli
from phaseset_core import experiments
from phaseset_core import host
from phaseset_core import latency_progress_journal as journal
from phaseset_core.experiments import base_run_ids
from phaseset_core.production import BackendExecution, PrivateReceiptAssertions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CONFIG = PROJECT_ROOT / "configs" / "phaseset" / "training.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CUDA_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _registry_raw(*, reverse: bool = False) -> bytes:
    run_ids = list(base_run_ids())
    if reverse:
        run_ids.reverse()
    return _canonical(
        {
            "authority": 0,
            "production": False,
            "result_claimed": False,
            "rows": [
                {
                    "attempt_id": f"actual-attempt-{index:02d}",
                    "run_id": run_id,
                    "terminal_outcome": "SUCCEEDED",
                    "terminal_sha256": hashlib.sha256(run_id.encode("ascii")).hexdigest(),
                }
                for index, run_id in enumerate(run_ids)
            ],
            "schema": controller.REGISTRY_SCHEMA,
        }
    )


def _controller_config(tmp_path: Path, registry: Path) -> controller.HostBaseQualificationConfig:
    output_root = tmp_path / "output"
    journal_parent = tmp_path / "journal"
    output_root.mkdir()
    journal_parent.mkdir()
    return controller.HostBaseQualificationConfig(
        registry_path=registry,
        registry_sha256=_digest(registry.read_bytes()),
        output_root=output_root,
        journal_parent=journal_parent,
        session_id="session-001",
        registered_cuda_uuid=CUDA_UUID,
        latency_protocol_sha256=SHA_A,
        launcher_sha256=SHA_B,
        wall_timeout_seconds=3600,
    )


def _admission():
    return controller.resolver_module.BaseCohortAdmission(
        plan_sha256=SHA_A,
        matrix_sha256=SHA_B,
        training_config_sha256=SHA_C,
        source_tree_sha256=_digest(b"source"),
        train_manifest_sha256=_digest(b"train"),
        val_manifest_sha256=_digest(b"val"),
        device="cuda",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32,
        checkpoint_every_updates=1,
    )


def _cuda_observation(captured_unix_ns: int = 1):
    return controller.latency_module.SharedCudaObservation(
        captured_unix_ns=captured_unix_ns,
        device_uuid=CUDA_UUID,
        memory_total_mib=49_000,
        memory_free_mib=40_000,
        memory_used_mib=100,
        gpu_utilization_percent=0,
        memory_utilization_percent=0,
        compute_mode="Default",
        driver_version="test-driver",
    )


def _cuda_identity():
    return controller.latency_module.BaseLatencyRuntimeIdentity(
        device="cuda:0",
        device_uuid=CUDA_UUID,
        device_name="fictional test GPU",
        torch_version="test-torch",
        cuda_runtime_version="test-cuda",
        cudnn_version=1,
        driver_version="test-driver",
        nvidia_total_memory_mib=49_000,
        compute_mode="Default",
        torch_total_memory_bytes=49_000 * 1024 * 1024,
        initial_allocated_bytes=0,
        initial_reserved_bytes=0,
        environment_sha256=_digest(b"environment"),
        precision_mode="FP32",
    )


class _FakeCaptureValidationSource:
    split = "val"
    census_sha256 = _digest(b"val")
    manifest_sha256 = _digest(b"capture-source")


def _emit_completed_journal(observer, session_sha256: str) -> None:
    observer.session_start(
        admission_observation_sha256=_digest(b"admission observation"),
        started_unix_ns=1,
    )
    observer.context_ready(
        runtime_sha256=_digest(b"runtime"),
        post_context_observation_sha256=_digest(b"post context"),
    )
    for round_index, round_rows in enumerate(
        controller.latency_module.cyclic_visit_schedule()
    ):
        for ordinal, run_id in enumerate(round_rows):
            event = {
                "round_index": round_index,
                "ordinal_in_round": ordinal,
                "run_id": run_id,
            }
            observer.visit_started(
                **event,
                before_observation_sha256=_digest(
                    f"before/{round_index}/{ordinal}".encode("ascii")
                ),
            )
            for warmup_index in range(5):
                observer.warmup_completed(**event, warmup_index=warmup_index)
            for sample_index in range(11):
                observer.timed_sample_completed(
                    **event,
                    sample_index=sample_index,
                    sample_ns=sample_index + 1,
                )
            observer.visit_ended(
                **event,
                outcome="COMPLETED",
                after_observation_sha256=_digest(
                    f"after/{round_index}/{ordinal}".encode("ascii")
                ),
                peak_allocated_bytes=1,
                peak_reserved_bytes=2,
                failure_code=None,
            )
    observer.terminal(
        outcome="COMPLETED",
        latency_session_sha256=session_sha256,
        failure_code=None,
    )


def _patch_pre_latency(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object]:
    cohort = SimpleNamespace(source_tree_sha256=_digest(b"source"))
    scored = SimpleNamespace(sha256=_digest(b"scored"))
    process_thread = object()
    signal_facade = SimpleNamespace(
        SIGALRM=14,
        ITIMER_REAL=0,
        getitimer=lambda _which: (0.0, 0.0),
        getsignal=lambda _signal_number: object(),
        setitimer=lambda _which, _seconds, _interval: None,
        signal=lambda _signal_number, _handler: None,
    )
    threading_facade = SimpleNamespace(
        current_thread=lambda: process_thread,
        main_thread=lambda: process_thread,
    )
    real_timer_preflight = controller._validate_formal_latency_timer_support
    real_timer_timeout = controller._formal_latency_timeout
    journal_writers: dict[Path, object] = {}

    class PortableJournalWriter:
        """Exercise the real journal schema/state machine without POSIX dir-fd I/O."""

        _EVENT_METHODS = {
            "session_start": "SESSION_START",
            "context_ready": "CONTEXT_READY",
            "visit_started": "VISIT_START",
            "warmup_completed": "WARMUP_COMPLETED",
            "timed_sample_completed": "TIMED_SAMPLE_COMPLETED",
            "visit_ended": "VISIT_END",
            "terminal": "TERMINAL",
        }

        def __init__(
            self,
            path: Path,
            session_id: str,
            bindings: journal.LatencyJournalBindings,
        ) -> None:
            self.path = path
            self.session_id = session_id
            self.bindings = bindings
            self.records: list[journal.LatencyJournalRecord] = []
            self.state = journal._JournalState()
            self.previous_sha256 = journal.ZERO_SHA256
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, _type, _value, _traceback) -> None:
            self.closed = True

        def __getattr__(self, name: str):
            try:
                event_type = self._EVENT_METHODS[name]
            except KeyError as error:
                raise AttributeError(name) from error

            def append(**payload: object) -> journal.LatencyJournalRecord:
                if self.closed or len(self.records) >= journal.MAX_EVENTS:
                    raise journal.LatencyJournalWriteError(
                        "portable structural journal is closed or exhausted"
                    )
                if event_type == "SESSION_START":
                    # The real writer injects its own admitted bindings here;
                    # session_start callers do not provide this field.
                    payload["bindings"] = self.bindings.payload()
                sequence = len(self.records)
                next_state = self.state.transition(event_type, payload)
                raw = journal._canonical_json_bytes(
                    journal._event_payload(
                        sequence=sequence,
                        session_id=self.session_id,
                        event_type=event_type,
                        previous_event_sha256=self.previous_sha256,
                        binding_sha256=self.bindings.sha256,
                        recorded_unix_ns=sequence + 1,
                        payload=payload,
                    )
                )
                event_sha256 = _digest(raw)
                record = journal.LatencyJournalRecord(
                    sequence=sequence,
                    event_type=event_type,
                    previous_event_sha256=self.previous_sha256,
                    binding_sha256=self.bindings.sha256,
                    raw=raw,
                    marker_raw=journal._completion_marker_raw(sequence, event_sha256),
                )
                self.records.append(record)
                self.state = next_state
                self.previous_sha256 = event_sha256
                return record

            return append

    def portable_create_journal(
        parent: Path,
        *,
        session_id: str,
        bindings: journal.LatencyJournalBindings,
        **_ignored: object,
    ) -> PortableJournalWriter:
        path = parent / session_id
        path.mkdir(mode=0o700, exist_ok=False)
        writer = PortableJournalWriter(path, session_id, bindings)
        journal_writers[path.resolve()] = writer
        return writer

    def portable_read_journal(
        path: Path,
        *,
        expected_bindings: journal.LatencyJournalBindings,
    ) -> journal.LatencyJournalReadResult:
        writer = journal_writers.get(path.resolve())
        if writer is None or writer.bindings != expected_bindings:
            raise journal.LatencyJournalError(
                "portable structural journal binding is unavailable"
            )
        return journal._read_result(
            session_id=writer.session_id,
            bindings=writer.bindings,
            records=writer.records,
            tail=(),
            terminal_outcome=writer.state.terminal_outcome,
        )

    def portable_timer_preflight(**_ignored: object) -> None:
        real_timer_preflight(
            _os_name="posix",
            _signal_api=signal_facade,
            _threading_api=threading_facade,
        )

    @contextmanager
    def portable_timer_timeout(seconds: int):
        with real_timer_timeout(
            seconds,
            _os_name="posix",
            _signal_api=signal_facade,
            _threading_api=threading_facade,
        ):
            yield

    monkeypatch.setattr(controller, "CaptureValidationSource", _FakeCaptureValidationSource)
    monkeypatch.setattr(
        controller.resolver_module,
        "resolve_completed_base_cohort",
        lambda *_args, **_kwargs: cohort,
    )
    monkeypatch.setattr(
        controller.scoring_module,
        "score_resolved_base_cohort",
        lambda *_args, **_kwargs: scored,
    )
    monkeypatch.setattr(
        controller.latency_module,
        "_resolved_cohort_sha256",
        lambda value: _digest(b"resolved") if value is cohort else pytest.fail("cohort drift"),
    )
    monkeypatch.setattr(
        controller,
        "_validate_formal_latency_timer_support",
        portable_timer_preflight,
    )
    monkeypatch.setattr(controller, "_formal_latency_timeout", portable_timer_timeout)
    monkeypatch.setattr(
        controller.journal_module,
        "create_latency_progress_journal",
        portable_create_journal,
    )
    monkeypatch.setattr(
        controller.journal_module,
        "read_latency_progress_journal",
        portable_read_journal,
    )
    return cohort, scored


def test_public_cli_qualify_base_carries_only_terminal_root(tmp_path: Path) -> None:
    namespace = cli.build_parser().parse_args(
        [
            "qualify-base",
            "--config",
            str(PUBLIC_CONFIG),
            "--terminal-root",
            str(tmp_path / "terminals"),
        ]
    )
    request = cli._build_command_request(namespace)
    assert request.command == "qualify-base"
    assert request.terminal_root == tmp_path / "terminals"
    assert {
        field.name
        for field in fields(request)
        if field.name not in {"command", "terminal_root"}
        and getattr(request, field.name) is not None
    } == set()
    signature = inspect.signature(controller.run_host_base_qualification)
    assert not {
        "winner",
        "scores",
        "latencies",
        "parameter_count",
        "qualification",
        "authorization",
    } & set(signature.parameters)


@pytest.mark.parametrize(
    "changed",
    [
        {"registered_cuda_uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
        {"registered_cuda_uuid": "GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"},
        {"registered_cuda_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeef" + " "},
        {"wall_timeout_seconds": True},
        {"wall_timeout_seconds": controller.MAX_WALL_TIMEOUT_SECONDS + 1},
        {"session_id": "../escape"},
    ],
)
def test_controller_configuration_rejects_noncanonical_values(
    tmp_path: Path,
    changed: dict[str, object],
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    values = {
        "registry_path": registry,
        "registry_sha256": _digest(registry.read_bytes()),
        "output_root": tmp_path,
        "journal_parent": tmp_path,
        "session_id": "session-001",
        "registered_cuda_uuid": CUDA_UUID,
        "latency_protocol_sha256": SHA_A,
        "launcher_sha256": SHA_B,
        "wall_timeout_seconds": 3600,
    }
    values.update(changed)
    with pytest.raises((controller.HostBaseQualificationError, TypeError)):
        controller.HostBaseQualificationConfig(**values)


def test_attempt_registry_is_exact_canonical_nine_run_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    raw = _registry_raw()
    path.write_bytes(raw)
    rows, consumed, digest = controller.load_base_attempt_registry(
        path,
        expected_sha256=_digest(raw),
    )
    assert consumed == raw
    assert digest == _digest(raw)
    assert tuple(row.run_id for row in rows) == base_run_ids()

    path.write_bytes(_registry_raw(reverse=True))
    with pytest.raises(controller.HostBaseQualificationError, match="run order"):
        controller.load_base_attempt_registry(
            path,
            expected_sha256=_digest(path.read_bytes()),
        )

    noncanonical = raw[:-1] + b" \n"
    path.write_bytes(noncanonical)
    with pytest.raises(controller.HostBaseQualificationError, match="canonical"):
        controller.load_base_attempt_registry(
            path,
            expected_sha256=_digest(noncanonical),
        )


def test_private_host_v2_loads_closed_qualification_configuration(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    value = {
        "base_qualification": {
            "journal_parent": "journals",
            "latency_protocol_sha256": SHA_A,
            "launcher_sha256": SHA_B,
            "output_root": "outputs",
            "registered_cuda_uuid": CUDA_UUID,
            "registry_path": "registry.json",
            "registry_sha256": _digest(registry.read_bytes()),
            "schema": controller.CONFIG_SCHEMA,
            "session_id": "qualification-001",
            "wall_timeout_seconds": 3600,
        },
        "prepared_index": "prepared-index.json",
        "receipts": {
            "artifacts": {
                name: f"receipts/{name}.json" for name in host.RECEIPT_DIGEST_FIELDS
            },
            "record": "receipts/record.json",
        },
        "runtime": {
            "bf16_runtime_qualified": False,
            "checkpoint_every_updates": 1,
            "corrective_change_sha256": SHA_C,
            "device": "cuda",
            "edge_budget": 32,
            "max_batch_bytes": 1024,
            "request_bf16": False,
            "resume_retry_class": "RESOURCE",
        },
        "schema": host.HOST_CONFIG_QUALIFICATION_SCHEMA,
        "source_tree_sha256": _digest(b"source tree"),
    }
    config_path = tmp_path / "host.json"
    config_path.write_bytes(_canonical(value))
    loaded = host.HostConfig.load(config_path)
    assert loaded.base_qualification is not None
    assert loaded.base_qualification.registry_path == registry.resolve()
    assert loaded.base_qualification.wall_timeout_seconds == 3600
    with pytest.raises(host.HostConfigurationError, match="only one"):
        loaded.validate_base_qualification_request(
            cli.CLICommandRequest(
                command="qualify-base",
                terminal_root=tmp_path / "terminals",
                base_checkpoint=tmp_path / "incompatible-checkpoint.pt",
            )
        )

    value["schema"] = host.HOST_CONFIG_SCHEMA
    config_path.write_bytes(_canonical(value))
    with pytest.raises(host.HostConfigurationError, match="schema v2"):
        host.HostConfig.load(config_path)


def test_injected_latency_runtime_retains_complete_journal_but_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_root = tmp_path / "terminals"
    terminal_root.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    config = _controller_config(tmp_path, registry)
    output_directory = config.output_root / config.session_id
    output_directory.mkdir()
    cohort, scored = _patch_pre_latency(monkeypatch)
    session_sha = _digest(b"engineering latency session")

    def fake_latency(*_args, **kwargs):
        assert kwargs["_runtime"] is fake_runtime
        _emit_completed_journal(kwargs["observer"], session_sha)
        return SimpleNamespace(sha256=session_sha)

    def forbidden_assembly(*_args, **_kwargs):
        pytest.fail("nonformal latency reached qualification assembly")

    fake_runtime = object()
    monkeypatch.setattr(
        controller.latency_module,
        "run_base_cohort_latency_session",
        fake_latency,
    )
    monkeypatch.setattr(
        controller.assembly_module,
        "assemble_base_cohort_qualification",
        forbidden_assembly,
    )
    result = controller.run_host_base_qualification(
        terminal_root,
        config=config,
        admission=_admission(),
        validation_source=_FakeCaptureValidationSource(),
        output_directory=output_directory,
        adapter_sha256=SHA_C,
        _test_runtime=fake_runtime,
    )
    assert result.execution.outcome == "HELD"
    assert result.formal is False
    assert result.authentication is None
    assert result.execution.semantic_evidence is None
    failure = json.loads((output_directory / "failure.json").read_bytes())
    assert failure["failure_code"] == "NONFORMAL_TEST_RUNTIME"
    recovered = journal.read_latency_progress_journal(
        result.journal_path,
        expected_bindings=journal.LatencyJournalBindings(
            resolved_cohort_sha256=(
                controller.latency_module._resolved_cohort_sha256(cohort)
            ),
            scored_cohort_sha256=scored.sha256,
            source_tree_sha256=cohort.source_tree_sha256,
            latency_protocol_sha256=config.latency_protocol_sha256,
            wrapper_sha256=config.launcher_sha256,
            registered_cuda_uuid=config.registered_cuda_uuid,
        ),
    )
    assert recovered.status == journal.COMPLETED_STATUS
    assert recovered.formal_latency is False


def test_formal_call_never_passes_runtime_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_root = tmp_path / "terminals"
    terminal_root.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    config = _controller_config(tmp_path, registry)
    output_directory = config.output_root / config.session_id
    output_directory.mkdir()
    cohort, _old_scored = _patch_pre_latency(monkeypatch)
    session_sha = _digest(b"substituted formal latency session")
    observed_kwargs: dict[str, object] = {}
    ordered_events: list[str] = []
    identity = _cuda_identity()
    state = controller._ScoringCudaState(
        runtime=SimpleNamespace(),
        admission_observation=_cuda_observation(1),
        runtime_identity=identity,
        post_context_observation=_cuda_observation(2),
    )
    scored = SimpleNamespace(
        sha256=_digest(b"scored"),
        observations=(
            SimpleNamespace(device="cuda:0", precision_mode="FP32"),
        ),
    )
    monkeypatch.setattr(
        controller.resolver_module,
        "resolve_completed_base_cohort",
        lambda *_args, **_kwargs: cohort,
    )
    monkeypatch.setattr(
        controller.scoring_module,
        "score_resolved_base_cohort",
        lambda *_args, **_kwargs: (ordered_events.append("score"), scored)[1],
    )
    monkeypatch.setattr(
        controller.scoring_module,
        "BaseCohortValidationResult",
        SimpleNamespace,
    )
    monkeypatch.setattr(
        controller,
        "_admit_scoring_cuda",
        lambda *_args: (ordered_events.append("admit"), state)[1],
    )
    monkeypatch.setattr(
        controller,
        "_release_scoring_cuda",
        lambda _state: (ordered_events.append("release"), _cuda_observation(3))[1],
    )

    def fake_latency(*_args, **kwargs):
        ordered_events.append("latency")
        observed_kwargs.update(kwargs)
        _emit_completed_journal(kwargs["observer"], session_sha)
        return SimpleNamespace(sha256=session_sha)

    monkeypatch.setattr(
        controller.latency_module,
        "run_base_cohort_latency_session",
        fake_latency,
    )
    result = controller.run_host_base_qualification(
        terminal_root,
        config=config,
        admission=_admission(),
        validation_source=_FakeCaptureValidationSource(),
        output_directory=output_directory,
        adapter_sha256=SHA_C,
    )
    assert "_runtime" not in observed_kwargs
    assert ordered_events == ["admit", "score", "release", "latency"]
    assert result.execution.outcome != "COMPLETED"
    assert result.authentication is None


def test_scoring_cuda_admission_and_release_keep_original_zero_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    config = _controller_config(tmp_path, registry)
    observations = iter((_cuda_observation(1), _cuda_observation(2)))
    identity = _cuda_identity()
    runtime_events: list[str] = []
    cuda_initialized_before = controller.latency_module.torch.cuda.is_initialized()
    monkeypatch.setattr(controller, "_scoring_precision", lambda _admission: "FP32")

    class FakeRuntime:
        def snapshot(self):
            runtime_events.append("snapshot")
            return next(observations)

        def initialize(self, *, device, precision_mode, admission_observation):
            runtime_events.append("initialize")
            assert str(device) == "cuda"
            assert precision_mode == "FP32"
            assert admission_observation.captured_unix_ns == 1
            return identity

    admitted = controller._admit_scoring_cuda(
        config,
        _admission(),
        _runtime=FakeRuntime(),
    )
    assert admitted.runtime_identity is identity
    assert runtime_events == ["snapshot", "initialize", "snapshot"]
    assert controller.latency_module.torch.cuda.is_initialized() is cuda_initialized_before

    release_events: list[str] = []

    class ReleaseRuntime:
        def synchronize(self, device):
            assert device == "cuda:0"
            release_events.append("synchronize")

        def snapshot(self):
            release_events.append("snapshot")
            return _cuda_observation(3)

    fake_torch = SimpleNamespace(
        device=lambda value: value,
        _C=SimpleNamespace(
            _cuda_clearCublasWorkspaces=lambda: release_events.append(
                "clear_cublas_workspaces"
            )
        ),
        cuda=SimpleNamespace(
            empty_cache=lambda: release_events.append("empty_cache"),
            memory_allocated=lambda _device: 0,
            memory_reserved=lambda _device: 0,
        ),
    )
    released = controller._release_scoring_cuda(
        controller._ScoringCudaState(
            runtime=ReleaseRuntime(),
            admission_observation=_cuda_observation(1),
            runtime_identity=identity,
            post_context_observation=_cuda_observation(2),
        ),
        _torch_api=fake_torch,
    )
    assert released.captured_unix_ns == 3
    assert release_events == [
        "synchronize",
        "clear_cublas_workspaces",
        "synchronize",
        "empty_cache",
        "snapshot",
    ]

    fake_torch.cuda.memory_reserved = lambda _device: 1
    with pytest.raises(controller._ScoringCudaHold, match="remained"):
        controller._release_scoring_cuda(
            controller._ScoringCudaState(
                runtime=ReleaseRuntime(),
                admission_observation=_cuda_observation(1),
                runtime_identity=identity,
                post_context_observation=_cuda_observation(2),
            ),
            _torch_api=fake_torch,
        )

    fake_torch._C = SimpleNamespace()
    with pytest.raises(controller._ScoringCudaHold) as missing:
        controller._release_scoring_cuda(
            controller._ScoringCudaState(
                runtime=ReleaseRuntime(),
                admission_observation=_cuda_observation(1),
                runtime_identity=identity,
                post_context_observation=_cuda_observation(2),
            ),
            _torch_api=fake_torch,
        )
    assert missing.value.failure_code == "RUNTIME_DRIFT"


@pytest.mark.parametrize("mode", ["admission-observation", "identity", "post-context"])
def test_scoring_cuda_malformed_runtime_values_are_runtime_drift_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    config = _controller_config(tmp_path, registry)
    cuda_initialized_before = controller.latency_module.torch.cuda.is_initialized()
    monkeypatch.setattr(controller, "_scoring_precision", lambda _admission: "FP32")
    observations = iter(
        (
            object() if mode == "admission-observation" else _cuda_observation(1),
            object() if mode == "post-context" else _cuda_observation(2),
        )
    )

    class MalformedRuntime:
        def snapshot(self):
            return next(observations)

        def initialize(self, **_kwargs):
            return object() if mode == "identity" else _cuda_identity()

    with pytest.raises(controller._ScoringCudaHold) as held:
        controller._admit_scoring_cuda(
            config,
            _admission(),
            _runtime=MalformedRuntime(),
        )
    assert held.value.failure_code == "RUNTIME_DRIFT"
    assert controller.latency_module.torch.cuda.is_initialized() is cuda_initialized_before


def test_scoring_precision_forwards_actual_admission_to_real_cpu_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = replace(_admission(), device="cpu")
    original = controller.training_module.resolve_precision
    observed: list[controller.training_module.TrainingConfig] = []
    cuda_initialized_before = controller.latency_module.torch.cuda.is_initialized()

    def observed_resolution(config):
        observed.append(config)
        return original(config)

    monkeypatch.setattr(
        controller.training_module,
        "resolve_precision",
        observed_resolution,
    )
    assert controller._scoring_precision(admission) == "FP32"
    assert len(observed) == 1
    config = observed[0]
    assert type(config) is controller.training_module.TrainingConfig
    assert config.stage == "base"
    assert config.seed == 1729
    assert config.device == admission.device
    assert config.request_bf16 is admission.request_bf16
    assert config.bf16_runtime_qualified is admission.bf16_runtime_qualified
    assert config.edge_budget == admission.edge_budget
    assert config.checkpoint_every_updates == admission.checkpoint_every_updates
    assert config.synthetic_contract is False
    assert controller.latency_module.torch.cuda.is_initialized() is cuda_initialized_before


def test_scoring_cuda_malformed_release_snapshot_is_runtime_drift_hold() -> None:
    class MalformedReleaseRuntime:
        def synchronize(self, _device):
            return None

        def snapshot(self):
            return object()

    fake_torch = SimpleNamespace(
        device=lambda value: value,
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: None),
        cuda=SimpleNamespace(
            empty_cache=lambda: None,
            memory_allocated=lambda _device: 0,
            memory_reserved=lambda _device: 0,
        ),
    )
    state = controller._ScoringCudaState(
        runtime=MalformedReleaseRuntime(),
        admission_observation=_cuda_observation(1),
        runtime_identity=_cuda_identity(),
        post_context_observation=_cuda_observation(2),
    )
    with pytest.raises(controller._ScoringCudaHold) as held:
        controller._release_scoring_cuda(state, _torch_api=fake_torch)
    assert held.value.failure_code == "RUNTIME_DRIFT"


def test_scoring_cuda_malformed_release_device_is_runtime_drift_hold() -> None:
    fake_torch = SimpleNamespace(
        device=lambda _value: (_ for _ in ()).throw(ValueError("bad device")),
    )
    state = controller._ScoringCudaState(
        runtime=SimpleNamespace(),
        admission_observation=_cuda_observation(1),
        runtime_identity=_cuda_identity(),
        post_context_observation=_cuda_observation(2),
    )
    with pytest.raises(controller._ScoringCudaHold) as held:
        controller._release_scoring_cuda(state, _torch_api=fake_torch)
    assert held.value.failure_code == "RUNTIME_DRIFT"


def test_scoring_resource_failure_retains_original_cause_and_secondary_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_root = tmp_path / "terminals"
    terminal_root.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    config = _controller_config(tmp_path, registry)
    output_directory = config.output_root / config.session_id
    output_directory.mkdir()
    _patch_pre_latency(monkeypatch)
    state = controller._ScoringCudaState(
        runtime=SimpleNamespace(),
        admission_observation=_cuda_observation(1),
        runtime_identity=_cuda_identity(),
        post_context_observation=_cuda_observation(2),
    )

    class CleanupBoom(RuntimeError):
        pass

    resource_error = controller.CaptureValidationResourceLimit("capture budget")
    score_error = controller.scoring_module.BaseCohortValidationError("score failed")
    score_error.__cause__ = resource_error

    monkeypatch.setattr(controller, "_admit_scoring_cuda", lambda *_args: state)
    monkeypatch.setattr(
        controller.scoring_module,
        "score_resolved_base_cohort",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(score_error),
    )
    monkeypatch.setattr(
        controller,
        "_release_scoring_cuda",
        lambda _state: (_ for _ in ()).throw(CleanupBoom("cleanup")),
    )
    result = controller.run_host_base_qualification(
        terminal_root,
        config=config,
        admission=_admission(),
        validation_source=_FakeCaptureValidationSource(),
        output_directory=output_directory,
        adapter_sha256=SHA_C,
    )
    assert result.execution.outcome == "HELD"
    failure = json.loads((output_directory / "failure.json").read_bytes())
    assert failure["stage"] == "SCORE"
    assert failure["error_type"] == "BaseCohortValidationError"
    assert failure["failure_code"] == "BASE_CAPTURE_SCORE_RESOURCE_LIMIT"
    assert failure["secondary_cleanup_error_type"] == CleanupBoom.__qualname__
    assert score_error.__cause__ is resource_error
    assert result.journal_path is None


def test_structural_controller_completion_persists_typed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Software branch coverage only; every result-bearing dependency is patched."""

    terminal_root = tmp_path / "terminals"
    terminal_root.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    config = _controller_config(tmp_path, registry)
    output_directory = config.output_root / config.session_id
    output_directory.mkdir()
    cohort, _old_scored = _patch_pre_latency(monkeypatch)
    structural = _typed_structural_assembly()
    identity = _cuda_identity()
    state = controller._ScoringCudaState(
        runtime=SimpleNamespace(),
        admission_observation=_cuda_observation(1),
        runtime_identity=identity,
        post_context_observation=_cuda_observation(2),
    )
    scored = SimpleNamespace(
        sha256=_digest(b"scored"),
        source_manifest_sha256=_FakeCaptureValidationSource.manifest_sha256,
        observations=(
            SimpleNamespace(device="cuda:0", precision_mode="FP32"),
        ),
    )
    session_sha = structural.dependency_binding.latency_session_sha256
    latency = SimpleNamespace(
        sha256=session_sha,
        actual_runtime=True,
        complete=True,
        formal_samples_complete=True,
        failure_code=None,
        status=controller.latency_module.ACTUAL_COMPLETE_STATUS,
    )
    monkeypatch.setattr(
        controller.resolver_module,
        "resolve_completed_base_cohort",
        lambda *_args, **_kwargs: cohort,
    )
    monkeypatch.setattr(
        controller.scoring_module,
        "score_resolved_base_cohort",
        lambda *_args, **_kwargs: scored,
    )
    monkeypatch.setattr(
        controller.scoring_module,
        "BaseCohortValidationResult",
        SimpleNamespace,
    )
    monkeypatch.setattr(controller, "_admit_scoring_cuda", lambda *_args: state)
    monkeypatch.setattr(
        controller,
        "_release_scoring_cuda",
        lambda _state: _cuda_observation(3),
    )

    def fake_latency(*_args, **kwargs):
        assert "_runtime" not in kwargs
        _emit_completed_journal(kwargs["observer"], session_sha)
        return latency

    monkeypatch.setattr(
        controller.latency_module,
        "run_base_cohort_latency_session",
        fake_latency,
    )
    monkeypatch.setattr(
        controller.latency_module,
        "BaseCohortLatencySession",
        SimpleNamespace,
    )
    monkeypatch.setattr(
        controller.assembly_module,
        "assemble_base_cohort_qualification",
        lambda *_args, **_kwargs: structural,
    )
    result = controller.run_host_base_qualification(
        terminal_root,
        config=config,
        admission=_admission(),
        validation_source=_FakeCaptureValidationSource(),
        output_directory=output_directory,
        adapter_sha256=SHA_C,
    )
    assert result.execution.outcome == "COMPLETED"
    assert result.formal is True
    assert result.authentication is not None
    assert result.authentication.verifies(
        result.authentication.authorization,
        qualification_sha256=_digest(structural.artifact),
        adapter_sha256=SHA_C,
    )
    assert result.execution.semantic_evidence is structural.execution_evidence
    assert (output_directory / "scoring-cuda.json") in result.execution.artifacts
    assert (output_directory / "base-qualification.json").read_bytes() == structural.artifact


def test_authentication_reopens_exact_controller_bytes(tmp_path: Path) -> None:
    grant = b"grant bytes\n"
    qualification = b"qualification bytes\n"
    grant_path = tmp_path / "grant.json"
    qualification_path = tmp_path / "qualification.json"
    grant_path.write_bytes(grant)
    qualification_path.write_bytes(qualification)
    authorization = controller.production_module.PrivateBaseQualificationAuthorization(
        authorized_at_utc="2026-09-08T00:00:00Z",
        external_grant_sha256=_digest(grant),
        validation_manifest_sha256=SHA_A,
        query_census_sha256=SHA_B,
        evaluator_sha256=SHA_C,
        score_rows_sha256=_digest(b"rows"),
    )
    value = controller.HostBaseQualificationAuthentication(
        authorization=authorization,
        qualification_sha256=_digest(qualification),
        adapter_sha256=_digest(b"adapter"),
        grant_path=grant_path,
        qualification_path=qualification_path,
        grant_raw=grant,
        qualification_raw=qualification,
    )
    assert value.verifies(
        authorization,
        qualification_sha256=_digest(qualification),
        adapter_sha256=_digest(b"adapter"),
    )
    qualification_path.write_bytes(qualification + b"tamper")
    assert not value.verifies(
        authorization,
        qualification_sha256=_digest(qualification),
        adapter_sha256=_digest(b"adapter"),
    )


def _receipts() -> PrivateReceiptAssertions:
    return PrivateReceiptAssertions(
        admitted_at_utc="2026-09-08T00:00:00Z",
        data_manifest_sha256=_digest(b"data"),
        prepared_data_manifest_sha256=_digest(b"prepared"),
        split_audit_sha256=_digest(b"split"),
        rights_assertion_sha256=_digest(b"rights"),
        caption_manifest_sha256=_digest(b"captions"),
        score_execution_census_sha256=_digest(b"score census"),
        validation_evaluation_census_sha256=_digest(b"val census"),
        hard_gallery_freeze_binding_sha256=_digest(b"freeze"),
        validation_hard_gallery_collection_sha256=_digest(b"gallery"),
        runtime_assertion_sha256=_digest(b"runtime assertion"),
        execution_assertion_sha256=_digest(b"execution assertion"),
        authority=1,
    )


def _typed_structural_assembly() -> assembly.BaseCohortQualificationAssembly:
    counts = {"B0": 100, "B1": 200, "B2": 300}
    fractions = {"B0": Fraction(1, 2), "B1": Fraction(3, 4), "B2": Fraction(3, 4)}
    rows = []
    for ordinal, run_id in enumerate(base_run_ids()):
        _role, seed, system_id = experiments.parse_run_id(run_id)
        rows.append(
            assembly.CombinedBaseScoreBinding(
                run_id=run_id,
                attempt_id=f"structural-attempt-{ordinal}",
                system_id=system_id,
                seed=seed,
                terminal_sha256=_digest(f"terminal-{ordinal}".encode("ascii")),
                selected_checkpoint_sha256=_digest(
                    f"checkpoint-{ordinal}".encode("ascii")
                ),
                selected_checkpoint_state_digest=_digest(
                    f"state-{ordinal}".encode("ascii")
                ),
                validation_score_artifact_sha256=_digest(
                    f"score-{ordinal}".encode("ascii")
                ),
                latency_row_sha256=_digest(f"latency-{ordinal}".encode("ascii")),
                latency_samples_sha256=_digest(
                    f"samples-{ordinal}".encode("ascii")
                ),
                parameter_census_sha256=_digest(
                    f"parameters-{system_id}".encode("ascii")
                ),
                parameter_count=counts[system_id],
                frozen_runtime_latency_ns=100 + ordinal,
                bidirectional_capture_r1=fractions[system_id],
                validation_manifest_sha256=_digest(b"val-census"),
                validation_source_manifest_sha256=_digest(b"val-source"),
                query_census_sha256=_digest(b"val-census"),
                evaluator_sha256=_digest(b"placeholder-evaluator"),
                runtime_sha256=_digest(b"runtime"),
                environment_sha256=_digest(b"environment"),
                precision_mode="FP32",
            )
        )
    selector = controller.production_module.current_base_qualification_artifact_bytes()
    evaluator = _digest(selector)
    rebound = tuple(replace(row, evaluator_sha256=evaluator) for row in rows)
    score_rows = assembly._base_scores(rebound)
    qualification = experiments.qualify_base(score_rows)
    dependency = assembly.BaseQualificationDependencyBinding(
        admission_sha256=_digest(b"admission"),
        resolved_cohort_sha256=_digest(b"resolved"),
        scored_cohort_sha256=_digest(b"scored"),
        latency_session_sha256=_digest(b"latency-session"),
        plan_sha256=_digest(b"plan"),
        matrix_sha256=_digest(b"matrix"),
        training_config_sha256=_digest(b"config"),
        source_tree_sha256=_digest(b"source-tree"),
        train_manifest_sha256=_digest(b"train"),
        admitted_val_census_sha256=_digest(b"val-census"),
        validation_source_manifest_sha256=_digest(b"val-source"),
        query_census_sha256=_digest(b"val-census"),
        resolver_source_sha256=_digest(b"resolver-source"),
        scorer_source_sha256=_digest(b"scorer-source"),
        latency_source_sha256=_digest(b"latency-source"),
        assembler_source_sha256=_digest(b"assembler-source"),
        production_source_sha256=_digest(b"production-source"),
        selector_artifact_sha256=evaluator,
        runtime_sha256=_digest(b"runtime"),
        environment_sha256=_digest(b"environment"),
        precision_mode="FP32",
    )
    evidence = controller.production_module.BaseQualificationExecutionEvidence(
        score_rows=score_rows,
        selector_artifact=selector,
    )
    return assembly.BaseCohortQualificationAssembly(
        qualification=qualification,
        execution_evidence=evidence,
        artifact=experiments.canonical_base_qualification_bytes(qualification),
        row_bindings=rebound,
        dependency_binding=dependency,
    )


def test_cli_factory_adapter_backend_reaches_typed_controller_before_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_root = tmp_path / "terminals"
    output_root = tmp_path / "outputs"
    journal_parent = tmp_path / "journals"
    terminal_root.mkdir()
    output_root.mkdir()
    journal_parent.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    qualification = controller.HostBaseQualificationConfig(
        registry_path=registry,
        registry_sha256=_digest(registry.read_bytes()),
        output_root=output_root,
        journal_parent=journal_parent,
        session_id="cli-session",
        registered_cuda_uuid=CUDA_UUID,
        latency_protocol_sha256=SHA_A,
        launcher_sha256=SHA_B,
        wall_timeout_seconds=7200,
    )
    receipts = _receipts()
    fake_val = _FakeCaptureValidationSource()
    fake_train = SimpleNamespace(manifest_sha256=_digest(b"train"))
    calls: list[dict[str, object]] = []

    class FakeConfig:
        source_tree_sha256 = _digest(b"source")
        device = "cuda"
        request_bf16 = False
        bf16_runtime_qualified = False
        edge_budget = 32
        checkpoint_every_updates = 1
        base_qualification = qualification

        def load_receipts(self):
            return receipts

        def validate_runtime(self):
            return None

        def validate_base_qualification_request(self, request):
            assert request.command == "qualify-base"
            assert request.terminal_root == terminal_root

        def load_sources(self):
            return fake_train, fake_val

    def fake_controller(attempt_root: Path, **kwargs):
        calls.append({"attempt_root": attempt_root, **kwargs})
        artifact = kwargs["output_directory"] / "nonformal-hold.json"
        artifact.write_bytes(b"nonformal host test hold\n")
        return controller.HostBaseQualificationRun(
            execution=BackendExecution(
                outcome="HELD",
                completed_at_utc="2026-09-08T00:00:01Z",
                artifacts=(artifact,),
            ),
            authentication=None,
            journal_path=None,
            registry_sha256=qualification.registry_sha256,
            formal=True,
        )

    monkeypatch.setattr(host.HostConfig, "load", lambda _path: FakeConfig())
    monkeypatch.setattr(host, "CaptureValidationSource", _FakeCaptureValidationSource)
    monkeypatch.setattr(host, "run_host_base_qualification", fake_controller)
    output = io.StringIO()
    errors = io.StringIO()
    exit_code = cli.main(
        [
            "qualify-base",
            "--config",
            str(PUBLIC_CONFIG),
            "--terminal-root",
            str(terminal_root),
        ],
        runtime_adapter_factory=host.RuntimeAdapterFactory(tmp_path / "host.json"),
        stdout=output,
        stderr=errors,
    )
    assert exit_code == cli.EXIT_HOLD
    assert errors.getvalue() == ""
    emitted = json.loads(output.getvalue())
    assert emitted["command"] == "qualify-base"
    assert emitted["outcome"] == "HELD"
    assert len(calls) == 1
    call = calls[0]
    assert call["attempt_root"] == terminal_root.resolve()
    assert type(call["admission"]) is controller.resolver_module.BaseCohortAdmission
    assert call["validation_source"] is fake_val
    assert call["config"] is qualification
    assert call["adapter_sha256"] == call["adapter_sha256"].lower()


@pytest.mark.parametrize(
    ("failure_site", "expected_holds"),
    [
        (
            "source",
            [
                "HOLD_PREPARED_DATA_MANIFEST_ABSENT",
                "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
            ],
        ),
        ("output", ["HOLD_EXTERNAL_RECEIPTS_UNVERIFIED"]),
        ("lease-acquire", ["HOLD_EXTERNAL_RECEIPTS_UNVERIFIED"]),
        ("lease-release", ["HOLD_EXTERNAL_RECEIPTS_UNVERIFIED"]),
    ],
)
def test_cli_qualify_base_post_admission_runtime_drift_is_hold_not_contract_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    expected_holds: list[str],
) -> None:
    terminal_root = tmp_path / "terminals"
    output_root = tmp_path / "outputs"
    journal_parent = tmp_path / "journals"
    terminal_root.mkdir()
    output_root.mkdir()
    journal_parent.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    qualification = controller.HostBaseQualificationConfig(
        registry_path=registry,
        registry_sha256=_digest(registry.read_bytes()),
        output_root=output_root,
        journal_parent=journal_parent,
        session_id=f"runtime-drift-{failure_site}",
        registered_cuda_uuid=CUDA_UUID,
        latency_protocol_sha256=SHA_A,
        launcher_sha256=SHA_B,
        wall_timeout_seconds=7200,
    )
    receipts = _receipts()
    fake_val = _FakeCaptureValidationSource()
    fake_train = SimpleNamespace(manifest_sha256=_digest(b"train"))
    source_calls = 0

    class FakeConfig:
        source_tree_sha256 = _digest(b"source")
        device = "cuda"
        request_bf16 = False
        bf16_runtime_qualified = False
        edge_budget = 32
        checkpoint_every_updates = 1
        base_qualification = qualification

        def load_receipts(self):
            return receipts

        def validate_runtime(self):
            return None

        def validate_base_qualification_request(self, request):
            assert request.command == "qualify-base"
            assert request.terminal_root == terminal_root

        def load_sources(self):
            nonlocal source_calls
            source_calls += 1
            if source_calls == 1 and failure_site == "output":
                (output_root / qualification.session_id).mkdir()
            if source_calls == 2 and failure_site == "source":
                raise host.HostConfigurationError(
                    "prepared manifest changed after admission"
                )
            return fake_train, fake_val

    class FailingReleaseLease:
        def release(self):
            raise OSError("lease release failed")

    if failure_site == "lease-acquire":
        monkeypatch.setattr(
            host._AttemptLease,
            "acquire",
            lambda _directory: (_ for _ in ()).throw(
                host.HostConfigurationError("lease acquisition failed")
            ),
        )
    elif failure_site == "lease-release":
        monkeypatch.setattr(
            host._AttemptLease,
            "acquire",
            lambda _directory: FailingReleaseLease(),
        )

    def controller_call(*_args, **kwargs):
        if failure_site != "lease-release":
            pytest.fail("controller ran after pre-controller runtime drift")
        artifact = kwargs["output_directory"] / "retained-controller-hold.json"
        artifact.write_bytes(b"retained controller hold\n")
        return controller.HostBaseQualificationRun(
            execution=BackendExecution(
                outcome="HELD",
                completed_at_utc="2026-09-08T00:00:01Z",
                artifacts=(artifact,),
            ),
            authentication=None,
            journal_path=None,
            registry_sha256=qualification.registry_sha256,
            formal=True,
        )

    monkeypatch.setattr(host.HostConfig, "load", lambda _path: FakeConfig())
    monkeypatch.setattr(host, "CaptureValidationSource", _FakeCaptureValidationSource)
    monkeypatch.setattr(host, "run_host_base_qualification", controller_call)
    output = io.StringIO()
    errors = io.StringIO()
    exit_code = cli.main(
        [
            "qualify-base",
            "--config",
            str(PUBLIC_CONFIG),
            "--terminal-root",
            str(terminal_root),
        ],
        runtime_adapter_factory=host.RuntimeAdapterFactory(tmp_path / "host.json"),
        stdout=output,
        stderr=errors,
    )
    assert exit_code == cli.EXIT_HOLD
    assert errors.getvalue() == ""
    emitted = json.loads(output.getvalue())
    assert emitted["status"] == "HELD"
    assert emitted["hold_codes"] == expected_holds
    assert source_calls == 2


def test_structural_completion_reaches_production_recompute_and_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract fixture only: no model, cohort, score, or latency was executed."""

    terminal_root = tmp_path / "terminals"
    output_root = tmp_path / "outputs"
    journal_parent = tmp_path / "journals"
    terminal_root.mkdir()
    output_root.mkdir()
    journal_parent.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    qualification_config = controller.HostBaseQualificationConfig(
        registry_path=registry,
        registry_sha256=_digest(registry.read_bytes()),
        output_root=output_root,
        journal_parent=journal_parent,
        session_id="structural-session",
        registered_cuda_uuid=CUDA_UUID,
        latency_protocol_sha256=SHA_A,
        launcher_sha256=SHA_B,
        wall_timeout_seconds=7200,
    )
    receipts = _receipts()
    fake_val = _FakeCaptureValidationSource()
    fake_train = SimpleNamespace(manifest_sha256=_digest(b"train"))
    structural = _typed_structural_assembly()
    callback_observed: list[str] = []

    class FakeConfig:
        source_tree_sha256 = _digest(b"source")
        device = "cuda"
        request_bf16 = False
        bf16_runtime_qualified = False
        edge_budget = 32
        checkpoint_every_updates = 1
        base_qualification = qualification_config

        def load_receipts(self):
            return receipts

        def validate_runtime(self):
            return None

        def validate_base_qualification_request(self, request):
            assert request.command == "qualify-base"
            assert request.terminal_root == terminal_root

        def load_sources(self):
            return fake_train, fake_val

    original_authenticate = host.PhaseSetHostBackend.authenticate_base_qualification

    def observed_authenticate(self, *args, **kwargs):
        result = original_authenticate(self, *args, **kwargs)
        callback_observed.append("accepted" if result else "rejected")
        return result

    def structural_controller(attempt_root: Path, **kwargs):
        assert attempt_root == terminal_root.resolve()
        output_directory = kwargs["output_directory"]
        qualification_path = output_directory / "base-qualification.json"
        grant_path = output_directory / "structural-grant.json"
        qualification_path.write_bytes(structural.artifact)
        grant_raw = b"structural software fixture; not a formal grant\n"
        grant_path.write_bytes(grant_raw)
        authorization = (
            controller.production_module.PrivateBaseQualificationAuthorization(
                authorized_at_utc="2026-09-08T00:00:01Z",
                external_grant_sha256=_digest(grant_raw),
                validation_manifest_sha256=(
                    structural.qualification.validation_manifest_sha256
                ),
                query_census_sha256=structural.qualification.query_census_sha256,
                evaluator_sha256=structural.qualification.evaluator_sha256,
                score_rows_sha256=structural.qualification.score_rows_sha256,
            )
        )
        authentication = controller.HostBaseQualificationAuthentication(
            authorization=authorization,
            qualification_sha256=_digest(structural.artifact),
            adapter_sha256=kwargs["adapter_sha256"],
            grant_path=grant_path,
            qualification_path=qualification_path,
            grant_raw=grant_raw,
            qualification_raw=structural.artifact,
        )
        return controller.HostBaseQualificationRun(
            execution=BackendExecution(
                outcome="COMPLETED",
                completed_at_utc="2026-09-08T00:00:02Z",
                artifacts=(qualification_path, grant_path),
                semantic_evidence=structural.execution_evidence,
            ),
            authentication=authentication,
            journal_path=None,
            registry_sha256=qualification_config.registry_sha256,
            formal=True,
        )

    monkeypatch.setattr(host.HostConfig, "load", lambda _path: FakeConfig())
    monkeypatch.setattr(host, "CaptureValidationSource", _FakeCaptureValidationSource)
    monkeypatch.setattr(host, "run_host_base_qualification", structural_controller)
    monkeypatch.setattr(
        host.PhaseSetHostBackend,
        "authenticate_base_qualification",
        observed_authenticate,
    )
    output = io.StringIO()
    errors = io.StringIO()
    exit_code = cli.main(
        [
            "qualify-base",
            "--config",
            str(PUBLIC_CONFIG),
            "--terminal-root",
            str(terminal_root),
        ],
        runtime_adapter_factory=host.RuntimeAdapterFactory(tmp_path / "host.json"),
        stdout=output,
        stderr=errors,
    )
    assert exit_code == cli.EXIT_OK
    assert errors.getvalue() == ""
    emitted = json.loads(output.getvalue())
    assert emitted["outcome"] == "COMPLETED"
    assert emitted["production"] is True
    assert emitted["result_claimed"] is False
    assert callback_observed == ["accepted"]


def test_formal_timer_runtime_hold_precedes_registry_and_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_root = tmp_path / "terminals"
    terminal_root.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_bytes(_registry_raw())
    config = _controller_config(tmp_path, registry)
    output_directory = config.output_root / config.session_id
    output_directory.mkdir()
    monkeypatch.setattr(controller, "CaptureValidationSource", _FakeCaptureValidationSource)
    monkeypatch.setattr(
        controller,
        "_validate_formal_latency_timer_support",
        lambda: (_ for _ in ()).throw(
            controller._FormalLatencyTimerHold("timer unavailable")
        ),
    )
    monkeypatch.setattr(
        controller,
        "load_base_attempt_registry",
        lambda *_args, **_kwargs: pytest.fail("registry opened before timer preflight"),
    )
    monkeypatch.setattr(
        controller.scoring_module,
        "score_resolved_base_cohort",
        lambda *_args, **_kwargs: pytest.fail("scorer ran before timer preflight"),
    )
    result = controller.run_host_base_qualification(
        terminal_root,
        config=config,
        admission=_admission(),
        validation_source=_FakeCaptureValidationSource(),
        output_directory=output_directory,
        adapter_sha256=SHA_C,
    )
    assert result.execution.outcome == "HELD"
    failure = json.loads((output_directory / "failure.json").read_bytes())
    assert failure["stage"] == "LATENCY_PREFLIGHT"
    assert failure["failure_code"] == "LATENCY_RUNTIME_DRIFT"
    assert not (output_directory / "controller-start.json").exists()


def test_process_timer_refuses_inherited_deadline_without_mutating_os() -> None:
    signal_api = SimpleNamespace(
        SIGALRM=14,
        ITIMER_REAL=0,
        getitimer=lambda _timer: (1.0, 0.0),
        getsignal=lambda _signal: None,
        setitimer=lambda *_args: pytest.fail("inherited timer was replaced"),
        signal=lambda *_args: pytest.fail("inherited handler was replaced"),
    )
    with pytest.raises(controller._FormalLatencyTimerHold, match="inherited"):
        with controller._formal_latency_timeout(
            3600,
            _os_name="posix",
            _signal_api=signal_api,
        ):
            pytest.fail("inherited timer was replaced")


def test_process_timer_fails_closed_when_platform_has_no_interval_timer() -> None:
    with pytest.raises(controller._FormalLatencyTimerHold, match="requires POSIX"):
        with controller._formal_latency_timeout(
            3600,
            _os_name="nt",
            _signal_api=SimpleNamespace(),
        ):
            pytest.fail("unsupported timer platform was admitted")


def test_process_timer_preflight_requires_the_python_main_thread() -> None:
    signal_api = SimpleNamespace(
        SIGALRM=14,
        ITIMER_REAL=0,
        getitimer=lambda _timer: (0.0, 0.0),
        getsignal=lambda _signal: None,
        setitimer=lambda *_args: None,
        signal=lambda *_args: None,
    )
    current = object()
    main = object()
    with pytest.raises(controller._FormalLatencyTimerHold, match="main thread"):
        controller._validate_formal_latency_timer_support(
            _os_name="posix",
            _signal_api=signal_api,
            _threading_api=SimpleNamespace(
                current_thread=lambda: current,
                main_thread=lambda: main,
            ),
        )


@pytest.mark.parametrize("failure_site", ["install", "cleanup"])
def test_process_timer_restores_handler_after_timer_api_failure(
    failure_site: str,
) -> None:
    original_handler = object()

    class SignalFacade:
        SIGALRM = 14
        ITIMER_REAL = 0

        def __init__(self) -> None:
            self.handler = original_handler
            self.armed = False

        def getitimer(self, _timer):
            return (0.0, 0.0)

        def getsignal(self, _signal):
            return self.handler

        def signal(self, _signal, handler):
            self.handler = handler

        def setitimer(self, _timer, seconds, _interval):
            if seconds > 0.0:
                if failure_site == "install":
                    raise OSError("arm failed")
                self.armed = True
                return None
            self.armed = False
            if failure_site == "cleanup":
                raise OSError("disarm failed")
            return None

    facade = SignalFacade()
    with pytest.raises(controller._FormalLatencyTimerHold):
        with controller._formal_latency_timeout(
            3600,
            _os_name="posix",
            _signal_api=facade,
        ):
            assert failure_site == "cleanup"
    assert facade.handler is original_handler
    assert facade.armed is False


def test_process_timer_cleanup_does_not_replace_primary_error() -> None:
    original_handler = object()

    class PrimaryBoom(RuntimeError):
        pass

    class CleanupFailingSignalFacade:
        SIGALRM = 14
        ITIMER_REAL = 0

        def __init__(self) -> None:
            self.handler = original_handler

        def getitimer(self, _timer):
            return (0.0, 0.0)

        def getsignal(self, _signal):
            return self.handler

        def signal(self, _signal, handler):
            self.handler = handler

        def setitimer(self, _timer, seconds, _interval):
            if seconds == 0.0:
                raise OSError("disarm failed")
            return None

    facade = CleanupFailingSignalFacade()
    with pytest.raises(PrimaryBoom, match="primary"):
        with controller._formal_latency_timeout(
            3600,
            _os_name="posix",
            _signal_api=facade,
        ):
            raise PrimaryBoom("primary")
    assert facade.handler is original_handler


@pytest.mark.skipif(
    controller.os.name != "posix",
    reason="the real interval-timer API is POSIX-only",
)
def test_real_posix_interval_timer_restores_process_state() -> None:
    if controller.signal.getitimer(controller.signal.ITIMER_REAL) != (0.0, 0.0):
        pytest.skip("test process already has an interval timer")
    original_handler = controller.signal.getsignal(controller.signal.SIGALRM)
    with controller._formal_latency_timeout(60):
        assert controller.signal.getitimer(controller.signal.ITIMER_REAL)[0] > 0.0
    assert controller.signal.getitimer(controller.signal.ITIMER_REAL) == (0.0, 0.0)
    assert controller.signal.getsignal(controller.signal.SIGALRM) == original_handler
