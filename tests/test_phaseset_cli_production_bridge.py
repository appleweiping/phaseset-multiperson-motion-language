from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path

from phaseset_core import cli, execution, experiments
from phaseset_core.production import (
    BackendExecution,
    PrivateReceiptAssertions,
    ProductionRuntimeAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phaseset" / "training.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _invoke(
    arguments: list[str],
    *,
    runtime_adapter: execution.RuntimeAdapter | None = None,
    runtime_adapter_factory: cli.RuntimeAdapterFactory | None = None,
) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = cli.main(
        arguments,
        stdout=stdout,
        stderr=stderr,
        runtime_adapter=runtime_adapter,
        runtime_adapter_factory=runtime_adapter_factory,
    )
    return code, stdout.getvalue(), stderr.getvalue()


class _PrivateBackend:
    backend_sha256 = _digest("cli-production-backend-v1")

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.authenticate_calls = 0
        self.intents: list[execution.CommandIntent] = []

    def authenticate(
        self,
        request: execution.RuntimeAdmissionRequest,
        receipts: PrivateReceiptAssertions,
        adapter_sha256: str,
    ) -> bool:
        assert request.handler_manifest_sha256 == execution.runtime_handler_manifest_sha256()
        assert receipts.authority == 7
        assert len(adapter_sha256) == 64
        self.authenticate_calls += 1
        return True

    def execute(self, intent: execution.CommandIntent) -> BackendExecution:
        self.intents.append(intent)
        return BackendExecution(
            outcome="COMPLETED",
            completed_at_utc="2026-08-25T23:45:00Z",
            artifacts=(self.artifact,),
        )


def _receipts() -> PrivateReceiptAssertions:
    return PrivateReceiptAssertions(
        admitted_at_utc="2026-08-25T23:44:00Z",
        data_manifest_sha256=_digest("data"),
        prepared_data_manifest_sha256=_digest("prepared"),
        split_audit_sha256=_digest("split"),
        rights_assertion_sha256=_digest("rights"),
        caption_manifest_sha256=_digest("caption-manifest"),
        score_execution_census_sha256=_digest("score-execution-census"),
        validation_evaluation_census_sha256=_digest("validation-evaluation-census"),
        hard_gallery_freeze_binding_sha256=_digest("hard-gallery-freeze-binding"),
        validation_hard_gallery_collection_sha256=_digest(
            "validation-hard-gallery-collection"
        ),
        runtime_assertion_sha256=_digest("runtime"),
        execution_assertion_sha256=_digest("execution"),
        authority=7,
    )


class _SyntheticAdapter:
    commands = execution.COMMANDS
    adapter_sha256 = _digest("cli-request-aware-synthetic")
    handler_manifest_sha256 = execution.runtime_handler_manifest_sha256(commands)

    def admit(
        self, request: execution.RuntimeAdmissionRequest
    ) -> execution.RuntimeAdmission:
        return execution.RuntimeAdmission(
            runtime_mode="SYNTHETIC_DATA_FREE",
            admitted_at_utc="2026-08-25T23:44:00Z",
            adapter_sha256=self.adapter_sha256,
            handler_manifest_sha256=request.handler_manifest_sha256,
            plan_sha256=request.plan_sha256,
            matrix_sha256=request.matrix_sha256,
            training_config_sha256=request.training_config_sha256,
            data_manifest_sha256=_digest("synthetic-data"),
            prepared_data_manifest_sha256=_digest("synthetic-prepared"),
            split_audit_sha256=_digest("synthetic-split"),
            rights_assertion_sha256=_digest("synthetic-rights-not-a-grant"),
            runtime_assertion_sha256=_digest("synthetic-runtime"),
            execution_assertion_sha256=_digest("synthetic-execution"),
            authority=0,
            execution_authorized=True,
            production=False,
            external_authentication_asserted=False,
            public_verification_performed=False,
            result_claimed=False,
            status="SYNTHETIC_DATA_FREE_NO_SCIENTIFIC_RESULT",
        )

    def handle(self, intent: execution.CommandIntent) -> execution.CommandResult:
        assert intent.admission_sha256 is not None
        return execution.CommandResult(
            command=intent.command,
            outcome="COMPLETED",
            completed_at_utc="2026-08-25T23:45:00Z",
            run_id=intent.run_id,
            seed=intent.seed,
            split=intent.split,
            artifact_sha256s=(_digest("synthetic-artifact"),),
            admission_sha256=intent.admission_sha256,
            runtime_mode=intent.runtime_mode,
            authority=intent.authority,
            production=intent.production,
        )


def test_private_parameters_without_backend_remain_authority_zero_hold(tmp_path: Path) -> None:
    private_root = tmp_path / "private-prepared-root"
    private_manifest = tmp_path / "private-source.json"
    code, stdout, stderr = _invoke(
        [
            "prepare-data",
            "--source-manifest",
            str(private_manifest),
            "--prepared-root",
            str(private_root),
            "--config",
            str(CONFIG),
        ]
    )
    payload = json.loads(stdout)
    assert code == cli.EXIT_HOLD
    assert stderr == ""
    assert payload["authority"] == 0
    assert payload["execution_authorized"] is False
    assert str(private_root) not in stdout
    assert str(private_manifest) not in stdout


def test_request_aware_factory_receives_residual_and_checkpoint_parameters(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "terminal.json"
    artifact.write_text("{}\n", encoding="utf-8")
    backend = _PrivateBackend(artifact)
    captured: list[cli.CLICommandRequest] = []

    def factory(request: cli.CLICommandRequest) -> execution.RuntimeAdapter:
        captured.append(request)
        return ProductionRuntimeAdapter(backend, _receipts())

    run_id = next(value for value in experiments.residual_run_ids() if value.endswith("/08"))
    checkpoint_directory = tmp_path / "checkpoints"
    resume_checkpoint = tmp_path / "checkpoint-0001.pt"
    base_checkpoint = tmp_path / "base.pt"
    periodic_cache = tmp_path / "periodic-cache.npz"
    code, stdout, stderr = _invoke(
        [
            "run-residual",
            "--run-id",
            run_id,
            "--system-id",
            "08",
            "--checkpoint-dir",
            str(checkpoint_directory),
            "--resume-checkpoint",
            str(resume_checkpoint),
            "--stop-after-global-step",
            "11",
            "--base-checkpoint",
            str(base_checkpoint),
            "--periodic-cache",
            str(periodic_cache),
            "--config",
            str(CONFIG),
        ],
        runtime_adapter_factory=factory,
    )
    payload = json.loads(stdout)
    assert code == cli.EXIT_OK
    assert stderr == ""
    assert payload["production"] is True
    assert payload["authority"] == 7
    assert payload["notice"].startswith("PRIVATE ADAPTER OUTPUT")
    assert len(captured) == 1
    request = captured[0]
    assert request.run_id == run_id
    assert request.system_id == "08"
    assert request.checkpoint_directory == checkpoint_directory
    assert request.resume_checkpoint == resume_checkpoint
    assert request.stop_after_global_step == 11
    assert request.base_checkpoint == base_checkpoint
    assert request.periodic_cache == periodic_cache
    assert backend.authenticate_calls == 1
    assert backend.intents[0].run_id == run_id
    for private_path in (
        checkpoint_directory,
        resume_checkpoint,
        base_checkpoint,
        periodic_cache,
    ):
        assert str(private_path) not in stdout


def test_resume_and_system_zero_reach_factory_without_path_serialization(tmp_path: Path) -> None:
    captured: list[cli.CLICommandRequest] = []

    def factory(request: cli.CLICommandRequest) -> execution.RuntimeAdapter:
        captured.append(request)
        return _SyntheticAdapter()

    attempt = tmp_path / "attempt-private"
    checkpoint = tmp_path / "checkpoint-private.pt"
    code, stdout, stderr = _invoke(
        [
            "resume",
            "--attempt-dir",
            str(attempt),
            "--checkpoint",
            str(checkpoint),
            "--system-id",
            "00",
            "--config",
            str(CONFIG),
        ],
        runtime_adapter_factory=factory,
    )
    payload = json.loads(stdout)
    assert code == cli.EXIT_OK
    assert stderr == ""
    assert payload["production"] is False
    assert payload["authority"] == 0
    assert payload["notice"] == "SYNTHETIC / NO SCIENTIFIC RESULT"
    assert captured[0].attempt_directory == attempt
    assert captured[0].resume_checkpoint == checkpoint
    assert captured[0].system_id == "00"
    assert str(attempt) not in stdout
    assert str(checkpoint) not in stdout


def test_base_resume_system_id_reaches_request_aware_factory(tmp_path: Path) -> None:
    captured: list[cli.CLICommandRequest] = []

    def factory(request: cli.CLICommandRequest) -> execution.RuntimeAdapter:
        captured.append(request)
        return _SyntheticAdapter()

    attempt = tmp_path / "base-resume-attempt"
    checkpoint = tmp_path / "base-checkpoint.pt"
    code, stdout, stderr = _invoke(
        [
            "resume", "--attempt-dir", str(attempt), "--checkpoint", str(checkpoint),
            "--system-id", "B1", "--config", str(CONFIG),
        ],
        runtime_adapter_factory=factory,
    )
    payload = json.loads(stdout)
    assert code == cli.EXIT_OK
    assert stderr == ""
    assert payload["production"] is False
    assert payload["authority"] == 0
    assert len(captured) == 1
    assert captured[0].attempt_directory == attempt
    assert captured[0].resume_checkpoint == checkpoint
    assert captured[0].system_id == "B1"
    assert str(attempt) not in stdout
    assert str(checkpoint) not in stdout


def test_private_parameters_cannot_be_silently_dropped_by_prebuilt_adapter(
    tmp_path: Path,
) -> None:
    run_id = experiments.base_run_ids()[0]
    code, stdout, stderr = _invoke(
        [
            "run-base",
            "--run-id",
            run_id,
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--config",
            str(CONFIG),
        ],
        runtime_adapter=_SyntheticAdapter(),
    )
    assert code == cli.EXIT_USAGE_OR_CONTRACT
    assert stdout == ""
    assert json.loads(stderr)["error_code"] == "EXECUTION_CONTRACT_REJECTED"
    assert str(tmp_path) not in stderr


def test_explicit_system_id_must_match_registered_run_before_factory_call() -> None:
    captured: list[cli.CLICommandRequest] = []

    def factory(request: cli.CLICommandRequest) -> execution.RuntimeAdapter:
        captured.append(request)
        return _SyntheticAdapter()

    run_id = next(value for value in experiments.residual_run_ids() if value.endswith("/07"))
    code, stdout, stderr = _invoke(
        [
            "run-residual",
            "--run-id",
            run_id,
            "--system-id",
            "08",
            "--config",
            str(CONFIG),
        ],
        runtime_adapter_factory=factory,
    )
    assert code == cli.EXIT_USAGE_OR_CONTRACT
    assert stdout == ""
    assert json.loads(stderr)["error_code"] == "EXECUTION_CONTRACT_REJECTED"
    assert captured == []


def test_negative_training_stop_is_rejected_before_adapter_factory_call() -> None:
    captured: list[cli.CLICommandRequest] = []

    def factory(request: cli.CLICommandRequest) -> execution.RuntimeAdapter:
        captured.append(request)
        return _SyntheticAdapter()

    code, stdout, stderr = _invoke(
        [
            "run-base",
            "--run-id",
            experiments.base_run_ids()[0],
            "--stop-after-global-step",
            "-1",
            "--config",
            str(CONFIG),
        ],
        runtime_adapter_factory=factory,
    )
    assert code == cli.EXIT_USAGE_OR_CONTRACT
    assert stdout == ""
    assert json.loads(stderr)["error_code"] == "EXECUTION_CONTRACT_REJECTED"
    assert captured == []


def test_invalid_cache_seed_is_rejected_before_adapter_factory_call() -> None:
    captured: list[cli.CLICommandRequest] = []

    def factory(request: cli.CLICommandRequest) -> execution.RuntimeAdapter:
        captured.append(request)
        return _SyntheticAdapter()

    code, stdout, stderr = _invoke(
        ["build-periodic-cache", "--seed", "999", "--config", str(CONFIG)],
        runtime_adapter_factory=factory,
    )
    assert code == cli.EXIT_USAGE_OR_CONTRACT
    assert stdout == ""
    assert json.loads(stderr)["error_code"] == "EXECUTION_CONTRACT_REJECTED"
    assert captured == []


def test_missing_config_stops_before_adapter_factory_call(tmp_path: Path) -> None:
    captured: list[cli.CLICommandRequest] = []

    def factory(request: cli.CLICommandRequest) -> execution.RuntimeAdapter:
        captured.append(request)
        return _SyntheticAdapter()

    missing = tmp_path / "missing-private-config.json"
    code, stdout, stderr = _invoke(
        ["prepare-data", "--config", str(missing)],
        runtime_adapter_factory=factory,
    )
    assert code == cli.EXIT_HOLD
    assert stderr == ""
    assert json.loads(stdout)["hold_codes"] == ["HOLD_TRAINING_CONFIG_ABSENT"]
    assert captured == []
    assert str(missing) not in stdout
