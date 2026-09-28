from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phaseset_core import execution, experiments, statistics


ROOT = Path(__file__).resolve().parents[1]
DIGESTS = tuple(character * 64 for character in "abcdef1234567890")


def _paired_capture(
    label: str,
    treatment_text: tuple[int, ...],
    control_text: tuple[int, ...],
    treatment_motion: tuple[int, ...] = (1,),
    control_motion: tuple[int, ...] = (0,),
) -> statistics.PairedCapture:
    identity = lambda domain, ordinal: hashlib.sha256(  # noqa: E731
        f"{label}-{domain}-{ordinal}".encode("ascii")
    ).hexdigest()
    return statistics.PairedCapture(
        capture_id=identity("capture", 0),
        caption_ids=tuple(
            sorted(identity("caption", ordinal) for ordinal in range(len(treatment_text)))
        ),
        motion_ids=tuple(
            sorted(identity("motion", ordinal) for ordinal in range(len(treatment_motion)))
        ),
        treatment_text_to_motion_hits=treatment_text,
        control_text_to_motion_hits=control_text,
        treatment_motion_to_text_hits=treatment_motion,
        control_motion_to_text_hits=control_motion,
    )


def _attempt(attempt_id: str = "attempt-0001") -> execution.AttemptRecord:
    return execution.AttemptRecord(
        attempt_id=attempt_id,
        run_id=experiments.base_run_ids()[0],
        created_at_utc="2026-08-25T20:00:00Z",
        plan_sha256=DIGESTS[0],
        matrix_sha256=DIGESTS[1],
        training_config_sha256=DIGESTS[2],
        source_tree_sha256=DIGESTS[3],
    )


def _checkpoint(parent: str | None = None, step: int = 10) -> execution.CheckpointRecord:
    return execution.CheckpointRecord(
        attempt_id="attempt-0001",
        run_id=experiments.base_run_ids()[0],
        epoch_index=0,
        global_step=step,
        written_at_utc="2026-08-25T20:02:00Z",
        model_state_sha256=DIGESTS[4],
        optimizer_state_sha256=DIGESTS[5],
        rng_state_sha256=DIGESTS[6],
        sampler_state_sha256=DIGESTS[7],
        dataloader_state_sha256=DIGESTS[8],
        dropout_state_sha256=DIGESTS[9],
        validation_state_sha256=DIGESTS[10],
        checkpoint_payload_sha256=DIGESTS[11],
        parent_checkpoint_receipt_sha256=parent,
    )


def test_artifact_dataclasses_and_canonical_roundtrip_are_immutable() -> None:
    record = _attempt()
    with pytest.raises(FrozenInstanceError):
        record.attempt_id = "changed"  # type: ignore[misc]
    raw = execution.canonical_attempt_bytes(record)
    assert execution.parse_attempt_bytes(raw) == record
    with pytest.raises(execution.ExecutionContractError):
        execution.parse_attempt_bytes(raw + b"\n")


def test_attempt_store_is_hash_chained_write_once(tmp_path: Path) -> None:
    store = execution.AttemptStore.create(tmp_path, _attempt())
    heartbeat0 = execution.HeartbeatRecord(
        attempt_id="attempt-0001",
        run_id=experiments.base_run_ids()[0],
        sequence=0,
        global_step=0,
        observed_at_utc="2026-08-25T20:01:00Z",
        phase="STARTING",
        previous_heartbeat_sha256=None,
    )
    heartbeat0_sha = store.write_heartbeat(heartbeat0)
    heartbeat1 = execution.HeartbeatRecord(
        attempt_id="attempt-0001",
        run_id=experiments.base_run_ids()[0],
        sequence=1,
        global_step=10,
        observed_at_utc="2026-08-25T20:02:00Z",
        phase="CHECKPOINTING",
        previous_heartbeat_sha256=heartbeat0_sha,
    )
    heartbeat1_sha = store.write_heartbeat(heartbeat1)
    checkpoint_sha = store.write_checkpoint(_checkpoint())
    attempt_sha = execution.artifact_sha256((store.path / "attempt.json").read_bytes())
    terminal = execution.TerminalRecord(
        attempt_id="attempt-0001",
        run_id=experiments.base_run_ids()[0],
        completed_at_utc="2026-08-25T20:03:00Z",
        outcome="SUCCEEDED",
        attempt_receipt_sha256=attempt_sha,
        latest_heartbeat_sha256=heartbeat1_sha,
        latest_checkpoint_receipt_sha256=checkpoint_sha,
        failure_code=None,
    )
    terminal_sha = store.write_terminal(terminal)
    assert len(terminal_sha) == 64
    assert execution.parse_terminal_bytes((store.path / "terminal.json").read_bytes()) == terminal
    with pytest.raises(execution.ExecutionContractError, match="terminal"):
        store.write_terminal(terminal)
    with pytest.raises(execution.ExecutionContractError, match="terminal"):
        store.write_heartbeat(heartbeat1)


def test_checkpoint_parent_and_step_cannot_be_forged(tmp_path: Path) -> None:
    store = execution.AttemptStore.create(tmp_path, _attempt())
    first_sha = store.write_checkpoint(_checkpoint())
    with pytest.raises(execution.ExecutionContractError, match="parent digest"):
        store.write_checkpoint(_checkpoint(parent=DIGESTS[11], step=20))
    assert store.write_checkpoint(_checkpoint(parent=first_sha, step=20))


def test_open_revalidates_existing_heartbeat_chain(tmp_path: Path) -> None:
    store = execution.AttemptStore.create(tmp_path, _attempt())
    first = execution.HeartbeatRecord(
        attempt_id=store.attempt.attempt_id,
        run_id=store.attempt.run_id,
        sequence=0,
        global_step=0,
        observed_at_utc="2026-08-25T20:01:00Z",
        phase="STARTING",
        previous_heartbeat_sha256=None,
    )
    first_sha = store.write_heartbeat(first)
    store.write_heartbeat(
        execution.HeartbeatRecord(
            attempt_id=store.attempt.attempt_id,
            run_id=store.attempt.run_id,
            sequence=1,
            global_step=1,
            observed_at_utc="2026-08-25T20:01:01Z",
            phase="RUNNING",
            previous_heartbeat_sha256=first_sha,
        )
    )
    execution.AttemptStore.open(tmp_path, store.attempt.attempt_id)
    first_path = store.path / "heartbeats" / "heartbeat-00000000.json"
    first_path.write_bytes(
        execution.canonical_heartbeat_bytes(replace(first, phase="RUNNING"))
    )
    with pytest.raises(execution.ExecutionContractError, match="predecessor digest"):
        execution.AttemptStore.open(tmp_path, store.attempt.attempt_id)


def test_completed_attempt_evidence_returns_owned_bytes_without_second_path_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = execution.AttemptStore.create(tmp_path, _attempt())
    checkpoint = _checkpoint()
    checkpoint_raw = execution.canonical_checkpoint_bytes(checkpoint)
    checkpoint_sha = store.write_checkpoint(checkpoint)
    terminal = execution.TerminalRecord(
        attempt_id=store.attempt.attempt_id,
        run_id=store.attempt.run_id,
        completed_at_utc="2026-08-25T20:03:00Z",
        outcome="SUCCEEDED",
        attempt_receipt_sha256=execution.artifact_sha256(
            (store.path / "attempt.json").read_bytes()
        ),
        latest_heartbeat_sha256=None,
        latest_checkpoint_receipt_sha256=checkpoint_sha,
        failure_code=None,
    )
    terminal_raw = execution.canonical_terminal_bytes(terminal)
    store.write_terminal(terminal)

    reads: dict[Path, int] = {}
    original_read_regular = execution._read_regular

    def counted_read(path: Path, label: str) -> bytes:
        reads[path] = reads.get(path, 0) + 1
        return original_read_regular(path, label)

    monkeypatch.setattr(execution, "_read_regular", counted_read)
    evidence = execution.verified_completed_attempt_evidence(
        tmp_path,
        store.attempt.attempt_id,
    )
    assert evidence.attempt == store.attempt
    assert evidence.latest_checkpoint == checkpoint
    assert evidence.latest_checkpoint_raw == checkpoint_raw
    assert evidence.terminal == terminal
    assert evidence.terminal_raw == terminal_raw
    assert evidence.resume is None
    assert evidence.resume_raw is None
    assert evidence.terminal_sha256 == execution.artifact_sha256(terminal_raw)
    assert evidence.latest_checkpoint_receipt_sha256 == checkpoint_sha
    assert evidence.latest_checkpoint_payload_sha256 == checkpoint.checkpoint_payload_sha256
    assert reads == {
        store.path / "attempt.json": 1,
        store.path / "checkpoints" / "checkpoint-000000000010.json": 1,
        store.path / "terminal.json": 1,
    }

    (store.path / "terminal.json").write_bytes(b"changed after verified return")
    (store.path / "checkpoints" / "checkpoint-000000000010.json").write_bytes(
        b"changed after verified return"
    )
    assert evidence.terminal_raw == terminal_raw
    assert evidence.latest_checkpoint_raw == checkpoint_raw
    with pytest.raises(FrozenInstanceError):
        evidence.terminal = terminal  # type: ignore[misc]

    rebound_terminal = replace(terminal, attempt_receipt_sha256=DIGESTS[15])
    with pytest.raises(execution.ExecutionContractError, match="identity is inconsistent"):
        execution.CompletedAttemptEvidence(
            attempt=evidence.attempt,
            attempt_raw=evidence.attempt_raw,
            latest_checkpoint=evidence.latest_checkpoint,
            latest_checkpoint_raw=evidence.latest_checkpoint_raw,
            terminal=rebound_terminal,
            terminal_raw=execution.canonical_terminal_bytes(rebound_terminal),
            resume=evidence.resume,
            resume_raw=evidence.resume_raw,
        )


def test_completed_attempt_evidence_rejects_open_and_failed_attempts(
    tmp_path: Path,
) -> None:
    failed = execution.AttemptStore.create(tmp_path, _attempt("attempt-failed"))
    with pytest.raises(execution.ExecutionContractError, match="no immutable terminal"):
        execution.verified_completed_attempt_evidence(tmp_path, "attempt-failed")
    failed_checkpoint = replace(_checkpoint(), attempt_id="attempt-failed")
    failed_checkpoint_sha = failed.write_checkpoint(failed_checkpoint)
    failed.write_terminal(
        execution.TerminalRecord(
            attempt_id=failed.attempt.attempt_id,
            run_id=failed.attempt.run_id,
            completed_at_utc="2026-08-25T20:03:00Z",
            outcome="FAILED",
            attempt_receipt_sha256=execution.artifact_sha256(
                (failed.path / "attempt.json").read_bytes()
            ),
            latest_heartbeat_sha256=None,
            latest_checkpoint_receipt_sha256=failed_checkpoint_sha,
            failure_code="INFRA_TRANSIENT",
        )
    )
    with pytest.raises(execution.ExecutionContractError, match="not SUCCEEDED"):
        execution.verified_completed_attempt_evidence(tmp_path, "attempt-failed")


def test_completed_attempt_evidence_revalidates_resume_predecessor_chain(
    tmp_path: Path,
) -> None:
    predecessor = execution.AttemptStore.create(tmp_path, _attempt())
    predecessor_checkpoint = _checkpoint()
    predecessor_checkpoint_sha = predecessor.write_checkpoint(predecessor_checkpoint)
    predecessor_terminal_sha = predecessor.write_terminal(
        execution.TerminalRecord(
            attempt_id=predecessor.attempt.attempt_id,
            run_id=predecessor.attempt.run_id,
            completed_at_utc="2026-08-25T20:03:00Z",
            outcome="FAILED",
            attempt_receipt_sha256=execution.artifact_sha256(
                (predecessor.path / "attempt.json").read_bytes()
            ),
            latest_heartbeat_sha256=None,
            latest_checkpoint_receipt_sha256=predecessor_checkpoint_sha,
            failure_code="INFRA_TRANSIENT",
        )
    )
    resumed_attempt = replace(
        predecessor.attempt,
        attempt_id="attempt-0002",
        created_at_utc="2026-08-25T20:04:00Z",
    )
    resume = execution.ResumeRecord(
        new_attempt_id=resumed_attempt.attempt_id,
        predecessor_attempt_id=predecessor.attempt.attempt_id,
        run_id=resumed_attempt.run_id,
        created_at_utc=resumed_attempt.created_at_utc,
        predecessor_terminal_sha256=predecessor_terminal_sha,
        checkpoint_receipt_sha256=predecessor_checkpoint_sha,
        corrective_change_sha256=DIGESTS[13],
        retry_class="INFRA_TRANSIENT",
    )
    resumed = execution.AttemptStore.create_resumed(
        tmp_path,
        resumed_attempt,
        resume,
    )
    resumed_checkpoint = replace(
        _checkpoint(step=20),
        attempt_id=resumed_attempt.attempt_id,
        checkpoint_payload_sha256=DIGESTS[12],
    )
    resumed_checkpoint_sha = resumed.write_checkpoint(resumed_checkpoint)
    resumed.write_terminal(
        execution.TerminalRecord(
            attempt_id=resumed.attempt.attempt_id,
            run_id=resumed.attempt.run_id,
            completed_at_utc="2026-08-25T20:05:00Z",
            outcome="SUCCEEDED",
            attempt_receipt_sha256=execution.artifact_sha256(
                (resumed.path / "attempt.json").read_bytes()
            ),
            latest_heartbeat_sha256=None,
            latest_checkpoint_receipt_sha256=resumed_checkpoint_sha,
            failure_code=None,
        )
    )
    evidence = execution.verified_completed_attempt_evidence(
        tmp_path,
        resumed_attempt.attempt_id,
    )
    assert evidence.resume == resume
    assert evidence.resume_raw == execution.canonical_resume_bytes(resume)
    assert evidence.latest_checkpoint == resumed_checkpoint

    (predecessor.path / "checkpoints" / "checkpoint-000000000010.json").write_bytes(
        execution.canonical_checkpoint_bytes(
            replace(predecessor_checkpoint, checkpoint_payload_sha256=DIGESTS[14])
        )
    )
    with pytest.raises(execution.ExecutionContractError):
        execution.verified_completed_attempt_evidence(
            tmp_path,
            resumed_attempt.attempt_id,
        )


def test_resume_payload_binding_revalidates_recursive_predecessor_chain(
    tmp_path: Path,
) -> None:
    predecessor = execution.AttemptStore.create(tmp_path, _attempt())
    first_sha = predecessor.write_checkpoint(_checkpoint(step=10))
    latest = replace(
        _checkpoint(parent=first_sha, step=20),
        checkpoint_payload_sha256=DIGESTS[12],
    )
    latest_sha = predecessor.write_checkpoint(latest)
    terminal_sha = predecessor.write_terminal(
        execution.TerminalRecord(
            attempt_id=predecessor.attempt.attempt_id,
            run_id=predecessor.attempt.run_id,
            completed_at_utc="2026-08-25T20:03:00Z",
            outcome="FAILED",
            attempt_receipt_sha256=execution.artifact_sha256(
                (predecessor.path / "attempt.json").read_bytes()
            ),
            latest_heartbeat_sha256=None,
            latest_checkpoint_receipt_sha256=latest_sha,
            failure_code="INFRA_TRANSIENT",
        )
    )
    new_attempt = replace(
        predecessor.attempt,
        attempt_id="attempt-0002",
        created_at_utc="2026-08-25T20:04:00Z",
    )
    resume = execution.ResumeRecord(
        new_attempt_id=new_attempt.attempt_id,
        predecessor_attempt_id=predecessor.attempt.attempt_id,
        run_id=new_attempt.run_id,
        created_at_utc=new_attempt.created_at_utc,
        predecessor_terminal_sha256=terminal_sha,
        checkpoint_receipt_sha256=latest_sha,
        corrective_change_sha256=DIGESTS[13],
        retry_class="INFRA_TRANSIENT",
    )
    execution.AttemptStore.create_resumed(tmp_path, new_attempt, resume)
    assert (
        execution.verified_resume_checkpoint_payload_sha256(tmp_path, resume)
        == DIGESTS[12]
    )

    first_path = predecessor.path / "checkpoints" / "checkpoint-000000000010.json"
    first_path.write_bytes(
        execution.canonical_checkpoint_bytes(
            replace(_checkpoint(step=10), checkpoint_payload_sha256=DIGESTS[14])
        )
    )
    with pytest.raises(execution.ExecutionContractError, match="parent digest"):
        execution.verified_resume_checkpoint_payload_sha256(tmp_path, resume)


@pytest.mark.parametrize("attempt_id", ("../escape", "a/b", "A", "space value", ""))
def test_attempt_id_rejects_path_and_noncanonical_values(tmp_path: Path, attempt_id: str) -> None:
    with pytest.raises(execution.ExecutionContractError):
        execution.AttemptStore.create(tmp_path, _attempt(attempt_id))


def test_sealed_test_consumption_is_single_use_and_not_external_authority(
    tmp_path: Path,
) -> None:
    gate = execution.SealedTestGate(tmp_path)
    receipt_sha = gate.consume(
        external_grant_sha256=DIGESTS[0],
        test_manifest_sha256=DIGESTS[1],
        caption_manifest_sha256=DIGESTS[5],
        evaluator_sha256=DIGESTS[2],
        validation_freeze_sha256=DIGESTS[3],
        aggregate_code_sha256=DIGESTS[4],
        evaluation_census_sha256=DIGESTS[6],
        hard_gallery_freeze_binding_sha256=DIGESTS[7],
        hard_gallery_collection_sha256=DIGESTS[8],
        consumed_at_utc="2026-08-25T21:00:00Z",
    )
    assert len(receipt_sha) == 64
    raw = (tmp_path / "sealed-test-consumption.json").read_bytes()
    assert b'"external_receipt_verified":false' in raw
    assert gate.consumed is True
    with pytest.raises(execution.ExecutionContractError, match="already exists"):
        gate.consume(
            external_grant_sha256=DIGESTS[0],
            test_manifest_sha256=DIGESTS[1],
            caption_manifest_sha256=DIGESTS[5],
            evaluator_sha256=DIGESTS[2],
            validation_freeze_sha256=DIGESTS[3],
            aggregate_code_sha256=DIGESTS[4],
            evaluation_census_sha256=DIGESTS[6],
            hard_gallery_freeze_binding_sha256=DIGESTS[7],
            hard_gallery_collection_sha256=DIGESTS[8],
            consumed_at_utc="2026-08-25T21:01:00Z",
        )


def test_public_training_config_and_runner_fail_closed() -> None:
    raw = execution.load_training_config(ROOT / "configs" / "phaseset" / "training.json")
    config = json.loads(raw)
    assert config["base_training"] == {
        "effective_global_batch": 128,
        "epochs": 30,
        "gradient_clip_decimal": "1.0",
        "learning_rate_decimal": "0.0002",
        "optimizer": "AdamW",
        "schedule": "linear-warmup-cosine-decay",
        "warmup_fraction_decimal": "0.05",
        "weight_decay_decimal": "0.01",
    }
    assert config["residual_training"] == {
        "base_frozen": True,
        "effective_global_batch": 128,
        "epochs": 20,
        "gradient_clip_decimal": "1.0",
        "learning_rate_decimal": "0.0003",
        "optimizer": "AdamW",
        "schedule": "linear-warmup-cosine-decay",
        "warmup_fraction_decimal": "0.05",
        "weight_decay_decimal": "0.01",
    }
    assert config["control_parameter_matching"] == {
        "control_system_ids": ["01", "02", "03", "04", "05", "06", "07"],
        "full_system_id": "08",
        "relative_tolerance_decimal": "0.01",
    }
    assert config["precision_policy"] == {
        "bf16_requires_target_runtime_qualification": True,
        "comparable_systems_share_precision": True,
        "default": "FP32",
    }
    report = execution.public_preflight(raw)
    assert report.ready is False
    assert report.external_receipt_verified is False
    assert len(report.checked_commands) == 11
    assert "prepare-data" in report.checked_commands
    assert "HOLD_SERVER_ENDPOINT_ABSENT" in report.hold_codes
    runner = execution.DataFreeRunner(raw)
    with pytest.raises(execution.ExecutionHold) as caught:
        runner.command_intent("run-base", run_id=experiments.base_run_ids()[0])
    assert caught.value.hold_codes == report.hold_codes


def test_public_artifact_cannot_claim_external_verification() -> None:
    forged = execution.AttemptRecord(
        attempt_id="attempt-0001",
        run_id=experiments.base_run_ids()[0],
        created_at_utc="2026-08-25T20:00:00Z",
        plan_sha256=DIGESTS[0],
        matrix_sha256=DIGESTS[1],
        training_config_sha256=DIGESTS[2],
        source_tree_sha256=DIGESTS[3],
        external_receipt_verified=True,
    )
    with pytest.raises(execution.ExecutionContractError, match="external receipt"):
        execution.canonical_attempt_bytes(forged)


def test_resume_creates_a_new_attempt_and_binds_failed_terminal(
    tmp_path: Path,
) -> None:
    predecessor = execution.AttemptStore.create(tmp_path, _attempt())
    checkpoint_sha = predecessor.write_checkpoint(_checkpoint())
    attempt_sha = execution.artifact_sha256((predecessor.path / "attempt.json").read_bytes())
    terminal = execution.TerminalRecord(
        attempt_id="attempt-0001",
        run_id=experiments.base_run_ids()[0],
        completed_at_utc="2026-08-25T20:03:00Z",
        outcome="FAILED",
        attempt_receipt_sha256=attempt_sha,
        latest_heartbeat_sha256=None,
        latest_checkpoint_receipt_sha256=checkpoint_sha,
        failure_code="INFRA_TRANSIENT",
    )
    terminal_sha = predecessor.write_terminal(terminal)
    new_attempt = execution.AttemptRecord(
        attempt_id="attempt-0002",
        run_id=experiments.base_run_ids()[0],
        created_at_utc="2026-08-25T20:04:00Z",
        plan_sha256=DIGESTS[0],
        matrix_sha256=DIGESTS[1],
        training_config_sha256=DIGESTS[2],
        source_tree_sha256=DIGESTS[3],
    )
    resume = execution.ResumeRecord(
        new_attempt_id="attempt-0002",
        predecessor_attempt_id="attempt-0001",
        run_id=experiments.base_run_ids()[0],
        created_at_utc="2026-08-25T20:04:00Z",
        predecessor_terminal_sha256=terminal_sha,
        checkpoint_receipt_sha256=checkpoint_sha,
        corrective_change_sha256=DIGESTS[12],
        retry_class="INFRA_TRANSIENT",
    )
    resumed = execution.AttemptStore.create_resumed(tmp_path, new_attempt, resume)
    assert resumed.path.name == "attempt-0002"
    assert execution.parse_resume_bytes((resumed.path / "resume.json").read_bytes()) == resume
    assert (predecessor.path / "terminal.json").read_bytes() == execution.canonical_terminal_bytes(
        terminal
    )
    with pytest.raises(FileExistsError):
        execution.AttemptStore.create_resumed(tmp_path, new_attempt, resume)


def test_resume_rejects_scientific_retry_and_wrong_terminal_digest() -> None:
    common = {
        "new_attempt_id": "attempt-0002",
        "predecessor_attempt_id": "attempt-0001",
        "run_id": experiments.base_run_ids()[0],
        "created_at_utc": "2026-08-25T20:04:00Z",
        "predecessor_terminal_sha256": DIGESTS[0],
        "checkpoint_receipt_sha256": DIGESTS[1],
        "corrective_change_sha256": DIGESTS[2],
    }
    with pytest.raises(execution.ExecutionContractError, match="not eligible"):
        execution.canonical_resume_bytes(execution.ResumeRecord(**common, retry_class="SCIENTIFIC"))


def test_periodic_cache_binds_seed_base_terminal_and_qualification() -> None:
    record = execution.PeriodicCacheRecord(
        seed=1729,
        qualified_base_system_id="B1",
        source_run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B1",
        created_at_utc="2026-08-25T22:00:00Z",
        base_terminal_sha256=DIGESTS[0],
        qualification_sha256=DIGESTS[1],
        cache_content_sha256=DIGESTS[2],
    )
    raw = execution.canonical_periodic_cache_bytes(record)
    assert execution.parse_periodic_cache_bytes(raw) == record
    with pytest.raises(execution.ExecutionContractError, match="seed-specific"):
        execution.canonical_periodic_cache_bytes(
            execution.PeriodicCacheRecord(
                seed=1729,
                qualified_base_system_id="B1",
                source_run_id=("phaseset-run-v1/BASE_QUALIFICATION/2718/B1"),
                created_at_utc="2026-08-25T22:00:00Z",
                base_terminal_sha256=DIGESTS[0],
                qualification_sha256=DIGESTS[1],
                cache_content_sha256=DIGESTS[2],
            )
        )


def _synthetic_digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _SyntheticPipelineAdapter:
    """Test-only adapter exercising the real ledger/statistics surfaces."""

    commands = execution.COMMANDS
    adapter_sha256 = _synthetic_digest("phaseset-test-only-synthetic-adapter-v1")
    handler_manifest_sha256 = execution.runtime_handler_manifest_sha256(commands)

    def __init__(self, root: Path) -> None:
        self.root = root / "synthetic-runtime-output"
        self.root.mkdir()
        self.attempt_root = self.root / "attempts"
        self.attempt_root.mkdir()
        self.admission: execution.RuntimeAdmission | None = None
        self.base_weights: np.ndarray | None = None
        self.qualification: experiments.BaseQualification | None = None
        self.failed_attempt: execution.AttemptStore | None = None
        self.failed_checkpoint_sha256: str | None = None
        self.failed_terminal_sha256: str | None = None
        self.bootstrap_result: statistics.BootstrapResult | None = None
        self.captures: tuple[statistics.PairedCapture, ...] | None = None
        self.rendered_paths: tuple[Path, Path] | None = None

    def _write(self, name: str, raw: bytes) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
        return execution.artifact_sha256(raw)

    def _json_artifact(self, name: str, payload: dict[str, object]) -> str:
        raw = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
        return self._write(name, raw)

    def admit(self, request: execution.RuntimeAdmissionRequest) -> execution.RuntimeAdmission:
        self.admission = execution.RuntimeAdmission(
            runtime_mode="SYNTHETIC_DATA_FREE",
            admitted_at_utc="2026-08-25T23:00:00Z",
            adapter_sha256=self.adapter_sha256,
            handler_manifest_sha256=request.handler_manifest_sha256,
            plan_sha256=request.plan_sha256,
            matrix_sha256=request.matrix_sha256,
            training_config_sha256=request.training_config_sha256,
            data_manifest_sha256=_synthetic_digest("synthetic-data-manifest"),
            prepared_data_manifest_sha256=_synthetic_digest("synthetic-prepared-data-manifest"),
            split_audit_sha256=_synthetic_digest("synthetic-split-audit"),
            rights_assertion_sha256=_synthetic_digest("synthetic-rights-not-a-grant"),
            runtime_assertion_sha256=_synthetic_digest("synthetic-cpu-runtime"),
            execution_assertion_sha256=_synthetic_digest("synthetic-test-execution-only"),
            authority=0,
            execution_authorized=True,
            production=False,
            external_authentication_asserted=False,
            public_verification_performed=False,
            result_claimed=False,
            status="SYNTHETIC_DATA_FREE_NO_SCIENTIFIC_RESULT",
        )
        return self.admission

    def _result(
        self,
        intent: execution.CommandIntent,
        outcome: str,
        *artifact_sha256s: str,
    ) -> execution.CommandResult:
        if intent.admission_sha256 is None:
            raise AssertionError("synthetic intent must be admitted")
        return execution.CommandResult(
            command=intent.command,
            outcome=outcome,
            completed_at_utc="2026-08-25T23:59:00Z",
            run_id=intent.run_id,
            seed=intent.seed,
            split=intent.split,
            artifact_sha256s=tuple(artifact_sha256s),
            admission_sha256=intent.admission_sha256,
            runtime_mode=intent.runtime_mode,
            authority=intent.authority,
            production=intent.production,
        )

    def _attempt_record(
        self,
        attempt_id: str,
        run_id: str,
        created_at_utc: str,
        intent: execution.CommandIntent,
    ) -> execution.AttemptRecord:
        if self.admission is None:
            raise AssertionError("synthetic admission must precede attempts")
        return execution.AttemptRecord(
            attempt_id=attempt_id,
            run_id=run_id,
            created_at_utc=created_at_utc,
            plan_sha256=intent.plan_sha256,
            matrix_sha256=self.admission.matrix_sha256,
            training_config_sha256=self.admission.training_config_sha256,
            source_tree_sha256=self.adapter_sha256,
        )

    def _checkpoint(
        self,
        *,
        attempt_id: str,
        run_id: str,
        global_step: int,
        weights: np.ndarray,
        written_at_utc: str,
    ) -> execution.CheckpointRecord:
        return execution.CheckpointRecord(
            attempt_id=attempt_id,
            run_id=run_id,
            epoch_index=0,
            global_step=global_step,
            written_at_utc=written_at_utc,
            model_state_sha256=hashlib.sha256(
                weights.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            optimizer_state_sha256=_synthetic_digest(f"optimizer-{attempt_id}"),
            rng_state_sha256=_synthetic_digest(f"rng-{attempt_id}"),
            sampler_state_sha256=_synthetic_digest(f"sampler-{attempt_id}"),
            dataloader_state_sha256=_synthetic_digest(f"loader-{attempt_id}"),
            dropout_state_sha256=_synthetic_digest(f"dropout-{attempt_id}"),
            validation_state_sha256=_synthetic_digest(f"validation-{attempt_id}"),
            checkpoint_payload_sha256=_synthetic_digest(
                f"checkpoint-payload-{attempt_id}"
            ),
            parent_checkpoint_receipt_sha256=None,
        )

    def _prepare_data(self, intent: execution.CommandIntent) -> execution.CommandResult:
        digest = self._json_artifact(
            "prepared-data.json",
            {
                "notice": "SYNTHETIC / NO SCIENTIFIC RESULT",
                "row_count": 4,
                "schema": "phaseset-test-synthetic-prepared-data-v1",
            },
        )
        return self._result(intent, "COMPLETED", digest)

    def _audit_split(self, intent: execution.CommandIntent) -> execution.CommandResult:
        digest = self._json_artifact(
            "split-audit.json",
            {
                "cross_split_identity_count": 0,
                "notice": "SYNTHETIC / NO SCIENTIFIC RESULT",
                "schema": "phaseset-test-synthetic-split-audit-v1",
            },
        )
        return self._result(intent, "COMPLETED", digest)

    def _run_base(self, intent: execution.CommandIntent) -> execution.CommandResult:
        if intent.run_id is None:
            raise AssertionError("base run identity is required")
        features = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=np.float64,
        )
        targets = np.asarray([1.0, -1.0, 0.0, 1.0], dtype=np.float64)
        weights = np.zeros(2, dtype=np.float64)
        for _ in range(8):
            residual = features @ weights - targets
            gradient = (features.T @ residual) / features.shape[0]
            weights -= 0.2 * gradient
        self.base_weights = weights.copy()

        attempt = self._attempt_record(
            "synthetic-base-0001",
            intent.run_id,
            "2026-08-25T23:01:00Z",
            intent,
        )
        store = execution.AttemptStore.create(self.attempt_root, attempt)
        heartbeat_sha = store.write_heartbeat(
            execution.HeartbeatRecord(
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                sequence=0,
                global_step=8,
                observed_at_utc="2026-08-25T23:01:01Z",
                phase="RUNNING",
                previous_heartbeat_sha256=None,
            )
        )
        checkpoint_sha = store.write_checkpoint(
            self._checkpoint(
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                global_step=8,
                weights=weights,
                written_at_utc="2026-08-25T23:01:02Z",
            )
        )
        terminal_sha = store.write_terminal(
            execution.TerminalRecord(
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                completed_at_utc="2026-08-25T23:01:03Z",
                outcome="SUCCEEDED",
                attempt_receipt_sha256=execution.artifact_sha256(
                    (store.path / "attempt.json").read_bytes()
                ),
                latest_heartbeat_sha256=heartbeat_sha,
                latest_checkpoint_receipt_sha256=checkpoint_sha,
                failure_code=None,
            )
        )
        return self._result(intent, "COMPLETED", checkpoint_sha, terminal_sha)

    def _qualify_base(self, intent: execution.CommandIntent) -> execution.CommandResult:
        score_by_system = {"B0": 60, "B1": 70, "B2": 65}
        parameters = {"B0": 100, "B1": 110, "B2": 120}
        latencies = {"B0": 1_000, "B1": 1_100, "B2": 1_200}
        scores = tuple(
            experiments.BaseScore(
                run_id=run_id,
                bidirectional_r1_numerator=score_by_system[experiments.parse_run_id(run_id)[2]],
                bidirectional_r1_denominator=100,
                parameter_count=parameters[experiments.parse_run_id(run_id)[2]],
                frozen_runtime_latency_ns=latencies[experiments.parse_run_id(run_id)[2]],
                terminal_sha256=_synthetic_digest(f"terminal-{run_id}"),
                selected_checkpoint_sha256=_synthetic_digest(f"checkpoint-{run_id}"),
                split="validation",
                validation_manifest_sha256=_synthetic_digest("base-validation-manifest"),
                query_census_sha256=_synthetic_digest("base-validation-census"),
                evaluator_sha256=_synthetic_digest("base-evaluator"),
                score_artifact_sha256=_synthetic_digest(f"base-score-{run_id}"),
            )
            for run_id in experiments.base_run_ids()
        )
        self.qualification = experiments.qualify_base(scores)
        raw = experiments.canonical_base_qualification_bytes(self.qualification)
        digest = self._write("base-qualification.json", raw)
        return self._result(intent, "COMPLETED", digest)

    def _build_periodic_cache(self, intent: execution.CommandIntent) -> execution.CommandResult:
        if self.qualification is None or intent.seed is None:
            raise AssertionError("qualification and cache seed are required")
        winner = self.qualification.winner_system_id
        source_run_id = f"phaseset-run-v1/BASE_QUALIFICATION/{intent.seed}/{winner}"
        index = experiments.base_run_ids().index(source_run_id)
        record = execution.PeriodicCacheRecord(
            seed=intent.seed,
            qualified_base_system_id=winner,
            source_run_id=source_run_id,
            created_at_utc="2026-08-25T23:02:00Z",
            base_terminal_sha256=self.qualification.completion_sha256s[index],
            qualification_sha256=execution.artifact_sha256(
                experiments.canonical_base_qualification_bytes(self.qualification)
            ),
            cache_content_sha256=_synthetic_digest("synthetic-periodic-cache-content"),
        )
        raw = execution.canonical_periodic_cache_bytes(record)
        digest = self._write("periodic-cache.json", raw)
        return self._result(intent, "COMPLETED", digest)

    def _run_residual(self, intent: execution.CommandIntent) -> execution.CommandResult:
        if intent.run_id is None or self.base_weights is None:
            raise AssertionError("base state and residual run identity are required")
        weights = self.base_weights + np.asarray([0.01, -0.01], dtype=np.float64)
        attempt = self._attempt_record(
            "synthetic-residual-0001",
            intent.run_id,
            "2026-08-25T23:03:00Z",
            intent,
        )
        store = execution.AttemptStore.create(self.attempt_root, attempt)
        checkpoint_sha = store.write_checkpoint(
            self._checkpoint(
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                global_step=9,
                weights=weights,
                written_at_utc="2026-08-25T23:03:01Z",
            )
        )
        terminal_sha = store.write_terminal(
            execution.TerminalRecord(
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                completed_at_utc="2026-08-25T23:03:02Z",
                outcome="FAILED",
                attempt_receipt_sha256=execution.artifact_sha256(
                    (store.path / "attempt.json").read_bytes()
                ),
                latest_heartbeat_sha256=None,
                latest_checkpoint_receipt_sha256=checkpoint_sha,
                failure_code="INFRA_TRANSIENT",
            )
        )
        self.failed_attempt = store
        self.failed_checkpoint_sha256 = checkpoint_sha
        self.failed_terminal_sha256 = terminal_sha
        self.base_weights = weights
        return self._result(intent, "FAILED", checkpoint_sha, terminal_sha)

    def _resume(self, intent: execution.CommandIntent) -> execution.CommandResult:
        if (
            self.failed_attempt is None
            or self.failed_checkpoint_sha256 is None
            or self.failed_terminal_sha256 is None
            or self.base_weights is None
        ):
            raise AssertionError("failed synthetic checkpoint is required")
        run_id = self.failed_attempt.attempt.run_id
        attempt = self._attempt_record(
            "synthetic-residual-0002",
            run_id,
            "2026-08-25T23:04:00Z",
            intent,
        )
        resume = execution.ResumeRecord(
            new_attempt_id=attempt.attempt_id,
            predecessor_attempt_id=self.failed_attempt.attempt.attempt_id,
            run_id=run_id,
            created_at_utc=attempt.created_at_utc,
            predecessor_terminal_sha256=self.failed_terminal_sha256,
            checkpoint_receipt_sha256=self.failed_checkpoint_sha256,
            corrective_change_sha256=_synthetic_digest("synthetic-retry-fix"),
            retry_class="INFRA_TRANSIENT",
        )
        store = execution.AttemptStore.create_resumed(
            self.attempt_root,
            attempt,
            resume,
        )
        resumed_weights = self.base_weights + np.asarray([0.005, 0.005])
        checkpoint_sha = store.write_checkpoint(
            self._checkpoint(
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                global_step=10,
                weights=resumed_weights,
                written_at_utc="2026-08-25T23:04:01Z",
            )
        )
        terminal_sha = store.write_terminal(
            execution.TerminalRecord(
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                completed_at_utc="2026-08-25T23:04:02Z",
                outcome="SUCCEEDED",
                attempt_receipt_sha256=execution.artifact_sha256(
                    (store.path / "attempt.json").read_bytes()
                ),
                latest_heartbeat_sha256=None,
                latest_checkpoint_receipt_sha256=checkpoint_sha,
                failure_code=None,
            )
        )
        return self._result(intent, "COMPLETED", checkpoint_sha, terminal_sha)

    def _evaluate(self, intent: execution.CommandIntent) -> execution.CommandResult:
        motion = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        text = motion.copy()
        motion /= np.linalg.norm(motion, axis=1, keepdims=True)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        similarities = motion @ text.T
        recall_at_1 = int(np.equal(np.argmax(similarities, axis=1), np.arange(3)).sum())
        digest = self._json_artifact(
            "gallery-evaluation.json",
            {
                "denominator": 3,
                "notice": "SYNTHETIC / NO SCIENTIFIC RESULT",
                "recall_at_1_numerator": recall_at_1,
                "schema": "phaseset-test-synthetic-gallery-v1",
                "split": intent.split,
            },
        )
        self.captures = (
            _paired_capture("synthetic-a", (1,), (0,)),
            _paired_capture("synthetic-b", (1, 0), (0, 0)),
            _paired_capture("synthetic-c", (1, 1, 0), (1, 0, 0), (0,), (0,)),
        )
        return self._result(intent, "COMPLETED", digest)

    def _bootstrap(self, intent: execution.CommandIntent) -> execution.CommandResult:
        if self.captures is None:
            raise AssertionError("synthetic gallery must precede bootstrap")
        self.bootstrap_result = statistics.paired_capture_bootstrap(
            self.captures,
            seed=20260825,
        )
        raw = statistics.canonical_bootstrap_result_bytes(self.bootstrap_result)
        digest = self._write("bootstrap.json", raw)
        return self._result(intent, "COMPLETED", digest)

    def _render_paper(self, intent: execution.CommandIntent) -> execution.CommandResult:
        if self.bootstrap_result is None:
            raise AssertionError("synthetic bootstrap must precede rendering")
        bootstrap_sha = execution.artifact_sha256(
            statistics.canonical_bootstrap_result_bytes(self.bootstrap_result)
        )
        hypotheses = tuple(
            statistics.HypothesisResult(
                hypothesis_id=f"H{index}",
                effect=index / 100,
                raw_p_value=index / 20,
                bootstrap_sha256=bootstrap_sha,
            )
            for index in range(1, 9)
        )
        report = statistics.build_statistical_report(hypotheses)
        report_sha = self._write(
            "synthetic-statistical-report.json",
            statistics.canonical_statistical_report_bytes(report),
        )
        marker = "SYNTHETIC / NO SCIENTIFIC RESULT"
        table_path = self.root / "SYNTHETIC_TABLE.txt"
        paper_path = self.root / "SYNTHETIC_PAPER.txt"
        table_raw = f"{marker}\nH1-H8 synthetic fixture table\n".encode("ascii")
        paper_raw = (f"{marker}\nThis fixture cannot support a publication claim.\n").encode(
            "ascii"
        )
        table_sha = self._write(table_path.name, table_raw)
        paper_sha = self._write(paper_path.name, paper_raw)
        self.rendered_paths = (table_path, paper_path)
        return self._result(intent, "COMPLETED", table_sha, paper_sha, report_sha)

    def handle(self, intent: execution.CommandIntent) -> execution.CommandResult:
        handlers = {
            "prepare-data": self._prepare_data,
            "audit-split": self._audit_split,
            "run-base": self._run_base,
            "qualify-base": self._qualify_base,
            "build-periodic-cache": self._build_periodic_cache,
            "run-residual": self._run_residual,
            "resume": self._resume,
            "evaluate": self._evaluate,
            "bootstrap": self._bootstrap,
            "render-paper": self._render_paper,
        }
        if intent.command not in handlers:
            raise execution.ExecutionContractError(
                "preflight is handled by adapter admission, not dispatch"
            )
        return handlers[intent.command](intent)


def test_synthetic_adapter_runs_tiny_failure_resume_gallery_and_render(
    tmp_path: Path,
) -> None:
    registered_paper = ROOT / "paper" / "main.tex"
    registered_paper_before = registered_paper.read_bytes() if registered_paper.is_file() else None
    raw = execution.load_training_config(ROOT / "configs" / "phaseset" / "training.json")
    adapter = _SyntheticPipelineAdapter(tmp_path)
    runner = execution.DataFreeRunner(raw, runtime_adapter=adapter)

    admission = runner.preflight()
    assert type(admission) is execution.RuntimeAdmission
    assert admission.ready is True
    assert admission.hold_codes == ()
    assert admission.external_receipt_verified is False
    admission_raw = execution.canonical_runtime_admission_bytes(admission)
    assert execution.parse_runtime_admission_bytes(admission_raw) == admission
    assert b"SYNTHETIC / NO SCIENTIFIC RESULT" in admission_raw
    assert b'"public_verification_performed":false' in admission_raw

    results = [
        runner.execute_command("prepare-data"),
        runner.execute_command("audit-split"),
        runner.execute_command("run-base", run_id=experiments.base_run_ids()[0]),
        runner.execute_command("qualify-base"),
        runner.execute_command("build-periodic-cache", seed=1729),
    ]
    failed = runner.execute_command(
        "run-residual",
        run_id=experiments.residual_run_ids()[0],
    )
    assert failed.outcome == "FAILED"
    results.extend(
        (
            failed,
            runner.execute_command("resume"),
            runner.execute_command("evaluate", split="validation"),
            runner.execute_command("bootstrap"),
            runner.execute_command("render-paper"),
        )
    )

    assert len(results) == 10
    assert all(result.runtime_mode == "SYNTHETIC_DATA_FREE" for result in results)
    assert all(result.result_claimed is False for result in results)
    assert all(
        execution.parse_command_result_bytes(execution.canonical_command_result_bytes(result))
        == result
        for result in results
    )
    assert adapter.bootstrap_result is not None
    assert adapter.bootstrap_result.draws == 100_000
    assert adapter.rendered_paths is not None
    for path in adapter.rendered_paths:
        assert path.is_file()
        assert b"SYNTHETIC / NO SCIENTIFIC RESULT" in path.read_bytes()
    assert not (adapter.root / "paper" / "main.tex").exists()
    if registered_paper_before is None:
        assert not registered_paper.exists()
    else:
        assert registered_paper.read_bytes() == registered_paper_before


def test_private_admission_is_an_assertion_not_public_verification() -> None:
    record = execution.RuntimeAdmission(
        runtime_mode="PRIVATE_AUTHORIZED",
        admitted_at_utc="2026-08-25T23:00:00Z",
        adapter_sha256=DIGESTS[0],
        handler_manifest_sha256=DIGESTS[1],
        plan_sha256=DIGESTS[2],
        matrix_sha256=DIGESTS[3],
        training_config_sha256=DIGESTS[4],
        data_manifest_sha256=DIGESTS[5],
        prepared_data_manifest_sha256=DIGESTS[6],
        split_audit_sha256=DIGESTS[7],
        rights_assertion_sha256=DIGESTS[8],
        runtime_assertion_sha256=DIGESTS[9],
        execution_assertion_sha256=DIGESTS[10],
        authority=1,
        execution_authorized=True,
        production=True,
        external_authentication_asserted=True,
        public_verification_performed=False,
        result_claimed=False,
        status="PRIVATE_ADAPTER_ASSERTED_AUTHORIZATION_NOT_PUBLICLY_VERIFIED",
    )
    raw = execution.canonical_runtime_admission_bytes(record)
    assert b"NOT PUBLICLY VERIFIED" in raw
    assert b'"public_verification_performed":false' in raw
    with pytest.raises(execution.ExecutionContractError, match="cannot claim"):
        execution.canonical_runtime_admission_bytes(
            replace(record, public_verification_performed=True)
        )


def test_runtime_adapter_requires_exact_command_census_and_digest_bindings(
    tmp_path: Path,
) -> None:
    raw = execution.load_training_config(ROOT / "configs" / "phaseset" / "training.json")

    incomplete_root = tmp_path / "incomplete"
    incomplete_root.mkdir()
    incomplete = _SyntheticPipelineAdapter(incomplete_root)
    incomplete.commands = execution.COMMANDS[:-1]
    with pytest.raises(execution.ExecutionContractError, match="eleven-command"):
        execution.DataFreeRunner(raw, runtime_adapter=incomplete)

    misbound_root = tmp_path / "misbound"
    misbound_root.mkdir()
    misbound = _SyntheticPipelineAdapter(misbound_root)
    original_admit = misbound.admit

    def admit_with_wrong_plan(
        request: execution.RuntimeAdmissionRequest,
    ) -> execution.RuntimeAdmission:
        return replace(
            original_admit(request),
            plan_sha256=_synthetic_digest("wrong-plan"),
        )

    misbound.admit = admit_with_wrong_plan  # type: ignore[method-assign]
    runner = execution.DataFreeRunner(raw, runtime_adapter=misbound)
    with pytest.raises(execution.ExecutionContractError, match="do not match"):
        runner.preflight()

    with pytest.raises(execution.ExecutionContractError, match="outside B0-B2"):
        execution.canonical_periodic_cache_bytes(
            execution.PeriodicCacheRecord(
                seed=1729,
                qualified_base_system_id="00",
                source_run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/00",
                created_at_utc="2026-08-25T22:00:00Z",
                base_terminal_sha256=DIGESTS[0],
                qualification_sha256=DIGESTS[1],
                cache_content_sha256=DIGESTS[2],
            )
        )
