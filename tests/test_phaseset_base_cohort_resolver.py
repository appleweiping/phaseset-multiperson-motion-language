from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from phaseset_core import base_cohort_resolver as resolver
from phaseset_core import execution
from phaseset_core import training
from phaseset_core.experiments import (
    base_run_ids,
    build_experiment_plan,
    experiment_plan_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _admission() -> resolver.BaseCohortAdmission:
    plan = build_experiment_plan()
    return resolver.BaseCohortAdmission(
        plan_sha256=experiment_plan_sha256(plan),
        matrix_sha256=plan.matrix_sha256,
        training_config_sha256=_sha("host-config"),
        source_tree_sha256=_sha("source"),
        train_manifest_sha256=_sha("train"),
        val_manifest_sha256=_sha("val"),
        device="cpu",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32_768,
        checkpoint_every_updates=1,
    )


def _registry() -> tuple[resolver.BaseAttemptRegistryRow, ...]:
    return tuple(
        resolver.BaseAttemptRegistryRow(
            attempt_id=f"attempt-{index}",
            run_id=run_id,
            terminal_outcome="SUCCEEDED",
            terminal_sha256=_sha(f"terminal-{index}"),
        )
        for index, run_id in enumerate(base_run_ids())
    )


def _dummy_resolved(run_id: str, row: resolver.BaseAttemptRegistryRow) -> resolver.ResolvedBaseRun:
    _role, seed, system_id = run_id.split("/")[1:]
    return resolver.ResolvedBaseRun(
        run_id=run_id,
        attempt_id=row.attempt_id,
        system_id=system_id,
        seed=int(seed),
        terminal_sha256=row.terminal_sha256,
        latest_checkpoint_sha256=_sha(f"latest-{run_id}"),
        latest_checkpoint_name="checkpoint-000000000075-000150-validation.pt",
        selected_checkpoint=resolver.SelectedBaseCheckpoint(
            name="checkpoint-000000000001-000002-validation.pt",
            sha256=_sha(f"best-{run_id}"),
            raw=f"best-{run_id}".encode("ascii"),
            epoch=1,
            global_step=1,
            state_digest=_sha(f"state-{run_id}"),
        ),
        failed_predecessors=(),
    )


def test_resolver_restores_exact_registered_order_without_claiming_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = tuple(reversed(_registry()))

    def fake_resolve(
        _root: Path,
        run_id: str,
        history: tuple[resolver.BaseAttemptRegistryRow, ...],
        _admitted: resolver.BaseCohortAdmission,
        *,
        selected_maximum_bytes: int,
    ) -> resolver.ResolvedBaseRun:
        assert len(history) == 1
        assert selected_maximum_bytes > 0
        return _dummy_resolved(run_id, history[0])

    monkeypatch.setattr(resolver, "_resolve_registered_run", fake_resolve)
    cohort = resolver.resolve_completed_base_cohort(
        tmp_path,
        registry_rows=registry,
        admission=_admission(),
    )
    assert tuple(row.run_id for row in cohort.rows) == base_run_ids()
    assert cohort.authority == 0
    assert cohort.production is False
    assert cohort.result_claimed is False
    assert "NO_SCORE_NO_QUALIFICATION" in cohort.status


@pytest.mark.parametrize("outcome", ["FAILED", "HELD"])
def test_registry_requires_one_success_per_registered_row(outcome: str) -> None:
    rows = list(_registry())
    rows[0] = replace(rows[0], terminal_outcome=outcome)
    with pytest.raises(resolver.BaseCohortResolutionError, match="exactly nine"):
        resolver._checked_registry_rows(tuple(rows))


def test_registry_rejects_duplicate_success_and_unregistered_rows() -> None:
    rows = _registry()
    duplicate = replace(rows[0], attempt_id="another-attempt")
    with pytest.raises(resolver.BaseCohortResolutionError, match="exactly nine"):
        resolver._checked_registry_rows(rows + (duplicate,))
    foreign = replace(
        rows[0],
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B9",
        terminal_outcome="FAILED",
    )
    with pytest.raises(resolver.BaseCohortResolutionError, match="unregistered"):
        resolver._checked_registry_rows((foreign,) + rows[1:])


def test_registry_rejects_history_before_chain_verification_can_exceed_bound() -> None:
    successes = _registry()
    run_id = successes[0].run_id
    failed = tuple(
        resolver.BaseAttemptRegistryRow(
            f"failed-{index}",
            run_id,
            "FAILED",
            _sha(f"failed-terminal-{index}"),
        )
        for index in range(resolver.MAX_ATTEMPTS_PER_RUN)
    )
    with pytest.raises(resolver.BaseCohortResolutionError, match="closed attempt bound"):
        resolver._checked_registry_rows(failed + successes)


def test_success_must_be_last_registry_history(tmp_path: Path) -> None:
    run_id = base_run_ids()[0]
    rows = (
        resolver.BaseAttemptRegistryRow("success", run_id, "SUCCEEDED", _sha("success")),
        resolver.BaseAttemptRegistryRow("later-failure", run_id, "FAILED", _sha("failure")),
    )
    with pytest.raises(resolver.BaseCohortResolutionError, match="close its run history"):
        resolver._resolve_registered_run(tmp_path, run_id, rows, _admission())


def _checkpoint_record(
    attempt_id: str,
    run_id: str,
    *,
    payload_sha256: str,
    parent: str | None = None,
) -> execution.CheckpointRecord:
    return execution.CheckpointRecord(
        attempt_id=attempt_id,
        run_id=run_id,
        epoch_index=1,
        global_step=1,
        written_at_utc="2026-09-08T12:00:01Z",
        model_state_sha256=_sha(f"{attempt_id}-model"),
        optimizer_state_sha256=_sha(f"{attempt_id}-optimizer"),
        rng_state_sha256=_sha(f"{attempt_id}-rng"),
        sampler_state_sha256=_sha(f"{attempt_id}-sampler"),
        dataloader_state_sha256=_sha(f"{attempt_id}-loader"),
        dropout_state_sha256=_sha(f"{attempt_id}-dropout"),
        validation_state_sha256=_sha(f"{attempt_id}-validation"),
        checkpoint_payload_sha256=payload_sha256,
        parent_checkpoint_receipt_sha256=parent,
    )


def _attempt(
    attempt_id: str,
    run_id: str,
    admission: resolver.BaseCohortAdmission,
    *,
    created_at: str,
) -> execution.AttemptRecord:
    return execution.AttemptRecord(
        attempt_id=attempt_id,
        run_id=run_id,
        created_at_utc=created_at,
        plan_sha256=admission.plan_sha256,
        matrix_sha256=admission.matrix_sha256,
        training_config_sha256=admission.training_config_sha256,
        source_tree_sha256=admission.source_tree_sha256,
    )


def test_failed_and_held_predecessors_are_retained_in_resume_order(tmp_path: Path) -> None:
    admission = _admission()
    run_id = base_run_ids()[0]
    failed_id = "attempt-failed"
    held_id = "attempt-held"
    success_id = "attempt-success"

    failed_attempt = _attempt(
        failed_id, run_id, admission, created_at="2026-09-08T12:00:00Z"
    )
    failed_attempt_raw = execution.canonical_attempt_bytes(failed_attempt)
    failed_checkpoint_sha = _sha("failed-checkpoint-receipt")
    failed_terminal = execution.TerminalRecord(
        attempt_id=failed_id,
        run_id=run_id,
        completed_at_utc="2026-09-08T12:01:00Z",
        outcome="FAILED",
        attempt_receipt_sha256=hashlib.sha256(failed_attempt_raw).hexdigest(),
        latest_heartbeat_sha256=None,
        latest_checkpoint_receipt_sha256=failed_checkpoint_sha,
        failure_code="HOST_EXECUTION_FAILED",
    )
    failed_terminal_raw = execution.canonical_terminal_bytes(failed_terminal)

    held_attempt = _attempt(
        held_id, run_id, admission, created_at="2026-09-08T12:02:00Z"
    )
    held_attempt_raw = execution.canonical_attempt_bytes(held_attempt)
    held_checkpoint_sha = _sha("held-checkpoint-receipt")
    held_resume = execution.ResumeRecord(
        new_attempt_id=held_id,
        predecessor_attempt_id=failed_id,
        run_id=run_id,
        created_at_utc=held_attempt.created_at_utc,
        predecessor_terminal_sha256=hashlib.sha256(failed_terminal_raw).hexdigest(),
        checkpoint_receipt_sha256=failed_checkpoint_sha,
        corrective_change_sha256=_sha("first-correction"),
        retry_class="IMPLEMENTATION",
    )
    held_terminal = execution.TerminalRecord(
        attempt_id=held_id,
        run_id=run_id,
        completed_at_utc="2026-09-08T12:03:00Z",
        outcome="HELD",
        attempt_receipt_sha256=hashlib.sha256(held_attempt_raw).hexdigest(),
        latest_heartbeat_sha256=None,
        latest_checkpoint_receipt_sha256=held_checkpoint_sha,
        failure_code="RESOURCE_UNAVAILABLE",
    )
    held_terminal_raw = execution.canonical_terminal_bytes(held_terminal)

    for attempt_id, attempt_raw, terminal_raw, resume in (
        (failed_id, failed_attempt_raw, failed_terminal_raw, None),
        (held_id, held_attempt_raw, held_terminal_raw, held_resume),
    ):
        directory = tmp_path / attempt_id
        directory.mkdir()
        (directory / "attempt.json").write_bytes(attempt_raw)
        (directory / "terminal.json").write_bytes(terminal_raw)
        if resume is not None:
            (directory / "resume.json").write_bytes(execution.canonical_resume_bytes(resume))

    success_attempt = _attempt(
        success_id, run_id, admission, created_at="2026-09-08T12:04:00Z"
    )
    success_attempt_raw = execution.canonical_attempt_bytes(success_attempt)
    success_resume = execution.ResumeRecord(
        new_attempt_id=success_id,
        predecessor_attempt_id=held_id,
        run_id=run_id,
        created_at_utc=success_attempt.created_at_utc,
        predecessor_terminal_sha256=hashlib.sha256(held_terminal_raw).hexdigest(),
        checkpoint_receipt_sha256=held_checkpoint_sha,
        corrective_change_sha256=_sha("second-correction"),
        retry_class="RESOURCE",
    )
    latest = _checkpoint_record(
        success_id,
        run_id,
        payload_sha256=_sha("success-payload"),
    )
    latest_raw = execution.canonical_checkpoint_bytes(latest)
    success_terminal = execution.TerminalRecord(
        attempt_id=success_id,
        run_id=run_id,
        completed_at_utc="2026-09-08T12:05:00Z",
        outcome="SUCCEEDED",
        attempt_receipt_sha256=hashlib.sha256(success_attempt_raw).hexdigest(),
        latest_heartbeat_sha256=None,
        latest_checkpoint_receipt_sha256=hashlib.sha256(latest_raw).hexdigest(),
        failure_code=None,
    )
    evidence = execution.CompletedAttemptEvidence(
        attempt=success_attempt,
        attempt_raw=success_attempt_raw,
        latest_checkpoint=latest,
        latest_checkpoint_raw=latest_raw,
        terminal=success_terminal,
        terminal_raw=execution.canonical_terminal_bytes(success_terminal),
        resume=success_resume,
        resume_raw=execution.canonical_resume_bytes(success_resume),
    )

    observed = resolver._owned_failed_predecessors(
        tmp_path,
        evidence,
        admission=admission,
    )
    assert tuple(row.attempt_id for row in observed) == (failed_id, held_id)
    assert tuple(row.terminal_outcome for row in observed) == ("FAILED", "HELD")
    assert tuple(row.failure_code for row in observed) == (
        "HOST_EXECUTION_FAILED",
        "RESOURCE_UNAVAILABLE",
    )


def test_owned_reader_rejects_symlink_bound_and_digest_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"payload")
    with pytest.raises(resolver.BaseCohortResolutionError, match="byte bound"):
        resolver._read_owned_file(
            artifact,
            label="artifact",
            maximum_bytes=2,
        )
    with pytest.raises(resolver.BaseCohortResolutionError, match="digest mismatch"):
        resolver._read_owned_file(
            artifact,
            label="artifact",
            maximum_bytes=16,
            expected_sha256=_sha("wrong"),
        )
    link = tmp_path / "link.pt"
    try:
        link.symlink_to(artifact)
    except OSError:
        pytest.skip("test account cannot create a symlink")
    with pytest.raises(resolver.BaseCohortResolutionError, match="safely|symlink"):
        resolver._read_owned_file(link, label="artifact", maximum_bytes=16)


def test_checkpoint_directory_census_rejects_before_unbounded_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    class FakeEntries:
        def __enter__(self) -> FakeEntries:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> object:
            for index in range(4):
                name = f"unrelated-{index}"
                observed.append(name)
                yield SimpleNamespace(name=name)

    monkeypatch.setattr(resolver, "MAX_CHECKPOINT_DIRECTORY_ENTRIES", 2)
    monkeypatch.setattr(resolver.os, "scandir", lambda _path: FakeEntries())
    with pytest.raises(resolver.BaseCohortResolutionError, match="closed entry bound"):
        resolver._latest_checkpoint_file(tmp_path, global_step=75)
    assert observed == ["unrelated-0", "unrelated-1", "unrelated-2"]


def test_checkpoint_decoder_uses_restricted_owned_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_load(stream: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        assert hasattr(stream, "read")
        return {}

    monkeypatch.setattr(resolver.torch, "load", fake_load)
    with pytest.raises(resolver.BaseCohortResolutionError, match="keys are not closed"):
        resolver._decode_checkpoint(b"not-a-real-checkpoint", "checkpoint")
    assert observed == {"map_location": "cpu", "weights_only": True}


def _checkpoint_payload(
    *,
    admission: resolver.BaseCohortAdmission,
    run_id: str,
    name: str,
    epoch: int,
    global_step: int,
    sequence: int,
    best_name: str,
    best_sha256: str | None,
) -> dict[str, object]:
    _role, seed_text, system_id = run_id.split("/")[1:]
    config = training.TrainingConfig(
        stage="base",
        seed=int(seed_text),
        device=admission.device,
        request_bf16=admission.request_bf16,
        bf16_runtime_qualified=admission.bf16_runtime_qualified,
        edge_budget=admission.edge_budget,
        checkpoint_every_updates=admission.checkpoint_every_updates,
        synthetic_contract=False,
    )
    payload: dict[str, object] = {
        "authority": 0,
        "behavior_sha256": _sha("behavior"),
        "best_checkpoint_name": best_name,
        "best_checkpoint_sha256": best_sha256,
        "best_validation_metric": 0.5,
        "checkpoint_sequence": sequence,
        "code_artifact_sha256": _sha("code"),
        "config_sha256": config.sha256,
        "environment_sha256": _sha("environment"),
        "epoch": epoch,
        "external_receipt_verified": False,
        "factory_sha256": _sha("factory"),
        "frozen_base_checkpoint_sha256": None,
        "frozen_base_state_sha256": None,
        "global_step": global_step,
        "initial_optimizable_state_sha256": _sha("initial"),
        "initialization_binding_sha256": _sha("binding"),
        "last_train_loss": 0.25,
        "max_microbatch_edges": 1,
        "model": {"weight": torch.zeros(1)},
        "optimizer": {
            "state": {0: {"step": global_step}},
            "param_groups": [],
        },
        "precision_mode": training.resolve_precision(config).mode,
        "production": False,
        "qualified_base_selection_sha256": None,
        "residual_capacity_audit_sha256": None,
        "result_claimed": False,
        "rng": {},
        "scheduler": {"last_epoch": global_step},
        "schema": training.CHECKPOINT_SCHEMA,
        "seed": int(seed_text),
        "stage": "base",
        "system_id": system_id,
        "total_steps": training.BASE_EPOCHS,
        "train_edges_seen": max(1, global_step),
        "train_manifest_sha256": admission.train_manifest_sha256,
        "update_index": 0,
        "updates_per_epoch": 1,
        "val_manifest_sha256": admission.val_manifest_sha256,
        "validation_edges_seen": max(1, epoch),
    }
    payload["state_digest"] = training._stable_hash(payload)
    assert resolver._CHECKPOINT_NAME.fullmatch(name) is not None
    return payload


def test_latest_and_selected_checkpoint_semantics_remain_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission()
    run_id = base_run_ids()[0]
    latest_name = f"checkpoint-{training.BASE_EPOCHS:012d}-000150-validation.pt"
    best_name = "checkpoint-000000000001-000002-validation.pt"
    latest_raw = b"latest-owned-bytes"
    best_raw = b"selected-owned-bytes"
    latest_sha = hashlib.sha256(latest_raw).hexdigest()
    best_sha = hashlib.sha256(best_raw).hexdigest()
    latest_payload = _checkpoint_payload(
        admission=admission,
        run_id=run_id,
        name=latest_name,
        epoch=training.BASE_EPOCHS,
        global_step=training.BASE_EPOCHS,
        sequence=150,
        best_name=best_name,
        best_sha256=best_sha,
    )
    selected_payload = _checkpoint_payload(
        admission=admission,
        run_id=run_id,
        name=best_name,
        epoch=1,
        global_step=1,
        sequence=2,
        best_name=best_name,
        best_sha256=None,
    )
    payloads = {latest_raw: latest_payload, best_raw: selected_payload}
    monkeypatch.setattr(resolver, "_decode_checkpoint", lambda raw, _label: payloads[raw])
    monkeypatch.setattr(resolver, "_validate_model_and_optimizer", lambda *_args, **_kwargs: None)

    latest = resolver._validate_checkpoint_payload(
        latest_raw,
        sha256=latest_sha,
        name=latest_name,
        run_id=run_id,
        admission=admission,
        ledger_record=None,
        require_completed_latest=True,
        require_selected_best=False,
    )
    selected = resolver._validate_checkpoint_payload(
        best_raw,
        sha256=best_sha,
        name=best_name,
        run_id=run_id,
        admission=admission,
        ledger_record=None,
        require_completed_latest=False,
        require_selected_best=True,
    )
    assert latest.sha256 != selected.sha256
    assert latest.best_name == selected.name
    assert latest.best_sha256 == selected.sha256
    assert selected.best_sha256 == selected.sha256
    resolver._validate_latest_selected_link(latest, selected)

    with pytest.raises(resolver.BaseCohortResolutionError, match="metric mismatch"):
        resolver._validate_latest_selected_link(
            latest,
            replace(selected, best_validation_metric=0.25),
        )
    with pytest.raises(resolver.BaseCohortResolutionError, match="schedule mismatch"):
        resolver._validate_latest_selected_link(
            latest,
            replace(
                selected,
                updates_per_epoch=2,
                total_steps=2 * training.BASE_EPOCHS,
                global_step=2,
            ),
        )
    with pytest.raises(resolver.BaseCohortResolutionError, match="counters exceed"):
        resolver._validate_latest_selected_link(
            latest,
            replace(selected, validation_edges_seen=latest.validation_edges_seen + 1),
        )


def test_selected_validation_checkpoint_must_be_at_epoch_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission()
    run_id = base_run_ids()[0]
    name = "checkpoint-000000000002-000002-validation.pt"
    raw = b"selected-with-update-cursor"
    digest = hashlib.sha256(raw).hexdigest()
    payload = _checkpoint_payload(
        admission=admission,
        run_id=run_id,
        name=name,
        epoch=1,
        global_step=2,
        sequence=2,
        best_name=name,
        best_sha256=None,
    )
    payload["update_index"] = 1
    without_digest = dict(payload)
    without_digest.pop("state_digest")
    payload["state_digest"] = training._stable_hash(without_digest)
    monkeypatch.setattr(resolver, "_decode_checkpoint", lambda *_args: payload)
    monkeypatch.setattr(
        resolver,
        "_validate_model_and_optimizer",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(resolver.BaseCohortResolutionError, match="best-at-write"):
        resolver._validate_checkpoint_payload(
            raw,
            sha256=digest,
            name=name,
            run_id=run_id,
            admission=admission,
            ledger_record=None,
            require_completed_latest=False,
            require_selected_best=True,
        )


def test_checkpoint_manifest_and_exact_header_cannot_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission()
    run_id = base_run_ids()[0]
    name = "checkpoint-000000000075-000150-validation.pt"
    raw = b"latest"
    digest = hashlib.sha256(raw).hexdigest()
    payload = _checkpoint_payload(
        admission=admission,
        run_id=run_id,
        name=name,
        epoch=training.BASE_EPOCHS,
        global_step=training.BASE_EPOCHS,
        sequence=150,
        best_name=name,
        best_sha256=None,
    )
    payload["train_manifest_sha256"] = _sha("substituted-train")
    without_digest = dict(payload)
    without_digest.pop("state_digest")
    payload["state_digest"] = training._stable_hash(without_digest)
    monkeypatch.setattr(resolver, "_decode_checkpoint", lambda *_args: payload)
    monkeypatch.setattr(resolver, "_validate_model_and_optimizer", lambda *_args, **_kwargs: None)
    with pytest.raises(resolver.BaseCohortResolutionError, match="train_manifest"):
        resolver._validate_checkpoint_payload(
            raw,
            sha256=digest,
            name=name,
            run_id=run_id,
            admission=admission,
            ledger_record=None,
            require_completed_latest=True,
            require_selected_best=False,
        )
    payload["train_manifest_sha256"] = admission.train_manifest_sha256
    payload["authority"] = False
    without_digest = dict(payload)
    without_digest.pop("state_digest")
    payload["state_digest"] = training._stable_hash(without_digest)
    with pytest.raises(resolver.BaseCohortResolutionError, match="authority"):
        resolver._validate_checkpoint_payload(
            raw,
            sha256=digest,
            name=name,
            run_id=run_id,
            admission=admission,
            ledger_record=None,
            require_completed_latest=True,
            require_selected_best=False,
        )


def test_ledger_state_digests_are_recomputed_from_runtime_payload() -> None:
    run_id = base_run_ids()[0]
    payload = {
        "epoch": 1,
        "global_step": 1,
        "update_index": 0,
        "train_manifest_sha256": _sha("train"),
        "val_manifest_sha256": _sha("val"),
        "best_validation_metric": 0.5,
        "best_checkpoint_name": "checkpoint-000000000001-000002-validation.pt",
        "best_checkpoint_sha256": None,
        "model": {"weight": torch.zeros(1)},
        "optimizer": {"state": {}},
        "rng": {"torch_cpu": torch.zeros(1, dtype=torch.uint8)},
    }
    record = _checkpoint_record("attempt-ledger", run_id, payload_sha256=_sha("payload"))
    with pytest.raises(resolver.BaseCohortResolutionError, match="model_state"):
        resolver._validate_ledger_payload_binding(payload, record)
