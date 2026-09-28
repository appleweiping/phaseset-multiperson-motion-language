"""NO_RESULT CPU integration tests for the checkpoint-observer seam."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from phaseset_core import cli, host
from phaseset_core.execution import AttemptRecord, AttemptStore, parse_checkpoint_bytes
from phaseset_core.training import PhaseSetTrainingRuntime, TrainingConfig
from test_phaseset_training import (
    _assert_state_equal,
    _base_system,
    _source,
    _state_snapshot,
    _training_batch,
)


PUBLIC_ROOT = Path(cli.__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> TrainingConfig:
    return TrainingConfig(
        stage="base",
        seed=1729,
        effective_global_batch=1,
        edge_budget=1,
        checkpoint_every_updates=1,
        synthetic_contract=True,
    )


def _sources(label: str):
    return (
        _source(_training_batch(split="train"), f"{label}-train"),
        _source(_training_batch(split="val"), f"{label}-val"),
    )


def test_real_fit_observer_sees_durable_digest_correct_update_and_validation(
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, str, str, str]] = []

    def observer(artifact, reason: str) -> None:
        assert artifact.path.is_file()
        observed.append(
            (reason, artifact.sha256, _sha256(artifact.path), artifact.path.name)
        )

    train, val = _sources("events")
    report = PhaseSetTrainingRuntime(
        _base_system(initialization_seed=91, dropout=0.2),
        _config(),
        tmp_path / "events",
        checkpoint_observer=observer,
    ).fit(train, val, stop_after_global_step=1)

    assert [row[0] for row in observed] == ["update", "validation"]
    assert all(declared == actual for _, declared, actual, _ in observed)
    assert observed[0][3].endswith("-update.pt")
    assert observed[1][3].endswith("-validation.pt")
    assert report.latest_checkpoint.sha256 == observed[-1][1]
    assert report.authority == 0
    assert report.production is False
    assert report.result_claimed is False


def test_observer_error_propagates_after_digest_valid_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    observed = []

    class ObserverFailure(RuntimeError):
        pass

    def observer(artifact, reason: str) -> None:
        observed.append((artifact, reason))
        raise ObserverFailure("injected observer failure")

    train, val = _sources("observer-error")
    with pytest.raises(ObserverFailure, match="injected observer failure"):
        PhaseSetTrainingRuntime(
            _base_system(initialization_seed=101, dropout=0.2),
            _config(),
            tmp_path / "observer-error",
            checkpoint_observer=observer,
        ).fit(train, val, stop_after_global_step=1)

    assert len(observed) == 1
    artifact, reason = observed[0]
    assert reason == "update"
    assert artifact.path.is_file()
    assert _sha256(artifact.path) == artifact.sha256

    resume_train, resume_val = _sources("observer-error")
    resumed = PhaseSetTrainingRuntime(
        _base_system(initialization_seed=999, dropout=0.2),
        _config(),
        tmp_path / "observer-error-resume",
    ).fit(
        resume_train,
        resume_val,
        resume_checkpoint=artifact.path,
        stop_after_global_step=artifact.global_step,
    )
    assert resumed.latest_checkpoint.sha256 == artifact.sha256
    assert resumed.authority == 0
    assert resumed.result_claimed is False


def test_attached_observer_preserves_baseline_model_and_checkpoint_state(
    tmp_path: Path,
) -> None:
    baseline_system = _base_system(initialization_seed=777, dropout=0.25)
    baseline_train, baseline_val = _sources("equivalence")
    baseline_report = PhaseSetTrainingRuntime(
        baseline_system,
        _config(),
        tmp_path / "baseline",
    ).fit(baseline_train, baseline_val, stop_after_global_step=1)

    observed_system = _base_system(initialization_seed=777, dropout=0.25)
    observed_events: list[tuple[str, str]] = []

    def observer(artifact, reason: str) -> None:
        observed_events.append((reason, _sha256(artifact.path)))

    observed_train, observed_val = _sources("equivalence")
    observed_report = PhaseSetTrainingRuntime(
        observed_system,
        _config(),
        tmp_path / "observed",
        checkpoint_observer=observer,
    ).fit(observed_train, observed_val, stop_after_global_step=1)

    _assert_state_equal(
        _state_snapshot(baseline_system), _state_snapshot(observed_system)
    )
    assert baseline_report.latest_checkpoint.sha256 == (
        observed_report.latest_checkpoint.sha256
    )
    assert baseline_report.last_train_loss == observed_report.last_train_loss
    assert baseline_report.best_validation_metric == (
        observed_report.best_validation_metric
    )
    baseline_payload = torch.load(
        baseline_report.latest_checkpoint.path,
        map_location="cpu",
        weights_only=False,
    )
    observed_payload = torch.load(
        observed_report.latest_checkpoint.path,
        map_location="cpu",
        weights_only=False,
    )
    assert baseline_payload["state_digest"] == observed_payload["state_digest"]
    assert observed_events == [
        ("update", observed_events[0][1]),
        ("validation", observed_report.latest_checkpoint.sha256),
    ]


def test_live_attempt_ledger_receipts_actual_runtime_observer_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    attempt = AttemptRecord(
        attempt_id="observer-ledger",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        created_at_utc="2026-09-08T00:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(root, attempt)
    ledger = host._LiveAttemptLedger(store, attempt)
    ledger.start()
    try:
        train, val = _sources("live-ledger")
        report = PhaseSetTrainingRuntime(
            _base_system(initialization_seed=123, dropout=0.2),
            _config(),
            store.path / "model-checkpoints",
            checkpoint_observer=ledger.observe_checkpoint,
        ).fit(train, val, stop_after_global_step=1)
        ledger.ensure_checkpoint(report.latest_checkpoint)
    finally:
        ledger.stop()

    receipt_paths = sorted((store.path / "checkpoints").glob("checkpoint-*.json"))
    assert len(receipt_paths) == 1
    receipt = parse_checkpoint_bytes(receipt_paths[0].read_bytes())
    assert receipt.checkpoint_payload_sha256 == report.latest_checkpoint.sha256
    assert receipt.global_step == report.global_step == 1
    assert ledger.latest_checkpoint_payload_sha256 == report.latest_checkpoint.sha256
    assert report.authority == 0
    assert report.result_claimed is False
