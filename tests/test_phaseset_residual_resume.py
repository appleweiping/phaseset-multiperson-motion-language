"""Host residual-resume contract and tiny real-runtime integration tests."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phaseset_core import cli, host, training
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.execution import (
    AttemptRecord,
    AttemptStore,
    CommandIntent,
    ResumeRecord,
    TerminalRecord,
    artifact_sha256,
    canonical_attempt_bytes,
    parse_checkpoint_bytes,
)
from phaseset_core.models import ActorMeanBase, PhaseSetEncoder
from phaseset_core.objectives import PhaseSetRetrievalHead
from phaseset_core.production import BackendExecution
from phaseset_core.training import (
    PhaseSetTrainingRuntime,
    ResidualRetrievalSystem,
    RetrievalTrainingBatch,
    TrainingConfig,
)


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intent() -> CommandIntent:
    return CommandIntent(
        command="resume",
        run_id=None,
        seed=None,
        split=None,
        plan_sha256="a" * 64,
    )


def _host_config(tmp_path: Path) -> host.HostConfig:
    return host.HostConfig(
        path=tmp_path / "host.json",
        prepared_index=tmp_path / "prepared-index.json",
        receipt_record=tmp_path / "receipts.json",
        receipt_artifacts={},
        source_tree_sha256="d" * 64,
        device="cpu",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=4,
        checkpoint_every_updates=1,
        max_batch_bytes=1_000_000,
        resume_retry_class="INFRA_TRANSIENT",
        corrective_change_sha256="e" * 64,
        residual_artifacts=None,
    )


def _predecessor(
    root: Path,
    *,
    attempt_id: str,
    run_id: str,
) -> tuple[AttemptStore, Path]:
    attempt = AttemptRecord(
        attempt_id=attempt_id,
        run_id=run_id,
        created_at_utc="2026-09-08T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(root, attempt)
    checkpoint = store.path / "model-checkpoints" / "checkpoint-000000000001-000002-validation.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "best_checkpoint_name": checkpoint.name,
            "best_checkpoint_sha256": None,
            "best_validation_metric": 1.0,
            "epoch": 1,
            "global_step": 1,
            "model": {},
            "optimizer": {},
            "rng": {},
            "train_manifest_sha256": "5" * 64,
            "update_index": 0,
            "updates_per_epoch": 1,
            "val_manifest_sha256": "6" * 64,
        },
        checkpoint,
    )
    receipt = host._checkpoint_record(
        checkpoint,
        _sha(checkpoint),
        attempt_id=attempt.attempt_id,
        run_id=attempt.run_id,
        written_at_utc="2026-09-08T12:00:01Z",
        parent_checkpoint_receipt_sha256=None,
    )
    receipt_sha256 = store.write_checkpoint(receipt)
    store.write_terminal(
        TerminalRecord(
            attempt_id=attempt.attempt_id,
            run_id=attempt.run_id,
            completed_at_utc="2026-09-08T12:00:02Z",
            outcome="FAILED",
            attempt_receipt_sha256=artifact_sha256(canonical_attempt_bytes(attempt)),
            latest_heartbeat_sha256=None,
            latest_checkpoint_receipt_sha256=receipt_sha256,
            failure_code="CONTROLLED_INTERRUPTION",
        )
    )
    return store, checkpoint


def _predecessor_with_prior_best(
    root: Path,
    *,
    attempt_id: str,
    run_id: str,
    best_case: str = "valid",
) -> tuple[AttemptStore, Path, Path]:
    attempt = AttemptRecord(
        attempt_id=attempt_id,
        run_id=run_id,
        created_at_utc="2026-09-08T12:10:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(root, attempt)
    checkpoint_directory = store.path / "model-checkpoints"
    checkpoint_directory.mkdir()
    best = checkpoint_directory / "checkpoint-000000000002-000003-validation.pt"
    torch.save(
        {
            "best_checkpoint_name": best.name,
            "best_checkpoint_sha256": None,
            "best_validation_metric": 0.75,
            "epoch": 1,
            "global_step": 2,
            "model": {"best": torch.tensor([2.0])},
            "optimizer": {"state": {}},
            "rng": {},
            "train_manifest_sha256": "5" * 64,
            "update_index": 0,
            "updates_per_epoch": 2,
            "val_manifest_sha256": "6" * 64,
        },
        best,
    )
    best_sha256 = _sha(best)
    if best_case == "missing":
        best.unlink()
    elif best_case == "wrong-digest":
        best_sha256 = "0" * 64
    elif best_case != "valid":
        raise AssertionError("unknown best-checkpoint fixture")
    latest = checkpoint_directory / "checkpoint-000000000003-000004-update.pt"
    torch.save(
        {
            "best_checkpoint_name": best.name,
            "best_checkpoint_sha256": best_sha256,
            "best_validation_metric": 0.75,
            "epoch": 1,
            "global_step": 3,
            "model": {"latest": torch.tensor([3.0])},
            "optimizer": {"state": {}},
            "rng": {},
            "train_manifest_sha256": "5" * 64,
            "update_index": 1,
            "updates_per_epoch": 2,
            "val_manifest_sha256": "6" * 64,
        },
        latest,
    )
    receipt = host._checkpoint_record(
        latest,
        _sha(latest),
        attempt_id=attempt.attempt_id,
        run_id=attempt.run_id,
        written_at_utc="2026-09-08T12:10:01Z",
        parent_checkpoint_receipt_sha256=None,
    )
    receipt_sha256 = store.write_checkpoint(receipt)
    store.write_terminal(
        TerminalRecord(
            attempt_id=attempt.attempt_id,
            run_id=attempt.run_id,
            completed_at_utc="2026-09-08T12:10:02Z",
            outcome="FAILED",
            attempt_receipt_sha256=artifact_sha256(canonical_attempt_bytes(attempt)),
            latest_heartbeat_sha256=None,
            latest_checkpoint_receipt_sha256=receipt_sha256,
            failure_code="CONTROLLED_INTERRUPTION",
        )
    )
    return store, latest, best


def _resume_request(
    root: Path,
    checkpoint: Path,
    *,
    attempt_id: str,
    system_id: str,
    base_checkpoint: Path | None = None,
    periodic_cache: Path | None = None,
) -> cli.CLICommandRequest:
    return cli.CLICommandRequest(
        command="resume",
        system_id=system_id,
        attempt_directory=root / attempt_id,
        resume_checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        periodic_cache=periodic_cache,
    )


def test_resume_cli_carries_residual_artifacts_but_has_no_stop_option(
    tmp_path: Path,
) -> None:
    parser = cli.build_parser()
    checkpoint = tmp_path / "checkpoint.pt"
    base = tmp_path / "base.pt"
    cache = tmp_path / "cache"
    attempt = tmp_path / "attempt"
    namespace = parser.parse_args(
        [
            "resume",
            "--attempt-dir",
            str(attempt),
            "--checkpoint",
            str(checkpoint),
            "--system-id",
            "08",
            "--base-checkpoint",
            str(base),
            "--periodic-cache",
            str(cache),
        ]
    )
    request = cli._build_command_request(namespace)
    assert request.attempt_directory == attempt
    assert request.resume_checkpoint == checkpoint
    assert request.base_checkpoint == base
    assert request.periodic_cache == cache
    assert request.stop_after_global_step is None

    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "--stop-after-global-step" not in subparsers.choices["resume"].format_help()


def test_base_resume_rejects_residual_artifact_paths(tmp_path: Path) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    _store, checkpoint = _predecessor(
        root,
        attempt_id="base-first",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
    )
    extra = tmp_path / "extra-base.pt"
    extra.write_bytes(b"not-consumed")
    request = _resume_request(
        root,
        checkpoint,
        attempt_id="base-resume",
        system_id="B0",
        base_checkpoint=extra,
    )
    backend = host.PhaseSetHostBackend(_host_config(tmp_path), request)
    with pytest.raises(host.HostConfigurationError, match="base resume rejects"):
        backend._resume_base(_intent())
    assert not request.attempt_directory.exists()


@pytest.mark.parametrize("case", ["missing", "base-is-directory", "cache-is-file"])
def test_residual_resume_requires_explicit_regular_artifacts(
    tmp_path: Path,
    case: str,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    _store, checkpoint = _predecessor(
        root,
        attempt_id="residual-first",
        run_id="phaseset-run-v1/RESIDUAL_TRAIN/1729/08",
    )
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")
    cache = tmp_path / "cache"
    cache.mkdir()
    if case == "missing":
        base_value = None
        cache_value = None
        message = "requires --base-checkpoint"
    elif case == "base-is-directory":
        base_value = cache
        cache_value = cache
        message = "regular existing file"
    else:
        base_value = base
        cache_value = base
        message = "regular existing directory"
    request = _resume_request(
        root,
        checkpoint,
        attempt_id=f"residual-resume-{case}",
        system_id="08",
        base_checkpoint=base_value,
        periodic_cache=cache_value,
    )
    backend = host.PhaseSetHostBackend(_host_config(tmp_path), request)
    with pytest.raises(host.HostConfigurationError, match=message):
        backend._resume_base(_intent())
    assert not request.attempt_directory.exists()


def test_residual_resume_dispatches_verified_new_ledger_and_runtime_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    _store, checkpoint, prior_best = _predecessor_with_prior_best(
        root,
        attempt_id="residual-first",
        run_id="phaseset-run-v1/RESIDUAL_TRAIN/1729/08",
    )
    base = tmp_path / "base.pt"
    base.write_bytes(b"qualified-base")
    cache = tmp_path / "cache"
    cache.mkdir()
    request = _resume_request(
        root,
        checkpoint,
        attempt_id="residual-second",
        system_id="08",
        base_checkpoint=base,
        periodic_cache=cache,
    )
    backend = host.PhaseSetHostBackend(_host_config(tmp_path), request)
    captured: dict[str, object] = {}
    fake_report = SimpleNamespace(latest_checkpoint=object())
    artifact = tmp_path / "dispatch.json"
    artifact.write_bytes(b"{}\n")

    def execute_residual(_self: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return fake_report

    def close_attempt(
        _self: object,
        _store: object,
        _attempt: object,
        report: object,
        _ledger: object,
    ) -> BackendExecution:
        assert report is fake_report
        return BackendExecution(
            outcome="COMPLETED",
            completed_at_utc="2026-09-08T12:00:03Z",
            artifacts=(artifact,),
        )

    monkeypatch.setattr(
        host.PhaseSetHostBackend,
        "_execute_residual_runtime",
        execute_residual,
    )
    monkeypatch.setattr(host._LiveAttemptLedger, "ensure_checkpoint", lambda *_: None)
    monkeypatch.setattr(host.PhaseSetHostBackend, "_close_attempt", close_attempt)
    result = backend._resume_base(_intent())

    assert result.outcome == "COMPLETED"
    assert captured["system_id"] == "08"
    assert captured["seed"] == 1729
    assert captured["base_checkpoint"] == base.resolve()
    assert captured["periodic_cache"] == cache.resolve()
    assert captured["stop_after_global_step"] is None
    assert captured["resume_attempt_root"] == root.resolve()
    assert type(captured["resume_record"]) is ResumeRecord
    copied = captured["resume_checkpoint"]
    assert copied == request.attempt_directory / "model-checkpoints" / checkpoint.name
    assert isinstance(copied, Path) and _sha(copied) == _sha(checkpoint)
    materialized = request.attempt_directory / "model-checkpoints"
    assert {path.name for path in materialized.iterdir()} == {
        checkpoint.name,
        prior_best.name,
    }
    assert _sha(materialized / prior_best.name) == _sha(prior_best)
    assert (request.attempt_directory / "resume.json").is_file()


def test_materialization_rejects_digest_change_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    _store, checkpoint = _predecessor(
        root,
        attempt_id="residual-first",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
    )
    load_calls = 0
    original_materialize = host._materialize_resume_checkpoints
    changed_bytes = b"changed-after-predecessor-validation"

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("digest mismatch must precede checkpoint deserialization")

    def mutate_then_materialize(
        source: Path,
        receipt: object,
        destination: Path,
    ) -> Path:
        source.write_bytes(changed_bytes)
        return original_materialize(source, receipt, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(host.torch, "load", forbidden_load)
    monkeypatch.setattr(host, "_materialize_resume_checkpoints", mutate_then_materialize)
    request = _resume_request(
        root,
        checkpoint,
        attempt_id="base-resume-copy-race",
        system_id="B0",
    )
    result = host.PhaseSetHostBackend(_host_config(tmp_path), request)._resume_base(
        _intent()
    )

    assert result.outcome == "FAILED"
    assert load_calls == 0
    materialized = request.attempt_directory / "model-checkpoints" / checkpoint.name
    assert materialized.read_bytes() == changed_bytes
    failure = json.loads((request.attempt_directory / "failure.json").read_text())
    terminal = host.parse_terminal_bytes(
        (request.attempt_directory / "terminal.json").read_bytes()
    )
    assert failure["failure_code"] == "HOST_EXECUTION_FAILED"
    assert failure["result_claimed"] is False
    assert terminal.outcome == "FAILED"
    assert terminal.failure_code == "HOST_EXECUTION_FAILED"


@pytest.mark.parametrize("best_case", ["missing", "wrong-digest"])
def test_bad_prior_best_terminalizes_new_attempt_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    best_case: str,
) -> None:
    root = tmp_path / "attempts"
    root.mkdir()
    _store, checkpoint, prior_best = _predecessor_with_prior_best(
        root,
        attempt_id="residual-first",
        run_id="phaseset-run-v1/RESIDUAL_TRAIN/1729/08",
        best_case=best_case,
    )
    base = tmp_path / "base.pt"
    base.write_bytes(b"qualified-base")
    cache = tmp_path / "cache"
    cache.mkdir()
    request = _resume_request(
        root,
        checkpoint,
        attempt_id=f"residual-resume-{best_case}",
        system_id="08",
        base_checkpoint=base,
        periodic_cache=cache,
    )
    backend = host.PhaseSetHostBackend(_host_config(tmp_path), request)

    runtime_calls: list[bool] = []

    def forbidden_runtime(*_args: object, **_kwargs: object) -> object:
        runtime_calls.append(True)
        raise AssertionError("bad prior best must fail before runtime construction")

    monkeypatch.setattr(
        host.PhaseSetHostBackend,
        "_execute_residual_runtime",
        forbidden_runtime,
    )
    result = backend._resume_base(_intent())

    assert result.outcome == "FAILED"
    assert runtime_calls == []
    failure = json.loads((request.attempt_directory / "failure.json").read_text())
    terminal = host.parse_terminal_bytes(
        (request.attempt_directory / "terminal.json").read_bytes()
    )
    assert failure["failure_code"] == "HOST_EXECUTION_FAILED"
    assert failure["result_claimed"] is False
    assert terminal.outcome == "FAILED"
    assert terminal.failure_code == "HOST_EXECUTION_FAILED"
    assert terminal.latest_checkpoint_receipt_sha256 is None
    materialized = request.attempt_directory / "model-checkpoints"
    expected_names = {checkpoint.name}
    if best_case == "wrong-digest":
        expected_names.add(prior_best.name)
        assert (materialized / prior_best.name).read_bytes() == prior_best.read_bytes()
    assert {path.name for path in materialized.iterdir()} == expected_names


def test_residual_runtime_rebuilds_registered_chain_and_forwards_resume_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_path = tmp_path / "qualification.json"
    audit_path = tmp_path / "capacity-audit.json"
    base_checkpoint = tmp_path / "selected-base.pt"
    periodic_cache = tmp_path / "periodic-cache"
    checkpoint_directory = tmp_path / "model-checkpoints"
    resume_checkpoint = tmp_path / "resume.pt"
    for path in (qualification_path, audit_path, base_checkpoint, resume_checkpoint):
        path.write_bytes(b"fixture")
    periodic_cache.mkdir()
    checkpoint_directory.mkdir()
    config = replace(
        _host_config(tmp_path),
        residual_artifacts=host.ResidualArtifacts(
            qualification=qualification_path,
            capacity_audits={1729: audit_path},
        ),
    )
    backend = host.PhaseSetHostBackend(
        config,
        cli.CLICommandRequest(command="resume", system_id="08"),
    )
    resume_record = object()
    train_source = object()
    val_source = object()
    qualification = object()
    qualified_base = object()
    capacity_audit = object()
    system = object()
    binding = object()
    report = object()
    energy_floors = np.arange(6, dtype=np.float64)
    events: list[tuple[str, object]] = []

    def load_qualification(path: Path) -> tuple[object, str]:
        events.append(("qualification", path))
        return qualification, "a" * 64

    def load_base(path: Path, **kwargs: object) -> object:
        events.append(("base", (path, kwargs)))
        return qualified_base

    def load_cache(path: Path, **kwargs: object) -> np.ndarray:
        events.append(("cache", (path, kwargs)))
        return energy_floors

    def load_audit(path: Path) -> tuple[object, str]:
        events.append(("audit", path))
        return capacity_audit, "b" * 64

    def load_sources(_self: host.HostConfig) -> tuple[object, object]:
        events.append(("sources", _self))
        return train_source, val_source

    def construct(system_id: str, *args: object, **kwargs: object):
        events.append(("constructor", (system_id, args, kwargs)))
        return system, binding

    class FakeRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append(("runtime", (args, kwargs)))

        def fit(self, *args: object, **kwargs: object) -> object:
            events.append(("fit", (args, kwargs)))
            return report

    monkeypatch.setattr(host, "_load_base_qualification", load_qualification)
    monkeypatch.setattr(host, "load_qualified_frozen_base", load_base)
    monkeypatch.setattr(host, "_load_periodic_cache", load_cache)
    monkeypatch.setattr(host, "_load_capacity_audit", load_audit)
    monkeypatch.setattr(host.HostConfig, "load_sources", load_sources)
    monkeypatch.setattr(host, "construct_registered_residual_seed_bound_system", construct)
    monkeypatch.setattr(host, "PhaseSetTrainingRuntime", FakeRuntime)
    ledger = SimpleNamespace(observe_checkpoint=object())

    result = backend._execute_residual_runtime(
        system_id="08",
        seed=1729,
        checkpoint_directory=checkpoint_directory,
        live_ledger=ledger,
        base_checkpoint=base_checkpoint,
        periodic_cache=periodic_cache,
        stop_after_global_step=None,
        resume_checkpoint=resume_checkpoint,
        resume_attempt_root=tmp_path,
        resume_record=resume_record,  # type: ignore[arg-type]
    )

    assert result is report
    assert [name for name, _ in events] == [
        "qualification",
        "base",
        "cache",
        "audit",
        "sources",
        "constructor",
        "runtime",
        "fit",
    ]
    assert events[0][1] == qualification_path
    base_args = events[1][1]
    assert isinstance(base_args, tuple) and base_args[0] == base_checkpoint
    assert base_args[1] == {
        "qualification": qualification,
        "seed": 1729,
        "expected_qualification_sha256": "a" * 64,
    }
    cache_args = events[2][1]
    assert isinstance(cache_args, tuple) and cache_args[0] == periodic_cache
    assert cache_args[1] == {
        "seed": 1729,
        "qualification": qualification,
        "qualification_sha256": "a" * 64,
    }
    constructor_args = events[5][1]
    assert isinstance(constructor_args, tuple) and constructor_args[0] == "08"
    assert constructor_args[1][0] is qualified_base
    assert constructor_args[2]["energy_floors"] is energy_floors
    assert constructor_args[2]["residual_capacity_audit"] is capacity_audit
    assert constructor_args[2]["expected_qualification_sha256"] == "a" * 64
    assert constructor_args[2]["expected_capacity_audit_sha256"] == "b" * 64
    runtime_args = events[6][1]
    assert isinstance(runtime_args, tuple)
    assert runtime_args[0][0] is system
    assert runtime_args[0][2] == checkpoint_directory
    assert runtime_args[1]["initialization_binding"] is binding
    assert runtime_args[1]["checkpoint_observer"] is ledger.observe_checkpoint
    fit_args = events[7][1]
    assert isinstance(fit_args, tuple)
    assert fit_args[0] == (train_source, val_source)
    assert fit_args[1] == {
        "resume_checkpoint": resume_checkpoint,
        "resume_attempt_root": tmp_path,
        "resume_record": resume_record,
        "stop_after_global_step": None,
    }


@pytest.mark.parametrize(
    ("broken", "expected_calls"),
    [("base", ["qualification", "base"]), ("cache", ["qualification", "base", "cache"])],
)
def test_residual_runtime_stops_before_training_on_semantically_wrong_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken: str,
    expected_calls: list[str],
) -> None:
    qualification_path = tmp_path / "qualification.json"
    audit_path = tmp_path / "capacity-audit.json"
    base_checkpoint = tmp_path / "selected-base.pt"
    periodic_cache = tmp_path / "periodic-cache"
    for path in (qualification_path, audit_path, base_checkpoint):
        path.write_bytes(b"fixture")
    periodic_cache.mkdir()
    config = replace(
        _host_config(tmp_path),
        residual_artifacts=host.ResidualArtifacts(
            qualification=qualification_path,
            capacity_audits={1729: audit_path},
        ),
    )
    backend = host.PhaseSetHostBackend(
        config,
        cli.CLICommandRequest(command="resume", system_id="08"),
    )
    calls: list[str] = []

    def load_qualification(_path: Path) -> tuple[object, str]:
        calls.append("qualification")
        return object(), "a" * 64

    def load_base(*_args: object, **_kwargs: object) -> object:
        calls.append("base")
        if broken == "base":
            raise host.HostConfigurationError("qualified base is wrong")
        return object()

    def load_cache(*_args: object, **_kwargs: object) -> np.ndarray:
        calls.append("cache")
        raise host.HostConfigurationError("periodic cache is wrong")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("semantic artifact failure must precede training")

    monkeypatch.setattr(host, "_load_base_qualification", load_qualification)
    monkeypatch.setattr(host, "load_qualified_frozen_base", load_base)
    monkeypatch.setattr(host, "_load_periodic_cache", load_cache)
    monkeypatch.setattr(host, "_load_capacity_audit", forbidden)
    monkeypatch.setattr(host.HostConfig, "load_sources", forbidden)
    monkeypatch.setattr(host, "construct_registered_residual_seed_bound_system", forbidden)
    monkeypatch.setattr(host, "PhaseSetTrainingRuntime", forbidden)

    with pytest.raises(host.HostConfigurationError, match="is wrong"):
        backend._execute_residual_runtime(
            system_id="08",
            seed=1729,
            checkpoint_directory=tmp_path / "model-checkpoints",
            live_ledger=SimpleNamespace(observe_checkpoint=object()),
            base_checkpoint=base_checkpoint,
            periodic_cache=periodic_cache,
            stop_after_global_step=None,
        )
    assert calls == expected_calls


def _groups() -> PreparedGroupBatch:
    rows = 2
    actors = 2
    frames = 9
    skeletons = np.zeros((rows, actors, frames, 22, 3), dtype=np.float32)
    grid = np.arange(frames, dtype=np.float32)
    actor_commitments: list[tuple[bytes, ...]] = []
    group_commitments: list[bytes] = []
    for row in range(rows):
        actor_ids = tuple(_digest(f"actor-{row}-{actor}") for actor in range(actors))
        actor_commitments.append(actor_ids)
        group_commitments.append(group_commitment(actor_ids))
        for actor in range(actors):
            for joint in range(22):
                skeletons[row, actor, :, joint, 0] = np.sin(
                    np.float32(0.19 + 0.03 * actor) * grid + np.float32(0.01 * joint + 0.02 * row)
                )
                skeletons[row, actor, :, joint, 2] = np.cos(
                    np.float32(0.11 + 0.02 * row) * grid + np.float32(0.01 * joint)
                )
    return PreparedGroupBatch(
        skeletons,
        np.ones((rows, actors), dtype=np.bool_),
        np.ones((rows, frames), dtype=np.bool_),
        np.ones((rows, actors, frames, 22), dtype=np.bool_),
        tuple(actor_commitments),
        tuple(group_commitments),
    )


def _batch(split: str) -> RetrievalTrainingBatch:
    groups = _groups()
    positive_ids = (_digest("positive-0"), _digest("positive-1"))
    text = torch.randn(
        (2, 8),
        generator=torch.Generator().manual_seed(701),
        dtype=torch.float32,
    ).contiguous()
    return RetrievalTrainingBatch(
        groups,
        text,
        positive_ids,
        positive_ids,
        (_digest("text-0"), _digest("text-1")),
        split,  # type: ignore[arg-type]
    )


@dataclass
class _Source:
    split: str
    manifest_sha256: str

    def iter_epoch(self, *, epoch: int, seed: int):
        del epoch, seed
        return (_batch(self.split),)


def _sources() -> tuple[_Source, _Source]:
    return (
        _Source("train", hashlib.sha256(b"tiny-train").hexdigest()),
        _Source("val", hashlib.sha256(b"tiny-val").hexdigest()),
    )


def _residual_system(initialization_seed: int) -> ResidualRetrievalSystem:
    torch.manual_seed(initialization_seed)
    return ResidualRetrievalSystem(
        ActorMeanBase(hidden_dim=8, heads=2, ffn_dim=16, dropout=0.2),
        PhaseSetEncoder(embedding_dim=8, hidden_dim=8, edge_budget=4),
        PhaseSetRetrievalHead(
            embedding_dim=8,
            text_hidden_dim=8,
            residual_lambda_init=0.2,
        ),
        embedding_dim=8,
    )


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _report_semantics(report: object) -> dict[str, object]:
    value = asdict(report)  # type: ignore[arg-type]
    for name in ("latest_checkpoint", "best_checkpoint"):
        checkpoint = value[name]
        if checkpoint is not None:
            checkpoint["path"] = Path(checkpoint["path"]).name
    return value


def test_tiny_real_residual_new_ledger_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        stage="residual",
        seed=1729,
        effective_global_batch=2,
        edge_budget=4,
        checkpoint_every_updates=1,
        synthetic_contract=True,
    )
    uninterrupted_system = _residual_system(991)
    uninterrupted_runtime = PhaseSetTrainingRuntime(
        uninterrupted_system,
        config,
        tmp_path / "uninterrupted",
    )
    uninterrupted_report = uninterrupted_runtime.fit(*_sources(), stop_after_global_step=2)

    attempt_root = tmp_path / "attempts"
    attempt_root.mkdir()
    run_id = "phaseset-run-v1/RESIDUAL_TRAIN/1729/08"
    predecessor_attempt = AttemptRecord(
        attempt_id="tiny-residual-first",
        run_id=run_id,
        created_at_utc="2026-09-08T13:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    predecessor = AttemptStore.create(attempt_root, predecessor_attempt)
    predecessor_ledger = host._LiveAttemptLedger(predecessor, predecessor_attempt)
    predecessor_ledger.start()
    first_system = _residual_system(991)
    first_runtime = PhaseSetTrainingRuntime(
        first_system,
        config,
        predecessor.path / "model-checkpoints",
        checkpoint_observer=predecessor_ledger.observe_checkpoint,
    )
    first_report = first_runtime.fit(*_sources(), stop_after_global_step=1)
    predecessor_ledger.ensure_checkpoint(first_report.latest_checkpoint)
    predecessor_ledger.stop()
    assert predecessor_ledger.latest_checkpoint_receipt_sha256 is not None
    predecessor_terminal_sha256 = predecessor.write_terminal(
        TerminalRecord(
            attempt_id=predecessor_attempt.attempt_id,
            run_id=run_id,
            completed_at_utc="2026-09-08T13:00:01Z",
            outcome="FAILED",
            attempt_receipt_sha256=artifact_sha256(canonical_attempt_bytes(predecessor_attempt)),
            latest_heartbeat_sha256=predecessor_ledger.latest_heartbeat_sha256,
            latest_checkpoint_receipt_sha256=(predecessor_ledger.latest_checkpoint_receipt_sha256),
            failure_code="CONTROLLED_INTERRUPTION",
        )
    )

    resumed_attempt = AttemptRecord(
        attempt_id="tiny-residual-second",
        run_id=run_id,
        created_at_utc="2026-09-08T13:00:02Z",
        plan_sha256=predecessor_attempt.plan_sha256,
        matrix_sha256=predecessor_attempt.matrix_sha256,
        training_config_sha256=predecessor_attempt.training_config_sha256,
        source_tree_sha256=predecessor_attempt.source_tree_sha256,
    )
    resume = ResumeRecord(
        new_attempt_id=resumed_attempt.attempt_id,
        predecessor_attempt_id=predecessor_attempt.attempt_id,
        run_id=run_id,
        created_at_utc=resumed_attempt.created_at_utc,
        predecessor_terminal_sha256=predecessor_terminal_sha256,
        checkpoint_receipt_sha256=(predecessor_ledger.latest_checkpoint_receipt_sha256),
        corrective_change_sha256="7" * 64,
        retry_class="INFRA_TRANSIENT",
    )
    resumed_store = AttemptStore.create_resumed(attempt_root, resumed_attempt, resume)
    checkpoint_receipt_path = sorted((predecessor.path / "checkpoints").glob("checkpoint-*.json"))[
        -1
    ]
    checkpoint_receipt = parse_checkpoint_bytes(checkpoint_receipt_path.read_bytes())
    resumed_ledger = host._LiveAttemptLedger(resumed_store, resumed_attempt)
    resumed_ledger.start()
    materialized = host._materialize_resume_checkpoints(
        first_report.latest_checkpoint.path,
        checkpoint_receipt,
        resumed_store.path / "model-checkpoints",
    )
    resumed_system = _residual_system(991)
    resumed_runtime = PhaseSetTrainingRuntime(
        resumed_system,
        config,
        resumed_store.path / "model-checkpoints",
        checkpoint_observer=resumed_ledger.observe_checkpoint,
    )
    resumed_report = resumed_runtime.fit(
        *_sources(),
        resume_checkpoint=materialized,
        resume_attempt_root=attempt_root,
        resume_record=resume,
        stop_after_global_step=2,
    )
    resumed_ledger.ensure_checkpoint(resumed_report.latest_checkpoint)
    resumed_ledger.stop()

    _assert_state_equal(_state(uninterrupted_system), _state(resumed_system))
    assert training._stable_hash(
        uninterrupted_runtime._checked_optimizer.state_dict()
    ) == training._stable_hash(resumed_runtime._checked_optimizer.state_dict())
    assert training._stable_hash(
        uninterrupted_runtime._checked_scheduler.state_dict()
    ) == training._stable_hash(resumed_runtime._checked_scheduler.state_dict())
    assert _report_semantics(uninterrupted_report) == _report_semantics(resumed_report)
    assert resumed_ledger.latest_checkpoint_payload_sha256 == (
        resumed_report.latest_checkpoint.sha256
    )
    payload = torch.load(
        resumed_report.latest_checkpoint.path,
        map_location="cpu",
        weights_only=True,
    )
    assert {
        "frozen_base_checkpoint_sha256",
        "frozen_base_state_sha256",
        "qualified_base_selection_sha256",
        "residual_capacity_audit_sha256",
    }.issubset(payload)
    assert not any("periodic_cache" in key or "descriptor" in key for key in payload)
