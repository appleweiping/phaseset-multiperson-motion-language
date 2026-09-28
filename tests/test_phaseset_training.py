"""Tiny real-forward tests for the executable PhaseSet training runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import threading
import types

import numpy as np
import pytest
import torch
from torch import nn

from phaseset_core import execution, experiments, training as training_module
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.controls import build_phaseset_system
from phaseset_core.models import ActorMeanBase, PhaseSetEncoder
from phaseset_core.objectives import PhaseSetRetrievalHead
from phaseset_core.training import (
    BASE_EPOCHS,
    BASE_LEARNING_RATE,
    EFFECTIVE_GLOBAL_BATCH,
    RESIDUAL_EPOCHS,
    RESIDUAL_LEARNING_RATE,
    BaseRetrievalSystem,
    PhaseSetTrainingRuntime,
    ResidualCapacityAudit,
    ResidualRetrievalSystem,
    RetrievalTrainingBatch,
    TrainingCheckpointError,
    TrainingConfig,
    TrainingInitializationBinding,
    TrainingRuntimeError,
    audit_residual_training_parameter_counts,
    build_registered_base_training_system,
    build_registered_residual_capacity_audit,
    construct_registered_base_seed_bound_system,
    construct_registered_residual_seed_bound_system,
    construct_seed_bound_system,
    load_qualified_frozen_base,
    resolve_precision,
    training_system_behavior_sha256,
    training_system_state_sha256,
)


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _prepared_groups(*, rows: int = 1, actors: int = 2) -> PreparedGroupBatch:
    time_steps = 9
    skeletons = np.zeros((rows, actors, time_steps, 22, 3), dtype=np.float32)
    actor_mask = np.ones((rows, actors), dtype=np.bool_)
    frame_mask = np.ones((rows, time_steps), dtype=np.bool_)
    track_mask = np.ones((rows, actors, time_steps, 22), dtype=np.bool_)
    commitments: list[tuple[bytes, ...]] = []
    groups: list[bytes] = []
    grid = np.arange(time_steps, dtype=np.float32)
    for row in range(rows):
        keys = tuple(_digest(f"actor-{row}-{actor}") for actor in range(actors))
        commitments.append(keys)
        groups.append(group_commitment(keys))
        for actor in range(actors):
            for joint in range(22):
                skeletons[row, actor, :, joint, 0] = np.sin(
                    np.float32(0.31 + 0.02 * actor) * grid
                    + np.float32(0.01 * joint + 0.03 * row)
                )
                skeletons[row, actor, :, joint, 1] = np.float32(
                    0.02 * actor + 0.001 * joint
                )
                skeletons[row, actor, :, joint, 2] = np.cos(
                    np.float32(0.17 + 0.01 * row) * grid
                    + np.float32(0.015 * joint)
                )
    return PreparedGroupBatch(
        skeletons,
        actor_mask,
        frame_mask,
        track_mask,
        tuple(commitments),
        tuple(groups),
    )


def _training_batch(
    *,
    split: str,
    rows: int = 1,
    actors: int = 2,
    variable_positives: bool = False,
    embedding_dim: int = 8,
) -> RetrievalTrainingBatch:
    groups = _prepared_groups(rows=rows, actors=actors)
    motion_ids = tuple(_digest(f"positive-{row}") for row in range(rows))
    if variable_positives:
        assert rows == 2
        text_ids = (motion_ids[0], motion_ids[0], motion_ids[1])
    else:
        text_ids = motion_ids
    generator = torch.Generator().manual_seed(701 + rows + actors)
    text = torch.randn(
        (len(text_ids), embedding_dim),
        generator=generator,
        dtype=torch.float32,
    )
    return RetrievalTrainingBatch(
        groups,
        text.contiguous(),
        motion_ids,
        text_ids,
        tuple(_digest(f"text-commitment-{index}") for index in range(len(text_ids))),
        split,  # type: ignore[arg-type]
    )


@dataclass
class _Source:
    split: str
    batches: tuple[RetrievalTrainingBatch, ...] | None
    manifest_sha256: str
    calls: list[tuple[int, int]] = field(default_factory=list)

    def iter_epoch(self, *, epoch: int, seed: int):
        self.calls.append((epoch, seed))
        if self.batches is None:
            raise AssertionError("sealed test source must never be iterated")
        return self.batches


def _source(batch: RetrievalTrainingBatch, label: str) -> _Source:
    return _Source(
        batch.split,
        (batch,),
        hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _multi_source(
    batches: tuple[RetrievalTrainingBatch, ...], label: str
) -> _Source:
    assert batches and len({batch.split for batch in batches}) == 1
    return _Source(
        batches[0].split,
        batches,
        hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _base_system(*, initialization_seed: int = 91, dropout: float = 0.2) -> BaseRetrievalSystem:
    torch.manual_seed(initialization_seed)
    base = ActorMeanBase(hidden_dim=8, heads=2, ffn_dim=16, dropout=dropout)
    return BaseRetrievalSystem(base, embedding_dim=8)


def _factory_base() -> BaseRetrievalSystem:
    return BaseRetrievalSystem(
        ActorMeanBase(hidden_dim=8, heads=2, ffn_dim=16, dropout=0.2),
        embedding_dim=8,
    )


def _state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


class _RandomConsumingHead(PhaseSetRetrievalHead):
    def __init__(self) -> None:
        super().__init__(embedding_dim=8, text_hidden_dim=8, residual_lambda_init=0.2)
        self.draws: list[float] = []

    def forward(
        self,
        group_tokens: torch.Tensor,
        band_mask: torch.Tensor,
        text_embeddings: torch.Tensor,
        frozen_base_scores: torch.Tensor,
    ):
        if self.training:
            self.draws.append(float(torch.rand((), device=group_tokens.device).item()))
        return super().forward(
            group_tokens,
            band_mask,
            text_embeddings,
            frozen_base_scores,
        )


def test_frozen_hyperparameters_variable_positive_batch_and_precision_contract() -> None:
    batch = _training_batch(
        split="train",
        rows=2,
        actors=3,
        variable_positives=True,
    )
    assert batch.motion_count == 2
    assert batch.edge_count == 6
    assert batch.text_embeddings.requires_grad is False
    assert tuple(batch.motion_positive_ids).count(batch.text_positive_ids[0]) == 1

    base = TrainingConfig(stage="base", seed=1729)
    residual = TrainingConfig(stage="residual", seed=2718)
    assert base.epochs == BASE_EPOCHS == 30
    assert residual.epochs == RESIDUAL_EPOCHS == 20
    assert base.learning_rate == BASE_LEARNING_RATE == 2e-4
    assert residual.learning_rate == RESIDUAL_LEARNING_RATE == 3e-4
    assert base.effective_global_batch == EFFECTIVE_GLOBAL_BATCH == 128

    fallback = TrainingConfig(
        stage="base",
        seed=1729,
        request_bf16=True,
        bf16_runtime_qualified=False,
    )
    assert resolve_precision(fallback).mode == "FP32"
    assert resolve_precision(fallback).reason == "BF16_NOT_RUNTIME_QUALIFIED"
    cpu_qualified = TrainingConfig(
        stage="base",
        seed=1729,
        request_bf16=True,
        bf16_runtime_qualified=True,
    )
    assert resolve_precision(cpu_qualified).mode == "FP32"
    assert resolve_precision(cpu_qualified).reason == "BF16_QUALIFICATION_IS_CUDA_ONLY"

    with pytest.raises(TrainingRuntimeError, match="128"):
        TrainingConfig(stage="base", seed=1729, effective_global_batch=2)


def test_base_training_uses_real_forward_optimizer_validation_and_edge_accounting(
    tmp_path: Path,
) -> None:
    train_batch = _training_batch(
        split="train",
        rows=2,
        actors=3,
        variable_positives=True,
    )
    val_batch = _training_batch(split="val", rows=2, actors=3, variable_positives=True)
    train = _source(train_batch, "base-train")
    val = _source(val_batch, "base-val")
    system = _base_system(dropout=0.1)
    before = _state_snapshot(system)
    config = TrainingConfig(
        stage="base",
        seed=1729,
        effective_global_batch=2,
        edge_budget=6,
        synthetic_contract=True,
    )
    report = PhaseSetTrainingRuntime(system, config, tmp_path / "base").fit(
        train,
        val,
        stop_after_global_step=1,
    )
    after = _state_snapshot(system)

    assert report.status == "INTERRUPTED"
    assert report.global_step == 1
    assert report.completed_epochs == 1
    assert report.best_validation_metric is not None
    assert report.best_checkpoint is not None
    assert report.best_checkpoint.path.is_file()
    assert report.latest_checkpoint.path.is_file()
    assert report.train_edges_seen == 6
    assert report.validation_edges_seen == 6
    assert report.max_microbatch_edges == 6
    assert report.authority == 0
    assert report.production is False
    assert report.result_claimed is False
    assert report.external_receipt_verified is False
    assert any(not torch.equal(before[name], after[name]) for name in before)
    assert train.calls == [(0, 1729)]
    assert val.calls == [(0, 1729)]


def test_edge_microbatches_accumulate_into_one_global_negative_gallery(tmp_path: Path) -> None:
    first = _training_batch(split="train")
    second_raw = _training_batch(split="train")
    second_id = _digest("second-positive-family")
    second = RetrievalTrainingBatch(
        second_raw.groups,
        second_raw.text_embeddings,
        (second_id,),
        (second_id,),
        (_digest("second-text-commitment"),),
        "train",
    )
    train = _multi_source((first, second), "two-microbatches")
    val = _source(_training_batch(split="val", rows=2), "two-microbatches-val")
    config = TrainingConfig(
        stage="base",
        seed=1729,
        effective_global_batch=2,
        edge_budget=2,
        synthetic_contract=True,
    )
    report = PhaseSetTrainingRuntime(
        _base_system(dropout=0.1),
        config,
        tmp_path / "accumulation",
    ).fit(train, val, stop_after_global_step=1)
    assert report.global_step == 1
    assert report.train_edges_seen == 2
    assert report.max_microbatch_edges == 2
    assert train.calls == [(0, 1729)]


def test_checkpoint_resume_is_bitwise_equivalent_and_truncation_fails_closed(
    tmp_path: Path,
) -> None:
    train = _source(_training_batch(split="train"), "resume-train")
    val = _source(_training_batch(split="val"), "resume-val")
    config = TrainingConfig(
        stage="base",
        seed=2718,
        effective_global_batch=1,
        edge_budget=1,
        synthetic_contract=True,
    )

    uninterrupted = _base_system(initialization_seed=777, dropout=0.25)
    uninterrupted_report = PhaseSetTrainingRuntime(
        uninterrupted,
        config,
        tmp_path / "uninterrupted",
    ).fit(train, val, stop_after_global_step=2)

    first_leg = _base_system(initialization_seed=777, dropout=0.25)
    first_report = PhaseSetTrainingRuntime(
        first_leg,
        config,
        tmp_path / "resumed",
    ).fit(train, val, stop_after_global_step=1)
    resumed = _base_system(initialization_seed=5, dropout=0.25)
    resumed_report = PhaseSetTrainingRuntime(
        resumed,
        config,
        tmp_path / "resumed",
    ).fit(
        train,
        val,
        resume_checkpoint=first_report.latest_checkpoint.path,
        stop_after_global_step=2,
    )

    _assert_state_equal(_state_snapshot(uninterrupted), _state_snapshot(resumed))
    assert resumed_report.global_step == uninterrupted_report.global_step == 2
    assert resumed_report.completed_epochs == uninterrupted_report.completed_epochs == 2
    assert resumed_report.last_train_loss == uninterrupted_report.last_train_loss
    assert resumed_report.best_validation_metric == uninterrupted_report.best_validation_metric
    assert resumed_report.train_edges_seen == uninterrupted_report.train_edges_seen == 2

    raw = first_report.latest_checkpoint.path.read_bytes()
    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(raw[: max(1, len(raw) // 3)])
    broken = _base_system(initialization_seed=999, dropout=0.25)
    with pytest.raises(TrainingCheckpointError, match="truncated|unreadable"):
        PhaseSetTrainingRuntime(broken, config, tmp_path / "broken").fit(
            train,
            val,
            resume_checkpoint=truncated,
            stop_after_global_step=2,
        )

    payload = torch.load(
        first_report.latest_checkpoint.path,
        map_location="cpu",
        weights_only=True,
    )
    payload.pop("state_digest")
    payload["global_step"] = 0
    payload["state_digest"] = training_module._stable_hash(payload)
    forged_directory = tmp_path / "forged-cursor"
    forged_directory.mkdir()
    forged = forged_directory / first_report.latest_checkpoint.path.name
    torch.save(payload, forged)
    with pytest.raises(TrainingCheckpointError, match="cursor violates"):
        PhaseSetTrainingRuntime(
            _base_system(initialization_seed=999, dropout=0.25),
            config,
            tmp_path / "forged-runtime",
        ).fit(
            train,
            val,
            resume_checkpoint=forged,
            stop_after_global_step=2,
        )


def test_checkpoint_publish_is_write_once_for_existing_and_concurrent_targets(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.pt"
    original = b"existing-checkpoint-bytes"
    existing.write_bytes(original)
    with pytest.raises(TrainingCheckpointError, match="already exists"):
        training_module._atomic_torch_save(existing, {"global_step": 0})
    assert existing.read_bytes() == original

    target = tmp_path / "concurrent.pt"
    barrier = threading.Barrier(2)

    def publish(step: int) -> tuple[str, object]:
        barrier.wait()
        try:
            artifact = training_module._atomic_torch_save(
                target,
                {"global_step": step, "writer": step},
            )
        except TrainingCheckpointError as error:
            return "rejected", error
        return "published", artifact

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(publish, (1, 2)))
    published = [value for status, value in results if status == "published"]
    rejected = [value for status, value in results if status == "rejected"]
    assert len(published) == 1
    assert len(rejected) == 1
    assert "already exists" in str(rejected[0])
    artifact = published[0]
    assert isinstance(artifact, training_module.CheckpointArtifact)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == artifact.sha256
    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert payload["writer"] == artifact.global_step
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_resume_rejects_changed_optimizer_and_scheduler_state(tmp_path: Path) -> None:
    train = _source(_training_batch(split="train"), "state-validation-train")
    val = _source(_training_batch(split="val"), "state-validation-val")
    config = TrainingConfig(
        stage="base",
        seed=2718,
        effective_global_batch=1,
        edge_budget=1,
        synthetic_contract=True,
    )
    report = PhaseSetTrainingRuntime(
        _base_system(initialization_seed=333, dropout=0.25),
        config,
        tmp_path / "state-source",
    ).fit(train, val, stop_after_global_step=1)

    def change_weight_decay(payload: dict[str, object]) -> None:
        optimizer = payload["optimizer"]
        assert isinstance(optimizer, dict)
        groups = optimizer["param_groups"]
        assert isinstance(groups, list)
        groups[0]["weight_decay"] = 0.02

    def remove_adamw_moment(payload: dict[str, object]) -> None:
        optimizer = payload["optimizer"]
        assert isinstance(optimizer, dict)
        states = optimizer["state"]
        assert isinstance(states, dict)
        first = next(iter(states.values()))
        assert isinstance(first, dict)
        first.pop("exp_avg")

    def change_scheduler_step_count(payload: dict[str, object]) -> None:
        scheduler = payload["scheduler"]
        assert isinstance(scheduler, dict)
        scheduler["_step_count"] += 1

    cases = (
        (change_weight_decay, "parameter-group.*weight_decay"),
        (remove_adamw_moment, "incomplete or malformed"),
        (change_scheduler_step_count, "scheduler _step_count"),
    )
    for index, (mutate, message) in enumerate(cases):
        payload = torch.load(
            report.latest_checkpoint.path,
            map_location="cpu",
            weights_only=True,
        )
        mutate(payload)
        payload.pop("state_digest")
        payload["state_digest"] = training_module._stable_hash(payload)
        source = tmp_path / f"changed-state-{index}"
        source.mkdir()
        changed = source / report.latest_checkpoint.path.name
        torch.save(payload, changed)
        with pytest.raises(TrainingCheckpointError, match=message):
            PhaseSetTrainingRuntime(
                _base_system(initialization_seed=999, dropout=0.25),
                config,
                tmp_path / f"changed-state-runtime-{index}",
            ).fit(
                train,
                val,
                resume_checkpoint=changed,
                stop_after_global_step=2,
            )


def test_residual_training_freezes_base_and_updates_periodic_path(tmp_path: Path) -> None:
    train = _source(_training_batch(split="train", actors=3), "residual-train")
    val = _source(_training_batch(split="val", actors=3), "residual-val")
    torch.manual_seed(404)
    frozen_base = ActorMeanBase(hidden_dim=8, heads=2, ffn_dim=16, dropout=0.1)
    periodic = PhaseSetEncoder(embedding_dim=8, hidden_dim=8, edge_budget=3)
    head = PhaseSetRetrievalHead(
        embedding_dim=8,
        text_hidden_dim=8,
        residual_lambda_init=0.2,
    )
    system = ResidualRetrievalSystem(
        frozen_base,
        periodic,
        head,
        embedding_dim=8,
    )
    base_before = _state_snapshot(system.frozen_base)
    periodic_before = _state_snapshot(system.periodic_encoder)
    config = TrainingConfig(
        stage="residual",
        seed=31415,
        effective_global_batch=1,
        edge_budget=3,
        synthetic_contract=True,
    )
    report = PhaseSetTrainingRuntime(system, config, tmp_path / "residual").fit(
        train,
        val,
        stop_after_global_step=1,
    )

    _assert_state_equal(base_before, _state_snapshot(system.frozen_base))
    assert all(not parameter.requires_grad for parameter in system.frozen_base.parameters())
    assert system.frozen_base.training is False
    periodic_after = _state_snapshot(system.periodic_encoder)
    assert any(
        not torch.equal(periodic_before[name], periodic_after[name])
        for name in periodic_before
    )
    assert report.stage == "residual"
    assert report.system_id == "08"
    assert report.train_edges_seen == 3


def test_non_full_control_runs_and_head_rng_advances_across_updates(tmp_path: Path) -> None:
    train = _source(_training_batch(split="train", actors=3), "control-train")
    val = _source(_training_batch(split="val", actors=3), "control-val")
    torch.manual_seed(911)
    frozen_base = ActorMeanBase(hidden_dim=8, heads=2, ffn_dim=16, dropout=0.1)
    control = build_phaseset_system(
        "04",
        embedding_dim=8,
        hidden_dim=8,
        edge_budget=3,
    )
    head = _RandomConsumingHead()
    system = ResidualRetrievalSystem(
        frozen_base,
        control,
        head,
        embedding_dim=8,
    )
    config = TrainingConfig(
        stage="residual",
        seed=1729,
        effective_global_batch=1,
        edge_budget=3,
        synthetic_contract=True,
    )
    report = PhaseSetTrainingRuntime(system, config, tmp_path / "control-04").fit(
        train,
        val,
        stop_after_global_step=2,
    )
    assert report.system_id == "04"
    assert report.global_step == 2
    assert len(head.draws) == 2
    assert head.draws[0] != head.draws[1]


def test_unqualified_bf16_is_update_identical_and_test_or_resource_misuse_is_rejected(
    tmp_path: Path,
) -> None:
    train = _source(_training_batch(split="train"), "precision-train")
    val = _source(_training_batch(split="val"), "precision-val")
    fp32_config = TrainingConfig(
        stage="base",
        seed=1729,
        effective_global_batch=1,
        edge_budget=1,
        synthetic_contract=True,
    )
    fallback_config = TrainingConfig(
        stage="base",
        seed=1729,
        request_bf16=True,
        bf16_runtime_qualified=False,
        effective_global_batch=1,
        edge_budget=1,
        synthetic_contract=True,
    )
    fp32 = _base_system(initialization_seed=818, dropout=0.15)
    fallback = _base_system(initialization_seed=818, dropout=0.15)
    PhaseSetTrainingRuntime(fp32, fp32_config, tmp_path / "fp32").fit(
        train, val, stop_after_global_step=1
    )
    fallback_report = PhaseSetTrainingRuntime(
        fallback,
        fallback_config,
        tmp_path / "fallback",
    ).fit(train, val, stop_after_global_step=1)
    _assert_state_equal(_state_snapshot(fp32), _state_snapshot(fallback))
    assert fallback_report.precision.mode == "FP32"

    sealed = _Source(
        "test",
        None,
        hashlib.sha256(b"sealed-test").hexdigest(),
    )
    with pytest.raises(TrainingRuntimeError, match="sealed"):
        PhaseSetTrainingRuntime(
            _base_system(),
            fp32_config,
            tmp_path / "sealed",
        ).fit(train, sealed)  # type: ignore[arg-type]
    assert sealed.calls == []

    too_many_edges = _source(
        _training_batch(split="train", actors=3),
        "too-many-edges",
    )
    with pytest.raises(TrainingRuntimeError, match="edge count"):
        PhaseSetTrainingRuntime(
            _base_system(),
            fp32_config,
            tmp_path / "edges",
        ).fit(too_many_edges, val)


def test_formal_runtime_requires_seed_bound_initialization_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(stage="base", seed=1729)
    with pytest.raises(TrainingRuntimeError, match="seed-bound"):
        PhaseSetTrainingRuntime(
            build_registered_base_training_system("B0"),
            config,
            tmp_path / "unbound",
        )

    torch.manual_seed(1)
    first, first_binding = construct_registered_base_seed_bound_system("B0", config)
    torch.manual_seed(999)
    second, second_binding = construct_registered_base_seed_bound_system("B0", config)
    _assert_state_equal(_state_snapshot(first), _state_snapshot(second))
    assert first_binding == second_binding
    assert first_binding.seed == 1729
    assert first_binding.initial_optimizable_state_sha256 == training_system_state_sha256(
        first,
        optimizable_only=True,
    )
    runtime = PhaseSetTrainingRuntime(
        first,
        config,
        tmp_path / "bound",
        initialization_binding=first_binding,
    )
    assert runtime.initialization_binding == first_binding

    torch.manual_seed(1729)
    low_dropout = _factory_base()
    torch.manual_seed(1729)
    high_dropout = BaseRetrievalSystem(
        ActorMeanBase(hidden_dim=8, heads=2, ffn_dim=16, dropout=0.75),
        embedding_dim=8,
    )
    assert training_system_state_sha256(low_dropout) == training_system_state_sha256(
        high_dropout
    )
    assert training_system_behavior_sha256(low_dropout) != training_system_behavior_sha256(
        high_dropout
    )

    with pytest.raises(TrainingRuntimeError, match="closed registered"):
        construct_seed_bound_system(
            lambda: build_registered_base_training_system("B0"),
            config,
            factory_sha256=hashlib.sha256(b"self-signed").hexdigest(),
        )

    with torch.no_grad():
        next(second.parameters()).add_(1.0)
    with pytest.raises(TrainingRuntimeError, match="live training system"):
        PhaseSetTrainingRuntime(
            second,
            config,
            tmp_path / "tampered",
            initialization_binding=second_binding,
        )

    live_dropout = next(module for module in first.modules() if isinstance(module, nn.Dropout))
    live_dropout.p = 0.8
    first._phaseset_registered_behavior_sha256 = training_system_behavior_sha256(first)
    with pytest.raises(TrainingRuntimeError, match="behavior changed"):
        runtime.fit(_source(_training_batch(split="train"), "train"), _source(
            _training_batch(split="val"),
            "val",
        ))


def test_formal_resume_requires_expected_digest_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingConfig(stage="base", seed=1729)
    system, binding = construct_registered_base_seed_bound_system("B0", config)
    train = _source(_training_batch(split="train", rows=128), "formal-resume-train")
    val = _source(_training_batch(split="val"), "formal-resume-val")
    report = PhaseSetTrainingRuntime(
        system,
        config,
        tmp_path / "formal-resume-source",
        initialization_binding=binding,
    ).fit(train, val, stop_after_global_step=0)

    ledger_root = tmp_path / "formal-resume-ledger"
    ledger_root.mkdir()
    run_id = experiments.base_run_ids()[0]
    ledger_digest = lambda label: hashlib.sha256(label.encode("ascii")).hexdigest()  # noqa: E731
    predecessor_attempt = execution.AttemptRecord(
        attempt_id="formal-resume-source",
        run_id=run_id,
        created_at_utc="2026-08-26T00:00:00Z",
        plan_sha256=ledger_digest("resume-plan"),
        matrix_sha256=ledger_digest("resume-matrix"),
        training_config_sha256=ledger_digest("resume-config"),
        source_tree_sha256=ledger_digest("resume-source"),
    )
    predecessor = execution.AttemptStore.create(ledger_root, predecessor_attempt)
    checkpoint_receipt_sha256 = predecessor.write_checkpoint(
        execution.CheckpointRecord(
            attempt_id=predecessor_attempt.attempt_id,
            run_id=run_id,
            epoch_index=0,
            global_step=report.latest_checkpoint.global_step,
            written_at_utc="2026-08-26T00:00:01Z",
            model_state_sha256=ledger_digest("resume-model"),
            optimizer_state_sha256=ledger_digest("resume-optimizer"),
            rng_state_sha256=ledger_digest("resume-rng"),
            sampler_state_sha256=ledger_digest("resume-sampler"),
            dataloader_state_sha256=ledger_digest("resume-loader"),
            dropout_state_sha256=ledger_digest("resume-dropout"),
            validation_state_sha256=ledger_digest("resume-validation"),
            checkpoint_payload_sha256=report.latest_checkpoint.sha256,
            parent_checkpoint_receipt_sha256=None,
        )
    )
    predecessor_terminal_sha256 = predecessor.write_terminal(
        execution.TerminalRecord(
            attempt_id=predecessor_attempt.attempt_id,
            run_id=run_id,
            completed_at_utc="2026-08-26T00:00:02Z",
            outcome="FAILED",
            attempt_receipt_sha256=execution.artifact_sha256(
                (predecessor.path / "attempt.json").read_bytes()
            ),
            latest_heartbeat_sha256=None,
            latest_checkpoint_receipt_sha256=checkpoint_receipt_sha256,
            failure_code="INFRA_TRANSIENT",
        )
    )
    resumed_attempt = execution.AttemptRecord(
        attempt_id="formal-resume-target",
        run_id=run_id,
        created_at_utc="2026-08-26T00:00:03Z",
        plan_sha256=predecessor_attempt.plan_sha256,
        matrix_sha256=predecessor_attempt.matrix_sha256,
        training_config_sha256=predecessor_attempt.training_config_sha256,
        source_tree_sha256=predecessor_attempt.source_tree_sha256,
    )
    resume_record = execution.ResumeRecord(
        new_attempt_id=resumed_attempt.attempt_id,
        predecessor_attempt_id=predecessor_attempt.attempt_id,
        run_id=run_id,
        created_at_utc=resumed_attempt.created_at_utc,
        predecessor_terminal_sha256=predecessor_terminal_sha256,
        checkpoint_receipt_sha256=checkpoint_receipt_sha256,
        corrective_change_sha256=ledger_digest("resume-corrective-change"),
        retry_class="INFRA_TRANSIENT",
    )
    execution.AttemptStore.create_resumed(ledger_root, resumed_attempt, resume_record)

    load_calls = 0

    def forbidden_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("checkpoint bytes must be checked before deserialization")

    with monkeypatch.context() as digest_check:
        digest_check.setattr(training_module.torch, "load", forbidden_load)
        missing_system, missing_binding = construct_registered_base_seed_bound_system(
            "B0", config
        )
        with pytest.raises(TrainingCheckpointError, match="attempt-ledger binding"):
            PhaseSetTrainingRuntime(
                missing_system,
                config,
                tmp_path / "formal-resume-missing",
                initialization_binding=missing_binding,
            ).fit(
                train,
                val,
                resume_checkpoint=report.latest_checkpoint.path,
                stop_after_global_step=0,
            )
        wrong_system, wrong_binding = construct_registered_base_seed_bound_system(
            "B0", config
        )
        with pytest.raises(
            TrainingCheckpointError,
            match="caller checkpoint digest differs from verified attempt ledger",
        ):
            PhaseSetTrainingRuntime(
                wrong_system,
                config,
                tmp_path / "formal-resume-wrong",
                initialization_binding=wrong_binding,
            ).fit(
                train,
                val,
                resume_checkpoint=report.latest_checkpoint.path,
                resume_checkpoint_sha256="0" * 64,
                resume_attempt_root=ledger_root,
                resume_record=resume_record,
                stop_after_global_step=0,
            )
    assert load_calls == 0

    resumed_system, resumed_binding = construct_registered_base_seed_bound_system(
        "B0", config
    )
    resumed = PhaseSetTrainingRuntime(
        resumed_system,
        config,
        tmp_path / "formal-resume-correct",
        initialization_binding=resumed_binding,
    ).fit(
        train,
        val,
        resume_checkpoint=report.latest_checkpoint.path,
        resume_attempt_root=ledger_root,
        resume_record=resume_record,
        stop_after_global_step=0,
    )
    assert resumed.latest_checkpoint.sha256 == report.latest_checkpoint.sha256


def test_formal_runtime_rejects_instance_and_class_method_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingConfig(stage="base", seed=1729)
    system, binding = construct_registered_base_seed_bound_system("B0", config)

    def fake_scores(self: object, motion: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        del self
        return torch.zeros((motion.shape[0], text.shape[0]), dtype=torch.float32)

    system.scores = types.MethodType(fake_scores, system)  # type: ignore[method-assign]
    with pytest.raises(TrainingRuntimeError, match="callable override"):
        PhaseSetTrainingRuntime(
            system,
            config,
            tmp_path / "instance-method-override",
            initialization_binding=binding,
        )

    clean, clean_binding = construct_registered_base_seed_bound_system("B0", config)
    with monkeypatch.context() as method_patch:
        method_patch.setattr(BaseRetrievalSystem, "scores", fake_scores)
        with pytest.raises(TrainingRuntimeError, match="method identity changed"):
            PhaseSetTrainingRuntime(
                clean,
                config,
                tmp_path / "class-method-override",
                initialization_binding=clean_binding,
            )

    hooked, hooked_binding = construct_registered_base_seed_bound_system("B0", config)
    hooked.group_base.register_forward_hook(lambda _module, _args, output: output)
    with pytest.raises(TrainingRuntimeError, match="runtime hook"):
        PhaseSetTrainingRuntime(
            hooked,
            config,
            tmp_path / "forward-hook",
            initialization_binding=hooked_binding,
        )

    gradient_hooked, gradient_binding = construct_registered_base_seed_bound_system(
        "B0", config
    )
    next(gradient_hooked.parameters()).register_hook(lambda gradient: gradient * 0.0)
    with pytest.raises(TrainingRuntimeError, match="gradient hook"):
        PhaseSetTrainingRuntime(
            gradient_hooked,
            config,
            tmp_path / "gradient-hook",
            initialization_binding=gradient_binding,
        )

    torch_class, torch_binding = construct_registered_base_seed_bound_system(
        "B0", config
    )
    original_linear_forward = nn.Linear.forward

    def fake_linear_forward(layer: nn.Linear, value: torch.Tensor) -> torch.Tensor:
        return original_linear_forward(layer, value) * 0.0

    with monkeypatch.context() as torch_patch:
        torch_patch.setattr(nn.Linear, "forward", fake_linear_forward)
        with pytest.raises(TrainingRuntimeError, match="method identity changed"):
            PhaseSetTrainingRuntime(
                torch_class,
                config,
                tmp_path / "torch-class-method-override",
                initialization_binding=torch_binding,
            )

    constant_system, constant_binding = construct_registered_base_seed_bound_system(
        "B0", config
    )
    with monkeypatch.context() as constant_patch:
        constant_patch.setattr(training_module, "WEIGHT_DECAY", 0.99)
        with pytest.raises(TrainingRuntimeError, match="runtime constant changed"):
            PhaseSetTrainingRuntime(
                constant_system,
                config,
                tmp_path / "weight-decay-override",
                initialization_binding=constant_binding,
            )

    objective_system, objective_binding = construct_registered_base_seed_bound_system(
        "B0", config
    )

    def fake_objective(logits: torch.Tensor, positives: torch.Tensor):
        zero = logits.sum() * 0.0 + positives.sum() * 0.0
        return zero, zero, zero

    with monkeypatch.context() as objective_patch:
        objective_patch.setattr(
            training_module,
            "variable_positive_symmetric_infonce",
            fake_objective,
        )
        with pytest.raises(TrainingRuntimeError, match="training function identity changed"):
            PhaseSetTrainingRuntime(
                objective_system,
                config,
                tmp_path / "objective-override",
                initialization_binding=objective_binding,
            )


def test_numerical_runtime_is_frozen_checked_and_restored(
    tmp_path: Path,
) -> None:
    original = training_module._capture_numerical_runtime_flags()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.set_float32_matmul_precision("medium")
        torch.backends.mha.set_fastpath_enabled(True)
        ambient = training_module._capture_numerical_runtime_flags()

        formal_config = TrainingConfig(stage="base", seed=1729)
        formal_system, formal_binding = construct_registered_base_seed_bound_system(
            "B0",
            formal_config,
        )
        formal_runtime = PhaseSetTrainingRuntime(
            formal_system,
            formal_config,
            tmp_path / "numerical-formal",
            initialization_binding=formal_binding,
        )
        with training_module._frozen_numerical_runtime(torch.device("cpu")):
            training_module._assert_frozen_numerical_runtime(torch.device("cpu"))
            assert torch.backends.mha.get_fastpath_enabled() is False
            formal_runtime._numeric_runtime_active = True
            torch.backends.mha.set_fastpath_enabled(True)
            with pytest.raises(TrainingRuntimeError, match="numerical runtime flags changed"):
                formal_runtime._assert_live_formal_binding(require_initial_state=True)
            formal_runtime._numeric_runtime_active = False
        assert training_module._capture_numerical_runtime_flags() == ambient

        with pytest.raises(RuntimeError, match="intentional numerical runtime exit"):
            with training_module._frozen_numerical_runtime(torch.device("cpu")):
                assert torch.backends.mha.get_fastpath_enabled() is False
                raise RuntimeError("intentional numerical runtime exit")
        assert training_module._capture_numerical_runtime_flags() == ambient

        synthetic_config = TrainingConfig(
            stage="base",
            seed=1729,
            effective_global_batch=1,
            edge_budget=1,
            synthetic_contract=True,
        )
        report = PhaseSetTrainingRuntime(
            _base_system(dropout=0.1),
            synthetic_config,
            tmp_path / "numerical-synthetic",
        ).fit(
            _source(_training_batch(split="train"), "numerical-train"),
            _source(_training_batch(split="val"), "numerical-val"),
            stop_after_global_step=1,
        )
        assert report.global_step == 1
        assert training_module._capture_numerical_runtime_flags() == ambient
    finally:
        training_module._restore_numerical_runtime_flags(original)


def test_validation_endpoint_is_capture_macro_not_query_micro() -> None:
    family_a = _digest("capture-family-a")
    family_b = _digest("capture-family-b")
    motion_ids = (family_a, family_a, family_a, family_b)
    text_ids = (family_a, family_a, family_a, family_b)
    positive = torch.tensor(
        [[left == right for right in text_ids] for left in motion_ids],
        dtype=torch.bool,
    )
    logits = torch.tensor(
        [
            [30.0, 0.0, 0.0, 10.0],
            [0.0, 20.0, 0.0, 0.0],
            [0.0, 0.0, 20.0, 0.0],
            [25.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    motion_commitments = tuple(_digest(f"macro-motion-{index}") for index in range(4))
    text_commitments = tuple(_digest(f"macro-text-{index}") for index in range(4))
    metric = training_module._capture_macro_bidirectional_r1(
        logits,
        positive,
        motion_ids,
        text_ids,
        motion_commitments,
        text_commitments,
    )
    row_hit = positive.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean()
    column_hit = positive.gather(0, logits.argmax(dim=0, keepdim=True)).float().mean()
    query_micro = float((0.5 * (row_hit + column_hit)).item())
    assert metric == 0.5
    assert query_micro == 0.75


def test_formal_resume_rejects_verifier_monkeypatch_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingConfig(stage="base", seed=1729)
    system, binding = construct_registered_base_seed_bound_system("B0", config)
    runtime = PhaseSetTrainingRuntime(
        system,
        config,
        tmp_path / "formal-resume-verifier-binding",
        initialization_binding=binding,
    )
    forged_verifier_called = False

    def forged_verifier(*_args: object, **_kwargs: object) -> str:
        nonlocal forged_verifier_called
        forged_verifier_called = True
        return "0" * 64

    with monkeypatch.context() as verifier_patch:
        verifier_patch.setattr(
            execution,
            "verified_resume_checkpoint_payload_sha256",
            forged_verifier,
        )
        with pytest.raises(
            TrainingRuntimeError,
            match="formal external function identity changed",
        ):
            runtime.fit(
                _source(_training_batch(split="train"), "never-read-train"),
                _source(_training_batch(split="val"), "never-read-val"),
                resume_checkpoint=tmp_path / "never-deserialized.pt",
            )
    assert forged_verifier_called is False


def test_residual_seed_binding_requires_and_binds_frozen_base_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingConfig(stage="residual", seed=2718)
    base_config = TrainingConfig(stage="base", seed=config.seed)
    base_system, base_binding = construct_registered_base_seed_bound_system(
        "B0",
        base_config,
    )
    base_report = PhaseSetTrainingRuntime(
        base_system,
        base_config,
        tmp_path / "formal-base",
        initialization_binding=base_binding,
    ).fit(
        _source(
            _training_batch(split="train", rows=128, embedding_dim=512),
            "formal-base-train",
        ),
        _source(
            _training_batch(split="val", embedding_dim=512),
            "formal-base-val",
        ),
        stop_after_global_step=1,
    )
    assert base_report.best_checkpoint is not None
    selected_base_checkpoint = base_report.best_checkpoint
    def qualification_for(checkpoint_sha256: str) -> experiments.BaseQualification:
        rows: list[experiments.BaseScore] = []
        for index, run_id in enumerate(experiments.base_run_ids(), start=1):
            _, row_seed, system_id = experiments.parse_run_id(run_id)
            selected = hashlib.sha256(f"selected-{run_id}".encode("ascii")).hexdigest()
            if system_id == "B0" and row_seed == config.seed:
                selected = checkpoint_sha256
            rows.append(
                experiments.BaseScore(
                    run_id=run_id,
                    bidirectional_r1_numerator={"B0": 80, "B1": 70, "B2": 60}[
                        system_id
                    ],
                    bidirectional_r1_denominator=100,
                    parameter_count={"B0": 100, "B1": 110, "B2": 120}[system_id],
                    frozen_runtime_latency_ns={"B0": 1_000, "B1": 1_100, "B2": 1_200}[
                        system_id
                    ],
                    terminal_sha256=hashlib.sha256(
                        f"terminal-{index}-{run_id}".encode("ascii")
                    ).hexdigest(),
                    selected_checkpoint_sha256=selected,
                    split="validation",
                    validation_manifest_sha256=hashlib.sha256(
                        b"base-validation-manifest"
                    ).hexdigest(),
                    query_census_sha256=hashlib.sha256(
                        b"base-validation-census"
                    ).hexdigest(),
                    evaluator_sha256=hashlib.sha256(b"base-evaluator").hexdigest(),
                    score_artifact_sha256=hashlib.sha256(
                        f"base-score-{run_id}".encode("ascii")
                    ).hexdigest(),
                )
            )
        return experiments.qualify_base(tuple(rows))

    qualification = qualification_for(selected_base_checkpoint.sha256)
    selection_sha = hashlib.sha256(
        experiments.canonical_base_qualification_bytes(qualification)
    ).hexdigest()
    update_path = next(
        path
        for path in selected_base_checkpoint.path.parent.glob("*-update.pt")
        if path.is_file()
    )
    update_sha = hashlib.sha256(update_path.read_bytes()).hexdigest()
    update_qualification = qualification_for(update_sha)
    update_selection_sha = hashlib.sha256(
        experiments.canonical_base_qualification_bytes(update_qualification)
    ).hexdigest()
    with pytest.raises(TrainingCheckpointError, match="validation-selected best"):
        load_qualified_frozen_base(
            update_path,
            qualification=update_qualification,
            seed=config.seed,
            expected_qualification_sha256=update_selection_sha,
        )

    qualified = load_qualified_frozen_base(
        selected_base_checkpoint.path,
        qualification=qualification,
        seed=config.seed,
        expected_qualification_sha256=selection_sha,
    )
    alternate_rows = tuple(
        replace(
            row,
            bidirectional_r1_numerator=59,
            bidirectional_r1_denominator=100,
        )
        if row.run_id.endswith("/B2")
        else row
        for row in qualification.score_rows
    )
    alternate_qualification = experiments.qualify_base(alternate_rows)
    alternate_selection_sha = hashlib.sha256(
        experiments.canonical_base_qualification_bytes(alternate_qualification)
    ).hexdigest()
    assert alternate_qualification.winner_system_id == qualification.winner_system_id
    forged_qualified = replace(
        qualified,
        qualification=alternate_qualification,
        selection_receipt_sha256=alternate_selection_sha,
    )
    with monkeypatch.context() as early_gate_patch:
        early_gate_patch.setattr(
            training_module,
            "_rebuild_registered_residual_capacity_audit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cohort construction crossed the external qualification gate")
            ),
        )
        with pytest.raises(TrainingCheckpointError, match="trusted selection"):
            build_registered_residual_capacity_audit(
                forged_qualified,
                config,
                energy_floors=np.zeros(6, dtype=np.float64),
                expected_qualification_sha256=selection_sha,
            )
    audit = build_registered_residual_capacity_audit(
        qualified,
        config,
        energy_floors=np.zeros(6, dtype=np.float64),
        expected_qualification_sha256=selection_sha,
    )
    assert isinstance(audit, ResidualCapacityAudit)
    with monkeypatch.context() as early_gate_patch:
        early_gate_patch.setattr(
            training_module,
            "_rebuild_registered_residual_capacity_audit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cohort construction crossed the external qualification gate")
            ),
        )
        with pytest.raises(TrainingCheckpointError, match="trusted selection"):
            construct_registered_residual_seed_bound_system(
                "04",
                forged_qualified,
                config,
                energy_floors=np.zeros(6, dtype=np.float64),
                residual_capacity_audit=audit,
                expected_qualification_sha256=selection_sha,
                expected_capacity_audit_sha256=audit.sha256,
            )

    tampered_group = copy.deepcopy(qualified.group_base)
    with torch.no_grad():
        next(tampered_group.parameters()).add_(1.0)
    tampered_group.eval()
    for parameter in tampered_group.parameters():
        parameter.requires_grad_(False)
    tampered_state = training_module._frozen_base_score_state_sha256(
        tampered_group,
        qualified.frozen_logit_scale,
    )
    with pytest.raises(
        TrainingCheckpointError,
        match="module and temperature differ from checkpoint evidence",
    ):
        replace(
            qualified,
            group_base=tampered_group,
            frozen_state_sha256=tampered_state,
        )

    tampered_behavior = copy.deepcopy(qualified.group_base)
    next(
        module for module in tampered_behavior.modules() if isinstance(module, nn.Dropout)
    ).p = 0.25
    tampered_behavior.eval()
    for parameter in tampered_behavior.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(
        TrainingCheckpointError,
        match="behavior differs from checkpoint evidence",
    ):
        replace(qualified, group_base=tampered_behavior)

    tampered_audit = replace(
        audit,
        rows=tuple(
            (system_id, count, "f" * 64)
            for system_id, count, _behavior_sha256 in audit.rows
        ),
    )
    assert tampered_audit.sha256 != audit.sha256
    with monkeypatch.context() as early_gate_patch:
        early_gate_patch.setattr(
            training_module,
            "_rebuild_registered_residual_capacity_audit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cohort construction crossed the external capacity gate")
            ),
        )
        with pytest.raises(TrainingRuntimeError, match="trusted expected digest"):
            construct_registered_residual_seed_bound_system(
                "04",
                qualified,
                config,
                energy_floors=np.zeros(6, dtype=np.float64),
                residual_capacity_audit=tampered_audit,
                expected_qualification_sha256=selection_sha,
                expected_capacity_audit_sha256=audit.sha256,
            )

    with pytest.raises(TrainingCheckpointError, match="digest mismatch"):
        load_qualified_frozen_base(
            selected_base_checkpoint.path,
            qualification=qualification_for("0" * 64),
            seed=config.seed,
            expected_qualification_sha256=hashlib.sha256(
                experiments.canonical_base_qualification_bytes(
                    qualification_for("0" * 64)
                )
            ).hexdigest(),
        )
    system, binding = construct_registered_residual_seed_bound_system(
        "04",
        qualified,
        config,
        energy_floors=np.zeros(6, dtype=np.float64),
        residual_capacity_audit=audit,
        expected_qualification_sha256=selection_sha,
        expected_capacity_audit_sha256=audit.sha256,
    )
    assert isinstance(binding, TrainingInitializationBinding)
    assert binding.system_id == "04"
    assert binding.frozen_base_checkpoint_sha256 == selected_base_checkpoint.sha256
    assert binding.frozen_base_state_sha256 is not None
    assert binding.qualified_base_selection_sha256 == selection_sha
    assert binding.residual_capacity_audit_sha256 == audit.sha256
    runtime = PhaseSetTrainingRuntime(
        system,
        config,
        tmp_path / "residual-bound",
        initialization_binding=binding,
    )
    assert runtime.initialization_binding == binding

    groups = _prepared_groups(actors=2)
    text = torch.randn(
        (1, 512),
        generator=torch.Generator().manual_seed(8128),
        dtype=torch.float32,
    ).contiguous()
    base_system.eval()
    system.eval()
    with torch.no_grad():
        selected_base_scores = base_system.scores(
            base_system.encode_trainable(groups),
            text,
        )
        tokens, band_mask = system.encode_trainable(groups)
        residual_scores = system.scores(
            tokens,
            band_mask,
            system.encode_frozen_base(groups),
            text,
        )
    assert torch.equal(selected_base_scores, residual_scores)
    assert not hasattr(system, "logit_scale")
    assert float(system._frozen_base_logit_scale.item()) == qualified.frozen_logit_scale


def test_complete_residual_capacity_audit_includes_text_head_and_bounded_scale() -> None:
    def system(system_id: str, *, text_hidden_dim: int = 8) -> ResidualRetrievalSystem:
        return ResidualRetrievalSystem(
            ActorMeanBase(hidden_dim=8, heads=2, ffn_dim=16, dropout=0.1),
            build_phaseset_system(
                system_id,
                embedding_dim=8,
                hidden_dim=8,
                energy_floors=np.zeros(6, dtype=np.float64),
            ),
            PhaseSetRetrievalHead(
                embedding_dim=8,
                text_hidden_dim=text_hidden_dim,
                residual_lambda_init=0.0,
            ),
            embedding_dim=8,
        )

    matched = {system_id: system(system_id) for system_id in (f"{i:02d}" for i in range(1, 9))}
    audit = audit_residual_training_parameter_counts(matched)
    assert len({count for _, count in audit}) == 1

    mismatched = dict(matched)
    mismatched["01"] = system("01", text_hidden_dim=2)
    with pytest.raises(TrainingRuntimeError, match="capacity bound"):
        audit_residual_training_parameter_counts(mismatched)
