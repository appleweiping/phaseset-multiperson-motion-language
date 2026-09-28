"""Focused engineering checks for the cached residual lifecycle seam.

The fixtures are analytic and grant no data, training, or result authority.
The complete descriptor plan is the real sealed 20-epoch plan; test-only
monkeypatches keep checkpoint lifecycle work small without weakening its
production admission rules.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from phaseset_core import capture_validation, execution, prepared_data_v2, training
from phaseset_core.contracts import PreparedGroupBatch, validate_prepared_group_batch
from phaseset_core.controls import PhaseSetSystem
from phaseset_core.models import GroupBaseOutput
from phaseset_core.objectives import PhaseSetRetrievalHead
from phaseset_core.periodic_capture_training_cache import (
    PeriodicCaptureTrainingCacheError,
)
from test_periodic_capture_training_cache import _FLOORS, _admit, _bundle


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _TinyFrozenBase(nn.Module):
    """Small deterministic base used only by the engineering lifecycle tests."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer(
            "template",
            torch.linspace(0.125, 0.5, width, dtype=torch.float32),
        )

    def forward(self, groups: PreparedGroupBatch) -> GroupBaseOutput:
        checked = validate_prepared_group_batch(groups)
        device = self.template.device
        actor_mask = torch.tensor(checked.actor_mask, dtype=torch.bool, device=device)
        group = self.template[None, :].expand(checked.batch_size, -1).contiguous()
        actors = group[:, None, :].expand(-1, checked.skeletons.shape[1], -1).contiguous()
        actors = torch.where(actor_mask[..., None], actors, torch.zeros_like(actors))
        return GroupBaseOutput(group, actors, actor_mask.contiguous())


def _tiny_system(system_id: str = "08") -> training.ResidualRetrievalSystem:
    torch.manual_seed(7301)
    periodic = PhaseSetSystem(
        system_id,
        embedding_dim=4,
        hidden_dim=4,
        energy_floors=_FLOORS,
        edge_budget=32_768,
    )
    return training.ResidualRetrievalSystem(
        _TinyFrozenBase(4),
        periodic,
        PhaseSetRetrievalHead(4, text_hidden_dim=4, residual_lambda_init=0.0),
        embedding_dim=4,
        frozen_base_logit_scale=1.0,
    )


def _binding(
    system: training.ResidualRetrievalSystem,
    config: training.TrainingConfig,
    plan_sha256: str | None,
) -> training.TrainingInitializationBinding:
    return training.TrainingInitializationBinding(
        stage="residual",
        seed=config.seed,
        system_id=system.system_id,
        factory_sha256=_digest("fixture/factory"),
        code_artifact_sha256=training.training_code_artifact_sha256(),
        environment_sha256=training.training_environment_sha256(),
        behavior_sha256=training.training_system_behavior_sha256(system),
        initial_optimizable_state_sha256=training.training_system_state_sha256(
            system,
            optimizable_only=True,
        ),
        frozen_base_checkpoint_sha256=_digest("fixture/base-checkpoint"),
        frozen_base_state_sha256=training._frozen_base_score_state_sha256(
            system.frozen_base,
            float(system._frozen_base_logit_scale.item()),
        ),
        qualified_base_selection_sha256=_digest("fixture/base-selection"),
        residual_capacity_audit_sha256=_digest("fixture/capacity"),
        periodic_descriptor_capture_plan_sha256=plan_sha256,
        schema=(
            training.CACHED_INITIALIZATION_SCHEMA
            if plan_sha256 is not None
            else "phaseset-training-initialization-binding-v1"
        ),
    )


def _runtime(
    root: Path,
    plan,
    monkeypatch: pytest.MonkeyPatch,
    *,
    system_id: str = "08",
) -> training.PhaseSetTrainingRuntime:
    monkeypatch.setattr(training, "_validate_registered_formal_system", lambda *_: None)
    system = _tiny_system(system_id)
    config = plan_config(plan)
    binding = _binding(system, config, plan.sha256)
    return training.PhaseSetTrainingRuntime(
        system,
        config,
        root,
        initialization_binding=binding,
        descriptor_plan=plan,
    )


def plan_config(plan) -> training.TrainingConfig:
    return training.TrainingConfig(
        stage="residual",
        seed=plan.seed,
        edge_budget=plan.edge_budget,
    )


def _narrow_text(
    batch: training.RetrievalTrainingBatch,
) -> training.RetrievalTrainingBatch:
    return training.RetrievalTrainingBatch(
        groups=batch.groups,
        text_embeddings=batch.text_embeddings[:, :4].clone().contiguous(),
        motion_positive_ids=batch.motion_positive_ids,
        text_positive_ids=batch.text_positive_ids,
        text_commitments=batch.text_commitments,
        split=batch.split,
        descriptor_contexts=batch.descriptor_contexts,
    )


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _tiny_optimizer_update(
    self: training.PhaseSetTrainingRuntime,
    _microbatches: tuple[training.RetrievalTrainingBatch, ...],
) -> float:
    optimizer = self._checked_optimizer
    optimizer.zero_grad(set_to_none=True)
    terms = [parameter.reshape(-1)[0] for parameter in self._trainable_parameters()]
    loss = torch.stack(terms).sum()
    loss.backward()
    optimizer.step()
    self._checked_scheduler.step()
    return float(loss.detach().cpu().item())


def _fixed_validation(
    _self: training.PhaseSetTrainingRuntime,
    _source: object,
) -> tuple[float, float, int, int]:
    return 0.5, 0.25, 3, 3


def _resume_record() -> execution.ResumeRecord:
    return execution.ResumeRecord(
        new_attempt_id="a" * 32,
        predecessor_attempt_id="b" * 32,
        run_id="08-seed1729-20990101T000000Z",
        created_at_utc="2099-01-01T00:00:00Z",
        predecessor_terminal_sha256=_digest("fixture/terminal"),
        checkpoint_receipt_sha256=_digest("fixture/checkpoint-receipt"),
        corrective_change_sha256=_digest("fixture/corrective-change"),
        retry_class="IMPLEMENTATION",
    )


def test_uncached_initialization_and_checkpoint_keep_v1_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _tiny_system()
    config = training.TrainingConfig(stage="residual", seed=1729)
    binding = _binding(system, config, None)
    value = json.loads(binding.canonical_bytes())

    assert value["schema"] == "phaseset-training-initialization-binding-v1"
    assert "periodic_descriptor_capture_plan_sha256" not in value
    assert binding.periodic_descriptor_capture_plan_sha256 is None
    positional = training.TrainingInitializationBinding(
        binding.stage,
        binding.seed,
        binding.system_id,
        binding.factory_sha256,
        binding.code_artifact_sha256,
        binding.environment_sha256,
        binding.behavior_sha256,
        binding.initial_optimizable_state_sha256,
        binding.frozen_base_checkpoint_sha256,
        binding.frozen_base_state_sha256,
        binding.qualified_base_selection_sha256,
        binding.residual_capacity_audit_sha256,
        binding.schema,
    )
    assert positional == binding
    monkeypatch.setattr(training, "_validate_registered_formal_system", lambda *_: None)
    runtime = training.PhaseSetTrainingRuntime(
        system,
        config,
        tmp_path / "uncached",
        initialization_binding=binding,
    )
    runtime._train_manifest = _digest("fixture/train")
    runtime._val_manifest = _digest("fixture/val")
    runtime._initialize_optimizer(1)
    runtime._seed_new_run()
    artifact = runtime._write_checkpoint("update")
    payload, _ = training._load_torch_checkpoint(
        artifact.path,
        expected_sha256=artifact.sha256,
    )
    assert payload["schema"] == training.CHECKPOINT_SCHEMA
    assert "periodic_descriptor_capture_plan_sha256" not in payload


def test_cached_gradient_cache_reopens_stream_for_first_pass_and_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    plan = _admit(bundle, system_id="08")
    runtime = _runtime(tmp_path / "runtime", plan, monkeypatch)
    runtime._initialize_optimizer(1)
    batch = _narrow_text(next(bundle.train_source.iter_epoch(epoch=0, seed=1729)))
    calls: list[training.RetrievalTrainingBatch] = []
    plan_type = type(plan)
    original = plan_type.open_training_batch

    def counted(self, value):
        calls.append(value)
        return original(self, value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cached lifecycle must not call the uncached encoder")

    monkeypatch.setattr(plan_type, "open_training_batch", counted)
    monkeypatch.setattr(training.ResidualRetrievalSystem, "encode_trainable", forbidden)
    loss = runtime._train_update((batch,))

    assert np.isfinite(loss)
    assert calls == [batch, batch]


def test_cached_replay_selector_failure_restores_rng_and_never_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    plan = _admit(bundle, system_id="08")
    runtime = _runtime(tmp_path / "runtime", plan, monkeypatch)
    runtime._initialize_optimizer(1)
    batch = _narrow_text(next(bundle.train_source.iter_epoch(epoch=0, seed=1729)))
    before_parameters = _state(runtime.system)
    before_scheduler = runtime._checked_scheduler.state_dict()
    plan_type = type(plan)
    original_open = plan_type.open_training_batch
    original_restore = training._restore_rng
    open_calls = 0
    restored_rng_sha256: list[str] = []

    def fail_replay(self, value):
        nonlocal open_calls
        open_calls += 1
        if open_calls == 2:
            raise PeriodicCaptureTrainingCacheError("fixture replay selector failure")
        return original_open(self, value)

    def record_restore(value):
        restored_rng_sha256.append(training._stable_hash(value))
        original_restore(value)

    def forbidden_step(*_args, **_kwargs):
        raise AssertionError("optimizer step must not run after replay failure")

    monkeypatch.setattr(plan_type, "open_training_batch", fail_replay)
    monkeypatch.setattr(training, "_restore_rng", record_restore)
    monkeypatch.setattr(runtime._checked_optimizer, "step", forbidden_step)

    with pytest.raises(
        PeriodicCaptureTrainingCacheError,
        match="fixture replay selector failure",
    ):
        runtime._train_update((batch,))

    assert open_calls == 2
    assert len(restored_rng_sha256) == 2
    assert training._stable_hash(training._capture_rng()) == restored_rng_sha256[-1]
    _assert_state_equal(before_parameters, _state(runtime.system))
    assert runtime._checked_scheduler.state_dict() == before_scheduler


def test_capture_validation_dispatches_exact_cached_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    plan = _admit(bundle, system_id="08")
    runtime = _runtime(tmp_path / "runtime", plan, monkeypatch)
    runtime._val_manifest = bundle.capture_source.census_sha256
    observed: list[tuple[object, ...]] = []

    def cached(system, config, source, descriptor_plan):
        observed.append((system, config, source, descriptor_plan))
        return SimpleNamespace(
            primary_capture_r1=Fraction(1, 2),
            loss=0.75,
            source_census_sha256=source.census_sha256,
        )

    monkeypatch.setattr(
        capture_validation,
        "_verified_capture_validation_bindings",
        lambda: (
            capture_validation.CaptureValidationSource,
            capture_validation.run_capture_validation,
        ),
    )
    monkeypatch.setattr(capture_validation, "run_capture_validation_cached", cached)
    metric, loss, edges, maximum = runtime._validation(bundle.capture_source)

    expected_edges = tuple(
        window.groups.actor_counts[0] * (window.groups.actor_counts[0] - 1) // 2
        for capture in bundle.capture_source.captures
        for window in capture.windows
    )
    assert len(observed) == 1
    actual_system, actual_config, actual_source, actual_plan = observed[0]
    assert actual_system is runtime.system
    assert actual_config is runtime.config
    assert actual_plan is plan
    assert type(actual_source) is capture_validation.CaptureValidationSource
    assert actual_source.split == bundle.capture_source.split
    assert actual_source.manifest_sha256 == bundle.capture_source.manifest_sha256
    assert actual_source.census_sha256 == bundle.capture_source.census_sha256
    assert (metric, loss, edges, maximum) == (
        0.5,
        0.75,
        sum(expected_edges),
        max(expected_edges),
    )


def test_cached_checkpoint_resume_matches_uninterrupted_and_binds_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    plan = _admit(bundle, system_id="08")
    source_type = prepared_data_v2.PreparedTrainingDataSourceV2
    original_iter = source_type.iter_epoch

    def full_effective_batch(self, *, epoch: int, seed: int):
        values = tuple(original_iter(self, epoch=epoch, seed=seed))
        assert len(values) == 1 and values[0].motion_count == 1
        return values * training.EFFECTIVE_GLOBAL_BATCH

    monkeypatch.setattr(source_type, "iter_epoch", full_effective_batch)
    monkeypatch.setattr(training.PhaseSetTrainingRuntime, "_train_update", _tiny_optimizer_update)
    monkeypatch.setattr(training.PhaseSetTrainingRuntime, "_validation", _fixed_validation)

    uninterrupted = _runtime(tmp_path / "uninterrupted", plan, monkeypatch)
    uninterrupted_report = uninterrupted.fit(
        bundle.train_source,
        bundle.capture_source,
        stop_after_global_step=2,
    )

    partial = _runtime(tmp_path / "resumed", plan, monkeypatch)
    partial_report = partial.fit(
        bundle.train_source,
        bundle.capture_source,
        stop_after_global_step=1,
    )
    monkeypatch.setattr(
        training.execution_module,
        "verified_resume_checkpoint_payload_sha256",
        lambda *_args: partial_report.latest_checkpoint.sha256,
    )
    resumed = _runtime(tmp_path / "resumed", plan, monkeypatch)
    resumed_report = resumed.fit(
        bundle.train_source,
        bundle.capture_source,
        resume_checkpoint=partial_report.latest_checkpoint.path,
        resume_checkpoint_sha256=partial_report.latest_checkpoint.sha256,
        resume_attempt_root=tmp_path / "attempt",
        resume_record=_resume_record(),
        stop_after_global_step=2,
    )

    _assert_state_equal(_state(uninterrupted.system), _state(resumed.system))
    assert training._stable_hash(uninterrupted._checked_optimizer.state_dict()) == (
        training._stable_hash(resumed._checked_optimizer.state_dict())
    )
    assert uninterrupted._checked_scheduler.state_dict() == (
        resumed._checked_scheduler.state_dict()
    )
    assert uninterrupted_report.global_step == resumed_report.global_step == 2
    assert uninterrupted_report.periodic_descriptor_capture_plan_sha256 == plan.sha256
    assert resumed_report.periodic_descriptor_capture_plan_sha256 == plan.sha256
    assert uninterrupted_report.dataloader_state_sha256 == (resumed_report.dataloader_state_sha256)
    assert uninterrupted_report.schema == training.CACHED_REPORT_SCHEMA
    payload, _artifact = training._load_torch_checkpoint(
        resumed_report.latest_checkpoint.path,
        expected_sha256=resumed_report.latest_checkpoint.sha256,
    )
    assert payload["schema"] == training.CACHED_CHECKPOINT_SCHEMA
    assert payload["periodic_descriptor_capture_plan_sha256"] == plan.sha256
    assert payload["dataloader_state_sha256"] == resumed_report.dataloader_state_sha256


def test_cached_checkpoint_digest_drift_and_uncached_resume_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    plan = _admit(bundle, system_id="08")
    runtime = _runtime(tmp_path / "runtime", plan, monkeypatch)
    runtime._train_manifest = plan.train_source_manifest_sha256
    runtime._val_manifest = plan.capture_source_census_sha256
    runtime._initialize_optimizer(1)
    runtime._seed_new_run()
    artifact = runtime._write_checkpoint("update")
    payload, _ = training._load_torch_checkpoint(
        artifact.path,
        expected_sha256=artifact.sha256,
    )
    payload.pop("state_digest")
    payload["periodic_descriptor_capture_plan_sha256"] = _digest("wrong-plan")
    payload["state_digest"] = training._stable_hash(payload)
    changed = training._atomic_torch_save(tmp_path / "changed.pt", payload)

    replacement = _runtime(tmp_path / "replacement", plan, monkeypatch)
    replacement._train_manifest = plan.train_source_manifest_sha256
    replacement._val_manifest = plan.capture_source_census_sha256
    replacement._initialize_optimizer(1)
    with pytest.raises(training.TrainingCheckpointError, match="plan_sha256 binding"):
        replacement._restore_checkpoint(changed.path, expected_sha256=changed.sha256)

    uncached_system = _tiny_system()
    uncached = training.PhaseSetTrainingRuntime(
        uncached_system,
        bundle.config,
        tmp_path / "uncached",
        initialization_binding=_binding(uncached_system, bundle.config, None),
    )
    uncached._train_manifest = plan.train_source_manifest_sha256
    uncached._val_manifest = plan.capture_source_census_sha256
    uncached._initialize_optimizer(1)
    with pytest.raises(training.TrainingCheckpointError, match="uncached runtime"):
        uncached._restore_checkpoint(artifact.path, expected_sha256=artifact.sha256)


def test_plan_source_and_system_mismatch_fail_before_fit_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    plan = _admit(bundle, system_id="08")
    runtime = _runtime(tmp_path / "runtime", plan, monkeypatch)
    changed_capture = capture_validation.CaptureValidationSource(
        "val",
        _digest("changed-capture-manifest"),
        bundle.capture_source.captures,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid sources must fail before formal model checks")

    monkeypatch.setattr(runtime, "_assert_live_formal_binding", forbidden)
    with pytest.raises(PeriodicCaptureTrainingCacheError):
        runtime.fit(bundle.train_source, changed_capture)

    system01 = _tiny_system("01")
    binding01 = _binding(system01, bundle.config, plan.sha256)
    with pytest.raises(training.TrainingRuntimeError, match="differs from the residual runtime"):
        training.PhaseSetTrainingRuntime(
            system01,
            bundle.config,
            tmp_path / "system01",
            initialization_binding=binding01,
            descriptor_plan=plan,
        )


def test_cached_binding_digest_is_required_before_formal_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    plan = _admit(bundle, system_id="08")
    system = _tiny_system()
    binding = _binding(system, bundle.config, None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("binding mismatch must precede formal model validation")

    monkeypatch.setattr(training, "_validate_registered_formal_system", forbidden)
    with pytest.raises(training.TrainingRuntimeError, match="binding and descriptor plan"):
        training.PhaseSetTrainingRuntime(
            system,
            bundle.config,
            tmp_path / "runtime",
            initialization_binding=binding,
            descriptor_plan=plan,
        )
