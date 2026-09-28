"""Server-only contracts for selected-best base cohort capture scoring."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from phaseset_core import base_cohort_resolver as resolver
from phaseset_core import base_cohort_validation as scoring
from phaseset_core import capture_validation as capture
from phaseset_core import frozen_clip_text as clip
from phaseset_core import training
from phaseset_core.capture_pooling import CaptureWindowPlan
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.evaluation import RetrievalDataset
from phaseset_core.experiments import (
    base_run_ids,
    build_experiment_plan,
    experiment_plan_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _id(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _groups(label: str) -> PreparedGroupBatch:
    actors = (_id(f"{label}/actor/0"), _id(f"{label}/actor/1"))
    skeletons = np.zeros((1, 2, 200, 22, 3), dtype=np.float32)
    skeletons[0, :, :, :, 0] = 1.0
    return PreparedGroupBatch(
        skeletons,
        np.ones((1, 2), dtype=np.bool_),
        np.ones((1, 200), dtype=np.bool_),
        np.ones((1, 2, 200, 22), dtype=np.bool_),
        (actors,),
        (group_commitment(actors),),
    )


def _text(label: str) -> clip.FrozenClipTextBatch:
    values = torch.zeros((1, 512), dtype=torch.float32)
    values[0, 0] = 1.0
    commitment = _id(f"{label}/caption")
    raw = _tensor_bytes(values)
    receipt = clip.FrozenClipTextReceipt(
        batch_size=1,
        caption_rows=(
            (
                0,
                commitment.hex(),
                _id(f"{label}/text").hex(),
                3,
                3,
                False,
                _id(f"{label}/tokens").hex(),
                _id(f"{label}/mask").hex(),
                hashlib.sha256(raw).hexdigest(),
            ),
        ),
        chunk_ranges=((0, 1),),
        frozen_embedding_cache_key_sha256=_sha(f"{label}/cache"),
        live_model_manifest_sha256=_sha(f"{label}/model"),
        output_bytes=len(raw),
        output_sha256=hashlib.sha256(raw).hexdigest(),
        output_shape=tuple(values.shape),
        output_stride=tuple(values.stride()),
        runtime_identity=(("fixture", "server-only"),),
        runtime_manifest_sha256=_sha(f"{label}/runtime"),
        snapshot_files=(),
        snapshot_manifest_sha256=_sha(f"{label}/snapshot"),
        source_files=(),
        source_manifest_sha256=_sha(f"{label}/source"),
    )
    return clip.FrozenClipTextBatch(
        values,
        (commitment,),
        receipt,
        _seal=clip._CONSTRUCTION_SEAL,
    )


def _capture(label: str, frame: int) -> capture.CaptureValidationCapture:
    window = _id(f"{label}/window")
    return capture.CaptureValidationCapture(
        CaptureWindowPlan(_id(label), (window,), (frame,)),
        (capture.CaptureValidationWindow(window, _groups(label)),),
        _text(label),
        "C00",
    )


def _source() -> capture.CaptureValidationSource:
    return capture.CaptureValidationSource(
        "val",
        _sha("upstream-validation-manifest"),
        (_capture("capture-a", 0), _capture("capture-b", 300)),
    )


def _admission(source: capture.CaptureValidationSource) -> resolver.BaseCohortAdmission:
    plan = build_experiment_plan()
    return resolver.BaseCohortAdmission(
        plan_sha256=experiment_plan_sha256(plan),
        matrix_sha256=plan.matrix_sha256,
        training_config_sha256=_sha("host-training-config"),
        source_tree_sha256=_sha("source-tree"),
        train_manifest_sha256=_sha("train-manifest"),
        val_manifest_sha256=source.census_sha256,
        device="cpu",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32_768,
        checkpoint_every_updates=1,
    )


def _resolved_row(run_id: str, index: int) -> resolver.ResolvedBaseRun:
    _role, seed_text, system_id = run_id.split("/")[1:]
    raw = f"selected-best-{index}".encode("ascii")
    return resolver.ResolvedBaseRun(
        run_id=run_id,
        attempt_id=f"attempt-{index}",
        system_id=system_id,
        seed=int(seed_text),
        terminal_sha256=_sha(f"terminal-{index}"),
        latest_checkpoint_sha256=_sha(f"latest-{index}"),
        latest_checkpoint_name="checkpoint-000000000030-000060-validation.pt",
        selected_checkpoint=resolver.SelectedBaseCheckpoint(
            name="checkpoint-000000000001-000002-validation.pt",
            sha256=hashlib.sha256(raw).hexdigest(),
            raw=raw,
            epoch=1,
            global_step=1,
            state_digest=_sha(f"checkpoint-state-{index}"),
        ),
        failed_predecessors=(),
    )


def _cohort(source: capture.CaptureValidationSource) -> resolver.ResolvedBaseCohort:
    admission = _admission(source)
    return resolver.ResolvedBaseCohort(
        rows=tuple(
            _resolved_row(run_id, index)
            for index, run_id in enumerate(base_run_ids())
        ),
        plan_sha256=admission.plan_sha256,
        matrix_sha256=admission.matrix_sha256,
        training_config_sha256=admission.training_config_sha256,
        source_tree_sha256=admission.source_tree_sha256,
        train_manifest_sha256=admission.train_manifest_sha256,
        val_manifest_sha256=admission.val_manifest_sha256,
    )


def _score_dataset(source: capture.CaptureValidationSource) -> RetrievalDataset:
    motions, captions, positives, sizes, labels = scoring._expected_dataset_identity(source)
    return RetrievalDataset(
        scores=np.ascontiguousarray(((2.0, 0.0), (0.0, 2.0)), dtype=np.float64),
        motion_commitments=motions,
        caption_commitments=captions,
        positive_motion_indices=positives,
        group_sizes=np.ascontiguousarray(sizes, dtype=np.int64),
        component_labels=labels,
    )


def _observation(
    row: resolver.ResolvedBaseRun,
    source: capture.CaptureValidationSource,
) -> scoring.BaseValidationScoreObservation:
    dataset = _score_dataset(source)
    score_bytes = np.ascontiguousarray(dataset.scores, dtype="<f8").tobytes()
    width = {"B0": 3, "B1": 5, "B2": 7}[row.system_id]
    parameters = scoring.BaseParameterCensus(
        row.system_id,
        (
            scoring.BaseParameterRow(
                "weight",
                (width,),
                "torch.float32",
                width,
            ),
        ),
        width,
        _sha(f"loaded-{row.run_id}"),
    )
    return scoring.BaseValidationScoreObservation(
        run_id=row.run_id,
        attempt_id=row.attempt_id,
        system_id=row.system_id,
        seed=row.seed,
        terminal_sha256=row.terminal_sha256,
        selected_checkpoint_sha256=row.selected_checkpoint.sha256,
        selected_checkpoint_state_digest=row.selected_checkpoint.state_digest,
        initialization_binding_sha256=_sha(f"binding-{row.run_id}"),
        runtime_config_sha256=_sha(f"runtime-config-{row.run_id}"),
        factory_sha256=_sha(f"factory-{row.system_id}"),
        code_artifact_sha256=_sha("code"),
        environment_sha256=_sha("environment"),
        behavior_sha256=_sha(f"behavior-{row.system_id}"),
        source_manifest_sha256=source.manifest_sha256,
        admitted_val_census_sha256=source.census_sha256,
        query_census_sha256=source.census_sha256,
        parameter_census=parameters,
        scores_float64le=score_bytes,
        score_shape=(2, 2),
        motion_commitments=dataset.motion_commitments,
        caption_commitments=dataset.caption_commitments,
        positive_motion_indices=dataset.positive_motion_indices,
        group_sizes=tuple(int(value) for value in dataset.group_sizes),
        component_labels=dataset.component_labels,
        text_to_motion_capture_r1=Fraction(1, 1),
        motion_to_text_capture_r1=Fraction(1, 1),
        primary_capture_r1=Fraction(1, 1),
        device="cpu",
        precision_mode="FP32",
        encoded_bytes=4096,
    )


def test_cohort_orchestration_preserves_nine_rows_without_selecting_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    cohort = _cohort(source)
    admission = _admission(source)
    monkeypatch.setattr(
        scoring,
        "_score_resolved_run",
        lambda row, *, admission, source: _observation(row, source),
    )

    result = scoring.score_resolved_base_cohort(
        cohort,
        admission=admission,
        validation_source=source,
    )

    assert tuple(row.run_id for row in result.observations) == base_run_ids()
    assert len({row.score_artifact_sha256 for row in result.observations}) == 9
    assert result.authority == 0
    assert result.production is False
    assert result.result_claimed is False
    assert result.training_config_sha256 == admission.training_config_sha256
    assert all(row.runtime_config_sha256 for row in result.observations)
    assert "NO_WINNER_NO_QUALIFICATION" in result.status
    assert not hasattr(result, "winner_system_id")


def test_resolved_owned_checkpoint_digest_is_rechecked_before_scoring() -> None:
    source = _source()
    cohort = _cohort(source)
    first = cohort.rows[0]
    changed_selected = replace(first.selected_checkpoint, raw=b"substituted")
    changed = replace(cohort, rows=(replace(first, selected_checkpoint=changed_selected),) + cohort.rows[1:])

    with pytest.raises(scoring.BaseCohortValidationError, match="owned checkpoint digest"):
        scoring._checked_cohort(changed, _admission(source))


def test_checkpoint_admitted_capture_census_cannot_be_relabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    admission = replace(_admission(source), val_manifest_sha256=_sha("different-gallery"))
    cohort = replace(_cohort(source), val_manifest_sha256=admission.val_manifest_sha256)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(scoring, "_score_resolved_run", forbidden)
    with pytest.raises(scoring.BaseCohortValidationError, match="capture census"):
        scoring.score_resolved_base_cohort(
            cohort,
            admission=admission,
            validation_source=source,
        )
    assert called is False


def test_score_observation_recomputes_metric_from_actual_score_bytes() -> None:
    source = _source()
    observation = _observation(_cohort(source).rows[0], source)

    with pytest.raises(scoring.BaseCohortValidationError, match="differs from score bytes"):
        replace(observation, primary_capture_r1=Fraction(0, 1))
    changed = np.ascontiguousarray(((0.0, 2.0), (2.0, 0.0)), dtype="<f8")
    with pytest.raises(scoring.BaseCohortValidationError, match="differs from score bytes"):
        replace(observation, scores_float64le=changed.tobytes(order="C"))


def test_changed_scores_with_same_ranking_remain_valid_without_external_hash() -> None:
    source = _source()
    observation = _observation(_cohort(source).rows[0], source)
    changed = np.ascontiguousarray(((3.0, 0.0), (0.0, 3.0)), dtype="<f8")
    updated = replace(observation, scores_float64le=changed.tobytes(order="C"))
    assert updated.primary_capture_r1 == observation.primary_capture_r1
    assert updated.scores_float64_sha256 != observation.scores_float64_sha256
    assert updated.score_artifact_sha256 != observation.score_artifact_sha256


def test_resolved_cohort_rejects_reordered_registered_rows_before_scoring() -> None:
    source = _source()
    cohort = _cohort(source)
    changed = replace(cohort, rows=tuple(reversed(cohort.rows)))
    with pytest.raises(scoring.BaseCohortValidationError, match="nine-row order"):
        scoring._checked_cohort(changed, _admission(source))


def test_capture_result_metric_and_gallery_identity_are_not_trusted() -> None:
    source = _source()
    row = _cohort(source).rows[0]
    config = training.TrainingConfig(stage="base", seed=row.seed)
    dataset = _score_dataset(source)
    result = capture.CaptureValidationResult(
        dataset=dataset,
        text_to_motion_capture_r1=Fraction(1, 1),
        motion_to_text_capture_r1=Fraction(1, 1),
        primary_capture_r1=Fraction(0, 1),
        loss=0.0,
        source_census_sha256=source.census_sha256,
        scores_float64_sha256=hashlib.sha256(dataset.scores.tobytes()).hexdigest(),
        system_id=row.system_id,
        stage="base",
        device="cpu",
        precision_mode="FP32",
        encoded_bytes=4096,
        cuda_static_tensor_bytes=None,
    )

    with pytest.raises(scoring.BaseCohortValidationError, match="recomputed gallery"):
        scoring._validated_capture_result(result, row=row, source=source, config=config)
    changed_dataset = replace(
        dataset,
        motion_commitments=tuple(reversed(dataset.motion_commitments)),
    )
    changed = replace(
        result,
        dataset=changed_dataset,
        primary_capture_r1=Fraction(1, 1),
    )
    with pytest.raises(scoring.BaseCohortValidationError, match="frozen source"):
        scoring._validated_capture_result(changed, row=row, source=source, config=config)


class _ParameterOnlyBase(nn.Module):
    system_id = "B0"

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, _groups):
        raise AssertionError("fake capture runner must own this test's scoring result")


def _selected_score_fixture(
    source: capture.CaptureValidationSource,
    row: resolver.ResolvedBaseRun,
):
    system = training.BaseRetrievalSystem(_ParameterOnlyBase(), embedding_dim=512)
    initial_state = training.training_system_state_sha256(system)
    behavior = training.training_system_behavior_sha256(system)
    model = {
        name: torch.ones_like(value)
        for name, value in system.state_dict().items()
    }
    binding = SimpleNamespace(
        behavior_sha256=behavior,
        code_artifact_sha256=_sha("code"),
        environment_sha256=_sha("environment"),
        factory_sha256=_sha("factory"),
        initial_optimizable_state_sha256=initial_state,
        sha256=_sha("initialization-binding"),
    )
    payload = {
        "behavior_sha256": binding.behavior_sha256,
        "code_artifact_sha256": binding.code_artifact_sha256,
        "environment_sha256": binding.environment_sha256,
        "factory_sha256": binding.factory_sha256,
        "initial_optimizable_state_sha256": binding.initial_optimizable_state_sha256,
        "initialization_binding_sha256": binding.sha256,
        "model": model,
    }
    dataset = _score_dataset(source)
    result = capture.CaptureValidationResult(
        dataset=dataset,
        text_to_motion_capture_r1=Fraction(1, 1),
        motion_to_text_capture_r1=Fraction(1, 1),
        primary_capture_r1=Fraction(1, 1),
        loss=0.0,
        source_census_sha256=source.census_sha256,
        scores_float64_sha256=hashlib.sha256(dataset.scores.tobytes()).hexdigest(),
        system_id=row.system_id,
        stage="base",
        device="cpu",
        precision_mode="FP32",
        encoded_bytes=4096,
        cuda_static_tensor_bytes=None,
    )
    return system, binding, payload, result


def test_selected_model_state_is_strictly_loaded_and_parameters_are_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    row = _cohort(source).rows[0]
    admission = _admission(source)
    system, binding, payload, result = _selected_score_fixture(source, row)
    monkeypatch.setattr(scoring, "_validate_selected_payload", lambda *_args: payload)
    monkeypatch.setattr(
        training,
        "construct_registered_base_seed_bound_system",
        lambda *_args: (system, binding),
    )

    def runner(live, _config, _source):
        assert all(torch.equal(value, torch.ones_like(value)) for value in live.state_dict().values())
        return result

    monkeypatch.setattr(
        capture,
        "_verified_capture_validation_bindings",
        lambda: (capture.CaptureValidationSource, runner),
    )
    observation = scoring._score_resolved_run(
        row,
        admission=admission,
        source=source,
    )
    assert observation.runtime_config_sha256 == training.TrainingConfig(
        stage="base",
        seed=row.seed,
        device=admission.device,
        request_bf16=admission.request_bf16,
        bf16_runtime_qualified=admission.bf16_runtime_qualified,
        edge_budget=admission.edge_budget,
        checkpoint_every_updates=admission.checkpoint_every_updates,
        synthetic_contract=False,
    ).sha256
    assert observation.parameter_census.total == 3
    assert tuple(item.name for item in observation.parameter_census.rows) == (
        "group_base.weight",
        "logit_scale",
    )
    assert observation.parameter_census.total != 0


def test_incomplete_selected_state_has_no_random_checkpoint_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    row = _cohort(source).rows[0]
    admission = _admission(source)
    system, binding, payload, _result = _selected_score_fixture(source, row)
    payload["model"] = {"group_base.weight": torch.ones(2)}
    monkeypatch.setattr(scoring, "_validate_selected_payload", lambda *_args: payload)
    monkeypatch.setattr(
        training,
        "construct_registered_base_seed_bound_system",
        lambda *_args: (system, binding),
    )
    with pytest.raises(scoring.BaseCohortValidationError, match="could not be loaded"):
        scoring._score_resolved_run(row, admission=admission, source=source)


def test_owned_validation_snapshot_detects_between_row_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    cohort = _cohort(source)
    calls = 0

    def mutate(row, *, admission, source):
        nonlocal calls
        calls += 1
        result = _observation(row, source)
        if calls == 1:
            object.__setattr__(source, "manifest_sha256", _sha("mutated-source"))
        return result

    monkeypatch.setattr(scoring, "_score_resolved_run", mutate)
    with pytest.raises(scoring.BaseCohortValidationError, match="between cohort rows"):
        scoring.score_resolved_base_cohort(
            cohort,
            admission=_admission(source),
            validation_source=source,
        )
    assert calls == 1


@pytest.mark.parametrize("system_id", ("B0", "B1", "B2"))
def test_registered_factory_is_actual_width_and_fully_trainable(system_id: str) -> None:
    config = training.TrainingConfig(stage="base", seed=1729, synthetic_contract=False)
    system, binding = training.construct_registered_base_seed_bound_system(system_id, config)
    assert system.system_id == system_id
    assert system.embedding_dim == 512
    assert binding.system_id == system_id
    assert all(parameter.requires_grad for parameter in system.parameters())
    census = scoring._parameter_census(system)
    assert census.total == sum(parameter.numel() for parameter in system.parameters())
