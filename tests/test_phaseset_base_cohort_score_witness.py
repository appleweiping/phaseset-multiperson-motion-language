"""Server-only real-model witness from selected state to full capture gallery.

The attempt registry and completed-ledger cohort are intentionally outside this
bounded integration test.  Everything after one parsed selected checkpoint row
uses the real registered implementation and is not monkeypatched.
"""

from __future__ import annotations

from fractions import Fraction
import gc
import hashlib
import io

import numpy as np
import pytest
import torch

from phaseset_core import base_cohort_resolver as resolver
from phaseset_core import base_cohort_validation as scoring
from phaseset_core import capture_validation as capture
from phaseset_core import evaluation
from phaseset_core import frozen_clip_text as clip
from phaseset_core import training
from phaseset_core.capture_pooling import CaptureWindowPlan
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.experiments import (
    base_run_ids,
    build_experiment_plan,
    experiment_plan_sha256,
    parse_run_id,
)


_SEED = 1729


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _identity(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).numpy().tobytes(order="C")


def _analytic_groups(label: str, k: int) -> PreparedGroupBatch:
    actor = np.arange(k, dtype=np.int64).reshape(1, k, 1, 1, 1)
    frame = np.arange(200, dtype=np.int64).reshape(1, 1, 200, 1, 1)
    joint = np.arange(22, dtype=np.int64).reshape(1, 1, 1, 22, 1)
    coordinate = np.arange(3, dtype=np.int64).reshape(1, 1, 1, 1, 3)
    integer = ((actor * 131 + frame * 17 + joint * 7 + coordinate * 3) % 2048) - 1024
    skeletons = np.ascontiguousarray(
        integer.astype(np.float32) * np.float32(2**-10),
        dtype=np.float32,
    )
    actors = tuple(_identity(f"{label}/actor/{index}") for index in range(k))
    return PreparedGroupBatch(
        skeletons,
        np.ones((1, k), dtype=np.bool_),
        np.ones((1, 200), dtype=np.bool_),
        np.ones((1, k, 200, 22), dtype=np.bool_),
        (actors,),
        (group_commitment(actors),),
    )


def _analytic_text(label: str, coordinate: int) -> clip.FrozenClipTextBatch:
    embeddings = torch.zeros((1, training.REGISTERED_EMBEDDING_DIM), dtype=torch.float32)
    embeddings[0, coordinate] = 1.0
    commitment = _identity(f"{label}/caption")
    raw = _tensor_bytes(embeddings)
    receipt = clip.FrozenClipTextReceipt(
        batch_size=1,
        caption_rows=(
            (
                0,
                commitment.hex(),
                _identity(f"{label}/analytic-text-not-a-caption").hex(),
                3,
                3,
                False,
                _identity(f"{label}/analytic-token-row").hex(),
                _identity(f"{label}/analytic-mask-row").hex(),
                hashlib.sha256(raw).hexdigest(),
            ),
        ),
        chunk_ranges=((0, 1),),
        frozen_embedding_cache_key_sha256=_sha(f"{label}/analytic-cache"),
        live_model_manifest_sha256=_sha(f"{label}/no-live-model"),
        output_bytes=len(raw),
        output_sha256=hashlib.sha256(raw).hexdigest(),
        output_shape=tuple(embeddings.shape),
        output_stride=tuple(embeddings.stride()),
        runtime_identity=(("fixture", "server-only-no-clip"),),
        runtime_manifest_sha256=_sha(f"{label}/analytic-runtime"),
        snapshot_files=(),
        snapshot_manifest_sha256=_sha(f"{label}/no-snapshot"),
        source_files=(),
        source_manifest_sha256=_sha(f"{label}/analytic-source"),
    )
    return clip.FrozenClipTextBatch(
        embeddings,
        (commitment,),
        receipt,
        _seal=clip._CONSTRUCTION_SEAL,
    )


def _analytic_capture(
    label: str,
    *,
    k: int,
    start_frame: int,
    text_coordinate: int,
    component: str,
) -> capture.CaptureValidationCapture:
    capture_id = _identity(f"{label}/capture")
    window_id = _identity(f"{label}/window")
    return capture.CaptureValidationCapture(
        CaptureWindowPlan(capture_id, (window_id,), (start_frame,)),
        (
            capture.CaptureValidationWindow(
                window_id,
                _analytic_groups(label, k),
            ),
        ),
        _analytic_text(label, text_coordinate),
        component,
    )


def _analytic_source() -> capture.CaptureValidationSource:
    return capture.CaptureValidationSource(
        "val",
        _sha("analytic-two-capture-manifest-no-rights-claim"),
        (
            _analytic_capture(
                "capture-k2",
                k=2,
                start_frame=0,
                text_coordinate=0,
                component="C00",
            ),
            _analytic_capture(
                "capture-k3",
                k=3,
                start_frame=300,
                text_coordinate=1,
                component="C01",
            ),
        ),
    )


def _admission(source: capture.CaptureValidationSource) -> resolver.BaseCohortAdmission:
    plan = build_experiment_plan()
    return resolver.BaseCohortAdmission(
        plan_sha256=experiment_plan_sha256(plan),
        matrix_sha256=plan.matrix_sha256,
        training_config_sha256=_sha("analytic-host-training-config-no-receipt"),
        source_tree_sha256=_sha("analytic-source-tree-comparison-only"),
        train_manifest_sha256=_sha("analytic-train-manifest-no-training-run"),
        val_manifest_sha256=source.census_sha256,
        device="cpu",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32_768,
        checkpoint_every_updates=1,
    )


def _run_id(system_id: str) -> str:
    matches = tuple(
        run_id
        for run_id in base_run_ids()
        if parse_run_id(run_id)[1:] == (_SEED, system_id)
    )
    assert len(matches) == 1
    return matches[0]


def _analytic_selected_row(
    system_id: str,
    admission: resolver.BaseCohortAdmission,
) -> tuple[resolver.ResolvedBaseRun, int, tuple[tuple[str, tuple[int, ...], int], ...], str]:
    """Build one real closed checkpoint payload without claiming a training attempt."""

    run_id = _run_id(system_id)
    config = training.TrainingConfig(
        stage="base",
        seed=_SEED,
        device=admission.device,
        request_bf16=admission.request_bf16,
        bf16_runtime_qualified=admission.bf16_runtime_qualified,
        edge_budget=admission.edge_budget,
        checkpoint_every_updates=admission.checkpoint_every_updates,
        synthetic_contract=False,
    )
    assert config.epochs == training.BASE_EPOCHS
    system, binding = training.construct_registered_base_seed_bound_system(system_id, config)
    initial_state_sha256 = training.training_system_state_sha256(system)
    expected_rows = tuple(
        (name, tuple(parameter.shape), int(parameter.numel()))
        for name, parameter in sorted(system.named_parameters())
    )
    expected_parameter_count = sum(row[2] for row in expected_rows)
    assert expected_parameter_count == sum(
        int(parameter.numel()) for parameter in system.parameters()
    )
    assert all(parameter.requires_grad for parameter in system.parameters())

    parameters = [parameter for parameter in system.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=training.WEIGHT_DECAY,
    )
    total_steps = training.BASE_EPOCHS

    def multiplier(step: int) -> float:
        return training._learning_rate_multiplier(step, total_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    optimizer.zero_grad(set_to_none=True)
    system.logit_scale.grad = torch.zeros_like(system.logit_scale)
    optimizer.step()
    scheduler.step()
    selected_state_sha256 = training.training_system_state_sha256(system)
    assert selected_state_sha256 != initial_state_sha256

    checkpoint_name = "checkpoint-000000000001-000001-validation.pt"
    payload: dict[str, object] = {
        "schema": training.CHECKPOINT_SCHEMA,
        "authority": 0,
        "production": False,
        "result_claimed": False,
        "external_receipt_verified": False,
        "stage": "base",
        "seed": _SEED,
        "system_id": system_id,
        "config_sha256": config.sha256,
        "train_manifest_sha256": admission.train_manifest_sha256,
        "val_manifest_sha256": admission.val_manifest_sha256,
        "updates_per_epoch": 1,
        "total_steps": total_steps,
        "precision_mode": training.resolve_precision(config).mode,
        "initialization_binding_sha256": binding.sha256,
        "behavior_sha256": binding.behavior_sha256,
        "initial_optimizable_state_sha256": binding.initial_optimizable_state_sha256,
        "factory_sha256": binding.factory_sha256,
        "code_artifact_sha256": binding.code_artifact_sha256,
        "environment_sha256": binding.environment_sha256,
        "frozen_base_checkpoint_sha256": None,
        "frozen_base_state_sha256": None,
        "qualified_base_selection_sha256": None,
        "residual_capacity_audit_sha256": None,
        "epoch": 1,
        "update_index": 0,
        "global_step": 1,
        "checkpoint_sequence": 1,
        "train_edges_seen": 1,
        "validation_edges_seen": 1,
        "max_microbatch_edges": 1,
        "best_validation_metric": 0.5,
        "best_checkpoint_name": checkpoint_name,
        "best_checkpoint_sha256": None,
        "last_train_loss": 0.0,
        "model": {
            name: value.detach().cpu().clone()
            for name, value in system.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": training._capture_rng(),
    }
    payload["state_digest"] = training._stable_hash(payload)
    assert frozenset(payload) == resolver._CHECKPOINT_KEYS
    selected_model_sha256 = training._stable_hash(payload["model"])
    assert selected_model_sha256 == selected_state_sha256
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    row = resolver.ResolvedBaseRun(
        run_id=run_id,
        attempt_id=f"analytic-no-ledger-{system_id.lower()}",
        system_id=system_id,
        seed=_SEED,
        terminal_sha256=_sha(f"{system_id}/no-terminal-evidence"),
        latest_checkpoint_sha256=raw_sha256,
        latest_checkpoint_name=checkpoint_name,
        selected_checkpoint=resolver.SelectedBaseCheckpoint(
            name=checkpoint_name,
            sha256=raw_sha256,
            raw=raw,
            epoch=1,
            global_step=1,
            state_digest=str(payload["state_digest"]),
        ),
        failed_predecessors=(),
    )
    del payload, scheduler, optimizer, system
    gc.collect()
    return row, expected_parameter_count, expected_rows, selected_model_sha256


def _independent_capture_fractions(
    observation: scoring.BaseValidationScoreObservation,
) -> tuple[Fraction, Fraction, Fraction]:
    scores = (
        np.frombuffer(observation.scores_float64le, dtype="<f8")
        .reshape(observation.score_shape)
        .astype(np.float64, copy=True, order="C")
    )
    dataset = evaluation.validate_retrieval_dataset(
        evaluation.RetrievalDataset(
            scores=scores,
            motion_commitments=observation.motion_commitments,
            caption_commitments=observation.caption_commitments,
            positive_motion_indices=observation.positive_motion_indices,
            group_sizes=np.ascontiguousarray(observation.group_sizes, dtype=np.int64),
            component_labels=observation.component_labels,
        )
    )
    rows = evaluation.capture_r1_contributions(dataset)
    assert len(rows) == 2
    count = len(rows)
    text = sum(
        (
            Fraction(sum(row.text_to_motion_hits), len(row.text_to_motion_hits))
            for row in rows
        ),
        start=Fraction(0, 1),
    ) / count
    motion = sum(
        (
            Fraction(sum(row.motion_to_text_hits), len(row.motion_to_text_hits))
            for row in rows
        ),
        start=Fraction(0, 1),
    ) / count
    return text, motion, (text + motion) / 2


@pytest.mark.parametrize("system_id", ("B0", "B1", "B2"))
def test_real_registered_selected_state_scores_complete_analytic_gallery(
    system_id: str,
) -> None:
    source = _analytic_source()
    source_census_before = source.census_sha256
    admission = resolver._checked_admission(_admission(source))
    row, expected_count, expected_rows, selected_model_sha256 = _analytic_selected_row(
        system_id,
        admission,
    )
    raw_before = hashlib.sha256(row.selected_checkpoint.raw).hexdigest()
    source_type, runner = capture._verified_capture_validation_bindings()
    assert source_type is capture.CaptureValidationSource
    assert runner is capture.run_capture_validation

    observation = scoring._score_resolved_run(
        row,
        admission=admission,
        source=source,
    )

    assert observation.authority == 0
    assert observation.production is False
    assert observation.result_claimed is False
    assert observation.system_id == system_id
    assert observation.score_shape == (2, 2)
    assert np.isfinite(
        np.frombuffer(observation.scores_float64le, dtype="<f8")
    ).all()
    assert hashlib.sha256(observation.scores_float64le).hexdigest() == (
        observation.scores_float64_sha256
    )
    assert _independent_capture_fractions(observation) == (
        observation.text_to_motion_capture_r1,
        observation.motion_to_text_capture_r1,
        observation.primary_capture_r1,
    )
    assert set(observation.group_sizes) == {2, 3}
    assert observation.positive_motion_indices == ((0,), (1,))
    assert observation.parameter_census.total == expected_count
    assert tuple(
        (item.name, item.shape, item.numel)
        for item in observation.parameter_census.rows
    ) == expected_rows
    assert observation.parameter_census.loaded_state_sha256 == selected_model_sha256
    assert observation.selected_checkpoint_sha256 == raw_before
    assert hashlib.sha256(row.selected_checkpoint.raw).hexdigest() == raw_before
    assert observation.query_census_sha256 == source_census_before
    assert source.census_sha256 == source_census_before
