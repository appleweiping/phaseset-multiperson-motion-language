"""Data-free old/A9 complete-parent score and host-boundary qualifications."""

from __future__ import annotations

import copy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from phaseset_core.capture_validation import _validate_text_batch
from phaseset_core.continuous_parent_host import ContinuousParentTrainingHost
from phaseset_core.continuous_parent_host import ParentHostRunIdentity
from phaseset_core.continuous_parent_training import backward_loaded_parent_batch
from phaseset_core.controls import PhaseSetSystem
from phaseset_core.legacy_continuous_retrieval import LegacyWholeCaptureRetrievalSystem
from phaseset_core.legacy_scalar_calibration import (
    LEGACY_SCALAR_FLOOR_SCHEMA,
    LegacyScalarFloorReceipt,
    legacy_scalar_config_sha256,
)
from phaseset_core.periodic import BAND_COUNT
from phaseset_core.temporal_coordination import coordination_objective
from phaseset_core.training import _capture_rng, _restore_rng, _stable_hash
from test_continuous_parent_host import _setup
from test_continuous_parent_training import _case, _empty_cf


def _legacy(
    base, mode: str, components: tuple[str, ...] = ("C01",),
) -> LegacyWholeCaptureRetrievalSystem:
    floors = np.full((BAND_COUNT,), 1e-10, dtype=np.float64)
    encoder = PhaseSetSystem(
        "08",
        embedding_dim=512,
        hidden_dim=8,
        energy_floors=floors,
        edge_budget=64,
    )
    # Synthetic binding fixture only: this is not an admitted native training receipt.
    receipt = LegacyScalarFloorReceipt(
        LEGACY_SCALAR_FLOOR_SCHEMA,
        "main",
        components,
        "f" * 64,
        legacy_scalar_config_sha256(),
        floors,
        1,
        1,
        3,
        (3,) * BAND_COUNT,
        (0,) * BAND_COUNT,
        (0,) * BAND_COUNT,
    )
    return LegacyWholeCaptureRetrievalSystem(
        copy.deepcopy(base), encoder, score_mode=mode,
        floor_receipt=receipt, edge_chunk_size=64,
    )


@pytest.mark.parametrize("mode", ["old", "A9"])
def test_complete_legacy_score_uses_frozen_b2_and_registered_head_math(tmp_path, mode):
    v2, views, _labels, text, _clip = _case(tmp_path)
    system = _legacy(v2.frozen_b2, mode)
    capture = views[0].capture
    expected_count = len(capture.windows())
    assert expected_count > 0
    output = system.score((capture,), text)
    assert output.scores.shape == (1, len(text.receipt.caption_rows))
    assert output.global_cosine.shape == output.scores.shape
    assert output.legacy_relation_cosine.shape == output.scores.shape
    assert output.capture_keys == (capture.source_sha256,)
    assert output.periodic_support.shape == (1,)
    assert bool(torch.isfinite(output.scores).all())
    assert bool((output.global_cosine.abs() <= 1.00001).all())
    text_embeddings, _, _ = _validate_text_batch(text, allow_row_selection=True)
    global_embedding = system.encode_global(capture)[None]
    if mode == "old":
        base_logits = system.frozen_b2.scores(global_embedding, text_embeddings)
        expected_scores = (
            base_logits
            + system.head.residual_lambda.tanh()[0] * output.legacy_relation_cosine
        )
    else:
        alpha = system.head.calibration.mixture_logit.sigmoid()
        alpha = torch.where(output.periodic_support[:, None], alpha, torch.zeros_like(alpha))
        scale = system.head.calibration.log_scale.clamp(math.log(1e-3), math.log(100)).exp()
        expected_scores = scale * (
            (1 - alpha) * output.global_cosine
            + alpha * output.legacy_relation_cosine
        )
    torch.testing.assert_close(output.scores, expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(
        output.global_cosine,
        F.normalize(global_embedding, dim=-1) @ F.normalize(text_embeddings, dim=-1).T,
        rtol=0,
        atol=0,
    )
    output.scores.sum().backward()
    assert all(parameter.grad is None for parameter in system.frozen_b2.parameters())
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in system.legacy_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in system.head.parameters())


@pytest.mark.parametrize("mode", ["old", "A9"])
def test_legacy_parent_host_rejects_unbound_floor_source(tmp_path, mode):
    v2, task, source, config, bindings = _setup(tmp_path)
    system = _legacy(v2.frozen_b2, mode)
    with pytest.raises(ValueError, match="one typed floor receipt"):
        ContinuousParentTrainingHost(system, task, source, config, bindings)
    assert system.get_extra_state().dtype == torch.uint8
    saved = system.get_extra_state().clone()
    system.set_extra_state(saved)
    with pytest.raises(ValueError, match="changed on resume"):
        system.set_extra_state(torch.empty(0, dtype=torch.uint8))
    system.legacy_encoder.encoder._energy_floors = np.full((6,), 0.1, dtype=np.float64)
    with pytest.raises(ValueError, match="changed after construction"):
        system.score((source.views[task.parents[0].source_sha256].capture,), None)
    system.legacy_encoder.encoder._energy_floors = system.floor_receipt.floors.copy()
    system._bound_old_scalar_floors[0] = 0.1
    with pytest.raises(ValueError, match="differs from receipt"):
        system.set_extra_state(saved)


@pytest.mark.parametrize("mode", ["old", "A9"])
def test_legacy_complete_parent_score_vjp_matches_dense_synthetic_population(
    tmp_path, mode,
):
    v2, views, labels, text, _ = _case(tmp_path)
    system = _legacy(v2.frozen_b2, mode, components=("C01", "C02"))
    dense = copy.deepcopy(system)
    captures = tuple(view.capture for view in views)
    by_source = {capture.source_sha256: capture for capture in captures}
    starting_rng = _capture_rng()
    dense_scores = torch.cat([dense.score((capture,), text).scores for capture in captures])
    empty = dense_scores.new_empty((0,))
    dense_loss = coordination_objective(
        dense_scores,
        labels.positive_mask(device=dense_scores.device),
        cf_positive_scores=empty,
        cf_negative_scores=empty,
        verified_negative_mask=torch.empty((0,), dtype=torch.bool),
        cf_weight=0.2,
        margin=0.2,
    )
    dense_loss.backward()
    _restore_rng(starting_rng)
    result = backward_loaded_parent_batch(
        system, labels, text,
        load_view=lambda source: by_source[source],
        counterfactuals=_empty_cf(), cf_weight=0.2, margin=0.2,
    )
    torch.testing.assert_close(result.scores, dense_scores.detach(), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        torch.tensor(result.loss), dense_loss.detach(), rtol=2e-6, atol=2e-7,
    )
    assert result.replayed_captures == 2
    assert all(parameter.grad is None for parameter in system.frozen_b2.parameters())
    assert any(parameter.grad is not None for parameter in system.head.parameters())
    for (name, actual), (expected_name, expected) in zip(
        system.named_parameters(), dense.named_parameters(), strict=True,
    ):
        assert name == expected_name
        if name.startswith("frozen_b2."):
            assert actual.grad is expected.grad is None
        else:
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-6)


def test_legacy_host_cannot_label_short_synthetic_run_as_registered_matrix_row(tmp_path):
    v2, task, source, config, bindings = _setup(tmp_path)
    system = _legacy(v2.frozen_b2, "old", components=("C01", "C02"))
    source.views = {key: view.capture for key, view in source.views.items()}
    source.legacy_floor_receipt = system.floor_receipt
    source.legacy_training_source_manifest_sha256 = (
        system.floor_receipt.training_source_manifest_sha256
    )
    bindings = replace(
        bindings,
        legacy_floor_receipt_sha256=system.floor_receipt.sha256,
        legacy_training_source_manifest_sha256=system.floor_receipt.training_source_manifest_sha256,
        legacy_frozen_b2_state_sha256=_stable_hash(system.frozen_b2.state_dict()),
    )
    identity = ParentHostRunIdentity("V2-019", "PhaseSet-v0.2-on-B2", "V2-007")
    with pytest.raises(ValueError, match="schedule/population"):
        ContinuousParentTrainingHost(
            system, task, source, config, bindings, run_identity=identity,
        )
    with pytest.raises(ValueError, match="one typed floor receipt"):
        ContinuousParentTrainingHost(
            system, task, source, config,
            replace(bindings, legacy_training_source_manifest_sha256="e" * 64),
            run_identity=identity,
        )
    with pytest.raises(ValueError, match="nonlegacy host"):
        ContinuousParentTrainingHost(v2, task, source, config, bindings)
    main_components = (
        "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
        "C10", "C12", "C13", "C14",
    )
    registered = _legacy(v2.frozen_b2, "old", components=main_components)
    source.legacy_floor_receipt = registered.floor_receipt
    source.legacy_training_source_manifest_sha256 = (
        registered.floor_receipt.training_source_manifest_sha256
    )
    registered_bindings = replace(
        bindings,
        legacy_floor_receipt_sha256=registered.floor_receipt.sha256,
        legacy_frozen_b2_state_sha256=_stable_hash(registered.frozen_b2.state_dict()),
    )
    full_config = replace(
        config, train_components=main_components, validation_components=("C00",),
        epochs=20, parent_batch_size=128,
    )
    with pytest.raises(ValueError, match="exact frozen-matrix"):
        ContinuousParentTrainingHost(
            registered, task, source, full_config, registered_bindings,
            run_identity=ParentHostRunIdentity("V2-049", "A9", "V2-007"),
        )
    with pytest.raises(ValueError, match="missing a requested learning component"):
        ContinuousParentTrainingHost(
            registered, task, source, full_config, registered_bindings,
            run_identity=identity,
        )
