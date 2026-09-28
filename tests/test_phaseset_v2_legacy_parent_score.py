"""Data-free old/A9 complete-parent score and host-boundary qualifications."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from phaseset_core.capture_validation import _validate_text_batch
from phaseset_core.continuous_parent_host import ContinuousParentTrainingHost
from phaseset_core.controls import PhaseSetSystem
from phaseset_core.legacy_continuous_retrieval import LegacyWholeCaptureRetrievalSystem
from phaseset_core.legacy_scalar_calibration import (
    LEGACY_SCALAR_FLOOR_SCHEMA,
    LegacyScalarFloorReceipt,
    legacy_scalar_config_sha256,
)
from phaseset_core.periodic import BAND_COUNT
from test_continuous_parent_host import _setup
from test_continuous_parent_training import _case


def _legacy(base, mode: str) -> LegacyWholeCaptureRetrievalSystem:
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
        ("C01",),
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
def test_legacy_parent_score_is_not_yet_a_registered_training_host(tmp_path, mode):
    v2, task, source, config, bindings = _setup(tmp_path)
    system = _legacy(v2.frozen_b2, mode)
    with pytest.raises(TypeError, match="base or V2 scorer"):
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
