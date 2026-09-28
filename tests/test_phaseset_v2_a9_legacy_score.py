"""A9 changes the old PhaseSet score calibration, not its six-token features."""

from __future__ import annotations

import pytest
import torch

from phaseset_core.objectives import PhaseSetRetrievalHead
from phaseset_core.temporal_coordination import LegacyPhaseSetCalibratedHead


def test_a9_preserves_old_band_text_representation_bitwise() -> None:
    torch.manual_seed(1729)
    old = PhaseSetRetrievalHead(8)
    calibrated = LegacyPhaseSetCalibratedHead(8)
    calibrated.text_band_mlp.load_state_dict(old.text_band_mlp.state_dict())
    tokens = torch.randn((3, 6, 8), dtype=torch.float32).contiguous()
    mask = torch.tensor(
        [[True] * 6, [True, False, True, False, False, False], [False] * 6],
        dtype=torch.bool,
    ).contiguous()
    text = torch.randn((4, 8), dtype=torch.float32).contiguous()
    base_cosine = torch.zeros((3, 4), dtype=torch.float32).contiguous()

    old_output = old(tokens, mask, text, base_cosine)
    new_output = calibrated(tokens, mask, text, base_cosine)
    assert torch.equal(new_output.text_band_embeddings, old_output.text_band_embeddings)
    assert torch.equal(new_output.legacy_relation_cosine, old_output.periodic_scores)
    assert torch.equal(new_output.periodic_support, mask.any(dim=1))


def test_a9_uses_common_cosine_scale_and_exact_global_fallback() -> None:
    torch.manual_seed(2718)
    head = LegacyPhaseSetCalibratedHead(8)
    tokens = torch.randn((2, 6, 8), dtype=torch.float32, requires_grad=True)
    mask = torch.tensor(
        [[False] * 6, [True] * 6], dtype=torch.bool
    ).contiguous()
    text = torch.randn((3, 8), dtype=torch.float32).contiguous()
    base_cosine = torch.tensor(
        [[0.2, -0.3, 0.4], [-0.1, 0.5, 0.7]], dtype=torch.float32
    ).contiguous()

    output = head(tokens, mask, text, base_cosine)
    alpha = head.calibration.mixture_logit.sigmoid()
    scale = head.calibration.log_scale.exp()
    torch.testing.assert_close(alpha, torch.tensor(0.1))
    torch.testing.assert_close(output.scores[0], scale * base_cosine[0])
    torch.testing.assert_close(
        output.scores[1],
        scale * ((1 - alpha) * base_cosine[1] + alpha * output.legacy_relation_cosine[1]),
    )
    output.scores.sum().backward()
    assert torch.count_nonzero(tokens.grad[0]) == 0
    assert torch.count_nonzero(tokens.grad[1]) > 0
    assert head.calibration.mixture_logit.grad is not None
    assert torch.isfinite(head.calibration.mixture_logit.grad).all()
    assert torch.count_nonzero(head.calibration.mixture_logit.grad) > 0
    assert head.calibration.log_scale.grad is not None
    assert torch.isfinite(head.calibration.log_scale.grad).all()
    assert torch.count_nonzero(head.calibration.log_scale.grad) > 0
    assert head.text_band_mlp.band_embedding.grad is not None


def test_a9_refuses_base_logits_or_trainable_base_scores() -> None:
    head = LegacyPhaseSetCalibratedHead(8)
    tokens = torch.zeros((1, 6, 8), dtype=torch.float32).contiguous()
    mask = torch.ones((1, 6), dtype=torch.bool).contiguous()
    text = torch.ones((2, 8), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="cosines"):
        head(tokens, mask, text, torch.tensor([[5.0, -2.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="frozen"):
        head(
            tokens,
            mask,
            text,
            torch.zeros((1, 2), dtype=torch.float32, requires_grad=True),
        )
    with pytest.raises(ValueError, match="frozen"):
        head(tokens, mask, text, torch.zeros((2, 1), dtype=torch.float32))
