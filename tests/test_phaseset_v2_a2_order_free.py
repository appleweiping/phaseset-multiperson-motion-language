"""A2 removes neural local/capture order without removing the physical input."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from phaseset_core.continuous_retrieval import ContinuousRetrievalSystem
from phaseset_core.temporal_coordination import (
    MaskedTemporalEncoder,
    OrderFreeTemporalEncoder,
    TemporalIncidenceEncoder,
)
from phaseset_core.training import BaseRetrievalSystem
from test_continuous_parent_training import _MockB2
from test_phaseset_v2_coordination import field, motion


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def test_a2_independent_temporal_mlp_matches_capacity_and_ignores_patch_order() -> None:
    ratio = _count(OrderFreeTemporalEncoder(512)) / _count(MaskedTemporalEncoder(512))
    assert abs(ratio - 1) < 0.01
    torch.manual_seed(1729)
    model = OrderFreeTemporalEncoder(8)
    sequence = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, False, True, True, False], [True, True, False, True, True]])
    permutation = torch.tensor([4, 2, 0, 3, 1])
    original, embedding = model(sequence, mask)
    reordered, permuted_embedding = model(sequence[:, permutation], mask[:, permutation])
    torch.testing.assert_close(reordered, original[:, permutation], rtol=0, atol=0)
    torch.testing.assert_close(permuted_embedding, embedding, rtol=0, atol=2e-7)
    assert not bool(original[~mask].view(torch.int32).any())
    embedding.sum().backward()
    assert bool(torch.isfinite(sequence.grad).all())
    assert not bool(sequence.grad[~mask].view(torch.int32).any())
    assert bool(sequence.grad[mask].abs().sum() > 0)


def test_a2_complete_phase_field_is_order_free_but_full_temporal_is_not() -> None:
    torch.manual_seed(2718)
    physical = field(motion((0.0, 0.7, 1.8)))
    order = np.arange(physical.patch_count - 1, -1, -1)
    reordered = replace(
        physical,
        intervals=tuple(physical.intervals[index] for index in order),
        actor_features=physical.actor_features[:, order].copy(),
        actor_patch_mask=physical.actor_patch_mask[:, order].copy(),
        root_positions=physical.root_positions[:, order].copy(),
        root_patch_mask=physical.root_patch_mask[:, order].copy(),
    )
    a2 = TemporalIncidenceEncoder(8, order_free=True, checkpoint_blocks=False).eval()
    full = TemporalIncidenceEncoder(8, checkpoint_blocks=False).eval()
    assert abs(_count(a2) / _count(full) - 1) < 0.05
    text = torch.randn(3, 8)
    first = a2.score_text(physical, text)
    second = a2.score_text(reordered, text)
    torch.testing.assert_close(
        first.coordination.embedding, second.coordination.embedding, rtol=0, atol=2e-7
    )
    torch.testing.assert_close(first.cosine, second.cosine, rtol=0, atol=2e-7)
    torch.testing.assert_close(
        first.coordination.patch_tokens[order],
        second.coordination.patch_tokens,
        rtol=0,
        atol=2e-7,
    )
    assert not torch.equal(full(physical).embedding, full(reordered).embedding)
    actor_reordered = field(motion((0.0, 0.7, 1.8), order=(2, 0, 1), padding=2))
    torch.testing.assert_close(
        first.coordination.embedding,
        a2(actor_reordered).embedding,
        rtol=0,
        atol=2e-7,
    )
    torch.testing.assert_close(
        first.cosine,
        a2.score_text(actor_reordered, text).cosine,
        rtol=0,
        atol=2e-7,
    )
    first.cosine.sum().backward()
    for module in (a2.actor_temporal, a2.node_temporal, a2.edge_temporal, a2.group_temporal):
        assert any(
            parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            for parameter in module.parameters()
        )


def test_a2_a3_a5_checkpoint_control_identity_and_a6_exclusion() -> None:
    def system(**kwargs):
        return ContinuousRetrievalSystem(
            BaseRetrievalSystem(_MockB2(), embedding_dim=512), **kwargs
        )

    full = system()
    a2 = system(order_free=True)
    a3 = system(pair_bag=True)
    legacy_pair_only = system(use_topology=False)
    a5 = system(strip_phase=True)
    for variant in (a2, a3, legacy_pair_only, a5):
        with pytest.raises(ValueError, match="relation kind"):
            variant.set_extra_state(full.get_extra_state())
        with pytest.raises(ValueError, match="relation kind"):
            full.set_extra_state(variant.get_extra_state())
    with pytest.raises(ValueError, match="separate phase control"):
        system(relation_kind="true_mean_difference_dct", order_free=True)
    with pytest.raises(ValueError, match="otherwise full phase"):
        system(order_free=True, use_topology=False)
    with pytest.raises(ValueError, match="otherwise full phase"):
        system(order_free=True, strip_phase=True)
    with pytest.raises(ValueError, match="otherwise full phase"):
        system(order_free=True, incidence_shuffle_seed=1729)
    with pytest.raises(ValueError, match="relation kind"):
        a2.load_state_dict(full.state_dict())
