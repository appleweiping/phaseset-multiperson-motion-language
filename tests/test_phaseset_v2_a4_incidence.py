"""A4 shuffles only explicit actor-incidence routing, not physical packets."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from phaseset_core.continuous_retrieval import ContinuousRetrievalSystem
from phaseset_core.temporal_coordination import (
    TemporalIncidenceEncoder,
    _shuffle_incidence_values,
)
from phaseset_core.training import BaseRetrievalSystem
from test_continuous_parent_training import _MockB2
from test_phaseset_v2_coordination import field, motion


def test_a4_reassigns_only_supported_half_edges_and_keeps_gradients() -> None:
    a = torch.tensor([[[1.0], [7.0]], [[2.0], [8.0]], [[3.0], [9.0]]], requires_grad=True)
    b = torch.tensor([[[4.0], [10.0]], [[5.0], [11.0]], [[6.0], [12.0]]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, True], [True, True]])
    kwargs = {
        "seed": 1729,
        "block_start": 0,
    }
    left, right = _shuffle_incidence_values(a, b, mask, **kwargs)
    repeated = _shuffle_incidence_values(a, b, mask, **kwargs)
    assert torch.equal(left, repeated[0]) and torch.equal(right, repeated[1])
    assert left[0, 1].item() == 7.0 and right[0, 1].item() == 10.0
    for patch in range(2):
        original = sorted(
            value
            for edge in range(3)
            if bool(mask[edge, patch])
            for value in (a[edge, patch, 0].item(), b[edge, patch, 0].item())
        )
        shuffled = sorted(
            value
            for edge in range(3)
            if bool(mask[edge, patch])
            for value in (left[edge, patch, 0].item(), right[edge, patch, 0].item())
        )
        assert shuffled == original
    assert not torch.equal(left, a) or not torch.equal(right, b)
    (left.sum() + right.sum()).backward()
    assert torch.equal(a.grad, torch.ones_like(a))
    assert torch.equal(b.grad, torch.ones_like(b))


def test_a4_changes_incidence_nodes_but_not_packets_or_actor_order() -> None:
    torch.manual_seed(1729)
    full = TemporalIncidenceEncoder(16, checkpoint_blocks=False).eval()
    shuffled = copy.deepcopy(full)
    shuffled.incidence_shuffle_seed = 1729
    physical = field(motion((0.0, 0.7, 1.8)))
    original_hidden, original_nodes, original_mask = full._encode_context(physical)
    shuffled_hidden, shuffled_nodes, shuffled_mask = shuffled._encode_context(physical)
    assert torch.equal(original_hidden, shuffled_hidden)
    assert torch.equal(original_mask, shuffled_mask)
    assert bool(original_mask.any())
    assert not torch.equal(original_nodes, shuffled_nodes)
    before = full._half_edges(physical, original_hidden, 0, physical.pair_count)
    after = shuffled._half_edges(physical, shuffled_hidden, 0, physical.pair_count)
    assert all(torch.equal(x, y) for x, y in zip(before, after, strict=True))
    first = shuffled(physical).embedding
    renamed = replace(
        physical,
        actor_commitments=tuple(bytes([index]) * 32 for index in (4, 5, 6)),
    )
    assert torch.equal(first, shuffled(renamed).embedding)
    reordered = field(motion((0.0, 0.7, 1.8), order=(2, 0, 1), padding=2))
    permuted = shuffled(reordered).embedding
    assert torch.equal(first, permuted)


def test_a4_identity_is_checkpoint_bound_and_separate_from_a6() -> None:
    phase = ContinuousRetrievalSystem(
        BaseRetrievalSystem(_MockB2(), embedding_dim=512),
        incidence_shuffle_seed=1729,
    )
    same = ContinuousRetrievalSystem(
        BaseRetrievalSystem(_MockB2(), embedding_dim=512),
        incidence_shuffle_seed=1729,
    )
    different = ContinuousRetrievalSystem(
        BaseRetrievalSystem(_MockB2(), embedding_dim=512),
        incidence_shuffle_seed=2718,
    )
    same.set_extra_state(phase.get_extra_state())
    with pytest.raises(ValueError, match="relation kind"):
        different.set_extra_state(phase.get_extra_state())
    with pytest.raises(ValueError, match="separate phase control"):
        ContinuousRetrievalSystem(
            BaseRetrievalSystem(_MockB2(), embedding_dim=512),
            relation_kind="true_mean_difference_dct",
            incidence_shuffle_seed=1729,
        )
