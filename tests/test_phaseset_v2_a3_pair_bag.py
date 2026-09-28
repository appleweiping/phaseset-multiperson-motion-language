"""A3 keeps the full head's capacity while removing actor--edge incidence."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from phaseset_core.continuous_retrieval import ContinuousRetrievalSystem
from phaseset_core.directional_phase import canonical_edge_pairs
from phaseset_core.temporal_coordination import TemporalIncidenceEncoder
from phaseset_core.training import BaseRetrievalSystem
from test_continuous_parent_training import _MockB2
from test_phaseset_v2_coordination import field, motion


def _parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def test_a3_full_capacity_active_topology_and_actor_permutation() -> None:
    torch.manual_seed(1729)
    full = TemporalIncidenceEncoder(32, checkpoint_blocks=False)
    bag = copy.deepcopy(full)
    bag.pair_bag = True
    assert _parameters(bag) == _parameters(full)
    source = field(motion((0.0, 0.3, 1.1, 2.0)))
    output = bag(source)
    assert bool(output.topology_mask.any())
    assert bool(output.topology_nodes.abs().sum() > 0)
    permuted = bag(field(motion((0.0, 0.3, 1.1, 2.0), order=(3, 1, 0, 2), padding=5)))
    torch.testing.assert_close(output.embedding, permuted.embedding, rtol=0, atol=2e-7)
    output.embedding.sum().backward()
    for module in (bag.node_mlp, bag.node_temporal, bag.edge_delta):
        assert any(
            parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            for parameter in module.parameters()
        )
    assert bag.topology_scale.grad is not None
    assert bool(bag.topology_scale.grad.abs().sum() > 0)


def test_a3_identical_pair_packet_multiset_removes_incidence_only() -> None:
    torch.manual_seed(2718)
    source = field(motion((0.0, 0.3, 1.1, 2.0)))
    endpoints = torch.from_numpy(canonical_edge_pairs(4, 0, 6).copy())
    patches, width = source.patch_count, 16
    values = torch.arange(6 * patches * width, dtype=torch.float32).reshape(6, patches, width)
    a = (values.remainder(31) - 15) / 7
    b = (values.remainder(19) - 9) / 5
    mask = torch.ones((6, patches), dtype=torch.bool)
    routing = (5, 1, 4, 3, 2, 0)
    text = torch.randn(3, width)

    def patched(model: TemporalIncidenceEncoder, order: tuple[int, ...]):
        def half_edges(_field, hidden, start, stop):
            index = torch.tensor(order[start:stop], device=hidden.device)
            return (
                a.to(hidden.device)[index],
                b.to(hidden.device)[index],
                endpoints.to(hidden.device)[start:stop],
                mask.to(hidden.device)[start:stop],
            )

        model._half_edges = half_edges
        return model(source), model.score_text(source, text)

    bag = TemporalIncidenceEncoder(width, pair_bag=True, checkpoint_blocks=False).eval()
    bag_rerouted = copy.deepcopy(bag)
    baseline, baseline_score = patched(bag, tuple(range(6)))
    rerouted, rerouted_score = patched(bag_rerouted, routing)
    torch.testing.assert_close(baseline.topology_nodes, rerouted.topology_nodes, rtol=0, atol=2e-7)
    torch.testing.assert_close(baseline.embedding, rerouted.embedding, rtol=0, atol=2e-7)
    torch.testing.assert_close(baseline_score.cosine, rerouted_score.cosine, rtol=0, atol=2e-7)
    assert torch.equal(baseline_score.periodic_support, rerouted_score.periodic_support)

    full = TemporalIncidenceEncoder(width, checkpoint_blocks=False).eval()
    full_rerouted = copy.deepcopy(full)
    (attached, _), (changed, _) = patched(full, tuple(range(6))), patched(full_rerouted, routing)
    assert not torch.equal(attached.topology_nodes, changed.topology_nodes)


def test_a3_k2_topology_and_gradients_are_positive_zero_and_full_equal() -> None:
    torch.manual_seed(31415)
    full = TemporalIncidenceEncoder(16, checkpoint_blocks=False)
    bag = copy.deepcopy(full)
    bag.pair_bag = True
    source = field(motion((0.0, np.pi / 2)))
    expected, result = full(source), bag(source)
    assert torch.equal(expected.embedding, result.embedding)
    assert torch.equal(expected.patch_tokens, result.patch_tokens)
    assert torch.count_nonzero(result.topology_nodes) == 0
    assert not bool(torch.signbit(result.topology_nodes).any())
    result.embedding.sum().backward()
    topology = list(bag.node_mlp.parameters()) + list(bag.node_temporal.parameters())
    topology += list(bag.edge_delta.parameters()) + [bag.topology_scale]
    for parameter in topology:
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        assert not bool(torch.signbit(parameter.grad).any())


def test_a3_crosses_64_edge_boundary_without_actor_order_shortcut() -> None:
    torch.manual_seed(1729)
    phases = tuple(index / 11 for index in range(13))
    model = TemporalIncidenceEncoder(8, pair_bag=True, checkpoint_blocks=False).eval()
    original = model(field(motion(phases)))
    reordered = model(field(motion(phases, order=tuple(reversed(range(13))), padding=2)))
    assert torch.equal(original.embedding, reordered.embedding)
    assert torch.equal(original.patch_tokens, reordered.patch_tokens)
    assert bool(original.topology_mask.any())
    checkpointed = copy.deepcopy(model)
    checkpointed.checkpoint_blocks = True
    replay = checkpointed(field(motion(phases)))
    assert torch.equal(original.embedding, replay.embedding)
    source = field(motion(phases))
    text = torch.randn(2, 8)
    eager_scores = model.score_text(source, text)
    replay_scores = checkpointed.score_text(source, text)
    assert torch.equal(eager_scores.cosine, replay_scores.cosine)
    assert torch.equal(eager_scores.periodic_support, replay_scores.periodic_support)
    eager_scores.cosine.sum().backward()
    replay_scores.cosine.sum().backward()
    for eager_parameter, replay_parameter in zip(model.parameters(), checkpointed.parameters(), strict=True):
        assert eager_parameter.grad is not None and replay_parameter.grad is not None
        torch.testing.assert_close(eager_parameter.grad, replay_parameter.grad, rtol=1e-6, atol=1e-7)


def test_a3_control_identity_exclusivity_and_exact_512d_capacity() -> None:
    def system(**kwargs):
        return ContinuousRetrievalSystem(
            BaseRetrievalSystem(_MockB2(), embedding_dim=512), **kwargs
        )

    full = system()
    a3 = system(pair_bag=True)
    assert _parameters(a3.coordination) == _parameters(full.coordination)
    with pytest.raises(ValueError, match="relation kind"):
        a3.set_extra_state(full.get_extra_state())
    with pytest.raises(ValueError, match="relation kind"):
        full.set_extra_state(a3.get_extra_state())
    with pytest.raises(ValueError, match="relation kind"):
        a3.load_state_dict(full.state_dict())
    with pytest.raises(ValueError, match="separate phase control"):
        system(relation_kind="true_mean_difference_dct", pair_bag=True)
    for conflict in (
        {"use_topology": False},
        {"strip_phase": True},
        {"incidence_shuffle_seed": 1729},
        {"order_free": True},
    ):
        with pytest.raises(ValueError, match="otherwise full phase"):
            system(pair_bag=True, **conflict)
