"""Mechanism checks, not retrieval results or native-data training."""

from __future__ import annotations

import copy
import itertools
import json
import os

import numpy as np
import pytest
import torch

from phaseset_core.contracts import (
    PreparedGroupBatch,
    group_commitment,
    skeleton_to_activity,
)
from phaseset_core.directional_phase import (
    LocalPhaseConfig,
    canonical_edge_pairs,
    directional_phase_fields,
    iter_local_pair_chunks,
    local_pair_chunk,
    mean_difference_signal_dct,
    signed_joint_velocity,
)
from phaseset_core.periodic import ResourceLimitError
from phaseset_core.temporal_coordination import (
    CalibratedCoordinationScore,
    OrderedCaptureReadout,
    TemporalIncidenceEncoder,
    complete_relation_text_scores,
    coordination_objective,
    verified_counterfactual_loss,
)


def motion(phases=(0.0, 0.0, np.pi / 2), *, order=None, padding=0, missing=False):
    count, length = len(phases), 200
    skeletons = np.zeros((1, count + padding, length + 1, 22, 3), dtype=np.float32)
    actor_mask = np.zeros((1, count + padding), dtype=np.bool_)
    track = np.zeros(skeletons.shape[:-1], dtype=np.bool_)
    keys = tuple(bytes([index + 1]) * 32 for index in range(count))
    rows = tuple(range(count)) if order is None else tuple(order)
    identifiers = [None] * (count + padding)
    time = np.arange(length) / 20
    for destination, actor in enumerate(rows):
        actor_mask[0, destination] = True
        identifiers[destination] = keys[actor]
        skeletons[0, destination, :length, :, 1] = np.float32(actor * 2)
        displacement = np.sin(2 * np.pi * 1.125 * time + phases[actor]).astype(np.float32)
        skeletons[0, destination, :length, 1:, 0] = displacement[:, None]
        track[0, destination, :length] = not missing
    if missing:
        skeletons[:] = 0.0
        # A prepared actor must have at least one tracked joint. Isolated
        # samples provide no observed two-frame velocity or Morlet support.
        track[0, :count, 0, 0] = True
    frame = np.zeros((1, length + 1), dtype=np.bool_)
    frame[:, :length] = True
    return PreparedGroupBatch(
        skeletons,
        actor_mask,
        frame,
        track,
        (tuple(identifiers),),
        (group_commitment(keys),),
    )


def field(batch=None, **kwargs):
    return directional_phase_fields(
        motion() if batch is None else batch,
        energy_floors=np.zeros(6, dtype=np.float64),
        **kwargs,
    )[0]


def test_speed_frontend_collision_but_signed_phase_separates_antiphase():
    inphase = motion((0.0, 0.0))
    antiphase = motion((0.0, np.pi))
    old_a, old_b = skeleton_to_activity(inphase), skeleton_to_activity(antiphase)
    np.testing.assert_allclose(old_a.activities, old_b.activities, atol=2e-6, rtol=1e-6)
    a, b = (
        local_pair_chunk(field(inphase), 0, 1),
        local_pair_chunk(field(antiphase), 0, 1),
    )
    assert a.phase_mask[0, :, 1].all() and b.phase_mask[0, :, 1].all()
    assert float(a.features[0, :, 1, 0].mean()) > 0.999
    assert float(b.features[0, :, 1, 0].mean()) < -0.999


def test_endpoint_exchange_is_complex_conjugation_with_energy_swap():
    values = local_pair_chunk(field(), 0, 3)
    reversed_values = values.reverse_features()
    np.testing.assert_array_equal(reversed_values[..., 0], values.features[..., 0])
    np.testing.assert_allclose(reversed_values[..., 1], -values.features[..., 1], atol=0)
    np.testing.assert_array_equal(reversed_values[..., 3], values.features[..., 4])
    np.testing.assert_array_equal(reversed_values[..., 4], values.features[..., 3])


def test_stationary_motion_has_unobservable_phase_and_observed_support():
    batch = motion((0.0, 0.0))
    skeletons = np.array(batch.skeletons)
    skeletons[:, :, :, :, 0] = 0.0
    stationary = PreparedGroupBatch(
        skeletons,
        batch.actor_mask.copy(),
        batch.frame_mask.copy(),
        batch.track_mask.copy(),
        batch.actor_commitments,
        batch.group_commitments,
    )
    values = local_pair_chunk(field(stationary), 0, 1)
    assert values.support_mask.any()
    assert not values.phase_mask.any()
    assert (values.features[..., :5] == 0).all()
    assert not np.signbit(values.features[..., :5]).any()


def test_missing_pelvis_invalidates_relative_velocity_without_fabrication():
    batch = motion((0.0, 0.0))
    track = np.array(batch.track_mask)
    track[0, 0, 20:25, 0] = False
    skeletons = np.array(batch.skeletons)
    skeletons[0, 0, 20:25, 0] = 0.0
    batch = PreparedGroupBatch(
        skeletons,
        batch.actor_mask.copy(),
        batch.frame_mask.copy(),
        track,
        batch.actor_commitments,
        batch.group_commitments,
    )
    values, mask = signed_joint_velocity(batch)
    assert not mask[0, 0, 20:26].any()
    assert np.all(values[0, 0, 20:26] == 0)
    assert mask[0, 1, 20:26].all()


def test_delay_changes_only_incident_relations():
    unchanged = local_pair_chunk(field(motion((0.0, 0.0, 0.0))), 0, 3)
    delayed = local_pair_chunk(field(motion((0.0, 0.0, np.pi / 2))), 0, 3)
    np.testing.assert_array_equal(unchanged.features[0], delayed.features[0])
    assert not np.array_equal(unchanged.features[1:], delayed.features[1:])


def test_all_actor_permutations_and_padding_leave_physical_relations_exact():
    reference = local_pair_chunk(field(), 0, 3)
    for order in itertools.permutations(range(3)):
        candidate = local_pair_chunk(field(motion(order=order, padding=4)), 0, 3)
        np.testing.assert_array_equal(reference.features, candidate.features)
        np.testing.assert_array_equal(reference.support_mask, candidate.support_mask)


def test_edge_addressing_and_chunks_cover_every_pair_without_dense_graph():
    expected = np.array(list(itertools.combinations(range(8), 2)))
    for start in range(len(expected)):
        np.testing.assert_array_equal(
            canonical_edge_pairs(8, start, len(expected)), expected[start:]
        )
    source = field(motion(tuple(index / 7 for index in range(8))))
    chunks = list(iter_local_pair_chunks(source, edge_chunk_size=5))
    np.testing.assert_array_equal(np.concatenate([value.endpoints for value in chunks]), expected)
    np.testing.assert_array_equal(
        np.concatenate([value.features for value in chunks]),
        local_pair_chunk(source, 0, len(expected)).features,
    )
    with pytest.raises(ResourceLimitError):
        field(motion(), config=LocalPhaseConfig(edge_budget=2))


def test_true_mean_difference_dct_retains_cross_terms():
    signal = np.arange(12, dtype=np.float64).reshape(6, 2) - 4
    same_mean, same_difference = mean_difference_signal_dct(signal, signal)
    opposite_mean, opposite_difference = mean_difference_signal_dct(signal, -signal)
    np.testing.assert_allclose(same_difference, 0, atol=0)
    np.testing.assert_allclose(opposite_mean, 0, atol=0)
    np.testing.assert_allclose(same_mean, opposite_difference, atol=0)
    np.testing.assert_allclose(np.square(same_mean).sum(), np.square(signal).sum(), rtol=1e-14)


def test_k2_topology_and_parameter_gradients_are_exact_zero_and_pair_only_equal():
    torch.manual_seed(1729)
    full = TemporalIncidenceEncoder(32)
    pair = copy.deepcopy(full)
    pair.use_topology = False
    source = field(motion((0.0, np.pi / 2)))
    result, pair_result = full(source), pair(source)
    assert torch.equal(result.embedding, pair_result.embedding)
    assert torch.count_nonzero(result.topology_nodes) == 0
    assert not bool(torch.signbit(result.topology_nodes).any())
    result.embedding.sum().backward()
    topology = list(full.node_mlp.parameters()) + list(full.node_temporal.parameters())
    topology += list(full.edge_delta.parameters()) + [full.topology_scale]
    for parameter in topology:
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        assert not bool(torch.signbit(parameter.grad).any())


@pytest.mark.parametrize("actor_count", [3, 13])
def test_temporal_encoder_is_actor_invariant_and_checkpoint_gradients_match(actor_count):
    torch.manual_seed(2718)
    model = TemporalIncidenceEncoder(32)
    eager = copy.deepcopy(model)
    eager.checkpoint_blocks = False
    phases = tuple(index / 7 for index in range(actor_count))
    source = field(motion(phases))
    first, second = model(source), eager(source)
    assert torch.equal(first.embedding, second.embedding)
    permuted = model(field(motion(phases, order=tuple(reversed(range(actor_count))), padding=3)))
    assert torch.equal(first.embedding, permuted.embedding)
    first.embedding.sum().backward()
    second.embedding.sum().backward()
    for left, right in zip(model.parameters(), eager.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)


def test_group_with_no_derived_support_has_exact_zero_neural_output():
    torch.manual_seed(31415)
    result = TemporalIncidenceEncoder(32)(field(motion(missing=True)))
    assert not result.patch_mask.any()
    assert torch.count_nonzero(result.embedding) == 0
    assert not torch.signbit(result.embedding).any()


def test_capture_readout_distinguishes_AB_from_BA_and_sorts_storage_order():
    torch.manual_seed(1729)
    reader = OrderedCaptureReadout(8)
    windows = torch.randn(2, 8)
    starts, mask = torch.tensor([0.0, 10.0]), torch.ones(2, dtype=torch.bool)
    ab, ba = reader(windows, starts, mask), reader(windows.flip(0), starts, mask)
    assert not torch.allclose(ab, ba)
    assert torch.equal(ab, reader(windows.flip(0), starts.flip(0), mask))
    with pytest.raises(ValueError, match="distinct"):
        reader(windows, torch.tensor([0.0, 0.0]), mask)


def test_calibration_can_correct_a_three_logit_error_and_rejects_scaled_inputs():
    score = CalibratedCoordinationScore(initial_temperature=0.1)
    base = torch.tensor([[0.5, 0.8]])  # wrong candidate leads by 3 base logits
    coordination = torch.tensor([[1.0, -1.0]])
    result = score(base, coordination)
    assert result[0, 0] > result[0, 1]
    with pytest.raises(ValueError, match="cosines"):
        score(base * 10, coordination)


def test_complete_relation_matching_never_recombines_components_of_two_relations():
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    text = torch.tensor([[1.0, 1.0]])
    result = complete_relation_text_scores(tokens, torch.ones(1, 2, dtype=torch.bool), text)
    torch.testing.assert_close(result, torch.tensor([[2**-0.5]]))


def test_counterfactual_loss_excludes_unverified_examples_and_has_finite_gradients():
    positive = torch.tensor([0.2, -100.0], requires_grad=True)
    negative = torch.tensor([0.1, 100.0], requires_grad=True)
    valid = torch.tensor([True, False])
    loss = verified_counterfactual_loss(positive, negative, valid)
    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(0.1)))
    loss.backward()
    assert positive.grad[1] == negative.grad[1] == 0
    empty = verified_counterfactual_loss(positive, negative, torch.zeros(2, dtype=torch.bool))
    assert empty == 0
    logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, False, True], [False, True, False]])
    objective = coordination_objective(
        logits,
        positives,
        cf_positive_scores=positive,
        cf_negative_scores=negative,
        verified_negative_mask=valid,
    )
    objective.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(
    os.environ.get("PHASESET_V2_CUDA_WITNESS") != "1", reason="server CUDA witness only"
)
def test_full_width_cuda_forward_backward_under_two_gib():
    torch.manual_seed(1729)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / total, 0)
    model = TemporalIncidenceEncoder(512).cuda()
    output = model(field())
    assert output.embedding.shape == (512,)
    assert torch.isfinite(output.embedding).all()
    output.embedding.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) for p in model.node_mlp.parameters())
    assert torch.cuda.max_memory_allocated() < 2 * 1024**3
    print(
        json.dumps(
            {
                "witness": "mechanism_only_not_retrieval_result",
                "width": 512,
                "actors": 3,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        )
    )
