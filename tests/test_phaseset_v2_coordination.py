"""Mechanism checks, not retrieval results or native-data training."""

from __future__ import annotations

import copy
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path

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
    MaskedTemporalEncoder,
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


def test_handoff_fold_components_are_exact_disjoint_and_bound_to_87_stage_matrix():
    matrix = json.loads(
        (Path(__file__).parents[1] / "configs/phaseset_v2_experiment_matrix.json").read_text()
    )
    components = matrix["pilot_and_group_fold_components"]
    assert components["pilot_train"] == ["C01", "C02"]
    assert components["pilot_validation"] == components["fold_validation"] == ["C03"]
    assert [row["held_out"] for row in components["folds"]] == [
        ["C00", "C06", "C10", "C14"],
        ["C04", "C07", "C12"],
        ["C05", "C08", "C13"],
    ]
    sealed = set(matrix["primary_split"]["sealed_test_component_labels"])
    assert sealed == {"C09", "C11", "C15"}
    development = set(components["development"])
    assert not development & sealed
    assert len(development) == 13
    held_out = []
    for fold in components["folds"]:
        train, test = set(fold["train"]), set(fold["held_out"])
        assert train == development - test - {"C03"}
        assert not (train | test) & sealed
        held_out.extend(fold["held_out"])
        rows = [row for row in matrix["runs"] if row["split"] == fold["run_split"]]
        assert len(rows) == 9
        assert {row["seed"] for row in rows} == {1729, 2718, 31415}
    assert len(held_out) == len(set(held_out)) == 10
    assert len(matrix["runs"]) == 87
    assert sum(matrix["pilot_allocation"].values()) == 12
    assert matrix["pilot_allocation"]["B2"] == 1
    assert matrix["budget"]["pilot_max_formal_step_fraction"] == 0.2
    assert matrix["frozen_strong_baseline"] is None  # selection requires real pilot
    assert all(row["status"] == "PLANNED" for row in matrix["runs"])


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


@pytest.mark.parametrize("chunk_frames", [1, 7, 31, 64])
def test_halo_convolution_matches_unchunked_responses_and_gap_masks(chunk_frames):
    batch = motion()
    coordinates, tracked = np.array(batch.skeletons), np.array(batch.track_mask)
    tracked[0, 1, 63:67] = False
    coordinates[0, 1, 63:67] = 0.0
    batch = PreparedGroupBatch(
        coordinates,
        batch.actor_mask.copy(),
        batch.frame_mask.copy(),
        tracked,
        batch.actor_commitments,
        batch.group_commitments,
    )
    reference = field(batch, config=LocalPhaseConfig(convolution_chunk_frames=1000))
    candidate = field(batch, config=LocalPhaseConfig(convolution_chunk_frames=chunk_frames))
    for left, right in zip(reference.responses, candidate.responses, strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(reference.response_masks, candidate.response_masks, strict=True):
        np.testing.assert_array_equal(left, right)
    assert not candidate.response_masks[0][1].all()
    np.testing.assert_array_equal(
        local_pair_chunk(reference, 0, 3).features,
        local_pair_chunk(candidate, 0, 3).features,
    )


def test_actor_temporal_history_is_encoded_before_incident_messages():
    torch.manual_seed(31415)
    model = TemporalIncidenceEncoder(8)
    source = field()
    captured = []
    handle = model.actor_temporal.register_forward_hook(
        lambda module, inputs, output: captured.append((inputs, output))
    )
    try:
        model(source)
    finally:
        handle.remove()
    assert len(captured) == 1
    (values, mask), (outputs, final) = captured[0]
    whole, whole_states = model.actor_temporal.encode_chunk(values, mask)
    first, carried = model.actor_temporal.encode_chunk(values[:, :3], mask[:, :3])
    tail, carried = model.actor_temporal.encode_chunk(values[:, 3:], mask[:, 3:], carried)
    assert torch.equal(outputs, torch.cat((first, tail), dim=1))
    assert torch.equal(outputs, whole) and torch.equal(final, whole_states[-1])
    assert torch.equal(carried, whole_states)


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
    np.testing.assert_allclose(2 * same_mean, opposite_difference, atol=0)
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


def test_a7_generic_local_uses_actor_kinematics_but_never_physical_relations(monkeypatch):
    torch.manual_seed(1729)
    source = field()
    generic = TemporalIncidenceEncoder(32, generic_local=True, checkpoint_blocks=False)
    text = torch.randn(2, 32)
    original = generic.score_text(source, text)
    # A7 must not depend on Morlet arrays, masks, centers or fitted floors.
    phase_free = replace(
        source,
        responses=(),
        response_masks=(),
        response_centers=(),
        energy_floors=np.full(6, np.inf),
    )

    def reject_physical(*_args):
        raise AssertionError("A7 read a physical pair relation")

    monkeypatch.setattr("phaseset_core.temporal_coordination.local_pair_chunk", reject_physical)
    changed = generic.score_text(phase_free, text)
    assert torch.equal(original.cosine, changed.cosine)
    assert torch.equal(original.coordination.embedding, changed.coordination.embedding)
    assert bool(changed.periodic_support)
    with pytest.raises(AssertionError, match="physical pair"):
        TemporalIncidenceEncoder(32, checkpoint_blocks=False)(source)


def test_a7_permutation_padding_gradient_and_capacity_contract():
    torch.manual_seed(2718)
    generic = TemporalIncidenceEncoder(32, generic_local=True, checkpoint_blocks=False)
    full = TemporalIncidenceEncoder(32, checkpoint_blocks=False)
    phase = (0.0, 0.25, 0.75, 1.5)
    original = generic(field(motion(phase)))
    permuted = generic(field(motion(phase, order=(3, 1, 0, 2), padding=5)))
    assert torch.equal(original.embedding, permuted.embedding)
    assert torch.equal(original.patch_mask, permuted.patch_mask)
    assert generic.half_edge[0].in_features == 2 * 32 + 3
    assert full.half_edge[0].in_features == 2 * 32 + 6 * 7 + 3
    full_count = sum(parameter.numel() for parameter in full.parameters())
    generic_count = sum(parameter.numel() for parameter in generic.parameters())
    assert abs(full_count - generic_count) / full_count < 0.05
    original.embedding.sum().backward()
    assert generic.half_edge[0].weight.grad is not None
    assert torch.count_nonzero(generic.half_edge[0].weight.grad) > 0


def test_a7_missing_tracks_are_exact_zero_and_k2_topology_stays_gated():
    torch.manual_seed(31415)
    generic = TemporalIncidenceEncoder(32, generic_local=True, checkpoint_blocks=False)
    missing = generic(field(motion(missing=True)))
    assert not missing.patch_mask.any()
    assert torch.count_nonzero(missing.embedding) == 0
    assert not torch.signbit(missing.embedding).any()
    dyad = field(motion((0.0, 0.3)))
    paired = copy.deepcopy(generic)
    paired.use_topology = False
    group_output, pair_output = generic(dyad), paired(dyad)
    assert torch.equal(group_output.embedding, pair_output.embedding)
    assert torch.count_nonzero(group_output.topology_nodes) == 0
    group_output.embedding.sum().backward()
    for parameter in generic.node_mlp.parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
    with pytest.raises(ValueError, match="A7 must use"):
        TemporalIncidenceEncoder(32, generic_local=True, use_topology=False)


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
    score = CalibratedCoordinationScore(initial_temperature=0.1, initial_mixture=0.2)
    base = torch.tensor([[0.5, 0.8]])  # wrong candidate leads by 3 base logits
    coordination = torch.tensor([[1.0, -1.0]])
    result = score(base, coordination)
    assert result[0, 0] > result[0, 1]
    with pytest.raises(ValueError, match="cosines"):
        score(base * 10, coordination)


def test_two_layer_temporal_state_continues_across_chunks_without_order_loss():
    torch.manual_seed(1729)
    encoder = MaskedTemporalEncoder(8)
    sequence = torch.randn(2, 9, 8)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 3:5] = False
    whole, final_states = encoder.encode_chunk(sequence, mask)
    first, states = encoder.encode_chunk(sequence[:, :4], mask[:, :4])
    second, continued_states = encoder.encode_chunk(sequence[:, 4:], mask[:, 4:], states)
    assert states.shape == (2, 2, 8)
    assert torch.equal(whole, torch.cat((first, second), dim=1))
    assert torch.equal(final_states, continued_states)
    assert torch.count_nonzero(whole[0, 3:5]) == 0
    whole.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_calibration_default_and_unsupported_motion_falls_back_without_relation_gradient():
    score = CalibratedCoordinationScore(initial_temperature=0.1)
    torch.testing.assert_close(score.mixture_logit.sigmoid(), torch.tensor(0.1))
    base = torch.tensor([[0.5, -0.8], [0.2, -0.1]])
    relation = torch.tensor([[-1.0, 1.0], [0.9, 0.8]], requires_grad=True)
    result = score(base, relation, coordination_support=torch.tensor([False, True]))
    torch.testing.assert_close(result[0], 10 * base[0])
    result.sum().backward()
    assert torch.count_nonzero(relation.grad[0]) == 0
    assert torch.count_nonzero(relation.grad[1]) == 2
    assert score.mixture_logit.grad is not None and score.mixture_logit.grad != 0
    with pytest.raises(ValueError, match="boolean"):
        score(base, relation, coordination_support=torch.ones(2))


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
