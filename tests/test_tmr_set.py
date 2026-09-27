"""Group adaptation invariants, not a native learning/replication result."""

import copy
from dataclasses import replace
from itertools import permutations

import pytest
import torch

from phaseset_core.tmr_set import (
    TMRGroupCapture,
    TMRGroupWindow,
    TMRSet,
    TMRTextBatch,
    absolute_sincos,
    minimum_assignment,
    track_costs,
)


def _capture(actors=3, frames=9, *, starts=(0.0, 30.0), input_grad=False):
    windows = []
    for start in starts:
        values = torch.randn(actors, frames, 66)
        observed = torch.ones_like(values, dtype=torch.bool)
        observed[-1, 2, :3] = False
        values[~observed] = 0
        values.requires_grad_(input_grad)
        windows.append(
            TMRGroupWindow(
                values, observed, tuple(bytes([i + 1]) * 32 for i in range(actors)), start
            )
        )
    return TMRGroupCapture(tuple(windows))


def _text(rows=3):
    tokens = torch.randn(rows, 5, 768)
    mask = torch.ones(rows, 5, dtype=torch.bool)
    sentences = torch.eye(rows, 768)
    return TMRTextBatch(tokens, mask, sentences)


def _small_model(**kwargs):
    # Analytic algorithm fixture only; never a registered TMR-Set experiment.
    return TMRSet(latent_dim=16, ff_size=32, num_layers=1, num_heads=4, **kwargs)


@pytest.mark.parametrize("size", range(1, 7))
def test_hungarian_against_exhaustive_assignment(size):
    torch.manual_seed(1729 + size)
    cost = torch.randn(size, size, dtype=torch.float64, requires_grad=True)
    assignment = minimum_assignment(cost)
    actual = cost[torch.arange(size), torch.tensor(assignment)].sum()
    expected = min(
        sum(cost[row, column].item() for row, column in enumerate(candidate))
        for candidate in permutations(range(size))
    )
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    actual.backward()
    reference = torch.zeros_like(cost)
    reference[torch.arange(size), torch.tensor(assignment)] = 1
    assert torch.equal(cost.grad, reference)
    assert minimum_assignment(torch.zeros(size, size)) == tuple(range(size))


def test_parent_assignment_cannot_switch_identity_between_windows():
    target = torch.stack((torch.zeros(7, 66), torch.ones(7, 66)))
    observed = torch.ones_like(target, dtype=torch.bool)
    first = track_costs(target, target, observed)
    second = track_costs(target.flip(0), target, observed)
    rows = torch.arange(2)
    assert first[rows, torch.tensor(minimum_assignment(first))].sum() == 0
    assert second[rows, torch.tensor(minimum_assignment(second))].sum() == 0
    total = first + second
    assert total[rows, torch.tensor(minimum_assignment(total))].sum() > 0


def test_track_cost_observed_mean_matches_selected_dense_reference():
    torch.manual_seed(2718)
    target = torch.randn(3, 11, 66)
    prediction = torch.randn_like(target, requires_grad=True)
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[1, 3:6, :9] = False
    costs = track_costs(prediction, target, observed)
    assignment = minimum_assignment(costs)
    loss = costs[torch.arange(3), torch.tensor(assignment)].sum() / observed.sum()
    selected = target[list(assignment)]
    selected_mask = observed[list(assignment)]
    expected = torch.nn.functional.smooth_l1_loss(
        prediction[selected_mask], selected[selected_mask]
    )
    torch.testing.assert_close(loss.to(torch.float32), expected, rtol=2e-6, atol=2e-7)
    got_grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(got_grad, expected_grad, rtol=3e-5, atol=3e-6)


def test_group_permutation_scores_reconstruction_and_input_gradient_equivariance():
    torch.manual_seed(31415)
    model = _small_model(dropout=0, checkpoint_windows=False).eval()
    capture = _capture(input_grad=True)
    text = _text(2)
    positive = torch.ones(1, 2, dtype=torch.bool)
    rng = torch.get_rng_state()
    score = model.score((capture,), text)
    losses = model.compute_loss((capture,), text, positive)
    original_grads = torch.autograd.grad(
        losses["loss"], [window.features for window in capture.windows]
    )
    for order in permutations(range(3)):
        changed = TMRGroupCapture(
            tuple(
                replace(
                    window,
                    features=window.features.detach()[list(order)].clone().requires_grad_(),
                    observed=window.observed[list(order)],
                    actor_commitments=tuple(window.actor_commitments[index] for index in order),
                )
                for window in capture.windows
            )
        )
        torch.set_rng_state(rng)
        got_score = model.score((changed,), text)
        got = model.compute_loss((changed,), text, positive)
        assert torch.equal(got_score, score)
        for key in losses:
            assert torch.equal(got[key], losses[key])
        gradients = torch.autograd.grad(
            got["loss"], [window.features for window in changed.windows]
        )
        for actual, expected in zip(gradients, original_grads, strict=True):
            assert torch.equal(actual, expected[list(order)])


def test_all_windows_absolute_gap_and_late_input_affect_group_encoding():
    torch.manual_seed(1729)
    model = _small_model(dropout=0, checkpoint_windows=False).eval()
    capture = _capture(starts=(0.0, 40.0))
    actual = model.encode_capture(capture)[0]
    without_gap = TMRGroupCapture(
        (capture.windows[0], replace(capture.windows[1], start_seconds=10.0))
    )
    assert not torch.equal(actual, model.encode_capture(without_gap)[0])
    changed_features = capture.windows[-1].features.clone()
    changed_features[:, -1] += 0.5
    changed = TMRGroupCapture(
        (capture.windows[0], replace(capture.windows[-1], features=changed_features))
    )
    assert not torch.equal(actual, model.encode_capture(changed)[0])
    assert not torch.equal(actual, model.encode_capture(TMRGroupCapture(capture.windows[:1]))[0])


def test_checkpointed_and_direct_full_losses_all_gradients_and_rng():
    torch.manual_seed(2718)
    direct = _small_model(dropout=0.1, checkpoint_windows=False).train()
    replay = copy.deepcopy(direct)
    replay.checkpoint_windows = True
    captures = (_capture(3), _capture(4))
    text = _text(3)
    positive = torch.tensor([[True, True, False], [False, False, True]])
    rng = torch.get_rng_state()
    first = direct.compute_loss(captures, text, positive)
    first_rng = torch.get_rng_state()
    first["loss"].backward()
    torch.set_rng_state(rng)
    second = replay.compute_loss(captures, text, positive)
    second["loss"].backward()
    assert torch.equal(first_rng, torch.get_rng_state())
    for key in first:
        assert torch.equal(first[key], second[key])
    for a, b in zip(direct.parameters(), replay.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None
        assert torch.equal(a.grad, b.grad) and torch.isfinite(a.grad).all()


def test_actual_default_set_vae_has_complete_objective_and_all_trainable_gradients():
    torch.manual_seed(1729)
    model = TMRSet(checkpoint_windows=True)
    assert model.latent_dim == 256
    for encoder in (model.actor_encoder, model.capture_encoder, model.text_encoder):
        assert len(encoder.seqTransEncoder.layers) == 6
        assert encoder.seqTransEncoder.layers[0].self_attn.num_heads == 4
        assert encoder.seqTransEncoder.layers[0].linear1.out_features == 1024
    assert len(model.motion_decoder.seqTransDecoder.layers) == 6
    captures = (_capture(3, frames=7, starts=(0.0,)), _capture(4, frames=7, starts=(0.0,)))
    losses = model.compute_loss(
        captures, _text(3), torch.tensor([[True, True, False], [False, False, True]])
    )
    assert all(torch.isfinite(value) and value > 0 for value in losses.values())
    losses["loss"].backward()
    assert all(
        parameter.requires_grad
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for parameter in (
        model.actor_encoder.projection.weight,
        model.capture_encoder.projection.weight,
        model.text_encoder.projection.weight,
        model.motion_decoder.final_layer.weight,
        model.slot_projection[0].weight,
        model.time_projection[0].weight,
    ):
        assert parameter.grad.abs().sum() > 0


def test_no_actor_count_semantic_limit_or_finite_slot_embedding_bank():
    torch.manual_seed(31415)
    model = _small_model(dropout=0, checkpoint_windows=False).eval()
    capture = _capture(17, frames=3, starts=(0.0,))
    with torch.no_grad():
        mean, logvar = model.encode_capture(capture)
        loss = model.reconstruct_capture(mean, capture)
    assert mean.shape == logvar.shape == (16,) and torch.isfinite(loss)
    assert not any(isinstance(module, torch.nn.Embedding) for module in model.modules())
    codes = absolute_sincos(torch.arange(257), 16)
    assert codes.shape == (257, 16)


def test_invalid_chronology_unknown_coordinates_and_nonfinite_matching_fail_explicitly():
    capture = _capture()
    with pytest.raises(ValueError, match="chronology"):
        TMRGroupCapture(tuple(reversed(capture.windows)))
    window = capture.windows[0]
    changed = window.features.clone()
    changed[~window.observed] = 999
    with pytest.raises(ValueError, match="exact"):
        replace(window, features=changed)
    with pytest.raises(ValueError, match="nonfinite"):
        minimum_assignment(torch.full((2, 2), torch.inf))
