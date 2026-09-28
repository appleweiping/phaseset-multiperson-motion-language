"""Analytic software checks; real-data qualification is a separate attempt."""

from dataclasses import replace
import itertools

import pytest
import torch
from torch.nn import functional as F

from phaseset_core.tmr_set import TMRGroupCapture, TMRGroupWindow
from phaseset_core.wamo_set import (
    LearnableSWT,
    WaMoConfig,
    WaMoSet,
    partial_frame_permutation,
    positive_contrastive,
    temporal_labels,
)


def fixture_capture(*, k=3, length=19, starts=(0.0, 30.0), grad=False):
    windows = []
    keys = tuple(bytes([index + 1]) * 32 for index in range(k))
    for start in starts:
        x = torch.randn(k, length, 66)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, 2:4, :3] = False
        if length > 4:
            mask[0, 0, :] = False
        x = x.masked_fill(~mask, 0).requires_grad_(grad)
        windows.append(TMRGroupWindow(x, mask, keys, start))
    return TMRGroupCapture(tuple(windows))


def permute(capture, order):
    return TMRGroupCapture(
        tuple(
            TMRGroupWindow(
                window.features[list(order)],
                window.observed[list(order)],
                tuple(window.actor_commitments[i] for i in order),
                window.start_seconds,
            )
            for window in capture.windows
        )
    )


def small(*, dropout=0.0, checkpoint=True):
    return WaMoSet(
        WaMoConfig(width=16, ff_size=32, layers=1, dropout=dropout), checkpoint_windows=checkpoint
    )


def scalar_filter(x, taps, dilation, direction):
    # Independent direct coordinate indexing of paper equations, no roll/conv.
    return torch.stack(
        [
            sum(
                taps[j] * x[:, (n - direction * dilation * j) % x.shape[1]]
                for j in range(len(taps))
            )
            for n in range(x.shape[1])
        ],
        dim=1,
    )


@pytest.mark.parametrize("length", [1, 7, 32, 200])
def test_swt_equations_no_decimation_haar_roundtrip_and_shift(length):
    torch.manual_seed(1729)
    wavelet = LearnableSWT()
    x = torch.randn(3, length, 66, dtype=torch.float64)
    wavelet = wavelet.double()
    bands = wavelet(x)
    low = x
    oracle = []
    for level in range(3):
        oracle.append(scalar_filter(low, wavelet.analysis_high, 2**level, -1))
        low = scalar_filter(low, wavelet.analysis_low, 2**level, -1)
    oracle.append(low)
    assert all(band.shape == x.shape for band in bands)
    for actual, expected in zip(bands, oracle, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Initialization was created in float32 then cast to float64.
    torch.testing.assert_close(wavelet.inverse(bands), x, rtol=3e-7, atol=3e-7)
    shifted = wavelet(torch.roll(x, 2, 1))
    for actual, original in zip(shifted, bands, strict=True):
        assert torch.equal(actual, torch.roll(original, 2, 1))


def test_learned_synthesis_independent_oracle_and_all_filter_gradients():
    torch.manual_seed(2718)
    wavelet = LearnableSWT().double()
    with torch.no_grad():
        for parameter in wavelet.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.03)
    x = torch.randn(2, 13, 5, dtype=torch.float64, requires_grad=True)
    bands = wavelet(x)
    low = bands[-1]
    for level in reversed(range(3)):
        low = scalar_filter(low, wavelet.synthesis_low, 2**level, 1) + scalar_filter(
            bands[level], wavelet.synthesis_high, 2**level, 1
        )
    torch.testing.assert_close(wavelet.inverse(bands), low, rtol=0, atol=0)
    wavelet.inverse(bands).square().mean().backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert all(p.grad.isfinite().all() and p.grad.abs().sum() > 0 for p in wavelet.parameters())


def test_partial_frame_shuffle_keeps_masks_origin_labels_and_unselected_frames():
    torch.manual_seed(31415)
    order = partial_frame_permutation(200, 0.25, torch.device("cpu"))
    assert sorted(order.tolist()) == list(range(200))
    assert 0 < int((order != torch.arange(200)).sum()) <= 50
    x = torch.arange(200).expand(3, 200)
    mask = x % 7 != 0
    labels = temporal_labels(200, 16, x.device)
    assert torch.equal(labels[order], x[:, order][0] * 16 // 200)
    assert torch.equal(mask[:, order], x[:, order] % 7 != 0)
    assert torch.equal(partial_frame_permutation(1, 0.25, x.device), torch.tensor([0]))


def test_contrastive_is_original_directional_sum_and_known_multi_positive():
    torch.manual_seed(1729)
    scores = torch.randn(3, 3, requires_grad=True)
    labels = torch.arange(3)
    expected = F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)
    torch.testing.assert_close(
        positive_contrastive(scores, torch.eye(3, dtype=torch.bool)), expected
    )
    mask = torch.tensor([[True, True, False], [False, False, True]])
    result = positive_contrastive(scores[:2], mask)
    assert torch.isfinite(result)
    result.backward()
    assert scores.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="known positive"):
        positive_contrastive(scores.detach(), torch.zeros(3, 3, dtype=torch.bool))


def test_all_k3_permutations_loss_score_and_input_gradient_equivariance():
    torch.manual_seed(1729)
    model = small().eval()
    capture = fixture_capture(grad=True)
    other = fixture_capture()
    text = torch.randn(3, 768)
    positive = torch.tensor([[True, True, False], [False, False, True]])
    torch.manual_seed(2718)
    expected = model.compute_loss((capture, other), text, positive)
    expected["loss"].backward()
    gradients = [window.features.grad.clone() for window in capture.windows]
    scores = model.score((capture, other), text).detach()
    for order in itertools.permutations(range(3)):
        permuted = permute(capture, order)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(2718)
        actual = model.compute_loss((permuted, other), text, positive)
        assert all(torch.equal(actual[key], expected[key]) for key in expected)
        assert torch.equal(model.score((permuted, other), text).detach(), scores)
        # Nonleaf permuted tensors retain their own derivative for the check.
        for window in permuted.windows:
            window.features.retain_grad()
        actual["loss"].backward()
        for window, gradient in zip(permuted.windows, gradients, strict=True):
            assert torch.equal(window.features.grad, gradient[list(order)])


def test_checkpoint_dropout_loss_rng_and_every_parameter_gradient_equal():
    torch.manual_seed(1729)
    direct = small(dropout=0.1, checkpoint=False).train()
    replay = small(dropout=0.1, checkpoint=True).train()
    replay.load_state_dict(direct.state_dict())
    captures = (fixture_capture(), fixture_capture())
    text = torch.randn(3, 768)
    positive = torch.tensor([[True, True, False], [False, False, True]])
    gradients, losses, rngs = [], [], []
    for model in (direct, replay):
        torch.manual_seed(31415)
        loss = model.compute_loss(captures, text, positive)
        loss["loss"].backward()
        losses.append(loss)
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters()})
        rngs.append(torch.get_rng_state())
    assert torch.equal(rngs[0], rngs[1])
    assert all(torch.equal(losses[0][key], losses[1][key]) for key in losses[0])
    for name in gradients[0]:
        assert torch.equal(gradients[0][name], gradients[1][name]), name


def test_default_complete_objective_all_trainable_parameter_gradients():
    torch.manual_seed(1729)
    model = WaMoSet().train()
    assert model.config == WaMoConfig()
    captures = (fixture_capture(length=32), fixture_capture(length=32))
    text = torch.randn(3, 768)
    positive = torch.tensor([[True, True, False], [False, False, True]])
    loss = model.compute_loss(captures, text, positive)
    assert set(loss) == {"loss", "contrastive", "reconstruction", "ordering"}
    loss["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0, name


def test_all_windows_absolute_gaps_dynamic_k_and_short_windows():
    torch.manual_seed(1729)
    model = small().eval()
    capture = fixture_capture(k=17, length=1)
    text = torch.randn(2, 768)
    score = model.score((capture,), text)
    assert score.shape == (1, 2) and score.isfinite().all()
    short = TMRGroupCapture((capture.windows[0],))
    gap = TMRGroupCapture((capture.windows[0], replace(capture.windows[1], start_seconds=300.0)))
    assert not torch.equal(model.score((short,), text), score)
    assert not torch.equal(model.score((gap,), text), score)
    late = capture.windows[1]
    changed = TMRGroupCapture((capture.windows[0], replace(late, features=late.features + 1)))
    assert not torch.equal(model.score((changed,), text), score)
    assert not any(isinstance(module, torch.nn.Embedding) for module in model.modules())


def test_observed_coordinate_and_frame_loss_normalization():
    torch.manual_seed(1729)
    model = small(checkpoint=False).eval()
    capture = fixture_capture(starts=(0.0,))
    window = capture.windows[0]
    x, observed = window.canonical(torch.device("cpu"))
    order = torch.arange(x.shape[1])
    _, rec, ce = model._window(x, observed, order)
    intra, inter = model.frame_features(x, observed)
    predictions = (
        model.wavelet.inverse(
            tuple(
                decoder(features)
                for decoder, features in zip(model.intra_decoders, intra, strict=True)
            )
        ),
        model.inter_decoder(inter),
    )
    expected_rec = sum(
        F.smooth_l1_loss(prediction[observed], x[observed], reduction="sum")
        for prediction in predictions
    )
    torch.testing.assert_close(rec.float(), expected_rec)
    frame_mask = observed.any(-1)
    labels = temporal_labels(x.shape[1], 16, x.device)[None].expand(len(x), -1)
    expected_ce = 2 * F.cross_entropy(
        model.temporal_classifier(inter)[frame_mask], labels[frame_mask], reduction="sum"
    )
    torch.testing.assert_close(ce.float(), expected_ce)
    # Independent nonidentity shuffled CE oracle exercises the actual branch,
    # not only the helper's mask/label indexing.
    order = torch.arange(x.shape[1]).roll(3)
    _, _, actual_ce = model._window(x, observed, order)
    _, shuffled = model.frame_features(x[:, order], observed[:, order])
    shuffled_mask = frame_mask[:, order]
    independent = F.cross_entropy(
        model.temporal_classifier(inter)[frame_mask], labels[frame_mask], reduction="sum"
    ) + F.cross_entropy(
        model.temporal_classifier(shuffled)[shuffled_mask],
        labels[:, order][shuffled_mask],
        reduction="sum",
    )
    torch.testing.assert_close(actual_ce.float(), independent)


def test_invalid_config_and_language_rejected():
    with pytest.raises(ValueError, match="expects"):
        LearnableSWT()(torch.zeros(2, 0, 66))
    for kwargs in ({"low_kernel": 3}, {"temperature": 0}, {"levels": 0}, {"width": 15}):
        with pytest.raises(ValueError):
            WaMoConfig(**kwargs)
    model = small()
    with pytest.raises(ValueError, match="frozen"):
        model.encode_text(torch.randn(2, 768, requires_grad=True))
