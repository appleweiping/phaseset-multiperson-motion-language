"""Data-free implementation checks; no dataset/replication/pilot authority."""

import pytest
import torch
from torch.nn import functional as F

from phaseset_core.tmr_components import (
    ACTORStyleDecoder,
    ACTORStyleEncoder,
    PositionalEncoding,
    TMRLossWeights,
    TMRSinglePerson,
    diagonal_kl,
    filtered_contrastive_loss,
    reconstruction_loss,
    sample_diagonal_gaussian,
    semantic_negative_mask,
    tmr_auxiliary_losses,
)


def test_default_actual_yaml_architecture_and_all_gradients():
    torch.manual_seed(1729)
    model = TMRSinglePerson(12)
    assert model.text_encoder.projection.in_features == 768
    for encoder in (model.text_encoder, model.motion_encoder):
        assert encoder.tokens.shape == (2, 256)
        assert len(encoder.seqTransEncoder.layers) == 6
        assert encoder.seqTransEncoder.layers[0].self_attn.num_heads == 4
        assert encoder.seqTransEncoder.layers[0].linear1.out_features == 1024
        assert encoder.seqTransEncoder.layers[0].dropout.p == 0.1
    assert len(model.motion_decoder.seqTransDecoder.layers) == 6
    motion = torch.randn(2, 7, 12)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, -2:] = False
    motion[~mask] = 0
    text = torch.randn(2, 5, 768)
    text_mask = torch.ones(2, 5, dtype=torch.bool)
    sentence = torch.eye(2)
    losses = model.compute_loss(motion, mask, text, text_mask, sentence)
    assert set(losses) == {"recons", "kl", "latent", "contrastive", "loss"}
    assert all(torch.isfinite(value) and value > 0 for value in losses.values())
    losses["loss"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.motion_encoder.projection.weight.grad.abs().sum() > 0
    assert model.text_encoder.projection.weight.grad.abs().sum() > 0
    assert model.motion_decoder.final_layer.weight.grad.abs().sum() > 0


def test_four_kls_two_reconstructions_and_weights_match_independent_formula():
    torch.manual_seed(2718)
    mu_m, mu_t = torch.randn(3, 8), torch.randn(3, 8)
    lv_m, lv_t = torch.randn(3, 8) * 0.2, torch.randn(3, 8) * 0.2
    z_m, z_t = torch.randn(3, 8), torch.randn(3, 8)
    target, rec_m, rec_t = (torch.randn(3, 9, 4) for _ in range(3))
    got = tmr_auxiliary_losses(
        text_reconstruction=rec_t,
        motion_reconstruction=rec_m,
        target=target,
        text_latent=z_t,
        motion_latent=z_m,
        text_distribution=(mu_t, lv_t),
        motion_distribution=(mu_m, lv_m),
    )
    qm = torch.distributions.Normal(mu_m, (lv_m / 2).exp())
    qt = torch.distributions.Normal(mu_t, (lv_t / 2).exp())
    unit = torch.distributions.Normal(torch.zeros_like(mu_m), torch.ones_like(mu_m))
    expected_kl = sum(
        torch.distributions.kl_divergence(q, p).mean()
        for q, p in ((qt, qm), (qm, qt), (qm, unit), (qt, unit))
    )
    torch.testing.assert_close(got["kl"], expected_kl, rtol=1e-6, atol=1e-6)
    assert got["recons"] == F.smooth_l1_loss(rec_t, target) + F.smooth_l1_loss(rec_m, target)
    assert got["latent"] == F.smooth_l1_loss(z_t, z_m)
    got["contrastive"] = torch.tensor(3.0)
    expected = got["recons"] + 1e-5 * got["kl"] + 1e-5 * got["latent"] + 0.1 * got["contrastive"]
    assert TMRLossWeights().total(got) == expected
    assert diagonal_kl((mu_m, lv_m), (mu_m, lv_m)) == 0


def test_diagonal_filter_is_original_symmetric_cross_entropy_and_gradients():
    torch.manual_seed(31415)
    motion = torch.randn(3, 8, requires_grad=True)
    text = torch.randn(3, 8, requires_grad=True)
    # Off-diagonal cosine .7 is above the actual .6, but below misread .8.
    sentences = torch.tensor([[1.0, 0, 0], [0.7, (1 - 0.7**2) ** 0.5, 0], [0, 0, 1.0]])
    positive = torch.eye(3, dtype=torch.bool)
    filtered = semantic_negative_mask(positive, sentences, 0.8)
    assert filtered[0, 1] and filtered[1, 0] and not filtered.diagonal().any()
    logits = F.normalize(motion, dim=1) @ F.normalize(text, dim=1).T / 0.1
    logits = logits.masked_fill(filtered, -torch.inf)
    expected = (
        F.cross_entropy(logits, torch.arange(3)) + F.cross_entropy(logits.T, torch.arange(3))
    ) / 2
    got = filtered_contrastive_loss(motion, text, positive, sentence_embeddings=sentences)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    actual_grad = torch.autograd.grad(got, (motion, text), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (motion, text))
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference, rtol=2e-6, atol=1e-6)


def test_rectangular_variable_positives_are_preserved_and_filter_is_negative_only():
    motion = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    text = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[True, True, False], [False, False, True]])
    sentences = F.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]), dim=1
    ).requires_grad_()
    filtered = semantic_negative_mask(positive, sentences, 0.8)
    assert not (filtered & positive).any()
    assert filtered[0, 2] and filtered[1, 0] and filtered[1, 1]
    got = filtered_contrastive_loss(motion, text, positive, sentence_embeddings=sentences)
    assert got == 0 and torch.isfinite(got)
    got.backward()
    assert torch.isfinite(motion.grad).all() and torch.isfinite(text.grad).all()
    assert sentences.grad is None
    unfiltered = filtered_contrastive_loss(motion.detach(), text.detach(), positive)
    logits = F.normalize(motion.detach(), dim=1) @ F.normalize(text.detach(), dim=1).T / 0.1
    expected = 0.5 * (
        (
            torch.logsumexp(logits, 1)
            - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), 1)
        ).mean()
        + (
            torch.logsumexp(logits, 0)
            - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), 0)
        ).mean()
    )
    assert unfiltered == expected and unfiltered > 0


def test_gap_mask_excludes_unknown_targets_and_their_gradients():
    prediction = torch.tensor([[[2.0, 999.0], [3.0, -999.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, float("nan")], [1.0, float("nan")]]])
    observed = torch.tensor([[[True, False], [True, False]]])
    got = reconstruction_loss(prediction, target, observed)
    assert got == 1.0
    got.backward()
    assert torch.equal(prediction.grad[~observed], torch.zeros(2))
    with pytest.raises(ValueError, match="without observed"):
        reconstruction_loss(prediction, target, torch.zeros_like(observed))


def test_extended_positions_keep_original_prefix_and_do_not_truncate():
    short = PositionalEncoding(256, dropout=0)
    extended = PositionalEncoding(256, dropout=0, max_len=6002)
    assert torch.equal(short.pe, extended.pe[:5000])
    assert extended(torch.zeros(1, 6002, 256)).shape == (1, 6002, 256)
    with pytest.raises(ValueError, match="never truncate"):
        short(torch.zeros(1, 5001, 256))
    assert "pe" not in extended.state_dict()


def test_decoder_padding_and_sample_mean_are_exact():
    torch.manual_seed(1729)
    decoder = ACTORStyleDecoder(4, latent_dim=16, ff_size=32, num_layers=1, dropout=0)
    encoder = ACTORStyleEncoder(4, latent_dim=16, ff_size=32, num_layers=1, dropout=0)
    mask = torch.tensor([[True, True, False]])
    encoded = encoder(torch.randn(1, 3, 4), mask)
    dist = (encoded[:, 0], encoded[:, 1])
    state = torch.get_rng_state()
    z = sample_diagonal_gaussian(dist, sample_mean=True)
    assert z is dist[0] and torch.equal(state, torch.get_rng_state())
    decoded = decoder(z, mask)
    assert torch.equal(decoded[~mask], torch.zeros(1, 4))
    assert not torch.signbit(decoded[~mask]).any()
    with pytest.raises(ValueError, match="valid position"):
        decoder(z, torch.zeros_like(mask))


@pytest.mark.parametrize("failure", ["empty_positive", "unnormalized_sentence", "temperature"])
def test_invalid_loss_inputs_are_not_silently_changed(failure):
    x = torch.eye(2)
    positive = torch.eye(2, dtype=torch.bool)
    kwargs = {}
    if failure == "empty_positive":
        positive[0] = False
    elif failure == "unnormalized_sentence":
        kwargs["sentence_embeddings"] = x * 2
    else:
        kwargs["temperature"] = 0
    with pytest.raises(ValueError):
        filtered_contrastive_loss(x, x, positive, **kwargs)
