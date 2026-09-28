"""Analytic dense-VAE replay oracles, not native convergence/optimizer evidence."""

import copy
from dataclasses import replace

import pytest
import torch

from phaseset_core.tmr_parent_training import backward_loaded_tmr_parent_batch
from phaseset_core.tmr_set import TMRGroupCapture, TMRSet, TMRTextBatch
from phaseset_core.training import _capture_rng, _restore_rng, _stable_hash
from test_literature_parent_training import fixture


def model():
    return TMRSet(latent_dim=16, ff_size=32, num_layers=1, num_heads=4, dropout=0.1)


def text_batch(labels):
    rows = len(labels.captions)
    tokens = torch.randn(rows, 5, 768)
    # Enable a real off-diagonal semantic filter for two different parents;
    # similarity never changes the registered family-derived positives.
    sentences = torch.eye(rows, 768)
    sentences[2] = sentences[0]
    return TMRTextBatch(tokens, torch.ones(rows, 5, dtype=torch.bool), sentences)


def test_all_vae_terms_dense_gradients_rng_and_full_parent_replay():
    torch.manual_seed(1729)
    captures, labels = fixture()
    actual = model().eval()
    dense = copy.deepcopy(actual).train()
    text = text_batch(labels)
    physical = tuple(
        TMRGroupCapture.from_prepared(captures[key]) for key in labels.motion_source_keys
    )
    state = _capture_rng()
    expected = dense.compute_loss(physical, text, labels.positive_mask(device=torch.device("cpu")))
    expected["loss"].backward()
    after = _stable_hash(_capture_rng())
    _restore_rng(state)
    loaded, texts, events = [], [], []

    def load(source):
        loaded.append(source)
        return captures[source]

    def load_text(captions):
        texts.append(captions)
        return text

    result = backward_loaded_tmr_parent_batch(
        actual,
        labels,
        learning_components=("C01",),
        load_capture=load,
        load_text=load_text,
        progress=lambda phase, index: events.append((phase, index)),
    )
    assert not actual.training and _stable_hash(_capture_rng()) == after
    assert texts == [labels.captions]
    assert loaded == list(labels.motion_source_keys) * 4
    assert events == [
        (phase, index)
        for phase in ("encoder-cache", "decoder-cache", "decoder-replay", "encoder-replay")
        for index in range(3)
    ]
    assert result.scores.shape == (3, 6) and result.parent_count == result.replayed_parents == 3
    assert result.text_count == 6
    assert set(result.loss_components) == {"loss", "recons", "kl", "latent", "contrastive"}
    for name, loss in expected.items():
        assert result.loss_components[name] == float(loss.detach())
    for (name, parameter), (_, reference) in zip(
        actual.named_parameters(), dense.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and reference.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=3e-5, atol=3e-6, msg=name)


def test_wrong_population_and_wrong_physical_source_rejected():
    captures, labels = fixture()
    actual, text = model().eval(), text_batch(labels)
    with pytest.raises(ValueError, match="learning parents"):
        backward_loaded_tmr_parent_batch(
            actual,
            labels,
            learning_components=("C02",),
            load_capture=lambda key: captures[key],
            load_text=lambda rows: text,
        )
    with pytest.raises(ValueError, match="complete prepared source"):
        backward_loaded_tmr_parent_batch(
            actual,
            labels,
            learning_components=("C01",),
            load_capture=lambda key: replace(captures[key], source_sha256="b" * 64),
            load_text=lambda rows: text,
        )
    assert not actual.training


def test_decoder_cache_replay_drift_fails_and_restores_eval_mode():
    captures, labels = fixture()
    actual, text = model().eval(), text_batch(labels)
    seen = {}

    def load(key):
        seen[key] = seen.get(key, 0) + 1
        capture = captures[key]
        if seen[key] == 3:  # encoder cache, decoder cache, then decoder replay
            x = capture.skeletons.copy()
            x[capture.track_mask] += 0.25
            return replace(capture, skeletons=x)
        return capture

    with pytest.raises(RuntimeError, match="reconstruction replay changed"):
        backward_loaded_tmr_parent_batch(
            actual,
            labels,
            learning_components=("C01",),
            load_capture=load,
            load_text=lambda rows: text,
        )
    assert not actual.training
