"""Analytic VJP oracles, never native optimizer/pilot/performance evidence."""

import copy
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.literature_parent_training import (
    backward_loaded_mime_parent_batch,
    backward_loaded_wamo_parent_batch,
)
from phaseset_core.mime_set import MIMERetrieval, MIMESetMotion
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.pipeline import PrivateCaptureArrays
from phaseset_core.tmr_set import TMRGroupCapture
from phaseset_core.training import _capture_rng, _restore_rng, _stable_hash
from test_mime_set import config, language
from test_wamo_set import small


def fixture():
    # Entire synthetic timelines, including a rejected middle interval. No
    # released siblings are presented as independent real data.
    rng = np.random.default_rng(1729)
    captures, records = {}, []
    for index, caption_count in enumerate((2, 1, 3)):
        k = 3
        x = rng.normal(0, 0.1, (900, k, 22, 3)).astype(np.float32)
        x += np.arange(k, dtype=np.float32)[None, :, None, None] * 0.2
        observed = np.ones(x.shape[:-1], dtype=np.bool_)
        observed[300:600] = False
        source = hashlib.sha256(f"analytic-source-{index}".encode()).hexdigest()
        family = hashlib.sha256(f"analytic-family-{index}".encode()).hexdigest()
        capture = prepare_continuous_capture(
            PrivateCaptureArrays(x, observed, tuple(bytes([i + 1]) * 32 for i in range(k)), source)
        )
        assert len(capture.windows()) == 2
        captures[source] = capture
        records.append(
            ParentCaptionRecord(
                source,
                family,
                "C01",
                "train",
                tuple(f"analytic human-like row {index} {row}" for row in range(caption_count)),
            )
        )
    task = ParentRetrievalTask(
        tuple(records), expected_family_keys=tuple(row.annotation_family_sha256 for row in records)
    )
    return captures, task.batch(tuple(row.annotation_family_sha256 for row in records))


@pytest.mark.parametrize("kind", ["wamo", "mime"])
def test_full_negative_replay_matches_original_objective_all_gradients_and_rng(kind):
    torch.manual_seed(1729)
    captures, labels = fixture()
    model = (
        small(dropout=0.1)
        if kind == "wamo"
        else MIMERetrieval(MIMESetMotion(config(dropout=0.1)), language())
    ).eval()
    dense = copy.deepcopy(model).train()
    physical = tuple(
        TMRGroupCapture.from_prepared(captures[key]) for key in labels.motion_source_keys
    )
    cls = torch.randn(len(labels.captions), 768)
    positive = labels.positive_mask(device=torch.device("cpu"))
    state = _capture_rng()
    if kind == "wamo":
        expected = dense.compute_loss(physical, cls, positive)
    else:
        expected = {
            "loss": dense.compute_loss(physical, dense.language.tokenize(labels.captions), positive)
        }
    expected["loss"].backward()
    after = _stable_hash(_capture_rng())
    _restore_rng(state)
    loaded, text_loaded = [], []

    def load(source):
        loaded.append(source)
        return captures[source]

    def load_cls(captions):
        text_loaded.append(captions)
        return cls

    args = dict(learning_components=("C01",), load_capture=load)
    result = (
        backward_loaded_wamo_parent_batch(model, labels, load_cls=load_cls, **args)
        if kind == "wamo"
        else backward_loaded_mime_parent_batch(model, labels, **args)
    )
    assert _stable_hash(_capture_rng()) == after and not model.training
    assert loaded == list(labels.motion_source_keys) * 2
    assert text_loaded == ([labels.captions] if kind == "wamo" else [])
    assert result.scores.shape == (3, 6) and result.parent_count == result.replayed_parents == 3
    assert result.text_count == 6
    for name, value in expected.items():
        assert result.loss_components[name] == float(value.detach())
    for actual, oracle in zip(model.parameters(), dense.parameters(), strict=True):
        assert actual.grad is not None and oracle.grad is not None
        # Dense autograd sums shared capture gradients in reverse traversal;
        # replay visits fixed source order. Only this FP32 accumulation differs.
        torch.testing.assert_close(actual.grad, oracle.grad, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("kind", ["wamo", "mime"])
def test_wrong_population_and_source_are_rejected_without_silent_substitution(kind):
    captures, labels = fixture()
    model = small() if kind == "wamo" else MIMERetrieval(MIMESetMotion(config()), language())
    function = (
        backward_loaded_wamo_parent_batch if kind == "wamo" else backward_loaded_mime_parent_batch
    )
    text = {"load_cls": lambda captions: torch.randn(len(captions), 768)} if kind == "wamo" else {}
    calls = []

    def load(source):
        calls.append(source)
        return replace(captures[source], source_sha256="b" * 64)

    with pytest.raises(ValueError, match="learning parents"):
        function(model, labels, learning_components=("C02",), load_capture=load, **text)
    assert not calls
    with pytest.raises(ValueError, match="complete prepared source"):
        function(model, labels, learning_components=("C01",), load_capture=load, **text)


def test_replay_detects_physical_drift_and_restores_model_mode():
    captures, labels = fixture()
    model = small().eval()
    seen = set()

    def load(source):
        capture = captures[source]
        if source in seen:
            changed = capture.skeletons.copy()
            changed[capture.track_mask] += np.float32(0.25)
            return replace(capture, skeletons=changed)
        seen.add(source)
        return capture

    with pytest.raises(RuntimeError, match="replay changed"):
        backward_loaded_wamo_parent_batch(
            model,
            labels,
            learning_components=("C01",),
            load_capture=load,
            load_cls=lambda captions: torch.ones(len(captions), 768),
        )
    assert not model.training
