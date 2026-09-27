"""Analytic sampler/component checks, not original MIME reproduction."""

from dataclasses import replace
import hashlib
import math
import random

import pytest
import torch

from phaseset_core.continuous_parent_training import parent_epoch_batches
from phaseset_core.mime_parent_curriculum import (
    MimeParentAnchors,
    mime_curriculum_hardness,
    mime_parent_epoch_batches,
    mime_training_parent_anchors,
)
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from test_phaseset_frozen_clip_text import _fixture


def _key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _population(count=202, *, component="C01", split="train"):
    parents = tuple(
        ParentCaptionRecord(
            _key(f"source/{i}"),
            _key(f"family/{i}"),
            component,
            split,
            (f"human row {i}", f"second human row {i}"),
        )
        for i in range(count)
    )
    return ParentRetrievalTask(
        parents, expected_family_keys=tuple(p.annotation_family_sha256 for p in parents)
    )


def _anchors(task):
    # Nonuniform analytic angles avoid intentionally tied neighbor ranks in
    # the independent scalar-versus-matrix cosine ordering oracle.
    angles = torch.linspace(0, 1, len(task.parents)).pow(3) * math.pi
    embeddings = torch.zeros(len(task.parents), 512)
    embeddings[:, 0], embeddings[:, 1] = angles.cos(), angles.sin()
    return MimeParentAnchors(
        tuple(p.annotation_family_sha256 for p in task.parents), embeddings.contiguous()
    )


def test_paper_cosine_ramp_has_explicit_zero_based_epoch_convention():
    assert [mime_curriculum_hardness(i) for i in range(4)] == [0, 0, 0, 0]
    ramp = [mime_curriculum_hardness(i) for i in range(3, 13)]
    assert all(a < b for a, b in zip(ramp, ramp[1:]))
    assert ramp[-1] == mime_curriculum_hardness(25) == 0.25
    for invalid in (-1, True, 1.2):
        with pytest.raises(ValueError):
            mime_curriculum_hardness(invalid)


@pytest.mark.parametrize("seed", [1729, 2718, 31415])
def test_curriculum_never_repeats_or_drops_parent_or_human_rows(seed):
    task = _population()
    anchors = _anchors(task)
    for epoch in (0, 3, 7, 12):
        kwargs = dict(
            train_components=("C01",), batch_size=128, neighbor_window=128, seed=seed, epoch=epoch
        )
        rng_before = random.getstate()
        batches = mime_parent_epoch_batches(task, anchors, **kwargs)
        assert random.getstate() == rng_before
        assert batches == mime_parent_epoch_batches(task, anchors, **kwargs)
        keys = tuple(k for batch in batches for k in batch.motion_positive_keys)
        assert len(keys) == len(set(keys)) == 202
        assert set(keys) == set(anchors.family_keys)
        assert tuple(len(b.parents) for b in batches) == (128, 74)
        assert sum(len(b.captions) for b in batches) == 404
        if epoch < 3:
            assert batches == parent_epoch_batches(
                task, **{k: v for k, v in kwargs.items() if k != "neighbor_window"}
            )


def test_window_sampling_agrees_with_independent_paper_equation():
    task = _population(20)
    anchors = _anchors(task)
    seed, epoch, size, window = 1729, 12, 4, 5
    batches = mime_parent_epoch_batches(
        task,
        anchors,
        train_components=("C01",),
        batch_size=size,
        neighbor_window=window,
        seed=seed,
        epoch=epoch,
    )
    rng = random.Random((seed << 32) + epoch)
    anchor = rng.choice(list(range(20)))
    vectors = anchors.embeddings
    neighbors = [i for i in range(20) if i != anchor]
    # Independent scalar Euclidean cosine and Eq.22, not sampler's matrix.
    neighbors.sort(
        key=lambda i: (-sum(float(a) * float(b) for a, b in zip(vectors[anchor], vectors[i])), i)
    )
    center = math.floor((1 - 0.25) * 18)
    start = max(0, min(center - window // 2, 19 - window))
    selected = [anchor, *rng.sample(neighbors[start : start + window], size - 1)]
    assert batches[0].motion_positive_keys == tuple(anchors.family_keys[i] for i in selected)


def test_tail_rebalances_and_test_or_wrong_learning_anchors_are_rejected():
    task = _population(129, component="C00", split="validation")
    anchors = _anchors(task)
    kwargs = dict(
        train_components=("C00",), batch_size=128, neighbor_window=128, seed=1729, epoch=12
    )
    batches = mime_parent_epoch_batches(task, anchors, **kwargs)
    assert tuple(len(b.parents) for b in batches) == (127, 2)
    assert len({k for b in batches for k in b.motion_positive_keys}) == 129
    for bad_task, bad_anchors, bad_kwargs in (
        (_population(4, component="C00", split="test"), anchors, kwargs),
        (task, replace(anchors, family_keys=tuple(reversed(anchors.family_keys))), kwargs),
        (task, anchors, kwargs | dict(neighbor_window=126)),
        (task, anchors, kwargs | dict(train_components=("C01",))),
    ):
        with pytest.raises(ValueError):
            mime_parent_epoch_batches(bad_task, bad_anchors, **bad_kwargs)


def test_human_anchor_pool_uses_every_learning_row_not_validation_or_changed_text(tmp_path):
    task = _population(4)
    labels = task.batch(tuple(p.annotation_family_sha256 for p in task.parents))
    clip, _, _, _ = _fixture(tmp_path)
    text = clip.encode(labels.captions, labels.caption_commitments, batch_size=3)
    anchors = mime_training_parent_anchors(task, text, train_components=("C01",))
    assert anchors.family_keys == labels.motion_positive_keys
    assert anchors.embeddings.shape == (4, 512) and not anchors.embeddings.requires_grad
    assert torch.allclose(anchors.embeddings.norm(dim=1), torch.ones(4))
    wrong_text = clip.encode(
        ("changed human text", *labels.captions[1:]), labels.caption_commitments, batch_size=3
    )
    with pytest.raises(ValueError):
        mime_training_parent_anchors(task, wrong_text, train_components=("C01",))
    with pytest.raises(ValueError):
        mime_training_parent_anchors(task, text, train_components=("C00",))
