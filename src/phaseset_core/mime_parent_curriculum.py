"""MIME paper-defined training-only semantic parent curriculum.

One sampler component of the explicitly adapted MIME-Set, not the original
MIME architecture, its original task replication or a qualified baseline.
Anchors use the mean of all admitted frozen CLIP human rows per complete
parent. The private study host freezes this adaptation and the explicit
neighbor-window width during its bounded pilot; the paper gives no width.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

import torch
from torch import Tensor
from torch.nn import functional as F

from .capture_validation import _validate_text_batch
from .continuous_parent_training import parent_epoch_batches
from .frozen_clip_text import FrozenClipTextBatch
from .parent_retrieval_task import ParentRetrievalBatch, ParentRetrievalTask
from .training import OFFICIAL_SEEDS


@dataclass(frozen=True)
class MimeParentAnchors:
    family_keys: tuple[str, ...]
    embeddings: Tensor


def _learning_batch(
    task: ParentRetrievalTask,
    components: tuple[str, ...],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> ParentRetrievalBatch:
    # Reuse the qualified sampler's full-population/fold/test admission, not
    # historical main train/validation labels as an outer-fold fallback.
    parent_epoch_batches(
        task, train_components=components, batch_size=batch_size, seed=seed, epoch=epoch
    )
    return task.batch(
        tuple(p.annotation_family_sha256 for p in task.parents if p.component in components)
    )


@torch.no_grad()
def mime_training_parent_anchors(
    task: ParentRetrievalTask,
    text_batch: FrozenClipTextBatch,
    *,
    train_components: tuple[str, ...],
) -> MimeParentAnchors:
    """Only this run's learning components and their complete human rows.

    No validation/held-out/test text or CF sentences enter the neighbor graph.
    Original MIME text finetuning is not claimed by this frozen-CLIP adaptation.
    Private rights and official-file provenance still precede this interface.
    """
    labels = _learning_batch(
        task, train_components, batch_size=len(task.parents), seed=1729, epoch=0
    )
    # A real parent host selects its exact training subset from the closed
    # complete-development CLIP batch. Accept that authenticated selection,
    # then require all and only learning-human commitments and original bytes.
    # Extra weak/validation columns cannot pass this exact equality.
    text, commitments, _ = _validate_text_batch(text_batch, allow_row_selection=True)
    if commitments != labels.caption_commitments:
        raise ValueError("MIME anchors need all and only learning-parent human rows in order")
    for caption, row in zip(labels.captions, text_batch.receipt.caption_rows, strict=True):
        if hashlib.sha256(caption.encode("utf-8")).hexdigest() != row[2]:
            raise ValueError("MIME anchor text differs from the admitted official human row")
    rows, start = [], 0
    for parent in labels.parents:
        stop = start + len(parent.captions)
        mean = text[start:stop].to(device="cpu", dtype=torch.float64).mean(dim=0).float()
        if not bool(torch.isfinite(mean).all()) or float(mean.norm()) == 0:
            raise ValueError("MIME parent anchor must have a finite nonzero human-row mean")
        rows.append(F.normalize(mean, dim=0))
        start = stop
    return MimeParentAnchors(labels.motion_positive_keys, torch.stack(rows).contiguous())


def mime_curriculum_hardness(epoch: int) -> float:
    """Paper v1: three uniform epochs, ten-epoch cosine ramp, maximum 0.25.

    Epochs are zero based: 0..2 uniform, 3 starts ramp at zero, 12 reaches
    maximum. This explicit convention is part of the disclosed adaptation.
    """
    if type(epoch) is not int or epoch < 0:
        raise ValueError("MIME curriculum epoch must be a nonnegative integer")
    if epoch < 3:
        return 0.0
    progress = min((epoch - 3) / 9, 1.0)
    return 0.25 * (1 - math.cos(math.pi * progress)) / 2


def mime_parent_epoch_batches(
    task: ParentRetrievalTask,
    anchors: MimeParentAnchors,
    *,
    train_components: tuple[str, ...],
    batch_size: int,
    neighbor_window: int,
    seed: int,
    epoch: int,
) -> tuple[ParentRetrievalBatch, ...]:
    """No-replacement anchor/window batches; exactly one visit per parent.

    Eligible neighbors are sorted by decreasing cosine. The window center is
    floor((1-hardness)*(N_i-1)), moving from far to closer neighbors as in MIME
    Eq.22. Ties use canonical family order, never actor ordinals. Window width
    is explicit, not represented as an undocumented author default. With a
    small final pool it shrinks; a singleton tail is rebalanced without losing
    or repeating observations. Returns all human positives of each parent.
    """
    if type(seed) is not int or seed not in OFFICIAL_SEEDS:
        raise ValueError("MIME sampler requires a fixed study seed")
    labels = _learning_batch(task, train_components, batch_size=batch_size, seed=seed, epoch=epoch)
    if type(neighbor_window) is not int or neighbor_window < batch_size - 1:
        raise ValueError("explicit MIME neighbor window must cover the batch's other parents")
    values = anchors.embeddings
    if (
        type(anchors) is not MimeParentAnchors
        or anchors.family_keys != labels.motion_positive_keys
        or type(values) is not Tensor
        or values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape != (len(labels.parents), 512)
        or not values.is_contiguous()
        or values.requires_grad
        or not bool(torch.isfinite(values).all())
        or bool((values.norm(dim=1) == 0).any())
    ):
        raise ValueError("MIME anchors must cover the exact learning population as frozen CPU rows")
    if epoch < 3:
        return parent_epoch_batches(
            task, train_components=train_components, batch_size=batch_size, seed=seed, epoch=epoch
        )
    similarity = F.normalize(values, dim=1) @ F.normalize(values, dim=1).T
    rng = random.Random((seed << 32) + epoch)
    remaining = list(range(len(labels.parents)))
    batches = []
    hardness = mime_curriculum_hardness(epoch)
    while remaining:
        anchor = rng.choice(remaining)
        remaining.remove(anchor)
        take = min(batch_size - 1, len(remaining))
        if take:
            ranked = sorted(remaining, key=lambda i: (-float(similarity[anchor, i]), i))
            width = min(neighbor_window, len(ranked))
            center = math.floor((1 - hardness) * (len(ranked) - 1))
            start = max(0, min(center - width // 2, len(ranked) - width))
            selected = rng.sample(ranked[start : start + width], take)
            for index in selected:
                remaining.remove(index)
        else:
            selected = []
        batches.append([anchor, *selected])
    if len(batches[-1]) == 1:
        # The qualified common sampler already rejected impossible ceilings.
        batches[-1].insert(0, batches[-2].pop())
    return tuple(
        task.batch(tuple(labels.motion_positive_keys[i] for i in batch)) for batch in batches
    )
