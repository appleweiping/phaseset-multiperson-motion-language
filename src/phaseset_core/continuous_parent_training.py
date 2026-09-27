"""Parent-unit sampling and memory-bounded complete-capture loss backward.

This is the numerical seam for the private training host, not its scientific
freeze, budget ledger, scheduler or checkpoint selector. It uses all human
rows once per admitted parent, preserves a full effective-batch negative
gallery, and replays one complete capture graph at a time. No optimizer is
started here. Human verification of counterfactuals remains external.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import torch
from torch import Tensor

from .capture_validation import _validate_text_batch
from .continuous_retrieval import ContinuousRetrievalSystem
from .continuous_training_input import ContinuousTrainingView
from .frozen_clip_text import FrozenClipTextBatch
from .parent_retrieval_task import ParentRetrievalBatch, ParentRetrievalTask
from .temporal_coordination import coordination_objective
from .training import OFFICIAL_SEEDS, _capture_rng, _restore_rng


def parent_epoch_batches(
    task: ParentRetrievalTask,
    *,
    train_components: tuple[str, ...],
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[ParentRetrievalBatch, ...]:
    """Visit every parent of the explicit learning components exactly once.

    Components, not the historical main-split name, select the population:
    an outer fold can train on C00 while keeping its held-out groups separate.
    The private host freezes those components from the dated study matrix.
    This does not open test data or choose epoch/step counts. Batch size is in
    parents, not windows, and still awaits the common bounded-pilot freeze.
    """
    if type(seed) is not int or seed not in OFFICIAL_SEEDS:
        raise ValueError("parent epochs require one of the three fixed seeds")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("contrastive parent batch size must be at least two")
    if (
        type(train_components) is not tuple
        or not train_components
        or len(set(train_components)) != len(train_components)
        or any(type(component) is not str or not component for component in train_components)
    ):
        raise ValueError("the learning components must be explicit and unique")
    selected = tuple(parent for parent in task.parents if parent.component in train_components)
    if {parent.component for parent in selected} != set(train_components):
        raise ValueError("the admitted task is missing a requested learning component")
    if any(parent.split == "test" for parent in selected):
        raise ValueError("final-test parents cannot enter a training epoch")
    if len(selected) < 2:
        raise ValueError("contrastive learning needs two distinct admitted parents")
    keys = [parent.annotation_family_sha256 for parent in selected]
    random.Random((seed << 32) + epoch).shuffle(keys)
    batches = [keys[start : start + batch_size] for start in range(0, len(keys), batch_size)]
    # Keep the full census, rather than dropping/repeating a singleton. Move
    # one parent from the previous batch only if both still have negatives.
    if len(batches[-1]) == 1:
        if len(batches[-2]) < 3:
            raise ValueError("batch ceiling cannot cover this census without a singleton")
        batches[-1].insert(0, batches[-2].pop())
    return tuple(task.batch(tuple(batch)) for batch in batches)


@dataclass(frozen=True)
class ParentCounterfactualRows:
    """Explicit score columns and caller-admitted human-false masks.

    Extra positive/negative CF sentences may follow the human retrieval rows
    in the same CLIP pool. They never receive fabricated retrieval positives.
    The private host supplies blind two-human provenance; these indices or a
    True bit are not proof of that provenance. Unverified pairs are excluded.
    """

    source_keys: tuple[str, ...]
    positive_columns: tuple[int, ...]
    negative_columns: tuple[int, ...]
    verified_false: tuple[bool, ...]

    def indices(
        self, labels: ParentRetrievalBatch, *, text_count: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        count = len(self.source_keys)
        if any(
            type(value) is not tuple or len(value) != count
            for value in (
                self.source_keys,
                self.positive_columns,
                self.negative_columns,
                self.verified_false,
            )
        ):
            raise ValueError("CF source, column and verified tuples must align")
        by_source = {key: index for index, key in enumerate(labels.motion_source_keys)}
        for source, positive, negative, verified in zip(
            self.source_keys,
            self.positive_columns,
            self.negative_columns,
            self.verified_false,
            strict=True,
        ):
            if source not in by_source or type(verified) is not bool:
                raise ValueError("CF rows need an admitted source and explicit bool verification")
            if (
                any(
                    type(column) is not int or not 0 <= column < text_count
                    for column in (positive, negative)
                )
                or positive == negative
            ):
                raise ValueError("CF positive/negative columns must be distinct and in the pool")
            if (
                verified
                and negative < len(labels.text_positive_keys)
                and (
                    labels.text_positive_keys[negative]
                    == labels.motion_positive_keys[by_source[source]]
                )
            ):
                raise ValueError("a retrieval positive cannot also be verified-false CF text")
        return (
            torch.tensor(
                [by_source[key] for key in self.source_keys], dtype=torch.int64, device=device
            ),
            torch.tensor(self.positive_columns, dtype=torch.int64, device=device),
            torch.tensor(self.negative_columns, dtype=torch.int64, device=device),
            torch.tensor(self.verified_false, dtype=torch.bool, device=device),
        )


@dataclass(frozen=True)
class ParentBackwardResult:
    loss: float
    scores: Tensor
    parent_count: int
    retrieval_caption_count: int
    verified_counterfactual_count: int
    replayed_captures: int


def backward_parent_batch(
    system: ContinuousRetrievalSystem,
    views: tuple[ContinuousTrainingView, ...],
    text_batch: FrozenClipTextBatch,
    labels: ParentRetrievalBatch,
    *,
    counterfactuals: ParentCounterfactualRows,
    cf_weight: float,
    margin: float,
) -> ParentBackwardResult:
    """Accumulate the exact full-gallery objective, with one capture graph live.

    The no-grad pass stores only B-by-Q scores and RNG snapshots; a leaf-score
    objective supplies the VJP for each complete parent. The replay includes
    calibration and the shared text adapter, so their gradients are not lost.
    The existing per-capture time/edge checkpointing remains in use. Caller
    clips/steps/checkpoints only after this succeeds. FP32 qualification only;
    native BF16 must be qualified and integrated separately before use.
    """
    if not isinstance(system, ContinuousRetrievalSystem):
        raise TypeError("complete parent training requires the V2 retrieval system")
    if not views or len(views) != len(labels.parents):
        raise ValueError("one complete input view is required per parent label")
    if tuple(view.positive_capture_key for view in views) != labels.motion_source_keys:
        raise ValueError("complete physical source order differs from parent label order")
    if len(set(labels.motion_source_keys)) != len(views) or len(
        set(labels.motion_positive_keys)
    ) != len(views):
        raise ValueError("released siblings cannot enter a parent batch twice")
    text, commitments, _ = _validate_text_batch(text_batch)
    retrieval_count = len(labels.captions)
    if commitments[:retrieval_count] != labels.caption_commitments:
        raise ValueError("CLIP pool must start with every admitted human row in label order")
    for caption, row in zip(labels.captions, text_batch.receipt.caption_rows, strict=False):
        if hashlib.sha256(caption.encode("utf-8")).hexdigest() != row[2]:
            raise ValueError("CLIP human text differs from the official parent caption")
    device = next(system.parameters()).device
    indices = counterfactuals.indices(labels, text_count=len(text), device=device)
    positive_mask = labels.positive_mask(device=device)
    was_training = system.training
    system.train()
    system.zero_grad(set_to_none=True)
    cached, states = [], []
    try:
        for view in views:
            states.append(_capture_rng())
            with torch.no_grad():
                cached.append(system.score((view,), text_batch).scores.detach())
        after = _capture_rng()
        scores = torch.cat(cached).detach().requires_grad_(True)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("complete parent score cache is nonfinite")
        motion, pos, neg, verified = indices
        loss = coordination_objective(
            scores[:, :retrieval_count].contiguous(),
            positive_mask,
            cf_positive_scores=scores[motion, pos],
            cf_negative_scores=scores[motion, neg],
            verified_negative_mask=verified,
            cf_weight=cf_weight,
            margin=margin,
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("complete parent objective is nonfinite")
        loss.backward()
        if scores.grad is None:
            raise RuntimeError("parent score cache has no objective gradient")
        try:
            for index, (view, state) in enumerate(zip(views, states, strict=True)):
                _restore_rng(state)
                replay = system.score((view,), text_batch)
                if not torch.equal(replay.scores.detach(), cached[index]):
                    raise RuntimeError("complete parent replay changed its score row")
                torch.autograd.backward(replay.scores, scores.grad[index : index + 1])
                del replay
        finally:
            _restore_rng(after)
        if any(parameter.grad is not None for parameter in system.frozen_b2.parameters()):
            raise RuntimeError("the shared B2 anchor must remain frozen")
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in system.parameters()
        ):
            raise ValueError("complete parent gradients are nonfinite")
        return ParentBackwardResult(
            float(loss.detach().cpu()),
            scores.detach().cpu(),
            len(views),
            retrieval_count,
            int(verified.sum()),
            len(views),
        )
    finally:
        system.train(was_training)
