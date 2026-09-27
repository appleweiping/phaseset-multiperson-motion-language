"""Exact human CLIP rows for a caller-admitted complete development task."""

from __future__ import annotations

import hashlib

from .capture_validation import _validate_text_batch
from .continuous_parent_training import ParentCounterfactualRows, validate_parent_text_batch
from .frozen_clip_text import FrozenClipTextBatch, select_frozen_clip_text_rows
from .parent_retrieval_task import ParentRetrievalBatch, ParentRetrievalTask


class ParentHumanClipRows:
    """Keep one authenticated preparation batch; select without a CLIP tower.

    Every official human occurrence retains its own source/family/ordinal
    commitment. Equal sentence strings are not deduplicated. No motion or
    learning statistics enter this cache. The complete task must contain only
    development parents. No test rows or arbitrary additional text are loaded.
    Returned empty CF rows mean *no CF evidence*, not verified negative truth.
    This serves base training and human-only evaluation; full CF training still
    requires the actual blind two-human records and a separate admitted pool.
    """

    def __init__(self, task: ParentRetrievalTask, original: FrozenClipTextBatch):
        if type(task) is not ParentRetrievalTask or any(
            parent.split == "test" for parent in task.parents
        ):
            raise ValueError("human CLIP rows require the complete admitted development task")
        all_labels = task.batch(tuple(parent.annotation_family_sha256 for parent in task.parents))
        _validate_text_batch(original)  # Original encoded/rehydrated schema only.
        if len(original.caption_commitments) != len(all_labels.captions):
            raise ValueError("original CLIP batch must cover every human occurrence exactly once")
        validate_parent_text_batch(all_labels, original)
        self.original = original
        self.parents = {parent.annotation_family_sha256: parent for parent in task.parents}
        self.indices = {
            commitment: index for index, commitment in enumerate(original.caption_commitments)
        }

    def text(self, labels: ParentRetrievalBatch, *, training: bool):
        if (
            type(training) is not bool
            or type(labels) is not ParentRetrievalBatch
            or not labels.parents
        ):
            raise ValueError("human text selection requires explicit labels and training bool")
        families = tuple(parent.annotation_family_sha256 for parent in labels.parents)
        if len(set(families)) != len(families) or any(
            self.parents.get(parent.annotation_family_sha256) != parent for parent in labels.parents
        ):
            raise ValueError("human text labels differ from the admitted development population")
        rows = select_frozen_clip_text_rows(
            self.original,
            tuple(self.indices[commitment] for commitment in labels.caption_commitments),
        )
        validate_parent_text_batch(labels, rows)
        if any(
            hashlib.sha256(caption.encode("utf-8")).hexdigest() != row[2]
            for caption, row in zip(labels.captions, rows.receipt.caption_rows, strict=True)
        ):
            raise ValueError("selected human text changed its official bytes")
        return rows, ParentCounterfactualRows((), (), (), ())
