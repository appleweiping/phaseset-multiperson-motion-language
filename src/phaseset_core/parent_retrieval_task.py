"""One complete native parent per retrieval row, with every human holistic row.

This is a provenance/label index, not permission, physical-floor admission or a
task seal. The private host supplies complete-parent source identities and an
explicit expected population. No data discovery, test opening, semantic label
generation, motion encoding or caption-count assumption takes place here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .continuous_retrieval import capture_family_positive_mask, human_holistic_captions


def _sha_key(value: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("parent/source/family identity must be a lowercase SHA-256 key")


@dataclass(frozen=True)
class ParentCaptionRecord:
    source_sha256: str
    annotation_family_sha256: str
    component: str
    split: str
    captions: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _sha_key(self.source_sha256)
        _sha_key(self.annotation_family_sha256)
        if type(self.component) is not str or not self.component:
            raise ValueError("parent component must be explicit")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("parent split must be explicit")
        if (
            type(self.captions) is not tuple
            or not self.captions
            or any(type(row) is not str or not row.strip() for row in self.captions)
        ):
            raise ValueError("every official human holistic row must be nonempty")

    @classmethod
    def from_official_bytes(
        cls,
        *,
        source_sha256: str,
        annotation_family_sha256: str,
        component: str,
        split: str,
        annotation_bytes: bytes,
        expected_human_rows: int,
    ) -> ParentCaptionRecord:
        """Bind the native file and all scene_explained rows, without fusion.

        The private host verifies official provenance and rights. A digest does
        not make an arbitrary file official or verify skeleton-observable truth.
        """
        if type(annotation_bytes) is not bytes or (
            hashlib.sha256(annotation_bytes).hexdigest() != annotation_family_sha256
        ):
            raise ValueError("official annotation bytes disagree with admitted family")
        captions = human_holistic_captions(json.loads(annotation_bytes))
        if type(expected_human_rows) is not int or expected_human_rows != len(captions):
            raise ValueError("official human row count disagrees with admitted census")
        return cls(source_sha256, annotation_family_sha256, component, split, captions)


@dataclass(frozen=True)
class ParentRetrievalBatch:
    parents: tuple[ParentCaptionRecord, ...]

    @property
    def motion_source_keys(self) -> tuple[str, ...]:
        return tuple(parent.source_sha256 for parent in self.parents)

    @property
    def motion_positive_keys(self) -> tuple[str, ...]:
        return tuple(parent.annotation_family_sha256 for parent in self.parents)

    @property
    def captions(self) -> tuple[str, ...]:
        return tuple(row for parent in self.parents for row in parent.captions)

    @property
    def text_positive_keys(self) -> tuple[str, ...]:
        return tuple(
            parent.annotation_family_sha256 for parent in self.parents for _ in parent.captions
        )

    @property
    def caption_commitments(self) -> tuple[bytes, ...]:
        # Lineage only: not inputs to the text adapter or motion network.
        return tuple(
            hashlib.sha256(
                bytes.fromhex(parent.source_sha256)
                + bytes.fromhex(parent.annotation_family_sha256)
                + ordinal.to_bytes(8, "big")
            ).digest()
            for parent in self.parents
            for ordinal in range(len(parent.captions))
        )

    def positive_mask(self, *, device: torch.device) -> Tensor:
        return capture_family_positive_mask(
            self.motion_positive_keys, self.text_positive_keys, device=device
        )


class ParentRetrievalTask:
    """Validate a complete caller-admitted population before selecting batches.

    Released siblings are not separate motion rows. Duplicate source/family
    records are rejected, not silently dropped. Component split consistency is
    checked on the complete population, before selecting a training/gallery
    subset. Captions are never deduplicated, normalized, or sampled.
    """

    def __init__(
        self,
        parents: tuple[ParentCaptionRecord, ...],
        *,
        expected_family_keys: tuple[str, ...],
    ) -> None:
        if (
            type(parents) is not tuple
            or not parents
            or any(type(parent) is not ParentCaptionRecord for parent in parents)
        ):
            raise ValueError("task requires a nonempty exact parent record tuple")
        if type(expected_family_keys) is not tuple or not expected_family_keys:
            raise ValueError("the complete expected parent population must be explicit")
        for key in expected_family_keys:
            _sha_key(key)
        families = tuple(parent.annotation_family_sha256 for parent in parents)
        sources = tuple(parent.source_sha256 for parent in parents)
        if len(set(families)) != len(families) or len(set(sources)) != len(sources):
            raise ValueError("released siblings or repeated parents cannot be gallery rows")
        if len(set(expected_family_keys)) != len(expected_family_keys) or (
            set(families) != set(expected_family_keys)
        ):
            raise ValueError("parent population is incomplete or contains unexpected families")
        component_splits = {}
        for parent in parents:
            previous = component_splits.setdefault(parent.component, parent.split)
            if previous != parent.split:
                raise ValueError("a participant component cannot cross task splits")
        self.parents = tuple(sorted(parents, key=lambda parent: parent.annotation_family_sha256))
        self._by_family = {parent.annotation_family_sha256: parent for parent in self.parents}

    def batch(self, family_keys: tuple[str, ...]) -> ParentRetrievalBatch:
        """Preserve caller batch order and include every row of each parent once."""
        if type(family_keys) is not tuple or not family_keys:
            raise ValueError("batch parent identities must be explicit and nonempty")
        for key in family_keys:
            _sha_key(key)
        if len(set(family_keys)) != len(family_keys) or any(
            key not in self._by_family for key in family_keys
        ):
            raise ValueError("batch contains duplicate or unadmitted parent families")
        return ParentRetrievalBatch(tuple(self._by_family[key] for key in family_keys))

    def gallery(self, *, split: str) -> ParentRetrievalBatch:
        """Use the complete requested split, not a released-window gallery."""
        return self.batch(
            tuple(
                parent.annotation_family_sha256 for parent in self.parents if parent.split == split
            )
        )
