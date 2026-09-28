"""Disclosed machine-caption training rows, never human motion truth."""

from dataclasses import dataclass, field
import hashlib

from .capture_validation import _validate_text_batch
from .continuous_parent_training import ParentWeakCounterfactualRows
from .frozen_clip_text import FrozenClipTextBatch, pool_frozen_clip_text_rows
from .parent_clip_rows import ParentHumanClipRows
from .parent_retrieval_task import _sha_key


@dataclass(frozen=True)
class WeakCaptionRecord:
    """One caller-admitted machine output and its original human occurrence.

    The private caller verifies the actual frozen generation record/policy.
    A digest or inclusion bit here is not proof of its semantic correctness.
    Rejected generation outputs remain in the private log, not fabricated rows.
    """

    source_sha256: str
    annotation_family_sha256: str
    positive_ordinal: int
    negative_caption: str = field(repr=False)
    generation_record_sha256: str
    policy_sha256: str
    included_weak: bool
    label_source: str = "machine_caption_contradiction"
    generator_model: str = "gpt-5.6-sol"

    def __post_init__(self):
        for key in (
            self.source_sha256,
            self.annotation_family_sha256,
            self.generation_record_sha256,
            self.policy_sha256,
        ):
            _sha_key(key)
        if (
            type(self.positive_ordinal) is not int
            or not 0 <= self.positive_ordinal < 2**64
            or type(self.negative_caption) is not str
            or not self.negative_caption.strip()
            or type(self.included_weak) is not bool
            or self.label_source != "machine_caption_contradiction"
            or self.generator_model != "gpt-5.6-sol"
        ):
            raise ValueError("weak rows require explicit occurrence and machine provenance")

    @property
    def caption_commitment(self) -> bytes:
        return hashlib.sha256(
            b"phaseset-parent-machine-caption-v1\0"
            + bytes.fromhex(self.source_sha256)
            + bytes.fromhex(self.annotation_family_sha256)
            + self.positive_ordinal.to_bytes(8, "big")
            + hashlib.sha256(self.negative_caption.encode("utf-8")).digest()
            + bytes.fromhex(self.generation_record_sha256)
            + bytes.fromhex(self.policy_sha256)
        ).digest()


class ParentWeakClipRows:
    """Stage-specific frozen weak pool, with an unchanged original human prefix.

    Stage training components are explicit: an outer fold may learn an old
    main-validation component, but this pool must not contain that stage's
    held-out or final-test rows. Evaluation delegates to human-only selection.
    No language model, CLIP tower, motion, score, or RNG is accessed here.
    """

    def __init__(
        self,
        human: ParentHumanClipRows,
        extra_original: FrozenClipTextBatch,
        records: tuple[WeakCaptionRecord, ...],
        *,
        train_components: tuple[str, ...],
        policy_sha256: str,
    ):
        _sha_key(policy_sha256)
        if type(human) is not ParentHumanClipRows:
            raise TypeError("need the original admitted human row provider")
        if (
            type(train_components) is not tuple
            or not train_components
            or len(set(train_components)) != len(train_components)
            or any(type(c) is not str or not c for c in train_components)
            or not set(train_components) <= {p.component for p in human.parents.values()}
        ):
            raise ValueError("weak pool needs explicit stage training components")
        if (
            type(records) is not tuple
            or not records
            or any(type(r) is not WeakCaptionRecord for r in records)
        ):
            raise ValueError("weak pool needs an exact nonempty record tuple")
        _validate_text_batch(extra_original)  # No derived or relabeled pool.
        if len(records) != len(extra_original.caption_commitments):
            raise ValueError("weak machine encoding must cover exactly its admitted records")
        for record, commitment, row in zip(
            records,
            extra_original.caption_commitments,
            extra_original.receipt.caption_rows,
            strict=True,
        ):
            parent = human.parents.get(record.annotation_family_sha256)
            if (
                parent is None
                or parent.source_sha256 != record.source_sha256
                or parent.component not in train_components
            ):
                raise ValueError("weak row is outside this stage's admitted training parents")
            if (
                record.positive_ordinal >= len(parent.captions)
                or record.policy_sha256 != policy_sha256
            ):
                raise ValueError("weak positive occurrence or frozen policy differs")
            if record.negative_caption in parent.captions:
                raise ValueError("weak negative repeats an original human positive")
            if (
                commitment != record.caption_commitment
                or row[2] != hashlib.sha256(record.negative_caption.encode("utf-8")).hexdigest()
            ):
                raise ValueError("weak cached sentence differs from the admitted machine record")
        self.human, self.extra_original, self.records = human, extra_original, records
        self.train_components, self.policy_sha256 = train_components, policy_sha256
        self.by_family = {
            family: tuple(i for i, r in enumerate(records) if r.annotation_family_sha256 == family)
            for family in human.parents
        }

    def text(self, labels, *, training):
        human_rows, human_cf = self.human.text(labels, training=training)
        if not training:
            return human_rows, human_cf
        if any(p.component not in self.train_components for p in labels.parents):
            raise ValueError("weak training cannot read stage-held-out labels")
        indices = tuple(
            i for p in labels.parents for i in self.by_family[p.annotation_family_sha256]
        )
        if not indices:
            return human_rows, ParentWeakCounterfactualRows((), (), (), ())
        rows = pool_frozen_clip_text_rows(
            self.human.original,
            self.extra_original,
            prefix_indices=tuple(self.human.indices[c] for c in labels.caption_commitments),
            extra_indices=indices,
        )
        offsets, cursor = {}, 0
        for parent in labels.parents:
            offsets[parent.annotation_family_sha256] = cursor
            cursor += len(parent.captions)
        records = tuple(self.records[i] for i in indices)
        cf = ParentWeakCounterfactualRows(
            tuple(r.source_sha256 for r in records),
            tuple(offsets[r.annotation_family_sha256] + r.positive_ordinal for r in records),
            tuple(range(cursor, cursor + len(records))),
            tuple(r.included_weak for r in records),
        )
        return rows, cf
