"""Authority-zero binding for a candidate machine-fused window training policy.

This private candidate is a consistency validator, not a policy selection,
rights authenticator, semantic-equivalence review, or training authorization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Final

import torch

from phaseset_core.caption_fusion import (
    ANNOTATION_KIND,
    TASK_SCOPE,
    CaptionFusionInput,
    CaptionFusionReceipt,
    MachineFusedCaption,
    _input_digest as _caption_input_digest,
    _output_digest as _caption_output_digest,
    canonical_caption_fusion_receipt_bytes,
)
from phaseset_core.contracts import group_commitment
from phaseset_core.prepared_data_v2 import (
    PreparedMotionTextBlock,
    PreparedTrainingDataSourceV2,
)
from phaseset_core.pipeline import PreparedGroupSample
from phaseset_core.training import RetrievalTrainingBatch


SCHEMA: Final = "phaseset-private-caption-role-validation-v2"
WINDOW_SUMMARY_SCHEMA: Final = "phaseset-private-window-manifest-content-summary-v1"
CANDIDATE_POLICY_ID: Final = "candidate-machine-fused-window-supervision-v1"
STATUS: Final = "VALIDATED_CANDIDATE_NOT_SELECTED_AUTHORITY0"
CAPTION_ROLES: Final = ("group_description", "semantic_paraphrase")
INSPECTION_SEED: Final = 1729
MAX_WINDOWS: Final = 16_384
MAX_CAPTIONS: Final = 32_768
MAX_MICROBATCHES: Final = 16_384
MAX_EMBEDDING_BYTES: Final = 64 * 1024 * 1024


class CaptionRoleProvenanceError(ValueError):
    """The proposed private supervision binding is inconsistent."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptionRoleProvenanceError(f"{label} must be lowercase SHA-256 hex")
    return value


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise CaptionRoleProvenanceError(f"{label} must be exact bytes[32]")
    return value


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _manifest_digest(domain: bytes, values: tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        raw = value.encode("ascii")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True, slots=True, repr=False)
class ActorCaptionContentSummary:
    """Verified-manifest content digest for one same-window actor caption."""

    actor_commitment: bytes = field(repr=False)
    source_record_sha256: str = field(repr=False)
    text_utf8_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actor_commitment",
            _raw32(self.actor_commitment, "actor caption commitment"),
        )
        _lower_sha256(self.source_record_sha256, "actor caption source record sha256")
        _lower_sha256(self.text_utf8_sha256, "actor caption text sha256")

    def _private_dict(self) -> dict[str, str]:
        return {
            "actor_commitment": self.actor_commitment.hex(),
            "source_record_sha256": self.source_record_sha256,
            "text_utf8_sha256": self.text_utf8_sha256,
        }

    def __repr__(self) -> str:
        return "ActorCaptionContentSummary(<private lineage>)"


@dataclass(frozen=True, slots=True, repr=False)
class HolisticContextContentSummary:
    """Verified-manifest content digest for optional same-window context."""

    source_record_sha256: str = field(repr=False)
    text_utf8_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        _lower_sha256(self.source_record_sha256, "holistic source record sha256")
        _lower_sha256(self.text_utf8_sha256, "holistic text sha256")

    def _private_dict(self) -> dict[str, str]:
        return {
            "source_record_sha256": self.source_record_sha256,
            "text_utf8_sha256": self.text_utf8_sha256,
        }

    def __repr__(self) -> str:
        return "HolisticContextContentSummary(<private lineage>)"


@dataclass(frozen=True, slots=True, repr=False)
class WindowManifestRow:
    """Private content summary of one already-verified train-window row."""

    window_sha256: str
    window_commitment: bytes = field(repr=False)
    actor_commitments: tuple[bytes, ...] = field(repr=False)
    group_commitment: bytes = field(repr=False)
    source_sha256: str = field(repr=False)
    source_start_frame: int
    actor_caption_summaries: tuple[ActorCaptionContentSummary, ...] = field(repr=False)
    holistic_context_summary: HolisticContextContentSummary | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        _lower_sha256(self.window_sha256, "window_sha256")
        window = _raw32(self.window_commitment, "window_commitment")
        if type(self.actor_commitments) is not tuple or len(self.actor_commitments) < 3:
            raise CaptionRoleProvenanceError(
                "window summary requires an exact tuple of at least three actors"
            )
        actors = tuple(
            _raw32(value, f"actor_commitments[{index}]")
            for index, value in enumerate(self.actor_commitments)
        )
        if len(set(actors)) != len(actors):
            raise CaptionRoleProvenanceError("window summary repeats an actor commitment")
        if (
            type(self.actor_caption_summaries) is not tuple
            or len(self.actor_caption_summaries) != len(actors)
            or any(
                type(value) is not ActorCaptionContentSummary
                for value in self.actor_caption_summaries
            )
        ):
            raise CaptionRoleProvenanceError(
                "window summary actor-caption census is inconsistent"
            )
        actor_caption_summaries = tuple(
            sorted(self.actor_caption_summaries, key=lambda value: value.actor_commitment)
        )
        if tuple(value.actor_commitment for value in actor_caption_summaries) != tuple(
            sorted(actors)
        ):
            raise CaptionRoleProvenanceError(
                "window summary actor-caption lineage is inconsistent"
            )
        if self.holistic_context_summary is not None and type(
            self.holistic_context_summary
        ) is not HolisticContextContentSummary:
            raise CaptionRoleProvenanceError(
                "holistic_context_summary must be exact or absent"
            )
        group = _raw32(self.group_commitment, "group_commitment")
        if group_commitment(actors) != group:
            raise CaptionRoleProvenanceError("window summary group commitment is inconsistent")
        _lower_sha256(self.source_sha256, "source_sha256")
        if type(self.source_start_frame) is not int or self.source_start_frame < 0:
            raise CaptionRoleProvenanceError(
                "source_start_frame must be an exact nonnegative int"
            )
        object.__setattr__(self, "window_commitment", window)
        object.__setattr__(self, "actor_commitments", actors)
        object.__setattr__(self, "group_commitment", group)
        object.__setattr__(self, "actor_caption_summaries", actor_caption_summaries)

    def _private_dict(self) -> dict[str, object]:
        return {
            "actor_commitments": [value.hex() for value in self.actor_commitments],
            "actor_caption_summaries": [
                value._private_dict() for value in self.actor_caption_summaries
            ],
            "group_commitment": self.group_commitment.hex(),
            "source_sha256": self.source_sha256,
            "source_start_frame": self.source_start_frame,
            "holistic_context_summary": (
                None
                if self.holistic_context_summary is None
                else self.holistic_context_summary._private_dict()
            ),
            "window_commitment": self.window_commitment.hex(),
            "window_sha256": self.window_sha256,
        }

    def __repr__(self) -> str:
        return "WindowManifestRow(<private lineage>)"


@dataclass(frozen=True, slots=True, repr=False)
class WindowManifestContentSummary:
    """Content extracted from one verified private train-window manifest."""

    manifest_sha256: str
    rows: tuple[WindowManifestRow, ...] = field(repr=False)
    split: str = "train"
    schema: str = WINDOW_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        _lower_sha256(self.manifest_sha256, "window manifest sha256")
        if self.split != "train" or self.schema != WINDOW_SUMMARY_SCHEMA:
            raise CaptionRoleProvenanceError("window summary schema or split is not registered")
        if (
            type(self.rows) is not tuple
            or not self.rows
            or len(self.rows) > MAX_WINDOWS
            or any(type(row) is not WindowManifestRow for row in self.rows)
        ):
            raise CaptionRoleProvenanceError("window summary rows are invalid or exceed the bound")
        windows = tuple(row.window_sha256 for row in self.rows)
        commitments = tuple(row.window_commitment for row in self.rows)
        if len(set(windows)) != len(windows) or len(set(commitments)) != len(commitments):
            raise CaptionRoleProvenanceError("window summary repeats a window identity")

    @property
    def content_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "manifest_sha256": self.manifest_sha256,
                    "rows": [row._private_dict() for row in self.rows],
                    "schema": self.schema,
                    "split": self.split,
                }
            )
        )

    def __repr__(self) -> str:
        return (
            f"WindowManifestContentSummary(windows={len(self.rows)}, "
            "<private lineage>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class WindowFusionHandoff:
    """Ephemeral fusion-to-CLIP handoff retained for one private validation."""

    fusion_input: CaptionFusionInput = field(repr=False)
    fused_caption: MachineFusedCaption = field(repr=False)
    clip_caption_commitments: tuple[bytes, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.fusion_input) is not CaptionFusionInput:
            raise CaptionRoleProvenanceError("fusion_input must be exact CaptionFusionInput")
        if type(self.fused_caption) is not MachineFusedCaption:
            raise CaptionRoleProvenanceError("fused_caption must be exact MachineFusedCaption")
        if (
            type(self.clip_caption_commitments) is not tuple
            or len(self.clip_caption_commitments) != 2
        ):
            raise CaptionRoleProvenanceError(
                "clip_caption_commitments must contain exactly two rows"
            )
        checked = tuple(
            _raw32(value, f"clip_caption_commitments[{index}]")
            for index, value in enumerate(self.clip_caption_commitments)
        )
        if len(set(checked)) != 2:
            raise CaptionRoleProvenanceError("fusion handoff repeats a caption commitment")
        object.__setattr__(self, "clip_caption_commitments", checked)

    def __repr__(self) -> str:
        return "WindowFusionHandoff(<private text and lineage>)"


@dataclass(frozen=True, slots=True)
class CaptionRoleValidationReceipt:
    """Canonical private digest receipt with no authority or policy selection."""

    window_manifest_sha256: str
    window_manifest_content_sha256: str
    prepared_train_manifest_sha256: str
    dataset_manifest_sha256: str
    caption_manifest_sha256: str
    human_text_rights_assertion_sha256: str
    row_binding_manifest_sha256: str
    fusion_receipt_manifest_sha256: str
    clip_receipt_manifest_sha256: str
    positive_family_manifest_sha256: str
    window_count: int
    caption_count: int
    microbatch_count: int
    inspection_seed: int = INSPECTION_SEED
    candidate_policy_id: str = CANDIDATE_POLICY_ID
    formal_supervision_policy_selected: bool = False
    rights_authenticated_by_this_receipt: bool = False
    semantic_equivalence_proved: bool = False
    training_authorized: bool = False
    scientific_result_claimed: bool = False
    authority: int = 0
    status: str = STATUS
    schema: str = SCHEMA

    def canonical_json_bytes(self) -> bytes:
        for name in (
            "window_manifest_sha256",
            "window_manifest_content_sha256",
            "prepared_train_manifest_sha256",
            "dataset_manifest_sha256",
            "caption_manifest_sha256",
            "human_text_rights_assertion_sha256",
            "row_binding_manifest_sha256",
            "fusion_receipt_manifest_sha256",
            "clip_receipt_manifest_sha256",
            "positive_family_manifest_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if (
            type(self.window_count) is not int
            or not 1 <= self.window_count <= MAX_WINDOWS
            or type(self.caption_count) is not int
            or self.caption_count != 2 * self.window_count
            or type(self.microbatch_count) is not int
            or not 1 <= self.microbatch_count <= MAX_MICROBATCHES
            or type(self.inspection_seed) is not int
            or self.inspection_seed != INSPECTION_SEED
            or self.formal_supervision_policy_selected is not False
            or self.rights_authenticated_by_this_receipt is not False
            or self.semantic_equivalence_proved is not False
            or self.training_authorized is not False
            or self.scientific_result_claimed is not False
            or type(self.authority) is not int
            or self.authority != 0
        ):
            raise CaptionRoleProvenanceError(
                "validation receipt census or frozen field type is invalid"
            )
        frozen = {
            "authority": 0,
            "candidate_policy_id": CANDIDATE_POLICY_ID,
            "formal_supervision_policy_selected": False,
            "rights_authenticated_by_this_receipt": False,
            "schema": SCHEMA,
            "scientific_result_claimed": False,
            "semantic_equivalence_proved": False,
            "status": STATUS,
            "training_authorized": False,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise CaptionRoleProvenanceError(f"receipt frozen field {name} changed")
        return _canonical_json(
            {
                **frozen,
                "caption_count": self.caption_count,
                "caption_manifest_sha256": self.caption_manifest_sha256,
                "clip_receipt_manifest_sha256": self.clip_receipt_manifest_sha256,
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "fusion_receipt_manifest_sha256": self.fusion_receipt_manifest_sha256,
                "human_text_rights_assertion_sha256": (
                    self.human_text_rights_assertion_sha256
                ),
                "microbatch_count": self.microbatch_count,
                "positive_family_manifest_sha256": self.positive_family_manifest_sha256,
                "prepared_train_manifest_sha256": self.prepared_train_manifest_sha256,
                "row_binding_manifest_sha256": self.row_binding_manifest_sha256,
                "inspection_seed": self.inspection_seed,
                "window_count": self.window_count,
                "window_manifest_content_sha256": self.window_manifest_content_sha256,
                "window_manifest_sha256": self.window_manifest_sha256,
            }
        )

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.canonical_json_bytes())


@dataclass(frozen=True, slots=True, repr=False)
class _BoundCaption:
    role: str
    commitment: bytes
    text_sha256: str
    embedding_sha256: str
    embedding: torch.Tensor = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class _BoundWindow:
    family: bytes
    row: WindowManifestRow
    captions: tuple[_BoundCaption, _BoundCaption]
    fusion_receipt_sha256: str
    clip_receipt_sha256: str
    row_binding_sha256: str


def _validate_handoff(
    *,
    sample: PreparedGroupSample,
    row: WindowManifestRow,
    handoff: WindowFusionHandoff,
    clip_commitments: tuple[bytes, bytes],
    clip_text_digests: tuple[str, str],
    clip_embedding_digests: tuple[str, str],
    clip_embeddings: torch.Tensor,
    clip_receipt_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_caption_manifest_sha256: str,
    expected_human_text_rights_assertion_sha256: str,
) -> _BoundWindow:
    # PreparedMotionTextBlock has already reconstructed exact PreparedGroupSample
    # instances; avoid importing its implementation-only row helper.
    if sample.window_sha256 != row.window_sha256:
        raise CaptionRoleProvenanceError("prepared and summarized window identity differ")
    if (
        sample.source_sha256 != row.source_sha256
        or sample.source_start_frame != row.source_start_frame
        or sample.actor_commitments != row.actor_commitments
        or sample.group_commitment != row.group_commitment
    ):
        raise CaptionRoleProvenanceError("prepared and summarized window lineage differ")

    fusion_input = handoff.fusion_input
    fused = handoff.fused_caption
    receipt = fused.receipt
    if fusion_input.window_commitment != row.window_commitment:
        raise CaptionRoleProvenanceError("fusion and summarized window commitments differ")
    fusion_actors = tuple(sorted(item.actor_commitment for item in fusion_input.actor_captions))
    if fusion_actors != tuple(sorted(row.actor_commitments)):
        raise CaptionRoleProvenanceError("fusion and summarized actor lineages differ")
    fusion_actor_content = tuple(
        sorted(
            (
                item.actor_commitment,
                item.source_record_sha256,
                _sha256_bytes(item.text.encode("utf-8")),
            )
            for item in fusion_input.actor_captions
        )
    )
    summarized_actor_content = tuple(
        (
            item.actor_commitment,
            item.source_record_sha256,
            item.text_utf8_sha256,
        )
        for item in row.actor_caption_summaries
    )
    if fusion_actor_content != summarized_actor_content:
        raise CaptionRoleProvenanceError(
            "fusion actor-caption content differs from verified window summary"
        )
    holistic = fusion_input.holistic_context
    holistic_summary = row.holistic_context_summary
    if (holistic is None) != (holistic_summary is None):
        raise CaptionRoleProvenanceError(
            "fusion holistic-context presence differs from verified window summary"
        )
    if holistic is not None and holistic_summary is not None and (
        holistic.window_commitment != row.window_commitment
        or holistic.source_record_sha256 != holistic_summary.source_record_sha256
        or _sha256_bytes(holistic.text.encode("utf-8"))
        != holistic_summary.text_utf8_sha256
    ):
        raise CaptionRoleProvenanceError(
            "fusion holistic content differs from verified window summary"
        )
    provenance = fusion_input.provenance
    if (
        provenance.split != "train"
        or provenance.dataset_manifest_sha256 != expected_dataset_manifest_sha256
        or provenance.caption_manifest_sha256 != expected_caption_manifest_sha256
        or provenance.human_text_rights_assertion_sha256
        != expected_human_text_rights_assertion_sha256
        or provenance.sealed_test_manifest_sha256 is not None
        or provenance.sealed_test_caption_sha256 is not None
    ):
        raise CaptionRoleProvenanceError("fusion provenance differs from admitted train inputs")

    try:
        fusion_receipt_raw = canonical_caption_fusion_receipt_bytes(receipt)
    except Exception:
        raise CaptionRoleProvenanceError("caption-fusion receipt is invalid") from None
    expected_admission = _sha256_bytes(
        b"phaseset-caption-fusion-local-admission-v1\x00"
        + provenance.digest().encode("ascii")
    )
    if (
        type(receipt) is not CaptionFusionReceipt
        or receipt.input_sha256 != _caption_input_digest(fusion_input)
        or receipt.output_sha256
        != _caption_output_digest(fused.group_description, fused.semantic_paraphrase)
        or receipt.provenance_sha256 != provenance.digest()
        or receipt.admission_sha256 != expected_admission
        or receipt.actor_count != len(row.actor_commitments)
        or receipt.annotation_kind != ANNOTATION_KIND
        or receipt.task_scope != TASK_SCOPE
        or receipt.machine_fused is not True
        or receipt.human_annotation is not False
        or receipt.public_verification_performed is not False
        or type(receipt.authority) is not int
        or receipt.authority != 0
        or receipt.scientific_result_claimed is not False
        or receipt.result_claimed is not False
    ):
        raise CaptionRoleProvenanceError("caption-fusion receipt does not bind this handoff")

    texts = (fused.group_description, fused.semantic_paraphrase)
    text_digests = tuple(_sha256_bytes(value.encode("utf-8")) for value in texts)
    if (
        handoff.clip_caption_commitments != clip_commitments
        or text_digests != clip_text_digests
    ):
        raise CaptionRoleProvenanceError("fusion output differs from ordered frozen-CLIP rows")
    caption_values = tuple(
        _BoundCaption(
            role=role,
            commitment=commitment,
            text_sha256=text_digest,
            embedding_sha256=embedding_digest,
            embedding=clip_embeddings[index : index + 1].detach().clone().contiguous(),
        )
        for index, (role, commitment, text_digest, embedding_digest) in enumerate(
            zip(
                CAPTION_ROLES,
                clip_commitments,
                clip_text_digests,
                clip_embedding_digests,
                strict=True,
            )
        )
    )
    family = bytes.fromhex(row.window_sha256)
    fusion_receipt_sha256 = _sha256_bytes(fusion_receipt_raw)
    row_binding = {
        "actor_commitments": [value.hex() for value in row.actor_commitments],
        "captions": [
            {
                "clip_caption_commitment": caption.commitment.hex(),
                "embedding_sha256": caption.embedding_sha256,
                "role": caption.role,
                "text_utf8_sha256": caption.text_sha256,
            }
            for caption in caption_values
        ],
        "clip_receipt_sha256": clip_receipt_sha256,
        "fusion_input_sha256": receipt.input_sha256,
        "fusion_output_sha256": receipt.output_sha256,
        "fusion_provenance_sha256": receipt.provenance_sha256,
        "fusion_receipt_sha256": fusion_receipt_sha256,
        "group_commitment": row.group_commitment.hex(),
        "positive_family": family.hex(),
        "source_sha256": row.source_sha256,
        "source_start_frame": row.source_start_frame,
        "window_commitment": row.window_commitment.hex(),
        "window_sha256": row.window_sha256,
    }
    return _BoundWindow(
        family=family,
        row=row,
        captions=(caption_values[0], caption_values[1]),
        fusion_receipt_sha256=fusion_receipt_sha256,
        clip_receipt_sha256=clip_receipt_sha256,
        row_binding_sha256=_sha256_bytes(_canonical_json(row_binding)),
    )


def validate_candidate_machine_fused_window_training(
    *,
    blocks: tuple[PreparedMotionTextBlock, ...],
    handoffs: tuple[WindowFusionHandoff, ...],
    window_summary: WindowManifestContentSummary,
    train_source: PreparedTrainingDataSourceV2,
    expected_window_manifest_sha256: str,
    expected_prepared_train_manifest_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_caption_manifest_sha256: str,
    expected_human_text_rights_assertion_sha256: str,
) -> CaptionRoleValidationReceipt:
    """Validate one complete candidate auxiliary-window supervision census."""

    expected_window = _lower_sha256(
        expected_window_manifest_sha256, "expected window manifest sha256"
    )
    expected_prepared = _lower_sha256(
        expected_prepared_train_manifest_sha256,
        "expected prepared train manifest sha256",
    )
    expected_dataset = _lower_sha256(
        expected_dataset_manifest_sha256, "expected dataset manifest sha256"
    )
    expected_caption = _lower_sha256(
        expected_caption_manifest_sha256, "expected caption manifest sha256"
    )
    expected_rights = _lower_sha256(
        expected_human_text_rights_assertion_sha256,
        "expected human text rights assertion sha256",
    )
    if type(window_summary) is not WindowManifestContentSummary:
        raise CaptionRoleProvenanceError(
            "window_summary must be exact WindowManifestContentSummary"
        )
    if window_summary.manifest_sha256 != expected_window:
        raise CaptionRoleProvenanceError("window summary differs from expected manifest digest")
    if (
        type(blocks) is not tuple
        or not blocks
        or any(type(block) is not PreparedMotionTextBlock for block in blocks)
    ):
        raise CaptionRoleProvenanceError("blocks must be a nonempty exact prepared-v2 tuple")
    if type(handoffs) is not tuple or any(
        type(value) is not WindowFusionHandoff for value in handoffs
    ):
        raise CaptionRoleProvenanceError("handoffs must be an exact handoff tuple")
    if type(train_source) is not PreparedTrainingDataSourceV2:
        raise CaptionRoleProvenanceError("train_source must be exact prepared-data-v2 source")
    if train_source.split != "train" or train_source.manifest_sha256 != expected_prepared:
        raise CaptionRoleProvenanceError("prepared train source identity differs from expected")

    prepared_rows: list[tuple[PreparedGroupSample, int, int]] = []
    caption_offset = 0
    embedding_bytes = 0
    clip_receipt_digests: list[str] = []
    block_values: list[tuple[torch.Tensor, tuple[tuple[object, ...], ...], tuple[bytes, ...]]] = []
    for block_index, block in enumerate(blocks):
        if len(block.samples) + len(prepared_rows) > MAX_WINDOWS:
            raise CaptionRoleProvenanceError("prepared window census exceeds the bound")
        if any(count != 2 for count in block.text_counts):
            raise CaptionRoleProvenanceError(
                "candidate policy requires exactly two text rows per motion window"
            )
        embeddings = block.text_batch.embeddings
        embedding_bytes += embeddings.numel() * embeddings.element_size()
        if embedding_bytes > MAX_EMBEDDING_BYTES:
            raise CaptionRoleProvenanceError("prepared frozen embeddings exceed the byte bound")
        receipt = block.text_batch.receipt
        try:
            clip_receipt_sha256 = receipt.sha256
        except Exception:
            raise CaptionRoleProvenanceError("frozen-CLIP receipt is invalid") from None
        clip_receipt_digests.append(clip_receipt_sha256)
        rows = receipt.caption_rows
        commitments = block.text_batch.caption_commitments
        local_offset = 0
        for sample, count in zip(block.samples, block.text_counts, strict=True):
            prepared_rows.append((sample, block_index, local_offset))
            local_offset += count
            caption_offset += count
        if local_offset != len(commitments) or len(rows) != len(commitments):
            raise CaptionRoleProvenanceError("prepared block text census is inconsistent")
        block_values.append((embeddings, rows, commitments))
    if (
        not prepared_rows
        or len(prepared_rows) != len(handoffs)
        or len(prepared_rows) != len(window_summary.rows)
        or caption_offset != 2 * len(prepared_rows)
        or caption_offset > MAX_CAPTIONS
    ):
        raise CaptionRoleProvenanceError("prepared, fusion, and window censuses differ")

    bound: list[_BoundWindow] = []
    all_caption_commitments: set[bytes] = set()
    all_families: set[bytes] = set()
    all_fusion_receipts: set[str] = set()
    for index, ((sample, block_index, local_offset), row, handoff) in enumerate(
        zip(prepared_rows, window_summary.rows, handoffs, strict=True)
    ):
        embeddings, receipt_rows, block_commitments = block_values[block_index]
        row_slice = receipt_rows[local_offset : local_offset + 2]
        commitment_slice = block_commitments[local_offset : local_offset + 2]
        if (
            len(row_slice) != 2
            or len(commitment_slice) != 2
            or tuple(item[0] for item in row_slice) != (local_offset, local_offset + 1)
            or tuple(item[1] for item in row_slice)
            != tuple(value.hex() for value in commitment_slice)
        ):
            raise CaptionRoleProvenanceError(
                f"frozen-CLIP ordered rows differ at prepared window {index}"
            )
        checked = _validate_handoff(
            sample=sample,
            row=row,
            handoff=handoff,
            clip_commitments=(commitment_slice[0], commitment_slice[1]),
            clip_text_digests=(row_slice[0][2], row_slice[1][2]),
            clip_embedding_digests=(row_slice[0][8], row_slice[1][8]),
            clip_embeddings=embeddings[local_offset : local_offset + 2],
            clip_receipt_sha256=clip_receipt_digests[block_index],
            expected_dataset_manifest_sha256=expected_dataset,
            expected_caption_manifest_sha256=expected_caption,
            expected_human_text_rights_assertion_sha256=expected_rights,
        )
        commitments = {caption.commitment for caption in checked.captions}
        if (
            checked.family in all_families
            or all_caption_commitments.intersection(commitments)
            or checked.fusion_receipt_sha256 in all_fusion_receipts
        ):
            raise CaptionRoleProvenanceError("candidate binding repeats a row identity")
        all_families.add(checked.family)
        all_caption_commitments.update(commitments)
        all_fusion_receipts.add(checked.fusion_receipt_sha256)
        bound.append(checked)

    by_family = {value.family: value for value in bound}
    by_caption = {
        caption.commitment: (value.family, caption)
        for value in bound
        for caption in value.captions
    }
    seen_families: set[bytes] = set()
    seen_captions: set[bytes] = set()
    microbatch_count = 0
    observed_embedding_bytes = 0
    try:
        iterator = train_source.iter_epoch(epoch=0, seed=INSPECTION_SEED)
        for batch_index, batch in enumerate(iterator):
            microbatch_count += 1
            if microbatch_count > MAX_MICROBATCHES:
                raise CaptionRoleProvenanceError("prepared source exceeds the microbatch bound")
            if type(batch) is not RetrievalTrainingBatch or batch.split != "train":
                raise CaptionRoleProvenanceError("prepared source emitted a non-train batch")
            observed_embedding_bytes += (
                batch.text_embeddings.numel() * batch.text_embeddings.element_size()
            )
            if observed_embedding_bytes > MAX_EMBEDDING_BYTES:
                raise CaptionRoleProvenanceError("materialized embeddings exceed the byte bound")
            if len(seen_families) + batch.motion_count > MAX_WINDOWS:
                raise CaptionRoleProvenanceError("materialized window census exceeds the bound")
            if len(seen_captions) + len(batch.text_commitments) > MAX_CAPTIONS:
                raise CaptionRoleProvenanceError("materialized caption census exceeds the bound")
            for motion_index, family in enumerate(batch.motion_positive_ids):
                expected = by_family.get(family)
                if expected is None or family in seen_families:
                    raise CaptionRoleProvenanceError(
                        f"materialized motion family differs at batch {batch_index}"
                    )
                group_row = batch.groups.group_commitments[motion_index]
                actor_row = tuple(
                    value
                    for value in batch.groups.actor_commitments[motion_index]
                    if value is not None
                )
                if (
                    group_row != expected.row.group_commitment
                    or actor_row != tuple(sorted(expected.row.actor_commitments))
                ):
                    raise CaptionRoleProvenanceError(
                        f"materialized motion lineage differs at batch {batch_index}"
                    )
                seen_families.add(family)
            for text_index, (family, commitment) in enumerate(
                zip(batch.text_positive_ids, batch.text_commitments, strict=True)
            ):
                expected_pair = by_caption.get(commitment)
                if expected_pair is None or commitment in seen_captions:
                    raise CaptionRoleProvenanceError(
                        f"materialized caption lineage differs at batch {batch_index}"
                    )
                expected_family, expected_caption_row = expected_pair
                if family != expected_family or not torch.equal(
                    batch.text_embeddings[text_index : text_index + 1],
                    expected_caption_row.embedding,
                ):
                    raise CaptionRoleProvenanceError(
                        f"materialized caption value or family differs at batch {batch_index}"
                    )
                if (
                    _sha256_bytes(_tensor_bytes(batch.text_embeddings[text_index : text_index + 1]))
                    != expected_caption_row.embedding_sha256
                ):
                    raise CaptionRoleProvenanceError(
                        f"materialized caption digest differs at batch {batch_index}"
                    )
                seen_captions.add(commitment)
    except CaptionRoleProvenanceError:
        raise
    except Exception:
        raise CaptionRoleProvenanceError("prepared train source iteration failed") from None

    if not microbatch_count:
        raise CaptionRoleProvenanceError("prepared train source emitted no microbatches")
    if seen_families != all_families or seen_captions != all_caption_commitments:
        raise CaptionRoleProvenanceError("materialized prepared source has an incomplete census")

    result = CaptionRoleValidationReceipt(
        window_manifest_sha256=expected_window,
        window_manifest_content_sha256=window_summary.content_sha256,
        prepared_train_manifest_sha256=expected_prepared,
        dataset_manifest_sha256=expected_dataset,
        caption_manifest_sha256=expected_caption,
        human_text_rights_assertion_sha256=expected_rights,
        row_binding_manifest_sha256=_manifest_digest(
            b"phaseset-caption-role-row-binding-manifest-v1\x00",
            tuple(value.row_binding_sha256 for value in bound),
        ),
        fusion_receipt_manifest_sha256=_manifest_digest(
            b"phaseset-caption-role-fusion-receipt-manifest-v1\x00",
            tuple(value.fusion_receipt_sha256 for value in bound),
        ),
        clip_receipt_manifest_sha256=_manifest_digest(
            b"phaseset-caption-role-clip-receipt-manifest-v1\x00",
            tuple(value.clip_receipt_sha256 for value in bound),
        ),
        positive_family_manifest_sha256=_manifest_digest(
            b"phaseset-caption-role-positive-family-manifest-v1\x00",
            tuple(value.family.hex() for value in bound),
        ),
        window_count=len(bound),
        caption_count=len(all_caption_commitments),
        microbatch_count=microbatch_count,
    )
    result.canonical_json_bytes()
    return result


__all__ = [
    "ActorCaptionContentSummary",
    "CANDIDATE_POLICY_ID",
    "CAPTION_ROLES",
    "CaptionRoleProvenanceError",
    "CaptionRoleValidationReceipt",
    "HolisticContextContentSummary",
    "INSPECTION_SEED",
    "STATUS",
    "WindowFusionHandoff",
    "WindowManifestContentSummary",
    "WindowManifestRow",
    "validate_candidate_machine_fused_window_training",
]
