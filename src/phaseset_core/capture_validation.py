"""Capture-level validation seam for already prepared PhaseSet windows.

This module composes the registered motion-system interfaces, frozen CLIP text
batches, and the separately audited fixed-tree capture pooling kernel.  It does
not select windows or captions, load checkpoints, qualify weights, or expose a
test split.  Capture and window commitments are source identities; actor-group
commitments remain participant-lineage metadata and are never substituted for
window identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import torch
from torch import Tensor

from .capture_pooling import (
    BAND_COUNT,
    EMBEDDING_DIM,
    CapturePoolingError,
    CapturePoolingResourceLimit,
    CaptureWindowPlan,
    pool_capture_windows,
)
from .contracts import PreparedGroupBatch, validate_prepared_group_batch
from .evaluation import (
    RetrievalDataset,
    capture_r1_contributions,
    validate_retrieval_dataset,
)
from .frozen_clip_text import (
    FrozenClipTextBatch,
    FrozenClipTextReceipt,
    FrozenClipTextRowSelectionReceipt,
)
from .objectives import variable_positive_symmetric_infonce
from .periodic_descriptor_cache_v2 import DescriptorWindowContext
from .training import (
    BaseRetrievalSystem,
    ResidualRetrievalSystem,
    TrainingConfig,
    TrainingRuntimeError,
    _assert_frozen_numerical_runtime,
    _autocast_context,
    _frozen_numerical_runtime,
    resolve_precision,
)

if TYPE_CHECKING:
    from .periodic_capture_training_cache import (
        PeriodicDescriptorCaptureTrainingPlan,
    )


STATUS: Final = "CAPTURE_VALIDATION_SEAM_AUTHORITY0"
SCHEMA: Final = "phaseset-capture-validation-source-v1"
DEFAULT_MAX_ENCODED_BYTES: Final = 256 * 1024 * 1024
DEFAULT_MAX_CUDA_STATIC_TENSOR_BYTES: Final = 2 * 1024 * 1024 * 1024


class CaptureValidationError(ValueError):
    """The supplied validation census or runtime result is inconsistent."""


class CaptureValidationResourceLimit(CaptureValidationError):
    """A declared resource budget was exceeded without sampling the census."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise CaptureValidationError(f"{label} must be exact built-in bytes[32]")
    return value


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptureValidationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_json_bytes(value: object) -> bytes:
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


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(value.tobytes(order="C"))


def _tensor_bytes(value: Tensor) -> bytes:
    return value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes(order="C")


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise CaptureValidationError(f"{label} must be an exact positive int")
    return value


def _validate_text_batch(
    value: object,
    *,
    allow_row_selection: bool = False,
) -> tuple[Tensor, tuple[bytes, ...], str]:
    """Rebind a sealed CLIP batch to its receipt and return a CPU snapshot."""

    if type(value) is not FrozenClipTextBatch:
        raise CaptureValidationError("holistic_text must be exact FrozenClipTextBatch")
    embeddings = value.embeddings
    lineage = value.caption_commitments
    receipt = value.receipt
    if type(embeddings) is not Tensor:
        raise CaptureValidationError("frozen text embeddings must be exact torch.Tensor")
    if (
        embeddings.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or embeddings.ndim != 2
        or tuple(embeddings.shape)[1:] != (EMBEDDING_DIM,)
        or int(embeddings.shape[0]) < 1
        or embeddings.requires_grad
        or not embeddings.is_contiguous()
        or not bool(torch.isfinite(embeddings).all().item())
    ):
        raise CaptureValidationError(
            "frozen text must be finite contiguous non-grad CPU float32 [Q,512]"
        )
    count = int(embeddings.shape[0])
    if type(lineage) is not tuple or len(lineage) != count:
        raise CaptureValidationError("frozen text lineage does not match its rows")
    commitments = tuple(
        _raw32(item, f"caption_commitments[{index}]") for index, item in enumerate(lineage)
    )
    if len(set(commitments)) != count:
        raise CaptureValidationError("caption commitments must be unique within a capture")
    selected = type(receipt) is FrozenClipTextRowSelectionReceipt
    if selected:
        if not allow_row_selection or (
            receipt.schema != "phaseset-frozen-clip-row-selection-v1"
            or receipt.status != "ROW_SELECTION_AUTHORITY0"
        ):
            raise CaptureValidationError("derived frozen text selection is not admitted here")
        origin = receipt.origin_receipt
        indices = receipt.selected_indices
        if (
            type(origin) is not FrozenClipTextReceipt
            or origin.schema != "phaseset-frozen-clip-text-receipt-v1"
            or origin.status != "ENCODED_AUTHORITY0"
            or type(indices) is not tuple
            or len(indices) != count
            or any(
                type(index) is not int or not 0 <= index < len(origin.caption_rows)
                for index in indices
            )
            or len(set(indices)) != count
            or receipt.caption_rows
            != tuple(
                (position, *origin.caption_rows[index][1:])
                for position, index in enumerate(indices)
            )
        ):
            raise CaptureValidationError("derived frozen text selection ancestry differs")
    else:
        if type(receipt) is not FrozenClipTextReceipt:
            raise CaptureValidationError("frozen text receipt type is invalid")
        if receipt.schema != "phaseset-frozen-clip-text-receipt-v1":
            raise CaptureValidationError("frozen text receipt schema is invalid")
        if receipt.status != "ENCODED_AUTHORITY0":
            raise CaptureValidationError("frozen text receipt status is invalid")
    raw = _tensor_bytes(embeddings)
    if (
        receipt.output_shape != tuple(embeddings.shape)
        or receipt.output_stride != tuple(embeddings.stride())
        or receipt.output_bytes != len(raw)
        or receipt.output_sha256 != _sha256(raw)
    ):
        raise CaptureValidationError("frozen text output differs from its receipt")
    if not selected and (type(receipt.batch_size) is not int or receipt.batch_size < 1):
        raise CaptureValidationError("frozen text receipt batch size is invalid")
    if type(receipt.caption_rows) is not tuple or len(receipt.caption_rows) != count:
        raise CaptureValidationError("frozen text receipt caption census is invalid")
    for index, (commitment, row) in enumerate(zip(commitments, receipt.caption_rows, strict=True)):
        if type(row) is not tuple or len(row) != 9:
            raise CaptureValidationError("frozen text receipt caption row is invalid")
        if row[0] != index or row[1] != commitment.hex():
            raise CaptureValidationError("frozen text receipt lineage is reordered")
        row_raw = _tensor_bytes(embeddings[index : index + 1].contiguous())
        if row[8] != _sha256(row_raw):
            raise CaptureValidationError("frozen text receipt row digest mismatch")
    if not selected and (type(receipt.chunk_ranges) is not tuple or not receipt.chunk_ranges):
        raise CaptureValidationError("frozen text receipt chunk census is invalid")
    position = 0
    for chunk in () if selected else receipt.chunk_ranges:
        if (
            type(chunk) is not tuple
            or len(chunk) != 2
            or type(chunk[0]) is not int
            or type(chunk[1]) is not int
            or chunk[0] != position
            or not chunk[0] < chunk[1] <= count
            or chunk[1] - chunk[0] > receipt.batch_size
        ):
            raise CaptureValidationError("frozen text receipt chunks do not partition rows")
        position = chunk[1]
    if not selected and position != count:
        raise CaptureValidationError("frozen text receipt chunks do not cover all rows")
    try:
        receipt_sha256 = _lower_sha256(receipt.sha256, "frozen text receipt SHA-256")
    except (TypeError, ValueError, OverflowError) as error:
        raise CaptureValidationError("frozen text receipt is not canonicalizable") from error
    return embeddings.detach().clone().contiguous(), commitments, receipt_sha256


@dataclass(frozen=True, slots=True, repr=False)
class CaptureDescriptorWindowSource:
    """Authenticated storage identities needed to resolve one descriptor row.

    These values are integrity observations, not dataset or execution
    authority.  The v2 storage loader populates them from bytes admitted under
    its externally expected manifest digest.
    """

    storage_manifest_sha256: str
    source_window_npz_sha256: str
    window_ordinal: int

    def __post_init__(self) -> None:
        storage = _lower_sha256(
            self.storage_manifest_sha256,
            "storage_manifest_sha256",
        )
        source = _lower_sha256(
            self.source_window_npz_sha256,
            "source_window_npz_sha256",
        )
        if type(self.window_ordinal) is not int or not 0 <= self.window_ordinal < 2**64:
            raise CaptureValidationError("window_ordinal must be an exact uint64")
        object.__setattr__(self, "storage_manifest_sha256", storage)
        object.__setattr__(self, "source_window_npz_sha256", source)


@dataclass(frozen=True, slots=True, repr=False)
class CaptureValidationWindow:
    """One actual accepted-window identity and its B=1 prepared motion row."""

    window_commitment: bytes
    groups: PreparedGroupBatch = field(repr=False)
    descriptor_source: CaptureDescriptorWindowSource | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        window = _raw32(self.window_commitment, "window_commitment")
        try:
            groups = validate_prepared_group_batch(self.groups)
        except (TypeError, ValueError) as error:
            raise CaptureValidationError("window prepared group is invalid") from error
        if groups.batch_size != 1:
            raise CaptureValidationError("capture validation encodes exactly B=1 per window")
        if groups.skeletons.shape[2] != 200 or not bool(groups.frame_mask.all()):
            raise CaptureValidationError("capture validation requires complete 200-frame windows")
        descriptor_source = self.descriptor_source
        if (
            descriptor_source is not None
            and type(descriptor_source) is not CaptureDescriptorWindowSource
        ):
            raise CaptureValidationError(
                "descriptor_source must be None or exact CaptureDescriptorWindowSource"
            )
        object.__setattr__(self, "window_commitment", window)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "descriptor_source", descriptor_source)

    def descriptor_context(self, *, seed: int) -> DescriptorWindowContext:
        """Return the formal val context for an authenticated v2 storage row."""

        if type(seed) is not int or not 0 <= seed < 2**64:
            raise CaptureValidationError("descriptor seed must be an exact uint64")
        source = self.descriptor_source
        if type(source) is not CaptureDescriptorWindowSource:
            raise CaptureValidationError(
                "capture window has no authenticated descriptor source lineage"
            )
        try:
            return DescriptorWindowContext.from_yaw(
                prepared_manifest_sha256=source.storage_manifest_sha256,
                source_batch_sha256=source.source_window_npz_sha256,
                window_sha256=self.window_commitment.hex(),
                window_ordinal=source.window_ordinal,
                split="val",
                seed=seed,
                epoch=0,
                augmentation_yaw=0.0,
            )
        except (TypeError, ValueError) as error:
            raise CaptureValidationError(
                "capture descriptor context could not be constructed"
            ) from error

    def __repr__(self) -> str:
        return "CaptureValidationWindow(B=1, <identity/features hidden>)"


@dataclass(frozen=True, slots=True, repr=False)
class CaptureValidationCapture:
    """One complete capture plan, all accepted windows, and holistic text rows.

    ``plan.capture_commitment`` is the source/capture identity.  The CLIP type
    and receipt prove an exact frozen embedding batch, not that its unseen text
    is the official holistic annotation; the caller must bind that fact in the
    supplied source manifest.
    """

    plan: CaptureWindowPlan
    windows: tuple[CaptureValidationWindow, ...] = field(repr=False)
    holistic_text: FrozenClipTextBatch = field(repr=False)
    component_label: str

    def __post_init__(self) -> None:
        if type(self.plan) is not CaptureWindowPlan:
            raise CaptureValidationError("plan must be exact CaptureWindowPlan")
        try:
            plan = CaptureWindowPlan(
                self.plan.capture_commitment,
                self.plan.window_commitments,
                self.plan.source_start_frames,
            )
        except (TypeError, ValueError) as error:
            raise CaptureValidationError("capture window plan is invalid") from error
        if type(self.windows) is not tuple or not self.windows:
            raise CaptureValidationError("capture windows must be a nonempty tuple")
        rebuilt: list[CaptureValidationWindow] = []
        for value in self.windows:
            if type(value) is not CaptureValidationWindow:
                raise CaptureValidationError("capture window type is invalid")
            rebuilt.append(
                CaptureValidationWindow(
                    value.window_commitment,
                    value.groups,
                    value.descriptor_source,
                )
            )
        by_identity = {item.window_commitment: item for item in rebuilt}
        if (
            len(by_identity) != len(rebuilt)
            or len(rebuilt) != len(plan.window_commitments)
            or set(by_identity) != set(plan.window_commitments)
        ):
            raise CaptureValidationError(
                "capture windows must exactly equal the complete accepted-window plan"
            )
        canonical = tuple(by_identity[value] for value in plan.window_commitments)
        reference = canonical[0].groups
        reference_actors = frozenset(
            item for item in reference.actor_commitments[0] if item is not None
        )
        reference_group = reference.group_commitments[0]
        reference_k = reference.actor_counts[0]
        for window in canonical[1:]:
            groups = window.groups
            actors = frozenset(item for item in groups.actor_commitments[0] if item is not None)
            if (
                actors != reference_actors
                or groups.group_commitments[0] != reference_group
                or groups.actor_counts[0] != reference_k
            ):
                raise CaptureValidationError(
                    "all windows in a capture must preserve valid actor set, group, and K"
                )
        _validate_text_batch(self.holistic_text)
        if type(self.component_label) is not str or not self.component_label:
            raise CaptureValidationError("component_label must be a nonempty exact string")
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "windows", canonical)

    @property
    def capture_commitment(self) -> bytes:
        return self.plan.capture_commitment

    @property
    def group_commitment(self) -> bytes:
        return self.windows[0].groups.group_commitments[0]

    @property
    def actor_commitment_set(self) -> frozenset[bytes]:
        return frozenset(
            item for item in self.windows[0].groups.actor_commitments[0] if item is not None
        )

    @property
    def group_size(self) -> int:
        return self.windows[0].groups.actor_counts[0]

    def __repr__(self) -> str:
        return (
            f"CaptureValidationCapture(windows={len(self.windows)}, "
            f"captions={len(self.holistic_text.caption_commitments)}, "
            "<identities/features hidden>)"
        )


def _window_census_row(
    window: CaptureValidationWindow,
    source_start_frame: int,
) -> dict[str, object]:
    groups = window.groups
    return {
        "actor_commitments": sorted(
            item.hex() for item in groups.actor_commitments[0] if item is not None
        ),
        "actor_mask_sha256": _array_sha256(groups.actor_mask),
        "frame_mask_sha256": _array_sha256(groups.frame_mask),
        "group_commitment": groups.group_commitments[0].hex(),
        "group_size": groups.actor_counts[0],
        "skeletons_float32_sha256": _array_sha256(groups.skeletons),
        "source_start_frame": source_start_frame,
        "track_mask_sha256": _array_sha256(groups.track_mask),
        "window_commitment": window.window_commitment.hex(),
    }


def _capture_census_row(capture: CaptureValidationCapture) -> dict[str, object]:
    _, captions, receipt_sha256 = _validate_text_batch(capture.holistic_text)
    return {
        "caption_commitments": [value.hex() for value in sorted(captions)],
        "capture_commitment": capture.capture_commitment.hex(),
        "component_label": capture.component_label,
        "frozen_text_receipt_sha256": receipt_sha256,
        "group_commitment": capture.group_commitment.hex(),
        "group_size": capture.group_size,
        "plan_sha256": capture.plan.sha256,
        "windows": [
            _window_census_row(window, start)
            for window, start in zip(
                capture.windows,
                capture.plan.source_start_frames,
                strict=True,
            )
        ],
    }


@dataclass(frozen=True, slots=True, repr=False)
class CaptureValidationSource:
    """Validated val-only source with a complete canonical capture census."""

    split: Literal["val"]
    manifest_sha256: str
    captures: tuple[CaptureValidationCapture, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.split) is not str or self.split != "val":
            raise CaptureValidationError("capture validation source must be exactly split='val'")
        manifest = _lower_sha256(self.manifest_sha256, "manifest_sha256")
        if type(self.captures) is not tuple or len(self.captures) < 2:
            raise CaptureValidationError("capture validation needs at least two captures")
        rebuilt: list[CaptureValidationCapture] = []
        for value in self.captures:
            if type(value) is not CaptureValidationCapture:
                raise CaptureValidationError("capture source row type is invalid")
            rebuilt.append(
                CaptureValidationCapture(
                    value.plan,
                    value.windows,
                    value.holistic_text,
                    value.component_label,
                )
            )
        ordered = tuple(sorted(rebuilt, key=lambda value: value.capture_commitment))
        capture_ids = [value.capture_commitment for value in ordered]
        if len(set(capture_ids)) != len(capture_ids):
            raise CaptureValidationError("capture identities must be globally unique")
        seen_windows: set[bytes] = set()
        seen_captions: set[bytes] = set()
        descriptor_rows: list[CaptureDescriptorWindowSource] = []
        window_count = 0
        for capture in ordered:
            windows = set(capture.plan.window_commitments)
            window_count += len(capture.windows)
            descriptor_rows.extend(
                window.descriptor_source
                for window in capture.windows
                if window.descriptor_source is not None
            )
            _, captions, _ = _validate_text_batch(capture.holistic_text)
            caption_set = set(captions)
            if seen_windows.intersection(windows):
                raise CaptureValidationError("window identities must be globally unique")
            if seen_captions.intersection(caption_set):
                raise CaptureValidationError("caption identities must be globally unique")
            seen_windows.update(windows)
            seen_captions.update(caption_set)
        if descriptor_rows:
            if len(descriptor_rows) != window_count:
                raise CaptureValidationError("capture source cannot mix v1 and v2 window lineage")
            descriptor_manifest = descriptor_rows[0].storage_manifest_sha256
            if any(row.storage_manifest_sha256 != descriptor_manifest for row in descriptor_rows):
                raise CaptureValidationError(
                    "capture descriptor rows must share one storage manifest"
                )
            if tuple(row.window_ordinal for row in descriptor_rows) != tuple(range(window_count)):
                raise CaptureValidationError(
                    "capture descriptor ordinals must equal canonical global order"
                )
        object.__setattr__(self, "manifest_sha256", manifest)
        object.__setattr__(self, "captures", ordered)

    @property
    def census_sha256(self) -> str:
        value = {
            "authority": 0,
            "captures": [_capture_census_row(capture) for capture in self.captures],
            "manifest_sha256": self.manifest_sha256,
            "schema": SCHEMA,
            "split": self.split,
        }
        return _sha256(_canonical_json_bytes(value))

    def __repr__(self) -> str:
        return (
            f"CaptureValidationSource(split='val', captures={len(self.captures)}, "
            f"census_sha256={self.census_sha256!r}, <identities/features hidden>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class CaptureValidationResult:
    """Actual capture-gallery scores and exact endpoint accounting."""

    dataset: RetrievalDataset = field(repr=False)
    text_to_motion_capture_r1: Fraction
    motion_to_text_capture_r1: Fraction
    primary_capture_r1: Fraction
    loss: float
    source_census_sha256: str
    scores_float64_sha256: str
    system_id: str
    stage: Literal["base", "residual"]
    device: str
    precision_mode: Literal["FP32", "BF16"]
    encoded_bytes: int
    cuda_static_tensor_bytes: int | None
    status: str = STATUS

    def __repr__(self) -> str:
        return (
            f"CaptureValidationResult(stage={self.stage!r}, system_id={self.system_id!r}, "
            f"shape={self.dataset.scores.shape}, primary={self.primary_capture_r1}, "
            f"source_census_sha256={self.source_census_sha256!r})"
        )


def _snapshot_source(value: object) -> CaptureValidationSource:
    if type(value) is not CaptureValidationSource:
        raise CaptureValidationError("source must be exact CaptureValidationSource")
    return CaptureValidationSource(value.split, value.manifest_sha256, value.captures)


def _system_tensor_bytes(system: torch.nn.Module) -> int:
    return sum(
        int(value.numel()) * int(value.element_size())
        for value in (*tuple(system.parameters()), *tuple(system.buffers()))
    )


def _require_system_device(system: torch.nn.Module, device: torch.device) -> None:
    tensors = (*tuple(system.parameters()), *tuple(system.buffers()))
    if not tensors or any(value.device != device for value in tensors):
        raise CaptureValidationError(
            "system parameters and buffers must already reside on the configured device"
        )


def _concrete_device(specification: str) -> torch.device:
    device = torch.device(specification)
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _expected_encoded_bytes(
    source: CaptureValidationSource,
    *,
    residual: bool,
) -> int:
    windows = sum(len(capture.windows) for capture in source.captures)
    per_window = EMBEDDING_DIM * np.dtype(np.float32).itemsize
    if residual:
        per_window += BAND_COUNT * EMBEDDING_DIM * np.dtype(np.float32).itemsize
        per_window += BAND_COUNT * np.dtype(np.bool_).itemsize
    return windows * per_window


def _cuda_static_bytes(
    system: torch.nn.Module,
    source: CaptureValidationSource,
    *,
    residual: bool,
) -> int:
    capture_count = len(source.captures)
    caption_count = sum(
        len(capture.holistic_text.caption_commitments) for capture in source.captures
    )
    total = _system_tensor_bytes(system)
    total += capture_count * EMBEDDING_DIM * 4
    if residual:
        total += capture_count * BAND_COUNT * EMBEDDING_DIM * 4
        total += capture_count * BAND_COUNT
    total += caption_count * EMBEDDING_DIM * 4
    total += capture_count * caption_count * (4 + 1)
    return total


def _finite_cpu_float32(value: Tensor, shape: tuple[int, ...], label: str) -> np.ndarray:
    if type(value) is not Tensor or value.dtype != torch.float32:
        raise CaptureValidationError(f"{label} must return an exact float32 tensor")
    output = value.detach().to(device="cpu").contiguous()
    if tuple(output.shape) != shape or not bool(torch.isfinite(output).all().item()):
        raise CaptureValidationError(f"{label} output shape or finiteness is invalid")
    return np.array(output.numpy(), dtype=np.float32, copy=True, order="C", subok=False)


def _capture_fractions(
    dataset: RetrievalDataset,
) -> tuple[Fraction, Fraction, Fraction]:
    rows = capture_r1_contributions(dataset)
    count = len(rows)
    text = (
        sum(
            (Fraction(sum(row.text_to_motion_hits), len(row.text_to_motion_hits)) for row in rows),
            start=Fraction(0, 1),
        )
        / count
    )
    motion = (
        sum(
            (Fraction(sum(row.motion_to_text_hits), len(row.motion_to_text_hits)) for row in rows),
            start=Fraction(0, 1),
        )
        / count
    )
    return text, motion, (text + motion) / 2


def run_capture_validation(
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
    config: TrainingConfig,
    source: CaptureValidationSource,
    *,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    max_cuda_static_tensor_bytes: int = DEFAULT_MAX_CUDA_STATIC_TENSOR_BYTES,
) -> CaptureValidationResult:
    """Encode all planned windows, pool captures, and score one full gallery.

    Each accepted window is encoded independently at B=1.  The byte budgets
    reject the complete workload; they never select a subset.  The CUDA budget
    covers model parameters/buffers plus explicit final gallery tensors.  It is
    not a claim about transient workspace or allocator peaks; CUDA OOM is
    translated to ``RESOURCE_LIMIT``.
    """

    return _run_capture_validation_core(
        system,
        config,
        source,
        descriptor_plan=None,
        max_encoded_bytes=max_encoded_bytes,
        max_cuda_static_tensor_bytes=max_cuda_static_tensor_bytes,
    )


def run_capture_validation_cached(
    system: ResidualRetrievalSystem,
    config: TrainingConfig,
    source: CaptureValidationSource,
    descriptor_plan: PeriodicDescriptorCaptureTrainingPlan,
    *,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    max_cuda_static_tensor_bytes: int = DEFAULT_MAX_CUDA_STATIC_TENSOR_BYTES,
) -> CaptureValidationResult:
    """Run the same full-capture oracle using explicit verified descriptor streams."""

    # Local import avoids a module cycle: the plan owns capture-source checks.
    from .periodic_capture_training_cache import (
        PeriodicDescriptorCaptureTrainingPlan,
    )

    if type(system) is not ResidualRetrievalSystem:
        raise CaptureValidationError(
            "cached capture validation requires exact ResidualRetrievalSystem"
        )
    if type(config) is not TrainingConfig:
        raise CaptureValidationError("config must be exact TrainingConfig")
    if type(descriptor_plan) is not PeriodicDescriptorCaptureTrainingPlan:
        raise CaptureValidationError(
            "descriptor_plan must be an admitted exact capture training plan"
        )
    if (
        config.stage != "residual"
        or descriptor_plan.system_id != system.system_id
        or descriptor_plan.config_sha256 != config.sha256
        or descriptor_plan.seed != config.seed
        or descriptor_plan.epochs != config.epochs
        or descriptor_plan.edge_budget != config.edge_budget
    ):
        raise CaptureValidationError(
            "descriptor plan differs from the residual system or training config"
        )
    checked_source = descriptor_plan.validate_capture_validation_source(source)
    return _run_capture_validation_core(
        system,
        config,
        checked_source,
        descriptor_plan=descriptor_plan,
        max_encoded_bytes=max_encoded_bytes,
        max_cuda_static_tensor_bytes=max_cuda_static_tensor_bytes,
    )


def _run_capture_validation_core(
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
    config: TrainingConfig,
    source: CaptureValidationSource,
    *,
    descriptor_plan: PeriodicDescriptorCaptureTrainingPlan | None,
    max_encoded_bytes: int,
    max_cuda_static_tensor_bytes: int,
) -> CaptureValidationResult:
    """Shared full-capture implementation; only periodic descriptor acquisition varies."""

    if not isinstance(system, (BaseRetrievalSystem, ResidualRetrievalSystem)):
        raise CaptureValidationError("system must use a registered retrieval interface")
    if type(config) is not TrainingConfig:
        raise CaptureValidationError("config must be exact TrainingConfig")
    checked_source = _snapshot_source(source)
    encoded_limit = _positive_int(max_encoded_bytes, "max_encoded_bytes")
    cuda_limit = _positive_int(
        max_cuda_static_tensor_bytes,
        "max_cuda_static_tensor_bytes",
    )
    residual = isinstance(system, ResidualRetrievalSystem)
    expected_stage = "residual" if residual else "base"
    if config.stage != expected_stage:
        raise CaptureValidationError("training config stage differs from retrieval system")
    if system.embedding_dim != EMBEDDING_DIM:
        raise CaptureValidationError("capture validation requires registered width 512")
    if type(system.system_id) is not str or not system.system_id:
        raise CaptureValidationError("retrieval system_id is invalid")
    device = _concrete_device(config.device)
    _require_system_device(system, device)
    precision = resolve_precision(config)
    expected_bytes = _expected_encoded_bytes(checked_source, residual=residual)
    if expected_bytes > encoded_limit:
        raise CaptureValidationResourceLimit(
            "RESOURCE_LIMIT: complete encoded-window arrays exceed max_encoded_bytes"
        )
    cuda_static_bytes = None
    if device.type == "cuda":
        cuda_static_bytes = _cuda_static_bytes(
            system,
            checked_source,
            residual=residual,
        )
        if cuda_static_bytes > cuda_limit:
            raise CaptureValidationResourceLimit(
                "RESOURCE_LIMIT: explicit CUDA tensors exceed max_cuda_static_tensor_bytes"
            )
    for capture in checked_source.captures:
        for window in capture.windows:
            k = window.groups.actor_counts[0]
            edges = k * (k - 1) // 2
            if edges > config.edge_budget:
                raise CaptureValidationResourceLimit(
                    "RESOURCE_LIMIT: an accepted window exceeds config.edge_budget"
                )

    modes = tuple((module, bool(module.training)) for module in system.modules())
    pooled_base: list[np.ndarray] = []
    pooled_tokens: list[np.ndarray] = []
    pooled_masks: list[np.ndarray] = []
    text_rows: list[Tensor] = []
    caption_commitments: list[bytes] = []
    positive_motion_indices: list[tuple[int, ...]] = []
    logits: Tensor | None = None
    loss_value: float | None = None
    try:
        with _frozen_numerical_runtime(device), torch.no_grad():
            system.eval()
            for motion_index, capture in enumerate(checked_source.captures):
                base_rows: list[np.ndarray] = []
                token_rows: list[np.ndarray] = []
                mask_rows: list[np.ndarray] = []
                for window_position, window in enumerate(capture.windows):
                    with _autocast_context(precision, device):
                        if residual:
                            assert isinstance(system, ResidualRetrievalSystem)
                            if descriptor_plan is None:
                                tokens, band_mask = system.encode_trainable(window.groups)
                            else:
                                descriptor_stream = descriptor_plan.open_capture_validation_window(
                                    capture,
                                    window_position,
                                )
                                tokens, band_mask = system.encode_trainable_cached(
                                    window.groups,
                                    descriptor_stream,
                                )
                            base = system.encode_frozen_base(window.groups)
                        else:
                            assert isinstance(system, BaseRetrievalSystem)
                            base = system.encode_trainable(window.groups)
                    base_rows.append(
                        _finite_cpu_float32(base, (1, EMBEDDING_DIM), "base encoder")[0]
                    )
                    _assert_frozen_numerical_runtime(device)
                    if residual:
                        token_array = _finite_cpu_float32(
                            tokens,
                            (1, BAND_COUNT, EMBEDDING_DIM),
                            "periodic encoder",
                        )[0]
                        if (
                            type(band_mask) is not Tensor
                            or band_mask.dtype != torch.bool
                            or band_mask.device != device
                            or tuple(band_mask.shape) != (1, BAND_COUNT)
                        ):
                            raise CaptureValidationError(
                                "periodic encoder must return a bool [1,6] device mask"
                            )
                        mask_array = np.array(
                            band_mask.detach().to(device="cpu").contiguous().numpy(),
                            dtype=np.bool_,
                            copy=True,
                            order="C",
                            subok=False,
                        )
                        if mask_array.shape != (1, BAND_COUNT):
                            raise CaptureValidationError(
                                "periodic encoder band mask shape is invalid"
                            )
                        token_rows.append(token_array)
                        mask_rows.append(mask_array[0])
                try:
                    pooled = pool_capture_windows(
                        capture.plan,
                        window_commitments=tuple(
                            window.window_commitment for window in capture.windows
                        ),
                        base_embeddings=np.ascontiguousarray(np.stack(base_rows), dtype=np.float32),
                        tokens=(
                            np.ascontiguousarray(np.stack(token_rows), dtype=np.float32)
                            if residual
                            else None
                        ),
                        band_mask=(
                            np.ascontiguousarray(np.stack(mask_rows), dtype=np.bool_)
                            if residual
                            else None
                        ),
                        max_encoded_bytes=encoded_limit,
                    )
                except CapturePoolingResourceLimit as error:
                    raise CaptureValidationResourceLimit(str(error)) from error
                except CapturePoolingError as error:
                    raise CaptureValidationError(
                        "capture pooling rejected encoder output"
                    ) from error
                pooled_base.append(pooled.base_embedding)
                if residual:
                    if pooled.tokens is None or pooled.band_mask is None:
                        raise CaptureValidationError("residual capture pooling omitted tokens")
                    pooled_tokens.append(pooled.tokens)
                    pooled_masks.append(pooled.band_mask)
                text, commitments, _ = _validate_text_batch(capture.holistic_text)
                order = tuple(sorted(range(len(commitments)), key=lambda i: commitments[i]))
                text_rows.extend(text[index : index + 1] for index in order)
                caption_commitments.extend(commitments[index] for index in order)
                positive_motion_indices.extend((motion_index,) for _ in order)

            motion = (
                torch.from_numpy(np.ascontiguousarray(np.stack(pooled_base), dtype=np.float32))
                .to(device=device)
                .contiguous()
            )
            text_gallery = torch.cat(text_rows).to(device=device).contiguous()
            positive = torch.zeros(
                (len(checked_source.captures), len(caption_commitments)),
                dtype=torch.bool,
                device=device,
            )
            for caption_index, row in enumerate(positive_motion_indices):
                positive[row[0], caption_index] = True
            if residual:
                assert isinstance(system, ResidualRetrievalSystem)
                tokens_gallery = (
                    torch.from_numpy(
                        np.ascontiguousarray(np.stack(pooled_tokens), dtype=np.float32)
                    )
                    .to(device=device)
                    .contiguous()
                )
                masks_gallery = (
                    torch.from_numpy(np.ascontiguousarray(np.stack(pooled_masks), dtype=np.bool_))
                    .to(device=device)
                    .contiguous()
                )
                logits = system.scores(
                    tokens_gallery,
                    masks_gallery,
                    motion,
                    text_gallery,
                )
            else:
                assert isinstance(system, BaseRetrievalSystem)
                logits = system.scores(motion, text_gallery)
            if (
                type(logits) is not Tensor
                or logits.dtype != torch.float32
                or logits.device != device
                or tuple(logits.shape) != (len(checked_source.captures), len(caption_commitments))
                or not logits.is_contiguous()
                or logits.requires_grad
                or not bool(torch.isfinite(logits).all().item())
            ):
                raise CaptureValidationError("full-gallery score tensor is invalid")
            _, _, objective = variable_positive_symmetric_infonce(logits, positive)
            loss_value = float(objective.detach().cpu().item())
            if not np.isfinite(loss_value):
                raise CaptureValidationError("capture-gallery loss is not finite")
            _assert_frozen_numerical_runtime(device)
    except torch.cuda.OutOfMemoryError as error:
        raise CaptureValidationResourceLimit(
            "RESOURCE_LIMIT: CUDA exhausted memory while evaluating the complete census"
        ) from error
    finally:
        for module, mode in modes:
            module.training = mode

    if logits is None or loss_value is None:
        raise TrainingRuntimeError("capture validation did not produce a gallery")
    score_array = np.ascontiguousarray(
        logits.detach().to(device="cpu").numpy(),
        dtype=np.float64,
    )
    dataset = validate_retrieval_dataset(
        RetrievalDataset(
            scores=score_array,
            motion_commitments=tuple(
                capture.capture_commitment for capture in checked_source.captures
            ),
            caption_commitments=tuple(caption_commitments),
            positive_motion_indices=tuple(positive_motion_indices),
            group_sizes=np.ascontiguousarray(
                [capture.group_size for capture in checked_source.captures],
                dtype=np.int64,
            ),
            component_labels=tuple(capture.component_label for capture in checked_source.captures),
        )
    )
    text_r1, motion_r1, primary_r1 = _capture_fractions(dataset)
    return CaptureValidationResult(
        dataset=dataset,
        text_to_motion_capture_r1=text_r1,
        motion_to_text_capture_r1=motion_r1,
        primary_capture_r1=primary_r1,
        loss=loss_value,
        source_census_sha256=checked_source.census_sha256,
        scores_float64_sha256=_array_sha256(dataset.scores),
        system_id=system.system_id,
        stage=expected_stage,
        device=str(device),
        precision_mode=precision.mode,
        encoded_bytes=expected_bytes,
        cuda_static_tensor_bytes=cuda_static_bytes,
    )


# Match the existing formal runtime's live-function drift checks. This is not
# a security boundary against arbitrary Python execution or a rights grant.
_CAPTURE_RUNTIME_BINDINGS = tuple(
    (name, globals()[name])
    for name in (
        "CaptureValidationSource",
        "run_capture_validation",
        "run_capture_validation_cached",
        "_run_capture_validation_core",
        "_snapshot_source",
        "_capture_fractions",
        "pool_capture_windows",
        "capture_r1_contributions",
        "validate_retrieval_dataset",
        "variable_positive_symmetric_infonce",
    )
)


def _verified_capture_validation_bindings():
    for name, expected in _CAPTURE_RUNTIME_BINDINGS:
        if globals().get(name) is not expected:
            raise TrainingRuntimeError(f"capture runtime function identity changed: {name}")
    return _CAPTURE_RUNTIME_BINDINGS[0][1], _CAPTURE_RUNTIME_BINDINGS[1][1]


__all__ = [
    "CaptureDescriptorWindowSource",
    "CaptureValidationCapture",
    "CaptureValidationError",
    "CaptureValidationResourceLimit",
    "CaptureValidationResult",
    "CaptureValidationSource",
    "CaptureValidationWindow",
    "run_capture_validation",
    "run_capture_validation_cached",
]
