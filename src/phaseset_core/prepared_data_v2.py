"""Private-server prepared-data v2 writer and epoch-yaw data source.

This authority-zero module composes already-prepared numeric motion windows
with already-frozen CLIP text batches.  It never locates a dataset, reads raw
caption text, loads a body model, grants rights, or executes a text model.

The ten-array NPZ payload is an unaugmented canonical float32 window.  A v2
split manifest binds one independent global uint64 window ordinal per motion.
The training source rereads the authenticated NPZ for every epoch and applies
the existing group_yaw_v2 angle once from (seed, epoch, ordinal), then attaches
the matching immutable descriptor contexts to that exact returned batch.
Validation is never augmented and uses canonical epoch zero/+0 yaw contexts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Final, Literal, Mapping
import zipfile

import numpy as np
import torch

from phaseset_core.contracts import PreparedGroupBatch
from phaseset_core.frozen_clip_text import (
    MAX_BATCH_SIZE as CLIP_MAX_BATCH_SIZE,
    FrozenClipTextBatch,
    FrozenClipTextReceipt,
)
from phaseset_core.pipeline import (
    PreparedGroupSample,
    collate_group_samples,
    deterministic_group_yaw,
    edge_budget_batches,
)
from phaseset_core.periodic_descriptor_cache_v2 import DescriptorWindowContext
from phaseset_core.training import RetrievalTrainingBatch


AUTHORITY: Final = 0
PREPARED_INDEX_V2_SCHEMA: Final = "phaseset-prepared-index-v2"
PREPARED_SPLIT_V2_SCHEMA: Final = "phaseset-prepared-training-split-v2"
TRAIN_AUGMENTATION: Final = "group_yaw_v2"
VALIDATION_AUGMENTATION: Final = "none"
EMBEDDING_DIM: Final = 512
TARGET_FRAMES: Final = 200
BODY_JOINTS: Final = 22
MAX_JSON_BYTES: Final = 64 * 1024 * 1024
DEFAULT_MAX_BATCH_BYTES: Final = 2**31
DEFAULT_MAX_DECODED_BATCH_BYTES: Final = 2**31
ABSOLUTE_MAX_DECODED_BATCH_BYTES: Final = 8 * 2**30
NPZ_KEY_ORDER: Final = (
    "skeletons",
    "actor_mask",
    "frame_mask",
    "track_mask",
    "actor_commitments",
    "group_commitments",
    "text_embeddings",
    "motion_positive_ids",
    "text_positive_ids",
    "text_commitments",
)
NPZ_KEYS: Final = frozenset(NPZ_KEY_ORDER)
NPZ_DTYPES: Final = {
    "skeletons": np.dtype(np.float32),
    "actor_mask": np.dtype(np.bool_),
    "frame_mask": np.dtype(np.bool_),
    "track_mask": np.dtype(np.bool_),
    "actor_commitments": np.dtype(np.uint8),
    "group_commitments": np.dtype(np.uint8),
    "text_embeddings": np.dtype(np.float32),
    "motion_positive_ids": np.dtype(np.uint8),
    "text_positive_ids": np.dtype(np.uint8),
    "text_commitments": np.dtype(np.uint8),
}
Split = Literal["train", "val"]


class PreparedDataV2Error(ValueError):
    """The typed input, immutable artifact, or v2 manifest is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreparedDataV2Error(f"{label} must be lowercase SHA-256 hex")
    return value


def _bytes32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise PreparedDataV2Error(f"{label} must be exact bytes[32]")
    return value


def _uint64(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise PreparedDataV2Error(f"{label} must be an exact uint64")
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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise PreparedDataV2Error("JSON contains a duplicate key")
        output[key] = value
    return output


def _reject_constant(value: str) -> object:
    raise PreparedDataV2Error(f"JSON contains forbidden constant {value}")


def _read_bounded(path: Path, *, maximum: int, label: str) -> bytes:
    if type(maximum) is not int or maximum < 1:
        raise PreparedDataV2Error("byte maximum must be a positive exact int")
    if path.is_symlink() or not path.is_file():
        raise PreparedDataV2Error(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        raise PreparedDataV2Error(f"{label} is empty or exceeds its byte bound")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) != size or len(raw) > maximum:
        raise PreparedDataV2Error(f"{label} changed while it was read")
    return raw


def _load_json(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    raw = _read_bounded(path, maximum=MAX_JSON_BYTES, label=label)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except PreparedDataV2Error:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreparedDataV2Error(f"{label} must be strict ASCII JSON") from error
    if type(value) is not dict:
        raise PreparedDataV2Error(f"{label} must be a JSON object")
    if raw != _canonical_json_bytes(value):
        raise PreparedDataV2Error(f"{label} must use canonical JSON bytes")
    return raw, value


def _closed(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise PreparedDataV2Error(f"{label} keys are not closed")


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise PreparedDataV2Error(f"{label} may not traverse a symlink")


def _resolve_under(root: Path, value: object, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value or Path(value).is_absolute():
        raise PreparedDataV2Error(f"{label} must be a relative path")
    unresolved = root / value
    _reject_symlink_components(unresolved, label)
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise PreparedDataV2Error(f"{label} escapes its artifact root") from error
    if not candidate.is_file():
        raise PreparedDataV2Error(f"{label} must be an existing regular file")
    return candidate


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True, slots=True, repr=False)
class PreparedMotionTextBlock:
    """One frozen CLIP batch partitioned across one or more motion windows.

    ``text_counts[i]`` assigns a nonempty consecutive text-row range to
    ``samples[i]``.  This supports official holistic text (Q=1), machine-fused
    variants (often Q=2), and other already-authorized variable-positive text
    without fixing a caption count in the writer.
    """

    samples: tuple[PreparedGroupSample, ...] = field(repr=False)
    text_batch: FrozenClipTextBatch = field(repr=False)
    text_counts: tuple[int, ...]
    window_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.samples) is not tuple or not self.samples:
            raise PreparedDataV2Error("samples must be one nonempty exact tuple")
        if any(type(sample) is not PreparedGroupSample for sample in self.samples):
            raise PreparedDataV2Error("samples must contain exact PreparedGroupSample values")
        for sample in self.samples:
            if struct.pack(">d", float(sample.augmentation_yaw)) != b"\x00" * 8:
                raise PreparedDataV2Error("stored motion samples must have exact +0 yaw")
        if type(self.text_batch) is not FrozenClipTextBatch:
            raise PreparedDataV2Error("text_batch must be exact FrozenClipTextBatch")
        if (
            type(self.text_counts) is not tuple
            or len(self.text_counts) != len(self.samples)
            or any(type(count) is not int or count < 1 for count in self.text_counts)
        ):
            raise PreparedDataV2Error("text_counts must assign at least one row per motion")
        if type(self.window_ordinals) is not tuple or len(self.window_ordinals) != len(
            self.samples
        ):
            raise PreparedDataV2Error("window_ordinals must cover every motion")
        ordinals = tuple(
            _uint64(value, f"window_ordinals[{index}]")
            for index, value in enumerate(self.window_ordinals)
        )
        if len(set(ordinals)) != len(ordinals):
            raise PreparedDataV2Error("window ordinals must be unique within a block")
        windows = tuple(sample.window_sha256 for sample in self.samples)
        if len(set(windows)) != len(windows):
            raise PreparedDataV2Error("motion windows must be unique within a block")

        embeddings = self.text_batch.embeddings
        lineage = self.text_batch.caption_commitments
        receipt = self.text_batch.receipt
        if (
            type(embeddings) is not torch.Tensor
            or embeddings.dtype != torch.float32
            or embeddings.device.type != "cpu"
            or embeddings.requires_grad
            or embeddings.ndim != 2
            or embeddings.shape[1] != EMBEDDING_DIM
            or not embeddings.is_contiguous()
            or not bool(torch.isfinite(embeddings).all().item())
        ):
            raise PreparedDataV2Error("frozen text embedding contract is invalid")
        if sum(self.text_counts) != int(embeddings.shape[0]):
            raise PreparedDataV2Error("text_counts must consume every frozen text row")
        if (
            type(lineage) is not tuple
            or len(lineage) != int(embeddings.shape[0])
            or any(type(value) is not bytes or len(value) != 32 for value in lineage)
            or len(set(lineage)) != len(lineage)
        ):
            raise PreparedDataV2Error("frozen text lineage is invalid")
        if type(receipt) is not FrozenClipTextReceipt:
            raise PreparedDataV2Error("frozen text receipt type is invalid")
        output = _tensor_bytes(embeddings)
        rows = receipt.caption_rows
        if (
            type(rows) is not tuple
            or len(rows) != len(lineage)
            or any(type(row) is not tuple or len(row) != 9 for row in rows)
        ):
            raise PreparedDataV2Error("frozen text receipt caption census is invalid")
        receipt_lineage = tuple(row[1] for row in rows)
        row_embedding_digests = tuple(row[8] for row in rows)
        expected_row_digests = tuple(
            _sha256_bytes(_tensor_bytes(embeddings[index : index + 1]))
            for index in range(len(lineage))
        )
        # The receipt records configured inference chunk size, not total rows.
        if (
            type(receipt.batch_size) is not int
            or not 1 <= receipt.batch_size <= CLIP_MAX_BATCH_SIZE
            or type(receipt.chunk_ranges) is not tuple
            or any(
                type(chunk) is not tuple
                or len(chunk) != 2
                or any(type(index) is not int for index in chunk)
                for chunk in receipt.chunk_ranges
            )
        ):
            raise PreparedDataV2Error("frozen text receipt chunk census is invalid")
        expected_chunks = tuple(
            (start, min(start + receipt.batch_size, len(lineage)))
            for start in range(0, len(lineage), receipt.batch_size)
        )
        if receipt.chunk_ranges != expected_chunks:
            raise PreparedDataV2Error("frozen text receipt chunk census is incomplete")
        if (
            tuple(row[0] for row in rows) != tuple(range(len(lineage)))
            or receipt.output_shape != tuple(embeddings.shape)
            or receipt.output_stride != tuple(embeddings.stride())
            or receipt.output_bytes != len(output)
            or receipt.output_sha256 != _sha256_bytes(output)
            or receipt_lineage != tuple(value.hex() for value in lineage)
            or row_embedding_digests != expected_row_digests
        ):
            raise PreparedDataV2Error("frozen text receipt does not bind the supplied batch")

    def __repr__(self) -> str:
        return (
            f"PreparedMotionTextBlock(motions={len(self.samples)}, "
            f"texts={sum(self.text_counts)}, <numeric/text hidden>)"
        )


@dataclass(frozen=True, slots=True)
class PreparedDataBuildResult:
    """Authority-zero identity of one newly written artifact tree."""

    root: Path
    index_path: Path
    index_sha256: str
    census_path: Path
    census_sha256: str
    train_manifest_sha256: str
    val_manifest_sha256: str
    batch_count: int
    motion_count: int
    text_count: int
    authority: int = AUTHORITY


@dataclass(frozen=True, slots=True, repr=False)
class _MotionTextRow:
    sample: PreparedGroupSample = field(repr=False)
    embeddings: np.ndarray = field(repr=False)
    text_commitments: tuple[bytes, ...] = field(repr=False)
    positive_family: bytes = field(repr=False)
    window_ordinal: int
    clip_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _ArtifactEntry:
    path: Path
    sha256: str
    window_ordinals: tuple[int, ...]


def _expand_blocks(blocks: object) -> tuple[_MotionTextRow, ...]:
    if type(blocks) is not tuple or not blocks:
        raise PreparedDataV2Error("each split requires a nonempty exact block tuple")
    if any(type(block) is not PreparedMotionTextBlock for block in blocks):
        raise PreparedDataV2Error("blocks must contain exact PreparedMotionTextBlock values")
    rows: list[_MotionTextRow] = []
    seen_windows: set[str] = set()
    seen_ordinals: set[int] = set()
    seen_text: set[bytes] = set()
    for block in blocks:
        embeddings = block.text_batch.embeddings.detach().cpu().contiguous().numpy()
        commitments = block.text_batch.caption_commitments
        start = 0
        for sample, count, ordinal in zip(
            block.samples,
            block.text_counts,
            block.window_ordinals,
            strict=True,
        ):
            stop = start + count
            if sample.window_sha256 in seen_windows:
                raise PreparedDataV2Error("a split repeats a motion window")
            if ordinal in seen_ordinals:
                raise PreparedDataV2Error("a split repeats a global window ordinal")
            selected_commitments = tuple(commitments[start:stop])
            if seen_text.intersection(selected_commitments):
                raise PreparedDataV2Error("a split repeats a text commitment")
            family = bytes.fromhex(sample.window_sha256)
            rows.append(
                _MotionTextRow(
                    sample=sample,
                    embeddings=np.ascontiguousarray(embeddings[start:stop]),
                    text_commitments=selected_commitments,
                    positive_family=family,
                    window_ordinal=ordinal,
                    clip_receipt_sha256=block.text_batch.receipt.sha256,
                )
            )
            seen_windows.add(sample.window_sha256)
            seen_ordinals.add(ordinal)
            seen_text.update(selected_commitments)
            start = stop
        if start != len(commitments):
            raise AssertionError("text partition did not consume its frozen batch")
    return tuple(rows)


def _pack32(values: tuple[bytes, ...]) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray([list(_bytes32(value, "commitment")) for value in values], dtype=np.uint8)
    )


def _batch_arrays(rows: tuple[_MotionTextRow, ...]) -> dict[str, np.ndarray]:
    samples = tuple(row.sample for row in rows)
    groups = collate_group_samples(samples)
    batch_size, padded_actors = groups.actor_mask.shape
    actor_commitments = np.zeros((batch_size, padded_actors, 32), dtype=np.uint8)
    for batch_index, commitments in enumerate(groups.actor_commitments):
        for actor_index, commitment in enumerate(commitments):
            if commitment is not None:
                actor_commitments[batch_index, actor_index] = np.frombuffer(
                    commitment, dtype=np.uint8
                )
    text_embeddings = np.ascontiguousarray(
        np.concatenate(tuple(row.embeddings for row in rows), axis=0),
        dtype=np.float32,
    )
    motion_positive_ids = _pack32(tuple(row.positive_family for row in rows))
    text_positive_ids = _pack32(
        tuple(
            row.positive_family
            for row in rows
            for _ in range(len(row.text_commitments))
        )
    )
    text_commitments = _pack32(
        tuple(value for row in rows for value in row.text_commitments)
    )
    arrays = {
        "skeletons": np.ascontiguousarray(groups.skeletons, dtype=np.float32),
        "actor_mask": np.ascontiguousarray(groups.actor_mask, dtype=np.bool_),
        "frame_mask": np.ascontiguousarray(groups.frame_mask, dtype=np.bool_),
        "track_mask": np.ascontiguousarray(groups.track_mask, dtype=np.bool_),
        "actor_commitments": actor_commitments,
        "group_commitments": _pack32(groups.group_commitments),
        "text_embeddings": text_embeddings,
        "motion_positive_ids": motion_positive_ids,
        "text_positive_ids": text_positive_ids,
        "text_commitments": text_commitments,
    }
    _arrays_to_training_batch(arrays, split="train")
    return arrays


def _decoded_nbytes(arrays: Mapping[str, np.ndarray]) -> int:
    return sum(int(value.nbytes) for value in arrays.values())


def _write_npz_once(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    max_decoded_bytes: int,
) -> str:
    if set(arrays) != NPZ_KEYS:
        raise AssertionError("internal NPZ key census changed")
    if _decoded_nbytes(arrays) > max_decoded_bytes:
        raise PreparedDataV2Error("decoded prepared batch exceeds its byte bound")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w+b", closefd=True) as output:
            with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
                for name in NPZ_KEY_ORDER:
                    info = zipfile.ZipInfo(
                        filename=name + ".npy",
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    with archive.open(info, mode="w", force_zip64=True) as member:
                        np.lib.format.write_array(
                            member,
                            np.ascontiguousarray(arrays[name]),
                            version=(1, 0),
                            allow_pickle=False,
                        )
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    os.chmod(path, 0o400)
    return _sha256_file(path)


def _write_json_once(path: Path, value: object) -> str:
    raw = _canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o400)
    return _sha256_bytes(raw)


def _validate_resource_bounds(max_batch_bytes: object, max_decoded_bytes: object) -> None:
    if type(max_batch_bytes) is not int or not 1 <= max_batch_bytes <= 2**63 - 1:
        raise PreparedDataV2Error("max_batch_bytes is outside its safety bound")
    if type(max_decoded_bytes) is not int or not (
        1 <= max_decoded_bytes <= ABSOLUTE_MAX_DECODED_BATCH_BYTES
    ):
        raise PreparedDataV2Error("max_decoded_batch_bytes is outside its safety bound")


def write_prepared_training_tree_v2(
    root: str | Path,
    *,
    train_blocks: tuple[PreparedMotionTextBlock, ...],
    val_blocks: tuple[PreparedMotionTextBlock, ...],
    max_total_edges: int,
    max_batch_size: int | None = None,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    max_decoded_batch_bytes: int = DEFAULT_MAX_DECODED_BATCH_BYTES,
) -> PreparedDataBuildResult:
    """Write one new immutable train/val tree; existing roots are never reused."""

    _validate_resource_bounds(max_batch_bytes, max_decoded_batch_bytes)
    destination = Path(root)
    if not destination.is_absolute():
        raise PreparedDataV2Error("prepared root must be absolute")
    if destination != destination.resolve(strict=False):
        raise PreparedDataV2Error("prepared root must be canonical")
    _reject_symlink_components(destination.parent, "prepared root parent")
    if destination.exists() or destination.is_symlink():
        raise PreparedDataV2Error("prepared root must be new")
    expanded = {
        "train": _expand_blocks(train_blocks),
        "val": _expand_blocks(val_blocks),
    }
    all_windows: set[str] = set()
    all_ordinals: set[int] = set()
    all_text: set[bytes] = set()
    all_actors: dict[str, set[bytes]] = {}
    for split in ("train", "val"):
        rows = expanded[split]
        windows = {row.sample.window_sha256 for row in rows}
        ordinals = {row.window_ordinal for row in rows}
        texts = {value for row in rows for value in row.text_commitments}
        actors = {value for row in rows for value in row.sample.actor_commitments}
        if (
            all_windows.intersection(windows)
            or all_ordinals.intersection(ordinals)
            or all_text.intersection(texts)
            or any(all_actors[prior].intersection(actors) for prior in all_actors)
        ):
            raise PreparedDataV2Error("train and val inputs are not strictly isolated")
        all_windows.update(windows)
        all_ordinals.update(ordinals)
        all_text.update(texts)
        all_actors[split] = actors

    plans = {
        split: edge_budget_batches(
            tuple(row.sample for row in expanded[split]),
            max_total_edges=max_total_edges,
            max_batch_size=max_batch_size,
        )
        for split in ("train", "val")
    }

    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    split_results: dict[str, tuple[str, int, int, int]] = {}
    census_splits: dict[str, list[dict[str, object]]] = {}
    for split, augmentation in (
        ("train", TRAIN_AUGMENTATION),
        ("val", VALIDATION_AUGMENTATION),
    ):
        rows = expanded[split]

        row_by_window = {row.sample.window_sha256: row for row in rows}
        planned = plans[split]
        split_directory = destination / split
        split_directory.mkdir(mode=0o700, exist_ok=False)
        manifest_rows = []
        census_rows = []
        split_motion_count = 0
        split_text_count = 0
        for batch_index, samples in enumerate(planned):
            selected = tuple(row_by_window[sample.window_sha256] for sample in samples)
            arrays = _batch_arrays(selected)
            batch_path = split_directory / f"batch-{batch_index:08d}.npz"
            digest = _write_npz_once(
                batch_path,
                arrays,
                max_decoded_bytes=max_decoded_batch_bytes,
            )
            if batch_path.stat().st_size > max_batch_bytes:
                raise PreparedDataV2Error("prepared batch exceeds max_batch_bytes")
            # Parse exactly what was written before admitting it to a manifest.
            _load_npz_training_batch(
                batch_path,
                expected_sha256=digest,
                split=split,
                max_batch_bytes=max_batch_bytes,
                max_decoded_bytes=max_decoded_batch_bytes,
            )
            relative = batch_path.relative_to(destination).as_posix()
            manifest_rows.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "window_ordinals": [row.window_ordinal for row in selected],
                }
            )
            census_rows.append(
                {
                    "batch_sha256": digest,
                    "motions": [
                        {
                            "clip_receipt_sha256": row.clip_receipt_sha256,
                            "group_commitment": row.sample.group_commitment.hex(),
                            "positive_family": row.positive_family.hex(),
                            "source_sha256": row.sample.source_sha256,
                            "text_commitments": [
                                value.hex() for value in row.text_commitments
                            ],
                            "window_ordinal": row.window_ordinal,
                            "window_sha256": row.sample.window_sha256,
                        }
                        for row in selected
                    ],
                    "path": relative,
                }
            )
            split_motion_count += len(selected)
            split_text_count += int(arrays["text_embeddings"].shape[0])
        manifest_path = destination / f"{split}.json"
        manifest_sha = _write_json_once(
            manifest_path,
            {
                "augmentation": augmentation,
                "batches": manifest_rows,
                "schema": PREPARED_SPLIT_V2_SCHEMA,
                "split": split,
            },
        )
        split_results[split] = (
            manifest_sha,
            len(manifest_rows),
            split_motion_count,
            split_text_count,
        )
        census_splits[split] = census_rows

    index_path = destination / "prepared-index.json"
    index_sha = _write_json_once(
        index_path,
        {
            "schema": PREPARED_INDEX_V2_SCHEMA,
            "train": {"path": "train.json", "sha256": split_results["train"][0]},
            "val": {"path": "val.json", "sha256": split_results["val"][0]},
        },
    )
    census_path = destination / "prepared-census.json"
    census_sha = _write_json_once(
        census_path,
        {
            "authority": AUTHORITY,
            "index_sha256": index_sha,
            "result_claimed": False,
            "schema": "phaseset-prepared-data-build-census-v2",
            "splits": census_splits,
        },
    )
    return PreparedDataBuildResult(
        root=destination.resolve(),
        index_path=index_path.resolve(),
        index_sha256=index_sha,
        census_path=census_path.resolve(),
        census_sha256=census_sha,
        train_manifest_sha256=split_results["train"][0],
        val_manifest_sha256=split_results["val"][0],
        batch_count=split_results["train"][1] + split_results["val"][1],
        motion_count=split_results["train"][2] + split_results["val"][2],
        text_count=split_results["train"][3] + split_results["val"][3],
    )


def _zip_member_census(raw: bytes, *, max_decoded_bytes: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if (
                names != tuple(name + ".npy" for name in NPZ_KEY_ORDER)
                or len(set(names)) != len(names)
            ):
                raise PreparedDataV2Error("prepared NPZ member census or order changed")
            total = 0
            decoded = 0
            for name, info in zip(NPZ_KEY_ORDER, infos, strict=True):
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.compress_size != info.file_size
                    or info.file_size < 1
                ):
                    raise PreparedDataV2Error("prepared NPZ member is not bounded ZIP_STORED")
                total += info.file_size
                if total > max_decoded_bytes:
                    raise PreparedDataV2Error("prepared NPZ decoded size exceeds its bound")
                with archive.open(info, mode="r") as member:
                    version = np.lib.format.read_magic(member)
                    if version != (1, 0):
                        raise PreparedDataV2Error("prepared NPY header version changed")
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        member
                    )
                    if (
                        fortran_order
                        or dtype != NPZ_DTYPES[name]
                        or dtype.hasobject
                        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                    ):
                        raise PreparedDataV2Error("prepared NPY header contract changed")
                    payload_bytes = math.prod(shape) * dtype.itemsize
                    if member.tell() + payload_bytes != info.file_size:
                        raise PreparedDataV2Error("prepared NPY payload length is inconsistent")
                    decoded += payload_bytes
                    if decoded > max_decoded_bytes:
                        raise PreparedDataV2Error(
                            "prepared array payloads exceed their decoded byte bound"
                        )
            if archive.testzip() is not None:
                raise PreparedDataV2Error("prepared NPZ CRC check failed")
    except PreparedDataV2Error:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise PreparedDataV2Error("prepared batch is not a valid bounded NPZ") from error


def _bytes32_rows(value: np.ndarray, label: str) -> tuple[bytes, ...]:
    if value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] != 32:
        raise PreparedDataV2Error(f"{label} must be uint8 [N,32]")
    return tuple(bytes(row) for row in np.ascontiguousarray(value))


def _actor_rows(
    value: np.ndarray,
    actor_mask: np.ndarray,
) -> tuple[tuple[bytes | None, ...], ...]:
    if (
        value.dtype != np.uint8
        or value.ndim != 3
        or value.shape != (*actor_mask.shape, 32)
    ):
        raise PreparedDataV2Error("actor_commitments must be uint8 [B,K,32]")
    rows = []
    for batch_index in range(value.shape[0]):
        commitments = []
        for actor_index in range(value.shape[1]):
            packed = bytes(value[batch_index, actor_index])
            if bool(actor_mask[batch_index, actor_index]):
                commitments.append(packed)
            else:
                if packed != b"\x00" * 32:
                    raise PreparedDataV2Error(
                        "invalid actor commitment slots must be exact zero"
                    )
                commitments.append(None)
        rows.append(tuple(commitments))
    return tuple(rows)


def _arrays_to_training_batch(
    arrays: Mapping[str, np.ndarray],
    *,
    split: Split,
) -> RetrievalTrainingBatch:
    if set(arrays) != NPZ_KEYS:
        raise PreparedDataV2Error("prepared batch NPZ keys are not closed")
    if any(arrays[name].dtype != dtype for name, dtype in NPZ_DTYPES.items()):
        raise PreparedDataV2Error("prepared batch contains an unexpected dtype")
    actor_mask = np.ascontiguousarray(arrays["actor_mask"])
    groups = PreparedGroupBatch(
        skeletons=np.ascontiguousarray(arrays["skeletons"]),
        actor_mask=actor_mask,
        frame_mask=np.ascontiguousarray(arrays["frame_mask"]),
        track_mask=np.ascontiguousarray(arrays["track_mask"]),
        actor_commitments=_actor_rows(arrays["actor_commitments"], actor_mask),
        group_commitments=_bytes32_rows(
            arrays["group_commitments"], "group_commitments"
        ),
    )
    text = np.ascontiguousarray(arrays["text_embeddings"])
    if text.ndim != 2 or text.shape[1] != EMBEDDING_DIM:
        raise PreparedDataV2Error("text_embeddings must be float32 [Q,512]")
    return RetrievalTrainingBatch(
        groups=groups,
        text_embeddings=torch.from_numpy(text),
        motion_positive_ids=_bytes32_rows(
            arrays["motion_positive_ids"], "motion_positive_ids"
        ),
        text_positive_ids=_bytes32_rows(
            arrays["text_positive_ids"], "text_positive_ids"
        ),
        text_commitments=_bytes32_rows(
            arrays["text_commitments"], "text_commitments"
        ),
        split=split,
    )


def _load_npz_training_batch(
    path: Path,
    *,
    expected_sha256: str,
    split: Split,
    max_batch_bytes: int,
    max_decoded_bytes: int,
) -> RetrievalTrainingBatch:
    raw = _read_bounded(path, maximum=max_batch_bytes, label="prepared batch")
    if _sha256_bytes(raw) != expected_sha256:
        raise PreparedDataV2Error("prepared batch digest mismatch")
    _zip_member_census(raw, max_decoded_bytes=max_decoded_bytes)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != NPZ_KEYS:
                raise PreparedDataV2Error("prepared batch NPZ keys are not closed")
            arrays = {
                name: np.ascontiguousarray(np.array(archive[name], copy=True, subok=False))
                for name in NPZ_KEY_ORDER
            }
    except PreparedDataV2Error:
        raise
    except (OSError, ValueError, MemoryError) as error:
        raise PreparedDataV2Error("prepared batch is not a safe numeric NPZ") from error
    if _decoded_nbytes(arrays) > max_decoded_bytes:
        raise PreparedDataV2Error("decoded prepared arrays exceed their byte bound")
    return _arrays_to_training_batch(arrays, split=split)


def _descriptor_contexts(
    batch: RetrievalTrainingBatch,
    *,
    prepared_manifest_sha256: str,
    source_batch_sha256: str,
    window_ordinals: tuple[int, ...],
    seed: int,
    epoch: int,
) -> tuple[DescriptorWindowContext, ...]:
    if len(window_ordinals) != batch.motion_count:
        raise PreparedDataV2Error("window ordinal count differs from motion count")
    context_epoch = epoch if batch.split == "train" else 0
    contexts = []
    for row, (positive_id, ordinal) in enumerate(
        zip(batch.motion_positive_ids, window_ordinals, strict=True)
    ):
        yaw = (
            deterministic_group_yaw(seed=seed, epoch=epoch, window_ordinal=ordinal)
            if batch.split == "train"
            else 0.0
        )
        try:
            context = DescriptorWindowContext.from_yaw(
                prepared_manifest_sha256=prepared_manifest_sha256,
                source_batch_sha256=source_batch_sha256,
                window_sha256=positive_id.hex(),
                window_ordinal=ordinal,
                split=batch.split,
                seed=seed,
                epoch=context_epoch,
                augmentation_yaw=yaw,
            )
        except (TypeError, ValueError) as error:
            raise PreparedDataV2Error(
                f"prepared descriptor context {row} is invalid"
            ) from error
        contexts.append(context)
    return tuple(contexts)


def _with_descriptor_contexts(
    batch: RetrievalTrainingBatch,
    contexts: tuple[DescriptorWindowContext, ...],
) -> RetrievalTrainingBatch:
    return RetrievalTrainingBatch(
        groups=batch.groups,
        text_embeddings=batch.text_embeddings,
        motion_positive_ids=batch.motion_positive_ids,
        text_positive_ids=batch.text_positive_ids,
        text_commitments=batch.text_commitments,
        split=batch.split,
        descriptor_contexts=contexts,
    )


def _rotate_training_batch(
    batch: RetrievalTrainingBatch,
    *,
    seed: int,
    epoch: int,
    ordinals: tuple[int, ...],
) -> RetrievalTrainingBatch:
    if batch.descriptor_contexts is not None:
        raise PreparedDataV2Error(
            "a batch with descriptor contexts cannot be rotated again"
        )
    groups = batch.groups
    if len(ordinals) != groups.batch_size:
        raise PreparedDataV2Error("window ordinal count differs from motion count")
    source = np.asarray(groups.skeletons, dtype=np.float64)
    rotated = np.empty(source.shape, dtype=np.float64)
    for row, ordinal in enumerate(ordinals):
        yaw = deterministic_group_yaw(seed=seed, epoch=epoch, window_ordinal=ordinal)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        x_axis = source[row, ..., 0]
        z_axis = source[row, ..., 2]
        rotated[row, ..., 0] = cosine * x_axis + sine * z_axis
        rotated[row, ..., 1] = source[row, ..., 1]
        rotated[row, ..., 2] = -sine * x_axis + cosine * z_axis
    skeletons = np.ascontiguousarray(rotated, dtype=np.float32)
    valid = (
        groups.actor_mask[:, :, None, None]
        & groups.frame_mask[:, None, :, None]
        & groups.track_mask
    )
    skeletons[~np.broadcast_to(valid[..., None], skeletons.shape)] = np.float32(0.0)
    rebuilt = PreparedGroupBatch(
        skeletons=skeletons,
        actor_mask=np.ascontiguousarray(groups.actor_mask),
        frame_mask=np.ascontiguousarray(groups.frame_mask),
        track_mask=np.ascontiguousarray(groups.track_mask),
        actor_commitments=groups.actor_commitments,
        group_commitments=groups.group_commitments,
    )
    return RetrievalTrainingBatch(
        groups=rebuilt,
        text_embeddings=batch.text_embeddings,
        motion_positive_ids=batch.motion_positive_ids,
        text_positive_ids=batch.text_positive_ids,
        text_commitments=batch.text_commitments,
        split=batch.split,
        descriptor_contexts=batch.descriptor_contexts,
    )


class PreparedTrainingDataSourceV2:
    """Strict v2 TrainingDataSource with nonaccumulating epoch yaw."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        expected_manifest_sha256: str,
        max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
        max_decoded_batch_bytes: int = DEFAULT_MAX_DECODED_BATCH_BYTES,
    ) -> None:
        _validate_resource_bounds(max_batch_bytes, max_decoded_batch_bytes)
        expected_manifest = _lower_sha256(
            expected_manifest_sha256,
            "expected_manifest_sha256",
        )
        unresolved = Path(manifest)
        _reject_symlink_components(unresolved, "prepared split manifest")
        path = unresolved.resolve(strict=True)
        raw, value = _load_json(path, label="prepared split manifest")
        manifest_sha256 = _sha256_bytes(raw)
        if manifest_sha256 != expected_manifest:
            raise PreparedDataV2Error("prepared split manifest digest mismatch")
        _closed(
            value,
            {"augmentation", "batches", "schema", "split"},
            "prepared split manifest",
        )
        if value["schema"] != PREPARED_SPLIT_V2_SCHEMA:
            raise PreparedDataV2Error("prepared split schema is not v2")
        split = value["split"]
        if split not in ("train", "val"):
            raise PreparedDataV2Error("prepared training split must be train or val")
        expected_augmentation = (
            TRAIN_AUGMENTATION if split == "train" else VALIDATION_AUGMENTATION
        )
        if value["augmentation"] != expected_augmentation:
            raise PreparedDataV2Error("prepared split augmentation policy changed")
        rows = value["batches"]
        if type(rows) is not list or not rows:
            raise PreparedDataV2Error("prepared split must list at least one batch")
        entries: list[_ArtifactEntry] = []
        seen_paths: set[Path] = set()
        seen_ordinals: set[int] = set()
        motion_families: set[bytes] = set()
        text_commitments: set[bytes] = set()
        actor_commitments: set[bytes] = set()
        for index, item in enumerate(rows):
            if type(item) is not dict:
                raise PreparedDataV2Error(f"batch row {index} must be an object")
            _closed(item, {"path", "sha256", "window_ordinals"}, f"batch row {index}")
            batch_path = _resolve_under(path.parent, item["path"], f"batch row {index}")
            digest = _lower_sha256(item["sha256"], f"batch row {index} sha256")
            raw_ordinals = item["window_ordinals"]
            if type(raw_ordinals) is not list or not raw_ordinals:
                raise PreparedDataV2Error("every batch must bind window ordinals")
            ordinals = tuple(
                _uint64(value, f"batch row {index} ordinal") for value in raw_ordinals
            )
            if batch_path in seen_paths or seen_ordinals.intersection(ordinals):
                raise PreparedDataV2Error("prepared split repeats a path or window ordinal")
            batch = _load_npz_training_batch(
                batch_path,
                expected_sha256=digest,
                split=split,
                max_batch_bytes=max_batch_bytes,
                max_decoded_bytes=max_decoded_batch_bytes,
            )
            if len(ordinals) != batch.motion_count:
                raise PreparedDataV2Error("window ordinal count differs from motion count")
            families = set(batch.motion_positive_ids)
            captions = set(batch.text_commitments)
            actors = {
                value
                for actor_row in batch.groups.actor_commitments
                for value in actor_row
                if value is not None
            }
            if (
                len(families) != batch.motion_count
                or motion_families.intersection(families)
                or len(captions) != len(batch.text_commitments)
                or text_commitments.intersection(captions)
            ):
                raise PreparedDataV2Error("prepared split repeats a motion or text identity")
            seen_paths.add(batch_path)
            seen_ordinals.update(ordinals)
            motion_families.update(families)
            text_commitments.update(captions)
            actor_commitments.update(actors)
            entries.append(_ArtifactEntry(batch_path, digest, ordinals))
        self.split: Split = split
        self.manifest_sha256 = manifest_sha256
        self._entries = tuple(entries)
        self._max_batch_bytes = max_batch_bytes
        self._max_decoded_bytes = max_decoded_batch_bytes
        self._window_ordinals = frozenset(seen_ordinals)
        self._motion_families = frozenset(motion_families)
        self._text_commitments = frozenset(text_commitments)
        self._actor_commitments = frozenset(actor_commitments)

    def iter_epoch(self, *, epoch: int, seed: int):
        checked_epoch = _uint64(epoch, "epoch")
        checked_seed = _uint64(seed, "seed")
        prefix = bytes.fromhex(self.manifest_sha256) + checked_seed.to_bytes(
            16, "big", signed=True
        )
        ordered = sorted(
            enumerate(self._entries),
            key=lambda row: hashlib.sha256(
                prefix
                + checked_epoch.to_bytes(8, "big")
                + row[0].to_bytes(8, "big")
            ).digest(),
        )
        for _, entry in ordered:
            original = _load_npz_training_batch(
                entry.path,
                expected_sha256=entry.sha256,
                split=self.split,
                max_batch_bytes=self._max_batch_bytes,
                max_decoded_bytes=self._max_decoded_bytes,
            )
            if self.split == "train":
                prepared = _rotate_training_batch(
                    original,
                    seed=checked_seed,
                    epoch=checked_epoch,
                    ordinals=entry.window_ordinals,
                )
            else:
                prepared = original
            contexts = _descriptor_contexts(
                prepared,
                prepared_manifest_sha256=self.manifest_sha256,
                source_batch_sha256=entry.sha256,
                window_ordinals=entry.window_ordinals,
                seed=checked_seed,
                epoch=checked_epoch,
            )
            yield _with_descriptor_contexts(prepared, contexts)


def load_prepared_training_sources_v2(
    index: str | Path,
    *,
    expected_index_sha256: str,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    max_decoded_batch_bytes: int = DEFAULT_MAX_DECODED_BATCH_BYTES,
) -> tuple[PreparedTrainingDataSourceV2, PreparedTrainingDataSourceV2]:
    """Load one exact v2 index and prove strict train/validation isolation."""

    unresolved = Path(index)
    _reject_symlink_components(unresolved, "prepared index")
    path = unresolved.resolve(strict=True)
    raw, value = _load_json(path, label="prepared index")
    if _sha256_bytes(raw) != _lower_sha256(
        expected_index_sha256, "expected_index_sha256"
    ):
        raise PreparedDataV2Error("prepared index differs from its expected digest")
    _closed(value, {"schema", "train", "val"}, "prepared index")
    if value["schema"] != PREPARED_INDEX_V2_SCHEMA:
        raise PreparedDataV2Error("prepared index schema is not v2")
    sources = []
    for split in ("train", "val"):
        row = value[split]
        if type(row) is not dict:
            raise PreparedDataV2Error(f"prepared index {split} row must be an object")
        _closed(row, {"path", "sha256"}, f"prepared index {split} row")
        manifest = _resolve_under(path.parent, row["path"], f"{split} manifest")
        digest = _lower_sha256(row["sha256"], f"{split} manifest sha256")
        source = PreparedTrainingDataSourceV2(
            manifest,
            expected_manifest_sha256=digest,
            max_batch_bytes=max_batch_bytes,
            max_decoded_batch_bytes=max_decoded_batch_bytes,
        )
        if source.split != split:
            raise PreparedDataV2Error(f"{split} manifest declares the wrong split")
        sources.append(source)
    train, val = sources
    if (
        train._window_ordinals & val._window_ordinals
        or train._motion_families & val._motion_families
        or train._text_commitments & val._text_commitments
        or train._actor_commitments & val._actor_commitments
    ):
        raise PreparedDataV2Error("prepared train and val identities are not disjoint")
    return train, val


__all__ = [
    "ABSOLUTE_MAX_DECODED_BATCH_BYTES",
    "AUTHORITY",
    "DEFAULT_MAX_BATCH_BYTES",
    "DEFAULT_MAX_DECODED_BATCH_BYTES",
    "NPZ_KEYS",
    "NPZ_KEY_ORDER",
    "PREPARED_INDEX_V2_SCHEMA",
    "PREPARED_SPLIT_V2_SCHEMA",
    "PreparedDataBuildResult",
    "PreparedDataV2Error",
    "PreparedMotionTextBlock",
    "PreparedTrainingDataSourceV2",
    "TRAIN_AUGMENTATION",
    "VALIDATION_AUGMENTATION",
    "load_prepared_training_sources_v2",
    "write_prepared_training_tree_v2",
]
