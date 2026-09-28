"""Immutable val-only storage for prepared capture-validation sources.

This authority-zero seam persists already-prepared numeric windows and already
frozen CLIP features.  It neither selects accepted windows nor handles caption
text, datasets, body-model assets, checkpoints, or evaluation test data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from typing import Final, Mapping
import zipfile

import numpy as np
import torch

from .capture_pooling import CaptureWindowPlan
from .capture_validation import (
    CaptureDescriptorWindowSource,
    CaptureValidationCapture,
    CaptureValidationSource,
    CaptureValidationWindow,
)
from .contracts import PreparedGroupBatch
from .frozen_clip_text import rehydrate_frozen_clip_text_batch


AUTHORITY: Final = 0
SCHEMA_V1: Final = "phaseset-capture-prepared-storage-v1"
SCHEMA_V2: Final = "phaseset-capture-prepared-storage-v2"
SCHEMA: Final = SCHEMA_V1
MANIFEST_NAME: Final = "capture-validation-manifest.json"
WINDOW_KEYS: Final = (
    "skeletons",
    "actor_mask",
    "frame_mask",
    "track_mask",
    "actor_commitments",
    "group_commitments",
)
TEXT_KEYS: Final = ("embeddings",)
WINDOW_DTYPES: Final = {
    "skeletons": np.dtype(np.float32),
    "actor_mask": np.dtype(np.bool_),
    "frame_mask": np.dtype(np.bool_),
    "track_mask": np.dtype(np.bool_),
    "actor_commitments": np.dtype(np.uint8),
    "group_commitments": np.dtype(np.uint8),
}
TEXT_DTYPES: Final = {"embeddings": np.dtype(np.float32)}
MAX_JSON_BYTES: Final = 64 * 1024 * 1024
ABSOLUTE_MAX_FILE_BYTES: Final = 8 * 2**30
ABSOLUTE_MAX_TOTAL_MATERIALIZED_BYTES: Final = 64 * 2**30


class CapturePreparedStorageError(ValueError):
    """A source or consumed artifact violates the closed storage contract."""


class CapturePreparedStorageResourceLimit(CapturePreparedStorageError):
    """A complete source exceeds a declared bound; no rows are sampled."""


@dataclass(frozen=True, slots=True)
class CaptureStorageLimits:
    """Explicit complete-census allocation and cardinality bounds."""

    max_file_bytes: int = 2**31
    max_decoded_file_bytes: int = 2**31
    max_total_materialized_bytes: int = 8 * 2**30
    max_captures: int = 100_000
    max_windows: int = 1_000_000
    max_captions: int = 1_000_000

    def __post_init__(self) -> None:
        values = (
            self.max_file_bytes,
            self.max_decoded_file_bytes,
            self.max_total_materialized_bytes,
            self.max_captures,
            self.max_windows,
            self.max_captions,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise CapturePreparedStorageError("storage limits must be positive exact ints")
        if self.max_file_bytes > ABSOLUTE_MAX_FILE_BYTES:
            raise CapturePreparedStorageError("max_file_bytes exceeds the absolute bound")
        if self.max_decoded_file_bytes > ABSOLUTE_MAX_FILE_BYTES:
            raise CapturePreparedStorageError(
                "max_decoded_file_bytes exceeds the absolute bound"
            )
        if self.max_total_materialized_bytes > ABSOLUTE_MAX_TOTAL_MATERIALIZED_BYTES:
            raise CapturePreparedStorageError(
                "max_total_materialized_bytes exceeds the absolute bound"
            )


@dataclass(frozen=True, slots=True)
class CaptureStorageBuildResult:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    source_census_sha256: str
    capture_count: int
    window_count: int
    caption_count: int
    total_materialized_bytes: int
    authority: int = AUTHORITY


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CapturePreparedStorageError(f"{label} must be lowercase SHA-256 hex")
    return value


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise CapturePreparedStorageError(f"{label} must be exact bytes[32]")
    return value


def _commitment_hex(value: object, label: str) -> bytes:
    return bytes.fromhex(_lower_sha256(value, label))


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
            raise CapturePreparedStorageError("JSON contains a duplicate key")
        output[key] = value
    return output


def _reject_constant(value: str) -> object:
    raise CapturePreparedStorageError(f"JSON contains forbidden constant {value}")


def _parse_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        canonical = _canonical_json_bytes(value)
    except CapturePreparedStorageError:
        raise
    except (UnicodeError, TypeError, ValueError, OverflowError, RecursionError):
        raise CapturePreparedStorageError(f"{label} must be strict ASCII JSON") from None
    if type(value) is not dict or raw != canonical:
        raise CapturePreparedStorageError(f"{label} must be a canonical JSON object")
    return value


def _closed(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise CapturePreparedStorageError(f"{label} keys are not closed")


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise CapturePreparedStorageError(f"{label} must be a positive exact int")
    return value


def _uint64(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise CapturePreparedStorageError(f"{label} must be an exact uint64")
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise CapturePreparedStorageError(f"{label} may not traverse a symlink")


def _canonical_existing_root(root: Path) -> Path:
    if not isinstance(root, Path) or not root.is_absolute():
        raise CapturePreparedStorageError("artifact root must be an absolute pathlib.Path")
    _reject_symlink_components(root, "artifact root")
    resolved = root.resolve()
    if not resolved.is_dir():
        raise CapturePreparedStorageError("artifact root must be an existing directory")
    return resolved


def _resolve_under(root: Path, value: object, label: str) -> Path:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise CapturePreparedStorageError(f"{label} must be a canonical relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CapturePreparedStorageError(f"{label} contains an invalid path component")
    unresolved = root.joinpath(*parts)
    _reject_symlink_components(unresolved, label)
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CapturePreparedStorageError(f"{label} escapes the artifact root") from error
    if not resolved.is_file():
        raise CapturePreparedStorageError(f"{label} must be a regular file")
    return resolved


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _cross_interface_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    # Windows path and descriptor APIs can expose different ctime semantics.
    # Check ctime within each API before/after, never across the two APIs.
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_bounded(path: Path, *, maximum: int, label: str) -> bytes:
    descriptor = -1
    try:
        path_before = os.lstat(path)
        if not stat.S_ISREG(path_before.st_mode):
            raise CapturePreparedStorageError(
                f"{label} must be a regular non-symlink file"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or _cross_interface_identity(path_before)
            != _cross_interface_identity(opened_before)
        ):
            raise CapturePreparedStorageError(f"{label} changed before it was opened")
        size = opened_before.st_size
        if not 1 <= size <= maximum:
            raise CapturePreparedStorageResourceLimit(f"{label} exceeds its byte bound")
        raw = bytearray()
        while len(raw) <= size:
            block = os.read(descriptor, min(1024 * 1024, size + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        opened_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if (
            not stat.S_ISREG(path_after.st_mode)
            or _stat_identity(opened_before) != _stat_identity(opened_after)
            or _stat_identity(path_before) != _stat_identity(path_after)
            or _cross_interface_identity(opened_after)
            != _cross_interface_identity(path_after)
            or len(raw) != size
        ):
            raise CapturePreparedStorageError(f"{label} changed while it was read")
        return bytes(raw)
    except CapturePreparedStorageError:
        raise
    except OSError:
        raise CapturePreparedStorageError(f"{label} could not be read safely") from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path.chmod(stat.S_IRUSR)


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, value, version=(1, 0), allow_pickle=False)
    return output.getvalue()


def _npz_bytes(arrays: Mapping[str, np.ndarray], keys: tuple[str, ...]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in keys:
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, _npy_bytes(arrays[key]))
    return output.getvalue()


def _array_decoded_bytes(arrays: Mapping[str, np.ndarray]) -> int:
    return sum(int(value.nbytes) for value in arrays.values())


def _decode_npz(
    raw: bytes,
    *,
    keys: tuple[str, ...],
    dtypes: Mapping[str, np.dtype],
    limits: CaptureStorageLimits,
    materialized_byte_limit: int,
    label: str,
) -> tuple[dict[str, np.ndarray], int]:
    if type(materialized_byte_limit) is not int or materialized_byte_limit < 1:
        raise CapturePreparedStorageResourceLimit(
            f"{label} has no remaining materialized-byte budget"
        )
    expected_names = [f"{key}.npy" for key in keys]
    try:
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as archive:
            infos = archive.infolist()
            if [value.filename for value in infos] != expected_names:
                raise CapturePreparedStorageError(f"{label} member census is invalid")
            decoded_from_headers = 0
            for key, info in zip(keys, infos, strict=True):
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.compress_size != info.file_size
                    or info.flag_bits & 0x1
                    or info.is_dir()
                    or info.file_size < 1
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.create_system != 3
                    or info.external_attr != (stat.S_IFREG | 0o600) << 16
                    or info.file_size > limits.max_decoded_file_bytes
                ):
                    raise CapturePreparedStorageError(f"{label} ZIP metadata is invalid")
            if sum(info.file_size for info in infos) > limits.max_decoded_file_bytes:
                raise CapturePreparedStorageResourceLimit(
                    f"{label} decoded archive exceeds its byte bound"
                )
            for key, info in zip(keys, infos, strict=True):
                with archive.open(info, mode="r") as member:
                    version = np.lib.format.read_magic(member)
                    if version != (1, 0):
                        raise CapturePreparedStorageError(
                            f"{label} NPY header version is invalid"
                        )
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        member
                    )
                    if (
                        fortran_order
                        or dtype != dtypes[key]
                        or dtype.hasobject
                        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                    ):
                        raise CapturePreparedStorageError(
                            f"{label} NPY header contract is invalid"
                        )
                    payload_bytes = math.prod(shape) * dtype.itemsize
                    if member.tell() + payload_bytes != info.file_size:
                        raise CapturePreparedStorageError(
                            f"{label} NPY payload length is inconsistent"
                        )
                    decoded_from_headers += payload_bytes
                    if (
                        decoded_from_headers > limits.max_decoded_file_bytes
                        or decoded_from_headers > materialized_byte_limit
                    ):
                        raise CapturePreparedStorageResourceLimit(
                            f"{label} decoded arrays exceed their byte bound"
                        )
            if archive.testzip() is not None:
                raise CapturePreparedStorageError(f"{label} CRC check failed")
            arrays: dict[str, np.ndarray] = {}
            with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
                if list(loaded.files) != list(keys):
                    raise CapturePreparedStorageError(f"{label} array census is invalid")
                for key in keys:
                    value = loaded[key]
                    if (
                        type(value) is not np.ndarray
                        or value.dtype != dtypes[key]
                        or not value.flags.c_contiguous
                    ):
                        raise CapturePreparedStorageError(
                            f"{label} array {key!r} has an invalid dtype or layout"
                        )
                    arrays[key] = np.array(value, copy=True, order="C", subok=False)
    except CapturePreparedStorageError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise CapturePreparedStorageError(f"{label} is not a valid canonical NPZ") from error
    decoded = _array_decoded_bytes(arrays)
    if (
        decoded != decoded_from_headers
        or decoded > limits.max_decoded_file_bytes
        or decoded > materialized_byte_limit
    ):
        raise CapturePreparedStorageResourceLimit(
            f"{label} decoded arrays exceed their byte bound"
        )
    return arrays, decoded


def _window_arrays(groups: PreparedGroupBatch) -> dict[str, np.ndarray]:
    k_pad = groups.padded_actor_count
    actors = np.zeros((1, k_pad, 32), dtype=np.uint8)
    for index, commitment in enumerate(groups.actor_commitments[0]):
        if commitment is not None:
            actors[0, index] = np.frombuffer(commitment, dtype=np.uint8)
    group = np.frombuffer(groups.group_commitments[0], dtype=np.uint8).reshape(1, 32)
    return {
        "skeletons": np.array(groups.skeletons, copy=True, order="C"),
        "actor_mask": np.array(groups.actor_mask, copy=True, order="C"),
        "frame_mask": np.array(groups.frame_mask, copy=True, order="C"),
        "track_mask": np.array(groups.track_mask, copy=True, order="C"),
        "actor_commitments": actors,
        "group_commitments": np.array(group, copy=True, order="C"),
    }


def _window_expected_decoded_bytes(groups: PreparedGroupBatch) -> int:
    return (
        int(groups.skeletons.nbytes)
        + int(groups.actor_mask.nbytes)
        + int(groups.frame_mask.nbytes)
        + int(groups.track_mask.nbytes)
        + groups.padded_actor_count * 32
        + 32
    )


def _groups_from_arrays(arrays: Mapping[str, np.ndarray]) -> PreparedGroupBatch:
    skeletons = arrays["skeletons"]
    if skeletons.ndim != 5 or skeletons.shape[0] != 1:
        raise CapturePreparedStorageError("window skeleton shape must be [1,K_pad,T,22,3]")
    _, k_pad, frames, joints, coordinates = skeletons.shape
    if frames != 200 or joints != 22 or coordinates != 3 or k_pad < 2:
        raise CapturePreparedStorageError("window skeleton dimensions are invalid")
    shapes = {
        "actor_mask": (1, k_pad),
        "frame_mask": (1, 200),
        "track_mask": (1, k_pad, 200, 22),
        "actor_commitments": (1, k_pad, 32),
        "group_commitments": (1, 32),
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise CapturePreparedStorageError(f"window array {key!r} shape is invalid")
    actors: list[bytes | None] = []
    for index in range(k_pad):
        raw = arrays["actor_commitments"][0, index].tobytes(order="C")
        actors.append(raw if bool(arrays["actor_mask"][0, index]) else None)
        if not bool(arrays["actor_mask"][0, index]) and raw != bytes(32):
            raise CapturePreparedStorageError(
                "invalid actor commitment slots must be exact zero"
            )
    try:
        return PreparedGroupBatch(
            skeletons,
            arrays["actor_mask"],
            arrays["frame_mask"],
            arrays["track_mask"],
            (tuple(actors),),
            (arrays["group_commitments"][0].tobytes(order="C"),),
        )
    except (TypeError, ValueError) as error:
        raise CapturePreparedStorageError("window batch contract is invalid") from error


def _snapshot_source(source: CaptureValidationSource) -> CaptureValidationSource:
    if type(source) is not CaptureValidationSource:
        raise CapturePreparedStorageError("source must be exact CaptureValidationSource")
    try:
        checked = CaptureValidationSource(
            source.split,
            source.manifest_sha256,
            source.captures,
        )
    except (TypeError, ValueError) as error:
        raise CapturePreparedStorageError("capture validation source is invalid") from error
    rebuilt: list[CaptureValidationCapture] = []
    for capture in checked.captures:
        receipt_raw = capture.holistic_text.receipt.canonical_json_bytes()
        try:
            text = rehydrate_frozen_clip_text_batch(
                capture.holistic_text.embeddings,
                receipt_json_bytes=receipt_raw,
                expected_receipt_sha256=_sha256(receipt_raw),
            )
            rebuilt.append(
                CaptureValidationCapture(
                    capture.plan,
                    capture.windows,
                    text,
                    capture.component_label,
                )
            )
        except Exception as error:
            raise CapturePreparedStorageError(
                "capture source contains a non-rehydratable frozen text batch"
            ) from error
    try:
        snapshot = CaptureValidationSource("val", checked.manifest_sha256, tuple(rebuilt))
    except (TypeError, ValueError) as error:
        raise CapturePreparedStorageError("capture source snapshot is invalid") from error
    if snapshot.census_sha256 != checked.census_sha256:
        raise CapturePreparedStorageError("capture source changed during snapshotting")
    return snapshot


def _count_source(source: CaptureValidationSource) -> tuple[int, int, int]:
    return (
        len(source.captures),
        sum(len(capture.windows) for capture in source.captures),
        sum(len(capture.holistic_text.caption_commitments) for capture in source.captures),
    )


def _check_counts(counts: tuple[int, int, int], limits: CaptureStorageLimits) -> None:
    if counts[0] > limits.max_captures:
        raise CapturePreparedStorageResourceLimit("capture census exceeds its bound")
    if counts[1] > limits.max_windows:
        raise CapturePreparedStorageResourceLimit("window census exceeds its bound")
    if counts[2] > limits.max_captions:
        raise CapturePreparedStorageResourceLimit("caption census exceeds its bound")


def _add_materialized(total: int, amount: int, limits: CaptureStorageLimits) -> int:
    result = total + amount
    if result > limits.max_total_materialized_bytes:
        raise CapturePreparedStorageResourceLimit(
            "complete source exceeds the total materialized-byte bound"
        )
    return result


def _write_capture_validation_source(
    root: Path,
    source: CaptureValidationSource,
    *,
    limits: CaptureStorageLimits,
    schema: str,
) -> CaptureStorageBuildResult:
    """Write one new immutable artifact tree under an exact storage schema."""

    if type(limits) is not CaptureStorageLimits:
        raise CapturePreparedStorageError("limits must be exact CaptureStorageLimits")
    if schema not in (SCHEMA_V1, SCHEMA_V2):
        raise CapturePreparedStorageError("storage writer schema is not registered")
    checked = _snapshot_source(source)
    counts = _count_source(checked)
    _check_counts(counts, limits)
    if not isinstance(root, Path) or not root.is_absolute():
        raise CapturePreparedStorageError("new artifact root must be an absolute Path")
    if root != root.resolve(strict=False):
        raise CapturePreparedStorageError("new artifact root must be canonical")
    _reject_symlink_components(root.parent, "artifact root parent")
    parent = root.parent.resolve()
    if not parent.is_dir():
        raise CapturePreparedStorageError("artifact root parent must exist")
    canonical = parent / root.name
    if canonical.exists() or canonical.is_symlink():
        raise CapturePreparedStorageError("new artifact root must not exist")
    canonical.mkdir(mode=0o700, parents=False, exist_ok=False)

    capture_rows: list[dict[str, object]] = []
    total = 0
    next_window_ordinal = 0
    for capture_index, capture in enumerate(checked.captures):
        window_rows: list[dict[str, object]] = []
        for window_index, (window, source_start) in enumerate(
            zip(capture.windows, capture.plan.source_start_frames, strict=True)
        ):
            relative = (
                f"captures/capture-{capture_index:06d}/window-{window_index:06d}.npz"
            )
            expected_decoded = _window_expected_decoded_bytes(window.groups)
            if expected_decoded > limits.max_decoded_file_bytes:
                raise CapturePreparedStorageResourceLimit(
                    "window decoded arrays exceed their byte bound"
                )
            total = _add_materialized(total, expected_decoded, limits)
            arrays = _window_arrays(window.groups)
            decoded = _array_decoded_bytes(arrays)
            if decoded != expected_decoded:
                raise CapturePreparedStorageError("window decoded-byte preflight changed")
            raw = _npz_bytes(arrays, WINDOW_KEYS)
            if len(raw) > limits.max_file_bytes:
                raise CapturePreparedStorageResourceLimit("window NPZ exceeds its byte bound")
            path = canonical.joinpath(*relative.split("/"))
            _write_once(path, raw)
            consumed = _read_bounded(path, maximum=limits.max_file_bytes, label="window NPZ")
            rebuilt, rebuilt_bytes = _decode_npz(
                consumed,
                keys=WINDOW_KEYS,
                dtypes=WINDOW_DTYPES,
                limits=limits,
                materialized_byte_limit=decoded,
                label="window NPZ",
            )
            _groups_from_arrays(rebuilt)
            if rebuilt_bytes != decoded:
                raise CapturePreparedStorageError("window decoded-byte census changed")
            window_row = {
                "decoded_bytes": decoded,
                "path": relative,
                "sha256": _sha256(consumed),
                "source_start_frame": source_start,
                "window_commitment": window.window_commitment.hex(),
            }
            if schema == SCHEMA_V2:
                window_row["window_ordinal"] = next_window_ordinal
            window_rows.append(window_row)
            next_window_ordinal += 1

        expected_text_decoded = capture.holistic_text.receipt.output_bytes
        if (
            type(expected_text_decoded) is not int
            or expected_text_decoded < 1
            or expected_text_decoded > limits.max_decoded_file_bytes
        ):
            raise CapturePreparedStorageResourceLimit(
                "text decoded arrays exceed their byte bound"
            )
        total = _add_materialized(total, expected_text_decoded, limits)
        embeddings = capture.holistic_text.embeddings
        text_array = embeddings.detach().cpu().contiguous().numpy().copy(order="C")
        text_arrays = {"embeddings": text_array}
        text_decoded = _array_decoded_bytes(text_arrays)
        if text_decoded != expected_text_decoded:
            raise CapturePreparedStorageError("text decoded-byte preflight changed")
        text_relative = f"captures/capture-{capture_index:06d}/text.npz"
        text_raw = _npz_bytes(text_arrays, TEXT_KEYS)
        if len(text_raw) > limits.max_file_bytes:
            raise CapturePreparedStorageResourceLimit("text NPZ exceeds its byte bound")
        text_path = canonical.joinpath(*text_relative.split("/"))
        _write_once(text_path, text_raw)
        consumed_text = _read_bounded(
            text_path, maximum=limits.max_file_bytes, label="text NPZ"
        )
        _, rebuilt_text_bytes = _decode_npz(
            consumed_text,
            keys=TEXT_KEYS,
            dtypes=TEXT_DTYPES,
            limits=limits,
            materialized_byte_limit=text_decoded,
            label="text NPZ",
        )
        if rebuilt_text_bytes != text_decoded:
            raise CapturePreparedStorageError("text decoded-byte census changed")

        receipt_raw = capture.holistic_text.receipt.canonical_json_bytes()
        if len(receipt_raw) > min(MAX_JSON_BYTES, limits.max_file_bytes):
            raise CapturePreparedStorageResourceLimit("text receipt exceeds its byte bound")
        total = _add_materialized(total, len(receipt_raw), limits)
        receipt_relative = f"captures/capture-{capture_index:06d}/text-receipt.json"
        receipt_path = canonical.joinpath(*receipt_relative.split("/"))
        _write_once(receipt_path, receipt_raw)
        consumed_receipt = _read_bounded(
            receipt_path,
            maximum=min(MAX_JSON_BYTES, limits.max_file_bytes),
            label="text receipt",
        )
        if consumed_receipt != receipt_raw:
            raise CapturePreparedStorageError("text receipt bytes changed after writing")
        capture_rows.append(
            {
                "capture_commitment": capture.capture_commitment.hex(),
                "component_label": capture.component_label,
                "plan_sha256": capture.plan.sha256,
                "text": {
                    "cache_key_sha256": (
                        capture.holistic_text.receipt.frozen_embedding_cache_key_sha256
                    ),
                    "caption_commitments": [
                        value.hex() for value in capture.holistic_text.caption_commitments
                    ],
                    "caption_count": len(capture.holistic_text.caption_commitments),
                    "decoded_bytes": text_decoded,
                    "path": text_relative,
                    "receipt_path": receipt_relative,
                    "receipt_sha256": _sha256(consumed_receipt),
                    "sha256": _sha256(consumed_text),
                    "output_sha256": capture.holistic_text.receipt.output_sha256,
                },
                "windows": window_rows,
            }
        )

    manifest_value = {
        "authority": AUTHORITY,
        "captures": capture_rows,
        "counts": {
            "captions": counts[2],
            "captures": counts[0],
            "windows": counts[1],
        },
        "schema": schema,
        "source_census_sha256": checked.census_sha256,
        "split": "val",
        "upstream_manifest_sha256": checked.manifest_sha256,
    }
    manifest_raw = _canonical_json_bytes(manifest_value)
    if len(manifest_raw) > min(MAX_JSON_BYTES, limits.max_file_bytes):
        raise CapturePreparedStorageResourceLimit("storage manifest exceeds its byte bound")
    total = _add_materialized(total, len(manifest_raw), limits)
    manifest_path = canonical / MANIFEST_NAME
    _write_once(manifest_path, manifest_raw)
    consumed_manifest = _read_bounded(
        manifest_path,
        maximum=min(MAX_JSON_BYTES, limits.max_file_bytes),
        label="storage manifest",
    )
    _parse_json(consumed_manifest, "storage manifest")
    return CaptureStorageBuildResult(
        root=canonical,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(consumed_manifest),
        source_census_sha256=checked.census_sha256,
        capture_count=counts[0],
        window_count=counts[1],
        caption_count=counts[2],
        total_materialized_bytes=total,
    )


def write_capture_validation_source(
    root: Path,
    source: CaptureValidationSource,
    *,
    limits: CaptureStorageLimits = CaptureStorageLimits(),
) -> CaptureStorageBuildResult:
    """Write the unchanged v1 storage format for uncached validation."""

    if type(source) is CaptureValidationSource and any(
        window.descriptor_source is not None
        for capture in source.captures
        for window in capture.windows
    ):
        raise CapturePreparedStorageError(
            "v1 storage cannot discard v2 descriptor source lineage"
        )
    return _write_capture_validation_source(
        root,
        source,
        limits=limits,
        schema=SCHEMA_V1,
    )


def write_capture_validation_source_v2(
    root: Path,
    source: CaptureValidationSource,
    *,
    limits: CaptureStorageLimits = CaptureStorageLimits(),
) -> CaptureStorageBuildResult:
    """Write v2 with canonical global descriptor-window ordinals."""

    return _write_capture_validation_source(
        root,
        source,
        limits=limits,
        schema=SCHEMA_V2,
    )


def _manifest_counts(value: object, limits: CaptureStorageLimits) -> tuple[int, int, int]:
    if type(value) is not dict:
        raise CapturePreparedStorageError("manifest counts must be an object")
    _closed(value, {"captures", "windows", "captions"}, "manifest counts")
    counts = (
        _positive_int(value["captures"], "capture count"),
        _positive_int(value["windows"], "window count"),
        _positive_int(value["captions"], "caption count"),
    )
    _check_counts(counts, limits)
    return counts


def load_capture_validation_source(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    limits: CaptureStorageLimits = CaptureStorageLimits(),
) -> CaptureValidationSource:
    """Consume one authenticated canonical manifest and all referenced bytes."""

    if type(limits) is not CaptureStorageLimits:
        raise CapturePreparedStorageError("limits must be exact CaptureStorageLimits")
    expected = _lower_sha256(expected_manifest_sha256, "expected manifest SHA-256")
    if not isinstance(manifest_path, Path) or not manifest_path.is_absolute():
        raise CapturePreparedStorageError("manifest_path must be an absolute Path")
    root = _canonical_existing_root(manifest_path.parent)
    resolved_manifest = _resolve_under(root, manifest_path.name, "storage manifest")
    if resolved_manifest.name != MANIFEST_NAME:
        raise CapturePreparedStorageError("storage manifest filename is not canonical")
    manifest_raw = _read_bounded(
        resolved_manifest,
        maximum=min(
            MAX_JSON_BYTES,
            limits.max_file_bytes,
            limits.max_total_materialized_bytes,
        ),
        label="storage manifest",
    )
    consumed_manifest_sha256 = _sha256(manifest_raw)
    if consumed_manifest_sha256 != expected:
        raise CapturePreparedStorageError("consumed storage manifest SHA-256 mismatch")
    manifest = _parse_json(manifest_raw, "storage manifest")
    _closed(
        manifest,
        {
            "authority",
            "captures",
            "counts",
            "schema",
            "source_census_sha256",
            "split",
            "upstream_manifest_sha256",
        },
        "storage manifest",
    )
    storage_schema = manifest["schema"]
    if (
        type(manifest["authority"]) is not int
        or manifest["authority"] != AUTHORITY
        or type(storage_schema) is not str
        or storage_schema not in (SCHEMA_V1, SCHEMA_V2)
        or manifest["split"] != "val"
    ):
        raise CapturePreparedStorageError("storage manifest literals are invalid")
    source_census = _lower_sha256(manifest["source_census_sha256"], "source census")
    upstream = _lower_sha256(manifest["upstream_manifest_sha256"], "upstream manifest")
    counts = _manifest_counts(manifest["counts"], limits)
    rows = manifest["captures"]
    if type(rows) is not list or len(rows) != counts[0]:
        raise CapturePreparedStorageError("capture rows do not match their census")
    total = _add_materialized(0, len(manifest_raw), limits)
    captures: list[CaptureValidationCapture] = []
    seen_paths: set[str] = {MANIFEST_NAME}
    observed_windows = 0
    observed_captions = 0
    prior_capture_id: bytes | None = None
    next_window_ordinal = 0
    for capture_index, raw_capture in enumerate(rows):
        if type(raw_capture) is not dict:
            raise CapturePreparedStorageError("capture row must be an object")
        _closed(
            raw_capture,
            {
                "capture_commitment",
                "component_label",
                "plan_sha256",
                "text",
                "windows",
            },
            "capture row",
        )
        capture_id = _commitment_hex(
            raw_capture["capture_commitment"],
            "capture commitment",
        )
        if prior_capture_id is not None and capture_id <= prior_capture_id:
            raise CapturePreparedStorageError(
                "capture rows must be in unique commitment order"
            )
        prior_capture_id = capture_id
        component = raw_capture["component_label"]
        if type(component) is not str or not component:
            raise CapturePreparedStorageError("component label must be a nonempty string")
        plan_sha = _lower_sha256(raw_capture["plan_sha256"], "plan SHA-256")
        window_rows = raw_capture["windows"]
        if type(window_rows) is not list or not window_rows:
            raise CapturePreparedStorageError("capture must contain every planned window")
        observed_windows += len(window_rows)
        if observed_windows > limits.max_windows:
            raise CapturePreparedStorageResourceLimit("window census exceeds its bound")
        window_ids: list[bytes] = []
        starts: list[int] = []
        windows: list[CaptureValidationWindow] = []
        for window_index, raw_window in enumerate(window_rows):
            if type(raw_window) is not dict:
                raise CapturePreparedStorageError("window row must be an object")
            window_keys = {
                "decoded_bytes",
                "path",
                "sha256",
                "source_start_frame",
                "window_commitment",
            }
            if storage_schema == SCHEMA_V2:
                window_keys.add("window_ordinal")
            _closed(raw_window, window_keys, "window row")
            decoded_expected = _positive_int(
                raw_window["decoded_bytes"], "window decoded bytes"
            )
            if decoded_expected > limits.max_decoded_file_bytes:
                raise CapturePreparedStorageResourceLimit(
                    "window decoded-byte declaration exceeds its bound"
                )
            expected_relative = (
                f"captures/capture-{capture_index:06d}/window-{window_index:06d}.npz"
            )
            if raw_window["path"] != expected_relative or expected_relative in seen_paths:
                raise CapturePreparedStorageError("window path census is not canonical")
            seen_paths.add(expected_relative)
            path = _resolve_under(root, expected_relative, "window NPZ")
            raw = _read_bounded(path, maximum=limits.max_file_bytes, label="window NPZ")
            source_window_npz_sha256 = _lower_sha256(
                raw_window["sha256"],
                "window SHA-256",
            )
            if _sha256(raw) != source_window_npz_sha256:
                raise CapturePreparedStorageError("consumed window NPZ SHA-256 mismatch")
            arrays, decoded = _decode_npz(
                raw,
                keys=WINDOW_KEYS,
                dtypes=WINDOW_DTYPES,
                limits=limits,
                materialized_byte_limit=(
                    limits.max_total_materialized_bytes - total
                ),
                label="window NPZ",
            )
            if decoded != decoded_expected:
                raise CapturePreparedStorageError("window decoded-byte census mismatch")
            total = _add_materialized(total, decoded, limits)
            window_id = _commitment_hex(
                raw_window["window_commitment"],
                "window commitment",
            )
            window_ids.append(window_id)
            starts.append(_uint64(raw_window["source_start_frame"], "window start"))
            if storage_schema == SCHEMA_V2:
                window_ordinal = _uint64(
                    raw_window["window_ordinal"],
                    "window ordinal",
                )
                if window_ordinal != next_window_ordinal:
                    raise CapturePreparedStorageError(
                        "window ordinals must equal canonical global order"
                    )
                descriptor_source = CaptureDescriptorWindowSource(
                    storage_manifest_sha256=consumed_manifest_sha256,
                    source_window_npz_sha256=source_window_npz_sha256,
                    window_ordinal=window_ordinal,
                )
            else:
                descriptor_source = None
            next_window_ordinal += 1
            windows.append(
                CaptureValidationWindow(
                    window_id,
                    _groups_from_arrays(arrays),
                    descriptor_source,
                )
            )

        try:
            plan = CaptureWindowPlan(capture_id, tuple(window_ids), tuple(starts))
        except (TypeError, ValueError) as error:
            raise CapturePreparedStorageError("capture window plan is invalid") from error
        if plan.sha256 != plan_sha:
            raise CapturePreparedStorageError("capture plan SHA-256 mismatch")

        text_row = raw_capture["text"]
        if type(text_row) is not dict:
            raise CapturePreparedStorageError("text row must be an object")
        _closed(
            text_row,
            {
                "caption_commitments",
                "caption_count",
                "cache_key_sha256",
                "decoded_bytes",
                "output_sha256",
                "path",
                "receipt_path",
                "receipt_sha256",
                "sha256",
            },
            "text row",
        )
        caption_count = _positive_int(text_row["caption_count"], "caption count")
        observed_captions += caption_count
        if observed_captions > limits.max_captions:
            raise CapturePreparedStorageResourceLimit("caption census exceeds its bound")
        declared_commitments = text_row["caption_commitments"]
        if type(declared_commitments) is not list or len(declared_commitments) != caption_count:
            raise CapturePreparedStorageError("caption commitment census is invalid")
        commitments: list[bytes] = []
        for value in declared_commitments:
            commitments.append(_commitment_hex(value, "caption commitment"))
        if len(set(commitments)) != caption_count:
            raise CapturePreparedStorageError("caption commitments are not unique")
        text_relative = f"captures/capture-{capture_index:06d}/text.npz"
        receipt_relative = f"captures/capture-{capture_index:06d}/text-receipt.json"
        if (
            text_row["path"] != text_relative
            or text_row["receipt_path"] != receipt_relative
            or text_relative in seen_paths
            or receipt_relative in seen_paths
        ):
            raise CapturePreparedStorageError("text path census is not canonical")
        seen_paths.update((text_relative, receipt_relative))
        text_raw = _read_bounded(
            _resolve_under(root, text_relative, "text NPZ"),
            maximum=limits.max_file_bytes,
            label="text NPZ",
        )
        if _sha256(text_raw) != _lower_sha256(text_row["sha256"], "text SHA-256"):
            raise CapturePreparedStorageError("consumed text NPZ SHA-256 mismatch")
        text_arrays, text_decoded = _decode_npz(
            text_raw,
            keys=TEXT_KEYS,
            dtypes=TEXT_DTYPES,
            limits=limits,
            materialized_byte_limit=(limits.max_total_materialized_bytes - total),
            label="text NPZ",
        )
        if text_decoded != _positive_int(text_row["decoded_bytes"], "text decoded bytes"):
            raise CapturePreparedStorageError("text decoded-byte census mismatch")
        embeddings = text_arrays["embeddings"]
        if embeddings.shape != (caption_count, 512) or not bool(np.isfinite(embeddings).all()):
            raise CapturePreparedStorageError("text embeddings must be finite [Q,512]")
        total = _add_materialized(total, text_decoded, limits)
        remaining = limits.max_total_materialized_bytes - total
        receipt_raw = _read_bounded(
            _resolve_under(root, receipt_relative, "text receipt"),
            maximum=min(MAX_JSON_BYTES, limits.max_file_bytes, remaining),
            label="text receipt",
        )
        receipt_sha = _lower_sha256(text_row["receipt_sha256"], "receipt SHA-256")
        cache_key_sha = _lower_sha256(text_row["cache_key_sha256"], "cache key SHA-256")
        output_sha = _lower_sha256(text_row["output_sha256"], "output SHA-256")
        if _sha256(receipt_raw) != receipt_sha:
            raise CapturePreparedStorageError("consumed text receipt SHA-256 mismatch")
        total = _add_materialized(total, len(receipt_raw), limits)
        try:
            text_batch = rehydrate_frozen_clip_text_batch(
                torch.from_numpy(embeddings),
                receipt_json_bytes=receipt_raw,
                expected_receipt_sha256=receipt_sha,
            )
        except Exception as error:
            raise CapturePreparedStorageError("frozen text cache rehydration failed") from error
        if text_batch.caption_commitments != tuple(commitments):
            raise CapturePreparedStorageError("rehydrated caption lineage differs from manifest")
        if (
            text_batch.receipt.frozen_embedding_cache_key_sha256 != cache_key_sha
            or text_batch.receipt.output_sha256 != output_sha
        ):
            raise CapturePreparedStorageError("rehydrated text digests differ from manifest")
        try:
            captures.append(
                CaptureValidationCapture(plan, tuple(windows), text_batch, component)
            )
        except (TypeError, ValueError) as error:
            raise CapturePreparedStorageError("capture validation row is invalid") from error

    if (len(captures), observed_windows, observed_captions) != counts:
        raise CapturePreparedStorageError("observed artifact census differs from manifest")
    expected_directories = {"captures"} | {
        f"captures/capture-{index:06d}" for index in range(counts[0])
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CapturePreparedStorageError("artifact tree contains a symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise CapturePreparedStorageError("artifact tree contains a special file")
    if observed_files != seen_paths or observed_directories != expected_directories:
        raise CapturePreparedStorageError("artifact tree census is not closed")
    try:
        source = CaptureValidationSource("val", upstream, tuple(captures))
    except (TypeError, ValueError) as error:
        raise CapturePreparedStorageError("rehydrated capture source is invalid") from error
    if source.census_sha256 != source_census:
        raise CapturePreparedStorageError("rehydrated source census SHA-256 mismatch")
    return source
