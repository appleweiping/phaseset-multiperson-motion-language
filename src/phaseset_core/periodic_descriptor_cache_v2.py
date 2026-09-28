"""Private candidate for immutable PhaseSet periodic descriptor shards.

This module has no execution or scientific authority.  It freezes only the
weight-independent ``PairChunk`` inputs produced by the existing public
descriptor streams.  The sibling private model candidate consumes its sealed
reader handle explicitly; the host, training runtime, checkpoints, and resume
path remain unwired.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Final, Literal

import numpy as np

from .contracts import PreparedGroupBatch, validate_prepared_group_batch
from .controls import iter_mean_difference_dct_chunks
from .morlet import MORLET_ORACLE_SHA256
from .periodic import (
    BAND_COUNT,
    CANONICAL_MICROBLOCK_SIZE,
    TOKEN_WIDTH,
    PairChunk,
    iter_marginal_power_pair_chunks,
    iter_unordered_pair_chunks,
    validate_edge_chunk_size,
    validate_energy_floors,
)
from .pipeline import deterministic_group_yaw


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
STATUS: Final = "PRIVATE_CANDIDATE_NO_EXECUTION_AUTHORITY"

INDEX_SCHEMA: Final = "phaseset-periodic-descriptor-cache-index-v2"
ENTRY_SCHEMA: Final = "phaseset-periodic-descriptor-cache-entry-v2"
SHARD_SCHEMA: Final = "phaseset-periodic-descriptor-cache-shard-v2"
MAGIC: Final = b"PHASESET-PERIODIC-DESCRIPTOR-CACHE-V2\x00"

DescriptorStreamKind = Literal[
    "FULL_RELATION",
    "MARGINAL_POWER",
    "MEAN_DIFFERENCE_DCT",
]
STREAM_KINDS: Final[tuple[DescriptorStreamKind, ...]] = (
    "FULL_RELATION",
    "MARGINAL_POWER",
    "MEAN_DIFFERENCE_DCT",
)

_SHARED_ARRAYS: Final = ("batch_indices", "actor_i", "actor_j")
_STREAM_SUFFIXES: Final = ("tokens_ij", "tokens_ji", "support_mask")
_ARRAY_NAMES: Final = _SHARED_ARRAYS + tuple(
    f"{kind}__{suffix}" for kind in STREAM_KINDS for suffix in _STREAM_SUFFIXES
)
_EXPECTED_DTYPES: Final = {
    "batch_indices": "<i8",
    "actor_i": "<i8",
    "actor_j": "<i8",
    **{
        f"{kind}__{suffix}": "|b1" if suffix == "support_mask" else "<f4"
        for kind in STREAM_KINDS
        for suffix in _STREAM_SUFFIXES
    },
}
_DECODED_BYTES_PER_EDGE: Final = len(_SHARED_ARRAYS) * np.dtype("<i8").itemsize + len(
    STREAM_KINDS
) * (
    2 * BAND_COUNT * TOKEN_WIDTH * np.dtype("<f4").itemsize
    + BAND_COUNT * np.dtype("|b1").itemsize
)
_CACHE_BATCH_TOKEN: Final = object()


class DescriptorCacheV2Error(ValueError):
    """The private descriptor-cache contract is invalid."""


class DescriptorCacheV2Miss(DescriptorCacheV2Error):
    """The exact explicit descriptor entry is absent; no fallback is allowed."""


@dataclass(frozen=True, slots=True)
class DescriptorCacheBounds:
    """Caller-visible resource bounds for index, shards, and decoded arrays."""

    max_index_bytes: int = 64 * 1024 * 1024
    max_shard_bytes: int = 512 * 1024 * 1024
    max_decoded_shard_bytes: int = 256 * 1024 * 1024
    max_shards: int = 262_144
    max_total_shard_bytes: int = 1 << 40
    max_edges_per_shard: int = 32_768

    def __post_init__(self) -> None:
        for name in (
            "max_index_bytes",
            "max_shard_bytes",
            "max_decoded_shard_bytes",
            "max_shards",
            "max_total_shard_bytes",
            "max_edges_per_shard",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TypeError(f"{name} must be an exact positive int")
        if self.max_decoded_shard_bytes > self.max_shard_bytes:
            raise DescriptorCacheV2Error(
                "decoded shard bound exceeds stored shard bound"
            )


@dataclass(frozen=True, slots=True)
class DescriptorCacheSourceBinding:
    """Exact upstream identities supplied by a trusted caller, not authority."""

    prepared_manifest_sha256: str
    source_tree_sha256: str
    environment_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "prepared_manifest_sha256",
            "source_tree_sha256",
            "environment_sha256",
        ):
            _lower_sha256(getattr(self, name), name)

    def _value(self) -> dict[str, str]:
        return {
            "environment_sha256": self.environment_sha256,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "source_tree_sha256": self.source_tree_sha256,
        }


@dataclass(frozen=True, slots=True)
class DescriptorWindowContext:
    """One explicit prepared-window identity and augmentation context."""

    prepared_manifest_sha256: str
    source_batch_sha256: str
    window_sha256: str
    window_ordinal: int
    split: Literal["train", "val"]
    seed: int
    epoch: int
    augmentation_yaw_be_hex: str

    def __post_init__(self) -> None:
        _lower_sha256(self.prepared_manifest_sha256, "prepared_manifest_sha256")
        _lower_sha256(self.source_batch_sha256, "source_batch_sha256")
        _lower_sha256(self.window_sha256, "window_sha256")
        _uint64(self.window_ordinal, "window_ordinal")
        _uint64(self.seed, "seed")
        _uint64(self.epoch, "epoch")
        if self.split not in ("train", "val"):
            raise DescriptorCacheV2Error("split must be exactly train or val")
        yaw = _yaw_from_hex(self.augmentation_yaw_be_hex)
        if self.split == "train":
            expected = deterministic_group_yaw(
                seed=self.seed,
                epoch=self.epoch,
                window_ordinal=self.window_ordinal,
            )
            if struct.pack(">d", expected).hex() != self.augmentation_yaw_be_hex:
                raise DescriptorCacheV2Error(
                    "train yaw differs from deterministic_group_yaw"
                )
        elif self.epoch != 0 or struct.pack(">d", yaw) != b"\x00" * 8:
            raise DescriptorCacheV2Error(
                "validation context requires epoch zero and +0 yaw"
            )

    @classmethod
    def from_yaw(
        cls,
        *,
        prepared_manifest_sha256: str,
        source_batch_sha256: str,
        window_sha256: str,
        window_ordinal: int,
        split: Literal["train", "val"],
        seed: int,
        epoch: int,
        augmentation_yaw: float,
    ) -> DescriptorWindowContext:
        if type(augmentation_yaw) is not float or not math.isfinite(augmentation_yaw):
            raise TypeError("augmentation_yaw must be an exact finite float")
        return cls(
            prepared_manifest_sha256=prepared_manifest_sha256,
            source_batch_sha256=source_batch_sha256,
            window_sha256=window_sha256,
            window_ordinal=window_ordinal,
            split=split,
            seed=seed,
            epoch=epoch,
            augmentation_yaw_be_hex=struct.pack(">d", augmentation_yaw).hex(),
        )

    def _value(self) -> dict[str, object]:
        return {
            "augmentation_yaw_be_hex": self.augmentation_yaw_be_hex,
            "epoch": self.epoch,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "seed": self.seed,
            "source_batch_sha256": self.source_batch_sha256,
            "split": self.split,
            "window_ordinal": self.window_ordinal,
            "window_sha256": self.window_sha256,
        }


@dataclass(frozen=True, slots=True)
class DescriptorCacheArtifact:
    root: Path
    index_path: Path
    index_sha256: str
    shard_count: int
    total_shard_bytes: int


@dataclass(frozen=True, slots=True)
class _IndexRow:
    batch_input_sha256: str
    bytes: int
    cache_key_sha256: str
    edge_count: int
    epoch: int
    path: str
    seed: int
    sha256: str
    source_batch_sha256: str
    split: str
    window_contexts_sha256: str
    window_count: int

    def _value(self) -> dict[str, object]:
        return {
            "batch_input_sha256": self.batch_input_sha256,
            "bytes": self.bytes,
            "cache_key_sha256": self.cache_key_sha256,
            "edge_count": self.edge_count,
            "epoch": self.epoch,
            "path": self.path,
            "seed": self.seed,
            "sha256": self.sha256,
            "source_batch_sha256": self.source_batch_sha256,
            "split": self.split,
            "window_contexts_sha256": self.window_contexts_sha256,
            "window_count": self.window_count,
        }


def _lower_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{name} must be a lowercase SHA-256")
    return value


def _uint64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise TypeError(f"{name} must be an exact uint64")
    return value


def _yaw_from_hex(value: object) -> float:
    if (
        type(value) is not str
        or len(value) != 16
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError("augmentation_yaw_be_hex must be 16 lowercase hex characters")
    yaw = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(yaw) or not -math.pi <= yaw < math.pi:
        raise DescriptorCacheV2Error("augmentation yaw must be finite in [-pi,pi)")
    return yaw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DescriptorCacheV2Error(
            "value cannot be encoded as canonical ASCII JSON"
        ) from error


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise DescriptorCacheV2Error(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_float(value: str) -> object:
    raise DescriptorCacheV2Error(f"floating-point JSON is forbidden: {value}")


def _parse_canonical_json(raw: bytes, label: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw.endswith(b"\n") or b"\r" in raw:
        raise DescriptorCacheV2Error(f"{label} is not LF-terminated canonical JSON")
    try:
        value = json.loads(
            raw.decode("ascii", "strict"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DescriptorCacheV2Error(f"{label} is not strict ASCII JSON") from error
    if type(value) is not dict or _canonical_json(value) != raw:
        raise DescriptorCacheV2Error(f"{label} bytes are not canonical")
    return value


def _closed(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise DescriptorCacheV2Error(f"{label} keys are not closed")
    return value


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise DescriptorCacheV2Error(
            "descriptor cache v2 requires a little-endian runtime"
        )


def prepared_group_batch_sha256(value: PreparedGroupBatch) -> str:
    """Hash the complete canonical numeric, mask, padding, and commitment input."""

    checked = validate_prepared_group_batch(value)
    digest = hashlib.sha256(b"phaseset-prepared-group-batch-cache-v2\x00")
    for name, array in (
        ("skeletons", checked.skeletons),
        ("actor_mask", checked.actor_mask),
        ("frame_mask", checked.frame_mask),
        ("track_mask", checked.track_mask),
    ):
        encoded_name = name.encode("ascii")
        encoded_dtype = array.dtype.str.encode("ascii")
        digest.update(struct.pack(">H", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">H", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">H", array.ndim))
        for extent in array.shape:
            digest.update(struct.pack(">Q", extent))
        raw = array.tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    digest.update(struct.pack(">Q", len(checked.actor_commitments)))
    for row in checked.actor_commitments:
        digest.update(struct.pack(">Q", len(row)))
        for commitment in row:
            if commitment is None:
                digest.update(b"\x00")
            else:
                digest.update(b"\x01" + commitment)
    digest.update(struct.pack(">Q", len(checked.group_commitments)))
    for commitment in checked.group_commitments:
        digest.update(commitment)
    return digest.hexdigest()


def energy_floors_sha256(value: np.ndarray) -> str:
    _require_little_endian()
    floors = validate_energy_floors(value)
    return _sha256(np.asarray(floors, dtype="<f8", order="C").tobytes(order="C"))


def _validate_contexts(
    contexts: object,
    *,
    batch_size: int,
    binding: DescriptorCacheSourceBinding,
) -> tuple[DescriptorWindowContext, ...]:
    if type(contexts) is not tuple or len(contexts) != batch_size:
        raise DescriptorCacheV2Error(
            "contexts must be an exact tuple covering every batch row"
        )
    if any(type(context) is not DescriptorWindowContext for context in contexts):
        raise TypeError("contexts must contain exact DescriptorWindowContext values")
    checked = contexts
    if any(
        context.prepared_manifest_sha256 != binding.prepared_manifest_sha256
        for context in checked
    ):
        raise DescriptorCacheV2Error(
            "window context prepared manifest differs from binding"
        )
    first = checked[0]
    for context in checked[1:]:
        if (
            context.source_batch_sha256 != first.source_batch_sha256
            or context.split != first.split
            or context.seed != first.seed
            or context.epoch != first.epoch
        ):
            raise DescriptorCacheV2Error(
                "one shard cannot mix batch, split, seed, or epoch"
            )
    if len({context.window_sha256 for context in checked}) != len(checked):
        raise DescriptorCacheV2Error("one shard repeats a window identity")
    if len({context.window_ordinal for context in checked}) != len(checked):
        raise DescriptorCacheV2Error("one shard repeats a window ordinal")
    return checked


def _contexts_value(
    contexts: tuple[DescriptorWindowContext, ...],
) -> list[dict[str, object]]:
    return [context._value() for context in contexts]


def _contexts_sha256(contexts: tuple[DescriptorWindowContext, ...]) -> str:
    return _sha256(_canonical_json({"windows": _contexts_value(contexts)}))


def _entry_value(
    *,
    binding: DescriptorCacheSourceBinding,
    contexts: tuple[DescriptorWindowContext, ...],
    batch_input_sha256: str,
    floors_sha256: str,
) -> dict[str, object]:
    return {
        "batch_input_sha256": batch_input_sha256,
        "energy_floors_sha256": floors_sha256,
        "morlet_oracle_sha256": MORLET_ORACLE_SHA256,
        "schema": ENTRY_SCHEMA,
        "source_binding": binding._value(),
        "streams": list(STREAM_KINDS),
        "windows": _contexts_value(contexts),
    }


def _entry_key(
    *,
    binding: DescriptorCacheSourceBinding,
    contexts: tuple[DescriptorWindowContext, ...],
    batch_input_sha256: str,
    floors_sha256: str,
) -> str:
    return _sha256(
        _canonical_json(
            _entry_value(
                binding=binding,
                contexts=contexts,
                batch_input_sha256=batch_input_sha256,
                floors_sha256=floors_sha256,
            )
        )
    )


def iter_uncached_pair_chunks(
    batch: PreparedGroupBatch,
    stream_kind: DescriptorStreamKind,
    *,
    energy_floors: np.ndarray,
    edge_chunk_size: int = CANONICAL_MICROBLOCK_SIZE,
) -> Iterator[PairChunk]:
    """Return the existing public stream as the uncached comparison oracle."""

    from .contracts import skeleton_to_activity

    checked = validate_prepared_group_batch(batch)
    floors = validate_energy_floors(energy_floors)
    chunk_size = validate_edge_chunk_size(edge_chunk_size)
    activity = skeleton_to_activity(checked)
    edge_budget = sum(count * (count - 1) // 2 for count in checked.actor_counts)
    if stream_kind == "FULL_RELATION":
        return iter_unordered_pair_chunks(
            activity,
            energy_floors=floors,
            edge_chunk_size=chunk_size,
            edge_budget=edge_budget,
        )
    if stream_kind == "MARGINAL_POWER":
        return iter_marginal_power_pair_chunks(
            activity,
            edge_chunk_size=chunk_size,
            edge_budget=edge_budget,
        )
    if stream_kind == "MEAN_DIFFERENCE_DCT":
        return iter_mean_difference_dct_chunks(
            activity,
            energy_floors=floors,
            edge_chunk_size=chunk_size,
            edge_budget=edge_budget,
        )
    raise DescriptorCacheV2Error("unknown descriptor stream kind")


def _expected_edge_census(
    batch: PreparedGroupBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: list[int] = []
    left: list[int] = []
    right: list[int] = []
    for group, actor_count in enumerate(batch.actor_counts):
        for actor_i in range(actor_count):
            for actor_j in range(actor_i + 1, actor_count):
                groups.append(group)
                left.append(actor_i)
                right.append(actor_j)
    if not groups:
        raise DescriptorCacheV2Error("descriptor batch contains no unordered edges")
    return (
        np.ascontiguousarray(groups, dtype=np.int64),
        np.ascontiguousarray(left, dtype=np.int64),
        np.ascontiguousarray(right, dtype=np.int64),
    )


def _collect_stream(
    batch: PreparedGroupBatch,
    stream_kind: DescriptorStreamKind,
    floors: np.ndarray,
) -> dict[str, np.ndarray]:
    chunks = tuple(
        iter_uncached_pair_chunks(
            batch,
            stream_kind,
            energy_floors=floors,
            edge_chunk_size=CANONICAL_MICROBLOCK_SIZE,
        )
    )
    if not chunks:
        raise DescriptorCacheV2Error("uncached stream emitted no PairChunk")
    return {
        "batch_indices": np.ascontiguousarray(
            np.concatenate([chunk.batch_indices for chunk in chunks]), dtype=np.int64
        ),
        "actor_i": np.ascontiguousarray(
            np.concatenate([chunk.actor_i for chunk in chunks]), dtype=np.int64
        ),
        "actor_j": np.ascontiguousarray(
            np.concatenate([chunk.actor_j for chunk in chunks]), dtype=np.int64
        ),
        "tokens_ij": np.ascontiguousarray(
            np.concatenate([chunk.tokens_ij for chunk in chunks]), dtype=np.float32
        ),
        "tokens_ji": np.ascontiguousarray(
            np.concatenate([chunk.tokens_ji for chunk in chunks]), dtype=np.float32
        ),
        "support_mask": np.ascontiguousarray(
            np.concatenate([chunk.support_mask for chunk in chunks]), dtype=np.bool_
        ),
    }


def _collect_all_arrays(
    batch: PreparedGroupBatch, floors: np.ndarray
) -> dict[str, np.ndarray]:
    expected = _expected_edge_census(batch)
    arrays: dict[str, np.ndarray] = {
        "batch_indices": expected[0],
        "actor_i": expected[1],
        "actor_j": expected[2],
    }
    for kind in STREAM_KINDS:
        stream = _collect_stream(batch, kind, floors)
        for name, expected_values in zip(_SHARED_ARRAYS, expected, strict=True):
            if stream[name].tobytes(order="C") != expected_values.tobytes(order="C"):
                raise DescriptorCacheV2Error(
                    f"{kind} changed the canonical edge census"
                )
        for suffix in _STREAM_SUFFIXES:
            arrays[f"{kind}__{suffix}"] = stream[suffix]
    if tuple(arrays) != _ARRAY_NAMES:
        raise AssertionError("descriptor array insertion order drifted")
    return arrays


def _preflight_descriptor_resources(
    batch: PreparedGroupBatch,
    *,
    bounds: DescriptorCacheBounds,
    stored_bytes_before: int,
) -> tuple[int, int]:
    edge_count = sum(count * (count - 1) // 2 for count in batch.actor_counts)
    if edge_count < 1:
        raise DescriptorCacheV2Error("descriptor batch contains no unordered edges")
    if edge_count > bounds.max_edges_per_shard:
        raise DescriptorCacheV2Error("descriptor shard exceeds its edge-count bound")
    decoded_nbytes = edge_count * _DECODED_BYTES_PER_EDGE
    if decoded_nbytes > bounds.max_decoded_shard_bytes:
        raise DescriptorCacheV2Error("decoded descriptor shard exceeds its byte bound")
    minimum_stored_nbytes = len(MAGIC) + 8 + decoded_nbytes
    if minimum_stored_nbytes > bounds.max_shard_bytes:
        raise DescriptorCacheV2Error("stored descriptor shard exceeds its byte bound")
    if stored_bytes_before + minimum_stored_nbytes > bounds.max_total_shard_bytes:
        raise DescriptorCacheV2Error("descriptor cache exceeds its total byte bound")
    return edge_count, decoded_nbytes


def _storage_array(name: str, value: np.ndarray) -> np.ndarray:
    dtype = np.dtype(_EXPECTED_DTYPES[name])
    stored = np.asarray(value, dtype=dtype, order="C")
    if not stored.flags.c_contiguous:
        stored = np.ascontiguousarray(stored, dtype=dtype)
    return stored


def _serialize_shard(
    *,
    binding: DescriptorCacheSourceBinding,
    contexts: tuple[DescriptorWindowContext, ...],
    batch_input_sha256: str,
    floors_sha256: str,
    cache_key_sha256: str,
    arrays: dict[str, np.ndarray],
    bounds: DescriptorCacheBounds,
) -> bytes:
    metadata: list[dict[str, object]] = []
    payload = bytearray()
    for name in _ARRAY_NAMES:
        stored = _storage_array(name, arrays[name])
        raw = stored.tobytes(order="C")
        metadata.append(
            {
                "dtype": _EXPECTED_DTYPES[name],
                "name": name,
                "nbytes": len(raw),
                "offset": len(payload),
                "sha256": _sha256(raw),
                "shape": list(stored.shape),
            }
        )
        payload.extend(raw)
    edge_count = int(arrays["batch_indices"].shape[0])
    expected_decoded_nbytes = edge_count * _DECODED_BYTES_PER_EDGE
    if len(payload) != expected_decoded_nbytes:
        raise DescriptorCacheV2Error("descriptor shard array byte census changed")
    if len(payload) > bounds.max_decoded_shard_bytes:
        raise DescriptorCacheV2Error("decoded descriptor shard exceeds its byte bound")
    header = _canonical_json(
        {
            "arrays": metadata,
            "batch_input_sha256": batch_input_sha256,
            "cache_key_sha256": cache_key_sha256,
            "decoded_nbytes": len(payload),
            "edge_count": edge_count,
            "energy_floors_sha256": floors_sha256,
            "morlet_oracle_sha256": MORLET_ORACLE_SHA256,
            "schema": SHARD_SCHEMA,
            "source_binding": binding._value(),
            "streams": list(STREAM_KINDS),
            "windows": _contexts_value(contexts),
        }
    )
    raw = MAGIC + struct.pack(">Q", len(header)) + header + bytes(payload)
    if len(raw) > bounds.max_shard_bytes:
        raise DescriptorCacheV2Error("stored descriptor shard exceeds its byte bound")
    return raw


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise DescriptorCacheV2Error(f"{label} cannot traverse a symlink")


def _read_owned(path: Path, *, maximum: int, label: str) -> bytes:
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DescriptorCacheV2Error(f"{label} cannot be opened safely") from error
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise DescriptorCacheV2Error(
                    f"{label} must be a single-link regular file"
                )
            if before.st_size < 1 or before.st_size > maximum:
                raise DescriptorCacheV2Error(f"{label} exceeds its stored byte bound")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise DescriptorCacheV2Error(
                        f"{label} ended before its declared size"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise DescriptorCacheV2Error(f"{label} grew during its bounded read")
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise DescriptorCacheV2Error(f"{label} changed during its owned read")
        finally:
            os.close(descriptor)
    except DescriptorCacheV2Error:
        raise
    except OSError as error:
        raise DescriptorCacheV2Error(f"{label} owned read failed") from error
    raw = b"".join(chunks)
    _reject_symlink_components(path, label)
    try:
        current = path.stat()
    except OSError as error:
        raise DescriptorCacheV2Error(
            f"{label} disappeared after its owned read"
        ) from error
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise DescriptorCacheV2Error(
            f"{label} path identity changed during its owned read"
        )
    return raw


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, stat.S_IREAD)
    except OSError as error:
        raise DescriptorCacheV2Error(
            "descriptor artifact exclusive write failed"
        ) from error


def _row_from_value(value: object, *, bounds: DescriptorCacheBounds) -> _IndexRow:
    row = _closed(
        value,
        {
            "batch_input_sha256",
            "bytes",
            "cache_key_sha256",
            "edge_count",
            "epoch",
            "path",
            "seed",
            "sha256",
            "source_batch_sha256",
            "split",
            "window_contexts_sha256",
            "window_count",
        },
        "descriptor index row",
    )
    for name in (
        "batch_input_sha256",
        "cache_key_sha256",
        "sha256",
        "source_batch_sha256",
        "window_contexts_sha256",
    ):
        _lower_sha256(row[name], name)
    for name, maximum in (
        ("bytes", bounds.max_shard_bytes),
        ("edge_count", bounds.max_edges_per_shard),
        ("window_count", bounds.max_edges_per_shard),
    ):
        if type(row[name]) is not int or not 1 <= row[name] <= maximum:
            raise DescriptorCacheV2Error(
                f"descriptor index row {name} is out of bounds"
            )
    _uint64(row["seed"], "seed")
    _uint64(row["epoch"], "epoch")
    if row["split"] not in ("train", "val"):
        raise DescriptorCacheV2Error("descriptor index row split is invalid")
    expected_path = f"shards/{row['cache_key_sha256']}.pdc2"
    if row["path"] != expected_path:
        raise DescriptorCacheV2Error("descriptor shard path is not canonical")
    return _IndexRow(**row)  # type: ignore[arg-type]


def _shape_tuple(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise DescriptorCacheV2Error(f"{name} shape must be a nonempty list")
    if any(type(extent) is not int or extent < 0 for extent in value):
        raise DescriptorCacheV2Error(f"{name} shape contains an invalid extent")
    return tuple(value)


def _contexts_from_values(
    value: object,
    *,
    expected_count: int,
) -> tuple[DescriptorWindowContext, ...]:
    if type(value) is not list or len(value) != expected_count:
        raise DescriptorCacheV2Error("descriptor shard windows must be a nonempty list")
    contexts: list[DescriptorWindowContext] = []
    for index, item in enumerate(value):
        row = _closed(
            item,
            {
                "augmentation_yaw_be_hex",
                "epoch",
                "prepared_manifest_sha256",
                "seed",
                "source_batch_sha256",
                "split",
                "window_ordinal",
                "window_sha256",
            },
            f"descriptor window {index}",
        )
        try:
            contexts.append(DescriptorWindowContext(**row))  # type: ignore[arg-type]
        except (TypeError, DescriptorCacheV2Error) as error:
            raise DescriptorCacheV2Error(
                f"descriptor window {index} is malformed"
            ) from error
    return tuple(contexts)


def _decode_shard(
    raw: bytes,
    *,
    row: _IndexRow,
    binding: DescriptorCacheSourceBinding,
    contexts: tuple[DescriptorWindowContext, ...],
    batch_input_sha256: str,
    floors_sha256: str,
    bounds: DescriptorCacheBounds,
) -> dict[str, np.ndarray]:
    prefix_size = len(MAGIC) + 8
    if len(raw) < prefix_size or not raw.startswith(MAGIC):
        raise DescriptorCacheV2Error("descriptor shard magic is invalid")
    header_size = struct.unpack(">Q", raw[len(MAGIC) : prefix_size])[0]
    if header_size < 2 or prefix_size + header_size > len(raw):
        raise DescriptorCacheV2Error("descriptor shard header length is invalid")
    header_raw = raw[prefix_size : prefix_size + header_size]
    payload = memoryview(raw)[prefix_size + header_size :]
    header = _closed(
        _parse_canonical_json(header_raw, "descriptor shard header"),
        {
            "arrays",
            "batch_input_sha256",
            "cache_key_sha256",
            "decoded_nbytes",
            "edge_count",
            "energy_floors_sha256",
            "morlet_oracle_sha256",
            "schema",
            "source_binding",
            "streams",
            "windows",
        },
        "descriptor shard header",
    )
    decoded_nbytes = header["decoded_nbytes"]
    if type(decoded_nbytes) is not int or decoded_nbytes < 0:
        raise DescriptorCacheV2Error("descriptor shard decoded byte census is invalid")
    decoded_contexts = _contexts_from_values(
        header["windows"],
        expected_count=row.window_count,
    )
    if (
        header["schema"] != SHARD_SCHEMA
        or header["source_binding"] != binding._value()
        or header["streams"] != list(STREAM_KINDS)
        or header["batch_input_sha256"] != batch_input_sha256
        or header["cache_key_sha256"] != row.cache_key_sha256
        or header["energy_floors_sha256"] != floors_sha256
        or header["morlet_oracle_sha256"] != MORLET_ORACLE_SHA256
        or decoded_contexts != contexts
        or header["edge_count"] != row.edge_count
        or decoded_nbytes != len(payload)
        or decoded_nbytes > bounds.max_decoded_shard_bytes
    ):
        raise DescriptorCacheV2Error("descriptor shard binding or census mismatch")
    items = header["arrays"]
    if type(items) is not list or len(items) != len(_ARRAY_NAMES):
        raise DescriptorCacheV2Error("descriptor shard array census is invalid")
    arrays: dict[str, np.ndarray] = {}
    cursor = 0
    for expected_name, item in zip(_ARRAY_NAMES, items, strict=True):
        metadata = _closed(
            item,
            {"dtype", "name", "nbytes", "offset", "sha256", "shape"},
            f"descriptor array {expected_name}",
        )
        if (
            metadata["name"] != expected_name
            or metadata["dtype"] != _EXPECTED_DTYPES[expected_name]
        ):
            raise DescriptorCacheV2Error("descriptor array name or dtype changed")
        shape = _shape_tuple(metadata["shape"], expected_name)
        dtype = np.dtype(_EXPECTED_DTYPES[expected_name])
        element_count = math.prod(shape)
        nbytes = element_count * dtype.itemsize
        if (
            type(metadata["offset"]) is not int
            or metadata["offset"] != cursor
            or type(metadata["nbytes"]) is not int
            or metadata["nbytes"] != nbytes
            or cursor + nbytes > len(payload)
        ):
            raise DescriptorCacheV2Error("descriptor array offsets are not gap-free")
        _lower_sha256(metadata["sha256"], f"{expected_name} sha256")
        selected = payload[cursor : cursor + nbytes]
        if _sha256(selected.tobytes()) != metadata["sha256"]:
            raise DescriptorCacheV2Error(
                f"descriptor array {expected_name} digest mismatch"
            )
        array = np.frombuffer(selected, dtype=dtype).reshape(shape)
        if array.flags.writeable:
            raise DescriptorCacheV2Error(
                "decoded descriptor array unexpectedly writeable"
            )
        arrays[expected_name] = array
        cursor += nbytes
    if cursor != len(payload):
        raise DescriptorCacheV2Error("descriptor shard contains trailing payload bytes")
    return arrays


def _validate_decoded_arrays(
    batch: PreparedGroupBatch,
    arrays: dict[str, np.ndarray],
    *,
    edge_count: int,
) -> None:
    expected = _expected_edge_census(batch)
    for name, expected_values in zip(_SHARED_ARRAYS, expected, strict=True):
        value = arrays[name]
        if (
            value.dtype != np.dtype(np.int64)
            or value.shape != (edge_count,)
            or not value.flags.c_contiguous
            or value.tobytes(order="C") != expected_values.tobytes(order="C")
        ):
            raise DescriptorCacheV2Error(
                "cached canonical edge census differs from batch"
            )
    for kind in STREAM_KINDS:
        for suffix in ("tokens_ij", "tokens_ji"):
            value = arrays[f"{kind}__{suffix}"]
            if (
                value.dtype != np.dtype(np.float32)
                or value.shape != (edge_count, BAND_COUNT, TOKEN_WIDTH)
                or not value.flags.c_contiguous
                or not bool(np.isfinite(value).all())
            ):
                raise DescriptorCacheV2Error(f"cached {kind} token tensor is invalid")
        support = arrays[f"{kind}__support_mask"]
        if (
            support.dtype != np.dtype(np.bool_)
            or support.shape != (edge_count, BAND_COUNT)
            or not support.flags.c_contiguous
        ):
            raise DescriptorCacheV2Error(f"cached {kind} support tensor is invalid")


@dataclass(frozen=True, slots=True)
class CachedDescriptorBatch:
    """One verified immutable shard supporting arbitrarily many stream replays."""

    cache_key_sha256: str
    batch_input_sha256: str
    energy_floors_sha256: str
    contexts: tuple[DescriptorWindowContext, ...]
    edge_count: int
    _arrays: tuple[tuple[str, np.ndarray], ...] = field(repr=False)
    _reader_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reader_token is not _CACHE_BATCH_TOKEN:
            raise DescriptorCacheV2Error(
                "cached descriptor batches must come from the verified reader"
            )
        _lower_sha256(self.cache_key_sha256, "cache_key_sha256")
        _lower_sha256(self.batch_input_sha256, "batch_input_sha256")
        _lower_sha256(self.energy_floors_sha256, "energy_floors_sha256")
        if (
            type(self.contexts) is not tuple
            or not self.contexts
            or any(
                type(context) is not DescriptorWindowContext
                for context in self.contexts
            )
        ):
            raise DescriptorCacheV2Error("cached descriptor contexts are invalid")
        if type(self.edge_count) is not int or self.edge_count < 1:
            raise DescriptorCacheV2Error("cached descriptor edge count is invalid")
        if (
            type(self._arrays) is not tuple
            or tuple(name for name, _ in self._arrays) != _ARRAY_NAMES
        ):
            raise DescriptorCacheV2Error("cached descriptor array census is invalid")
        for name, array in self._arrays:
            if (
                type(array) is not np.ndarray
                or array.dtype != np.dtype(_EXPECTED_DTYPES[name])
                or not array.flags.c_contiguous
                or array.flags.writeable
            ):
                raise DescriptorCacheV2Error(
                    "cached descriptor array storage is mutable"
                )

    def _mapping(self) -> dict[str, np.ndarray]:
        return dict(self._arrays)

    def stream(self, stream_kind: DescriptorStreamKind) -> CachedPairChunkStream:
        if stream_kind not in STREAM_KINDS:
            raise DescriptorCacheV2Error("unknown descriptor stream kind")
        return CachedPairChunkStream(self, stream_kind)

    def iter_chunks(
        self,
        stream_kind: DescriptorStreamKind,
        *,
        edge_chunk_size: int,
    ) -> Iterator[PairChunk]:
        return self.stream(stream_kind).iter_chunks(edge_chunk_size=edge_chunk_size)


@dataclass(frozen=True, slots=True)
class CachedPairChunkStream:
    """A re-iterable exact ``PairChunk`` source for the explicit model seam."""

    batch: CachedDescriptorBatch
    stream_kind: DescriptorStreamKind

    def __post_init__(self) -> None:
        if (
            type(self.batch) is not CachedDescriptorBatch
            or self.stream_kind not in STREAM_KINDS
        ):
            raise DescriptorCacheV2Error("cached PairChunk stream identity is invalid")

    def iter_chunks(self, *, edge_chunk_size: int) -> Iterator[PairChunk]:
        chunk_size = validate_edge_chunk_size(edge_chunk_size)
        arrays = self.batch._mapping()
        prefix = f"{self.stream_kind}__"
        for start in range(0, self.batch.edge_count, chunk_size):
            stop = min(self.batch.edge_count, start + chunk_size)
            yield PairChunk(
                np.ascontiguousarray(arrays["batch_indices"][start:stop]),
                np.ascontiguousarray(arrays["actor_i"][start:stop]),
                np.ascontiguousarray(arrays["actor_j"][start:stop]),
                np.ascontiguousarray(arrays[f"{prefix}tokens_ij"][start:stop]),
                np.ascontiguousarray(arrays[f"{prefix}tokens_ji"][start:stop]),
                np.ascontiguousarray(arrays[f"{prefix}support_mask"][start:stop]),
            )


class PeriodicDescriptorCacheWriterV2:
    """Write a fresh bounded private cache tree and publish its index last."""

    def __init__(
        self,
        root: str | Path,
        *,
        source_binding: DescriptorCacheSourceBinding,
        energy_floors: np.ndarray,
        bounds: DescriptorCacheBounds = DescriptorCacheBounds(),
    ) -> None:
        _require_little_endian()
        if type(source_binding) is not DescriptorCacheSourceBinding:
            raise TypeError("source_binding must be exact DescriptorCacheSourceBinding")
        if type(bounds) is not DescriptorCacheBounds:
            raise TypeError("bounds must be exact DescriptorCacheBounds")
        unresolved = Path(root)
        _reject_symlink_components(unresolved, "descriptor cache root")
        try:
            unresolved.mkdir(parents=False, exist_ok=False)
            shards = unresolved / "shards"
            shards.mkdir(exist_ok=False)
        except OSError as error:
            raise DescriptorCacheV2Error("descriptor cache root must be new") from error
        self.root = unresolved.resolve()
        self.shards = (self.root / "shards").resolve()
        self.source_binding = source_binding
        self.energy_floors = validate_energy_floors(energy_floors)
        self.energy_floors_sha256 = energy_floors_sha256(self.energy_floors)
        self.bounds = bounds
        self._rows: list[_IndexRow] = []
        self._keys: set[str] = set()
        self._total_bytes = 0
        self._finalized = False

    def add_batch(
        self,
        batch: PreparedGroupBatch,
        contexts: tuple[DescriptorWindowContext, ...],
    ) -> str:
        if self._finalized:
            raise DescriptorCacheV2Error("descriptor cache index is already finalized")
        checked = validate_prepared_group_batch(batch)
        checked_contexts = _validate_contexts(
            contexts,
            batch_size=checked.batch_size,
            binding=self.source_binding,
        )
        batch_digest = prepared_group_batch_sha256(checked)
        cache_key = _entry_key(
            binding=self.source_binding,
            contexts=checked_contexts,
            batch_input_sha256=batch_digest,
            floors_sha256=self.energy_floors_sha256,
        )
        if cache_key in self._keys:
            raise DescriptorCacheV2Error("descriptor cache repeats an exact entry key")
        if len(self._rows) >= self.bounds.max_shards:
            raise DescriptorCacheV2Error(
                "descriptor cache exceeds its shard-count bound"
            )
        expected_edge_count, _ = _preflight_descriptor_resources(
            checked,
            bounds=self.bounds,
            stored_bytes_before=self._total_bytes,
        )
        arrays = _collect_all_arrays(checked, self.energy_floors)
        edge_count = int(arrays["batch_indices"].shape[0])
        if edge_count != expected_edge_count:
            raise DescriptorCacheV2Error("descriptor shard preflight census changed")
        raw = _serialize_shard(
            binding=self.source_binding,
            contexts=checked_contexts,
            batch_input_sha256=batch_digest,
            floors_sha256=self.energy_floors_sha256,
            cache_key_sha256=cache_key,
            arrays=arrays,
            bounds=self.bounds,
        )
        if self._total_bytes + len(raw) > self.bounds.max_total_shard_bytes:
            raise DescriptorCacheV2Error(
                "descriptor cache exceeds its total byte bound"
            )
        relative = f"shards/{cache_key}.pdc2"
        _write_exclusive(self.root / relative, raw)
        first = checked_contexts[0]
        row = _IndexRow(
            batch_input_sha256=batch_digest,
            bytes=len(raw),
            cache_key_sha256=cache_key,
            edge_count=edge_count,
            epoch=first.epoch,
            path=relative,
            seed=first.seed,
            sha256=_sha256(raw),
            source_batch_sha256=first.source_batch_sha256,
            split=first.split,
            window_contexts_sha256=_contexts_sha256(checked_contexts),
            window_count=len(checked_contexts),
        )
        self._rows.append(row)
        self._keys.add(cache_key)
        self._total_bytes += len(raw)
        return cache_key

    def finalize(self) -> DescriptorCacheArtifact:
        if self._finalized:
            raise DescriptorCacheV2Error("descriptor cache index is already finalized")
        if not self._rows:
            raise DescriptorCacheV2Error(
                "descriptor cache cannot finalize without shards"
            )
        rows = sorted(self._rows, key=lambda row: row.cache_key_sha256)
        raw = _canonical_json(
            {
                "authority": AUTHORITY,
                "energy_floors_sha256": self.energy_floors_sha256,
                "environment_sha256": self.source_binding.environment_sha256,
                "external_authentication_asserted": False,
                "morlet_oracle_sha256": MORLET_ORACLE_SHA256,
                "prepared_manifest_sha256": (
                    self.source_binding.prepared_manifest_sha256
                ),
                "production": PRODUCTION,
                "result_claimed": RESULT_CLAIMED,
                "schema": INDEX_SCHEMA,
                "shards": [row._value() for row in rows],
                "source_tree_sha256": self.source_binding.source_tree_sha256,
                "streams": list(STREAM_KINDS),
                "total_shard_bytes": self._total_bytes,
            }
        )
        if len(raw) > self.bounds.max_index_bytes:
            raise DescriptorCacheV2Error(
                "descriptor cache index exceeds its byte bound"
            )
        index = self.root / "index.json"
        _write_exclusive(index, raw)
        self._finalized = True
        return DescriptorCacheArtifact(
            root=self.root,
            index_path=index,
            index_sha256=_sha256(raw),
            shard_count=len(rows),
            total_shard_bytes=self._total_bytes,
        )


class PeriodicDescriptorCacheV2:
    """Read and resolve exact immutable descriptor shards without fallback."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_index_sha256: str,
        expected_source_binding: DescriptorCacheSourceBinding,
        energy_floors: np.ndarray,
        bounds: DescriptorCacheBounds = DescriptorCacheBounds(),
    ) -> None:
        _require_little_endian()
        _lower_sha256(expected_index_sha256, "expected_index_sha256")
        if type(expected_source_binding) is not DescriptorCacheSourceBinding:
            raise TypeError(
                "expected_source_binding must be exact DescriptorCacheSourceBinding"
            )
        if type(bounds) is not DescriptorCacheBounds:
            raise TypeError("bounds must be exact DescriptorCacheBounds")
        unresolved = Path(root)
        _reject_symlink_components(unresolved, "descriptor cache root")
        try:
            directory = unresolved.resolve(strict=True)
        except OSError as error:
            raise DescriptorCacheV2Error("descriptor cache root is absent") from error
        if not directory.is_dir() or directory.is_symlink():
            raise DescriptorCacheV2Error(
                "descriptor cache root must be a non-symlink directory"
            )
        self.root = directory
        self.source_binding = expected_source_binding
        self.energy_floors = validate_energy_floors(energy_floors)
        self.energy_floors_sha256 = energy_floors_sha256(self.energy_floors)
        self.bounds = bounds
        index_raw = _read_owned(
            self.root / "index.json",
            maximum=bounds.max_index_bytes,
            label="descriptor cache index",
        )
        if _sha256(index_raw) != expected_index_sha256:
            raise DescriptorCacheV2Error("descriptor cache index digest mismatch")
        value = _closed(
            _parse_canonical_json(index_raw, "descriptor cache index"),
            {
                "authority",
                "energy_floors_sha256",
                "environment_sha256",
                "external_authentication_asserted",
                "morlet_oracle_sha256",
                "prepared_manifest_sha256",
                "production",
                "result_claimed",
                "schema",
                "shards",
                "source_tree_sha256",
                "streams",
                "total_shard_bytes",
            },
            "descriptor cache index",
        )
        if (
            type(value["authority"]) is not int
            or value["authority"] != AUTHORITY
            or value["production"] is not PRODUCTION
            or value["result_claimed"] is not RESULT_CLAIMED
            or value["external_authentication_asserted"] is not False
            or value["schema"] != INDEX_SCHEMA
            or value["streams"] != list(STREAM_KINDS)
            or value["energy_floors_sha256"] != self.energy_floors_sha256
            or value["morlet_oracle_sha256"] != MORLET_ORACLE_SHA256
            or value["prepared_manifest_sha256"]
            != expected_source_binding.prepared_manifest_sha256
            or value["source_tree_sha256"] != expected_source_binding.source_tree_sha256
            or value["environment_sha256"] != expected_source_binding.environment_sha256
        ):
            raise DescriptorCacheV2Error("descriptor cache index binding mismatch")
        rows_value = value["shards"]
        if (
            type(rows_value) is not list
            or not rows_value
            or len(rows_value) > bounds.max_shards
        ):
            raise DescriptorCacheV2Error("descriptor cache shard census is invalid")
        rows = tuple(_row_from_value(row, bounds=bounds) for row in rows_value)
        if tuple(row.cache_key_sha256 for row in rows) != tuple(
            sorted(row.cache_key_sha256 for row in rows)
        ):
            raise DescriptorCacheV2Error("descriptor cache index rows are not sorted")
        if len({row.cache_key_sha256 for row in rows}) != len(rows):
            raise DescriptorCacheV2Error("descriptor cache index repeats a key")
        total = sum(row.bytes for row in rows)
        if (
            type(value["total_shard_bytes"]) is not int
            or value["total_shard_bytes"] != total
            or total > bounds.max_total_shard_bytes
        ):
            raise DescriptorCacheV2Error(
                "descriptor cache total byte census is invalid"
            )
        self._rows = {row.cache_key_sha256: row for row in rows}
        self.index_sha256 = expected_index_sha256
        self._validate_tree_census()

    @property
    def cache_key_census(self) -> tuple[str, ...]:
        """Return the closed sorted key census from the authenticated index."""

        return tuple(self._rows)

    def _validate_tree_census(self) -> None:
        shards_path = self.root / "shards"
        _reject_symlink_components(shards_path, "descriptor shard directory")
        if not shards_path.is_dir() or shards_path.is_symlink():
            raise DescriptorCacheV2Error(
                "descriptor shard directory must be a non-symlink directory"
            )
        try:
            root_names: set[str] = set()
            for entry in os.scandir(self.root):
                root_names.add(entry.name)
                if len(root_names) > 2:
                    raise DescriptorCacheV2Error(
                        "descriptor cache root entries are not closed"
                    )
            shard_entries: list[os.DirEntry[str]] = []
            for entry in os.scandir(shards_path):
                shard_entries.append(entry)
                if len(shard_entries) > self.bounds.max_shards:
                    raise DescriptorCacheV2Error(
                        "descriptor shard directory exceeds its entry bound"
                    )
        except OSError as error:
            raise DescriptorCacheV2Error(
                "descriptor cache tree cannot be enumerated"
            ) from error
        if root_names != {"index.json", "shards"}:
            raise DescriptorCacheV2Error("descriptor cache root entries are not closed")
        if (
            len(shard_entries) != len(self._rows)
            or len(shard_entries) > self.bounds.max_shards
        ):
            raise DescriptorCacheV2Error("descriptor shard directory census mismatch")
        expected = {Path(row.path).name for row in self._rows.values()}
        observed = {entry.name for entry in shard_entries}
        if observed != expected or any(
            not entry.is_file(follow_symlinks=False) for entry in shard_entries
        ):
            raise DescriptorCacheV2Error(
                "descriptor shard directory entries are not closed"
            )

    def open_batch(
        self,
        batch: PreparedGroupBatch,
        contexts: tuple[DescriptorWindowContext, ...],
    ) -> CachedDescriptorBatch:
        self._validate_tree_census()
        checked = validate_prepared_group_batch(batch)
        checked_contexts = _validate_contexts(
            contexts,
            batch_size=checked.batch_size,
            binding=self.source_binding,
        )
        batch_digest = prepared_group_batch_sha256(checked)
        cache_key = _entry_key(
            binding=self.source_binding,
            contexts=checked_contexts,
            batch_input_sha256=batch_digest,
            floors_sha256=self.energy_floors_sha256,
        )
        row = self._rows.get(cache_key)
        if row is None:
            raise DescriptorCacheV2Miss("exact descriptor cache entry is absent")
        first = checked_contexts[0]
        if (
            row.batch_input_sha256 != batch_digest
            or row.source_batch_sha256 != first.source_batch_sha256
            or row.split != first.split
            or row.seed != first.seed
            or row.epoch != first.epoch
            or row.window_count != len(checked_contexts)
            or row.window_contexts_sha256 != _contexts_sha256(checked_contexts)
        ):
            raise DescriptorCacheV2Error(
                "descriptor index row differs from explicit request"
            )
        raw = _read_owned(
            self.root / row.path,
            maximum=self.bounds.max_shard_bytes,
            label="descriptor cache shard",
        )
        if len(raw) != row.bytes or _sha256(raw) != row.sha256:
            raise DescriptorCacheV2Error(
                "descriptor cache shard bytes differ from index"
            )
        arrays = _decode_shard(
            raw,
            row=row,
            binding=self.source_binding,
            contexts=checked_contexts,
            batch_input_sha256=batch_digest,
            floors_sha256=self.energy_floors_sha256,
            bounds=self.bounds,
        )
        _validate_decoded_arrays(checked, arrays, edge_count=row.edge_count)
        return CachedDescriptorBatch(
            cache_key_sha256=cache_key,
            batch_input_sha256=batch_digest,
            energy_floors_sha256=self.energy_floors_sha256,
            contexts=checked_contexts,
            edge_count=row.edge_count,
            _arrays=tuple((name, arrays[name]) for name in _ARRAY_NAMES),
            _reader_token=_CACHE_BATCH_TOKEN,
        )


__all__ = [
    "AUTHORITY",
    "CachedDescriptorBatch",
    "CachedPairChunkStream",
    "DescriptorCacheArtifact",
    "DescriptorCacheBounds",
    "DescriptorCacheSourceBinding",
    "DescriptorCacheV2Error",
    "DescriptorCacheV2Miss",
    "DescriptorStreamKind",
    "DescriptorWindowContext",
    "ENTRY_SCHEMA",
    "INDEX_SCHEMA",
    "PRODUCTION",
    "PeriodicDescriptorCacheV2",
    "PeriodicDescriptorCacheWriterV2",
    "RESULT_CLAIMED",
    "SHARD_SCHEMA",
    "STATUS",
    "STREAM_KINDS",
    "energy_floors_sha256",
    "iter_uncached_pair_chunks",
    "prepared_group_batch_sha256",
]
