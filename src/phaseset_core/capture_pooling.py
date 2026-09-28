"""Deterministic capture-level pooling of already encoded motion windows.

Window identity is independent of actor-set identity. This numeric seam does
not encode motion, score a gallery, establish participant identity, select
windows, or authorize access to sealed evaluation data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

EMBEDDING_DIM = 512
BAND_COUNT = 6
SOURCE_WINDOW_FRAMES = 300
DEFAULT_MAX_ENCODED_BYTES = 256 * 1024 * 1024


class CapturePoolingError(ValueError):
    """The supplied encoded windows differ from the complete capture plan."""


class CapturePoolingResourceLimit(CapturePoolingError):
    """An explicit memory bound was exceeded; no windows were sampled."""


def _commitment(value: object) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise CapturePoolingError("identity must be exact bytes[32]")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class CaptureWindowPlan:
    """The complete accepted-window census, frozen before any model score.

    Rejected windows and trailing source frames belong to the upstream
    preprocessing census. This plan may not be reconstructed by filtering
    available model outputs. A capture with no accepted window is ineligible.
    """

    capture_commitment: bytes
    window_commitments: tuple[bytes, ...]
    source_start_frames: tuple[int, ...]

    def __post_init__(self) -> None:
        _commitment(self.capture_commitment)
        if type(self.window_commitments) is not tuple or not self.window_commitments:
            raise CapturePoolingError("a capture must have its complete window census")
        for value in self.window_commitments:
            _commitment(value)
        if len(set(self.window_commitments)) != len(self.window_commitments):
            raise CapturePoolingError("window identities must be unique")
        if (
            type(self.source_start_frames) is not tuple
            or len(self.source_start_frames) != len(self.window_commitments)
            or any(
                type(value) is not int
                or not 0 <= value < 2**64
                or value % SOURCE_WINDOW_FRAMES
                for value in self.source_start_frames
            )
            or tuple(sorted(set(self.source_start_frames))) != self.source_start_frames
        ):
            raise CapturePoolingError("window starts must be ordered nonoverlapping 10s grid positions")

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            {
                "capture_commitment": self.capture_commitment.hex(),
                "schema": "phaseset-capture-window-plan-v1",
                "source_start_frames": self.source_start_frames,
                "window_commitments": tuple(value.hex() for value in self.window_commitments),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()

    def __repr__(self) -> str:
        return f"CaptureWindowPlan(windows={len(self.window_commitments)}, <identities hidden>)"


@dataclass(frozen=True, slots=True, repr=False)
class CapturePooledEncoding:
    """One unnormalized capture encoding; scoring normalizes after pooling."""

    capture_commitment: bytes
    plan_sha256: str
    window_count: int
    base_embedding: np.ndarray
    tokens: np.ndarray | None
    band_mask: np.ndarray | None
    valid_window_count: np.ndarray | None

    def __repr__(self) -> str:
        return (
            f"CapturePooledEncoding(windows={self.window_count}, "
            f"periodic={self.tokens is not None}, <identity/features hidden>)"
        )


def _require_array(value: object, dtype: np.dtype, shape: tuple[int, ...], label: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise CapturePoolingError(f"{label} dtype, shape, or contiguous layout is invalid")
    return value


def _tree_sum(values: np.ndarray) -> np.ndarray:
    """Fixed adjacent binary tree in float64, carrying an odd final node."""
    level = np.array(values, dtype=np.float64, order="C", copy=True)
    while len(level) > 1:
        pairs = len(level) // 2
        next_level = np.empty(((len(level) + 1) // 2, *level.shape[1:]), dtype=np.float64)
        np.add(level[: 2 * pairs : 2], level[1 : 2 * pairs : 2], out=next_level[:pairs])
        if len(level) % 2:
            next_level[-1] = level[-1]
        level = next_level
    return level[0]


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def pool_capture_windows(
    plan: CaptureWindowPlan,
    *,
    window_commitments: tuple[bytes, ...],
    base_embeddings: np.ndarray,
    tokens: np.ndarray | None = None,
    band_mask: np.ndarray | None = None,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
) -> CapturePooledEncoding:
    """Pool exactly every planned window, independent of input row ordering.

    Inputs are already encoded CPU float32 features. Canonical source-time
    order fixes the float64 reduction tree. Base embeddings average over all
    accepted windows. Each periodic band averages over its valid windows;
    an unsupported capture band has exact positive-zero output. No per-window
    normalization or score averaging is performed.

    The resource bound counts supplied encoded-array bytes, not total process
    RSS. Temporary float64 reduction storage is O(W*D), never actor-pair dense.
    """

    if type(plan) is not CaptureWindowPlan:
        raise CapturePoolingError("plan must be exact CaptureWindowPlan")
    if type(max_encoded_bytes) is not int or max_encoded_bytes < 1:
        raise CapturePoolingError("max_encoded_bytes must be a positive integer")
    if type(window_commitments) is not tuple:
        raise CapturePoolingError("window identities must be a tuple")
    for value in window_commitments:
        _commitment(value)
    count = len(plan.window_commitments)
    if (
        len(window_commitments) != count
        or len(set(window_commitments)) != count
        or set(window_commitments) != set(plan.window_commitments)
    ):
        raise CapturePoolingError("encoded windows must exactly match the complete capture census")
    base = _require_array(base_embeddings, np.dtype(np.float32), (count, EMBEDDING_DIM), "base")
    if (tokens is None) != (band_mask is None):
        raise CapturePoolingError("periodic tokens and mask must be supplied together")
    periodic = None
    mask = None
    encoded_bytes = base.nbytes
    if tokens is not None:
        periodic = _require_array(tokens, np.dtype(np.float32), (count, BAND_COUNT, EMBEDDING_DIM), "tokens")
        mask = _require_array(band_mask, np.dtype(np.bool_), (count, BAND_COUNT), "band_mask")
        encoded_bytes += periodic.nbytes + mask.nbytes
    if encoded_bytes > max_encoded_bytes:
        raise CapturePoolingResourceLimit("RESOURCE_LIMIT: encoded capture exceeds the byte budget")
    if not np.isfinite(base).all() or (periodic is not None and not np.isfinite(periodic).all()):
        raise CapturePoolingError("encoded features must be finite")
    if periodic is not None:
        invalid = periodic[~mask]
        if np.any(invalid != 0.0) or np.any(np.signbit(invalid)):
            raise CapturePoolingError("unsupported window bands must have exact positive-zero tokens")
    positions = {value: index for index, value in enumerate(window_commitments)}
    order = np.array([positions[value] for value in plan.window_commitments], dtype=np.int64)
    base_mean = np.ascontiguousarray((_tree_sum(base[order]) / count).astype(np.float32))
    pooled_tokens = None
    pooled_mask = None
    valid_counts = None
    if periodic is not None:
        canonical_mask = mask[order]
        valid_counts = canonical_mask.sum(axis=0, dtype=np.int64)
        pooled_mask = np.ascontiguousarray(valid_counts > 0)
        denominator = np.maximum(valid_counts, 1)[:, None]
        pooled_tokens = np.ascontiguousarray((_tree_sum(periodic[order]) / denominator).astype(np.float32))
        pooled_tokens[~pooled_mask] = np.float32(0.0)
        _readonly(pooled_tokens)
        _readonly(pooled_mask)
        _readonly(valid_counts)
    return CapturePooledEncoding(
        capture_commitment=plan.capture_commitment,
        plan_sha256=plan.sha256,
        window_count=count,
        base_embedding=_readonly(base_mean),
        tokens=pooled_tokens,
        band_mask=pooled_mask,
        valid_window_count=valid_counts,
    )
