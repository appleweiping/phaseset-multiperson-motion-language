"""Signal-level mean/difference DCT edge field for the PhaseSet-V2 A6 control.

The physical input is the shared-coordinate signed joint velocity, never a
Morlet response or separate endpoint self-powers. DCT energy floors are an
explicit, separately fitted training-population input; Morlet floors are not
interchangeable. This module does not fit or admit those floors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continuous_capture import PreparedContinuousCapture, physical_view_sha256
from .dct_calibration import DctFloorReceipt
from .directional_phase import (
    ACTOR_FEATURE_DIM,
    PHASE_FEATURE_DIM,
    DirectionalPhaseField,
    LocalPhaseConfig,
    LocalPairChunk,
    _patch_intervals,
    _readonly,
    _signed_velocity_arrays,
    canonical_edge_pairs,
    true_mean_difference_dct_features,
)
from .periodic import ResourceLimitError


def require_dct_working_budget(actor_count: int, frame_count: int, config: LocalPhaseConfig) -> None:
    """Reject a complete A6 timeline before loading or encoding it."""
    if (
        type(actor_count) is not int
        or actor_count < 2
        or type(frame_count) is not int
        or frame_count < 2
        or type(config) is not LocalPhaseConfig
    ):
        raise ValueError("A6 working-set preflight requires valid capture dimensions and config")
    if actor_count * frame_count * 66 * 128 > config.max_response_bytes:
        raise MemoryError("RESOURCE_LIMIT: A6 full-timeline DCT working set exceeds budget")


@dataclass(frozen=True)
class LocalDctPairChunk(LocalPairChunk):
    """The legacy ``phase_mask`` slot means calibrated DCT evidence here."""

    def reverse_features(self) -> np.ndarray:
        """DCT mean/difference energies are symmetric; only endpoint powers swap."""
        reversed_values = self.features.copy()
        reversed_values[..., 3] = self.features[..., 4]
        reversed_values[..., 4] = self.features[..., 3]
        reversed_values[reversed_values == 0.0] = 0.0
        return reversed_values


@dataclass(frozen=True)
class DctViewContext:
    """A6 lineage/patch context; it contains no Morlet response or cache."""

    config: LocalPhaseConfig
    intervals: tuple[tuple[int, int], ...]
    source_sha256: str
    actor_commitments: tuple[bytes, ...]
    physical_view_sha256: str
    dct_floor_receipt_sha256: str

    @classmethod
    def from_capture(
        cls,
        capture: PreparedContinuousCapture,
        config: LocalPhaseConfig,
        dct_floor_receipt: DctFloorReceipt,
    ) -> DctViewContext:
        if type(capture) is not PreparedContinuousCapture or type(config) is not LocalPhaseConfig:
            raise TypeError("A6 context requires an exact prepared capture and config")
        if type(dct_floor_receipt) is not DctFloorReceipt:
            raise TypeError("A6 context requires a typed DCT floor receipt")
        dct_floor_receipt.require_config(config)
        if not 2 <= config.patch_frames <= 40:
            raise ValueError("A6 context patch length must be in [2,40]")
        require_dct_working_budget(capture.actor_count, capture.frame_count, config)
        return cls(
            config,
            _patch_intervals(capture.frame_count, config),
            capture.source_sha256,
            capture.actor_commitments,
            physical_view_sha256(capture),
            dct_floor_receipt.sha256,
        )

    @property
    def actor_count(self) -> int:
        return len(self.actor_commitments)


@dataclass(frozen=True)
class DctRelationField:
    """Complete-capture A6 field with bounded, recomputed unordered edge chunks.

    Every actor feature and root is recomputed from the same body22 capture as
    the DCT velocities. A lightweight A6 context supplies frozen patch and
    source/view lineage without loading Morlet cache. The legacy phase-field
    context remains accepted for data-free compatibility tests only; neither
    path retains a Morlet response in this A6 field.
    """

    config: LocalPhaseConfig
    intervals: tuple[tuple[int, int], ...]
    velocities: np.ndarray  # float64 [K,T,66]
    velocity_mask: np.ndarray  # bool [K,T,66]
    actor_features: np.ndarray  # float32 [K,P,132]
    actor_patch_mask: np.ndarray  # bool [K,P]
    root_positions: np.ndarray  # float32 [K,P,3]
    root_patch_mask: np.ndarray  # bool [K,P]
    dct_floor_receipt: DctFloorReceipt  # declared, independently fitted population
    source_sha256: str

    @classmethod
    def from_capture(
        cls,
        capture: PreparedContinuousCapture,
        phase_field: DirectionalPhaseField | DctViewContext,
        *,
        dct_floor_receipt: DctFloorReceipt,
    ) -> DctRelationField:
        if type(capture) is not PreparedContinuousCapture or type(phase_field) not in (
            DirectionalPhaseField,
            DctViewContext,
        ):
            raise TypeError("A6 requires an admitted capture and typed relation context")
        if (
            phase_field.actor_count != capture.actor_count
            or phase_field.intervals[-1][1] != capture.frame_count
            or not 2 <= phase_field.config.patch_frames <= 40
            or phase_field.source_sha256 != capture.source_sha256
            or phase_field.actor_commitments != capture.actor_commitments
            or phase_field.physical_view_sha256 != physical_view_sha256(capture)
        ):
            raise ValueError("A6 source/actor/timeline/patch binding differs from admitted capture")
        if type(dct_floor_receipt) is not DctFloorReceipt:
            raise TypeError("A6 requires a typed DCT floor receipt")
        dct_floor_receipt.require_config(phase_field.config)
        if (
            type(phase_field) is DctViewContext
            and phase_field.dct_floor_receipt_sha256 != dct_floor_receipt.sha256
        ):
            raise ValueError("A6 context DCT floor receipt differs from the model")
        # The legacy config's response-byte gate is the shared physical
        # working-set budget. Count complete signed-velocity intermediates
        # before materializing any full-timeline float64 arrays.
        require_dct_working_budget(capture.actor_count, capture.frame_count, phase_field.config)
        values, masks = _signed_velocity_arrays(
            capture.skeletons[None],
            capture.track_mask[None],
            np.ones((1, capture.frame_count), dtype=np.bool_),
            np.ones((1, capture.actor_count), dtype=np.bool_),
        )
        shape = (capture.actor_count, capture.frame_count, 66)
        velocities = _readonly(np.ascontiguousarray(values[0].reshape(shape)))
        velocity_mask = _readonly(np.ascontiguousarray(masks[0].reshape(shape)))
        intervals = _patch_intervals(capture.frame_count, phase_field.config)
        if intervals != phase_field.intervals:
            raise ValueError("A6 physical patch intervals differ from the admitted view")
        actor_features = np.zeros(
            (capture.actor_count, len(intervals), ACTOR_FEATURE_DIM), dtype=np.float32
        )
        actor_mask = np.zeros((capture.actor_count, len(intervals)), dtype=np.bool_)
        roots = np.zeros((capture.actor_count, len(intervals), 3), dtype=np.float32)
        root_mask = np.zeros_like(actor_mask)
        for patch, (start, stop) in enumerate(intervals):
            local_values = velocities[:, start:stop]
            local_mask = velocity_mask[:, start:stop]
            denominator = local_mask.sum(axis=1).clip(min=1)
            mean = local_values.sum(axis=1) / denominator
            rms = np.sqrt((local_values**2).sum(axis=1) / denominator)
            actor_features[:, patch] = np.concatenate((mean, rms), axis=-1)
            actor_mask[:, patch] = local_mask.any(axis=(1, 2))
            pelvis_mask = capture.track_mask[:, start:stop, 0]
            pelvis_values = capture.skeletons[:, start:stop, 0].astype(np.float64)
            roots[:, patch] = (
                np.where(pelvis_mask[..., None], pelvis_values, 0.0).sum(axis=1)
                / pelvis_mask.sum(axis=1).clip(min=1)[:, None]
            )
            root_mask[:, patch] = pelvis_mask.any(axis=1)
        return cls(
            phase_field.config,
            intervals,
            velocities,
            velocity_mask,
            _readonly(actor_features),
            _readonly(actor_mask),
            _readonly(roots),
            _readonly(root_mask),
            dct_floor_receipt,
            capture.source_sha256,
        )

    def __post_init__(self) -> None:
        if (
            type(self.config) is not LocalPhaseConfig
            or type(self.intervals) is not tuple
            or not self.intervals
            or type(self.velocities) is not np.ndarray
            or self.velocities.dtype != np.dtype(np.float64)
            or self.velocities.ndim != 3
            or self.velocities.shape[0] < 2
            or self.velocities.shape[1:] != (self.intervals[-1][1], 66)
            or type(self.velocity_mask) is not np.ndarray
            or self.velocity_mask.dtype != np.dtype(np.bool_)
            or self.velocity_mask.shape != self.velocities.shape
            or type(self.actor_features) is not np.ndarray
            or self.actor_features.shape != (self.actor_count, self.patch_count, ACTOR_FEATURE_DIM)
            or self.actor_features.dtype != np.dtype(np.float32)
            or type(self.actor_patch_mask) is not np.ndarray
            or self.actor_patch_mask.shape != (self.actor_count, self.patch_count)
            or self.actor_patch_mask.dtype != np.dtype(np.bool_)
            or type(self.root_positions) is not np.ndarray
            or self.root_positions.shape != (self.actor_count, self.patch_count, 3)
            or self.root_positions.dtype != np.dtype(np.float32)
            or type(self.root_patch_mask) is not np.ndarray
            or self.root_patch_mask.shape != (self.actor_count, self.patch_count)
            or self.root_patch_mask.dtype != np.dtype(np.bool_)
            or not bool(np.isfinite(self.velocities).all())
            or not bool(np.isfinite(self.actor_features).all())
            or not bool(np.isfinite(self.root_positions).all())
            or bool(np.any(self.velocities[~self.velocity_mask].view(np.uint64) != 0))
            or type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("A6 signed velocity field or source identity is invalid")
        if type(self.dct_floor_receipt) is not DctFloorReceipt:
            raise TypeError("A6 requires a typed DCT floor receipt")
        self.dct_floor_receipt.require_config(self.config)
        self.velocities.setflags(write=False)
        self.velocity_mask.setflags(write=False)
        self.actor_features.setflags(write=False)
        self.actor_patch_mask.setflags(write=False)
        self.root_positions.setflags(write=False)
        self.root_patch_mask.setflags(write=False)

    @property
    def actor_count(self) -> int:
        return int(self.velocities.shape[0])

    @property
    def patch_count(self) -> int:
        return len(self.intervals)

    @property
    def pair_count(self) -> int:
        return self.actor_count * (self.actor_count - 1) // 2


def local_dct_pair_chunk(field: DctRelationField, start: int, stop: int) -> LocalDctPairChunk:
    """Recompute every edge/patch from the full signed timeline, without K×K storage."""
    if type(field) is not DctRelationField:
        raise TypeError("A6 pair chunk requires a DctRelationField")
    if field.pair_count > field.config.edge_budget:
        raise ResourceLimitError(
            required_edges=field.pair_count, edge_budget=field.config.edge_budget
        )
    pairs = canonical_edge_pairs(field.actor_count, start, stop)
    features = np.zeros((len(pairs), field.patch_count, 6, PHASE_FEATURE_DIM), dtype=np.float64)
    evidence = np.zeros(features.shape[:-1], dtype=np.bool_)
    support = np.zeros_like(evidence)
    log_floors = np.log1p(field.dct_floor_receipt.floors)
    for edge, (left, right) in enumerate(pairs):
        for patch, (window_start, window_stop) in enumerate(field.intervals):
            values, observed = true_mean_difference_dct_features(
                field.velocities[left, window_start:window_stop],
                field.velocities[right, window_start:window_stop],
                field.velocity_mask[left, window_start:window_stop],
                field.velocity_mask[right, window_start:window_stop],
            )
            valid = observed & (values[:, 3] > log_floors) & (values[:, 4] > log_floors)
            features[edge, patch] = values
            features[edge, patch, :, 6] = valid.astype(np.float64)
            evidence[edge, patch] = valid
            support[edge, patch] = observed
    features[features == 0.0] = 0.0
    return LocalDctPairChunk(
        _readonly(pairs), _readonly(features), _readonly(evidence), _readonly(support)
    )
