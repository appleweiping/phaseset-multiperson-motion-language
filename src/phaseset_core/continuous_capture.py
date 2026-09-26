"""Capture-wide coordinates and resampling, without ten-second state resets.

The original window API remains unchanged. This V2 seam preserves absolute
time, applies the same source-window missing-data decisions, and carries one
group origin/yaw across every accepted window. Rejected intervals stay on the
timeline as exact +0 with false masks; they are never concatenated away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .contracts import group_commitment
from .pipeline import (
    PrivateCaptureArrays,
    PreparationResourceLimit,
    PreparedGroupSample,
    _capture_lineage_sha256,
    _window_lineage_sha256,
)
from .preprocessing import (
    SOURCE_WINDOW_FRAMES,
    TARGET_WINDOW_FRAMES,
    PreprocessingError,
    WindowDecision,
    assess_missing_window,
    canonicalize_group,
    design_antialias_fir,
    interpolate_accepted_gaps,
)


@dataclass(frozen=True)
class PreparedContinuousCapture:
    skeletons: np.ndarray = field(repr=False)  # float32 [K,T,22,3], absolute 20 Hz
    track_mask: np.ndarray = field(repr=False)  # bool [K,T,22]
    actor_commitments: tuple[bytes, ...]
    source_sha256: str
    group_commitment: bytes
    decisions: tuple[WindowDecision, ...]
    trailing_source_frames: int
    reference_frame: int
    group_center: np.ndarray
    augmentation_yaw: float

    def __post_init__(self) -> None:
        if (
            type(self.skeletons) is not np.ndarray
            or self.skeletons.dtype != np.dtype(np.float32)
            or self.skeletons.ndim != 4
            or self.skeletons.shape[0] < 2
            or self.skeletons.shape[2:] != (22, 3)
            or not self.skeletons.flags.c_contiguous
        ):
            raise PreprocessingError("continuous skeletons must be float32 [K>=2,T,22,3]")
        if (
            type(self.track_mask) is not np.ndarray
            or self.track_mask.dtype != np.dtype(np.bool_)
            or self.track_mask.shape != self.skeletons.shape[:-1]
            or not self.track_mask.flags.c_contiguous
        ):
            raise PreprocessingError("continuous tracking must be bool [K,T,22]")
        if not bool(np.isfinite(self.skeletons).all()):
            raise PreprocessingError("continuous skeletons must be finite")
        if bool((self.skeletons[~self.track_mask].view(np.uint32) != 0).any()):
            raise PreprocessingError("untracked continuous skeletons must be exact +0")
        if (
            type(self.decisions) is not tuple
            or not self.decisions
            or any(type(decision) is not WindowDecision for decision in self.decisions)
            or self.frame_count != len(self.decisions) * TARGET_WINDOW_FRAMES
            or not any(decision.accepted for decision in self.decisions)
        ):
            raise PreprocessingError("window decisions must cover the complete timeline")
        if (
            type(self.actor_commitments) is not tuple
            or len(self.actor_commitments) != self.actor_count
            or tuple(sorted(self.actor_commitments)) != self.actor_commitments
            or group_commitment(self.actor_commitments) != self.group_commitment
        ):
            raise PreprocessingError("continuous actor lineage must be canonical")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise PreprocessingError("source_sha256 must be lowercase SHA-256 hex")
        if type(self.trailing_source_frames) is not int or not 0 <= self.trailing_source_frames < 300:
            raise PreprocessingError("trailing source frames must be in [0,300)")
        if (
            type(self.reference_frame) is not int
            or not 0 <= self.reference_frame < self.frame_count
            or not bool(self.track_mask[:, self.reference_frame].all())
        ):
            raise PreprocessingError("capture origin must use one fully observed target frame")
        if (
            type(self.group_center) is not np.ndarray
            or self.group_center.shape != (3,)
            or not bool(np.isfinite(self.group_center).all())
            or isinstance(self.augmentation_yaw, bool)
            or not isinstance(self.augmentation_yaw, (int, float))
            or not math.isfinite(self.augmentation_yaw)
        ):
            raise PreprocessingError("capture transform must be finite")
        self.skeletons.setflags(write=False)
        self.track_mask.setflags(write=False)
        self.group_center.setflags(write=False)

    @property
    def actor_count(self) -> int:
        return self.skeletons.shape[0]

    @property
    def frame_count(self) -> int:
        return self.skeletons.shape[1]

    def windows(self) -> tuple[PreparedGroupSample, ...]:
        """Base-model views of accepted windows, all in this capture's frame."""
        return tuple(
            PreparedGroupSample(
                skeletons=self.skeletons[:, index * 200 : (index + 1) * 200],
                track_mask=self.track_mask[:, index * 200 : (index + 1) * 200],
                actor_commitments=self.actor_commitments,
                group_commitment=self.group_commitment,
                source_sha256=self.source_sha256,
                window_sha256=_window_lineage_sha256(self.source_sha256, index * 300),
                source_start_frame=index * 300,
                augmentation_yaw=self.augmentation_yaw,
                decision=decision,
            )
            for index, decision in enumerate(self.decisions)
            if decision.accepted
        )


def _capture_filter(values: np.ndarray, kernel: np.ndarray, chunk_frames: int) -> np.ndarray:
    """Use the full FIR halo at chunk seams; pad only true capture endpoints."""
    half = kernel.size // 2
    flat = values.reshape(len(values), -1)
    padded = np.pad(flat, ((half, half), (0, 0)), mode="edge")
    filtered = np.empty_like(flat)
    for start in range(0, len(values), chunk_frames):
        stop = min(start + chunk_frames, len(values))
        windows = np.lib.stride_tricks.sliding_window_view(
            padded[start : stop + kernel.size - 1], kernel.size, axis=0
        )
        filtered[start:stop] = np.einsum("tcf,f->tc", windows, kernel, optimize=False)
    return filtered.reshape(values.shape)


def prepare_continuous_capture(
    capture: PrivateCaptureArrays,
    *,
    augmentation_yaw: float = 0.0,
    resample_chunk_frames: int = 1024,
    max_preparation_bytes: int = 1_073_741_824,
) -> PreparedContinuousCapture:
    """Prepare all complete ten-second intervals, retaining one shared frame.

    Interpolation eligibility is still assessed per original 300-frame window,
    including the 95%, seven-frame and bounded-gap rules. Anti-alias filtering
    spans adjacent accepted windows. A target touching a rejected interval's
    FIR footprint is conservatively masked; no long gap is imputed. Original
    short-gap observation masks are retained. No participant is removed.
    """
    if type(capture) is not PrivateCaptureArrays:
        raise TypeError("capture must be exactly PrivateCaptureArrays")
    for name, value in (
        ("resample_chunk_frames", resample_chunk_frames),
        ("max_preparation_bytes", max_preparation_bytes),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive exact int")
    full_windows = capture.frame_count // SOURCE_WINDOW_FRAMES
    if full_windows == 0:
        raise PreprocessingError("capture has no complete source window")
    source_length = full_windows * SOURCE_WINDOW_FRAMES
    # Conservative bound for the linear-sized preparation buffers, independent
    # of the all-edge budget checked by the physical relation frontend.
    if source_length * capture.actor_count * 66 * 64 > max_preparation_bytes:
        raise PreparationResourceLimit("RESOURCE_LIMIT: continuous preparation exceeds budget")
    order = sorted(range(capture.actor_count), key=lambda i: capture.actor_commitments[i])
    motion = capture.joints[:source_length, order, :22]
    joint_mask = capture.track_mask[:source_length, order, :22]
    observed = joint_mask[..., 0]
    if not np.array_equal(joint_mask, np.broadcast_to(observed[..., None], joint_mask.shape)):
        raise PreprocessingError("body-22 tracking must be uniform within each actor-frame")
    decisions = tuple(
        assess_missing_window(observed[start : start + SOURCE_WINDOW_FRAMES])
        for start in range(0, source_length, SOURCE_WINDOW_FRAMES)
    )
    if not any(decision.accepted for decision in decisions):
        raise PreprocessingError("capture has no accepted source window")
    work = np.zeros(motion.shape, dtype=np.float64)
    eligible = np.zeros(source_length, dtype=np.bool_)
    for index, decision in enumerate(decisions):
        if decision.accepted:
            start, stop = index * SOURCE_WINDOW_FRAMES, (index + 1) * SOURCE_WINDOW_FRAMES
            work[start:stop] = interpolate_accepted_gaps(
                motion[start:stop], observed_mask=observed[start:stop]
            ).values
            eligible[start:stop] = True
    kernel = design_antialias_fir()
    filtered = _capture_filter(work, kernel, resample_chunk_frames)
    output_length = full_windows * TARGET_WINDOW_FRAMES
    # Exact rational source locations: even target frames are source integers,
    # odd frames are half-integers. Avoid floating-point near-integer mask drift.
    twice_positions = np.arange(output_length, dtype=np.int64) * 3
    lower = twice_positions // 2
    fractional = twice_positions % 2 != 0
    upper = np.minimum(lower + 1, source_length - 1)
    fraction = (fractional.astype(np.float64) * 0.5).reshape(-1, 1, 1, 1)
    values = filtered[lower] * (1.0 - fraction) + filtered[upper] * fraction
    mask = observed[lower].copy()
    mask[fractional] &= observed[upper[fractional]]
    half = kernel.size // 2
    footprint_ok = np.lib.stride_tricks.sliding_window_view(
        np.pad(eligible, (half, half), mode="edge"), kernel.size
    ).all(axis=-1)
    target_ok = footprint_ok[lower].copy()
    target_ok[fractional] &= footprint_ok[upper[fractional]]
    mask &= target_ok[:, None]
    canonical = canonicalize_group(
        values,
        np.zeros((output_length, capture.actor_count), dtype=np.float64),
        observed_mask=mask,
        augmentation_yaw=augmentation_yaw,
    )
    skeletons = np.ascontiguousarray(
        np.where(mask[..., None, None], canonical.positions, 0.0).transpose(1, 0, 2, 3),
        dtype=np.float32,
    )
    track_mask = np.ascontiguousarray(np.broadcast_to(mask.T[..., None], skeletons.shape[:-1]))
    skeletons.setflags(write=False)
    track_mask.setflags(write=False)
    return PreparedContinuousCapture(
        skeletons=skeletons,
        track_mask=track_mask,
        actor_commitments=tuple(capture.actor_commitments[index] for index in order),
        source_sha256=_capture_lineage_sha256(capture),
        group_commitment=capture.group_commitment,
        decisions=decisions,
        trailing_source_frames=capture.frame_count - source_length,
        reference_frame=canonical.reference_frame,
        group_center=canonical.group_center,
        augmentation_yaw=canonical.augmentation_yaw,
    )
