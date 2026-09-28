"""Direction-preserving, local six-band relations for PhaseSet-V2.

Coordinates are shared world coordinates. Slot zero is pelvis velocity; the
remaining 21 slots are joint velocities relative to that actor's pelvis.
This convention does not identify anatomical mirror gestures as antiphase.
Physical features are fixed preprocessing, independent of text or actor IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from .contracts import PreparedGroupBatch, validate_prepared_group_batch
from .continuous_capture import PreparedContinuousCapture, physical_view_sha256
from .morlet import MORLET_FREQUENCIES_HZ, SAMPLE_RATE_HZ, morlet_kernel_bank
from .periodic import ResourceLimitError, validate_energy_floors


PHASE_FEATURE_DIM = 7
ACTOR_FEATURE_DIM = 132


@dataclass(frozen=True)
class LocalPhaseConfig:
    patch_frames: int = 40
    hop_frames: int = 20
    epsilon: float = 1e-12
    edge_budget: int = 32_768
    max_response_bytes: int = 1_073_741_824
    convolution_chunk_frames: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "patch_frames",
            "hop_frames",
            "edge_budget",
            "max_response_bytes",
            "convolution_chunk_frames",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class DirectionalPhaseField:
    """One canonical group; response memory grows with actors, not pairs."""

    responses: tuple[np.ndarray, ...]  # six [K,N_b,22,3] complex128 arrays
    response_masks: tuple[np.ndarray, ...]  # six [K,N_b,22,3] masks
    response_centers: tuple[np.ndarray, ...]
    intervals: tuple[tuple[int, int], ...]
    actor_features: np.ndarray  # [K,P,132], signed mean and per-channel RMS
    actor_patch_mask: np.ndarray  # [K,P]
    root_positions: np.ndarray  # [K,P,3]
    root_patch_mask: np.ndarray  # [K,P]
    energy_floors: np.ndarray
    config: LocalPhaseConfig
    source_sha256: str | None = None
    actor_commitments: tuple[bytes, ...] | None = None
    physical_view_sha256: str | None = None

    @property
    def actor_count(self) -> int:
        return int(self.actor_features.shape[0])

    @property
    def patch_count(self) -> int:
        return len(self.intervals)

    @property
    def pair_count(self) -> int:
        return self.actor_count * (self.actor_count - 1) // 2


@dataclass(frozen=True)
class LocalPairChunk:
    endpoints: np.ndarray  # [E,2], unordered canonical pairs
    features: np.ndarray  # [E,P,6,7]
    phase_mask: np.ndarray  # [E,P,6]
    support_mask: np.ndarray  # [E,P,6]; low coherence does not remove edges

    def reverse_features(self) -> np.ndarray:
        """Reverse both endpoints: conjugate C and exchange endpoint energies."""
        reversed_values = self.features.copy()
        reversed_values[..., 1] *= -1.0
        reversed_values[..., 3] = self.features[..., 4]
        reversed_values[..., 4] = self.features[..., 3]
        reversed_values[reversed_values == 0.0] = 0.0
        return reversed_values


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def signed_joint_velocity(batch: PreparedGroupBatch) -> tuple[np.ndarray, np.ndarray]:
    """Preserve signed XYZ; a velocity needs both frames and the pelvis tracked."""
    checked = validate_prepared_group_batch(batch)
    return _signed_velocity_arrays(
        checked.skeletons, checked.track_mask, checked.frame_mask, checked.actor_mask
    )


def _signed_velocity_arrays(
    skeletons: np.ndarray,
    track_mask: np.ndarray,
    frame_mask: np.ndarray,
    actor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = skeletons.astype(np.float64)
    values = np.zeros_like(coordinates)
    mask = np.zeros_like(coordinates, dtype=np.bool_)
    displacement = (coordinates[:, :, 1:] - coordinates[:, :, :-1]) * SAMPLE_RATE_HZ
    joint_valid = track_mask[:, :, 1:] & track_mask[:, :, :-1]
    frame_valid = frame_mask[:, 1:] & frame_mask[:, :-1]
    joint_valid &= frame_valid[:, None, :, None] & actor_mask[:, :, None, None]
    pelvis_valid = joint_valid[..., :1]
    relative = displacement - displacement[..., :1, :]
    relative[..., 0, :] = displacement[..., 0, :]
    observed = joint_valid & pelvis_valid
    values[:, :, 1:] = np.where(observed[..., None], relative, 0.0)
    mask[:, :, 1:] = np.broadcast_to(observed[..., None], relative.shape)
    values[values == 0.0] = 0.0
    return _readonly(values), _readonly(mask)


def _patch_intervals(length: int, config: LocalPhaseConfig) -> tuple[tuple[int, int], ...]:
    width = min(length, config.patch_frames)
    starts = list(range(0, length - width + 1, config.hop_frames))
    if starts[-1] != length - width:
        starts.append(length - width)
    return tuple((start, start + width) for start in starts)


def directional_phase_fields(
    batch: PreparedGroupBatch,
    *,
    energy_floors: np.ndarray,
    config: LocalPhaseConfig = LocalPhaseConfig(),
) -> tuple[DirectionalPhaseField, ...]:
    """Compute each actor/band once, preserving the unchanged physical kernels.

    Floors must be fitted for this signed-vector frontend using training only.
    Legacy five-speed-channel fitted floors are not interchangeable with these.
    """
    checked = validate_prepared_group_batch(batch)
    floors = validate_energy_floors(energy_floors)
    return _directional_fields_from_arrays(
        checked.skeletons,
        checked.track_mask,
        checked.frame_mask,
        checked.actor_mask,
        checked.actor_counts,
        checked.valid_lengths,
        floors,
        config,
    )


def continuous_directional_phase_field(
    capture: PreparedContinuousCapture,
    *,
    energy_floors: np.ndarray,
    config: LocalPhaseConfig = LocalPhaseConfig(),
) -> DirectionalPhaseField:
    """Analyze a whole capture without the legacy short-window time limit.

    The timeline, signed derivatives, Morlet support and patch locations span
    ten-second boundaries. This remains an explicitly bounded RAM response
    cache, not a disk-streaming training host. Floors remain training-only.
    """
    if type(capture) is not PreparedContinuousCapture:
        raise TypeError("capture must be exactly PreparedContinuousCapture")
    floors = validate_energy_floors(energy_floors)
    field = _directional_fields_from_arrays(
        capture.skeletons[None],
        capture.track_mask[None],
        np.ones((1, capture.frame_count), dtype=np.bool_),
        np.ones((1, capture.actor_count), dtype=np.bool_),
        (capture.actor_count,),
        (capture.frame_count,),
        floors,
        config,
    )[0]
    return replace(
        field,
        source_sha256=capture.source_sha256,
        actor_commitments=capture.actor_commitments,
        physical_view_sha256=physical_view_sha256(capture),
    )


def _directional_fields_from_arrays(
    skeletons: np.ndarray,
    track_mask: np.ndarray,
    frame_mask: np.ndarray,
    actor_mask: np.ndarray,
    actor_counts: tuple[int, ...],
    valid_lengths: tuple[int, ...],
    floors: np.ndarray,
    config: LocalPhaseConfig,
    response_directory: Path | None = None,
) -> tuple[DirectionalPhaseField, ...]:
    required_edges = sum(count * (count - 1) // 2 for count in actor_counts)
    if required_edges > config.edge_budget:
        raise ResourceLimitError(required_edges=required_edges, edge_budget=config.edge_budget)
    bands = morlet_kernel_bank()
    projected = sum(
        count * sum(max(0, length - band.length + 1) for band in bands) * 66 * 17
        for count, length in zip(actor_counts, valid_lengths, strict=True)
    )
    if response_directory is not None and len(actor_counts) != 1:
        raise ValueError("disk response storage is one complete capture per directory")
    if response_directory is None and projected > config.max_response_bytes:
        raise MemoryError("RESOURCE_LIMIT: actor Morlet responses exceed the byte budget")
    signed, signed_mask = _signed_velocity_arrays(skeletons, track_mask, frame_mask, actor_mask)
    results = []
    for row, (count, length) in enumerate(zip(actor_counts, valid_lengths, strict=True)):
        values = signed[row, :count, :length].reshape(count, length, 66)
        masks = signed_mask[row, :count, :length].reshape(count, length, 66)
        responses, response_masks, centers = [], [], []
        for band in bands:
            if length < band.length:
                transformed = np.zeros((count, 0, 22, 3), dtype=np.complex128)
                valid = np.zeros(transformed.shape, dtype=np.bool_)
                positions = np.zeros(0, dtype=np.float64)
            else:
                response_count = length - band.length + 1
                shape = (count, response_count, 22, 3)
                if response_directory is None:
                    transformed = np.zeros(shape, dtype=np.complex128)
                    valid = np.zeros(shape, dtype=np.bool_)
                else:
                    response_path = response_directory / f"response-{band.index}.npy"
                    mask_path = response_directory / f"response-mask-{band.index}.npy"
                    # Claim new files exclusively; never overwrite an earlier cache.
                    response_path.open("xb").close()
                    mask_path.open("xb").close()
                    transformed = np.lib.format.open_memmap(
                        response_path, mode="w+", dtype=np.complex128, shape=shape
                    )
                    valid = np.lib.format.open_memmap(
                        mask_path, mode="w+", dtype=np.bool_, shape=shape
                    )
                # Each central response is written once. The input extends by
                # kernel_length-1 past the response block: full physical halo,
                # never independent chunk convolution or a shortened kernel.
                for start in range(0, response_count, config.convolution_chunk_frames):
                    stop = min(response_count, start + config.convolution_chunk_frames)
                    halo_stop = stop + band.length - 1
                    windows = np.lib.stride_tricks.sliding_window_view(
                        values[:, start:halo_stop], band.length, axis=1
                    )
                    mask_windows = np.lib.stride_tricks.sliding_window_view(
                        masks[:, start:halo_stop], band.length, axis=1
                    )
                    local_valid = mask_windows.all(axis=-1).reshape(count, stop - start, 22, 3)
                    local_values = np.einsum("ktcl,l->ktc", windows, band.kernel).reshape(
                        count, stop - start, 22, 3
                    )
                    valid[:, start:stop] = local_valid
                    transformed[:, start:stop] = np.where(local_valid, local_values, 0.0 + 0.0j)
                if response_directory is not None:
                    transformed.flush()
                    valid.flush()
                positions = np.arange(length - band.length + 1) + (band.length - 1) / 2
            responses.append(_readonly(np.ascontiguousarray(transformed)))
            response_masks.append(_readonly(np.ascontiguousarray(valid)))
            centers.append(_readonly(positions))
        intervals = _patch_intervals(length, config)
        actor_features = np.zeros((count, len(intervals), ACTOR_FEATURE_DIM), dtype=np.float32)
        actor_mask = np.zeros((count, len(intervals)), dtype=np.bool_)
        roots = np.zeros((count, len(intervals), 3), dtype=np.float32)
        root_mask = np.zeros_like(actor_mask)
        for patch, (start, stop) in enumerate(intervals):
            local_values, local_mask = values[:, start:stop], masks[:, start:stop]
            denominator = local_mask.sum(axis=1).clip(min=1)
            mean = local_values.sum(axis=1) / denominator
            rms = np.sqrt((local_values**2).sum(axis=1) / denominator)
            actor_features[:, patch] = np.concatenate((mean, rms), axis=-1)
            actor_mask[:, patch] = local_mask.any(axis=(1, 2))
            pelvis_mask = track_mask[row, :count, start:stop, 0]
            pelvis_values = skeletons[row, :count, start:stop, 0].astype(np.float64)
            roots[:, patch] = (
                np.where(pelvis_mask[..., None], pelvis_values, 0.0).sum(axis=1)
                / pelvis_mask.sum(axis=1).clip(min=1)[:, None]
            )
            root_mask[:, patch] = pelvis_mask.any(axis=1)
        results.append(
            DirectionalPhaseField(
                tuple(responses),
                tuple(response_masks),
                tuple(centers),
                intervals,
                _readonly(actor_features),
                _readonly(actor_mask),
                _readonly(roots),
                _readonly(root_mask),
                floors,
                config,
            )
        )
    return tuple(results)


def canonical_edge_pairs(actor_count: int, start: int, stop: int) -> np.ndarray:
    """Address a contiguous lexicographic edge block without storing all edges."""
    total = actor_count * (actor_count - 1) // 2
    if not 0 <= start <= stop <= total:
        raise ValueError("edge interval is outside the complete unordered graph")
    result = np.empty((stop - start, 2), dtype=np.int64)
    if stop == start:
        return result
    low, high = 0, actor_count - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if middle * (2 * actor_count - middle - 1) // 2 <= start:
            low = middle
        else:
            high = middle
    left = low
    right = left + 1 + start - left * (2 * actor_count - left - 1) // 2
    for offset in range(stop - start):
        result[offset] = left, right
        right += 1
        if right == actor_count:
            left += 1
            right = left + 1
    return result


def local_pair_chunk(field: DirectionalPhaseField, start: int, stop: int) -> LocalPairChunk:
    pairs = canonical_edge_pairs(field.actor_count, start, stop)
    features = np.zeros((len(pairs), field.patch_count, 6, PHASE_FEATURE_DIM), dtype=np.float64)
    phase_mask = np.zeros(features.shape[:-1], dtype=np.bool_)
    support_mask = np.zeros_like(phase_mask)
    for band, (response, mask, centers) in enumerate(
        zip(field.responses, field.response_masks, field.response_centers, strict=True)
    ):
        for patch, (window_start, window_stop) in enumerate(field.intervals):
            selected = (centers >= window_start) & (centers < window_stop)
            for edge, (left, right) in enumerate(pairs):
                common = mask[left, selected] & mask[right, selected]
                count = int(common.sum())
                if count == 0:
                    continue
                a, b = (
                    response[left, selected][common],
                    response[right, selected][common],
                )
                aa = float(np.vdot(a, a).real)
                bb = float(np.vdot(b, b).real)
                cross = complex(np.sum(a * b.conj()))
                left_power, right_power = aa / count, bb / count
                observed = (
                    left_power > field.energy_floors[band]
                    and right_power > field.energy_floors[band]
                )
                correlation = cross / math.sqrt(aa * bb + field.config.epsilon) if observed else 0j
                features[edge, patch, band] = (
                    correlation.real,
                    correlation.imag,
                    abs(correlation),
                    math.log1p(left_power),
                    math.log1p(right_power),
                    count / max(1, int(selected.sum()) * 66),
                    float(observed),
                )
                phase_mask[edge, patch, band] = observed
                support_mask[edge, patch, band] = True
    if not np.isfinite(features).all():
        raise ValueError("local phase features became nonfinite")
    features[features == 0.0] = 0.0
    return LocalPairChunk(
        _readonly(pairs),
        _readonly(features),
        _readonly(phase_mask),
        _readonly(support_mask),
    )


def iter_local_pair_chunks(
    field: DirectionalPhaseField, *, edge_chunk_size: int = 256
) -> Iterator[LocalPairChunk]:
    if type(edge_chunk_size) is not int or edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be a positive integer")
    for start in range(0, field.pair_count, edge_chunk_size):
        yield local_pair_chunk(field, start, min(field.pair_count, start + edge_chunk_size))


def mean_difference_signal_dct(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, ...]:
    """Orthonormal DCT-II of mean/difference signals, including cross terms."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 1:
        raise ValueError("DCT inputs must have matching nonempty [T,C] shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("DCT inputs must be finite")
    length = left.shape[0]
    basis = np.cos(
        np.pi / length * (np.arange(length) + 0.5)[None, :] * np.arange(length)[:, None]
    ) * math.sqrt(2.0 / length)
    basis[0] /= math.sqrt(2.0)
    return basis @ ((left + right) / 2), basis @ (left - right)


def true_mean_difference_dct_features(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Six-band phase-free control from the *signals* (a+b)/2 and a-b.

    This is not the legacy independent-self-power control. A shared
    intersection mask is applied before either DCT so the sum/difference
    identity and its cross term remain exact for the observed channels.
    No Morlet phase, imaginary quadrature or directed lag enters the output.
    The returned support means observed DCT samples, not calibrated phase
    energy; a training host must not call it a fitted physical floor.

    DCT-II bins are partitioned into six complete positive-frequency bands
    at geometric midpoints between the frozen Morlet center frequencies.
    Feature slots: log mean-signal power, log difference-signal power,
    normalized cross-spectrum, log left power, log right power, common
    observation coverage, and an observed-channel indicator.
    """
    if (
        type(left) is not np.ndarray
        or type(right) is not np.ndarray
        or type(left_mask) is not np.ndarray
        or type(right_mask) is not np.ndarray
        or left.ndim != 2
        or left.shape != right.shape
        or left.shape != left_mask.shape
        or left.shape != right_mask.shape
        or not 2 <= left.shape[0] <= 40
        or left.shape[1] < 1
        or left_mask.dtype != np.bool_
        or right_mask.dtype != np.bool_
        or left.dtype.kind != "f"
        or right.dtype.kind != "f"
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("DCT control requires finite [T,C] values and matching bool masks")
    length = left.shape[0]
    common = left_mask & right_mask
    eligible = common.sum(axis=0) >= 2
    features = np.zeros((len(MORLET_FREQUENCIES_HZ), PHASE_FEATURE_DIM), dtype=np.float64)
    support = np.zeros((len(MORLET_FREQUENCIES_HZ),), dtype=np.bool_)
    if not bool(eligible.any()):
        return _readonly(features), _readonly(support)

    # Construct the actual paired signals before any frequency transform.
    a = np.where(common[:, eligible], left[:, eligible].astype(np.float64), 0.0)
    b = np.where(common[:, eligible], right[:, eligible].astype(np.float64), 0.0)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("DCT control endpoint conversion became nonfinite")
    try:
        with np.errstate(over="raise", invalid="raise"):
            mean_signal = (a + b) / 2.0
            difference_signal = a - b
    except FloatingPointError as exc:
        raise ValueError("DCT control paired signal became nonfinite") from exc
    samples = np.arange(length, dtype=np.float64) + 0.5
    bins = np.arange(1, length, dtype=np.int64)
    frequencies = bins * SAMPLE_RATE_HZ / (2 * length)
    centers = np.asarray(MORLET_FREQUENCIES_HZ, dtype=np.float64)
    boundaries = np.sqrt(centers[:-1] * centers[1:])
    band_for_bin = np.searchsorted(boundaries, frequencies, side="right")
    basis = math.sqrt(2.0 / length) * np.cos(
        np.pi * bins[:, None] * samples[None, :] / length
    )
    mean_dct = basis @ mean_signal
    difference_dct = basis @ difference_signal
    left_dct = basis @ a
    right_dct = basis @ b
    if not all(
        np.isfinite(value).all()
        for value in (mean_dct, difference_dct, left_dct, right_dct)
    ):
        raise ValueError("DCT control coefficients became nonfinite")
    coverage = float(common.sum()) / common.size
    for band in range(len(MORLET_FREQUENCIES_HZ)):
        selected = band_for_bin == band
        if not bool(selected.any()):
            continue
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean_power = float(np.square(mean_dct[selected]).sum(axis=0).mean())
                difference_power = float(np.square(difference_dct[selected]).sum(axis=0).mean())
                left_power = float(np.square(left_dct[selected]).sum(axis=0).mean())
                right_power = float(np.square(right_dct[selected]).sum(axis=0).mean())
        except FloatingPointError as exc:
            raise ValueError("DCT control band energy became nonfinite") from exc
        total = 4 * mean_power + difference_power
        cross = (4 * mean_power - difference_power) / total if total > 0 else 0.0
        features[band] = (
            math.log1p(mean_power),
            math.log1p(difference_power),
            cross,
            math.log1p(left_power),
            math.log1p(right_power),
            coverage,
            1.0,
        )
        support[band] = True
    features[features == 0.0] = 0.0
    if not np.isfinite(features).all():
        raise ValueError("DCT control features became nonfinite")
    return _readonly(features), _readonly(support)
