"""Typed, self-hashed declaration for an A6 training-only DCT floor fit.

This schema makes population/config drift visible at the model seam. It does
not independently prove that a fit ran or grant formal launch authority: the
private controller must authenticate the source-manifest hash and exact
registered training population before constructing a production model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable
import hashlib
import json

import numpy as np

from .directional_phase import LocalPhaseConfig
from .continuous_capture import PreparedContinuousCapture
from .directional_calibration import _linear_fifth_percentile
from .directional_phase import _patch_intervals, _signed_velocity_arrays, true_mean_difference_dct_features
from .periodic import validate_energy_floors


DCT_FLOOR_SCHEMA = "phaseset-v2-a6-dct-training-floor-v1"
DCT_CALIBRATION_METHOD = "signed-vector-actor-patch-DCT-power/linear-fifth-percentile-v1"
_POPULATIONS = frozenset({"main", "pilot", "fold_0", "fold_1", "fold_2"})
_SEALED_COMPONENTS = frozenset({"C09", "C11", "C15"})


def dct_config_sha256(config: LocalPhaseConfig) -> str:
    if type(config) is not LocalPhaseConfig:
        raise TypeError("DCT config digest requires LocalPhaseConfig")
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"phaseset-v2-dct-config-v1\x00" + encoded).hexdigest()


def _sha256_hex(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class DctFloorReceipt:
    """Declared fit lineage and immutable six-band floors, not a signed authority."""

    schema: str
    population_name: str
    training_components: tuple[str, ...]
    training_source_manifest_sha256: str
    config_sha256: str
    floors: np.ndarray

    def __post_init__(self) -> None:
        if self.schema != DCT_FLOOR_SCHEMA or self.population_name not in _POPULATIONS:
            raise ValueError("DCT floor schema or population differs")
        if (
            type(self.training_components) is not tuple
            or not self.training_components
            or len(set(self.training_components)) != len(self.training_components)
            or tuple(sorted(self.training_components)) != self.training_components
            or any(
                type(value) is not str
                or len(value) != 3
                or value[0] != "C"
                or not value[1:].isdigit()
                for value in self.training_components
            )
            or bool(set(self.training_components) & _SEALED_COMPONENTS)
            or not _sha256_hex(self.training_source_manifest_sha256)
            or not _sha256_hex(self.config_sha256)
        ):
            raise ValueError("DCT floor training population or manifest binding differs")
        checked = validate_energy_floors(self.floors)
        frozen = checked.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "floors", frozen)

    @property
    def sha256(self) -> str:
        payload = {
            "schema": self.schema,
            "population_name": self.population_name,
            "training_components": self.training_components,
            "training_source_manifest_sha256": self.training_source_manifest_sha256,
            "config_sha256": self.config_sha256,
            "floors_float64_le_hex": self.floors.astype("<f8", copy=False).tobytes().hex(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"phaseset-v2-dct-floor-receipt-v1\x00" + encoded).hexdigest()

    def require_config(self, config: LocalPhaseConfig) -> None:
        if self.config_sha256 != dct_config_sha256(config):
            raise ValueError("DCT floor receipt/config digest differs")

    def require_population(self, name: str, training_components: tuple[str, ...]) -> None:
        if self.population_name != name or self.training_components != training_components:
            raise ValueError("DCT floor receipt/training population differs")


@dataclass(frozen=True)
class DctCalibrationCapture:
    capture_id: str  # admitted private source identity, never an actor-set positive ID
    component: str
    capture: PreparedContinuousCapture

    def __post_init__(self) -> None:
        if (
            type(self.capture) is not PreparedContinuousCapture
            or self.capture_id != self.capture.source_sha256
            or type(self.component) is not str
            or not self.component
        ):
            raise ValueError("A6 calibration record must bind exact prepared source identity")


@dataclass(frozen=True)
class DctEnergyCalibration:
    floors: np.ndarray
    capture_count: int
    training_components: tuple[str, ...]
    observed_counts: tuple[int, ...]
    missing_counts: tuple[int, ...]
    exact_zero_counts: tuple[int, ...]
    method: str = DCT_CALIBRATION_METHOD


def actor_patch_dct_power_observations(
    capture: PreparedContinuousCapture,
    config: LocalPhaseConfig,
    *,
    max_working_bytes: int = 1_073_741_824,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpaired marginal actor DCT power in the exact A6 endpoint power units.

    A zero-valued second endpoint makes slot 3 of the already audited paired
    descriptor the first actor's power. This shares the binning/mask/DCT
    arithmetic with A6 while collecting no cross-actor or text information.
    Observed zero power is retained for the fixed fifth percentile.
    """
    if type(capture) is not PreparedContinuousCapture or type(config) is not LocalPhaseConfig:
        raise TypeError("DCT calibration requires an exact capture and config")
    if not 2 <= config.patch_frames <= 40:
        raise ValueError("DCT calibration patch length must be in [2,40]")
    if type(max_working_bytes) is not int or max_working_bytes < 1:
        raise ValueError("DCT calibration working-memory bound must be positive")
    # _signed_velocity_arrays creates float64 coordinates/velocities and
    # temporary derivative, relative-motion and validity arrays. The 128-byte
    # per scalar upper envelope also covers small local DCT temporaries; this
    # gate happens before any full-timeline float64 allocation.
    projected = capture.actor_count * capture.frame_count * 66 * 128
    if projected > max_working_bytes:
        raise MemoryError("RESOURCE_LIMIT: A6 full-capture velocity working set exceeds budget")
    values, masks = _signed_velocity_arrays(
        capture.skeletons[None],
        capture.track_mask[None],
        np.ones((1, capture.frame_count), dtype=np.bool_),
        np.ones((1, capture.actor_count), dtype=np.bool_),
    )
    shape = (capture.actor_count, capture.frame_count, 66)
    signed, observed = values[0].reshape(shape), masks[0].reshape(shape)
    intervals = _patch_intervals(capture.frame_count, config)
    powers = np.zeros((capture.actor_count, len(intervals), 6), dtype=np.float64)
    support = np.zeros(powers.shape, dtype=np.bool_)
    for actor in range(capture.actor_count):
        for patch, (start, stop) in enumerate(intervals):
            actor_signal = signed[actor, start:stop]
            actor_mask = observed[actor, start:stop]
            features, band_support = true_mean_difference_dct_features(
                actor_signal,
                np.zeros_like(actor_signal),
                actor_mask,
                actor_mask,
            )
            power = np.expm1(features[:, 3])
            if not bool(np.isfinite(power).all()) or bool(np.any(power < 0.0)):
                raise ValueError("A6 marginal DCT power became invalid")
            powers[actor, patch] = power
            support[actor, patch] = band_support
    powers[powers == 0.0] = 0.0
    powers.setflags(write=False)
    support.setflags(write=False)
    return powers, support


def fit_dct_energy_floors(
    captures: Iterable[DctCalibrationCapture],
    *,
    training_components: tuple[str, ...],
    expected_capture_ids: tuple[str, ...],
    config: LocalPhaseConfig,
    max_actor_patch_observations: int = 1_000_000,
    max_working_bytes: int = 1_073_741_824,
    max_total_velocity_scalars: int = 1_000_000_000,
) -> DctEnergyCalibration:
    """Fail closed unless exactly the declared complete training-parent census is read.

    The private host must independently authenticate component assignments,
    prepared capture files and the source-manifest digest before calling this
    pure fitter. Neither validation nor final-test rows are sampled or read.
    """
    if (
        type(training_components) is not tuple
        or not training_components
        or len(set(training_components)) != len(training_components)
        or any(type(value) is not str or not value for value in training_components)
        or bool(set(training_components) & _SEALED_COMPONENTS)
        or type(expected_capture_ids) is not tuple
        or not expected_capture_ids
        or len(set(expected_capture_ids)) != len(expected_capture_ids)
        or any(type(value) is not str or not value for value in expected_capture_ids)
        or type(config) is not LocalPhaseConfig
        or type(max_actor_patch_observations) is not int
        or max_actor_patch_observations < 1
        or type(max_working_bytes) is not int
        or max_working_bytes < 1
        or type(max_total_velocity_scalars) is not int
        or max_total_velocity_scalars < 1
    ):
        raise ValueError("A6 complete train-only calibration contract is invalid")
    allowed, expected = set(training_components), set(expected_capture_ids)
    seen: set[str] = set()
    observed_components: set[str] = set()
    values: list[list[np.ndarray]] = [[] for _ in range(6)]
    missing = np.zeros(6, dtype=np.int64)
    total = 0
    total_velocity_scalars = 0
    for item in captures:
        if type(item) is not DctCalibrationCapture:
            raise TypeError("A6 calibration stream must contain exact capture records")
        if item.component not in allowed or item.capture_id not in expected or item.capture_id in seen:
            raise ValueError("A6 calibration stream contains held-out, repeated or foreign parent")
        total += item.capture.actor_count * len(_patch_intervals(item.capture.frame_count, config))
        if total > max_actor_patch_observations:
            raise MemoryError("RESOURCE_LIMIT: A6 complete actor-patch population exceeds budget")
        total_velocity_scalars += item.capture.actor_count * item.capture.frame_count * 66
        if total_velocity_scalars > max_total_velocity_scalars:
            raise MemoryError("RESOURCE_LIMIT: A6 full training timeline exceeds scalar budget")
        powers, support = actor_patch_dct_power_observations(
            item.capture, config, max_working_bytes=max_working_bytes
        )
        for band in range(6):
            selected = support[..., band]
            values[band].append(powers[..., band][selected])
            missing[band] += int(selected.size - selected.sum())
        seen.add(item.capture_id)
        observed_components.add(item.component)
    if seen != expected or observed_components != allowed:
        raise ValueError("A6 calibration stream differs from the complete training census")
    selected_values = tuple(np.concatenate(rows) for rows in values)
    floors = validate_energy_floors(
        np.array([_linear_fifth_percentile(row) for row in selected_values], dtype=np.float64)
    ).copy()
    floors.setflags(write=False)
    return DctEnergyCalibration(
        floors,
        len(seen),
        tuple(sorted(allowed)),
        tuple(int(row.size) for row in selected_values),
        tuple(int(count) for count in missing),
        tuple(int(np.count_nonzero(row == 0.0)) for row in selected_values),
    )
