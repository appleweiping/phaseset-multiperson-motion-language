"""Train-only scalar Morlet floors for the unchanged old PhaseSet frontend.

One observation is one actor, accepted 200-frame window and physical band.
The complete admitted parent census is required; no pair, caption, score or
model output participates in fitting. A private source manifest must separately
prove that the supplied captures are the registered training population.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .continuous_capture import PreparedContinuousCapture
from .contracts import skeleton_to_activity
from .directional_calibration import _linear_fifth_percentile
from .morlet import MORLET_ORACLE_SHA256, SAMPLE_RATE_HZ
from .periodic import (
    BAND_COUNT,
    _actor_marginal_powers,
    _precompute_actor_responses,
    validate_energy_floors,
)
from .pipeline import collate_group_samples


LEGACY_SCALAR_CALIBRATION_METHOD = (
    "old-five-speed-channel-actor-window-Morlet-power/linear-fifth-percentile-v1"
)
LEGACY_SCALAR_FLOOR_SCHEMA = "phaseset-v2-legacy-scalar-training-floor-v1"
_SEALED = frozenset(("C09", "C11", "C15"))


def legacy_scalar_config_sha256() -> str:
    """Bind the old five-channel, 20-Hz physical kernel and estimator identity."""
    payload = {
        "activity": "five-masked-body-group-speed-channels/20Hz/v1",
        "morlet_oracle_sha256": MORLET_ORACLE_SHA256,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "method": LEGACY_SCALAR_CALIBRATION_METHOD,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"phaseset-v2-legacy-scalar-config-v1\0" + raw).hexdigest()


def _sha256_key(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class LegacyScalarCalibrationCapture:
    source_sha256: str
    component: str
    capture: PreparedContinuousCapture


@dataclass(frozen=True)
class LegacyScalarEnergyCalibration:
    floors: np.ndarray
    capture_count: int
    accepted_window_count: int
    actor_window_count: int
    training_components: tuple[str, ...]
    observed_counts: tuple[int, ...]
    missing_counts: tuple[int, ...]
    exact_zero_counts: tuple[int, ...]
    method: str = LEGACY_SCALAR_CALIBRATION_METHOD
    config_sha256: str = legacy_scalar_config_sha256()


def fit_legacy_scalar_energy_floors(
    captures: Iterable[LegacyScalarCalibrationCapture],
    *,
    training_components: tuple[str, ...],
    expected_capture_ids: tuple[str, ...],
    max_actor_window_observations: int = 1_000_000,
) -> LegacyScalarEnergyCalibration:
    """Fit six fifth-percentile floors from every admitted training parent.

    The old pair descriptor uses Morlet power over the shared endpoint mask.
    Calibration uses each actor's marginal response mask, as an independent
    actor-local observation, and includes observed exact-zero powers. Absence
    of physical support is counted but excluded from the percentile. A runtime
    limit raises RESOURCE_LIMIT; it never samples or truncates the population.
    """
    if (
        type(training_components) is not tuple
        or not training_components
        or tuple(sorted(set(training_components))) != training_components
        or set(training_components) & _SEALED
    ):
        raise ValueError("old scalar floors require sorted, unique training components")
    if (
        type(expected_capture_ids) is not tuple
        or not expected_capture_ids
        or len(set(expected_capture_ids)) != len(expected_capture_ids)
        or any(not _sha256_key(value) for value in expected_capture_ids)
    ):
        raise ValueError("old scalar floor parent census must be explicit and unique")
    if type(max_actor_window_observations) is not int or max_actor_window_observations < 1:
        raise ValueError("old scalar floor observation budget must be positive")
    allowed = set(training_components)
    expected = set(expected_capture_ids)
    seen: set[str] = set()
    seen_components: set[str] = set()
    samples: list[list[float]] = [[] for _ in range(BAND_COUNT)]
    missing = np.zeros((BAND_COUNT,), dtype=np.int64)
    accepted_windows = actor_windows = 0
    for row in captures:
        if type(row) is not LegacyScalarCalibrationCapture:
            raise TypeError("old scalar floor stream needs exact capture records")
        if (
            row.component not in allowed
            or row.source_sha256 not in expected
            or row.source_sha256 in seen
        ):
            raise ValueError("old scalar floor stream has nontraining, repeated or unknown parent")
        capture = row.capture
        if (
            type(capture) is not PreparedContinuousCapture
            or capture.source_sha256 != row.source_sha256
            or capture.augmentation_yaw != 0.0
        ):
            raise ValueError("old scalar floors require the admitted unaugmented physical parent")
        windows = capture.windows()
        actor_windows += len(windows) * capture.actor_count
        if actor_windows > max_actor_window_observations:
            raise MemoryError("RESOURCE_LIMIT: old scalar floor population exceeds budget")
        for window in windows:
            activity = skeleton_to_activity(collate_group_samples((window,)))
            actor_count = activity.actor_counts[0]
            length = activity.valid_lengths[0]
            responses, _bands, length_mask = _precompute_actor_responses(
                activity.activities[0, :actor_count],
                activity.activity_mask[0, :actor_count],
                length,
            )
            for response in responses:
                powers = _actor_marginal_powers(response, length_mask)
                for band in range(BAND_COUNT):
                    mask = response.response_mask[band]
                    if not bool(length_mask[band]) or mask is None or not bool(mask.any()):
                        missing[band] += 1
                    else:
                        power = float(powers[band])
                        if not np.isfinite(power) or power < 0.0:
                            raise ValueError("old scalar physical power is nonfinite or negative")
                        samples[band].append(power if power else 0.0)
            accepted_windows += 1
        seen.add(row.source_sha256)
        seen_components.add(row.component)
    if seen != expected or seen_components != allowed:
        raise ValueError("old scalar fit did not consume the complete training census")
    values = tuple(np.asarray(row, dtype=np.float64) for row in samples)
    if any(int(row.size) + int(missing[band]) != actor_windows for band, row in enumerate(values)):
        raise AssertionError("old scalar calibration lost an actor-window observation")
    floors = validate_energy_floors(
        np.asarray([_linear_fifth_percentile(row) for row in values], dtype=np.float64)
    ).copy()
    floors.setflags(write=False)
    return LegacyScalarEnergyCalibration(
        floors=floors,
        capture_count=len(seen),
        accepted_window_count=accepted_windows,
        actor_window_count=actor_windows,
        training_components=training_components,
        observed_counts=tuple(int(row.size) for row in values),
        missing_counts=tuple(int(value) for value in missing),
        exact_zero_counts=tuple(int(np.count_nonzero(row == 0.0)) for row in values),
    )


@dataclass(frozen=True)
class LegacyScalarFloorReceipt:
    schema: str
    population_name: str
    training_components: tuple[str, ...]
    training_source_manifest_sha256: str
    config_sha256: str
    floors: np.ndarray
    capture_count: int
    accepted_window_count: int
    actor_window_count: int
    observed_counts: tuple[int, ...]
    missing_counts: tuple[int, ...]
    exact_zero_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema != LEGACY_SCALAR_FLOOR_SCHEMA or self.population_name != "main":
            raise ValueError("old scalar floor schema or registered population differs")
        if (
            type(self.training_components) is not tuple
            or not self.training_components
            or tuple(sorted(set(self.training_components))) != self.training_components
            or bool(set(self.training_components) & _SEALED)
            or not _sha256_key(self.training_source_manifest_sha256)
            or self.config_sha256 != legacy_scalar_config_sha256()
        ):
            raise ValueError("old scalar floor training source or physical config differs")
        if (
            type(self.capture_count) is not int
            or self.capture_count < 1
            or type(self.accepted_window_count) is not int
            or self.accepted_window_count < self.capture_count
            or type(self.actor_window_count) is not int
            or self.actor_window_count < 2 * self.accepted_window_count
            or any(
                type(counts) is not tuple
                or len(counts) != BAND_COUNT
                or any(type(value) is not int or value < 0 for value in counts)
                for counts in (self.observed_counts, self.missing_counts, self.exact_zero_counts)
            )
            or any(
                self.observed_counts[band] + self.missing_counts[band] != self.actor_window_count
                or self.exact_zero_counts[band] > self.observed_counts[band]
                or self.observed_counts[band] == 0
                for band in range(BAND_COUNT)
            )
        ):
            raise ValueError("old scalar floor diagnostics do not cover every actor-window")
        checked = validate_energy_floors(self.floors)
        frozen = checked.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "floors", frozen)

    @classmethod
    def from_fit(
        cls,
        fit: LegacyScalarEnergyCalibration,
        *,
        training_source_manifest_sha256: str,
    ) -> LegacyScalarFloorReceipt:
        if (
            type(fit) is not LegacyScalarEnergyCalibration
            or fit.method != LEGACY_SCALAR_CALIBRATION_METHOD
            or fit.config_sha256 != legacy_scalar_config_sha256()
        ):
            raise ValueError("old scalar floor receipt requires the complete physical fit")
        return cls(
            LEGACY_SCALAR_FLOOR_SCHEMA,
            "main",
            fit.training_components,
            training_source_manifest_sha256,
            fit.config_sha256,
            fit.floors,
            fit.capture_count,
            fit.accepted_window_count,
            fit.actor_window_count,
            fit.observed_counts,
            fit.missing_counts,
            fit.exact_zero_counts,
        )

    @property
    def sha256(self) -> str:
        payload = {
            "schema": self.schema,
            "population_name": self.population_name,
            "training_components": self.training_components,
            "training_source_manifest_sha256": self.training_source_manifest_sha256,
            "config_sha256": self.config_sha256,
            "floors_float64_le_hex": self.floors.astype("<f8", copy=False).tobytes().hex(),
            "capture_count": self.capture_count,
            "accepted_window_count": self.accepted_window_count,
            "actor_window_count": self.actor_window_count,
            "observed_counts": self.observed_counts,
            "missing_counts": self.missing_counts,
            "exact_zero_counts": self.exact_zero_counts,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(b"phaseset-v2-legacy-scalar-floor-receipt-v1\0" + encoded).hexdigest()

    def require_population(self, components: tuple[str, ...]) -> None:
        if self.training_components != components:
            raise ValueError("old scalar floor training population differs")
