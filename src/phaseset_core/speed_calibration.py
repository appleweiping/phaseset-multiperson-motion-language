"""Typed train-only floor declaration for the A1 speed-magnitude Morlet frontend.

The receipt binds the declared population and input/configuration but is not
itself proof that the fit ran; private admission verifies the source census.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .directional_calibration import DirectionalEnergyCalibration, SPEED_CALIBRATION_METHOD
from .directional_phase import LocalPhaseConfig
from .periodic import validate_energy_floors


SPEED_FLOOR_SCHEMA = "phaseset-v2-a1-speed-training-floor-v1"
_POPULATIONS = frozenset({"main", "pilot", "fold_0", "fold_1", "fold_2"})
_SEALED = frozenset({"C09", "C11", "C15"})


def speed_config_sha256(config: LocalPhaseConfig) -> str:
    if type(config) is not LocalPhaseConfig:
        raise TypeError("A1 floor configuration requires LocalPhaseConfig")
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"phaseset-v2-speed-config-v1\0" + encoded).hexdigest()


def _sha256_key(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class SpeedFloorReceipt:
    schema: str
    population_name: str
    training_components: tuple[str, ...]
    training_source_manifest_sha256: str
    config_sha256: str
    floors: np.ndarray

    def __post_init__(self) -> None:
        if self.schema != SPEED_FLOOR_SCHEMA or self.population_name not in _POPULATIONS:
            raise ValueError("A1 floor schema or population differs")
        if (
            type(self.training_components) is not tuple
            or not self.training_components
            or len(set(self.training_components)) != len(self.training_components)
            or tuple(sorted(self.training_components)) != self.training_components
            or bool(set(self.training_components) & _SEALED)
            or not _sha256_key(self.training_source_manifest_sha256)
            or not _sha256_key(self.config_sha256)
        ):
            raise ValueError("A1 floor training population or source binding differs")
        checked = validate_energy_floors(self.floors)
        frozen = checked.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "floors", frozen)

    @classmethod
    def from_fit(
        cls,
        fit: DirectionalEnergyCalibration,
        *,
        population_name: str,
        training_source_manifest_sha256: str,
        config: LocalPhaseConfig,
    ) -> SpeedFloorReceipt:
        if type(fit) is not DirectionalEnergyCalibration or fit.method != SPEED_CALIBRATION_METHOD:
            raise ValueError("A1 floor receipt requires the complete speed-only fit")
        if fit.config != config:
            raise ValueError("A1 floor receipt/config differs from the actual training fit")
        return cls(
            SPEED_FLOOR_SCHEMA,
            population_name,
            fit.training_components,
            training_source_manifest_sha256,
            speed_config_sha256(config),
            fit.energy_floors,
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
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"phaseset-v2-speed-floor-receipt-v1\0" + encoded).hexdigest()

    def require_config(self, config: LocalPhaseConfig) -> None:
        if self.config_sha256 != speed_config_sha256(config):
            raise ValueError("A1 floor receipt/config differs")

    def require_population(self, name: str, components: tuple[str, ...]) -> None:
        if self.population_name != name or self.training_components != components:
            raise ValueError("A1 floor receipt/training population differs")
