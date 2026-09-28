"""Typed, self-hashed declaration for an A6 training-only DCT floor fit.

This schema makes population/config drift visible at the model seam. It does
not independently prove that a fit ran or grant formal launch authority: the
private controller must authenticate the source-manifest hash and exact
registered training population before constructing a production model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .directional_phase import LocalPhaseConfig
from .periodic import validate_energy_floors


DCT_FLOOR_SCHEMA = "phaseset-v2-a6-dct-training-floor-v1"
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
