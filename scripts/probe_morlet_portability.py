"""Bounded, NumPy-only cross-runtime probe for the two Morlet banks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np


MODULES = (
    ("phasepair_core.signal", "phasepair_core/_morlet_oracle_bytes.py"),
    ("phaseset_core.morlet", "phaseset_core/_morlet_oracle_bytes.py"),
)


def _load_artifact(path: Path) -> bytes:
    spec = importlib.util.spec_from_file_location("_private_morlet_artifact", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("artifact import specification failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CANONICAL_BANK_BYTES


def _parse(raw: bytes, lengths: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    cursor = 0
    kernels = []
    for expected_length in lengths:
        (observed_length,) = struct.unpack_from(">H", raw, cursor)
        cursor += 2
        if observed_length != expected_length:
            raise RuntimeError("artifact support mismatch")
        values = np.frombuffer(raw, dtype=">f8", count=2 * expected_length, offset=cursor)
        cursor += 16 * expected_length
        pairs = values.reshape(expected_length, 2)
        kernel = np.empty(expected_length, dtype=np.complex128)
        kernel.real = pairs[:, 0]
        kernel.imag = pairs[:, 1]
        kernels.append(kernel)
    if cursor != len(raw):
        raise RuntimeError("artifact trailing bytes")
    return tuple(kernels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    artifact_root = args.artifact_root.resolve(strict=True)
    sys.path.insert(0, str(source_root / "src"))
    observations = []
    for module_name, relative_artifact in MODULES:
        module = importlib.import_module(module_name)
        if module_name == "phasepair_core.signal":
            formula_receipt = module._build_morlet_formula_bank()
        else:
            formula_receipt = module._build_formula_bank()
        formula = formula_receipt.bands
        raw = _load_artifact(artifact_root / relative_artifact)
        canonical = _parse(raw, module.MORLET_LENGTHS)
        formula_sha = module.morlet_bank_sha256(formula)
        differences = [
            np.abs(band.kernel - kernel) for band, kernel in zip(formula, canonical, strict=True)
        ]
        observations.append(
            {
                "module": module_name,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "formula_sha256": formula_sha,
                "max_complex_abs_error": max(float(np.max(value)) for value in differences),
                "max_component_abs_error": max(
                    max(
                        float(np.max(np.abs(band.kernel.real - kernel.real))),
                        float(np.max(np.abs(band.kernel.imag - kernel.imag))),
                    )
                    for band, kernel in zip(formula, canonical, strict=True)
                ),
                "formula_zero_dc_errors": [
                    float(value) for value in formula_receipt.zero_dc_errors
                ],
                "formula_unit_energy_errors": [
                    float(value) for value in formula_receipt.unit_energy_errors
                ],
                "canonical_zero_dc_errors": [
                    float(np.abs(np.sum(kernel, dtype=np.complex128)))
                    for kernel in canonical
                ],
                "canonical_unit_energy_errors": [
                    float(
                        np.abs(
                            np.sum(
                                np.square(np.abs(kernel), dtype=np.float64),
                                dtype=np.float64,
                            )
                            - np.float64(1.0)
                        )
                    )
                    for kernel in canonical
                ],
            }
        )
    print(json.dumps(observations, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
