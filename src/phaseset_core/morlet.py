"""Independent 20-Hz Morlet oracle for PhaseSet skeleton activity.

PhasePair v0.1.0 remains a byte-frozen 30-Hz oracle.  PhaseSet consumes motion
resampled to 20 Hz, so reusing those tap bytes would shift every physical center
frequency by 2/3.  This module retains the registered six center frequencies
and the same zero-DC, unit-energy, three-cycle formula, while deriving a new
20-Hz bank with its own schema and immutable digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import platform
import struct
import sys
from typing import Final

import numpy as np

from phaseset_core._morlet_oracle_bytes import (
    CANONICAL_BANK_BYTES as _MORLET_CANONICAL_BANK_BYTES,
    ORACLE_EXPORT_FORMULA_MODULE as _MORLET_EXPORT_FORMULA_MODULE,
    ORACLE_EXPORT_RUNTIME as MORLET_ORACLE_EXPORT_RUNTIME,
    ORACLE_EXPORT_SCHEMA as MORLET_PORTABLE_SCHEMA,
    ORACLE_EXPORT_SHA256 as _MORLET_EXPORT_SHA256,
)


STATUS: Final = "DATA_FREE_PHASESET_20HZ_MORLET_ORACLE_AUTHORITY0"
PHASESET_MORLET_SCHEMA: Final = "phaseset-morlet-bank-v1/fs20/three-cycle"
SAMPLE_RATE_HZ: Final = 20.0
MORLET_FREQUENCIES_HZ: Final[tuple[float, ...]] = (
    0.75,
    1.125,
    1.6875,
    2.53125,
    3.796875,
    5.6953125,
)
MORLET_LENGTHS: Final[tuple[int, ...]] = (80, 54, 36, 24, 16, 11)
MORLET_SIGMAS: Final[tuple[float, ...]] = (
    13.333333333333334,
    8.88888888888889,
    5.925925925925926,
    3.950617283950617,
    2.633744855967078,
    1.755829903978052,
)
MORLET_ORACLE_SHA256: Final = (
    "633d754bba12dab0555722f4b8963409a110c8681796cc907d163096e2e42655"
)
MORLET_FORMULA_MAX_ABS_ERROR_BOUND: Final = float.fromhex(
    "0x1.0000000000000p-52"
)
MORLET_PORTABLE_LINUX_FORMULA_SHA256: Final = (
    "926c06aa57a27139d1ed1a9e0d9c3da191717b8a9495d6fff900bda44459d9a3"
)
MORLET_ZERO_DC_BOUND: Final = 2.220446049250313e-16
MORLET_UNIT_ENERGY_BOUND: Final = 2.220446049250313e-16
MORLET_EXPECTED_ZERO_DC_ERRORS: Final[tuple[float, ...]] = (
    1.1102230246251565e-16,
    1.0702444338622552e-16,
    1.747819990421485e-17,
    3.741738622089172e-17,
    1.6653345369377348e-16,
    1.1816510524180758e-16,
)
MORLET_EXPECTED_UNIT_ENERGY_ERRORS: Final[tuple[float, ...]] = (
    0.0,
    2.220446049250313e-16,
    1.1102230246251565e-16,
    0.0,
    2.220446049250313e-16,
    1.1102230246251565e-16,
)


class PhaseSetMorletError(ValueError):
    """A 20-Hz bank input or numerical invariant is invalid."""


class PhaseSetMorletOracleError(RuntimeError):
    """The rebuilt bank no longer matches the frozen PhaseSet oracle."""


@dataclass(frozen=True, slots=True)
class PhaseSetMorletBand:
    index: int
    frequency_hz: float
    length: int
    sigma: float
    kernel: np.ndarray


@dataclass(frozen=True, slots=True)
class _BuildReceipt:
    bands: tuple[PhaseSetMorletBand, ...]
    zero_dc_errors: tuple[float, ...]
    unit_energy_errors: tuple[float, ...]


def _build_formula_bank() -> _BuildReceipt:
    bands: list[PhaseSetMorletBand] = []
    zero_dc_errors: list[float] = []
    unit_energy_errors: list[float] = []
    for index, (frequency, length, sigma) in enumerate(
        zip(MORLET_FREQUENCIES_HZ, MORLET_LENGTHS, MORLET_SIGMAS, strict=True),
        start=1,
    ):
        if math.ceil(3.0 * SAMPLE_RATE_HZ / frequency) != length:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_SUPPORT_DRIFT")
        if SAMPLE_RATE_HZ / (2.0 * frequency) != sigma:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_SIGMA_DRIFT")
        offsets = np.ascontiguousarray(
            np.arange(length, dtype=np.float64) - (length - 1) / 2.0,
            dtype=np.float64,
        )
        gaussian = np.ascontiguousarray(
            np.exp(-(offsets * offsets) / (2.0 * sigma * sigma)),
            dtype=np.float64,
        )
        carrier = np.ascontiguousarray(
            gaussian
            * np.exp(
                -1j
                * 2.0
                * math.pi
                * frequency
                * offsets
                / SAMPLE_RATE_HZ
            ),
            dtype=np.complex128,
        )
        beta = np.sum(carrier, dtype=np.complex128) / np.sum(
            gaussian, dtype=np.float64
        )
        raw = np.ascontiguousarray(carrier - beta * gaussian, dtype=np.complex128)
        energy = np.sum(
            np.square(np.abs(raw), dtype=np.float64),
            dtype=np.float64,
        )
        if not math.isfinite(float(energy)) or energy <= 0.0:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_RAW_ENERGY_INVALID")
        kernel = np.ascontiguousarray(raw / np.sqrt(energy), dtype=np.complex128)
        zero_dc_errors.append(
            float(np.abs(np.sum(kernel, dtype=np.complex128)))
        )
        unit_energy_errors.append(
            float(
                np.abs(
                    np.sum(
                        np.square(np.abs(kernel), dtype=np.float64),
                        dtype=np.float64,
                    )
                    - np.float64(1.0)
                )
            )
        )
        kernel.setflags(write=False)
        bands.append(PhaseSetMorletBand(index, frequency, length, sigma, kernel))
    return _BuildReceipt(
        tuple(bands),
        tuple(zero_dc_errors),
        tuple(unit_energy_errors),
    )


def morlet_bank_sha256(
    bands: tuple[PhaseSetMorletBand, ...] | None = None,
) -> str:
    """Hash uint16be lengths followed by float64be real/imaginary taps."""

    selected = morlet_kernel_bank() if bands is None else bands
    if type(selected) is not tuple or len(selected) != 6:
        raise PhaseSetMorletError("bands must be the exact six-band tuple")
    digest = hashlib.sha256()
    for expected_index, band in enumerate(selected, start=1):
        if type(band) is not PhaseSetMorletBand or band.index != expected_index:
            raise PhaseSetMorletError("band type or order is invalid")
        digest.update(struct.pack(">H", band.length))
        for value in band.kernel:
            digest.update(struct.pack(">dd", float(value.real), float(value.imag)))
    return digest.hexdigest()


def morlet_formula_bank_diagnostic() -> tuple[PhaseSetMorletBand, ...]:
    """Replay the exact 20-Hz formula without granting empirical authority."""

    return _build_formula_bank().bands


def _canonical_morlet_bank() -> tuple[PhaseSetMorletBand, ...]:
    """Decode the Windows-oracle bytes without replaying platform libm calls."""

    if (
        MORLET_PORTABLE_SCHEMA != "portable-morlet-oracle-bytes-v1"
        or MORLET_ORACLE_EXPORT_RUNTIME != "CPython 3.14.5 / NumPy 2.4.6 / Windows"
        or _MORLET_EXPORT_FORMULA_MODULE != "phaseset_core.morlet"
        or _MORLET_EXPORT_SHA256 != MORLET_ORACLE_SHA256
        or hashlib.sha256(_MORLET_CANONICAL_BANK_BYTES).hexdigest() != MORLET_ORACLE_SHA256
    ):
        raise PhaseSetMorletOracleError("PHASESET_MORLET_PORTABLE_PROVENANCE_MISMATCH")
    cursor = 0
    bands: list[PhaseSetMorletBand] = []
    for index, (frequency, length, sigma) in enumerate(
        zip(MORLET_FREQUENCIES_HZ, MORLET_LENGTHS, MORLET_SIGMAS, strict=True),
        start=1,
    ):
        if cursor + 2 > len(_MORLET_CANONICAL_BANK_BYTES):
            raise PhaseSetMorletOracleError("PHASESET_MORLET_PORTABLE_BYTES_TRUNCATED")
        (encoded_length,) = struct.unpack_from(">H", _MORLET_CANONICAL_BANK_BYTES, cursor)
        cursor += 2
        if encoded_length != length:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_PORTABLE_SUPPORT_MISMATCH")
        stop = cursor + 16 * length
        if stop > len(_MORLET_CANONICAL_BANK_BYTES):
            raise PhaseSetMorletOracleError("PHASESET_MORLET_PORTABLE_BYTES_TRUNCATED")
        values = np.frombuffer(
            _MORLET_CANONICAL_BANK_BYTES,
            dtype=">f8",
            count=2 * length,
            offset=cursor,
        ).reshape(length, 2)
        kernel = np.empty(length, dtype=np.complex128)
        kernel.real = values[:, 0]
        kernel.imag = values[:, 1]
        kernel = np.ascontiguousarray(kernel, dtype=np.complex128)
        if not bool(np.isfinite(kernel).all()):
            raise PhaseSetMorletOracleError("PHASESET_MORLET_PORTABLE_NONFINITE")
        kernel.setflags(write=False)
        bands.append(PhaseSetMorletBand(index, frequency, length, sigma, kernel))
        cursor = stop
    if cursor != len(_MORLET_CANONICAL_BANK_BYTES):
        raise PhaseSetMorletOracleError("PHASESET_MORLET_PORTABLE_TRAILING_BYTES")
    result = tuple(bands)
    if morlet_bank_sha256(result) != MORLET_ORACLE_SHA256:
        raise PhaseSetMorletOracleError("PHASESET_MORLET_DIGEST_MISMATCH")
    return result


def _formula_max_abs_error(
    formula: tuple[PhaseSetMorletBand, ...],
    canonical: tuple[PhaseSetMorletBand, ...],
) -> float:
    if len(formula) != len(canonical):
        raise PhaseSetMorletOracleError("PHASESET_MORLET_FORMULA_EQUIVALENCE_SHAPE")
    maximum = 0.0
    for expected_index, (observed, frozen) in enumerate(
        zip(formula, canonical, strict=True), start=1
    ):
        if (
            observed.index != expected_index
            or frozen.index != expected_index
            or observed.frequency_hz != frozen.frequency_hz
            or observed.length != frozen.length
            or observed.sigma != frozen.sigma
            or observed.kernel.shape != frozen.kernel.shape
        ):
            raise PhaseSetMorletOracleError("PHASESET_MORLET_FORMULA_EQUIVALENCE_SCHEMA")
        difference = np.abs(observed.kernel - frozen.kernel)
        if not bool(np.isfinite(difference).all()):
            raise PhaseSetMorletOracleError("PHASESET_MORLET_FORMULA_EQUIVALENCE_NONFINITE")
        maximum = max(maximum, float(np.max(difference)))
    return maximum


def morlet_formula_max_abs_error() -> float:
    """Return the native-formula distance from the immutable canonical bytes."""

    return _formula_max_abs_error(
        morlet_formula_bank_diagnostic(),
        _canonical_morlet_bank(),
    )


def _morlet_runtime_identity() -> tuple[str, tuple[int, int, int], str, str, str]:
    """Return only the fields used to qualify a native formula replay."""

    machine = platform.machine().lower()
    if machine == "amd64":
        machine = "x86_64"
    return (
        platform.python_implementation(),
        sys.version_info[:3],
        np.__version__,
        platform.system(),
        machine,
    )


def _validate_morlet_formula_receipt(
    receipt: _BuildReceipt,
    canonical: tuple[PhaseSetMorletBand, ...],
) -> None:
    """Accept only a registered exact replay or the qualified Linux replay."""

    identity = _morlet_runtime_identity()
    implementation, python_version, numpy_version, system, machine = identity
    supported_family = (
        implementation == "CPython"
        and python_version[:2] in {(3, 12), (3, 13), (3, 14)}
        and numpy_version == "2.4.6"
        and machine == "x86_64"
    )
    formula_digest = morlet_bank_sha256(receipt.bands)
    if _formula_max_abs_error(
        receipt.bands, canonical
    ) > MORLET_FORMULA_MAX_ABS_ERROR_BOUND:
        raise PhaseSetMorletOracleError("PHASESET_MORLET_FORMULA_EQUIVALENCE_FAIL")
    if supported_family and system == "Windows":
        # Runtime identity, rather than an accidentally matching digest, selects
        # the original exact-oracle branch. Any one-ULP receipt drift must HOLD.
        if formula_digest != MORLET_ORACLE_SHA256:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_DIGEST_MISMATCH")
        if receipt.zero_dc_errors != MORLET_EXPECTED_ZERO_DC_ERRORS:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_ZERO_DC_ORACLE_FAIL")
        if receipt.unit_energy_errors != MORLET_EXPECTED_UNIT_ENERGY_ERRORS:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_UNIT_ENERGY_ORACLE_FAIL")
        if any(error > MORLET_ZERO_DC_BOUND for error in receipt.zero_dc_errors):
            raise PhaseSetMorletOracleError("PHASESET_MORLET_ZERO_DC_BOUND_FAIL")
        if any(error > MORLET_UNIT_ENERGY_BOUND for error in receipt.unit_energy_errors):
            raise PhaseSetMorletOracleError("PHASESET_MORLET_UNIT_ENERGY_BOUND_FAIL")
        return
    if supported_family and system == "Linux":
        if formula_digest != MORLET_PORTABLE_LINUX_FORMULA_SHA256:
            raise PhaseSetMorletOracleError(
                "PHASESET_MORLET_PORTABLE_FORMULA_DIGEST_MISMATCH"
            )
        return
    raise PhaseSetMorletOracleError("PHASESET_MORLET_PORTABLE_RUNTIME_UNQUALIFIED")


def morlet_kernel_bank() -> tuple[PhaseSetMorletBand, ...]:
    """Return portable oracle bytes after replaying the unchanged formula."""

    receipt = _build_formula_bank()
    canonical = _canonical_morlet_bank()
    _validate_morlet_formula_receipt(receipt, canonical)
    # The returned physical bank is canonical, not the native diagnostic. Its
    # verified original digest binds the original zero-DC and unit-energy bytes.
    return canonical


def morlet_length_mask(valid_length: object) -> np.ndarray:
    if type(valid_length) is not int:
        raise TypeError("valid_length must be an exact built-in int")
    if not 1 <= valid_length <= 299:
        raise PhaseSetMorletError("valid_length must satisfy 1<=N<=299")
    return np.asarray(
        [valid_length >= support for support in MORLET_LENGTHS],
        dtype=np.uint8,
    )


__all__ = [
    "MORLET_EXPECTED_UNIT_ENERGY_ERRORS",
    "MORLET_EXPECTED_ZERO_DC_ERRORS",
    "MORLET_FREQUENCIES_HZ",
    "MORLET_FORMULA_MAX_ABS_ERROR_BOUND",
    "MORLET_LENGTHS",
    "MORLET_ORACLE_SHA256",
    "MORLET_ORACLE_EXPORT_RUNTIME",
    "MORLET_PORTABLE_LINUX_FORMULA_SHA256",
    "MORLET_PORTABLE_SCHEMA",
    "MORLET_SIGMAS",
    "MORLET_UNIT_ENERGY_BOUND",
    "MORLET_ZERO_DC_BOUND",
    "PHASESET_MORLET_SCHEMA",
    "PhaseSetMorletBand",
    "PhaseSetMorletError",
    "PhaseSetMorletOracleError",
    "SAMPLE_RATE_HZ",
    "STATUS",
    "morlet_bank_sha256",
    "morlet_formula_bank_diagnostic",
    "morlet_formula_max_abs_error",
    "morlet_kernel_bank",
    "morlet_length_mask",
]
