"""DATA_FREE_NONPRODUCTION PhasePair primitives bound to contract 165840.

This module contains no dataset, model, optimizer, or training integration.  It
implements only the fixed numerical operators whose inheritance chain closes at
``PHASEPAIR_SIGNAL_SPEC_20260824_165840.md``, including its exact CPython 3.14.5
/ NumPy 2.4.6 Morlet reduction oracle.  The canonical pooled descriptor remains
held until a future contract freezes the training-final energy-floor receipt.

Scientific status: ``NO_RESULT``.  Returning a descriptor is not evidence of a
scientific effect, a passed data gate, or execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import platform
import struct
import sys
from typing import Final, Protocol

import numpy as np

from phasepair_core._morlet_oracle_bytes import (
    CANONICAL_BANK_BYTES as _MORLET_CANONICAL_BANK_BYTES,
    ORACLE_EXPORT_FORMULA_MODULE as _MORLET_EXPORT_FORMULA_MODULE,
    ORACLE_EXPORT_RUNTIME as MORLET_ORACLE_EXPORT_RUNTIME,
    ORACLE_EXPORT_SCHEMA as MORLET_PORTABLE_SCHEMA,
    ORACLE_EXPORT_SHA256 as _MORLET_EXPORT_SHA256,
)


SIGNAL_CONTRACT_FAMILY_ID: Final[str] = (
    "phasepair-scientific-contract-successor-v7/20260824_165840"
)
EXECUTION_SCOPE: Final[str] = "DATA_FREE_NONPRODUCTION"
SCIENTIFIC_STATUS: Final[str] = "NO_RESULT"
SAMPLE_RATE_HZ: Final[float] = 30.0
TOKEN_EPSILON: Final[float] = 1e-12
MAX_PADDED_LENGTH: Final[int] = 300
MAX_VALID_LENGTH: Final[int] = 299

MORLET_FREQUENCIES_HZ: Final[tuple[float, ...]] = (
    0.75,
    1.125,
    1.6875,
    2.53125,
    3.796875,
    5.6953125,
)
MORLET_LENGTHS: Final[tuple[int, ...]] = (120, 80, 54, 36, 24, 16)
MORLET_SIGMAS: Final[tuple[float, ...]] = (
    20.0,
    13.333333333333334,
    8.8888888888888893,
    5.9259259259259256,
    3.9506172839506171,
    2.6337448559670782,
)
MORLET_ORACLE_SHA256: Final[str] = (
    "4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7"
)
MORLET_FORMULA_MAX_ABS_ERROR_BOUND: Final[float] = float.fromhex(
    "0x1.0000000000000p-52"
)
MORLET_PORTABLE_LINUX_FORMULA_SHA256: Final[str] = (
    "3b4cec820d8202d4576d829d37626acc2414f0aeecf430e17c751d9bd1731f6c"
)
MORLET_ZERO_DC_BOUND: Final[float] = 2.22e-16
MORLET_UNIT_ENERGY_BOUND: Final[float] = 2.220446049250313e-16
MORLET_EXPECTED_UNIT_ENERGY_ERRORS: Final[tuple[float, ...]] = (
    MORLET_UNIT_ENERGY_BOUND,
    0.0,
    MORLET_UNIT_ENERGY_BOUND,
    MORLET_UNIT_ENERGY_BOUND,
    MORLET_UNIT_ENERGY_BOUND,
    0.0,
)
ENERGY_FLOOR_RECEIPT_HOLD_CODE: Final[str] = (
    "HOLD_ENERGY_FLOOR_RECEIPT_SCHEMA_UNIMPLEMENTED"
)


class SignalContractError(ValueError):
    """Base class for a fail-closed signal contract violation."""


class SignalValidationError(SignalContractError):
    """Raised when tensor, mask, length, or numeric validation fails."""


class SignalNumericError(SignalContractError):
    """Raised when a contract-bounded numerical invariant is violated."""


class SignalContractHold(SignalContractError):
    """Raised when a required external, frozen input has not been supplied."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ValidatedPair:
    """A strict, prefix-valid pair of five-channel activity tensors."""

    actor_a: np.ndarray
    actor_b: np.ndarray
    valid_mask: np.ndarray
    valid_length: int

    @property
    def valid_a(self) -> np.ndarray:
        return self.actor_a[: self.valid_length]

    @property
    def valid_b(self) -> np.ndarray:
        return self.actor_b[: self.valid_length]


class InterEditSource(Protocol):
    """The complete read allowlist for the pure-DCT InterEdit producer."""

    actor_a: np.ndarray
    actor_b: np.ndarray
    valid_mask: np.ndarray
    valid_length: int


@dataclass(frozen=True)
class InterEditDescriptor:
    """Pure-DCT InterEdit energies, tokens, and paired group mask."""

    group_indices: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    energies: np.ndarray  # [S/D, low/mid/high], float64
    tokens: np.ndarray  # [six slots, 13], float64
    mask: np.ndarray  # [six slots], uint8


@dataclass(frozen=True)
class MorletBand:
    """One immutable member of the fixed six-band Morlet bank."""

    index: int
    frequency_hz: float
    length: int
    sigma: float
    kernel: np.ndarray


@dataclass(frozen=True)
class _MorletBuildReceipt:
    """Internal KERNEL01 values produced by the five canonical reductions."""

    bands: tuple[MorletBand, ...]
    zero_dc_errors: tuple[float, ...]
    unit_energy_errors: tuple[float, ...]


@dataclass(frozen=True)
class RelationDescriptor:
    """Pooled cross-person fields and the 13D full-token input."""

    s_aa: np.ndarray
    s_bb: np.ndarray
    s_ab: np.ndarray
    coherence: np.ndarray
    phase: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    delay_seconds: np.ndarray
    length_mask: np.ndarray
    energy_mask: np.ndarray
    relation_valid: np.ndarray
    signed_phase_valid: np.ndarray
    descriptor_valid: np.ndarray
    tokens: np.ndarray


def _require_json_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SignalValidationError(f"{name} must be a non-boolean integer")
    return int(value)


def _validate_mask(valid_mask: object, padded_length: int, valid_length: object) -> tuple[np.ndarray, int]:
    if type(valid_mask) is not np.ndarray:
        raise SignalValidationError("valid_mask must be an exact numpy.ndarray")
    if valid_mask.dtype != np.bool_:
        raise SignalValidationError("valid_mask dtype must be bool")
    if valid_mask.ndim != 1 or valid_mask.shape != (padded_length,):
        raise SignalValidationError("valid_mask shape must equal [T]")
    if not valid_mask.flags.c_contiguous:
        raise SignalValidationError("valid_mask must be C-contiguous")

    n = _require_json_integer(valid_length, "valid_length")
    if not 1 <= n <= min(padded_length, MAX_VALID_LENGTH):
        raise SignalValidationError("valid_length must satisfy 1 <= N <= min(T, 299)")
    expected = np.zeros(padded_length, dtype=np.bool_)
    expected[:n] = True
    if not np.array_equal(valid_mask, expected):
        raise SignalValidationError("valid_mask must be the exact contiguous prefix mask")
    return valid_mask, n


def _validate_activity_array(value: object, name: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise SignalValidationError(f"{name} must be an exact numpy.ndarray")
    if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise SignalValidationError(f"{name} dtype must be float32 or float64")
    if value.ndim != 2 or value.shape[1] != 5:
        raise SignalValidationError(f"{name} shape must be [T,5]")
    if not value.flags.c_contiguous:
        raise SignalValidationError(f"{name} must be C-contiguous")
    return value


def validate_paired_activity(
    actor_a: object,
    actor_b: object,
    valid_mask: object,
    valid_length: object,
) -> ValidatedPair:
    """Validate an exact pair without copying or mutating it.

    Invalid rows must already be positive floating-point zero.  This prevents
    padding values, signed zero, or NaNs from becoming a hidden signal input.
    """

    a = _validate_activity_array(actor_a, "actor_a")
    b = _validate_activity_array(actor_b, "actor_b")
    if a.shape != b.shape or a.dtype != b.dtype:
        raise SignalValidationError("actor tensors must have identical shape and dtype")
    padded_length = int(a.shape[0])
    if not 1 <= padded_length <= MAX_PADDED_LENGTH:
        raise SignalValidationError("padded tensor length must satisfy 1 <= T <= 300")
    mask, n = _validate_mask(valid_mask, padded_length, valid_length)

    if not np.isfinite(a[:n]).all() or not np.isfinite(b[:n]).all():
        raise SignalValidationError("all valid activity samples must be finite")
    invalid = ~mask
    for value, name in ((a, "actor_a"), (b, "actor_b")):
        padding = value[invalid]
        if padding.size and (np.any(padding != 0.0) or np.any(np.signbit(padding))):
            raise SignalValidationError(f"{name} padding must be exact positive zero")
    return ValidatedPair(a, b, mask, n)


def zero_invalid_padding(
    actor_a: object,
    actor_b: object,
    valid_mask: object,
    valid_length: object,
) -> ValidatedPair:
    """Return C-contiguous copies whose invalid rows are exact positive zero."""

    a = _validate_activity_array(actor_a, "actor_a")
    b = _validate_activity_array(actor_b, "actor_b")
    if a.shape != b.shape or a.dtype != b.dtype:
        raise SignalValidationError("actor tensors must have identical shape and dtype")
    padded_length = int(a.shape[0])
    if not 1 <= padded_length <= MAX_PADDED_LENGTH:
        raise SignalValidationError("padded tensor length must satisfy 1 <= T <= 300")
    mask, n = _validate_mask(valid_mask, padded_length, valid_length)
    if not np.isfinite(a[:n]).all() or not np.isfinite(b[:n]).all():
        raise SignalValidationError("all valid activity samples must be finite")
    a_out = np.array(a, dtype=a.dtype, order="C", copy=True)
    b_out = np.array(b, dtype=b.dtype, order="C", copy=True)
    a_out[~mask] = a.dtype.type(0.0)
    b_out[~mask] = b.dtype.type(0.0)
    return validate_paired_activity(a_out, b_out, mask, n)


def swap_pair(pair: ValidatedPair) -> ValidatedPair:
    """Apply the actor-swap involution without rebuilding a coordinate frame."""

    if type(pair) is not ValidatedPair:
        raise SignalValidationError("pair must be an exact ValidatedPair")
    checked = validate_paired_activity(
        pair.actor_a, pair.actor_b, pair.valid_mask, pair.valid_length
    )
    return ValidatedPair(
        checked.actor_b, checked.actor_a, checked.valid_mask, checked.valid_length
    )


def dct_group_indices(n: object) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return exact non-DC low/mid/high DCT-II bins using integer inequalities."""

    length = _require_json_integer(n, "N")
    if not 1 <= length <= MAX_VALID_LENGTH:
        raise SignalValidationError("N must satisfy 1 <= N <= 299")
    low: list[int] = []
    mid: list[int] = []
    high: list[int] = []
    for m in range(1, length):
        twice_nu_numerator = 30 * m
        if length <= twice_nu_numerator < 3 * length:
            low.append(m)
        elif 3 * length <= twice_nu_numerator < 6 * length:
            mid.append(m)
        elif 6 * length <= twice_nu_numerator <= 12 * length:
            high.append(m)
    return tuple(low), tuple(mid), tuple(high)


def interedit_availability_mask(n: object) -> np.ndarray:
    """Return ``[aL,aL,aM,aM,aH,aH]`` for a finite valid input."""

    groups = dct_group_indices(n)
    available = tuple(np.uint8(bool(group)) for group in groups)
    return np.asarray(
        (available[0], available[0], available[1], available[1], available[2], available[2]),
        dtype=np.uint8,
    )


def _dct_coefficient(values: np.ndarray, channel: int, m: int) -> float:
    n = int(values.shape[0])
    total = 0.0
    for t in range(n):
        total = total + float(values[t, channel]) * math.cos(math.pi * (t + 0.5) * m / n)
    alpha = math.sqrt(1.0 / n) if m == 0 else math.sqrt(2.0 / n)
    return alpha * total


def _dct_group_energy(values: np.ndarray, group: tuple[int, ...]) -> float:
    if not group:
        return 0.0
    total = 0.0
    for channel in range(5):
        for m in group:
            coefficient = _dct_coefficient(values, channel, m)
            total = total + coefficient * coefficient
    return total / (5.0 * len(group))


def interedit_descriptor(pair: ValidatedPair) -> InterEditDescriptor:
    """Compute the pure-DCT InterEdit control without any Morlet dependency."""

    # Revalidate at the public numerical boundary so forged dataclasses fail closed.
    checked = validate_paired_activity(
        pair.actor_a, pair.actor_b, pair.valid_mask, pair.valid_length
    )
    n = checked.valid_length
    actor_a = np.asarray(checked.valid_a, dtype=np.float64, order="C")
    actor_b = np.asarray(checked.valid_b, dtype=np.float64, order="C")
    summed = 0.5 * (actor_a + actor_b)
    differenced = actor_a - actor_b
    groups = dct_group_indices(n)

    energies = np.zeros((2, 3), dtype=np.float64)
    for group_index, group in enumerate(groups):
        energies[0, group_index] = _dct_group_energy(summed, group)
        energies[1, group_index] = _dct_group_energy(differenced, group)

    tokens = np.zeros((6, 13), dtype=np.float64)
    slots = ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2))
    for slot, (component, group_index) in enumerate(slots):
        energy_s = float(energies[0, group_index])
        energy_d = float(energies[1, group_index])
        e_s = math.log(energy_s + TOKEN_EPSILON)
        e_d = math.log(energy_d + TOKEN_EPSILON)
        e_total = math.log(energy_s + energy_d + TOKEN_EPSILON)
        e_selected = e_s if component == 0 else e_d
        tokens[slot, :7] = (e_selected, e_s, e_d, e_total, e_selected - e_total, 0.0, 0.0)
        tokens[slot, 7 + slot] = 1.0

    mask = interedit_availability_mask(n)
    return InterEditDescriptor(groups, energies, tokens, mask)


def interedit_descriptor_from_source(source: InterEditSource) -> InterEditDescriptor:
    """Read exactly the four allowed source attributes and no Morlet fields."""

    pair = validate_paired_activity(
        source.actor_a,
        source.actor_b,
        source.valid_mask,
        source.valid_length,
    )
    return interedit_descriptor(pair)


def _build_morlet_formula_bank() -> _MorletBuildReceipt:
    bands: list[MorletBand] = []
    zero_dc_errors: list[float] = []
    unit_energy_errors: list[float] = []
    for index, (frequency, length, sigma) in enumerate(
        zip(MORLET_FREQUENCIES_HZ, MORLET_LENGTHS, MORLET_SIGMAS), start=1
    ):
        if math.ceil(3.0 * SAMPLE_RATE_HZ / frequency) != length:
            raise SignalNumericError("MORLET_LENGTH_CONTRACT_FAIL")
        u = np.ascontiguousarray(
            np.arange(length, dtype=np.float64) - (length - 1) / 2.0,
            dtype=np.float64,
        )
        gaussian = np.ascontiguousarray(
            np.exp(-(u * u) / (2.0 * sigma * sigma)), dtype=np.float64
        )
        carrier = np.ascontiguousarray(
            gaussian
            * np.exp(
                -1j * 2.0 * math.pi * frequency * u / SAMPLE_RATE_HZ
            ),
            dtype=np.complex128,
        )
        sum_c = np.sum(
            carrier,
            axis=0,
            dtype=np.complex128,
            out=None,
            keepdims=False,
        )
        sum_g = np.sum(
            gaussian,
            axis=0,
            dtype=np.float64,
            out=None,
            keepdims=False,
        )
        beta = sum_c / sum_g
        raw = np.ascontiguousarray(carrier - beta * gaussian, dtype=np.complex128)
        q_raw = np.square(np.abs(raw), dtype=np.float64)
        if not q_raw.flags.c_contiguous:
            raise SignalNumericError("MORLET_KERNEL_LAYOUT_FAIL")
        energy_0 = np.sum(
            q_raw,
            axis=0,
            dtype=np.float64,
            out=None,
            keepdims=False,
        )
        if not math.isfinite(float(energy_0)) or energy_0 <= 0.0:
            raise SignalNumericError("MORLET_KERNEL_ENERGY_FAIL")
        kernel = np.ascontiguousarray(raw / np.sqrt(energy_0), dtype=np.complex128)
        dc_sum = np.sum(
            kernel,
            axis=0,
            dtype=np.complex128,
            out=None,
            keepdims=False,
        )
        q_unit = np.square(np.abs(kernel), dtype=np.float64)
        if not q_unit.flags.c_contiguous:
            raise SignalNumericError("MORLET_KERNEL_LAYOUT_FAIL")
        energy_1 = np.sum(
            q_unit,
            axis=0,
            dtype=np.float64,
            out=None,
            keepdims=False,
        )
        zero_dc_errors.append(float(np.abs(dc_sum)))
        unit_energy_errors.append(
            float(np.abs(energy_1 - np.float64(1.0)))
        )
        kernel.setflags(write=False)
        bands.append(MorletBand(index, frequency, length, sigma, kernel))
    return _MorletBuildReceipt(
        tuple(bands), tuple(zero_dc_errors), tuple(unit_energy_errors)
    )


def morlet_unit_energy_errors(bands: tuple[MorletBand, ...]) -> tuple[float, ...]:
    """Replay the exact NumPy reduction for each unit-energy residual."""

    if len(bands) != 6:
        raise SignalValidationError("bands must contain the exact six Morlet bands")
    errors: list[float] = []
    for band in bands:
        q_unit = np.square(np.abs(band.kernel), dtype=np.float64)
        energy_1 = np.sum(
            q_unit,
            axis=0,
            dtype=np.float64,
            out=None,
            keepdims=False,
        )
        errors.append(float(np.abs(energy_1 - np.float64(1.0))))
    return tuple(errors)


def morlet_formula_bank_diagnostic() -> tuple[MorletBand, ...]:
    """Replay the exact formula for a data-free, ``NO_RESULT`` diagnostic."""

    return _build_morlet_formula_bank().bands


def _canonical_morlet_bank() -> tuple[MorletBand, ...]:
    """Decode the Windows-oracle bytes without replaying platform libm calls."""

    if (
        MORLET_PORTABLE_SCHEMA != "portable-morlet-oracle-bytes-v1"
        or MORLET_ORACLE_EXPORT_RUNTIME != "CPython 3.14.5 / NumPy 2.4.6 / Windows"
        or _MORLET_EXPORT_FORMULA_MODULE != "phasepair_core.signal"
        or _MORLET_EXPORT_SHA256 != MORLET_ORACLE_SHA256
        or hashlib.sha256(_MORLET_CANONICAL_BANK_BYTES).hexdigest() != MORLET_ORACLE_SHA256
    ):
        raise SignalContractHold("HOLD_MORLET_PORTABLE_PROVENANCE_MISMATCH")
    cursor = 0
    bands: list[MorletBand] = []
    for index, (frequency, length, sigma) in enumerate(
        zip(MORLET_FREQUENCIES_HZ, MORLET_LENGTHS, MORLET_SIGMAS), start=1
    ):
        if cursor + 2 > len(_MORLET_CANONICAL_BANK_BYTES):
            raise SignalContractHold("HOLD_MORLET_PORTABLE_BYTES_TRUNCATED")
        (encoded_length,) = struct.unpack_from(">H", _MORLET_CANONICAL_BANK_BYTES, cursor)
        cursor += 2
        if encoded_length != length:
            raise SignalContractHold("HOLD_MORLET_PORTABLE_SUPPORT_MISMATCH")
        stop = cursor + 16 * length
        if stop > len(_MORLET_CANONICAL_BANK_BYTES):
            raise SignalContractHold("HOLD_MORLET_PORTABLE_BYTES_TRUNCATED")
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
            raise SignalContractHold("HOLD_MORLET_PORTABLE_NONFINITE")
        kernel.setflags(write=False)
        bands.append(MorletBand(index, frequency, length, sigma, kernel))
        cursor = stop
    if cursor != len(_MORLET_CANONICAL_BANK_BYTES):
        raise SignalContractHold("HOLD_MORLET_PORTABLE_TRAILING_BYTES")
    result = tuple(bands)
    if morlet_bank_sha256(result) != MORLET_ORACLE_SHA256:
        raise SignalContractHold("HOLD_MORLET_ORACLE_DIGEST_MISMATCH")
    return result


def _formula_max_abs_error(
    formula: tuple[MorletBand, ...], canonical: tuple[MorletBand, ...]
) -> float:
    if len(formula) != len(canonical):
        raise SignalContractHold("HOLD_MORLET_FORMULA_EQUIVALENCE_SHAPE")
    maximum = 0.0
    for expected_index, (observed, frozen) in enumerate(
        zip(formula, canonical), start=1
    ):
        if (
            observed.index != expected_index
            or frozen.index != expected_index
            or observed.frequency_hz != frozen.frequency_hz
            or observed.length != frozen.length
            or observed.sigma != frozen.sigma
            or observed.kernel.shape != frozen.kernel.shape
        ):
            raise SignalContractHold("HOLD_MORLET_FORMULA_EQUIVALENCE_SCHEMA")
        difference = np.abs(observed.kernel - frozen.kernel)
        if not bool(np.isfinite(difference).all()):
            raise SignalContractHold("HOLD_MORLET_FORMULA_EQUIVALENCE_NONFINITE")
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
    receipt: _MorletBuildReceipt,
    canonical: tuple[MorletBand, ...],
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
        raise SignalContractHold("HOLD_MORLET_FORMULA_EQUIVALENCE_FAIL")
    if supported_family and system == "Windows":
        # Runtime identity, rather than an accidentally matching digest, selects
        # the original exact-oracle branch. Any one-ULP receipt drift must HOLD.
        if formula_digest != MORLET_ORACLE_SHA256:
            raise SignalContractHold("HOLD_MORLET_ORACLE_DIGEST_MISMATCH")
        if receipt.unit_energy_errors != MORLET_EXPECTED_UNIT_ENERGY_ERRORS:
            raise SignalNumericError("MORLET_UNIT_ENERGY_ORACLE_FAIL")
        if any(error > MORLET_ZERO_DC_BOUND for error in receipt.zero_dc_errors):
            raise SignalNumericError("MORLET_ZERO_DC_FAIL")
        if any(error > MORLET_UNIT_ENERGY_BOUND for error in receipt.unit_energy_errors):
            raise SignalNumericError("MORLET_UNIT_ENERGY_FAIL")
        return
    if supported_family and system == "Linux":
        if formula_digest != MORLET_PORTABLE_LINUX_FORMULA_SHA256:
            raise SignalContractHold("HOLD_MORLET_PORTABLE_FORMULA_DIGEST_MISMATCH")
        return
    raise SignalContractHold("HOLD_MORLET_PORTABLE_RUNTIME_UNQUALIFIED")


def morlet_kernel_bank() -> tuple[MorletBand, ...]:
    """Return portable oracle bytes after replaying the unchanged formula."""

    receipt = _build_morlet_formula_bank()
    canonical = _canonical_morlet_bank()
    _validate_morlet_formula_receipt(receipt, canonical)
    # The returned physical bank is canonical, not the native diagnostic. Its
    # verified original digest binds the original zero-DC and unit-energy bytes.
    return canonical


def morlet_bank_sha256(bands: tuple[MorletBand, ...] | None = None) -> str:
    """Serialize the bank as uint16be length plus float64be real/imag taps."""

    selected = morlet_kernel_bank() if bands is None else bands
    digest = hashlib.sha256()
    for band in selected:
        digest.update(struct.pack(">H", band.length))
        for value in band.kernel:
            digest.update(struct.pack(">dd", float(value.real), float(value.imag)))
    return digest.hexdigest()


def morlet_length_mask(n: object) -> np.ndarray:
    """Return the fixed three-cycle valid-length mask for all six bands."""

    length = _require_json_integer(n, "N")
    if not 1 <= length <= MAX_VALID_LENGTH:
        raise SignalValidationError("N must satisfy 1 <= N <= 299")
    return np.asarray([length >= support for support in MORLET_LENGTHS], dtype=np.uint8)


def wrap_phase(angle: object) -> float:
    """Canonicalize a finite angle to the half-open interval [-pi, pi)."""

    if isinstance(angle, (bool, np.bool_)) or not isinstance(angle, (int, float, np.number)):
        raise SignalValidationError("phase angle must be numeric")
    value = float(angle)
    if not math.isfinite(value):
        raise SignalValidationError("phase angle must be finite")
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == math.pi:
        wrapped = -math.pi
    return 0.0 if wrapped == 0.0 else wrapped


def principal_phase(value: object) -> float:
    """Return canonical Arg(z), mapping exact +pi to -pi."""

    if not isinstance(value, (complex, float, int, np.number)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise SignalValidationError("cross value must be numeric")
    z = complex(value)
    if not math.isfinite(z.real) or not math.isfinite(z.imag):
        raise SignalValidationError("cross value must be finite")
    phase = math.atan2(z.imag, z.real)
    if phase == math.pi:
        return -math.pi
    return 0.0 if phase == 0.0 else phase


def wrapped_delay_seconds(phase: object, frequency_hz: object) -> float:
    """Convert canonical relative phase to the principal wrapped-delay proxy."""

    if isinstance(frequency_hz, (bool, np.bool_)) or not isinstance(
        frequency_hz, (int, float, np.number)
    ):
        raise SignalValidationError("frequency_hz must be numeric")
    frequency = float(frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise SignalValidationError("frequency_hz must be finite and positive")
    return -wrap_phase(phase) / (2.0 * math.pi * frequency)


def phase_from_delay(delay_seconds: object, frequency_hz: object) -> float:
    """Map a delay to the corresponding canonical relative phase."""

    if isinstance(delay_seconds, (bool, np.bool_)) or not isinstance(
        delay_seconds, (int, float, np.number)
    ):
        raise SignalValidationError("delay_seconds must be numeric")
    delay = float(delay_seconds)
    if not math.isfinite(delay):
        raise SignalValidationError("delay_seconds must be finite")
    if type(frequency_hz) is not float:
        raise SignalValidationError("frequency_hz must be an exact builtin float")
    frequency = frequency_hz
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise SignalValidationError("frequency_hz must be finite and positive")
    return wrap_phase(-2.0 * math.pi * frequency * delay)


def clamp_coherence(value: object) -> float:
    """Apply the exact 2^-40 numerical tolerance around [0,1]."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise SignalValidationError("coherence must be numeric")
    coherence = float(value)
    if not math.isfinite(coherence):
        raise SignalNumericError("COHERENCE_NUMERIC_FAIL")
    tolerance = 2.0**-40
    if coherence < -tolerance or coherence > 1.0 + tolerance:
        raise SignalNumericError("COHERENCE_NUMERIC_FAIL")
    return min(1.0, max(0.0, coherence))


def _sliding_complex_dot(activity: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(activity, len(kernel), axis=0)
    return np.asarray(windows @ kernel, dtype=np.complex128)


def _pooled_cross_fields(
    transformed_a: np.ndarray,
    transformed_b: np.ndarray,
) -> tuple[float, float, complex]:
    if transformed_a.shape != transformed_b.shape or transformed_a.ndim != 2:
        raise SignalNumericError("MORLET_TRANSFORM_SHAPE_FAIL")
    m, channels = transformed_a.shape
    if channels != 5 or m < 1:
        raise SignalNumericError("MORLET_TRANSFORM_SHAPE_FAIL")
    total_aa = 0.0
    total_bb = 0.0
    total_ab = 0.0 + 0.0j
    for channel in range(5):
        for t in range(m):
            value_a = complex(transformed_a[t, channel])
            value_b = complex(transformed_b[t, channel])
            total_aa = total_aa + value_a.real * value_a.real + value_a.imag * value_a.imag
            total_bb = total_bb + value_b.real * value_b.real + value_b.imag * value_b.imag
            total_ab = total_ab + value_a.conjugate() * value_b
    denominator = 5.0 * m
    return total_aa / denominator, total_bb / denominator, total_ab / denominator


def _validate_energy_floors(value: object) -> np.ndarray:
    if value is None:
        raise SignalContractHold("HOLD_TRAIN_ENERGY_FLOORS")
    if type(value) is not np.ndarray:
        raise SignalValidationError("energy_floors must be an exact numpy.ndarray")
    if value.dtype != np.float64 or value.shape != (6,) or not value.flags.c_contiguous:
        raise SignalValidationError("energy_floors must be C-contiguous float64[6]")
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise SignalValidationError("energy_floors must be finite and nonnegative")
    if np.any((value == 0.0) & np.signbit(value)):
        raise SignalValidationError("energy_floors zero values must be exact positive zero")
    return value


def _pooled_relation_descriptor_from_bank(
    checked: ValidatedPair,
    floors: np.ndarray,
    selected_bands: tuple[MorletBand, ...],
) -> RelationDescriptor:
    """Compute from already validated inputs and an internally selected bank."""

    if len(selected_bands) != 6:
        raise SignalValidationError("bands must contain the exact six Morlet bands")

    actor_a = np.asarray(checked.valid_a, dtype=np.float64, order="C")
    actor_b = np.asarray(checked.valid_b, dtype=np.float64, order="C")
    n = checked.valid_length

    s_aa = np.zeros(6, dtype=np.float64)
    s_bb = np.zeros(6, dtype=np.float64)
    s_ab = np.zeros(6, dtype=np.complex128)
    coherence = np.zeros(6, dtype=np.float64)
    phase = np.zeros(6, dtype=np.float64)
    cosine = np.ones(6, dtype=np.float64)
    sine = np.zeros(6, dtype=np.float64)
    delay = np.zeros(6, dtype=np.float64)
    length_mask = morlet_length_mask(n)
    energy_mask = np.zeros(6, dtype=np.uint8)
    relation_valid = np.zeros(6, dtype=np.uint8)
    signed_phase_valid = np.zeros(6, dtype=np.uint8)

    for offset, band in enumerate(selected_bands):
        if band.index != offset + 1 or band.length != MORLET_LENGTHS[offset]:
            raise SignalValidationError("Morlet band order or length drift")
        if not length_mask[offset]:
            continue
        transformed_a = _sliding_complex_dot(actor_a, band.kernel)
        transformed_b = _sliding_complex_dot(actor_b, band.kernel)
        auto_a, auto_b, cross = _pooled_cross_fields(transformed_a, transformed_b)
        if not (
            math.isfinite(auto_a)
            and math.isfinite(auto_b)
            and math.isfinite(cross.real)
            and math.isfinite(cross.imag)
            and auto_a >= 0.0
            and auto_b >= 0.0
        ):
            raise SignalNumericError("MORLET_POOLED_NUMERIC_FAIL")
        s_aa[offset] = auto_a
        s_bb[offset] = auto_b
        s_ab[offset] = cross
        energy_mask[offset] = np.uint8(auto_a >= floors[offset] and auto_b >= floors[offset])

        raw_coherence = abs(cross) ** 2 / (
            (auto_a + TOKEN_EPSILON) * (auto_b + TOKEN_EPSILON)
        )
        coherence[offset] = clamp_coherence(raw_coherence)
        threshold = TOKEN_EPSILON * math.sqrt(auto_a * auto_b)
        relation_valid[offset] = np.uint8(abs(cross) > threshold)
        if not relation_valid[offset]:
            continue

        band_phase = principal_phase(cross)
        phase[offset] = band_phase
        cosine[offset] = math.cos(band_phase)
        branch_valid = abs(math.pi - abs(band_phase)) > 1e-6
        signed_phase_valid[offset] = np.uint8(branch_valid)
        if branch_valid:
            sine[offset] = math.sin(band_phase)
            delay[offset] = wrapped_delay_seconds(band_phase, band.frequency_hz)

    descriptor_valid = (
        length_mask.astype(np.uint8)
        * energy_mask.astype(np.uint8)
        * relation_valid.astype(np.uint8)
    )
    tokens = np.zeros((6, 13), dtype=np.float64)
    for offset, band in enumerate(selected_bands):
        tokens[offset, :7] = (
            math.log(float(s_aa[offset]) + TOKEN_EPSILON),
            math.log(float(s_bb[offset]) + TOKEN_EPSILON),
            float(coherence[offset]),
            float(cosine[offset]),
            float(sine[offset]),
            2.0 * band.frequency_hz * float(delay[offset]),
            float(signed_phase_valid[offset]),
        )
        tokens[offset, 7 + offset] = 1.0

    return RelationDescriptor(
        s_aa,
        s_bb,
        s_ab,
        coherence,
        phase,
        cosine,
        sine,
        delay,
        length_mask,
        energy_mask,
        relation_valid,
        signed_phase_valid,
        descriptor_valid,
        tokens,
    )


def pooled_relation_descriptor(
    pair: ValidatedPair,
    *,
    energy_floors: object = None,
) -> RelationDescriptor:
    """Enter canonical gates while rejecting every raw-floor payload.

    The canonical path always selects :func:`morlet_kernel_bank` itself.  If that
    gate passes, execution still fails closed because the inherited contract does
    not freeze an exact training-final energy-floor receipt schema, allowlist,
    adapter/crop/seed binding, dimensions, or receipt-byte digest.  Raw vectors
    are accepted only by :func:`pooled_relation_descriptor_diagnostic` and have
    no canonical or scientific authority.  ``energy_floors`` is retained solely
    as a fail-closed legacy seam: its value is never read and can never produce a
    descriptor.
    """

    validate_paired_activity(
        pair.actor_a, pair.actor_b, pair.valid_mask, pair.valid_length
    )
    morlet_kernel_bank()
    raise SignalContractHold(ENERGY_FLOOR_RECEIPT_HOLD_CODE)


def pooled_relation_descriptor_diagnostic(
    pair: ValidatedPair,
    *,
    energy_floors: np.ndarray | None = None,
) -> RelationDescriptor:
    """Compute a synthetic portable-oracle replay marked ``NO_RESULT``.

    This diagnostic is not a canonical-bank PASS and cannot grant data, cache,
    training, or scientific eligibility. It replays and bounds the inherited
    native formula internally, then computes with the canonical frozen bytes;
    callers cannot supply a substitute bank.
    """

    checked = validate_paired_activity(
        pair.actor_a, pair.actor_b, pair.valid_mask, pair.valid_length
    )
    floors = _validate_energy_floors(energy_floors)
    bands = morlet_kernel_bank()
    return _pooled_relation_descriptor_from_bank(checked, floors, bands)


def _validate_descriptor_array(
    value: object,
    name: str,
    *,
    dtype: object,
    shape: tuple[int, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise SignalValidationError(f"{name} must be an exact numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype or value.shape != shape or not value.flags.c_contiguous:
        raise SignalValidationError(
            f"{name} must be C-contiguous {expected_dtype.name}{list(shape)}"
        )
    if not np.isfinite(value).all():
        raise SignalValidationError(f"{name} must be finite")
    return value


def _validate_descriptor_mask(value: object, name: str) -> np.ndarray:
    mask = _validate_descriptor_array(value, name, dtype=np.uint8, shape=(6,))
    if np.any((mask != 0) & (mask != 1)):
        raise SignalValidationError(f"{name} must contain only binary 0/1 values")
    return mask


def _require_positive_zero(values: np.ndarray, name: str) -> None:
    if np.any(values != 0.0) or np.any(np.signbit(values)):
        raise SignalValidationError(f"{name} padding must be exact positive zero")


def _require_positive_complex_zero(values: np.ndarray, name: str) -> None:
    if (
        np.any(values.real != 0.0)
        or np.any(values.imag != 0.0)
        or np.any(np.signbit(values.real))
        or np.any(np.signbit(values.imag))
    ):
        raise SignalValidationError(f"{name} padding must be exact positive complex zero")


def _require_bitwise_equal(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if actual.tobytes(order="C") != expected.tobytes(order="C"):
        raise SignalValidationError(f"{name} is inconsistent with descriptor algebra")


def _require_two_ulp_equal(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    for observed, canonical in zip(actual, expected):
        observed_value = float(observed)
        canonical_value = float(canonical)
        if canonical_value == 0.0:
            if observed_value != 0.0 or math.copysign(1.0, observed_value) < 0.0:
                raise SignalValidationError(
                    f"{name} is inconsistent with descriptor algebra"
                )
            continue
        tolerance = 2.0 * max(math.ulp(observed_value), math.ulp(canonical_value))
        if abs(observed_value - canonical_value) > tolerance:
            raise SignalValidationError(f"{name} is inconsistent with descriptor algebra")


def _validate_relation_descriptor(value: object) -> RelationDescriptor:
    """Revalidate a caller-owned descriptor before applying analytic swap."""

    if type(value) is not RelationDescriptor:
        raise SignalValidationError("descriptor must be an exact RelationDescriptor")
    descriptor = value
    s_aa = _validate_descriptor_array(
        descriptor.s_aa, "s_aa", dtype=np.float64, shape=(6,)
    )
    s_bb = _validate_descriptor_array(
        descriptor.s_bb, "s_bb", dtype=np.float64, shape=(6,)
    )
    s_ab = _validate_descriptor_array(
        descriptor.s_ab, "s_ab", dtype=np.complex128, shape=(6,)
    )
    coherence = _validate_descriptor_array(
        descriptor.coherence, "coherence", dtype=np.float64, shape=(6,)
    )
    phase = _validate_descriptor_array(
        descriptor.phase, "phase", dtype=np.float64, shape=(6,)
    )
    cosine = _validate_descriptor_array(
        descriptor.cosine, "cosine", dtype=np.float64, shape=(6,)
    )
    sine = _validate_descriptor_array(
        descriptor.sine, "sine", dtype=np.float64, shape=(6,)
    )
    delay = _validate_descriptor_array(
        descriptor.delay_seconds, "delay_seconds", dtype=np.float64, shape=(6,)
    )
    length_mask = _validate_descriptor_mask(descriptor.length_mask, "length_mask")
    energy_mask = _validate_descriptor_mask(descriptor.energy_mask, "energy_mask")
    relation_valid = _validate_descriptor_mask(
        descriptor.relation_valid, "relation_valid"
    )
    signed_phase_valid = _validate_descriptor_mask(
        descriptor.signed_phase_valid, "signed_phase_valid"
    )
    descriptor_valid = _validate_descriptor_mask(
        descriptor.descriptor_valid, "descriptor_valid"
    )
    tokens = _validate_descriptor_array(
        descriptor.tokens, "tokens", dtype=np.float64, shape=(6, 13)
    )

    if np.any(s_aa < 0.0) or np.any(s_bb < 0.0):
        raise SignalValidationError("auto spectra must be nonnegative")
    allowed_length_masks = {
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 0, 1, 1, 1),
        (0, 0, 1, 1, 1, 1),
        (0, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
    }
    if tuple(int(item) for item in length_mask) not in allowed_length_masks:
        raise SignalValidationError("length_mask is not a canonical Morlet support mask")
    if np.any(energy_mask > length_mask):
        raise SignalValidationError("energy_mask cannot activate a length-masked band")

    padding = length_mask == 0
    _require_positive_zero(s_aa[padding], "s_aa")
    _require_positive_zero(s_bb[padding], "s_bb")
    _require_positive_complex_zero(s_ab[padding], "s_ab")

    expected_coherence = np.zeros(6, dtype=np.float64)
    expected_phase = np.zeros(6, dtype=np.float64)
    expected_cosine = np.ones(6, dtype=np.float64)
    expected_sine = np.zeros(6, dtype=np.float64)
    expected_delay = np.zeros(6, dtype=np.float64)
    expected_relation = np.zeros(6, dtype=np.uint8)
    expected_signed = np.zeros(6, dtype=np.uint8)
    for index, frequency in enumerate(MORLET_FREQUENCIES_HZ):
        if not length_mask[index]:
            continue
        auto_a = float(s_aa[index])
        auto_b = float(s_bb[index])
        cross = complex(s_ab[index])
        cross_power = abs(cross) ** 2
        denominator = (auto_a + TOKEN_EPSILON) * (auto_b + TOKEN_EPSILON)
        auto_product = auto_a * auto_b
        if not (
            math.isfinite(cross_power)
            and math.isfinite(denominator)
            and math.isfinite(auto_product)
            and denominator > 0.0
        ):
            raise SignalValidationError("descriptor spectral algebra must remain finite")
        try:
            expected_coherence[index] = clamp_coherence(cross_power / denominator)
        except SignalNumericError as exc:
            raise SignalValidationError("coherence is inconsistent with spectra") from exc
        threshold = TOKEN_EPSILON * math.sqrt(auto_product)
        expected_relation[index] = np.uint8(abs(cross) > threshold)
        if not expected_relation[index]:
            continue
        expected_phase[index] = principal_phase(cross)
        expected_cosine[index] = math.cos(float(expected_phase[index]))
        branch_valid = abs(math.pi - abs(float(expected_phase[index]))) > 1e-6
        expected_signed[index] = np.uint8(branch_valid)
        if branch_valid:
            expected_sine[index] = math.sin(float(expected_phase[index]))
            expected_delay[index] = wrapped_delay_seconds(
                float(expected_phase[index]), frequency
            )

    expected_descriptor_valid = (
        length_mask.astype(np.uint8)
        * energy_mask.astype(np.uint8)
        * expected_relation.astype(np.uint8)
    )
    expected_tokens = np.zeros((6, 13), dtype=np.float64)
    for index, frequency in enumerate(MORLET_FREQUENCIES_HZ):
        expected_tokens[index, :7] = (
            math.log(float(s_aa[index]) + TOKEN_EPSILON),
            math.log(float(s_bb[index]) + TOKEN_EPSILON),
            float(expected_coherence[index]),
            float(expected_cosine[index]),
            float(expected_sine[index]),
            2.0 * frequency * float(delay[index]),
            float(expected_signed[index]),
        )
        expected_tokens[index, 7 + index] = 1.0

    _require_bitwise_equal(coherence, expected_coherence, "coherence")
    _require_bitwise_equal(phase, expected_phase, "phase")
    _require_bitwise_equal(cosine, expected_cosine, "cosine")
    _require_bitwise_equal(sine, expected_sine, "sine")
    _require_two_ulp_equal(delay, expected_delay, "delay_seconds")
    _require_bitwise_equal(relation_valid, expected_relation, "relation_valid")
    _require_bitwise_equal(signed_phase_valid, expected_signed, "signed_phase_valid")
    _require_bitwise_equal(
        descriptor_valid, expected_descriptor_valid, "descriptor_valid"
    )
    _require_bitwise_equal(tokens, expected_tokens, "tokens")
    return descriptor


def swap_relation_descriptor(descriptor: RelationDescriptor) -> RelationDescriptor:
    """Derive BA fields analytically from AB fields without re-accumulation."""

    descriptor = _validate_relation_descriptor(descriptor)
    swapped_cross = np.asarray(np.conjugate(descriptor.s_ab), dtype=np.complex128)
    swapped_cross[descriptor.length_mask == 0] = np.complex128(0.0 + 0.0j)
    phase = np.zeros(6, dtype=np.float64)
    sine = np.zeros(6, dtype=np.float64)
    delay = np.zeros(6, dtype=np.float64)
    for index in range(6):
        if descriptor.relation_valid[index]:
            phase[index] = principal_phase(complex(swapped_cross[index]))
        if descriptor.signed_phase_valid[index]:
            sine[index] = -float(descriptor.sine[index])
            delay[index] = -float(descriptor.delay_seconds[index])

    tokens = np.zeros((6, 13), dtype=np.float64)
    for index, frequency in enumerate(MORLET_FREQUENCIES_HZ):
        tokens[index, :7] = (
            math.log(float(descriptor.s_bb[index]) + TOKEN_EPSILON),
            math.log(float(descriptor.s_aa[index]) + TOKEN_EPSILON),
            float(descriptor.coherence[index]),
            float(descriptor.cosine[index]),
            float(sine[index]),
            2.0 * frequency * float(delay[index]),
            float(descriptor.signed_phase_valid[index]),
        )
        tokens[index, 7 + index] = 1.0

    swapped = RelationDescriptor(
        np.array(descriptor.s_bb, copy=True),
        np.array(descriptor.s_aa, copy=True),
        swapped_cross,
        np.array(descriptor.coherence, copy=True),
        phase,
        np.array(descriptor.cosine, copy=True),
        sine,
        delay,
        np.array(descriptor.length_mask, copy=True),
        np.array(descriptor.energy_mask, copy=True),
        np.array(descriptor.relation_valid, copy=True),
        np.array(descriptor.signed_phase_valid, copy=True),
        np.array(descriptor.descriptor_valid, copy=True),
        tokens,
    )
    return _validate_relation_descriptor(swapped)


__all__ = [
    "ENERGY_FLOOR_RECEIPT_HOLD_CODE",
    "EXECUTION_SCOPE",
    "InterEditDescriptor",
    "InterEditSource",
    "MAX_PADDED_LENGTH",
    "MAX_VALID_LENGTH",
    "MORLET_FREQUENCIES_HZ",
    "MORLET_LENGTHS",
    "MORLET_EXPECTED_UNIT_ENERGY_ERRORS",
    "MORLET_FORMULA_MAX_ABS_ERROR_BOUND",
    "MORLET_ORACLE_SHA256",
    "MORLET_ORACLE_EXPORT_RUNTIME",
    "MORLET_PORTABLE_LINUX_FORMULA_SHA256",
    "MORLET_PORTABLE_SCHEMA",
    "MORLET_SIGMAS",
    "MORLET_UNIT_ENERGY_BOUND",
    "MORLET_ZERO_DC_BOUND",
    "MorletBand",
    "RelationDescriptor",
    "SAMPLE_RATE_HZ",
    "SCIENTIFIC_STATUS",
    "SIGNAL_CONTRACT_FAMILY_ID",
    "SignalContractError",
    "SignalContractHold",
    "SignalNumericError",
    "SignalValidationError",
    "ValidatedPair",
    "clamp_coherence",
    "dct_group_indices",
    "interedit_availability_mask",
    "interedit_descriptor",
    "interedit_descriptor_from_source",
    "morlet_bank_sha256",
    "morlet_formula_bank_diagnostic",
    "morlet_formula_max_abs_error",
    "morlet_kernel_bank",
    "morlet_length_mask",
    "morlet_unit_energy_errors",
    "phase_from_delay",
    "pooled_relation_descriptor",
    "pooled_relation_descriptor_diagnostic",
    "principal_phase",
    "swap_pair",
    "swap_relation_descriptor",
    "validate_paired_activity",
    "wrap_phase",
    "wrapped_delay_seconds",
    "zero_invalid_padding",
]
