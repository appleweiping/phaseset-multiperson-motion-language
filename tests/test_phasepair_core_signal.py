"""Contract tests for the data-free PhasePair signal core.

These tests use synthetic arrays only.  They produce no dataset metric, model
result, scientific result, or execution authority: ``NO_RESULT``.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import struct

import numpy as np
import pytest

import phasepair_core.signal as signal_module
from phasepair_core.signal import (
    ENERGY_FLOOR_RECEIPT_HOLD_CODE,
    EXECUTION_SCOPE,
    MORLET_EXPECTED_UNIT_ENERGY_ERRORS,
    MORLET_FREQUENCIES_HZ,
    MORLET_LENGTHS,
    MORLET_ORACLE_SHA256,
    MORLET_SIGMAS,
    MORLET_UNIT_ENERGY_BOUND,
    MORLET_ZERO_DC_BOUND,
    MorletBand,
    RelationDescriptor,
    SAMPLE_RATE_HZ,
    SCIENTIFIC_STATUS,
    SIGNAL_CONTRACT_FAMILY_ID,
    SignalContractHold,
    SignalNumericError,
    SignalValidationError,
    ValidatedPair,
    clamp_coherence,
    dct_group_indices,
    interedit_availability_mask,
    interedit_descriptor,
    interedit_descriptor_from_source,
    morlet_bank_sha256,
    morlet_formula_bank_diagnostic,
    morlet_kernel_bank,
    morlet_length_mask,
    morlet_unit_energy_errors,
    phase_from_delay,
    pooled_relation_descriptor,
    pooled_relation_descriptor_diagnostic,
    principal_phase,
    swap_pair,
    swap_relation_descriptor,
    validate_paired_activity,
    wrap_phase,
    wrapped_delay_seconds,
    zero_invalid_padding,
)


MODULE_STATUS = "NO_RESULT"
LEFT_TO_RIGHT_MUTANT_SHA256 = (
    "d6ead906d84238cf64fb6009e8009ac11b16b0002f30b4294a0a373ed7fd5524"
)


def _pair(actor_a: np.ndarray, actor_b: np.ndarray):
    assert actor_a.shape == actor_b.shape
    mask = np.ones(actor_a.shape[0], dtype=np.bool_)
    return validate_paired_activity(actor_a, actor_b, mask, actor_a.shape[0])


def _five_channel_tone(n: int, bin_index: int, delay: int = 0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    tone = np.cos(2.0 * math.pi * bin_index * (t - delay) / n)
    return np.ascontiguousarray(np.repeat(tone[:, None], 5, axis=1))


def _serialize_morlet_bands(bands: tuple[MorletBand, ...]) -> str:
    digest = hashlib.sha256()
    for band in bands:
        digest.update(struct.pack(">H", band.length))
        for value in band.kernel:
            digest.update(struct.pack(">dd", float(value.real), float(value.imag)))
    return digest.hexdigest()


def _left_to_right_morlet_mutant() -> tuple[
    tuple[MorletBand, ...], tuple[float, ...], tuple[float, ...]
]:
    """Build the contract's explicit scalar-accumulator must-kill mutant."""

    def complex_sum(values: np.ndarray) -> np.complex128:
        accumulator = np.complex128(0.0 + 0.0j)
        for value in values:
            accumulator = np.complex128(accumulator + np.complex128(value))
        return accumulator

    def float_sum(values: np.ndarray) -> np.float64:
        accumulator = np.float64(0.0)
        for value in values:
            accumulator = np.float64(accumulator + np.float64(value))
        return accumulator

    def magnitude_sum(values: np.ndarray) -> np.float64:
        accumulator = np.float64(0.0)
        for value in values:
            squared = np.float64(abs(complex(value)) ** 2)
            accumulator = np.float64(accumulator + squared)
        return accumulator

    sample_rate = np.float64(SAMPLE_RATE_HZ)
    bands: list[MorletBand] = []
    zero_dc_errors: list[float] = []
    unit_energy_errors: list[float] = []
    for index, (frequency, length, sigma) in enumerate(
        zip(MORLET_FREQUENCIES_HZ, MORLET_LENGTHS, MORLET_SIGMAS), start=1
    ):
        u = np.arange(length, dtype=np.float64) - np.float64((length - 1) / 2.0)
        gaussian = np.ascontiguousarray(
            np.exp(
                -(u * u)
                / (np.float64(2.0) * np.float64(sigma) * np.float64(sigma))
            ),
            dtype=np.float64,
        )
        carrier = np.ascontiguousarray(
            gaussian
            * np.exp(
                -1j
                * np.float64(2.0)
                * np.float64(math.pi)
                * np.float64(frequency)
                * u
                / sample_rate
            ),
            dtype=np.complex128,
        )
        beta = complex_sum(carrier) / float_sum(gaussian)
        raw = np.ascontiguousarray(carrier - beta * gaussian, dtype=np.complex128)
        energy_0 = magnitude_sum(raw)
        kernel = np.ascontiguousarray(raw / np.sqrt(energy_0), dtype=np.complex128)
        dc_sum = complex_sum(kernel)
        energy_1 = magnitude_sum(kernel)
        zero_dc_errors.append(float(np.abs(dc_sum)))
        unit_energy_errors.append(
            float(np.abs(energy_1 - np.float64(1.0)))
        )
        kernel.setflags(write=False)
        bands.append(MorletBand(index, frequency, length, sigma, kernel))
    return tuple(bands), tuple(zero_dc_errors), tuple(unit_energy_errors)


def test_module_is_explicitly_no_result() -> None:
    assert SCIENTIFIC_STATUS == MODULE_STATUS == "NO_RESULT"
    assert EXECUTION_SCOPE == "DATA_FREE_NONPRODUCTION"
    assert SIGNAL_CONTRACT_FAMILY_ID.endswith("/20260824_165840")


def test_strict_pair_tensor_mask_and_length_validation() -> None:
    actor_a = np.zeros((4, 5), dtype=np.float64)
    actor_b = np.zeros((4, 5), dtype=np.float64)
    actor_a[:3, 0] = (1.0, 2.0, 3.0)
    actor_b[:3, 0] = (3.0, 2.0, 1.0)
    mask = np.asarray((True, True, True, False), dtype=np.bool_)

    checked = validate_paired_activity(actor_a, actor_b, mask, 3)
    assert checked.actor_a is actor_a
    assert checked.actor_b is actor_b
    assert checked.valid_length == 3

    bad_mask = np.asarray((True, False, True, False), dtype=np.bool_)
    with pytest.raises(SignalValidationError, match="contiguous prefix"):
        validate_paired_activity(actor_a, actor_b, bad_mask, 2)
    with pytest.raises(SignalValidationError, match="non-boolean integer"):
        validate_paired_activity(actor_a, actor_b, mask, True)
    with pytest.raises(SignalValidationError, match="identical shape and dtype"):
        validate_paired_activity(actor_a.astype(np.float32), actor_b, mask, 3)
    with pytest.raises(SignalValidationError, match="float32 or float64"):
        validate_paired_activity(actor_a.astype(np.float16), actor_b.astype(np.float16), mask, 3)
    with pytest.raises(SignalValidationError, match="C-contiguous"):
        validate_paired_activity(actor_a[:, ::-1], actor_b[:, ::-1], mask, 3)

    class SliceLie(np.ndarray):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(value, np.ndarray):
                return np.ones(value.shape, dtype=value.dtype)
            return value

    class MetadataLie(np.ndarray):
        def __getattribute__(self, name):
            if name in {"dtype", "ndim", "shape", "flags"}:
                raise RuntimeError("DYNAMIC_METADATA_READ")
            return super().__getattribute__(name)

    with pytest.raises(SignalValidationError, match="exact numpy.ndarray"):
        validate_paired_activity(actor_a.view(SliceLie), actor_b, mask, 3)
    with pytest.raises(SignalValidationError, match="exact numpy.ndarray"):
        validate_paired_activity(actor_a.view(MetadataLie), actor_b, mask, 3)
    with pytest.raises(SignalValidationError, match="exact numpy.ndarray"):
        validate_paired_activity(actor_a, actor_b, mask.view(MetadataLie), 3)

    nonfinite = actor_a.copy()
    nonfinite[1, 2] = np.nan
    with pytest.raises(SignalValidationError, match="valid activity samples"):
        validate_paired_activity(nonfinite, actor_b, mask, 3)

    nonzero_padding = actor_a.copy()
    nonzero_padding[3, 0] = 1.0
    with pytest.raises(SignalValidationError, match="positive zero"):
        validate_paired_activity(nonzero_padding, actor_b, mask, 3)
    negative_zero_padding = actor_a.copy()
    negative_zero_padding[3, 0] = -0.0
    with pytest.raises(SignalValidationError, match="positive zero"):
        validate_paired_activity(negative_zero_padding, actor_b, mask, 3)


def test_zero_invalid_padding_is_exact_and_does_not_touch_valid_rows() -> None:
    actor_a = np.arange(30, dtype=np.float32).reshape(6, 5)
    actor_b = -actor_a
    valid_a = actor_a[:4].copy()
    valid_b = actor_b[:4].copy()
    actor_a[4:] = np.nan
    actor_b[4:] = -0.0
    mask = np.asarray((True, True, True, True, False, False), dtype=np.bool_)

    checked = zero_invalid_padding(actor_a, actor_b, mask, 4)
    assert np.array_equal(checked.actor_a[:4], valid_a)
    assert np.array_equal(checked.actor_b[:4], valid_b)
    assert np.array_equal(checked.actor_a[4:], np.zeros((2, 5), dtype=np.float32))
    assert np.array_equal(checked.actor_b[4:], np.zeros((2, 5), dtype=np.float32))
    assert not np.signbit(checked.actor_a[4:]).any()
    assert not np.signbit(checked.actor_b[4:]).any()


def test_swap_pair_revalidates_forged_public_dataclass() -> None:
    actor_a = np.zeros((4, 5), dtype=np.float64)
    actor_b = np.ones((4, 5), dtype=np.float64)
    actor_b[3] = 0.0
    mask = np.asarray((True, True, True, False), dtype=np.bool_)
    checked = validate_paired_activity(actor_a, actor_b, mask, 3)

    swapped = swap_pair(checked)
    assert swapped.actor_a is actor_b
    assert swapped.actor_b is actor_a
    assert swap_pair(swapped).actor_a is actor_a

    class ForgedValidatedPair(ValidatedPair):
        pass

    subclass = ForgedValidatedPair(actor_a, actor_b, mask, 3)
    with pytest.raises(SignalValidationError, match="exact ValidatedPair"):
        swap_pair(subclass)

    nonfinite = actor_a.copy()
    nonfinite[1, 2] = np.nan
    with pytest.raises(SignalValidationError, match="valid activity samples"):
        swap_pair(ValidatedPair(nonfinite, actor_b, mask, 3))

    nonzero_padding = actor_a.copy()
    nonzero_padding[3, 0] = 1.0
    with pytest.raises(SignalValidationError, match="positive zero"):
        swap_pair(ValidatedPair(nonzero_padding, actor_b, mask, 3))

    forged_mask = np.asarray((True, False, True, False), dtype=np.bool_)
    with pytest.raises(SignalValidationError, match="contiguous prefix"):
        swap_pair(ValidatedPair(actor_a, actor_b, forged_mask, 2))


@pytest.mark.parametrize(
    ("n", "groups", "mask"),
    (
        (1, ((), (), ()), (0, 0, 0, 0, 0, 0)),
        (3, ((), (), (1,)), (0, 0, 0, 0, 1, 1)),
        (10, ((), (1,), (2, 3, 4)), (0, 0, 1, 1, 1, 1)),
        (
            299,
            (tuple(range(10, 30)), tuple(range(30, 60)), tuple(range(60, 120))),
            (1, 1, 1, 1, 1, 1),
        ),
    ),
)
def test_interedit_group_and_mask_goldens(
    n: int,
    groups: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    mask: tuple[int, ...],
) -> None:
    observed_groups = dct_group_indices(n)
    assert observed_groups == groups
    assert tuple(interedit_availability_mask(n).tolist()) == mask
    assert len(observed_groups[0]) in (0, 20)
    if n == 299:
        assert tuple(map(len, observed_groups)) == (20, 30, 60)


def test_interedit_zero_energy_stays_active_when_groups_exist() -> None:
    zeros = np.zeros((299, 5), dtype=np.float64)
    result = interedit_descriptor(_pair(zeros, zeros.copy()))
    assert np.array_equal(result.energies, np.zeros((2, 3), dtype=np.float64))
    assert np.array_equal(result.mask, np.ones(6, dtype=np.uint8))
    assert np.isfinite(result.tokens).all()


def test_interedit_source_reads_no_morlet_field() -> None:
    class Source:
        def __init__(self) -> None:
            self.actor_a = np.zeros((10, 5), dtype=np.float64)
            self.actor_b = np.zeros((10, 5), dtype=np.float64)
            self.valid_mask = np.ones(10, dtype=np.bool_)
            self.valid_length = 10
            self.morlet_read_count = 0

        def __getattr__(self, name: str):
            if name.startswith(("morlet", "length_mask", "energy_mask", "s_ab", "coherence")):
                self.morlet_read_count += 1
                raise AssertionError(f"forbidden Morlet read: {name}")
            raise AttributeError(name)

    source = Source()
    result = interedit_descriptor_from_source(source)
    assert source.morlet_read_count == 0
    assert tuple(result.mask.tolist()) == (0, 0, 1, 1, 1, 1)


def test_interedit_actor_swap_is_bitwise_invariant() -> None:
    n = 299
    t = np.arange(n, dtype=np.float64)
    actor_a = np.empty((n, 5), dtype=np.float64)
    actor_b = np.empty((n, 5), dtype=np.float64)
    for channel in range(5):
        actor_a[:, channel] = np.cos(2.0 * math.pi * (14 + channel) * t / n + 0.1 * channel)
        actor_b[:, channel] = np.cos(2.0 * math.pi * (19 + channel) * t / n - 0.2 * channel)
    actor_a = np.ascontiguousarray(actor_a)
    actor_b = np.ascontiguousarray(actor_b)

    result_ab = interedit_descriptor(_pair(actor_a, actor_b))
    result_ba = interedit_descriptor(_pair(actor_b, actor_a))
    assert result_ab.energies.tobytes() == result_ba.energies.tobytes()
    assert result_ab.tokens.tobytes() == result_ba.tokens.tobytes()
    assert result_ab.mask.tobytes() == result_ba.mask.tobytes()


def test_morlet_kernel_bank_invariants_and_oracle_digest() -> None:
    bank = morlet_kernel_bank()
    assert tuple(band.length for band in bank) == MORLET_LENGTHS
    assert tuple(band.frequency_hz for band in bank) == MORLET_FREQUENCIES_HZ
    for band in bank:
        assert not band.kernel.flags.writeable
        assert band.kernel.dtype == np.complex128
        assert band.kernel.ndim == 1
        assert band.kernel.flags.c_contiguous
        assert abs(np.sum(band.kernel, dtype=np.complex128)) <= MORLET_ZERO_DC_BOUND
    assert morlet_bank_sha256(bank) == MORLET_ORACLE_SHA256
    assert morlet_unit_energy_errors(bank) == MORLET_EXPECTED_UNIT_ENERGY_ERRORS
    assert MORLET_UNIT_ENERGY_BOUND == np.finfo(np.float64).eps
    assert MORLET_UNIT_ENERGY_BOUND.hex() == "0x1.0000000000000p-52"
    assert 2.22e-16 < MORLET_UNIT_ENERGY_BOUND


def test_canonical_morlet_uses_exact_five_numpy_reductions_per_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sum = np.sum
    real_square = np.square
    real_sqrt = np.sqrt
    sum_calls: list[tuple[np.ndarray, tuple[object, ...], dict[str, object]]] = []
    square_calls: list[tuple[np.ndarray, tuple[object, ...], dict[str, object]]] = []
    sqrt_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def sum_spy(
        values: np.ndarray, *args: object, **kwargs: object
    ) -> np.generic:
        sum_calls.append((values, args, kwargs))
        return real_sum(values, *args, **kwargs)

    def square_spy(
        values: np.ndarray, *args: object, **kwargs: object
    ) -> np.ndarray:
        square_calls.append((values, args, kwargs))
        return real_square(values, *args, **kwargs)

    def sqrt_spy(value: object, *args: object, **kwargs: object) -> np.generic:
        sqrt_calls.append((value, args, kwargs))
        return real_sqrt(value, *args, **kwargs)

    monkeypatch.setattr(signal_module.np, "sum", sum_spy)
    monkeypatch.setattr(signal_module.np, "square", square_spy)
    monkeypatch.setattr(signal_module.np, "sqrt", sqrt_spy)
    bank = morlet_kernel_bank()

    assert len(bank) == 6
    assert len(sum_calls) == 30
    expected_dtypes = (
        np.complex128,
        np.float64,
        np.float64,
        np.complex128,
        np.float64,
    ) * 6
    for (values, args, kwargs), expected_dtype in zip(
        sum_calls, expected_dtypes, strict=True
    ):
        assert type(values) is np.ndarray
        assert values.ndim == 1
        assert values.flags.c_contiguous
        assert args == ()
        assert tuple(kwargs) == ("axis", "dtype", "out", "keepdims")
        assert kwargs == {
            "axis": 0,
            "dtype": expected_dtype,
            "out": None,
            "keepdims": False,
        }
        assert "initial" not in kwargs
        assert "where" not in kwargs

    assert len(square_calls) == 12
    for values, args, kwargs in square_calls:
        assert type(values) is np.ndarray
        assert values.dtype == np.float64
        assert values.ndim == 1
        assert values.flags.c_contiguous
        assert args == ()
        assert kwargs == {"dtype": np.float64}

    assert len(sqrt_calls) == 6
    for value, args, kwargs in sqrt_calls:
        assert type(value) is np.float64
        assert args == ()
        assert kwargs == {}


def test_kernel01_kills_explicit_left_to_right_reduction_mutant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands, zero_dc_errors, unit_energy_errors = _left_to_right_morlet_mutant()
    observed_sha = _serialize_morlet_bands(bands)
    one_ulp = MORLET_UNIT_ENERGY_BOUND

    assert morlet_bank_sha256(bands) == observed_sha
    assert observed_sha != MORLET_ORACLE_SHA256
    assert any(
        not np.array_equal(mutant.kernel, canonical.kernel)
        for mutant, canonical in zip(bands, morlet_kernel_bank(), strict=True)
    )
    if signal_module._morlet_runtime_identity() in {
        ("CPython", (3, 12, 0), "2.4.6", "Windows", "x86_64"),
        ("CPython", (3, 14, 5), "2.4.6", "Windows", "x86_64"),
    }:
        assert observed_sha == LEFT_TO_RIGHT_MUTANT_SHA256
        assert tuple(error / one_ulp for error in unit_energy_errors) == (
            2.5,
            1.5,
            0.5,
            1.0,
            1.0,
            0.0,
        )
        assert zero_dc_errors[1] == 3.0493334034159067e-16
        assert zero_dc_errors[1] > MORLET_ZERO_DC_BOUND

    receipt = signal_module._MorletBuildReceipt(
        bands,
        zero_dc_errors,
        unit_energy_errors,
    )
    monkeypatch.setattr(
        signal_module,
        "_morlet_runtime_identity",
        lambda: ("CPython", (3, 12, 12), "2.4.6", "Linux", "x86_64"),
    )
    monkeypatch.setattr(signal_module, "_build_morlet_formula_bank", lambda: receipt)
    with pytest.raises(SignalContractHold, match="PORTABLE_FORMULA_DIGEST_MISMATCH"):
        morlet_kernel_bank()


def test_kernel01_kills_decimal_rounded_energy_bound_mutant() -> None:
    errors = morlet_unit_energy_errors(morlet_kernel_bank())
    rounded_mutant = 2.22e-16
    assert rounded_mutant.hex() == "0x1.ffe5ab7e8ad5ep-53"
    assert tuple(index + 1 for index, error in enumerate(errors) if error > rounded_mutant) == (
        1,
        3,
        4,
        5,
    )


@pytest.mark.parametrize(
    ("n", "expected"),
    (
        (1, (0, 0, 0, 0, 0, 0)),
        (15, (0, 0, 0, 0, 0, 0)),
        (16, (0, 0, 0, 0, 0, 1)),
        (24, (0, 0, 0, 0, 1, 1)),
        (54, (0, 0, 1, 1, 1, 1)),
        (80, (0, 1, 1, 1, 1, 1)),
        (120, (1, 1, 1, 1, 1, 1)),
        (299, (1, 1, 1, 1, 1, 1)),
    ),
)
def test_morlet_valid_length_mask(n: int, expected: tuple[int, ...]) -> None:
    assert tuple(morlet_length_mask(n).tolist()) == expected


def test_missing_diagnostic_floor_is_an_explicit_hold() -> None:
    tone = _five_channel_tone(299, 17)
    with pytest.raises(SignalContractHold, match="HOLD_TRAIN_ENERGY_FLOORS") as caught:
        pooled_relation_descriptor_diagnostic(_pair(tone, tone.copy()))
    assert caught.value.code == "HOLD_TRAIN_ENERGY_FLOORS"


def test_canonical_descriptor_cannot_accept_a_caller_supplied_bank() -> None:
    tone = _five_channel_tone(299, 17)
    pair = _pair(tone, tone.copy())
    floors = np.zeros(6, dtype=np.float64)
    diagnostic_bands = morlet_formula_bank_diagnostic()
    forged_bands = tuple(
        replace(band, kernel=np.zeros_like(band.kernel)) for band in diagnostic_bands
    )

    with pytest.raises(
        SignalContractHold, match=ENERGY_FLOOR_RECEIPT_HOLD_CODE
    ) as raw_floor_hold:
        pooled_relation_descriptor(pair, energy_floors=floors)
    assert raw_floor_hold.value.code == ENERGY_FLOOR_RECEIPT_HOLD_CODE
    with pytest.raises(TypeError, match="unexpected keyword argument 'bands'"):
        pooled_relation_descriptor(  # type: ignore[call-arg]
            pair, bands=forged_bands
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'bands'"):
        pooled_relation_descriptor_diagnostic(  # type: ignore[call-arg]
            pair, energy_floors=floors, bands=diagnostic_bands
        )
    with pytest.raises(
        SignalContractHold, match=ENERGY_FLOOR_RECEIPT_HOLD_CODE
    ) as no_floor_hold:
        pooled_relation_descriptor(pair)
    assert no_floor_hold.value.code == ENERGY_FLOOR_RECEIPT_HOLD_CODE

    diagnostic = pooled_relation_descriptor_diagnostic(pair, energy_floors=floors)
    assert diagnostic.descriptor_valid.shape == (6,)
    assert SCIENTIFIC_STATUS == "NO_RESULT"


def test_canonical_floor_receipt_schema_holds_after_morlet_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tone = _five_channel_tone(299, 17)
    pair = _pair(tone, tone.copy())
    floors = np.zeros(6, dtype=np.float64)
    calls = 0

    def accepted_morlet_bank():
        nonlocal calls
        calls += 1
        return morlet_formula_bank_diagnostic()

    monkeypatch.setattr(signal_module, "morlet_kernel_bank", accepted_morlet_bank)
    with pytest.raises(
        SignalContractHold, match=ENERGY_FLOOR_RECEIPT_HOLD_CODE
    ) as caught:
        pooled_relation_descriptor(pair, energy_floors=floors)
    assert caught.value.code == ENERGY_FLOOR_RECEIPT_HOLD_CODE
    assert calls == 1

    with pytest.raises(
        SignalContractHold, match=ENERGY_FLOOR_RECEIPT_HOLD_CODE
    ) as no_payload:
        pooled_relation_descriptor(pair)
    assert no_payload.value.code == ENERGY_FLOOR_RECEIPT_HOLD_CODE
    assert calls == 2

    with pytest.raises(
        SignalContractHold, match=ENERGY_FLOOR_RECEIPT_HOLD_CODE
    ) as forged_payload:
        pooled_relation_descriptor(pair, energy_floors={"untrusted": floors})
    assert forged_payload.value.code == ENERGY_FLOOR_RECEIPT_HOLD_CODE
    assert calls == 3

    with pytest.raises(TypeError, match="unexpected keyword argument 'floor_receipt'"):
        pooled_relation_descriptor(  # type: ignore[call-arg]
            pair, floor_receipt={"energy_floors": floors}
        )
    assert calls == 3


def test_diagnostic_energy_floor_validation_is_exact_and_fail_closed() -> None:
    zeros = np.zeros((16, 5), dtype=np.float64)
    pair = _pair(zeros, zeros.copy())
    floors = np.zeros(6, dtype=np.float64)
    result = pooled_relation_descriptor_diagnostic(pair, energy_floors=floors)
    assert result.tokens.shape == (6, 13)

    class FloorArraySubclass(np.ndarray):
        pass

    with pytest.raises(SignalValidationError, match="exact numpy.ndarray"):
        pooled_relation_descriptor_diagnostic(
            pair, energy_floors=floors.view(FloorArraySubclass)
        )

    negative_zero = floors.copy()
    negative_zero[2] = -0.0
    with pytest.raises(SignalValidationError, match="exact positive zero"):
        pooled_relation_descriptor_diagnostic(pair, energy_floors=negative_zero)

    noncontiguous = np.zeros(12, dtype=np.float64)[::2]
    with pytest.raises(SignalValidationError, match="C-contiguous float64"):
        pooled_relation_descriptor_diagnostic(pair, energy_floors=noncontiguous)

    nonfinite = floors.copy()
    nonfinite[0] = np.nan
    with pytest.raises(SignalValidationError, match="finite and nonnegative"):
        pooled_relation_descriptor_diagnostic(pair, energy_floors=nonfinite)

    negative = floors.copy()
    negative[0] = -1.0
    with pytest.raises(SignalValidationError, match="finite and nonnegative"):
        pooled_relation_descriptor_diagnostic(pair, energy_floors=negative)


@pytest.mark.parametrize(
    ("delay_frames", "expected_phase", "expected_cos", "expected_sin", "expected_delay"),
    (
        (
            4,
            -1.428951909581721,
            0.1413692476828594,
            -0.9899569363409614,
            0.1347702280595757,
        ),
        (
            -4,
            1.428951660692440,
            0.1413694940725253,
            0.9899569011556403,
            -0.1347702045858220,
        ),
    ),
)
def test_known_delay_and_analytic_actor_swap(
    delay_frames: int,
    expected_phase: float,
    expected_cos: float,
    expected_sin: float,
    expected_delay: float,
) -> None:
    actor_a = _five_channel_tone(299, 17)
    actor_b = _five_channel_tone(299, 17, delay_frames)
    floors = np.zeros(6, dtype=np.float64)
    descriptor = pooled_relation_descriptor_diagnostic(
        _pair(actor_a, actor_b), energy_floors=floors
    )
    target = 2

    assert descriptor.coherence[target] >= 0.9999998
    assert descriptor.phase[target] == pytest.approx(expected_phase, abs=1e-9)
    assert descriptor.cosine[target] == pytest.approx(expected_cos, abs=1e-9)
    assert descriptor.sine[target] == pytest.approx(expected_sin, abs=1e-9)
    assert descriptor.delay_seconds[target] == pytest.approx(expected_delay, abs=1e-10)
    assert descriptor.descriptor_valid[target] == 1

    swapped = swap_relation_descriptor(descriptor)
    assert swapped.s_aa[target] == descriptor.s_bb[target]
    assert swapped.s_bb[target] == descriptor.s_aa[target]
    assert swapped.coherence[target] == descriptor.coherence[target]
    assert swapped.cosine[target] == descriptor.cosine[target]
    assert swapped.sine[target] == -descriptor.sine[target]
    assert swapped.delay_seconds[target] == -descriptor.delay_seconds[target]

    direct_ba = pooled_relation_descriptor_diagnostic(
        swap_pair(_pair(actor_a, actor_b)),
        energy_floors=floors,
    )
    assert direct_ba.coherence[target] == pytest.approx(swapped.coherence[target], abs=1e-12)
    assert direct_ba.sine[target] == pytest.approx(swapped.sine[target], abs=1e-12)
    assert direct_ba.delay_seconds[target] == pytest.approx(swapped.delay_seconds[target], abs=1e-12)


def test_antiphase_branch_cut_keeps_direction_free_relation() -> None:
    actor_a = _five_channel_tone(299, 17)
    descriptor = pooled_relation_descriptor_diagnostic(
        _pair(actor_a, np.ascontiguousarray(-actor_a)),
        energy_floors=np.zeros(6, dtype=np.float64),
    )
    target = 2
    assert descriptor.phase[target] == -math.pi
    assert descriptor.coherence[target] >= 0.999999999999
    assert descriptor.cosine[target] <= -0.999999999999
    assert descriptor.relation_valid[target] == 1
    assert descriptor.signed_phase_valid[target] == 0
    assert descriptor.sine[target] == 0.0
    assert descriptor.delay_seconds[target] == 0.0
    assert descriptor.descriptor_valid[target] == 1


def test_phase_wrap_delay_helpers_and_coherence_clamp() -> None:
    assert wrap_phase(math.pi) == -math.pi
    assert wrap_phase(-math.pi) == -math.pi
    assert principal_phase(complex(-1.0, 0.0)) == -math.pi
    assert wrapped_delay_seconds(-math.pi, 2.0) == pytest.approx(0.25)
    delay = 0.1347702280595757
    phase = phase_from_delay(delay, MORLET_FREQUENCIES_HZ[2])
    assert wrapped_delay_seconds(phase, MORLET_FREQUENCIES_HZ[2]) == pytest.approx(delay)

    assert clamp_coherence(-(2.0**-41)) == 0.0
    assert clamp_coherence(1.0 + 2.0**-41) == 1.0
    with pytest.raises(SignalNumericError, match="COHERENCE_NUMERIC_FAIL"):
        clamp_coherence(-(2.0**-39))
    with pytest.raises(SignalNumericError, match="COHERENCE_NUMERIC_FAIL"):
        clamp_coherence(1.0 + 2.0**-39)


@pytest.mark.parametrize(
    "frequency_hz",
    ("2.0", True, 2, np.float64(2.0)),
)
def test_phase_from_delay_rejects_non_exact_builtin_frequency(
    frequency_hz: object,
) -> None:
    with pytest.raises(SignalValidationError, match="exact builtin float"):
        phase_from_delay(0.125, frequency_hz)


@pytest.mark.parametrize("frequency_hz", (float("nan"), float("inf"), 0.0, -2.0))
def test_phase_from_delay_rejects_nonfinite_or_nonpositive_builtin_float(
    frequency_hz: float,
) -> None:
    with pytest.raises(SignalValidationError, match="finite and positive"):
        phase_from_delay(0.125, frequency_hz)


def test_swap_relation_descriptor_rejects_forged_class_arrays_and_fields() -> None:
    actor_a = _five_channel_tone(299, 17)
    actor_b = _five_channel_tone(299, 17, 4)
    descriptor = pooled_relation_descriptor_diagnostic(
        _pair(actor_a, actor_b), energy_floors=np.zeros(6, dtype=np.float64)
    )

    class ForgedRelationDescriptor(RelationDescriptor):
        pass

    forged_class = ForgedRelationDescriptor(
        descriptor.s_aa,
        descriptor.s_bb,
        descriptor.s_ab,
        descriptor.coherence,
        descriptor.phase,
        descriptor.cosine,
        descriptor.sine,
        descriptor.delay_seconds,
        descriptor.length_mask,
        descriptor.energy_mask,
        descriptor.relation_valid,
        descriptor.signed_phase_valid,
        descriptor.descriptor_valid,
        descriptor.tokens,
    )
    with pytest.raises(SignalValidationError, match="exact RelationDescriptor"):
        swap_relation_descriptor(forged_class)

    class ArraySubclass(np.ndarray):
        pass

    forged_array = replace(descriptor, s_aa=descriptor.s_aa.view(ArraySubclass))
    with pytest.raises(SignalValidationError, match="exact numpy.ndarray"):
        swap_relation_descriptor(forged_array)

    wrong_dtype = replace(descriptor, phase=descriptor.phase.astype(np.float32))
    with pytest.raises(SignalValidationError, match="float64"):
        swap_relation_descriptor(wrong_dtype)

    wrong_shape = replace(descriptor, coherence=descriptor.coherence[:5].copy())
    with pytest.raises(SignalValidationError, match="float64"):
        swap_relation_descriptor(wrong_shape)

    nonfinite = descriptor.phase.copy()
    nonfinite[2] = np.nan
    with pytest.raises(SignalValidationError, match="phase must be finite"):
        swap_relation_descriptor(replace(descriptor, phase=nonfinite))

    nonbinary = descriptor.length_mask.copy()
    nonbinary[0] = np.uint8(2)
    with pytest.raises(SignalValidationError, match="binary 0/1"):
        swap_relation_descriptor(replace(descriptor, length_mask=nonbinary))

    coherence_lie = descriptor.coherence.copy()
    coherence_lie[2] = coherence_lie[2] - 0.125
    with pytest.raises(SignalValidationError, match="coherence is inconsistent"):
        swap_relation_descriptor(replace(descriptor, coherence=coherence_lie))

    phase_lie = descriptor.phase.copy()
    phase_lie[2] = wrap_phase(float(phase_lie[2]) + 0.125)
    with pytest.raises(SignalValidationError, match="phase is inconsistent"):
        swap_relation_descriptor(replace(descriptor, phase=phase_lie))

    token_lie = descriptor.tokens.copy()
    token_lie[2, 3] = token_lie[2, 3] + 0.125
    with pytest.raises(SignalValidationError, match="tokens is inconsistent"):
        swap_relation_descriptor(replace(descriptor, tokens=token_lie))

    descriptor_mask_lie = descriptor.descriptor_valid.copy()
    descriptor_mask_lie[2] = np.uint8(1 - descriptor_mask_lie[2])
    with pytest.raises(SignalValidationError, match="descriptor_valid is inconsistent"):
        swap_relation_descriptor(
            replace(descriptor, descriptor_valid=descriptor_mask_lie)
        )


def test_swap_relation_descriptor_rejects_negative_zero_padding() -> None:
    zeros = np.zeros((16, 5), dtype=np.float64)
    descriptor = pooled_relation_descriptor_diagnostic(
        _pair(zeros, zeros.copy()), energy_floors=np.zeros(6, dtype=np.float64)
    )
    assert tuple(descriptor.length_mask.tolist()) == (0, 0, 0, 0, 0, 1)
    forged_padding = descriptor.s_aa.copy()
    forged_padding[0] = -0.0
    with pytest.raises(SignalValidationError, match="exact positive zero"):
        swap_relation_descriptor(replace(descriptor, s_aa=forged_padding))


def test_swap_relation_descriptor_is_a_bitwise_involution() -> None:
    actor_a = _five_channel_tone(299, 17)
    actor_b = _five_channel_tone(299, 17, 4)
    descriptor = pooled_relation_descriptor_diagnostic(
        _pair(actor_a, actor_b), energy_floors=np.zeros(6, dtype=np.float64)
    )
    round_trip = swap_relation_descriptor(swap_relation_descriptor(descriptor))
    for field in RelationDescriptor.__dataclass_fields__:
        original = getattr(descriptor, field)
        observed = getattr(round_trip, field)
        assert original.dtype == observed.dtype
        assert original.shape == observed.shape
        assert original.tobytes(order="C") == observed.tobytes(order="C")


def test_padding_canonicalization_preserves_valid_descriptor() -> None:
    valid_a = _five_channel_tone(299, 17)
    valid_b = _five_channel_tone(299, 17, 4)
    padded_a = np.empty((300, 5), dtype=np.float64)
    padded_b = np.empty((300, 5), dtype=np.float64)
    padded_a[:299] = valid_a
    padded_b[:299] = valid_b
    padded_a[299] = np.nan
    padded_b[299] = -123.0
    mask = np.zeros(300, dtype=np.bool_)
    mask[:299] = True

    canonical = zero_invalid_padding(padded_a, padded_b, mask, 299)
    unpadded = _pair(valid_a, valid_b)
    floors = np.zeros(6, dtype=np.float64)
    from_padded = pooled_relation_descriptor_diagnostic(
        canonical, energy_floors=floors
    )
    from_unpadded = pooled_relation_descriptor_diagnostic(
        unpadded, energy_floors=floors
    )

    assert from_padded.s_aa.tobytes() == from_unpadded.s_aa.tobytes()
    assert from_padded.s_bb.tobytes() == from_unpadded.s_bb.tobytes()
    assert from_padded.s_ab.tobytes() == from_unpadded.s_ab.tobytes()
    assert from_padded.tokens.tobytes() == from_unpadded.tokens.tobytes()
