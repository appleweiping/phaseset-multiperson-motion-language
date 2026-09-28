"""Portable canonical-byte checks kept separate from frozen Morlet tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from phasepair_core import signal as legacy
from phaseset_core import morlet as phaseset


ORACLE_RUNTIME = "CPython 3.14.5 / NumPy 2.4.6 / Windows"


def _is_registered_oracle_runtime() -> bool:
    return legacy._morlet_runtime_identity() in {
        ("CPython", (3, 12, 0), "2.4.6", "Windows", "x86_64"),
        ("CPython", (3, 14, 5), "2.4.6", "Windows", "x86_64"),
    }


@pytest.mark.parametrize("module", (legacy, phaseset))
def test_canonical_bytes_keep_the_original_digest_and_formula_is_bounded(module) -> None:
    canonical = module.morlet_kernel_bank()
    formula = module.morlet_formula_bank_diagnostic()
    assert module.MORLET_PORTABLE_SCHEMA == "portable-morlet-oracle-bytes-v1"
    assert module.MORLET_ORACLE_EXPORT_RUNTIME == ORACLE_RUNTIME
    assert module.morlet_bank_sha256(canonical) == module.MORLET_ORACLE_SHA256
    assert module.morlet_formula_max_abs_error() <= module.MORLET_FORMULA_MAX_ABS_ERROR_BOUND
    zero_dc_errors = tuple(
        float(np.abs(np.sum(band.kernel, dtype=np.complex128))) for band in canonical
    )
    unit_energy_errors = tuple(
        float(
            np.abs(
                np.sum(
                    np.square(np.abs(band.kernel), dtype=np.float64),
                    dtype=np.float64,
                )
                - np.float64(1.0)
            )
        )
        for band in canonical
    )
    assert all(error <= module.MORLET_ZERO_DC_BOUND for error in zero_dc_errors)
    assert all(error <= module.MORLET_UNIT_ENERGY_BOUND for error in unit_energy_errors)
    if _is_registered_oracle_runtime():
        assert module.morlet_bank_sha256(formula) == module.MORLET_ORACLE_SHA256
        assert module.morlet_formula_max_abs_error() == 0.0


@pytest.mark.parametrize(
    ("module", "builder_name", "exception", "message", "python_version"),
    (
        (
            legacy,
            "_build_morlet_formula_bank",
            legacy.SignalContractHold,
            "HOLD_MORLET_ORACLE_DIGEST_MISMATCH",
            (3, 12, 0),
        ),
        (
            legacy,
            "_build_morlet_formula_bank",
            legacy.SignalContractHold,
            "HOLD_MORLET_ORACLE_DIGEST_MISMATCH",
            (3, 14, 5),
        ),
        (
            phaseset,
            "_build_formula_bank",
            phaseset.PhaseSetMorletOracleError,
            "PHASESET_MORLET_DIGEST_MISMATCH",
            (3, 12, 0),
        ),
        (
            phaseset,
            "_build_formula_bank",
            phaseset.PhaseSetMorletOracleError,
            "PHASESET_MORLET_DIGEST_MISMATCH",
            (3, 14, 5),
        ),
    ),
)
def test_registered_windows_identity_rejects_within_bound_receipt_drift(
    monkeypatch: pytest.MonkeyPatch,
    module,
    builder_name: str,
    exception: type[Exception],
    message: str,
    python_version: tuple[int, int, int],
) -> None:
    canonical = module._canonical_morlet_bank()
    changed_kernel = np.array(canonical[0].kernel, copy=True, order="C")
    changed_kernel.real[0] = np.nextafter(changed_kernel.real[0], np.inf)
    changed_kernel.setflags(write=False)
    changed_band = replace(canonical[0], kernel=changed_kernel)
    changed_bands = (changed_band, *canonical[1:])
    difference = module._formula_max_abs_error(changed_bands, canonical)
    assert 0.0 < difference <= module.MORLET_FORMULA_MAX_ABS_ERROR_BOUND

    original_receipt = getattr(module, builder_name)()
    changed_receipt = replace(original_receipt, bands=changed_bands)
    monkeypatch.setattr(
        module,
        "_morlet_runtime_identity",
        lambda: ("CPython", python_version, "2.4.6", "Windows", "x86_64"),
    )
    monkeypatch.setattr(module, builder_name, lambda: changed_receipt)
    with pytest.raises(exception, match=message):
        module.morlet_kernel_bank()


@pytest.mark.parametrize(
    ("module", "exception"),
    (
        (legacy, legacy.SignalContractHold),
        (phaseset, phaseset.PhaseSetMorletOracleError),
    ),
)
def test_unqualified_hardware_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    module,
    exception: type[Exception],
) -> None:
    monkeypatch.setattr(
        module,
        "_morlet_runtime_identity",
        lambda: ("CPython", (3, 12, 12), "2.4.6", "Linux", "aarch64"),
    )
    with pytest.raises(exception, match="PORTABLE_RUNTIME_UNQUALIFIED"):
        module.morlet_kernel_bank()


def test_legacy_canonical_bytes_reject_a_single_byte_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = bytearray(legacy._MORLET_CANONICAL_BANK_BYTES)
    changed[-1] ^= 1
    monkeypatch.setattr(legacy, "_MORLET_CANONICAL_BANK_BYTES", bytes(changed))
    with pytest.raises(legacy.SignalContractHold, match="PORTABLE_PROVENANCE"):
        legacy.morlet_kernel_bank()


def test_phaseset_canonical_bytes_reject_a_single_byte_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = bytearray(phaseset._MORLET_CANONICAL_BANK_BYTES)
    changed[-1] ^= 1
    monkeypatch.setattr(phaseset, "_MORLET_CANONICAL_BANK_BYTES", bytes(changed))
    with pytest.raises(phaseset.PhaseSetMorletOracleError, match="PORTABLE_PROVENANCE"):
        phaseset.morlet_kernel_bank()


@pytest.mark.parametrize("module", (legacy, phaseset))
@pytest.mark.parametrize("python_version", ((3, 12, 1), (3, 13, 2), (3, 14, 6)))
def test_supported_python_patch_has_no_arbitrary_identity_gate(
    monkeypatch: pytest.MonkeyPatch, module, python_version: tuple[int, int, int]
) -> None:
    """Receipt routing test, not qualification of the simulated interpreter."""
    canonical = module._canonical_morlet_bank()
    builder = (
        module._build_morlet_formula_bank
        if module is legacy else module._build_formula_bank
    )
    receipt = builder()
    windows_receipt = replace(
        receipt, bands=canonical,
        zero_dc_errors=tuple(
            float(np.abs(np.sum(band.kernel, dtype=np.complex128)))
            for band in canonical
        ),
        unit_energy_errors=module.MORLET_EXPECTED_UNIT_ENERGY_ERRORS,
    )
    monkeypatch.setattr(
        module, "_morlet_runtime_identity",
        lambda: ("CPython", python_version, "2.4.6", "Windows", "x86_64"),
    )
    module._validate_morlet_formula_receipt(windows_receipt, canonical)
    if module.morlet_bank_sha256(receipt.bands) == module.MORLET_PORTABLE_LINUX_FORMULA_SHA256:
        monkeypatch.setattr(
            module, "_morlet_runtime_identity",
            lambda: ("CPython", python_version, "2.4.6", "Linux", "x86_64"),
        )
        module._validate_morlet_formula_receipt(receipt, canonical)
