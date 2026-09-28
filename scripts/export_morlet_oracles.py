"""Export the two frozen Morlet formula banks from the registered Windows oracle.

This build-only script imports the unchanged public baseline, replays its
diagnostic formulas, verifies the already-frozen bank digests, and writes only
Python modules containing the canonical serialized bytes and source-runtime
provenance.  It consumes no model, dataset, or experiment input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
import platform
import struct
import sys
from types import ModuleType

import numpy as np


EXPECTED_PYTHON = (3, 14, 5)
EXPECTED_NUMPY = "2.4.6"
EXPECTED_PLATFORM = "Windows"
EXPECTED_DIGESTS = {
    "phasepair_core.signal": "4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7",
    "phaseset_core.morlet": "633d754bba12dab0555722f4b8963409a110c8681796cc907d163096e2e42655",
}
OUTPUTS = {
    "phasepair_core.signal": ("phasepair_core", "_morlet_oracle_bytes.py"),
    "phaseset_core.morlet": ("phaseset_core", "_morlet_oracle_bytes.py"),
}


def _serialize(module: ModuleType) -> bytes:
    bands = module.morlet_formula_bank_diagnostic()
    output = bytearray()
    for band in bands:
        output.extend(struct.pack(">H", band.length))
        for value in band.kernel:
            output.extend(struct.pack(">dd", float(value.real), float(value.imag)))
    return bytes(output)


def _render(*, module_name: str, raw: bytes, digest: str) -> str:
    wrapped = "\n".join(
        f'    "{raw.hex()[offset:offset + 96]}"'
        for offset in range(0, len(raw.hex()), 96)
    )
    return f'''"""Generated frozen Morlet bytes; do not hand-edit.

Source formula module: {module_name}
Exporter runtime: CPython 3.14.5 / NumPy 2.4.6 / Windows
Canonical serialized SHA-256: {digest}
"""

from __future__ import annotations

from typing import Final


ORACLE_EXPORT_SCHEMA: Final = "portable-morlet-oracle-bytes-v1"
ORACLE_EXPORT_RUNTIME: Final = "CPython 3.14.5 / NumPy 2.4.6 / Windows"
ORACLE_EXPORT_FORMULA_MODULE: Final = "{module_name}"
ORACLE_EXPORT_SHA256: Final = "{digest}"
CANONICAL_BANK_BYTES: Final = bytes.fromhex(
{wrapped}
)


__all__ = [
    "CANONICAL_BANK_BYTES",
    "ORACLE_EXPORT_FORMULA_MODULE",
    "ORACLE_EXPORT_RUNTIME",
    "ORACLE_EXPORT_SCHEMA",
    "ORACLE_EXPORT_SHA256",
]
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise SystemExit(f"wrong Python oracle: {sys.version}")
    if np.__version__ != EXPECTED_NUMPY:
        raise SystemExit(f"wrong NumPy oracle: {np.__version__}")
    if platform.system() != EXPECTED_PLATFORM:
        raise SystemExit(f"wrong platform oracle: {platform.system()}")
    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    sys.path.insert(0, str(source_root / "src"))

    for module_name, expected_digest in EXPECTED_DIGESTS.items():
        module = importlib.import_module(module_name)
        module_file = Path(module.__file__).resolve(strict=True)
        if source_root not in module_file.parents:
            raise SystemExit(f"module escaped baseline source root: {module_name}")
        raw = _serialize(module)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_digest or module.morlet_bank_sha256(
            module.morlet_formula_bank_diagnostic()
        ) != expected_digest:
            raise SystemExit(f"frozen formula mismatch: {module_name} {digest}")
        package, filename = OUTPUTS[module_name]
        destination = output_root / "src" / package / filename
        if not destination.parent.is_dir():
            raise SystemExit(f"missing staging package directory: {destination.parent}")
        destination.write_text(
            _render(module_name=module_name, raw=raw, digest=digest),
            encoding="ascii",
            newline="\n",
        )
        print(f"EXPORTED {module_name} bytes={len(raw)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
