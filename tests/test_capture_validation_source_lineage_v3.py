"""Focused server-only contracts for capture descriptor source lineage v2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from phaseset_core import capture_prepared_storage as storage
from phaseset_core import capture_validation
from phaseset_core.capture_validation import (
    CaptureDescriptorWindowSource,
    CaptureValidationCapture,
    CaptureValidationSource,
)
from test_phaseset_capture_prepared_storage import _source


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _write_v1(tmp_path: Path) -> tuple[Path, storage.CaptureStorageBuildResult]:
    root = (tmp_path / "capture-store-v1").resolve()
    return root, storage.write_capture_validation_source(root, _source())


def _write_v2(tmp_path: Path) -> tuple[Path, storage.CaptureStorageBuildResult]:
    root = (tmp_path / "capture-store-v2").resolve()
    return root, storage.write_capture_validation_source_v2(root, _source())


def _rewrite_manifest(path: Path, mutate) -> str:
    value = json.loads(path.read_text(encoding="ascii"))
    mutate(value)
    raw = _canonical(value)
    os.chmod(path, 0o600)
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def _replace_first_descriptor(
    source: CaptureValidationSource,
    descriptor: CaptureDescriptorWindowSource | None,
) -> tuple[CaptureValidationCapture, ...]:
    first = source.captures[0]
    changed_window = replace(first.windows[0], descriptor_source=descriptor)
    changed_capture = replace(first, windows=(changed_window, *first.windows[1:]))
    return (changed_capture, *source.captures[1:])


def test_v1_storage_remains_uncached_and_uses_the_original_closed_schema(
    tmp_path: Path,
) -> None:
    _, result = _write_v1(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="ascii"))
    assert manifest["schema"] == storage.SCHEMA == storage.SCHEMA_V1
    assert all(
        "window_ordinal" not in window
        for capture in manifest["captures"]
        for window in capture["windows"]
    )
    loaded = storage.load_capture_validation_source(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    )
    for capture in loaded.captures:
        for window in capture.windows:
            assert window.descriptor_source is None
            with pytest.raises(
                capture_validation.CaptureValidationError,
                match="no authenticated descriptor source lineage",
            ):
                window.descriptor_context(seed=1729)


def test_v2_roundtrip_binds_consumed_manifest_npz_and_canonical_global_ordinal(
    tmp_path: Path,
) -> None:
    root, result = _write_v2(tmp_path)
    manifest_raw = result.manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode("ascii"))
    assert manifest_raw == _canonical(manifest)
    assert manifest["schema"] == storage.SCHEMA_V2
    rows = [window for capture in manifest["captures"] for window in capture["windows"]]
    assert [row["window_ordinal"] for row in rows] == [0, 1, 2]
    assert sorted(row["source_start_frame"] for row in rows) == [0, 0, 300]
    assert [row["window_ordinal"] for row in rows] != [row["source_start_frame"] for row in rows]

    loaded = storage.load_capture_validation_source(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    )
    assert loaded.manifest_sha256 == _source().manifest_sha256
    assert loaded.census_sha256 == result.source_census_sha256
    windows = [window for capture in loaded.captures for window in capture.windows]
    assert len(windows) == len(rows)
    for invalid_seed in (False, -1, 2**64):
        with pytest.raises(
            capture_validation.CaptureValidationError,
            match="exact uint64",
        ):
            windows[0].descriptor_context(seed=invalid_seed)
    for ordinal, (window, row) in enumerate(zip(windows, rows, strict=True)):
        provenance = window.descriptor_source
        assert type(provenance) is CaptureDescriptorWindowSource
        assert provenance.storage_manifest_sha256 == result.manifest_sha256
        assert provenance.source_window_npz_sha256 == row["sha256"]
        assert provenance.window_ordinal == ordinal
        path = root.joinpath(*row["path"].split("/"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        context = window.descriptor_context(seed=2718)
        assert context.prepared_manifest_sha256 == result.manifest_sha256
        assert context.source_batch_sha256 == row["sha256"]
        assert context.window_sha256 == window.window_commitment.hex()
        assert (context.window_ordinal, context.split, context.seed, context.epoch) == (
            ordinal,
            "val",
            2718,
            0,
        )
        assert context.augmentation_yaw_be_hex == "0000000000000000"

    rebuilt = capture_validation._snapshot_source(loaded)
    assert tuple(
        window.descriptor_source for capture in rebuilt.captures for window in capture.windows
    ) == tuple(
        window.descriptor_source for capture in loaded.captures for window in capture.windows
    )


@pytest.mark.parametrize("changed_ordinal", [False, 1, 4])
def test_v2_loader_rejects_nonexact_or_noncanonical_global_ordinals(
    tmp_path: Path,
    changed_ordinal: object,
) -> None:
    _, result = _write_v2(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        value["captures"][0]["windows"][0]["window_ordinal"] = changed_ordinal

    rebound = _rewrite_manifest(result.manifest_path, mutate)
    expected = "exact uint64" if changed_ordinal is False else "canonical global order"
    with pytest.raises(storage.CapturePreparedStorageError, match=expected):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=rebound,
        )


def test_source_rejects_mixed_v1_v2_lineage_and_v1_writer_cannot_discard_v2(
    tmp_path: Path,
) -> None:
    _, result = _write_v2(tmp_path)
    loaded = storage.load_capture_validation_source(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    )
    with pytest.raises(storage.CapturePreparedStorageError, match="cannot discard"):
        storage.write_capture_validation_source(
            (tmp_path / "forbidden-downgrade").resolve(),
            loaded,
        )
    with pytest.raises(
        capture_validation.CaptureValidationError,
        match="cannot mix v1 and v2",
    ):
        CaptureValidationSource(
            loaded.split,
            loaded.manifest_sha256,
            _replace_first_descriptor(loaded, None),
        )


def test_source_directly_rejects_cross_manifest_descriptor_lineage(tmp_path: Path) -> None:
    _, result = _write_v2(tmp_path)
    loaded = storage.load_capture_validation_source(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    )
    original = loaded.captures[0].windows[0].descriptor_source
    assert type(original) is CaptureDescriptorWindowSource
    changed_manifest = "f" * 64 if original.storage_manifest_sha256 != "f" * 64 else "e" * 64
    changed = replace(original, storage_manifest_sha256=changed_manifest)
    with pytest.raises(
        capture_validation.CaptureValidationError,
        match="share one storage manifest",
    ):
        CaptureValidationSource(
            loaded.split,
            loaded.manifest_sha256,
            _replace_first_descriptor(loaded, changed),
        )


def test_source_directly_rejects_noncanonical_global_ordinal(tmp_path: Path) -> None:
    _, result = _write_v2(tmp_path)
    loaded = storage.load_capture_validation_source(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    )
    original = loaded.captures[0].windows[0].descriptor_source
    assert type(original) is CaptureDescriptorWindowSource
    changed = replace(original, window_ordinal=original.window_ordinal + 1)
    with pytest.raises(
        capture_validation.CaptureValidationError,
        match="ordinals must equal canonical global order",
    ):
        CaptureValidationSource(
            loaded.split,
            loaded.manifest_sha256,
            _replace_first_descriptor(loaded, changed),
        )
