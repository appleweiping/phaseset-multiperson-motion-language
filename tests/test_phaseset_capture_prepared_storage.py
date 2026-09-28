"""Server-only contracts for immutable prepared capture-validation storage."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phaseset_core import capture_prepared_storage as storage
from phaseset_core import frozen_clip_text as clip
from phaseset_core.capture_pooling import CaptureWindowPlan
from phaseset_core.capture_validation import (
    CaptureValidationCapture,
    CaptureValidationSource,
    CaptureValidationWindow,
)
from phaseset_core.contracts import PreparedGroupBatch, group_commitment


EXPECTED_RUNTIME = (
    ("cuda_available", "false"),
    ("cuda_visible_devices", "hidden"),
    ("deterministic_algorithms", "true"),
    ("device", "cpu"),
    ("float32_matmul_precision", "highest"),
    ("huggingface_hub", "0.36.2"),
    ("interop_threads", "1"),
    ("intraop_threads", "1"),
    ("machine", "x86_64"),
    ("mha_fastpath_enabled", "false"),
    ("numpy", "2.4.6"),
    ("python", "3.12.12"),
    ("python_implementation", "CPython"),
    ("sys_byteorder", "little"),
    ("system", "Linux"),
    ("tokenizers", "0.22.2"),
    ("torch", "2.12.0+cu126"),
    ("torch_cuda_build", "12.6"),
    ("transformers", "4.57.3"),
)
EXPECTED_SOURCE_PREFIX = (
    (
        "transformers.modeling_clip",
        50_315,
        "3e779a99827618d6e6d57583ec4f5c0d765321dfa5054a4ffca6a56bb004d499",
    ),
    (
        "transformers.tokenization_clip_fast",
        6_766,
        "67dff0f21a56d0dc18a45f8764f3fafa02ee2b0aeafeb70761c9623f9114dd53",
    ),
    (
        "phaseset_core.training",
        157_160,
        "9f6c21e354a17facda89a542d53e39013477c5e1b1a259781e2446cd394f6d06",
    ),
)


def _id(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


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


def _digest(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).numpy().tobytes(order="C")


def _frozen_text(label: str, axes: tuple[int, ...]) -> clip.FrozenClipTextBatch:
    embeddings = torch.zeros((len(axes), 512), dtype=torch.float32)
    for row, axis in enumerate(axes):
        embeddings[row, axis] = float(row + 1)
    rows = tuple(
        (
            row,
            _id(f"{label}/caption/{row}").hex(),
            _id(f"{label}/caption-bytes/{row}").hex(),
            3,
            3,
            False,
            _id(f"{label}/tokens/{row}").hex(),
            _id(f"{label}/mask/{row}").hex(),
            hashlib.sha256(_tensor_bytes(embeddings[row : row + 1])).hexdigest(),
        )
        for row in range(len(axes))
    )
    snapshot = tuple(sorted(clip.PINNED_FILES))
    source = EXPECTED_SOURCE_PREFIX + (
        (
            "phaseset_core.frozen_clip_text",
            68_448,
            _id("candidate frozen clip source").hex(),
        ),
    )
    snapshot_manifest = _digest(
        {
            "files": [
                {"bytes": size, "name": name, "sha256": digest}
                for name, size, digest in snapshot
            ],
            "model_id": clip.MODEL_ID,
            "revision": clip.REVISION,
            "schema": "phaseset-frozen-clip-snapshot-v1",
        }
    )
    source_manifest = _digest(
        [
            {"bytes": size, "label": name, "sha256": digest}
            for name, size, digest in source
        ]
    )
    runtime_manifest = _digest(dict(EXPECTED_RUNTIME))
    chunks = ((0, len(axes)),)
    cache_key = _digest(
        {
            "batch_size": len(axes),
            "caption_rows": [
                {
                    "attention_mask_int64_sha256": row[7],
                    "caption_utf8_sha256": row[2],
                    "encoded_token_count_with_special_tokens": row[4],
                    "index": row[0],
                    "input_ids_int64_sha256": row[6],
                    "original_token_count_with_special_tokens": row[3],
                    "truncated": row[5],
                }
                for row in rows
            ],
            "chunk_ranges": [list(value) for value in chunks],
            "method_id": clip.METHOD_ID,
            "model_id": clip.MODEL_ID,
            "revision": clip.REVISION,
            "runtime_manifest_sha256": runtime_manifest,
            "snapshot_manifest_sha256": snapshot_manifest,
            "source_manifest_sha256": source_manifest,
            "token_length": clip.TOKEN_LENGTH,
            "truncation": True,
        }
    )
    raw = _tensor_bytes(embeddings)
    receipt = clip.FrozenClipTextReceipt(
        batch_size=len(axes),
        caption_rows=rows,
        chunk_ranges=chunks,
        frozen_embedding_cache_key_sha256=cache_key,
        live_model_manifest_sha256=_id("fixture live model manifest").hex(),
        output_bytes=len(raw),
        output_sha256=hashlib.sha256(raw).hexdigest(),
        output_shape=tuple(embeddings.shape),
        output_stride=tuple(embeddings.stride()),
        runtime_identity=EXPECTED_RUNTIME,
        runtime_manifest_sha256=runtime_manifest,
        snapshot_files=snapshot,
        snapshot_manifest_sha256=snapshot_manifest,
        source_files=source,
        source_manifest_sha256=source_manifest,
    )
    receipt_raw = receipt.canonical_json_bytes()
    return clip.rehydrate_frozen_clip_text_batch(
        embeddings,
        receipt_json_bytes=receipt_raw,
        expected_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
    )


def _groups(label: str, value: float, *, pad: int) -> PreparedGroupBatch:
    actors = tuple(_id(f"{label}/actor/{index}") for index in range(3))
    skeletons = np.zeros((1, 3 + pad, 200, 22, 3), dtype=np.float32)
    skeletons[0, :3, :, :, 0] = value
    skeletons[0, :3, :, :, 1] = np.arange(200, dtype=np.float32)[None, :, None]
    actor_mask = np.zeros((1, 3 + pad), dtype=np.bool_)
    actor_mask[:, :3] = True
    frame_mask = np.ones((1, 200), dtype=np.bool_)
    track_mask = np.zeros((1, 3 + pad, 200, 22), dtype=np.bool_)
    track_mask[:, :3] = True
    return PreparedGroupBatch(
        skeletons,
        actor_mask,
        frame_mask,
        track_mask,
        (actors + (None,) * pad,),
        (group_commitment(actors),),
    )


def _capture(
    label: str,
    values: tuple[float, ...],
    axes: tuple[int, ...],
) -> CaptureValidationCapture:
    window_ids = tuple(_id(f"{label}/window/{index}") for index in range(len(values)))
    plan = CaptureWindowPlan(
        _id(f"{label}/capture"),
        window_ids,
        tuple(300 * index for index in range(len(values))),
    )
    windows = tuple(
        CaptureValidationWindow(
            window_id,
            _groups(label, value, pad=index % 2),
        )
        for index, (window_id, value) in enumerate(zip(window_ids, values, strict=True))
    )
    return CaptureValidationCapture(plan, windows[::-1], _frozen_text(label, axes), "C00")


def _source() -> CaptureValidationSource:
    return CaptureValidationSource(
        "val",
        _id("accepted-window-and-caption-manifest").hex(),
        (
            _capture("capture-b", (3.0,), (7,)),
            _capture("capture-a", (1.0, 2.0), (2, 4)),
        ),
    )


def _write(tmp_path: Path) -> tuple[Path, storage.CaptureStorageBuildResult]:
    root = (tmp_path / "capture-store").resolve()
    result = storage.write_capture_validation_source(root, _source())
    return root, result


def _rewrite_manifest(path: Path, mutate) -> str:
    value = json.loads(path.read_text(encoding="ascii"))
    mutate(value)
    raw = _canonical(value)
    os.chmod(path, 0o600)
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def test_roundtrip_preserves_complete_capture_source(tmp_path: Path) -> None:
    original = _source()
    root, result = _write(tmp_path)
    loaded = storage.load_capture_validation_source(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    )

    assert loaded.census_sha256 == original.census_sha256 == result.source_census_sha256
    assert loaded.manifest_sha256 == original.manifest_sha256
    assert result.capture_count == 2
    assert result.window_count == 3
    assert result.caption_count == 3
    assert sorted(len(value.windows) for value in loaded.captures) == [1, 2]
    multi_window = next(value for value in loaded.captures if len(value.windows) == 2)
    assert tuple(value.groups.padded_actor_count for value in multi_window.windows) == (
        3,
        4,
    )
    for before, after in zip(original.captures, loaded.captures, strict=True):
        assert before.plan == after.plan
        assert before.holistic_text.caption_commitments == (
            after.holistic_text.caption_commitments
        )
        assert before.holistic_text.receipt.canonical_json_bytes() == (
            after.holistic_text.receipt.canonical_json_bytes()
        )
        assert torch.equal(before.holistic_text.embeddings, after.holistic_text.embeddings)
    manifest_raw = result.manifest_path.read_bytes()
    assert str(root).encode("utf-8") not in manifest_raw
    assert b"caption-a" not in manifest_raw


def test_writer_is_new_root_only_and_count_bounds_are_complete(tmp_path: Path) -> None:
    root, _ = _write(tmp_path)
    with pytest.raises(storage.CapturePreparedStorageError):
        storage.write_capture_validation_source(root, _source())
    with pytest.raises(storage.CapturePreparedStorageResourceLimit):
        storage.write_capture_validation_source(
            (tmp_path / "bounded").resolve(),
            _source(),
            limits=storage.CaptureStorageLimits(max_windows=2),
        )
    assert not (tmp_path / "bounded").exists()


def test_manifest_digest_and_referenced_file_bytes_are_authenticated(tmp_path: Path) -> None:
    root, result = _write(tmp_path)
    with pytest.raises(storage.CapturePreparedStorageError, match="manifest SHA-256"):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256="0" * 64,
        )

    window = root / "captures" / "capture-000000" / "window-000000.npz"
    os.chmod(window, 0o600)
    raw = bytearray(window.read_bytes())
    raw[-1] ^= 1
    window.write_bytes(raw)
    with pytest.raises(storage.CapturePreparedStorageError, match="window NPZ SHA-256"):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=result.manifest_sha256,
        )


@pytest.mark.parametrize(
    ("window_count", "error_pattern"),
    [(1, "every planned window"), (2, "plan SHA-256")],
)
def test_manifest_cannot_silently_drop_an_accepted_window(
    tmp_path: Path, window_count: int, error_pattern: str,
) -> None:
    _, result = _write(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        captures = value["captures"]
        assert isinstance(captures, list)
        row = next(row for row in captures if len(row["windows"]) == window_count)
        assert isinstance(row, dict)
        windows = row["windows"]
        assert isinstance(windows, list)
        windows.pop()
        counts = value["counts"]
        assert isinstance(counts, dict)
        counts["windows"] -= 1

    rebound = _rewrite_manifest(result.manifest_path, mutate)
    with pytest.raises(storage.CapturePreparedStorageError, match=error_pattern):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=rebound,
        )


def test_text_feature_swap_fails_against_bound_receipt(tmp_path: Path) -> None:
    root, result = _write(tmp_path)
    left = root / "captures" / "capture-000000" / "text.npz"
    right = root / "captures" / "capture-000001" / "text.npz"
    left_raw = left.read_bytes()
    right_raw = right.read_bytes()
    os.chmod(left, 0o600)
    os.chmod(right, 0o600)
    left.write_bytes(right_raw)
    right.write_bytes(left_raw)

    def mutate(value: dict[str, object]) -> None:
        captures = value["captures"]
        assert isinstance(captures, list)
        first = captures[0]
        second = captures[1]
        assert isinstance(first, dict) and isinstance(second, dict)
        first_text = first["text"]
        second_text = second["text"]
        assert isinstance(first_text, dict) and isinstance(second_text, dict)
        first_text["sha256"], second_text["sha256"] = (
            hashlib.sha256(right_raw).hexdigest(),
            hashlib.sha256(left_raw).hexdigest(),
        )
        first_text["decoded_bytes"], second_text["decoded_bytes"] = (
            second_text["decoded_bytes"],
            first_text["decoded_bytes"],
        )

    rebound = _rewrite_manifest(result.manifest_path, mutate)
    with pytest.raises(storage.CapturePreparedStorageError, match="text embeddings|rehydration"):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=rebound,
        )


def test_artifact_tree_is_closed_and_text_limits_are_enforced(tmp_path: Path) -> None:
    root, result = _write(tmp_path)
    extra = root / "unreferenced.bin"
    extra.write_bytes(b"not admitted")
    with pytest.raises(storage.CapturePreparedStorageError, match="tree census"):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=result.manifest_sha256,
        )
    extra.unlink()

    with pytest.raises(storage.CapturePreparedStorageResourceLimit):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=result.manifest_sha256,
            limits=replace(storage.CaptureStorageLimits(), max_captions=2),
        )


def test_bounded_read_rejects_leaf_replacement_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = tmp_path / "leaf.bin"
    replacement = tmp_path / "replacement.bin"
    leaf.write_bytes(b"original")
    replacement.write_bytes(b"replaced")
    real_open = storage.os.open
    calls = 0

    def swapping_open(path: Path, flags: int, *args: object) -> int:
        nonlocal calls
        calls += 1
        if hasattr(os, "O_NOFOLLOW"):
            assert flags & os.O_NOFOLLOW
        os.replace(replacement, leaf)
        return real_open(path, flags, *args)

    monkeypatch.setattr(storage.os, "open", swapping_open)
    with pytest.raises(storage.CapturePreparedStorageError, match="changed before"):
        storage._read_bounded(leaf, maximum=64, label="fixture")
    assert calls == 1


def _stat_view(value: os.stat_result, **changes: int) -> SimpleNamespace:
    fields = {
        name: getattr(value, name)
        for name in (
            "st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"
        )
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_bounded_read_allows_stable_cross_api_ctime_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = tmp_path / "ctime.bin"
    leaf.write_bytes(b"unchanged")
    real_fstat = storage.os.fstat

    def distinct_ctime(descriptor: int) -> SimpleNamespace:
        observed = real_fstat(descriptor)
        return _stat_view(observed, st_ctime_ns=observed.st_ctime_ns + 1234567)

    monkeypatch.setattr(storage.os, "fstat", distinct_ctime)
    assert storage._read_bounded(leaf, maximum=64, label="fixture") == b"unchanged"


@pytest.mark.parametrize("interface", ["lstat", "fstat"])
def test_bounded_read_still_rejects_same_api_ctime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interface: str,
) -> None:
    leaf = tmp_path / "ctime.bin"
    leaf.write_bytes(b"unchanged")
    original = getattr(storage.os, interface)
    calls = 0

    def drifting_ctime(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        observed = original(*args, **kwargs)
        calls += 1
        return _stat_view(observed, st_ctime_ns=observed.st_ctime_ns + calls)

    monkeypatch.setattr(storage.os, interface, drifting_ctime)
    with pytest.raises(storage.CapturePreparedStorageError, match="changed while"):
        storage._read_bounded(leaf, maximum=64, label="fixture")
    assert calls == 2


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_size", "st_mtime_ns"])
def test_bounded_read_still_binds_cross_api_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    leaf = tmp_path / "identity.bin"
    leaf.write_bytes(b"unchanged")
    real_fstat = storage.os.fstat

    def mismatched_identity(descriptor: int) -> SimpleNamespace:
        observed = real_fstat(descriptor)
        return _stat_view(observed, **{field: getattr(observed, field) + 1})

    monkeypatch.setattr(storage.os, "fstat", mismatched_identity)
    with pytest.raises(storage.CapturePreparedStorageError, match="changed before"):
        storage._read_bounded(leaf, maximum=64, label="fixture")


@pytest.mark.parametrize("mutation", ["authority", "capture", "window", "caption"])
def test_manifest_literals_and_commitment_hex_are_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, result = _write(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        captures = value["captures"]
        assert isinstance(captures, list)
        first = captures[0]
        assert isinstance(first, dict)
        if mutation == "authority":
            value["authority"] = False
        elif mutation == "capture":
            first["capture_commitment"] = first["capture_commitment"].upper()
        elif mutation == "window":
            windows = first["windows"]
            assert isinstance(windows, list) and isinstance(windows[0], dict)
            windows[0]["window_commitment"] += " "
        else:
            text = first["text"]
            assert isinstance(text, dict)
            commitments = text["caption_commitments"]
            assert isinstance(commitments, list)
            commitments[0] = commitments[0].upper()

    rebound = _rewrite_manifest(result.manifest_path, mutate)
    with pytest.raises(storage.CapturePreparedStorageError):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=rebound,
        )


@pytest.mark.parametrize("relabel_only", [False, True])
def test_capture_rows_must_remain_in_canonical_commitment_order(
    tmp_path: Path, relabel_only: bool,
) -> None:
    _, result = _write(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        captures = value["captures"]
        assert isinstance(captures, list)
        if not relabel_only:
            captures.reverse()
            return
        reversed_ids = [row["capture_commitment"] for row in captures][::-1]
        for row, capture_id in zip(captures, reversed_ids, strict=True):
            row["capture_commitment"] = capture_id
            windows = row["windows"]
            row["plan_sha256"] = CaptureWindowPlan(
                bytes.fromhex(capture_id),
                tuple(bytes.fromhex(window["window_commitment"]) for window in windows),
                tuple(window["source_start_frame"] for window in windows),
            ).sha256

    rebound = _rewrite_manifest(result.manifest_path, mutate)
    expected_error = "commitment order" if relabel_only else "window path census"
    with pytest.raises(storage.CapturePreparedStorageError, match=expected_error):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=rebound,
        )


def test_total_budget_rejects_header_before_numpy_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _write(tmp_path)
    manifest_bytes = len(result.manifest_path.read_bytes())
    calls = 0

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("NumPy materialized an over-budget member")

    monkeypatch.setattr(storage.np, "load", forbidden_load)
    with pytest.raises(storage.CapturePreparedStorageResourceLimit):
        storage.load_capture_validation_source(
            result.manifest_path,
            expected_manifest_sha256=result.manifest_sha256,
            limits=replace(
                storage.CaptureStorageLimits(),
                max_total_materialized_bytes=manifest_bytes + 1,
            ),
        )
    assert calls == 0


def test_writer_checks_total_before_window_array_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_arrays(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("writer copied an over-budget window")

    monkeypatch.setattr(storage, "_window_arrays", forbidden_arrays)
    with pytest.raises(storage.CapturePreparedStorageResourceLimit):
        storage.write_capture_validation_source(
            (tmp_path / "writer-budget").resolve(),
            _source(),
            limits=replace(
                storage.CaptureStorageLimits(),
                max_total_materialized_bytes=1,
            ),
        )
    assert calls == 0
