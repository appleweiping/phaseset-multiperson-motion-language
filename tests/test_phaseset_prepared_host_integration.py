"""Private host dispatch tests for canonical prepared-data v1 and v2 sources."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from phaseset_core import frozen_clip_text as clip
from phaseset_core import host
from phaseset_core import prepared_data_v2 as prepared
from phaseset_core.contracts import group_commitment
from phaseset_core.pipeline import PreparedGroupSample
from phaseset_core.preprocessing import WindowDecision


MAX_BATCH_BYTES = 32 * 1024 * 1024


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _canonical_json(value: object) -> bytes:
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def _write_v1_batch(path: Path, label: str) -> None:
    actor_ids = (_digest(f"{label}/actor/0"), _digest(f"{label}/actor/1"))
    family = _digest(f"{label}/family")
    np.savez(
        path,
        skeletons=np.zeros((1, 2, 200, 22, 3), dtype=np.float32),
        actor_mask=np.ones((1, 2), dtype=np.bool_),
        frame_mask=np.ones((1, 200), dtype=np.bool_),
        track_mask=np.ones((1, 2, 200, 22), dtype=np.bool_),
        actor_commitments=np.asarray([[list(row) for row in actor_ids]], dtype=np.uint8),
        group_commitments=np.asarray(
            [list(group_commitment(actor_ids))], dtype=np.uint8
        ),
        text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        motion_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_commitments=np.asarray([list(_digest(f"{label}/text"))], dtype=np.uint8),
    )


def _write_v1_tree(root: Path) -> tuple[Path, Path, Path]:
    manifests: dict[str, Path] = {}
    for split, labels in (("train", ("a", "b")), ("val", ("a",))):
        rows = []
        for label in labels:
            batch = root / f"{split}-{label}.npz"
            _write_v1_batch(batch, f"{split}/{label}")
            rows.append({"path": batch.name, "sha256": _sha(batch)})
        manifest = root / f"{split}.json"
        _write_json(
            manifest,
            {
                "batches": rows,
                "schema": host.PREPARED_SPLIT_SCHEMA,
                "split": split,
            },
        )
        manifests[split] = manifest
    index = root / "prepared-index.json"
    _write_json(
        index,
        {
            "schema": host.PREPARED_INDEX_SCHEMA,
            "train": {
                "path": manifests["train"].name,
                "sha256": _sha(manifests["train"]),
            },
            "val": {
                "path": manifests["val"].name,
                "sha256": _sha(manifests["val"]),
            },
        },
    )
    return manifests["train"], manifests["val"], index


def _sample(
    label: str,
    *,
    capture_label: str | None = None,
    source_start_frame: int = 0,
) -> PreparedGroupSample:
    capture = capture_label or label
    actors = tuple(
        sorted((_digest(f"{capture}/actor/0"), _digest(f"{capture}/actor/1")))
    )
    skeletons = np.zeros((2, 200, 22, 3), dtype=np.float32)
    skeletons[:, :, :, 0] = np.arange(200, dtype=np.float32)[None, :, None]
    skeletons[1, :, :, 2] = np.float32(1.0)
    return PreparedGroupSample(
        skeletons=np.ascontiguousarray(skeletons),
        track_mask=np.ones((2, 200, 22), dtype=np.bool_),
        actor_commitments=actors,
        group_commitment=group_commitment(actors),
        source_sha256=hashlib.sha256(f"{capture}/source".encode("ascii")).hexdigest(),
        window_sha256=hashlib.sha256(f"{label}/window".encode("ascii")).hexdigest(),
        source_start_frame=source_start_frame,
        augmentation_yaw=0.0,
        decision=WindowDecision(True, "ACCEPT", 300, 2, 0, 1.0, 0, False),
    )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _text_batch(label: str, count: int = 1) -> clip.FrozenClipTextBatch:
    embeddings = (
        torch.arange(count * 512, dtype=torch.float32).reshape(count, 512).contiguous()
    )
    commitments = tuple(_digest(f"{label}/caption/{index}") for index in range(count))
    output = _tensor_bytes(embeddings)
    receipt = clip.FrozenClipTextReceipt(
        batch_size=count,
        caption_rows=tuple(
            (
                index,
                commitment.hex(),
                hashlib.sha256(f"{label}/text/{index}".encode("ascii")).hexdigest(),
                3,
                3,
                False,
                hashlib.sha256(f"{label}/tokens/{index}".encode("ascii")).hexdigest(),
                hashlib.sha256(f"{label}/mask/{index}".encode("ascii")).hexdigest(),
                hashlib.sha256(_tensor_bytes(embeddings[index : index + 1])).hexdigest(),
            )
            for index, commitment in enumerate(commitments)
        ),
        chunk_ranges=((0, count),),
        frozen_embedding_cache_key_sha256="1" * 64,
        live_model_manifest_sha256="2" * 64,
        output_bytes=len(output),
        output_sha256=hashlib.sha256(output).hexdigest(),
        output_shape=tuple(embeddings.shape),
        output_stride=tuple(embeddings.stride()),
        runtime_identity=(("fixture", "server-only"),),
        runtime_manifest_sha256="3" * 64,
        snapshot_files=(),
        snapshot_manifest_sha256="4" * 64,
        source_files=(),
        source_manifest_sha256="5" * 64,
    )
    return clip.FrozenClipTextBatch(
        embeddings,
        commitments,
        receipt,
        _seal=clip._CONSTRUCTION_SEAL,
    )


def _write_v2_tree(root: Path):
    train = prepared.PreparedMotionTextBlock(
        samples=(
            _sample("train/window-0", capture_label="train/capture", source_start_frame=0),
            _sample(
                "train/window-1",
                capture_label="train/capture",
                source_start_frame=300,
            ),
        ),
        text_batch=_text_batch("train", 2),
        text_counts=(1, 1),
        window_ordinals=(10, 11),
    )
    val = prepared.PreparedMotionTextBlock(
        samples=(_sample("val"),),
        text_batch=_text_batch("val"),
        text_counts=(1,),
        window_ordinals=(20,),
    )
    return prepared.write_prepared_training_tree_v2(
        root,
        train_blocks=(train,),
        val_blocks=(val,),
        max_total_edges=4,
        max_batch_size=2,
        max_batch_bytes=MAX_BATCH_BYTES,
        max_decoded_batch_bytes=MAX_BATCH_BYTES,
    )


def _host_config(root: Path, index: Path) -> host.HostConfig:
    common = root / "receipt-artifact.bin"
    common.write_bytes(b"fixture-only receipt artifact\n")
    artifacts: dict[str, str] = {}
    digests: dict[str, str] = {}
    for field in host.RECEIPT_DIGEST_FIELDS:
        artifact = index if field == "prepared_data_manifest_sha256" else common
        artifacts[field] = artifact.relative_to(root).as_posix()
        digests[field] = _sha(artifact)
    receipt = root / "receipt-record.json"
    _write_json(
        receipt,
        {
            "admitted_at_utc": "2026-09-08T12:00:00Z",
            "authority": 1,
            **digests,
        },
    )
    config = root / "host.json"
    _write_json(
        config,
        {
            "prepared_index": index.relative_to(root).as_posix(),
            "receipts": {
                "artifacts": artifacts,
                "record": receipt.name,
            },
            "runtime": {
                "bf16_runtime_qualified": False,
                "checkpoint_every_updates": 1,
                "corrective_change_sha256": hashlib.sha256(b"none").hexdigest(),
                "device": "cpu",
                "edge_budget": 32768,
                "max_batch_bytes": MAX_BATCH_BYTES,
                "request_bf16": False,
                "resume_retry_class": "INFRA_TRANSIENT",
            },
            "schema": host.HOST_CONFIG_SCHEMA,
            "source_tree_sha256": hashlib.sha256(b"source-tree").hexdigest(),
        },
    )
    return host.HostConfig.load(config)


def _replace_v2_positive_family(result: prepared.PreparedDataBuildResult) -> None:
    batch = result.root / "train" / "batch-00000000.npz"
    with np.load(batch, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["text_positive_ids"][0] = np.frombuffer(
        _digest("foreign-positive-family"), dtype=np.uint8
    )
    os.chmod(batch, 0o600)
    np.savez(batch, **arrays)

    manifest = result.root / "train.json"
    os.chmod(manifest, 0o600)
    manifest_value = json.loads(manifest.read_text(encoding="ascii"))
    manifest_value["batches"][0]["sha256"] = _sha(batch)
    manifest.write_bytes(_canonical_json(manifest_value))

    os.chmod(result.index_path, 0o600)
    index_value = json.loads(result.index_path.read_text(encoding="ascii"))
    index_value["train"]["sha256"] = _sha(manifest)
    result.index_path.write_bytes(_canonical_json(index_value))


def test_host_preserves_v1_dispatch_and_binds_consumed_manifests(tmp_path: Path) -> None:
    train_manifest, val_manifest, index = _write_v1_tree(tmp_path)
    train, val = _host_config(tmp_path, index).load_sources()
    assert type(train) is host.PrivatePreparedDataSource
    assert type(val) is host.PrivatePreparedDataSource
    assert train.manifest_sha256 == _sha(train_manifest)
    assert val.manifest_sha256 == _sha(val_manifest)
    assert len(tuple(train.iter_epoch(epoch=0, seed=1729))) == 2
    assert len(tuple(val.iter_epoch(epoch=0, seed=1729))) == 1


def test_host_v1_rejects_manifest_swap_at_constructor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_manifest, _, index = _write_v1_tree(tmp_path)
    changed = json.loads(train_manifest.read_text(encoding="ascii"))
    changed["batches"] = list(reversed(changed["batches"]))
    changed_raw = _canonical_json(changed)
    real_constructor = host.PrivatePreparedDataSource
    swapped = False

    def swapping_constructor(
        manifest: str | Path,
        *,
        max_batch_bytes: int,
    ) -> host.PrivatePreparedDataSource:
        nonlocal swapped
        if Path(manifest).resolve() == train_manifest.resolve() and not swapped:
            train_manifest.write_bytes(changed_raw)
            swapped = True
        return real_constructor(manifest, max_batch_bytes=max_batch_bytes)

    monkeypatch.setattr(host, "PrivatePreparedDataSource", swapping_constructor)
    with pytest.raises(host.HostConfigurationError, match="manifest digest mismatch"):
        _host_config(tmp_path, index).load_sources()
    assert swapped


def test_host_dispatches_v2_and_returns_training_sources(tmp_path: Path) -> None:
    result = _write_v2_tree(tmp_path / "prepared")
    train, val = _host_config(tmp_path, result.index_path).load_sources()
    assert type(train) is prepared.PreparedTrainingDataSourceV2
    assert type(val) is prepared.PreparedTrainingDataSourceV2
    assert train.split == "train"
    assert val.split == "val"
    train_batches = tuple(train.iter_epoch(epoch=2, seed=1729))
    assert len(train_batches) == 1
    assert train_batches[0].motion_count == 2
    assert train_batches[0].groups.group_commitments[0] == (
        train_batches[0].groups.group_commitments[1]
    )
    assert train_batches[0].motion_positive_ids[0] != (
        train_batches[0].motion_positive_ids[1]
    )
    assert len(tuple(val.iter_epoch(epoch=2, seed=1729))) == 1


def test_host_v2_translates_invalid_positive_family_to_hold_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _write_v2_tree(tmp_path / "prepared")
    _replace_v2_positive_family(result)
    config = _host_config(tmp_path, result.index_path)
    with pytest.raises(host.HostConfigurationError, match="prepared v2 source") as error:
        config.load_sources()
    assert isinstance(error.value.__cause__, host.TrainingRuntimeError)

    def forbidden_backend(*args: object, **kwargs: object) -> object:
        raise AssertionError("runtime backend was constructed for invalid prepared data")

    monkeypatch.setattr(host, "PhaseSetHostBackend", forbidden_backend)
    factory = host.RuntimeAdapterFactory(config.path)
    with pytest.raises(host.ExecutionHold) as held:
        factory(host.cli.CLICommandRequest(command="preflight"))
    assert held.value.hold_codes == (
        "HOLD_PREPARED_DATA_MANIFEST_ABSENT",
        "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
    )


def test_host_rejects_unknown_index_schema(tmp_path: Path) -> None:
    result = _write_v2_tree(tmp_path / "prepared")
    os.chmod(result.index_path, 0o600)
    value = json.loads(result.index_path.read_text(encoding="ascii"))
    value["schema"] = "phaseset-prepared-index-unknown"
    result.index_path.write_bytes(_canonical_json(value))
    with pytest.raises(host.HostConfigurationError, match="schema is not registered"):
        _host_config(tmp_path, result.index_path).load_sources()


def test_host_rejects_test_split_in_authenticated_v2_tree(tmp_path: Path) -> None:
    result = _write_v2_tree(tmp_path / "prepared")
    val_manifest = result.root / "val.json"
    os.chmod(val_manifest, 0o600)
    value = json.loads(val_manifest.read_text(encoding="ascii"))
    value["split"] = "test"
    val_manifest.write_bytes(_canonical_json(value))
    os.chmod(result.index_path, 0o600)
    index_value = json.loads(result.index_path.read_text(encoding="ascii"))
    index_value["val"]["sha256"] = _sha(val_manifest)
    result.index_path.write_bytes(_canonical_json(index_value))
    with pytest.raises(host.HostConfigurationError, match="prepared v2 source") as error:
        _host_config(tmp_path, result.index_path).load_sources()
    assert isinstance(error.value.__cause__, prepared.PreparedDataV2Error)
    assert "train or val" in str(error.value.__cause__)


def test_host_v2_rejects_index_swap_after_receipt_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _write_v2_tree(tmp_path / "prepared")
    original_loader = host.load_prepared_training_sources_v2
    changed = json.loads(result.index_path.read_text(encoding="ascii"))
    changed["train"]["sha256"] = "0" * 64
    changed_raw = _canonical_json(changed)
    swapped = False

    def swapping_loader(
        index: str | Path,
        *,
        expected_index_sha256: str,
        max_batch_bytes: int,
    ) -> tuple[
        prepared.PreparedTrainingDataSourceV2,
        prepared.PreparedTrainingDataSourceV2,
    ]:
        nonlocal swapped
        if not swapped:
            os.chmod(result.index_path, 0o600)
            result.index_path.write_bytes(changed_raw)
            swapped = True
        return original_loader(
            index,
            expected_index_sha256=expected_index_sha256,
            max_batch_bytes=max_batch_bytes,
        )

    monkeypatch.setattr(host, "load_prepared_training_sources_v2", swapping_loader)
    with pytest.raises(host.HostConfigurationError, match="prepared v2 source") as error:
        _host_config(tmp_path, result.index_path).load_sources()
    assert isinstance(error.value.__cause__, prepared.PreparedDataV2Error)
    assert "expected digest" in str(error.value.__cause__)
    assert swapped
