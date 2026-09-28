"""Server-only numeric contract tests for prepared_data_v2.

The fixtures contain only analytic arrays and a sealed synthetic
FrozenClipTextBatch.  They do not load CLIP/SMPL-X or access a dataset.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from phaseset_core import frozen_clip_text as clip
from phaseset_core import prepared_data_v2 as prepared
from phaseset_core.contracts import group_commitment
from phaseset_core.pipeline import PreparedGroupSample, deterministic_group_yaw
from phaseset_core.preprocessing import WindowDecision


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _sample(label: str, *, actors: int) -> PreparedGroupSample:
    commitments = tuple(sorted(_digest(f"{label}/actor/{index}") for index in range(actors)))
    skeletons = np.zeros((actors, 200, 22, 3), dtype=np.float32)
    grid = np.arange(200, dtype=np.float32)
    for actor in range(actors):
        for joint in range(22):
            skeletons[actor, :, joint, 0] = grid * np.float32(0.01) + actor
            skeletons[actor, :, joint, 1] = np.float32(joint * 0.03)
            skeletons[actor, :, joint, 2] = grid * np.float32(-0.004) + actor * 2
    track = np.ones((actors, 200, 22), dtype=np.bool_)
    source = hashlib.sha256((label + "/source").encode("ascii")).hexdigest()
    window = hashlib.sha256((label + "/window").encode("ascii")).hexdigest()
    return PreparedGroupSample(
        skeletons=np.ascontiguousarray(skeletons),
        track_mask=track,
        actor_commitments=commitments,
        group_commitment=group_commitment(commitments),
        source_sha256=source,
        window_sha256=window,
        source_start_frame=0,
        augmentation_yaw=0.0,
        decision=WindowDecision(True, "ACCEPT", 300, actors, 0, 1.0, 0, False),
    )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _text_batch(label: str, count: int) -> clip.FrozenClipTextBatch:
    values = torch.arange(count * 512, dtype=torch.float32).reshape(count, 512)
    embeddings = (torch.sin(values * 0.001) + len(label)).contiguous()
    commitments = tuple(_digest(f"{label}/caption/{index}") for index in range(count))
    rows = tuple(
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
    )
    output = _tensor_bytes(embeddings)
    receipt = clip.FrozenClipTextReceipt(
        batch_size=count,
        caption_rows=rows,
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


def _block(
    label: str,
    counts: tuple[int, ...],
    ordinals: tuple[int, ...],
    *,
    actors: tuple[int, ...] | None = None,
) -> prepared.PreparedMotionTextBlock:
    actor_counts = actors or tuple(2 + index % 2 for index in range(len(counts)))
    samples = tuple(
        _sample(f"{label}/{index}", actors=actor_count)
        for index, actor_count in enumerate(actor_counts)
    )
    return prepared.PreparedMotionTextBlock(
        samples=samples,
        text_batch=_text_batch(label, sum(counts)),
        text_counts=counts,
        window_ordinals=ordinals,
    )


def _write_tree(root: Path):
    return prepared.write_prepared_training_tree_v2(
        root,
        train_blocks=(_block("train", (1, 3), (10, 11), actors=(2, 3)),),
        val_blocks=(_block("val", (2,), (20,), actors=(2,)),),
        max_total_edges=4,
        max_batch_size=2,
        max_batch_bytes=32 * 1024 * 1024,
        max_decoded_batch_bytes=32 * 1024 * 1024,
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        + b"\n"
    )


@pytest.mark.parametrize("chunk_size", (1, 2, 3, 4, 8))
def test_text_receipt_chunk_size_is_not_the_total_caption_census(chunk_size: int) -> None:
    block = _block("chunked", (1, 3), (10, 11))
    original = block.text_batch
    chunk_ranges = tuple((start, min(start + chunk_size, 4)) for start in range(0, 4, chunk_size))
    changed = clip.FrozenClipTextBatch(
        original.embeddings,
        original.caption_commitments,
        replace(original.receipt, batch_size=chunk_size, chunk_ranges=chunk_ranges),
        _seal=clip._CONSTRUCTION_SEAL,
    )
    rebuilt = replace(block, text_batch=changed)
    assert rebuilt.text_counts == (1, 3)
    assert len(rebuilt.text_batch.receipt.caption_rows) == 4
    assert torch.equal(rebuilt.text_batch.embeddings, original.embeddings)


@pytest.mark.parametrize(
    ("chunk_size", "chunks"),
    (
        (False, ((0, 4),)),
        (0, ((0, 4),)),
        (257, ((0, 4),)),
        (2, ((0, 2),)),
        (2, ((2, 4), (0, 2))),
        (2, ((0, 2), (1, 4))),
        (2, ((0, 2), (2, 5))),
        (2, ((False, 2), (2, 4))),
        (2, [(0, 2), (2, 4)]),
    ),
)
def test_text_receipt_rejects_invalid_or_incomplete_chunks(chunk_size, chunks) -> None:
    block = _block("invalid-chunks", (1, 3), (10, 11))
    original = block.text_batch
    changed = clip.FrozenClipTextBatch(
        original.embeddings,
        original.caption_commitments,
        replace(original.receipt, batch_size=chunk_size, chunk_ranges=chunks),
        _seal=clip._CONSTRUCTION_SEAL,
    )
    with pytest.raises(prepared.PreparedDataV2Error, match="chunk census"):
        replace(block, text_batch=changed)


def test_write_load_variable_q_and_exact_ten_key_npz(tmp_path: Path) -> None:
    result = _write_tree(tmp_path / "prepared")
    assert result.authority == 0
    assert (result.batch_count, result.motion_count, result.text_count) == (2, 3, 6)
    assert hashlib.sha256(result.index_path.read_bytes()).hexdigest() == result.index_sha256
    assert hashlib.sha256(result.census_path.read_bytes()).hexdigest() == result.census_sha256

    with np.load(result.root / "train" / "batch-00000000.npz", allow_pickle=False) as data:
        assert tuple(data.files) == prepared.NPZ_KEY_ORDER
        assert set(data.files) == prepared.NPZ_KEYS
        assert data["text_embeddings"].shape == (4, 512)
        families = tuple(bytes(row) for row in data["text_positive_ids"])
        assert families.count(bytes(data["motion_positive_ids"][0])) == 1
        assert families.count(bytes(data["motion_positive_ids"][1])) == 3

    train, val = prepared.load_prepared_training_sources_v2(
        result.index_path,
        expected_index_sha256=result.index_sha256,
        max_batch_bytes=32 * 1024 * 1024,
        max_decoded_batch_bytes=32 * 1024 * 1024,
    )
    assert train.split == "train"
    assert val.split == "val"
    assert sum(batch.motion_count for batch in train.iter_epoch(epoch=0, seed=1729)) == 2
    assert sum(batch.motion_count for batch in val.iter_epoch(epoch=0, seed=1729)) == 1


def test_train_yaw_is_epoch_bound_nonaccumulating_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    result = _write_tree(tmp_path / "prepared")
    train, _ = prepared.load_prepared_training_sources_v2(
        result.index_path,
        expected_index_sha256=result.index_sha256,
        max_batch_bytes=32 * 1024 * 1024,
        max_decoded_batch_bytes=32 * 1024 * 1024,
    )
    epoch0 = tuple(train.iter_epoch(epoch=0, seed=1729))[0]
    epoch1 = tuple(train.iter_epoch(epoch=1, seed=1729))[0]
    replay = tuple(train.iter_epoch(epoch=0, seed=1729))[0]
    assert np.array_equal(epoch0.groups.skeletons, replay.groups.skeletons)
    assert not np.array_equal(epoch0.groups.skeletons, epoch1.groups.skeletons)
    assert np.array_equal(epoch0.groups.actor_mask, epoch1.groups.actor_mask)
    assert np.array_equal(epoch0.groups.frame_mask, epoch1.groups.frame_mask)
    assert np.array_equal(epoch0.groups.track_mask, epoch1.groups.track_mask)
    assert epoch0.groups.actor_commitments == epoch1.groups.actor_commitments
    assert epoch0.motion_positive_ids == epoch1.motion_positive_ids
    assert epoch0.text_positive_ids == epoch1.text_positive_ids
    assert epoch0.text_commitments == epoch1.text_commitments
    assert torch.equal(epoch0.text_embeddings, epoch1.text_embeddings)
    with np.load(result.root / "train" / "batch-00000000.npz", allow_pickle=False) as data:
        source = np.asarray(data["skeletons"], dtype=np.float64)
        expected = np.empty_like(source)
        for row, ordinal in enumerate((10, 11)):
            yaw = deterministic_group_yaw(seed=1729, epoch=0, window_ordinal=ordinal)
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            expected[row, ..., 0] = cosine * source[row, ..., 0] + sine * source[row, ..., 2]
            expected[row, ..., 1] = source[row, ..., 1]
            expected[row, ..., 2] = -sine * source[row, ..., 0] + cosine * source[row, ..., 2]
        expected = np.ascontiguousarray(expected, dtype=np.float32)
        valid = (
            data["actor_mask"][:, :, None, None]
            & data["frame_mask"][:, None, :, None]
            & data["track_mask"]
        )
        expected[~np.broadcast_to(valid[..., None], expected.shape)] = np.float32(0.0)
    assert np.array_equal(epoch0.groups.skeletons, expected)
    invalid = ~epoch0.groups.actor_mask[:, :, None, None, None]
    padded = np.broadcast_to(invalid, epoch0.groups.skeletons.shape)
    assert np.all(epoch0.groups.skeletons.view(np.uint32)[padded] == 0)


def test_validation_is_exact_unaugmented_npz_content(tmp_path: Path) -> None:
    result = _write_tree(tmp_path / "prepared")
    _, val = prepared.load_prepared_training_sources_v2(
        result.index_path,
        expected_index_sha256=result.index_sha256,
        max_batch_bytes=32 * 1024 * 1024,
        max_decoded_batch_bytes=32 * 1024 * 1024,
    )
    batch = tuple(val.iter_epoch(epoch=29, seed=31415))[0]
    with np.load(result.root / "val" / "batch-00000000.npz", allow_pickle=False) as data:
        assert np.array_equal(batch.groups.skeletons, data["skeletons"])
        assert torch.equal(batch.text_embeddings, torch.from_numpy(data["text_embeddings"]))


def test_writer_is_new_root_only_and_leaves_no_overwrite_path(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    _write_tree(root)
    before = hashlib.sha256((root / "prepared-index.json").read_bytes()).hexdigest()
    with pytest.raises(prepared.PreparedDataV2Error, match="must be new"):
        _write_tree(root)
    assert hashlib.sha256((root / "prepared-index.json").read_bytes()).hexdigest() == before


def test_block_rejects_noncanonical_yaw_partition_and_receipt_drift() -> None:
    sample = _sample("bad", actors=2)
    text = _text_batch("bad", 2)
    with pytest.raises(prepared.PreparedDataV2Error, match="text_counts"):
        prepared.PreparedMotionTextBlock((sample,), text, (1,), (7,))
    with pytest.raises(prepared.PreparedDataV2Error, match=r"exact \+0 yaw"):
        prepared.PreparedMotionTextBlock(
            (replace(sample, augmentation_yaw=0.25),), text, (2,), (7,)
        )
    bad_receipt = replace(text.receipt, output_sha256="0" * 64)
    forged = clip.FrozenClipTextBatch(
        text.embeddings,
        text.caption_commitments,
        bad_receipt,
        _seal=clip._CONSTRUCTION_SEAL,
    )
    with pytest.raises(prepared.PreparedDataV2Error, match="receipt"):
        prepared.PreparedMotionTextBlock((sample,), forged, (2,), (7,))


def test_writer_rejects_train_val_participant_or_ordinal_leakage(tmp_path: Path) -> None:
    train = _block("train", (1,), (10,))
    same_actor_sample = replace(
        _sample("val", actors=2),
        actor_commitments=train.samples[0].actor_commitments,
        group_commitment=train.samples[0].group_commitment,
    )
    val = prepared.PreparedMotionTextBlock(
        (same_actor_sample,), _text_batch("val", 1), (1,), (20,)
    )
    with pytest.raises(prepared.PreparedDataV2Error, match="strictly isolated"):
        prepared.write_prepared_training_tree_v2(
            tmp_path / "actor-leak",
            train_blocks=(train,),
            val_blocks=(val,),
            max_total_edges=4,
        )
    with pytest.raises(prepared.PreparedDataV2Error, match="strictly isolated"):
        prepared.write_prepared_training_tree_v2(
            tmp_path / "ordinal-leak",
            train_blocks=(train,),
            val_blocks=(_block("fresh-val", (1,), (10,)),),
            max_total_edges=4,
        )


def test_loader_rejects_test_or_wrong_augmentation_manifest(tmp_path: Path) -> None:
    result = _write_tree(tmp_path / "prepared")
    manifest = result.root / "val.json"
    os.chmod(manifest, 0o600)
    value = json.loads(manifest.read_text(encoding="ascii"))
    value["split"] = "test"
    value["augmentation"] = "none"
    manifest.write_bytes(_canonical_json(value))
    os.chmod(manifest, 0o400)
    with pytest.raises(prepared.PreparedDataV2Error, match="train or val"):
        prepared.PreparedTrainingDataSourceV2(
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            max_batch_bytes=32 * 1024 * 1024,
            max_decoded_batch_bytes=32 * 1024 * 1024,
        )

    os.chmod(manifest, 0o600)
    value["split"] = "val"
    value["augmentation"] = "group_yaw_v2"
    manifest.write_bytes(_canonical_json(value))
    os.chmod(manifest, 0o400)
    with pytest.raises(prepared.PreparedDataV2Error, match="policy changed"):
        prepared.PreparedTrainingDataSourceV2(
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            max_batch_bytes=32 * 1024 * 1024,
            max_decoded_batch_bytes=32 * 1024 * 1024,
        )


def test_index_rejects_manifest_swapped_at_constructor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _write_tree(tmp_path / "prepared")
    target = (result.root / "train.json").resolve()
    changed = json.loads(target.read_text(encoding="ascii"))
    changed["batches"][0]["sha256"] = "0" * 64
    changed_raw = _canonical_json(changed)
    real_constructor = prepared.PreparedTrainingDataSourceV2
    swapped = False

    def swapping_constructor(
        manifest: str | Path,
        *,
        expected_manifest_sha256: str,
        max_batch_bytes: int,
        max_decoded_batch_bytes: int,
    ) -> prepared.PreparedTrainingDataSourceV2:
        nonlocal swapped
        if Path(manifest).resolve() == target and not swapped:
            os.chmod(target, 0o600)
            target.write_bytes(changed_raw)
            os.chmod(target, 0o400)
            swapped = True
        return real_constructor(
            manifest,
            expected_manifest_sha256=expected_manifest_sha256,
            max_batch_bytes=max_batch_bytes,
            max_decoded_batch_bytes=max_decoded_batch_bytes,
        )

    monkeypatch.setattr(prepared, "PreparedTrainingDataSourceV2", swapping_constructor)
    with pytest.raises(prepared.PreparedDataV2Error, match="manifest digest mismatch"):
        prepared.load_prepared_training_sources_v2(
            result.index_path,
            expected_index_sha256=result.index_sha256,
            max_batch_bytes=32 * 1024 * 1024,
            max_decoded_batch_bytes=32 * 1024 * 1024,
        )
    assert swapped


def test_loader_rejects_compressed_or_oversized_npz_before_numpy_load(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in prepared.NPZ_KEY_ORDER:
            archive.writestr(name + ".npy", b"not-an-array")
    raw = path.read_bytes()
    with pytest.raises(prepared.PreparedDataV2Error, match="ZIP_STORED"):
        prepared._zip_member_census(raw, max_decoded_bytes=1024 * 1024)
    oversized = tmp_path / "oversized.npz"
    with zipfile.ZipFile(oversized, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in prepared.NPZ_KEY_ORDER:
            archive.writestr(name + ".npy", b"not-an-array")
    with pytest.raises(prepared.PreparedDataV2Error, match="decoded size"):
        prepared._zip_member_census(
            oversized.read_bytes(),
            max_decoded_bytes=1,
        )
