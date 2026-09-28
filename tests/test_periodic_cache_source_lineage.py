"""Static-candidate tests for prepared-data descriptor source lineage.

The fixtures are analytic arrays and sealed synthetic text embeddings.  They
do not execute CLIP, SMPL-X, a training model, or a descriptor encoder.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from phaseset_core import frozen_clip_text as clip
from phaseset_core import prepared_data_v2 as prepared
from phaseset_core import training
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.periodic_descriptor_cache_v2 import (
    DescriptorWindowContext,
    prepared_group_batch_sha256,
)
from phaseset_core.pipeline import PreparedGroupSample, deterministic_group_yaw
from phaseset_core.preprocessing import WindowDecision


_MAX_BYTES = 32 * 1024 * 1024


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
    return PreparedGroupSample(
        skeletons=np.ascontiguousarray(skeletons),
        track_mask=track,
        actor_commitments=commitments,
        group_commitment=group_commitment(commitments),
        source_sha256=hashlib.sha256((label + "/source").encode("ascii")).hexdigest(),
        window_sha256=hashlib.sha256((label + "/window").encode("ascii")).hexdigest(),
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
    actors: tuple[int, ...],
) -> prepared.PreparedMotionTextBlock:
    samples = tuple(
        _sample(f"{label}/{index}", actors=actor_count) for index, actor_count in enumerate(actors)
    )
    return prepared.PreparedMotionTextBlock(
        samples=samples,
        text_batch=_text_batch(label, sum(counts)),
        text_counts=counts,
        window_ordinals=ordinals,
    )


def _write_tree(root: Path) -> prepared.PreparedDataBuildResult:
    return prepared.write_prepared_training_tree_v2(
        root,
        train_blocks=(_block("train", (1, 3), (10, 11), actors=(2, 3)),),
        val_blocks=(_block("val", (2,), (20,), actors=(2,)),),
        max_total_edges=4,
        max_batch_size=2,
        max_batch_bytes=_MAX_BYTES,
        max_decoded_batch_bytes=_MAX_BYTES,
    )


def _load_sources(root: Path):
    result = _write_tree(root)
    sources = prepared.load_prepared_training_sources_v2(
        result.index_path,
        expected_index_sha256=result.index_sha256,
        max_batch_bytes=_MAX_BYTES,
        max_decoded_batch_bytes=_MAX_BYTES,
    )
    return result, *sources


def _expected_rotation(source: np.ndarray, yaws: tuple[float, ...], arrays) -> np.ndarray:
    expected = np.empty(source.shape, dtype=np.float64)
    for row, yaw in enumerate(yaws):
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        expected[row, ..., 0] = cosine * source[row, ..., 0] + sine * source[row, ..., 2]
        expected[row, ..., 1] = source[row, ..., 1]
        expected[row, ..., 2] = -sine * source[row, ..., 0] + cosine * source[row, ..., 2]
    result = np.ascontiguousarray(expected, dtype=np.float32)
    valid = (
        arrays["actor_mask"][:, :, None, None]
        & arrays["frame_mask"][:, None, :, None]
        & arrays["track_mask"]
    )
    result[~np.broadcast_to(valid[..., None], result.shape)] = np.float32(0.0)
    return result


def test_all_formal_train_epochs_bind_actual_source_and_returned_yaw(tmp_path: Path) -> None:
    result, train_source, _ = _load_sources(tmp_path / "prepared")
    manifest = json.loads((result.root / "train.json").read_text(encoding="ascii"))
    assert len(manifest["batches"]) == 1
    row = manifest["batches"][0]
    source_path = result.root / row["path"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == row["sha256"]
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    source = np.asarray(arrays["skeletons"], dtype=np.float64)

    for seed in training.OFFICIAL_SEEDS:
        for epoch in range(training.RESIDUAL_EPOCHS):
            (batch,) = tuple(train_source.iter_epoch(epoch=epoch, seed=seed))
            contexts = batch.descriptor_contexts
            assert contexts is not None
            assert len(contexts) == batch.motion_count == 2
            assert tuple(context.window_ordinal for context in contexts) == (10, 11)
            yaws = tuple(
                deterministic_group_yaw(
                    seed=seed,
                    epoch=epoch,
                    window_ordinal=context.window_ordinal,
                )
                for context in contexts
            )
            for index, (context, yaw) in enumerate(zip(contexts, yaws, strict=True)):
                assert type(context) is DescriptorWindowContext
                assert context.prepared_manifest_sha256 == train_source.manifest_sha256
                assert context.source_batch_sha256 == row["sha256"]
                assert context.window_sha256 == batch.motion_positive_ids[index].hex()
                assert (context.split, context.seed, context.epoch) == (
                    "train",
                    seed,
                    epoch,
                )
                assert context.augmentation_yaw_be_hex == struct.pack(">d", yaw).hex()
            assert np.array_equal(
                batch.groups.skeletons,
                _expected_rotation(source, yaws, arrays),
            )


def test_validation_is_unaugmented_with_canonical_epoch_zero_context(tmp_path: Path) -> None:
    result, _, val_source = _load_sources(tmp_path / "prepared")
    manifest = json.loads((result.root / "val.json").read_text(encoding="ascii"))
    row = manifest["batches"][0]
    first = tuple(val_source.iter_epoch(epoch=0, seed=1729))[0]
    historical_call = tuple(val_source.iter_epoch(epoch=29, seed=31415))[0]
    with np.load(result.root / row["path"], allow_pickle=False) as archive:
        assert np.array_equal(first.groups.skeletons, archive["skeletons"])
        assert np.array_equal(historical_call.groups.skeletons, archive["skeletons"])
    for batch, seed in ((first, 1729), (historical_call, 31415)):
        assert batch.descriptor_contexts is not None
        (context,) = batch.descriptor_contexts
        assert context.prepared_manifest_sha256 == val_source.manifest_sha256
        assert context.source_batch_sha256 == row["sha256"]
        assert context.window_sha256 == batch.motion_positive_ids[0].hex()
        assert (context.window_ordinal, context.split, context.seed, context.epoch) == (
            20,
            "val",
            seed,
            0,
        )
        assert context.augmentation_yaw_be_hex == "0000000000000000"


def test_batch_rejects_mixed_or_inconsistent_descriptor_contexts(tmp_path: Path) -> None:
    _, train_source, _ = _load_sources(tmp_path / "prepared")
    batch = tuple(train_source.iter_epoch(epoch=0, seed=1729))[0]
    assert batch.descriptor_contexts is not None
    first, second = batch.descriptor_contexts

    with pytest.raises(TypeError, match="exact tuple"):
        replace(batch, descriptor_contexts=list(batch.descriptor_contexts))
    with pytest.raises(TypeError, match="exact DescriptorWindowContext"):
        replace(batch, descriptor_contexts=(first, object()))
    with pytest.raises(training.TrainingRuntimeError, match="window identity"):
        replace(
            batch,
            descriptor_contexts=(first, replace(second, window_sha256="f" * 64)),
        )
    with pytest.raises(training.TrainingRuntimeError, match="source, seed, or epoch"):
        replace(
            batch,
            descriptor_contexts=(first, replace(second, source_batch_sha256="e" * 64)),
        )
    duplicate_ordinal = DescriptorWindowContext.from_yaw(
        prepared_manifest_sha256=second.prepared_manifest_sha256,
        source_batch_sha256=second.source_batch_sha256,
        window_sha256=second.window_sha256,
        window_ordinal=first.window_ordinal,
        split="train",
        seed=second.seed,
        epoch=second.epoch,
        augmentation_yaw=deterministic_group_yaw(
            seed=second.seed,
            epoch=second.epoch,
            window_ordinal=first.window_ordinal,
        ),
    )
    with pytest.raises(training.TrainingRuntimeError, match="repeat.*ordinal"):
        replace(batch, descriptor_contexts=(first, duplicate_ordinal))
    val_context = DescriptorWindowContext.from_yaw(
        prepared_manifest_sha256=first.prepared_manifest_sha256,
        source_batch_sha256=first.source_batch_sha256,
        window_sha256=first.window_sha256,
        window_ordinal=first.window_ordinal,
        split="val",
        seed=first.seed,
        epoch=0,
        augmentation_yaw=0.0,
    )
    with pytest.raises(training.TrainingRuntimeError, match="split differs"):
        replace(batch, descriptor_contexts=(val_context, second))


def _changed_numeric(groups: PreparedGroupBatch) -> PreparedGroupBatch:
    skeletons = np.array(groups.skeletons, copy=True, order="C")
    skeletons[0, 0, 0, 0, 0] += np.float32(0.125)
    return PreparedGroupBatch(
        skeletons=skeletons,
        actor_mask=np.array(groups.actor_mask, copy=True, order="C"),
        frame_mask=np.array(groups.frame_mask, copy=True, order="C"),
        track_mask=np.array(groups.track_mask, copy=True, order="C"),
        actor_commitments=groups.actor_commitments,
        group_commitments=groups.group_commitments,
    )


def _extra_padding(groups: PreparedGroupBatch) -> PreparedGroupBatch:
    batch_size, padded_actors, time, joints, coordinates = groups.skeletons.shape
    skeletons = np.zeros(
        (batch_size, padded_actors + 1, time, joints, coordinates),
        dtype=np.float32,
    )
    skeletons[:, :padded_actors] = groups.skeletons
    actor_mask = np.zeros((batch_size, padded_actors + 1), dtype=np.bool_)
    actor_mask[:, :padded_actors] = groups.actor_mask
    track_mask = np.zeros(
        (batch_size, padded_actors + 1, time, joints),
        dtype=np.bool_,
    )
    track_mask[:, :padded_actors] = groups.track_mask
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(actor_mask),
        frame_mask=np.array(groups.frame_mask, copy=True, order="C"),
        track_mask=np.ascontiguousarray(track_mask),
        actor_commitments=tuple(row + (None,) for row in groups.actor_commitments),
        group_commitments=groups.group_commitments,
    )


def test_numeric_and_padding_changes_preserve_lineage_but_change_complete_batch_key(
    tmp_path: Path,
) -> None:
    _, train_source, _ = _load_sources(tmp_path / "prepared")
    batch = tuple(train_source.iter_epoch(epoch=3, seed=2718))[0]
    original_digest = prepared_group_batch_sha256(batch.groups)
    numeric = replace(batch, groups=_changed_numeric(batch.groups))
    padded = replace(batch, groups=_extra_padding(batch.groups))
    assert numeric.descriptor_contexts == batch.descriptor_contexts
    assert padded.descriptor_contexts == batch.descriptor_contexts
    assert prepared_group_batch_sha256(numeric.groups) != original_digest
    assert prepared_group_batch_sha256(padded.groups) != original_digest


def test_legacy_batch_defaults_to_none_and_contextualized_batch_cannot_rotate_again(
    tmp_path: Path,
) -> None:
    _, train_source, _ = _load_sources(tmp_path / "prepared")
    batch = tuple(train_source.iter_epoch(epoch=2, seed=31415))[0]
    legacy = training.RetrievalTrainingBatch(
        batch.groups,
        batch.text_embeddings,
        batch.motion_positive_ids,
        batch.text_positive_ids,
        batch.text_commitments,
        batch.split,
    )
    assert legacy.descriptor_contexts is None
    legacy_rotated = prepared._rotate_training_batch(
        legacy,
        seed=31415,
        epoch=3,
        ordinals=(10, 11),
    )
    assert legacy_rotated.descriptor_contexts is None

    for seed, epoch in ((31415, 2), (1729, 19)):
        with pytest.raises(prepared.PreparedDataV2Error, match="cannot be rotated again"):
            prepared._rotate_training_batch(
                batch,
                seed=seed,
                epoch=epoch,
                ordinals=(10, 11),
            )
