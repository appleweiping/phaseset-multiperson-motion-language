"""Cached row/host integrity on analytic fixtures, not native learning."""

import copy
from dataclasses import replace
import hashlib

import pytest
import torch

from phaseset_core import frozen_clip_text as clip
from phaseset_core.capture_validation import CaptureValidationError, _validate_text_batch
from phaseset_core.continuous_parent_host import ContinuousParentTrainingHost
from phaseset_core.parent_clip_rows import ParentHumanClipRows
from phaseset_core.parent_disk_source import DevelopmentParentDiskSource
from phaseset_core.parent_retrieval_task import (
    ParentCaptionRecord,
    ParentRetrievalBatch,
    ParentRetrievalTask,
)
from phaseset_core.training import _capture_rng, _load_torch_checkpoint, _stable_hash
from test_phaseset_frozen_clip_cache_rehydrate import _fixture


def _task():
    parents = tuple(
        ParentCaptionRecord(
            hashlib.sha256(f"source/{index}".encode()).hexdigest(),
            hashlib.sha256(f"family/{index}".encode()).hexdigest(),
            "C01" if index < 2 else "C03",
            "train" if index < 2 else "validation",
            ("same exact sentence", "additional human row")
            if index == 0
            else ("same exact sentence",),
        )
        for index in range(3)
    )
    return ParentRetrievalTask(
        parents, expected_family_keys=tuple(p.annotation_family_sha256 for p in parents)
    )


def _original(task):
    labels = task.batch(tuple(parent.annotation_family_sha256 for parent in task.parents))
    embeddings, receipt = _fixture(len(labels.captions), batch_size=2)
    rows = tuple(
        (index, commitment.hex(), hashlib.sha256(caption.encode()).hexdigest(), *old[3:])
        for index, (commitment, caption, old) in enumerate(
            zip(labels.caption_commitments, labels.captions, receipt.caption_rows, strict=True)
        )
    )
    receipt = replace(receipt, caption_rows=rows)
    receipt = replace(
        receipt,
        frozen_embedding_cache_key_sha256=clip._rehydration_cache_key(
            batch_size=receipt.batch_size,
            caption_rows=rows,
            chunk_ranges=receipt.chunk_ranges,
            runtime_manifest_sha256=receipt.runtime_manifest_sha256,
            snapshot_manifest_sha256=receipt.snapshot_manifest_sha256,
            source_manifest_sha256=receipt.source_manifest_sha256,
        ),
    )
    return clip.rehydrate_frozen_clip_text_batch(
        embeddings,
        receipt_json_bytes=receipt.canonical_json_bytes(),
        expected_receipt_sha256=receipt.sha256,
    )


def test_selection_is_exact_owned_and_honest_without_encoder_runtime_or_rng(monkeypatch):
    original = _original(_task())
    before = _stable_hash(_capture_rng())

    def forbidden(*args, **kwargs):
        raise AssertionError("cached selection attempted a model/runtime probe")

    monkeypatch.setattr(clip, "_real_backend", forbidden)
    monkeypatch.setattr(clip, "_load_components", forbidden)
    selected = clip.select_frozen_clip_text_rows(original, (3, 1, 0))
    rows, commitments, digest = _validate_text_batch(selected, allow_row_selection=True)
    assert torch.equal(rows, original.embeddings[[3, 1, 0]])
    assert commitments == tuple(original.caption_commitments[i] for i in (3, 1, 0))
    assert selected.receipt.origin_receipt is original.receipt
    assert selected.receipt.selected_indices == (3, 1, 0)
    public = selected.receipt.to_public_dict()
    assert public["origin_receipt_sha256"] == original.receipt.sha256
    assert public["status"] == "ROW_SELECTION_AUTHORITY0" and public["authority"] == 0
    assert public["training_authorized"] is False and public["production"] is False
    assert digest == selected.receipt.sha256 and digest != original.receipt.sha256
    assert b"same exact sentence" not in selected.receipt.canonical_json_bytes()
    leaked = selected.embeddings
    leaked.zero_()
    assert not torch.equal(leaked, selected.embeddings)
    assert _stable_hash(_capture_rng()) == before
    with pytest.raises(CaptureValidationError, match="not admitted"):
        _validate_text_batch(selected)
    with pytest.raises(clip.FrozenClipTextAdapterError):
        clip.rehydrate_frozen_clip_text_batch(
            selected.embeddings,
            receipt_json_bytes=selected.receipt.canonical_json_bytes(),
            expected_receipt_sha256=selected.receipt.sha256,
        )


@pytest.mark.parametrize("indices", [(), (True,), (-1,), (4,), (0, 0), [0]])
def test_row_selection_rejects_ambiguous_or_duplicate_lineage(indices):
    with pytest.raises(ValueError, match="indices"):
        clip.select_frozen_clip_text_rows(_original(_task()), indices)


def test_selection_does_not_accept_nested_gathers_or_changed_ancestry():
    selected = clip.select_frozen_clip_text_rows(_original(_task()), (0, 2))
    with pytest.raises(ValueError, match="original"):
        clip.select_frozen_clip_text_rows(selected, (0,))
    object.__setattr__(
        selected,
        "_FrozenClipTextBatch__receipt",
        replace(selected.receipt, selected_indices=(2, 0)),
    )
    with pytest.raises(CaptureValidationError, match="ancestry"):
        _validate_text_batch(selected, allow_row_selection=True)


def test_every_human_occurrence_and_reordered_parent_batch_are_preserved():
    task = _task()
    original = _original(task)
    rows = ParentHumanClipRows(task, original)
    labels = task.batch(tuple(parent.annotation_family_sha256 for parent in reversed(task.parents)))
    selected, cf = rows.text(labels, training=True)
    assert selected.caption_commitments == labels.caption_commitments
    assert len(selected.embeddings) == 4  # Equal text is not deduplicated.
    assert not cf.source_keys and not cf.verified_false
    again, _ = rows.text(labels, training=False)
    assert selected.receipt.sha256 == again.receipt.sha256
    assert torch.equal(selected.embeddings, again.embeddings)
    changed = replace(labels.parents[0], captions=("altered human bytes",))
    with pytest.raises(ValueError, match="population"):
        rows.text(ParentRetrievalBatch((changed,)), training=True)
    with pytest.raises(ValueError, match="population"):
        rows.text(ParentRetrievalBatch((labels.parents[0], labels.parents[0])), training=False)
    with pytest.raises(ValueError, match="bool"):
        rows.text(labels, training=1)
    with pytest.raises(ValueError):
        ParentHumanClipRows(task, selected)
    wrong = replace(task.parents[0], split="test", component="C09")
    test_task = ParentRetrievalTask(
        (wrong,), expected_family_keys=(wrong.annotation_family_sha256,)
    )
    with pytest.raises(ValueError, match="development"):
        ParentHumanClipRows(test_task, original)


def test_disk_source_binds_the_same_task_and_returns_explicitly_no_cf(tmp_path):
    from test_parent_disk_source import _source

    source, _, _, task, record = _source(tmp_path)
    rows = ParentHumanClipRows(task, _original(task))
    connected = DevelopmentParentDiskSource(
        task,
        (record,),
        yaw_eligibility=source.yaw_eligibility,
        language_rows=source.language_rows,
        human_clip_rows=rows,
    )
    labels = task.batch(tuple(rows.parents))
    text, cf = connected.text(labels, training=False)
    assert text.caption_commitments == labels.caption_commitments and not cf.verified_false
    with pytest.raises(ValueError, match="closed original"):
        source.text(labels, training=True)
    other = _task()
    with pytest.raises(ValueError, match="same complete"):
        DevelopmentParentDiskSource(
            task,
            (record,),
            yaw_eligibility=source.yaw_eligibility,
            language_rows=source.language_rows,
            human_clip_rows=ParentHumanClipRows(other, _original(other)),
        )


@pytest.mark.parametrize("stage", ["base", "residual"])
def test_cached_text_host_optimizer_validation_resume_is_bitwise(stage, tmp_path):
    if stage == "base":
        from test_continuous_base_retrieval import _setup
    else:
        from test_continuous_parent_host import _setup
    system, task, source, config, bindings = _setup(tmp_path)
    labels = task.batch(tuple(parent.annotation_family_sha256 for parent in task.parents))
    original = source.clip.encode(
        labels.captions, caption_commitments=labels.caption_commitments, batch_size=3
    )
    cached = ParentHumanClipRows(task, original)
    source.text = cached.text
    initial = copy.deepcopy(system)
    full = ContinuousParentTrainingHost(system, task, source, config, bindings).fit(
        tmp_path / "full"
    )
    interrupted = ContinuousParentTrainingHost(
        copy.deepcopy(initial), task, source, config, bindings
    ).fit(tmp_path / "interrupted", stop_after_steps=1)
    resumed_model = copy.deepcopy(initial)
    resumed = ContinuousParentTrainingHost(resumed_model, task, source, config, bindings).fit(
        tmp_path / "resumed",
        resume_checkpoint=interrupted.latest_checkpoint.path,
        resume_sha256=interrupted.latest_checkpoint.sha256,
    )
    assert full.outcome == resumed.outcome == "COMPLETED"
    assert full.validation_history == resumed.validation_history
    expected, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[key]) == _stable_hash(expected[key])
