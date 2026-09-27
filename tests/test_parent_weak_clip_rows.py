"""Analytic cached weak inputs; not generated native labels or convergence."""

import copy
from dataclasses import replace
import hashlib

import pytest
import torch

from phaseset_core import frozen_clip_text as clip
from phaseset_core.capture_validation import CaptureValidationError, _validate_text_batch
from phaseset_core.continuous_parent_host import ContinuousParentTrainingHost
from phaseset_core.continuous_parent_training import ParentWeakCounterfactualRows
from phaseset_core.parent_clip_rows import ParentHumanClipRows
from phaseset_core.parent_disk_source import DevelopmentParentDiskSource
from phaseset_core.parent_weak_clip_rows import ParentWeakClipRows, WeakCaptionRecord
from phaseset_core.training import _capture_rng, _load_torch_checkpoint, _stable_hash
from test_parent_clip_rows import _original, _task
from test_phaseset_frozen_clip_cache_rehydrate import _fixture


POLICY = "c" * 64


def _encoded(captions, commitments):
    embeddings, receipt = _fixture(len(captions), batch_size=2)
    rows = tuple(
        (i, c.hex(), hashlib.sha256(text.encode()).hexdigest(), *old[3:])
        for i, (c, text, old) in enumerate(
            zip(commitments, captions, receipt.caption_rows, strict=True)
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


def _record(parent, ordinal=0, *, included=True):
    return WeakCaptionRecord(
        parent.source_sha256,
        parent.annotation_family_sha256,
        ordinal,
        f"analytic machine alternative {ordinal} for {parent.source_sha256[:8]}",
        "d" * 64,
        POLICY,
        included,
    )


def _extra(records):
    return _encoded(
        tuple(r.negative_caption for r in records), tuple(r.caption_commitment for r in records)
    )


def _pool(task=None):
    task = _task() if task is None else task
    human = ParentHumanClipRows(task, _original(task))
    records = tuple(_record(p) for p in task.parents if p.component == "C01")
    return ParentWeakClipRows(
        human, _extra(records), records, train_components=("C01",), policy_sha256=POLICY
    )


def test_two_original_pool_keeps_prefix_bytes_without_encoder_or_rng(monkeypatch):
    pool = _pool()
    before = _stable_hash(_capture_rng())

    def forbidden(*args, **kwargs):
        raise AssertionError("cached pool attempted model/runtime loading")

    monkeypatch.setattr(clip, "_real_backend", forbidden)
    monkeypatch.setattr(clip, "_load_components", forbidden)
    result = clip.pool_frozen_clip_text_rows(
        pool.human.original, pool.extra_original, prefix_indices=(3, 1, 0), extra_indices=(1, 0)
    )
    embeddings, lineage, digest = _validate_text_batch(result, allow_row_selection=True)
    assert torch.equal(embeddings[:3], pool.human.original.embeddings[[3, 1, 0]])
    assert torch.equal(embeddings[3:], pool.extra_original.embeddings[[1, 0]])
    assert lineage == tuple(pool.human.original.caption_commitments[i] for i in (3, 1, 0)) + tuple(
        pool.extra_original.caption_commitments[i] for i in (1, 0)
    )
    assert result.receipt.origin_receipts == (
        pool.human.original.receipt,
        pool.extra_original.receipt,
    )
    assert result.receipt.prefix_count == 3 and digest == result.receipt.sha256
    public = result.receipt.to_public_dict()
    assert public["training_authorized"] is False and public["authority"] == 0
    assert public["operation"].endswith("no_encoder_call")
    assert b"analytic machine" not in result.receipt.canonical_json_bytes()
    snapshot = result.embeddings
    snapshot.zero_()
    assert not torch.equal(snapshot, result.embeddings)
    assert _stable_hash(_capture_rng()) == before
    with pytest.raises(CaptureValidationError, match="not admitted"):
        _validate_text_batch(result)
    with pytest.raises(clip.FrozenClipTextAdapterError):
        clip.rehydrate_frozen_clip_text_batch(
            result.embeddings,
            receipt_json_bytes=result.receipt.canonical_json_bytes(),
            expected_receipt_sha256=result.receipt.sha256,
        )
    with pytest.raises(ValueError, match="original"):
        clip.select_frozen_clip_text_rows(result, (0,))


@pytest.mark.parametrize(
    "identity",
    ["snapshot_manifest_sha256", "live_model_manifest_sha256", "runtime_manifest_sha256"],
)
def test_pool_rejects_mismatched_encoding_identity(identity):
    pool = _pool()
    changed = copy.deepcopy(pool.extra_original)
    object.__setattr__(
        changed, "_FrozenClipTextBatch__receipt", replace(changed.receipt, **{identity: "f" * 64})
    )
    with pytest.raises(ValueError, match="same model"):
        clip.pool_frozen_clip_text_rows(
            pool.human.original, changed, prefix_indices=(0,), extra_indices=(0,)
        )


@pytest.mark.parametrize("change", ["prefix", "indices", "runtime"])
def test_pool_receipt_ancestry_is_bound_to_original_rows(change):
    pool = _pool()
    result = clip.pool_frozen_clip_text_rows(
        pool.human.original, pool.extra_original, prefix_indices=(0, 2), extra_indices=(1, 0)
    )
    receipt = result.receipt
    if change == "prefix":
        receipt = replace(receipt, prefix_count=1)
    elif change == "indices":
        receipt = replace(receipt, selected_indices=((0, 2), (0, 1)))
    else:
        receipt = replace(
            receipt,
            origin_receipts=(
                receipt.origin_receipts[0],
                replace(receipt.origin_receipts[1], runtime_manifest_sha256="f" * 64),
            ),
        )
    object.__setattr__(result, "_FrozenClipTextBatch__receipt", receipt)
    with pytest.raises(CaptureValidationError, match="ancestry"):
        _validate_text_batch(result, allow_row_selection=True)


def test_pool_rejects_repeated_commitments_between_originals():
    pool = _pool()
    with pytest.raises(ValueError, match="repeat"):
        clip.pool_frozen_clip_text_rows(
            pool.human.original, pool.human.original, prefix_indices=(0,), extra_indices=(0,)
        )


def test_stage_pool_parent_reordering_preserves_human_occurrences_and_cf_columns():
    pool = _pool()
    parents = tuple(p for p in reversed(tuple(pool.human.parents.values())) if p.component == "C01")
    from phaseset_core.parent_retrieval_task import ParentRetrievalBatch

    labels = ParentRetrievalBatch(parents)
    rows, cf = pool.text(labels, training=True)
    expected, _ = pool.human.text(labels, training=True)
    assert rows.caption_commitments[: len(labels.captions)] == labels.caption_commitments
    assert torch.equal(rows.embeddings[: len(labels.captions)], expected.embeddings)
    assert len(labels.captions) == 3  # Equal human strings remain separate occurrences.
    assert type(cf) is ParentWeakCounterfactualRows and not hasattr(cf, "verified_false")
    assert cf.source_keys == labels.motion_source_keys
    assert cf.negative_columns == (3, 4)
    assert cf.positive_columns == (0, len(parents[0].captions))
    assert cf.included_weak == (True, True)


def test_evaluation_never_reads_machine_rows_or_changes_original_gallery():
    pool = _pool()
    from phaseset_core.parent_retrieval_task import ParentRetrievalBatch

    labels = ParentRetrievalBatch(tuple(pool.human.parents.values()))
    expected, _ = pool.human.text(labels, training=False)
    pool.extra_original = None  # An unusable machine store must be irrelevant to eval.
    rows, cf = pool.text(labels, training=False)
    assert torch.equal(rows.embeddings, expected.embeddings)
    assert rows.receipt.sha256 == expected.receipt.sha256
    assert not cf.verified_false and len(rows.embeddings) == len(labels.captions)
    with pytest.raises(ValueError, match="held-out"):
        pool.text(labels, training=True)


@pytest.mark.parametrize(
    "drift", ["source", "policy", "ordinal", "positive_alias", "encoded_text", "commitment"]
)
def test_weak_records_reject_stage_and_cached_text_drift(drift):
    pool = _pool()
    records = list(pool.records)
    extra = pool.extra_original
    if drift == "source":
        records[0] = replace(records[0], source_sha256="e" * 64)
    elif drift == "policy":
        records[0] = replace(records[0], policy_sha256="e" * 64)
    elif drift == "ordinal":
        records[0] = replace(records[0], positive_ordinal=100)
    elif drift == "positive_alias":
        records[0] = replace(
            records[0],
            negative_caption=pool.human.parents[records[0].annotation_family_sha256].captions[0],
        )
    elif drift == "encoded_text":
        extra = _encoded(
            ("changed text", records[1].negative_caption),
            tuple(r.caption_commitment for r in records),
        )
    else:
        extra = _encoded(
            tuple(r.negative_caption for r in records), (b"x" * 32, records[1].caption_commitment)
        )
    with pytest.raises(ValueError):
        ParentWeakClipRows(
            pool.human, extra, tuple(records), train_components=("C01",), policy_sha256=POLICY
        )


def test_component_admission_allows_outer_fold_learning_but_not_held_out():
    pool = _pool()
    parent = next(p for p in pool.human.parents.values() if p.component == "C03")
    record = _record(parent)
    with pytest.raises(ValueError, match="training parents"):
        ParentWeakClipRows(
            pool.human,
            _extra((record,)),
            (record,),
            train_components=("C01",),
            policy_sha256=POLICY,
        )
    outer = ParentWeakClipRows(
        pool.human, _extra((record,)), (record,), train_components=("C03",), policy_sha256=POLICY
    )
    from phaseset_core.parent_retrieval_task import ParentRetrievalBatch

    rows, cf = outer.text(ParentRetrievalBatch((parent,)), training=True)
    assert len(rows.embeddings) == 2 and cf.negative_columns == (1,)


def test_no_candidate_parent_returns_weak_empty_without_fabricating_negatives():
    pool = _pool()
    record = pool.records[0]
    pool = ParentWeakClipRows(
        pool.human, _extra((record,)), (record,), train_components=("C01",), policy_sha256=POLICY
    )
    parent = next(
        p
        for p in pool.human.parents.values()
        if p.component == "C01" and p.annotation_family_sha256 != record.annotation_family_sha256
    )
    from phaseset_core.parent_retrieval_task import ParentRetrievalBatch

    rows, cf = pool.text(ParentRetrievalBatch((parent,)), training=True)
    assert len(rows.embeddings) == len(parent.captions) and not cf.source_keys
    assert type(cf) is ParentWeakCounterfactualRows


def test_existing_disk_source_uses_separate_pool_only_for_training(tmp_path):
    from test_parent_disk_source import _source

    original, _, inputs, task, disk_record = _source(tmp_path)
    human = ParentHumanClipRows(task, _original(task))
    record = _record(task.parents[0], included=False)
    pool = ParentWeakClipRows(
        human, _extra((record,)), (record,), train_components=("C01",), policy_sha256=POLICY
    )
    source = DevelopmentParentDiskSource(
        task,
        (disk_record,),
        yaw_eligibility=original.yaw_eligibility,
        language_rows=original.language_rows,
        human_clip_rows=human,
        weak_clip_rows=pool,
    )
    labels = task.batch(tuple(human.parents))
    text, cf = source.text(labels, training=True)
    assert len(text.embeddings) == 2 and cf.included_weak == (False,)
    text, cf = source.text(labels, training=False)
    assert len(text.embeddings) == 1 and not cf.verified_false
    assert (
        source.capture(task.parents[0], seed=1729, epoch=0, training=False).frame_count
        == inputs.capture.frame_count
    )
    with pytest.raises(ValueError, match="same admitted human"):
        DevelopmentParentDiskSource(
            task,
            (disk_record,),
            yaw_eligibility=original.yaw_eligibility,
            language_rows=original.language_rows,
            weak_clip_rows=pool,
        )


def test_cached_pool_actual_optimizer_and_resume_preserve_human_validation(tmp_path):
    from test_continuous_parent_host import _setup, _events

    system, task, source, config, bindings = _setup(tmp_path)
    human = ParentHumanClipRows(task, _original(task))
    records = tuple(_record(p) for p in task.parents if p.component in config.train_components)
    pool = ParentWeakClipRows(
        human,
        _extra(records),
        records,
        train_components=config.train_components,
        policy_sha256=POLICY,
    )
    source.text = pool.text
    initial = copy.deepcopy(system)
    full = ContinuousParentTrainingHost(system, task, source, config, bindings).fit(
        tmp_path / "full"
    )
    stopped = ContinuousParentTrainingHost(
        copy.deepcopy(initial), task, source, config, bindings
    ).fit(tmp_path / "stopped", stop_after_steps=1)
    resumed_model = copy.deepcopy(initial)
    resumed = ContinuousParentTrainingHost(resumed_model, task, source, config, bindings).fit(
        tmp_path / "resumed",
        resume_checkpoint=stopped.latest_checkpoint.path,
        resume_sha256=stopped.latest_checkpoint.sha256,
    )
    assert full.outcome == resumed.outcome == "COMPLETED"
    assert full.validation_history == resumed.validation_history
    expected, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[key]) == _stable_hash(expected[key])
    updates = [event for event in _events(tmp_path / "full") if event["phase"] == "update"]
    assert len(updates) == 2 and all(
        event["weak_cf"] == 4 and event["verified_cf"] == 0 and event["human_rows"] == 5
        for event in updates
    )
    # Real analytic optimizer execution is not native learning or label evidence.


def test_base_stage_rejects_cached_weak_pool_before_loading_motion(tmp_path):
    from test_continuous_base_retrieval import _setup
    from phaseset_core.continuous_parent_training import backward_loaded_parent_batch

    system, task, source, config, _ = _setup(tmp_path)
    human = ParentHumanClipRows(task, _original(task))
    records = tuple(_record(p) for p in task.parents if p.component in config.train_components)
    pool = ParentWeakClipRows(
        human,
        _extra(records),
        records,
        train_components=config.train_components,
        policy_sha256=POLICY,
    )
    labels = task.batch(
        tuple(
            p.annotation_family_sha256
            for p in task.parents
            if p.component in config.train_components
        )
    )
    rows, cf = pool.text(labels, training=True)

    def forbidden(*args):
        raise AssertionError("base loaded motion despite inadmissible weak inputs")

    with pytest.raises(ValueError, match="base"):
        backward_loaded_parent_batch(
            system, labels, rows, load_view=forbidden, counterfactuals=cf, cf_weight=0.0, margin=0.0
        )
