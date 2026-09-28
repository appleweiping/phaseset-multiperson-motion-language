"""Data-free provenance/label fixtures, not human relation ground truth."""

import hashlib
import json

import pytest
import torch
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def record(name, rows=2, component="C01", split="train"):
    raw = json.dumps(
        {"scene_explained": [f"{name} fixture row {i}" for i in range(rows)], "mood": ["ignored"]}
    ).encode()
    return ParentCaptionRecord.from_official_bytes(
        source_sha256=key(f"source-{name}"),
        annotation_family_sha256=hashlib.sha256(raw).hexdigest(),
        component=component,
        split=split,
        annotation_bytes=raw,
        expected_human_rows=rows,
    )


def task(*parents):
    return ParentRetrievalTask(
        tuple(parents), expected_family_keys=tuple(p.annotation_family_sha256 for p in parents)
    )


def test_complete_gallery_uses_one_parent_and_all_variable_human_rows():
    a, b, c = record("a", 2), record("b", 5), record("c", 1, "C00", "validation")
    index = task(b, c, a)
    batch = index.gallery(split="train")
    assert len(batch.parents) == 2 and len(batch.captions) == 7
    assert len(set(batch.motion_source_keys)) == 2
    mask = batch.positive_mask(device=torch.device("cpu"))
    assert mask.shape == (2, 7) and mask.sum() == 7
    assert sorted(mask.sum(dim=1).tolist()) == [2, 5]
    assert bool(mask.any(dim=0).all())
    assert index.gallery(split="validation").captions == c.captions
    assert "ignored" not in batch.captions


def test_batch_order_and_caption_lineage_match_existing_native_clip_contract():
    a, b = record("a"), record("b", 3)
    batch = task(a, b).batch((b.annotation_family_sha256, a.annotation_family_sha256))
    assert batch.motion_source_keys == (b.source_sha256, a.source_sha256)
    assert batch.captions == b.captions + a.captions
    expected = hashlib.sha256(
        bytes.fromhex(b.source_sha256) + bytes.fromhex(b.annotation_family_sha256) + bytes(8)
    ).digest()
    assert batch.caption_commitments[0] == expected
    assert len(set(batch.caption_commitments)) == 5
    assert (
        batch.text_positive_keys
        == (b.annotation_family_sha256,) * 3 + (a.annotation_family_sha256,) * 2
    )


def test_task_input_reordering_has_same_canonical_gallery():
    a, b = record("a"), record("b")
    assert task(a, b).gallery(split="train") == task(b, a).gallery(split="train")


def test_released_sibling_and_source_alias_are_not_independent_gallery_rows():
    a, b = record("a"), record("b")
    with pytest.raises(ValueError, match="siblings"):
        task(a, a)
    alias = ParentCaptionRecord(
        a.source_sha256, b.annotation_family_sha256, "C01", "train", b.captions
    )
    with pytest.raises(ValueError, match="siblings"):
        task(a, alias)


def test_expected_population_rejects_partial_and_extra_parent_inputs():
    a, b = record("a"), record("b")
    with pytest.raises(ValueError, match="population"):
        ParentRetrievalTask(
            (a,), expected_family_keys=(a.annotation_family_sha256, b.annotation_family_sha256)
        )
    with pytest.raises(ValueError, match="population"):
        ParentRetrievalTask((a, b), expected_family_keys=(a.annotation_family_sha256,))


def test_component_cross_split_is_rejected_before_subset_selection():
    a, b = record("a"), record("b", split="validation")
    with pytest.raises(ValueError, match="component"):
        task(a, b)


def test_batch_rejects_repetition_unknown_and_empty_gallery():
    a = record("a")
    index = task(a)
    for keys in ((a.annotation_family_sha256,) * 2, (key("unknown"),), ()):
        with pytest.raises(ValueError):
            index.batch(keys)
    with pytest.raises(ValueError):
        index.gallery(split="validation")


def test_native_bytes_and_row_census_are_not_replaced_or_normalized():
    raw = b'{"scene_explained": ["  unnormalized human fixture  "]}'
    kwargs = dict(
        source_sha256=key("source"),
        annotation_family_sha256=hashlib.sha256(raw).hexdigest(),
        component="C01",
        split="train",
        annotation_bytes=raw,
        expected_human_rows=1,
    )
    assert ParentCaptionRecord.from_official_bytes(**kwargs).captions == (
        "  unnormalized human fixture  ",
    )
    with pytest.raises(ValueError, match="bytes"):
        ParentCaptionRecord.from_official_bytes(**{**kwargs, "annotation_bytes": raw + b" "})
    with pytest.raises(ValueError, match="census"):
        ParentCaptionRecord.from_official_bytes(**{**kwargs, "expected_human_rows": 3})
