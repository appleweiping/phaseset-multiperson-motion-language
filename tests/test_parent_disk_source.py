"""Analytic disk source and independent replay checks, not native training."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from phaseset_core.parent_disk_source import (
    DevelopmentParentDiskSource,
    ParentDiskRecord,
    deterministic_parent_yaw,
)
from phaseset_core.dct_calibration import DCT_FLOOR_SCHEMA, DctFloorReceipt, dct_config_sha256
from phaseset_core.dct_relations import DctViewContext
from phaseset_core.directional_phase import LocalPhaseConfig
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.tmr_feature_cache import FrozenTMRRowCache, save_frozen_tmr_rows
from phaseset_core.tmr_set import TMRTextBatch
from phaseset_core.training import _capture_rng, _stable_hash
from test_continuous_training_input import _input


def _source(tmp_path, *, allowed=True, phase=True, split="train"):
    inputs, body, physical, record = _input(tmp_path)
    parent = ParentCaptionRecord("a" * 64, "b" * 64, "C01", split, ("group moves together",))
    task = ParentRetrievalTask((parent,), expected_family_keys=("b" * 64,))
    text = TMRTextBatch(
        torch.ones(1, 4, 768), torch.ones(1, 4, dtype=torch.bool), torch.full((1, 768), 768**-0.5)
    )
    cached = save_frozen_tmr_rows(tmp_path / "language", parent.captions, text)
    row = ParentDiskRecord(parent, body, record["prepared_record"], physical, record)
    source = DevelopmentParentDiskSource(
        task,
        (row,),
        yaw_eligibility={"b" * 64: allowed},
        language_rows=FrozenTMRRowCache(tmp_path / "language", cached),
        energy_floors=inputs.cached_field.energy_floors if phase else None,
    )
    return source, parent, inputs, task, row


def test_replayed_complete_body_is_identical_after_model_rng_draws(tmp_path):
    source, parent, original, _, _ = _source(tmp_path, phase=False)
    args = dict(seed=1729, epoch=3, training=True)
    before = _stable_hash(_capture_rng())
    first = source.capture(parent, **args)
    assert _stable_hash(_capture_rng()) == before
    torch.randn(127)
    np.random.random(127)
    next_rng = _stable_hash(_capture_rng())
    second = source.capture(parent, **args)
    assert _stable_hash(_capture_rng()) == next_rng
    np.testing.assert_array_equal(first.skeletons, second.skeletons)
    np.testing.assert_array_equal(first.track_mask, original.capture.track_mask)
    assert first.decisions == original.capture.decisions and len(first.windows()) == 2
    assert first.frame_count == 600 and first.source_sha256 == parent.source_sha256
    assert first.augmentation_yaw == deterministic_parent_yaw("a" * 64, seed=1729, epoch=3)
    assert first.augmentation_yaw != source.capture(parent, **{**args, "epoch": 4}).augmentation_yaw
    validation = source.capture(parent, **{**args, "training": False})
    np.testing.assert_array_equal(validation.skeletons, original.capture.skeletons)
    assert validation.augmentation_yaw == 0.0
    assert not first.skeletons[~first.track_mask].view(np.uint32).any()
    assert not first.skeletons.flags.writeable


def test_phase_views_recompute_every_field_and_validation_reuses_cache(tmp_path):
    source, parent, _, _, _ = _source(tmp_path)
    args = dict(seed=2718, epoch=0, training=True)
    first, second = source.view(parent, **args), source.view(parent, **args)
    body = source.capture(parent, **args)
    np.testing.assert_array_equal(first.capture.skeletons, body.skeletons)
    assert not first.reused_physical_cache
    for name in ("actor_features", "actor_patch_mask", "root_positions", "root_patch_mask"):
        np.testing.assert_array_equal(
            getattr(first.phase_field, name), getattr(second.phase_field, name)
        )
    for a, b in zip(first.phase_field.responses, second.phase_field.responses, strict=True):
        np.testing.assert_array_equal(a, b)
    assert source.view(parent, **{**args, "training": False}).reused_physical_cache
    text = source.tmr_text(parent.captions * 2)
    assert len(text.tokens) == 2 and torch.equal(text.tokens[0], text.tokens[1])
    assert torch.equal(source.wamo_cls(parent.captions), text.tokens[:1, 0])


def test_a6_disk_views_use_body22_without_morlet_cache(tmp_path, monkeypatch):
    source, parent, original, task, row = _source(tmp_path, phase=False)
    receipt = DctFloorReceipt(
        DCT_FLOOR_SCHEMA,
        "main",
        ("C01",),
        "a" * 64,
        dct_config_sha256(LocalPhaseConfig()),
        np.full(6, 1e-3, dtype=np.float64),
    )
    body_only = replace(row, physical_directory=None, physical_record=None)
    dct_source = DevelopmentParentDiskSource(
        task,
        (body_only,),
        yaw_eligibility={"b" * 64: True},
        language_rows=source.language_rows,
        dct_floor_receipt=receipt,
    )
    view = dct_source.view(parent, seed=1729, epoch=0, training=False)
    assert type(view.phase_field) is DctViewContext
    assert not view.reused_physical_cache and not hasattr(view.phase_field, "responses")
    np.testing.assert_array_equal(view.capture.skeletons, original.capture.skeletons)
    assert view.phase_field.physical_view_sha256
    assert view.phase_field.dct_floor_receipt_sha256 == receipt.sha256
    rotated = dct_source.view(parent, seed=1729, epoch=1, training=True)
    assert type(rotated.phase_field) is DctViewContext
    assert rotated.capture.augmentation_yaw != 0.0
    enormous = replace(
        body_only,
        prepared_record={**body_only.prepared_record, "shape": [2, 1_000_000, 22, 3]},
    )
    dct_source.records[parent.annotation_family_sha256] = enormous
    monkeypatch.setattr(
        dct_source, "capture", lambda *args, **kwargs: pytest.fail("body load started")
    )
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        dct_source.view(parent, seed=1729, epoch=0, training=False)
    with pytest.raises(ValueError, match="not be admitted together"):
        DevelopmentParentDiskSource(
            task,
            (row,),
            yaw_eligibility={"b" * 64: True},
            language_rows=source.language_rows,
            energy_floors=np.zeros(6),
            dct_floor_receipt=receipt,
        )


def test_unapproved_caption_yaw_and_task_drift_do_not_get_silent_defaults(tmp_path):
    source, parent, original, task, row = _source(tmp_path, allowed=False)
    rotated = source.view(parent, seed=31415, epoch=0, training=True)
    assert rotated.reused_physical_cache and rotated.capture.augmentation_yaw == 0.0
    np.testing.assert_array_equal(rotated.capture.skeletons, original.capture.skeletons)
    for changed in (
        replace(parent, source_sha256="c" * 64),
        replace(parent, captions=("different label",)),
        replace(parent, annotation_family_sha256="c" * 64),
    ):
        with pytest.raises(ValueError, match="admitted"):
            source.capture(changed, seed=1729, epoch=0, training=True)
    for rules in ({}, {"b" * 64: 1}):
        with pytest.raises(ValueError, match="eligibility"):
            DevelopmentParentDiskSource(
                task, (row,), yaw_eligibility=rules, language_rows=source.language_rows
            )
    with pytest.raises(ValueError, match="cover"):
        DevelopmentParentDiskSource(
            task, (), yaw_eligibility={}, language_rows=source.language_rows
        )
    with pytest.raises(ValueError, match="complete matching"):
        DevelopmentParentDiskSource(
            task,
            (replace(row, physical_record=None),),
            yaw_eligibility={"b" * 64: True},
            language_rows=source.language_rows,
            energy_floors=np.zeros(6),
        )


def test_final_test_is_never_opened_by_development_source(tmp_path):
    with pytest.raises(ValueError, match="final test"):
        _source(tmp_path, split="test")


@pytest.mark.parametrize("seed,epoch", [(0, 0), (True, 0), (1729, -1), (1729, True)])
def test_yaw_seed_and_epoch_are_explicit(seed, epoch):
    with pytest.raises(ValueError):
        deterministic_parent_yaw("a" * 64, seed=seed, epoch=epoch)
