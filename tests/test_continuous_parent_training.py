"""Data-free qualifications; no native training or human-CF truth is claimed."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import pytest
import torch

from phaseset_core.continuous_parent_training import (
    ParentCounterfactualRows,
    backward_parent_batch,
    parent_epoch_batches,
)
from phaseset_core.continuous_retrieval import ContinuousRetrievalSystem
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.training import (
    BaseRetrievalSystem,
    _atomic_torch_save,
    _capture_rng,
    _load_torch_checkpoint,
    _restore_rng,
)
from test_continuous_retrieval import _MockB2
from test_continuous_training_input import _input
from test_phaseset_frozen_clip_text import _fixture


def _key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _population(count=202, component="C01", split="train"):
    parents = tuple(
        ParentCaptionRecord(
            _key(f"source/{index}"),
            _key(f"family/{index}"),
            component,
            split,
            (f"human row {index}",),
        )
        for index in range(count)
    )
    return ParentRetrievalTask(
        parents, expected_family_keys=tuple(parent.annotation_family_sha256 for parent in parents)
    )


def test_parent_epoch_visits_every_family_once_with_no_window_weighting():
    task = _population()
    kwargs = dict(train_components=("C01",), batch_size=128, seed=1729, epoch=0)
    batches = parent_epoch_batches(task, **kwargs)
    assert tuple(len(batch.parents) for batch in batches) == (128, 74)
    keys = tuple(key for batch in batches for key in batch.motion_positive_keys)
    assert len(keys) == len(set(keys)) == 202
    assert set(keys) == {parent.annotation_family_sha256 for parent in task.parents}
    assert parent_epoch_batches(task, **kwargs) == batches
    for seed, epoch in ((2718, 0), (1729, 1)):
        other = parent_epoch_batches(task, **(kwargs | dict(seed=seed, epoch=epoch)))
        assert other != batches
        assert {key for batch in other for key in batch.motion_positive_keys} == set(keys)


def test_singleton_tail_is_rebalanced_without_dropping_or_repeating_parents():
    task = _population(129)
    batches = parent_epoch_batches(
        task, train_components=("C01",), batch_size=128, seed=31415, epoch=2
    )
    assert tuple(len(batch.parents) for batch in batches) == (127, 2)
    assert len({key for batch in batches for key in batch.motion_positive_keys}) == 129


def test_fold_component_selection_does_not_confuse_old_main_validation_with_test():
    task = _population(4, component="C00", split="validation")
    assert (
        len(parent_epoch_batches(task, train_components=("C00",), batch_size=4, seed=1729, epoch=0))
        == 1
    )
    for bad_task, components, size in (
        (_population(4, split="test"), ("C01",), 4),
        (_population(1), ("C01",), 4),
        (_population(3), ("C01",), 2),
        (task, ("C00", "C00"), 4),
        (task, ("C02",), 4),
    ):
        with pytest.raises(ValueError):
            parent_epoch_batches(
                bad_task, train_components=components, batch_size=size, seed=1729, epoch=0
            )


def _case(tmp_path, *, extra_cf=False):
    (tmp_path / "input").mkdir()
    (tmp_path / "clip").mkdir()
    source, _, _, _ = _input(tmp_path / "input")
    first = source.view(allow_shared_yaw=False)
    second = replace(first, capture=replace(first.capture, source_sha256=_key("other-source")))
    # Analytic duplicated motion is deliberately a fixture, not native data or
    # an independent real capture, and cannot support paper performance claims.
    records = (
        ParentCaptionRecord(
            first.positive_capture_key, _key("first-family"), "C01", "train", ("walking", "moving")
        ),
        ParentCaptionRecord(
            second.positive_capture_key, _key("second-family"), "C02", "train", ("turning",)
        ),
    )
    task = ParentRetrievalTask(
        records, expected_family_keys=tuple(row.annotation_family_sha256 for row in records)
    )
    labels = task.batch(tuple(row.annotation_family_sha256 for row in records))
    clip, _, _, _ = _fixture(tmp_path / "clip")
    captions = labels.captions
    commitments = labels.caption_commitments
    if extra_cf:
        captions += ("first leads", "first follows")
        commitments += (bytes.fromhex(_key("cf-positive")), bytes.fromhex(_key("cf-negative")))
    text = clip.encode(captions, caption_commitments=commitments, batch_size=3)
    torch.manual_seed(1729)
    system = ContinuousRetrievalSystem(BaseRetrievalSystem(_MockB2(), embedding_dim=512))
    return system, (first, second), labels, text, clip


def _empty_cf():
    return ParentCounterfactualRows((), (), (), ())


@pytest.mark.parametrize("verified", [None, False, True])
def test_complete_gallery_replay_matches_dense_loss_and_all_parameter_gradients(tmp_path, verified):
    system, views, labels, text, _ = _case(tmp_path, extra_cf=verified is not None)
    dense = copy.deepcopy(system)
    base = copy.deepcopy(system.frozen_b2.state_dict())
    cf = (
        _empty_cf()
        if verified is None
        else ParentCounterfactualRows((views[0].positive_capture_key,), (3,), (4,), (verified,))
    )
    output = (
        dense(
            views,
            text,
            motion_positive_keys=labels.motion_positive_keys,
            text_positive_keys=labels.text_positive_keys,
        )
        if verified is None
        else None
    )
    if output is None:
        scores = dense.score(views, text).scores
        from phaseset_core.temporal_coordination import coordination_objective

        expected_loss = coordination_objective(
            scores[:, :3].contiguous(),
            labels.positive_mask(device=scores.device),
            cf_positive_scores=scores[0, 3:4],
            cf_negative_scores=scores[0, 4:5],
            verified_negative_mask=torch.tensor([verified]),
            cf_weight=0.2,
            margin=0.2,
        )
    else:
        scores = output.scores
        empty = scores.new_empty(0)
        expected_loss = dense.objective(
            output,
            cf_positive_scores=empty,
            cf_negative_scores=empty,
            verified_negative_mask=torch.empty(0, dtype=torch.bool),
            cf_weight=0.2,
            margin=0.2,
        )
    expected_loss.backward()
    result = backward_parent_batch(
        system, views, text, labels, counterfactuals=cf, cf_weight=0.2, margin=0.2
    )
    # Independent dense and parent-row GEMMs can round differently. Preserve
    # bitwise checks for same-shape replay and checkpoint/resume, not this oracle.
    torch.testing.assert_close(
        torch.tensor(result.loss), expected_loss.detach(), rtol=2e-6, atol=2e-7
    )
    torch.testing.assert_close(result.scores, scores.detach(), rtol=2e-6, atol=2e-7)
    assert result.replayed_captures == result.parent_count == 2
    assert result.retrieval_caption_count == 3
    assert result.verified_counterfactual_count == (1 if verified is True else 0)
    for (name, actual), (expected_name, expected) in zip(
        system.named_parameters(), dense.named_parameters(), strict=True
    ):
        assert name == expected_name
        if name.startswith("frozen_b2."):
            assert actual.grad is expected.grad is None
        else:
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-6)
    for name, value in system.frozen_b2.state_dict().items():
        assert torch.equal(value, base[name])


def test_parent_sources_and_actual_human_text_are_checked_before_backward(tmp_path):
    system, views, labels, text, clip = _case(tmp_path)
    for wrong_views, wrong_text in (
        (tuple(reversed(views)), text),
        (
            views,
            clip.encode(
                ("not walking", "moving", "turning"),
                caption_commitments=labels.caption_commitments,
                batch_size=3,
            ),
        ),
    ):
        with pytest.raises(ValueError):
            backward_parent_batch(
                system,
                wrong_views,
                wrong_text,
                labels,
                counterfactuals=_empty_cf(),
                cf_weight=0.2,
                margin=0.2,
            )


def test_cf_rows_cannot_fabricate_retrieval_positives_or_use_unadmitted_sources(tmp_path):
    _, views, labels, text, _ = _case(tmp_path)
    for cf in (
        ParentCounterfactualRows((views[0].positive_capture_key,), (1,), (0,), (True,)),
        ParentCounterfactualRows((_key("unadmitted"),), (0,), (2,), (True,)),
        ParentCounterfactualRows((views[0].positive_capture_key,), (0,), (0,), (False,)),
        ParentCounterfactualRows((views[0].positive_capture_key,), (0,), (3,), (True,)),
    ):
        with pytest.raises(ValueError):
            cf.indices(labels, text_count=len(text.caption_commitments), device=torch.device("cpu"))


def test_rng_advances_once_and_mode_restores_after_complete_capture_replays(tmp_path):
    system, views, labels, text, _ = _case(tmp_path)
    # Stochastic fixture probes RNG replay; no production architecture change.
    system.text_adapter[1] = torch.nn.Dropout(0.25)
    system.eval()
    torch.manual_seed(31415)
    initial = _capture_rng()
    system.train()
    with torch.no_grad():
        for view in views:
            system.score((view,), text)
    expected = torch.get_rng_state().clone()
    system.eval()
    _restore_rng(initial)
    backward_parent_batch(
        system, views, text, labels, counterfactuals=_empty_cf(), cf_weight=0.2, margin=0.2
    )
    assert not system.training and not system.frozen_b2.training
    assert torch.equal(torch.get_rng_state(), expected)


def test_parent_replay_optimizer_checkpoint_resume_is_bitwise_exact(tmp_path):
    system, views, labels, text, _ = _case(tmp_path)
    optimizer = torch.optim.AdamW(
        [p for p in system.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
    )

    def step(model, opt):
        result = backward_parent_batch(
            model, views, text, labels, counterfactuals=_empty_cf(), cf_weight=0.2, margin=0.2
        )
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        return result.loss

    step(system, optimizer)
    saved = _atomic_torch_save(
        tmp_path / "step1.pt",
        {
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _capture_rng(),
            "parent_keys": labels.motion_positive_keys,
            "epoch": 0,
            "parent_offset": 2,
            "global_step": 1,
        },
    )
    expected = step(system, optimizer)
    resumed = copy.deepcopy(system)
    resumed_opt = torch.optim.AdamW(
        [p for p in resumed.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
    )
    payload, _ = _load_torch_checkpoint(saved.path, expected_sha256=saved.sha256)
    assert payload["parent_keys"] == labels.motion_positive_keys
    resumed.load_state_dict(payload["model"])
    resumed_opt.load_state_dict(payload["optimizer"])
    _restore_rng(payload["rng"])
    actual = step(resumed, resumed_opt)
    assert actual == expected
    for actual_parameter, expected_parameter in zip(
        resumed.parameters(), system.parameters(), strict=True
    ):
        assert torch.equal(actual_parameter, expected_parameter)
