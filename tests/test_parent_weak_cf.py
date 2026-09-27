"""Analytic weak-supervision interface checks; not native labels or results."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from phaseset_core.continuous_parent_host import ContinuousParentTrainingHost
from phaseset_core.continuous_base_retrieval import ContinuousBaseRetrievalSystem
from phaseset_core.continuous_parent_training import (
    ParentCounterfactualRows,
    ParentWeakCounterfactualRows,
    backward_loaded_parent_batch,
    backward_parent_batch,
)
from phaseset_core.objectives import variable_positive_symmetric_infonce
from phaseset_core.temporal_coordination import (
    weak_coordination_objective,
    weak_counterfactual_loss,
)
from phaseset_core.training import _load_torch_checkpoint, _stable_hash
from test_continuous_parent_host import _Source, _setup
from test_continuous_parent_training import _case, _key, _population


def test_weak_margin_has_independent_formula_and_no_human_mask():
    positive = torch.tensor([1.0, 2.0, -0.5], requires_grad=True)
    negative = torch.tensor([0.0, 3.0, 0.5], requires_grad=True)
    mask = torch.tensor([True, False, True])
    actual = weak_counterfactual_loss(positive, negative, mask, margin=0.3)
    expected = (
        F.softplus(0.3 + negative[0] - positive[0]) + F.softplus(0.3 + negative[2] - positive[2])
    ) / 2
    assert torch.equal(actual, expected)
    actual.backward()
    assert positive.grad[1] == negative.grad[1] == 0
    assert positive.grad[0] < 0 and negative.grad[0] > 0
    row = ParentWeakCounterfactualRows((), (), (), ())
    assert not hasattr(row, "verified_false")


@pytest.mark.parametrize("size", [0, 3])
def test_excluded_weak_targets_have_zero_loss_and_gradient(size):
    pos = torch.arange(size, dtype=torch.float32, requires_grad=True)
    neg = torch.arange(size, dtype=torch.float32, requires_grad=True)
    value = weak_counterfactual_loss(pos, neg, torch.zeros(size, dtype=torch.bool))
    value.backward()
    assert value == 0 and torch.count_nonzero(pos.grad) == torch.count_nonzero(neg.grad) == 0


@pytest.mark.parametrize("bad", ["shape", "mask", "rank", "negative_margin", "nan_margin"])
def test_weak_margin_rejects_malformed_inputs(bad):
    pos, neg, mask = torch.ones(2), torch.zeros(2), torch.ones(2, dtype=torch.bool)
    margin = 0.2
    if bad == "shape":
        neg = torch.zeros(1)
    elif bad == "mask":
        mask = mask.to(torch.float32)
    elif bad == "rank":
        pos, neg, mask = pos[None], neg[None], mask[None]
    else:
        margin = -0.1 if bad == "negative_margin" else float("nan")
    with pytest.raises(ValueError):
        weak_counterfactual_loss(pos, neg, mask, margin=margin)


def test_weak_objective_keeps_original_retrieval_prefix_and_weight_zero_control():
    scores = torch.tensor([[2.0, 1.0, -1.0], [0.0, 1.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, True]])
    positive, negative = scores[0, :1], scores[0, 2:3]
    contrastive = variable_positive_symmetric_infonce(scores, mask)[2]
    for weight in (0.0, 0.2):
        actual = weak_coordination_objective(
            scores,
            mask,
            cf_positive_scores=positive,
            cf_negative_scores=negative,
            included_weak_mask=torch.tensor([True]),
            cf_weight=weight,
            margin=0.1,
        )
        expected = contrastive + weight * F.softplus(0.1 + negative - positive).mean()
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="cf_weight"):
        weak_coordination_objective(
            scores,
            mask,
            cf_positive_scores=positive,
            cf_negative_scores=negative,
            included_weak_mask=torch.tensor([True]),
            cf_weight=-1,
        )


def _labels():
    task = _population(2)
    return task.batch(tuple(row.annotation_family_sha256 for row in task.parents))


@pytest.mark.parametrize("included", [False, True])
def test_weak_columns_bind_same_parent_human_evidence_outside_gallery(included):
    labels = _labels()
    rows = ParentWeakCounterfactualRows((labels.motion_source_keys[0],), (0,), (2,), (included,))
    indices = rows.indices(labels, text_count=3, device=torch.device("cpu"))
    assert [value.tolist() for value in indices] == [[0], [0], [2], [included]]


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "other_positive",
        "generated_positive",
        "gallery_negative",
        "outside_pool",
        "bool_column",
        "nonbool",
        "length",
        "not_tuple",
    ],
)
def test_weak_columns_reject_fake_positive_labels_and_invalid_admission(bad):
    labels = _labels()
    rows = ParentWeakCounterfactualRows((labels.motion_source_keys[0],), (0,), (2,), (True,))
    settings = {
        "unknown": dict(source_keys=(_key("unknown"),)),
        "other_positive": dict(positive_columns=(1,)),
        "generated_positive": dict(positive_columns=(2,)),
        "gallery_negative": dict(negative_columns=(1,)),
        "outside_pool": dict(negative_columns=(3,)),
        "bool_column": dict(positive_columns=(False,)),
        "nonbool": dict(included_weak=(1,)),
        "length": dict(included_weak=()),
        "not_tuple": dict(source_keys=[labels.motion_source_keys[0]]),
    }
    with pytest.raises(ValueError):
        replace(rows, **settings[bad]).indices(labels, text_count=3, device=torch.device("cpu"))


@pytest.mark.parametrize("included,weight", [(False, 0.2), (True, 0.2), (True, 0.0)])
def test_weak_complete_parent_replay_matches_independent_dense_gradients(
    tmp_path, included, weight
):
    system, views, labels, text, _ = _case(tmp_path, extra_cf=True)
    dense = copy.deepcopy(system)
    cf = ParentWeakCounterfactualRows((views[0].positive_capture_key,), (0,), (4,), (included,))
    scores = dense.score(views, text).scores
    expected = variable_positive_symmetric_infonce(
        scores[:, :3].contiguous(), labels.positive_mask(device=scores.device)
    )[2]
    if included:
        expected = expected + weight * F.softplus(0.2 + scores[0, 4] - scores[0, 0])
    expected.backward()
    result = backward_parent_batch(
        system, views, text, labels, counterfactuals=cf, cf_weight=weight, margin=0.2
    )
    torch.testing.assert_close(torch.tensor(result.loss), expected.detach(), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(result.scores, scores.detach(), rtol=2e-6, atol=2e-7)
    assert result.verified_counterfactual_count == 0
    assert result.weak_counterfactual_count == int(included)
    assert result.retrieval_caption_count == 3 and result.replayed_captures == 2
    for (name, actual), (other_name, oracle) in zip(
        system.named_parameters(), dense.named_parameters(), strict=True
    ):
        assert name == other_name
        if name.startswith("frozen_b2."):
            assert actual.grad is oracle.grad is None
        else:
            torch.testing.assert_close(actual.grad, oracle.grad, rtol=3e-5, atol=3e-6)


def test_weak_negative_cannot_alias_known_original_positive_and_kind_is_explicit(tmp_path):
    system, views, labels, _, clip = _case(tmp_path)
    text = clip.encode(
        labels.captions + ("walking",),
        caption_commitments=labels.caption_commitments + (bytes.fromhex(_key("alias")),),
        batch_size=3,
    )
    for rows, error in (
        (
            ParentWeakCounterfactualRows((views[0].positive_capture_key,), (0,), (3,), (True,)),
            ValueError,
        ),
        (SimpleNamespace(source_keys=()), TypeError),
    ):
        with pytest.raises(error):
            backward_parent_batch(
                system, views, text, labels, counterfactuals=rows, cf_weight=0.2, margin=0.2
            )


class _WeakSource(_Source):
    """Synthetic generation for tests only; no actual label or language model."""

    def text(self, labels, *, training):
        if not training:
            return super().text(labels, training=False)
        self.text_loaded.append((labels.motion_source_keys, training))
        extras = tuple(
            f"synthetic weak alternative {index}" for index in range(len(labels.parents))
        )
        commits = tuple(
            bytes.fromhex(_key("weak/" + source)) for source in labels.motion_source_keys
        )
        offsets, offset = [], 0
        for parent in labels.parents:
            offsets.append(offset)
            offset += len(parent.captions)
        text = self.clip.encode(
            labels.captions + extras,
            caption_commitments=labels.caption_commitments + commits,
            batch_size=3,
        )
        rows = ParentWeakCounterfactualRows(
            labels.motion_source_keys,
            tuple(offsets),
            tuple(range(offset, offset + len(extras))),
            (True,) * len(extras),
        )
        return text, rows


def test_weak_host_actual_updates_resume_and_disjoint_counts_are_bitwise(tmp_path):
    system, task, source, config, bindings = _setup(tmp_path)
    initial = copy.deepcopy(system)

    def make_source():
        return _WeakSource(tuple(source.views.values()), source.clip)

    full = ContinuousParentTrainingHost(system, task, make_source(), config, bindings).fit(
        tmp_path / "full"
    )
    stopped = ContinuousParentTrainingHost(
        copy.deepcopy(initial), task, make_source(), config, bindings
    ).fit(tmp_path / "stopped", stop_after_steps=1)
    resumed_model = copy.deepcopy(initial)
    resumed_source = make_source()
    resumed = ContinuousParentTrainingHost(
        resumed_model, task, resumed_source, config, bindings
    ).fit(
        tmp_path / "resumed",
        resume_checkpoint=stopped.latest_checkpoint.path,
        resume_sha256=stopped.latest_checkpoint.sha256,
    )
    assert full.outcome == resumed.outcome == "COMPLETED"
    assert full.global_step == resumed.global_step == 2
    for name, value in system.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name])
    a, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    b, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(a[key]) == _stable_hash(b[key])
    events = [
        json.loads(path.read_text()) for path in sorted((tmp_path / "full").glob("event-*.json"))
    ]
    updates = [event for event in events if event["phase"] == "update"]
    assert len(updates) == 2
    assert all(
        event["weak_cf"] == 4 and event["verified_cf"] == 0 and event["human_rows"] == 5
        for event in updates
    )
    assert any(not training for _, training in resumed_source.text_loaded)
    assert not any(parent.split == "test" for parent in task.parents)
    # Actual AdamW/resume on analytic fixtures is software evidence only.


def test_original_human_row_type_retains_its_distinct_verification_field():
    original = ParentCounterfactualRows((), (), (), ())
    assert hasattr(original, "verified_false") and not hasattr(original, "included_weak")


@pytest.mark.parametrize("kind", ["tmr", "wamo", "mime"])
def test_new_shared_event_counts_preserve_literature_hosts_without_weak_labels(kind, tmp_path):
    from test_literature_parent_host import test_real_objective_optimizer_and_full_gallery_endpoint

    # Only the new shared event/result field motivates this existing analytic
    # optimizer call. It is not a repeated native comparator quality review.
    test_real_objective_optimizer_and_full_gallery_endpoint(kind, tmp_path)
    events = [
        json.loads(path.read_text()) for path in sorted((tmp_path / "run").glob("event-*.json"))
    ]
    updates = [event for event in events if event["phase"] == "update"]
    assert len(updates) == 2 and all(
        event["weak_cf"] == event["verified_cf"] == 0 for event in updates
    )


def test_base_training_rejects_weak_cf_before_any_capture_load(tmp_path):
    residual, views, labels, text, _ = _case(tmp_path, extra_cf=True)
    base = ContinuousBaseRetrievalSystem(copy.deepcopy(residual.frozen_b2).requires_grad_(True))
    cf = ParentWeakCounterfactualRows((views[0].positive_capture_key,), (0,), (4,), (True,))
    loaded = []

    def load(key):
        loaded.append(key)
        raise AssertionError("weak CF must be rejected before loading a base input")

    with pytest.raises(ValueError, match="base training"):
        backward_loaded_parent_batch(
            base, labels, text, load_view=load, counterfactuals=cf, cf_weight=0, margin=0
        )
    assert not loaded


def test_validation_rejects_weak_cf_gallery_leak_with_original_failure_evidence(tmp_path):
    system, task, source, config, bindings = _setup(tmp_path)

    class LeakyWeakSource(_WeakSource):
        def text(self, labels, *, training):
            return super().text(labels, training=True)

    bad_source = LeakyWeakSource(tuple(source.views.values()), source.clip)
    host = ContinuousParentTrainingHost(
        system, task, bad_source, replace(config, epochs=1), bindings
    )
    with pytest.raises(ValueError, match="only all official human"):
        host.fit(tmp_path / "failure")
    terminal = json.loads((tmp_path / "failure/terminal.json").read_text())
    assert terminal["outcome"] == "FAILED" and terminal["global_step"] == 1
