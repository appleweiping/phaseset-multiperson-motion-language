"""Real sampler/optimizer wiring on analytic fixtures, not native pilot results."""

import copy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from phaseset_core.continuous_parent_host import ParentHostConfig
from phaseset_core.frozen_clip_text import select_frozen_clip_text_rows
from phaseset_core.literature_parent_host import MIMEParentTrainingHost
from phaseset_core.mime_parent_curriculum import (
    mime_parent_epoch_batches,
    mime_training_parent_anchors,
)
from phaseset_core.parent_clip_rows import ParentHumanClipRows
from phaseset_core.parent_retrieval_task import ParentRetrievalTask
from phaseset_core.training import TrainingCheckpointError, _load_torch_checkpoint, _stable_hash
from test_literature_parent_host import setup
from test_phaseset_frozen_clip_text import _fixture


def human_rows(task, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    clip, _, _, _ = _fixture(tmp_path)
    all_labels = task.batch(tuple(p.annotation_family_sha256 for p in task.parents))
    original = clip.encode(all_labels.captions, all_labels.caption_commitments, batch_size=3)
    provider = ParentHumanClipRows(task, original)
    labels = task.batch(
        tuple(p.annotation_family_sha256 for p in task.parents if p.component == "C01")
    )
    text, no_cf = provider.text(labels, training=True)
    assert not no_cf.source_keys
    return labels, text, provider


def test_closed_human_row_selection_builds_train_only_anchors_with_independent_mean(tmp_path):
    _, task, _, _, _ = setup("mime")
    labels, text, provider = human_rows(task, tmp_path)
    rng = _stable_hash(torch.get_rng_state())
    anchors = mime_training_parent_anchors(task, text, train_components=("C01",))
    assert _stable_hash(torch.get_rng_state()) == rng
    assert anchors.family_keys == labels.motion_positive_keys
    expected = []
    for parent in labels.parents:
        own = task.batch((parent.annotation_family_sha256,))
        rows = [provider.original.embeddings[provider.indices[c]] for c in own.caption_commitments]
        expected.append(F.normalize(torch.stack(rows).double().mean(0).float(), dim=0))
    assert torch.equal(anchors.embeddings, torch.stack(expected))


def test_cached_validation_or_changed_human_selection_is_not_a_training_anchor(tmp_path):
    _, task, _, _, _ = setup("mime")
    labels, text, provider = human_rows(task, tmp_path)
    val = task.batch(
        tuple(p.annotation_family_sha256 for p in task.parents if p.component == "C03")
    )
    # Equal-sized authentic cache subset, but from selection/held-out text.
    wrong = select_frozen_clip_text_rows(
        provider.original,
        tuple(provider.indices[c] for c in val.caption_commitments[: len(labels.captions)]),
    )
    with pytest.raises(ValueError, match="all and only learning"):
        mime_training_parent_anchors(task, wrong, train_components=("C01",))
    changed = tuple(
        replace(p, captions=("changed human", *p.captions[1:])) if p.component == "C01" else p
        for p in task.parents
    )
    changed_task = ParentRetrievalTask(
        changed, expected_family_keys=tuple(p.annotation_family_sha256 for p in changed)
    )
    with pytest.raises(ValueError, match="differs from"):
        mime_training_parent_anchors(changed_task, text, train_components=("C01",))


def test_actual_mime_host_uses_curriculum_epochs_and_resumes_all_state_bitwise(tmp_path):
    model, task, source, _, bindings = setup("mime")
    _, text, _ = human_rows(task, tmp_path / "text")
    anchors = mime_training_parent_anchors(task, text, train_components=("C01",))
    config = ParentHostConfig.for_literature(
        "MIME-Set", 1729, ("C01",), ("C03",), epochs=4, parent_batch_size=2
    )
    initial = copy.deepcopy(model)

    def host(system, *, width=2, anchors=anchors):
        return MIMEParentTrainingHost(
            system, task, source, config, bindings, anchors=anchors, neighbor_window=width
        )

    caller_anchors = replace(anchors, embeddings=anchors.embeddings.clone())
    complete_host = host(model, anchors=caller_anchors)
    caller_anchors.embeddings[0, 0] += 0.125
    assert torch.equal(complete_host._curriculum_anchors.embeddings, anchors.embeddings)
    assert complete_host._manifest["mime_curriculum"]["anchor_embeddings_sha256"] == _stable_hash(
        anchors.embeddings
    )
    for epoch in (0, 3, 12):
        assert complete_host._batches(epoch) == mime_parent_epoch_batches(
            task,
            anchors,
            train_components=("C01",),
            batch_size=2,
            neighbor_window=2,
            seed=1729,
            epoch=epoch,
        )
    assert (
        complete_host._manifest["mime_curriculum"]["source"]
        == "all_learning_parent_human_frozen_clip_mean"
    )
    full = complete_host.fit(tmp_path / "full", checkpoint_keep_recent=2)
    split = host(copy.deepcopy(initial)).fit(
        tmp_path / "split", stop_after_steps=3, checkpoint_keep_recent=2
    )
    resumed = host(copy.deepcopy(initial)).fit(
        tmp_path / "resumed",
        resume_checkpoint=split.latest_checkpoint.path,
        resume_sha256=split.latest_checkpoint.sha256,
        checkpoint_keep_recent=2,
    )
    assert full.global_step == resumed.global_step == 4 and len(full.validation_history) == 4
    expected, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[key]) == _stable_hash(expected[key])
    assert resumed.validation_history == full.validation_history
    source.loaded.clear()
    source.texts.clear()
    with pytest.raises(TrainingCheckpointError, match="execution inputs drifted"):
        host(copy.deepcopy(initial), width=3).fit(
            tmp_path / "width-drift",
            resume_checkpoint=split.latest_checkpoint.path,
            resume_sha256=split.latest_checkpoint.sha256,
        )
    assert source.loaded == source.texts == []
    changed_anchors = anchors.embeddings.clone()
    changed_anchors[0, 0] += 0.125
    bad = replace(anchors, embeddings=changed_anchors)
    with pytest.raises(TrainingCheckpointError, match="execution inputs drifted"):
        host(copy.deepcopy(initial), anchors=bad).fit(
            tmp_path / "anchor-drift",
            resume_checkpoint=split.latest_checkpoint.path,
            resume_sha256=split.latest_checkpoint.sha256,
        )
    assert source.loaded == source.texts == []
