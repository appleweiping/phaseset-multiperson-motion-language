"""Analytic optimizer/resume checks; never native pilot/convergence evidence."""

import copy
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from phaseset_core.continuous_parent_host import ParentHostBindings, ParentHostConfig
from phaseset_core.literature_parent_host import LiteratureParentTrainingHost
from phaseset_core.mime_set import MIMERetrieval, MIMESetMotion
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.tmr_set import TMRGroupCapture, TMRSet, TMRTextBatch
from phaseset_core.training import (
    TrainingCheckpointError,
    _capture_macro_bidirectional_r1,
    _capture_rng,
    _load_torch_checkpoint,
    _stable_hash,
)
from test_literature_parent_training import fixture
from test_mime_set import config as mime_config, language
from test_wamo_set import small


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Source:
    def __init__(self, captures):
        self.captures = captures
        self.loaded, self.texts = [], []

    def capture(self, parent, *, seed, epoch, training):
        self.loaded.append((parent.source_sha256, epoch, training))
        assert seed == 1729
        return self.captures[parent.source_sha256]

    def _features(self, captions, shape):
        # Independent per-row generators: identical rows/features under batching
        # and replay, without consuming any model/dropout/sampling RNG.
        return torch.stack(
            [
                torch.randn(shape, generator=torch.Generator().manual_seed(int(key(row)[:12], 16)))
                for row in captions
            ]
        )

    def tmr_text(self, captions):
        self.texts.append(("tmr", captions))
        tokens = self._features(captions, (5, 768))
        sentence = torch.nn.functional.normalize(self._features(captions, (768,)), dim=1)
        return TMRTextBatch(tokens, torch.ones(len(captions), 5, dtype=torch.bool), sentence)

    def wamo_cls(self, captions):
        self.texts.append(("wamo", captions))
        return self._features(captions, (768,))


def setup(kind):
    torch.manual_seed(1729)
    captures, labels = fixture()
    records = list(labels.parents[:2])
    validation = replace(labels.parents[2], component="C03", split="validation")
    records.append(validation)
    second = replace(
        validation,
        source_sha256=key("validation-source"),
        annotation_family_sha256=key("validation-family"),
        captions=("other validation a", "other b"),
    )
    records.append(second)
    captures[second.source_sha256] = replace(
        captures[validation.source_sha256], source_sha256=second.source_sha256
    )
    task = ParentRetrievalTask(
        tuple(records), expected_family_keys=tuple(row.annotation_family_sha256 for row in records)
    )
    model = (
        TMRSet(latent_dim=16, ff_size=32, num_layers=1, num_heads=4, dropout=0.1)
        if kind == "tmr"
        else small(dropout=0.1)
        if kind == "wamo"
        else MIMERetrieval(MIMESetMotion(mime_config(dropout=0.1)), language())
    )
    config = ParentHostConfig.for_base(1729, ("C01",), ("C03",), epochs=2, parent_batch_size=2)
    bindings = ParentHostBindings(key("inputs"), key("code"), key("runtime"), None)
    return model, task, Source(captures), config, bindings


@pytest.mark.parametrize("kind", ["tmr", "wamo", "mime"])
def test_real_objective_optimizer_and_full_gallery_endpoint(kind, tmp_path):
    model, task, source, cfg, bindings = setup(kind)
    initial = copy.deepcopy(model.state_dict())
    host = LiteratureParentTrainingHost(model, task, source, cfg, bindings)
    result = host.fit(tmp_path / "run")
    assert result.outcome == "COMPLETED" and result.global_step == 2
    assert result.parents_seen == 4 and result.completed_epochs == 2
    assert len(result.validation_history) == 2
    assert result.best_validation_r1 == max(value for _, value in result.validation_history)
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items())
    training_loads = [row for row in source.loaded if row[2]]
    assert len(training_loads) == (16 if kind == "tmr" else 8)
    assert len(source.loaded) == len(training_loads) + 4
    events = [json.loads(p.read_text()) for p in sorted((tmp_path / "run").glob("event-*.json"))]
    updates = [row for row in events if row["phase"] == "update"]
    assert [row["human_rows"] for row in updates] == [3, 3]
    assert all(row["verified_cf"] == 0 and row["learning_rate"] == 2e-4 for row in updates)
    losses = [row["loss_components"] for row in events if row["phase"] == "literature_objective"]
    expected = (
        {"loss", "recons", "kl", "latent", "contrastive"}
        if kind == "tmr"
        else {"loss", "reconstruction", "ordering", "contrastive"}
        if kind == "wamo"
        else {"loss"}
    )
    assert len(losses) == 2 and all(set(row) == expected for row in losses)
    # Independent public model.score against ground-truth family positives.
    # No model-generated pseudo-label, caption-weighted endpoint or shortened timeline.
    gallery = host._gallery
    physical = tuple(
        TMRGroupCapture.from_prepared(source.captures[p.source_sha256]) for p in gallery.parents
    )
    model.eval()
    with torch.no_grad():
        text = (
            source.tmr_text(gallery.captions)
            if kind == "tmr"
            else source.wamo_cls(gallery.captions)
            if kind == "wamo"
            else model.language.tokenize(gallery.captions)
        )
        scores = model.score(physical, text)
    assert scores.shape == (2, 5)
    oracle = _capture_macro_bidirectional_r1(
        scores.cpu(),
        gallery.positive_mask(device=torch.device("cpu")),
        tuple(bytes.fromhex(k) for k in gallery.motion_positive_keys),
        tuple(bytes.fromhex(k) for k in gallery.text_positive_keys),
        tuple(bytes.fromhex(k) for k in gallery.motion_source_keys),
        gallery.caption_commitments,
    )
    model.train()
    state = _stable_hash(_capture_rng())
    assert host._validate() == oracle == result.validation_history[-1][1]
    assert model.training and _stable_hash(_capture_rng()) == state
    payload, _ = _load_torch_checkpoint(
        result.best_checkpoint.path, expected_sha256=result.best_checkpoint.sha256
    )
    assert payload["validation_history"][-1][1] == result.best_validation_r1
    assert (
        host._manifest["literature_method"]
        == {"tmr": "TMR-Set", "wamo": "WaMo-Set", "mime": "MIME-Set"}[kind]
    )


@pytest.mark.parametrize("kind", ["tmr", "wamo", "mime"])
def test_optimizer_dropout_vae_interruption_resume_bitwise(kind, tmp_path):
    model, task, source, cfg, bindings = setup(kind)
    initial = copy.deepcopy(model)
    full = LiteratureParentTrainingHost(model, task, source, cfg, bindings).fit(tmp_path / "full")
    stopped = LiteratureParentTrainingHost(copy.deepcopy(initial), task, source, cfg, bindings).fit(
        tmp_path / "stop", stop_after_steps=1
    )
    resumed_model = copy.deepcopy(initial)
    resumed = LiteratureParentTrainingHost(resumed_model, task, source, cfg, bindings).fit(
        tmp_path / "resumed",
        resume_checkpoint=stopped.latest_checkpoint.path,
        resume_sha256=stopped.latest_checkpoint.sha256,
    )
    assert resumed.outcome == full.outcome == "COMPLETED"
    assert resumed.parents_seen == full.parents_seen == 4
    assert resumed.validation_history == full.validation_history
    for name, value in model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name]), name
    expected, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for name in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[name]) == _stable_hash(expected[name]), name


@pytest.mark.parametrize("kind", ["tmr", "wamo", "mime"])
def test_loss_configuration_drift_rejected_on_resume(kind, tmp_path):
    model, task, source, cfg, bindings = setup(kind)
    initial = copy.deepcopy(model)
    stopped = LiteratureParentTrainingHost(model, task, source, cfg, bindings).fit(
        tmp_path / "stop", stop_after_steps=0
    )
    if kind == "tmr":
        initial.loss_weights = replace(initial.loss_weights, latent=0.02)
    elif kind == "wamo":
        initial.config = replace(initial.config, reconstruction_weight=2.0)
    else:
        initial.motion.config = replace(initial.motion.config, epsilon=1e-6)
    with pytest.raises(TrainingCheckpointError, match="drifted"):
        LiteratureParentTrainingHost(initial, task, source, cfg, bindings).fit(
            tmp_path / "drift",
            resume_checkpoint=stopped.latest_checkpoint.path,
            resume_sha256=stopped.latest_checkpoint.sha256,
        )
    assert json.loads((tmp_path / "drift" / "terminal.json").read_text())["outcome"] == "FAILED"


def test_test_and_held_out_rows_never_loaded_or_persisted(tmp_path):
    model, task, source, cfg, bindings = setup("wamo")
    sealed = ParentCaptionRecord(
        key("sealed-source"), key("sealed-family"), "C09", "test", ("sealed",)
    )
    held = replace(
        sealed,
        source_sha256=key("held-source"),
        annotation_family_sha256=key("held-family"),
        component="C04",
        split="train",
        captions=("held",),
    )
    rows = task.parents + (sealed, held)
    task = ParentRetrievalTask(
        rows, expected_family_keys=tuple(row.annotation_family_sha256 for row in rows)
    )
    host = LiteratureParentTrainingHost(model, task, source, cfg, bindings)
    host.fit(tmp_path / "not-started", stop_after_steps=0)
    assert not source.loaded and not source.texts
    assert all(row["component"] in {"C01", "C03"} for row in host._manifest["task"])
    assert "sealed" not in (tmp_path / "not-started" / "run.json").read_text()
    for wrong in (
        replace(cfg, train_components=("C09",)),
        replace(cfg, validation_components=("C09",)),
    ):
        with pytest.raises(ValueError):
            LiteratureParentTrainingHost(model, task, source, wrong, bindings)
    with pytest.raises(ValueError, match="original losses"):
        LiteratureParentTrainingHost(
            model, task, source, replace(cfg, stage="residual", cf_weight=0.2), bindings
        )
    with pytest.raises(ValueError, match="independently"):
        LiteratureParentTrainingHost(
            model, task, source, cfg, replace(bindings, frozen_base_checkpoint_sha256=key("base"))
        )
