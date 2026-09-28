"""Actual tiny optimization/checkpoint/resume with the holistic capture caller."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from phaseset_core import training
from phaseset_core.capture_validation import CaptureValidationSource, run_capture_validation
from phaseset_core.models import GroupBaseOutput
from test_phaseset_capture_validation import _groups, _id, _source


class _TinyLearnedBase(nn.Module):
    """A small trainable interface fixture, not a registered architecture result."""
    system_id = "B0"

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 512)
        self.dropout = nn.Dropout(0.2)

    def forward(self, groups):
        values = torch.tensor(np.array(groups.skeletons[:, 0, 0, 0, :2], copy=True))
        embeddings = self.projection(self.dropout(values)).contiguous()
        actors = embeddings[:, None, :].expand(-1, 3, -1).contiguous()
        return GroupBaseOutput(embeddings, actors, torch.ones(actors.shape[:2], dtype=torch.bool))


@dataclass
class _TrainSource:
    split: str = "train"
    manifest_sha256: str = "b" * 64

    def iter_epoch(self, *, epoch, seed):
        for index, coordinate in enumerate(((1.0, 0.0), (0.0, 1.0))):
            text = torch.zeros((1, 512), dtype=torch.float32)
            text[0, index] = 1.0
            key = _id(f"train-window/{index}")
            yield training.RetrievalTrainingBatch(_groups(*coordinate), text, (key,), (key,),
                (_id(f"train-text/{index}"),), "train")


def _runtime(path: Path):
    with torch.random.fork_rng():
        torch.manual_seed(811)
        system = training.BaseRetrievalSystem(_TinyLearnedBase(), embedding_dim=512)
    config = training.TrainingConfig(stage="base", seed=1729, synthetic_contract=True,
                                     effective_global_batch=2)
    return training.PhaseSetTrainingRuntime(system, config, checkpoint_directory=path)


def _payload(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal_nested(a, b)
    else:
        assert left == right


def test_capture_gallery_selects_validation_checkpoint_and_binds_complete_census(tmp_path):
    runtime = _runtime(tmp_path / "one")
    source = _source()
    report = runtime.fit(_TrainSource(), source, stop_after_global_step=1)
    assert report.status == "INTERRUPTED"
    assert report.completed_epochs == 1
    assert report.best_checkpoint is not None
    assert report.val_manifest_sha256 == source.census_sha256
    assert report.validation_edges_seen == 9  # three K=3 windows, not two captures
    payload = _payload(report.latest_checkpoint.path)
    actual = run_capture_validation(runtime.system, runtime.config, source)
    assert payload["best_validation_metric"] == float(actual.primary_capture_r1)
    assert payload["val_manifest_sha256"] == source.census_sha256
    assert payload["val_manifest_sha256"] != source.manifest_sha256


def test_capture_checkpoint_resume_equals_uninterrupted_optimizer_and_model(tmp_path):
    source = _source()
    complete = _runtime(tmp_path / "complete")
    expected = complete.fit(_TrainSource(), source, stop_after_global_step=2)
    resumed = _runtime(tmp_path / "resumed")
    partial = resumed.fit(_TrainSource(), source, stop_after_global_step=1)
    # A new runtime reconstructs and checks state from the checkpoint, rather
    # than continuing the in-memory optimizer left by the first fit call.
    restarted = _runtime(tmp_path / "resumed")
    observed = restarted.fit(_TrainSource(), source,
        resume_checkpoint=partial.latest_checkpoint.path,
        resume_checkpoint_sha256=partial.latest_checkpoint.sha256, stop_after_global_step=2)
    left, right = _payload(expected.latest_checkpoint.path), _payload(observed.latest_checkpoint.path)
    for field in ("model", "optimizer", "scheduler", "rng", "epoch", "update_index",
                  "global_step", "best_validation_metric", "val_manifest_sha256"):
        _equal_nested(left[field], right[field])
    assert expected.best_validation_metric == observed.best_validation_metric
    assert expected.validation_edges_seen == observed.validation_edges_seen == 18


def test_changed_capture_census_is_rejected_on_resume(tmp_path):
    source = _source()
    runtime = _runtime(tmp_path / "changed")
    report = runtime.fit(_TrainSource(), source, stop_after_global_step=1)
    changed = CaptureValidationSource("val", "d" * 64, source.captures)
    restarted = _runtime(tmp_path / "changed")
    with pytest.raises(training.TrainingCheckpointError, match="val_manifest"):
        restarted.fit(_TrainSource(), changed, resume_checkpoint=report.latest_checkpoint.path,
            resume_checkpoint_sha256=report.latest_checkpoint.sha256, stop_after_global_step=2)


def test_capture_source_cannot_enter_training_iterator(tmp_path):
    runtime = _runtime(tmp_path / "invalid-train")
    with pytest.raises(training.TrainingRuntimeError, match="cannot be a training source"):
        runtime.fit(_source(), _source(), stop_after_global_step=1)


def test_live_capture_census_drift_rejected_before_encoding(tmp_path):
    runtime = _runtime(tmp_path / "drift")
    source = _source()
    runtime._val_manifest = source.census_sha256
    changed = replace(source, manifest_sha256="e" * 64)
    with pytest.raises(training.TrainingRuntimeError, match="census changed"):
        runtime._validation(changed)


def test_scored_capture_census_is_checked_before_using_metric(tmp_path, monkeypatch):
    from phaseset_core import capture_validation as capture_module

    runtime = _runtime(tmp_path / "scored-drift")
    source = _source()
    runtime._val_manifest = source.census_sha256
    original = capture_module.run_capture_validation

    def swapped(system, config, checked):
        return original(system, config, replace(checked, manifest_sha256="f" * 64))

    monkeypatch.setattr(capture_module, "run_capture_validation", swapped)
    with pytest.raises(training.TrainingRuntimeError, match="scored capture census"):
        runtime._validation(source)


def test_edge_count_uses_the_scored_snapshot_not_live_original(tmp_path, monkeypatch):
    from phaseset_core import capture_validation as capture_module

    runtime = _runtime(tmp_path / "edge-snapshot")
    source = _source()
    runtime._val_manifest = source.census_sha256
    original = capture_module.run_capture_validation

    def mutate_original(system, config, checked):
        assert checked is not source
        result = original(system, config, checked)
        object.__setattr__(source, "captures", source.captures[:1])
        return result

    monkeypatch.setattr(capture_module, "run_capture_validation", mutate_original)
    _, _, total_edges, maximum_edges = runtime._validation(source)
    assert total_edges == 9 and maximum_edges == 3


@pytest.mark.parametrize("name", (
    "CaptureValidationSource", "run_capture_validation", "_snapshot_source",
    "_capture_fractions", "pool_capture_windows", "capture_r1_contributions",
    "validate_retrieval_dataset", "variable_positive_symmetric_infonce",
))
def test_formal_runtime_rejects_capture_function_drift(tmp_path, monkeypatch, name):
    from phaseset_core import capture_validation as capture_module

    config = training.TrainingConfig(stage="base", seed=1729)
    system, binding = training.construct_registered_base_seed_bound_system("B0", config)
    runtime = training.PhaseSetTrainingRuntime(system, config, tmp_path / name,
                                              initialization_binding=binding)
    monkeypatch.setattr(capture_module, name, lambda *args, **kwargs: None)
    with pytest.raises(training.TrainingRuntimeError, match="capture runtime function identity"):
        runtime._assert_live_formal_binding(require_initial_state=True)
