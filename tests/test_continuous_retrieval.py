"""Data-free directed-packet and capture/text training seam checks."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from phaseset_core.continuous_retrieval import (
    ContinuousRetrievalSystem,
    capture_family_positive_mask,
    human_holistic_captions,
)
from phaseset_core.models import GroupBaseOutput
from phaseset_core.temporal_coordination import TemporalIncidenceEncoder
from phaseset_core.training import (
    BaseRetrievalSystem,
    _atomic_torch_save,
    _capture_rng,
    _load_torch_checkpoint,
    _restore_rng,
)
from test_continuous_training_input import _input
from test_phaseset_frozen_clip_text import _commitment, _fixture
from test_phaseset_v2_coordination import field, motion


def _directed_oracle(model, physical, text):
    """Unblocked small-graph oracle retaining both ordered endpoints and states."""
    hidden, nodes, topology = model._encode_context(physical)
    output = model._readout(physical, hidden, nodes, topology)
    a, b, endpoints, track = model._half_edges(physical, hidden, 0, physical.pair_count)
    left, right = endpoints[:, 0], endpoints[:, 1]
    eligible = (topology[left] | topology[right])[..., None]
    values = []
    for half, source, target in ((a, left, right), (b, right, left)):
        delta = model.edge_delta(torch.cat((half, nodes[source], nodes[target]), dim=-1))
        delta = torch.where(eligible, delta, torch.zeros_like(delta))
        encoded, _ = model.edge_temporal(half + model.topology_scale.tanh() * delta, track)
        values.append(encoded)
    from phaseset_core.directional_phase import local_pair_chunk

    observed = torch.from_numpy(
        local_pair_chunk(physical, 0, physical.pair_count).phase_mask.any(-1)
    )
    mask = torch.cat((observed & track, observed & track))
    cosines = F.normalize(torch.cat(values), dim=-1) @ F.normalize(text, dim=-1).T
    best = cosines.masked_fill(~mask[..., None], -torch.inf).amax((0, 1))
    return 0.5 * (output.embedding @ F.normalize(text, dim=-1).T + best)


def test_directed_stream_matches_complete_packet_oracle_and_preserves_old_readout():
    torch.manual_seed(1729)
    physical = field()
    model = TemporalIncidenceEncoder(width=8).eval()
    text = torch.randn(4, 8, requires_grad=True)
    result = model.score_text(physical, text)
    old_output = model(physical)
    for name in ("embedding", "patch_tokens", "topology_nodes", "valid_pair_count"):
        assert torch.equal(getattr(result.coordination, name), getattr(old_output, name))
    torch.testing.assert_close(
        result.cosine, _directed_oracle(model, physical, text), rtol=0, atol=0
    )
    assert bool(result.periodic_support) and bool((result.cosine.abs() <= 1.00001).all())
    result.cosine.sum().backward()
    assert bool(torch.isfinite(text.grad).all())
    for module in (
        model.actor_temporal,
        model.half_edge,
        model.node_mlp,
        model.node_temporal,
        model.edge_delta,
        model.edge_temporal,
        model.group_temporal,
    ):
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())


def test_directed_score_k2_full_equals_pair_only_and_topology_gradients_exact_zero():
    torch.manual_seed(2718)
    physical = field(motion((0.0, 0.9)))
    full = TemporalIncidenceEncoder(width=8)
    pair = copy.deepcopy(full)
    pair.use_topology = False
    text = torch.randn(3, 8)
    a, b = full.score_text(physical, text), pair.score_text(physical, text)
    assert torch.equal(a.cosine, b.cosine)
    a.cosine.sum().backward()
    for name, p in full.named_parameters():
        if name.startswith(("node_mlp", "node_temporal", "edge_delta", "topology_scale")):
            assert p.grad is not None
            assert not bool(p.grad.view(torch.int32).any())


def test_directed_checkpoint_backward_matches_uncheckpointed_across_64_edges():
    torch.manual_seed(31415)
    physical = field(motion(tuple(np.arange(13) * 0.37)))
    a = TemporalIncidenceEncoder(width=8)
    b = copy.deepcopy(a)
    b.checkpoint_blocks = False
    text_a = torch.randn(2, 8, requires_grad=True)
    text_b = text_a.detach().clone().requires_grad_()
    x, y = a.score_text(physical, text_a), b.score_text(physical, text_b)
    assert torch.equal(x.cosine, y.cosine)
    x.cosine.sum().backward()
    y.cosine.sum().backward()
    torch.testing.assert_close(text_a.grad, text_b.grad, rtol=0, atol=0)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_directed_scores_actor_permutation_and_padding_invariant():
    torch.manual_seed(1729)
    model = TemporalIncidenceEncoder(width=8).eval()
    text = torch.randn(3, 8)
    expected = model.score_text(field(), text)
    for order in ((2, 0, 1), (1, 2, 0)):
        actual = model.score_text(field(motion(order=order, padding=5)), text)
        assert torch.equal(actual.cosine, expected.cosine)


def test_observed_stationary_support_is_not_periodic_evidence():
    batch = motion()
    from phaseset_core.contracts import PreparedGroupBatch

    stationary = PreparedGroupBatch(
        np.zeros_like(batch.skeletons),
        batch.actor_mask,
        batch.frame_mask,
        batch.track_mask,
        batch.actor_commitments,
        batch.group_commitments,
    )
    model = TemporalIncidenceEncoder(width=8)
    result = model.score_text(field(stationary), torch.randn(3, 8))
    assert bool(result.coordination.patch_mask.any())
    assert not bool(result.periodic_support)
    assert not bool(result.cosine.detach().view(torch.int32).any())
    result.cosine.sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            # A masked score has zero derivative; products with signed weights
            # may represent that derivative as -0. This is not the distinct
            # K=2 topology +0 bit-pattern contract tested above.
            assert bool(torch.isfinite(p.grad).all())
            assert not bool(torch.count_nonzero(p.grad))


def test_directed_score_resource_limit_and_text_boundary_are_explicit():
    model = TemporalIncidenceEncoder(width=8)
    physical = field()
    huge = replace(physical, intervals=physical.intervals * 10000)
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        model.score_text(huge, torch.ones(1000, 8))
    for text in (torch.ones(3, 7), torch.ones(0, 8), torch.ones(1, 8, dtype=torch.float64)):
        with pytest.raises(ValueError, match="text requires"):
            model.score_text(physical, text)


class _MockB2(nn.Module):
    """Mock only; never a qualified real B2 checkpoint or literature baseline."""

    system_id = "B2"

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, groups):
        signal = torch.tensor(groups.skeletons.copy()).mean((1, 2, 3, 4))[:, None]
        embedding = self.dropout(self.projection(signal))
        return GroupBaseOutput(
            embedding, embedding[:, None], torch.ones(len(signal), 1, dtype=torch.bool)
        )


def test_capture_text_variable_positives_frozen_base_and_optimizer_resume(tmp_path):
    torch.manual_seed(1729)
    (tmp_path / "input").mkdir()
    (tmp_path / "clip").mkdir()
    source, _, _, _ = _input(tmp_path / "input")
    first = source.view(allow_shared_yaw=False)
    second = replace(first, capture=replace(first.capture, source_sha256="b" * 64))
    assert first.capture.group_commitment == second.capture.group_commitment
    clip, _, _, _ = _fixture(tmp_path / "clip")
    captions = clip.encode(
        ("walking", "moving", "turning"),
        caption_commitments=tuple(_commitment(x) for x in ("a", "b", "c")),
        batch_size=3,
    )
    model = ContinuousRetrievalSystem(BaseRetrievalSystem(_MockB2(), embedding_dim=512)).train()
    assert not model.frozen_b2.training and not any(
        p.requires_grad for p in model.frozen_b2.parameters()
    )
    base_before = copy.deepcopy(model.frozen_b2.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    with torch.no_grad():
        arbitrary_gallery = model.score((first,), captions)
    assert arbitrary_gallery.scores.shape == (1, 3)
    assert not hasattr(arbitrary_gallery, "positive_mask")

    def step(system, opt):
        opt.zero_grad(set_to_none=True)
        output = system(
            (first, second),
            captions,
            motion_positive_keys=("c" * 64, "d" * 64),
            text_positive_keys=("c" * 64, "c" * 64, "d" * 64),
        )
        assert output.positive_mask.tolist() == [[True, True, False], [False, False, True]]
        empty = output.scores.new_empty((0,))
        loss = system.objective(
            output,
            cf_positive_scores=empty,
            cf_negative_scores=empty,
            verified_negative_mask=torch.empty(0, dtype=torch.bool),
        )
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is None for p in system.frozen_b2.parameters())
        assert any(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for p in system.text_adapter.parameters()
        )
        torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad], 1.0)
        opt.step()
        return loss.detach()

    step(model, optimizer)
    saved = _atomic_torch_save(
        tmp_path / "step1.pt",
        {
            "global_step": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _capture_rng(),
        },
    )
    expected = step(model, optimizer)
    resumed = copy.deepcopy(model)
    resumed_opt = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=3e-4)
    payload, _ = _load_torch_checkpoint(saved.path, expected_sha256=saved.sha256)
    resumed.load_state_dict(payload["model"])
    resumed_opt.load_state_dict(payload["optimizer"])
    _restore_rng(payload["rng"])
    actual = step(resumed, resumed_opt)
    assert torch.equal(actual, expected)
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(a, b)
    for name, value in model.frozen_b2.state_dict().items():
        assert torch.equal(value, base_before[name])


def test_target_families_include_siblings_and_reject_unmatched_rows():
    keys = ("a" * 64, "b" * 64)
    assert capture_family_positive_mask(
        keys, (keys[0], keys[0], keys[1]), device=torch.device("cpu")
    ).tolist() == [[True, True, False], [False, False, True]]
    assert capture_family_positive_mask(
        (keys[0], keys[0], keys[1]), keys, device=torch.device("cpu")
    ).tolist() == [[True, False], [True, False], [False, True]]
    for motions, captions in ((keys, (keys[0],)), (keys, ("invalid",))):
        with pytest.raises(ValueError):
            capture_family_positive_mask(motions, captions, device=torch.device("cpu"))


def test_official_holistic_schema_keeps_variable_human_rows_without_mood_fusion():
    assert human_holistic_captions(
        {
            "scene_explained": ["They move together.", "They walk."],
            "scene_mood_explained": ["They feel happy."],
        }
    ) == ("They move together.", "They walk.")
    for value in ({}, {"scene_explained": []}, {"scene_explained": [""]}, []):
        with pytest.raises(ValueError):
            human_holistic_captions(value)


def test_pre_pilot_language_rule_is_recorded_without_changing_the_87_stage_plan():
    matrix = json.loads(
        (Path(__file__).parents[1] / "configs/phaseset_v2_experiment_matrix.json").read_text()
    )
    contract = matrix["mechanism_contract"]
    assert (
        contract["relation_language_readout"]
        == "0.5 * ordered_capture_cosine + 0.5 * max_whole_directed_packet_cosine"
    )
    assert contract["relation_language_edge_block"] == 64
    assert (
        contract["human_holistic_caption_field"]
        == "all_scene_explained_rows_no_questionnaire_fusion"
    )
    assert len(matrix["runs"]) == 87 and matrix["status"] == "PLANNED_NOT_LAUNCHABLE"
