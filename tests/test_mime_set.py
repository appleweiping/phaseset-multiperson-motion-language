"""Analytic software checks, not original-task or human-data performance."""

from dataclasses import replace
import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from phaseset_core.mime_language import MIMETextTokens, TrainableMIMECLIP
from phaseset_core.mime_set import (
    CoAttentionLayer,
    MIMEConfig,
    MIMERetrieval,
    MIMESetMotion,
    OriginalMIMEMotion,
    pair_features,
)
from phaseset_core.tmr_set import TMRGroupCapture, TMRGroupWindow


def capture(*, k=3, length=13, grad=False):
    keys = tuple(bytes([i + 1]) * 32 for i in range(k))
    windows = []
    for start in (0.0, 40.0):
        x = torch.randn(k, length, 66)
        mask = torch.ones_like(x, dtype=torch.bool)
        if length > 1:
            mask[0, 0] = False
        windows.append(
            TMRGroupWindow(x.masked_fill(~mask, 0).requires_grad_(grad), mask, keys, start)
        )
    return TMRGroupCapture(tuple(windows))


def reordered(parent, order):
    return TMRGroupCapture(
        tuple(
            replace(
                window,
                features=window.features[list(order)],
                observed=window.observed[list(order)],
                actor_commitments=tuple(window.actor_commitments[i] for i in order),
            )
            for window in parent.windows
        )
    )


def config(dropout=0.0):
    return MIMEConfig(width=16, layers=1, ff_size=32, dropout=dropout)


class FixtureTokenizer:
    """Deterministic content tokens for software-only segmentation tests."""

    def __call__(self, captions, *, add_special_tokens, truncation):
        assert not add_special_tokens and not truncation
        return {"input_ids": [[i % 127 for i, _ in enumerate(c.split())] for c in captions]}


class FixtureTower(nn.Module):
    """Tiny differentiable seam, never claimed to be actual pretrained CLIP."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.token_embedding = nn.Embedding(128, 512)
        self.mix = nn.Linear(512, 512)

    def forward(self, input_ids, attention_mask, return_dict):
        x = self.embeddings.token_embedding(input_ids % 128)
        x = (x * attention_mask[..., None]).sum(1) / attention_mask.sum(1)[:, None]
        return SimpleNamespace(pooler_output=self.mix(x))


def language(width=16, checkpoint=True, group=True):
    return TrainableMIMECLIP(
        FixtureTower(),
        FixtureTokenizer(),
        width=width,
        checkpoint_segments=checkpoint,
        group_long_text=group,
    )


def manual_attention(module, query, key, value, key_mask):
    width, heads = module.embed_dim, module.num_heads
    projections = []
    for index, x in enumerate((query, key, value)):
        projected = F.linear(
            x,
            module.in_proj_weight[index * width : (index + 1) * width],
            module.in_proj_bias[index * width : (index + 1) * width],
        )
        projections.append(
            projected.reshape(len(x), x.shape[1], heads, width // heads).transpose(1, 2)
        )
    q, k, v = projections
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(width / heads)
    probabilities = scores.masked_fill(~key_mask[:, None, None], -torch.inf).softmax(-1)
    attended = (probabilities @ v).transpose(1, 2).reshape(len(query), query.shape[1], width)
    return F.linear(attended, module.out_proj.weight, module.out_proj.bias)


@pytest.mark.parametrize("shared", [False, True])
def test_bidirectional_full_time_attention_matches_independent_equations(shared):
    torch.manual_seed(1729)
    block = CoAttentionLayer(config(), shared=shared).eval()
    a, b = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
    ma, mb = torch.ones(2, 9, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool)
    mb[0, -2:] = False
    updates = []
    for endpoint, (x, mask) in enumerate(((a, ma), (b, mb))):
        index = 0 if shared else endpoint
        z = block.self_norms[index](x)
        updates.append(
            (x + manual_attention(block.self_attention[index], z, z, z, mask)).masked_fill(
                ~mask[..., None], 0
            )
        )
    sources = [block.cross_norms[0 if shared else i](x) for i, x in enumerate(updates)]
    expected = []
    for endpoint, mask in enumerate((ma, mb)):
        index = 0 if shared else endpoint
        other = 1 - endpoint
        crossed = updates[endpoint] + manual_attention(
            block.cross_attention[index],
            sources[endpoint],
            sources[other],
            sources[other],
            (ma, mb)[other],
        )
        expected.append(
            (crossed + block.ffns[index](block.ff_norms[index](crossed))).masked_fill(
                ~mask[..., None], 0
            )
        )
    for actual, oracle in zip(block(a, b, ma, mb), expected, strict=True):
        torch.testing.assert_close(actual, oracle, rtol=2e-6, atol=3e-7)


def test_original_root_distance_displacement_delta_ratio_and_dimensions():
    torch.manual_seed(2718)
    a, b = torch.randn(2, 7, 135), torch.randn(2, 7, 135)
    ra, rb = torch.randn(2, 7, 3), torch.randn(2, 7, 3)
    mask = torch.ones(2, 7, dtype=torch.bool)
    aa, bb, r = pair_features(a, b, ra, rb, a[..., :3], b[..., :3], mask)
    assert aa.shape[-1] == bb.shape[-1] == 139 and r.shape[-1] == 418
    assert torch.equal(aa[..., :135], a) and torch.equal(bb[..., :135], b)
    assert torch.equal(aa[..., -3:], ra - rb) and torch.equal(bb[..., -3:], rb - ra)
    torch.testing.assert_close(
        r[..., -1],
        a[..., :3].square().sum(-1)
        / (a[..., :3].square().sum(-1) + b[..., :3].square().sum(-1) + 1e-8),
    )
    aa, bb, r = pair_features(a, b, ra, rb, a[..., :3], b[..., :3], ~mask)
    assert torch.count_nonzero(r) == 0
    assert torch.count_nonzero(aa[..., -4:]) == torch.count_nonzero(bb[..., -4:]) == 0


def test_original_default_roles_and_full_dyadic_loss_all_gradients():
    torch.manual_seed(1729)
    motion = OriginalMIMEMotion()
    assert motion.config == MIMEConfig()
    assert motion.projections[0].weight is not motion.projections[1].weight
    model = MIMERetrieval(motion, language(512))
    tokens = model.language.tokenize(
        ("first software sentence", "second test sentence", "third fixture")
    )
    x, roots = torch.randn(2, 2, 13, 135), torch.randn(2, 2, 13, 3)
    mask = torch.ones(2, 2, 13, dtype=torch.bool)
    positive = torch.tensor([[True, True, False], [False, False, True]])
    loss = model.compute_loss_dyads(x, roots, mask, tokens, positive)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        assert parameter.grad.abs().sum() > 0, name
    with pytest.raises(ValueError, match="135"):
        motion(x[..., :66], roots, mask)
    # Enforce rejection at the integrated original scoring/loss boundary even
    # if a caller supplies tokens from the group-compatible language factory.
    long_tokens = model.language.tokenize((" ".join("word" for _ in range(180)),))
    assert long_tokens.segment_counts == (3,)
    with pytest.raises(ValueError, match="TEXT_CONTEXT_LIMIT"):
        model.score_dyads(x, roots, mask, long_tokens)
    with pytest.raises(ValueError, match="TEXT_CONTEXT_LIMIT"):
        model.compute_loss_dyads(x, roots, mask, long_tokens, torch.ones(2, 1, dtype=torch.bool))


def test_group_all_permutations_full_loss_and_input_gradients():
    torch.manual_seed(1729)
    model = MIMERetrieval(MIMESetMotion(config()), language()).eval()
    parent = capture(grad=True)
    other = capture()
    tokens = model.language.tokenize(("one software fixture", "two fixture", "three"))
    positive = torch.tensor([[True, True, False], [False, False, True]])
    score = model.score((parent, other), tokens).detach()
    expected = model.compute_loss((parent, other), tokens, positive)
    expected.backward()
    gradients = [window.features.grad.clone() for window in parent.windows]
    for order in itertools.permutations(range(3)):
        moved = reordered(parent, order)
        assert torch.equal(model.score((moved, other), tokens).detach(), score)
        actual = model.compute_loss((moved, other), tokens, positive)
        assert torch.equal(actual, expected)
        for window in moved.windows:
            window.features.retain_grad()
        actual.backward()
        for window, grad in zip(moved.windows, gradients, strict=True):
            assert torch.equal(window.features.grad, grad[list(order)])


def test_group_pair_endpoint_swap_is_structurally_symmetric():
    torch.manual_seed(1729)
    model = MIMESetMotion(config()).eval()
    window = capture().windows[0]
    x, observed = window.features, window.observed
    delta = model.root_delta(x, observed)
    ab = model.pair(x[0:1], x[1:2], observed[0:1], observed[1:2], delta[0:1], delta[1:2])
    ba = model.pair(x[1:2], x[0:1], observed[1:2], observed[0:1], delta[1:2], delta[0:1])
    assert torch.equal(ab, ba)


def test_checkpoint_edge_and_trainable_language_loss_all_gradients_rng():
    torch.manual_seed(1729)
    direct = MIMERetrieval(
        MIMESetMotion(config(0.1), checkpoint_edges=False), language(checkpoint=False)
    )
    replay = MIMERetrieval(
        MIMESetMotion(config(0.1), checkpoint_edges=True), language(checkpoint=True)
    )
    replay.load_state_dict(direct.state_dict())
    parents = (capture(), capture())
    tokens = direct.language.tokenize(("one two three", "four", "five six"))
    positive = torch.tensor([[True, True, False], [False, False, True]])
    values = []
    for model in (direct, replay):
        torch.manual_seed(31415)
        loss = model.compute_loss(parents, tokens, positive)
        loss.backward()
        values.append(
            (loss, torch.get_rng_state(), {n: p.grad.clone() for n, p in model.named_parameters()})
        )
    assert torch.equal(values[0][0], values[1][0]) and torch.equal(values[0][1], values[1][1])
    for name in values[0][2]:
        assert torch.equal(values[0][2][name], values[1][2][name]), name


def test_group_default_complete_objective_all_trainable_gradients():
    torch.manual_seed(1729)
    model = MIMERetrieval(MIMESetMotion(), language(512))
    tokens = model.language.tokenize(("one software sentence", "two words", "three fixture"))
    positive = torch.tensor([[True, True, False], [False, False, True]])
    model.compute_loss((capture(), capture()), tokens, positive).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_all_windows_absolute_gaps_and_dynamic_k_without_ordinal_bank():
    torch.manual_seed(1729)
    model = MIMESetMotion(config()).eval()
    parent = capture(k=8, length=1)
    score = model(parent)
    short = TMRGroupCapture((parent.windows[0],))
    gap = TMRGroupCapture((parent.windows[0], replace(parent.windows[1], start_seconds=400)))
    assert not torch.equal(score, model(short)) and not torch.equal(score, model(gap))
    late = TMRGroupCapture(
        (parent.windows[0], replace(parent.windows[1], features=parent.windows[1].features + 1))
    )
    assert not torch.equal(score, model(late))
    assert not any(isinstance(module, nn.Embedding) for module in model.modules())


def test_missing_root_steps_do_not_bridge_unobserved_frames():
    x = torch.arange(6, dtype=torch.float32)[None, :, None].expand(2, 6, 66)
    observed = torch.ones_like(x, dtype=torch.bool)
    observed[:, 2] = False
    delta = MIMESetMotion.root_delta(x.masked_fill(~observed, 0), observed)
    assert torch.count_nonzero(delta[:, [0, 2, 3]]) == 0
    assert torch.equal(delta[:, [1, 4, 5]], torch.ones(2, 3, 3))


def test_complete_caption_segments_keep_every_content_token_and_original_rejects():
    module = language()
    short, long = "first short", " ".join("word" for _ in range(180))
    tokens = module.tokenize((short, long))
    assert tokens.segment_counts == (1, 3)
    content = []
    for ids, mask in zip(tokens.input_ids[1:], tokens.attention_mask[1:], strict=True):
        valid = ids[mask]
        assert valid[0] == 49406 and valid[-1] == 49407
        content.extend(valid[1:-1].tolist())
    assert content == [i % 127 for i in range(180)]
    with pytest.raises(ValueError, match="TEXT_CONTEXT_LIMIT"):
        language(group=False).tokenize((long,))
    output = module(tokens)
    assert output.shape == (2, 16)
    output.square().mean().backward()
    assert all(
        p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        for p in module.tower.parameters()
    )
    with pytest.raises(ValueError, match="exactly cover"):
        MIMETextTokens(tokens.input_ids, tokens.attention_mask, (1, 1))
