"""Analytic cache IO checks, not proof of actual pretrained assets."""

import copy

import numpy as np
import pytest
import torch

from phaseset_core.tmr_feature_cache import FrozenTMRRowCache, save_frozen_tmr_rows
from phaseset_core.tmr_set import TMRSet, TMRTextBatch
from phaseset_core.training import _capture_rng, _stable_hash


def fixture():
    torch.manual_seed(1729)
    tokens = torch.randn(3, 8, 768)
    mask = torch.arange(8)[None] < torch.tensor([3, 8, 5])[:, None]
    sentences = torch.nn.functional.normalize(torch.randn(3, 768), dim=1)
    return ("exact text", "中文  unchanged whitespace ", "a different row"), TMRTextBatch(
        tokens, mask, sentences
    )


def test_complete_rows_padding_reordering_duplicates_cls_and_rng(tmp_path):
    captions, text = fixture()
    state = _stable_hash(_capture_rng())
    records = save_frozen_tmr_rows(tmp_path / "rows", captions, text)
    cache = FrozenTMRRowCache(tmp_path / "rows", records)
    order = [2, 0, 1, 0]
    actual = cache.tmr_text(tuple(captions[index] for index in order))
    for i, j in enumerate(order):
        valid = text.token_mask[j]
        assert torch.equal(actual.tokens[i, : int(valid.sum())], text.tokens[j, valid])
        assert torch.equal(actual.sentences[i], text.sentences[j])
        assert torch.equal(actual.token_mask[i], text.token_mask[j])
        assert (actual.tokens[i, ~actual.token_mask[i]].view(torch.int32) == 0).all()
    assert torch.equal(cache.wamo_cls(tuple(captions[j] for j in order)), text.tokens[order, 0])
    assert _stable_hash(_capture_rng()) == state
    assert len(records) == 3 and len(actual.tokens) == 4


def test_closed_files_and_unknown_human_rows_are_not_silently_replaced(tmp_path):
    captions, text = fixture()
    records = save_frozen_tmr_rows(tmp_path, captions, text)
    cache = FrozenTMRRowCache(tmp_path, records)
    with pytest.raises(ValueError, match="outside"):
        cache.tmr_text(("new machine-fused row",))
    with pytest.raises(FileExistsError):
        save_frozen_tmr_rows(tmp_path, captions, text)
    path = tmp_path / f"{records[0]['caption_utf8_sha256']}.npz"
    with path.open("wb") as stream:
        np.savez(stream, tokens=np.zeros((3, 768), np.float32), sentence=np.zeros(768, np.float32))
    with pytest.raises(ValueError, match="differs"):
        cache.wamo_cls(captions)


def test_removing_only_right_padding_preserves_tmr_latents_and_all_text_gradients(tmp_path):
    captions, text = fixture()
    records = save_frozen_tmr_rows(tmp_path, captions, text)
    cached = FrozenTMRRowCache(tmp_path, records).tmr_text(captions)
    model = TMRSet(latent_dim=16, ff_size=32, num_layers=1, num_heads=4, dropout=0.0).eval()
    oracle = copy.deepcopy(model)
    actual = model.text_distributions(cached)
    expected = oracle.text_distributions(text)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in expected).backward()
    for (name, parameter), (_, reference) in zip(
        model.text_encoder.named_parameters(), oracle.text_encoder.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and reference.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-5, atol=1e-6, msg=name)


def test_no_content_or_masked_interior_tokens_can_be_removed(tmp_path):
    captions, text = fixture()
    mask = text.token_mask.clone()
    mask[0, 0], mask[0, 3] = False, True
    with pytest.raises(ValueError, match="right padding"):
        save_frozen_tmr_rows(tmp_path, captions, TMRTextBatch(text.tokens, mask, text.sentences))
    with pytest.raises(ValueError, match="unique"):
        save_frozen_tmr_rows(tmp_path, (captions[0],) * 3, text)
    records = save_frozen_tmr_rows(tmp_path / "good", captions, text)
    with pytest.raises(ValueError, match="unique"):
        FrozenTMRRowCache(tmp_path / "good", records + records[:1])
    changed = copy.deepcopy(records)
    changed[0]["token_count"] = 2
    with pytest.raises(ValueError, match="headers"):
        FrozenTMRRowCache(tmp_path / "good", changed).tmr_text(captions)
