"""Software seams only; real pinned language outputs are qualified privately."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from phaseset_core.tmr_language import (
    FrozenTMRLanguage,
    tokenize_without_truncation,
    verify_asset_directory,
)


class AnalyticTokenizer:
    model_max_length = 8

    def __call__(self, text, **kwargs):
        assert kwargs == dict(return_tensors="pt", padding=True, truncation=False)
        lengths = [len(item.split()) + 2 for item in text]
        mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
        return {"input_ids": mask.to(torch.int64), "attention_mask": mask.to(torch.int64)}


class AnalyticLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(768).float())
        self.dropout = torch.nn.Dropout(0.9)
        self.config = SimpleNamespace(max_position_embeddings=10)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=self.dropout(input_ids[..., None] + self.weight[None, None])
        )


def test_actual_asset_roster_and_digest_are_required(tmp_path):
    import hashlib

    path = tmp_path / "config.json"
    path.write_bytes(b"analytic software fixture, never a language model")
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_asset_directory(tmp_path, expected)
    with pytest.raises(ValueError, match="digest"):
        verify_asset_directory(tmp_path, {path.name: "0" * 64})
    with pytest.raises(ValueError, match="roster"):
        verify_asset_directory(tmp_path, {})
    with pytest.raises(ValueError, match="missing"):
        verify_asset_directory(Path(tmp_path / "missing"), expected)


def test_frozen_language_cannot_train_and_preserves_rows_masks_and_batching():
    language = FrozenTMRLanguage(
        AnalyticTokenizer(), AnalyticLanguage(), AnalyticTokenizer(), AnalyticLanguage()
    ).train(True)
    assert not any(module.training for module in language.modules())
    assert not any(parameter.requires_grad for parameter in language.parameters())
    text = ("group walks", "one follows another", "all turn")
    a, b = language(text, microbatch=1), language(text, microbatch=16)
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.sentences, b.sentences)
    assert torch.equal(a.token_mask.sum(1), torch.tensor([4, 5, 4]))
    torch.testing.assert_close(a.sentences.norm(dim=1), torch.ones(3))
    assert not a.tokens.requires_grad and not a.sentences.requires_grad


def test_never_quietly_truncate_or_encode_empty_caption():
    tokenizer, model = AnalyticTokenizer(), AnalyticLanguage()
    with pytest.raises(ValueError, match="never truncate"):
        tokenize_without_truncation(tokenizer, model, ("word " * 8,))
    with pytest.raises(ValueError, match="nonempty"):
        tokenize_without_truncation(tokenizer, model, (" ",))
