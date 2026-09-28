"""Pinned, frozen, local-only language assets used by the TMR adaptation.

Token features use DistilBERT; semantic negative filtering uses a separate
masked-mean, normalized MPNet representation, as in the upstream TextToEmb.
No CLIP substitution, remote code, network download, or quiet truncation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .tmr_set import TMRTextBatch

DISTILBERT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MPNET_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
# Public checkpoint/file digests independently obtained from the official Hub.
# Model bytes remain in the caller's private asset directories, never the wheel.
DISTILBERT_FILES = {
    "LICENSE": "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
    "README.md": "b925a22d8efd7a29608a0e10d3f2fb33d5d80df7de4ac6e34093853cd8e0274e",
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
MPNET_FILES = {
    "README.md": "89a1a9c3290fe58e76c939b578c48a14331dc7bfcaaf5a53102adb183da6f96a",
    "config.json": "d46a3e04ded82bba22528424480697d394eeda6a27484e08c5bb2bdf5906cfa0",
    "model.safetensors": "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
    "special_tokens_map.json": "9ef40e9c160511bf3f46ceb71f1471dafa1e9473d5120bb816c36b2efa75f8ba",
    "tokenizer.json": "b8be2c30ba5dd723a6d5ee26d013da103d5408d92ddcb23747622f9e48f1d842",
    "tokenizer_config.json": "67f2ff7e223518e729869bb3a70f0caf8368fe549383fc11cfe2dfb42fffc268",
    "vocab.txt": "dbd90cb94e2247bd4d4ccaecbf616d2290e66691d7d5e5bb81f063c2d0649ada",
}


def verify_asset_directory(directory: Path, expected: dict[str, str]) -> None:
    """Validate the actual tokenizer/config/checkpoint bytes before loading."""
    if not directory.is_dir():
        raise ValueError("private pinned language directory is missing")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != set(expected):
        raise ValueError("language asset file roster differs from the pinned snapshot")
    for name, digest in expected.items():
        with (directory / name).open("rb") as stream:
            actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_digest != digest:
            raise ValueError(f"pinned language asset digest mismatch: {name}")


def tokenize_without_truncation(tokenizer, model, captions: tuple[str, ...]):
    if not captions or any(type(text) is not str or not text.strip() for text in captions):
        raise ValueError("language input needs nonempty, unmodified human text rows")
    encoded = tokenizer(list(captions), return_tensors="pt", padding=True, truncation=False)
    # MPNet has 514 positional entries but padding-offset positions leave only
    # 512 usable tokens; the pinned tokenizer supplies that actual input bound.
    bound = min(tokenizer.model_max_length, model.config.max_position_embeddings)
    if encoded["input_ids"].shape[1] > bound:
        raise ValueError("caption exceeds the pinned language token bound; never truncate")
    return encoded


class FrozenTMRLanguage(nn.Module):
    """Actual frozen DistilBERT tokens plus MPNet, with an explicit asset factory."""

    @classmethod
    def from_private_assets(cls, distilbert: Path, mpnet: Path, *, device="cpu"):
        verify_asset_directory(distilbert, DISTILBERT_FILES)
        verify_asset_directory(mpnet, MPNET_FILES)
        from transformers import AutoModel, AutoTokenizer

        def load(path):
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            model = AutoModel.from_pretrained(
                path, local_files_only=True, trust_remote_code=False, use_safetensors=True
            )
            if model.config.hidden_size != 768:
                raise ValueError("pinned TMR language width must remain 768")
            return tokenizer, model

        dt, dm = load(distilbert)
        mt, mm = load(mpnet)
        return cls(dt, dm, mt, mm).to(device)

    def __init__(self, distilbert_tokenizer, distilbert_model, mpnet_tokenizer, mpnet_model):
        super().__init__()
        # Direct construction is a test seam, not evidence of pinned assets.
        self.distilbert_tokenizer = distilbert_tokenizer
        self.distilbert_model = distilbert_model
        self.mpnet_tokenizer = mpnet_tokenizer
        self.mpnet_model = mpnet_model
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, captions: tuple[str, ...], *, microbatch: int = 16) -> TMRTextBatch:
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("language microbatch must be a positive integer")
        distilbert = tokenize_without_truncation(
            self.distilbert_tokenizer, self.distilbert_model, captions
        )
        mpnet = tokenize_without_truncation(self.mpnet_tokenizer, self.mpnet_model, captions)
        device = next(self.distilbert_model.parameters()).device
        tokens, sentences = [], []
        for start in range(0, len(captions), microbatch):
            di = {
                key: value[start : start + microbatch].to(device)
                for key, value in distilbert.items()
            }
            mi = {key: value[start : start + microbatch].to(device) for key, value in mpnet.items()}
            tokens.append(self.distilbert_model(**di).last_hidden_state)
            hidden = self.mpnet_model(**mi).last_hidden_state
            expanded = mi["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)
            sentences.append(F.normalize(pooled, p=2, dim=1))
        return TMRTextBatch(
            torch.cat(tokens).to(torch.float32),
            distilbert["attention_mask"].to(device=device, dtype=torch.bool),
            torch.cat(sentences).to(torch.float32),
        )
