"""Private frozen-language rows for complete-parent TMR/WaMo training.

This stores model-derived text features, not a new annotation or retrieval
label. Every human caption occurrence remains in the batch; identical text
may reuse frozen features. Files/manifest stay private. Cache generation uses
the admitted actual frozen language tower, never this module as a mock encoder.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .tmr_set import TMRTextBatch


def caption_key(caption: str) -> str:
    if type(caption) is not str or not caption.strip():
        raise ValueError("cache rows require exact nonempty caption strings")
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_frozen_tmr_rows(
    directory: str | Path, captions: tuple[str, ...], text: TMRTextBatch
) -> tuple[dict, ...]:
    """Write each unique input row once, stripping only masked right padding.

    No content token, CLS/SEP, or sentence feature is removed. The caller
    pre-admits a unique-text preparation batch but retains all annotation
    occurrences and their positive family lineage in the task manifest.
    """
    if type(text) is not TMRTextBatch or len(captions) != len(text.tokens):
        raise ValueError("all captions must have their actual frozen TMR features")
    keys = tuple(caption_key(row) for row in captions)
    if len(set(keys)) != len(keys):
        raise ValueError("preparation writes each exact unique text once")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, key in enumerate(keys):
        mask = text.token_mask[index].detach().cpu()
        length = int(mask.sum())
        if not bool(mask[:length].all()) or bool(mask[length:].any()):
            raise ValueError("the admitted frozen tokenizer must use contiguous right padding")
        path = root / f"{key}.npz"
        tokens = text.tokens[index, :length].detach().cpu().contiguous().numpy()
        sentence = text.sentences[index].detach().cpu().contiguous().numpy()
        with path.open("xb") as stream:
            np.savez(stream, tokens=tokens, sentence=sentence)
        records.append(
            dict(
                caption_utf8_sha256=key,
                file_sha256=file_sha256(path),
                file_bytes=path.stat().st_size,
                token_count=length,
            )
        )
    return tuple(records)


class FrozenTMRRowCache:
    """Load exact rows in caller order, with no batch/caption multiplicity loss.

    ``records`` comes from the closed private cache manifest bound to actual
    source, frozen model, code, precision and runtime receipts. An arbitrary
    matching file is not official provenance or permission. No final-test
    discovery, learned statistic or row generation occurs here.
    """

    def __init__(self, directory: str | Path, records: tuple[Mapping, ...]):
        self.directory = Path(directory)
        self.records = {row["caption_utf8_sha256"]: dict(row) for row in records}
        if not records or len(self.records) != len(records):
            raise ValueError("cache manifest needs each unique exact text once")

    def _row(self, caption):
        key = caption_key(caption)
        if key not in self.records:
            raise ValueError("caption is outside the admitted frozen human feature cache")
        record, path = self.records[key], self.directory / f"{key}.npz"
        if (
            file_sha256(path) != record["file_sha256"]
            or path.stat().st_size != record["file_bytes"]
        ):
            raise ValueError("private frozen language artifact differs from its closed manifest")
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {"tokens", "sentence"}:
                raise ValueError("frozen language artifact has an unexpected array roster")
            tokens, sentence = values["tokens"], values["sentence"]
        if (
            tokens.dtype != np.float32
            or tokens.shape != (record["token_count"], 768)
            or sentence.dtype != np.float32
            or sentence.shape != (768,)
            or not 1 <= len(tokens) <= 512
            or not np.isfinite(tokens).all()
            or not np.isfinite(sentence).all()
        ):
            raise ValueError("private frozen language arrays have invalid headers or values")
        return torch.from_numpy(tokens), torch.from_numpy(sentence)

    def tmr_text(self, captions: tuple[str, ...]) -> TMRTextBatch:
        rows = [self._row(caption) for caption in captions]
        if not rows:
            raise ValueError("frozen text batch cannot be empty")
        length = max(len(tokens) for tokens, _ in rows)
        features = torch.zeros(len(rows), length, 768, dtype=torch.float32)
        mask = torch.zeros(len(rows), length, dtype=torch.bool)
        for index, (tokens, _) in enumerate(rows):
            features[index, : len(tokens)] = tokens
            mask[index, : len(tokens)] = True
        return TMRTextBatch(features, mask, torch.stack([sentence for _, sentence in rows]))

    def wamo_cls(self, captions: tuple[str, ...]) -> torch.Tensor:
        # The actual frozen DistilBERT CLS, not MPNet/CLIP or the learned TMR projection.
        return torch.stack([self._row(caption)[0][0] for caption in captions]).contiguous()
