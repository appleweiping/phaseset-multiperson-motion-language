"""Verified offline CLIP text fine-tuning for MIME, with explicit long text.

The frozen CLIP adapter remains unchanged. This is an independent trainable
tower, never reuses/mutates its opaque frozen instance or embedding receipts.
Original dyadic task rejects over-77 tokens; group adaptation preserves all BPE
content via 75-content-token segments and equal segment pooling per caption.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .frozen_clip_text import PINNED_FILES, TOKEN_BOS_ID, TOKEN_EOS_ID, TOKEN_LENGTH


@dataclass(frozen=True)
class MIMETextTokens:
    input_ids: Tensor  # [segments,77], CPU integers (not embeddings)
    attention_mask: Tensor
    segment_counts: tuple[int, ...]  # one positive count per complete caption

    def __post_init__(self):
        ids = self.input_ids
        if ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[1] != TOKEN_LENGTH:
            raise ValueError("MIME CLIP needs integer [segments,77] tokens")
        if self.attention_mask.dtype != torch.bool or self.attention_mask.shape != ids.shape:
            raise ValueError("token masks must match every segment")
        if not self.segment_counts or any(type(n) is not int or n < 1 for n in self.segment_counts):
            raise ValueError("each caption needs at least one segment")
        if sum(self.segment_counts) != len(ids) or not bool(self.attention_mask.any(1).all()):
            raise ValueError("segments must exactly cover every caption")


class TrainableMIMECLIP(nn.Module):
    def __init__(
        self,
        tower: nn.Module,
        tokenizer,
        *,
        width: int = 512,
        group_long_text: bool = True,
        checkpoint_segments: bool = True,
    ):
        super().__init__()
        self.tower = tower
        self.tokenizer = tokenizer
        self.projection = nn.Linear(512, width)
        self.group_long_text = group_long_text
        self.checkpoint_segments = checkpoint_segments
        self.tower.requires_grad_(True)

    @classmethod
    def from_private_snapshot(
        cls,
        path: Path,
        *,
        width: int = 512,
        group_long_text: bool = True,
        checkpoint_segments: bool = True,
    ):
        # Verify existing official fixed assets before deserializing; never fetch.
        for name, size, digest in PINNED_FILES:
            entry = path / name
            if entry.stat().st_size != size:
                raise ValueError("CLIP asset size mismatch")
            actual = hashlib.sha256()
            with entry.open("rb") as handle:
                while block := handle.read(8 * 1024**2):
                    actual.update(block)
            if actual.hexdigest() != digest:
                raise ValueError("CLIP asset digest mismatch")
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

        tokenizer = CLIPTokenizerFast.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            source, info = CLIPTextModelWithProjection.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                output_loading_info=True,
                use_safetensors=False,
            )
        if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
            raise ValueError("CLIP text initialization is incomplete")
        # Vision/logit keys exist in the fixed full CLIP checkpoint but no vision
        # model is instantiated. Drop pretrained projection: MIME learns its own.
        if any(
            not (
                key.startswith("vision_model.")
                or key in ("visual_projection.weight", "logit_scale")
            )
            for key in info["unexpected_keys"]
        ):
            raise ValueError("unexpected CLIP checkpoint key")
        if source.config.hidden_size != 512 or source.config.max_position_embeddings != 77:
            raise ValueError("wrong CLIP text configuration")
        if tokenizer.bos_token_id != TOKEN_BOS_ID or tokenizer.eos_token_id != TOKEN_EOS_ID:
            raise ValueError("wrong CLIP tokenizer")
        return cls(
            source.text_model,
            tokenizer,
            width=width,
            group_long_text=group_long_text,
            checkpoint_segments=checkpoint_segments,
        )

    def tokenize(self, captions: tuple[str, ...]) -> MIMETextTokens:
        if not captions or any(type(c) is not str or not c.strip() for c in captions):
            raise ValueError("MIME requires real nonempty caption rows")
        raw = self.tokenizer(list(captions), add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        segments, masks, counts = [], [], []
        for content in raw:
            if len(content) > 75 and not self.group_long_text:
                raise ValueError("TEXT_CONTEXT_LIMIT: original MIME caption exceeds CLIP context")
            pieces = [content[start : start + 75] for start in range(0, len(content), 75)] or [[]]
            counts.append(len(pieces))
            for piece in pieces:
                values = [TOKEN_BOS_ID, *piece, TOKEN_EOS_ID]
                segments.append(values + [TOKEN_EOS_ID] * (77 - len(values)))
                masks.append([True] * len(values) + [False] * (77 - len(values)))
        return MIMETextTokens(
            torch.tensor(segments, dtype=torch.int64),
            torch.tensor(masks, dtype=torch.bool),
            tuple(counts),
        )

    def forward(self, tokens: MIMETextTokens) -> Tensor:
        witness = self.tower.embeddings.token_embedding.weight
        rows = []
        for index in range(len(tokens.input_ids)):
            ids = tokens.input_ids[index : index + 1].to(witness.device)
            mask = tokens.attention_mask[index : index + 1].to(witness.device)

            def encode(parameter, ids=ids, mask=mask):
                # The extracted CLIPTextTransformer returns a pooled ModelOutput
                # directly; unlike the outer pretrained model it has no
                # return_dict keyword in the frozen Transformers runtime.
                return self.tower(input_ids=ids, attention_mask=mask).pooler_output[0]

            rows.append(
                checkpoint(encode, witness, use_reentrant=False, preserve_rng_state=True)
                if self.checkpoint_segments and torch.is_grad_enabled()
                else encode(witness)
            )
        pooled, start = [], 0
        for count in tokens.segment_counts:
            pooled.append(
                torch.stack(rows[start : start + count]).to(torch.float64).mean(0).float()
            )
            start += count
        return self.projection(torch.stack(pooled))
