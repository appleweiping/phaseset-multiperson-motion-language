"""Complete-capture V2 retrieval against admitted, frozen CLIP sentence rows.

This is a trainable score/loss seam, not a formal-attempt scheduler. The caller
admits split, rights, caption provenance, population floors and the frozen B2
checkpoint. It must not label machine-fused text or unverified counterfactuals
as human evidence. No final-test or data discovery happens inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .capture_validation import _validate_text_batch
from .continuous_training_input import ContinuousTrainingView
from .dct_calibration import DctFloorReceipt
from .dct_relations import DctRelationField, DctViewContext, require_dct_working_budget
from .directional_phase import DirectionalPhaseField
from .frozen_clip_text import FrozenClipTextBatch
from .pipeline import collate_group_samples, edge_budget_batches
from .temporal_coordination import (
    CalibratedCoordinationScore,
    TemporalIncidenceEncoder,
    coordination_objective,
)
from .training import BaseRetrievalSystem


def human_holistic_captions(annotation: dict) -> tuple[str, ...]:
    """Use every official ``scene_explained`` row, without machine fusion.

    Other parallel questionnaire fields describe mood/atmosphere/relationships;
    they are not silently concatenated into motion ground truth. This selects
    a schema field, not proof that an assertion is skeleton-observable. Rights,
    official-file provenance and the two-human relation audit remain external.
    """
    if type(annotation) is not dict:
        raise ValueError("official holistic annotation must be a JSON object")
    rows = annotation.get("scene_explained")
    if (
        type(rows) is not list
        or not rows
        or any(type(row) is not str or not row.strip() for row in rows)
    ):
        raise ValueError("scene_explained must contain nonempty human sentence rows")
    return tuple(rows)


def capture_family_positive_mask(
    motion_keys: tuple[str, ...], text_keys: tuple[str, ...], *, device: torch.device
) -> Tensor:
    """Variable positives use admitted target families, never a repeated actor set.

    Released segments can share one parent holistic annotation. These siblings
    must not become false negatives. Source identity remains separate; the host
    binds parent/annotation-family membership from real provenance.
    """
    for keys in (motion_keys, text_keys):
        if not keys or any(
            type(key) is not str or len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
            for key in keys
        ):
            raise ValueError("positive family keys must be nonempty SHA-256 tuples")
    mask = torch.tensor(
        [[motion == caption for caption in text_keys] for motion in motion_keys],
        dtype=torch.bool,
        device=device,
    )
    if not bool(mask.any(dim=0).all() and mask.any(dim=1).all()):
        raise ValueError("every motion and caption row needs a positive target family")
    return mask.contiguous()


@dataclass(frozen=True)
class ContinuousScoreOutput:
    scores: Tensor
    global_cosine: Tensor
    coordination_cosine: Tensor
    periodic_support: Tensor
    text_receipt_sha256: str
    capture_keys: tuple[str, ...]


@dataclass(frozen=True)
class ContinuousRetrievalOutput:
    scores: Tensor
    positive_mask: Tensor
    global_cosine: Tensor
    coordination_cosine: Tensor
    periodic_support: Tensor
    text_receipt_sha256: str
    capture_keys: tuple[str, ...]


class ContinuousRetrievalSystem(nn.Module):
    """Fixed B2 global anchor plus full ordered, directed-packet coordination.

    B2 consumes every accepted ten-second view in the capture's common frame.
    Its fixed window mean is only the legacy global branch; the coordination
    branch preserves the complete absolute timeline, gaps and two-layer state.
    The new text adapter is shared, 512->512->512 with GELU. The frozen CLIP
    tower and B2 remain non-trainable; no base logit temperature is reused.
    """

    def __init__(
        self,
        frozen_b2: BaseRetrievalSystem,
        *,
        use_topology: bool = True,
        strip_phase: bool = False,
        relation_kind: str = "phase",
        dct_floor_receipt: DctFloorReceipt | None = None,
        checkpoint_blocks: bool = True,
        base_window_batch_size: int = 1,
        base_edge_budget: int = 32768,
    ) -> None:
        super().__init__()
        if (
            not isinstance(frozen_b2, BaseRetrievalSystem)
            or frozen_b2.system_id != "B2"
            or frozen_b2.embedding_dim != 512
        ):
            raise ValueError("V2 requires the fixed 512D B2 anchor, not a selected legacy winner")
        if type(base_window_batch_size) is not int or base_window_batch_size < 1:
            raise ValueError("base_window_batch_size must be a positive integer")
        if type(base_edge_budget) is not int or base_edge_budget < 1:
            raise ValueError("base_edge_budget must be a positive integer")
        if relation_kind not in ("phase", "true_mean_difference_dct"):
            raise ValueError("relation_kind must be phase or true_mean_difference_dct")
        if relation_kind == "phase":
            if dct_floor_receipt is not None:
                raise ValueError("DCT receipt must not enter the phase system")
            self.register_buffer("dct_energy_floors", None)
            self._dct_floor_receipt = None
            self._dct_receipt_sha256 = None
        else:
            if strip_phase or type(dct_floor_receipt) is not DctFloorReceipt:
                raise ValueError("A6 requires a typed independent DCT floor receipt")
            # Persistent state binds the physical calibration to every model
            # checkpoint; a changed external array cannot alter later scoring.
            self.register_buffer(
                "dct_energy_floors", torch.from_numpy(dct_floor_receipt.floors.copy())
            )
            self._dct_floor_receipt = dct_floor_receipt
            self._dct_receipt_sha256 = dct_floor_receipt.sha256
        self.relation_kind = relation_kind
        self.frozen_b2 = frozen_b2.requires_grad_(False).eval()
        self.coordination = TemporalIncidenceEncoder(
            width=512,
            use_topology=use_topology,
            strip_phase=strip_phase,
            checkpoint_blocks=checkpoint_blocks,
        )
        self.text_adapter = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512))
        self.calibration = CalibratedCoordinationScore()
        self.base_window_batch_size = base_window_batch_size
        self.base_edge_budget = base_edge_budget

    def get_extra_state(self) -> Tensor:
        """Tensor-encode the relation identity for tensor-only checkpoint hosts."""
        payload = {
            "schema": "phaseset-v2-relation-state-v1",
            "relation_kind": self.relation_kind,
            "dct_floor_receipt_sha256": self._dct_receipt_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return torch.tensor(list(encoded), dtype=torch.uint8)

    def set_extra_state(self, state: object) -> None:
        if (
            type(state) is not Tensor
            or state.dtype != torch.uint8
            or state.ndim != 1
            or not torch.equal(state.detach().cpu(), self.get_extra_state())
        ):
            raise ValueError("relation kind or DCT floor receipt changed across checkpoint resume")
        self._check_dct_floor_binding()

    def _check_dct_floor_binding(self) -> None:
        if self.relation_kind == "phase":
            return
        if (
            self._dct_floor_receipt.sha256 != self._dct_receipt_sha256
            or not torch.equal(
                self.dct_energy_floors.detach().cpu(),
                torch.from_numpy(self._dct_floor_receipt.floors.copy()),
            )
        ):
            raise ValueError("DCT calibration buffer or receipt changed after construction")

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_b2.eval()
        return self

    @torch.no_grad()
    def encode_global(self, view: ContinuousTrainingView) -> Tensor:
        """Use all accepted windows, with fixed reduction order across microbatches."""
        self.frozen_b2.eval()
        windows = view.capture.windows()
        batches = edge_budget_batches(
            windows,
            max_total_edges=self.base_edge_budget,
            max_batch_size=self.base_window_batch_size,
        )
        embeddings = [
            self.frozen_b2.encode_trainable(collate_group_samples(batch)) for batch in batches
        ]
        # edge_budget_batches sorts this single capture in absolute start order.
        rows = torch.cat(embeddings)
        if len(rows) != len(windows):
            raise RuntimeError("complete global capture encoding omitted a window")
        return F.normalize(rows.to(torch.float64).mean(dim=0).to(torch.float32), dim=0)

    def score(
        self,
        views: tuple[ContinuousTrainingView, ...],
        text_batch: FrozenClipTextBatch,
    ) -> ContinuousScoreOutput:
        """Score arbitrary gallery/CF sentences without assigning positive labels.

        This also supports gallery text batches whose matching motion rows are
        elsewhere. Counterfactual text must never be given fabricated positive
        capture keys merely to pass the contrastive training interface.
        """
        if (
            type(views) is not tuple
            or not views
            or any(type(view) is not ContinuousTrainingView for view in views)
        ):
            raise ValueError("views must be a nonempty tuple of admitted continuous inputs")
        text, _, receipt_sha = _validate_text_batch(text_batch, allow_row_selection=True)
        self._check_dct_floor_binding()
        device = next(self.coordination.parameters()).device
        keys = tuple(view.positive_capture_key for view in views)
        text = text.to(device)
        adapted_text = self.text_adapter(text)
        base_rows, relation_rows, supports = [], [], []
        for view in views:
            if view.phase_field.actor_count != view.capture.actor_count or (
                view.phase_field.intervals[-1][1] != view.capture.frame_count
            ):
                raise ValueError("the coordination field must span the complete capture")
            if self.relation_kind != "phase":
                if type(view.phase_field) is not DctViewContext:
                    raise ValueError("A6 retrieval requires the DCT-only context")
                if view.phase_field.dct_floor_receipt_sha256 != self._dct_receipt_sha256:
                    raise ValueError("A6 view DCT floor receipt differs from the model")
                require_dct_working_budget(
                    view.capture.actor_count, view.capture.frame_count, view.phase_field.config
                )
            base_rows.append(self.encode_global(view))
            if self.relation_kind == "phase":
                if type(view.phase_field) is not DirectionalPhaseField:
                    raise ValueError("phase retrieval requires the complete Morlet field")
                physical = view.phase_field
            else:
                physical = DctRelationField.from_capture(
                    view.capture,
                    view.phase_field,
                    dct_floor_receipt=self._dct_floor_receipt,
                )
            output = self.coordination.score_text(physical, adapted_text)
            relation_rows.append(output.cosine)
            supports.append(output.periodic_support)
        global_cosine = F.normalize(torch.stack(base_rows), dim=-1) @ F.normalize(text, dim=-1).T
        relation_cosine = torch.stack(relation_rows)
        support = torch.stack(supports)
        scores = self.calibration(global_cosine, relation_cosine, coordination_support=support)
        return ContinuousScoreOutput(
            scores, global_cosine, relation_cosine, support, receipt_sha, keys
        )

    def forward(
        self,
        views: tuple[ContinuousTrainingView, ...],
        text_batch: FrozenClipTextBatch,
        *,
        motion_positive_keys: tuple[str, ...],
        text_positive_keys: tuple[str, ...],
    ) -> ContinuousRetrievalOutput:
        output = self.score(views, text_batch)
        if len(set(output.capture_keys)) != len(output.capture_keys):
            raise ValueError("training views must be distinct source captures")
        if (
            type(motion_positive_keys) is not tuple
            or len(motion_positive_keys) != output.scores.shape[0]
            or type(text_positive_keys) is not tuple
            or len(text_positive_keys) != output.scores.shape[1]
        ):
            raise ValueError("admitted positive family keys must cover every motion/text row")
        positives = capture_family_positive_mask(
            motion_positive_keys, text_positive_keys, device=output.scores.device
        )
        return ContinuousRetrievalOutput(
            output.scores,
            positives,
            output.global_cosine,
            output.coordination_cosine,
            output.periodic_support,
            output.text_receipt_sha256,
            output.capture_keys,
        )

    @staticmethod
    def objective(
        output: ContinuousRetrievalOutput,
        *,
        cf_positive_scores: Tensor,
        cf_negative_scores: Tensor,
        verified_negative_mask: Tensor,
        cf_weight: float = 0.2,
        margin: float = 0.2,
    ) -> Tensor:
        """No implicit false labels; empty explicit CF tensors are only no-CF use.

        The full-training host must supply genuinely verified-false relation
        examples with their provenance. This function cannot certify their
        human truth. Margin/weight/sampling are frozen by the bounded pilot.
        """
        return coordination_objective(
            output.scores,
            output.positive_mask,
            cf_positive_scores=cf_positive_scores,
            cf_negative_scores=cf_negative_scores,
            verified_negative_mask=verified_negative_mask,
            cf_weight=cf_weight,
            margin=margin,
        )
