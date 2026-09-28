"""Executable, data-free PyTorch training runtime for PhaseSet.

The runtime accepts host-provided :class:`PreparedGroupBatch` objects and
already-frozen text embeddings.  It deliberately knows nothing about dataset
paths, remote machines, credentials, or external execution receipts.  Base
qualification and frozen-base residual fitting share one deterministic
gradient-cache loop so that edge-budget microbatches still see the complete
effective-batch negative gallery.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from fractions import Fraction
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import random
import struct
import sys
import types
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable
import uuid

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phaseset_core import execution as execution_module

if TYPE_CHECKING:
    from .capture_validation import CaptureValidationSource
    from .periodic_capture_training_cache import PeriodicDescriptorCaptureTrainingPlan

from .contracts import PreparedGroupBatch, validate_prepared_group_batch
from .execution import (
    ExecutionContractError,
    ResumeRecord,
)
from .experiments import (
    SEEDS as EXPERIMENT_SEEDS,
    BaseQualification,
    canonical_base_qualification_bytes,
)
from .models import (
    ActorMeanBase,
    GroupBaseOutput,
    GroupTokenOutput,
    PhaseSetEncoder,
    SetPMABase,
    SocialTemporalBase,
    build_group_base,
)
from .objectives import PhaseSetRetrievalHead, variable_positive_symmetric_infonce
from .periodic_descriptor_cache_v2 import CachedPairChunkStream, DescriptorWindowContext


STATUS: Final = "DATA_FREE_EXECUTABLE_TRAINING_RUNTIME_NONPRODUCTION_AUTHORITY0"
CHECKPOINT_SCHEMA: Final = "phaseset-training-checkpoint-v1"
REPORT_SCHEMA: Final = "phaseset-training-report-v1"
CACHED_CHECKPOINT_SCHEMA: Final = "phaseset-training-checkpoint-v2-cached-descriptor-plan"
CACHED_INITIALIZATION_SCHEMA: Final = (
    "phaseset-training-initialization-binding-v2-cached-descriptor-plan"
)
CACHED_REPORT_SCHEMA: Final = "phaseset-training-report-v2-cached-descriptor-plan"
OFFICIAL_SEEDS: Final = (1729, 2718, 31415)
RESIDUAL_SYSTEM_IDS: Final = tuple(f"{index:02d}" for index in range(1, 9))
REGISTERED_BASE_SYSTEM_IDS: Final = ("B0", "B1", "B2")
REGISTERED_EMBEDDING_DIM: Final = 512
REGISTERED_HEADS: Final = 8
REGISTERED_FFN_DIM: Final = 2048
REGISTERED_DROPOUT: Final = 0.1
REGISTERED_PERIODIC_HIDDEN_DIM: Final = 256
REGISTERED_TEXT_HIDDEN_DIM: Final = 512
BASE_EPOCHS: Final = 30
RESIDUAL_EPOCHS: Final = 20
BASE_LEARNING_RATE: Final = 2e-4
RESIDUAL_LEARNING_RATE: Final = 3e-4
WEIGHT_DECAY: Final = 0.01
WARMUP_FRACTION: Final = 0.05
EFFECTIVE_GLOBAL_BATCH: Final = 128
GRADIENT_CLIP_NORM: Final = 1.0

_FORMAL_RUNTIME_CONSTANTS: Final = {
    "BASE_EPOCHS": 30,
    "BASE_LEARNING_RATE": 2e-4,
    "EFFECTIVE_GLOBAL_BATCH": 128,
    "GRADIENT_CLIP_NORM": 1.0,
    "REGISTERED_DROPOUT": 0.1,
    "REGISTERED_EMBEDDING_DIM": 512,
    "REGISTERED_FFN_DIM": 2048,
    "REGISTERED_HEADS": 8,
    "REGISTERED_PERIODIC_HIDDEN_DIM": 256,
    "REGISTERED_TEXT_HIDDEN_DIM": 512,
    "RESIDUAL_EPOCHS": 20,
    "RESIDUAL_LEARNING_RATE": 3e-4,
    "WARMUP_FRACTION": 0.05,
    "WEIGHT_DECAY": 0.01,
}

Stage = Literal["base", "residual"]
Split = Literal["train", "val"]


class TrainingRuntimeError(RuntimeError):
    """Raised when an executable training contract cannot be honored."""


class TrainingCheckpointError(TrainingRuntimeError):
    """Raised when an immutable checkpoint is missing, corrupt, or mismatched."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be one lowercase SHA-256 hex digest")
    return value


def _positive_id(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be exact built-in bytes[32]")
    return value


@dataclass(frozen=True, slots=True)
class RetrievalTrainingBatch:
    """One edge-budget microbatch with frozen text semantics.

    Positive-family identifiers are opaque commitments used only to construct
    the variable-positive mask.  They never enter either neural tower.  An
    optional descriptor context tuple preserves authenticated prepared-source
    lineage; its presence alone grants no cache, data, or execution authority.
    """

    groups: PreparedGroupBatch
    text_embeddings: Tensor
    motion_positive_ids: tuple[bytes, ...]
    text_positive_ids: tuple[bytes, ...]
    text_commitments: tuple[bytes, ...]
    split: Split
    descriptor_contexts: tuple[DescriptorWindowContext, ...] | None = None

    def __post_init__(self) -> None:
        checked = validate_prepared_group_batch(self.groups)
        if self.split not in ("train", "val"):
            raise TrainingRuntimeError("training batches may only use train or val; test is sealed")
        if type(self.text_embeddings) is not Tensor:
            raise TypeError("text_embeddings must be an exact torch.Tensor")
        if (
            self.text_embeddings.dtype != torch.float32
            or self.text_embeddings.ndim != 2
            or self.text_embeddings.shape[0] < 1
            or self.text_embeddings.shape[1] < 2
            or self.text_embeddings.device.type != "cpu"
            or self.text_embeddings.requires_grad
            or not self.text_embeddings.is_contiguous()
            or not bool(torch.isfinite(self.text_embeddings).all().item())
        ):
            raise TrainingRuntimeError(
                "frozen text_embeddings must be finite contiguous CPU float32 [Q,D]"
            )
        if (
            type(self.motion_positive_ids) is not tuple
            or len(self.motion_positive_ids) != checked.batch_size
        ):
            raise TypeError("motion_positive_ids must contain one bytes[32] per motion")
        if type(self.text_positive_ids) is not tuple or len(self.text_positive_ids) != int(
            self.text_embeddings.shape[0]
        ):
            raise TypeError("text_positive_ids must contain one bytes[32] per text")
        if type(self.text_commitments) is not tuple or len(self.text_commitments) != int(
            self.text_embeddings.shape[0]
        ):
            raise TypeError("text_commitments must contain one bytes[32] per text")
        motions = tuple(
            _positive_id(value, f"motion_positive_ids[{index}]")
            for index, value in enumerate(self.motion_positive_ids)
        )
        texts = tuple(
            _positive_id(value, f"text_positive_ids[{index}]")
            for index, value in enumerate(self.text_positive_ids)
        )
        text_commitments = tuple(
            _positive_id(value, f"text_commitments[{index}]")
            for index, value in enumerate(self.text_commitments)
        )
        if len(text_commitments) != len(set(text_commitments)):
            raise TrainingRuntimeError("text commitments must be unique within a batch")
        if any(value not in texts for value in motions):
            raise TrainingRuntimeError("every motion must have a positive text in its microbatch")
        if any(value not in motions for value in texts):
            raise TrainingRuntimeError("every text must have a positive motion in its microbatch")
        contexts = self.descriptor_contexts
        if contexts is not None:
            if type(contexts) is not tuple or len(contexts) != checked.batch_size:
                raise TypeError(
                    "descriptor_contexts must be None or an exact tuple covering every motion"
                )
            if any(type(context) is not DescriptorWindowContext for context in contexts):
                raise TypeError(
                    "descriptor_contexts must contain exact DescriptorWindowContext values"
                )
            first = contexts[0]
            for index, context in enumerate(contexts):
                if context.window_sha256 != motions[index].hex():
                    raise TrainingRuntimeError(
                        "descriptor window identity differs from its motion positive family"
                    )
                if context.split != self.split:
                    raise TrainingRuntimeError(
                        "descriptor context split differs from its training batch"
                    )
                if (
                    context.prepared_manifest_sha256 != first.prepared_manifest_sha256
                    or context.source_batch_sha256 != first.source_batch_sha256
                    or context.seed != first.seed
                    or context.epoch != first.epoch
                ):
                    raise TrainingRuntimeError(
                        "one training batch cannot mix descriptor source, seed, or epoch"
                    )
            if len({context.window_sha256 for context in contexts}) != len(contexts):
                raise TrainingRuntimeError("descriptor contexts repeat a window identity")
            if len({context.window_ordinal for context in contexts}) != len(contexts):
                raise TrainingRuntimeError("descriptor contexts repeat a window ordinal")
        # Snapshot text values so a host iterator cannot mutate an in-flight epoch.
        frozen_text = self.text_embeddings.detach().clone(memory_format=torch.contiguous_format)
        object.__setattr__(self, "groups", checked)
        object.__setattr__(self, "text_embeddings", frozen_text)
        object.__setattr__(self, "motion_positive_ids", motions)
        object.__setattr__(self, "text_positive_ids", texts)
        object.__setattr__(self, "text_commitments", text_commitments)
        object.__setattr__(self, "descriptor_contexts", contexts)

    @property
    def motion_count(self) -> int:
        return self.groups.batch_size

    @property
    def edge_count(self) -> int:
        return sum(count * (count - 1) // 2 for count in self.groups.actor_counts)


@runtime_checkable
class TrainingDataSource(Protocol):
    """Minimal deterministic host seam for real or synthetic prepared data."""

    split: str
    manifest_sha256: str

    def iter_epoch(self, *, epoch: int, seed: int) -> Iterable[RetrievalTrainingBatch]:
        """Return the deterministic microbatch stream for one zero-based epoch."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Frozen training settings; scientific hyperparameters are not caller knobs."""

    stage: Stage
    seed: int
    device: str = "cpu"
    request_bf16: bool = False
    bf16_runtime_qualified: bool = False
    effective_global_batch: int = EFFECTIVE_GLOBAL_BATCH
    edge_budget: int = 32_768
    checkpoint_every_updates: int = 1
    synthetic_contract: bool = False

    def __post_init__(self) -> None:
        if self.stage not in ("base", "residual"):
            raise TypeError("stage must be exactly 'base' or 'residual'")
        if type(self.seed) is not int:
            raise TypeError("seed must be an exact int")
        if not self.synthetic_contract and self.seed not in OFFICIAL_SEEDS:
            raise TrainingRuntimeError("formal runs must use one of the three frozen seeds")
        if type(self.device) is not str or not self.device:
            raise TypeError("device must be a nonempty torch device string")
        try:
            parsed_device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise TypeError("device is not a valid torch device string") from error
        if parsed_device.type not in ("cpu", "cuda"):
            raise TrainingRuntimeError("training runtime supports only CPU or CUDA")
        if parsed_device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingRuntimeError("requested CUDA device is unavailable")
        if type(self.request_bf16) is not bool or type(self.bf16_runtime_qualified) is not bool:
            raise TypeError("precision qualification switches must be exact bool values")
        if type(self.effective_global_batch) is not int or self.effective_global_batch < 1:
            raise TypeError("effective_global_batch must be an exact positive int")
        if not self.synthetic_contract and self.effective_global_batch != EFFECTIVE_GLOBAL_BATCH:
            raise TrainingRuntimeError("formal effective global batch is frozen at 128 windows")
        if type(self.edge_budget) is not int or self.edge_budget < 1:
            raise TypeError("edge_budget must be an exact positive int")
        if type(self.checkpoint_every_updates) is not int or self.checkpoint_every_updates < 1:
            raise TypeError("checkpoint_every_updates must be an exact positive int")
        if type(self.synthetic_contract) is not bool:
            raise TypeError("synthetic_contract must be an exact bool")

    @property
    def epochs(self) -> int:
        return BASE_EPOCHS if self.stage == "base" else RESIDUAL_EPOCHS

    @property
    def learning_rate(self) -> float:
        return BASE_LEARNING_RATE if self.stage == "base" else RESIDUAL_LEARNING_RATE

    def canonical_bytes(self) -> bytes:
        value = {
            "authority": 0,
            "bf16_runtime_qualified": self.bf16_runtime_qualified,
            "checkpoint_every_updates": self.checkpoint_every_updates,
            "device": str(torch.device(self.device)),
            "edge_budget": self.edge_budget,
            "effective_global_batch": self.effective_global_batch,
            "epochs": self.epochs,
            "gradient_clip_decimal": "1.0",
            "learning_rate_decimal": "0.0002" if self.stage == "base" else "0.0003",
            "optimizer": "AdamW",
            "request_bf16": self.request_bf16,
            "schedule": "linear-warmup-cosine-decay",
            "seed": self.seed,
            "stage": self.stage,
            "synthetic_contract": self.synthetic_contract,
            "warmup_fraction_decimal": "0.05",
            "weight_decay_decimal": "0.01",
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class PrecisionDecision:
    mode: Literal["FP32", "BF16"]
    autocast_enabled: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TrainingInitializationBinding:
    """Seed-bound construction receipt checked before a formal optimizer exists."""

    stage: Stage
    seed: int
    system_id: str
    factory_sha256: str
    code_artifact_sha256: str
    environment_sha256: str
    behavior_sha256: str
    initial_optimizable_state_sha256: str
    frozen_base_checkpoint_sha256: str | None = None
    frozen_base_state_sha256: str | None = None
    qualified_base_selection_sha256: str | None = None
    residual_capacity_audit_sha256: str | None = None
    schema: str = "phaseset-training-initialization-binding-v1"
    periodic_descriptor_capture_plan_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("base", "residual"):
            raise TypeError("initialization stage must be base or residual")
        if type(self.seed) is not int:
            raise TypeError("initialization seed must be an exact int")
        if type(self.system_id) is not str or not self.system_id:
            raise TypeError("initialization system_id must be a nonempty string")
        _lower_sha256(self.factory_sha256, "factory_sha256")
        _lower_sha256(self.code_artifact_sha256, "code_artifact_sha256")
        _lower_sha256(self.environment_sha256, "environment_sha256")
        _lower_sha256(self.behavior_sha256, "behavior_sha256")
        _lower_sha256(
            self.initial_optimizable_state_sha256,
            "initial_optimizable_state_sha256",
        )
        frozen_values = (
            self.frozen_base_checkpoint_sha256,
            self.frozen_base_state_sha256,
            self.qualified_base_selection_sha256,
            self.residual_capacity_audit_sha256,
        )
        if self.stage == "base":
            if frozen_values != (None, None, None, None):
                raise TrainingRuntimeError(
                    "base initialization cannot bind a frozen base or residual capacity audit"
                )
        else:
            if any(value is None for value in frozen_values):
                raise TrainingRuntimeError(
                    "residual initialization requires checkpoint and frozen-base state digests"
                )
            _lower_sha256(
                self.frozen_base_checkpoint_sha256,
                "frozen_base_checkpoint_sha256",
            )
            _lower_sha256(
                self.frozen_base_state_sha256,
                "frozen_base_state_sha256",
            )
            _lower_sha256(
                self.residual_capacity_audit_sha256,
                "residual_capacity_audit_sha256",
            )
            _lower_sha256(
                self.qualified_base_selection_sha256,
                "qualified_base_selection_sha256",
            )
        plan_sha256 = self.periodic_descriptor_capture_plan_sha256
        if plan_sha256 is not None:
            if self.stage != "residual":
                raise TrainingRuntimeError(
                    "base initialization cannot bind a periodic descriptor capture plan"
                )
            _lower_sha256(
                plan_sha256,
                "periodic_descriptor_capture_plan_sha256",
            )
        expected_schema = (
            CACHED_INITIALIZATION_SCHEMA
            if plan_sha256 is not None
            else "phaseset-training-initialization-binding-v1"
        )
        if self.schema != expected_schema:
            raise TrainingRuntimeError("initialization binding schema changed")

    def canonical_bytes(self) -> bytes:
        value = {
            "behavior_sha256": self.behavior_sha256,
            "code_artifact_sha256": self.code_artifact_sha256,
            "environment_sha256": self.environment_sha256,
            "factory_sha256": self.factory_sha256,
            "frozen_base_checkpoint_sha256": self.frozen_base_checkpoint_sha256,
            "frozen_base_state_sha256": self.frozen_base_state_sha256,
            "initial_optimizable_state_sha256": self.initial_optimizable_state_sha256,
            "qualified_base_selection_sha256": self.qualified_base_selection_sha256,
            "residual_capacity_audit_sha256": self.residual_capacity_audit_sha256,
            "schema": self.schema,
            "seed": self.seed,
            "stage": self.stage,
            "system_id": self.system_id,
        }
        if self.periodic_descriptor_capture_plan_sha256 is not None:
            value["periodic_descriptor_capture_plan_sha256"] = (
                self.periodic_descriptor_capture_plan_sha256
            )
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True, repr=False)
class QualifiedFrozenBase:
    """A registered group base strictly loaded from one verified checkpoint."""

    group_base: nn.Module
    qualification: BaseQualification = field(repr=False, compare=False)
    checkpoint_path: Path = field(repr=False, compare=False)
    system_id: str
    seed: int
    checkpoint_sha256: str
    frozen_state_sha256: str
    frozen_logit_scale: float
    selection_receipt_sha256: str
    code_artifact_sha256: str
    environment_sha256: str
    checkpoint_config_sha256: str

    def __post_init__(self) -> None:
        expected_types = {
            "B0": ActorMeanBase,
            "B1": SetPMABase,
            "B2": SocialTemporalBase,
        }
        if self.system_id not in expected_types or type(self.group_base) is not expected_types.get(
            self.system_id
        ):
            raise TrainingRuntimeError("qualified frozen base type/system identity mismatch")
        if type(self.seed) is not int or self.seed not in OFFICIAL_SEEDS:
            raise TrainingRuntimeError("qualified frozen base seed is outside the census")
        for name in (
            "checkpoint_sha256",
            "frozen_state_sha256",
            "selection_receipt_sha256",
            "code_artifact_sha256",
            "environment_sha256",
            "checkpoint_config_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if any(parameter.requires_grad for parameter in self.group_base.parameters()):
            raise TrainingRuntimeError("qualified frozen base parameters must be frozen")
        if type(self.frozen_logit_scale) is not float or not math.isfinite(self.frozen_logit_scale):
            raise TrainingRuntimeError("qualified frozen base logit scale must be finite float")
        if (
            _frozen_base_score_state_sha256(
                self.group_base,
                self.frozen_logit_scale,
            )
            != self.frozen_state_sha256
        ):
            raise TrainingRuntimeError("qualified frozen base state digest mismatch")
        _validate_qualified_frozen_base(self)

    def __repr__(self) -> str:
        return (
            f"QualifiedFrozenBase(system_id={self.system_id!r}, seed={self.seed}, "
            "checkpoint=<digest>, state=<digest>)"
        )


def resolve_precision(config: TrainingConfig) -> PrecisionDecision:
    """Resolve BF16 only after an explicit target-CUDA qualification."""

    device = torch.device(config.device)
    if not config.request_bf16:
        return PrecisionDecision("FP32", False, "FP32_REQUESTED")
    if not config.bf16_runtime_qualified:
        return PrecisionDecision("FP32", False, "BF16_NOT_RUNTIME_QUALIFIED")
    if device.type != "cuda":
        return PrecisionDecision("FP32", False, "BF16_QUALIFICATION_IS_CUDA_ONLY")
    if not torch.cuda.is_bf16_supported():
        return PrecisionDecision("FP32", False, "CUDA_BF16_UNSUPPORTED")
    return PrecisionDecision("BF16", True, "CUDA_BF16_RUNTIME_QUALIFIED")


class BaseRetrievalSystem(nn.Module):
    """Train one B0/B1/B2 motion base against frozen global text embeddings."""

    def __init__(self, group_base: nn.Module, *, embedding_dim: int) -> None:
        super().__init__()
        if not isinstance(group_base, nn.Module):
            raise TypeError("group_base must be a torch module")
        if type(embedding_dim) is not int or embedding_dim < 2:
            raise TypeError("embedding_dim must be an exact int >=2")
        self.group_base = group_base
        self.embedding_dim = embedding_dim
        self.logit_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    @property
    def system_id(self) -> str:
        value = getattr(self.group_base, "system_id", "BASE")
        return value if type(value) is str else "BASE"

    def encode_trainable(self, groups: PreparedGroupBatch) -> Tensor:
        output = self.group_base(groups)
        if type(output) is not GroupBaseOutput:
            raise TrainingRuntimeError("group base must return exact GroupBaseOutput")
        embedding = output.group_embedding.float().contiguous()
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dim:
            raise TrainingRuntimeError("group base embedding width differs from training system")
        return embedding

    def scores(self, motion_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
        _validate_embedding_pair(motion_embeddings, text_embeddings, self.embedding_dim)
        scale = torch.exp(torch.clamp(self.logit_scale[0], min=0.0, max=math.log(100.0)))
        return (
            scale
            * F.normalize(motion_embeddings, dim=-1, eps=1e-12)
            @ F.normalize(text_embeddings, dim=-1, eps=1e-12).T
        ).contiguous()


class ResidualRetrievalSystem(nn.Module):
    """Train PhaseSet periodic tokens while keeping a qualified base frozen."""

    def __init__(
        self,
        frozen_base: nn.Module,
        periodic_encoder: nn.Module,
        retrieval_head: PhaseSetRetrievalHead,
        *,
        embedding_dim: int,
        frozen_base_logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(frozen_base, nn.Module):
            raise TypeError("frozen_base must be a torch module")
        if not isinstance(periodic_encoder, nn.Module):
            raise TypeError("periodic_encoder must be a torch module")
        if not isinstance(retrieval_head, PhaseSetRetrievalHead):
            raise TypeError("retrieval_head must be PhaseSetRetrievalHead")
        if type(embedding_dim) is not int or embedding_dim < 2:
            raise TypeError("embedding_dim must be an exact int >=2")
        if type(frozen_base_logit_scale) is not float or not math.isfinite(frozen_base_logit_scale):
            raise TypeError("frozen_base_logit_scale must be a finite exact float")
        self.frozen_base = frozen_base
        self.periodic_encoder = periodic_encoder
        self.retrieval_head = retrieval_head
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "_frozen_base_logit_scale",
            torch.tensor([frozen_base_logit_scale], dtype=torch.float32),
            persistent=True,
        )
        candidate_id = getattr(periodic_encoder, "system_id", None)
        if candidate_id is None and isinstance(periodic_encoder, PhaseSetEncoder):
            candidate_id = "08"
        if type(candidate_id) is not str or candidate_id not in RESIDUAL_SYSTEM_IDS:
            raise TrainingRuntimeError(
                "residual encoder must bind one registered system_id in 01--08"
            )
        self._system_id = candidate_id
        encoder_width = getattr(periodic_encoder, "embedding_dim", embedding_dim)
        if type(encoder_width) is not int or encoder_width != embedding_dim:
            raise TrainingRuntimeError("residual encoder width differs from training system")
        for parameter in self.frozen_base.parameters():
            parameter.requires_grad_(False)
        self.frozen_base.eval()

    @property
    def system_id(self) -> str:
        return self._system_id

    def train(self, mode: bool = True) -> ResidualRetrievalSystem:
        super().train(mode)
        self.frozen_base.eval()
        return self

    def _validated_trainable_output(
        self,
        output: object,
    ) -> tuple[Tensor, Tensor]:
        """Apply the unchanged residual-token checks to either explicit path."""

        if type(output) is not GroupTokenOutput:
            raise TrainingRuntimeError("residual encoder must return exact GroupTokenOutput")
        if output.tokens.ndim != 3 or output.tokens.shape[-1] != self.embedding_dim:
            raise TrainingRuntimeError("residual token width differs from training system")
        return output.tokens.float().contiguous(), output.band_mask.contiguous()

    def encode_trainable(self, groups: PreparedGroupBatch) -> tuple[Tensor, Tensor]:
        return self._validated_trainable_output(self.periodic_encoder(groups))

    def encode_trainable_cached(
        self,
        groups: PreparedGroupBatch,
        descriptor_stream: CachedPairChunkStream,
    ) -> tuple[Tensor, Tensor]:
        """Encode only from one explicit verified descriptor stream.

        The cache remains a substitute for the fixed periodic descriptors only.
        Frozen-base outputs and learned residual outputs are never cached here.
        """

        from .controls import PhaseSetSystem, _ControlledPhaseSetEncoder

        if self.system_id not in ("02", "03", "04", "06", "07", "08"):
            raise TrainingRuntimeError(
                "this registered residual system does not admit cached descriptors"
            )
        periodic = self.periodic_encoder
        if (
            type(periodic) is not PhaseSetSystem
            or periodic.system_id != self.system_id
            or type(periodic.encoder) is not _ControlledPhaseSetEncoder
        ):
            raise TrainingRuntimeError(
                "cached residual encoding requires the exact registered PhaseSet encoder"
            )
        if type(descriptor_stream) is not CachedPairChunkStream:
            raise TypeError("descriptor_stream must be exact CachedPairChunkStream")
        output = periodic.forward_cached(
            groups,
            descriptor_stream=descriptor_stream,
        )
        return self._validated_trainable_output(output)

    def encode_frozen_base(self, groups: PreparedGroupBatch) -> Tensor:
        with torch.no_grad():
            output = self.frozen_base(groups)
        if type(output) is not GroupBaseOutput:
            raise TrainingRuntimeError("frozen group base must return exact GroupBaseOutput")
        embedding = output.group_embedding.float().contiguous().detach()
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dim:
            raise TrainingRuntimeError("frozen base width differs from residual system")
        return embedding

    def scores(
        self,
        group_tokens: Tensor,
        band_mask: Tensor,
        base_embeddings: Tensor,
        text_embeddings: Tensor,
    ) -> Tensor:
        _validate_embedding_pair(base_embeddings, text_embeddings, self.embedding_dim)
        base_scores = (
            (
                torch.exp(
                    torch.clamp(
                        self._frozen_base_logit_scale[0],
                        min=0.0,
                        max=math.log(100.0),
                    )
                )
                * F.normalize(base_embeddings, dim=-1, eps=1e-12)
                @ F.normalize(text_embeddings, dim=-1, eps=1e-12).T
            )
            .detach()
            .float()
            .contiguous()
        )
        retrieval = self.retrieval_head(
            group_tokens.float().contiguous(),
            band_mask.contiguous(),
            text_embeddings,
            base_scores,
        )
        return retrieval.scores.contiguous()


def _periodic_descriptor_capture_plan_identity(
    descriptor_plan: object,
    *,
    system_id: str,
    config: TrainingConfig,
    energy_floors_sha256: str,
) -> tuple[PeriodicDescriptorCaptureTrainingPlan, str]:
    """Validate one sealed cache plan without opening a descriptor shard."""

    from .periodic_capture_training_cache import (
        PeriodicDescriptorCaptureTrainingPlan,
    )

    if type(descriptor_plan) is not PeriodicDescriptorCaptureTrainingPlan:
        raise TypeError("descriptor_plan must be exact PeriodicDescriptorCaptureTrainingPlan")
    if (
        config.stage != "residual"
        or config.synthetic_contract
        or descriptor_plan.system_id != system_id
        or descriptor_plan.seed != config.seed
        or descriptor_plan.epochs != config.epochs
        or descriptor_plan.config_sha256 != config.sha256
        or descriptor_plan.edge_budget != config.edge_budget
        or descriptor_plan.energy_floors_sha256 != energy_floors_sha256
        or descriptor_plan.authority != 0
        or descriptor_plan.production is not False
        or descriptor_plan.result_claimed is not False
    ):
        raise TrainingRuntimeError(
            "periodic descriptor capture plan differs from the residual runtime"
        )
    try:
        digest = _lower_sha256(
            descriptor_plan.sha256,
            "periodic_descriptor_capture_plan_sha256",
        )
    except (TypeError, ValueError) as error:
        raise TrainingRuntimeError(
            "periodic descriptor capture plan digest is malformed"
        ) from error
    return descriptor_plan, digest


def _residual_energy_floors_sha256(
    system: ResidualRetrievalSystem,
    config: TrainingConfig,
) -> str:
    """Read the exact registered residual floor identity without a forward."""

    from .controls import PhaseSetSystem, _ControlledPhaseSetEncoder
    from .periodic_descriptor_cache_v2 import energy_floors_sha256

    periodic = system.periodic_encoder
    if (
        type(periodic) is not PhaseSetSystem
        or periodic.system_id != system.system_id
        or type(periodic.encoder) is not _ControlledPhaseSetEncoder
        or periodic.encoder.edge_budget != config.edge_budget
    ):
        raise TrainingRuntimeError(
            "cached residual runtime requires the exact registered PhaseSet encoder"
        )
    return energy_floors_sha256(periodic.encoder._energy_floors)


def _capture_project_method_identities() -> tuple[tuple[type[object], str, object], ...]:
    """Freeze Python method descriptors before a formal system can be constructed."""

    from . import controls as controls_module
    from . import models as models_module
    from . import objectives as objectives_module

    classes: set[type[object]] = {
        BaseRetrievalSystem,
        ResidualRetrievalSystem,
        nn.Dropout,
        nn.GELU,
        nn.LayerNorm,
        nn.Linear,
        nn.Module,
        nn.ModuleList,
        nn.MultiheadAttention,
        nn.Sequential,
        nn.TransformerEncoder,
        nn.TransformerEncoderLayer,
    }
    for module in (controls_module, models_module, objectives_module):
        classes.update(
            value
            for value in vars(module).values()
            if isinstance(value, type) and value.__module__ == module.__name__
        )
    descriptors: list[tuple[type[object], str, object]] = []
    for class_type in sorted(classes, key=lambda value: (value.__module__, value.__qualname__)):
        for name, value in sorted(vars(class_type).items()):
            if isinstance(value, (types.FunctionType, staticmethod, classmethod, property)):
                descriptors.append((class_type, name, value))
    return tuple(descriptors)


_FORMAL_PROJECT_METHOD_IDENTITIES = _capture_project_method_identities()
_FORMAL_TORCH_FUNCTION_IDENTITIES = (
    (F, "conv1d", F.conv1d),
    (F, "dropout", F.dropout),
    (F, "gelu", F.gelu),
    (F, "linear", F.linear),
    (F, "normalize", F.normalize),
    (F, "pad", F.pad),
)


def _assert_registered_method_integrity(
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
) -> None:
    """Reject class monkeypatching and instance-level method shadowing."""

    from .capture_validation import _verified_capture_validation_bindings

    _verified_capture_validation_bindings()
    for name, expected in _FORMAL_RUNTIME_CONSTANTS.items():
        if globals().get(name) != expected or type(globals().get(name)) is not type(expected):
            raise TrainingRuntimeError(f"formal runtime constant changed: {name}")
    for name, expected in _FORMAL_TRAINING_FUNCTION_IDENTITIES:
        if globals().get(name) is not expected:
            raise TrainingRuntimeError(f"formal training function identity changed: {name}")
    for owner, name, expected in _FORMAL_EXTERNAL_FUNCTION_IDENTITIES:
        if getattr(owner, name, None) is not expected:
            raise TrainingRuntimeError(f"formal external function identity changed: {name}")
    for class_type, name, expected in _FORMAL_PROJECT_METHOD_IDENTITIES:
        if vars(class_type).get(name) is not expected:
            raise TrainingRuntimeError("registered method identity changed")
    for module, name, expected in _FORMAL_TORCH_FUNCTION_IDENTITIES:
        if getattr(module, name, None) is not expected:
            raise TrainingRuntimeError("registered torch function identity changed")
    from torch.nn.modules import module as torch_module

    global_hook_names = (
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
        "_global_buffer_registration_hooks",
    )
    if any(bool(getattr(torch_module, name, {})) for name in global_hook_names):
        raise TrainingRuntimeError("formal runtime rejects global torch module hooks")
    for parameter_name, parameter in system.named_parameters():
        if bool(getattr(parameter, "_backward_hooks", {})) or bool(
            getattr(parameter, "_post_accumulate_grad_hooks", {})
        ):
            raise TrainingRuntimeError(
                f"formal parameter {parameter_name} contains a gradient hook"
            )
    local_hook_names = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    for module_name, module in system.named_modules():
        if any(bool(getattr(module, name, {})) for name in local_hook_names):
            label = module_name or "<root>"
            raise TrainingRuntimeError(f"formal module {label} contains a runtime hook")
        if hasattr(module, "parametrizations") and bool(module.parametrizations):
            label = module_name or "<root>"
            raise TrainingRuntimeError(f"formal module {label} contains a parametrization")
        shadowable: set[str] = set()
        for class_type in type(module).__mro__:
            for name, value in vars(class_type).items():
                if isinstance(
                    value,
                    (types.FunctionType, staticmethod, classmethod, property),
                ) or callable(value):
                    shadowable.add(name)
        storage_names = set(vars(module))
        storage_names.update(module._modules)
        storage_names.update(module._parameters)
        storage_names.update(module._buffers)
        shadowed = sorted(shadowable & storage_names)
        if shadowed:
            label = module_name or "<root>"
            raise TrainingRuntimeError(
                f"formal module {label} contains a callable override: {shadowed[0]}"
            )


def _registered_spec_sha256(value: dict[str, object]) -> str:
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    return _sha256(raw)


def build_registered_base_training_system(system_id: str) -> BaseRetrievalSystem:
    """Build one exact B0/B1/B2 formal architecture (initialization is separate)."""

    if type(system_id) is not str or system_id not in REGISTERED_BASE_SYSTEM_IDS:
        raise TrainingRuntimeError("registered base system_id must be B0, B1, or B2")
    system = BaseRetrievalSystem(
        build_group_base(
            system_id,
            hidden_dim=REGISTERED_EMBEDDING_DIM,
            heads=REGISTERED_HEADS,
            ffn_dim=REGISTERED_FFN_DIM,
            dropout=REGISTERED_DROPOUT,
        ),
        embedding_dim=REGISTERED_EMBEDDING_DIM,
    )
    spec = {
        "dropout_decimal": "0.1",
        "embedding_dim": REGISTERED_EMBEDDING_DIM,
        "ffn_dim": REGISTERED_FFN_DIM,
        "heads": REGISTERED_HEADS,
        "schema": "phaseset-registered-base-system-v1",
        "system_id": system_id,
    }
    system._phaseset_registered_spec_sha256 = _registered_spec_sha256(spec)
    return system


def _residual_spec(
    *,
    system_id: str,
    seed: int,
    energy_floors: np.ndarray,
    edge_budget: int,
) -> dict[str, object]:
    floors = np.ascontiguousarray(energy_floors, dtype=np.float64)
    return {
        "edge_budget": edge_budget,
        "embedding_dim": REGISTERED_EMBEDDING_DIM,
        "energy_floors_sha256": _sha256(floors.tobytes(order="C")),
        "incidence_seed": seed,
        "periodic_hidden_dim": REGISTERED_PERIODIC_HIDDEN_DIM,
        "residual_lambda_init_decimal": "0.0",
        "schema": "phaseset-registered-residual-system-v1",
        "system_id": system_id,
        "text_hidden_dim": REGISTERED_TEXT_HIDDEN_DIM,
    }


def build_registered_residual_training_system(
    system_id: str,
    frozen_base: nn.Module,
    *,
    seed: int,
    energy_floors: np.ndarray,
    edge_budget: int = 32_768,
    frozen_base_logit_scale: float = 1.0,
) -> ResidualRetrievalSystem:
    """Build one exact capacity-matched formal residual system 01--08."""

    from .controls import build_phaseset_system
    from .periodic import validate_edge_budget, validate_energy_floors

    if type(system_id) is not str or system_id not in RESIDUAL_SYSTEM_IDS:
        raise TrainingRuntimeError("registered residual system_id must be 01--08")
    if type(seed) is not int or seed not in OFFICIAL_SEEDS:
        raise TrainingRuntimeError("registered residual seed is outside the census")
    if type(frozen_base) not in (ActorMeanBase, SetPMABase, SocialTemporalBase):
        raise TrainingRuntimeError("frozen base must be one exact registered group-base class")
    if getattr(frozen_base, "hidden_dim", None) != REGISTERED_EMBEDDING_DIM:
        raise TrainingRuntimeError("frozen base width is outside the registered architecture")
    floors = validate_energy_floors(energy_floors)
    budget = validate_edge_budget(edge_budget)
    periodic = build_phaseset_system(
        system_id,
        embedding_dim=REGISTERED_EMBEDDING_DIM,
        hidden_dim=REGISTERED_PERIODIC_HIDDEN_DIM,
        energy_floors=floors,
        edge_budget=budget,
        incidence_seed=seed,
    )
    head = PhaseSetRetrievalHead(
        REGISTERED_EMBEDDING_DIM,
        text_hidden_dim=REGISTERED_TEXT_HIDDEN_DIM,
        residual_lambda_init=0.0,
    )
    system = ResidualRetrievalSystem(
        frozen_base,
        periodic,
        head,
        embedding_dim=REGISTERED_EMBEDDING_DIM,
        frozen_base_logit_scale=frozen_base_logit_scale,
    )
    system._phaseset_registered_spec_sha256 = _registered_spec_sha256(
        _residual_spec(
            system_id=system_id,
            seed=seed,
            energy_floors=floors,
            edge_budget=budget,
        )
    )
    system._phaseset_registered_seed = seed
    return system


def _canonical_registered_behavior_sha256(
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
    config: TrainingConfig,
) -> str:
    """Derive expected behavior from the closed registry, never a live marker."""

    ambient_rng = _capture_rng()
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if isinstance(system, BaseRetrievalSystem):
            reference: BaseRetrievalSystem | ResidualRetrievalSystem = (
                build_registered_base_training_system(system.system_id)
            )
        else:
            base_ids = {
                ActorMeanBase: "B0",
                SetPMABase: "B1",
                SocialTemporalBase: "B2",
            }
            base_id = base_ids.get(type(system.frozen_base))
            if base_id is None:
                raise TrainingRuntimeError("frozen base class is outside the registry")
            periodic = system.periodic_encoder
            from .controls import PhaseSetSystem

            if type(periodic) is not PhaseSetSystem or periodic.encoder is None:
                raise TrainingRuntimeError("residual encoder is outside the registry")
            reference = build_registered_residual_training_system(
                system.system_id,
                build_registered_base_training_system(base_id).group_base,
                seed=config.seed,
                energy_floors=periodic.encoder._energy_floors,
                edge_budget=config.edge_budget,
                frozen_base_logit_scale=float(
                    system._frozen_base_logit_scale.detach().cpu().item()
                ),
            )
        return training_system_behavior_sha256(reference)
    finally:
        _restore_rng(ambient_rng)


def _validate_registered_formal_system(
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
    config: TrainingConfig,
) -> None:
    _assert_registered_method_integrity(system)
    marker = getattr(system, "_phaseset_registered_spec_sha256", None)
    _lower_sha256(marker, "registered_spec_sha256")
    if training_system_behavior_sha256(system) != _canonical_registered_behavior_sha256(
        system,
        config,
    ):
        raise TrainingRuntimeError("formal system behavior changed after registered construction")
    if isinstance(system, BaseRetrievalSystem):
        system_id = system.system_id
        expected = _registered_spec_sha256(
            {
                "dropout_decimal": "0.1",
                "embedding_dim": REGISTERED_EMBEDDING_DIM,
                "ffn_dim": REGISTERED_FFN_DIM,
                "heads": REGISTERED_HEADS,
                "schema": "phaseset-registered-base-system-v1",
                "system_id": system_id,
            }
        )
        if (
            system_id not in REGISTERED_BASE_SYSTEM_IDS
            or type(system.group_base) not in (ActorMeanBase, SetPMABase, SocialTemporalBase)
            or system.embedding_dim != REGISTERED_EMBEDDING_DIM
            or marker != expected
        ):
            raise TrainingRuntimeError("base system is outside the registered formal factory")
        return

    from .controls import PhaseSetSystem

    periodic = system.periodic_encoder
    if type(periodic) is not PhaseSetSystem or periodic.encoder is None:
        raise TrainingRuntimeError("residual encoder is outside the registered formal factory")
    floors = periodic.encoder._energy_floors
    expected = _registered_spec_sha256(
        _residual_spec(
            system_id=system.system_id,
            seed=config.seed,
            energy_floors=floors,
            edge_budget=periodic.encoder.edge_budget,
        )
    )
    text_projection = system.retrieval_head.text_band_mlp.shared_projection
    frozen_logit = system._frozen_base_logit_scale
    if (
        system.system_id not in RESIDUAL_SYSTEM_IDS
        or system.embedding_dim != REGISTERED_EMBEDDING_DIM
        or periodic.embedding_dim != REGISTERED_EMBEDDING_DIM
        or periodic.encoder.half_edge_encoder[0].out_features != REGISTERED_PERIODIC_HIDDEN_DIM
        or periodic.encoder.edge_budget != config.edge_budget
        or system.retrieval_head.text_band_mlp.embedding_dim != REGISTERED_EMBEDDING_DIM
        or text_projection[0].out_features != REGISTERED_TEXT_HIDDEN_DIM
        or tuple(frozen_logit.shape) != (1,)
        or frozen_logit.dtype != torch.float32
        or frozen_logit.requires_grad
        or not bool(torch.isfinite(frozen_logit).all().item())
        or getattr(system, "_phaseset_registered_seed", None) != config.seed
        or marker != expected
    ):
        raise TrainingRuntimeError("residual system is outside the registered formal factory")


@dataclass(frozen=True, slots=True)
class ResidualCapacityAudit:
    """Immutable all-system capacity receipt required by every formal residual."""

    rows: tuple[tuple[str, int, str], ...]
    seed: int
    edge_budget: int
    energy_floors_sha256: str
    frozen_base_checkpoint_sha256: str
    frozen_base_state_sha256: str
    qualified_base_selection_sha256: str
    code_artifact_sha256: str
    environment_sha256: str
    schema: str = "phaseset-residual-capacity-audit-v1"

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple:
            raise TypeError("capacity audit rows must be an exact tuple")
        for row in self.rows:
            if (
                type(row) is not tuple
                or len(row) != 3
                or type(row[0]) is not str
                or type(row[1]) is not int
                or row[1] < 1
                or type(row[2]) is not str
            ):
                raise TypeError(
                    "capacity rows must be exact (system_id, count, behavior_sha256) tuples"
                )
            _lower_sha256(row[2], f"capacity behavior {row[0]}")
        if tuple(row[0] for row in self.rows) != RESIDUAL_SYSTEM_IDS:
            raise TrainingRuntimeError("capacity audit must contain ordered systems 01--08")
        if type(self.seed) is not int or self.seed not in OFFICIAL_SEEDS:
            raise TrainingRuntimeError("capacity audit seed is outside the census")
        if type(self.edge_budget) is not int or self.edge_budget < 1:
            raise TrainingRuntimeError("capacity audit edge budget is invalid")
        for name in (
            "energy_floors_sha256",
            "frozen_base_checkpoint_sha256",
            "frozen_base_state_sha256",
            "qualified_base_selection_sha256",
            "code_artifact_sha256",
            "environment_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        full = self.count_for("08")
        for system_id, count, _ in self.rows[:-1]:
            if abs(count - full) > 0.01 * full:
                raise TrainingRuntimeError(
                    f"complete residual system {system_id} exceeds the +/-1% capacity bound"
                )
        if self.schema != "phaseset-residual-capacity-audit-v1":
            raise TrainingRuntimeError("residual capacity audit schema changed")

    def count_for(self, system_id: str) -> int:
        if type(system_id) is not str or system_id not in RESIDUAL_SYSTEM_IDS:
            raise TrainingRuntimeError("capacity lookup system_id must be 01--08")
        return {row[0]: row[1] for row in self.rows}[system_id]

    def behavior_for(self, system_id: str) -> str:
        if type(system_id) is not str or system_id not in RESIDUAL_SYSTEM_IDS:
            raise TrainingRuntimeError("capacity lookup system_id must be 01--08")
        return {row[0]: row[2] for row in self.rows}[system_id]

    def canonical_bytes(self) -> bytes:
        value = {
            "full_system_id": "08",
            "rows": [
                {
                    "behavior_sha256": behavior,
                    "system_id": system_id,
                    "trainable_parameters": count,
                }
                for system_id, count, behavior in self.rows
            ],
            "code_artifact_sha256": self.code_artifact_sha256,
            "edge_budget": self.edge_budget,
            "environment_sha256": self.environment_sha256,
            "energy_floors_sha256": self.energy_floors_sha256,
            "frozen_base_checkpoint_sha256": self.frozen_base_checkpoint_sha256,
            "frozen_base_state_sha256": self.frozen_base_state_sha256,
            "schema": self.schema,
            "qualified_base_selection_sha256": self.qualified_base_selection_sha256,
            "seed": self.seed,
            "tolerance_decimal": "0.01",
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())


def audit_residual_training_parameter_counts(
    systems: Mapping[str, ResidualRetrievalSystem],
) -> tuple[tuple[str, int], ...]:
    """Audit the complete residual system, including text head and logit scale."""

    if type(systems) is not dict or set(systems) != set(RESIDUAL_SYSTEM_IDS):
        raise TrainingRuntimeError("systems must be an exact 01--08 residual dict")
    counts: dict[str, int] = {}
    for system_id in RESIDUAL_SYSTEM_IDS:
        system = systems[system_id]
        if not isinstance(system, ResidualRetrievalSystem) or system.system_id != system_id:
            raise TrainingRuntimeError("residual capacity mapping identity mismatch")
        counts[system_id] = sum(
            parameter.numel() for parameter in system.parameters() if parameter.requires_grad
        )
    full = counts["08"]
    if full <= 0:
        raise TrainingRuntimeError("registered full residual has no trainable parameters")
    for system_id in RESIDUAL_SYSTEM_IDS[:-1]:
        if abs(counts[system_id] - full) > 0.01 * full:
            raise TrainingRuntimeError(
                f"complete residual system {system_id} exceeds the +/-1% capacity bound"
            )
    return tuple((system_id, counts[system_id]) for system_id in RESIDUAL_SYSTEM_IDS)


def _rebuild_registered_residual_capacity_audit(
    qualified_base: QualifiedFrozenBase,
    config: TrainingConfig,
    *,
    energy_floors: np.ndarray,
) -> ResidualCapacityAudit:
    """Recompute the complete formal 01--08 capacity evidence from sources."""

    from .periodic import validate_energy_floors

    floors = validate_energy_floors(energy_floors)
    floor_sha = _sha256(np.ascontiguousarray(floors, dtype=np.float64).tobytes(order="C"))
    rows: list[tuple[str, int, str]] = []
    for system_id in RESIDUAL_SYSTEM_IDS:
        system = build_registered_residual_training_system(
            system_id,
            qualified_base.group_base,
            seed=config.seed,
            energy_floors=floors,
            edge_budget=config.edge_budget,
            frozen_base_logit_scale=qualified_base.frozen_logit_scale,
        )
        _validate_registered_formal_system(system, config)
        rows.append(
            (
                system_id,
                sum(
                    parameter.numel()
                    for parameter in system.parameters()
                    if parameter.requires_grad
                ),
                training_system_behavior_sha256(system),
            )
        )
    return ResidualCapacityAudit(
        rows=tuple(rows),
        seed=config.seed,
        edge_budget=config.edge_budget,
        energy_floors_sha256=floor_sha,
        frozen_base_checkpoint_sha256=qualified_base.checkpoint_sha256,
        frozen_base_state_sha256=qualified_base.frozen_state_sha256,
        qualified_base_selection_sha256=qualified_base.selection_receipt_sha256,
        code_artifact_sha256=training_code_artifact_sha256(),
        environment_sha256=training_environment_sha256(),
    )


def _validate_registered_residual_capacity_audit(
    value: ResidualCapacityAudit,
    qualified_base: QualifiedFrozenBase,
    config: TrainingConfig,
    *,
    energy_floors: np.ndarray,
    expected_qualification_sha256: str,
    expected_capacity_audit_sha256: str,
) -> ResidualCapacityAudit:
    """Require one receipt to equal a fresh closed-registry reconstruction."""

    if type(value) is not ResidualCapacityAudit:
        raise TypeError("capacity audit must be exact ResidualCapacityAudit")
    try:
        expected_capacity = _lower_sha256(
            expected_capacity_audit_sha256,
            "expected capacity audit sha256",
        )
    except TypeError as error:
        raise TrainingRuntimeError("expected capacity audit digest is malformed") from error
    if value.sha256 != expected_capacity:
        raise TrainingRuntimeError("capacity audit differs from the trusted expected digest")
    _validate_qualified_frozen_base(
        qualified_base,
        expected_selection_sha256=expected_qualification_sha256,
    )
    rebuilt = _rebuild_registered_residual_capacity_audit(
        qualified_base,
        config,
        energy_floors=energy_floors,
    )
    if rebuilt != value or rebuilt.canonical_bytes() != value.canonical_bytes():
        raise TrainingRuntimeError(
            "residual capacity audit differs from fresh registered cohort evidence"
        )
    return rebuilt


def build_registered_residual_capacity_audit(
    qualified_base: QualifiedFrozenBase,
    config: TrainingConfig,
    *,
    energy_floors: np.ndarray,
    expected_qualification_sha256: str,
) -> ResidualCapacityAudit:
    """Build and validate the complete formal 01--08 cohort in closed code."""

    if type(qualified_base) is not QualifiedFrozenBase:
        raise TypeError("qualified_base must be exact QualifiedFrozenBase")
    if type(config) is not TrainingConfig or config.stage != "residual":
        raise TrainingRuntimeError("capacity audit requires a formal residual config")
    if config.synthetic_contract or qualified_base.seed != config.seed:
        raise TrainingRuntimeError("capacity audit seed/base context mismatch")
    _validate_qualified_frozen_base(
        qualified_base,
        expected_selection_sha256=expected_qualification_sha256,
    )
    return _rebuild_registered_residual_capacity_audit(
        qualified_base,
        config,
        energy_floors=energy_floors,
    )


def _validate_embedding_pair(motion: Tensor, text: Tensor, width: int) -> None:
    if (
        type(motion) is not Tensor
        or type(text) is not Tensor
        or motion.dtype != torch.float32
        or text.dtype != torch.float32
        or motion.ndim != 2
        or text.ndim != 2
        or motion.shape[1] != width
        or text.shape[1] != width
        or motion.device != text.device
        or not motion.is_contiguous()
        or not text.is_contiguous()
        or not bool(torch.isfinite(motion).all().item())
        or not bool(torch.isfinite(text).all().item())
    ):
        raise TrainingRuntimeError("motion/text embeddings must be finite contiguous float32 [N,D]")


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    path: Path
    sha256: str
    global_step: int


CheckpointObserver = Callable[[CheckpointArtifact, Literal["update", "validation"]], None]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Local runtime report; it intentionally cannot attest external execution."""

    status: Literal["COMPLETED", "INTERRUPTED"]
    stage: Stage
    system_id: str
    global_step: int
    completed_epochs: int
    best_validation_metric: float | None
    latest_checkpoint: CheckpointArtifact
    best_checkpoint: CheckpointArtifact | None
    train_edges_seen: int
    validation_edges_seen: int
    max_microbatch_edges: int
    precision: PrecisionDecision
    initialization_binding_sha256: str | None
    code_artifact_sha256: str
    environment_sha256: str
    behavior_sha256: str
    initial_optimizable_state_sha256: str
    frozen_base_checkpoint_sha256: str | None
    frozen_base_state_sha256: str | None
    qualified_base_selection_sha256: str | None
    residual_capacity_audit_sha256: str | None
    config_sha256: str
    train_manifest_sha256: str
    val_manifest_sha256: str
    last_train_loss: float | None
    authority: int = 0
    production: bool = False
    result_claimed: bool = False
    external_receipt_verified: bool = False
    schema: str = REPORT_SCHEMA


@dataclass(frozen=True, slots=True)
class CachedResidualTrainingReport(TrainingReport):
    """A residual report that binds one complete descriptor-cache plan."""

    periodic_descriptor_capture_plan_sha256: str = ""
    dataloader_state_sha256: str = ""
    schema: str = CACHED_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.stage != "residual":
            raise TrainingRuntimeError("cached descriptor reports must be residual")
        _lower_sha256(
            self.periodic_descriptor_capture_plan_sha256,
            "periodic_descriptor_capture_plan_sha256",
        )
        _lower_sha256(self.dataloader_state_sha256, "dataloader_state_sha256")
        if self.schema != CACHED_REPORT_SCHEMA:
            raise TrainingRuntimeError("cached descriptor report schema changed")


@dataclass(slots=True)
class _CursorState:
    epoch: int = 0
    update_index: int = 0
    global_step: int = 0
    checkpoint_sequence: int = 0
    train_edges_seen: int = 0
    validation_edges_seen: int = 0
    max_microbatch_edges: int = 0
    best_validation_metric: float | None = None
    best_checkpoint_name: str | None = None
    best_checkpoint_sha256: str | None = None
    last_train_loss: float | None = None


@dataclass(slots=True)
class _CacheRow:
    batch: RetrievalTrainingBatch
    primary_leaf: Tensor
    band_mask: Tensor | None
    base_embedding: Tensor | None
    rng_before: dict[str, object]


def _scoped_capture_validation_functions() -> tuple[object, object, object]:
    """Return the exact capture callables after their module closes live drift."""

    from .capture_validation import (
        CaptureValidationSource,
        _verified_capture_validation_bindings,
        run_capture_validation,
        run_capture_validation_cached,
    )

    checked_source, checked_uncached = _verified_capture_validation_bindings()
    if (
        checked_source is not CaptureValidationSource
        or checked_uncached is not run_capture_validation
    ):
        raise TrainingRuntimeError("capture runtime binding verifier changed its ABI")
    return CaptureValidationSource, run_capture_validation, run_capture_validation_cached


def _source_identity(
    source: object,
    split: Split,
) -> tuple[TrainingDataSource | CaptureValidationSource, str]:
    # Lazy import avoids the capture evaluator's intentional system-interface
    # dependency on this module. Only the concrete val-only source is admitted.
    CaptureValidationSource, _, _ = _scoped_capture_validation_functions()

    if type(source) is CaptureValidationSource:
        if split != "val":
            raise TrainingRuntimeError("capture validation cannot be a training source")
        checked = CaptureValidationSource(source.split, source.manifest_sha256, source.captures)
        # The census binds the complete window/feature/text plan AND upstream
        # manifest, so resume cannot substitute a different capture gallery.
        return checked, checked.census_sha256
    if not isinstance(source, TrainingDataSource):
        raise TypeError("data source must implement TrainingDataSource")
    if source.split != split:
        if source.split == "test":
            raise TrainingRuntimeError("test split is sealed and cannot enter model selection")
        raise TrainingRuntimeError(f"expected {split} data source")
    digest = _lower_sha256(source.manifest_sha256, f"{split} manifest_sha256")
    return source, digest


def _descriptor_plan_sources(
    plan: PeriodicDescriptorCaptureTrainingPlan,
    train_source: object,
    val_source: object,
) -> tuple[TrainingDataSource, CaptureValidationSource]:
    """Bind the complete prepared-train and holistic-capture inputs pre-forward."""

    from .prepared_data_v2 import PreparedTrainingDataSourceV2

    CaptureValidationSource, _, _ = _scoped_capture_validation_functions()

    if type(train_source) is not PreparedTrainingDataSourceV2:
        raise TypeError("cached descriptor training requires exact PreparedTrainingDataSourceV2")
    if (
        train_source.split != "train"
        or train_source.manifest_sha256 != plan.train_source_manifest_sha256
    ):
        raise TrainingRuntimeError(
            "cached descriptor training source differs from the admitted plan"
        )
    if type(val_source) is not CaptureValidationSource:
        raise TypeError("cached descriptor training requires exact CaptureValidationSource")
    checked = plan.validate_capture_validation_source(val_source)
    if (
        checked.split != "val"
        or checked.manifest_sha256 != plan.capture_source_manifest_sha256
        or checked.census_sha256 != plan.capture_source_census_sha256
    ):
        raise TrainingRuntimeError(
            "cached capture validation source differs from the admitted plan"
        )
    return train_source, checked


def _cached_dataloader_state_sha256(
    train_manifest_sha256: str,
    val_manifest_sha256: str,
    descriptor_plan_sha256: str,
) -> str:
    value = {
        "periodic_descriptor_capture_plan_sha256": _lower_sha256(
            descriptor_plan_sha256,
            "periodic_descriptor_capture_plan_sha256",
        ),
        "schema": "phaseset-training-dataloader-state-v2-cached-descriptor-plan",
        "train_manifest_sha256": _lower_sha256(
            train_manifest_sha256,
            "train_manifest_sha256",
        ),
        "val_manifest_sha256": _lower_sha256(
            val_manifest_sha256,
            "val_manifest_sha256",
        ),
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    return _sha256(raw)


def _runtime_descriptor_plan_identity(
    descriptor_plan: object | None,
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
    config: TrainingConfig,
) -> tuple[PeriodicDescriptorCaptureTrainingPlan | None, str | None]:
    if descriptor_plan is None:
        return None, None
    if not isinstance(system, ResidualRetrievalSystem):
        raise TrainingRuntimeError("base runtime cannot use a periodic descriptor capture plan")
    return _periodic_descriptor_capture_plan_identity(
        descriptor_plan,
        system_id=system.system_id,
        config=config,
        energy_floors_sha256=_residual_energy_floors_sha256(system, config),
    )


def _materialize_epoch(
    source: TrainingDataSource,
    *,
    split: Split,
    epoch: int,
    seed: int,
    edge_budget: int,
) -> tuple[RetrievalTrainingBatch, ...]:
    try:
        values = tuple(source.iter_epoch(epoch=epoch, seed=seed))
    except Exception as error:
        raise TrainingRuntimeError(f"{split} iterator failed for epoch {epoch}") from error
    if not values:
        raise TrainingRuntimeError(f"{split} iterator produced no microbatches")
    for index, batch in enumerate(values):
        if type(batch) is not RetrievalTrainingBatch:
            raise TypeError(f"{split} iterator item {index} is not RetrievalTrainingBatch")
        if batch.split != split:
            if batch.split == "test":
                raise TrainingRuntimeError("test split is sealed and cannot enter training")
            raise TrainingRuntimeError(f"{split} iterator emitted a {batch.split} batch")
        if batch.edge_count > edge_budget:
            raise TrainingRuntimeError(
                f"microbatch edge count {batch.edge_count} exceeds frozen budget {edge_budget}"
            )
    return values


def _group_effective_batches(
    values: tuple[RetrievalTrainingBatch, ...],
    target: int,
) -> tuple[tuple[RetrievalTrainingBatch, ...], ...]:
    groups: list[tuple[RetrievalTrainingBatch, ...]] = []
    current: list[RetrievalTrainingBatch] = []
    count = 0
    for batch in values:
        if batch.motion_count > target:
            raise TrainingRuntimeError("one microbatch exceeds the effective global batch")
        if count + batch.motion_count > target:
            raise TrainingRuntimeError(
                "microbatch plan overshoots effective batch; sampler must close at 128 windows"
            )
        current.append(batch)
        count += batch.motion_count
        if count == target:
            groups.append(tuple(current))
            current = []
            count = 0
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _positive_mask(
    motion_ids: tuple[bytes, ...],
    text_ids: tuple[bytes, ...],
    device: torch.device,
) -> Tensor:
    mask = torch.tensor(
        [[motion == text for text in text_ids] for motion in motion_ids],
        dtype=torch.bool,
        device=device,
    ).contiguous()
    if not bool(mask.any(dim=1).all().item()) or not bool(mask.any(dim=0).all().item()):
        raise TrainingRuntimeError("effective batch contains an unbound motion or text")
    return mask


def _capture_macro_bidirectional_r1(
    logits: Tensor,
    positive: Tensor,
    motion_positive_ids: tuple[bytes, ...],
    text_positive_ids: tuple[bytes, ...],
    motion_commitments: tuple[bytes, ...],
    text_commitments: tuple[bytes, ...],
) -> float:
    """Exact capture-macro validation endpoint with commitment tie breaks."""

    motion_count, text_count = positive.shape
    if (
        logits.shape != positive.shape
        or len(motion_positive_ids) != motion_count
        or len(text_positive_ids) != text_count
        or len(motion_commitments) != motion_count
        or len(text_commitments) != text_count
        or len(set(motion_commitments)) != motion_count
        or len(set(text_commitments)) != text_count
    ):
        raise TrainingRuntimeError("validation endpoint census is malformed")
    families = tuple(sorted(set(motion_positive_ids)))
    if not families or set(families) != set(text_positive_ids):
        raise TrainingRuntimeError("validation capture-family census differs by direction")
    motion_hits: list[int] = []
    for motion_index in range(motion_count):
        row = logits[motion_index]
        maximum = torch.max(row)
        tied = tuple(
            int(index) for index in torch.nonzero(row == maximum, as_tuple=False).flatten().tolist()
        )
        chosen = min(tied, key=lambda index: text_commitments[index])
        motion_hits.append(int(bool(positive[motion_index, chosen].item())))
    text_hits: list[int] = []
    for text_index in range(text_count):
        column = logits[:, text_index]
        maximum = torch.max(column)
        tied = tuple(
            int(index)
            for index in torch.nonzero(column == maximum, as_tuple=False).flatten().tolist()
        )
        chosen = min(tied, key=lambda index: motion_commitments[index])
        text_hits.append(int(bool(positive[chosen, text_index].item())))
    capture_values: list[Fraction] = []
    for family in families:
        motion_indices = tuple(
            index for index, value in enumerate(motion_positive_ids) if value == family
        )
        text_indices = tuple(
            index for index, value in enumerate(text_positive_ids) if value == family
        )
        motion_value = Fraction(
            sum(motion_hits[index] for index in motion_indices),
            len(motion_indices),
        )
        text_value = Fraction(
            sum(text_hits[index] for index in text_indices),
            len(text_indices),
        )
        capture_values.append((motion_value + text_value) / 2)
    return float(sum(capture_values, start=Fraction(0, 1)) / len(capture_values))


def _capture_rng() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_algorithm": numpy_state[0],
        "numpy_keys": torch.tensor(numpy_state[1].astype(np.uint32).astype(np.int64)),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng(value: Mapping[str, object]) -> None:
    try:
        random.setstate(value["python"])  # type: ignore[arg-type]
        numpy_keys = value["numpy_keys"]
        if type(numpy_keys) is not Tensor:
            raise TypeError
        np.random.set_state(
            (
                str(value["numpy_algorithm"]),
                numpy_keys.cpu().numpy().astype(np.uint32),
                int(value["numpy_position"]),
                int(value["numpy_has_gauss"]),
                float(value["numpy_cached_gaussian"]),
            )
        )
        cpu_state = value["torch_cpu"]
        if type(cpu_state) is not Tensor:
            raise TypeError
        torch.set_rng_state(cpu_state.cpu())
        cuda_states = value["torch_cuda"]
        if torch.cuda.is_available():
            if type(cuda_states) is not list or any(
                type(item) is not Tensor for item in cuda_states
            ):
                raise TypeError
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise TrainingCheckpointError("checkpoint RNG state is malformed") from error


def _stable_hash(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if item is None:
            digest.update(b"N")
        elif type(item) is bool:
            digest.update(b"B1" if item else b"B0")
        elif type(item) is int:
            raw = str(item).encode("ascii")
            digest.update(b"I" + struct.pack(">I", len(raw)) + raw)
        elif type(item) is float:
            digest.update(b"F" + struct.pack(">d", item))
        elif type(item) is str:
            raw = item.encode("utf-8")
            digest.update(b"S" + struct.pack(">I", len(raw)) + raw)
        elif type(item) is bytes:
            digest.update(b"Y" + struct.pack(">I", len(item)) + item)
        elif type(item) is Tensor:
            tensor = item.detach().cpu().contiguous()
            dtype = str(tensor.dtype).encode("ascii")
            digest.update(b"T" + struct.pack(">I", len(dtype)) + dtype)
            update(tuple(int(axis) for axis in tensor.shape))
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            digest.update(struct.pack(">Q", len(raw)) + raw)
        elif isinstance(item, Mapping):
            digest.update(b"M" + struct.pack(">I", len(item)))
            ordered = sorted(item.items(), key=lambda pair: repr(pair[0]))
            for key, nested in ordered:
                update(key)
                update(nested)
        elif type(item) in (tuple, list):
            digest.update((b"Q" if type(item) is tuple else b"L") + struct.pack(">I", len(item)))
            for nested in item:
                update(nested)
        else:
            raise TypeError(f"unsupported checkpoint value type: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def training_system_state_sha256(
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
    *,
    optimizable_only: bool = False,
) -> str:
    """Hash a canonical CPU snapshot of one training system's registered state.

    For a residual system, ``optimizable_only`` excludes the frozen base and
    its frozen temperature while retaining every trainable residual parameter
    and mutable residual buffer. For a base system the two views are identical.
    This is a construction receipt, not a model weight.
    """

    if not isinstance(system, (BaseRetrievalSystem, ResidualRetrievalSystem)):
        raise TypeError("system must be a registered training system")
    state = system.state_dict()
    selected = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(state.items())
        if not (
            optimizable_only
            and isinstance(system, ResidualRetrievalSystem)
            and (name.startswith("frozen_base.") or name == "_frozen_base_logit_scale")
        )
    }
    if not selected:
        raise TrainingRuntimeError("training system state is empty")
    return _stable_hash(selected)


_UNSUPPORTED_BEHAVIOR_VALUE = object()


def _behavior_value(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TrainingRuntimeError("model behavior contains a nonfinite float")
        return {"float_hex": value.hex()}
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": contiguous.dtype.str,
            "sha256": _sha256(contiguous.tobytes(order="C")),
            "shape": list(contiguous.shape),
        }
    if type(value) in (tuple, list, torch.Size):
        nested = [_behavior_value(item) for item in value]
        if any(item is _UNSUPPORTED_BEHAVIOR_VALUE for item in nested):
            return _UNSUPPORTED_BEHAVIOR_VALUE
        return nested
    if is_dataclass(value) and not isinstance(value, type):
        encoded: dict[str, object] = {}
        for item in dataclass_fields(value):
            nested = _behavior_value(getattr(value, item.name))
            if nested is _UNSUPPORTED_BEHAVIOR_VALUE:
                return _UNSUPPORTED_BEHAVIOR_VALUE
            encoded[item.name] = nested
        return encoded
    return _UNSUPPORTED_BEHAVIOR_VALUE


def training_system_behavior_sha256(
    system: BaseRetrievalSystem | ResidualRetrievalSystem,
) -> str:
    """Hash non-state behavior that can change a resumed training trajectory.

    State tensors alone omit dropout probabilities, attention configuration,
    energy floors, control incidence seeds, and similar Python attributes. The
    closed inventory below binds those values together with the exact module
    class tree. Opaque attributes are excluded rather than represented by an
    unstable ``repr``.
    """

    if not isinstance(system, (BaseRetrievalSystem, ResidualRetrievalSystem)):
        raise TypeError("system must be a registered training system")
    rows: list[dict[str, object]] = []
    excluded = {
        "training",
        "dump_patches",
        "call_super_init",
    }
    explicit_names = (
        "system_id",
        "embedding_dim",
        "hidden_dim",
        "edge_budget",
        "incidence_seed",
        "embed_dim",
        "num_heads",
        "dropout",
        "batch_first",
        "norm_first",
        "activation_relu_or_gelu",
        "p",
        "inplace",
        "in_features",
        "out_features",
        "normalized_shape",
        "eps",
        "elementwise_affine",
        "approximate",
        "control_spec",
        "spec",
        "_system_id",
        "_energy_floors",
        "_phaseset_registered_spec_sha256",
        "_phaseset_registered_seed",
    )
    for name, module in system.named_modules():
        attributes: dict[str, object] = {}
        candidates = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_") and key not in excluded
        }
        for key in explicit_names:
            if hasattr(module, key):
                candidates[key] = getattr(module, key)
        for key, value in sorted(candidates.items()):
            encoded = _behavior_value(value)
            if encoded is not _UNSUPPORTED_BEHAVIOR_VALUE:
                attributes[key] = encoded
        activation = getattr(module, "activation", None)
        if callable(activation):
            attributes["activation_callable"] = (
                f"{getattr(activation, '__module__', '')}."
                f"{getattr(activation, '__qualname__', type(activation).__qualname__)}"
            )
        rows.append(
            {
                "attributes": attributes,
                "class": f"{type(module).__module__}.{type(module).__qualname__}",
                "name": name,
            }
        )
    parameters = [
        {
            "dtype": str(parameter.dtype),
            "name": name,
            "requires_grad": parameter.requires_grad,
            "shape": list(parameter.shape),
        }
        for name, parameter in system.named_parameters()
    ]
    payload = {
        "parameters": parameters,
        "rows": rows,
        "schema": "phaseset-training-system-behavior-v1",
        "stage": "residual" if isinstance(system, ResidualRetrievalSystem) else "base",
        "system_id": system.system_id,
    }
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        + b"\n"
    )
    return _sha256(raw)


def _module_state_sha256(module: nn.Module) -> str:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(module.state_dict().items())
    }
    if not state:
        raise TrainingRuntimeError("frozen base state is empty")
    return _stable_hash(state)


def _frozen_base_score_state_sha256(
    group_base: nn.Module,
    frozen_logit_scale: float,
) -> str:
    if type(frozen_logit_scale) is not float or not math.isfinite(frozen_logit_scale):
        raise TrainingRuntimeError("frozen base logit scale must be a finite exact float")
    state = {
        "group_base": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in sorted(group_base.state_dict().items())
        },
        "logit_scale": torch.tensor([frozen_logit_scale], dtype=torch.float32),
    }
    if not state["group_base"]:
        raise TrainingRuntimeError("frozen base state is empty")
    return _stable_hash(state)


def _construct_seed_bound_system(
    factory: Callable[[], BaseRetrievalSystem | ResidualRetrievalSystem],
    config: TrainingConfig,
    *,
    factory_sha256: str,
    frozen_base_checkpoint_sha256: str | None = None,
    qualified_base_selection_sha256: str | None = None,
    residual_capacity_audit: ResidualCapacityAudit | None = None,
    periodic_descriptor_capture_plan_sha256: str | None = None,
) -> tuple[
    BaseRetrievalSystem | ResidualRetrievalSystem,
    TrainingInitializationBinding,
]:
    """Construct a formal system only after installing its registered seed.

    Ambient Python, NumPy, CPU, and CUDA RNG states are restored after the
    constructor returns, so construction cannot silently perturb a host. The
    returned binding is subsequently mandatory for non-synthetic runtimes.
    """

    if not callable(factory):
        raise TypeError("factory must be callable")
    if type(config) is not TrainingConfig:
        raise TypeError("config must be exact TrainingConfig")
    checked_factory = _lower_sha256(factory_sha256, "factory_sha256")
    if config.stage == "residual":
        checked_checkpoint = _lower_sha256(
            frozen_base_checkpoint_sha256,
            "frozen_base_checkpoint_sha256",
        )
        checked_selection = _lower_sha256(
            qualified_base_selection_sha256,
            "qualified_base_selection_sha256",
        )
        if type(residual_capacity_audit) is not ResidualCapacityAudit:
            raise TrainingRuntimeError(
                "formal residual construction requires an all-system capacity audit"
            )
        checked_plan_sha256 = (
            _lower_sha256(
                periodic_descriptor_capture_plan_sha256,
                "periodic_descriptor_capture_plan_sha256",
            )
            if periodic_descriptor_capture_plan_sha256 is not None
            else None
        )
    else:
        if frozen_base_checkpoint_sha256 is not None:
            raise TrainingRuntimeError("base construction cannot bind a frozen checkpoint")
        checked_checkpoint = None
        checked_selection = None
        if qualified_base_selection_sha256 is not None:
            raise TrainingRuntimeError("base construction cannot bind a base selection receipt")
        if residual_capacity_audit is not None:
            raise TrainingRuntimeError("base construction cannot bind a residual capacity audit")
        if periodic_descriptor_capture_plan_sha256 is not None:
            raise TrainingRuntimeError(
                "base construction cannot bind a periodic descriptor capture plan"
            )
        checked_plan_sha256 = None

    ambient_rng = _capture_rng()
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        system = factory()
    finally:
        _restore_rng(ambient_rng)
    if config.stage == "base" and not isinstance(system, BaseRetrievalSystem):
        raise TrainingRuntimeError("base factory returned the wrong system type")
    if config.stage == "residual" and not isinstance(system, ResidualRetrievalSystem):
        raise TrainingRuntimeError("residual factory returned the wrong system type")
    if not config.synthetic_contract:
        _validate_registered_formal_system(system, config)
    if isinstance(system, ResidualRetrievalSystem) and residual_capacity_audit is not None:
        live_count = sum(
            parameter.numel() for parameter in system.parameters() if parameter.requires_grad
        )
        if residual_capacity_audit.count_for(system.system_id) != live_count:
            raise TrainingRuntimeError(
                "residual system trainable count differs from the bound capacity audit"
            )
    frozen_state = (
        _frozen_base_score_state_sha256(
            system.frozen_base,
            float(system._frozen_base_logit_scale.detach().cpu().item()),
        )
        if isinstance(system, ResidualRetrievalSystem)
        else None
    )
    binding = TrainingInitializationBinding(
        stage=config.stage,
        seed=config.seed,
        system_id=system.system_id,
        factory_sha256=checked_factory,
        code_artifact_sha256=training_code_artifact_sha256(),
        environment_sha256=training_environment_sha256(),
        behavior_sha256=training_system_behavior_sha256(system),
        initial_optimizable_state_sha256=training_system_state_sha256(
            system,
            optimizable_only=True,
        ),
        frozen_base_checkpoint_sha256=checked_checkpoint,
        frozen_base_state_sha256=frozen_state,
        qualified_base_selection_sha256=checked_selection,
        residual_capacity_audit_sha256=(
            residual_capacity_audit.sha256 if residual_capacity_audit is not None else None
        ),
        periodic_descriptor_capture_plan_sha256=checked_plan_sha256,
        schema=(
            CACHED_INITIALIZATION_SCHEMA
            if checked_plan_sha256 is not None
            else "phaseset-training-initialization-binding-v1"
        ),
    )
    return system, binding


def construct_seed_bound_system(
    factory: Callable[[], BaseRetrievalSystem | ResidualRetrievalSystem],
    config: TrainingConfig,
    *,
    factory_sha256: str,
) -> tuple[
    BaseRetrievalSystem | ResidualRetrievalSystem,
    TrainingInitializationBinding,
]:
    """Construct only a synthetic custom factory under a deterministic seed.

    Formal systems deliberately do not accept an arbitrary callable or a
    caller-asserted factory digest. Use the closed registered constructors
    below so architecture and initialization provenance cannot be self-signed.
    """

    if type(config) is not TrainingConfig:
        raise TypeError("config must be exact TrainingConfig")
    if not config.synthetic_contract:
        raise TrainingRuntimeError(
            "formal construction must use a closed registered system constructor"
        )
    return _construct_seed_bound_system(
        factory,
        config,
        factory_sha256=factory_sha256,
    )


def _registered_factory_sha256(stage: Stage, system_id: str) -> str:
    return _registered_spec_sha256(
        {
            "registry": "phaseset-training-registered-factories-v1",
            "stage": stage,
            "system_id": system_id,
        }
    )


def training_code_artifact_sha256() -> str:
    """Hash the installed PhaseSet Python implementation without local paths."""

    package_root = Path(__file__).resolve().parent
    roots = (package_root, package_root.parent / "phasepair_core")
    files = sorted(
        (path for root in roots for path in root.rglob("*.py")),
        key=lambda path: path.relative_to(package_root.parent).as_posix(),
    )
    if not files or not all(path.is_file() and not path.is_symlink() for path in files):
        raise TrainingRuntimeError("installed PhaseSet source artifact is not a regular file set")
    digest = hashlib.sha256(b"phaseset-installed-python-artifact-v1\x00")
    for path in files:
        name = path.relative_to(package_root.parent).as_posix().encode("ascii")
        raw = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    metadata = package_root.parent.parent / "pyproject.toml"
    if metadata.is_file() and not metadata.is_symlink():
        raw = metadata.read_bytes()
        digest.update(b"pyproject.toml\x00" + len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def _backend_flag_targets() -> tuple[tuple[object, str, bool], ...]:
    candidates = (
        (torch.backends.cuda.matmul, "allow_tf32", False),
        (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False),
        (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False),
        (torch.backends.cuda.matmul, "allow_fp16_accumulation", False),
        (torch.backends.cudnn, "allow_tf32", False),
        (torch.backends.cudnn, "benchmark", False),
        (torch.backends.cudnn, "deterministic", True),
    )
    return tuple(
        (owner, name, expected) for owner, name, expected in candidates if hasattr(owner, name)
    )


def _capture_numerical_runtime_flags() -> dict[str, object]:
    flags: dict[str, object] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (torch.is_deterministic_algorithms_warn_only_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mha_fastpath_enabled": bool(torch.backends.mha.get_fastpath_enabled()),
    }
    for owner, name, _expected in _backend_flag_targets():
        owner_name = "cuda.matmul" if owner is torch.backends.cuda.matmul else "cudnn"
        flags[f"{owner_name}.{name}"] = bool(getattr(owner, name))
    return flags


def _install_frozen_numerical_runtime() -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mha.set_fastpath_enabled(False)
    for owner, name, expected in _backend_flag_targets():
        setattr(owner, name, expected)


def _restore_numerical_runtime_flags(flags: Mapping[str, object]) -> None:
    for owner, name, _expected in _backend_flag_targets():
        owner_name = "cuda.matmul" if owner is torch.backends.cuda.matmul else "cudnn"
        value = flags.get(f"{owner_name}.{name}")
        if type(value) is bool:
            setattr(owner, name, value)
    precision = flags.get("float32_matmul_precision")
    if type(precision) is str:
        torch.set_float32_matmul_precision(precision)
    mha_fastpath = flags.get("mha_fastpath_enabled")
    if type(mha_fastpath) is bool:
        torch.backends.mha.set_fastpath_enabled(mha_fastpath)
    deterministic = flags.get("deterministic_algorithms")
    warn_only = flags.get("deterministic_warn_only")
    if type(deterministic) is bool and type(warn_only) is bool:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _assert_frozen_numerical_runtime(device: torch.device) -> None:
    observed = _capture_numerical_runtime_flags()
    expected: dict[str, object] = {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "mha_fastpath_enabled": False,
    }
    for owner, name, value in _backend_flag_targets():
        owner_name = "cuda.matmul" if owner is torch.backends.cuda.matmul else "cudnn"
        expected[f"{owner_name}.{name}"] = value
    if observed != expected:
        raise TrainingRuntimeError("frozen numerical runtime flags changed")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
        ":4096:8",
        ":16:8",
    ):
        raise TrainingRuntimeError(
            "CUDA formal runtime requires a deterministic CUBLAS_WORKSPACE_CONFIG"
        )


@contextmanager
def _frozen_numerical_runtime(device: torch.device) -> Iterator[None]:
    previous = _capture_numerical_runtime_flags()
    try:
        _install_frozen_numerical_runtime()
        _assert_frozen_numerical_runtime(device)
        yield
    finally:
        _restore_numerical_runtime_flags(previous)


def _cuda_environment_inventory() -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return []
    rows: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        rows.append(
            {
                "capability": list(torch.cuda.get_device_capability(index)),
                "index": index,
                "name": properties.name,
                "total_memory": properties.total_memory,
            }
        )
    return rows


def training_environment_sha256() -> str:
    """Bind hardware/runtime identity and the installed deterministic policy."""

    frozen_flags = {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "mha_fastpath_enabled": False,
        "backend_flags": {
            (
                f"{'cuda.matmul' if owner is torch.backends.cuda.matmul else 'cudnn'}.{name}"
            ): expected
            for owner, name, expected in _backend_flag_targets()
        },
    }
    value = {
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_devices": _cuda_environment_inventory(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "frozen_numerical_runtime": frozen_flags,
        "numpy_version": np.__version__,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "schema": "phaseset-training-environment-v2",
        "torch_build_sha256": _sha256(torch.__config__.show().encode("utf-8")),
        "torch_version": torch.__version__,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    return _sha256(raw)


def construct_registered_base_seed_bound_system(
    system_id: str,
    config: TrainingConfig,
) -> tuple[BaseRetrievalSystem, TrainingInitializationBinding]:
    """Construct one formal B0--B2 system exclusively from the closed registry."""

    if type(config) is not TrainingConfig or config.stage != "base":
        raise TrainingRuntimeError("registered base construction requires a base config")
    if config.synthetic_contract:
        raise TrainingRuntimeError("registered formal construction rejects synthetic configs")
    system, binding = _construct_seed_bound_system(
        lambda: build_registered_base_training_system(system_id),
        config,
        factory_sha256=_registered_factory_sha256("base", system_id),
    )
    if not isinstance(system, BaseRetrievalSystem):
        raise AssertionError("closed base registry returned the wrong type")
    return system, binding


def construct_registered_residual_seed_bound_system(
    system_id: str,
    qualified_base: QualifiedFrozenBase,
    config: TrainingConfig,
    *,
    energy_floors: np.ndarray,
    residual_capacity_audit: ResidualCapacityAudit,
    expected_qualification_sha256: str,
    expected_capacity_audit_sha256: str,
    descriptor_plan: PeriodicDescriptorCaptureTrainingPlan | None = None,
) -> tuple[ResidualRetrievalSystem, TrainingInitializationBinding]:
    """Construct one formal 01--08 residual exclusively from the closed registry."""

    if type(config) is not TrainingConfig or config.stage != "residual":
        raise TrainingRuntimeError("registered residual construction requires a residual config")
    if config.synthetic_contract:
        raise TrainingRuntimeError("registered formal construction rejects synthetic configs")
    if type(qualified_base) is not QualifiedFrozenBase:
        raise TypeError("qualified_base must be exact QualifiedFrozenBase")
    if qualified_base.seed != config.seed:
        raise TrainingRuntimeError("qualified base seed differs from residual seed")
    checked_plan_sha256: str | None = None
    if descriptor_plan is not None:
        from .periodic import validate_energy_floors
        from .periodic_descriptor_cache_v2 import energy_floors_sha256

        checked_floors = validate_energy_floors(energy_floors)
        _, checked_plan_sha256 = _periodic_descriptor_capture_plan_identity(
            descriptor_plan,
            system_id=system_id,
            config=config,
            energy_floors_sha256=energy_floors_sha256(checked_floors),
        )
    _validate_registered_residual_capacity_audit(
        residual_capacity_audit,
        qualified_base,
        config,
        energy_floors=energy_floors,
        expected_qualification_sha256=expected_qualification_sha256,
        expected_capacity_audit_sha256=expected_capacity_audit_sha256,
    )
    if (
        _frozen_base_score_state_sha256(
            qualified_base.group_base,
            qualified_base.frozen_logit_scale,
        )
        != qualified_base.frozen_state_sha256
    ):
        raise TrainingRuntimeError("qualified base changed after checkpoint loading")
    from .periodic import validate_energy_floors

    floors = validate_energy_floors(energy_floors)
    floor_sha = _sha256(np.ascontiguousarray(floors, dtype=np.float64).tobytes(order="C"))
    expected_context = (
        residual_capacity_audit.seed == config.seed
        and residual_capacity_audit.edge_budget == config.edge_budget
        and residual_capacity_audit.energy_floors_sha256 == floor_sha
        and residual_capacity_audit.frozen_base_checkpoint_sha256
        == qualified_base.checkpoint_sha256
        and residual_capacity_audit.frozen_base_state_sha256 == qualified_base.frozen_state_sha256
        and residual_capacity_audit.qualified_base_selection_sha256
        == qualified_base.selection_receipt_sha256
        and residual_capacity_audit.code_artifact_sha256 == training_code_artifact_sha256()
        and residual_capacity_audit.environment_sha256 == training_environment_sha256()
    )
    if not expected_context:
        raise TrainingRuntimeError("residual capacity audit context mismatch")
    system, binding = _construct_seed_bound_system(
        lambda: build_registered_residual_training_system(
            system_id,
            qualified_base.group_base,
            seed=config.seed,
            energy_floors=floors,
            edge_budget=config.edge_budget,
            frozen_base_logit_scale=qualified_base.frozen_logit_scale,
        ),
        config,
        factory_sha256=_registered_factory_sha256("residual", system_id),
        frozen_base_checkpoint_sha256=qualified_base.checkpoint_sha256,
        qualified_base_selection_sha256=qualified_base.selection_receipt_sha256,
        residual_capacity_audit=residual_capacity_audit,
        periodic_descriptor_capture_plan_sha256=checked_plan_sha256,
    )
    if not isinstance(system, ResidualRetrievalSystem):
        raise AssertionError("closed residual registry returned the wrong type")
    if residual_capacity_audit.behavior_for(system_id) != training_system_behavior_sha256(system):
        raise TrainingRuntimeError("residual behavior differs from the capacity cohort")
    return system, binding


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> CheckpointArtifact:
    if path.exists() or path.is_symlink():
        raise TrainingCheckpointError("immutable checkpoint target already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise TrainingCheckpointError("checkpoint directory cannot be a symlink")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        raw = temporary.read_bytes()
        temporary_metadata = temporary.stat()
        try:
            # A same-directory hard-link publish is atomic and, unlike replace,
            # cannot overwrite a checkpoint won by a concurrent writer.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise TrainingCheckpointError("immutable checkpoint target already exists") from error
        except OSError as error:
            if path.exists() or path.is_symlink():
                raise TrainingCheckpointError(
                    "immutable checkpoint target already exists"
                ) from error
            raise TrainingCheckpointError("checkpoint atomic publish failed") from error
        published_metadata = path.stat()
        if (
            not path.is_file()
            or path.is_symlink()
            or (published_metadata.st_dev, published_metadata.st_ino)
            != (temporary_metadata.st_dev, temporary_metadata.st_ino)
        ):
            raise TrainingCheckpointError("checkpoint atomic publish identity mismatch")
        temporary.unlink()
        published = path.read_bytes()
        if published != raw:
            raise TrainingCheckpointError("checkpoint post-publish verification failed")
    finally:
        if temporary.exists():
            temporary.unlink()
    return CheckpointArtifact(path.resolve(), _sha256(published), int(payload["global_step"]))


def _load_torch_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], CheckpointArtifact]:
    if not path.is_file() or path.is_symlink():
        raise TrainingCheckpointError("checkpoint must be a regular non-symlink file")
    raw = path.read_bytes()
    artifact_sha256 = _sha256(raw)
    if expected_sha256 is not None:
        try:
            expected = _lower_sha256(expected_sha256, "expected checkpoint sha256")
        except TypeError as error:
            raise TrainingCheckpointError("expected checkpoint digest is malformed") from error
        if artifact_sha256 != expected:
            raise TrainingCheckpointError("checkpoint artifact digest mismatch")
    try:
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise TrainingCheckpointError("checkpoint is truncated or unreadable") from error
    if type(value) is not dict:
        raise TrainingCheckpointError("checkpoint payload must be an exact dict")
    global_step = value.get("global_step")
    if type(global_step) is not int or global_step < 0:
        raise TrainingCheckpointError("checkpoint global_step is malformed")
    return value, CheckpointArtifact(path.resolve(), artifact_sha256, global_step)


def _validate_qualified_frozen_base(
    value: QualifiedFrozenBase,
    *,
    expected_selection_sha256: str | None = None,
) -> str:
    """Revalidate every qualified-base field against its nine rows and file.

    The checkpoint itself and the complete canonical qualification are the
    source evidence.  No copied construction marker or caller-recomputed
    summary digest is treated as proof of origin.
    """

    if type(value) is not QualifiedFrozenBase:
        raise TypeError("qualified base must be exact QualifiedFrozenBase")
    try:
        qualification_bytes = canonical_base_qualification_bytes(value.qualification)
    except (TypeError, ValueError) as error:
        raise TrainingCheckpointError(
            "qualified base canonical nine-row evidence is malformed"
        ) from error
    selection = _sha256(qualification_bytes)
    if expected_selection_sha256 is not None:
        try:
            trusted_selection = _lower_sha256(
                expected_selection_sha256,
                "expected base selection sha256",
            )
        except TypeError as error:
            raise TrainingCheckpointError("expected base selection digest is malformed") from error
        if selection != trusted_selection:
            raise TrainingCheckpointError(
                "qualified base differs from the trusted selection digest"
            )
    if value.selection_receipt_sha256 != selection:
        raise TrainingCheckpointError("qualified base selection receipt mismatch")
    if value.seed not in OFFICIAL_SEEDS:
        raise TrainingCheckpointError("qualified base seed is outside the census")
    expected_system_id = value.qualification.winner_system_id
    if value.system_id != expected_system_id:
        raise TrainingCheckpointError("qualified base winner identity changed")
    expected_checkpoint = value.qualification.winner_checkpoint_sha256s[
        OFFICIAL_SEEDS.index(value.seed)
    ]
    if value.checkpoint_sha256 != expected_checkpoint:
        raise TrainingCheckpointError(
            "qualified base checkpoint differs from nine-row selection evidence"
        )
    if not isinstance(value.checkpoint_path, Path):
        raise TrainingCheckpointError("qualified base checkpoint path is malformed")
    payload, artifact = _load_torch_checkpoint(
        value.checkpoint_path,
        expected_sha256=expected_checkpoint,
    )
    if artifact.path != value.checkpoint_path or artifact.sha256 != value.checkpoint_sha256:
        raise TrainingCheckpointError("qualified base checkpoint source changed")
    payload_state_digest = payload.get("state_digest")
    payload_without_digest = dict(payload)
    payload_without_digest.pop("state_digest", None)
    if type(payload_state_digest) is not str or payload_state_digest != _stable_hash(
        payload_without_digest
    ):
        raise TrainingCheckpointError("qualified base checkpoint state digest mismatch")
    updates_per_epoch = _checkpoint_exact_int(
        payload,
        "updates_per_epoch",
        minimum=1,
    )
    total_steps = _checkpoint_exact_int(payload, "total_steps", minimum=1)
    if total_steps != BASE_EPOCHS * updates_per_epoch:
        raise TrainingCheckpointError(
            "qualified base checkpoint schedule differs from the formal base schedule"
        )
    selection_cursor = _validate_checkpoint_progress(
        payload,
        epochs=BASE_EPOCHS,
        updates_per_epoch=updates_per_epoch,
        total_steps=total_steps,
        artifact_path=artifact.path,
        artifact_sha256=artifact.sha256,
    )
    if (
        not artifact.path.name.endswith("-validation.pt")
        or selection_cursor.epoch < 1
        or selection_cursor.best_validation_metric is None
        or selection_cursor.best_checkpoint_name != artifact.path.name
        or selection_cursor.best_checkpoint_sha256 != artifact.sha256
    ):
        raise TrainingCheckpointError(
            "qualified base artifact is not its payload's validation-selected best"
        )
    # This proves the artifact was selected when its immutable validation
    # checkpoint was written. The externally trusted qualification digest,
    # which binds all terminal and selected-checkpoint receipts, decides which
    # such artifact is the authorized final winner across the completed runs.
    current_code = training_code_artifact_sha256()
    current_environment = training_environment_sha256()
    expected_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "base",
        "system_id": expected_system_id,
        "seed": value.seed,
        "factory_sha256": _registered_factory_sha256("base", expected_system_id),
        "code_artifact_sha256": current_code,
        "environment_sha256": current_environment,
        "config_sha256": value.checkpoint_config_sha256,
    }
    for name, expected in expected_payload.items():
        if payload.get(name) != expected:
            raise TrainingCheckpointError(f"qualified base checkpoint {name} evidence mismatch")
    if value.code_artifact_sha256 != current_code:
        raise TrainingCheckpointError("qualified base code artifact changed")
    if value.environment_sha256 != current_environment:
        raise TrainingCheckpointError("qualified base environment changed")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise TrainingCheckpointError("qualified base checkpoint model is malformed")
    current_model: dict[str, Tensor] = {
        f"group_base.{name}": tensor.detach().cpu().contiguous()
        for name, tensor in sorted(value.group_base.state_dict().items())
    }
    current_model["logit_scale"] = torch.tensor(
        [value.frozen_logit_scale],
        dtype=torch.float32,
    )
    if _stable_hash(model) != _stable_hash(current_model):
        raise TrainingCheckpointError(
            "qualified base module and temperature differ from checkpoint evidence"
        )
    behavior_group = copy.deepcopy(value.group_base)
    for parameter in behavior_group.parameters():
        parameter.requires_grad_(True)
    behavior_system = BaseRetrievalSystem(
        behavior_group,
        embedding_dim=REGISTERED_EMBEDDING_DIM,
    )
    behavior_system._phaseset_registered_spec_sha256 = _registered_spec_sha256(
        {
            "dropout_decimal": "0.1",
            "embedding_dim": REGISTERED_EMBEDDING_DIM,
            "ffn_dim": REGISTERED_FFN_DIM,
            "heads": REGISTERED_HEADS,
            "schema": "phaseset-registered-base-system-v1",
            "system_id": expected_system_id,
        }
    )
    if training_system_behavior_sha256(behavior_system) != payload.get("behavior_sha256"):
        raise TrainingCheckpointError("qualified base behavior differs from checkpoint evidence")
    expected_frozen_state = _frozen_base_score_state_sha256(
        value.group_base,
        value.frozen_logit_scale,
    )
    if value.frozen_state_sha256 != expected_frozen_state:
        raise TrainingCheckpointError("qualified base frozen state changed")
    if value.group_base.training or any(
        parameter.requires_grad for parameter in value.group_base.parameters()
    ):
        raise TrainingCheckpointError("qualified base is no longer frozen in eval mode")
    return selection


def load_qualified_frozen_base(
    checkpoint_path: str | Path,
    *,
    qualification: BaseQualification,
    seed: int,
    expected_qualification_sha256: str,
) -> QualifiedFrozenBase:
    """Strictly load the group base from a selected registered base checkpoint."""

    try:
        qualification_bytes = canonical_base_qualification_bytes(
            qualification,
            expected_sha256=expected_qualification_sha256,
        )
    except (TypeError, ValueError) as error:
        raise TrainingCheckpointError("base qualification receipt is malformed") from error
    selection = _sha256(qualification_bytes)
    if tuple(EXPERIMENT_SEEDS) != OFFICIAL_SEEDS:
        raise TrainingCheckpointError("experiment/training seed censuses disagree")
    if seed not in OFFICIAL_SEEDS:
        raise TrainingCheckpointError("qualified base identity is outside the census")
    system_id = qualification.winner_system_id
    if system_id not in REGISTERED_BASE_SYSTEM_IDS:
        raise TrainingCheckpointError("qualified base winner is outside the census")
    expected_checkpoint = qualification.winner_checkpoint_sha256s[OFFICIAL_SEEDS.index(seed)]
    payload, artifact = _load_torch_checkpoint(
        Path(checkpoint_path),
        expected_sha256=expected_checkpoint,
    )
    if artifact.sha256 != expected_checkpoint:
        raise TrainingCheckpointError("qualified base checkpoint digest mismatch")
    state_digest = payload.pop("state_digest", None)
    if type(state_digest) is not str or state_digest != _stable_hash(payload):
        raise TrainingCheckpointError("qualified base checkpoint state digest mismatch")
    payload["state_digest"] = state_digest
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "base",
        "system_id": system_id,
        "seed": seed,
        "factory_sha256": _registered_factory_sha256("base", system_id),
        "code_artifact_sha256": training_code_artifact_sha256(),
        "environment_sha256": training_environment_sha256(),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise TrainingCheckpointError(f"qualified base checkpoint {name} mismatch")
    for name in (
        "config_sha256",
        "initialization_binding_sha256",
        "behavior_sha256",
        "initial_optimizable_state_sha256",
    ):
        try:
            _lower_sha256(payload.get(name), name)
        except TypeError as error:
            raise TrainingCheckpointError(
                f"qualified base checkpoint {name} is malformed"
            ) from error
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise TrainingCheckpointError("qualified base checkpoint model is malformed")
    ambient_rng = _capture_rng()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        system = build_registered_base_training_system(system_id)
        system.load_state_dict(model, strict=True)  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise TrainingCheckpointError("qualified base checkpoint model cannot be loaded") from error
    finally:
        _restore_rng(ambient_rng)
    if training_system_state_sha256(system) != _stable_hash(model):
        raise TrainingCheckpointError("qualified base loaded state differs from checkpoint")
    group_base = system.group_base
    frozen_logit_scale = float(system.logit_scale.detach().cpu().item())
    for parameter in group_base.parameters():
        parameter.requires_grad_(False)
    group_base.eval()
    frozen_state = _frozen_base_score_state_sha256(
        group_base,
        frozen_logit_scale,
    )
    return QualifiedFrozenBase(
        group_base=group_base,
        qualification=qualification,
        checkpoint_path=artifact.path,
        system_id=system_id,
        seed=seed,
        checkpoint_sha256=artifact.sha256,
        frozen_state_sha256=frozen_state,
        frozen_logit_scale=frozen_logit_scale,
        selection_receipt_sha256=selection,
        code_artifact_sha256=training_code_artifact_sha256(),
        environment_sha256=training_environment_sha256(),
        checkpoint_config_sha256=payload["config_sha256"],  # type: ignore[arg-type]
    )


def _checkpoint_exact_int(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: int = 0,
) -> int:
    value = payload.get(name)
    if type(value) is not int or value < minimum:
        raise TrainingCheckpointError(f"checkpoint {name} must be an exact int >= {minimum}")
    return value


def _optimizer_step_value(value: object) -> int:
    if type(value) is int:
        step = value
    elif type(value) is Tensor and value.numel() == 1:
        scalar = float(value.detach().cpu().item())
        if not math.isfinite(scalar) or not scalar.is_integer():
            raise TrainingCheckpointError("checkpoint optimizer step is nonintegral")
        step = int(scalar)
    else:
        raise TrainingCheckpointError("checkpoint optimizer step is malformed")
    if step < 0:
        raise TrainingCheckpointError("checkpoint optimizer step is negative")
    return step


def _validate_checkpoint_progress(
    payload: Mapping[str, object],
    *,
    epochs: int,
    updates_per_epoch: int,
    total_steps: int,
    artifact_path: Path,
    artifact_sha256: str,
) -> _CursorState:
    """Validate one exact optimizer/scheduler/cursor state before loading it."""

    epoch = _checkpoint_exact_int(payload, "epoch")
    update_index = _checkpoint_exact_int(payload, "update_index")
    global_step = _checkpoint_exact_int(payload, "global_step")
    sequence = _checkpoint_exact_int(payload, "checkpoint_sequence", minimum=1)
    if epoch > epochs or update_index > updates_per_epoch or global_step > total_steps:
        raise TrainingCheckpointError("checkpoint progress cursor is outside the schedule")
    if global_step != epoch * updates_per_epoch + update_index:
        raise TrainingCheckpointError(
            "checkpoint cursor violates global_step = epoch * updates + update_index"
        )
    if epoch == epochs and update_index != 0:
        raise TrainingCheckpointError("completed checkpoint cannot carry an update cursor")
    expected_prefix = f"checkpoint-{global_step:012d}-{sequence:06d}-"
    if (
        not artifact_path.name.startswith(expected_prefix)
        or artifact_path.suffix != ".pt"
        or artifact_path.name[len(expected_prefix) : -3] not in ("update", "validation")
    ):
        raise TrainingCheckpointError("checkpoint filename disagrees with its progress cursor")

    train_edges = _checkpoint_exact_int(payload, "train_edges_seen")
    validation_edges = _checkpoint_exact_int(payload, "validation_edges_seen")
    maximum_edges = _checkpoint_exact_int(payload, "max_microbatch_edges")
    if (global_step == 0) is not (train_edges == 0):
        raise TrainingCheckpointError("checkpoint train-edge counter disagrees with global_step")
    if (epoch == 0) is not (validation_edges == 0):
        raise TrainingCheckpointError("checkpoint validation-edge counter disagrees with epoch")
    if (global_step > 0 or epoch > 0) and maximum_edges < 1:
        raise TrainingCheckpointError("checkpoint maximum edge count is inconsistent")

    metric = payload.get("best_validation_metric")
    if metric is not None and (
        type(metric) is not float or not math.isfinite(metric) or not 0.0 <= metric <= 1.0
    ):
        raise TrainingCheckpointError("checkpoint validation metric is malformed")
    loss = payload.get("last_train_loss")
    if global_step == 0:
        if loss is not None:
            raise TrainingCheckpointError("zero-step checkpoint cannot carry a train loss")
    elif type(loss) is not float or not math.isfinite(loss) or loss < 0.0:
        raise TrainingCheckpointError("checkpoint train loss is malformed")
    best_name = payload.get("best_checkpoint_name")
    best_sha = payload.get("best_checkpoint_sha256")
    if best_name is not None and (
        type(best_name) is not str
        or not best_name.startswith("checkpoint-")
        or not best_name.endswith(".pt")
        or Path(best_name).name != best_name
    ):
        raise TrainingCheckpointError("checkpoint best-checkpoint name is malformed")
    if best_sha is not None:
        try:
            _lower_sha256(best_sha, "best_checkpoint_sha256")
        except TypeError as error:
            raise TrainingCheckpointError("checkpoint best digest is malformed") from error
        if best_name is None:
            raise TrainingCheckpointError("checkpoint best digest has no checkpoint name")
    elif best_name is not None:
        if best_name != artifact_path.name:
            raise TrainingCheckpointError(
                "checkpoint omits the digest of a different selected checkpoint"
            )
        best_sha = _lower_sha256(artifact_sha256, "artifact_sha256")

    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise TrainingCheckpointError("checkpoint scheduler state is malformed")
    if _checkpoint_exact_int(scheduler, "last_epoch") != global_step:
        raise TrainingCheckpointError("checkpoint scheduler step disagrees with global_step")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
        raise TrainingCheckpointError("checkpoint optimizer state is malformed")
    optimizer_steps = [
        _optimizer_step_value(state["step"])
        for state in optimizer["state"].values()  # type: ignore[union-attr]
        if isinstance(state, Mapping) and "step" in state
    ]
    if global_step > 0 and not optimizer_steps:
        raise TrainingCheckpointError("checkpoint optimizer contains no parameter steps")
    if any(step != global_step for step in optimizer_steps):
        raise TrainingCheckpointError("checkpoint optimizer step disagrees with global_step")

    return _CursorState(
        epoch=epoch,
        update_index=update_index,
        global_step=global_step,
        checkpoint_sequence=sequence,
        train_edges_seen=train_edges,
        validation_edges_seen=validation_edges,
        max_microbatch_edges=maximum_edges,
        best_validation_metric=metric,
        best_checkpoint_name=best_name,
        best_checkpoint_sha256=best_sha,
        last_train_loss=loss,
    )


def _learning_rate_multiplier(step: int, total_steps: int) -> float:
    warmup_steps = max(1, math.ceil(WARMUP_FRACTION * total_steps))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _same_checkpoint_scalar(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _same_checkpoint_scalar(first, second)
            for first, second in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return bool(left == right)


def _validate_restored_optimizer_scheduler(
    payload: Mapping[str, object],
    *,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainingConfig,
    global_step: int,
    total_steps: int,
) -> None:
    """Validate the complete frozen AdamW/LambdaLR state before mutation."""

    optimizer_payload = payload.get("optimizer")
    if not isinstance(optimizer_payload, Mapping) or set(optimizer_payload) != {
        "state",
        "param_groups",
    }:
        raise TrainingCheckpointError("checkpoint optimizer key census is malformed")
    states = optimizer_payload.get("state")
    groups = optimizer_payload.get("param_groups")
    if not isinstance(states, Mapping) or type(groups) is not list:
        raise TrainingCheckpointError("checkpoint optimizer state is malformed")

    expected_optimizer = optimizer.state_dict()
    expected_groups = expected_optimizer["param_groups"]
    if len(groups) != len(expected_groups) or len(groups) != len(optimizer.param_groups):
        raise TrainingCheckpointError("checkpoint optimizer parameter-group count mismatch")
    expected_lr = config.learning_rate * _learning_rate_multiplier(
        global_step,
        total_steps,
    )
    parameter_by_id: dict[int, nn.Parameter] = {}
    for group_index, (saved_group, expected_group, live_group) in enumerate(
        zip(groups, expected_groups, optimizer.param_groups, strict=True)
    ):
        if not isinstance(saved_group, Mapping) or set(saved_group) != set(expected_group):
            raise TrainingCheckpointError(
                f"checkpoint optimizer parameter-group {group_index} key census mismatch"
            )
        saved_parameters = saved_group.get("params")
        expected_parameters = expected_group["params"]
        live_parameters = live_group["params"]
        if (
            type(saved_parameters) is not list
            or saved_parameters != expected_parameters
            or len(saved_parameters) != len(live_parameters)
            or any(type(parameter_id) is not int for parameter_id in saved_parameters)
        ):
            raise TrainingCheckpointError(
                f"checkpoint optimizer parameter-group {group_index} parameter census mismatch"
            )
        for parameter_id, parameter in zip(
            saved_parameters,
            live_parameters,
            strict=True,
        ):
            if parameter_id in parameter_by_id or not isinstance(parameter, nn.Parameter):
                raise TrainingCheckpointError(
                    "checkpoint optimizer parameter identity census is malformed"
                )
            parameter_by_id[parameter_id] = parameter
        for key, expected_value in expected_group.items():
            if key == "params":
                continue
            required_value = expected_lr if key == "lr" else expected_value
            if not _same_checkpoint_scalar(saved_group.get(key), required_value):
                raise TrainingCheckpointError(
                    f"checkpoint optimizer parameter-group {group_index} {key} mismatch"
                )

    if any(type(parameter_id) is not int for parameter_id in states):
        raise TrainingCheckpointError("checkpoint optimizer state parameter id is malformed")
    state_ids = set(states)
    if (
        (global_step == 0 and state_ids)
        or (global_step > 0 and not state_ids)
        or not state_ids.issubset(parameter_by_id)
    ):
        raise TrainingCheckpointError("checkpoint optimizer parameter-state census mismatch")
    for parameter_id in state_ids:
        parameter = parameter_by_id[parameter_id]
        record = states[parameter_id]
        if not isinstance(record, Mapping) or set(record) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise TrainingCheckpointError(
                "checkpoint AdamW parameter state is incomplete or malformed"
            )
        if _optimizer_step_value(record["step"]) != global_step:
            raise TrainingCheckpointError("checkpoint optimizer step disagrees with global_step")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = record[name]
            if (
                type(moment) is not Tensor
                or moment.layout is not torch.strided
                or tuple(moment.shape) != tuple(parameter.shape)
                or moment.dtype != parameter.dtype
                or not bool(torch.isfinite(moment).all().item())
            ):
                raise TrainingCheckpointError(f"checkpoint AdamW {name} tensor is malformed")
        if bool((record["exp_avg_sq"] < 0).any().item()):  # type: ignore[operator]
            raise TrainingCheckpointError("checkpoint AdamW exp_avg_sq is negative")

    scheduler_payload = payload.get("scheduler")
    if not isinstance(scheduler_payload, Mapping):
        raise TrainingCheckpointError("checkpoint scheduler state is malformed")
    expected_scheduler = scheduler.state_dict()
    if set(scheduler_payload) != set(expected_scheduler):
        raise TrainingCheckpointError("checkpoint scheduler key census mismatch")
    for key, expected_value in expected_scheduler.items():
        if key == "last_epoch":
            required_value: object = global_step
        elif key == "_step_count":
            required_value = global_step + 1
        elif key == "_last_lr":
            required_value = [expected_lr] * len(optimizer.param_groups)
        else:
            required_value = expected_value
        if not _same_checkpoint_scalar(scheduler_payload.get(key), required_value):
            raise TrainingCheckpointError(f"checkpoint scheduler {key} mismatch")


_FORMAL_TRAINING_FUNCTION_IDENTITIES = (
    ("_assert_frozen_numerical_runtime", _assert_frozen_numerical_runtime),
    ("_atomic_torch_save", _atomic_torch_save),
    ("_backend_flag_targets", _backend_flag_targets),
    ("_capture_numerical_runtime_flags", _capture_numerical_runtime_flags),
    ("_capture_rng", _capture_rng),
    ("_capture_macro_bidirectional_r1", _capture_macro_bidirectional_r1),
    ("_cached_dataloader_state_sha256", _cached_dataloader_state_sha256),
    ("_cuda_environment_inventory", _cuda_environment_inventory),
    ("_descriptor_plan_sources", _descriptor_plan_sources),
    ("_frozen_numerical_runtime", _frozen_numerical_runtime),
    ("_group_effective_batches", _group_effective_batches),
    ("_install_frozen_numerical_runtime", _install_frozen_numerical_runtime),
    ("_learning_rate_multiplier", _learning_rate_multiplier),
    ("_load_torch_checkpoint", _load_torch_checkpoint),
    ("_materialize_epoch", _materialize_epoch),
    ("_positive_mask", _positive_mask),
    (
        "_periodic_descriptor_capture_plan_identity",
        _periodic_descriptor_capture_plan_identity,
    ),
    ("_residual_energy_floors_sha256", _residual_energy_floors_sha256),
    ("_restore_numerical_runtime_flags", _restore_numerical_runtime_flags),
    ("_restore_rng", _restore_rng),
    ("_runtime_descriptor_plan_identity", _runtime_descriptor_plan_identity),
    ("_scoped_capture_validation_functions", _scoped_capture_validation_functions),
    (
        "_rebuild_registered_residual_capacity_audit",
        _rebuild_registered_residual_capacity_audit,
    ),
    ("_source_identity", _source_identity),
    ("_stable_hash", _stable_hash),
    ("_validate_checkpoint_progress", _validate_checkpoint_progress),
    ("_validate_qualified_frozen_base", _validate_qualified_frozen_base),
    (
        "_validate_registered_residual_capacity_audit",
        _validate_registered_residual_capacity_audit,
    ),
    ("_validate_restored_optimizer_scheduler", _validate_restored_optimizer_scheduler),
    ("canonical_base_qualification_bytes", canonical_base_qualification_bytes),
    ("resolve_precision", resolve_precision),
    ("training_environment_sha256", training_environment_sha256),
    ("variable_positive_symmetric_infonce", variable_positive_symmetric_infonce),
)
_FORMAL_EXTERNAL_FUNCTION_IDENTITIES = (
    (
        execution_module,
        "verified_resume_checkpoint_payload_sha256",
        execution_module.verified_resume_checkpoint_payload_sha256,
    ),
    (torch.nn.utils, "clip_grad_norm_", torch.nn.utils.clip_grad_norm_),
    (torch.optim, "AdamW", torch.optim.AdamW),
    (torch.optim.lr_scheduler, "LambdaLR", torch.optim.lr_scheduler.LambdaLR),
)


def _autocast_context(decision: PrecisionDecision, device: torch.device):
    if not decision.autocast_enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)


class PhaseSetTrainingRuntime:
    """Deterministic optimizer/checkpoint loop for base or residual systems."""

    def __init__(
        self,
        system: BaseRetrievalSystem | ResidualRetrievalSystem,
        config: TrainingConfig,
        checkpoint_directory: str | Path,
        *,
        initialization_binding: TrainingInitializationBinding | None = None,
        checkpoint_observer: CheckpointObserver | None = None,
        descriptor_plan: PeriodicDescriptorCaptureTrainingPlan | None = None,
    ) -> None:
        if type(config) is not TrainingConfig:
            raise TypeError("config must be exact TrainingConfig")
        if config.stage == "base" and not isinstance(system, BaseRetrievalSystem):
            raise TrainingRuntimeError("base stage requires BaseRetrievalSystem")
        if config.stage == "residual" and not isinstance(system, ResidualRetrievalSystem):
            raise TrainingRuntimeError("residual stage requires ResidualRetrievalSystem")
        if initialization_binding is not None and type(initialization_binding) is not (
            TrainingInitializationBinding
        ):
            raise TypeError("initialization_binding must be exact TrainingInitializationBinding")
        if checkpoint_observer is not None and not callable(checkpoint_observer):
            raise TypeError("checkpoint_observer must be callable or None")
        if not config.synthetic_contract and initialization_binding is None:
            raise TrainingRuntimeError(
                "formal training requires a seed-bound initialization binding"
            )
        checked_plan, checked_plan_sha256 = _runtime_descriptor_plan_identity(
            descriptor_plan,
            system,
            config,
        )
        binding_plan_sha256 = (
            initialization_binding.periodic_descriptor_capture_plan_sha256
            if initialization_binding is not None
            else None
        )
        if binding_plan_sha256 != checked_plan_sha256:
            raise TrainingRuntimeError("initialization binding and descriptor plan identity differ")
        if not config.synthetic_contract:
            _validate_registered_formal_system(system, config)
        actual_initial_state = training_system_state_sha256(
            system,
            optimizable_only=True,
        )
        actual_behavior = training_system_behavior_sha256(system)
        if initialization_binding is not None:
            if (
                initialization_binding.stage != config.stage
                or initialization_binding.seed != config.seed
                or initialization_binding.system_id != system.system_id
                or initialization_binding.initial_optimizable_state_sha256 != actual_initial_state
                or initialization_binding.behavior_sha256 != actual_behavior
                or initialization_binding.code_artifact_sha256 != training_code_artifact_sha256()
                or initialization_binding.environment_sha256 != training_environment_sha256()
            ):
                raise TrainingRuntimeError(
                    "initialization binding does not match the live training system"
                )
            if isinstance(system, ResidualRetrievalSystem):
                if initialization_binding.frozen_base_state_sha256 != (
                    _frozen_base_score_state_sha256(
                        system.frozen_base,
                        float(system._frozen_base_logit_scale.detach().cpu().item()),
                    )
                ):
                    raise TrainingRuntimeError(
                        "initialization binding does not match the frozen base"
                    )
            elif (
                initialization_binding.frozen_base_checkpoint_sha256 is not None
                or initialization_binding.frozen_base_state_sha256 is not None
            ):
                raise TrainingRuntimeError("base runtime received a residual binding")
        root = Path(checkpoint_directory)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir() or root.is_symlink():
            raise TrainingCheckpointError("checkpoint directory must be a non-symlink directory")
        self.system = system
        self.config = config
        self.initialization_binding = initialization_binding
        self.descriptor_plan = checked_plan
        self.periodic_descriptor_capture_plan_sha256 = checked_plan_sha256
        self.initial_optimizable_state_sha256 = actual_initial_state
        self.behavior_sha256 = actual_behavior
        self.device = torch.device(config.device)
        self.system.to(self.device)
        self.precision = resolve_precision(config)
        self.checkpoint_directory = root.resolve()
        self.checkpoint_observer = checkpoint_observer
        self._optimizer: torch.optim.AdamW | None = None
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._state = _CursorState()
        self._latest: CheckpointArtifact | None = None
        self._train_manifest = ""
        self._val_manifest = ""
        self._updates_per_epoch = 0
        self._total_steps = 0
        self._numeric_runtime_active = False

    def _assert_live_descriptor_plan(self) -> None:
        plan, digest = _runtime_descriptor_plan_identity(
            self.descriptor_plan,
            self.system,
            self.config,
        )
        binding_digest = (
            self.initialization_binding.periodic_descriptor_capture_plan_sha256
            if self.initialization_binding is not None
            else None
        )
        if (
            plan is not self.descriptor_plan
            or digest != self.periodic_descriptor_capture_plan_sha256
            or binding_digest != digest
        ):
            raise TrainingRuntimeError("live periodic descriptor capture plan changed")

    def _assert_live_formal_binding(self, *, require_initial_state: bool) -> None:
        self._assert_live_descriptor_plan()
        if self.config.synthetic_contract:
            return
        if self._numeric_runtime_active:
            _assert_frozen_numerical_runtime(self.device)
        binding = self.initialization_binding
        if binding is None:
            raise TrainingRuntimeError("formal training lost its initialization binding")
        _validate_registered_formal_system(self.system, self.config)
        if training_code_artifact_sha256() != binding.code_artifact_sha256:
            raise TrainingRuntimeError("installed training code artifact changed")
        if training_environment_sha256() != binding.environment_sha256:
            raise TrainingRuntimeError("training environment changed")
        if training_system_behavior_sha256(self.system) != binding.behavior_sha256:
            raise TrainingRuntimeError("live formal behavior diverged from initialization binding")
        if (
            require_initial_state
            and training_system_state_sha256(
                self.system,
                optimizable_only=True,
            )
            != binding.initial_optimizable_state_sha256
        ):
            raise TrainingRuntimeError("live formal state diverged before optimizer creation")
        if isinstance(self.system, ResidualRetrievalSystem):
            frozen_state = _frozen_base_score_state_sha256(
                self.system.frozen_base,
                float(self.system._frozen_base_logit_scale.detach().cpu().item()),
            )
            if frozen_state != binding.frozen_base_state_sha256:
                raise TrainingRuntimeError("live frozen base diverged from initialization binding")

    def _trainable_parameters(self) -> list[nn.Parameter]:
        parameters = [
            parameter for parameter in self.system.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise TrainingRuntimeError("training system has no trainable parameters")
        return parameters

    def _initialize_optimizer(self, updates_per_epoch: int) -> None:
        if updates_per_epoch < 1:
            raise TrainingRuntimeError("training source must yield at least one update per epoch")
        self._updates_per_epoch = updates_per_epoch
        self._total_steps = self.config.epochs * updates_per_epoch
        optimizer = torch.optim.AdamW(
            self._trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

        def multiplier(step: int) -> float:
            return _learning_rate_multiplier(step, self._total_steps)

        self._optimizer = optimizer
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    @property
    def _checked_optimizer(self) -> torch.optim.AdamW:
        if self._optimizer is None:
            raise TrainingRuntimeError("optimizer is not initialized")
        return self._optimizer

    @property
    def _checked_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        if self._scheduler is None:
            raise TrainingRuntimeError("scheduler is not initialized")
        return self._scheduler

    def _seed_new_run(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _checkpoint_payload(self) -> dict[str, object]:
        state = self._state
        payload: dict[str, object] = {
            "schema": (
                CACHED_CHECKPOINT_SCHEMA if self.descriptor_plan is not None else CHECKPOINT_SCHEMA
            ),
            "authority": 0,
            "production": False,
            "result_claimed": False,
            "external_receipt_verified": False,
            "stage": self.config.stage,
            "seed": self.config.seed,
            "system_id": self.system.system_id,
            "config_sha256": self.config.sha256,
            "train_manifest_sha256": self._train_manifest,
            "val_manifest_sha256": self._val_manifest,
            "updates_per_epoch": self._updates_per_epoch,
            "total_steps": self._total_steps,
            "precision_mode": self.precision.mode,
            "initialization_binding_sha256": (
                self.initialization_binding.sha256
                if self.initialization_binding is not None
                else None
            ),
            "behavior_sha256": self.behavior_sha256,
            "initial_optimizable_state_sha256": self.initial_optimizable_state_sha256,
            "factory_sha256": (
                self.initialization_binding.factory_sha256
                if self.initialization_binding is not None
                else None
            ),
            "code_artifact_sha256": (
                self.initialization_binding.code_artifact_sha256
                if self.initialization_binding is not None
                else training_code_artifact_sha256()
            ),
            "environment_sha256": (
                self.initialization_binding.environment_sha256
                if self.initialization_binding is not None
                else training_environment_sha256()
            ),
            "frozen_base_checkpoint_sha256": (
                self.initialization_binding.frozen_base_checkpoint_sha256
                if self.initialization_binding is not None
                else None
            ),
            "frozen_base_state_sha256": (
                self.initialization_binding.frozen_base_state_sha256
                if self.initialization_binding is not None
                else None
            ),
            "qualified_base_selection_sha256": (
                self.initialization_binding.qualified_base_selection_sha256
                if self.initialization_binding is not None
                else None
            ),
            "residual_capacity_audit_sha256": (
                self.initialization_binding.residual_capacity_audit_sha256
                if self.initialization_binding is not None
                else None
            ),
            "epoch": state.epoch,
            "update_index": state.update_index,
            "global_step": state.global_step,
            "checkpoint_sequence": state.checkpoint_sequence,
            "train_edges_seen": state.train_edges_seen,
            "validation_edges_seen": state.validation_edges_seen,
            "max_microbatch_edges": state.max_microbatch_edges,
            "best_validation_metric": state.best_validation_metric,
            "best_checkpoint_name": state.best_checkpoint_name,
            "best_checkpoint_sha256": state.best_checkpoint_sha256,
            "last_train_loss": state.last_train_loss,
            "model": self.system.state_dict(),
            "optimizer": self._checked_optimizer.state_dict(),
            "scheduler": self._checked_scheduler.state_dict(),
            "rng": _capture_rng(),
        }
        if self.periodic_descriptor_capture_plan_sha256 is not None:
            payload["periodic_descriptor_capture_plan_sha256"] = (
                self.periodic_descriptor_capture_plan_sha256
            )
            payload["dataloader_state_sha256"] = _cached_dataloader_state_sha256(
                self._train_manifest,
                self._val_manifest,
                self.periodic_descriptor_capture_plan_sha256,
            )
        payload["state_digest"] = _stable_hash(payload)
        return payload

    def _write_checkpoint(self, reason: Literal["update", "validation"]) -> CheckpointArtifact:
        self._assert_live_formal_binding(require_initial_state=False)
        self._state.checkpoint_sequence += 1
        name = (
            f"checkpoint-{self._state.global_step:012d}-"
            f"{self._state.checkpoint_sequence:06d}-{reason}.pt"
        )
        if reason == "validation" and self._state.best_checkpoint_name == "__CURRENT__":
            self._state.best_checkpoint_name = name
        payload = self._checkpoint_payload()
        artifact = _atomic_torch_save(self.checkpoint_directory / name, payload)
        self._latest = artifact
        if self._state.best_checkpoint_name == name:
            self._state.best_checkpoint_sha256 = artifact.sha256
        if self.checkpoint_observer is not None:
            self.checkpoint_observer(artifact, reason)
        return artifact

    def _restore_checkpoint(
        self,
        path: Path,
        *,
        expected_sha256: str | None,
    ) -> None:
        payload, artifact = _load_torch_checkpoint(
            path,
            expected_sha256=expected_sha256,
        )
        digest = payload.pop("state_digest", None)
        if type(digest) is not str or digest != _stable_hash(payload):
            raise TrainingCheckpointError("checkpoint state digest mismatch")
        payload["state_digest"] = digest
        if self.descriptor_plan is None and "periodic_descriptor_capture_plan_sha256" in payload:
            raise TrainingCheckpointError(
                "uncached runtime cannot resume a cached descriptor checkpoint"
            )
        bindings = {
            "schema": (
                CACHED_CHECKPOINT_SCHEMA if self.descriptor_plan is not None else CHECKPOINT_SCHEMA
            ),
            "stage": self.config.stage,
            "seed": self.config.seed,
            "system_id": self.system.system_id,
            "config_sha256": self.config.sha256,
            "train_manifest_sha256": self._train_manifest,
            "val_manifest_sha256": self._val_manifest,
            "updates_per_epoch": self._updates_per_epoch,
            "total_steps": self._total_steps,
            "precision_mode": self.precision.mode,
            "behavior_sha256": self.behavior_sha256,
            "code_artifact_sha256": training_code_artifact_sha256(),
            "environment_sha256": training_environment_sha256(),
        }
        if self.periodic_descriptor_capture_plan_sha256 is not None:
            bindings["periodic_descriptor_capture_plan_sha256"] = (
                self.periodic_descriptor_capture_plan_sha256
            )
            bindings["dataloader_state_sha256"] = _cached_dataloader_state_sha256(
                self._train_manifest,
                self._val_manifest,
                self.periodic_descriptor_capture_plan_sha256,
            )
        if self.initialization_binding is None:
            if any(
                payload.get(key) is not None
                for key in (
                    "initialization_binding_sha256",
                    "factory_sha256",
                    "frozen_base_checkpoint_sha256",
                    "frozen_base_state_sha256",
                    "qualified_base_selection_sha256",
                    "residual_capacity_audit_sha256",
                )
            ):
                raise TrainingCheckpointError(
                    "synthetic checkpoint unexpectedly carries a formal initialization binding"
                )
            try:
                _lower_sha256(
                    payload.get("initial_optimizable_state_sha256"),
                    "initial_optimizable_state_sha256",
                )
            except TypeError as error:
                raise TrainingCheckpointError(
                    "checkpoint initial state digest is malformed"
                ) from error
        else:
            bindings.update(
                {
                    "initialization_binding_sha256": self.initialization_binding.sha256,
                    "initial_optimizable_state_sha256": (self.initial_optimizable_state_sha256),
                    "factory_sha256": self.initialization_binding.factory_sha256,
                    "code_artifact_sha256": self.initialization_binding.code_artifact_sha256,
                    "frozen_base_checkpoint_sha256": (
                        self.initialization_binding.frozen_base_checkpoint_sha256
                    ),
                    "frozen_base_state_sha256": (
                        self.initialization_binding.frozen_base_state_sha256
                    ),
                    "qualified_base_selection_sha256": (
                        self.initialization_binding.qualified_base_selection_sha256
                    ),
                    "residual_capacity_audit_sha256": (
                        self.initialization_binding.residual_capacity_audit_sha256
                    ),
                }
            )
        for key, expected in bindings.items():
            if payload.get(key) != expected:
                raise TrainingCheckpointError(f"checkpoint {key} binding mismatch")
        state = _validate_checkpoint_progress(
            payload,
            epochs=self.config.epochs,
            updates_per_epoch=self._updates_per_epoch,
            total_steps=self._total_steps,
            artifact_path=artifact.path,
            artifact_sha256=artifact.sha256,
        )
        _validate_restored_optimizer_scheduler(
            payload,
            optimizer=self._checked_optimizer,
            scheduler=self._checked_scheduler,
            config=self.config,
            global_step=state.global_step,
            total_steps=self._total_steps,
        )
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping):
            raise TrainingCheckpointError("checkpoint model state is malformed")
        try:
            self.system.load_state_dict(model_payload, strict=True)  # type: ignore[arg-type]
            self._checked_optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
            self._checked_scheduler.load_state_dict(payload["scheduler"])  # type: ignore[arg-type]
            rng = payload["rng"]
            if not isinstance(rng, Mapping):
                raise TypeError
            _restore_rng(rng)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise TrainingCheckpointError("checkpoint training state is malformed") from error
        if training_system_state_sha256(self.system) != _stable_hash(model_payload):
            raise TrainingCheckpointError("loaded model state differs from checkpoint bytes")
        if self._checked_scheduler.last_epoch != state.global_step:
            raise TrainingCheckpointError("loaded scheduler step differs from checkpoint cursor")
        self._state = state
        self._latest = artifact
        self._assert_live_formal_binding(require_initial_state=False)

    def _encode_cached_training_batch(
        self,
        batch: RetrievalTrainingBatch,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(self.system, ResidualRetrievalSystem):
            raise TrainingRuntimeError("cached descriptor plan requires a residual system")
        self._assert_live_descriptor_plan()
        plan = self.descriptor_plan
        if plan is None:
            raise TrainingRuntimeError("cached descriptor plan was lost")
        stream = plan.open_training_batch(batch)
        return self.system.encode_trainable_cached(batch.groups, stream)

    def _prepare_cache(
        self, microbatches: tuple[RetrievalTrainingBatch, ...]
    ) -> tuple[list[_CacheRow], Tensor, tuple[bytes, ...], tuple[bytes, ...]]:
        rows: list[_CacheRow] = []
        texts: list[Tensor] = []
        motion_ids: list[bytes] = []
        text_ids: list[bytes] = []
        self.system.train(True)
        for batch in microbatches:
            rng_before = _capture_rng()
            with torch.no_grad(), _autocast_context(self.precision, self.device):
                if isinstance(self.system, BaseRetrievalSystem):
                    primary = self.system.encode_trainable(batch.groups)
                    band_mask = None
                    base_embedding = None
                else:
                    if self.descriptor_plan is None:
                        primary, band_mask = self.system.encode_trainable(batch.groups)
                    else:
                        primary, band_mask = self._encode_cached_training_batch(batch)
                    base_embedding = self.system.encode_frozen_base(batch.groups)
            primary_leaf = primary.detach().float().contiguous().requires_grad_(True)
            rows.append(
                _CacheRow(
                    batch=batch,
                    primary_leaf=primary_leaf,
                    band_mask=band_mask,
                    base_embedding=base_embedding,
                    rng_before=rng_before,
                )
            )
            texts.append(batch.text_embeddings.to(self.device).contiguous())
            motion_ids.extend(batch.motion_positive_ids)
            text_ids.extend(batch.text_positive_ids)
        return rows, torch.cat(texts).contiguous(), tuple(motion_ids), tuple(text_ids)

    def _scores_from_cache(self, rows: list[_CacheRow], text: Tensor) -> Tensor:
        primary = torch.cat([row.primary_leaf for row in rows]).contiguous()
        if isinstance(self.system, BaseRetrievalSystem):
            return self.system.scores(primary, text)
        masks = torch.cat([row.band_mask for row in rows if row.band_mask is not None]).contiguous()
        bases = torch.cat(
            [row.base_embedding for row in rows if row.base_embedding is not None]
        ).contiguous()
        return self.system.scores(primary, masks, bases, text)

    def _train_update(self, microbatches: tuple[RetrievalTrainingBatch, ...]) -> float:
        optimizer = self._checked_optimizer
        optimizer.zero_grad(set_to_none=True)
        rows, text, motion_ids, text_ids = self._prepare_cache(microbatches)
        positive = _positive_mask(motion_ids, text_ids, self.device)
        logits = self._scores_from_cache(rows, text)
        _, _, loss = variable_positive_symmetric_infonce(logits, positive)
        if not bool(torch.isfinite(loss).item()):
            raise TrainingRuntimeError("training loss became nonfinite")
        # Capture after every stochastic part of the objective/head forward.
        # Replays restore encoder RNG snapshots, then return to this state so a
        # dropout-bearing head cannot repeat the same mask on the next update.
        rng_after = _capture_rng()
        loss.backward()
        try:
            for row in rows:
                gradient = row.primary_leaf.grad
                if gradient is None:
                    raise TrainingRuntimeError(
                        "gradient cache did not receive representation gradients"
                    )
                _restore_rng(row.rng_before)
                with _autocast_context(self.precision, self.device):
                    if isinstance(self.system, BaseRetrievalSystem):
                        replay = self.system.encode_trainable(row.batch.groups)
                    else:
                        if self.descriptor_plan is None:
                            replay, replay_mask = self.system.encode_trainable(row.batch.groups)
                        else:
                            replay, replay_mask = self._encode_cached_training_batch(row.batch)
                        if row.band_mask is None or not torch.equal(replay_mask, row.band_mask):
                            raise TrainingRuntimeError(
                                "residual replay changed its deterministic band mask"
                            )
                torch.autograd.backward(replay, gradient)
        finally:
            _restore_rng(rng_after)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._trainable_parameters(),
            GRADIENT_CLIP_NORM,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise TrainingRuntimeError("training gradients became nonfinite")
        optimizer.step()
        self._checked_scheduler.step()
        return float(loss.detach().cpu().item())

    def _validation(
        self,
        source: TrainingDataSource | CaptureValidationSource,
    ) -> tuple[float, float, int, int]:
        from .capture_validation import CaptureValidationSource, run_capture_validation

        run_capture_validation_cached = None

        if not self.config.synthetic_contract:
            (
                CaptureValidationSource,
                run_capture_validation,
                run_capture_validation_cached,
            ) = _scoped_capture_validation_functions()
        if type(source) is CaptureValidationSource:
            checked = CaptureValidationSource(source.split, source.manifest_sha256, source.captures)
            if checked.census_sha256 != self._val_manifest:
                raise TrainingRuntimeError("capture validation census changed after fit admission")
            edges = tuple(
                window.groups.actor_counts[0] * (window.groups.actor_counts[0] - 1) // 2
                for capture in checked.captures
                for window in capture.windows
            )
            if self.descriptor_plan is None:
                result = run_capture_validation(self.system, self.config, checked)
            else:
                self._assert_live_descriptor_plan()
                if run_capture_validation_cached is None:
                    raise TrainingRuntimeError("cached capture validation binding is unavailable")
                result = run_capture_validation_cached(
                    self.system,
                    self.config,
                    checked,
                    self.descriptor_plan,
                )
            if result.source_census_sha256 != self._val_manifest:
                raise TrainingRuntimeError("scored capture census differs from admitted validation")
            return float(result.primary_capture_r1), result.loss, sum(edges), max(edges)
        batches = _materialize_epoch(
            source,
            split="val",
            epoch=0,
            seed=self.config.seed,
            edge_budget=self.config.edge_budget,
        )
        self.system.eval()
        primary: list[Tensor] = []
        masks: list[Tensor] = []
        bases: list[Tensor] = []
        texts: list[Tensor] = []
        motion_ids: list[bytes] = []
        text_ids: list[bytes] = []
        motion_commitments: list[bytes] = []
        text_commitments: list[bytes] = []
        with torch.no_grad():
            for batch in batches:
                with _autocast_context(self.precision, self.device):
                    if isinstance(self.system, BaseRetrievalSystem):
                        encoded = self.system.encode_trainable(batch.groups)
                    else:
                        encoded, mask = self.system.encode_trainable(batch.groups)
                        masks.append(mask)
                        bases.append(self.system.encode_frozen_base(batch.groups))
                primary.append(encoded.float().contiguous())
                texts.append(batch.text_embeddings.to(self.device).contiguous())
                motion_ids.extend(batch.motion_positive_ids)
                text_ids.extend(batch.text_positive_ids)
                motion_commitments.extend(batch.groups.group_commitments)
                text_commitments.extend(batch.text_commitments)
            motion = torch.cat(primary).contiguous()
            text = torch.cat(texts).contiguous()
            positive = _positive_mask(tuple(motion_ids), tuple(text_ids), self.device)
            if isinstance(self.system, BaseRetrievalSystem):
                logits = self.system.scores(motion, text)
            else:
                logits = self.system.scores(
                    motion,
                    torch.cat(masks).contiguous(),
                    torch.cat(bases).contiguous(),
                    text,
                )
            _, _, loss = variable_positive_symmetric_infonce(logits, positive)
            metric = _capture_macro_bidirectional_r1(
                logits,
                positive,
                tuple(motion_ids),
                tuple(text_ids),
                tuple(motion_commitments),
                tuple(text_commitments),
            )
        edge_count = sum(batch.edge_count for batch in batches)
        maximum = max(batch.edge_count for batch in batches)
        return metric, float(loss.item()), edge_count, maximum

    def _artifact_for_name(self, name: str | None, sha256: str | None) -> CheckpointArtifact | None:
        if name is None:
            return None
        path = self.checkpoint_directory / name
        expected = sha256
        if expected is None:
            if self._latest is None or self._latest.path.name != name:
                raise TrainingCheckpointError(
                    "selected validation checkpoint lacks an immutable digest"
                )
            expected = self._latest.sha256
        _, artifact = _load_torch_checkpoint(path, expected_sha256=expected)
        return artifact

    def _report(self, status: Literal["COMPLETED", "INTERRUPTED"]) -> TrainingReport:
        if self._latest is None or self._latest.global_step != self._state.global_step:
            # A zero-update interruption is still given an exact resumable checkpoint.
            self._latest = self._write_checkpoint("update")
        best = self._artifact_for_name(
            self._state.best_checkpoint_name,
            self._state.best_checkpoint_sha256,
        )
        report = TrainingReport(
            status=status,
            stage=self.config.stage,
            system_id=self.system.system_id,
            global_step=self._state.global_step,
            completed_epochs=self._state.epoch,
            best_validation_metric=self._state.best_validation_metric,
            latest_checkpoint=self._latest,
            best_checkpoint=best,
            train_edges_seen=self._state.train_edges_seen,
            validation_edges_seen=self._state.validation_edges_seen,
            max_microbatch_edges=self._state.max_microbatch_edges,
            precision=self.precision,
            initialization_binding_sha256=(
                self.initialization_binding.sha256
                if self.initialization_binding is not None
                else None
            ),
            code_artifact_sha256=training_code_artifact_sha256(),
            environment_sha256=training_environment_sha256(),
            behavior_sha256=self.behavior_sha256,
            initial_optimizable_state_sha256=self.initial_optimizable_state_sha256,
            frozen_base_checkpoint_sha256=(
                self.initialization_binding.frozen_base_checkpoint_sha256
                if self.initialization_binding is not None
                else None
            ),
            frozen_base_state_sha256=(
                self.initialization_binding.frozen_base_state_sha256
                if self.initialization_binding is not None
                else None
            ),
            qualified_base_selection_sha256=(
                self.initialization_binding.qualified_base_selection_sha256
                if self.initialization_binding is not None
                else None
            ),
            residual_capacity_audit_sha256=(
                self.initialization_binding.residual_capacity_audit_sha256
                if self.initialization_binding is not None
                else None
            ),
            config_sha256=self.config.sha256,
            train_manifest_sha256=self._train_manifest,
            val_manifest_sha256=self._val_manifest,
            last_train_loss=self._state.last_train_loss,
        )
        if self.periodic_descriptor_capture_plan_sha256 is None:
            return report
        values = {
            field.name: getattr(report, field.name)
            for field in dataclass_fields(TrainingReport)
            if field.name != "schema"
        }
        return CachedResidualTrainingReport(
            **values,
            periodic_descriptor_capture_plan_sha256=(self.periodic_descriptor_capture_plan_sha256),
            dataloader_state_sha256=_cached_dataloader_state_sha256(
                self._train_manifest,
                self._val_manifest,
                self.periodic_descriptor_capture_plan_sha256,
            ),
        )

    def fit(
        self,
        train_source: TrainingDataSource,
        val_source: TrainingDataSource | CaptureValidationSource,
        *,
        resume_checkpoint: str | Path | None = None,
        resume_checkpoint_sha256: str | None = None,
        resume_attempt_root: str | Path | None = None,
        resume_record: ResumeRecord | None = None,
        stop_after_global_step: int | None = None,
    ) -> TrainingReport:
        """Fit until the frozen epoch count or a controlled interruption point.

        Validation alone selects the best checkpoint.  There is intentionally
        no test-source argument, and any source or batch labelled ``test`` is
        rejected before a model forward. A concrete CaptureValidationSource
        uses the complete holistic capture gallery; the existing window source
        remains available for the separately declared auxiliary task. Capture
        checkpoints bind the complete source census rather than actor-set IDs.
        """

        if self.descriptor_plan is not None:
            self._assert_live_descriptor_plan()
            train_source, val_source = _descriptor_plan_sources(
                self.descriptor_plan,
                train_source,
                val_source,
            )
        self._assert_live_formal_binding(require_initial_state=True)
        train, self._train_manifest = _source_identity(train_source, "train")
        val, self._val_manifest = _source_identity(val_source, "val")
        if self.descriptor_plan is not None and (
            self._train_manifest != self.descriptor_plan.train_source_manifest_sha256
            or self._val_manifest != self.descriptor_plan.capture_source_census_sha256
        ):
            raise TrainingRuntimeError(
                "runtime data identities differ from the cached descriptor plan"
            )
        first_epoch = _materialize_epoch(
            train,
            split="train",
            epoch=0,
            seed=self.config.seed,
            edge_budget=self.config.edge_budget,
        )
        first_groups = _group_effective_batches(
            first_epoch,
            self.config.effective_global_batch,
        )
        if not self.config.synthetic_contract and any(
            sum(batch.motion_count for batch in group) != self.config.effective_global_batch
            for group in first_groups
        ):
            raise TrainingRuntimeError(
                "formal sampler must emit only complete 128-window effective batches"
            )
        self._initialize_optimizer(len(first_groups))
        if resume_checkpoint is None:
            if any(
                value is not None
                for value in (
                    resume_checkpoint_sha256,
                    resume_attempt_root,
                    resume_record,
                )
            ):
                raise TrainingCheckpointError("resume binding was provided without a checkpoint")
            if any(self.checkpoint_directory.glob("checkpoint-*.pt")):
                raise TrainingCheckpointError("new run requires an empty checkpoint directory")
            self._seed_new_run()
        else:
            expected_checkpoint_sha256 = resume_checkpoint_sha256
            if not self.config.synthetic_contract:
                if resume_attempt_root is None or type(resume_record) is not ResumeRecord:
                    raise TrainingCheckpointError(
                        "formal resume requires a verified attempt-ledger binding"
                    )
                try:
                    ledger_sha256 = execution_module.verified_resume_checkpoint_payload_sha256(
                        resume_attempt_root,
                        resume_record,
                    )
                except (ExecutionContractError, TypeError) as error:
                    raise TrainingCheckpointError(
                        "formal resume attempt-ledger verification failed"
                    ) from error
                if (
                    expected_checkpoint_sha256 is not None
                    and expected_checkpoint_sha256 != ledger_sha256
                ):
                    raise TrainingCheckpointError(
                        "caller checkpoint digest differs from verified attempt ledger"
                    )
                expected_checkpoint_sha256 = ledger_sha256
            elif (resume_attempt_root is None) != (resume_record is None):
                raise TrainingCheckpointError(
                    "synthetic resume ledger root and record must be provided together"
                )
            elif resume_attempt_root is not None:
                try:
                    ledger_sha256 = execution_module.verified_resume_checkpoint_payload_sha256(
                        resume_attempt_root,
                        resume_record,
                    )
                except (ExecutionContractError, TypeError) as error:
                    raise TrainingCheckpointError(
                        "synthetic resume attempt-ledger verification failed"
                    ) from error
                if (
                    expected_checkpoint_sha256 is not None
                    and expected_checkpoint_sha256 != ledger_sha256
                ):
                    raise TrainingCheckpointError(
                        "caller checkpoint digest differs from verified attempt ledger"
                    )
                expected_checkpoint_sha256 = ledger_sha256
            self._restore_checkpoint(
                Path(resume_checkpoint),
                expected_sha256=expected_checkpoint_sha256,
            )
        if stop_after_global_step is not None:
            if type(stop_after_global_step) is not int or stop_after_global_step < 0:
                raise TypeError("stop_after_global_step must be an exact nonnegative int")
            if stop_after_global_step < self._state.global_step:
                raise TrainingRuntimeError("stop point precedes the resumed global step")
            if stop_after_global_step == self._state.global_step:
                return self._report("INTERRUPTED")

        with _frozen_numerical_runtime(self.device):
            self._numeric_runtime_active = True
            try:
                while self._state.epoch < self.config.epochs:
                    epoch = self._state.epoch
                    values = (
                        first_epoch
                        if epoch == 0
                        else _materialize_epoch(
                            train,
                            split="train",
                            epoch=epoch,
                            seed=self.config.seed,
                            edge_budget=self.config.edge_budget,
                        )
                    )
                    groups = _group_effective_batches(
                        values,
                        self.config.effective_global_batch,
                    )
                    if len(groups) != self._updates_per_epoch:
                        raise TrainingRuntimeError(
                            "updates per epoch changed after schedule freeze"
                        )
                    if not self.config.synthetic_contract and any(
                        sum(batch.motion_count for batch in group)
                        != self.config.effective_global_batch
                        for group in groups
                    ):
                        raise TrainingRuntimeError(
                            "formal sampler emitted a partial 128-window effective batch"
                        )
                    if self._state.update_index > len(groups):
                        raise TrainingCheckpointError("resumed update cursor exceeds epoch plan")
                    for update_index in range(self._state.update_index, len(groups)):
                        microbatches = groups[update_index]
                        loss = self._train_update(microbatches)
                        edges = sum(batch.edge_count for batch in microbatches)
                        maximum = max(batch.edge_count for batch in microbatches)
                        self._state.global_step += 1
                        self._state.update_index = update_index + 1
                        self._state.train_edges_seen += edges
                        self._state.max_microbatch_edges = max(
                            self._state.max_microbatch_edges,
                            maximum,
                        )
                        self._state.last_train_loss = loss
                        if self._state.global_step % self.config.checkpoint_every_updates == 0:
                            self._write_checkpoint("update")
                        if (
                            stop_after_global_step is not None
                            and self._state.global_step >= stop_after_global_step
                            and self._state.update_index < len(groups)
                        ):
                            return self._report("INTERRUPTED")

                    metric, _, val_edges, val_maximum = self._validation(val)
                    self._state.validation_edges_seen += val_edges
                    self._state.max_microbatch_edges = max(
                        self._state.max_microbatch_edges,
                        val_maximum,
                    )
                    improved = (
                        self._state.best_validation_metric is None
                        or metric > self._state.best_validation_metric
                    )
                    if improved:
                        self._state.best_validation_metric = metric
                        self._state.best_checkpoint_name = "__CURRENT__"
                        self._state.best_checkpoint_sha256 = None
                    self._state.epoch += 1
                    self._state.update_index = 0
                    self._write_checkpoint("validation")
                    if (
                        stop_after_global_step is not None
                        and self._state.global_step >= stop_after_global_step
                    ):
                        return self._report("INTERRUPTED")
                return self._report("COMPLETED")
            finally:
                self._numeric_runtime_active = False


__all__ = [
    "BASE_EPOCHS",
    "BASE_LEARNING_RATE",
    "BaseRetrievalSystem",
    "CachedResidualTrainingReport",
    "CHECKPOINT_SCHEMA",
    "CheckpointArtifact",
    "CheckpointObserver",
    "EFFECTIVE_GLOBAL_BATCH",
    "GRADIENT_CLIP_NORM",
    "OFFICIAL_SEEDS",
    "PhaseSetTrainingRuntime",
    "PrecisionDecision",
    "QualifiedFrozenBase",
    "REPORT_SCHEMA",
    "RESIDUAL_EPOCHS",
    "RESIDUAL_LEARNING_RATE",
    "REGISTERED_BASE_SYSTEM_IDS",
    "REGISTERED_DROPOUT",
    "REGISTERED_EMBEDDING_DIM",
    "REGISTERED_FFN_DIM",
    "REGISTERED_HEADS",
    "REGISTERED_PERIODIC_HIDDEN_DIM",
    "REGISTERED_TEXT_HIDDEN_DIM",
    "RESIDUAL_SYSTEM_IDS",
    "ResidualCapacityAudit",
    "ResidualRetrievalSystem",
    "RetrievalTrainingBatch",
    "STATUS",
    "TrainingCheckpointError",
    "TrainingConfig",
    "TrainingDataSource",
    "TrainingInitializationBinding",
    "TrainingReport",
    "TrainingRuntimeError",
    "WARMUP_FRACTION",
    "WEIGHT_DECAY",
    "audit_residual_training_parameter_counts",
    "build_registered_base_training_system",
    "build_registered_residual_capacity_audit",
    "build_registered_residual_training_system",
    "construct_registered_base_seed_bound_system",
    "construct_registered_residual_seed_bound_system",
    "construct_seed_bound_system",
    "load_qualified_frozen_base",
    "resolve_precision",
    "training_system_behavior_sha256",
    "training_code_artifact_sha256",
    "training_environment_sha256",
    "training_system_state_sha256",
]
