"""Sealed authority-zero cache plans for prepared-window residual training.

This private candidate derives one exact plan from already-authenticated v2
prepared sources and periodic descriptor readers.  It does not construct a
model, authorize data, run training, or alter checkpoint/report schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import random
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
import torch

from .controls import system_spec
from .periodic import ResourceLimitError
from .periodic_descriptor_cache_v2 import (
    CachedDescriptorBatch,
    CachedPairChunkStream,
    DescriptorCacheSourceBinding,
    DescriptorStreamKind,
    DescriptorWindowContext,
    PeriodicDescriptorCacheV2,
    prepared_group_batch_sha256,
)
from .prepared_data_v2 import PreparedTrainingDataSourceV2
from .training import (
    OFFICIAL_SEEDS,
    RESIDUAL_EPOCHS,
    RetrievalTrainingBatch,
    TrainingConfig,
)


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
STATUS: Final = "PRIVATE_STATIC_CACHE_PLAN_CANDIDATE_UNEXECUTED"
PLAN_SCHEMA: Final = "phaseset-periodic-training-cache-plan-v1"
WINDOW_VALIDATION_SEMANTICS: Final = (
    "prepared-window-engineering-only-not-holistic-capture-validation"
)

_PLAN_TOKEN: Final = object()
_CACHEABLE_SYSTEMS: Final[Mapping[str, DescriptorStreamKind]] = MappingProxyType(
    {
        "02": "MARGINAL_POWER",
        "03": "MEAN_DIFFERENCE_DCT",
        "04": "FULL_RELATION",
        "06": "FULL_RELATION",
        "07": "FULL_RELATION",
        "08": "FULL_RELATION",
    }
)
_EXPECTED_POLICIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "02": "MARGINAL_POWER",
        "03": "MEAN_DIFFERENCE_DCT",
        "04": "FULL_RELATION",
        "06": "SHUFFLED_INCIDENCE",
        "07": "PHASE_STRIPPED",
        "08": "FULL_RELATION",
    }
)

Split = Literal["train", "val"]
RequestIdentity = tuple[str, int, int, str, str, str, int, int]


class PeriodicTrainingCacheError(ValueError):
    """An exact training cache plan cannot be admitted or selected."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        + b"\n"
    )


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _source_binding_value(binding: DescriptorCacheSourceBinding) -> dict[str, str]:
    return {
        "environment_sha256": binding.environment_sha256,
        "prepared_manifest_sha256": binding.prepared_manifest_sha256,
        "source_tree_sha256": binding.source_tree_sha256,
    }


def _context_value(context: DescriptorWindowContext) -> dict[str, object]:
    return {
        "augmentation_yaw_be_hex": context.augmentation_yaw_be_hex,
        "epoch": context.epoch,
        "prepared_manifest_sha256": context.prepared_manifest_sha256,
        "seed": context.seed,
        "source_batch_sha256": context.source_batch_sha256,
        "split": context.split,
        "window_ordinal": context.window_ordinal,
        "window_sha256": context.window_sha256,
    }


def _contexts_sha256(contexts: tuple[DescriptorWindowContext, ...]) -> str:
    return _sha256(_canonical_json({"windows": [_context_value(row) for row in contexts]}))


def _request_sha256(
    *,
    split: Split,
    epoch: int,
    seed: int,
    source_batch_sha256: str,
    contexts_sha256: str,
    batch_input_sha256: str,
    edge_count: int,
    window_count: int,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "batch_input_sha256": batch_input_sha256,
                "contexts_sha256": contexts_sha256,
                "edge_count": edge_count,
                "epoch": epoch,
                "seed": seed,
                "source_batch_sha256": source_batch_sha256,
                "split": split,
                "window_count": window_count,
            }
        )
    )


def _request_identity(
    *,
    split: Split,
    epoch: int,
    seed: int,
    source_batch_sha256: str,
    contexts_sha256: str,
    batch_input_sha256: str,
    edge_count: int,
    window_count: int,
) -> RequestIdentity:
    return (
        split,
        epoch,
        seed,
        source_batch_sha256,
        contexts_sha256,
        batch_input_sha256,
        edge_count,
        window_count,
    )


@dataclass(frozen=True, slots=True)
class PeriodicDescriptorPlanRow:
    """One actual prepared-source request proven against one descriptor shard."""

    split: Split
    epoch: int
    seed: int
    execution_position: int
    source_batch_sha256: str
    contexts_sha256: str
    batch_input_sha256: str
    cache_key_sha256: str
    request_sha256: str
    edge_count: int
    window_count: int

    def __post_init__(self) -> None:
        if self.split not in ("train", "val"):
            raise PeriodicTrainingCacheError("plan row split must be train or val")
        for name in ("epoch", "seed", "execution_position", "edge_count", "window_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative int")
        if self.edge_count < 1 or self.window_count < 1:
            raise PeriodicTrainingCacheError("plan rows require windows and edges")
        for name in (
            "source_batch_sha256",
            "contexts_sha256",
            "batch_input_sha256",
            "cache_key_sha256",
            "request_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        expected = _request_sha256(
            split=self.split,
            epoch=self.epoch,
            seed=self.seed,
            source_batch_sha256=self.source_batch_sha256,
            contexts_sha256=self.contexts_sha256,
            batch_input_sha256=self.batch_input_sha256,
            edge_count=self.edge_count,
            window_count=self.window_count,
        )
        if self.request_sha256 != expected:
            raise PeriodicTrainingCacheError("plan row request digest is inconsistent")

    def _value(self) -> dict[str, object]:
        return {
            "batch_input_sha256": self.batch_input_sha256,
            "cache_key_sha256": self.cache_key_sha256,
            "contexts_sha256": self.contexts_sha256,
            "edge_count": self.edge_count,
            "epoch": self.epoch,
            "execution_position": self.execution_position,
            "request_sha256": self.request_sha256,
            "seed": self.seed,
            "source_batch_sha256": self.source_batch_sha256,
            "split": self.split,
            "window_count": self.window_count,
        }

    def _identity(self) -> RequestIdentity:
        return _request_identity(
            split=self.split,
            epoch=self.epoch,
            seed=self.seed,
            source_batch_sha256=self.source_batch_sha256,
            contexts_sha256=self.contexts_sha256,
            batch_input_sha256=self.batch_input_sha256,
            edge_count=self.edge_count,
            window_count=self.window_count,
        )


@dataclass(frozen=True, slots=True)
class _RngSnapshot:
    python_state: object
    numpy_state: tuple[object, ...]
    torch_cpu_state: torch.Tensor
    cuda_initialized: bool
    torch_cuda_states: tuple[torch.Tensor, ...]


def _capture_rng() -> _RngSnapshot:
    numpy_state = np.random.get_state()
    frozen_numpy = (
        numpy_state[0],
        np.array(numpy_state[1], copy=True),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    cuda_initialized = torch.cuda.is_initialized()
    cuda_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if cuda_initialized else ()
    )
    return _RngSnapshot(
        python_state=random.getstate(),
        numpy_state=frozen_numpy,
        torch_cpu_state=torch.random.get_rng_state().clone(),
        cuda_initialized=cuda_initialized,
        torch_cuda_states=cuda_states,
    )


def _rng_matches(snapshot: _RngSnapshot) -> bool:
    current_numpy = np.random.get_state()
    if random.getstate() != snapshot.python_state:
        return False
    if (
        current_numpy[0] != snapshot.numpy_state[0]
        or not np.array_equal(current_numpy[1], snapshot.numpy_state[1])
        or current_numpy[2:] != snapshot.numpy_state[2:]
        or not torch.equal(torch.random.get_rng_state(), snapshot.torch_cpu_state)
        or torch.cuda.is_initialized() is not snapshot.cuda_initialized
    ):
        return False
    if not snapshot.cuda_initialized:
        return True
    current_cuda = tuple(torch.cuda.get_rng_state_all())
    return len(current_cuda) == len(snapshot.torch_cuda_states) and all(
        torch.equal(left, right)
        for left, right in zip(current_cuda, snapshot.torch_cuda_states, strict=True)
    )


def _restore_rng(snapshot: _RngSnapshot) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)  # type: ignore[arg-type]
    torch.random.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.cuda_initialized:
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda_states))


@dataclass(frozen=True, slots=True)
class PeriodicDescriptorTrainingPlan:
    """Sealed metadata plus readers for one exact formal residual seed."""

    system_id: str
    seed: int
    epochs: int
    stream_kind: DescriptorStreamKind
    config_sha256: str
    edge_budget: int
    energy_floors_sha256: str
    train_source_manifest_sha256: str
    validation_source_manifest_sha256: str
    train_cache_index_sha256: str
    validation_cache_index_sha256: str
    train_cache_source_binding: DescriptorCacheSourceBinding
    validation_cache_source_binding: DescriptorCacheSourceBinding
    train_cache_key_census: tuple[str, ...]
    validation_cache_key_census: tuple[str, ...]
    train_rows: tuple[PeriodicDescriptorPlanRow, ...]
    validation_rows: tuple[PeriodicDescriptorPlanRow, ...]
    _train_cache: PeriodicDescriptorCacheV2 = field(repr=False, compare=False)
    _validation_cache: PeriodicDescriptorCacheV2 = field(repr=False, compare=False)
    _train_lookup: Mapping[RequestIdentity, PeriodicDescriptorPlanRow] = field(
        repr=False, compare=False
    )
    _validation_lookup: Mapping[RequestIdentity, PeriodicDescriptorPlanRow] = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PLAN_TOKEN:
            raise PeriodicTrainingCacheError(
                "training cache plans must come from complete plan admission"
            )
        if self.system_id not in _CACHEABLE_SYSTEMS:
            raise PeriodicTrainingCacheError("plan system is not cacheable")
        if self.seed not in OFFICIAL_SEEDS or self.epochs != RESIDUAL_EPOCHS:
            raise PeriodicTrainingCacheError("plan seed or epoch census changed")
        if self.stream_kind != _CACHEABLE_SYSTEMS[self.system_id]:
            raise PeriodicTrainingCacheError("plan stream differs from its system")
        if type(self.edge_budget) is not int or self.edge_budget < 1:
            raise TypeError("edge_budget must be an exact positive int")
        for name in (
            "config_sha256",
            "energy_floors_sha256",
            "train_source_manifest_sha256",
            "validation_source_manifest_sha256",
            "train_cache_index_sha256",
            "validation_cache_index_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        for rows, split in ((self.train_rows, "train"), (self.validation_rows, "val")):
            if (
                type(rows) is not tuple
                or not rows
                or any(type(row) is not PeriodicDescriptorPlanRow for row in rows)
                or any(row.split != split for row in rows)
            ):
                raise PeriodicTrainingCacheError("plan row census is invalid")
        all_rows = self.train_rows + self.validation_rows
        if tuple(row.execution_position for row in all_rows) != tuple(range(len(all_rows))):
            raise PeriodicTrainingCacheError("plan execution positions are not canonical")
        if len({row.cache_key_sha256 for row in self.train_rows}) != len(self.train_rows):
            raise PeriodicTrainingCacheError("training plan repeats a cache key")
        if len({row.cache_key_sha256 for row in self.validation_rows}) != len(self.validation_rows):
            raise PeriodicTrainingCacheError("validation plan repeats a cache key")
        if tuple(sorted(row.cache_key_sha256 for row in self.train_rows)) != (
            self.train_cache_key_census
        ):
            raise PeriodicTrainingCacheError("training rows differ from cache key census")
        if tuple(sorted(row.cache_key_sha256 for row in self.validation_rows)) != (
            self.validation_cache_key_census
        ):
            raise PeriodicTrainingCacheError("validation rows differ from cache key census")
        for census, label in (
            (self.train_cache_key_census, "train"),
            (self.validation_cache_key_census, "validation"),
        ):
            if (
                type(census) is not tuple
                or not census
                or census != tuple(sorted(census))
                or len(set(census)) != len(census)
            ):
                raise PeriodicTrainingCacheError(f"{label} cache key census is invalid")
            for index, digest in enumerate(census):
                _lower_sha256(digest, f"{label} cache key {index}")
        if set(self._train_lookup) != {row._identity() for row in self.train_rows}:
            raise PeriodicTrainingCacheError("training lookup differs from plan rows")
        if set(self._validation_lookup) != {row._identity() for row in self.validation_rows}:
            raise PeriodicTrainingCacheError("validation lookup differs from plan rows")
        if any(self._train_lookup[row._identity()] != row for row in self.train_rows):
            raise PeriodicTrainingCacheError("training lookup values differ from plan rows")
        if any(self._validation_lookup[row._identity()] != row for row in self.validation_rows):
            raise PeriodicTrainingCacheError("validation lookup values differ from plan rows")

    @property
    def authority(self) -> int:
        return AUTHORITY

    @property
    def production(self) -> bool:
        return PRODUCTION

    @property
    def result_claimed(self) -> bool:
        return RESULT_CLAIMED

    def canonical_bytes(self) -> bytes:
        value = {
            "authority": AUTHORITY,
            "config_sha256": self.config_sha256,
            "edge_budget": self.edge_budget,
            "energy_floors_sha256": self.energy_floors_sha256,
            "epochs": self.epochs,
            "external_authentication_asserted": False,
            "production": PRODUCTION,
            "result_claimed": RESULT_CLAIMED,
            "schema": PLAN_SCHEMA,
            "seed": self.seed,
            "stream_kind": self.stream_kind,
            "system_id": self.system_id,
            "train_cache_index_sha256": self.train_cache_index_sha256,
            "train_cache_key_census": list(self.train_cache_key_census),
            "train_cache_source_binding": _source_binding_value(self.train_cache_source_binding),
            "train_rows": [row._value() for row in self.train_rows],
            "train_source_manifest_sha256": self.train_source_manifest_sha256,
            "validation_cache_index_sha256": self.validation_cache_index_sha256,
            "validation_cache_key_census": list(self.validation_cache_key_census),
            "validation_cache_source_binding": _source_binding_value(
                self.validation_cache_source_binding
            ),
            "validation_rows": [row._value() for row in self.validation_rows],
            "validation_source_manifest_sha256": (self.validation_source_manifest_sha256),
            "window_validation_semantics": WINDOW_VALIDATION_SEMANTICS,
        }
        return _canonical_json(value)

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    def open_training_batch(self, batch: RetrievalTrainingBatch) -> CachedPairChunkStream:
        """Reopen and verify one admitted training shard without fallback."""

        return self._open_batch(
            batch,
            split="train",
            cache=self._train_cache,
            lookup=self._train_lookup,
            manifest_sha256=self.train_source_manifest_sha256,
            cache_index_sha256=self.train_cache_index_sha256,
            source_binding=self.train_cache_source_binding,
            cache_key_census=self.train_cache_key_census,
        )

    def open_validation_batch(self, batch: RetrievalTrainingBatch) -> CachedPairChunkStream:
        """Reopen one admitted prepared-window validation shard without fallback."""

        return self._open_batch(
            batch,
            split="val",
            cache=self._validation_cache,
            lookup=self._validation_lookup,
            manifest_sha256=self.validation_source_manifest_sha256,
            cache_index_sha256=self.validation_cache_index_sha256,
            source_binding=self.validation_cache_source_binding,
            cache_key_census=self.validation_cache_key_census,
        )

    def _open_batch(
        self,
        batch: RetrievalTrainingBatch,
        *,
        split: Split,
        cache: PeriodicDescriptorCacheV2,
        lookup: Mapping[RequestIdentity, PeriodicDescriptorPlanRow],
        manifest_sha256: str,
        cache_index_sha256: str,
        source_binding: DescriptorCacheSourceBinding,
        cache_key_census: tuple[str, ...],
    ) -> CachedPairChunkStream:
        checked, identity = _batch_identity(
            batch,
            split=split,
            seed=self.seed,
            expected_manifest_sha256=manifest_sha256,
            epochs=self.epochs,
        )
        row = lookup.get(identity)
        if row is None:
            raise PeriodicTrainingCacheError("batch is not an exact admitted plan row")
        _validate_reader_snapshot(
            cache,
            index_sha256=cache_index_sha256,
            source_binding=source_binding,
            floors_sha256=self.energy_floors_sha256,
            cache_key_census=cache_key_census,
        )
        _enforce_edge_budget(row.edge_count, self.edge_budget)
        contexts = checked.descriptor_contexts
        assert contexts is not None
        opened = cache.open_batch(checked.groups, contexts)
        _validate_opened(
            opened,
            row=row,
            contexts=contexts,
            floors_sha256=self.energy_floors_sha256,
        )
        stream = opened.stream(self.stream_kind)
        if type(stream) is not CachedPairChunkStream:
            raise PeriodicTrainingCacheError("cache returned an unsealed descriptor stream")
        return stream


def _validate_reader_snapshot(
    cache: PeriodicDescriptorCacheV2,
    *,
    index_sha256: str,
    source_binding: DescriptorCacheSourceBinding,
    floors_sha256: str,
    cache_key_census: tuple[str, ...],
) -> None:
    if type(cache) is not PeriodicDescriptorCacheV2:
        raise TypeError("cache must be an exact PeriodicDescriptorCacheV2")
    if (
        cache.index_sha256 != index_sha256
        or type(cache.source_binding) is not DescriptorCacheSourceBinding
        or cache.source_binding != source_binding
        or cache.energy_floors_sha256 != floors_sha256
        or cache.cache_key_census != cache_key_census
    ):
        raise PeriodicTrainingCacheError("cache reader metadata changed after admission")


def _enforce_edge_budget(edge_count: int, edge_budget: int) -> None:
    if edge_count > edge_budget:
        raise ResourceLimitError(required_edges=edge_count, edge_budget=edge_budget)


def _batch_identity(
    batch: RetrievalTrainingBatch,
    *,
    split: Split,
    seed: int,
    expected_manifest_sha256: str,
    epochs: int,
) -> tuple[RetrievalTrainingBatch, RequestIdentity]:
    if type(batch) is not RetrievalTrainingBatch:
        raise TypeError("cache plan requires an exact RetrievalTrainingBatch")
    if batch.split != split:
        raise PeriodicTrainingCacheError("batch split differs from selector")
    contexts = batch.descriptor_contexts
    if (
        type(contexts) is not tuple
        or not contexts
        or any(type(context) is not DescriptorWindowContext for context in contexts)
    ):
        raise PeriodicTrainingCacheError("batch lacks exact descriptor contexts")
    first = contexts[0]
    if (
        first.prepared_manifest_sha256 != expected_manifest_sha256
        or first.seed != seed
        or first.split != split
        or (split == "train" and not 0 <= first.epoch < epochs)
        or (split == "val" and first.epoch != 0)
    ):
        raise PeriodicTrainingCacheError("batch descriptor context differs from the plan")
    if any(
        context.prepared_manifest_sha256 != expected_manifest_sha256
        or context.seed != seed
        or context.split != split
        or context.epoch != first.epoch
        or context.source_batch_sha256 != first.source_batch_sha256
        for context in contexts
    ):
        raise PeriodicTrainingCacheError("batch mixes descriptor context identities")
    edge_count = sum(count * (count - 1) // 2 for count in batch.groups.actor_counts)
    if edge_count < 1:
        raise PeriodicTrainingCacheError("descriptor cache rows require at least one edge")
    contexts_digest = _contexts_sha256(contexts)
    batch_digest = prepared_group_batch_sha256(batch.groups)
    identity = _request_identity(
        split=split,
        epoch=first.epoch,
        seed=seed,
        source_batch_sha256=first.source_batch_sha256,
        contexts_sha256=contexts_digest,
        batch_input_sha256=batch_digest,
        edge_count=edge_count,
        window_count=len(contexts),
    )
    return batch, identity


def _validate_opened(
    opened: CachedDescriptorBatch,
    *,
    row: PeriodicDescriptorPlanRow,
    contexts: tuple[DescriptorWindowContext, ...],
    floors_sha256: str,
) -> None:
    if type(opened) is not CachedDescriptorBatch:
        raise PeriodicTrainingCacheError("cache returned an unsealed descriptor batch")
    if (
        opened.cache_key_sha256 != row.cache_key_sha256
        or opened.batch_input_sha256 != row.batch_input_sha256
        or opened.energy_floors_sha256 != floors_sha256
        or opened.contexts != contexts
        or opened.edge_count != row.edge_count
    ):
        raise PeriodicTrainingCacheError("opened cache shard differs from its plan row")


def _plan_row(
    batch: RetrievalTrainingBatch,
    opened: CachedDescriptorBatch,
    *,
    split: Split,
    seed: int,
    execution_position: int,
) -> PeriodicDescriptorPlanRow:
    contexts = batch.descriptor_contexts
    assert contexts is not None
    first = contexts[0]
    contexts_digest = _contexts_sha256(contexts)
    batch_digest = prepared_group_batch_sha256(batch.groups)
    request_digest = _request_sha256(
        split=split,
        epoch=first.epoch,
        seed=seed,
        source_batch_sha256=first.source_batch_sha256,
        contexts_sha256=contexts_digest,
        batch_input_sha256=batch_digest,
        edge_count=opened.edge_count,
        window_count=len(contexts),
    )
    return PeriodicDescriptorPlanRow(
        split=split,
        epoch=first.epoch,
        seed=seed,
        execution_position=execution_position,
        source_batch_sha256=first.source_batch_sha256,
        contexts_sha256=contexts_digest,
        batch_input_sha256=batch_digest,
        cache_key_sha256=opened.cache_key_sha256,
        request_sha256=request_digest,
        edge_count=opened.edge_count,
        window_count=len(contexts),
    )


def _enumerate_rows(
    source: PreparedTrainingDataSourceV2,
    cache: PeriodicDescriptorCacheV2,
    *,
    split: Split,
    seed: int,
    epochs: tuple[int, ...],
    stream_kind: DescriptorStreamKind,
    edge_budget: int,
    start_position: int,
    source_binding: DescriptorCacheSourceBinding,
    floors_sha256: str,
    index_sha256: str,
    cache_key_census: tuple[str, ...],
) -> tuple[
    tuple[PeriodicDescriptorPlanRow, ...], Mapping[RequestIdentity, PeriodicDescriptorPlanRow]
]:
    rows: list[PeriodicDescriptorPlanRow] = []
    lookup: dict[RequestIdentity, PeriodicDescriptorPlanRow] = {}
    keys: set[str] = set()
    for epoch in epochs:
        for batch in source.iter_epoch(epoch=epoch, seed=seed):
            if len(rows) >= len(cache_key_census):
                raise PeriodicTrainingCacheError(
                    f"{split} source row census exceeds its closed cache census"
                )
            checked, identity = _batch_identity(
                batch,
                split=split,
                seed=seed,
                expected_manifest_sha256=source.manifest_sha256,
                epochs=RESIDUAL_EPOCHS,
            )
            edge_count = identity[-2]
            _enforce_edge_budget(edge_count, edge_budget)
            _validate_reader_snapshot(
                cache,
                index_sha256=index_sha256,
                source_binding=source_binding,
                floors_sha256=floors_sha256,
                cache_key_census=cache_key_census,
            )
            contexts = checked.descriptor_contexts
            assert contexts is not None
            opened = cache.open_batch(checked.groups, contexts)
            if opened.energy_floors_sha256 != floors_sha256:
                raise PeriodicTrainingCacheError("opened shard energy floors changed")
            stream = opened.stream(stream_kind)
            if type(stream) is not CachedPairChunkStream:
                raise PeriodicTrainingCacheError("cache returned an unsealed descriptor stream")
            row = _plan_row(
                checked,
                opened,
                split=split,
                seed=seed,
                execution_position=start_position + len(rows),
            )
            _validate_opened(
                opened,
                row=row,
                contexts=contexts,
                floors_sha256=floors_sha256,
            )
            if row._identity() != identity:
                raise PeriodicTrainingCacheError(
                    "opened cache shard request differs from the actual source batch"
                )
            if identity in lookup or row.cache_key_sha256 in keys:
                raise PeriodicTrainingCacheError(
                    f"{split} execution plan repeats an exact descriptor request"
                )
            rows.append(row)
            lookup[identity] = row
            keys.add(row.cache_key_sha256)
    if not rows:
        raise PeriodicTrainingCacheError(f"{split} source produced no cache plan rows")
    if tuple(sorted(keys)) != cache_key_census:
        raise PeriodicTrainingCacheError(
            f"{split} required cache-key set differs from the closed reader census"
        )
    return tuple(rows), MappingProxyType(lookup)


def _admit_periodic_descriptor_training_plan(
    *,
    system_id: str,
    config: TrainingConfig,
    train_source: PreparedTrainingDataSourceV2,
    val_source: PreparedTrainingDataSourceV2,
    train_cache: PeriodicDescriptorCacheV2,
    val_cache: PeriodicDescriptorCacheV2,
) -> PeriodicDescriptorTrainingPlan:
    if type(config) is not TrainingConfig:
        raise TypeError("config must be an exact TrainingConfig")
    if config.stage != "residual" or config.synthetic_contract:
        raise PeriodicTrainingCacheError(
            "descriptor plans require an exact formal residual configuration"
        )
    if config.seed not in OFFICIAL_SEEDS or config.epochs != RESIDUAL_EPOCHS:
        raise PeriodicTrainingCacheError("descriptor plan seed or epoch census changed")
    if type(system_id) is not str or system_id not in _CACHEABLE_SYSTEMS:
        raise PeriodicTrainingCacheError(
            "descriptor plans support residual systems 02, 03, 04, 06, 07, and 08 only"
        )
    stream_kind = _CACHEABLE_SYSTEMS[system_id]
    specification = system_spec(system_id)
    if (
        specification.feature_policy != _EXPECTED_POLICIES[system_id]
        or specification.residual_enabled is not True
    ):
        raise PeriodicTrainingCacheError("registered system feature policy changed")
    if (
        type(train_source) is not PreparedTrainingDataSourceV2
        or type(val_source) is not PreparedTrainingDataSourceV2
    ):
        raise TypeError("sources must be exact PreparedTrainingDataSourceV2 values")
    if train_source is val_source or train_source.split != "train" or val_source.split != "val":
        raise PeriodicTrainingCacheError("plan requires distinct train and validation sources")
    if (
        type(train_cache) is not PeriodicDescriptorCacheV2
        or type(val_cache) is not PeriodicDescriptorCacheV2
    ):
        raise TypeError("caches must be exact PeriodicDescriptorCacheV2 readers")
    if train_cache is val_cache:
        raise PeriodicTrainingCacheError("train and validation require distinct cache readers")
    train_binding = train_cache.source_binding
    validation_binding = val_cache.source_binding
    if (
        type(train_binding) is not DescriptorCacheSourceBinding
        or type(validation_binding) is not DescriptorCacheSourceBinding
    ):
        raise PeriodicTrainingCacheError("cache source binding type changed")
    if (
        train_binding.prepared_manifest_sha256 != train_source.manifest_sha256
        or validation_binding.prepared_manifest_sha256 != val_source.manifest_sha256
    ):
        raise PeriodicTrainingCacheError("cache source binding differs from prepared source")
    if train_cache.energy_floors_sha256 != val_cache.energy_floors_sha256:
        raise PeriodicTrainingCacheError("train and validation cache energy floors differ")
    if not np.array_equal(train_cache.energy_floors, val_cache.energy_floors):
        raise PeriodicTrainingCacheError("train and validation energy-floor bytes differ")
    train_census = train_cache.cache_key_census
    validation_census = val_cache.cache_key_census
    for census, label in ((train_census, "train"), (validation_census, "validation")):
        if (
            type(census) is not tuple
            or not census
            or census != tuple(sorted(census))
            or len(set(census)) != len(census)
            or any(type(value) is not str for value in census)
        ):
            raise PeriodicTrainingCacheError(f"{label} cache key census is invalid")
    config_sha256 = config.sha256
    train_manifest = train_source.manifest_sha256
    validation_manifest = val_source.manifest_sha256
    floors_sha256 = train_cache.energy_floors_sha256
    train_index = train_cache.index_sha256
    validation_index = val_cache.index_sha256
    train_rows, train_lookup = _enumerate_rows(
        train_source,
        train_cache,
        split="train",
        seed=config.seed,
        epochs=tuple(range(RESIDUAL_EPOCHS)),
        stream_kind=stream_kind,
        edge_budget=config.edge_budget,
        start_position=0,
        source_binding=train_binding,
        floors_sha256=floors_sha256,
        index_sha256=train_index,
        cache_key_census=train_census,
    )
    validation_rows, validation_lookup = _enumerate_rows(
        val_source,
        val_cache,
        split="val",
        seed=config.seed,
        epochs=(0,),
        stream_kind=stream_kind,
        edge_budget=config.edge_budget,
        start_position=len(train_rows),
        source_binding=validation_binding,
        floors_sha256=floors_sha256,
        index_sha256=validation_index,
        cache_key_census=validation_census,
    )
    if (
        config.sha256 != config_sha256
        or train_source.manifest_sha256 != train_manifest
        or val_source.manifest_sha256 != validation_manifest
        or train_source.split != "train"
        or val_source.split != "val"
    ):
        raise PeriodicTrainingCacheError(
            "configuration or source identity changed during admission"
        )
    _validate_reader_snapshot(
        train_cache,
        index_sha256=train_index,
        source_binding=train_binding,
        floors_sha256=floors_sha256,
        cache_key_census=train_census,
    )
    _validate_reader_snapshot(
        val_cache,
        index_sha256=validation_index,
        source_binding=validation_binding,
        floors_sha256=floors_sha256,
        cache_key_census=validation_census,
    )
    return PeriodicDescriptorTrainingPlan(
        system_id=system_id,
        seed=config.seed,
        epochs=RESIDUAL_EPOCHS,
        stream_kind=stream_kind,
        config_sha256=config_sha256,
        edge_budget=config.edge_budget,
        energy_floors_sha256=floors_sha256,
        train_source_manifest_sha256=train_manifest,
        validation_source_manifest_sha256=validation_manifest,
        train_cache_index_sha256=train_index,
        validation_cache_index_sha256=validation_index,
        train_cache_source_binding=train_binding,
        validation_cache_source_binding=validation_binding,
        train_cache_key_census=train_census,
        validation_cache_key_census=validation_census,
        train_rows=train_rows,
        validation_rows=validation_rows,
        _train_cache=train_cache,
        _validation_cache=val_cache,
        _train_lookup=train_lookup,
        _validation_lookup=validation_lookup,
        _seal=_PLAN_TOKEN,
    )


def admit_periodic_descriptor_training_plan(
    *,
    system_id: str,
    config: TrainingConfig,
    train_source: PreparedTrainingDataSourceV2,
    val_source: PreparedTrainingDataSourceV2,
    train_cache: PeriodicDescriptorCacheV2,
    val_cache: PeriodicDescriptorCacheV2,
) -> PeriodicDescriptorTrainingPlan:
    """Derive and preflight one exact authority-zero seed-specific cache plan."""

    snapshot = _capture_rng()
    try:
        plan = _admit_periodic_descriptor_training_plan(
            system_id=system_id,
            config=config,
            train_source=train_source,
            val_source=val_source,
            train_cache=train_cache,
            val_cache=val_cache,
        )
    except Exception as error:
        if not _rng_matches(snapshot):
            _restore_rng(snapshot)
            raise PeriodicTrainingCacheError(
                "cache plan admission changed caller RNG state"
            ) from error
        raise
    if not _rng_matches(snapshot):
        _restore_rng(snapshot)
        raise PeriodicTrainingCacheError("cache plan admission changed caller RNG state")
    return plan


__all__ = [
    "AUTHORITY",
    "PLAN_SCHEMA",
    "PRODUCTION",
    "PeriodicDescriptorPlanRow",
    "PeriodicDescriptorTrainingPlan",
    "PeriodicTrainingCacheError",
    "RESULT_CLAIMED",
    "STATUS",
    "WINDOW_VALIDATION_SEMANTICS",
    "admit_periodic_descriptor_training_plan",
]
