"""Sealed descriptor plans for prepared training plus holistic capture validation.

This authority-zero, data-free planning layer authenticates no path and runs no
model.  It derives one complete seed-specific descriptor plan from exact
already-loaded sources and readers.  The existing prepared-window plan keeps
its original engineering-only meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Final

import numpy as np

from .capture_validation import (
    CaptureDescriptorWindowSource,
    CaptureValidationCapture,
    CaptureValidationSource,
)
from .controls import system_spec
from .periodic_descriptor_cache_v2 import (
    CachedDescriptorBatch,
    CachedPairChunkStream,
    DescriptorCacheSourceBinding,
    DescriptorStreamKind,
    DescriptorWindowContext,
    PeriodicDescriptorCacheV2,
    prepared_group_batch_sha256,
)
from .periodic_training_cache import (
    _CACHEABLE_SYSTEMS,
    _EXPECTED_POLICIES,
    _batch_identity,
    _canonical_json,
    _capture_rng,
    _contexts_sha256,
    _enforce_edge_budget,
    _enumerate_rows,
    _lower_sha256,
    _restore_rng,
    _rng_matches,
    _source_binding_value,
    _validate_opened,
    _validate_reader_snapshot,
    PeriodicDescriptorPlanRow,
    PeriodicTrainingCacheError,
    RequestIdentity,
)
from .prepared_data_v2 import PreparedTrainingDataSourceV2
from .training import OFFICIAL_SEEDS, RESIDUAL_EPOCHS, RetrievalTrainingBatch, TrainingConfig


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
PLAN_SCHEMA: Final = "phaseset-periodic-descriptor-capture-training-plan-v1"
CAPTURE_VALIDATION_SEMANTICS: Final = "complete-holistic-capture-validation"

_PLAN_TOKEN: Final = object()

CaptureRequestIdentity = tuple[
    str,
    str,
    int,
    int,
    str,
    str,
    int,
    str,
    str,
    int,
    int,
]


class PeriodicCaptureTrainingCacheError(PeriodicTrainingCacheError):
    """An exact train-plus-capture descriptor plan could not be used."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _capture_request_sha256(
    *,
    capture_commitment: str,
    capture_plan_sha256: str,
    window_position: int,
    source_start_frame: int,
    window_commitment: str,
    source_window_npz_sha256: str,
    window_ordinal: int,
    contexts_sha256: str,
    batch_input_sha256: str,
    edge_count: int,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "batch_input_sha256": batch_input_sha256,
                "capture_commitment": capture_commitment,
                "capture_plan_sha256": capture_plan_sha256,
                "contexts_sha256": contexts_sha256,
                "edge_count": edge_count,
                "source_start_frame": source_start_frame,
                "source_window_npz_sha256": source_window_npz_sha256,
                "window_commitment": window_commitment,
                "window_ordinal": window_ordinal,
                "window_position": window_position,
            }
        )
    )


def _capture_request_identity(
    *,
    capture_commitment: str,
    capture_plan_sha256: str,
    window_position: int,
    source_start_frame: int,
    window_commitment: str,
    source_window_npz_sha256: str,
    window_ordinal: int,
    contexts_sha256: str,
    batch_input_sha256: str,
    edge_count: int,
) -> CaptureRequestIdentity:
    return (
        capture_commitment,
        capture_plan_sha256,
        window_position,
        source_start_frame,
        window_commitment,
        source_window_npz_sha256,
        window_ordinal,
        contexts_sha256,
        batch_input_sha256,
        edge_count,
        1,
    )


@dataclass(frozen=True, slots=True)
class PeriodicCaptureDescriptorPlanRow:
    """One exact B=1 capture window and its admitted descriptor shard."""

    execution_position: int
    capture_position: int
    capture_commitment: str
    capture_plan_sha256: str
    window_position: int
    source_start_frame: int
    window_commitment: str
    storage_manifest_sha256: str
    source_window_npz_sha256: str
    window_ordinal: int
    contexts_sha256: str
    batch_input_sha256: str
    cache_key_sha256: str
    request_sha256: str
    edge_count: int
    window_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "execution_position",
            "capture_position",
            "window_position",
            "source_start_frame",
            "window_ordinal",
            "edge_count",
            "window_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative int")
        if self.edge_count < 1 or self.window_count != 1:
            raise PeriodicCaptureTrainingCacheError(
                "capture descriptor rows require one window and at least one edge"
            )
        for name in (
            "capture_commitment",
            "capture_plan_sha256",
            "window_commitment",
            "storage_manifest_sha256",
            "source_window_npz_sha256",
            "contexts_sha256",
            "batch_input_sha256",
            "cache_key_sha256",
            "request_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        expected = _capture_request_sha256(
            capture_commitment=self.capture_commitment,
            capture_plan_sha256=self.capture_plan_sha256,
            window_position=self.window_position,
            source_start_frame=self.source_start_frame,
            window_commitment=self.window_commitment,
            source_window_npz_sha256=self.source_window_npz_sha256,
            window_ordinal=self.window_ordinal,
            contexts_sha256=self.contexts_sha256,
            batch_input_sha256=self.batch_input_sha256,
            edge_count=self.edge_count,
        )
        if self.request_sha256 != expected:
            raise PeriodicCaptureTrainingCacheError("capture row request digest is inconsistent")

    def _identity(self) -> CaptureRequestIdentity:
        return _capture_request_identity(
            capture_commitment=self.capture_commitment,
            capture_plan_sha256=self.capture_plan_sha256,
            window_position=self.window_position,
            source_start_frame=self.source_start_frame,
            window_commitment=self.window_commitment,
            source_window_npz_sha256=self.source_window_npz_sha256,
            window_ordinal=self.window_ordinal,
            contexts_sha256=self.contexts_sha256,
            batch_input_sha256=self.batch_input_sha256,
            edge_count=self.edge_count,
        )

    def _value(self) -> dict[str, object]:
        return {
            "batch_input_sha256": self.batch_input_sha256,
            "cache_key_sha256": self.cache_key_sha256,
            "capture_commitment": self.capture_commitment,
            "capture_plan_sha256": self.capture_plan_sha256,
            "capture_position": self.capture_position,
            "contexts_sha256": self.contexts_sha256,
            "edge_count": self.edge_count,
            "execution_position": self.execution_position,
            "request_sha256": self.request_sha256,
            "source_start_frame": self.source_start_frame,
            "source_window_npz_sha256": self.source_window_npz_sha256,
            "storage_manifest_sha256": self.storage_manifest_sha256,
            "window_commitment": self.window_commitment,
            "window_count": self.window_count,
            "window_ordinal": self.window_ordinal,
            "window_position": self.window_position,
        }


def _checked_cache_census(
    cache: PeriodicDescriptorCacheV2,
    label: str,
) -> tuple[str, ...]:
    census = cache.cache_key_census
    if (
        type(census) is not tuple
        or not census
        or census != tuple(sorted(census))
        or len(set(census)) != len(census)
    ):
        raise PeriodicCaptureTrainingCacheError(f"{label} cache key census is invalid")
    for index, value in enumerate(census):
        _lower_sha256(value, f"{label} cache key {index}")
    return census


def _snapshot_capture_source(value: object) -> CaptureValidationSource:
    if type(value) is not CaptureValidationSource:
        raise TypeError("capture source must be exact CaptureValidationSource")
    return CaptureValidationSource(value.split, value.manifest_sha256, value.captures)


def _capture_source_counts(
    source: CaptureValidationSource,
) -> tuple[int, int, int, str]:
    capture_count = len(source.captures)
    window_count = sum(len(capture.windows) for capture in source.captures)
    caption_count = sum(
        len(capture.holistic_text.caption_commitments) for capture in source.captures
    )
    descriptor_sources = tuple(
        window.descriptor_source for capture in source.captures for window in capture.windows
    )
    if not descriptor_sources or any(
        type(value) is not CaptureDescriptorWindowSource for value in descriptor_sources
    ):
        raise PeriodicCaptureTrainingCacheError(
            "capture plan requires v2 descriptor lineage on every window"
        )
    typed_sources = tuple(value for value in descriptor_sources if value is not None)
    storage_manifest = typed_sources[0].storage_manifest_sha256
    if any(value.storage_manifest_sha256 != storage_manifest for value in typed_sources):
        raise PeriodicCaptureTrainingCacheError(
            "capture descriptor rows do not share one storage manifest"
        )
    if tuple(value.window_ordinal for value in typed_sources) != tuple(range(window_count)):
        raise PeriodicCaptureTrainingCacheError(
            "capture descriptor ordinals differ from canonical global order"
        )
    return capture_count, window_count, caption_count, storage_manifest


def _checked_capture_values(
    capture: CaptureValidationCapture,
    window_position: int,
    *,
    seed: int,
    edge_budget: int,
    expected_storage_manifest_sha256: str,
) -> tuple[
    CaptureRequestIdentity,
    tuple[DescriptorWindowContext, ...],
    str,
    int,
]:
    assert type(capture) is CaptureValidationCapture
    if type(window_position) is not int:
        raise TypeError("window_position must be an exact int")
    if not 0 <= window_position < len(capture.windows):
        raise PeriodicCaptureTrainingCacheError("capture window position is out of range")
    window = capture.windows[window_position]
    descriptor_source = window.descriptor_source
    if type(descriptor_source) is not CaptureDescriptorWindowSource:
        raise PeriodicCaptureTrainingCacheError("capture window lacks exact v2 descriptor lineage")
    if descriptor_source.storage_manifest_sha256 != expected_storage_manifest_sha256:
        raise PeriodicCaptureTrainingCacheError(
            "capture window storage manifest differs from the plan"
        )
    context = window.descriptor_context(seed=seed)
    contexts = (context,)
    edge_count = sum(count * (count - 1) // 2 for count in window.groups.actor_counts)
    if edge_count < 1:
        raise PeriodicCaptureTrainingCacheError("capture descriptor rows require at least one edge")
    _enforce_edge_budget(edge_count, edge_budget)
    batch_digest = prepared_group_batch_sha256(window.groups)
    context_digest = _contexts_sha256(contexts)
    identity = _capture_request_identity(
        capture_commitment=capture.capture_commitment.hex(),
        capture_plan_sha256=capture.plan.sha256,
        window_position=window_position,
        source_start_frame=capture.plan.source_start_frames[window_position],
        window_commitment=window.window_commitment.hex(),
        source_window_npz_sha256=descriptor_source.source_window_npz_sha256,
        window_ordinal=descriptor_source.window_ordinal,
        contexts_sha256=context_digest,
        batch_input_sha256=batch_digest,
        edge_count=edge_count,
    )
    return identity, contexts, batch_digest, edge_count


def _capture_row(
    *,
    capture: CaptureValidationCapture,
    capture_position: int,
    window_position: int,
    execution_position: int,
    contexts: tuple[DescriptorWindowContext, ...],
    batch_input_sha256: str,
    edge_count: int,
    opened: CachedDescriptorBatch,
) -> PeriodicCaptureDescriptorPlanRow:
    window = capture.windows[window_position]
    descriptor_source = window.descriptor_source
    assert type(descriptor_source) is CaptureDescriptorWindowSource
    context_digest = _contexts_sha256(contexts)
    request_digest = _capture_request_sha256(
        capture_commitment=capture.capture_commitment.hex(),
        capture_plan_sha256=capture.plan.sha256,
        window_position=window_position,
        source_start_frame=capture.plan.source_start_frames[window_position],
        window_commitment=window.window_commitment.hex(),
        source_window_npz_sha256=descriptor_source.source_window_npz_sha256,
        window_ordinal=descriptor_source.window_ordinal,
        contexts_sha256=context_digest,
        batch_input_sha256=batch_input_sha256,
        edge_count=edge_count,
    )
    return PeriodicCaptureDescriptorPlanRow(
        execution_position=execution_position,
        capture_position=capture_position,
        capture_commitment=capture.capture_commitment.hex(),
        capture_plan_sha256=capture.plan.sha256,
        window_position=window_position,
        source_start_frame=capture.plan.source_start_frames[window_position],
        window_commitment=window.window_commitment.hex(),
        storage_manifest_sha256=descriptor_source.storage_manifest_sha256,
        source_window_npz_sha256=descriptor_source.source_window_npz_sha256,
        window_ordinal=descriptor_source.window_ordinal,
        contexts_sha256=context_digest,
        batch_input_sha256=batch_input_sha256,
        cache_key_sha256=opened.cache_key_sha256,
        request_sha256=request_digest,
        edge_count=edge_count,
    )


def _enumerate_capture_rows(
    source: CaptureValidationSource,
    cache: PeriodicDescriptorCacheV2,
    *,
    seed: int,
    stream_kind: DescriptorStreamKind,
    edge_budget: int,
    start_position: int,
    storage_manifest_sha256: str,
    source_binding: DescriptorCacheSourceBinding,
    floors_sha256: str,
    index_sha256: str,
    cache_key_census: tuple[str, ...],
) -> tuple[
    tuple[PeriodicCaptureDescriptorPlanRow, ...],
    Mapping[CaptureRequestIdentity, PeriodicCaptureDescriptorPlanRow],
]:
    rows: list[PeriodicCaptureDescriptorPlanRow] = []
    lookup: dict[CaptureRequestIdentity, PeriodicCaptureDescriptorPlanRow] = {}
    keys: set[str] = set()
    for capture_position, capture in enumerate(source.captures):
        for window_position, window in enumerate(capture.windows):
            if len(rows) >= len(cache_key_census):
                raise PeriodicCaptureTrainingCacheError(
                    "capture source row census exceeds its closed cache census"
                )
            identity, contexts, batch_digest, edge_count = _checked_capture_values(
                capture,
                window_position,
                seed=seed,
                edge_budget=edge_budget,
                expected_storage_manifest_sha256=storage_manifest_sha256,
            )
            _validate_reader_snapshot(
                cache,
                index_sha256=index_sha256,
                source_binding=source_binding,
                floors_sha256=floors_sha256,
                cache_key_census=cache_key_census,
            )
            opened = cache.open_batch(window.groups, contexts)
            stream = opened.stream(stream_kind)
            if type(stream) is not CachedPairChunkStream:
                raise PeriodicCaptureTrainingCacheError(
                    "capture cache returned an unsealed descriptor stream"
                )
            row = _capture_row(
                capture=capture,
                capture_position=capture_position,
                window_position=window_position,
                execution_position=start_position + len(rows),
                contexts=contexts,
                batch_input_sha256=batch_digest,
                edge_count=edge_count,
                opened=opened,
            )
            _validate_opened(
                opened,
                row=row,
                contexts=contexts,
                floors_sha256=floors_sha256,
            )
            if row._identity() != identity:
                raise PeriodicCaptureTrainingCacheError(
                    "capture shard request differs from the exact source window"
                )
            if identity in lookup or row.cache_key_sha256 in keys:
                raise PeriodicCaptureTrainingCacheError(
                    "capture plan repeats an exact descriptor request"
                )
            rows.append(row)
            lookup[identity] = row
            keys.add(row.cache_key_sha256)
    if not rows:
        raise PeriodicCaptureTrainingCacheError("capture source produced no plan rows")
    if tuple(sorted(keys)) != cache_key_census:
        raise PeriodicCaptureTrainingCacheError(
            "capture required key set differs from the closed reader census"
        )
    return tuple(rows), MappingProxyType(lookup)


@dataclass(frozen=True, slots=True)
class PeriodicDescriptorCaptureTrainingPlan:
    """Sealed complete train plus holistic capture descriptor plan."""

    system_id: str
    seed: int
    epochs: int
    stream_kind: DescriptorStreamKind
    config_sha256: str
    edge_budget: int
    energy_floors_sha256: str
    train_source_manifest_sha256: str
    capture_source_manifest_sha256: str
    capture_source_census_sha256: str
    capture_storage_manifest_sha256: str
    train_cache_index_sha256: str
    capture_cache_index_sha256: str
    train_cache_source_binding: DescriptorCacheSourceBinding
    capture_cache_source_binding: DescriptorCacheSourceBinding
    train_cache_key_census: tuple[str, ...]
    capture_cache_key_census: tuple[str, ...]
    capture_count: int
    window_count: int
    caption_count: int
    train_rows: tuple[PeriodicDescriptorPlanRow, ...]
    capture_rows: tuple[PeriodicCaptureDescriptorPlanRow, ...]
    _train_cache: PeriodicDescriptorCacheV2 = field(repr=False, compare=False)
    _capture_cache: PeriodicDescriptorCacheV2 = field(repr=False, compare=False)
    _train_lookup: Mapping[RequestIdentity, PeriodicDescriptorPlanRow] = field(
        repr=False,
        compare=False,
    )
    _capture_lookup: Mapping[
        CaptureRequestIdentity,
        PeriodicCaptureDescriptorPlanRow,
    ] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PLAN_TOKEN:
            raise PeriodicCaptureTrainingCacheError(
                "capture training plans must come from complete admission"
            )
        if (
            self.system_id not in _CACHEABLE_SYSTEMS
            or self.stream_kind != _CACHEABLE_SYSTEMS[self.system_id]
        ):
            raise PeriodicCaptureTrainingCacheError("plan system or stream is not cacheable")
        if self.seed not in OFFICIAL_SEEDS or self.epochs != RESIDUAL_EPOCHS:
            raise PeriodicCaptureTrainingCacheError("plan seed or epoch census changed")
        if type(self.edge_budget) is not int or self.edge_budget < 1:
            raise TypeError("edge_budget must be an exact positive int")
        for name in (
            "config_sha256",
            "energy_floors_sha256",
            "train_source_manifest_sha256",
            "capture_source_manifest_sha256",
            "capture_source_census_sha256",
            "capture_storage_manifest_sha256",
            "train_cache_index_sha256",
            "capture_cache_index_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        for name in ("capture_count", "window_count", "caption_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TypeError(f"{name} must be an exact positive int")
        if (
            type(self.train_cache_source_binding) is not DescriptorCacheSourceBinding
            or type(self.capture_cache_source_binding) is not DescriptorCacheSourceBinding
            or self.train_cache_source_binding.prepared_manifest_sha256
            != self.train_source_manifest_sha256
            or self.capture_cache_source_binding.prepared_manifest_sha256
            != self.capture_storage_manifest_sha256
        ):
            raise PeriodicCaptureTrainingCacheError(
                "cache source bindings differ from their consumed source manifests"
            )
        if (
            type(self.train_rows) is not tuple
            or not self.train_rows
            or any(type(row) is not PeriodicDescriptorPlanRow for row in self.train_rows)
            or any(row.split != "train" for row in self.train_rows)
        ):
            raise PeriodicCaptureTrainingCacheError("training row census is invalid")
        if (
            type(self.capture_rows) is not tuple
            or len(self.capture_rows) != self.window_count
            or any(type(row) is not PeriodicCaptureDescriptorPlanRow for row in self.capture_rows)
        ):
            raise PeriodicCaptureTrainingCacheError("capture row census is invalid")
        all_positions = tuple(row.execution_position for row in self.train_rows + self.capture_rows)
        if all_positions != tuple(range(len(all_positions))):
            raise PeriodicCaptureTrainingCacheError("plan execution positions are not canonical")
        if tuple(row.window_ordinal for row in self.capture_rows) != tuple(
            range(self.window_count)
        ):
            raise PeriodicCaptureTrainingCacheError("capture row ordinals are not canonical")
        capture_positions = tuple(sorted({row.capture_position for row in self.capture_rows}))
        if capture_positions != tuple(range(self.capture_count)):
            raise PeriodicCaptureTrainingCacheError(
                "capture positions are not complete and canonical"
            )
        if tuple(row.capture_position for row in self.capture_rows) != tuple(
            sorted(row.capture_position for row in self.capture_rows)
        ):
            raise PeriodicCaptureTrainingCacheError("capture rows are not grouped canonically")
        if len({row.capture_commitment for row in self.capture_rows}) != self.capture_count:
            raise PeriodicCaptureTrainingCacheError(
                "capture commitments are not complete and unique"
            )
        if any(
            row.storage_manifest_sha256 != self.capture_storage_manifest_sha256
            for row in self.capture_rows
        ):
            raise PeriodicCaptureTrainingCacheError(
                "capture rows differ from the consumed storage manifest"
            )
        for position in capture_positions:
            rows = tuple(row for row in self.capture_rows if row.capture_position == position)
            if (
                tuple(row.window_position for row in rows) != tuple(range(len(rows)))
                or len({row.capture_commitment for row in rows}) != 1
                or len({row.capture_plan_sha256 for row in rows}) != 1
            ):
                raise PeriodicCaptureTrainingCacheError(
                    "capture-local row order or ownership is invalid"
                )
        if tuple(sorted(row.cache_key_sha256 for row in self.train_rows)) != (
            self.train_cache_key_census
        ):
            raise PeriodicCaptureTrainingCacheError("training rows differ from cache key census")
        if tuple(sorted(row.cache_key_sha256 for row in self.capture_rows)) != (
            self.capture_cache_key_census
        ):
            raise PeriodicCaptureTrainingCacheError("capture rows differ from cache key census")
        for census, label in (
            (self.train_cache_key_census, "training"),
            (self.capture_cache_key_census, "capture"),
        ):
            if (
                type(census) is not tuple
                or not census
                or census != tuple(sorted(census))
                or len(set(census)) != len(census)
            ):
                raise PeriodicCaptureTrainingCacheError(f"{label} cache key census is invalid")
            for index, value in enumerate(census):
                _lower_sha256(value, f"{label} cache key {index}")
        if set(self._train_lookup) != {row._identity() for row in self.train_rows}:
            raise PeriodicCaptureTrainingCacheError("training lookup differs from rows")
        if set(self._capture_lookup) != {row._identity() for row in self.capture_rows}:
            raise PeriodicCaptureTrainingCacheError("capture lookup differs from rows")
        if any(self._train_lookup[row._identity()] != row for row in self.train_rows):
            raise PeriodicCaptureTrainingCacheError("training lookup values differ from rows")
        if any(self._capture_lookup[row._identity()] != row for row in self.capture_rows):
            raise PeriodicCaptureTrainingCacheError("capture lookup values differ from rows")

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
            "caption_count": self.caption_count,
            "capture_cache_index_sha256": self.capture_cache_index_sha256,
            "capture_cache_key_census": list(self.capture_cache_key_census),
            "capture_cache_source_binding": _source_binding_value(
                self.capture_cache_source_binding
            ),
            "capture_count": self.capture_count,
            "capture_rows": [row._value() for row in self.capture_rows],
            "capture_source_census_sha256": self.capture_source_census_sha256,
            "capture_source_manifest_sha256": self.capture_source_manifest_sha256,
            "capture_storage_manifest_sha256": self.capture_storage_manifest_sha256,
            "capture_validation_semantics": CAPTURE_VALIDATION_SEMANTICS,
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
            "window_count": self.window_count,
        }
        return _canonical_json(value)

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    def open_training_batch(
        self,
        batch: RetrievalTrainingBatch,
    ) -> CachedPairChunkStream:
        """Reopen one exact admitted training shard without fallback."""

        checked, identity = _batch_identity(
            batch,
            split="train",
            seed=self.seed,
            expected_manifest_sha256=self.train_source_manifest_sha256,
            epochs=self.epochs,
        )
        row = self._train_lookup.get(identity)
        if row is None:
            raise PeriodicCaptureTrainingCacheError(
                "training batch is not an exact admitted plan row"
            )
        _validate_reader_snapshot(
            self._train_cache,
            index_sha256=self.train_cache_index_sha256,
            source_binding=self.train_cache_source_binding,
            floors_sha256=self.energy_floors_sha256,
            cache_key_census=self.train_cache_key_census,
        )
        _enforce_edge_budget(row.edge_count, self.edge_budget)
        contexts = checked.descriptor_contexts
        assert contexts is not None
        opened = self._train_cache.open_batch(checked.groups, contexts)
        _validate_opened(
            opened,
            row=row,
            contexts=contexts,
            floors_sha256=self.energy_floors_sha256,
        )
        stream = opened.stream(self.stream_kind)
        if type(stream) is not CachedPairChunkStream:
            raise PeriodicCaptureTrainingCacheError(
                "training cache returned an unsealed descriptor stream"
            )
        return stream

    def validate_capture_validation_source(
        self,
        source: CaptureValidationSource,
    ) -> CaptureValidationSource:
        """Return a rebuilt exact source after checking full gallery and motion identity."""

        checked = _snapshot_capture_source(source)
        capture_count, window_count, caption_count, storage_manifest = _capture_source_counts(
            checked
        )
        if (
            checked.manifest_sha256 != self.capture_source_manifest_sha256
            or checked.census_sha256 != self.capture_source_census_sha256
            or storage_manifest != self.capture_storage_manifest_sha256
            or capture_count != self.capture_count
            or window_count != self.window_count
            or caption_count != self.caption_count
        ):
            raise PeriodicCaptureTrainingCacheError(
                "capture source differs from the complete admitted census"
            )
        actual_rows: list[CaptureRequestIdentity] = []
        for capture in checked.captures:
            for window_position in range(len(capture.windows)):
                identity, _contexts, _batch_digest, _edge_count = _checked_capture_values(
                    capture,
                    window_position,
                    seed=self.seed,
                    edge_budget=self.edge_budget,
                    expected_storage_manifest_sha256=(self.capture_storage_manifest_sha256),
                )
                actual_rows.append(identity)
        if tuple(actual_rows) != tuple(row._identity() for row in self.capture_rows):
            raise PeriodicCaptureTrainingCacheError(
                "capture source window rows differ from the admitted plan"
            )
        _validate_reader_snapshot(
            self._capture_cache,
            index_sha256=self.capture_cache_index_sha256,
            source_binding=self.capture_cache_source_binding,
            floors_sha256=self.energy_floors_sha256,
            cache_key_census=self.capture_cache_key_census,
        )
        return checked

    def open_capture_validation_window(
        self,
        capture: CaptureValidationCapture,
        window_position: int,
    ) -> CachedPairChunkStream:
        """Reopen one exact admitted capture-window shard without fallback."""

        if type(capture) is not CaptureValidationCapture:
            raise TypeError("capture must be exact CaptureValidationCapture")
        checked_capture = CaptureValidationCapture(
            capture.plan,
            capture.windows,
            capture.holistic_text,
            capture.component_label,
        )
        identity, contexts, _batch_digest, _edge_count = _checked_capture_values(
            checked_capture,
            window_position,
            seed=self.seed,
            edge_budget=self.edge_budget,
            expected_storage_manifest_sha256=self.capture_storage_manifest_sha256,
        )
        row = self._capture_lookup.get(identity)
        if row is None:
            raise PeriodicCaptureTrainingCacheError(
                "capture window is not an exact admitted plan row"
            )
        _validate_reader_snapshot(
            self._capture_cache,
            index_sha256=self.capture_cache_index_sha256,
            source_binding=self.capture_cache_source_binding,
            floors_sha256=self.energy_floors_sha256,
            cache_key_census=self.capture_cache_key_census,
        )
        _enforce_edge_budget(row.edge_count, self.edge_budget)
        opened = self._capture_cache.open_batch(
            checked_capture.windows[window_position].groups,
            contexts,
        )
        _validate_opened(
            opened,
            row=row,
            contexts=contexts,
            floors_sha256=self.energy_floors_sha256,
        )
        stream = opened.stream(self.stream_kind)
        if type(stream) is not CachedPairChunkStream:
            raise PeriodicCaptureTrainingCacheError(
                "capture cache returned an unsealed descriptor stream"
            )
        return stream


def _admit_periodic_descriptor_capture_training_plan(
    *,
    system_id: str,
    config: TrainingConfig,
    train_source: PreparedTrainingDataSourceV2,
    capture_validation_source: CaptureValidationSource,
    train_cache: PeriodicDescriptorCacheV2,
    capture_validation_cache: PeriodicDescriptorCacheV2,
) -> PeriodicDescriptorCaptureTrainingPlan:
    if type(config) is not TrainingConfig:
        raise TypeError("config must be an exact TrainingConfig")
    if config.stage != "residual" or config.synthetic_contract:
        raise PeriodicCaptureTrainingCacheError(
            "capture descriptor plans require an exact formal residual configuration"
        )
    if config.seed not in OFFICIAL_SEEDS or config.epochs != RESIDUAL_EPOCHS:
        raise PeriodicCaptureTrainingCacheError("plan seed or epoch census changed")
    if type(system_id) is not str or system_id not in _CACHEABLE_SYSTEMS:
        raise PeriodicCaptureTrainingCacheError(
            "capture plans support residual systems 02, 03, 04, 06, 07, and 08 only"
        )
    stream_kind = _CACHEABLE_SYSTEMS[system_id]
    specification = system_spec(system_id)
    if (
        specification.feature_policy != _EXPECTED_POLICIES[system_id]
        or specification.residual_enabled is not True
    ):
        raise PeriodicCaptureTrainingCacheError("registered system feature policy changed")
    if type(train_source) is not PreparedTrainingDataSourceV2:
        raise TypeError("train_source must be exact PreparedTrainingDataSourceV2")
    if train_source.split != "train":
        raise PeriodicCaptureTrainingCacheError("prepared source must be train-only")
    checked_capture_source = _snapshot_capture_source(capture_validation_source)
    capture_count, window_count, caption_count, storage_manifest = _capture_source_counts(
        checked_capture_source
    )
    if (
        type(train_cache) is not PeriodicDescriptorCacheV2
        or type(capture_validation_cache) is not PeriodicDescriptorCacheV2
    ):
        raise TypeError("caches must be exact PeriodicDescriptorCacheV2 readers")
    if train_cache is capture_validation_cache:
        raise PeriodicCaptureTrainingCacheError(
            "training and capture validation require distinct readers"
        )
    train_binding = train_cache.source_binding
    capture_binding = capture_validation_cache.source_binding
    if (
        type(train_binding) is not DescriptorCacheSourceBinding
        or type(capture_binding) is not DescriptorCacheSourceBinding
    ):
        raise PeriodicCaptureTrainingCacheError("cache source binding type changed")
    if train_binding.prepared_manifest_sha256 != train_source.manifest_sha256:
        raise PeriodicCaptureTrainingCacheError(
            "training cache binding differs from the prepared train source"
        )
    if capture_binding.prepared_manifest_sha256 != storage_manifest:
        raise PeriodicCaptureTrainingCacheError(
            "capture cache binding differs from the consumed storage manifest"
        )
    if train_cache.energy_floors_sha256 != capture_validation_cache.energy_floors_sha256:
        raise PeriodicCaptureTrainingCacheError(
            "training and capture cache energy-floor digests differ"
        )
    if not np.array_equal(train_cache.energy_floors, capture_validation_cache.energy_floors):
        raise PeriodicCaptureTrainingCacheError(
            "training and capture cache energy-floor bytes differ"
        )
    train_census = _checked_cache_census(train_cache, "training")
    capture_census = _checked_cache_census(capture_validation_cache, "capture")
    config_sha256 = config.sha256
    train_manifest = train_source.manifest_sha256
    capture_manifest = checked_capture_source.manifest_sha256
    capture_source_census = checked_capture_source.census_sha256
    floors_sha256 = train_cache.energy_floors_sha256
    train_index = train_cache.index_sha256
    capture_index = capture_validation_cache.index_sha256
    _validate_reader_snapshot(
        train_cache,
        index_sha256=train_index,
        source_binding=train_binding,
        floors_sha256=floors_sha256,
        cache_key_census=train_census,
    )
    _validate_reader_snapshot(
        capture_validation_cache,
        index_sha256=capture_index,
        source_binding=capture_binding,
        floors_sha256=floors_sha256,
        cache_key_census=capture_census,
    )
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
    capture_rows, capture_lookup = _enumerate_capture_rows(
        checked_capture_source,
        capture_validation_cache,
        seed=config.seed,
        stream_kind=stream_kind,
        edge_budget=config.edge_budget,
        start_position=len(train_rows),
        storage_manifest_sha256=storage_manifest,
        source_binding=capture_binding,
        floors_sha256=floors_sha256,
        index_sha256=capture_index,
        cache_key_census=capture_census,
    )
    if (
        config.sha256 != config_sha256
        or train_source.manifest_sha256 != train_manifest
        or checked_capture_source.manifest_sha256 != capture_manifest
        or checked_capture_source.census_sha256 != capture_source_census
        or train_source.split != "train"
        or checked_capture_source.split != "val"
    ):
        raise PeriodicCaptureTrainingCacheError(
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
        capture_validation_cache,
        index_sha256=capture_index,
        source_binding=capture_binding,
        floors_sha256=floors_sha256,
        cache_key_census=capture_census,
    )
    return PeriodicDescriptorCaptureTrainingPlan(
        system_id=system_id,
        seed=config.seed,
        epochs=RESIDUAL_EPOCHS,
        stream_kind=stream_kind,
        config_sha256=config_sha256,
        edge_budget=config.edge_budget,
        energy_floors_sha256=floors_sha256,
        train_source_manifest_sha256=train_manifest,
        capture_source_manifest_sha256=capture_manifest,
        capture_source_census_sha256=capture_source_census,
        capture_storage_manifest_sha256=storage_manifest,
        train_cache_index_sha256=train_index,
        capture_cache_index_sha256=capture_index,
        train_cache_source_binding=train_binding,
        capture_cache_source_binding=capture_binding,
        train_cache_key_census=train_census,
        capture_cache_key_census=capture_census,
        capture_count=capture_count,
        window_count=window_count,
        caption_count=caption_count,
        train_rows=train_rows,
        capture_rows=capture_rows,
        _train_cache=train_cache,
        _capture_cache=capture_validation_cache,
        _train_lookup=train_lookup,
        _capture_lookup=capture_lookup,
        _seal=_PLAN_TOKEN,
    )


def admit_periodic_descriptor_capture_training_plan(
    *,
    system_id: str,
    config: TrainingConfig,
    train_source: PreparedTrainingDataSourceV2,
    capture_validation_source: CaptureValidationSource,
    train_cache: PeriodicDescriptorCacheV2,
    capture_validation_cache: PeriodicDescriptorCacheV2,
) -> PeriodicDescriptorCaptureTrainingPlan:
    """Derive one exact authority-zero train-plus-capture cache plan."""

    snapshot = _capture_rng()
    try:
        plan = _admit_periodic_descriptor_capture_training_plan(
            system_id=system_id,
            config=config,
            train_source=train_source,
            capture_validation_source=capture_validation_source,
            train_cache=train_cache,
            capture_validation_cache=capture_validation_cache,
        )
    except Exception as error:
        if not _rng_matches(snapshot):
            _restore_rng(snapshot)
            raise PeriodicCaptureTrainingCacheError(
                "capture plan admission changed caller RNG state"
            ) from error
        raise
    if not _rng_matches(snapshot):
        _restore_rng(snapshot)
        raise PeriodicCaptureTrainingCacheError("capture plan admission changed caller RNG state")
    return plan


__all__ = [
    "AUTHORITY",
    "CAPTURE_VALIDATION_SEMANTICS",
    "PLAN_SCHEMA",
    "PRODUCTION",
    "PeriodicCaptureDescriptorPlanRow",
    "PeriodicCaptureTrainingCacheError",
    "PeriodicDescriptorCaptureTrainingPlan",
    "RESULT_CLAIMED",
    "admit_periodic_descriptor_capture_training_plan",
]
