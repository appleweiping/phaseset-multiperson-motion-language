"""Score a resolved nine-base cohort on one frozen capture validation census.

This private seam starts from ``base_cohort_resolver`` owned selected-best
checkpoint bytes.  It reconstructs each registered seed-bound B0/B1/B2 system,
revalidates the selected payload, loads exactly that model state, and delegates
the complete capture gallery to the audited capture-validation runtime.

The output is authority zero and deliberately stops before latency measurement,
``BaseScore`` construction, winner selection, or production authorization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from typing import Final

import numpy as np
import torch
from torch import nn

from . import base_cohort_resolver as resolver_module
from . import capture_validation as capture_module
from . import evaluation as evaluation_module
from . import training as training_module
from .experiments import base_run_ids, parse_run_id


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
OBSERVATION_SCHEMA: Final = "phaseset-base-capture-score-observation-v1"
COHORT_SCHEMA: Final = "phaseset-base-cohort-validation-v1"
OBSERVATION_STATUS: Final = "ACTUAL_GALLERY_SCORED_NO_LATENCY_NO_QUALIFICATION"
COHORT_STATUS: Final = "NINE_BASE_GALLERIES_SCORED_NO_WINNER_NO_QUALIFICATION"
MAX_SCORE_MATRIX_BYTES: Final = capture_module.DEFAULT_MAX_ENCODED_BYTES


class BaseCohortValidationError(ValueError):
    """The resolved cohort, selected model, or scored gallery is inconsistent."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaseCohortValidationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise BaseCohortValidationError(f"{label} must be exact bytes[32]")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1 or value > (1 << 63) - 1:
        raise BaseCohortValidationError(f"{label} must be an exact positive int")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


@dataclass(frozen=True, slots=True)
class BaseParameterRow:
    """One live named parameter in canonical module order."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise BaseCohortValidationError("parameter name must be a nonempty string")
        if (
            type(self.shape) is not tuple
            or any(type(axis) is not int or axis < 0 for axis in self.shape)
        ):
            raise BaseCohortValidationError("parameter shape must be an exact int tuple")
        if type(self.dtype) is not str or self.dtype != "torch.float32":
            raise BaseCohortValidationError("registered base parameter dtype must be float32")
        expected = 1
        for axis in self.shape:
            expected *= axis
        if _positive_int(self.numel, "parameter numel") != expected:
            raise BaseCohortValidationError("parameter numel differs from its shape")


@dataclass(frozen=True, slots=True)
class BaseParameterCensus:
    """Complete live parameter census derived after selected-state loading."""

    system_id: str
    rows: tuple[BaseParameterRow, ...]
    total: int
    loaded_state_sha256: str

    def __post_init__(self) -> None:
        if self.system_id not in training_module.REGISTERED_BASE_SYSTEM_IDS:
            raise BaseCohortValidationError("parameter census system_id is not registered")
        if type(self.rows) is not tuple or not self.rows:
            raise BaseCohortValidationError("parameter census must be a nonempty tuple")
        if any(type(row) is not BaseParameterRow for row in self.rows):
            raise BaseCohortValidationError("parameter census row type is invalid")
        names = tuple(row.name for row in self.rows)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise BaseCohortValidationError(
                "parameter census names must be unique and lexically ordered"
            )
        if _positive_int(self.total, "parameter total") != sum(
            row.numel for row in self.rows
        ):
            raise BaseCohortValidationError("parameter total differs from its rows")
        _lower_sha256(self.loaded_state_sha256, "loaded_state_sha256")

    @property
    def sha256(self) -> str:
        return _sha256(
            _canonical_json_bytes(
                {
                    "loaded_state_sha256": self.loaded_state_sha256,
                    "rows": [
                        {
                            "dtype": row.dtype,
                            "name": row.name,
                            "numel": row.numel,
                            "shape": list(row.shape),
                        }
                        for row in self.rows
                    ],
                    "schema": "phaseset-base-parameter-census-v1",
                    "system_id": self.system_id,
                    "total": self.total,
                }
            )
        )


def _fraction_payload(value: Fraction) -> dict[str, int]:
    if type(value) is not Fraction or not Fraction(0, 1) <= value <= Fraction(1, 1):
        raise BaseCohortValidationError("capture R1 must be an exact fraction in [0,1]")
    return {"denominator": value.denominator, "numerator": value.numerator}


def _dataset_from_observation(
    *,
    scores_float64le: bytes,
    score_shape: tuple[int, int],
    motion_commitments: tuple[bytes, ...],
    caption_commitments: tuple[bytes, ...],
    positive_motion_indices: tuple[tuple[int, ...], ...],
    group_sizes: tuple[int, ...],
    component_labels: tuple[str, ...],
) -> evaluation_module.RetrievalDataset:
    if type(scores_float64le) is not bytes:
        raise BaseCohortValidationError("score bytes must be exact immutable bytes")
    if (
        type(score_shape) is not tuple
        or len(score_shape) != 2
        or any(type(axis) is not int or axis < 2 for axis in score_shape)
    ):
        raise BaseCohortValidationError("score shape must be exact [M>=2,Q>=2]")
    expected = score_shape[0] * score_shape[1] * np.dtype("<f8").itemsize
    if expected > MAX_SCORE_MATRIX_BYTES:
        raise BaseCohortValidationError(
            "complete score matrix exceeds its fixed observation byte bound"
        )
    if len(scores_float64le) != expected:
        raise BaseCohortValidationError("score bytes differ from declared float64 shape")
    try:
        scores = (
            np.frombuffer(scores_float64le, dtype="<f8")
            .reshape(score_shape)
            .astype(np.float64, copy=True, order="C")
        )
        dataset = evaluation_module.validate_retrieval_dataset(
            evaluation_module.RetrievalDataset(
                scores=scores,
                motion_commitments=motion_commitments,
                caption_commitments=caption_commitments,
                positive_motion_indices=positive_motion_indices,
                group_sizes=np.ascontiguousarray(group_sizes, dtype=np.int64),
                component_labels=component_labels,
            )
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise BaseCohortValidationError("score observation dataset is invalid") from error
    return dataset


def _capture_fractions(
    dataset: evaluation_module.RetrievalDataset,
) -> tuple[Fraction, Fraction, Fraction]:
    rows = evaluation_module.capture_r1_contributions(dataset)
    if not rows:
        raise BaseCohortValidationError("capture score contribution census is empty")
    count = len(rows)
    text = sum(
        (
            Fraction(sum(row.text_to_motion_hits), len(row.text_to_motion_hits))
            for row in rows
        ),
        start=Fraction(0, 1),
    ) / count
    motion = sum(
        (
            Fraction(sum(row.motion_to_text_hits), len(row.motion_to_text_hits))
            for row in rows
        ),
        start=Fraction(0, 1),
    ) / count
    return text, motion, (text + motion) / 2


@dataclass(frozen=True, slots=True, repr=False)
class BaseValidationScoreObservation:
    """Immutable actual-score bytes plus checkpoint/runtime/query bindings."""

    run_id: str
    attempt_id: str
    system_id: str
    seed: int
    terminal_sha256: str
    selected_checkpoint_sha256: str
    selected_checkpoint_state_digest: str
    initialization_binding_sha256: str
    runtime_config_sha256: str
    factory_sha256: str
    code_artifact_sha256: str
    environment_sha256: str
    behavior_sha256: str
    source_manifest_sha256: str
    admitted_val_census_sha256: str
    query_census_sha256: str
    parameter_census: BaseParameterCensus
    scores_float64le: bytes = field(repr=False)
    score_shape: tuple[int, int]
    motion_commitments: tuple[bytes, ...] = field(repr=False)
    caption_commitments: tuple[bytes, ...] = field(repr=False)
    positive_motion_indices: tuple[tuple[int, ...], ...] = field(repr=False)
    group_sizes: tuple[int, ...]
    component_labels: tuple[str, ...]
    text_to_motion_capture_r1: Fraction
    motion_to_text_capture_r1: Fraction
    primary_capture_r1: Fraction
    device: str
    precision_mode: str
    encoded_bytes: int
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED
    schema: str = OBSERVATION_SCHEMA
    status: str = OBSERVATION_STATUS

    def __post_init__(self) -> None:
        role, seed, system_id = parse_run_id(self.run_id)
        if (
            role != "BASE_QUALIFICATION"
            or self.seed != seed
            or self.system_id != system_id
        ):
            raise BaseCohortValidationError("observation run identity is inconsistent")
        if type(self.attempt_id) is not str or not self.attempt_id:
            raise BaseCohortValidationError("observation attempt_id is invalid")
        for label in (
            "terminal_sha256",
            "selected_checkpoint_sha256",
            "selected_checkpoint_state_digest",
            "initialization_binding_sha256",
            "runtime_config_sha256",
            "factory_sha256",
            "code_artifact_sha256",
            "environment_sha256",
            "behavior_sha256",
            "source_manifest_sha256",
            "admitted_val_census_sha256",
            "query_census_sha256",
        ):
            _lower_sha256(getattr(self, label), label)
        if self.admitted_val_census_sha256 != self.query_census_sha256:
            raise BaseCohortValidationError(
                "checkpoint-admitted validation census differs from scored query census"
            )
        if (
            type(self.parameter_census) is not BaseParameterCensus
            or self.parameter_census.system_id != self.system_id
        ):
            raise BaseCohortValidationError("observation parameter census is incompatible")
        motions = tuple(
            _raw32(value, f"motion_commitments[{index}]")
            for index, value in enumerate(self.motion_commitments)
        )
        captions = tuple(
            _raw32(value, f"caption_commitments[{index}]")
            for index, value in enumerate(self.caption_commitments)
        )
        if type(self.group_sizes) is not tuple or any(
            type(value) is not int for value in self.group_sizes
        ):
            raise BaseCohortValidationError("group_sizes must be an exact int tuple")
        if type(self.component_labels) is not tuple:
            raise BaseCohortValidationError("component_labels must be an exact tuple")
        dataset = _dataset_from_observation(
            scores_float64le=self.scores_float64le,
            score_shape=self.score_shape,
            motion_commitments=motions,
            caption_commitments=captions,
            positive_motion_indices=self.positive_motion_indices,
            group_sizes=self.group_sizes,
            component_labels=self.component_labels,
        )
        expected_fractions = _capture_fractions(dataset)
        observed_fractions = (
            self.text_to_motion_capture_r1,
            self.motion_to_text_capture_r1,
            self.primary_capture_r1,
        )
        for value in observed_fractions:
            _fraction_payload(value)
        if observed_fractions != expected_fractions:
            raise BaseCohortValidationError("observation capture R1 differs from score bytes")
        if type(self.device) is not str or not self.device:
            raise BaseCohortValidationError("observation device is invalid")
        if self.precision_mode not in ("FP32", "BF16"):
            raise BaseCohortValidationError("observation precision mode is invalid")
        _positive_int(self.encoded_bytes, "encoded_bytes")
        if (
            type(self.authority) is not int
            or self.authority != AUTHORITY
            or self.production is not PRODUCTION
            or self.result_claimed is not RESULT_CLAIMED
            or self.schema != OBSERVATION_SCHEMA
            or self.status != OBSERVATION_STATUS
        ):
            raise BaseCohortValidationError("observation result header is invalid")
        object.__setattr__(self, "motion_commitments", motions)
        object.__setattr__(self, "caption_commitments", captions)

    @property
    def scores_float64_sha256(self) -> str:
        return _sha256(self.scores_float64le)

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "admitted_val_census_sha256": self.admitted_val_census_sha256,
                "attempt_id": self.attempt_id,
                "authority": self.authority,
                "behavior_sha256": self.behavior_sha256,
                "caption_commitments": [value.hex() for value in self.caption_commitments],
                "code_artifact_sha256": self.code_artifact_sha256,
                "component_labels": list(self.component_labels),
                "device": self.device,
                "encoded_bytes": self.encoded_bytes,
                "environment_sha256": self.environment_sha256,
                "factory_sha256": self.factory_sha256,
                "group_sizes": list(self.group_sizes),
                "initialization_binding_sha256": self.initialization_binding_sha256,
                "motion_commitments": [value.hex() for value in self.motion_commitments],
                "motion_to_text_capture_r1": _fraction_payload(
                    self.motion_to_text_capture_r1
                ),
                "parameter_census_sha256": self.parameter_census.sha256,
                "parameter_count": self.parameter_census.total,
                "positive_motion_indices": [
                    list(value) for value in self.positive_motion_indices
                ],
                "precision_mode": self.precision_mode,
                "primary_capture_r1": _fraction_payload(self.primary_capture_r1),
                "production": self.production,
                "query_census_sha256": self.query_census_sha256,
                "result_claimed": self.result_claimed,
                "run_id": self.run_id,
                "runtime_config_sha256": self.runtime_config_sha256,
                "schema": self.schema,
                "score_dtype": "<f8",
                "score_shape": list(self.score_shape),
                "scores_float64_sha256": self.scores_float64_sha256,
                "seed": self.seed,
                "selected_checkpoint_sha256": self.selected_checkpoint_sha256,
                "selected_checkpoint_state_digest": (
                    self.selected_checkpoint_state_digest
                ),
                "source_manifest_sha256": self.source_manifest_sha256,
                "status": self.status,
                "system_id": self.system_id,
                "terminal_sha256": self.terminal_sha256,
                "text_to_motion_capture_r1": _fraction_payload(
                    self.text_to_motion_capture_r1
                ),
            }
        )

    @property
    def score_artifact_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    def __repr__(self) -> str:
        return (
            f"BaseValidationScoreObservation(run_id={self.run_id!r}, "
            f"primary={self.primary_capture_r1}, "
            f"score_artifact_sha256={self.score_artifact_sha256!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class BaseCohortValidationResult:
    """All nine score observations, still without latency or a winner."""

    observations: tuple[BaseValidationScoreObservation, ...]
    plan_sha256: str
    matrix_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    train_manifest_sha256: str
    source_manifest_sha256: str
    query_census_sha256: str
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED
    schema: str = COHORT_SCHEMA
    status: str = COHORT_STATUS

    def __post_init__(self) -> None:
        if type(self.observations) is not tuple or any(
            type(value) is not BaseValidationScoreObservation
            for value in self.observations
        ):
            raise BaseCohortValidationError("cohort observations have an invalid type")
        if tuple(value.run_id for value in self.observations) != base_run_ids():
            raise BaseCohortValidationError(
                "cohort observations must cover the exact registered nine-row order"
            )
        for label in (
            "plan_sha256",
            "matrix_sha256",
            "training_config_sha256",
            "source_tree_sha256",
            "train_manifest_sha256",
            "source_manifest_sha256",
            "query_census_sha256",
        ):
            _lower_sha256(getattr(self, label), label)
        if any(
            row.source_manifest_sha256 != self.source_manifest_sha256
            or row.query_census_sha256 != self.query_census_sha256
            for row in self.observations
        ):
            raise BaseCohortValidationError("cohort rows do not share one validation census")
        if len({row.terminal_sha256 for row in self.observations}) != 9:
            raise BaseCohortValidationError("cohort terminal digests must be unique")
        if len({row.selected_checkpoint_sha256 for row in self.observations}) != 9:
            raise BaseCohortValidationError("cohort selected checkpoints must be unique")
        if len({row.score_artifact_sha256 for row in self.observations}) != 9:
            raise BaseCohortValidationError("cohort score artifacts must be unique")
        for system_id in training_module.REGISTERED_BASE_SYSTEM_IDS:
            counts = {
                row.parameter_census.total
                for row in self.observations
                if row.system_id == system_id
            }
            census_rows = {
                tuple((item.name, item.shape, item.dtype, item.numel) for item in row.parameter_census.rows)
                for row in self.observations
                if row.system_id == system_id
            }
            if len(counts) != 1 or len(census_rows) != 1:
                raise BaseCohortValidationError(
                    "base parameter census must agree across each system's three seeds"
                )
        if (
            type(self.authority) is not int
            or self.authority != AUTHORITY
            or self.production is not PRODUCTION
            or self.result_claimed is not RESULT_CLAIMED
            or self.schema != COHORT_SCHEMA
            or self.status != COHORT_STATUS
        ):
            raise BaseCohortValidationError("cohort result header is invalid")

    @property
    def sha256(self) -> str:
        return _sha256(
            _canonical_json_bytes(
                {
                    "authority": self.authority,
                    "matrix_sha256": self.matrix_sha256,
                    "observation_sha256s": [
                        value.score_artifact_sha256 for value in self.observations
                    ],
                    "plan_sha256": self.plan_sha256,
                    "production": self.production,
                    "query_census_sha256": self.query_census_sha256,
                    "result_claimed": self.result_claimed,
                    "schema": self.schema,
                    "source_manifest_sha256": self.source_manifest_sha256,
                    "source_tree_sha256": self.source_tree_sha256,
                    "status": self.status,
                    "training_config_sha256": self.training_config_sha256,
                    "train_manifest_sha256": self.train_manifest_sha256,
                }
            )
        )


def _checked_admission(
    value: object,
) -> resolver_module.BaseCohortAdmission:
    try:
        return resolver_module._checked_admission(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortValidationError("base cohort admission is invalid") from error


def _checked_cohort(
    value: object,
    admission: resolver_module.BaseCohortAdmission,
) -> resolver_module.ResolvedBaseCohort:
    if type(value) is not resolver_module.ResolvedBaseCohort:
        raise BaseCohortValidationError("cohort must be exact ResolvedBaseCohort")
    if (
        type(value.authority) is not int
        or value.authority != resolver_module.AUTHORITY
        or value.production is not resolver_module.PRODUCTION
        or value.result_claimed is not resolver_module.RESULT_CLAIMED
        or value.status != resolver_module.STATUS
    ):
        raise BaseCohortValidationError("resolved cohort header is invalid")
    expected_bindings = {
        "plan_sha256": admission.plan_sha256,
        "matrix_sha256": admission.matrix_sha256,
        "training_config_sha256": admission.training_config_sha256,
        "source_tree_sha256": admission.source_tree_sha256,
        "train_manifest_sha256": admission.train_manifest_sha256,
        "val_manifest_sha256": admission.val_manifest_sha256,
    }
    for name, expected in expected_bindings.items():
        if getattr(value, name) != expected:
            raise BaseCohortValidationError(f"resolved cohort {name} binding mismatch")
    if type(value.rows) is not tuple or tuple(row.run_id for row in value.rows) != base_run_ids():
        raise BaseCohortValidationError("resolved cohort lost the registered nine-row order")
    terminal_digests: set[str] = set()
    checkpoint_digests: set[str] = set()
    selected_bytes = 0
    for index, row in enumerate(value.rows):
        if type(row) is not resolver_module.ResolvedBaseRun:
            raise BaseCohortValidationError(f"resolved cohort row {index} type is invalid")
        role, seed, system_id = parse_run_id(row.run_id)
        if role != "BASE_QUALIFICATION" or row.seed != seed or row.system_id != system_id:
            raise BaseCohortValidationError("resolved row identity is inconsistent")
        if type(row.attempt_id) is not str or not row.attempt_id:
            raise BaseCohortValidationError("resolved row attempt_id is invalid")
        terminal = _lower_sha256(row.terminal_sha256, "terminal_sha256")
        _lower_sha256(row.latest_checkpoint_sha256, "latest_checkpoint_sha256")
        if type(row.latest_checkpoint_name) is not str or not row.latest_checkpoint_name:
            raise BaseCohortValidationError("latest checkpoint name is invalid")
        selected = row.selected_checkpoint
        if type(selected) is not resolver_module.SelectedBaseCheckpoint:
            raise BaseCohortValidationError("selected checkpoint type is invalid")
        if type(selected.raw) is not bytes or not selected.raw:
            raise BaseCohortValidationError("selected checkpoint bytes are invalid")
        selected_digest = _lower_sha256(selected.sha256, "selected checkpoint SHA-256")
        if _sha256(selected.raw) != selected_digest:
            raise BaseCohortValidationError("selected owned checkpoint digest mismatch")
        if type(selected.name) is not str or not selected.name:
            raise BaseCohortValidationError("selected checkpoint name is invalid")
        _positive_int(selected.epoch, "selected checkpoint epoch")
        _positive_int(selected.global_step, "selected checkpoint global_step")
        _lower_sha256(selected.state_digest, "selected checkpoint state_digest")
        if type(row.failed_predecessors) is not tuple or any(
            type(item) is not resolver_module.FailedBaseAttempt
            for item in row.failed_predecessors
        ):
            raise BaseCohortValidationError("failed predecessor census is invalid")
        terminal_digests.add(terminal)
        checkpoint_digests.add(selected_digest)
        selected_bytes += len(selected.raw)
        if selected_bytes > resolver_module.MAX_SELECTED_COHORT_BYTES:
            raise BaseCohortValidationError("selected cohort exceeds its owned-byte bound")
    if len(terminal_digests) != 9 or len(checkpoint_digests) != 9:
        raise BaseCohortValidationError(
            "resolved cohort terminals and selected checkpoints must be unique"
        )
    return value


def _snapshot_validation_source(value: object) -> capture_module.CaptureValidationSource:
    try:
        source_type, _runner = capture_module._verified_capture_validation_bindings()
        if type(value) is not source_type:
            raise BaseCohortValidationError(
                "validation source must be exact CaptureValidationSource"
            )
        return source_type(value.split, value.manifest_sha256, value.captures)
    except BaseCohortValidationError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortValidationError("capture validation source is invalid") from error


def _parameter_census(
    system: training_module.BaseRetrievalSystem,
) -> BaseParameterCensus:
    rows: list[BaseParameterRow] = []
    for name, parameter in sorted(system.named_parameters()):
        if (
            type(parameter) is not nn.Parameter
            or not parameter.requires_grad
            or parameter.dtype != torch.float32
            or parameter.numel() < 1
            or not bool(torch.isfinite(parameter.detach()).all().item())
        ):
            raise BaseCohortValidationError(
                "loaded registered base parameter census is invalid"
            )
        rows.append(
            BaseParameterRow(
                name=name,
                shape=tuple(int(axis) for axis in parameter.shape),
                dtype=str(parameter.dtype),
                numel=int(parameter.numel()),
            )
        )
    state_sha256 = training_module.training_system_state_sha256(system)
    return BaseParameterCensus(
        system_id=system.system_id,
        rows=tuple(rows),
        total=sum(row.numel for row in rows),
        loaded_state_sha256=state_sha256,
    )


def _validate_selected_payload(
    row: resolver_module.ResolvedBaseRun,
    admission: resolver_module.BaseCohortAdmission,
) -> dict[str, object]:
    selected = row.selected_checkpoint
    try:
        checked = resolver_module._validate_checkpoint_payload(
            selected.raw,
            sha256=selected.sha256,
            name=selected.name,
            run_id=row.run_id,
            admission=admission,
            ledger_record=None,
            require_completed_latest=False,
            require_selected_best=True,
        )
        if (
            checked.raw != selected.raw
            or checked.sha256 != selected.sha256
            or checked.name != selected.name
            or checked.epoch != selected.epoch
            or checked.global_step != selected.global_step
            or checked.state_digest != selected.state_digest
        ):
            raise BaseCohortValidationError(
                "revalidated selected checkpoint differs from resolved owned bytes"
            )
        payload = resolver_module._decode_checkpoint(selected.raw, selected.name)
    except BaseCohortValidationError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortValidationError(
            "selected checkpoint failed scoring-time revalidation"
        ) from error
    return payload


def _expected_dataset_identity(
    source: capture_module.CaptureValidationSource,
) -> tuple[
    tuple[bytes, ...],
    tuple[bytes, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    motions = tuple(capture.capture_commitment for capture in source.captures)
    captions: list[bytes] = []
    positives: list[tuple[int, ...]] = []
    for motion_index, capture in enumerate(source.captures):
        commitments = capture.holistic_text.caption_commitments
        order = sorted(range(len(commitments)), key=lambda index: commitments[index])
        captions.extend(commitments[index] for index in order)
        positives.extend((motion_index,) for _ in order)
    return (
        motions,
        tuple(captions),
        tuple(positives),
        tuple(capture.group_size for capture in source.captures),
        tuple(capture.component_label for capture in source.captures),
    )


def _preflight_score_matrix(source: capture_module.CaptureValidationSource) -> int:
    motion_count = len(source.captures)
    caption_count = sum(
        len(capture.holistic_text.caption_commitments)
        for capture in source.captures
    )
    expected = motion_count * caption_count * np.dtype("<f8").itemsize
    if expected > MAX_SCORE_MATRIX_BYTES:
        raise BaseCohortValidationError(
            "complete score matrix exceeds its fixed observation byte bound"
        )
    return expected


def _validated_capture_result(
    result: object,
    *,
    row: resolver_module.ResolvedBaseRun,
    source: capture_module.CaptureValidationSource,
    config: training_module.TrainingConfig,
) -> tuple[
    evaluation_module.RetrievalDataset,
    tuple[Fraction, Fraction, Fraction],
    bytes,
]:
    if type(result) is not capture_module.CaptureValidationResult:
        raise BaseCohortValidationError(
            "capture scorer did not return exact CaptureValidationResult"
        )
    try:
        dataset = evaluation_module.validate_retrieval_dataset(result.dataset)
    except (TypeError, ValueError) as error:
        raise BaseCohortValidationError("capture scorer dataset is invalid") from error
    expected_identity = _expected_dataset_identity(source)
    observed_identity = (
        dataset.motion_commitments,
        dataset.caption_commitments,
        dataset.positive_motion_indices,
        tuple(int(value) for value in dataset.group_sizes),
        dataset.component_labels,
    )
    if observed_identity != expected_identity:
        raise BaseCohortValidationError("scored gallery identity differs from frozen source")
    public_score_sha256 = _sha256(dataset.scores.tobytes(order="C"))
    scores = np.ascontiguousarray(dataset.scores, dtype="<f8")
    score_bytes = scores.tobytes(order="C")
    fractions = _capture_fractions(dataset)
    configured_device = torch.device(config.device)
    if configured_device.type == "cuda" and configured_device.index is None:
        configured_device = torch.device("cuda", torch.cuda.current_device())
    expected_encoded_bytes = (
        sum(len(capture.windows) for capture in source.captures)
        * training_module.REGISTERED_EMBEDDING_DIM
        * np.dtype(np.float32).itemsize
    )
    if (
        result.source_census_sha256 != source.census_sha256
        or result.scores_float64_sha256 != public_score_sha256
        or (
            result.text_to_motion_capture_r1,
            result.motion_to_text_capture_r1,
            result.primary_capture_r1,
        )
        != fractions
        or result.system_id != row.system_id
        or result.stage != "base"
        or result.device != str(configured_device)
        or result.precision_mode != training_module.resolve_precision(config).mode
        or result.encoded_bytes != expected_encoded_bytes
        or result.status != capture_module.STATUS
    ):
        raise BaseCohortValidationError(
            "capture scorer metadata or metric differs from recomputed gallery"
        )
    return dataset, fractions, score_bytes


def _score_resolved_run(
    row: resolver_module.ResolvedBaseRun,
    *,
    admission: resolver_module.BaseCohortAdmission,
    source: capture_module.CaptureValidationSource,
) -> BaseValidationScoreObservation:
    before_census = source.census_sha256
    if before_census != admission.val_manifest_sha256:
        raise BaseCohortValidationError(
            "validation source differs from checkpoint-admitted val census"
        )
    payload = _validate_selected_payload(row, admission)
    config = training_module.TrainingConfig(
        stage="base",
        seed=row.seed,
        device=admission.device,
        request_bf16=admission.request_bf16,
        bf16_runtime_qualified=admission.bf16_runtime_qualified,
        edge_budget=admission.edge_budget,
        checkpoint_every_updates=admission.checkpoint_every_updates,
        synthetic_contract=False,
    )
    try:
        system, binding = training_module.construct_registered_base_seed_bound_system(
            row.system_id,
            config,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortValidationError("registered base reconstruction failed") from error
    expected_bindings = {
        "behavior_sha256": binding.behavior_sha256,
        "code_artifact_sha256": binding.code_artifact_sha256,
        "environment_sha256": binding.environment_sha256,
        "factory_sha256": binding.factory_sha256,
        "initial_optimizable_state_sha256": binding.initial_optimizable_state_sha256,
        "initialization_binding_sha256": binding.sha256,
    }
    if any(payload.get(name) != value for name, value in expected_bindings.items()):
        raise BaseCohortValidationError(
            "selected checkpoint differs from live seed-bound initialization"
        )
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise BaseCohortValidationError("selected checkpoint model state is malformed")
    try:
        system.load_state_dict(model, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortValidationError("selected model state could not be loaded") from error
    loaded_state_sha256 = training_module.training_system_state_sha256(system)
    if (
        loaded_state_sha256 != training_module._stable_hash(model)
        or training_module.training_system_behavior_sha256(system)
        != binding.behavior_sha256
    ):
        raise BaseCohortValidationError(
            "loaded selected model differs from its checkpoint/runtime binding"
        )
    parameter_census = _parameter_census(system)
    if parameter_census.loaded_state_sha256 != loaded_state_sha256:
        raise AssertionError("parameter census lost the selected model state")
    try:
        system.to(torch.device(config.device))
        source_type, runner = capture_module._verified_capture_validation_bindings()
        if type(source) is not source_type:
            raise BaseCohortValidationError("capture source type changed before scoring")
        result = runner(system, config, source)
    except BaseCohortValidationError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortValidationError("complete capture gallery scoring failed") from error
    dataset, fractions, score_bytes = _validated_capture_result(
        result,
        row=row,
        source=source,
        config=config,
    )
    after_census = source.census_sha256
    if after_census != before_census or result.source_census_sha256 != before_census:
        raise BaseCohortValidationError("validation census changed during scoring")
    if _sha256(row.selected_checkpoint.raw) != row.selected_checkpoint.sha256:
        raise BaseCohortValidationError("selected checkpoint bytes changed during scoring")
    return BaseValidationScoreObservation(
        run_id=row.run_id,
        attempt_id=row.attempt_id,
        system_id=row.system_id,
        seed=row.seed,
        terminal_sha256=row.terminal_sha256,
        selected_checkpoint_sha256=row.selected_checkpoint.sha256,
        selected_checkpoint_state_digest=row.selected_checkpoint.state_digest,
        initialization_binding_sha256=binding.sha256,
        runtime_config_sha256=config.sha256,
        factory_sha256=binding.factory_sha256,
        code_artifact_sha256=binding.code_artifact_sha256,
        environment_sha256=binding.environment_sha256,
        behavior_sha256=binding.behavior_sha256,
        source_manifest_sha256=source.manifest_sha256,
        admitted_val_census_sha256=admission.val_manifest_sha256,
        query_census_sha256=result.source_census_sha256,
        parameter_census=parameter_census,
        scores_float64le=score_bytes,
        score_shape=tuple(int(axis) for axis in dataset.scores.shape),
        motion_commitments=dataset.motion_commitments,
        caption_commitments=dataset.caption_commitments,
        positive_motion_indices=dataset.positive_motion_indices,
        group_sizes=tuple(int(value) for value in dataset.group_sizes),
        component_labels=dataset.component_labels,
        text_to_motion_capture_r1=fractions[0],
        motion_to_text_capture_r1=fractions[1],
        primary_capture_r1=fractions[2],
        device=result.device,
        precision_mode=result.precision_mode,
        encoded_bytes=result.encoded_bytes,
    )


def score_resolved_base_cohort(
    cohort: resolver_module.ResolvedBaseCohort,
    *,
    admission: resolver_module.BaseCohortAdmission,
    validation_source: capture_module.CaptureValidationSource,
) -> BaseCohortValidationResult:
    """Score all nine selected-best bases without latency or winner selection."""

    checked_admission = _checked_admission(admission)
    checked_cohort = _checked_cohort(cohort, checked_admission)
    source = _snapshot_validation_source(validation_source)
    expected_census = source.census_sha256
    if expected_census != checked_admission.val_manifest_sha256:
        raise BaseCohortValidationError(
            "capture census differs from the cohort validation binding"
        )
    _preflight_score_matrix(source)
    observations: list[BaseValidationScoreObservation] = []
    for row in checked_cohort.rows:
        if source.census_sha256 != expected_census:
            raise BaseCohortValidationError("validation source changed between cohort rows")
        observations.append(
            _score_resolved_run(
                row,
                admission=checked_admission,
                source=source,
            )
        )
    if source.census_sha256 != expected_census:
        raise BaseCohortValidationError("validation source changed after cohort scoring")
    return BaseCohortValidationResult(
        observations=tuple(observations),
        plan_sha256=checked_cohort.plan_sha256,
        matrix_sha256=checked_cohort.matrix_sha256,
        training_config_sha256=checked_cohort.training_config_sha256,
        source_tree_sha256=checked_cohort.source_tree_sha256,
        train_manifest_sha256=checked_cohort.train_manifest_sha256,
        source_manifest_sha256=source.manifest_sha256,
        query_census_sha256=expected_census,
    )


__all__ = [
    "AUTHORITY",
    "BaseCohortValidationError",
    "BaseCohortValidationResult",
    "BaseParameterCensus",
    "BaseParameterRow",
    "BaseValidationScoreObservation",
    "COHORT_SCHEMA",
    "COHORT_STATUS",
    "MAX_SCORE_MATRIX_BYTES",
    "OBSERVATION_SCHEMA",
    "OBSERVATION_STATUS",
    "PRODUCTION",
    "RESULT_CLAIMED",
    "score_resolved_base_cohort",
]
