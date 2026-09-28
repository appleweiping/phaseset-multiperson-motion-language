"""Assemble one verified nine-row base cohort into the existing selector.

This module performs no scoring, latency measurement, training, winner override,
or authorization.  It accepts only the resolver, scorer, and formal-latency
objects, closes their shared bindings, derives every :class:`BaseScore` field,
and invokes the already installed deterministic qualification path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Final

from . import base_cohort_latency as latency_module
from . import base_cohort_resolver as resolver_module
from . import base_cohort_validation as scoring_module
from . import experiments as experiments_module
from . import production as production_module


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
ROW_BINDING_SCHEMA: Final = "phaseset-base-qualification-combined-row-v1"
DEPENDENCY_SCHEMA: Final = "phaseset-base-qualification-dependencies-v1"
ASSEMBLY_SCHEMA: Final = "phaseset-base-cohort-qualification-assembly-v1"
STATUS: Final = "BASE_QUALIFICATION_ASSEMBLED_NO_EXTERNAL_AUTHORIZATION"


class BaseCohortQualificationError(ValueError):
    """The three input evidence families cannot form one qualification."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaseCohortQualificationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
        raise BaseCohortQualificationError(f"{label} must be an exact positive int")
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


def _canonical_digest(value: object) -> str:
    return _sha256(_canonical_json_bytes(value))


def _fraction_payload(value: Fraction) -> dict[str, int]:
    if type(value) is not Fraction or not Fraction(0, 1) <= value <= Fraction(1, 1):
        raise BaseCohortQualificationError("base validation score must be an exact fraction")
    return {"denominator": value.denominator, "numerator": value.numerator}


def _source_bytes(module: ModuleType, label: str) -> bytes:
    module_file = getattr(module, "__file__", None)
    if type(module_file) is not str:
        raise BaseCohortQualificationError(f"installed {label} source path is unavailable")
    path = Path(module_file)
    if path.suffix in {".pyc", ".pyo"}:
        path = path.with_suffix(".py")
    if path.is_symlink() or not path.is_file():
        raise BaseCohortQualificationError(f"installed {label} source is irregular")
    raw = path.read_bytes()
    if not raw:
        raise BaseCohortQualificationError(f"installed {label} source is empty")
    return raw


def _admission_sha256(value: resolver_module.BaseCohortAdmission) -> str:
    return _canonical_digest(
        {
            "bf16_runtime_qualified": value.bf16_runtime_qualified,
            "checkpoint_every_updates": value.checkpoint_every_updates,
            "device": value.device,
            "edge_budget": value.edge_budget,
            "matrix_sha256": value.matrix_sha256,
            "plan_sha256": value.plan_sha256,
            "request_bf16": value.request_bf16,
            "source_tree_sha256": value.source_tree_sha256,
            "train_manifest_sha256": value.train_manifest_sha256,
            "training_config_sha256": value.training_config_sha256,
            "val_manifest_sha256": value.val_manifest_sha256,
        }
    )


def _samples_sha256(samples: tuple[int, ...]) -> str:
    if type(samples) is not tuple or len(samples) != latency_module.SAMPLES_PER_ROW:
        raise BaseCohortQualificationError("latency row must contain exact 99 samples")
    for value in samples:
        _positive_int(value, "latency sample")
    return _canonical_digest(
        {
            "samples_ns": list(samples),
            "schema": "phaseset-base-latency-raw-samples-v1",
        }
    )


def _median_from_samples(samples: tuple[int, ...]) -> int:
    _samples_sha256(samples)
    ordered = tuple(sorted(samples))
    return ordered[len(ordered) // 2]


def _visit_samples_for_run(
    latency: latency_module.BaseCohortLatencySession,
    run_id: str,
) -> tuple[int, ...]:
    matching = tuple(visit for visit in latency.visits if visit.run_id == run_id)
    if len(matching) != latency_module.ROUND_COUNT or any(
        visit.completed is not True or len(visit.samples_ns) != latency_module.TIMED_PER_VISIT
        for visit in matching
    ):
        raise BaseCohortQualificationError(
            "latency visit census is incomplete for a qualification row"
        )
    samples = tuple(sample for visit in matching for sample in visit.samples_ns)
    _samples_sha256(samples)
    return samples


@dataclass(frozen=True, slots=True)
class BaseQualificationDependencyBinding:
    admission_sha256: str
    resolved_cohort_sha256: str
    scored_cohort_sha256: str
    latency_session_sha256: str
    plan_sha256: str
    matrix_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    train_manifest_sha256: str
    admitted_val_census_sha256: str
    validation_source_manifest_sha256: str
    query_census_sha256: str
    resolver_source_sha256: str
    scorer_source_sha256: str
    latency_source_sha256: str
    assembler_source_sha256: str
    production_source_sha256: str
    selector_artifact_sha256: str
    runtime_sha256: str
    environment_sha256: str
    precision_mode: str
    schema: str = DEPENDENCY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "admission_sha256",
            "resolved_cohort_sha256",
            "scored_cohort_sha256",
            "latency_session_sha256",
            "plan_sha256",
            "matrix_sha256",
            "training_config_sha256",
            "source_tree_sha256",
            "train_manifest_sha256",
            "admitted_val_census_sha256",
            "validation_source_manifest_sha256",
            "query_census_sha256",
            "resolver_source_sha256",
            "scorer_source_sha256",
            "latency_source_sha256",
            "assembler_source_sha256",
            "production_source_sha256",
            "selector_artifact_sha256",
            "runtime_sha256",
            "environment_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if self.admitted_val_census_sha256 != self.query_census_sha256:
            raise BaseCohortQualificationError(
                "admitted validation census differs from scored query census"
            )
        if self.precision_mode not in ("FP32", "BF16"):
            raise BaseCohortQualificationError("qualification precision mode is invalid")
        if self.schema != DEPENDENCY_SCHEMA:
            raise BaseCohortQualificationError("qualification dependency schema changed")

    @property
    def sha256(self) -> str:
        return _canonical_digest({name: getattr(self, name) for name in self.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class CombinedBaseScoreBinding:
    run_id: str
    attempt_id: str
    system_id: str
    seed: int
    terminal_sha256: str
    selected_checkpoint_sha256: str
    selected_checkpoint_state_digest: str
    validation_score_artifact_sha256: str
    latency_row_sha256: str
    latency_samples_sha256: str
    parameter_census_sha256: str
    parameter_count: int
    frozen_runtime_latency_ns: int
    bidirectional_capture_r1: Fraction
    validation_manifest_sha256: str
    validation_source_manifest_sha256: str
    query_census_sha256: str
    evaluator_sha256: str
    runtime_sha256: str
    environment_sha256: str
    precision_mode: str
    schema: str = ROW_BINDING_SCHEMA

    def __post_init__(self) -> None:
        role, seed, system_id = experiments_module.parse_run_id(self.run_id)
        if role != "BASE_QUALIFICATION" or seed != self.seed or system_id != self.system_id:
            raise BaseCohortQualificationError("combined base row identity is inconsistent")
        if type(self.attempt_id) is not str or not self.attempt_id:
            raise BaseCohortQualificationError("combined base attempt ID is invalid")
        for name in (
            "terminal_sha256",
            "selected_checkpoint_sha256",
            "selected_checkpoint_state_digest",
            "validation_score_artifact_sha256",
            "latency_row_sha256",
            "latency_samples_sha256",
            "parameter_census_sha256",
            "validation_manifest_sha256",
            "validation_source_manifest_sha256",
            "query_census_sha256",
            "evaluator_sha256",
            "runtime_sha256",
            "environment_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        _positive_int(self.parameter_count, "parameter_count")
        _positive_int(self.frozen_runtime_latency_ns, "frozen_runtime_latency_ns")
        _fraction_payload(self.bidirectional_capture_r1)
        if self.precision_mode not in ("FP32", "BF16"):
            raise BaseCohortQualificationError("combined row precision mode is invalid")
        if self.schema != ROW_BINDING_SCHEMA:
            raise BaseCohortQualificationError("combined row schema changed")

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "attempt_id": self.attempt_id,
                "bidirectional_capture_r1": _fraction_payload(self.bidirectional_capture_r1),
                "environment_sha256": self.environment_sha256,
                "evaluator_sha256": self.evaluator_sha256,
                "frozen_runtime_latency_ns": self.frozen_runtime_latency_ns,
                "latency_row_sha256": self.latency_row_sha256,
                "latency_samples_sha256": self.latency_samples_sha256,
                "parameter_census_sha256": self.parameter_census_sha256,
                "parameter_count": self.parameter_count,
                "precision_mode": self.precision_mode,
                "query_census_sha256": self.query_census_sha256,
                "run_id": self.run_id,
                "runtime_sha256": self.runtime_sha256,
                "schema": self.schema,
                "seed": self.seed,
                "selected_checkpoint_sha256": self.selected_checkpoint_sha256,
                "selected_checkpoint_state_digest": (self.selected_checkpoint_state_digest),
                "system_id": self.system_id,
                "terminal_sha256": self.terminal_sha256,
                "validation_manifest_sha256": self.validation_manifest_sha256,
                "validation_score_artifact_sha256": (self.validation_score_artifact_sha256),
                "validation_source_manifest_sha256": (self.validation_source_manifest_sha256),
            }
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True, repr=False)
class BaseCohortQualificationAssembly:
    qualification: experiments_module.BaseQualification
    execution_evidence: production_module.BaseQualificationExecutionEvidence
    artifact: bytes = field(repr=False)
    row_bindings: tuple[CombinedBaseScoreBinding, ...]
    dependency_binding: BaseQualificationDependencyBinding
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED
    schema: str = ASSEMBLY_SCHEMA
    status: str = STATUS

    def __post_init__(self) -> None:
        if type(self.qualification) is not experiments_module.BaseQualification:
            raise BaseCohortQualificationError("assembly qualification type is invalid")
        if (
            type(self.execution_evidence)
            is not production_module.BaseQualificationExecutionEvidence
        ):
            raise BaseCohortQualificationError("assembly execution evidence type is invalid")
        if type(self.artifact) is not bytes or not self.artifact:
            raise BaseCohortQualificationError("assembly artifact must be nonempty bytes")
        if (
            type(self.row_bindings) is not tuple
            or len(self.row_bindings) != len(experiments_module.base_run_ids())
            or any(type(row) is not CombinedBaseScoreBinding for row in self.row_bindings)
        ):
            raise BaseCohortQualificationError("assembly combined row census is invalid")
        if type(self.dependency_binding) is not BaseQualificationDependencyBinding:
            raise BaseCohortQualificationError("assembly dependency binding is invalid")
        if tuple(row.run_id for row in self.row_bindings) != experiments_module.base_run_ids():
            raise BaseCohortQualificationError("assembly combined rows changed order")
        if self.execution_evidence.score_rows != self.qualification.score_rows:
            raise BaseCohortQualificationError("execution evidence differs from qualification rows")
        if tuple(row.sha256 for row in self.row_bindings) != tuple(
            row.score_artifact_sha256 for row in self.qualification.score_rows
        ):
            raise BaseCohortQualificationError("combined evidence differs from BaseScore rows")
        if _sha256(self.execution_evidence.selector_artifact) != (
            self.dependency_binding.selector_artifact_sha256
        ):
            raise BaseCohortQualificationError("execution selector differs from dependencies")
        try:
            canonical = experiments_module.canonical_base_qualification_bytes(self.qualification)
        except (TypeError, ValueError, RuntimeError) as error:
            raise BaseCohortQualificationError("assembled qualification is invalid") from error
        if self.artifact != canonical:
            raise BaseCohortQualificationError("assembly artifact is not canonical qualification")
        if (
            type(self.authority) is not int
            or self.authority != AUTHORITY
            or self.production is not PRODUCTION
            or self.result_claimed is not RESULT_CLAIMED
            or self.schema != ASSEMBLY_SCHEMA
            or self.status != STATUS
        ):
            raise BaseCohortQualificationError("assembly result header is invalid")

    @property
    def sha256(self) -> str:
        return _canonical_digest(
            {
                "artifact_sha256": _sha256(self.artifact),
                "authority": self.authority,
                "dependency_binding_sha256": self.dependency_binding.sha256,
                "production": self.production,
                "result_claimed": self.result_claimed,
                "row_binding_sha256s": [row.sha256 for row in self.row_bindings],
                "schema": self.schema,
                "status": self.status,
            }
        )

    def __repr__(self) -> str:
        return (
            "BaseCohortQualificationAssembly("
            f"winner={self.qualification.winner_system_id!r}, "
            f"sha256={self.sha256!r})"
        )


def _installed_code_snapshot() -> tuple[dict[str, str], bytes]:
    selector_artifact = production_module.current_base_qualification_artifact_bytes()
    if type(selector_artifact) is not bytes or not selector_artifact:
        raise BaseCohortQualificationError("installed selector artifact is invalid")
    sources = {
        "resolver_source_sha256": _sha256(_source_bytes(resolver_module, "base cohort resolver")),
        "scorer_source_sha256": _sha256(_source_bytes(scoring_module, "base cohort validation")),
        "latency_source_sha256": _sha256(_source_bytes(latency_module, "base cohort latency")),
        "assembler_source_sha256": _sha256(
            _source_bytes(
                __import__(__name__, fromlist=["*"]),
                "base cohort qualification",
            )
        ),
        "production_source_sha256": _sha256(_source_bytes(production_module, "production adapter")),
        "selector_artifact_sha256": _sha256(selector_artifact),
    }
    return sources, selector_artifact


def _checked_inputs(
    cohort: object,
    admission: object,
    scored_cohort: object,
    latency_session: object,
) -> tuple[
    resolver_module.ResolvedBaseCohort,
    resolver_module.BaseCohortAdmission,
    scoring_module.BaseCohortValidationResult,
    latency_module.BaseCohortLatencySession,
]:
    try:
        checked_cohort, checked_admission, checked_scores = latency_module._checked_inputs(
            cohort,
            admission,
            scored_cohort,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortQualificationError(
            "resolver and scorer inputs do not form one admitted cohort"
        ) from error
    if type(latency_session) is not latency_module.BaseCohortLatencySession:
        raise BaseCohortQualificationError("latency input must be exact formal session")
    if (
        latency_session.authority != latency_module.AUTHORITY
        or latency_session.production is not latency_module.PRODUCTION
        or latency_session.result_claimed is not latency_module.RESULT_CLAIMED
        or latency_session.status != latency_module.ACTUAL_COMPLETE_STATUS
        or latency_session.actual_runtime is not True
        or latency_session.complete is not True
        or latency_session.formal_samples_complete is not True
        or latency_session.failure_code is not None
    ):
        raise BaseCohortQualificationError("latency session is not one actual complete session")
    if (
        latency_session.resolved_cohort_sha256
        != latency_module._resolved_cohort_sha256(checked_cohort)
        or latency_session.scored_cohort_sha256 != checked_scores.sha256
    ):
        raise BaseCohortQualificationError("latency session differs from resolver or scorer")
    runtime = latency_session.runtime_identity
    if type(runtime) is not latency_module.BaseLatencyRuntimeIdentity:
        raise BaseCohortQualificationError("latency session runtime identity is absent")
    if (
        runtime.device_uuid != latency_session.registered_cuda_uuid
        or latency_session.admission_observation.device_uuid != runtime.device_uuid
        or latency_session.post_context_observation is None
        or latency_session.post_context_observation.device_uuid != runtime.device_uuid
    ):
        raise BaseCohortQualificationError("latency session CUDA identity is inconsistent")
    expected_runs = experiments_module.base_run_ids()
    if tuple(row.run_id for row in latency_session.rows) != expected_runs:
        raise BaseCohortQualificationError("latency session row order is incomplete")
    return checked_cohort, checked_admission, checked_scores, latency_session


def _build_row_bindings(
    cohort: resolver_module.ResolvedBaseCohort,
    scored: scoring_module.BaseCohortValidationResult,
    latency: latency_module.BaseCohortLatencySession,
    evaluator_sha256: str,
) -> tuple[CombinedBaseScoreBinding, ...]:
    runtime = latency.runtime_identity
    if type(runtime) is not latency_module.BaseLatencyRuntimeIdentity:
        raise BaseCohortQualificationError("formal latency runtime is absent")
    rows: list[CombinedBaseScoreBinding] = []
    for resolved, score, measured in zip(
        cohort.rows,
        scored.observations,
        latency.rows,
        strict=True,
    ):
        visit_samples = _visit_samples_for_run(latency, resolved.run_id)
        samples_sha256 = _samples_sha256(visit_samples)
        measured_median = _median_from_samples(visit_samples)
        if (
            resolved.run_id != score.run_id
            or resolved.run_id != measured.run_id
            or resolved.attempt_id != score.attempt_id
            or resolved.system_id != score.system_id
            or resolved.system_id != measured.system_id
            or resolved.seed != score.seed
            or resolved.seed != measured.seed
            or resolved.terminal_sha256 != score.terminal_sha256
            or resolved.selected_checkpoint.sha256 != score.selected_checkpoint_sha256
            or resolved.selected_checkpoint.sha256 != measured.selected_checkpoint_sha256
            or resolved.selected_checkpoint.state_digest != score.selected_checkpoint_state_digest
            or resolved.selected_checkpoint.state_digest
            != measured.selected_checkpoint_state_digest
        ):
            raise BaseCohortQualificationError("qualification row identity or checkpoint changed")
        if _sha256(resolved.selected_checkpoint.raw) != resolved.selected_checkpoint.sha256:
            raise BaseCohortQualificationError("selected-best checkpoint bytes changed")
        if (
            measured.parameter_census_sha256 != score.parameter_census.sha256
            or measured.parameter_count != score.parameter_census.total
            or measured.runtime_sha256 != runtime.sha256
            or measured.samples_ns != visit_samples
            or measured_median != measured.median_ns
            or score.environment_sha256 != runtime.environment_sha256
            or score.precision_mode != runtime.precision_mode
            or score.device != runtime.device
            or score.admitted_val_census_sha256 != cohort.val_manifest_sha256
            or score.query_census_sha256 != cohort.val_manifest_sha256
            or score.source_manifest_sha256 != scored.source_manifest_sha256
        ):
            raise BaseCohortQualificationError(
                "qualification row score, parameter, runtime, or census changed"
            )
        rows.append(
            CombinedBaseScoreBinding(
                run_id=resolved.run_id,
                attempt_id=resolved.attempt_id,
                system_id=resolved.system_id,
                seed=resolved.seed,
                terminal_sha256=resolved.terminal_sha256,
                selected_checkpoint_sha256=resolved.selected_checkpoint.sha256,
                selected_checkpoint_state_digest=(resolved.selected_checkpoint.state_digest),
                validation_score_artifact_sha256=score.score_artifact_sha256,
                latency_row_sha256=measured.sha256,
                latency_samples_sha256=samples_sha256,
                parameter_census_sha256=score.parameter_census.sha256,
                parameter_count=score.parameter_census.total,
                frozen_runtime_latency_ns=measured_median,
                bidirectional_capture_r1=score.primary_capture_r1,
                validation_manifest_sha256=cohort.val_manifest_sha256,
                validation_source_manifest_sha256=score.source_manifest_sha256,
                query_census_sha256=score.query_census_sha256,
                evaluator_sha256=evaluator_sha256,
                runtime_sha256=runtime.sha256,
                environment_sha256=runtime.environment_sha256,
                precision_mode=runtime.precision_mode,
            )
        )
    if tuple(row.run_id for row in rows) != experiments_module.base_run_ids():
        raise BaseCohortQualificationError("combined qualification rows are incomplete")
    return tuple(rows)


def _base_scores(
    row_bindings: tuple[CombinedBaseScoreBinding, ...],
) -> tuple[experiments_module.BaseScore, ...]:
    return tuple(
        experiments_module.BaseScore(
            run_id=row.run_id,
            bidirectional_r1_numerator=row.bidirectional_capture_r1.numerator,
            bidirectional_r1_denominator=row.bidirectional_capture_r1.denominator,
            parameter_count=row.parameter_count,
            frozen_runtime_latency_ns=row.frozen_runtime_latency_ns,
            terminal_sha256=row.terminal_sha256,
            selected_checkpoint_sha256=row.selected_checkpoint_sha256,
            split="validation",
            validation_manifest_sha256=row.validation_manifest_sha256,
            query_census_sha256=row.query_census_sha256,
            evaluator_sha256=row.evaluator_sha256,
            score_artifact_sha256=row.sha256,
        )
        for row in row_bindings
    )


def assemble_base_cohort_qualification(
    cohort: resolver_module.ResolvedBaseCohort,
    *,
    admission: resolver_module.BaseCohortAdmission,
    scored_cohort: scoring_module.BaseCohortValidationResult,
    latency_session: latency_module.BaseCohortLatencySession,
) -> BaseCohortQualificationAssembly:
    """Derive all nine BaseScore rows and invoke the existing selector once."""

    checked_cohort, checked_admission, checked_scores, checked_latency = _checked_inputs(
        cohort,
        admission,
        scored_cohort,
        latency_session,
    )
    before_sources, selector_artifact = _installed_code_snapshot()
    evaluator_sha256 = before_sources["selector_artifact_sha256"]
    row_bindings = _build_row_bindings(
        checked_cohort,
        checked_scores,
        checked_latency,
        evaluator_sha256,
    )
    score_rows = _base_scores(row_bindings)
    try:
        qualification = experiments_module.qualify_base(score_rows)
        evidence = production_module.BaseQualificationExecutionEvidence(
            score_rows=score_rows,
            selector_artifact=selector_artifact,
        )
        artifact = production_module.build_base_qualification_execution_artifact(evidence)
        canonical = experiments_module.canonical_base_qualification_bytes(qualification)
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortQualificationError(
            "installed base qualification recomputation failed"
        ) from error
    if artifact != canonical:
        raise BaseCohortQualificationError(
            "production evidence and deterministic selector artifacts differ"
        )
    runtime = checked_latency.runtime_identity
    if type(runtime) is not latency_module.BaseLatencyRuntimeIdentity:
        raise AssertionError("checked formal latency lost runtime identity")
    dependency = BaseQualificationDependencyBinding(
        admission_sha256=_admission_sha256(checked_admission),
        resolved_cohort_sha256=latency_module._resolved_cohort_sha256(checked_cohort),
        scored_cohort_sha256=checked_scores.sha256,
        latency_session_sha256=checked_latency.sha256,
        plan_sha256=checked_cohort.plan_sha256,
        matrix_sha256=checked_cohort.matrix_sha256,
        training_config_sha256=checked_cohort.training_config_sha256,
        source_tree_sha256=checked_cohort.source_tree_sha256,
        train_manifest_sha256=checked_cohort.train_manifest_sha256,
        admitted_val_census_sha256=checked_cohort.val_manifest_sha256,
        validation_source_manifest_sha256=checked_scores.source_manifest_sha256,
        query_census_sha256=checked_scores.query_census_sha256,
        resolver_source_sha256=before_sources["resolver_source_sha256"],
        scorer_source_sha256=before_sources["scorer_source_sha256"],
        latency_source_sha256=before_sources["latency_source_sha256"],
        assembler_source_sha256=before_sources["assembler_source_sha256"],
        production_source_sha256=before_sources["production_source_sha256"],
        selector_artifact_sha256=evaluator_sha256,
        runtime_sha256=runtime.sha256,
        environment_sha256=runtime.environment_sha256,
        precision_mode=runtime.precision_mode,
    )
    after_sources, after_selector = _installed_code_snapshot()
    if after_sources != before_sources or after_selector != selector_artifact:
        raise BaseCohortQualificationError("installed dependency bytes changed during assembly")
    if (
        latency_module._resolved_cohort_sha256(checked_cohort) != dependency.resolved_cohort_sha256
        or checked_scores.sha256 != dependency.scored_cohort_sha256
        or checked_latency.sha256 != dependency.latency_session_sha256
        or _admission_sha256(checked_admission) != dependency.admission_sha256
        or any(
            _sha256(row.selected_checkpoint.raw) != row.selected_checkpoint.sha256
            for row in checked_cohort.rows
        )
    ):
        raise BaseCohortQualificationError("qualification inputs changed during assembly")
    return BaseCohortQualificationAssembly(
        qualification=qualification,
        execution_evidence=evidence,
        artifact=artifact,
        row_bindings=row_bindings,
        dependency_binding=dependency,
    )


__all__ = [
    "ASSEMBLY_SCHEMA",
    "AUTHORITY",
    "BaseCohortQualificationAssembly",
    "BaseCohortQualificationError",
    "BaseQualificationDependencyBinding",
    "CombinedBaseScoreBinding",
    "PRODUCTION",
    "RESULT_CLAIMED",
    "STATUS",
    "assemble_base_cohort_qualification",
]
