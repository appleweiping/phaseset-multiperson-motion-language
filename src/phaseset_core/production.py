"""Host-injected production adapter for the closed PhaseSet runner.

This module supplies the concrete bridge that the authority-zero CLI omits by
design. A trusted host authenticates private receipts and implements the eleven
registered commands. Public serializers receive only SHA-256 digests—never
dataset paths, server locators, credentials, or artifact contents.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import types
from typing import Protocol

import numpy as np

from phaseset_core import evaluation as evaluation_module
from phaseset_core import experiments as experiments_module
from phaseset_core import publication as publication_module
from phaseset_core import statistics as statistics_module
from phaseset_core import training as training_module
from phaseset_core.evaluation import (
    EvaluationReport,
    HardGalleryCollection,
    RetrievalDataset,
)
from phaseset_core.experiments import (
    FINAL_SYSTEMS,
    HYPOTHESIS_COMPARISONS,
    SEEDS,
    BaseQualification,
    BaseScore,
)
from phaseset_core.publication import (
    PublicationBundle,
    ResourceMeasurement,
)

from phaseset_core.execution import (
    COMMANDS,
    CommandIntent,
    CommandResult,
    RuntimeAdmission,
    RuntimeAdmissionRequest,
    SealedTestGate,
    canonical_command_result_bytes,
    canonical_runtime_admission_bytes,
    runtime_admission_sha256,
    runtime_handler_manifest_sha256,
)
from phaseset_core.statistics import (
    HypothesisComparison,
    StatisticalReport,
)


_SEMANTIC_FUNCTION_IDENTITIES = (
    (
        evaluation_module,
        "canonical_evaluation_report_bytes",
        evaluation_module.canonical_evaluation_report_bytes,
    ),
    (
        evaluation_module,
        "capture_cluster_commitments",
        evaluation_module.capture_cluster_commitments,
    ),
    (evaluation_module, "evaluate_retrieval", evaluation_module.evaluate_retrieval),
    (
        evaluation_module,
        "evaluate_hard_text_to_motion",
        evaluation_module.evaluate_hard_text_to_motion,
    ),
    (
        evaluation_module,
        "hard_gallery_collection_sha256",
        evaluation_module.hard_gallery_collection_sha256,
    ),
    (
        evaluation_module,
        "hard_gallery_freeze_binding_sha256",
        evaluation_module.hard_gallery_freeze_binding_sha256,
    ),
    (
        evaluation_module,
        "retrieval_census_sha256",
        evaluation_module.retrieval_census_sha256,
    ),
    (
        evaluation_module,
        "validate_retrieval_dataset",
        evaluation_module.validate_retrieval_dataset,
    ),
    (
        statistics_module,
        "build_statistical_report",
        statistics_module.build_statistical_report,
    ),
    (
        statistics_module,
        "canonical_statistical_report_bytes",
        statistics_module.canonical_statistical_report_bytes,
    ),
    (experiments_module, "qualify_base", experiments_module.qualify_base),
    (
        experiments_module,
        "canonical_base_qualification_bytes",
        experiments_module.canonical_base_qualification_bytes,
    ),
    (
        publication_module,
        "build_resource_report",
        publication_module.build_resource_report,
    ),
    (
        publication_module,
        "render_publication_bundle",
        publication_module.render_publication_bundle,
    ),
    (
        publication_module,
        "canonical_publication_manifest_bytes",
        publication_module.canonical_publication_manifest_bytes,
    ),
)
_SEMANTIC_MODULE_FUNCTION_IDENTITIES = tuple(
    (module, name, value)
    for module in (
        evaluation_module,
        statistics_module,
        experiments_module,
        publication_module,
    )
    for name, value in sorted(vars(module).items())
    if isinstance(value, types.FunctionType) and value.__module__ == module.__name__
)
# Python 3.12 lazily adds this interpreter metadata on ordinary annotation
# inspection. Materialize it before the census so introspection is not mistaken
# for an injected semantic global; its binding remains protected like all others.
for _semantic_module in (
    evaluation_module,
    statistics_module,
    experiments_module,
    publication_module,
):
    vars(_semantic_module).setdefault("__annotations__", {})
del _semantic_module
_SEMANTIC_MODULE_GLOBAL_NAME_CENSUS = tuple(
    (module, frozenset(vars(module)))
    for module in (
        evaluation_module,
        statistics_module,
        experiments_module,
        publication_module,
    )
)
_SEMANTIC_MODULE_GLOBAL_BINDING_IDENTITIES = tuple(
    (module, name, value)
    for module in (
        evaluation_module,
        statistics_module,
        experiments_module,
        publication_module,
    )
    for name, value in sorted(vars(module).items())
)
_SEMANTIC_NUMPY_IDENTITIES = tuple(
    (np, name, getattr(np, name))
    for name in (
        "abs",
        "add",
        "all",
        "any",
        "array_equal",
        "asarray",
        "ascontiguousarray",
        "bool_",
        "count_nonzero",
        "dtype",
        "empty",
        "empty_like",
        "float64",
        "generic",
        "int64",
        "isfinite",
        "ix_",
        "log1p",
        "max",
        "mean",
        "median",
        "ndarray",
        "random",
        "signbit",
        "sort",
        "sqrt",
        "stack",
        "uint64",
        "zeros",
    )
)
_SEMANTIC_RANDOM_IDENTITIES = (
    (np.random, "Generator", np.random.Generator),
    (np.random, "PCG64", np.random.PCG64),
)
_SEMANTIC_IMPORTED_BINDINGS = tuple(
    (module, name, getattr(module, name))
    for module, names in (
        (evaluation_module, ("hashlib", "json", "math", "np", "re")),
        (
            statistics_module,
            ("Fraction", "hashlib", "json", "math", "np", "re"),
        ),
        (experiments_module, ("Fraction", "Path", "hashlib", "json")),
        (
            publication_module,
            ("Fraction", "hashlib", "json", "math"),
        ),
    )
    for name in names
)
_SEMANTIC_BUILTIN_BINDINGS = tuple(
    (name, value) for name, value in sorted(vars(builtins).items())
)
_PRODUCTION_IMPORTED_BINDINGS = {
    "builtins": builtins,
    "hashlib": hashlib,
    "json": json,
    "np": np,
}
_PRODUCTION_GLOBAL_NAME_CENSUS: frozenset[str] = frozenset()
_PRODUCTION_GLOBAL_BINDING_IDENTITIES: tuple[tuple[str, object], ...] = ()


def _assert_semantic_function_integrity() -> None:
    if frozenset(globals()) != _PRODUCTION_GLOBAL_NAME_CENSUS:
        raise ProductionAdapterError(
            "production semantic global-name census changed before evidence computation"
        )
    for name, expected in _PRODUCTION_GLOBAL_BINDING_IDENTITIES:
        if globals().get(name) is not expected:
            raise ProductionAdapterError(
                "production semantic global binding changed before evidence computation: "
                f"{name}"
            )
    for module, expected_names in _SEMANTIC_MODULE_GLOBAL_NAME_CENSUS:
        if frozenset(vars(module)) != expected_names:
            raise ProductionAdapterError(
                "semantic module global-name census changed before evidence computation: "
                f"{module.__name__}"
            )
    for owner, name, expected in (
        _SEMANTIC_FUNCTION_IDENTITIES
        + _SEMANTIC_MODULE_FUNCTION_IDENTITIES
        + _SEMANTIC_MODULE_GLOBAL_BINDING_IDENTITIES
        + _SEMANTIC_NUMPY_IDENTITIES
        + _SEMANTIC_RANDOM_IDENTITIES
        + _SEMANTIC_IMPORTED_BINDINGS
    ):
        if getattr(owner, name, None) is not expected:
            raise ProductionAdapterError(
                f"semantic function identity changed before evidence computation: {name}"
            )
    for name, expected in _SEMANTIC_BUILTIN_BINDINGS:
        if vars(builtins).get(name) is not expected:
            raise ProductionAdapterError(
                "semantic builtin binding changed before evidence computation: "
                f"{name}"
            )
    for name, expected in _PRODUCTION_IMPORTED_BINDINGS.items():
        if globals().get(name) is not expected:
            raise ProductionAdapterError(
                f"semantic imported binding changed before evidence computation: {name}"
            )


class ProductionAdapterError(RuntimeError):
    """A private backend failed authentication or its closed output contract."""


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionAdapterError(f"{label} must be lowercase SHA-256 hex")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PrivateReceiptAssertions:
    """Digest-only assertions authenticated outside the public package."""

    admitted_at_utc: str
    data_manifest_sha256: str
    prepared_data_manifest_sha256: str
    split_audit_sha256: str
    rights_assertion_sha256: str
    caption_manifest_sha256: str
    score_execution_census_sha256: str
    validation_evaluation_census_sha256: str
    hard_gallery_freeze_binding_sha256: str
    validation_hard_gallery_collection_sha256: str
    runtime_assertion_sha256: str
    execution_assertion_sha256: str
    authority: int

    def __post_init__(self) -> None:
        for name in (
            "data_manifest_sha256",
            "prepared_data_manifest_sha256",
            "split_audit_sha256",
            "rights_assertion_sha256",
            "caption_manifest_sha256",
            "score_execution_census_sha256",
            "validation_evaluation_census_sha256",
            "hard_gallery_freeze_binding_sha256",
            "validation_hard_gallery_collection_sha256",
            "runtime_assertion_sha256",
            "execution_assertion_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if type(self.authority) is not int or self.authority <= 0:
            raise ProductionAdapterError("private authority must be a positive exact int")
        # Reuse the public validator for the exact UTC grammar by constructing
        # the complete record during admission. Keep this dataclass path-free.
        if type(self.admitted_at_utc) is not str:
            raise ProductionAdapterError("admitted_at_utc must be an exact string")


@dataclass(frozen=True, slots=True)
class PrivateSealedTestAuthorization:
    """Digest-only identity of the externally authenticated one-use test grant."""

    consumed_at_utc: str
    external_grant_sha256: str
    test_manifest_sha256: str
    caption_manifest_sha256: str
    evaluator_sha256: str
    validation_freeze_sha256: str
    aggregate_code_sha256: str
    evaluation_census_sha256: str
    hard_gallery_freeze_binding_sha256: str
    hard_gallery_collection_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "external_grant_sha256",
            "test_manifest_sha256",
            "caption_manifest_sha256",
            "evaluator_sha256",
            "validation_freeze_sha256",
            "aggregate_code_sha256",
            "evaluation_census_sha256",
            "hard_gallery_freeze_binding_sha256",
            "hard_gallery_collection_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if type(self.consumed_at_utc) is not str:
            raise ProductionAdapterError("consumed_at_utc must be an exact string")


@dataclass(frozen=True, slots=True)
class PrivateBaseQualificationAuthorization:
    """External authorization of the completed nine-row validation census."""

    authorized_at_utc: str
    external_grant_sha256: str
    validation_manifest_sha256: str
    query_census_sha256: str
    evaluator_sha256: str
    score_rows_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "external_grant_sha256",
            "validation_manifest_sha256",
            "query_census_sha256",
            "evaluator_sha256",
            "score_rows_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if type(self.authorized_at_utc) is not str:
            raise ProductionAdapterError("authorized_at_utc must be an exact string")


@dataclass(frozen=True, slots=True)
class PrivatePublicationAuthorization:
    """External authorization of one content-bound final render."""

    authorized_at_utc: str
    external_grant_sha256: str
    evaluation_aggregate_sha256: str
    statistics_sha256: str
    resource_report_sha256: str
    claim_report_sha256: str
    renderer_code_sha256: str
    publication_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "external_grant_sha256",
            "evaluation_aggregate_sha256",
            "statistics_sha256",
            "resource_report_sha256",
            "claim_report_sha256",
            "renderer_code_sha256",
            "publication_manifest_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if type(self.authorized_at_utc) is not str:
            raise ProductionAdapterError("authorized_at_utc must be an exact string")


@dataclass(frozen=True, slots=True)
class BackendExecution:
    """Private backend output before paths are replaced by content digests."""

    outcome: str
    completed_at_utc: str
    artifacts: tuple[Path, ...]
    semantic_evidence: object | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"COMPLETED", "FAILED", "HELD"}:
            raise ProductionAdapterError("backend outcome is outside the closed census")
        if type(self.completed_at_utc) is not str:
            raise ProductionAdapterError("completed_at_utc must be an exact string")
        if type(self.artifacts) is not tuple or not self.artifacts:
            raise ProductionAdapterError("backend must return at least one artifact")
        if any(not isinstance(path, Path) for path in self.artifacts):
            raise ProductionAdapterError("backend artifacts must be exact pathlib.Path objects")
        if len(set(self.artifacts)) != len(self.artifacts):
            raise ProductionAdapterError("backend artifact paths must be unique")


@dataclass(frozen=True, slots=True, repr=False)
class EvaluationCorpusProvenance:
    """Closed identity of the preregistered human-holistic corpus endpoint."""

    evaluation_type: str
    split: str
    caption_kind: str
    source_manifest_sha256: str
    prepared_data_manifest_sha256: str
    split_audit_sha256: str
    rights_assertion_sha256: str
    caption_manifest_sha256: str
    test_manifest_sha256: str | None
    sealed_test_consumption_sha256: str | None
    capture_commitments: tuple[bytes, ...]


@dataclass(frozen=True, slots=True, repr=False)
class SeedScoreEvidence:
    """One registered seed/checkpoint score table before seed aggregation."""

    seed: int
    checkpoint_sha256: str
    run_manifest_sha256: str
    terminal_sha256: str
    qualification_sha256: str
    capacity_receipt_sha256: str
    environment_sha256: str
    text_tower_sha256: str
    scoring_code_sha256: str
    dataset: RetrievalDataset


@dataclass(frozen=True, slots=True, repr=False)
class SystemScoreEvidence:
    """The three seed rows for one registered final system."""

    system_id: str
    seed_scores: tuple[SeedScoreEvidence, ...]


@dataclass(frozen=True, slots=True, repr=False)
class EvaluationExecutionEvidence:
    """Twenty-seven score rows from which the adapter recomputes nine reports."""

    provenance: EvaluationCorpusProvenance
    system_scores: tuple[SystemScoreEvidence, ...]
    evaluator_artifact: bytes
    hard_gallery_collection: HardGalleryCollection


@dataclass(frozen=True, slots=True, repr=False)
class BaseQualificationExecutionEvidence:
    score_rows: tuple[BaseScore, ...]
    selector_artifact: bytes


@dataclass(frozen=True, slots=True, repr=False)
class BootstrapExecutionEvidence:
    """Private evaluation census bound to one frozen aggregate and bootstrap seed."""

    evaluation_evidence: EvaluationExecutionEvidence
    evaluation_aggregate_artifact: bytes
    seed: int
    sealed_test_consumption_sha256: str
    sealed_test_evaluation_binding_sha256: str


@dataclass(frozen=True, slots=True, repr=False)
class PublicationExecutionEvidence:
    """Sealed statistics plus the exact formal resource profiler census."""

    bootstrap_evidence: BootstrapExecutionEvidence
    resource_measurements: tuple[ResourceMeasurement, ...]
    renderer_artifact: bytes


def current_evaluator_artifact_bytes() -> bytes:
    """Read the installed evaluator source bytes used by the semantic gate."""

    module_file = getattr(evaluation_module, "__file__", None)
    if type(module_file) is not str:
        raise ProductionAdapterError("installed evaluator source path is unavailable")
    path = Path(module_file)
    if path.suffix in {".pyc", ".pyo"}:
        path = path.with_suffix(".py")
    if path.is_symlink() or not path.is_file():
        raise ProductionAdapterError("installed evaluator source must be a regular file")
    raw = path.read_bytes()
    if not raw:
        raise ProductionAdapterError("installed evaluator source is empty")
    return raw


def current_statistics_artifact_bytes() -> bytes:
    """Read the installed aggregate/statistics source bytes used by the gate."""

    module_file = getattr(statistics_module, "__file__", None)
    if type(module_file) is not str:
        raise ProductionAdapterError("installed statistics source path is unavailable")
    path = Path(module_file)
    if path.suffix in {".pyc", ".pyo"}:
        path = path.with_suffix(".py")
    if path.is_symlink() or not path.is_file():
        raise ProductionAdapterError("installed statistics source must be a regular file")
    raw = path.read_bytes()
    if not raw:
        raise ProductionAdapterError("installed statistics source is empty")
    return raw


def _module_source_bytes(module: object, label: str) -> bytes:
    module_file = getattr(module, "__file__", None)
    if type(module_file) is not str:
        raise ProductionAdapterError(f"installed {label} source path is unavailable")
    path = Path(module_file)
    if path.suffix in {".pyc", ".pyo"}:
        path = path.with_suffix(".py")
    if path.is_symlink() or not path.is_file():
        raise ProductionAdapterError(f"installed {label} source must be a regular file")
    raw = path.read_bytes()
    if not raw:
        raise ProductionAdapterError(f"installed {label} source is empty")
    return raw


def current_scoring_artifact_bytes() -> bytes:
    """Bind every installed PhaseSet Python source that can affect inference."""

    package_root = Path(__file__).parent
    if package_root.is_symlink() or not package_root.is_dir():
        raise ProductionAdapterError("installed PhaseSet package root is invalid")
    paths = tuple(sorted(package_root.glob("*.py"), key=lambda path: path.name))
    if not paths:
        raise ProductionAdapterError("installed PhaseSet source census is empty")
    chunks = [b"phaseset-installed-python-source-census-v2\n"]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ProductionAdapterError("installed PhaseSet source census is irregular")
        raw = path.read_bytes()
        if not raw:
            raise ProductionAdapterError("installed PhaseSet source census contains an empty file")
        name = path.name.encode("ascii")
        chunks.extend(
            (
                len(name).to_bytes(4, "big"),
                name,
                len(raw).to_bytes(8, "big"),
                raw,
            )
        )
    return b"".join(chunks)


def current_base_qualification_artifact_bytes() -> bytes:
    """Bind the validation metric implementation and deterministic selector."""

    chunks = [b"phaseset-base-qualification-code-v1\n"]
    for label, module in (
        ("training", training_module),
        ("experiments", experiments_module),
    ):
        name = label.encode("ascii")
        raw = _module_source_bytes(module, label)
        chunks.extend(
            (
                len(name).to_bytes(4, "big"),
                name,
                len(raw).to_bytes(8, "big"),
                raw,
            )
        )
    return b"".join(chunks)


def current_publication_renderer_artifact_bytes() -> bytes:
    """Read the installed content-bound publication renderer source."""

    return _module_source_bytes(publication_module, "publication renderer")


def _evaluation_census_sha256(dataset: RetrievalDataset) -> str:
    _assert_semantic_function_integrity()
    return evaluation_module.retrieval_census_sha256(dataset)


def _corpus_provenance_bytes(value: object) -> bytes:
    if type(value) is not EvaluationCorpusProvenance:
        raise TypeError("provenance must be exact EvaluationCorpusProvenance")
    if value.evaluation_type != "REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL":
        raise ProductionAdapterError("formal evaluation type must be real human holistic")
    if value.caption_kind != "HUMAN_HOLISTIC":
        raise ProductionAdapterError("formal primary captions must be human holistic")
    if value.split not in {"validation", "test"}:
        raise ProductionAdapterError("evaluation provenance split is invalid")
    digest_fields = (
        "source_manifest_sha256",
        "prepared_data_manifest_sha256",
        "split_audit_sha256",
        "rights_assertion_sha256",
        "caption_manifest_sha256",
    )
    for name in digest_fields:
        _lower_sha256(getattr(value, name), name)
    if value.split == "test":
        _lower_sha256(value.test_manifest_sha256, "test_manifest_sha256")
        _lower_sha256(
            value.sealed_test_consumption_sha256,
            "sealed_test_consumption_sha256",
        )
    elif value.test_manifest_sha256 is not None or (
        value.sealed_test_consumption_sha256 is not None
    ):
        raise ProductionAdapterError("validation evidence cannot bind sealed-test fields")
    commitments = value.capture_commitments
    if (
        type(commitments) is not tuple
        or len(commitments) < 2
        or any(type(item) is not bytes or len(item) != 32 for item in commitments)
        or commitments != tuple(sorted(commitments))
        or len(commitments) != len(set(commitments))
    ):
        raise ProductionAdapterError("capture commitments are not canonical")
    return _canonical_json_bytes(
        {
            "caption_kind": value.caption_kind,
            "caption_manifest_sha256": value.caption_manifest_sha256,
            "capture_commitments": [item.hex() for item in commitments],
            "evaluation_type": value.evaluation_type,
            "prepared_data_manifest_sha256": value.prepared_data_manifest_sha256,
            "rights_assertion_sha256": value.rights_assertion_sha256,
            "schema": "phaseset-evaluation-corpus-provenance-v1",
            "sealed_test_consumption_sha256": value.sealed_test_consumption_sha256,
            "source_manifest_sha256": value.source_manifest_sha256,
            "split": value.split,
            "split_audit_sha256": value.split_audit_sha256,
            "test_manifest_sha256": value.test_manifest_sha256,
        }
    )


def _same_query_census(left: RetrievalDataset, right: RetrievalDataset) -> bool:
    return (
        left.motion_commitments == right.motion_commitments
        and left.caption_commitments == right.caption_commitments
        and left.positive_motion_indices == right.positive_motion_indices
        and left.component_labels == right.component_labels
        and np.array_equal(left.group_sizes, right.group_sizes)
    )


def _recompute_evaluation_evidence(
    value: object,
) -> tuple[
    bytes,
    dict[str, tuple[EvaluationReport, ...]],
    str,
    str,
    str,
    str,
]:
    _assert_semantic_function_integrity()
    if type(value) is not EvaluationExecutionEvidence:
        raise TypeError("value must be exact EvaluationExecutionEvidence")
    provenance_raw = _corpus_provenance_bytes(value.provenance)
    evaluator_artifact = value.evaluator_artifact
    if type(evaluator_artifact) is not bytes or not evaluator_artifact:
        raise ProductionAdapterError("evaluator artifact must be nonempty exact bytes")
    if evaluator_artifact != current_evaluator_artifact_bytes():
        raise ProductionAdapterError("evaluator artifact differs from installed source bytes")
    expected_system_ids = tuple(system_id for system_id, _ in FINAL_SYSTEMS)
    if (
        type(value.system_scores) is not tuple
        or any(type(row) is not SystemScoreEvidence for row in value.system_scores)
        or tuple(row.system_id for row in value.system_scores) != expected_system_ids
    ):
        raise ProductionAdapterError("evaluation evidence must contain ordered systems 00-08")

    scoring_code_sha256 = hashlib.sha256(current_scoring_artifact_bytes()).hexdigest()
    reference: RetrievalDataset | None = None
    checked_by_system: dict[
        str,
        list[tuple[SeedScoreEvidence, RetrievalDataset, str, dict[str, object]]],
    ] = {}
    execution_rows: list[dict[str, object]] = []
    checkpoint_census: set[str] = set()
    run_manifest_census: set[str] = set()
    terminal_census: set[str] = set()
    for system in value.system_scores:
        if (
            type(system.seed_scores) is not tuple
            or any(type(row) is not SeedScoreEvidence for row in system.seed_scores)
            or tuple(row.seed for row in system.seed_scores) != SEEDS
        ):
            raise ProductionAdapterError(
                f"system {system.system_id} must bind seeds 1729,2718,31415"
            )
        checked_seed_rows: list[
            tuple[SeedScoreEvidence, RetrievalDataset, str, dict[str, object]]
        ] = []
        for seed_row in system.seed_scores:
            checkpoint_sha256 = _lower_sha256(
                seed_row.checkpoint_sha256,
                f"{system.system_id}/{seed_row.seed}.checkpoint_sha256",
            )
            if checkpoint_sha256 in checkpoint_census:
                raise ProductionAdapterError("the 27 checkpoint identities must be unique")
            checkpoint_census.add(checkpoint_sha256)
            row_digests = {
                name: _lower_sha256(
                    getattr(seed_row, name),
                    f"{system.system_id}/{seed_row.seed}.{name}",
                )
                for name in (
                    "run_manifest_sha256",
                    "terminal_sha256",
                    "qualification_sha256",
                    "capacity_receipt_sha256",
                    "environment_sha256",
                    "text_tower_sha256",
                )
            }
            if row_digests["run_manifest_sha256"] in run_manifest_census:
                raise ProductionAdapterError("the 27 run manifests must be unique")
            if row_digests["terminal_sha256"] in terminal_census:
                raise ProductionAdapterError("the 27 terminal receipts must be unique")
            run_manifest_census.add(row_digests["run_manifest_sha256"])
            terminal_census.add(row_digests["terminal_sha256"])
            if seed_row.scoring_code_sha256 != scoring_code_sha256:
                raise ProductionAdapterError("score row differs from installed scoring code")
            try:
                dataset = evaluation_module.validate_retrieval_dataset(
                    seed_row.dataset
                )
            except (TypeError, ValueError) as error:
                raise ProductionAdapterError(
                    f"system {system.system_id} seed {seed_row.seed} dataset is invalid"
                ) from error
            if reference is None:
                reference = dataset
            elif not _same_query_census(reference, dataset):
                raise ProductionAdapterError(
                    f"system {system.system_id} seed {seed_row.seed} changed the query census"
                )
            score_sha256 = hashlib.sha256(
                dataset.scores.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest()
            execution_row: dict[str, object] = {
                "capacity_receipt_sha256": row_digests["capacity_receipt_sha256"],
                "checkpoint_sha256": checkpoint_sha256,
                "environment_sha256": row_digests["environment_sha256"],
                "qualification_sha256": row_digests["qualification_sha256"],
                "run_manifest_sha256": row_digests["run_manifest_sha256"],
                "scoring_code_sha256": scoring_code_sha256,
                "seed": seed_row.seed,
                "system_id": system.system_id,
                "terminal_sha256": row_digests["terminal_sha256"],
                "text_tower_sha256": row_digests["text_tower_sha256"],
            }
            checked_seed_rows.append(
                (seed_row, dataset, score_sha256, execution_row)
            )
            execution_rows.append(execution_row)
        checked_by_system[system.system_id] = checked_seed_rows

    if reference is None:
        raise AssertionError("closed system census cannot be empty")
    collection = value.hard_gallery_collection
    if type(collection) is not HardGalleryCollection:
        raise ProductionAdapterError(
            "evaluation evidence must bind one exact frozen hard-gallery collection"
        )
    try:
        hard_collection_sha256 = evaluation_module.hard_gallery_collection_sha256(
            reference,
            collection,
        )
        hard_freeze_binding_sha256 = (
            evaluation_module.hard_gallery_freeze_binding_sha256(
            collection.freeze_binding
            )
        )
    except (TypeError, ValueError) as error:
        raise ProductionAdapterError("hard-gallery collection evidence is invalid") from error
    if (
        collection.freeze_binding.caption_manifest_sha256
        != value.provenance.caption_manifest_sha256
        or collection.freeze_binding.motion_manifest_sha256
        != value.provenance.prepared_data_manifest_sha256
    ):
        raise ProductionAdapterError(
            "hard-gallery freeze binding differs from the evaluation manifests"
        )
    expected_checkpoint_census = tuple(sorted(checkpoint_census))
    if (
        collection.scorer_provenance.evaluated_checkpoint_sha256s
        != expected_checkpoint_census
    ):
        raise ProductionAdapterError(
            "hard-gallery scorer did not exclude the exact 27 evaluated checkpoints"
        )
    if (
        evaluation_module.capture_cluster_commitments(reference)
        != value.provenance.capture_commitments
    ):
        raise ProductionAdapterError(
            "declared capture identities differ from the positive-relation census"
        )
    census_sha256 = _evaluation_census_sha256(reference)
    census_provenance_sha256 = hashlib.sha256(
        provenance_raw + census_sha256.encode("ascii") + b"\n"
    ).hexdigest()
    score_execution_census_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            {
                "rows": execution_rows,
                "schema": "phaseset-score-execution-census-v1",
            }
        )
    ).hexdigest()
    reports: dict[str, tuple[EvaluationReport, ...]] = {}
    score_rows: list[dict[str, object]] = []
    system_summaries: list[dict[str, object]] = []
    for system_id in expected_system_ids:
        seed_reports: list[EvaluationReport] = []
        seed_primary: list[float] = []
        seed_hard_r1: list[float] = []
        for seed_row, dataset, score_sha256, execution_row in checked_by_system[system_id]:
            score_provenance_sha256 = hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "census_provenance_sha256": census_provenance_sha256,
                        "execution": execution_row,
                        "schema": "phaseset-seed-score-provenance-v2",
                        "score_sha256": score_sha256,
                    }
                )
            ).hexdigest()
            report = replace(
                evaluation_module.evaluate_retrieval(dataset),
                evaluation_type=value.provenance.evaluation_type,
                census_provenance_sha256=census_provenance_sha256,
                score_provenance_sha256=score_provenance_sha256,
            )
            seed_reports.append(report)
            seed_primary.append(report.full_gallery.primary)
            try:
                hard_metrics = evaluation_module.evaluate_hard_text_to_motion(
                    dataset,
                    collection,
                )
            except (TypeError, ValueError) as error:
                raise ProductionAdapterError(
                    "hard-gallery evaluation changed its query census or lineage"
                ) from error
            seed_hard_r1.append(hard_metrics.recall_at_1)
            score_rows.append(
                {
                    **execution_row,
                    "evaluation_report_sha256": hashlib.sha256(
                        evaluation_module.canonical_evaluation_report_bytes(report)
                    ).hexdigest(),
                    "primary_hex": report.full_gallery.primary.hex(),
                    "hard_text_to_motion_r1_hex": hard_metrics.recall_at_1.hex(),
                    "score_provenance_sha256": score_provenance_sha256,
                    "score_sha256": score_sha256,
                }
            )
        reports[system_id] = tuple(seed_reports)
        primary_values = np.asarray(seed_primary, dtype=np.float64)
        primary_mean = float(np.add.reduce(primary_values, dtype=np.float64) / len(SEEDS))
        deviations = primary_values - primary_mean
        primary_population_std = float(
            np.sqrt(
                np.add.reduce(deviations * deviations, dtype=np.float64) / len(SEEDS)
            )
        )
        hard_values = np.asarray(seed_hard_r1, dtype=np.float64)
        hard_mean = float(np.add.reduce(hard_values, dtype=np.float64) / len(SEEDS))
        hard_deviations = hard_values - hard_mean
        hard_population_std = float(
            np.sqrt(
                np.add.reduce(
                    hard_deviations * hard_deviations,
                    dtype=np.float64,
                )
                / len(SEEDS)
            )
        )
        system_summaries.append(
            {
                "aggregation": "MEAN_OF_THREE_INDEPENDENT_SEED_METRICS_NO_LOGIT_ENSEMBLE",
                "primary_mean_hex": primary_mean.hex(),
                "primary_population_std_hex": primary_population_std.hex(),
                "hard_text_to_motion_r1_mean_hex": hard_mean.hex(),
                "hard_text_to_motion_r1_population_std_hex": (
                    hard_population_std.hex()
                ),
                "seed_evaluation_report_sha256s": [
                    hashlib.sha256(
                        evaluation_module.canonical_evaluation_report_bytes(report)
                    ).hexdigest()
                    for report in seed_reports
                ],
                "system_id": system_id,
            }
        )
    evaluator_sha256 = hashlib.sha256(evaluator_artifact).hexdigest()
    raw = _canonical_json_bytes(
        {
            "caption_kind": value.provenance.caption_kind,
            "caption_manifest_sha256": value.provenance.caption_manifest_sha256,
            "census_provenance_sha256": census_provenance_sha256,
            "census_sha256": census_sha256,
            "evaluation_type": value.provenance.evaluation_type,
            "evaluator_sha256": evaluator_sha256,
            "hard_gallery_collection_sha256": hard_collection_sha256,
            "hard_gallery_freeze_binding_sha256": hard_freeze_binding_sha256,
            "score_rows": score_rows,
            "schema": "phaseset-evaluation-aggregate-v3",
            "score_execution_census_sha256": score_execution_census_sha256,
            "sealed_test_consumption_sha256": (
                value.provenance.sealed_test_consumption_sha256
            ),
            "split": value.provenance.split,
            "systems": system_summaries,
            "test_manifest_sha256": value.provenance.test_manifest_sha256,
        }
    )
    return (
        raw,
        reports,
        score_execution_census_sha256,
        census_sha256,
        hard_freeze_binding_sha256,
        hard_collection_sha256,
    )


def build_evaluation_execution_artifact(value: object) -> bytes:
    """Recompute the exact nine-system validation/test aggregate."""

    if type(value) is not EvaluationExecutionEvidence:
        raise TypeError("value must be exact EvaluationExecutionEvidence")
    raw, _, _, _, _, _ = _recompute_evaluation_evidence(value)
    return raw


def _recompute_base_qualification_evidence(
    value: object,
) -> tuple[bytes, BaseQualification]:
    _assert_semantic_function_integrity()
    if type(value) is not BaseQualificationExecutionEvidence:
        raise TypeError("value must be exact BaseQualificationExecutionEvidence")
    selector_artifact = value.selector_artifact
    if type(selector_artifact) is not bytes or not selector_artifact:
        raise ProductionAdapterError(
            "base-qualification selector artifact must be nonempty exact bytes"
        )
    installed_artifact = current_base_qualification_artifact_bytes()
    if selector_artifact != installed_artifact:
        raise ProductionAdapterError(
            "base-qualification selector artifact differs from installed sources"
        )
    evaluator_sha256 = hashlib.sha256(installed_artifact).hexdigest()
    if (
        type(value.score_rows) is not tuple
        or any(type(row) is not BaseScore for row in value.score_rows)
        or any(row.evaluator_sha256 != evaluator_sha256 for row in value.score_rows)
    ):
        raise ProductionAdapterError(
            "base score rows must bind the installed validation evaluator/selector"
        )
    qualification = experiments_module.qualify_base(value.score_rows)
    raw = experiments_module.canonical_base_qualification_bytes(qualification)
    return raw, qualification


def build_base_qualification_execution_artifact(value: object) -> bytes:
    """Rebuild the authorized nine-row validation qualification artifact."""

    raw, _ = _recompute_base_qualification_evidence(value)
    return raw


def _build_formal_report(value: BootstrapExecutionEvidence) -> StatisticalReport:
    _assert_semantic_function_integrity()
    _lower_sha256(
        value.sealed_test_consumption_sha256,
        "sealed_test_consumption_sha256",
    )
    _lower_sha256(
        value.sealed_test_evaluation_binding_sha256,
        "sealed_test_evaluation_binding_sha256",
    )
    if value.evaluation_evidence.provenance.split != "test":
        raise ProductionAdapterError("formal bootstrap requires sealed test evidence")
    evaluation_raw, reports, _, _, _, _ = _recompute_evaluation_evidence(
        value.evaluation_evidence
    )
    if type(value.evaluation_aggregate_artifact) is not bytes or (
        value.evaluation_aggregate_artifact != evaluation_raw
    ):
        raise ProductionAdapterError(
            "bootstrap evidence differs from the sealed evaluation aggregate"
        )
    comparisons = tuple(
        HypothesisComparison(
            hypothesis_id=hypothesis_id,
            treatment_system_id=treatment_system_id,
            baseline_system_id=baseline_system_id,
            training_seeds=SEEDS,
            treatment_evaluations=reports[treatment_system_id],
            baseline_evaluations=reports[baseline_system_id],
            resampling_seed=value.seed,
            aggregate_artifact=evaluation_raw,
            evaluator_artifact=value.evaluation_evidence.evaluator_artifact,
        )
        for hypothesis_id, treatment_system_id, baseline_system_id in (
            HYPOTHESIS_COMPARISONS
        )
    )
    return statistics_module.build_statistical_report(comparisons)


def build_bootstrap_execution_artifact(value: object) -> bytes:
    """Rebuild the formal H1-H8 report from sealed evaluation evidence."""

    if type(value) is not BootstrapExecutionEvidence:
        raise TypeError("value must be exact BootstrapExecutionEvidence")
    _assert_semantic_function_integrity()
    return statistics_module.canonical_statistical_report_bytes(
        _build_formal_report(value)
    )


def _recompute_publication_evidence(
    value: object,
) -> tuple[PublicationBundle, bytes]:
    _assert_semantic_function_integrity()
    if type(value) is not PublicationExecutionEvidence:
        raise TypeError("value must be exact PublicationExecutionEvidence")
    if (
        type(value.renderer_artifact) is not bytes
        or not value.renderer_artifact
        or value.renderer_artifact != current_publication_renderer_artifact_bytes()
    ):
        raise ProductionAdapterError(
            "publication renderer artifact differs from installed source"
        )
    bootstrap = value.bootstrap_evidence
    if type(bootstrap) is not BootstrapExecutionEvidence:
        raise ProductionAdapterError(
            "publication evidence must contain exact sealed bootstrap evidence"
        )
    evaluation_raw, _, _, _, _, _ = _recompute_evaluation_evidence(
        bootstrap.evaluation_evidence
    )
    if bootstrap.evaluation_aggregate_artifact != evaluation_raw:
        raise ProductionAdapterError(
            "publication evaluation aggregate differs from sealed score evidence"
        )
    report = _build_formal_report(bootstrap)
    resources = publication_module.build_resource_report(
        value.resource_measurements,
        evaluation_aggregate=evaluation_raw,
    )
    bundle = publication_module.render_publication_bundle(
        evaluation_aggregate=evaluation_raw,
        statistics=report,
        resources=resources,
        require_real_resources=True,
    )
    manifest = publication_module.canonical_publication_manifest_bytes(bundle)
    return bundle, manifest


def build_publication_execution_artifacts(
    value: object,
) -> tuple[bytes, ...]:
    """Return the exact content set required from completed ``render-paper``."""

    bundle, manifest = _recompute_publication_evidence(value)
    return tuple(row.content for row in bundle.artifacts) + (manifest,)


def _require_semantic_artifact(
    private_result: BackendExecution,
    expected: bytes,
    label: str,
) -> None:
    matches = 0
    for path in private_result.artifacts:
        if path.is_symlink() or not path.is_file():
            continue
        if path.read_bytes() == expected:
            matches += 1
    if matches != 1:
        raise ProductionAdapterError(
            f"completed {label} must return exactly one recomputed canonical artifact"
        )


def _require_semantic_artifact_set(
    private_result: BackendExecution,
    expected: tuple[bytes, ...],
    label: str,
) -> None:
    if len(private_result.artifacts) != len(expected):
        raise ProductionAdapterError(
            f"completed {label} returned the wrong artifact count"
        )
    remaining = list(expected)
    for path in private_result.artifacts:
        if path.is_symlink() or not path.is_file():
            raise ProductionAdapterError(
                f"completed {label} returned a non-regular artifact"
            )
        raw = path.read_bytes()
        try:
            index = remaining.index(raw)
        except ValueError as error:
            raise ProductionAdapterError(
                f"completed {label} returned content outside the canonical bundle"
            ) from error
        del remaining[index]
    if remaining:
        raise ProductionAdapterError(
            f"completed {label} omitted canonical publication artifacts"
        )


class PrivateCommandBackend(Protocol):
    """Trusted host implementation; never imported from a public CLI option."""

    backend_sha256: str

    def authenticate(
        self,
        request: RuntimeAdmissionRequest,
        receipts: PrivateReceiptAssertions,
        adapter_sha256: str,
    ) -> bool:
        """Authenticate rights/runtime/execution assertions out of process."""

    def execute(self, intent: CommandIntent) -> BackendExecution:
        """Execute one already-admitted command and return immutable artifacts."""

    def authenticate_sealed_test(
        self,
        intent: CommandIntent,
        authorization: PrivateSealedTestAuthorization,
        adapter_sha256: str,
    ) -> bool:
        """Authenticate the frozen one-use test grant out of process."""

    def authenticate_base_qualification(
        self,
        intent: CommandIntent,
        authorization: PrivateBaseQualificationAuthorization,
        qualification_sha256: str,
        adapter_sha256: str,
    ) -> bool:
        """Authenticate the externally frozen nine-row validation census."""

    def authenticate_publication(
        self,
        intent: CommandIntent,
        authorization: PrivatePublicationAuthorization,
        publication_manifest_sha256: str,
        adapter_sha256: str,
    ) -> bool:
        """Authenticate one content-bound resource/claim/paper render."""


class ProductionRuntimeAdapter:
    """Digest-bound implementation of :class:`execution.RuntimeAdapter`."""

    commands = COMMANDS
    handler_manifest_sha256 = runtime_handler_manifest_sha256(COMMANDS)

    __slots__ = (
        "_admission",
        "_backend",
        "_base_qualification_authorization",
        "_receipts",
        "_request",
        "_publication_authorization",
        "_sealed_test_authorization",
        "_sealed_test_consumption_sha256",
        "_sealed_test_evaluation_binding_sha256",
        "_sealed_test_gate",
        "adapter_sha256",
    )

    def __init__(
        self,
        backend: PrivateCommandBackend,
        receipts: PrivateReceiptAssertions,
        *,
        sealed_test_ledger_root: Path | None = None,
        sealed_test_authorization: PrivateSealedTestAuthorization | None = None,
        base_qualification_authorization: (
            PrivateBaseQualificationAuthorization | None
        ) = None,
        publication_authorization: PrivatePublicationAuthorization | None = None,
    ) -> None:
        if type(receipts) is not PrivateReceiptAssertions:
            raise TypeError("receipts must be exactly PrivateReceiptAssertions")
        backend_digest = _lower_sha256(
            getattr(backend, "backend_sha256", None),
            "backend_sha256",
        )
        if not callable(getattr(backend, "authenticate", None)) or not callable(
            getattr(backend, "execute", None)
        ):
            raise ProductionAdapterError("private backend surface is incomplete")
        self._backend = backend
        self._receipts = receipts
        if base_qualification_authorization is not None and type(
            base_qualification_authorization
        ) is not PrivateBaseQualificationAuthorization:
            raise TypeError(
                "base_qualification_authorization must be exactly "
                "PrivateBaseQualificationAuthorization"
            )
        self._base_qualification_authorization = base_qualification_authorization
        if publication_authorization is not None and type(
            publication_authorization
        ) is not PrivatePublicationAuthorization:
            raise TypeError(
                "publication_authorization must be exactly "
                "PrivatePublicationAuthorization"
            )
        self._publication_authorization = publication_authorization
        self._request: RuntimeAdmissionRequest | None = None
        self._admission: RuntimeAdmission | None = None
        if (sealed_test_ledger_root is None) != (sealed_test_authorization is None):
            raise ProductionAdapterError(
                "sealed-test ledger and authorization must be configured together"
            )
        if sealed_test_authorization is not None and type(
            sealed_test_authorization
        ) is not PrivateSealedTestAuthorization:
            raise TypeError(
                "sealed_test_authorization must be exactly PrivateSealedTestAuthorization"
            )
        self._sealed_test_gate = (
            None
            if sealed_test_ledger_root is None
            else SealedTestGate(sealed_test_ledger_root)
        )
        self._sealed_test_authorization = sealed_test_authorization
        self._sealed_test_consumption_sha256: str | None = None
        self._sealed_test_evaluation_binding_sha256: str | None = None
        self.adapter_sha256 = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "backend_sha256": backend_digest,
                    "commands": list(COMMANDS),
                    "handler_manifest_sha256": self.handler_manifest_sha256,
                    "schema": "phaseset-production-runtime-adapter-v3",
                }
            )
        ).hexdigest()

    def admit(self, request: RuntimeAdmissionRequest) -> RuntimeAdmission:
        if type(request) is not RuntimeAdmissionRequest:
            raise TypeError("request must be exactly RuntimeAdmissionRequest")
        if self._request is not None:
            if request != self._request or self._admission is None:
                raise ProductionAdapterError("runtime admission request changed after authentication")
            return self._admission
        authenticated = self._backend.authenticate(
            request,
            self._receipts,
            self.adapter_sha256,
        )
        if authenticated is not True:
            raise ProductionAdapterError("private backend did not authenticate admission")
        receipts = self._receipts
        admission = RuntimeAdmission(
            runtime_mode="PRIVATE_AUTHORIZED",
            admitted_at_utc=receipts.admitted_at_utc,
            adapter_sha256=self.adapter_sha256,
            handler_manifest_sha256=request.handler_manifest_sha256,
            plan_sha256=request.plan_sha256,
            matrix_sha256=request.matrix_sha256,
            training_config_sha256=request.training_config_sha256,
            data_manifest_sha256=receipts.data_manifest_sha256,
            prepared_data_manifest_sha256=receipts.prepared_data_manifest_sha256,
            split_audit_sha256=receipts.split_audit_sha256,
            rights_assertion_sha256=receipts.rights_assertion_sha256,
            runtime_assertion_sha256=receipts.runtime_assertion_sha256,
            execution_assertion_sha256=receipts.execution_assertion_sha256,
            authority=receipts.authority,
            execution_authorized=True,
            production=True,
            external_authentication_asserted=True,
            public_verification_performed=False,
            result_claimed=False,
            status="PRIVATE_ADAPTER_ASSERTED_AUTHORIZATION_NOT_PUBLICLY_VERIFIED",
        )
        canonical_runtime_admission_bytes(admission)
        self._request = request
        self._admission = admission
        return admission

    def _validate_authorized_code(self) -> None:
        authorization = self._sealed_test_authorization
        if authorization is None:
            raise ProductionAdapterError("sealed-test authorization is absent")
        installed_evaluator_sha256 = hashlib.sha256(
            current_evaluator_artifact_bytes()
        ).hexdigest()
        installed_statistics_sha256 = hashlib.sha256(
            current_statistics_artifact_bytes()
        ).hexdigest()
        if (
            authorization.evaluator_sha256 != installed_evaluator_sha256
            or authorization.aggregate_code_sha256 != installed_statistics_sha256
        ):
            raise ProductionAdapterError(
                "sealed-test authorization code digests differ from installed sources"
            )

    def _validate_evaluation_binding(
        self,
        evidence: EvaluationExecutionEvidence,
        *,
        split: str,
        sealed_test_consumption_sha256: str | None,
        score_execution_census_sha256: str,
        evaluation_census_sha256: str,
        hard_gallery_freeze_binding_sha256: str,
        hard_gallery_collection_sha256: str,
    ) -> None:
        provenance = evidence.provenance
        receipts = self._receipts
        expected = {
            "source_manifest_sha256": receipts.data_manifest_sha256,
            "prepared_data_manifest_sha256": receipts.prepared_data_manifest_sha256,
            "split_audit_sha256": receipts.split_audit_sha256,
            "rights_assertion_sha256": receipts.rights_assertion_sha256,
            "caption_manifest_sha256": receipts.caption_manifest_sha256,
        }
        for name, expected_digest in expected.items():
            if getattr(provenance, name) != expected_digest:
                raise ProductionAdapterError(
                    f"evaluation provenance {name} differs from authenticated admission"
                )
        if score_execution_census_sha256 != receipts.score_execution_census_sha256:
            raise ProductionAdapterError(
                "score execution census differs from authenticated admission"
            )
        if (
            hard_gallery_freeze_binding_sha256
            != receipts.hard_gallery_freeze_binding_sha256
        ):
            raise ProductionAdapterError(
                "hard-gallery freeze binding differs from authenticated admission"
            )
        if provenance.split != split:
            raise ProductionAdapterError("evaluation evidence split differs from intent")
        if split == "test":
            authorization = self._sealed_test_authorization
            if authorization is None:
                raise ProductionAdapterError("test evaluation authorization is absent")
            self._validate_authorized_code()
            if (
                provenance.test_manifest_sha256
                != authorization.test_manifest_sha256
                or provenance.caption_manifest_sha256
                != authorization.caption_manifest_sha256
                or provenance.sealed_test_consumption_sha256
                != sealed_test_consumption_sha256
                or evaluation_census_sha256
                != authorization.evaluation_census_sha256
                or hard_gallery_freeze_binding_sha256
                != authorization.hard_gallery_freeze_binding_sha256
                or hard_gallery_collection_sha256
                != authorization.hard_gallery_collection_sha256
            ):
                raise ProductionAdapterError(
                    "test evaluation evidence differs from the consumed sealed grant"
                )
        else:
            if sealed_test_consumption_sha256 is not None:
                raise ProductionAdapterError(
                    "validation evaluation cannot consume the test grant"
                )
            if (
                evaluation_census_sha256
                != receipts.validation_evaluation_census_sha256
                or hard_gallery_collection_sha256
                != receipts.validation_hard_gallery_collection_sha256
            ):
                raise ProductionAdapterError(
                    "validation GT/gallery census differs from authenticated admission"
                )

    def _validate_base_qualification_binding(
        self,
        intent: CommandIntent,
        qualification: BaseQualification,
        qualification_raw: bytes,
    ) -> None:
        authorization = self._base_qualification_authorization
        if authorization is None:
            raise ProductionAdapterError(
                "completed base qualification requires external authorization"
            )
        expected = {
            "validation_manifest_sha256": qualification.validation_manifest_sha256,
            "query_census_sha256": qualification.query_census_sha256,
            "evaluator_sha256": qualification.evaluator_sha256,
            "score_rows_sha256": qualification.score_rows_sha256,
        }
        for name, actual in expected.items():
            if getattr(authorization, name) != actual:
                raise ProductionAdapterError(
                    f"base qualification {name} differs from external authorization"
                )
        authenticate = getattr(
            self._backend,
            "authenticate_base_qualification",
            None,
        )
        qualification_sha256 = hashlib.sha256(qualification_raw).hexdigest()
        if not callable(authenticate) or authenticate(
            intent,
            authorization,
            qualification_sha256,
            self.adapter_sha256,
        ) is not True:
            raise ProductionAdapterError(
                "private backend did not authenticate base qualification"
            )

    def _verify_sealed_bootstrap_evidence(
        self,
        evidence: object,
    ) -> bytes:
        if type(evidence) is not BootstrapExecutionEvidence:
            raise ProductionAdapterError(
                "bootstrap requires exact sealed evaluation evidence"
            )
        gate = self._sealed_test_gate
        authorization = self._sealed_test_authorization
        if gate is None or authorization is None:
            raise ProductionAdapterError(
                "bootstrap requires the persistent sealed-test evaluation ledger"
            )
        self._validate_authorized_code()
        (
            _,
            _,
            score_execution_census_sha256,
            evaluation_census_sha256,
            hard_freeze_binding_sha256,
            hard_collection_sha256,
        ) = _recompute_evaluation_evidence(evidence.evaluation_evidence)
        self._validate_evaluation_binding(
            evidence.evaluation_evidence,
            split="test",
            sealed_test_consumption_sha256=(
                evidence.sealed_test_consumption_sha256
            ),
            score_execution_census_sha256=score_execution_census_sha256,
            evaluation_census_sha256=evaluation_census_sha256,
            hard_gallery_freeze_binding_sha256=hard_freeze_binding_sha256,
            hard_gallery_collection_sha256=hard_collection_sha256,
        )
        try:
            consumption_sha256 = gate.verify_consumption(
                external_grant_sha256=authorization.external_grant_sha256,
                test_manifest_sha256=authorization.test_manifest_sha256,
                caption_manifest_sha256=authorization.caption_manifest_sha256,
                evaluator_sha256=authorization.evaluator_sha256,
                validation_freeze_sha256=authorization.validation_freeze_sha256,
                aggregate_code_sha256=authorization.aggregate_code_sha256,
                evaluation_census_sha256=authorization.evaluation_census_sha256,
                hard_gallery_freeze_binding_sha256=(
                    authorization.hard_gallery_freeze_binding_sha256
                ),
                hard_gallery_collection_sha256=(
                    authorization.hard_gallery_collection_sha256
                ),
                consumed_at_utc=authorization.consumed_at_utc,
            )
            aggregate_sha256, binding_sha256 = gate.verified_evaluation_binding(
                consumption_sha256=consumption_sha256,
                evaluator_sha256=authorization.evaluator_sha256,
                aggregate_code_sha256=authorization.aggregate_code_sha256,
            )
        except (ValueError, TypeError) as error:
            raise ProductionAdapterError(
                "bootstrap sealed-test evaluation ledger verification failed"
            ) from error
        if (
            evidence.sealed_test_consumption_sha256 != consumption_sha256
            or evidence.sealed_test_evaluation_binding_sha256 != binding_sha256
            or hashlib.sha256(evidence.evaluation_aggregate_artifact).hexdigest()
            != aggregate_sha256
        ):
            raise ProductionAdapterError(
                "bootstrap evidence is not the consumed sealed-test evaluation"
            )
        return build_bootstrap_execution_artifact(evidence)

    def _validate_publication_binding(
        self,
        intent: CommandIntent,
        bundle: PublicationBundle,
        manifest: bytes,
    ) -> None:
        authorization = self._publication_authorization
        if authorization is None:
            raise ProductionAdapterError(
                "completed publication render requires external authorization"
            )
        renderer_code_sha256 = hashlib.sha256(
            current_publication_renderer_artifact_bytes()
        ).hexdigest()
        actual = {
            "evaluation_aggregate_sha256": bundle.evaluation_aggregate_sha256,
            "statistics_sha256": bundle.statistics_sha256,
            "resource_report_sha256": bundle.resource_report_sha256,
            "claim_report_sha256": bundle.claim_report_sha256,
            "renderer_code_sha256": renderer_code_sha256,
            "publication_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        }
        for name, digest in actual.items():
            if getattr(authorization, name) != digest:
                raise ProductionAdapterError(
                    f"publication {name} differs from external authorization"
                )
        authenticate = getattr(self._backend, "authenticate_publication", None)
        if not callable(authenticate) or authenticate(
            intent,
            authorization,
            actual["publication_manifest_sha256"],
            self.adapter_sha256,
        ) is not True:
            raise ProductionAdapterError(
                "private backend did not authenticate publication render"
            )

    def consume_sealed_test(self, intent: CommandIntent) -> str:
        """Authenticate then atomically consume the sole persistent test grant."""

        if type(intent) is not CommandIntent:
            raise TypeError("intent must be exactly CommandIntent")
        if self._admission is None:
            raise ProductionAdapterError("private adapter must be admitted before test access")
        if (
            intent.command != "evaluate"
            or intent.split != "test"
            or intent.sealed_test_consumption_sha256 is not None
        ):
            raise ProductionAdapterError("sealed-test gate received a non-test intent")
        if self._sealed_test_consumption_sha256 is not None:
            raise ProductionAdapterError("sealed-test grant was already consumed")
        gate = self._sealed_test_gate
        authorization = self._sealed_test_authorization
        if gate is None or authorization is None:
            raise ProductionAdapterError("sealed-test grant and persistent ledger are absent")
        if gate.consumed:
            raise ProductionAdapterError("sealed-test persistent ledger is already consumed")
        self._validate_authorized_code()
        authenticate = getattr(self._backend, "authenticate_sealed_test", None)
        if not callable(authenticate) or authenticate(
            intent,
            authorization,
            self.adapter_sha256,
        ) is not True:
            raise ProductionAdapterError("private backend did not authenticate sealed-test grant")
        consumption_sha256 = gate.consume(
            external_grant_sha256=authorization.external_grant_sha256,
            test_manifest_sha256=authorization.test_manifest_sha256,
            caption_manifest_sha256=authorization.caption_manifest_sha256,
            evaluator_sha256=authorization.evaluator_sha256,
            validation_freeze_sha256=authorization.validation_freeze_sha256,
            aggregate_code_sha256=authorization.aggregate_code_sha256,
            evaluation_census_sha256=authorization.evaluation_census_sha256,
            hard_gallery_freeze_binding_sha256=(
                authorization.hard_gallery_freeze_binding_sha256
            ),
            hard_gallery_collection_sha256=(
                authorization.hard_gallery_collection_sha256
            ),
            consumed_at_utc=authorization.consumed_at_utc,
        )
        self._sealed_test_consumption_sha256 = consumption_sha256
        return consumption_sha256

    def handle(self, intent: CommandIntent) -> CommandResult:
        if type(intent) is not CommandIntent:
            raise TypeError("intent must be exactly CommandIntent")
        if self._admission is None:
            raise ProductionAdapterError("private adapter must be admitted before execution")
        admission_sha = runtime_admission_sha256(self._admission)
        if (
            intent.runtime_mode != "PRIVATE_AUTHORIZED"
            or intent.admission_sha256 != admission_sha
            or intent.authority != self._receipts.authority
            or intent.execution_authorized is not True
            or intent.production is not True
            or intent.result_claimed is not False
            or (
                intent.command == "evaluate"
                and intent.split == "test"
                and intent.sealed_test_consumption_sha256
                != self._sealed_test_consumption_sha256
            )
            or (
                (intent.command != "evaluate" or intent.split != "test")
                and intent.sealed_test_consumption_sha256 is not None
            )
        ):
            raise ProductionAdapterError("command intent is not bound to this admission")
        private_result = self._backend.execute(intent)
        if type(private_result) is not BackendExecution:
            raise ProductionAdapterError("backend returned an invalid execution object")
        if private_result.outcome == "COMPLETED" and intent.command == "evaluate":
            evidence = private_result.semantic_evidence
            if type(evidence) is not EvaluationExecutionEvidence:
                raise ProductionAdapterError(
                    "completed evaluation requires split-bound score-table evidence"
                )
            (
                evaluation_raw,
                _,
                score_execution_census_sha256,
                evaluation_census_sha256,
                hard_freeze_binding_sha256,
                hard_collection_sha256,
            ) = _recompute_evaluation_evidence(evidence)
            self._validate_evaluation_binding(
                evidence,
                split=intent.split,
                sealed_test_consumption_sha256=(
                    intent.sealed_test_consumption_sha256
                ),
                score_execution_census_sha256=score_execution_census_sha256,
                evaluation_census_sha256=evaluation_census_sha256,
                hard_gallery_freeze_binding_sha256=hard_freeze_binding_sha256,
                hard_gallery_collection_sha256=hard_collection_sha256,
            )
            _require_semantic_artifact(private_result, evaluation_raw, "evaluation")
            if intent.split == "test":
                gate = self._sealed_test_gate
                authorization = self._sealed_test_authorization
                if (
                    gate is None
                    or authorization is None
                    or intent.sealed_test_consumption_sha256 is None
                ):
                    raise ProductionAdapterError(
                        "completed test evaluation lost its sealed-test binding"
                    )
                binding_sha256 = gate.bind_evaluation(
                    consumption_sha256=intent.sealed_test_consumption_sha256,
                    evaluation_aggregate_sha256=hashlib.sha256(evaluation_raw).hexdigest(),
                    evaluator_sha256=authorization.evaluator_sha256,
                    aggregate_code_sha256=authorization.aggregate_code_sha256,
                    completed_at_utc=private_result.completed_at_utc,
                )
                self._sealed_test_evaluation_binding_sha256 = binding_sha256
        elif private_result.outcome == "COMPLETED" and intent.command == "qualify-base":
            evidence = private_result.semantic_evidence
            if type(evidence) is not BaseQualificationExecutionEvidence:
                raise ProductionAdapterError(
                    "completed base qualification requires nine-row semantic evidence"
                )
            qualification_raw, qualification = (
                _recompute_base_qualification_evidence(evidence)
            )
            self._validate_base_qualification_binding(
                intent,
                qualification,
                qualification_raw,
            )
            _require_semantic_artifact(
                private_result,
                qualification_raw,
                "base qualification",
            )
        elif private_result.outcome == "COMPLETED" and intent.command == "bootstrap":
            evidence = private_result.semantic_evidence
            bootstrap_raw = self._verify_sealed_bootstrap_evidence(evidence)
            _require_semantic_artifact(private_result, bootstrap_raw, "bootstrap")
        elif private_result.outcome == "COMPLETED" and intent.command == "render-paper":
            evidence = private_result.semantic_evidence
            if type(evidence) is not PublicationExecutionEvidence:
                raise ProductionAdapterError(
                    "completed publication render requires content-bound evidence"
                )
            self._verify_sealed_bootstrap_evidence(evidence.bootstrap_evidence)
            bundle, manifest = _recompute_publication_evidence(evidence)
            self._validate_publication_binding(intent, bundle, manifest)
            _require_semantic_artifact_set(
                private_result,
                tuple(row.content for row in bundle.artifacts) + (manifest,),
                "publication render",
            )
        elif private_result.semantic_evidence is not None:
            raise ProductionAdapterError(
                "semantic evidence is only valid for completed qualification, "
                "evaluation, bootstrap, or publication rendering"
            )
        digests: list[str] = []
        for artifact in private_result.artifacts:
            if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size <= 0:
                raise ProductionAdapterError("backend artifact must be a nonempty regular file")
            digests.append(_sha256_file(artifact))
        ordered_digests = tuple(sorted(digests))
        if len(set(ordered_digests)) != len(ordered_digests):
            raise ProductionAdapterError("backend artifacts must have unique content")
        result = CommandResult(
            command=intent.command,
            outcome=private_result.outcome,
            completed_at_utc=private_result.completed_at_utc,
            run_id=intent.run_id,
            seed=intent.seed,
            split=intent.split,
            artifact_sha256s=ordered_digests,
            admission_sha256=admission_sha,
            runtime_mode="PRIVATE_AUTHORIZED",
            authority=self._receipts.authority,
            production=True,
            sealed_test_consumption_sha256=(
                intent.sealed_test_consumption_sha256
            ),
            result_claimed=False,
        )
        canonical_command_result_bytes(result)
        return result


__all__ = [
    "BackendExecution",
    "BaseQualificationExecutionEvidence",
    "BootstrapExecutionEvidence",
    "EvaluationCorpusProvenance",
    "EvaluationExecutionEvidence",
    "PublicationExecutionEvidence",
    "PrivateCommandBackend",
    "PrivateBaseQualificationAuthorization",
    "PrivatePublicationAuthorization",
    "PrivateReceiptAssertions",
    "PrivateSealedTestAuthorization",
    "ProductionAdapterError",
    "ProductionRuntimeAdapter",
    "SeedScoreEvidence",
    "SystemScoreEvidence",
    "build_bootstrap_execution_artifact",
    "build_base_qualification_execution_artifact",
    "build_evaluation_execution_artifact",
    "build_publication_execution_artifacts",
    "current_base_qualification_artifact_bytes",
    "current_evaluator_artifact_bytes",
    "current_publication_renderer_artifact_bytes",
    "current_scoring_artifact_bytes",
    "current_statistics_artifact_bytes",
]


# Freeze the fully initialized production module, including names that would
# otherwise resolve through Python builtins.  The two guard containers are
# excluded from the identity tuple because this final initialization replaces
# their placeholders; their names remain part of the exact name census.
_PRODUCTION_GLOBAL_NAME_CENSUS = frozenset(globals())
_PRODUCTION_GLOBAL_BINDING_IDENTITIES = tuple(
    (name, value)
    for name, value in sorted(globals().items())
    if name
    not in {
        "_PRODUCTION_GLOBAL_NAME_CENSUS",
        "_PRODUCTION_GLOBAL_BINDING_IDENTITIES",
    }
)
