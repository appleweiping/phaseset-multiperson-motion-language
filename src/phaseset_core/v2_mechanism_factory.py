"""Bind implemented V2 residual mechanisms to exact registered run rows.

This constructs models and host configs; it cannot authorize an optimizer
step, GPU reservation, data read, pilot or formal run. Unsupported rows fail
closed instead of silently becoming the full model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch

from .continuous_parent_host import (
    ContinuousParentTrainingHost,
    ParentHostBindings,
    ParentHostConfig,
    ParentHostRunIdentity,
    ParentTrainingSource,
)
from .continuous_base_retrieval import ContinuousBaseRetrievalSystem
from .continuous_retrieval import ContinuousRetrievalSystem
from .dct_calibration import DctFloorReceipt
from .speed_calibration import SpeedFloorReceipt
from .directional_phase import LocalPhaseConfig
from .study_storage import V2_MATRIX_SHA256
from .parent_retrieval_task import ParentRetrievalTask
from .training import (
    OFFICIAL_SEEDS,
    BaseRetrievalSystem,
    CheckpointArtifact,
    _load_torch_checkpoint,
    _stable_hash,
    build_registered_base_training_system,
)


class V2MechanismHold(ValueError):
    """A row cannot yet be constructed as its registered scientific system."""


_SUPPORTED = frozenset({"PhaseSet-V2", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"})


def _sha256_key(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class V2SelectedB2Checkpoint:
    """Private admission's validation-selected complete-parent B2 artifact."""

    path: Path
    sha256: str
    terminal_path: Path
    terminal_sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.path, Path)
            or not _sha256_key(self.sha256)
            or not isinstance(self.terminal_path, Path)
            or not _sha256_key(self.terminal_sha256)
        ):
            raise V2MechanismHold("selected B2 checkpoint and terminal digests are required")


@dataclass(frozen=True)
class V2MechanismBuild:
    run_id: str
    system_id: str
    seed: int
    split: str
    max_epochs: int
    predecessor_run_id: str
    predecessor_checkpoint_sha256: str
    training_source_manifest_sha256: str
    training_components: tuple[str, ...] | None
    validation_components: tuple[str, ...] | None
    model: ContinuousRetrievalSystem
    counterfactual_objective_enabled: bool
    launch_authority: bool = False


def _registered_matrix(matrix_bytes: bytes) -> dict:
    if (
        type(matrix_bytes) is not bytes
        or hashlib.sha256(matrix_bytes).hexdigest() != V2_MATRIX_SHA256
    ):
        raise V2MechanismHold("exact frozen V2 matrix bytes required")
    matrix = json.loads(matrix_bytes)
    rows = matrix.get("runs") if type(matrix) is dict else None
    if (
        type(matrix) is not dict
        or matrix.get("schema") != "phaseset-v2-training-matrix-v1"
        or matrix.get("status") != "PLANNED_NOT_LAUNCHABLE"
        or type(rows) is not list
        or len(rows) != 87
        or any(type(row) is not dict or row.get("id") != f"V2-{i:03d}" for i, row in enumerate(rows, 1))
    ):
        raise V2MechanismHold("the complete registered 87-row matrix is required")
    return matrix


def _registered_row(matrix_bytes: bytes, run_id: str) -> tuple[dict, dict]:
    matrix = _registered_matrix(matrix_bytes)
    rows = matrix["runs"]
    if type(run_id) is not str or not run_id.startswith("V2-"):
        raise V2MechanismHold("an exact V2 run ID is required")
    candidates = [row for row in rows if row["id"] == run_id]
    if len(candidates) != 1:
        raise V2MechanismHold("V2 run ID is absent from the fixed matrix")
    row = candidates[0]
    if row.get("system") not in _SUPPORTED:
        raise V2MechanismHold(f"{row.get('system')} is not an implemented V2 mechanism")
    if row.get("seed") not in OFFICIAL_SEEDS or row.get("status") != "PLANNED":
        raise V2MechanismHold("V2 seed or row status differs from the plan")
    dependencies = row.get("depends_on")
    if type(dependencies) is not list or len(dependencies) != 1:
        raise V2MechanismHold("a residual must bind exactly one B2 predecessor")
    predecessor = next((item for item in rows if item["id"] == dependencies[0]), None)
    if (
        predecessor is None
        or predecessor.get("system") != "B2"
        or predecessor.get("seed") != row["seed"]
        or predecessor.get("split") != row.get("split")
    ):
        raise V2MechanismHold("residual B2 predecessor does not match seed and split")
    if row["system"] == "PhaseSet-V2":
        if row.get("group") not in {"main_residual", "group_held_out", "external_dyadic"}:
            raise V2MechanismHold("full V2 row belongs to an unexpected stage")
        if row.get("group") == "external_dyadic":
            raise V2MechanismHold("external dyadic population binding is not implemented")
    elif row.get("group") != "mechanism_ablation" or row.get("split") != "main":
        raise V2MechanismHold("mechanism control must belong to the main ablation group")
    return matrix, row


def make_v2_registered_base_host(
    matrix_bytes: bytes,
    run_id: str,
    task: ParentRetrievalTask,
    source: ParentTrainingSource,
    config: ParentHostConfig,
    bindings: ParentHostBindings,
) -> ContinuousParentTrainingHost:
    """Bind a main B0/B1/B2 qualification run to its exact matrix row."""
    matrix = _registered_matrix(matrix_bytes)
    row = next((item for item in matrix["runs"] if item["id"] == run_id), None)
    if (
        row is None
        or row.get("system") not in {"B0", "B1", "B2"}
        or row.get("group") != "internal_base"
        or row.get("split") != "main"
        or row.get("status") != "PLANNED"
        or row.get("seed") not in OFFICIAL_SEEDS
        or type(config) is not ParentHostConfig
        or type(bindings) is not ParentHostBindings
    ):
        raise V2MechanismHold("registered main B0/B1/B2 base row and model required")
    development = matrix["pilot_and_group_fold_components"]["development"]
    validation = tuple(matrix["primary_split"]["validation_component_labels"])
    train = tuple(component for component in development if component not in validation)
    if (
        row["seed"] != config.seed
        or config.stage != "base"
        or config.epochs != row["max_epochs"]
        or config.train_components != train
        or config.validation_components != validation
        or bindings.frozen_base_checkpoint_sha256 is not None
    ):
        raise V2MechanismHold("base host population, seed or stage differs from matrix")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(row["seed"])
        base_model = build_registered_base_training_system(row["system"])
    return ContinuousParentTrainingHost(
        ContinuousBaseRetrievalSystem(base_model),
        task,
        source,
        config,
        bindings,
        run_identity=ParentHostRunIdentity(run_id, row["system"], None),
    )


def make_v2_b2_base_host(
    matrix_bytes: bytes,
    run_id: str,
    task: ParentRetrievalTask,
    source: ParentTrainingSource,
    config: ParentHostConfig,
    bindings: ParentHostBindings,
) -> ContinuousParentTrainingHost:
    """Compatibility entry point that continues to admit B2 rows only."""
    matrix = _registered_matrix(matrix_bytes)
    row = next((item for item in matrix["runs"] if item["id"] == run_id), None)
    if row is None or row.get("system") != "B2":
        raise V2MechanismHold("registered main B2 base row required")
    return make_v2_registered_base_host(matrix_bytes, run_id, task, source, config, bindings)


def _load_selected_b2_predecessor(
    matrix: dict,
    row: dict,
    selected_b2: V2SelectedB2Checkpoint,
) -> tuple[BaseRetrievalSystem, CheckpointArtifact, dict]:
    """One terminal/checkpoint authority path shared by registered residuals."""
    if type(selected_b2) is not V2SelectedB2Checkpoint:
        raise V2MechanismHold("validation-selected B2 artifact is required")
    payload, artifact = _load_torch_checkpoint(
        selected_b2.path, expected_sha256=selected_b2.sha256
    )
    if (
        selected_b2.terminal_path.is_symlink()
        or not selected_b2.terminal_path.is_file()
        or selected_b2.terminal_path.parent.resolve() != artifact.path.parent
    ):
        raise V2MechanismHold("selected B2 terminal must be an owned regular file")
    terminal_bytes = selected_b2.terminal_path.read_bytes()
    if hashlib.sha256(terminal_bytes).hexdigest() != selected_b2.terminal_sha256:
        raise V2MechanismHold("selected B2 terminal digest mismatch")
    try:
        terminal = json.loads(terminal_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise V2MechanismHold("selected B2 terminal is unreadable") from error
    saved_state_digest = payload.get("state_sha256")
    without_digest = dict(payload)
    without_digest.pop("state_sha256", None)
    if saved_state_digest != _stable_hash(without_digest):
        raise V2MechanismHold("selected B2 checkpoint state digest is invalid")
    manifest = payload.get("manifest")
    expected_identity = {
        "run_id": row["depends_on"][0],
        "system_id": "B2",
        "predecessor_run_id": None,
    }
    if (
        type(manifest) is not dict
        or manifest.get("schema") != "phaseset-complete-parent-host-v1"
        or manifest.get("registered_run_identity") != expected_identity
        or type(manifest.get("config")) is not dict
        or manifest["config"].get("seed") != row["seed"]
        or manifest["config"].get("stage") != "base"
        or manifest["config"].get("epochs") != 30
    ):
        raise V2MechanismHold("selected B2 checkpoint does not bind the predecessor row")
    predecessor = next(item for item in matrix["runs"] if item["id"] == expected_identity["run_id"])
    if (
        type(terminal) is not dict
        or terminal.get("outcome") != "COMPLETED"
        or terminal.get("failure_class") is not None
        or terminal.get("registered_run_identity") != expected_identity
        or terminal.get("completed_epochs") != predecessor["max_epochs"]
        or type(manifest.get("total_steps")) is not int
        or terminal.get("global_step") != manifest["total_steps"]
        or terminal.get("best_checkpoint") != str(artifact.path)
        or terminal.get("best_checkpoint_sha256") != artifact.sha256
        or terminal.get("best_validation_r1") != payload.get("best_validation_r1")
    ):
        raise V2MechanismHold("B2 terminal does not attest the completed selected predecessor")
    best = payload.get("best")
    if (
        type(best) is not tuple
        or best != (str(artifact.path), None)
        or payload.get("best_validation_r1") is None
        or not payload.get("validation_history")
    ):
        raise V2MechanismHold("B2 artifact is not its validation-selected checkpoint")
    saved_model = payload.get("model")
    if not isinstance(saved_model, Mapping) or not saved_model or any(
        type(key) is not str or not key.startswith("base.") for key in saved_model
    ):
        raise V2MechanismHold("selected B2 model state is malformed")
    base_state = {key.removeprefix("base."): value for key, value in saved_model.items()}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(row["seed"])
        frozen_b2 = build_registered_base_training_system("B2")
    try:
        frozen_b2.load_state_dict(base_state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise V2MechanismHold("selected B2 weights do not match the B2 architecture") from error
    return frozen_b2, artifact, manifest


def build_v2_mechanism(
    matrix_bytes: bytes,
    run_id: str,
    *,
    selected_b2: V2SelectedB2Checkpoint,
    training_source_manifest_sha256: str,
    dct_floor_receipt: DctFloorReceipt | None = None,
    speed_floor_receipt: SpeedFloorReceipt | None = None,
    checkpoint_blocks: bool = True,
) -> V2MechanismBuild:
    """Construct the exact implemented mechanism, never a launch decision."""
    matrix, row = _registered_row(matrix_bytes, run_id)
    if not _sha256_key(training_source_manifest_sha256):
        raise V2MechanismHold("admitted training-source manifest digest is required")
    frozen_b2, artifact, manifest = _load_selected_b2_predecessor(
        matrix, row, selected_b2
    )
    system_id = row["system"]
    if system_id != "A6" and dct_floor_receipt is not None:
        raise V2MechanismHold("a DCT floor receipt may only enter A6")
    if system_id == "A6" and type(dct_floor_receipt) is not DctFloorReceipt:
        raise V2MechanismHold("A6 requires its typed training-only DCT floor receipt")
    if system_id != "A1" and speed_floor_receipt is not None:
        raise V2MechanismHold("a speed floor receipt may only enter A1")
    if system_id == "A1" and type(speed_floor_receipt) is not SpeedFloorReceipt:
        raise V2MechanismHold("A1 requires its typed training-only speed floor receipt")
    if type(checkpoint_blocks) is not bool:
        raise V2MechanismHold("checkpointing choice must be explicit boolean")
    training_components = validation_components = None
    if row["split"] == "main":
        development = matrix["pilot_and_group_fold_components"]["development"]
        validation = matrix["primary_split"]["validation_component_labels"]
        training_components = tuple(component for component in development if component not in validation)
        validation_components = tuple(validation)
        if len(training_components) != 12 or validation_components != ("C00",):
            raise V2MechanismHold("main training/validation components differ from the plan")
    elif row["split"].startswith("group_fold_"):
        folds = matrix["pilot_and_group_fold_components"]["folds"]
        fold = next((item for item in folds if item["run_split"] == row["split"]), None)
        if fold is None or matrix["pilot_and_group_fold_components"]["reuse_main_checkpoints"] is not False:
            raise V2MechanismHold("registered group fold population is unavailable")
        training_components = tuple(fold["train"])
        validation_components = tuple(matrix["pilot_and_group_fold_components"]["fold_validation"])
    if (
        tuple(manifest["config"].get("train_components", ())) != training_components
        or tuple(manifest["config"].get("validation_components", ())) != validation_components
    ):
        raise V2MechanismHold("selected B2 population differs from residual population")
    if system_id == "A6":
        dct_floor_receipt.require_config(LocalPhaseConfig())
        dct_floor_receipt.require_population("main", training_components)
        if dct_floor_receipt.training_source_manifest_sha256 != training_source_manifest_sha256:
            raise V2MechanismHold("A6 floor source differs from admitted training source")
    if system_id == "A1":
        speed_floor_receipt.require_config(LocalPhaseConfig())
        speed_floor_receipt.require_population("main", training_components)
        if speed_floor_receipt.training_source_manifest_sha256 != training_source_manifest_sha256:
            raise V2MechanismHold("A1 floor source differs from admitted training source")
    kwargs = {
        "order_free": system_id == "A2",
        "pair_bag": system_id == "A3",
        "incidence_shuffle_seed": row["seed"] if system_id == "A4" else None,
        "strip_phase": system_id == "A5",
        "generic_local": system_id == "A7",
        "relation_kind": "true_mean_difference_dct" if system_id == "A6" else "phase",
        "dct_floor_receipt": dct_floor_receipt,
        "periodic_velocity_mode": "speed_only" if system_id == "A1" else "signed_vector",
        "speed_floor_receipt": speed_floor_receipt,
        "checkpoint_blocks": checkpoint_blocks,
    }
    # The host seeds inside fit, which is too late for model initialization.
    # Forking CPU RNG makes same-seed ablations start from identical residual
    # parameters without changing the caller's ambient random stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(row["seed"])
        model = ContinuousRetrievalSystem(frozen_b2, **kwargs)
    return V2MechanismBuild(
        run_id=run_id,
        system_id=system_id,
        seed=row["seed"],
        split=row["split"],
        max_epochs=row["max_epochs"],
        predecessor_run_id=row["depends_on"][0],
        predecessor_checkpoint_sha256=artifact.sha256,
        training_source_manifest_sha256=training_source_manifest_sha256,
        training_components=training_components,
        validation_components=validation_components,
        model=model,
        counterfactual_objective_enabled=system_id != "A8",
    )


def bind_v2_parent_host_config(
    build: V2MechanismBuild,
    config: ParentHostConfig,
) -> ParentHostConfig:
    """A8 changes only CF loss weight, not model or weak training pool."""
    if type(build) is not V2MechanismBuild or type(config) is not ParentHostConfig:
        raise TypeError("V2 build and complete-parent host config required")
    if config.seed != build.seed or config.stage != "residual":
        raise V2MechanismHold("host seed/stage differs from the registered V2 row")
    if config.epochs != build.max_epochs:
        raise V2MechanismHold("registered residual schedule differs from matrix")
    if build.split != "main":
        raise V2MechanismHold("fold and Inter-X host population binding is not yet implemented")
    if (
        config.train_components != build.training_components
        or config.validation_components != build.validation_components
    ):
        raise V2MechanismHold("host training/validation components differ from the matrix")
    if build.system_id == "A8":
        if config.cf_weight <= 0:
            raise V2MechanismHold("A8 must be derived from the same nonzero CF candidate")
        return replace(config, cf_weight=0.0)
    if config.cf_weight <= 0:
        raise V2MechanismHold("full and other controls require the frozen CF objective")
    return config


def make_v2_parent_host(
    build: V2MechanismBuild,
    task: ParentRetrievalTask,
    source: ParentTrainingSource,
    config: ParentHostConfig,
    bindings: ParentHostBindings,
    *,
    admitted_training_source_manifest_sha256: str,
) -> ContinuousParentTrainingHost:
    """Single row-bound V2 host path; still not a formal-run admission."""
    if type(build) is not V2MechanismBuild or type(bindings) is not ParentHostBindings:
        raise TypeError("exact V2 build and execution bindings required")
    if (
        bindings.frozen_base_checkpoint_sha256 != build.predecessor_checkpoint_sha256
        or admitted_training_source_manifest_sha256 != build.training_source_manifest_sha256
    ):
        raise V2MechanismHold("host checkpoint or admitted training source differs from build")
    bound_config = bind_v2_parent_host_config(build, config)
    if build.system_id == "A1" and (
        getattr(source, "velocity_mode", None) != "speed_only"
        or getattr(getattr(source, "speed_floor_receipt", None), "sha256", None)
        != build.model._speed_receipt_sha256
    ):
        raise V2MechanismHold("A1 host requires the registered speed-only source and floor receipt")
    identity = ParentHostRunIdentity(
        build.run_id, build.system_id, build.predecessor_run_id
    )
    return ContinuousParentTrainingHost(
        build.model, task, source, bound_config, bindings, run_identity=identity
    )
