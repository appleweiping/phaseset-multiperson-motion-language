"""Construct registered old PhaseSet and A9 controls on the selected B2.

This binds scientific row identity but never grants GPU, pilot, data-rights or
formal-training authority. The private operator still verifies the complete
input manifest, resource budget and frozen runtime before any optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .continuous_parent_host import (
    ContinuousParentTrainingHost,
    ParentHostBindings,
    ParentHostConfig,
    ParentHostRunIdentity,
    ParentTrainingSource,
)
from .controls import PhaseSetSystem
from .legacy_continuous_retrieval import LegacyWholeCaptureRetrievalSystem
from .legacy_scalar_calibration import LegacyScalarFloorReceipt
from .parent_retrieval_task import ParentRetrievalTask
from .training import OFFICIAL_SEEDS
from .v2_mechanism_factory import (
    V2MechanismHold,
    V2SelectedB2Checkpoint,
    _load_selected_b2_predecessor,
    _registered_matrix,
    _sha256_key,
)


@dataclass(frozen=True)
class V2LegacyBuild:
    run_id: str
    system_id: str
    seed: int
    max_epochs: int
    predecessor_run_id: str
    predecessor_checkpoint_sha256: str
    training_source_manifest_sha256: str
    floor_receipt_sha256: str
    training_components: tuple[str, ...]
    validation_components: tuple[str, ...]
    model: LegacyWholeCaptureRetrievalSystem
    launch_authority: bool = False


def build_v2_legacy_control(
    matrix_bytes: bytes,
    run_id: str,
    *,
    selected_b2: V2SelectedB2Checkpoint,
    floor_receipt: LegacyScalarFloorReceipt,
    training_source_manifest_sha256: str,
    checkpoint_windows: bool = True,
) -> V2LegacyBuild:
    """Load one exact B2 predecessor and old/A9 row with a typed floor."""
    matrix = _registered_matrix(matrix_bytes)
    row = next((item for item in matrix["runs"] if item["id"] == run_id), None)
    if (
        type(run_id) is not str
        or row is None
        or row.get("system") not in ("PhaseSet-v0.2-on-B2", "A9")
        or row.get("group")
        != ("main_residual" if row["system"] == "PhaseSet-v0.2-on-B2" else "mechanism_ablation")
        or row.get("split") != "main"
        or row.get("seed") not in OFFICIAL_SEEDS
        or row.get("max_epochs") != 20
        or row.get("status") != "PLANNED"
        or type(row.get("depends_on")) is not list
        or len(row["depends_on"]) != 1
    ):
        raise V2MechanismHold("registered main old/A9 row is required")
    predecessor = next(
        (item for item in matrix["runs"] if item["id"] == row["depends_on"][0]), None
    )
    if (
        predecessor is None
        or predecessor.get("system") != "B2"
        or predecessor.get("seed") != row["seed"]
        or predecessor.get("split") != "main"
    ):
        raise V2MechanismHold("old/A9 row needs its same-seed B2 predecessor")
    if type(floor_receipt) is not LegacyScalarFloorReceipt:
        raise V2MechanismHold("old/A9 needs the typed training-only scalar floor")
    if not _sha256_key(training_source_manifest_sha256):
        raise V2MechanismHold("admitted training-source manifest digest is required")
    development = tuple(matrix["pilot_and_group_fold_components"]["development"])
    validation = tuple(matrix["primary_split"]["validation_component_labels"])
    train = tuple(component for component in development if component not in validation)
    if len(train) != 12 or validation != ("C00",):
        raise V2MechanismHold("old/A9 main development components differ from the matrix")
    floor_receipt.require_population(train)
    if floor_receipt.training_source_manifest_sha256 != training_source_manifest_sha256:
        raise V2MechanismHold("old/A9 floor source differs from admitted training source")
    frozen_b2, artifact, manifest = _load_selected_b2_predecessor(
        matrix, row, selected_b2
    )
    if (
        tuple(manifest["config"].get("train_components", ())) != train
        or tuple(manifest["config"].get("validation_components", ())) != validation
    ):
        raise V2MechanismHold("selected B2 population differs from old/A9 population")
    mode = "old" if row["system"] == "PhaseSet-v0.2-on-B2" else "A9"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(row["seed"])
        encoder = PhaseSetSystem("08", embedding_dim=512, energy_floors=floor_receipt.floors)
        model = LegacyWholeCaptureRetrievalSystem(
            frozen_b2,
            encoder,
            score_mode=mode,
            floor_receipt=floor_receipt,
            checkpoint_windows=checkpoint_windows,
        )
    return V2LegacyBuild(
        run_id=run_id,
        system_id=row["system"],
        seed=row["seed"],
        max_epochs=row["max_epochs"],
        predecessor_run_id=row["depends_on"][0],
        predecessor_checkpoint_sha256=artifact.sha256,
        training_source_manifest_sha256=training_source_manifest_sha256,
        floor_receipt_sha256=floor_receipt.sha256,
        training_components=train,
        validation_components=validation,
        model=model,
    )


def make_v2_legacy_parent_host(
    build: V2LegacyBuild,
    task: ParentRetrievalTask,
    source: ParentTrainingSource,
    config: ParentHostConfig,
    bindings: ParentHostBindings,
    *,
    admitted_training_source_manifest_sha256: str,
) -> ContinuousParentTrainingHost:
    """Connect the registered build to the complete-parent training host."""
    if (
        type(build) is not V2LegacyBuild
        or type(config) is not ParentHostConfig
        or type(bindings) is not ParentHostBindings
    ):
        raise TypeError("exact old/A9 build, host config and bindings are required")
    if (
        config.seed != build.seed
        or config.stage != "residual"
        or config.epochs != build.max_epochs
        or config.train_components != build.training_components
        or config.validation_components != build.validation_components
        or admitted_training_source_manifest_sha256 != build.training_source_manifest_sha256
        or bindings.frozen_base_checkpoint_sha256 != build.predecessor_checkpoint_sha256
        or bindings.legacy_floor_receipt_sha256 != build.floor_receipt_sha256
        or bindings.legacy_training_source_manifest_sha256
        != build.training_source_manifest_sha256
    ):
        raise V2MechanismHold("old/A9 host schedule, predecessor, floor or source differs from build")
    return ContinuousParentTrainingHost(
        build.model,
        task,
        source,
        config,
        bindings,
        run_identity=ParentHostRunIdentity(
            build.run_id, build.system_id, build.predecessor_run_id
        ),
    )
