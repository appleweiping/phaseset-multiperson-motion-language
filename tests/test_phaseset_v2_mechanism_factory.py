"""Exact V2 matrix rows construct the intended residual, never launch it."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phaseset_core.continuous_parent_host import ParentHostBindings, ParentHostConfig
from phaseset_core.dct_calibration import DCT_FLOOR_SCHEMA, DctFloorReceipt, dct_config_sha256
from phaseset_core.directional_phase import LocalPhaseConfig
from phaseset_core.speed_calibration import SPEED_FLOOR_SCHEMA, SpeedFloorReceipt, speed_config_sha256
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.training import (
    _atomic_torch_save,
    _stable_hash,
    build_registered_base_training_system,
)
from phaseset_core.v2_mechanism_factory import (
    V2MechanismHold,
    V2SelectedB2Checkpoint,
    bind_v2_parent_host_config,
    build_v2_mechanism,
    make_v2_b2_base_host,
    make_v2_parent_host,
)
MATRIX = (Path(__file__).parents[1] / "configs/phaseset_v2_experiment_matrix.json").read_bytes()
MAIN_TRAIN = ("C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C10", "C12", "C13", "C14")
FOLD_0_TRAIN = ("C01", "C02", "C04", "C05", "C07", "C08", "C12", "C13")
SOURCE_SHA = "a" * 64


def _selected(
    tmp_path, *, run_id="V2-007", seed=1729,
    train_components=MAIN_TRAIN, validation_components=("C00",),
):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        base = build_registered_base_training_system("B2")
    population_key = hashlib.sha256("|".join(train_components).encode()).hexdigest()[:8]
    path = tmp_path / f"{run_id}-{population_key}-best-validation.pt"
    payload = {
        "manifest": {
            "schema": "phaseset-complete-parent-host-v1",
            "config": {
                "stage": "base", "seed": seed, "epochs": 30,
                "train_components": train_components,
                "validation_components": validation_components,
            },
            "registered_run_identity": {
                "run_id": run_id, "system_id": "B2", "predecessor_run_id": None,
            },
        },
        "global_step": 1,
        "best": (str(path.resolve()), None),
        "best_validation_r1": 0.5,
        "validation_history": [(1, 0.5)],
        "model": {f"base.{key}": value for key, value in base.state_dict().items()},
    }
    payload["manifest"]["total_steps"] = 30
    payload["state_sha256"] = _stable_hash(payload)
    artifact = _atomic_torch_save(path, payload)
    terminal = {
        "outcome": "COMPLETED", "failure_class": None,
        "registered_run_identity": payload["manifest"]["registered_run_identity"],
        "completed_epochs": 30, "global_step": 30,
        "best_checkpoint": str(artifact.path),
        "best_checkpoint_sha256": artifact.sha256,
        "best_validation_r1": 0.5,
    }
    terminal_path = tmp_path / f"{run_id}-{population_key}-terminal.json"
    terminal_bytes = (json.dumps(terminal, sort_keys=True) + "\n").encode()
    terminal_path.write_bytes(terminal_bytes)
    return V2SelectedB2Checkpoint(
        artifact.path, artifact.sha256, terminal_path,
        hashlib.sha256(terminal_bytes).hexdigest(),
    )


def _build(tmp_path, run_id, *, selected=None, source_sha=SOURCE_SHA, **kwargs):
    if selected is None:
        selected = _selected(tmp_path)
    return build_v2_mechanism(
        MATRIX, run_id, selected_b2=selected,
        training_source_manifest_sha256=source_sha, **kwargs
    )


def _dct_receipt(*, population: str = "main") -> DctFloorReceipt:
    return DctFloorReceipt(
        DCT_FLOOR_SCHEMA,
        population,
        MAIN_TRAIN,
        "a" * 64,
        dct_config_sha256(LocalPhaseConfig()),
        np.ones(6, dtype=np.float64),
    )


def _speed_receipt(*, population: str = "main") -> SpeedFloorReceipt:
    return SpeedFloorReceipt(
        SPEED_FLOOR_SCHEMA,
        population,
        MAIN_TRAIN,
        SOURCE_SHA,
        speed_config_sha256(LocalPhaseConfig()),
        np.ones(6, dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("run_id", "system_id", "attribute", "expected"),
    (
        ("V2-022", "PhaseSet-V2", "pair_bag", False),
        ("V2-028", "A2", "order_free", True),
        ("V2-031", "A3", "pair_bag", True),
        ("V2-034", "A4", "incidence_shuffle_seed", 1729),
        ("V2-037", "A5", "strip_phase", True),
        ("V2-043", "A7", "generic_local", True),
        ("V2-046", "A8", "pair_bag", False),
    ),
)
def test_registered_main_mechanisms_bind_exact_run_and_live_model(
    tmp_path, run_id: str, system_id: str, attribute: str, expected: object
) -> None:
    build = _build(tmp_path, run_id)
    assert (build.run_id, build.system_id, build.seed, build.split) == (
        run_id, system_id, 1729, "main"
    )
    assert build.predecessor_run_id == "V2-007"
    assert len(build.predecessor_checkpoint_sha256) == 64
    assert build.training_components == MAIN_TRAIN
    assert build.validation_components == ("C00",)
    assert getattr(build.model.coordination, attribute) == expected
    assert build.counterfactual_objective_enabled is (system_id != "A8")
    assert build.launch_authority is False


def test_a6_requires_exact_population_and_uses_only_dct_relations(tmp_path) -> None:
    selected = _selected(tmp_path)
    with pytest.raises(V2MechanismHold, match="typed"):
        _build(tmp_path, "V2-040", selected=selected)
    with pytest.raises(ValueError, match="population"):
        _build(tmp_path, "V2-040", selected=selected, dct_floor_receipt=_dct_receipt(population="pilot"))
    with pytest.raises(V2MechanismHold, match="floor source"):
        _build(tmp_path, "V2-040", selected=selected, source_sha="b" * 64, dct_floor_receipt=_dct_receipt())
    build = _build(tmp_path, "V2-040", selected=selected, dct_floor_receipt=_dct_receipt())
    assert build.model.relation_kind == "true_mean_difference_dct"
    assert build.model.coordination.strip_phase is False
    assert build.model.coordination.pair_bag is False
    with pytest.raises(V2MechanismHold, match="only enter A6"):
        _build(tmp_path, "V2-031", selected=selected, dct_floor_receipt=_dct_receipt())


def test_a1_binds_independent_speed_floor_and_preserves_residual_capacity(tmp_path) -> None:
    selected = _selected(tmp_path)
    with pytest.raises(V2MechanismHold, match="typed"):
        _build(tmp_path, "V2-025", selected=selected)
    with pytest.raises(ValueError, match="population"):
        _build(tmp_path, "V2-025", selected=selected, speed_floor_receipt=_speed_receipt(population="pilot"))
    with pytest.raises(V2MechanismHold, match="floor source"):
        _build(tmp_path, "V2-025", selected=selected, source_sha="b" * 64, speed_floor_receipt=_speed_receipt())
    full = _build(tmp_path, "V2-022", selected=selected)
    a1 = _build(tmp_path, "V2-025", selected=selected, speed_floor_receipt=_speed_receipt())
    assert a1.system_id == "A1" and a1.model.periodic_velocity_mode == "speed_only"
    assert a1.model.relation_kind == "phase" and a1.counterfactual_objective_enabled
    assert sum(p.numel() for p in full.model.parameters() if p.requires_grad) == sum(
        p.numel() for p in a1.model.parameters() if p.requires_grad
    )
    with pytest.raises(ValueError, match="checkpoint resume"):
        a1.model.set_extra_state(full.model.get_extra_state())
    with pytest.raises(V2MechanismHold, match="only enter A1"):
        _build(tmp_path, "V2-022", selected=selected, speed_floor_receipt=_speed_receipt())


def test_a7_preserves_cf_host_and_checkpoint_identity(tmp_path) -> None:
    selected = _selected(tmp_path)
    full = _build(tmp_path, "V2-022", selected=selected)
    a7 = _build(tmp_path, "V2-043", selected=selected)
    assert a7.counterfactual_objective_enabled
    assert a7.model.coordination.generic_local
    assert a7.model.relation_kind == "phase"
    assert a7.model.periodic_velocity_mode == "signed_vector"
    config = ParentHostConfig(1729, MAIN_TRAIN, ("C00",), cf_weight=0.2, cf_margin=0.2)
    assert bind_v2_parent_host_config(a7, config) is config
    with pytest.raises(ValueError, match="checkpoint resume"):
        a7.model.set_extra_state(full.model.get_extra_state())
    with pytest.raises(ValueError, match="checkpoint resume"):
        full.model.set_extra_state(a7.model.get_extra_state())


def test_a8_disables_only_cf_objective_and_preserves_weak_pool_host_contract(tmp_path) -> None:
    selected = _selected(tmp_path)
    full = _build(tmp_path, "V2-022", selected=selected)
    a8 = _build(tmp_path, "V2-046", selected=selected)
    config = ParentHostConfig(1729, MAIN_TRAIN, ("C00",), cf_weight=0.2, cf_margin=0.2)
    assert bind_v2_parent_host_config(full, config) is config
    bound = bind_v2_parent_host_config(a8, config)
    assert bound == replace(config, cf_weight=0.0)
    assert bound.cf_margin == config.cf_margin
    assert a8.model.coordination.pair_bag is False
    full_trainable = {name: value for name, value in full.model.state_dict().items() if not name.startswith("frozen_b2.")}
    a8_trainable = {name: value for name, value in a8.model.state_dict().items() if not name.startswith("frozen_b2.")}
    assert full_trainable.keys() == a8_trainable.keys()
    assert all(torch.equal(value, a8_trainable[name]) for name, value in full_trainable.items())
    with pytest.raises(V2MechanismHold, match="nonzero CF candidate"):
        bind_v2_parent_host_config(a8, replace(config, cf_weight=0.0))
    with pytest.raises(V2MechanismHold, match="components"):
        bind_v2_parent_host_config(a8, replace(config, train_components=("C01", "C02")))


def test_unsupported_rows_and_matrix_drift_fail_without_full_model_fallback(tmp_path) -> None:
    selected = _selected(tmp_path)
    for run_id in ("V2-049", "V2-007"):
        with pytest.raises(V2MechanismHold, match="not an implemented V2 mechanism"):
            _build(tmp_path, run_id, selected=selected)
    with pytest.raises(V2MechanismHold, match="exact frozen"):
        build_v2_mechanism(MATRIX + b"\n", "V2-031", selected_b2=selected, training_source_manifest_sha256=SOURCE_SHA)
    with pytest.raises(V2MechanismHold, match="absent"):
        _build(tmp_path, "V2-999", selected=selected)


def test_fold_model_identity_is_available_but_fold_host_population_is_held(tmp_path) -> None:
    selected = _selected(
        tmp_path, run_id="V2-052",
        train_components=FOLD_0_TRAIN, validation_components=("C03",),
    )
    build = _build(tmp_path, "V2-053", selected=selected)
    assert build.system_id == "PhaseSet-V2"
    assert build.split == "group_fold_1"
    assert build.predecessor_run_id == "V2-052"
    assert build.training_components == FOLD_0_TRAIN
    assert build.validation_components == ("C03",)
    config = ParentHostConfig(1729, MAIN_TRAIN, ("C00",))
    with pytest.raises(V2MechanismHold, match="not yet implemented"):
        bind_v2_parent_host_config(build, config)
    selected_main_population = _selected(tmp_path, run_id="V2-052")
    with pytest.raises(V2MechanismHold, match="population"):
        _build(tmp_path, "V2-053", selected=selected_main_population)


def test_predecessor_row_and_artifact_drift_fail_closed(tmp_path) -> None:
    selected = _selected(tmp_path)
    with pytest.raises(V2MechanismHold, match="predecessor row"):
        _build(tmp_path, "V2-023", selected=selected)
    wrong_digest = replace(selected, sha256="0" * 64)
    with pytest.raises(Exception, match="digest mismatch"):
        _build(tmp_path, "V2-022", selected=wrong_digest)
    other = _selected(tmp_path, run_id="V2-008", seed=2718)
    a = _build(tmp_path, "V2-022", selected=selected)
    b = _build(tmp_path, "V2-023", selected=other)
    a_state = dict(a.model.named_parameters())
    b_state = dict(b.model.named_parameters())
    assert any(not torch.equal(a_state[key], b_state[key]) for key in a_state if not key.startswith("frozen_b2."))


def test_single_host_entrypoint_binds_a8_and_persists_row_identity(tmp_path) -> None:
    build = _build(tmp_path, "V2-046")
    rows = tuple(
        ParentCaptionRecord(
            hashlib.sha256(f"source-{i}".encode()).hexdigest(),
            hashlib.sha256(f"family-{i}".encode()).hexdigest(),
            component,
            "validation" if component == "C00" else "train",
            (f"human row {i}",),
        )
        for i, component in enumerate((*MAIN_TRAIN, "C00"))
    )
    task = ParentRetrievalTask(rows, expected_family_keys=tuple(row.annotation_family_sha256 for row in rows))
    config = ParentHostConfig(1729, MAIN_TRAIN, ("C00",), cf_weight=0.2)
    bindings = ParentHostBindings("b" * 64, "c" * 64, "d" * 64, build.predecessor_checkpoint_sha256)
    host = make_v2_parent_host(
        build, task, object(), config, bindings,
        admitted_training_source_manifest_sha256=SOURCE_SHA,
    )
    assert host.config.cf_weight == 0.0
    assert host._manifest["registered_run_identity"] == {
        "run_id": "V2-046", "system_id": "A8", "predecessor_run_id": "V2-007",
    }
    with pytest.raises(V2MechanismHold, match="checkpoint or admitted training source"):
        make_v2_parent_host(
            build, task, object(), config, replace(bindings, frozen_base_checkpoint_sha256="e" * 64),
            admitted_training_source_manifest_sha256=SOURCE_SHA,
        )
    base_host = make_v2_b2_base_host(
        MATRIX, "V2-007", task, object(),
        ParentHostConfig.for_base(1729, MAIN_TRAIN, ("C00",)),
        ParentHostBindings("b" * 64, "c" * 64, "d" * 64, None),
    )
    assert base_host._manifest["registered_run_identity"] == {
        "run_id": "V2-007", "system_id": "B2", "predecessor_run_id": None,
    }
