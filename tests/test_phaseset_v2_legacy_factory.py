"""Frozen-matrix old/A9 construction; synthetic receipts confer no authority."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phaseset_core.continuous_parent_host import ParentHostBindings, ParentHostConfig
from phaseset_core.legacy_scalar_calibration import (
    LEGACY_SCALAR_FLOOR_SCHEMA,
    LegacyScalarFloorReceipt,
    legacy_scalar_config_sha256,
)
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.training import _stable_hash
from phaseset_core.v2_legacy_factory import (
    V2LegacyBuild,
    build_v2_legacy_control,
    make_v2_legacy_parent_host,
)
from phaseset_core.v2_mechanism_factory import V2MechanismHold
from test_phaseset_v2_mechanism_factory import MATRIX, MAIN_TRAIN, _selected

SOURCE_SHA = "a" * 64


def _receipt(*, components=MAIN_TRAIN, source=SOURCE_SHA):
    # The fixture tests row binding only; it is not a real calibrated receipt.
    return LegacyScalarFloorReceipt(
        LEGACY_SCALAR_FLOOR_SCHEMA,
        "main",
        components,
        source,
        legacy_scalar_config_sha256(),
        np.full((6,), 1e-10, dtype=np.float64),
        202,
        4820,
        19280,
        (19280,) * 6,
        (0,) * 6,
        (0,) * 6,
    )


@pytest.mark.parametrize(
    ("run_id", "seed", "predecessor", "mode"),
    (
        ("V2-019", 1729, "V2-007", "old"),
        ("V2-020", 2718, "V2-008", "old"),
        ("V2-021", 31415, "V2-009", "old"),
        ("V2-049", 1729, "V2-007", "A9"),
        ("V2-050", 2718, "V2-008", "A9"),
        ("V2-051", 31415, "V2-009", "A9"),
    ),
)
def test_all_six_registered_rows_construct_only_their_selected_b2(
    tmp_path, run_id, seed, predecessor, mode,
):
    selected = _selected(tmp_path, run_id=predecessor, seed=seed)
    build = build_v2_legacy_control(
        MATRIX, run_id, selected_b2=selected,
        floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
    )
    assert type(build) is V2LegacyBuild
    assert (build.run_id, build.seed, build.predecessor_run_id) == (
        run_id, seed, predecessor,
    )
    assert build.model.score_mode == mode
    assert build.model.legacy_encoder.system_id == "08"
    assert build.training_components == MAIN_TRAIN
    assert build.validation_components == ("C00",)
    assert build.predecessor_checkpoint_sha256 == selected.sha256
    assert build.floor_receipt_sha256 == build.model.floor_receipt.sha256
    assert build.launch_authority is False
    assert all(not p.requires_grad for p in build.model.frozen_b2.parameters())
    assert build.model.head is not None


def test_old_and_a9_have_identical_same_seed_encoder_initialization(tmp_path):
    selected = _selected(tmp_path)
    old = build_v2_legacy_control(
        MATRIX, "V2-019", selected_b2=selected,
        floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
    )
    a9 = build_v2_legacy_control(
        MATRIX, "V2-049", selected_b2=selected,
        floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
    )
    for key, value in old.model.legacy_encoder.state_dict().items():
        assert torch.equal(value, a9.model.legacy_encoder.state_dict()[key])
    assert _stable_hash(old.model.frozen_b2.state_dict()) == _stable_hash(
        a9.model.frozen_b2.state_dict()
    )
    assert type(old.model.head) is not type(a9.model.head)


def test_nonlegacy_rows_or_wrong_b2_and_floor_fail_closed(tmp_path):
    selected = _selected(tmp_path)
    with pytest.raises(V2MechanismHold, match="registered main old/A9"):
        build_v2_legacy_control(
            MATRIX, "V2-022", selected_b2=selected,
            floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
        )
    with pytest.raises(V2MechanismHold, match="exact frozen"):
        build_v2_legacy_control(
            MATRIX + b"\n", "V2-019", selected_b2=selected,
            floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
        )
    with pytest.raises(V2MechanismHold, match="typed"):
        build_v2_legacy_control(
            MATRIX, "V2-019", selected_b2=selected,
            floor_receipt=None, training_source_manifest_sha256=SOURCE_SHA,
        )
    with pytest.raises(V2MechanismHold, match="floor source"):
        build_v2_legacy_control(
            MATRIX, "V2-019", selected_b2=selected,
            floor_receipt=_receipt(), training_source_manifest_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="population"):
        build_v2_legacy_control(
            MATRIX, "V2-019", selected_b2=selected,
            floor_receipt=_receipt(components=("C01",)),
            training_source_manifest_sha256=SOURCE_SHA,
        )
    with pytest.raises(V2MechanismHold, match="predecessor row"):
        build_v2_legacy_control(
            MATRIX, "V2-020", selected_b2=selected,
            floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
        )
    with pytest.raises(Exception, match="digest mismatch"):
        build_v2_legacy_control(
            MATRIX, "V2-019", selected_b2=replace(selected, sha256="0" * 64),
            floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
        )


def test_host_factory_rejects_unbound_config_and_predecessor_before_data_read(tmp_path):
    selected = _selected(tmp_path)
    build = build_v2_legacy_control(
        MATRIX, "V2-019", selected_b2=selected,
        floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
    )
    config = ParentHostConfig(1729, MAIN_TRAIN, ("C00",))
    bindings = ParentHostBindings(
        "b" * 64, "c" * 64, "d" * 64, selected.sha256,
        build.floor_receipt_sha256, SOURCE_SHA,
        _stable_hash(build.model.frozen_b2.state_dict()),
    )
    with pytest.raises(V2MechanismHold, match="host schedule"):
        make_v2_legacy_parent_host(
            build, None, None, replace(config, epochs=19), bindings,
            admitted_training_source_manifest_sha256=SOURCE_SHA,
        )
    with pytest.raises(V2MechanismHold, match="host schedule"):
        make_v2_legacy_parent_host(
            build, None, None, config,
            replace(bindings, frozen_base_checkpoint_sha256="e" * 64),
            admitted_training_source_manifest_sha256=SOURCE_SHA,
        )
    with pytest.raises(V2MechanismHold, match="host schedule"):
        make_v2_legacy_parent_host(
            build, None, None, config, bindings,
            admitted_training_source_manifest_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    ("run_id", "seed", "predecessor", "system_id"),
    (
        ("V2-019", 1729, "V2-007", "PhaseSet-v0.2-on-B2"),
        ("V2-020", 2718, "V2-008", "PhaseSet-v0.2-on-B2"),
        ("V2-021", 31415, "V2-009", "PhaseSet-v0.2-on-B2"),
        ("V2-049", 1729, "V2-007", "A9"),
        ("V2-050", 2718, "V2-008", "A9"),
        ("V2-051", 31415, "V2-009", "A9"),
    ),
)
def test_registered_host_persists_each_row_and_floor_without_loading_data(
    tmp_path, run_id, seed, predecessor, system_id,
):
    selected = _selected(tmp_path, run_id=predecessor, seed=seed)
    build = build_v2_legacy_control(
        MATRIX, run_id, selected_b2=selected,
        floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
    )
    rows = tuple(
        ParentCaptionRecord(
            hashlib.sha256(f"old-source-{i}".encode()).hexdigest(),
            hashlib.sha256(f"old-family-{i}".encode()).hexdigest(),
            component,
            "validation" if component == "C00" else "train",
            (f"synthetic human row {i}",),
        )
        for i, component in enumerate((*MAIN_TRAIN, "C00"))
    )
    task = ParentRetrievalTask(
        rows, expected_family_keys=tuple(row.annotation_family_sha256 for row in rows)
    )
    source = SimpleNamespace(
        legacy_floor_receipt=build.model.floor_receipt,
        legacy_training_source_manifest_sha256=SOURCE_SHA,
    )
    config = ParentHostConfig(seed, MAIN_TRAIN, ("C00",))
    bindings = ParentHostBindings(
        "b" * 64, "c" * 64, "d" * 64, selected.sha256,
        build.floor_receipt_sha256, SOURCE_SHA,
        _stable_hash(build.model.frozen_b2.state_dict()),
    )
    host = make_v2_legacy_parent_host(
        build, task, source, config, bindings,
        admitted_training_source_manifest_sha256=SOURCE_SHA,
    )
    assert host._manifest["registered_run_identity"] == {
        "run_id": run_id,
        "system_id": system_id,
        "predecessor_run_id": predecessor,
    }
    assert host.bindings.legacy_floor_receipt_sha256 == build.floor_receipt_sha256
    assert host._steps_per_epoch == 1


def test_shared_b2_helper_rejects_terminal_digest_and_semantic_drift(tmp_path):
    selected = _selected(tmp_path)
    original = json.loads(selected.terminal_path.read_bytes())
    selected.terminal_path.write_bytes(b"corrupt terminal")
    with pytest.raises(V2MechanismHold, match="terminal digest mismatch"):
        build_v2_legacy_control(
            MATRIX, "V2-019", selected_b2=selected,
            floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
        )
    mutations = (
        ("outcome", "FAILED"),
        ("registered_run_identity", {
            "run_id": "V2-008", "system_id": "B2", "predecessor_run_id": None,
        }),
        ("completed_epochs", 29),
        ("best_checkpoint", "not-the-selected-checkpoint.pt"),
    )
    for field, value in mutations:
        terminal = {**original, field: value}
        raw = (json.dumps(terminal, sort_keys=True) + "\n").encode()
        selected.terminal_path.write_bytes(raw)
        changed = replace(
            selected, terminal_sha256=hashlib.sha256(raw).hexdigest()
        )
        with pytest.raises(V2MechanismHold, match="B2 terminal does not attest"):
            build_v2_legacy_control(
                MATRIX, "V2-019", selected_b2=changed,
                floor_receipt=_receipt(), training_source_manifest_sha256=SOURCE_SHA,
            )
