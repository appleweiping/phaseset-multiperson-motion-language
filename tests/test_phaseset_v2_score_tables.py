"""Synthetic-only score-table geometry and pairing tests; no sealed scores."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from phaseset_core.evaluation import RetrievalDataset, retrieval_census_sha256
from phaseset_core.v2_score_tables import (
    STATUS,
    V2ScoreTable,
    assemble_v2_score_tables,
)
from phaseset_core.v2_statistics import SEEDS, V2StatisticsError, v2_algorithm_report


def _commitment(prefix: str, index: int) -> bytes:
    return hashlib.sha256(f"synthetic-{prefix}-{index}".encode()).digest()


@pytest.fixture(scope="module")
def score_fixture():
    motions = tuple(_commitment("motion", index) for index in range(40))
    captions = tuple(_commitment("caption", index) for index in range(200))
    positives = tuple((index // 5,) for index in range(200))
    sizes = np.asarray((3,) * 16 + (4,) * 24, dtype=np.int64)
    components = ("C09",) * 16 + ("C11",) * 8 + ("C15",) * 16
    systems = ("PhaseSet-V2", "TMR-Set", *(f"A{index}" for index in range(1, 10)))
    rows = []
    for system in systems:
        for seed in SEEDS:
            scores = np.full((40, 200), -1.0, dtype=np.float64)
            for parent in range(40):
                scores[parent, parent * 5:(parent + 1) * 5] = 1.0
            if system == "A1" and seed == SEEDS[0]:
                scores[1, 0] = 2.0
            rows.append(V2ScoreTable(
                system,
                seed,
                RetrievalDataset(scores, motions, captions, positives, sizes, components),
            ))
    return tuple(rows), retrieval_census_sha256(rows[0].dataset)


def test_assembles_33_same_census_tables_and_keeps_algorithm_unbound(score_fixture):
    tables, census = score_fixture
    assembly = assemble_v2_score_tables(
        tuple(reversed(tables)), selected_b_star="TMR-Set", expected_census_sha256=census
    )
    assert assembly.status == STATUS
    assert not assembly.scientific_or_test_authority
    assert assembly.parent_count == 40 and assembly.caption_count == 200
    assert len(assembly.score_table_sha256s) == 33
    assert tuple((system, seed) for system, seed, _ in assembly.score_table_sha256s) == tuple(
        (system, seed)
        for system in ("PhaseSet-V2", "TMR-Set", *(f"A{index}" for index in range(1, 10)))
        for seed in SEEDS
    )
    assert tuple(row.comparison_id for row in assembly.comparisons) == (
        "H1", *(f"A{index}" for index in range(1, 10))
    )
    assert all(
        tuple(seed for seed, _ in comparison.seed_blocks) == SEEDS
        for comparison in assembly.comparisons
    )
    assert assembly.comparisons[0].baseline_system_id == "TMR-Set"
    first_seed_a1 = assembly.comparisons[1].seed_blocks[0][1]
    assert len(first_seed_a1) == 40
    assert sum(sum(row.control_text_to_motion_hits) for row in first_seed_a1) == 199
    assert sum(sum(row.control_motion_to_text_hits) for row in first_seed_a1) == 39
    report = v2_algorithm_report(assembly.comparisons, selected_b_star="TMR-Set")
    assert report.parent_count == 40 and report.caption_count == 200
    assert not report.scientific_or_test_authority


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_system", "wrong_seed"])
def test_rejects_incomplete_or_ambiguous_formal_matrix(score_fixture, fault):
    tables, census = score_fixture
    if fault == "missing":
        bad = tables[:-1]
    elif fault == "duplicate":
        bad = (*tables[:-1], tables[0])
    elif fault == "wrong_system":
        bad = (*tables[:-1], replace(tables[-1], system_id="other"))
    else:
        bad = (*tables[:-1], replace(tables[-1], seed=999))
    with pytest.raises(V2StatisticsError):
        assemble_v2_score_tables(
            bad, selected_b_star="TMR-Set", expected_census_sha256=census
        )


def test_rejects_score_table_that_changes_pre_score_census(score_fixture):
    tables, census = score_fixture
    changed = replace(
        tables[-1].dataset,
        caption_commitments=(
            _commitment("different", 0), *tables[-1].dataset.caption_commitments[1:]
        ),
    )
    with pytest.raises(V2StatisticsError, match="pre-score frozen census"):
        assemble_v2_score_tables(
            (*tables[:-1], replace(tables[-1], dataset=changed)),
            selected_b_star="TMR-Set",
            expected_census_sha256=census,
        )


@pytest.mark.parametrize(
    "fault", ["shape", "multi_positive", "positive", "group_size", "component"]
)
def test_rejects_wrong_parent_task_geometry_before_scoring(score_fixture, fault):
    tables, census = score_fixture
    original = tables[-1].dataset
    if fault == "shape":
        changed = replace(
            original,
            scores=np.ascontiguousarray(original.scores[:, :-1]),
            caption_commitments=original.caption_commitments[:-1],
            positive_motion_indices=original.positive_motion_indices[:-1],
        )
    elif fault == "multi_positive":
        changed = replace(
            original,
            positive_motion_indices=((0, 1), *original.positive_motion_indices[1:]),
        )
    elif fault == "positive":
        changed = replace(
            original,
            positive_motion_indices=((1,), *original.positive_motion_indices[1:]),
        )
    elif fault == "group_size":
        sizes = original.group_sizes.copy()
        sizes[0] = 4
        changed = replace(original, group_sizes=sizes)
    else:
        changed = replace(original, component_labels=("C11", *original.component_labels[1:]))
    with pytest.raises(V2StatisticsError):
        assemble_v2_score_tables(
            (*tables[:-1], replace(tables[-1], dataset=changed)),
            selected_b_star="TMR-Set",
            expected_census_sha256=census,
        )


def test_requires_pilot_chosen_literature_identity_and_frozen_digest(score_fixture):
    tables, census = score_fixture
    with pytest.raises(V2StatisticsError, match="allowed literature family"):
        assemble_v2_score_tables(
            tables, selected_b_star="A1", expected_census_sha256=census
        )
    with pytest.raises(V2StatisticsError, match="lowercase SHA-256"):
        assemble_v2_score_tables(
            tables, selected_b_star="TMR-Set", expected_census_sha256="0" * 63
        )
