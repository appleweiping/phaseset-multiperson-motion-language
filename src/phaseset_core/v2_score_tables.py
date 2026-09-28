"""Assemble PhaseSet-V2 paired outcomes from one frozen 40-by-200 score census.

This module has no authority to decide whether scores came from official data,
qualified checkpoints, or a single sealed test pass. The private execution host
must verify those facts before passing the score tables and a pre-score census
digest. The returned comparisons remain algorithm-only inputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from phaseset_core.evaluation import (
    RetrievalDataset,
    evaluate_retrieval,
    retrieval_census_sha256,
    validate_retrieval_dataset,
)
from phaseset_core.statistics import paired_captures_from_contributions
from phaseset_core.v2_statistics import (
    COMPARISONS,
    SEEDS,
    V2PairedComparison,
    V2StatisticsError,
)


STATUS = "SCORE_TABLE_ASSEMBLY_ONLY_NO_SEALED_TEST_AUTHORITY"
_B_STAR = frozenset({"TMR-Set", "WaMo-Set", "MIME-Set"})
_K_COUNTS = {3: 16, 4: 24}
_COMPONENT_COUNTS = {"C09": 16, "C11": 8, "C15": 16}


@dataclass(frozen=True, slots=True)
class V2ScoreTable:
    system_id: str
    seed: int
    dataset: RetrievalDataset


@dataclass(frozen=True, slots=True)
class V2ScoreAssembly:
    comparisons: tuple[V2PairedComparison, ...]
    census_sha256: str
    score_table_sha256s: tuple[tuple[str, int, str], ...]
    parent_count: int = 40
    caption_count: int = 200
    status: str = STATUS
    scientific_or_test_authority: bool = False


def _sha256_hex(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise V2StatisticsError("pre-score census must be a lowercase SHA-256 digest")
    return value


def _check_task_geometry(dataset: RetrievalDataset) -> None:
    if dataset.scores.shape != (40, 200):
        raise V2StatisticsError("PhaseSet-V2 score geometry must be 40 parents by 200 rows")
    if Counter(dataset.group_sizes.tolist()) != _K_COUNTS:
        raise V2StatisticsError("parent K=3/K=4 census differs from frozen task")
    if Counter(dataset.component_labels) != _COMPONENT_COUNTS:
        raise V2StatisticsError("parent component census differs from frozen task")
    if any(len(row) != 1 for row in dataset.positive_motion_indices):
        raise V2StatisticsError("each caption row must have exactly one parent")
    positive_counts = np.bincount(
        [row[0] for row in dataset.positive_motion_indices], minlength=40
    )
    if not np.array_equal(positive_counts, np.full(40, 5, dtype=np.int64)):
        raise V2StatisticsError("each parent must have five caption rows")


def assemble_v2_score_tables(
    tables: object,
    *,
    selected_b_star: object,
    expected_census_sha256: object,
) -> V2ScoreAssembly:
    """Pair scored systems without giving caller-supplied scores result authority.

    The host must compute ``expected_census_sha256`` from its independently
    sealed, score-free full gallery before opening any model score. This function
    checks all 33 tables against that same ordered census, then ranks each seed
    separately using the public evaluator and pairs by parent commitments.
    """

    if type(selected_b_star) is not str or selected_b_star not in _B_STAR:
        raise V2StatisticsError("B* must name an allowed literature family; host owns qualification")
    expected_census = _sha256_hex(expected_census_sha256)
    systems = ("PhaseSet-V2", selected_b_star, *COMPARISONS[1:])
    expected_keys = {(system, seed) for system in systems for seed in SEEDS}
    if type(tables) is not tuple or len(tables) != len(expected_keys):
        raise V2StatisticsError("exactly 33 V2/B*/A1--A9 seed score tables are required")

    checked: dict[tuple[str, int], RetrievalDataset] = {}
    for row in tables:
        if type(row) is not V2ScoreTable:
            raise TypeError("every score table must be exact V2ScoreTable")
        key = (row.system_id, row.seed)
        if key not in expected_keys or key in checked:
            raise V2StatisticsError("score system/seed is unexpected or duplicated")
        dataset = validate_retrieval_dataset(row.dataset)
        _check_task_geometry(dataset)
        if retrieval_census_sha256(dataset) != expected_census:
            raise V2StatisticsError("score table differs from pre-score frozen census")
        checked[key] = dataset
    if set(checked) != expected_keys:
        raise V2StatisticsError("one or more fixed system/seed score tables are missing")

    evaluated = {key: evaluate_retrieval(checked[key]) for key in sorted(checked)}
    comparisons = []
    for comparison_id in COMPARISONS:
        baseline = selected_b_star if comparison_id == "H1" else comparison_id
        seed_blocks = tuple(
            (
                seed,
                paired_captures_from_contributions(
                    evaluated[("PhaseSet-V2", seed)].capture_contributions,
                    evaluated[(baseline, seed)].capture_contributions,
                ),
            )
            for seed in SEEDS
        )
        comparisons.append(V2PairedComparison(comparison_id, baseline, seed_blocks))
    return V2ScoreAssembly(
        comparisons=tuple(comparisons),
        census_sha256=expected_census,
        score_table_sha256s=tuple(
            (system, seed, evaluated[(system, seed)].dataset_sha256)
            for system in systems
            for seed in SEEDS
        ),
    )


__all__ = [
    "STATUS", "V2ScoreAssembly", "V2ScoreTable", "assemble_v2_score_tables",
]
