"""Data-free PhaseSet-V2 decision arithmetic, without result authority.

The caller must supply independently ranked, paired parent-level hit rows from
one frozen gallery. This module neither loads scores nor authenticates their
origin. Its output is an algorithm witness, never a sealed-test report.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from phaseset_core.statistics import PairedCapture, summarize_variable_captions


SEEDS = (1729, 2718, 31415)
COMPARISONS = ("H1",) + tuple(f"A{index}" for index in range(1, 10))
HOLM_FAMILY = COMPARISONS[1:]
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260928
H1_MARGIN = Fraction(2, 100)
STATUS = "ALGORITHM_ONLY_UNBOUND_NO_SEALED_TEST_AUTHORITY"


class V2StatisticsError(ValueError):
    """A predeclared comparison or paired query census was violated."""


@dataclass(frozen=True, slots=True)
class V2PairedComparison:
    comparison_id: str
    baseline_system_id: str
    seed_blocks: tuple[tuple[int, tuple[PairedCapture, ...]], ...]


@dataclass(frozen=True, slots=True)
class V2DecisionRow:
    comparison_id: str
    baseline_system_id: str
    seed_effects: tuple[float, float, float]
    observed_effect: float
    confidence_lower: float
    confidence_upper: float
    raw_p_value: float
    holm_adjusted_p_value: float | None
    claim_gate_passed: bool


@dataclass(frozen=True, slots=True)
class V2AlgorithmReport:
    rows: tuple[V2DecisionRow, ...]
    draws: int
    resampling_seed: int
    parent_count: int
    caption_count: int
    index_stream_sha256: str
    input_rows_sha256: str
    schema: str = "phaseset-v2-algorithm-witness-v1"
    status: str = STATUS
    scientific_or_test_authority: bool = False


def _effect(row: PairedCapture) -> Fraction:
    text = Fraction(
        sum(row.treatment_text_to_motion_hits) - sum(row.control_text_to_motion_hits),
        len(row.caption_ids),
    )
    motion = Fraction(
        sum(row.treatment_motion_to_text_hits) - sum(row.control_motion_to_text_hits),
        len(row.motion_ids),
    )
    return (text + motion) / 2


def _census(row: PairedCapture) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return row.capture_id, row.caption_ids, row.motion_ids


def _treatment_hits(row: PairedCapture) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return row.treatment_text_to_motion_hits, row.treatment_motion_to_text_hits


def _normalise(
    comparisons: object,
    *,
    selected_b_star: object,
) -> tuple[tuple[V2PairedComparison, ...], np.ndarray, tuple[tuple[float, ...], ...], int, str]:
    if type(selected_b_star) is not str or selected_b_star not in {
        "TMR-Set", "WaMo-Set", "MIME-Set"
    }:
        raise V2StatisticsError("B* must be one pilot-qualified literature family")
    if type(comparisons) is not tuple or len(comparisons) != len(COMPARISONS):
        raise V2StatisticsError("exactly H1 and A1--A9 comparisons are required")
    by_id: dict[str, V2PairedComparison] = {}
    for comparison in comparisons:
        if type(comparison) is not V2PairedComparison:
            raise TypeError("comparison must be exact V2PairedComparison")
        identifier = comparison.comparison_id
        if identifier not in COMPARISONS or identifier in by_id:
            raise V2StatisticsError("comparison IDs must be unique H1,A1--A9")
        expected_baseline = selected_b_star if identifier == "H1" else identifier
        if comparison.baseline_system_id != expected_baseline:
            raise V2StatisticsError(f"{identifier} baseline differs from frozen registry")
        by_id[identifier] = comparison
    ordered = tuple(by_id[identifier] for identifier in COMPARISONS)

    reference_census = None
    reference_treatment = None
    parent_count = None
    caption_count = None
    effects: list[list[float]] = []
    per_seed: list[tuple[float, ...]] = []
    digest_rows: list[dict[str, object]] = []
    for comparison in ordered:
        if (
            type(comparison.seed_blocks) is not tuple
            or len(comparison.seed_blocks) != len(SEEDS)
        ):
            raise V2StatisticsError("all three fixed seed blocks are required")
        seed_effect_vectors: list[list[Fraction]] = []
        digest_blocks: list[dict[str, object]] = []
        for expected_seed, block in zip(SEEDS, comparison.seed_blocks, strict=True):
            if type(block) is not tuple or len(block) != 2 or block[0] != expected_seed:
                raise V2StatisticsError("seed blocks must be ordered 1729,2718,31415")
            rows = block[1]
            summary = summarize_variable_captions(rows)
            if summary.capture_count < 2:
                raise V2StatisticsError("at least two independent parents are required")
            if tuple(row.capture_id for row in rows) != tuple(
                sorted(row.capture_id for row in rows)
            ):
                raise V2StatisticsError("parent rows must be sorted by commitment")
            census = tuple(_census(row) for row in rows)
            treatment = tuple(_treatment_hits(row) for row in rows)
            if reference_census is None:
                reference_census = census
                reference_treatment = {expected_seed: treatment}
                parent_count = summary.capture_count
                caption_count = summary.total_caption_count
            else:
                if census != reference_census or summary.total_caption_count != caption_count:
                    raise V2StatisticsError("comparison/seed parent-query census differs")
                assert reference_treatment is not None
                prior_treatment = reference_treatment.setdefault(expected_seed, treatment)
                if prior_treatment != treatment:
                    raise V2StatisticsError("V2 treatment hits differ across comparisons")
            fractions = [_effect(row) for row in rows]
            seed_effect_vectors.append(fractions)
            digest_blocks.append({
                "seed": expected_seed,
                "rows": [
                    {
                        "parent": row.capture_id,
                        "caption_ids": row.caption_ids,
                        "motion_ids": row.motion_ids,
                        "treatment_t2m": row.treatment_text_to_motion_hits,
                        "control_t2m": row.control_text_to_motion_hits,
                        "treatment_m2t": row.treatment_motion_to_text_hits,
                        "control_m2t": row.control_motion_to_text_hits,
                    }
                    for row in rows
                ],
            })
        assert parent_count is not None
        seed_means = tuple(float(sum(vector, start=Fraction()) / parent_count)
                           for vector in seed_effect_vectors)
        parent_effects = [
            float(sum((vector[index] for vector in seed_effect_vectors), start=Fraction())
                  / len(SEEDS))
            for index in range(parent_count)
        ]
        effects.append(parent_effects)
        per_seed.append(seed_means)
        digest_rows.append({
            "comparison_id": comparison.comparison_id,
            "baseline_system_id": comparison.baseline_system_id,
            "seed_blocks": digest_blocks,
        })
    payload = json.dumps(digest_rows, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return (
        ordered,
        np.asarray(effects, dtype=np.float64),
        tuple(per_seed),
        parent_count,
        hashlib.sha256(payload).hexdigest(),
    )


def _holm(raw_p: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p.items(), key=lambda item: (item[1], item[0]))
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (identifier, probability) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * probability))
        adjusted[identifier] = running
    return adjusted


def v2_algorithm_report(
    comparisons: object,
    *,
    selected_b_star: object,
) -> V2AlgorithmReport:
    """Apply the frozen arithmetic to paired hits, without granting result authority."""

    ordered, effects, per_seed, parent_count, input_hash = _normalise(
        comparisons, selected_b_star=selected_b_star
    )
    samples = np.empty((len(COMPARISONS), BOOTSTRAP_DRAWS), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    index_hash = hashlib.sha256()
    chunk = min(4096, max(1, 1_000_000 // parent_count))
    for offset in range(0, BOOTSTRAP_DRAWS, chunk):
        count = min(chunk, BOOTSTRAP_DRAWS - offset)
        indices = rng.integers(0, parent_count, size=(count, parent_count), dtype=np.uint64)
        index_hash.update(np.asarray(indices, dtype="<u8", order="C").tobytes())
        samples[:, offset:offset + count] = effects[:, indices].mean(axis=2)

    raw_p: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for index, identifier in enumerate(COMPARISONS):
        draws = samples[index]
        if not np.isfinite(draws).all():
            raise V2StatisticsError("bootstrap contains non-finite effects")
        if identifier == "H1":
            raw_p[identifier] = (1 + int(np.count_nonzero(draws <= float(H1_MARGIN)))) / (
                BOOTSTRAP_DRAWS + 1
            )
        else:
            left = (1 + int(np.count_nonzero(draws <= 0.0))) / (BOOTSTRAP_DRAWS + 1)
            right = (1 + int(np.count_nonzero(draws >= 0.0))) / (BOOTSTRAP_DRAWS + 1)
            raw_p[identifier] = min(1.0, 2.0 * min(left, right))
        draws.sort()
        intervals[identifier] = (float(draws[2_499]), float(draws[97_499]))
    adjusted = _holm({identifier: raw_p[identifier] for identifier in HOLM_FAMILY})
    rows = []
    for index, comparison in enumerate(ordered):
        identifier = comparison.comparison_id
        observed = float(effects[index].mean())
        if not math.isfinite(observed):
            raise V2StatisticsError("observed effect is not finite")
        lower, upper = intervals[identifier]
        if identifier == "H1":
            gate = observed >= float(H1_MARGIN) and raw_p[identifier] <= 0.05 and (
                lower > float(H1_MARGIN)
            )
            holm_p = None
        else:
            holm_p = adjusted[identifier]
            gate = lower > 0.0 and holm_p <= 0.05
        rows.append(V2DecisionRow(
            comparison_id=identifier,
            baseline_system_id=comparison.baseline_system_id,
            seed_effects=per_seed[index],
            observed_effect=observed,
            confidence_lower=lower,
            confidence_upper=upper,
            raw_p_value=raw_p[identifier],
            holm_adjusted_p_value=holm_p,
            claim_gate_passed=gate,
        ))
    return V2AlgorithmReport(
        rows=tuple(rows),
        draws=BOOTSTRAP_DRAWS,
        resampling_seed=BOOTSTRAP_SEED,
        parent_count=parent_count,
        caption_count=sum(len(row.caption_ids) for row in ordered[0].seed_blocks[0][1]),
        index_stream_sha256=index_hash.hexdigest(),
        input_rows_sha256=input_hash,
    )


__all__ = [
    "BOOTSTRAP_DRAWS", "BOOTSTRAP_SEED", "COMPARISONS", "H1_MARGIN", "HOLM_FAMILY",
    "SEEDS", "STATUS", "V2AlgorithmReport", "V2DecisionRow", "V2PairedComparison",
    "V2StatisticsError", "v2_algorithm_report",
]
