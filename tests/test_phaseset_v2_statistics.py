"""Synthetic-only V2 inference arithmetic; no sealed scores or result authority."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from phaseset_core.statistics import PairedCapture
from phaseset_core import statistics, v2_statistics


def _key(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _parent(index: int) -> PairedCapture:
    captions = tuple(sorted(_key(f"caption-{index}-{ordinal}") for ordinal in range(index + 1)))
    return PairedCapture(
        capture_id=_key(f"parent-{index}"),
        caption_ids=captions,
        motion_ids=(_key(f"motion-{index}"),),
        treatment_text_to_motion_hits=(1,) * len(captions),
        control_text_to_motion_hits=(0,) * len(captions),
        treatment_motion_to_text_hits=(1,),
        control_motion_to_text_hits=(0,),
    )


def _comparisons() -> tuple[v2_statistics.V2PairedComparison, ...]:
    parents = tuple(sorted((_parent(0), _parent(1), _parent(2)), key=lambda row: row.capture_id))
    blocks = tuple((seed, parents) for seed in v2_statistics.SEEDS)
    return tuple(
        v2_statistics.V2PairedComparison(
            comparison_id=identifier,
            baseline_system_id="TMR-Set" if identifier == "H1" else identifier,
            seed_blocks=blocks,
        )
        for identifier in v2_statistics.COMPARISONS
    )


@pytest.fixture(scope="module")
def witness() -> v2_statistics.V2AlgorithmReport:
    return v2_statistics.v2_algorithm_report(
        _comparisons(), selected_b_star="TMR-Set"
    )


def test_v2_algorithm_shared_parent_stream_nine_way_holm_and_no_authority(
    witness: v2_statistics.V2AlgorithmReport,
) -> None:
    assert witness.status == v2_statistics.STATUS
    assert witness.scientific_or_test_authority is False
    assert witness.draws == 100_000
    assert witness.resampling_seed == 20260928
    assert witness.parent_count == 3
    assert witness.caption_count == 6
    assert tuple(row.comparison_id for row in witness.rows) == v2_statistics.COMPARISONS
    assert witness.index_stream_sha256 == (
        "a1c6d56a77065d334654f04a3806dc9740fcd62d3e4383e1b51ca0e089af19cb"
    )
    assert len(witness.input_rows_sha256) == 64
    assert witness.rows[0].baseline_system_id == "TMR-Set"
    assert witness.rows[0].holm_adjusted_p_value is None
    assert all(row.observed_effect == 1.0 for row in witness.rows)
    assert all(row.confidence_lower == 1.0 for row in witness.rows)
    assert all(row.claim_gate_passed for row in witness.rows)
    assert all(row.holm_adjusted_p_value is not None for row in witness.rows[1:])
    assert witness.rows[1].holm_adjusted_p_value == pytest.approx(18 / 100_001)


def test_v2_comparison_order_is_canonical() -> None:
    comparisons = _comparisons()
    shuffled = comparisons[4:] + comparisons[:4]
    ordered = v2_statistics.v2_algorithm_report(
        shuffled, selected_b_star="TMR-Set"
    )
    assert tuple(row.comparison_id for row in ordered.rows) == v2_statistics.COMPARISONS


def test_v2_parent_resampling_stream_matches_legacy_cpu_oracle(
    witness: v2_statistics.V2AlgorithmReport,
) -> None:
    oracle = statistics.paired_seed_blocked_bootstrap(
        _comparisons()[1].seed_blocks,
        resampling_seed=v2_statistics.BOOTSTRAP_SEED,
    )
    assert witness.index_stream_sha256 == oracle.index_stream_sha256
    assert witness.rows[1].observed_effect == oracle.observed_effect
    assert witness.rows[1].confidence_lower == oracle.confidence_lower
    assert witness.rows[1].confidence_upper == oracle.confidence_upper
    assert witness.rows[1].raw_p_value == oracle.two_sided_p_value


def test_v2_missing_seed_or_wrong_comparator_fails_before_bootstrap() -> None:
    comparisons = _comparisons()
    missing_seed = replace(comparisons[0], seed_blocks=comparisons[0].seed_blocks[:2])
    with pytest.raises(v2_statistics.V2StatisticsError, match="three fixed seed"):
        v2_statistics.v2_algorithm_report(
            (missing_seed,) + comparisons[1:], selected_b_star="TMR-Set"
        )
    wrong_baseline = replace(comparisons[0], baseline_system_id="MIME-Set")
    with pytest.raises(v2_statistics.V2StatisticsError, match="baseline differs"):
        v2_statistics.v2_algorithm_report(
            (wrong_baseline,) + comparisons[1:], selected_b_star="TMR-Set"
        )


def test_v2_treatment_and_query_census_are_identical_across_controls() -> None:
    comparisons = list(_comparisons())
    seed, rows = comparisons[1].seed_blocks[0]
    changed = replace(rows[0], treatment_motion_to_text_hits=(0,))
    comparisons[1] = replace(
        comparisons[1],
        seed_blocks=((seed, (changed,) + rows[1:]),) + comparisons[1].seed_blocks[1:],
    )
    with pytest.raises(v2_statistics.V2StatisticsError, match="treatment hits differ"):
        v2_statistics.v2_algorithm_report(
            tuple(comparisons), selected_b_star="TMR-Set"
        )

    comparisons = list(_comparisons())
    seed, rows = comparisons[1].seed_blocks[2]
    changed = replace(rows[0], treatment_motion_to_text_hits=(0,))
    comparisons[1] = replace(
        comparisons[1],
        seed_blocks=comparisons[1].seed_blocks[:2] + ((seed, (changed,) + rows[1:]),),
    )
    with pytest.raises(v2_statistics.V2StatisticsError, match="treatment hits differ"):
        v2_statistics.v2_algorithm_report(
            tuple(comparisons), selected_b_star="TMR-Set"
        )

    comparisons = list(_comparisons())
    seed, rows = comparisons[1].seed_blocks[0]
    changed = replace(rows[0], caption_ids=(_key("wrong"),) + rows[0].caption_ids[1:])
    comparisons[1] = replace(
        comparisons[1],
        seed_blocks=((seed, (changed,) + rows[1:]),) + comparisons[1].seed_blocks[1:],
    )
    with pytest.raises(ValueError, match="caption_ids|census"):
        v2_statistics.v2_algorithm_report(
            tuple(comparisons), selected_b_star="TMR-Set"
        )


def test_v2_parent_macro_and_seed_mean_precede_common_parent_resampling() -> None:
    parents = tuple(sorted((_parent(0), _parent(2)), key=lambda row: row.capture_id))
    patterns = {
        1729: {1: ((1,), (1,)), 3: ((1, 0, 0), (0,))},
        2718: {1: ((0,), (0,)), 3: ((1, 1, 0), (1,))},
        31415: {1: ((1,), (0,)), 3: ((0, 0, 0), (0,))},
    }
    blocks = tuple(
        (
            seed,
            tuple(
                replace(
                    row,
                    treatment_text_to_motion_hits=patterns[seed][len(row.caption_ids)][0],
                    treatment_motion_to_text_hits=patterns[seed][len(row.caption_ids)][1],
                )
                for row in parents
            ),
        )
        for seed in v2_statistics.SEEDS
    )
    comparisons = tuple(
        v2_statistics.V2PairedComparison(
            comparison_id=identifier,
            baseline_system_id="TMR-Set" if identifier == "H1" else identifier,
            seed_blocks=blocks,
        )
        for identifier in v2_statistics.COMPARISONS
    )
    report = v2_statistics.v2_algorithm_report(
        comparisons, selected_b_star="TMR-Set"
    )
    assert report.caption_count == 4
    assert report.rows[0].seed_effects == pytest.approx((7 / 12, 5 / 12, 1 / 4))
    assert report.rows[0].observed_effect == pytest.approx(5 / 12)
    assert report.rows[0].confidence_lower == pytest.approx(1 / 3)
    assert report.rows[0].confidence_upper == pytest.approx(1 / 2)
    assert report.rows[0].raw_p_value == pytest.approx(1 / 100_001)


def test_v2_h1_strict_margin_boundary_and_zero_effect_holm() -> None:
    rows = tuple(
        PairedCapture(
            capture_id=_key(f"boundary-parent-{index}"),
            caption_ids=tuple(sorted(_key(f"boundary-{index}-{ordinal}") for ordinal in range(25))),
            motion_ids=(_key(f"boundary-motion-{index}"),),
            treatment_text_to_motion_hits=(1,) + (0,) * 24,
            control_text_to_motion_hits=(0,) * 25,
            treatment_motion_to_text_hits=(0,),
            control_motion_to_text_hits=(0,),
        )
        for index in range(2)
    )
    rows = tuple(sorted(rows, key=lambda row: row.capture_id))
    blocks = tuple((seed, rows) for seed in v2_statistics.SEEDS)
    comparisons = tuple(
        v2_statistics.V2PairedComparison(
            comparison_id=identifier,
            baseline_system_id="TMR-Set" if identifier == "H1" else identifier,
            seed_blocks=blocks,
        )
        for identifier in v2_statistics.COMPARISONS
    )
    report = v2_statistics.v2_algorithm_report(
        comparisons, selected_b_star="TMR-Set"
    )
    assert report.rows[0].observed_effect == pytest.approx(0.02)
    assert report.rows[0].confidence_lower == pytest.approx(0.02)
    assert report.rows[0].claim_gate_passed is False

    zero_blocks = tuple(
        (
            seed,
            tuple(replace(row, treatment_text_to_motion_hits=(0,) * 25) for row in rows),
        )
        for seed in v2_statistics.SEEDS
    )
    zero_comparisons = tuple(replace(row, seed_blocks=zero_blocks) for row in comparisons)
    zero_report = v2_statistics.v2_algorithm_report(
        zero_comparisons, selected_b_star="TMR-Set"
    )
    assert all(row.claim_gate_passed is False for row in zero_report.rows)
    assert all(row.raw_p_value == 1.0 for row in zero_report.rows[1:])
    assert all(row.holm_adjusted_p_value == 1.0 for row in zero_report.rows[1:])


def test_v2_holm_is_nine_way_step_down_with_fixed_tie_order() -> None:
    raw = {f"A{index}": index / 1000 for index in range(1, 10)}
    adjusted = v2_statistics._holm(raw)
    assert adjusted["A1"] == pytest.approx(0.009)
    assert adjusted["A2"] == pytest.approx(0.016)
    assert adjusted["A3"] == pytest.approx(0.021)
    assert adjusted["A9"] == pytest.approx(0.025)
