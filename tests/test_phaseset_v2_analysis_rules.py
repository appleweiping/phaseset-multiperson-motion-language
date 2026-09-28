"""Static pre-pilot V2 decision rules; this test never reads test motions or scores."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_pretest_rules_match_frozen_matrix_and_supervision() -> None:
    rules = json.loads((ROOT / "configs/phaseset_v2_analysis_rules.json").read_text())
    matrix = json.loads((ROOT / "configs/phaseset_v2_experiment_matrix.json").read_text())
    supervision = json.loads(
        (ROOT / "configs/phaseset_v2_relation_supervision.json").read_text()
    )

    assert rules["schema"] == "phaseset-v2-pretest-analysis-rules-v1"
    assert rules["status"] == "DRAFT_PRE_PILOT_NO_SCORES_REVIEW_REVISION"
    assert rules["pilot_comparator"]["selected_B_star"] is None
    assert rules["pilot_comparator"]["selected_candidate"] is None
    assert rules["pilot_comparator"]["families"] == [
        "TMR-Set",
        "WaMo-Set",
        "MIME-Set",
    ]
    assert rules["pilot_comparator"]["train_components"] == ["C01", "C02"]
    assert rules["pilot_comparator"]["validation_components"] == ["C03"]
    assert rules["v2_head_pilot_selection"]["candidate_ids"] == [
        "v2-cf-half",
        "v2-cf-fixed",
    ]
    assert rules["v2_head_pilot_selection"]["selected_candidate"] is None
    assert "original_task_sanity" in " ".join(
        rules["pilot_comparator"]["fidelity_qualification"]
    )
    assert matrix["primary_split"]["train_released_episodes"] == 400
    assert matrix["primary_split"]["validation_released_episodes"] == 96
    assert matrix["primary_split"]["test_released_episodes"] == 76
    assert matrix["primary_split"]["sealed_test_parent_captures"] == 40
    assert matrix["primary_split"]["sealed_test_official_human_rows"] == 200
    assert "relation_challenge_gain_pp" not in matrix["success_thresholds"]
    assert matrix["historical_superseded_relation_challenge"]["status"].startswith(
        "UNASSESSABLE"
    )
    assert rules["formal_comparison"]["training_seeds"] == matrix["seeds"]
    assert rules["task"]["sealed_test_uses_machine_weak_text"] is False
    assert supervision["primary_retrieval"]["machine_text_in_primary_gallery"] is False
    assert rules["claim_gates"]["H1"]["null_margin_pp"] == matrix[
        "success_thresholds"
    ]["retrieval_mean_r1_gain_pp"]
    assert rules["claim_gates"]["H1"]["point_estimate_at_least_pp"] == 2
    assert "intentional_conjunctive_conservative_gate" in rules["inference"][
        "h1_decision_calibration"
    ]
    assert rules["secondary"]["inference_cost_ratio_target"] == matrix[
        "success_thresholds"
    ]["inference_cost_ratio"]
    assert rules["inference"]["draws"] == 100_000
    assert rules["inference"]["holm_family"] == [
        f"A{number}" for number in range(1, 10)
    ]
    assert rules["claim_gates"]["relation_challenge"].startswith("UNASSESSABLE")
    assert len(matrix["runs"]) == matrix["budget"]["formal_stages"] == 87
