from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import inspect
from types import SimpleNamespace

import pytest

from phaseset_core import base_cohort_latency as latency
from phaseset_core import base_cohort_qualification as assembly
from phaseset_core import experiments, production


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _combined_rows() -> tuple[assembly.CombinedBaseScoreBinding, ...]:
    counts = {"B0": 100, "B1": 200, "B2": 300}
    fractions = {"B0": Fraction(1, 2), "B1": Fraction(3, 4), "B2": Fraction(3, 4)}
    latencies = {"B0": 300, "B1": 200, "B2": 100}
    rows = []
    for ordinal, run_id in enumerate(experiments.base_run_ids()):
        _role, seed, system_id = experiments.parse_run_id(run_id)
        rows.append(
            assembly.CombinedBaseScoreBinding(
                run_id=run_id,
                attempt_id=f"attempt-{ordinal}",
                system_id=system_id,
                seed=seed,
                terminal_sha256=_sha(f"terminal-{ordinal}"),
                selected_checkpoint_sha256=_sha(f"checkpoint-{ordinal}"),
                selected_checkpoint_state_digest=_sha(f"state-{ordinal}"),
                validation_score_artifact_sha256=_sha(f"score-{ordinal}"),
                latency_row_sha256=_sha(f"latency-{ordinal}"),
                latency_samples_sha256=_sha(f"samples-{ordinal}"),
                parameter_census_sha256=_sha(f"parameters-{system_id}"),
                parameter_count=counts[system_id],
                frozen_runtime_latency_ns=latencies[system_id] + ordinal,
                bidirectional_capture_r1=fractions[system_id],
                validation_manifest_sha256=_sha("val-census"),
                validation_source_manifest_sha256=_sha("val-source"),
                query_census_sha256=_sha("val-census"),
                evaluator_sha256=_sha("evaluator"),
                runtime_sha256=_sha("runtime"),
                environment_sha256=_sha("environment"),
                precision_mode="FP32",
            )
        )
    return tuple(rows)


def _dependency(selector_artifact: bytes) -> assembly.BaseQualificationDependencyBinding:
    return assembly.BaseQualificationDependencyBinding(
        admission_sha256=_sha("admission"),
        resolved_cohort_sha256=_sha("resolved"),
        scored_cohort_sha256=_sha("scored"),
        latency_session_sha256=_sha("latency-session"),
        plan_sha256=_sha("plan"),
        matrix_sha256=_sha("matrix"),
        training_config_sha256=_sha("config"),
        source_tree_sha256=_sha("source-tree"),
        train_manifest_sha256=_sha("train"),
        admitted_val_census_sha256=_sha("val-census"),
        validation_source_manifest_sha256=_sha("val-source"),
        query_census_sha256=_sha("val-census"),
        resolver_source_sha256=_sha("resolver-source"),
        scorer_source_sha256=_sha("scorer-source"),
        latency_source_sha256=_sha("latency-source"),
        assembler_source_sha256=_sha("assembler-source"),
        production_source_sha256=_sha("production-source"),
        selector_artifact_sha256=hashlib.sha256(selector_artifact).hexdigest(),
        runtime_sha256=_sha("runtime"),
        environment_sha256=_sha("environment"),
        precision_mode="FP32",
    )


def _structural_result() -> assembly.BaseCohortQualificationAssembly:
    rows = _combined_rows()
    score_rows = assembly._base_scores(rows)
    qualification = experiments.qualify_base(score_rows)
    selector = b"analytic-selector-bytes"
    dependency = _dependency(selector)
    evaluator = dependency.selector_artifact_sha256
    rebound = tuple(replace(row, evaluator_sha256=evaluator) for row in rows)
    score_rows = assembly._base_scores(rebound)
    qualification = experiments.qualify_base(score_rows)
    evidence = production.BaseQualificationExecutionEvidence(
        score_rows=score_rows,
        selector_artifact=selector,
    )
    return assembly.BaseCohortQualificationAssembly(
        qualification=qualification,
        execution_evidence=evidence,
        artifact=experiments.canonical_base_qualification_bytes(qualification),
        row_bindings=rebound,
        dependency_binding=dependency,
    )


def test_exact_99_sample_median_is_recomputed_from_raw_integers() -> None:
    samples = tuple(range(1, 100))

    assert assembly._median_from_samples(samples) == 50
    assert len(assembly._samples_sha256(samples)) == 64
    with pytest.raises(assembly.BaseCohortQualificationError, match="exact 99"):
        assembly._median_from_samples(samples[:-1])
    with pytest.raises(assembly.BaseCohortQualificationError, match="positive"):
        assembly._median_from_samples(samples[:-1] + (0,))


def test_combined_row_digest_changes_for_score_or_latency_evidence() -> None:
    row = _combined_rows()[0]

    assert replace(row, validation_score_artifact_sha256=_sha("other-score")).sha256 != row.sha256
    assert replace(row, latency_row_sha256=_sha("other-latency")).sha256 != row.sha256
    assert replace(row, latency_samples_sha256=_sha("other-samples")).sha256 != row.sha256


def test_base_scores_use_fraction_live_count_median_and_combined_digest() -> None:
    bindings = _combined_rows()
    scores = assembly._base_scores(bindings)

    assert tuple(row.run_id for row in scores) == experiments.base_run_ids()
    for binding, score in zip(bindings, scores, strict=True):
        assert (
            Fraction(
                score.bidirectional_r1_numerator,
                score.bidirectional_r1_denominator,
            )
            == binding.bidirectional_capture_r1
        )
        assert score.parameter_count == binding.parameter_count
        assert score.frozen_runtime_latency_ns == binding.frozen_runtime_latency_ns
        assert score.score_artifact_sha256 == binding.sha256
        assert score.evaluator_sha256 == binding.evaluator_sha256


def test_existing_selector_and_canonical_evidence_are_retained() -> None:
    result = _structural_result()

    assert result.qualification.winner_system_id == "B1"
    assert result.execution_evidence.score_rows == result.qualification.score_rows
    assert result.artifact == experiments.canonical_base_qualification_bytes(result.qualification)
    assert result.authority == 0
    assert result.production is False
    assert result.result_claimed is False
    assert len(result.sha256) == 64


def test_result_rejects_artifact_and_combined_row_drift() -> None:
    result = _structural_result()

    with pytest.raises(assembly.BaseCohortQualificationError, match="not canonical"):
        replace(result, artifact=result.artifact + b"\n")
    changed = list(result.row_bindings)
    changed[0] = replace(
        changed[0],
        validation_score_artifact_sha256=_sha("mutated-score"),
    )
    with pytest.raises(assembly.BaseCohortQualificationError, match="combined evidence"):
        replace(result, row_bindings=tuple(changed))


def test_row_builder_closes_selected_state_runtime_census_median_and_device() -> None:
    runtime = latency.BaseLatencyRuntimeIdentity(
        device="cuda:1",
        device_uuid="GPU-11111111-2222-3333-4444-555555555555",
        device_name="registered-device",
        torch_version="2.12.0+cu126",
        cuda_runtime_version="12.6",
        cudnn_version=90100,
        driver_version="570.00",
        nvidia_total_memory_mib=49140,
        compute_mode="Default",
        torch_total_memory_bytes=50_807_242_752,
        initial_allocated_bytes=0,
        initial_reserved_bytes=0,
        environment_sha256=_sha("environment"),
        precision_mode="FP32",
    )
    resolved_rows = []
    score_rows = []
    latency_rows = []
    visits = []
    val_census = _sha("val-census")
    val_source = _sha("val-source")
    for ordinal, run_id in enumerate(experiments.base_run_ids()):
        _role, seed, system_id = experiments.parse_run_id(run_id)
        raw = f"selected-{ordinal}".encode("ascii")
        checkpoint_sha = hashlib.sha256(raw).hexdigest()
        state = _sha(f"state-{ordinal}")
        parameter_sha = _sha(f"parameter-{system_id}")
        parameter_count = {"B0": 10, "B1": 20, "B2": 30}[system_id]
        resolved_rows.append(
            SimpleNamespace(
                run_id=run_id,
                attempt_id=f"attempt-{ordinal}",
                system_id=system_id,
                seed=seed,
                terminal_sha256=_sha(f"terminal-{ordinal}"),
                selected_checkpoint=SimpleNamespace(
                    raw=raw,
                    sha256=checkpoint_sha,
                    state_digest=state,
                ),
            )
        )
        score_rows.append(
            SimpleNamespace(
                run_id=run_id,
                attempt_id=f"attempt-{ordinal}",
                system_id=system_id,
                seed=seed,
                terminal_sha256=_sha(f"terminal-{ordinal}"),
                selected_checkpoint_sha256=checkpoint_sha,
                selected_checkpoint_state_digest=state,
                parameter_census=SimpleNamespace(
                    sha256=parameter_sha,
                    total=parameter_count,
                ),
                score_artifact_sha256=_sha(f"score-{ordinal}"),
                primary_capture_r1=Fraction(ordinal, 8),
                environment_sha256=runtime.environment_sha256,
                precision_mode=runtime.precision_mode,
                device=runtime.device,
                admitted_val_census_sha256=val_census,
                query_census_sha256=val_census,
                source_manifest_sha256=val_source,
            )
        )
        samples = tuple(1000 + ordinal + offset for offset in range(99))
        for visit_index in range(9):
            start = visit_index * 11
            visits.append(
                SimpleNamespace(
                    run_id=run_id,
                    completed=True,
                    samples_ns=samples[start : start + 11],
                )
            )
        latency_rows.append(
            SimpleNamespace(
                run_id=run_id,
                system_id=system_id,
                seed=seed,
                selected_checkpoint_sha256=checkpoint_sha,
                selected_checkpoint_state_digest=state,
                parameter_census_sha256=parameter_sha,
                parameter_count=parameter_count,
                runtime_sha256=runtime.sha256,
                samples_ns=samples,
                median_ns=sorted(samples)[49],
                sha256=_sha(f"latency-{ordinal}"),
            )
        )
    cohort = SimpleNamespace(rows=tuple(resolved_rows), val_manifest_sha256=val_census)
    scored = SimpleNamespace(
        observations=tuple(score_rows),
        source_manifest_sha256=val_source,
    )
    measured = SimpleNamespace(
        runtime_identity=runtime,
        rows=tuple(latency_rows),
        visits=tuple(visits),
    )

    rows = assembly._build_row_bindings(
        cohort,
        scored,
        measured,
        _sha("evaluator"),
    )
    assert tuple(row.run_id for row in rows) == experiments.base_run_ids()
    assert rows[0].frozen_runtime_latency_ns == 1049
    assert rows[0].parameter_count == 10
    assert score_rows[0].device == runtime.device == "cuda:1"
    score_rows[0].device = "cuda"
    with pytest.raises(assembly.BaseCohortQualificationError, match="runtime, or census"):
        assembly._build_row_bindings(
            cohort,
            scored,
            measured,
            _sha("evaluator"),
        )
    score_rows[0].device = runtime.device
    latency_rows[0].median_ns += 1
    with pytest.raises(assembly.BaseCohortQualificationError, match="runtime, or census"):
        assembly._build_row_bindings(
            cohort,
            scored,
            measured,
            _sha("evaluator"),
        )


def test_fake_or_held_latency_cannot_reach_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latency,
        "_checked_inputs",
        lambda *_args: (SimpleNamespace(), SimpleNamespace(), SimpleNamespace()),
    )
    fake_session = object.__new__(latency.BaseCohortLatencySession)
    object.__setattr__(fake_session, "authority", latency.AUTHORITY)
    object.__setattr__(fake_session, "production", latency.PRODUCTION)
    object.__setattr__(fake_session, "result_claimed", latency.RESULT_CLAIMED)
    object.__setattr__(fake_session, "status", latency.FAKE_COMPLETE_STATUS)
    object.__setattr__(fake_session, "actual_runtime", False)
    object.__setattr__(fake_session, "complete", True)
    object.__setattr__(fake_session, "formal_samples_complete", False)
    object.__setattr__(fake_session, "failure_code", None)

    with pytest.raises(assembly.BaseCohortQualificationError, match="not one actual"):
        assembly._checked_inputs(object(), object(), object(), fake_session)


def test_public_entry_accepts_no_submitted_metrics_or_dependencies() -> None:
    names = tuple(inspect.signature(assembly.assemble_base_cohort_qualification).parameters)

    assert names == ("cohort", "admission", "scored_cohort", "latency_session")
    assert not {
        "scores",
        "score_rows",
        "parameter_count",
        "latency_ns",
        "samples",
        "evaluator_sha256",
        "selector_artifact",
        "winner",
    }.intersection(names)
