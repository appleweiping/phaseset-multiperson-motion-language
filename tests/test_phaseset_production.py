from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phaseset_core import (
    evaluation,
    execution,
    experiments,
    production,
    publication,
    statistics,
)
from phaseset_core.experiments import FINAL_SYSTEMS, SEEDS


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _receipts(
    *,
    admitted_at: str = "2026-08-25T15:00:00Z",
    score_execution_census_sha256: str | None = None,
    validation_evaluation_census_sha256: str | None = None,
    hard_gallery_freeze_binding_sha256: str | None = None,
    validation_hard_gallery_collection_sha256: str | None = None,
) -> production.PrivateReceiptAssertions:
    return production.PrivateReceiptAssertions(
        admitted_at_utc=admitted_at,
        data_manifest_sha256=_digest("data"),
        prepared_data_manifest_sha256=_digest("prepared"),
        split_audit_sha256=_digest("split"),
        rights_assertion_sha256=_digest("rights"),
        caption_manifest_sha256=_digest("human-holistic-caption-manifest"),
        score_execution_census_sha256=(
            score_execution_census_sha256 or _digest("score-execution-census")
        ),
        validation_evaluation_census_sha256=(
            validation_evaluation_census_sha256
            or _digest("validation-evaluation-census")
        ),
        hard_gallery_freeze_binding_sha256=(
            hard_gallery_freeze_binding_sha256
            or _digest("hard-gallery-freeze-binding")
        ),
        validation_hard_gallery_collection_sha256=(
            validation_hard_gallery_collection_sha256
            or _digest("validation-hard-gallery-collection")
        ),
        runtime_assertion_sha256=_digest("runtime"),
        execution_assertion_sha256=_digest("execution"),
        authority=7,
    )


def _sealed_authorization(
    *,
    evaluation_census_sha256: str | None = None,
    hard_gallery_freeze_binding_sha256: str | None = None,
    hard_gallery_collection_sha256: str | None = None,
) -> production.PrivateSealedTestAuthorization:
    return production.PrivateSealedTestAuthorization(
        consumed_at_utc="2026-08-25T15:00:30Z",
        external_grant_sha256=_digest("test-grant"),
        test_manifest_sha256=_digest("sealed-test-manifest"),
        caption_manifest_sha256=_digest("human-holistic-caption-manifest"),
        evaluator_sha256=hashlib.sha256(
            production.current_evaluator_artifact_bytes()
        ).hexdigest(),
        validation_freeze_sha256=_digest("validation-freeze"),
        aggregate_code_sha256=hashlib.sha256(
            production.current_statistics_artifact_bytes()
        ).hexdigest(),
        evaluation_census_sha256=(
            evaluation_census_sha256 or _digest("test-evaluation-census")
        ),
        hard_gallery_freeze_binding_sha256=(
            hard_gallery_freeze_binding_sha256
            or _digest("hard-gallery-freeze-binding")
        ),
        hard_gallery_collection_sha256=(
            hard_gallery_collection_sha256
            or _digest("test-hard-gallery-collection")
        ),
    )


def _score_execution_census_sha256(
    systems: tuple[production.SystemScoreEvidence, ...],
) -> str:
    rows = []
    for system in systems:
        for seed_row in system.seed_scores:
            rows.append(
                {
                    "capacity_receipt_sha256": seed_row.capacity_receipt_sha256,
                    "checkpoint_sha256": seed_row.checkpoint_sha256,
                    "environment_sha256": seed_row.environment_sha256,
                    "qualification_sha256": seed_row.qualification_sha256,
                    "run_manifest_sha256": seed_row.run_manifest_sha256,
                    "scoring_code_sha256": seed_row.scoring_code_sha256,
                    "seed": seed_row.seed,
                    "system_id": system.system_id,
                    "terminal_sha256": seed_row.terminal_sha256,
                    "text_tower_sha256": seed_row.text_tower_sha256,
                }
            )
    raw = (
        json.dumps(
            {"rows": rows, "schema": "phaseset-score-execution-census-v1"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        .encode("ascii")
        + b"\n"
    )
    return hashlib.sha256(raw).hexdigest()


def _system_scores() -> tuple[production.SystemScoreEvidence, ...]:
    motion_commitments = tuple(bytes([index + 1]) * 32 for index in range(4))
    caption_commitments = tuple(bytes([index + 11]) * 32 for index in range(4))
    common = {
        "motion_commitments": motion_commitments,
        "caption_commitments": caption_commitments,
        "positive_motion_indices": tuple((index,) for index in range(4)),
        "group_sizes": np.asarray([3, 3, 4, 4], dtype=np.int64),
        "component_labels": ("C0", "C0", "C1", "C1"),
    }
    scoring_code_sha256 = hashlib.sha256(
        production.current_scoring_artifact_bytes()
    ).hexdigest()
    return tuple(
        production.SystemScoreEvidence(
            system_id=system_id,
            seed_scores=tuple(
                production.SeedScoreEvidence(
                    seed=seed,
                    checkpoint_sha256=_digest(f"checkpoint-{system_id}-{seed}"),
                    run_manifest_sha256=_digest(f"manifest-{system_id}-{seed}"),
                    terminal_sha256=_digest(f"terminal-{system_id}-{seed}"),
                    qualification_sha256=_digest(f"qualification-{seed}"),
                    capacity_receipt_sha256=_digest(f"capacity-{system_id}-{seed}"),
                    environment_sha256=_digest("frozen-environment"),
                    text_tower_sha256=_digest("frozen-clip-text-tower"),
                    scoring_code_sha256=scoring_code_sha256,
                    dataset=evaluation.RetrievalDataset(
                        scores=(
                            np.eye(4, dtype=np.float64)
                            if system_id == "08"
                            else np.roll(
                                np.eye(4, dtype=np.float64),
                                (seed % 3) + 1,
                                axis=0,
                            )
                        ),
                        **common,
                    ),
                )
                for seed in SEEDS
            ),
        )
        for system_id, _ in FINAL_SYSTEMS
    )


def _hard_collection(
    systems: tuple[production.SystemScoreEvidence, ...],
) -> evaluation.HardGalleryCollection:
    dataset = systems[0].seed_scores[0].dataset
    motion_count, caption_count = dataset.scores.shape
    k_pad = int(np.max(dataset.group_sizes))
    actor_mask = np.zeros((motion_count, k_pad), dtype=np.bool_)
    power = np.zeros((motion_count, k_pad, 6), dtype=np.float64)
    for motion_index, group_size in enumerate(dataset.group_sizes):
        actor_mask[motion_index, : int(group_size)] = True
        power[motion_index, : int(group_size)] = 2.0
    checkpoint_census = tuple(
        sorted(
            row.checkpoint_sha256
            for system in systems
            for row in system.seed_scores
        )
    )
    scorer = evaluation.HardNegativeScorerProvenance(
        scorer_id="production-test-independent-text-scorer",
        scorer_code_sha256=_digest("hard-scorer-code"),
        scorer_checkpoint_sha256=_digest("hard-scorer-checkpoint"),
        freeze_receipt_sha256=_digest("hard-freeze-receipt"),
        caption_manifest_sha256=_digest("human-holistic-caption-manifest"),
        motion_manifest_sha256=_digest("prepared"),
        frozen_at_utc="2026-08-25T14:00:00Z",
        evaluated_checkpoint_sha256s=checkpoint_census,
    )
    similarity = np.asarray(
        [
            [0.1, 0.9, 0.4, 0.7],
            [0.8, 0.2, 0.6, 0.3],
            [0.5, 0.7, 0.1, 0.9],
            [0.6, 0.3, 0.8, 0.2],
        ],
        dtype=np.float64,
    )
    assert similarity.shape == (caption_count, motion_count)
    features = evaluation.HardNegativeFeatures(
        duration_seconds=np.full(motion_count, 10.0, dtype=np.float64),
        per_actor_band_power=power,
        actor_mask=actor_mask,
        total_motion_energy=np.full(motion_count, 3.0, dtype=np.float64),
        root_speed=np.full(motion_count, 0.5, dtype=np.float64),
        frozen_text_similarity=similarity,
        scorer_provenance=scorer,
    )
    binding = evaluation.HardGalleryFreezeBinding(
        scorer_provenance_sha256=(
            evaluation.hard_negative_scorer_provenance_sha256(scorer)
        ),
        similarity_sha256=evaluation.hard_negative_similarity_sha256(similarity),
        freeze_receipt_sha256=scorer.freeze_receipt_sha256,
        caption_manifest_sha256=scorer.caption_manifest_sha256,
        motion_manifest_sha256=scorer.motion_manifest_sha256,
        external_authorization_sha256=_digest("hard-gallery-external-authorization"),
    )
    return evaluation.build_group_hard_galleries(
        dataset,
        features,
        evaluation.HardGalleryConfig(
            candidate_count=2,
            duration_absolute_max=0.0,
            band_power_log_linf_max=0.0,
            total_energy_log_max=0.0,
            root_speed_log_max=0.0,
        ),
        binding,
    )


def _corpus_provenance(
    systems: tuple[production.SystemScoreEvidence, ...],
    *,
    split: str,
    sealed_test_consumption_sha256: str | None = None,
) -> production.EvaluationCorpusProvenance:
    reference = systems[0].seed_scores[0].dataset
    authorization = _sealed_authorization()
    return production.EvaluationCorpusProvenance(
        evaluation_type="REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL",
        split=split,
        caption_kind="HUMAN_HOLISTIC",
        source_manifest_sha256=_digest("data"),
        prepared_data_manifest_sha256=_digest("prepared"),
        split_audit_sha256=_digest("split"),
        rights_assertion_sha256=_digest("rights"),
        caption_manifest_sha256=_digest("human-holistic-caption-manifest"),
        test_manifest_sha256=(
            authorization.test_manifest_sha256 if split == "test" else None
        ),
        sealed_test_consumption_sha256=(
            sealed_test_consumption_sha256 if split == "test" else None
        ),
        capture_commitments=evaluation.capture_cluster_commitments(reference),
    )


def _evaluation_evidence(
    systems: tuple[production.SystemScoreEvidence, ...],
    *,
    split: str,
    hard_gallery_collection: evaluation.HardGalleryCollection,
    sealed_test_consumption_sha256: str | None = None,
) -> production.EvaluationExecutionEvidence:
    return production.EvaluationExecutionEvidence(
        provenance=_corpus_provenance(
            systems,
            split=split,
            sealed_test_consumption_sha256=sealed_test_consumption_sha256,
        ),
        system_scores=systems,
        evaluator_artifact=production.current_evaluator_artifact_bytes(),
        hard_gallery_collection=hard_gallery_collection,
    )


class _Backend:
    backend_sha256 = _digest("private-backend-v1")

    def __init__(
        self,
        root: Path,
        *,
        authenticated: bool = True,
        duplicate_content: bool = False,
        directory_artifact: bool = False,
        sealed_authenticated: bool = True,
        outcome: str = "COMPLETED",
    ) -> None:
        self.root = root
        self.authenticated = authenticated
        self.duplicate_content = duplicate_content
        self.directory_artifact = directory_artifact
        self.sealed_authenticated = sealed_authenticated
        self.outcome = outcome
        self.auth_calls = 0
        self.sealed_auth_calls = 0
        self.execute_calls = 0

    def authenticate(
        self,
        request: execution.RuntimeAdmissionRequest,
        receipts: production.PrivateReceiptAssertions,
        adapter_sha256: str,
    ) -> bool:
        assert request.commands == execution.COMMANDS
        assert receipts.authority == 7
        assert len(adapter_sha256) == 64
        self.auth_calls += 1
        return self.authenticated

    def authenticate_sealed_test(
        self,
        intent: execution.CommandIntent,
        authorization: production.PrivateSealedTestAuthorization,
        adapter_sha256: str,
    ) -> bool:
        assert intent.command == "evaluate" and intent.split == "test"
        assert authorization.test_manifest_sha256 == _digest("sealed-test-manifest")
        assert len(adapter_sha256) == 64
        self.sealed_auth_calls += 1
        return self.sealed_authenticated

    def execute(self, intent: execution.CommandIntent) -> production.BackendExecution:
        self.execute_calls += 1
        self.root.mkdir(parents=True, exist_ok=True)
        if self.directory_artifact:
            return production.BackendExecution(
                "COMPLETED",
                "2026-08-25T15:01:00Z",
                (self.root,),
            )
        first = self.root / f"{intent.command}-terminal.json"
        first.write_bytes(b'{"outcome":"COMPLETED"}\n')
        artifacts = (first,)
        if self.duplicate_content:
            second = self.root / f"{intent.command}-copy.json"
            second.write_bytes(first.read_bytes())
            artifacts = (first, second)
        return production.BackendExecution(
            self.outcome,
            "2026-08-25T15:01:00Z",
            artifacts,
        )


def _request(label: str = "same") -> execution.RuntimeAdmissionRequest:
    return execution.RuntimeAdmissionRequest(
        plan_sha256=_digest(f"plan-{label}"),
        matrix_sha256=_digest("matrix"),
        training_config_sha256=_digest("training"),
        handler_manifest_sha256=execution.runtime_handler_manifest_sha256(),
    )


def _base_score_rows() -> tuple[experiments.BaseScore, ...]:
    evaluator_sha256 = hashlib.sha256(
        production.current_base_qualification_artifact_bytes()
    ).hexdigest()
    rows = []
    for run_id in experiments.base_run_ids():
        _, seed, system_id = experiments.parse_run_id(run_id)
        score = {"B0": 5, "B1": 8, "B2": 6}[system_id]
        rows.append(
            experiments.BaseScore(
                run_id=run_id,
                bidirectional_r1_numerator=score,
                bidirectional_r1_denominator=10,
                parameter_count={"B0": 1_000, "B1": 2_000, "B2": 3_000}[
                    system_id
                ],
                frozen_runtime_latency_ns={"B0": 10, "B1": 20, "B2": 30}[
                    system_id
                ],
                terminal_sha256=_digest(f"base-terminal-{system_id}-{seed}"),
                selected_checkpoint_sha256=_digest(
                    f"base-checkpoint-{system_id}-{seed}"
                ),
                split="validation",
                validation_manifest_sha256=_digest("base-validation-manifest"),
                query_census_sha256=_digest("base-validation-query-census"),
                evaluator_sha256=evaluator_sha256,
                score_artifact_sha256=_digest(f"base-score-{system_id}-{seed}"),
            )
        )
    return tuple(rows)


def _base_authorization(
    rows: tuple[experiments.BaseScore, ...],
) -> production.PrivateBaseQualificationAuthorization:
    qualification = experiments.qualify_base(rows)
    return production.PrivateBaseQualificationAuthorization(
        authorized_at_utc="2026-08-25T14:30:00Z",
        external_grant_sha256=_digest("base-qualification-grant"),
        validation_manifest_sha256=qualification.validation_manifest_sha256,
        query_census_sha256=qualification.query_census_sha256,
        evaluator_sha256=qualification.evaluator_sha256,
        score_rows_sha256=qualification.score_rows_sha256,
    )


def _resource_rows(
    systems: tuple[production.SystemScoreEvidence, ...],
) -> tuple[publication.ResourceMeasurement, ...]:
    rows = []
    for system_index, system in enumerate(systems):
        for score in system.seed_scores:
            rows.append(
                publication.ResourceMeasurement(
                    workload=publication.PRIMARY_WORKLOAD,
                    system_id=system.system_id,
                    seed=score.seed,
                    group_size=None,
                    parameter_count=1_000_000 + system_index,
                    forward_flops=10_000_000 + system_index,
                    peak_allocated_bytes=2_000_000 + score.seed,
                    window_count=64,
                    elapsed_ns=1_000_000 + score.seed,
                    checkpoint_sha256=score.checkpoint_sha256,
                    terminal_sha256=score.terminal_sha256,
                    environment_sha256=score.environment_sha256,
                    device_sha256=_digest("resource-device"),
                    profiler_code_sha256=_digest("resource-profiler"),
                    measurement_artifact_sha256=_digest(
                        f"resource-{system.system_id}-{score.seed}"
                    ),
                    precision="FP32",
                    measurement_type=publication.FORMAL_MEASUREMENT_TYPE,
                )
            )
    full = next(
        row
        for row in rows
        if row.system_id == "08" and row.seed == publication.PROFILE_SEED
    )
    for group_size in publication.SCALING_GROUP_SIZES:
        rows.append(
            publication.ResourceMeasurement(
                workload=publication.SCALING_WORKLOAD,
                system_id="08",
                seed=publication.PROFILE_SEED,
                group_size=group_size,
                parameter_count=full.parameter_count,
                forward_flops=10_000_000 * group_size,
                peak_allocated_bytes=2_000_000 * group_size,
                window_count=16,
                elapsed_ns=1_000_000 * group_size,
                checkpoint_sha256=full.checkpoint_sha256,
                terminal_sha256=full.terminal_sha256,
                environment_sha256=full.environment_sha256,
                device_sha256=full.device_sha256,
                profiler_code_sha256=full.profiler_code_sha256,
                measurement_artifact_sha256=_digest(
                    f"resource-scaling-{group_size}"
                ),
                precision=full.precision,
                measurement_type=full.measurement_type,
            )
        )
    return tuple(rows)


def test_private_adapter_runs_through_closed_runner_without_serializing_paths(tmp_path) -> None:
    private_root = tmp_path / "private-output-never-serialized"
    backend = _Backend(private_root)
    adapter = production.ProductionRuntimeAdapter(backend, _receipts())
    runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=adapter,
    )
    admission = runner.preflight()
    assert type(admission) is execution.RuntimeAdmission
    assert admission.runtime_mode == "PRIVATE_AUTHORIZED"
    assert admission.production is True
    assert admission.public_verification_performed is False
    assert backend.auth_calls == 1

    result = runner.execute_command("prepare-data")
    assert result.outcome == "COMPLETED"
    assert result.production is True
    assert result.result_claimed is False
    assert result.artifact_sha256s == (
        hashlib.sha256(b'{"outcome":"COMPLETED"}\n').hexdigest(),
    )
    assert backend.execute_calls == 1
    serialized = (
        execution.canonical_runtime_admission_bytes(admission)
        + execution.canonical_command_result_bytes(result)
    )
    assert str(tmp_path).encode() not in serialized
    assert b"private-output-never-serialized" not in serialized
    assert b"endpoint" not in serialized


def test_admission_is_authenticated_once_and_cannot_be_rebound(tmp_path) -> None:
    backend = _Backend(tmp_path)
    adapter = production.ProductionRuntimeAdapter(backend, _receipts())
    request = _request()
    first = adapter.admit(request)
    assert adapter.admit(request) is first
    assert backend.auth_calls == 1
    with pytest.raises(production.ProductionAdapterError, match="changed"):
        adapter.admit(replace(request, plan_sha256=_digest("different-plan")))


def test_failed_authentication_and_invalid_timestamp_fail_before_dispatch(tmp_path) -> None:
    rejected = production.ProductionRuntimeAdapter(
        _Backend(tmp_path, authenticated=False),
        _receipts(),
    )
    with pytest.raises(production.ProductionAdapterError, match="did not authenticate"):
        rejected.admit(_request())

    invalid_time = production.ProductionRuntimeAdapter(
        _Backend(tmp_path),
        _receipts(admitted_at="2026-08-25T15:00:00+00:00"),
    )
    with pytest.raises(execution.ExecutionContractError, match="exact second-resolution UTC"):
        invalid_time.admit(_request("invalid-time"))


def test_backend_artifacts_must_be_nonempty_regular_and_content_unique(tmp_path) -> None:
    directory_adapter = production.ProductionRuntimeAdapter(
        _Backend(tmp_path / "directory", directory_artifact=True),
        _receipts(),
    )
    directory_runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=directory_adapter,
    )
    directory_runner.preflight()
    with pytest.raises(production.ProductionAdapterError, match="regular file"):
        directory_runner.execute_command("prepare-data")

    duplicate_adapter = production.ProductionRuntimeAdapter(
        _Backend(tmp_path / "duplicate", duplicate_content=True),
        _receipts(),
    )
    duplicate_runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=duplicate_adapter,
    )
    duplicate_runner.preflight()
    with pytest.raises(production.ProductionAdapterError, match="unique content"):
        duplicate_runner.execute_command("prepare-data")


def test_adapter_identity_binds_backend_and_closed_command_manifest(tmp_path) -> None:
    first = production.ProductionRuntimeAdapter(_Backend(tmp_path / "a"), _receipts())
    second = production.ProductionRuntimeAdapter(_Backend(tmp_path / "b"), _receipts())
    assert first.adapter_sha256 == second.adapter_sha256
    assert first.commands == execution.COMMANDS
    assert first.handler_manifest_sha256 == execution.runtime_handler_manifest_sha256()

    class _ChangedBackend(_Backend):
        backend_sha256 = _digest("changed-private-backend")

    changed = production.ProductionRuntimeAdapter(_ChangedBackend(tmp_path / "c"), _receipts())
    assert changed.adapter_sha256 != first.adapter_sha256


def test_completed_base_qualification_is_recomputed_and_externally_authorized(
    tmp_path: Path,
) -> None:
    rows = _base_score_rows()
    authorization = _base_authorization(rows)

    class _QualificationBackend(_Backend):
        backend_sha256 = _digest("base-qualification-backend")

        def __init__(self, root: Path, *, authenticated: bool = True) -> None:
            super().__init__(root)
            self.qualification_authenticated = authenticated
            self.qualification_auth_calls = 0

        def authenticate_base_qualification(
            self,
            intent: execution.CommandIntent,
            supplied_authorization: production.PrivateBaseQualificationAuthorization,
            qualification_sha256: str,
            adapter_sha256: str,
        ) -> bool:
            assert intent.command == "qualify-base"
            assert supplied_authorization == authorization
            assert len(qualification_sha256) == len(adapter_sha256) == 64
            self.qualification_auth_calls += 1
            return self.qualification_authenticated

        def execute(self, intent: execution.CommandIntent) -> production.BackendExecution:
            self.execute_calls += 1
            evidence = production.BaseQualificationExecutionEvidence(
                score_rows=rows,
                selector_artifact=(
                    production.current_base_qualification_artifact_bytes()
                ),
            )
            raw = production.build_base_qualification_execution_artifact(evidence)
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / "base-qualification.json"
            path.write_bytes(raw)
            return production.BackendExecution(
                outcome="COMPLETED",
                completed_at_utc="2026-08-25T15:01:00Z",
                artifacts=(path,),
                semantic_evidence=evidence,
            )

    backend = _QualificationBackend(tmp_path / "qualified")
    runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=production.ProductionRuntimeAdapter(
            backend,
            _receipts(),
            base_qualification_authorization=authorization,
        ),
    )
    result = runner.execute_command("qualify-base")
    assert result.outcome == "COMPLETED"
    assert backend.qualification_auth_calls == 1

    missing = _QualificationBackend(tmp_path / "missing")
    missing_runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=production.ProductionRuntimeAdapter(missing, _receipts()),
    )
    with pytest.raises(
        production.ProductionAdapterError,
        match="requires external authorization",
    ):
        missing_runner.execute_command("qualify-base")

    rejected = _QualificationBackend(tmp_path / "rejected", authenticated=False)
    rejected_runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=production.ProductionRuntimeAdapter(
            rejected,
            _receipts(),
            base_qualification_authorization=authorization,
        ),
    )
    with pytest.raises(
        production.ProductionAdapterError,
        match="did not authenticate base qualification",
    ):
        rejected_runner.execute_command("qualify-base")


def test_sealed_test_is_authenticated_consumed_once_and_bound_to_result(tmp_path) -> None:
    ledger = tmp_path / "sealed-ledger"
    ledger.mkdir()
    backend = _Backend(tmp_path / "artifacts", outcome="HELD")
    adapter = production.ProductionRuntimeAdapter(
        backend,
        _receipts(),
        sealed_test_ledger_root=ledger,
        sealed_test_authorization=_sealed_authorization(),
    )
    runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=adapter,
    )
    result = runner.execute_command("evaluate", split="test")
    assert result.sealed_test_consumption_sha256 is not None
    assert len(result.sealed_test_consumption_sha256) == 64
    assert (ledger / "sealed-test-consumption.json").is_file()
    assert backend.sealed_auth_calls == 1
    assert backend.execute_calls == 1

    with pytest.raises(production.ProductionAdapterError, match="already consumed"):
        runner.execute_command("evaluate", split="test")
    assert backend.sealed_auth_calls == 1
    assert backend.execute_calls == 1

    second_backend = _Backend(tmp_path / "second-artifacts", outcome="HELD")
    second = production.ProductionRuntimeAdapter(
        second_backend,
        _receipts(),
        sealed_test_ledger_root=ledger,
        sealed_test_authorization=_sealed_authorization(),
    )
    second_runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=second,
    )
    with pytest.raises(production.ProductionAdapterError, match="already consumed"):
        second_runner.execute_command("evaluate", split="test")
    assert second_backend.sealed_auth_calls == 0
    assert second_backend.execute_calls == 0


def test_sealed_test_fails_before_dispatch_without_authenticated_gate(tmp_path) -> None:
    missing_backend = _Backend(tmp_path / "missing")
    missing = production.ProductionRuntimeAdapter(missing_backend, _receipts())
    missing_runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=missing,
    )
    with pytest.raises(production.ProductionAdapterError, match="grant.*absent"):
        missing_runner.execute_command("evaluate", split="test")
    assert missing_backend.execute_calls == 0

    ledger = tmp_path / "rejected-ledger"
    ledger.mkdir()
    rejected_backend = _Backend(
        tmp_path / "rejected",
        sealed_authenticated=False,
    )
    rejected = production.ProductionRuntimeAdapter(
        rejected_backend,
        _receipts(),
        sealed_test_ledger_root=ledger,
        sealed_test_authorization=_sealed_authorization(),
    )
    rejected_runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=rejected,
    )
    with pytest.raises(production.ProductionAdapterError, match="did not authenticate"):
        rejected_runner.execute_command("evaluate", split="test")
    assert not (ledger / "sealed-test-consumption.json").exists()
    assert rejected_backend.execute_calls == 0


def test_completed_evaluation_and_bootstrap_are_semantically_recomputed(
    tmp_path: Path,
) -> None:
    systems = _system_scores()
    score_execution_census_sha256 = _score_execution_census_sha256(systems)
    hard_collection = _hard_collection(systems)
    reference_dataset = systems[0].seed_scores[0].dataset
    evaluation_census_sha256 = evaluation.retrieval_census_sha256(reference_dataset)
    hard_freeze_binding_sha256 = evaluation.hard_gallery_freeze_binding_sha256(
        hard_collection.freeze_binding
    )
    hard_collection_sha256 = evaluation.hard_gallery_collection_sha256(
        reference_dataset,
        hard_collection,
    )
    sealed_authorization = _sealed_authorization(
        evaluation_census_sha256=evaluation_census_sha256,
        hard_gallery_freeze_binding_sha256=hard_freeze_binding_sha256,
        hard_gallery_collection_sha256=hard_collection_sha256,
    )
    ledger = tmp_path / "semantic-sealed-ledger"
    ledger.mkdir()

    class _SemanticBackend(_Backend):
        backend_sha256 = _digest("semantic-production-backend-v1")

        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.test_consumption_sha256: str | None = None
            self.test_evidence: production.EvaluationExecutionEvidence | None = None
            self.test_aggregate: bytes | None = None

        def execute(self, intent: execution.CommandIntent) -> production.BackendExecution:
            self.execute_calls += 1
            self.root.mkdir(parents=True, exist_ok=True)
            if intent.command == "evaluate":
                assert intent.split == "test"
                assert intent.sealed_test_consumption_sha256 is not None
                self.test_consumption_sha256 = intent.sealed_test_consumption_sha256
                self.test_evidence = _evaluation_evidence(
                    systems,
                    split="test",
                    hard_gallery_collection=hard_collection,
                    sealed_test_consumption_sha256=(
                        intent.sealed_test_consumption_sha256
                    ),
                )
                self.test_aggregate = production.build_evaluation_execution_artifact(
                    self.test_evidence
                )
                evidence: object = self.test_evidence
                raw = self.test_aggregate
                name = "evaluation-aggregate.json"
            elif intent.command == "bootstrap":
                assert self.test_consumption_sha256 is not None
                assert self.test_evidence is not None
                assert self.test_aggregate is not None
                binding_raw = (ledger / "sealed-test-evaluation.json").read_bytes()
                evidence = production.BootstrapExecutionEvidence(
                    evaluation_evidence=self.test_evidence,
                    evaluation_aggregate_artifact=self.test_aggregate,
                    seed=20260825,
                    sealed_test_consumption_sha256=self.test_consumption_sha256,
                    sealed_test_evaluation_binding_sha256=hashlib.sha256(
                        binding_raw
                    ).hexdigest(),
                )
                raw = production.build_bootstrap_execution_artifact(evidence)
                name = "statistical-report.json"
            else:  # pragma: no cover - closed test backend census
                raise AssertionError("unexpected semantic command")
            path = self.root / name
            path.write_bytes(raw)
            return production.BackendExecution(
                outcome="COMPLETED",
                completed_at_utc="2026-08-25T15:01:00Z",
                artifacts=(path,),
                semantic_evidence=evidence,
            )

    backend = _SemanticBackend(tmp_path / "semantic-artifacts")
    runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=production.ProductionRuntimeAdapter(
            backend,
            _receipts(
                score_execution_census_sha256=score_execution_census_sha256,
                hard_gallery_freeze_binding_sha256=hard_freeze_binding_sha256,
            ),
            sealed_test_ledger_root=ledger,
            sealed_test_authorization=sealed_authorization,
        ),
    )
    evaluation_result = runner.execute_command("evaluate", split="test")
    bootstrap_result = runner.execute_command("bootstrap")
    assert evaluation_result.outcome == bootstrap_result.outcome == "COMPLETED"
    assert backend.execute_calls == 2

    changed = list(systems)
    changed_seed_rows = list(changed[0].seed_scores)
    changed_dataset = changed_seed_rows[0].dataset
    changed_seed_rows[0] = replace(
        changed_seed_rows[0],
        dataset=evaluation.RetrievalDataset(
            scores=changed_dataset.scores,
            motion_commitments=changed_dataset.motion_commitments,
            caption_commitments=tuple(reversed(changed_dataset.caption_commitments)),
            positive_motion_indices=changed_dataset.positive_motion_indices,
            group_sizes=changed_dataset.group_sizes,
            component_labels=changed_dataset.component_labels,
        ),
    )
    changed[0] = replace(changed[0], seed_scores=tuple(changed_seed_rows))
    with pytest.raises(production.ProductionAdapterError, match="query census|invalid"):
        production.build_evaluation_execution_artifact(
            _evaluation_evidence(
                tuple(changed),
                split="validation",
                hard_gallery_collection=hard_collection,
            )
        )


def test_completed_publication_render_is_sealed_content_bound_and_authorized(
    tmp_path: Path,
) -> None:
    systems = _system_scores()
    score_execution_census_sha256 = _score_execution_census_sha256(systems)
    hard_collection = _hard_collection(systems)
    reference = systems[0].seed_scores[0].dataset
    evaluation_census_sha256 = evaluation.retrieval_census_sha256(reference)
    hard_freeze_sha256 = evaluation.hard_gallery_freeze_binding_sha256(
        hard_collection.freeze_binding
    )
    hard_collection_sha256 = evaluation.hard_gallery_collection_sha256(
        reference,
        hard_collection,
    )
    sealed = _sealed_authorization(
        evaluation_census_sha256=evaluation_census_sha256,
        hard_gallery_freeze_binding_sha256=hard_freeze_sha256,
        hard_gallery_collection_sha256=hard_collection_sha256,
    )

    pre_root = tmp_path / "precomputed-ledger"
    pre_root.mkdir()
    pre_gate = execution.SealedTestGate(pre_root)
    consumption_sha256 = pre_gate.consume(
        external_grant_sha256=sealed.external_grant_sha256,
        test_manifest_sha256=sealed.test_manifest_sha256,
        caption_manifest_sha256=sealed.caption_manifest_sha256,
        evaluator_sha256=sealed.evaluator_sha256,
        validation_freeze_sha256=sealed.validation_freeze_sha256,
        aggregate_code_sha256=sealed.aggregate_code_sha256,
        evaluation_census_sha256=sealed.evaluation_census_sha256,
        hard_gallery_freeze_binding_sha256=(
            sealed.hard_gallery_freeze_binding_sha256
        ),
        hard_gallery_collection_sha256=sealed.hard_gallery_collection_sha256,
        consumed_at_utc=sealed.consumed_at_utc,
    )
    evaluation_evidence = _evaluation_evidence(
        systems,
        split="test",
        hard_gallery_collection=hard_collection,
        sealed_test_consumption_sha256=consumption_sha256,
    )
    evaluation_raw = production.build_evaluation_execution_artifact(
        evaluation_evidence
    )
    completed_at = "2026-08-25T15:01:00Z"
    binding_sha256 = pre_gate.bind_evaluation(
        consumption_sha256=consumption_sha256,
        evaluation_aggregate_sha256=hashlib.sha256(evaluation_raw).hexdigest(),
        evaluator_sha256=sealed.evaluator_sha256,
        aggregate_code_sha256=sealed.aggregate_code_sha256,
        completed_at_utc=completed_at,
    )
    bootstrap_evidence = production.BootstrapExecutionEvidence(
        evaluation_evidence=evaluation_evidence,
        evaluation_aggregate_artifact=evaluation_raw,
        seed=20260825,
        sealed_test_consumption_sha256=consumption_sha256,
        sealed_test_evaluation_binding_sha256=binding_sha256,
    )
    publication_evidence = production.PublicationExecutionEvidence(
        bootstrap_evidence=bootstrap_evidence,
        resource_measurements=_resource_rows(systems),
        renderer_artifact=production.current_publication_renderer_artifact_bytes(),
    )
    publication_contents = production.build_publication_execution_artifacts(
        publication_evidence
    )
    publication_manifest = json.loads(publication_contents[-1])
    publication_authorization = production.PrivatePublicationAuthorization(
        authorized_at_utc="2026-08-25T15:02:00Z",
        external_grant_sha256=_digest("publication-grant"),
        evaluation_aggregate_sha256=publication_manifest[
            "evaluation_aggregate_sha256"
        ],
        statistics_sha256=publication_manifest["statistics_sha256"],
        resource_report_sha256=publication_manifest["resource_report_sha256"],
        claim_report_sha256=publication_manifest["claim_report_sha256"],
        renderer_code_sha256=hashlib.sha256(
            production.current_publication_renderer_artifact_bytes()
        ).hexdigest(),
        publication_manifest_sha256=hashlib.sha256(
            publication_contents[-1]
        ).hexdigest(),
    )

    class _PublicationBackend(_Backend):
        backend_sha256 = _digest("publication-production-backend")

        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.publication_auth_calls = 0

        def authenticate_publication(
            self,
            intent: execution.CommandIntent,
            supplied: production.PrivatePublicationAuthorization,
            publication_manifest_sha256: str,
            adapter_sha256: str,
        ) -> bool:
            assert intent.command == "render-paper"
            assert supplied == publication_authorization
            assert publication_manifest_sha256 == supplied.publication_manifest_sha256
            assert len(adapter_sha256) == 64
            self.publication_auth_calls += 1
            return True

        def execute(self, intent: execution.CommandIntent) -> production.BackendExecution:
            self.execute_calls += 1
            self.root.mkdir(parents=True, exist_ok=True)
            if intent.command == "evaluate":
                path = self.root / "evaluation-aggregate.json"
                path.write_bytes(evaluation_raw)
                return production.BackendExecution(
                    outcome="COMPLETED",
                    completed_at_utc=completed_at,
                    artifacts=(path,),
                    semantic_evidence=evaluation_evidence,
                )
            assert intent.command == "render-paper"
            paths = []
            for index, raw in enumerate(publication_contents):
                path = self.root / f"publication-{index:02d}.bin"
                path.write_bytes(raw)
                paths.append(path)
            return production.BackendExecution(
                outcome="COMPLETED",
                completed_at_utc="2026-08-25T15:03:00Z",
                artifacts=tuple(paths),
                semantic_evidence=publication_evidence,
            )

    live_root = tmp_path / "live-ledger"
    live_root.mkdir()
    backend = _PublicationBackend(tmp_path / "publication-artifacts")
    runner = execution.DataFreeRunner(
        execution.canonical_public_training_config_bytes(),
        runtime_adapter=production.ProductionRuntimeAdapter(
            backend,
            _receipts(
                score_execution_census_sha256=score_execution_census_sha256,
                hard_gallery_freeze_binding_sha256=hard_freeze_sha256,
            ),
            sealed_test_ledger_root=live_root,
            sealed_test_authorization=sealed,
            publication_authorization=publication_authorization,
        ),
    )
    runner.execute_command("evaluate", split="test")
    result = runner.execute_command("render-paper")
    assert result.outcome == "COMPLETED"
    assert len(result.artifact_sha256s) == len(publication_contents)
    assert backend.publication_auth_calls == 1


def test_annotation_introspection_does_not_inject_a_semantic_global():
    import inspect

    for module in (evaluation, statistics, experiments, publication):
        before = set(vars(module))
        inspect.get_annotations(module)
        assert set(vars(module)) == before
    production._assert_semantic_function_integrity()


def test_semantic_helper_monkeypatch_is_rejected_before_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    systems = _system_scores()
    collection = _hard_collection(systems)
    evidence = _evaluation_evidence(
        systems,
        split="validation",
        hard_gallery_collection=collection,
    )

    with monkeypatch.context() as evaluator_patch:
        evaluator_patch.setattr(
            evaluation,
            "_evaluate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("forged evaluator helper executed")
            ),
        )
        with pytest.raises(
            production.ProductionAdapterError,
            match="semantic function identity changed.*_evaluate",
        ):
            production.build_evaluation_execution_artifact(evidence)

    with monkeypatch.context() as statistics_patch:
        statistics_patch.setattr(
            statistics,
            "paired_seed_blocked_bootstrap",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("forged bootstrap helper executed")
            ),
        )
        with pytest.raises(
            production.ProductionAdapterError,
            match="semantic function identity changed.*paired_seed_blocked_bootstrap",
        ):
            production.build_evaluation_execution_artifact(evidence)

    class _NumpyProxy:
        def __getattr__(self, name: str) -> object:
            if name == "mean":
                return lambda *_args, **_kwargs: 0.0
            return getattr(np, name)

    with monkeypatch.context() as binding_patch:
        binding_patch.setattr(evaluation, "np", _NumpyProxy())
        with pytest.raises(
            production.ProductionAdapterError,
            match="semantic function identity changed before evidence computation: np",
        ):
            production.build_evaluation_execution_artifact(evidence)

    with monkeypatch.context() as builtin_shadow_patch:
        builtin_shadow_patch.setattr(
            evaluation,
            "float",
            lambda *_args, **_kwargs: 0.0,
            raising=False,
        )
        with pytest.raises(
            production.ProductionAdapterError,
            match="semantic module global-name census changed",
        ):
            production.build_evaluation_execution_artifact(evidence)

    with monkeypatch.context() as production_builtin_shadow_patch:
        production_builtin_shadow_patch.setattr(
            production,
            "float",
            lambda *_args, **_kwargs: 0.0,
            raising=False,
        )
        with pytest.raises(
            production.ProductionAdapterError,
            match="production semantic global-name census changed",
        ):
            production.build_evaluation_execution_artifact(evidence)

    with monkeypatch.context() as numpy_dtype_patch:
        numpy_dtype_patch.setattr(np, "float64", np.float32)
        with pytest.raises(
            production.ProductionAdapterError,
            match="semantic function identity changed.*float64",
        ):
            production.build_evaluation_execution_artifact(evidence)

    class _RandomProxy:
        Generator = np.random.Generator
        PCG64 = np.random.PCG64

    with monkeypatch.context() as numpy_random_patch:
        numpy_random_patch.setattr(np, "random", _RandomProxy())
        with pytest.raises(
            production.ProductionAdapterError,
            match="semantic function identity changed.*random",
        ):
            production.build_evaluation_execution_artifact(evidence)
