from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from phaseset_core.evaluation import (
    EvaluationContractError,
    HardGalleryFreezeBinding,
    HardGalleryConfig,
    HardNegativeFeatures,
    HardNegativeScorerProvenance,
    RetrievalDataset,
    assert_count_only_non_discriminative,
    build_group_hard_galleries,
    count_only_chance_recall_at_1,
    count_only_scores,
    evaluate_hard_text_to_motion,
    evaluate_retrieval,
    hard_gallery_collection_sha256,
    hard_negative_scorer_provenance_sha256,
    hard_negative_similarity_sha256,
)


def _commitment(domain: str, index: int) -> bytes:
    return hashlib.sha256(f"{domain}-{index}".encode()).digest()


def _dataset(count: int = 40, *, k3_count: int | None = None) -> RetrievalDataset:
    scores = np.full((count, count), -1.0, dtype=np.float64)
    np.fill_diagonal(scores, 1.0)
    # Deterministic off-diagonal variation prevents accidental tie-only tests.
    scores += np.arange(count * count, dtype=np.float64).reshape(count, count) * 1e-9
    if k3_count is None:
        k3_count = count // 2
    group_sizes = np.asarray([3] * k3_count + [4] * (count - k3_count), dtype=np.int64)
    components = tuple("C0" if index % 3 == 0 else "C1" for index in range(count))
    return RetrievalDataset(
        scores=np.ascontiguousarray(scores),
        motion_commitments=tuple(_commitment("motion", index) for index in range(count)),
        caption_commitments=tuple(_commitment("caption", index) for index in range(count)),
        positive_motion_indices=tuple((index,) for index in range(count)),
        group_sizes=group_sizes,
        component_labels=components,
    )


def _features(dataset: RetrievalDataset) -> HardNegativeFeatures:
    motion_count, caption_count = dataset.scores.shape
    k_pad = int(np.max(dataset.group_sizes))
    actor_mask = np.zeros((motion_count, k_pad), dtype=np.bool_)
    power = np.zeros((motion_count, k_pad, 6), dtype=np.float64)
    for motion_index, group_size in enumerate(dataset.group_sizes):
        actor_mask[motion_index, : int(group_size)] = True
        power[motion_index, : int(group_size)] = 2.0
    frozen_similarity = np.fromfunction(
        lambda caption, motion: (
            (caption * 17.0 + motion * 31.0 + caption * motion * 7.0) % 101.0
        ),
        (caption_count, motion_count),
        dtype=np.float64,
    )
    evaluated_checkpoints = tuple(
        sorted(
            hashlib.sha256(f"evaluated-{index}".encode("ascii")).hexdigest()
            for index in range(27)
        )
    )
    return HardNegativeFeatures(
        duration_seconds=np.full(motion_count, 10.0, dtype=np.float64),
        per_actor_band_power=np.ascontiguousarray(power),
        actor_mask=np.ascontiguousarray(actor_mask),
        total_motion_energy=np.full(motion_count, 3.0, dtype=np.float64),
        root_speed=np.full(motion_count, 0.5, dtype=np.float64),
        frozen_text_similarity=np.ascontiguousarray(frozen_similarity),
        scorer_provenance=HardNegativeScorerProvenance(
            scorer_id="frozen-independent-text-scorer-v1",
            scorer_code_sha256=hashlib.sha256(b"hard-scorer-code").hexdigest(),
            scorer_checkpoint_sha256=hashlib.sha256(
                b"independent-hard-scorer-checkpoint"
            ).hexdigest(),
            freeze_receipt_sha256=hashlib.sha256(b"hard-freeze-receipt").hexdigest(),
            caption_manifest_sha256=hashlib.sha256(b"hard-caption-manifest").hexdigest(),
            motion_manifest_sha256=hashlib.sha256(b"hard-motion-manifest").hexdigest(),
            frozen_at_utc="2026-08-25T14:00:00Z",
            evaluated_checkpoint_sha256s=evaluated_checkpoints,
        ),
    )


def _hard_config() -> HardGalleryConfig:
    return HardGalleryConfig(
        candidate_count=32,
        duration_absolute_max=0.0,
        band_power_log_linf_max=0.0,
        total_energy_log_max=0.0,
        root_speed_log_max=0.0,
    )


def _freeze_binding(features: HardNegativeFeatures) -> HardGalleryFreezeBinding:
    return HardGalleryFreezeBinding(
        scorer_provenance_sha256=hard_negative_scorer_provenance_sha256(
            features.scorer_provenance
        ),
        similarity_sha256=hard_negative_similarity_sha256(
            features.frozen_text_similarity
        ),
        freeze_receipt_sha256=features.scorer_provenance.freeze_receipt_sha256,
        caption_manifest_sha256=features.scorer_provenance.caption_manifest_sha256,
        motion_manifest_sha256=features.scorer_provenance.motion_manifest_sha256,
        external_authorization_sha256=hashlib.sha256(
            b"externally-authenticated-hard-gallery-freeze"
        ).hexdigest(),
    )


def test_full_variable_positive_k_macro_and_component_metrics() -> None:
    dataset = _dataset()
    report = evaluate_retrieval(dataset)
    assert report.full_gallery.primary == 1.0
    assert report.full_gallery.text_to_motion.recall_at_10 == 1.0
    assert report.full_gallery.motion_to_text.median_rank == 1.0
    assert tuple(group_size for group_size, _ in report.by_group_size) == (3, 4)
    assert report.macro_group_size.primary == 1.0
    assert tuple(label for label, _ in report.leave_one_component_out) == ("C0", "C1")
    assert len(report.dataset_sha256) == 64

    # Motion 0 has two positive captions, while caption 3 has two positive motions.
    scores = dataset.scores[:-1, :-1].copy()
    scores[0, 3] = 2.0
    scores[3, 3] = 2.0
    positives = ((0,), (1,), (2,), (0, 3), *tuple((index,) for index in range(4, 39)))
    variable = RetrievalDataset(
        scores=np.ascontiguousarray(scores),
        motion_commitments=dataset.motion_commitments[:-1],
        caption_commitments=dataset.caption_commitments[:-1],
        positive_motion_indices=positives,
        group_sizes=np.ascontiguousarray(dataset.group_sizes[:-1]),
        component_labels=dataset.component_labels[:-1],
    )
    variable_report = evaluate_retrieval(variable)
    assert variable_report.full_gallery.text_to_motion.query_count == 39
    assert variable_report.full_gallery.motion_to_text.query_count == 39
    assert variable_report.full_gallery.text_to_motion.aggregation_unit_count == 38
    assert variable_report.full_gallery.motion_to_text.aggregation_unit_count == 38


def test_primary_is_bidirectional_capture_macro_not_query_micro() -> None:
    motion_count = 8
    caption_count = 11
    scores = np.full((motion_count, caption_count), -10.0, dtype=np.float64)
    positives = ((0,), (0,), (0,), (0,), *((index,) for index in range(1, 8)))

    scores[0, :4] = 5.0
    scores[1, 0] = 6.0
    scores[1, 3] = 6.0
    scores[2, 1] = 6.0
    scores[3, 2] = 6.0
    for motion_index, caption_index in enumerate(range(4, 11), start=1):
        scores[motion_index, caption_index] = 7.0

    dataset = RetrievalDataset(
        scores=np.ascontiguousarray(scores),
        motion_commitments=tuple(
            _commitment("macro-motion", index) for index in range(motion_count)
        ),
        caption_commitments=tuple(
            _commitment("macro-caption", index) for index in range(caption_count)
        ),
        positive_motion_indices=positives,
        group_sizes=np.asarray([3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64),
        component_labels=("C0", "C0", "C0", "C0", "C1", "C1", "C1", "C1"),
    )
    report = evaluate_retrieval(dataset)
    assert report.full_gallery.text_to_motion.query_count == 11
    assert report.full_gallery.text_to_motion.aggregation_unit_count == 8
    assert report.full_gallery.text_to_motion.recall_at_1 == pytest.approx(7 / 8)
    assert report.full_gallery.motion_to_text.recall_at_1 == 1.0
    assert report.full_gallery.primary == pytest.approx(15 / 16)
    assert report.full_gallery.text_to_motion.recall_at_1 != pytest.approx(7 / 11)
    contribution_primary = np.mean(
        [
            0.5
            * (
                np.mean(row.text_to_motion_hits)
                + np.mean(row.motion_to_text_hits)
            )
            for row in report.capture_contributions
        ]
    )
    assert contribution_primary == pytest.approx(report.full_gallery.primary)


def test_sealed_parent_census_geometry_uses_40_motion_and_200_human_rows() -> None:
    """Synthetic scores lock the actual parent/query geometry, not test outcomes."""

    parent_count, rows_per_parent = 40, 5
    scores = np.full((parent_count, parent_count * rows_per_parent), -10.0)
    for parent in range(parent_count):
        scores[parent, parent * rows_per_parent : (parent + 1) * rows_per_parent] = 10.0
    # One C09 caption retrieves another parent; that parent also retrieves this
    # nonpositive caption ahead of its five positive rows.
    scores[1, 0] = 11.0
    dataset = RetrievalDataset(
        scores=np.ascontiguousarray(scores, dtype=np.float64),
        motion_commitments=tuple(
            _commitment("sealed-geometry-motion", parent) for parent in range(parent_count)
        ),
        caption_commitments=tuple(
            _commitment("sealed-geometry-caption", row)
            for row in range(parent_count * rows_per_parent)
        ),
        positive_motion_indices=tuple(
            (row // rows_per_parent,) for row in range(parent_count * rows_per_parent)
        ),
        group_sizes=np.asarray([4] * 24 + [3] * 16, dtype=np.int64),
        component_labels=("C09",) * 16 + ("C11",) * 8 + ("C15",) * 16,
    )
    report = evaluate_retrieval(dataset)
    assert report.full_gallery.text_to_motion.query_count == 200
    assert report.full_gallery.motion_to_text.query_count == 40
    assert report.full_gallery.text_to_motion.aggregation_unit_count == 40
    assert report.full_gallery.motion_to_text.aggregation_unit_count == 40
    assert len(report.capture_contributions) == 40
    assert report.full_gallery.text_to_motion.recall_at_1 == pytest.approx(39.8 / 40)
    assert report.full_gallery.motion_to_text.recall_at_1 == pytest.approx(39 / 40)
    assert report.full_gallery.primary == pytest.approx(0.985)


def test_variable_positive_ties_use_commitments_in_both_directions() -> None:
    scores = np.zeros((4, 4), dtype=np.float64)
    positives = ((0, 1), (0,), (2, 3), (2,))
    dataset = RetrievalDataset(
        scores=scores,
        motion_commitments=tuple(bytes([index + 1]) * 32 for index in range(4)),
        caption_commitments=tuple(bytes([index + 1]) * 32 for index in range(4)),
        positive_motion_indices=positives,
        group_sizes=np.full(4, 3, dtype=np.int64),
        component_labels=("C0", "C0", "C1", "C1"),
    )
    report = evaluate_retrieval(dataset)
    assert report.full_gallery.text_to_motion.recall_at_1 == 0.5
    assert report.full_gallery.motion_to_text.recall_at_1 == 0.5
    assert report.full_gallery.text_to_motion.aggregation_unit_count == 2
    assert report.full_gallery.motion_to_text.aggregation_unit_count == 2
    assert report.full_gallery.text_to_motion.median_rank == 2.0
    assert report.full_gallery.motion_to_text.median_rank == 2.0

    # Target table order changes, but commitment-based tie resolution does not.
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    inverse = {int(old): new for new, old in enumerate(permutation)}
    permuted = RetrievalDataset(
        scores=np.ascontiguousarray(scores[np.ix_(permutation, permutation)]),
        motion_commitments=tuple(dataset.motion_commitments[index] for index in permutation),
        caption_commitments=tuple(dataset.caption_commitments[index] for index in permutation),
        positive_motion_indices=tuple(
            tuple(sorted(inverse[index] for index in positives[old_caption]))
            for old_caption in permutation
        ),
        group_sizes=np.ascontiguousarray(dataset.group_sizes[permutation]),
        component_labels=tuple(dataset.component_labels[index] for index in permutation),
    )
    permuted_report = evaluate_retrieval(permuted)
    assert permuted_report.full_gallery == report.full_gallery


def test_group_hard32_caps_at_native_k_census_and_count_only_is_chance() -> None:
    # The frozen test census is 27 native K=3 and 49 native K=4 captures.
    dataset = _dataset(76, k3_count=27)
    features = _features(dataset)
    collection = build_group_hard_galleries(
        dataset,
        features,
        _hard_config(),
        _freeze_binding(features),
    )
    galleries = collection.galleries
    assert len(galleries) == 76
    for caption_index, gallery in enumerate(galleries):
        expected_size = 27 if int(dataset.group_sizes[caption_index]) == 3 else 32
        assert len(gallery.motion_indices) == expected_size
        assert gallery.maximum_candidate_count == 32
        sizes = dataset.group_sizes[list(gallery.motion_indices)]
        assert np.all(sizes == dataset.group_sizes[caption_index])

    # Matching calipers are applied first; frozen text difficulty then picks the
    # 31 hardest K=4 negatives (ties end in commitment order).
    caption_index = 27
    expected_negatives = sorted(
        (
            (
                -float(features.frozen_text_similarity[caption_index, motion_index]),
                dataset.motion_commitments[motion_index],
                motion_index,
            )
            for motion_index in range(27, 76)
            if motion_index != caption_index
        )
    )[:31]
    expected_set = {caption_index, *(row[2] for row in expected_negatives)}
    assert set(galleries[caption_index].motion_indices) == expected_set

    assert evaluate_hard_text_to_motion(dataset, collection).recall_at_1 == 1.0
    assert_count_only_non_discriminative(dataset, collection)
    expected_chance = (27 * (1.0 / 27.0) + 49 * (1.0 / 32.0)) / 76.0
    assert count_only_chance_recall_at_1(dataset, collection) == pytest.approx(
        expected_chance
    )
    scores = count_only_scores(dataset)
    assert scores.dtype == np.float64 and not scores.flags.writeable
    assert len(hard_gallery_collection_sha256(dataset, collection)) == 64


def test_hard_gallery_feature_matching_is_actor_permutation_invariant() -> None:
    dataset = _dataset(12, k3_count=6)
    features = _features(dataset)
    power = features.per_actor_band_power.copy()
    for motion_index, group_size in enumerate(dataset.group_sizes):
        for actor_index in range(int(group_size)):
            power[motion_index, actor_index] = actor_index + np.arange(6) / 10.0
    features = replace(features, per_actor_band_power=np.ascontiguousarray(power))
    permuted = power.copy()
    for motion_index, group_size in enumerate(dataset.group_sizes):
        permuted[motion_index, : int(group_size)] = permuted[motion_index, : int(group_size)][::-1]
    permuted_features = replace(
        features,
        per_actor_band_power=np.ascontiguousarray(permuted),
    )
    config = replace(_hard_config(), candidate_count=6)
    first = build_group_hard_galleries(
        dataset,
        features,
        config,
        _freeze_binding(features),
    )
    second = build_group_hard_galleries(
        dataset,
        permuted_features,
        config,
        _freeze_binding(permuted_features),
    )
    assert first == second


def test_hard_gallery_fails_only_when_no_caliper_matched_negative() -> None:
    dataset = _dataset(40)
    features = _features(dataset)
    unique_duration = np.arange(40, dtype=np.float64)
    features = replace(features, duration_seconds=unique_duration)
    with pytest.raises(EvaluationContractError, match="NO_MATCHED_NEGATIVE"):
        build_group_hard_galleries(
            dataset,
            features,
            _hard_config(),
            _freeze_binding(features),
        )


def test_hard_gallery_rejects_circular_scorer_and_binds_frozen_provenance() -> None:
    dataset = _dataset(40)
    features = _features(dataset)
    affine_self_scores = np.ascontiguousarray(2.0 * dataset.scores.T + 1.0)
    with pytest.raises(EvaluationContractError, match="full ranking"):
        build_group_hard_galleries(
            dataset,
            replace(features, frozen_text_similarity=affine_self_scores),
            _hard_config(),
            _freeze_binding(features),
        )

    negative_rank_leak = np.ascontiguousarray(dataset.scores.T.copy())
    for caption_index, positives in enumerate(dataset.positive_motion_indices):
        negative_rank_leak[caption_index, list(positives)] = -1.0e9
    with pytest.raises(EvaluationContractError, match="externally frozen expected"):
        build_group_hard_galleries(
            dataset,
            replace(features, frozen_text_similarity=negative_rank_leak),
            _hard_config(),
            _freeze_binding(features),
        )

    evaluated_checkpoint = features.scorer_provenance.evaluated_checkpoint_sha256s[0]
    with pytest.raises(EvaluationContractError, match="evaluated system checkpoint"):
        build_group_hard_galleries(
            dataset,
            replace(
                features,
                scorer_provenance=replace(
                    features.scorer_provenance,
                    scorer_checkpoint_sha256=evaluated_checkpoint,
                ),
            ),
            _hard_config(),
            _freeze_binding(features),
        )

    collection = build_group_hard_galleries(
        dataset,
        features,
        _hard_config(),
        _freeze_binding(features),
    )
    forged_provenance = replace(
        collection.scorer_provenance,
        freeze_receipt_sha256=hashlib.sha256(b"forged-freeze").hexdigest(),
    )
    with pytest.raises(EvaluationContractError, match="freeze binding|collection digest"):
        evaluate_hard_text_to_motion(
            dataset,
            replace(collection, scorer_provenance=forged_provenance),
        )
    with pytest.raises(TypeError, match="collection"):
        evaluate_hard_text_to_motion(dataset, collection.galleries)


def test_hard_gallery_evaluation_rejects_k_or_digest_tampering() -> None:
    dataset = _dataset(76, k3_count=27)
    features = _features(dataset)
    collection = build_group_hard_galleries(
        dataset,
        features,
        _hard_config(),
        _freeze_binding(features),
    )
    galleries = collection.galleries
    tampered_digest = list(galleries)
    tampered_digest[0] = replace(tampered_digest[0], gallery_sha256="0" * 64)
    with pytest.raises(EvaluationContractError, match="digest"):
        evaluate_hard_text_to_motion(
            dataset,
            replace(collection, galleries=tuple(tampered_digest)),
        )

    mixed_k = list(galleries)
    candidates = list(mixed_k[0].motion_indices)
    negative_position = next(
        index
        for index, motion_index in enumerate(candidates)
        if motion_index not in dataset.positive_motion_indices[0]
    )
    candidates[negative_position] = 27
    candidates.sort(key=lambda index: dataset.motion_commitments[index])
    mixed_k[0] = replace(mixed_k[0], motion_indices=tuple(candidates))
    with pytest.raises(EvaluationContractError, match="strict K"):
        evaluate_hard_text_to_motion(
            dataset,
            replace(collection, galleries=tuple(mixed_k)),
        )


def test_evaluation_rejects_nan_cross_k_or_cross_component_positive() -> None:
    dataset = _dataset()
    bad_scores = dataset.scores.copy()
    bad_scores[0, 0] = np.nan
    with pytest.raises(EvaluationContractError, match="finite"):
        evaluate_retrieval(replace(dataset, scores=bad_scores))

    cross_k = list(dataset.positive_motion_indices)
    cross_k[0] = (0, 20)
    with pytest.raises(EvaluationContractError, match="multiple K"):
        evaluate_retrieval(replace(dataset, positive_motion_indices=tuple(cross_k)))

    cross_component = list(dataset.positive_motion_indices)
    cross_component[0] = (0, 1)
    with pytest.raises(EvaluationContractError, match="multiple components"):
        evaluate_retrieval(replace(dataset, positive_motion_indices=tuple(cross_component)))


def test_strata_and_component_sensitivity_fail_closed() -> None:
    dataset = _dataset(6, k3_count=1)
    with pytest.raises(EvaluationContractError, match="K=3"):
        evaluate_retrieval(dataset)

    one_component = replace(dataset, group_sizes=np.full(6, 3, dtype=np.int64))
    one_component = replace(one_component, component_labels=("C0",) * 6)
    with pytest.raises(EvaluationContractError, match="at least two components"):
        evaluate_retrieval(one_component)
