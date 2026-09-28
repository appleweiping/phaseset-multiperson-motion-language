from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from phaseset_core import caption_fusion, frozen_clip_text
from phaseset_core.caption_role_provenance import (
    ActorCaptionContentSummary,
    CaptionRoleProvenanceError,
    HolisticContextContentSummary,
    WindowFusionHandoff,
    WindowManifestContentSummary,
    WindowManifestRow,
    validate_candidate_machine_fused_window_training,
)
from phaseset_core.contracts import group_commitment
from phaseset_core.pipeline import PreparedGroupSample
from phaseset_core.prepared_data_v2 import (
    PreparedMotionTextBlock,
    PreparedTrainingDataSourceV2,
    load_prepared_training_sources_v2,
    write_prepared_training_tree_v2,
)
from phaseset_core.preprocessing import WindowDecision


DATASET_SHA = hashlib.sha256(b"dataset-manifest").hexdigest()
CAPTION_SHA = hashlib.sha256(b"caption-manifest").hexdigest()
RIGHTS_SHA = hashlib.sha256(b"human-text-rights").hexdigest()
WINDOW_MANIFEST_SHA = hashlib.sha256(b"window-manifest").hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _raw(label: str) -> bytes:
    return hashlib.sha256(label.encode("utf-8")).digest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _sample(label: str, *, actor_prefix: str, ordinal: int) -> PreparedGroupSample:
    actors = tuple(_raw(f"{actor_prefix}-actor-{index}") for index in range(3))
    skeletons = np.zeros((3, 200, 22, 3), dtype=np.float32)
    mask = np.ones((3, 200, 22), dtype=np.bool_)
    return PreparedGroupSample(
        skeletons=skeletons,
        track_mask=mask,
        actor_commitments=actors,
        group_commitment=group_commitment(actors),
        source_sha256=_sha(f"{label}-source"),
        window_sha256=_sha(f"{label}-window"),
        source_start_frame=ordinal * 300,
        augmentation_yaw=0.0,
        decision=WindowDecision(True, "accepted", 300, 3, 0, 1.0, 0, False),
    )


def _fusion(
    sample: PreparedGroupSample,
    label: str,
) -> tuple[WindowFusionHandoff, WindowManifestRow, tuple[str, str]]:
    window_commitment = _raw(f"{label}-private-window-commitment")
    provenance = caption_fusion.CaptionFusionProvenance(
        split="train",
        dataset_manifest_sha256=DATASET_SHA,
        caption_manifest_sha256=CAPTION_SHA,
        human_text_rights_assertion_sha256=RIGHTS_SHA,
    )
    actors = tuple(
        caption_fusion.HumanActorCaption(
            text=f"A person performs action {label}-{index}.",
            actor_commitment=commitment,
            window_commitment=window_commitment,
            source_record_sha256=_sha(f"{label}-actor-source-{index}"),
        )
        for index, commitment in enumerate(reversed(sample.actor_commitments))
    )
    holistic = caption_fusion.HumanHolisticContext(
        text=f"The group jointly performs the {label} activity.",
        window_commitment=window_commitment,
        source_record_sha256=_sha(f"{label}-holistic-source"),
    )
    value = caption_fusion.CaptionFusionInput(actors, provenance, holistic)
    group = f"Three people coordinate the {label} movement."
    paraphrase = f"A trio jointly performs the {label} motion."
    output_sha256 = caption_fusion._output_digest(group, paraphrase)
    admission_sha256 = hashlib.sha256(
        b"phaseset-caption-fusion-local-admission-v1\x00"
        + provenance.digest().encode("ascii")
    ).hexdigest()
    receipt = caption_fusion.CaptionFusionReceipt(
        input_sha256=caption_fusion._input_digest(value),
        output_sha256=output_sha256,
        provenance_sha256=provenance.digest(),
        admission_sha256=admission_sha256,
        backend_manifest_sha256=_sha("frozen-backend"),
        actor_count=3,
        attempts_per_audit_run=(1, 1, 1),
        attempt_count=3,
        retry_count=0,
    )
    fused = caption_fusion.MachineFusedCaption(group, paraphrase, receipt)
    commitments = (_raw(f"{label}-group-text"), _raw(f"{label}-paraphrase-text"))
    handoff = WindowFusionHandoff(value, fused, commitments)
    row = WindowManifestRow(
        window_sha256=sample.window_sha256,
        window_commitment=window_commitment,
        actor_commitments=sample.actor_commitments,
        group_commitment=sample.group_commitment,
        source_sha256=sample.source_sha256,
        source_start_frame=sample.source_start_frame,
        actor_caption_summaries=tuple(
            ActorCaptionContentSummary(
                actor_commitment=item.actor_commitment,
                source_record_sha256=item.source_record_sha256,
                text_utf8_sha256=hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
            )
            for item in actors
        ),
        holistic_context_summary=HolisticContextContentSummary(
            source_record_sha256=holistic.source_record_sha256,
            text_utf8_sha256=hashlib.sha256(holistic.text.encode("utf-8")).hexdigest(),
        ),
    )
    return handoff, row, (group, paraphrase)


def _clip_batch(
    captions: tuple[str, ...],
    commitments: tuple[bytes, ...],
) -> frozen_clip_text.FrozenClipTextBatch:
    embeddings = torch.arange(len(captions) * 512, dtype=torch.float32).reshape(
        len(captions), 512
    )
    output = _tensor_bytes(embeddings)
    caption_rows = tuple(
        (
            index,
            commitment.hex(),
            hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            7,
            7,
            False,
            _sha(f"input-ids-{index}"),
            _sha(f"attention-mask-{index}"),
            hashlib.sha256(_tensor_bytes(embeddings[index : index + 1])).hexdigest(),
        )
        for index, (caption, commitment) in enumerate(
            zip(captions, commitments, strict=True)
        )
    )
    receipt = frozen_clip_text.FrozenClipTextReceipt(
        batch_size=len(captions),
        caption_rows=caption_rows,
        chunk_ranges=((0, len(captions)),),
        frozen_embedding_cache_key_sha256=_sha("cache-key"),
        live_model_manifest_sha256=_sha("model"),
        output_bytes=len(output),
        output_sha256=hashlib.sha256(output).hexdigest(),
        output_shape=tuple(embeddings.shape),
        output_stride=tuple(embeddings.stride()),
        runtime_identity=(("python", "test"),),
        runtime_manifest_sha256=_sha("runtime"),
        snapshot_files=(("config.json", 1, _sha("snapshot-file")),),
        snapshot_manifest_sha256=_sha("snapshot"),
        source_files=(("adapter", 1, _sha("source-file")),),
        source_manifest_sha256=_sha("source"),
    )
    return frozen_clip_text.FrozenClipTextBatch(
        embeddings,
        commitments,
        receipt,
        _seal=frozen_clip_text._CONSTRUCTION_SEAL,
    )


def _block_and_evidence(
    labels: tuple[str, ...],
    *,
    actor_prefix: str,
    ordinal_start: int,
) -> tuple[
    PreparedMotionTextBlock,
    tuple[WindowFusionHandoff, ...],
    tuple[WindowManifestRow, ...],
]:
    samples = tuple(
        _sample(label, actor_prefix=f"{actor_prefix}-{index}", ordinal=ordinal_start + index)
        for index, label in enumerate(labels)
    )
    built = tuple(_fusion(sample, label) for sample, label in zip(samples, labels, strict=True))
    handoffs = tuple(value[0] for value in built)
    rows = tuple(value[1] for value in built)
    captions = tuple(text for value in built for text in value[2])
    commitments = tuple(
        commitment for value in handoffs for commitment in value.clip_caption_commitments
    )
    block = PreparedMotionTextBlock(
        samples=samples,
        text_batch=_clip_batch(captions, commitments),
        text_counts=tuple(2 for _ in samples),
        window_ordinals=tuple(ordinal_start + index for index in range(len(samples))),
    )
    return block, handoffs, rows


def _case(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_block, handoffs, rows = _block_and_evidence(
        ("train-a", "train-b"), actor_prefix="train", ordinal_start=10
    )
    val_block, _, _ = _block_and_evidence(
        ("val-a",), actor_prefix="val", ordinal_start=100
    )
    result = write_prepared_training_tree_v2(
        (tmp_path / "prepared").resolve(),
        train_blocks=(train_block,),
        val_blocks=(val_block,),
        max_total_edges=32,
    )
    train_source, _ = load_prepared_training_sources_v2(
        result.index_path,
        expected_index_sha256=result.index_sha256,
    )
    summary = WindowManifestContentSummary(WINDOW_MANIFEST_SHA, rows)
    kwargs = {
        "blocks": (train_block,),
        "handoffs": handoffs,
        "window_summary": summary,
        "train_source": train_source,
        "expected_window_manifest_sha256": WINDOW_MANIFEST_SHA,
        "expected_prepared_train_manifest_sha256": train_source.manifest_sha256,
        "expected_dataset_manifest_sha256": DATASET_SHA,
        "expected_caption_manifest_sha256": CAPTION_SHA,
        "expected_human_text_rights_assertion_sha256": RIGHTS_SHA,
    }
    return kwargs


def test_validates_complete_candidate_chain_without_selecting_or_authorizing_policy(
    tmp_path,
) -> None:
    receipt = validate_candidate_machine_fused_window_training(**_case(tmp_path))
    assert receipt.window_count == 2
    assert receipt.caption_count == 4
    assert receipt.microbatch_count >= 1
    assert receipt.formal_supervision_policy_selected is False
    assert receipt.rights_authenticated_by_this_receipt is False
    assert receipt.semantic_equivalence_proved is False
    assert receipt.training_authorized is False
    assert receipt.scientific_result_claimed is False
    assert receipt.authority == 0
    assert receipt.schema == "phaseset-private-caption-role-validation-v2"
    assert receipt.caption_manifest_sha256 == CAPTION_SHA
    assert len(receipt.canonical_json_bytes()) > 1
    assert len(receipt.sha256) == 64


def test_rejects_fusion_window_or_actor_lineage_drift(tmp_path) -> None:
    kwargs = _case(tmp_path)
    summary = kwargs["window_summary"]
    first = summary.rows[0]
    changed = replace(first, window_commitment=_raw("wrong-window"))
    kwargs["window_summary"] = replace(summary, rows=(changed, *summary.rows[1:]))
    with pytest.raises(CaptionRoleProvenanceError, match="window commitments differ"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_rejects_human_caption_content_summary_drift(tmp_path) -> None:
    kwargs = _case(tmp_path)
    summary = kwargs["window_summary"]
    first = summary.rows[0]
    actor_rows = first.actor_caption_summaries
    changed_actor = replace(actor_rows[0], text_utf8_sha256=_sha("wrong-human-text"))
    changed = replace(
        first,
        actor_caption_summaries=(changed_actor, *actor_rows[1:]),
    )
    kwargs["window_summary"] = replace(summary, rows=(changed, *summary.rows[1:]))
    with pytest.raises(CaptionRoleProvenanceError, match="actor-caption content differs"):
        validate_candidate_machine_fused_window_training(**kwargs)

    kwargs = _case(tmp_path / "second")
    summary = kwargs["window_summary"]
    first = summary.rows[0]
    kwargs["window_summary"] = replace(
        summary,
        rows=(replace(first, holistic_context_summary=None), *summary.rows[1:]),
    )
    with pytest.raises(CaptionRoleProvenanceError, match="holistic-context presence"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_rejects_swapped_description_and_paraphrase_clip_rows(tmp_path) -> None:
    kwargs = _case(tmp_path)
    first = kwargs["handoffs"][0]
    swapped = replace(
        first,
        clip_caption_commitments=tuple(reversed(first.clip_caption_commitments)),
    )
    kwargs["handoffs"] = (swapped, *kwargs["handoffs"][1:])
    with pytest.raises(CaptionRoleProvenanceError, match="ordered frozen-CLIP rows"):
        validate_candidate_machine_fused_window_training(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("machine_fused", 1),
        ("human_annotation", 0),
        ("public_verification_performed", 0),
        ("authority", False),
        ("scientific_result_claimed", 0),
        ("result_claimed", 0),
    ),
)
def test_rejects_fusion_receipt_bool_int_aliases(tmp_path, field, value) -> None:
    kwargs = _case(tmp_path)
    first = kwargs["handoffs"][0]
    changed_receipt = replace(first.fused_caption.receipt, **{field: value})
    changed_caption = replace(first.fused_caption, receipt=changed_receipt)
    kwargs["handoffs"] = (replace(first, fused_caption=changed_caption), *kwargs["handoffs"][1:])
    with pytest.raises(CaptionRoleProvenanceError, match="caption-fusion receipt"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_rejects_fusion_receipt_live_input_mismatch(tmp_path) -> None:
    kwargs = _case(tmp_path)
    first = kwargs["handoffs"][0]
    changed_receipt = replace(first.fused_caption.receipt, input_sha256=_sha("wrong-input"))
    changed_caption = replace(first.fused_caption, receipt=changed_receipt)
    kwargs["handoffs"] = (replace(first, fused_caption=changed_caption), *kwargs["handoffs"][1:])
    with pytest.raises(CaptionRoleProvenanceError, match="does not bind this handoff"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_rejects_wrong_caption_provenance_or_non_two_caption_policy(tmp_path) -> None:
    kwargs = _case(tmp_path)
    kwargs["expected_caption_manifest_sha256"] = _sha("different-caption-manifest")
    with pytest.raises(CaptionRoleProvenanceError, match="fusion provenance"):
        validate_candidate_machine_fused_window_training(**kwargs)

    kwargs = _case(tmp_path / "second")
    block = kwargs["blocks"][0]
    object.__setattr__(block, "text_counts", (1, 3))
    with pytest.raises(CaptionRoleProvenanceError, match="exactly two"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_materialized_actor_lineage_accepts_canonical_reordering(tmp_path) -> None:
    kwargs = _case(tmp_path)
    original_row = kwargs["window_summary"].rows[0]
    assert original_row.actor_commitments != tuple(sorted(original_row.actor_commitments))
    batches = tuple(kwargs["train_source"].iter_epoch(epoch=0, seed=1729))
    assert batches[0].groups.actor_commitments[0] == tuple(
        sorted(original_row.actor_commitments)
    )
    receipt = validate_candidate_machine_fused_window_training(**kwargs)
    assert receipt.window_count == 2


def test_rejects_materialized_changed_actor_membership(tmp_path, monkeypatch) -> None:
    kwargs = _case(tmp_path)
    original = PreparedTrainingDataSourceV2.iter_epoch

    def mutant(self, *, epoch: int, seed: int):
        batches = tuple(original(self, epoch=epoch, seed=seed))
        first = batches[0]
        actor_rows = list(first.groups.actor_commitments)
        actor_rows[0] = (_raw("changed-materialized-actor"), *actor_rows[0][1:])
        group_rows = list(first.groups.group_commitments)
        group_rows[0] = group_commitment(actor_rows[0])
        changed_groups = replace(
            first.groups,
            actor_commitments=tuple(actor_rows),
            group_commitments=tuple(group_rows),
        )
        yield replace(first, groups=changed_groups)
        yield from batches[1:]

    monkeypatch.setattr(PreparedTrainingDataSourceV2, "iter_epoch", mutant)
    with pytest.raises(CaptionRoleProvenanceError, match="motion lineage differs"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_rejects_materialized_positive_family_mutant(tmp_path, monkeypatch) -> None:
    kwargs = _case(tmp_path)
    original = PreparedTrainingDataSourceV2.iter_epoch

    def mutant(self, *, epoch: int, seed: int):
        batches = tuple(original(self, epoch=epoch, seed=seed))
        first = batches[0]
        old = first.motion_positive_ids[0]
        wrong = _raw("wrong-positive-family")
        motion_ids = (wrong, *first.motion_positive_ids[1:])
        text_ids = tuple(wrong if value == old else value for value in first.text_positive_ids)
        assert first.descriptor_contexts is not None
        contexts = (
            replace(first.descriptor_contexts[0], window_sha256=wrong.hex()),
            *first.descriptor_contexts[1:],
        )
        yield replace(
            first,
            motion_positive_ids=motion_ids,
            text_positive_ids=text_ids,
            descriptor_contexts=contexts,
        )
        yield from batches[1:]

    monkeypatch.setattr(PreparedTrainingDataSourceV2, "iter_epoch", mutant)
    with pytest.raises(CaptionRoleProvenanceError, match="motion family differs"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_rejects_materialized_text_value_or_incomplete_census(tmp_path, monkeypatch) -> None:
    kwargs = _case(tmp_path)
    original = PreparedTrainingDataSourceV2.iter_epoch

    def changed_text(self, *, epoch: int, seed: int):
        batches = tuple(original(self, epoch=epoch, seed=seed))
        first = batches[0]
        values = first.text_embeddings
        values[0, 0] += 1.0
        yield replace(first, text_embeddings=values)
        yield from batches[1:]

    monkeypatch.setattr(PreparedTrainingDataSourceV2, "iter_epoch", changed_text)
    with pytest.raises(CaptionRoleProvenanceError, match="caption value or family"):
        validate_candidate_machine_fused_window_training(**kwargs)

    kwargs = _case(tmp_path / "second")

    def incomplete(self, *, epoch: int, seed: int):
        if False:
            yield from original(self, epoch=epoch, seed=seed)

    monkeypatch.setattr(PreparedTrainingDataSourceV2, "iter_epoch", incomplete)
    with pytest.raises(CaptionRoleProvenanceError, match="emitted no microbatches"):
        validate_candidate_machine_fused_window_training(**kwargs)


def test_receipt_canonicalization_rejects_claim_escalation(tmp_path) -> None:
    receipt = validate_candidate_machine_fused_window_training(**_case(tmp_path))
    for field, value in (
        ("formal_supervision_policy_selected", True),
        ("rights_authenticated_by_this_receipt", True),
        ("training_authorized", True),
        ("scientific_result_claimed", True),
        ("authority", 1),
        ("formal_supervision_policy_selected", 0),
        ("rights_authenticated_by_this_receipt", 0),
        ("semantic_equivalence_proved", 0),
        ("training_authorized", 0),
        ("scientific_result_claimed", 0),
        ("authority", False),
    ):
        with pytest.raises(
            CaptionRoleProvenanceError,
            match="frozen field|frozen field type",
        ):
            replace(receipt, **{field: value}).canonical_json_bytes()
