"""Pure cache-integrity tests; these do not claim actual CLIP inference."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from phaseset_core import frozen_clip_text as clip


HISTORICAL_ADAPTER_ROW = (
    "phaseset_core.frozen_clip_text",
    48_867,
    "9ccd454368f85eedb5eeebfdc2c87d0f4b5f76fc141426ea77cec0487444cc41",
)
V2_HISTORICAL_ADAPTER_ROW = (
    "phaseset_core.frozen_clip_text",
    48_867,
    "1fbe41cce5db731a5d96fc87f7426eacee52382a45ad5bc96a2c3fbd769756e7",
)
HISTORICAL_TRAINING_ROW = (
    "phaseset_core.training",
    156_450,
    "01f5a48be6b4dc9604770d570ff04c0614cdbb9ec90ebfea91afef642384da62",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes(order="C")


def _snapshot_manifest(rows: tuple[tuple[str, int, str], ...]) -> str:
    return _sha(
        _canonical(
            {
                "files": [
                    {"bytes": size, "name": name, "sha256": digest}
                    for name, size, digest in rows
                ],
                "model_id": clip.MODEL_ID,
                "revision": clip.REVISION,
                "schema": "phaseset-frozen-clip-snapshot-v1",
            }
        )
    )


def _source_manifest(rows: tuple[tuple[str, int, str], ...]) -> str:
    return _sha(
        _canonical(
            [
                {"bytes": size, "label": label, "sha256": digest}
                for label, size, digest in rows
            ]
        )
    )


def _runtime_manifest(rows: tuple[tuple[str, str], ...]) -> str:
    return _sha(_canonical(dict(rows)))


def _fixture(
    count: int = 2,
    batch_size: int = 1,
    *,
    training_row: tuple[str, int, str] | None = None,
    adapter_row: tuple[str, int, str] = HISTORICAL_ADAPTER_ROW,
):
    embeddings = torch.arange(count * clip.EMBEDDING_DIM, dtype=torch.float32).reshape(
        count,
        clip.EMBEDDING_DIM,
    )
    embeddings = (torch.sin(embeddings * 0.01) + 0.25).contiguous()
    raw = _tensor_bytes(embeddings)
    caption_rows = []
    for index in range(count):
        row_raw = raw[
            index * clip.EMBEDDING_DIM * 4 : (index + 1) * clip.EMBEDDING_DIM * 4
        ]
        caption_rows.append(
            (
                index,
                _sha(f"caption-lineage/{index}".encode("ascii")),
                _sha(f"caption-text/{index}".encode("ascii")),
                5 + index,
                5 + index,
                False,
                _sha(f"input-ids/{index}".encode("ascii")),
                _sha(f"attention-mask/{index}".encode("ascii")),
                _sha(row_raw),
            )
        )
    chunks = tuple(
        (start, min(start + batch_size, count))
        for start in range(0, count, batch_size)
    )
    snapshot = tuple(sorted(clip._OFFICIAL_PINNED_FILES))
    source = (
        *clip._EXPECTED_SOURCE_FILES[:2],
        training_row or clip._EXPECTED_SOURCE_FILES[2],
        adapter_row,
    )
    runtime = clip._EXPECTED_RUNTIME
    snapshot_sha = _snapshot_manifest(snapshot)
    source_sha = _source_manifest(source)
    runtime_sha = _runtime_manifest(runtime)
    cache_payload = {
        "batch_size": batch_size,
        "caption_rows": [
            {
                "attention_mask_int64_sha256": row[7],
                "caption_utf8_sha256": row[2],
                "encoded_token_count_with_special_tokens": row[4],
                "index": row[0],
                "input_ids_int64_sha256": row[6],
                "original_token_count_with_special_tokens": row[3],
                "truncated": row[5],
            }
            for row in caption_rows
        ],
        "chunk_ranges": [list(value) for value in chunks],
        "method_id": clip.METHOD_ID,
        "model_id": clip.MODEL_ID,
        "revision": clip.REVISION,
        "runtime_manifest_sha256": runtime_sha,
        "snapshot_manifest_sha256": snapshot_sha,
        "source_manifest_sha256": source_sha,
        "token_length": clip.TOKEN_LENGTH,
        "truncation": True,
    }
    receipt = clip.FrozenClipTextReceipt(
        batch_size=batch_size,
        caption_rows=tuple(caption_rows),
        chunk_ranges=chunks,
        frozen_embedding_cache_key_sha256=_sha(_canonical(cache_payload)),
        live_model_manifest_sha256=_sha(b"synthetic-structural-test-model"),
        output_bytes=len(raw),
        output_sha256=_sha(raw),
        output_shape=tuple(embeddings.shape),
        output_stride=tuple(embeddings.stride()),
        runtime_identity=runtime,
        runtime_manifest_sha256=runtime_sha,
        snapshot_files=snapshot,
        snapshot_manifest_sha256=snapshot_sha,
        source_files=source,
        source_manifest_sha256=source_sha,
    )
    return embeddings, receipt


def _rehydrate(embeddings: torch.Tensor, public: dict[str, object]):
    raw = _canonical(public) + b"\n"
    return clip.rehydrate_frozen_clip_text_batch(
        embeddings,
        receipt_json_bytes=raw,
        expected_receipt_sha256=_sha(raw),
    )


def test_rehydrates_owned_snapshot_without_model_or_current_source_relabel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings, receipt = _fixture()
    rng_before = torch.get_rng_state().clone()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("rehydration attempted model/runtime construction")

    monkeypatch.setattr(clip, "_real_backend", forbidden)
    monkeypatch.setattr(clip, "_load_components", forbidden)
    restored = clip.rehydrate_frozen_clip_text_batch(
        embeddings,
        receipt_json_bytes=receipt.canonical_json_bytes(),
        expected_receipt_sha256=receipt.sha256,
    )
    assert type(restored) is clip.FrozenClipTextBatch
    assert restored.receipt.canonical_json_bytes() == receipt.canonical_json_bytes()
    assert restored.receipt.source_files[-1] == HISTORICAL_ADAPTER_ROW
    assert restored.receipt.source_files[-1][2] != _sha(
        Path(clip.__file__).read_bytes()
    )
    assert restored.caption_commitments == tuple(
        bytes.fromhex(row[1]) for row in receipt.caption_rows
    )
    assert torch.equal(restored.embeddings, embeddings)
    embeddings.zero_()
    assert not torch.equal(restored.embeddings, embeddings)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_rehydrates_authenticated_older_training_source_without_relabel() -> None:
    embeddings, receipt = _fixture(
        training_row=HISTORICAL_TRAINING_ROW,
        adapter_row=V2_HISTORICAL_ADAPTER_ROW,
    )
    restored = clip.rehydrate_frozen_clip_text_batch(
        embeddings,
        receipt_json_bytes=receipt.canonical_json_bytes(),
        expected_receipt_sha256=receipt.sha256,
    )
    assert restored.receipt.source_files[2] == HISTORICAL_TRAINING_ROW
    assert restored.receipt.source_files[3] == V2_HISTORICAL_ADAPTER_ROW
    assert restored.receipt.source_manifest_sha256 == receipt.source_manifest_sha256
    assert restored.receipt.frozen_embedding_cache_key_sha256 == (
        receipt.frozen_embedding_cache_key_sha256
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("label", "phaseset_core.foreign"), ("bytes", "156450")],
)
def test_historical_source_rows_keep_exact_label_and_types(
    field: str,
    replacement: object,
) -> None:
    embeddings, receipt = _fixture(training_row=HISTORICAL_TRAINING_ROW)
    public = receipt.to_public_dict()
    public["source"]["files"][2][field] = replacement  # type: ignore[index]
    if field == "label":
        source = public["source"]  # type: ignore[assignment]
        source["manifest_sha256"] = _sha(  # type: ignore[index]
            _canonical(source["files"])  # type: ignore[index]
        )
    with pytest.raises(clip.FrozenClipTextAdapterError) as error:
        _rehydrate(embeddings, public)
    assert error.value.code == "HOLD_REHYDRATION_SOURCE"


def test_expected_receipt_digest_and_canonical_bytes_are_external_gates() -> None:
    embeddings, receipt = _fixture()
    with pytest.raises(clip.FrozenClipTextAdapterError) as mismatch:
        clip.rehydrate_frozen_clip_text_batch(
            embeddings,
            receipt_json_bytes=receipt.canonical_json_bytes(),
            expected_receipt_sha256="0" * 64,
        )
    assert mismatch.value.code == "HOLD_REHYDRATION_EXPECTED_RECEIPT"

    noncanonical = receipt.canonical_json_bytes()[:-1] + b" \n"
    with pytest.raises(clip.FrozenClipTextAdapterError) as whitespace:
        clip.rehydrate_frozen_clip_text_batch(
            embeddings,
            receipt_json_bytes=noncanonical,
            expected_receipt_sha256=_sha(noncanonical),
        )
    assert whitespace.value.code == "HOLD_REHYDRATION_RECEIPT_JSON"

    duplicate = receipt.canonical_json_bytes().replace(
        b'{"authority":0,',
        b'{"authority":0,"authority":0,',
        1,
    )
    with pytest.raises(clip.FrozenClipTextAdapterError) as duplicate_error:
        clip.rehydrate_frozen_clip_text_batch(
            embeddings,
            receipt_json_bytes=duplicate,
            expected_receipt_sha256=_sha(duplicate),
        )
    assert duplicate_error.value.code == "HOLD_REHYDRATION_RECEIPT_JSON"


@pytest.mark.parametrize(
    ("path", "replacement", "code"),
    [
        (("schema",), "foreign-schema", "HOLD_REHYDRATION_RECEIPT_SCHEMA"),
        (("caption_rows", 0, "index"), 1, "HOLD_REHYDRATION_CAPTION_ROWS"),
        (("chunk_ranges",), [[0, 2]], "HOLD_REHYDRATION_CHUNKS"),
        (("output", "stride"), [1, 512], "HOLD_REHYDRATION_OUTPUT"),
        (("runtime_manifest_sha256",), "0" * 64, "HOLD_REHYDRATION_PROVENANCE"),
        (
            ("frozen_embedding_cache_key_sha256",),
            "0" * 64,
            "HOLD_REHYDRATION_CACHE_KEY",
        ),
    ],
)
def test_closed_receipt_semantics_fail_after_authenticated_mutation(
    path: tuple[object, ...],
    replacement: object,
    code: str,
) -> None:
    embeddings, receipt = _fixture(batch_size=1)
    public = receipt.to_public_dict()
    target = public
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(clip.FrozenClipTextAdapterError) as error:
        _rehydrate(embeddings, public)
    assert error.value.code == code


def test_lineage_provenance_and_row_output_digests_are_rebound() -> None:
    embeddings, receipt = _fixture()
    public = receipt.to_public_dict()
    public["caption_rows"][1]["caption_commitment"] = public["caption_rows"][0][
        "caption_commitment"
    ]
    with pytest.raises(clip.FrozenClipTextAdapterError) as duplicate:
        _rehydrate(embeddings, public)
    assert duplicate.value.code == "HOLD_REHYDRATION_LINEAGE"

    public = receipt.to_public_dict()
    public["caption_rows"][0]["embedding_float32_sha256"] = "0" * 64
    with pytest.raises(clip.FrozenClipTextAdapterError) as row_digest:
        _rehydrate(embeddings, public)
    assert row_digest.value.code == "HOLD_REHYDRATION_OUTPUT"

    public = receipt.to_public_dict()
    public["source"]["manifest_sha256"] = "0" * 64
    with pytest.raises(clip.FrozenClipTextAdapterError) as source:
        _rehydrate(embeddings, public)
    assert source.value.code == "HOLD_REHYDRATION_SOURCE"

    public = receipt.to_public_dict()
    public["snapshot"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(clip.FrozenClipTextAdapterError) as snapshot:
        _rehydrate(embeddings, public)
    assert snapshot.value.code == "HOLD_REHYDRATION_SNAPSHOT"


def test_tensor_bytes_type_shape_finiteness_and_gradient_are_exact() -> None:
    embeddings, receipt = _fixture()

    def call(value: object) -> None:
        clip.rehydrate_frozen_clip_text_batch(
            value,
            receipt_json_bytes=receipt.canonical_json_bytes(),
            expected_receipt_sha256=receipt.sha256,
        )

    mutants = (
        embeddings.to(torch.float64),
        embeddings[:, ::2],
        embeddings.detach().clone().requires_grad_(True),
    )
    for mutant in mutants:
        with pytest.raises(clip.FrozenClipTextAdapterError) as error:
            call(mutant)
        assert error.value.code == "HOLD_REHYDRATION_OUTPUT"
    nonfinite = embeddings.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(clip.FrozenClipTextAdapterError) as error:
        call(nonfinite)
    assert error.value.code == "HOLD_REHYDRATION_OUTPUT"
    changed = embeddings.clone()
    changed[0, 0] += 1.0
    with pytest.raises(clip.FrozenClipTextAdapterError) as error:
        call(changed)
    assert error.value.code == "HOLD_REHYDRATION_OUTPUT"


def test_snapshot_requires_the_actual_encoder_filename_order() -> None:
    embeddings, receipt = _fixture()
    canonical_names = [row[0] for row in receipt.snapshot_files]
    assert canonical_names == sorted(canonical_names)
    assert canonical_names != [row[0] for row in clip._OFFICIAL_PINNED_FILES]
    public = receipt.to_public_dict()
    files_by_name = {row["name"]: row for row in public["snapshot"]["files"]}
    public["snapshot"]["files"] = [
        files_by_name[name] for name, _size, _digest in clip._OFFICIAL_PINNED_FILES
    ]
    public["snapshot"]["manifest_sha256"] = _snapshot_manifest(
        clip._OFFICIAL_PINNED_FILES
    )
    with pytest.raises(clip.FrozenClipTextAdapterError) as error:
        _rehydrate(embeddings, public)
    assert error.value.code == "HOLD_REHYDRATION_SNAPSHOT"


def test_direct_batch_construction_remains_sealed() -> None:
    embeddings, receipt = _fixture(1)
    with pytest.raises(TypeError, match="internal"):
        clip.FrozenClipTextBatch(
            embeddings,
            (bytes.fromhex(receipt.caption_rows[0][1]),),
            receipt,
            _seal=object(),
        )
