"""Server-only host dispatch tests for prepared capture validation storage."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from phaseset_core import capture_prepared_storage as capture_storage
from phaseset_core import host, prepared_data_v2
from phaseset_core.capture_validation import CaptureValidationSource
from phaseset_core.contracts import group_commitment
from phaseset_core.training import TrainingDataSource
from test_phaseset_capture_prepared_storage import _source as _capture_source
from test_phaseset_prepared_host_integration import _sample as _v2_sample
from test_phaseset_prepared_host_integration import _text_batch as _v2_text_batch


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
        newline="\n",
    )


def _write_batch(
    path: Path,
    split: str,
    *,
    actors: tuple[bytes, bytes] | None = None,
    text_commitment: bytes | None = None,
) -> None:
    actors = actors or (_id(f"{split}/actor/0"), _id(f"{split}/actor/1"))
    family = _id(f"{split}/positive")
    np.savez(
        path,
        skeletons=np.zeros((1, 2, 200, 22, 3), dtype=np.float32),
        actor_mask=np.ones((1, 2), dtype=np.bool_),
        frame_mask=np.ones((1, 200), dtype=np.bool_),
        track_mask=np.ones((1, 2, 200, 22), dtype=np.bool_),
        actor_commitments=np.asarray(
            [[list(actors[0]), list(actors[1])]],
            dtype=np.uint8,
        ),
        group_commitments=np.asarray(
            [list(group_commitment(actors))],
            dtype=np.uint8,
        ),
        text_embeddings=np.zeros((1, 512), dtype=np.float32),
        motion_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_commitments=np.asarray(
            [list(text_commitment or _id(f"{split}/text"))],
            dtype=np.uint8,
        ),
    )


def _prepared_v1_tree(
    root: Path,
    *,
    train_actors: tuple[bytes, bytes] | None = None,
    train_text_commitment: bytes | None = None,
) -> Path:
    root.mkdir(parents=True)
    rows: dict[str, dict[str, str]] = {}
    for split in ("train", "val"):
        batch = root / f"{split}-000000.npz"
        _write_batch(
            batch,
            split,
            actors=train_actors if split == "train" else None,
            text_commitment=(
                train_text_commitment if split == "train" else None
            ),
        )
        manifest = root / f"{split}.json"
        _write_json(
            manifest,
            {
                "batches": [{"path": batch.name, "sha256": _sha(batch)}],
                "schema": host.PREPARED_SPLIT_SCHEMA,
                "split": split,
            },
        )
        rows[split] = {"path": manifest.name, "sha256": _sha(manifest)}
    index = root / "prepared-index.json"
    _write_json(
        index,
        {
            "schema": host.PREPARED_INDEX_SCHEMA,
            "train": rows["train"],
            "val": rows["val"],
        },
    )
    return index


def _capture_host_index(
    tmp_path: Path,
    training_index: Path,
    *,
    train_kind: str,
) -> tuple[Path, capture_storage.CaptureStorageBuildResult]:
    source = _capture_source()
    capture_result = capture_storage.write_capture_validation_source(
        (tmp_path / "capture").resolve(),
        source,
    )
    index = tmp_path / "capture-host-index.json"
    _write_json(
        index,
        {
            "schema": host.CAPTURE_VALIDATION_INDEX_SCHEMA,
            "train": {
                "kind": train_kind,
                "path": training_index.relative_to(tmp_path).as_posix(),
                "sha256": _sha(training_index),
            },
            "val": {
                "kind": host.CAPTURE_VALIDATION_KIND,
                "path": capture_result.manifest_path.relative_to(tmp_path).as_posix(),
                "sha256": capture_result.manifest_sha256,
                "source_census_sha256": capture_result.source_census_sha256,
                "upstream_manifest_sha256": source.manifest_sha256,
            },
        },
    )
    return index, capture_result


def _combined_tree(
    tmp_path: Path,
    *,
    train_actors: tuple[bytes, bytes] | None = None,
    train_text_commitment: bytes | None = None,
) -> tuple[Path, capture_storage.CaptureStorageBuildResult]:
    training_index = _prepared_v1_tree(
        tmp_path / "prepared",
        train_actors=train_actors,
        train_text_commitment=train_text_commitment,
    )
    return _capture_host_index(
        tmp_path,
        training_index,
        train_kind=host.PREPARED_TRAIN_V1_KIND,
    )


def _prepared_v2_tree(root: Path, overlap: str | None = None):
    sample = _v2_sample(
        "train-v2",
        capture_label="capture-a" if overlap == "actor" else None,
    )
    if overlap == "window":
        sample = replace(
            sample,
            window_sha256=_id("capture-a/window/0").hex(),
        )
    train_text = _v2_text_batch("capture-a" if overlap == "caption" else "train-v2")
    train = prepared_data_v2.PreparedMotionTextBlock(
        samples=(sample,),
        text_batch=train_text,
        text_counts=(1,),
        window_ordinals=(10,),
    )
    auxiliary = prepared_data_v2.PreparedMotionTextBlock(
        samples=(_v2_sample("auxiliary-v2"),),
        text_batch=_v2_text_batch("auxiliary-v2"),
        text_counts=(1,),
        window_ordinals=(20,),
    )
    return prepared_data_v2.write_prepared_training_tree_v2(
        root,
        train_blocks=(train,),
        val_blocks=(auxiliary,),
        max_total_edges=1,
        max_batch_size=1,
        max_batch_bytes=10_000_000,
        max_decoded_batch_bytes=10_000_000,
    )


def _numpy_rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _config(
    index: Path,
    *,
    residual_artifacts: host.ResidualArtifacts | None = None,
) -> host.HostConfig:
    return host.HostConfig(
        path=index.parent / "host.json",
        prepared_index=index,
        receipt_record=index,
        receipt_artifacts={},
        source_tree_sha256=_id("source tree").hex(),
        device="cpu",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32_768,
        checkpoint_every_updates=1,
        max_batch_bytes=10_000_000,
        resume_retry_class="INFRA_TRANSIENT",
        corrective_change_sha256=_id("corrective change").hex(),
        residual_artifacts=residual_artifacts,
    )


def _bind_receipt(
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
) -> None:
    monkeypatch.setattr(
        host.HostConfig,
        "load_receipts",
        lambda _self: SimpleNamespace(prepared_data_manifest_sha256=digest),
    )


def _rewrite_index(path: Path, mutate) -> str:
    value = json.loads(path.read_text(encoding="ascii"))
    mutate(value)
    _write_json(path, value)
    return _sha(path)


def _make_noncanonical_json(raw: bytes, variant: str) -> bytes:
    if variant == "crlf":
        assert raw.endswith(b"\n") and not raw.endswith(b"\r\n")
        return raw[:-1] + b"\r\n"
    if variant == "whitespace":
        assert raw.startswith(b"{")
        return b"{ " + raw[1:]
    if variant == "duplicate":
        value = json.loads(raw)
        duplicate = json.dumps(value["schema"], ensure_ascii=True).encode("ascii")
        return b'{"schema":' + duplicate + b"," + raw[1:]
    if variant == "nonascii":
        assert b"\\u00e9" in raw
        return raw.replace(b"\\u00e9", "é".encode(), 1)
    raise AssertionError(f"unsupported fixture variant: {variant}")


def test_capture_index_returns_training_protocol_and_exact_capture_val(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, result = _combined_tree(tmp_path)
    _bind_receipt(monkeypatch, _sha(index))

    train, val = _config(index).load_sources()

    assert isinstance(train, TrainingDataSource)
    assert train.split == "train"
    assert type(val) is CaptureValidationSource
    assert val.split == "val"
    assert val.census_sha256 == result.source_census_sha256
    assert val.manifest_sha256 == _capture_source().manifest_sha256
    assert not hasattr(val, "iter_epoch")


@pytest.mark.parametrize("variant", ("whitespace", "duplicate", "nonascii", "crlf"))
def test_capture_index_requires_exact_canonical_bytes_after_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    prepared_root = tmp_path / ("prépared" if variant == "nonascii" else "prepared")
    training_index = _prepared_v1_tree(prepared_root)
    index, _ = _capture_host_index(
        tmp_path,
        training_index,
        train_kind=host.PREPARED_TRAIN_V1_KIND,
    )
    index.write_bytes(_make_noncanonical_json(index.read_bytes(), variant))
    _bind_receipt(monkeypatch, _sha(index))

    with pytest.raises(host.HostConfigurationError, match="bytes are not canonical"):
        _config(index).load_sources()


@pytest.mark.parametrize("variant", ("whitespace", "duplicate", "nonascii", "crlf"))
def test_capture_index_requires_canonical_referenced_training_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    training_index = _prepared_v1_tree(tmp_path / "prepared")
    if variant == "nonascii":
        original = training_index.parent / "train.json"
        renamed = training_index.parent / "tréin.json"
        original.rename(renamed)
        _rewrite_index(
            training_index,
            lambda value: value["train"].update(path=renamed.name),
        )
    index, _ = _capture_host_index(
        tmp_path,
        training_index,
        train_kind=host.PREPARED_TRAIN_V1_KIND,
    )
    training_index.write_bytes(
        _make_noncanonical_json(training_index.read_bytes(), variant)
    )
    outer_digest = _rewrite_index(
        index,
        lambda value: value["train"].update(sha256=_sha(training_index)),
    )
    _bind_receipt(monkeypatch, outer_digest)

    with pytest.raises(host.HostConfigurationError, match="bytes are not canonical"):
        _config(index).load_sources()


def test_existing_v1_index_behavior_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _prepared_v1_tree(tmp_path / "prepared")
    _bind_receipt(monkeypatch, _sha(index))

    train, val = _config(index).load_sources()

    assert type(train) is host.PrivatePreparedDataSource
    assert type(val) is host.PrivatePreparedDataSource
    assert (train.split, val.split) == ("train", "val")


def test_existing_v2_dispatch_keeps_its_authenticated_index_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "prepared-v2.json"
    _write_json(
        index,
        {
            "schema": host.PREPARED_INDEX_V2_SCHEMA,
            "train": {"fixture": "not consumed after registered loader dispatch"},
            "val": {"fixture": "not consumed after registered loader dispatch"},
        },
    )
    expected = _sha(index)
    _bind_receipt(monkeypatch, expected)
    v1_index = _prepared_v1_tree(tmp_path / "v1-fixture")
    v1_train = host.PrivatePreparedDataSource(v1_index.parent / "train.json")
    v1_val = host.PrivatePreparedDataSource(v1_index.parent / "val.json")
    calls: list[tuple[Path, str, int]] = []

    def fake_loader(
        path: Path,
        *,
        expected_index_sha256: str,
        max_batch_bytes: int,
    ) -> tuple[TrainingDataSource, TrainingDataSource]:
        calls.append((path, expected_index_sha256, max_batch_bytes))
        return v1_train, v1_val

    monkeypatch.setattr(host, "load_prepared_training_sources_v2", fake_loader)
    train, val = _config(index).load_sources()

    assert (train, val) == (v1_train, v1_val)
    assert calls == [(index, expected, 10_000_000)]


def test_capture_index_accepts_only_explicit_matching_v2_training_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, result = _combined_tree(tmp_path)
    inner = tmp_path / "prepared" / "prepared-index.json"
    inner_value = json.loads(inner.read_text(encoding="ascii"))
    inner_value["schema"] = host.PREPARED_INDEX_V2_SCHEMA
    _write_json(inner, inner_value)
    outer = json.loads(index.read_text(encoding="ascii"))
    outer["train"]["kind"] = host.PREPARED_TRAIN_V2_KIND
    outer["train"]["sha256"] = _sha(inner)
    _write_json(index, outer)
    _bind_receipt(monkeypatch, _sha(index))
    v1_train = host.PrivatePreparedDataSource(inner.parent / "train.json")
    auxiliary_val = host.PrivatePreparedDataSource(inner.parent / "val.json")
    calls: list[tuple[Path, str]] = []

    def fake_loader(
        path: Path,
        *,
        expected_index_sha256: str,
        max_batch_bytes: int,
    ) -> tuple[TrainingDataSource, TrainingDataSource]:
        assert max_batch_bytes == 10_000_000
        calls.append((path, expected_index_sha256))
        return v1_train, auxiliary_val

    monkeypatch.setattr(host, "load_prepared_training_sources_v2", fake_loader)
    train, val = _config(index).load_sources()

    assert train is v1_train
    assert type(val) is CaptureValidationSource
    assert val.census_sha256 == result.source_census_sha256
    assert calls == [(inner, _sha(inner))]


def test_capture_index_scans_actual_v2_without_consuming_global_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_v2_tree(tmp_path / "prepared-v2")
    index, result = _capture_host_index(
        tmp_path,
        prepared.index_path,
        train_kind=host.PREPARED_TRAIN_V2_KIND,
    )
    _bind_receipt(monkeypatch, _sha(index))
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = host.torch.get_rng_state().clone()

    train, val = _config(index).load_sources()

    assert train.split == "train"
    assert type(val) is CaptureValidationSource
    assert val.census_sha256 == result.source_census_sha256
    assert random.getstate() == python_rng
    assert _numpy_rng_equal(np.random.get_state(), numpy_rng)
    assert host.torch.equal(host.torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize(
    ("overlap", "message"),
    (
        ("actor", "actor/track identity"),
        ("caption", "caption identity"),
        ("window", "window identity"),
    ),
)
def test_capture_index_rejects_actual_v2_train_identity_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlap: str,
    message: str,
) -> None:
    prepared = _prepared_v2_tree(tmp_path / "prepared-v2", overlap)
    index, _ = _capture_host_index(
        tmp_path,
        prepared.index_path,
        train_kind=host.PREPARED_TRAIN_V2_KIND,
    )
    _bind_receipt(monkeypatch, _sha(index))

    with pytest.raises(host.HostConfigurationError, match=message):
        _config(index).load_sources()


@pytest.mark.parametrize(
    ("overlap", "message"),
    (
        ("actor", "actor/track identity"),
        ("caption", "caption identity"),
    ),
)
def test_capture_index_rejects_actual_v1_train_identity_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlap: str,
    message: str,
) -> None:
    source = _capture_source()
    val_actors = sorted(
        actor
        for capture in source.captures
        for actor in capture.actor_commitment_set
    )
    val_captions = sorted(
        commitment
        for capture in source.captures
        for commitment in capture.holistic_text.caption_commitments
    )
    index, _ = _combined_tree(
        tmp_path,
        train_actors=(val_actors[0], _id("train/distinct-actor"))
        if overlap == "actor"
        else None,
        train_text_commitment=val_captions[0] if overlap == "caption" else None,
    )
    _bind_receipt(monkeypatch, _sha(index))

    with pytest.raises(host.HostConfigurationError, match=message):
        _config(index).load_sources()


def test_capture_index_rejects_train_source_drift_during_full_identity_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _ = _combined_tree(tmp_path)
    inner = tmp_path / "prepared" / "prepared-index.json"
    actual = host.PrivatePreparedDataSource(inner.parent / "train.json")

    class DriftingSource:
        split = "train"

        def __init__(self) -> None:
            self.manifest_sha256 = actual.manifest_sha256

        def iter_epoch(self, *, epoch: int, seed: int):
            yield from actual.iter_epoch(epoch=epoch, seed=seed)
            self.manifest_sha256 = "0" * 64

    drifting = DriftingSource()
    monkeypatch.setattr(
        host.HostConfig,
        "_load_registered_training_index",
        lambda _self, _path, **_kwargs: drifting,
    )
    _bind_receipt(monkeypatch, _sha(index))

    with pytest.raises(host.HostConfigurationError, match="changed during identity scan"):
        _config(index).load_sources()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["train"].update(kind=[]), "training kind"),
        (lambda value: value["train"].update(kind=host.CAPTURE_VALIDATION_KIND), "training kind"),
        (lambda value: value["val"].update(kind=host.PREPARED_TRAIN_V1_KIND), "val kind"),
        (lambda value: value["train"].update(kind=host.PREPARED_TRAIN_V2_KIND), "schema"),
        (lambda value: value["val"].update(sha256="0" * 64), "val storage"),
        (
            lambda value: value["val"].update(source_census_sha256="0" * 64),
            "source census",
        ),
        (
            lambda value: value["val"].update(upstream_manifest_sha256="0" * 64),
            "upstream manifest",
        ),
    ),
)
def test_capture_index_rejects_mixed_kinds_and_identity_drift_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    index, _ = _combined_tree(tmp_path)
    rebound = _rewrite_index(index, mutation)
    _bind_receipt(monkeypatch, rebound)

    with pytest.raises(host.HostConfigurationError, match=message):
        _config(index).load_sources()


def test_outer_index_digest_is_required_before_capture_artifact_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _ = _combined_tree(tmp_path)
    _bind_receipt(monkeypatch, "0" * 64)
    reached = False

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal reached
        reached = True
        raise AssertionError("capture loader must not run")

    monkeypatch.setattr(host, "load_capture_validation_source", forbidden_loader)
    with pytest.raises(host.HostConfigurationError, match="authenticated receipt"):
        _config(index).load_sources()
    assert reached is False


class _FakeRuntime:
    calls: list[tuple[object, object, dict[str, object]]] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def fit(self, train: object, val: object, **kwargs: object) -> str:
        self.calls.append((train, val, kwargs))
        return "FIT_RETURNED"


def test_base_runtime_and_resume_forward_typed_capture_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, result = _combined_tree(tmp_path)
    _bind_receipt(monkeypatch, _sha(index))
    monkeypatch.setattr(
        host,
        "construct_registered_base_seed_bound_system",
        lambda _system_id, _config: (object(), object()),
    )
    _FakeRuntime.calls.clear()
    monkeypatch.setattr(host, "PhaseSetTrainingRuntime", _FakeRuntime)
    backend = host.PhaseSetHostBackend(_config(index), object())
    resume = tmp_path / "checkpoint.pt"

    returned = backend._execute_base_runtime(
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        system_id="B0",
        seed=1729,
        checkpoint_directory=tmp_path / "model-checkpoints",
        live_ledger=SimpleNamespace(observe_checkpoint=lambda _value: None),
        resume_checkpoint=resume,
        resume_attempt_root=tmp_path,
        resume_record=None,
    )

    assert returned == "FIT_RETURNED"
    train, val, kwargs = _FakeRuntime.calls[-1]
    assert isinstance(train, TrainingDataSource) and train.split == "train"
    assert type(val) is CaptureValidationSource
    assert val.census_sha256 == result.source_census_sha256
    assert kwargs["resume_checkpoint"] == resume
    assert kwargs["resume_attempt_root"] == tmp_path


def test_residual_runtime_and_resume_forward_typed_capture_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, result = _combined_tree(tmp_path)
    _bind_receipt(monkeypatch, _sha(index))
    qualification = tmp_path / "qualification.json"
    capacity = tmp_path / "capacity.json"
    qualification.write_bytes(b"qualification")
    capacity.write_bytes(b"capacity")
    artifacts = host.ResidualArtifacts(
        qualification=qualification,
        capacity_audits={1729: capacity},
    )
    monkeypatch.setattr(host, "_load_base_qualification", lambda _path: (object(), "1" * 64))
    monkeypatch.setattr(host, "load_qualified_frozen_base", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(host, "_load_periodic_cache", lambda *_args, **_kwargs: (1.0,) * 6)
    monkeypatch.setattr(host, "_load_capacity_audit", lambda _path: (object(), "2" * 64))
    monkeypatch.setattr(
        host,
        "construct_registered_residual_seed_bound_system",
        lambda *_args, **_kwargs: (object(), object()),
    )
    _FakeRuntime.calls.clear()
    monkeypatch.setattr(host, "PhaseSetTrainingRuntime", _FakeRuntime)
    backend = host.PhaseSetHostBackend(_config(index, residual_artifacts=artifacts), object())
    resume = tmp_path / "residual-checkpoint.pt"

    returned = backend._execute_residual_runtime(
        system_id="01",
        seed=1729,
        checkpoint_directory=tmp_path / "model-checkpoints",
        live_ledger=SimpleNamespace(observe_checkpoint=lambda _value: None),
        base_checkpoint=tmp_path / "base.pt",
        periodic_cache=tmp_path / "periodic-cache",
        stop_after_global_step=None,
        resume_checkpoint=resume,
        resume_attempt_root=tmp_path,
        resume_record=None,
    )

    assert returned == "FIT_RETURNED"
    train, val, kwargs = _FakeRuntime.calls[-1]
    assert isinstance(train, TrainingDataSource) and train.split == "train"
    assert type(val) is CaptureValidationSource
    assert val.census_sha256 == result.source_census_sha256
    assert kwargs["resume_checkpoint"] == resume
    assert kwargs["resume_attempt_root"] == tmp_path
