from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from phaseset_core import embody_adapter as adapter


def _commitment(index: int) -> bytes:
    return hashlib.sha256(f"contract-actor-{index}".encode("ascii")).digest()


def _provenance() -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, hashlib.sha256(f"contract-file-{key}".encode("ascii")).hexdigest())
        for key in adapter.EMBODY_FEATURE_KEYS
    )


def _actor(index: int, *, frames: int = 3) -> adapter.EmbodyActorArrays:
    body_pose = np.zeros((frames, 63), dtype=np.float32)
    global_orient = np.zeros((frames, 3), dtype=np.float32)
    transl = np.zeros((frames, 3), dtype=np.float32)
    transl[:, 0] = np.arange(frames, dtype=np.float32) + index * 10
    betas = np.zeros((frames, 300), dtype=np.float32)
    left_hand = np.zeros((frames, 45), dtype=np.float32)
    right_hand = np.zeros((frames, 45), dtype=np.float32)
    missing = np.ones((frames,), dtype=np.uint8)
    if index == 0:
        missing[1] = 0
        body_pose[1] = np.nan
    return adapter.EmbodyActorArrays(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        betas=betas,
        left_hand_pose=left_hand,
        right_hand_pose=right_hand,
        missing=missing,
        actor_commitment=_commitment(index),
        feature_sha256s=_provenance(),
    )


class _ContractEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.saw_only_finite = True

    def __call__(self, **tracks: np.ndarray) -> np.ndarray:
        assert set(tracks) == {
            "body_pose",
            "global_orient",
            "transl",
            "betas",
            "left_hand_pose",
            "right_hand_pose",
        }
        self.batch_sizes.append(len(tracks["body_pose"]))
        self.saw_only_finite &= all(np.isfinite(value).all() for value in tracks.values())
        output = np.zeros((len(tracks["body_pose"]), 55, 3), dtype=np.float32)
        output[:] = tracks["transl"][:, None, :]
        output[:, :, 1] = np.arange(55, dtype=np.float32)[None, :]
        return output


def test_native_tracks_convert_to_existing_private_capture_without_dropping_frames() -> None:
    evaluator = _ContractEvaluator()
    capture = adapter.embody_to_private_capture(
        (_actor(0), _actor(1)),
        evaluator=evaluator,
        evaluator_provenance_sha256=hashlib.sha256(b"contract-evaluator").hexdigest(),
        max_batch_frames=2,
    )
    assert capture.joints.shape == (3, 2, 22, 3)
    assert capture.track_mask.shape == (3, 2, 22)
    assert capture.track_mask[0, 0].all()
    assert not capture.track_mask[1, 0].any()
    assert capture.track_mask[:, 1].all()
    assert np.count_nonzero(capture.joints[1, 0]) == 0
    np.testing.assert_array_equal(capture.joints[0, 0, :, 1], np.arange(22))
    assert evaluator.batch_sizes == [2, 1, 2, 1]
    assert bool(evaluator.saw_only_finite)
    assert len(capture.source_sha256 or "") == 64


def test_actor_contract_rejects_missing_provenance_bad_shape_and_observed_nan() -> None:
    actor = _actor(0)
    with pytest.raises(adapter.EmbodyAdapterError, match="seven Embody tracks"):
        adapter.EmbodyActorArrays(
            actor.body_pose,
            actor.global_orient,
            actor.transl,
            actor.betas,
            actor.left_hand_pose,
            actor.right_hand_pose,
            actor.missing,
            actor.actor_commitment,
            actor.feature_sha256s[:-1],
        )
    with pytest.raises(adapter.EmbodyAdapterError, match=r"\[T,63\]"):
        adapter.EmbodyActorArrays(
            np.zeros((3, 66), dtype=np.float32),
            actor.global_orient,
            actor.transl,
            actor.betas,
            actor.left_hand_pose,
            actor.right_hand_pose,
            actor.missing,
            actor.actor_commitment,
            actor.feature_sha256s,
        )
    bad = np.array(actor.body_pose, copy=True)
    bad[0, 0] = np.nan
    with pytest.raises(adapter.EmbodyAdapterError, match="observed body_pose"):
        adapter.EmbodyActorArrays(
            bad,
            actor.global_orient,
            actor.transl,
            actor.betas,
            actor.left_hand_pose,
            actor.right_hand_pose,
            actor.missing,
            actor.actor_commitment,
            actor.feature_sha256s,
        )


def test_local_npy_loader_requires_exact_keys_hashes_and_disables_pickle(tmp_path) -> None:
    actor = _actor(1)
    arrays = {
        adapter.BODY_POSE: actor.body_pose,
        adapter.GLOBAL_ORIENT: actor.global_orient,
        adapter.TRANSL: actor.transl,
        adapter.BETAS: actor.betas,
        adapter.LEFT_HAND_POSE: actor.left_hand_pose,
        adapter.RIGHT_HAND_POSE: actor.right_hand_pose,
        adapter.MISSING: actor.missing,
    }
    paths = {}
    digests = {}
    for key, value in arrays.items():
        path = tmp_path / f"{key}.npy"
        np.save(path, value)
        paths[key] = path
        digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = adapter.load_embody_actor_npy(
        paths,
        expected_sha256s=digests,
        actor_commitment=actor.actor_commitment,
    )
    np.testing.assert_array_equal(loaded.body_pose, actor.body_pose)
    assert loaded.feature_sha256s == tuple(
        (key, digests[key]) for key in adapter.EMBODY_FEATURE_KEYS
    )

    with pytest.raises(adapter.EmbodyAdapterError, match="exactly the seven"):
        adapter.load_embody_actor_npy(
            {key: value for key, value in paths.items() if key != adapter.MISSING},
            expected_sha256s=digests,
            actor_commitment=actor.actor_commitment,
        )
    wrong = dict(digests)
    wrong[adapter.BODY_POSE] = "0" * 64
    with pytest.raises(adapter.EmbodyAdapterError, match="SHA-256 mismatch"):
        adapter.load_embody_actor_npy(
            paths,
            expected_sha256s=wrong,
            actor_commitment=actor.actor_commitment,
        )

    unsafe = tmp_path / "unsafe.npy"
    np.save(unsafe, np.asarray([object()], dtype=object))
    unsafe_paths = dict(paths)
    unsafe_paths[adapter.BODY_POSE] = unsafe
    unsafe_digests = dict(digests)
    unsafe_digests[adapter.BODY_POSE] = hashlib.sha256(unsafe.read_bytes()).hexdigest()
    with pytest.raises(adapter.EmbodyAdapterError, match="safe numeric NPY"):
        adapter.load_embody_actor_npy(
            unsafe_paths,
            expected_sha256s=unsafe_digests,
            actor_commitment=actor.actor_commitment,
        )


def test_group_and_evaluator_fail_closed_on_lineage_or_shape_drift() -> None:
    actors = (_actor(0), _actor(1))
    evaluator = _ContractEvaluator()
    with pytest.raises(adapter.EmbodyAdapterError, match="evaluator_provenance"):
        adapter.embody_to_private_capture(
            actors,
            evaluator=evaluator,
            evaluator_provenance_sha256="",
        )

    class _WrongOrderEvaluator:
        def __call__(self, **tracks: np.ndarray) -> np.ndarray:
            return np.zeros((len(tracks["body_pose"]), 21, 3), dtype=np.float32)

    with pytest.raises(adapter.EmbodyAdapterError, match="J>=22"):
        adapter.embody_to_private_capture(
            actors,
            evaluator=_WrongOrderEvaluator(),
            evaluator_provenance_sha256=hashlib.sha256(b"wrong-evaluator").hexdigest(),
        )


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)

    def to(self, *args: object, **kwargs: object) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def contiguous(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _FakeTorch:
    float32 = object()

    @staticmethod
    def from_numpy(value: np.ndarray) -> _FakeTensor:
        return _FakeTensor(value)

    @staticmethod
    def zeros(shape: tuple[int, int], **kwargs: object) -> _FakeTensor:
        return _FakeTensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def no_grad() -> _NoGrad:
        return _NoGrad()


class _FakeModel:
    gender = "neutral"
    num_betas = 300
    num_expression_coeffs = 100
    use_pca = False

    def __init__(self) -> None:
        self.batch_size = 1
        self.training = True
        self.calls: list[dict[str, object]] = []
        self.device = None

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> _FakeModel:
        self.training = False
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        body_pose = kwargs["body_pose"]
        assert isinstance(body_pose, _FakeTensor)
        joints = np.zeros((len(body_pose.value), 55, 3), dtype=np.float32)
        return SimpleNamespace(joints=_FakeTensor(joints))


class _FakeSMPLX:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create(self, path: str, **kwargs: object) -> _FakeModel:
        self.calls.append((path, kwargs))
        return self.model


def test_concrete_evaluator_binds_model_and_supplies_current_chunk_face_zeros(
    tmp_path, monkeypatch
) -> None:
    model_root = tmp_path / "models"
    model_file = model_root / "smplx" / "SMPLX_NEUTRAL.npz"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"contract-only-model-file")
    model = _FakeModel()
    smplx_module = _FakeSMPLX(model)
    monkeypatch.setattr(
        adapter,
        "_runtime_modules",
        lambda: (smplx_module, _FakeTorch()),
    )
    evaluator = adapter.LicensedSMPLXEvaluator(
        model_root, device="host-device",
        expected_model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
    )
    assert evaluator.model_file_sha256 == hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert len(evaluator.provenance_sha256) == 64
    assert str(model_root) not in repr(evaluator)
    assert model.device == "host-device"
    assert model.training is False
    create_path, create_kwargs = smplx_module.calls[0]
    assert create_path == str(model_root)
    assert create_kwargs["model_type"] == "smplx"
    assert create_kwargs["gender"] == "neutral"
    assert create_kwargs["flat_hand_mean"] is True
    assert create_kwargs["num_betas"] == 300
    assert create_kwargs["num_expression_coeffs"] == 100
    assert create_kwargs["use_pca"] is False
    assert create_kwargs["batch_size"] == 1
    assert all(
        create_kwargs[key] is False
        for key in (
            "create_betas",
            "create_global_orient",
            "create_body_pose",
            "create_transl",
            "create_left_hand_pose",
            "create_right_hand_pose",
            "create_expression",
            "create_jaw_pose",
            "create_leye_pose",
            "create_reye_pose",
        )
    )

    for batch_size in (2, 1):
        evaluator(
            body_pose=np.zeros((batch_size, 63), dtype=np.float32),
            global_orient=np.zeros((batch_size, 3), dtype=np.float32),
            transl=np.zeros((batch_size, 3), dtype=np.float32),
            betas=np.zeros((batch_size, 300), dtype=np.float32),
            left_hand_pose=np.zeros((batch_size, 45), dtype=np.float32),
            right_hand_pose=np.zeros((batch_size, 45), dtype=np.float32),
        )
        assert model.batch_size == batch_size
        call = model.calls[-1]
        assert isinstance(call["expression"], _FakeTensor)
        assert call["expression"].value.shape == (batch_size, 100)
        for key in ("jaw_pose", "leye_pose", "reye_pose"):
            assert isinstance(call[key], _FakeTensor)
            assert call[key].value.shape == (batch_size, 3)
        assert call["return_verts"] is False
        assert call["return_shaped"] is False
        assert call["pose2rot"] is True

    capture = adapter.embody_to_private_capture(
        (_actor(0), _actor(1)), evaluator=evaluator, max_batch_frames=2
    )
    assert capture.source_sha256 is not None


def test_untrusted_model_digest_is_rejected_before_import_or_model_construction(
    tmp_path, monkeypatch
) -> None:
    model_file = tmp_path / "SMPLX_NEUTRAL.npz"
    model_file.write_bytes(b"untrusted-contract-only-content")

    def forbidden_import():
        pytest.fail("untrusted model reached the optional pickle-capable loader")

    monkeypatch.setattr(adapter, "_runtime_modules", forbidden_import)
    with pytest.raises(adapter.EmbodyAdapterError, match="trusted expected digest"):
        adapter.LicensedSMPLXEvaluator(
            model_file, device="cpu", expected_model_sha256="0" * 64
        )
    with pytest.raises(adapter.EmbodyAdapterError, match="expected_model_sha256"):
        adapter.LicensedSMPLXEvaluator(
            model_file, device="cpu", expected_model_sha256="invalid"
        )


def test_float64_overflow_is_rejected_before_or_after_evaluation() -> None:
    actor = _actor(1)
    overflowing = np.array(actor.body_pose, dtype=np.float64, copy=True)
    overflowing[0, 0] = np.finfo(np.float64).max
    with pytest.raises(adapter.EmbodyAdapterError, match="representable as finite float32"):
        adapter.EmbodyActorArrays(
            overflowing,
            actor.global_orient,
            actor.transl,
            actor.betas,
            actor.left_hand_pose,
            actor.right_hand_pose,
            actor.missing,
            actor.actor_commitment,
            actor.feature_sha256s,
        )

    class _OverflowingEvaluator:
        def __call__(self, **tracks: np.ndarray) -> np.ndarray:
            result = np.zeros((len(tracks["body_pose"]), 22, 3), dtype=np.float64)
            result[0, 0, 0] = np.finfo(np.float64).max
            return result

    with pytest.raises(adapter.EmbodyAdapterError, match="representable as finite float32"):
        adapter.embody_to_private_capture(
            (_actor(0), _actor(1)),
            evaluator=_OverflowingEvaluator(),
            evaluator_provenance_sha256=hashlib.sha256(b"overflowing-output").hexdigest(),
        )
