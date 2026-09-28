"""Bounded Embody SMPL-X conversion seam for caller-licensed local assets.

This module contains no dataset locator, download credential, participant ID,
or SMPL-X model asset.  It validates the seven native per-actor Embody tracks,
calls a caller-supplied SMPL-X joint evaluator in bounded chunks, and returns
the existing :class:`PrivateCaptureArrays` seam.  Windowing, 30-to-20 Hz
resampling, body-22 selection, canonicalization, and batching remain owned by
``phaseset_core.pipeline``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from importlib import import_module
import io
from pathlib import Path
import struct
import threading
from typing import Final, Protocol

import numpy as np

from phaseset_core.pipeline import PrivateCaptureArrays


BODY_POSE: Final = "smplx_mesh_body_pose"
GLOBAL_ORIENT: Final = "smplx_mesh_global_orient"
TRANSL: Final = "smplx_mesh_transl"
BETAS: Final = "smplx_mesh_betas"
LEFT_HAND_POSE: Final = "smplx_mesh_left_hand_pose"
RIGHT_HAND_POSE: Final = "smplx_mesh_right_hand_pose"
MISSING: Final = "missing"
EMBODY_FEATURE_KEYS: Final = (
    BODY_POSE,
    GLOBAL_ORIENT,
    TRANSL,
    BETAS,
    LEFT_HAND_POSE,
    RIGHT_HAND_POSE,
    MISSING,
)
SMPLX_BODY22_NAMES: Final = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
DEFAULT_MAX_ACTOR_FILE_BYTES: Final = 512 * 1024 * 1024
ABSOLUTE_MAX_ACTOR_FILE_BYTES: Final = 2 * 1024 * 1024 * 1024
MAX_ACTOR_FRAMES: Final = 360_000
DEFAULT_EVALUATOR_BATCH_FRAMES: Final = 600
MAX_EVALUATOR_BATCH_FRAMES: Final = 4096
MAX_SMPLX_MODEL_BYTES: Final = 2 * 1024 * 1024 * 1024
LICENSED_SMPLX_CONFIG: Final = (
    "model_type=smplx;gender=neutral;flat_hand_mean=true;num_betas=300;"
    "num_expression_coeffs=100;use_pca=false;axis_angle=true"
)


class EmbodyAdapterError(ValueError):
    """Native Embody data, provenance, or evaluator output is invalid."""


class SMPLXBodyJointEvaluator(Protocol):
    """Licensed-host implementation of the official SMPL-X forward seam.

    Inputs are finite float32 axis-angle/shape/translation arrays for one
    actor and one bounded contiguous frame chunk.  The result must use the
    official SMPL-X joint order and contain at least its first 22 body joints.
    """

    def __call__(
        self,
        *,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray,
        left_hand_pose: np.ndarray,
        right_hand_pose: np.ndarray,
    ) -> np.ndarray: ...


def _runtime_modules() -> tuple[object, object]:
    """Import optional SMPL-X and the package's torch dependency lazily."""

    try:
        return import_module("smplx"), import_module("torch")
    except ImportError as exc:
        raise EmbodyAdapterError(
            "LicensedSMPLXEvaluator requires the caller-installed official smplx package"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_neutral_model_file(model_path: str | Path) -> tuple[Path, Path]:
    supplied = Path(model_path)
    if supplied.is_symlink():
        raise EmbodyAdapterError("SMPL-X model path must not be a symlink")
    if supplied.is_dir():
        model_file = supplied / "smplx" / "SMPLX_NEUTRAL.npz"
        create_path = supplied
    else:
        model_file = supplied
        create_path = supplied
    if (
        model_file.name != "SMPLX_NEUTRAL.npz"
        or model_file.is_symlink()
        or not model_file.is_file()
    ):
        raise EmbodyAdapterError("model_path must resolve to the neutral SMPL-X NPZ asset")
    size = model_file.stat().st_size
    if not 1 <= size <= MAX_SMPLX_MODEL_BYTES:
        raise EmbodyAdapterError("SMPL-X model asset is empty or exceeds the byte bound")
    return create_path, model_file


class LicensedSMPLXEvaluator:
    """Concrete official-SMPL-X evaluator for a caller-licensed local model.

    Construction performs no download and makes no license assertion.  It
    verifies a trusted expected NPZ digest before importing the optional loader
    and rechecks after ``smplx.create``. The expected digest must come from the
    caller's verified licensed-asset manifest, not an untrusted file itself.
    Facial tracks absent from Embody are supplied
    as exact zeros at the current chunk length instead of using model parameters
    whose constructor batch size can differ from the final chunk.
    """

    __slots__ = (
        "_device",
        "_lock",
        "_model",
        "_model_file_sha256",
        "_provenance_sha256",
        "_torch",
    )

    def __init__(
        self, model_path: str | Path, *, device: str, expected_model_sha256: str
    ) -> None:
        if type(device) is not str or not device or "\x00" in device:
            raise EmbodyAdapterError("device must be a nonempty NUL-free string")
        create_path, model_file = _resolve_neutral_model_file(model_path)
        expected = _sha256_hex(expected_model_sha256, "expected_model_sha256")
        digest_before = _sha256_file(model_file)
        # Official SMPL-X loads NPZ with pickle enabled. Establish the trusted
        # artifact binding BEFORE handing any bytes to that third-party loader.
        if digest_before != expected:
            raise EmbodyAdapterError("SMPL-X model asset differs from trusted expected digest")
        smplx_module, torch_module = _runtime_modules()
        try:
            model = smplx_module.create(
                str(create_path),
                model_type="smplx",
                gender="neutral",
                flat_hand_mean=True,
                num_betas=300,
                num_expression_coeffs=100,
                use_pca=False,
                dtype=torch_module.float32,
                batch_size=1,
                create_betas=False,
                create_global_orient=False,
                create_body_pose=False,
                create_transl=False,
                create_left_hand_pose=False,
                create_right_hand_pose=False,
                create_expression=False,
                create_jaw_pose=False,
                create_leye_pose=False,
                create_reye_pose=False,
            )
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise EmbodyAdapterError("official SMPL-X model construction failed") from exc
        digest_after = _sha256_file(model_file)
        if digest_after != digest_before:
            raise EmbodyAdapterError("SMPL-X model asset changed while it was loaded")
        if (
            getattr(model, "gender", None) != "neutral"
            or getattr(model, "num_betas", None) != 300
            or getattr(model, "num_expression_coeffs", None) != 100
            or getattr(model, "use_pca", None) is not False
        ):
            raise EmbodyAdapterError("loaded SMPL-X model does not match the Embody contract")
        provenance = hashlib.sha256(b"phaseset-licensed-smplx-evaluator-v1\x00")
        provenance.update(LICENSED_SMPLX_CONFIG.encode("ascii"))
        provenance.update(b"\x00" + digest_before.encode("ascii"))
        self._device = device
        self._lock = threading.Lock()
        self._model = model
        self._torch = torch_module
        self._model_file_sha256 = digest_before
        self._provenance_sha256 = provenance.hexdigest()

    @property
    def model_file_sha256(self) -> str:
        return self._model_file_sha256

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256

    def __repr__(self) -> str:
        return (
            "LicensedSMPLXEvaluator("
            f"model_file_sha256='{self.model_file_sha256}', device={self._device!r})"
        )

    def __call__(
        self,
        *,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray,
        left_hand_pose: np.ndarray,
        right_hand_pose: np.ndarray,
    ) -> np.ndarray:
        tracks = {
            "body_pose": (body_pose, 63),
            "global_orient": (global_orient, 3),
            "transl": (transl, 3),
            "betas": (betas, 300),
            "left_hand_pose": (left_hand_pose, 45),
            "right_hand_pose": (right_hand_pose, 45),
        }
        batch_size = len(body_pose) if type(body_pose) is np.ndarray else -1
        if not 1 <= batch_size <= MAX_EVALUATOR_BATCH_FRAMES:
            raise EmbodyAdapterError("SMPL-X evaluator batch length is outside its bound")
        tensors = {}
        for label, (value, width) in tracks.items():
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.float32)
                or value.shape != (batch_size, width)
                or not value.flags.c_contiguous
                or not bool(np.isfinite(value).all())
            ):
                raise EmbodyAdapterError(
                    f"{label} evaluator input must be finite C-contiguous float32 "
                    f"[{batch_size},{width}]"
                )
            tensors[label] = self._torch.from_numpy(value).to(
                device=self._device, dtype=self._torch.float32
            )
        zeros3 = self._torch.zeros(
            (batch_size, 3), device=self._device, dtype=self._torch.float32
        )
        expression = self._torch.zeros(
            (batch_size, 100), device=self._device, dtype=self._torch.float32
        )
        with self._lock, self._torch.no_grad():
            # SMPLX.forward uses this member for landmark barycentric repeats.
            # Updating it to the real chunk length prevents a short final chunk
            # from inheriting the constructor's fixed batch dimension.
            self._model.batch_size = batch_size
            output = self._model(
                **tensors,
                expression=expression,
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                return_verts=False,
                return_full_pose=False,
                return_shaped=False,
                pose2rot=True,
            )
        joints = getattr(output, "joints", None)
        if joints is None:
            raise EmbodyAdapterError("official SMPL-X output has no joints")
        try:
            result = (
                joints.detach()
                .to(device="cpu", dtype=self._torch.float32)
                .contiguous()
                .numpy()
            )
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise EmbodyAdapterError("official SMPL-X joints could not be exported") from exc
        result = np.array(result, dtype=np.float32, copy=True, order="C")
        if (
            result.ndim != 3
            or result.shape[0] != batch_size
            or result.shape[1] < 22
            or result.shape[2] != 3
            or not bool(np.isfinite(result).all())
        ):
            raise EmbodyAdapterError("official SMPL-X joints violate [B,J>=22,3]")
        return np.ascontiguousarray(result[:, :22])


def _sha256_hex(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EmbodyAdapterError(f"{label} must be lowercase SHA-256 hex")
    return value


def _commitment(value: object) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise EmbodyAdapterError("actor_commitment must be exact bytes32")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    snapshot = np.ascontiguousarray(np.array(value, copy=True, subok=False))
    snapshot.setflags(write=False)
    return snapshot


def _float_track(value: object, label: str, width: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray")
    if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise EmbodyAdapterError(f"{label} must be float32 or float64")
    if value.ndim != 2 or value.shape[1] != width:
        raise EmbodyAdapterError(f"{label} must have shape [T,{width}]")
    return _readonly(value)


def _missing_track(value: object) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("missing must be an exact numpy.ndarray")
    flattened = value[:, 0] if value.ndim == 2 and value.shape[1] == 1 else value
    if flattened.ndim != 1 or flattened.dtype.kind not in "bifu":
        raise EmbodyAdapterError("missing must be a numeric binary [T] or [T,1] array")
    if not bool(np.isfinite(flattened).all()) or not bool(
        np.logical_or(flattened == 0, flattened == 1).all()
    ):
        raise EmbodyAdapterError("missing must contain only the official binary values 0 or 1")
    return _readonly(np.asarray(flattened, dtype=np.bool_))


def _canonical_feature_sha256s(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise TypeError("feature_sha256s must be an exact tuple")
    if any(type(item) is not tuple or len(item) != 2 for item in value):
        raise EmbodyAdapterError(
            "feature_sha256s entries must be exact (feature, sha256) tuples"
        )
    supplied = {item[0]: item[1] for item in value}
    if len(supplied) != len(value) or set(supplied) != set(EMBODY_FEATURE_KEYS):
        raise EmbodyAdapterError("feature provenance must cover exactly the seven Embody tracks")
    return tuple(
        (key, _sha256_hex(supplied[key], f"{key} provenance"))
        for key in EMBODY_FEATURE_KEYS
    )


@dataclass(frozen=True, slots=True)
class EmbodyActorArrays:
    """One actor's complete native Embody SMPL-X tracks at 30 fps.

    ``missing`` follows the upstream convention: 1 is usable tracking and 0
    is a corrupted frame.  Missing frames are retained, never dropped.
    ``feature_sha256s`` contains only digests, never private paths or IDs.
    """

    body_pose: np.ndarray = field(repr=False)
    global_orient: np.ndarray = field(repr=False)
    transl: np.ndarray = field(repr=False)
    betas: np.ndarray = field(repr=False)
    left_hand_pose: np.ndarray = field(repr=False)
    right_hand_pose: np.ndarray = field(repr=False)
    missing: np.ndarray = field(repr=False)
    actor_commitment: bytes = field(repr=False)
    feature_sha256s: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        tracks = {
            "body_pose": _float_track(self.body_pose, BODY_POSE, 63),
            "global_orient": _float_track(self.global_orient, GLOBAL_ORIENT, 3),
            "transl": _float_track(self.transl, TRANSL, 3),
            "betas": _float_track(self.betas, BETAS, 300),
            "left_hand_pose": _float_track(self.left_hand_pose, LEFT_HAND_POSE, 45),
            "right_hand_pose": _float_track(self.right_hand_pose, RIGHT_HAND_POSE, 45),
        }
        missing = _missing_track(self.missing)
        frame_count = int(tracks["body_pose"].shape[0])
        if not 1 <= frame_count <= MAX_ACTOR_FRAMES:
            raise EmbodyAdapterError(f"actor frame count must be in [1,{MAX_ACTOR_FRAMES}]")
        if any(track.shape[0] != frame_count for track in tracks.values()):
            raise EmbodyAdapterError("all native Embody tracks must have the same frame count")
        if missing.shape != (frame_count,):
            raise EmbodyAdapterError("missing must have the same frame count as SMPL-X tracks")
        observed = missing
        if not bool(observed.any()):
            raise EmbodyAdapterError("actor has no usable tracked frame")
        for label, track in tracks.items():
            if not bool(np.isfinite(track[observed]).all()):
                raise EmbodyAdapterError(f"observed {label} contains non-finite data")
            with np.errstate(over="ignore", invalid="ignore"):
                observed_float32 = np.asarray(track[observed], dtype=np.float32)
            if not bool(np.isfinite(observed_float32).all()):
                raise EmbodyAdapterError(
                    f"observed {label} is not representable as finite float32"
                )
            object.__setattr__(self, label, track)
        object.__setattr__(self, "missing", missing)
        object.__setattr__(self, "actor_commitment", _commitment(self.actor_commitment))
        object.__setattr__(
            self,
            "feature_sha256s",
            _canonical_feature_sha256s(self.feature_sha256s),
        )

    @property
    def frame_count(self) -> int:
        return int(self.body_pose.shape[0])

    @property
    def observed_mask(self) -> np.ndarray:
        return self.missing


def _file_bytes(path: Path, *, feature: str, maximum: int) -> bytes:
    if path.suffix.lower() != ".npy" or path.is_symlink() or not path.is_file():
        raise EmbodyAdapterError(f"{feature} must be a regular non-symlink .npy file")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise EmbodyAdapterError(f"{feature} is empty or exceeds the byte bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise EmbodyAdapterError(f"{feature} changed while it was read")
    return raw


def load_embody_actor_npy(
    feature_paths: Mapping[str, str | Path],
    *,
    expected_sha256s: Mapping[str, str],
    actor_commitment: bytes,
    max_total_bytes: int = DEFAULT_MAX_ACTOR_FILE_BYTES,
) -> EmbodyActorArrays:
    """Load one caller-supplied licensed actor record without pickle support."""

    if not isinstance(feature_paths, Mapping) or set(feature_paths) != set(
        EMBODY_FEATURE_KEYS
    ):
        raise EmbodyAdapterError("feature_paths must cover exactly the seven Embody tracks")
    if not isinstance(expected_sha256s, Mapping) or set(expected_sha256s) != set(
        EMBODY_FEATURE_KEYS
    ):
        raise EmbodyAdapterError("expected_sha256s must cover exactly the seven Embody tracks")
    if type(max_total_bytes) is not int or not (
        1 <= max_total_bytes <= ABSOLUTE_MAX_ACTOR_FILE_BYTES
    ):
        raise EmbodyAdapterError("max_total_bytes is outside the public safety bound")
    commitment = _commitment(actor_commitment)
    arrays: dict[str, np.ndarray] = {}
    digests: list[tuple[str, str]] = []
    consumed = 0
    for feature in EMBODY_FEATURE_KEYS:
        expected = _sha256_hex(expected_sha256s[feature], f"{feature} expected digest")
        remaining = max_total_bytes - consumed
        if remaining <= 0:
            raise EmbodyAdapterError("actor files exceed max_total_bytes")
        raw = _file_bytes(Path(feature_paths[feature]), feature=feature, maximum=remaining)
        consumed += len(raw)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise EmbodyAdapterError(f"{feature} SHA-256 mismatch")
        try:
            loaded = np.load(io.BytesIO(raw), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise EmbodyAdapterError(f"{feature} is not a safe numeric NPY") from exc
        if type(loaded) is not np.ndarray:
            raise EmbodyAdapterError(f"{feature} must contain exactly one numeric array")
        arrays[feature] = np.array(loaded, copy=True, subok=False)
        digests.append((feature, actual))
    return EmbodyActorArrays(
        body_pose=arrays[BODY_POSE],
        global_orient=arrays[GLOBAL_ORIENT],
        transl=arrays[TRANSL],
        betas=arrays[BETAS],
        left_hand_pose=arrays[LEFT_HAND_POSE],
        right_hand_pose=arrays[RIGHT_HAND_POSE],
        missing=arrays[MISSING],
        actor_commitment=commitment,
        feature_sha256s=tuple(digests),
    )


def _array_digest(digest: object, label: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(label.encode("ascii"))
    digest.update(b"\x00")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(struct.pack(">I", contiguous.ndim))
    for dimension in contiguous.shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(contiguous.tobytes(order="C"))


def _actor_source_digest(actor: EmbodyActorArrays) -> bytes:
    digest = hashlib.sha256(b"phaseset-embody-actor-source-v1\x00")
    for feature, file_digest in actor.feature_sha256s:
        digest.update(feature.encode("ascii") + b"\x00" + file_digest.encode("ascii"))
    for label, value in (
        (BODY_POSE, actor.body_pose),
        (GLOBAL_ORIENT, actor.global_orient),
        (TRANSL, actor.transl),
        (BETAS, actor.betas),
        (LEFT_HAND_POSE, actor.left_hand_pose),
        (RIGHT_HAND_POSE, actor.right_hand_pose),
        (MISSING, actor.missing),
    ):
        _array_digest(digest, label, value)
    return digest.digest()


def _safe_chunk(track: np.ndarray, observed: np.ndarray, start: int, stop: int) -> np.ndarray:
    source = track[start:stop]
    present = observed[start:stop]
    chunk = np.zeros(source.shape, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        chunk[present] = source[present]
    if not bool(np.isfinite(chunk).all()):
        raise EmbodyAdapterError("observed SMPL-X track is not finite after float32 conversion")
    return np.ascontiguousarray(chunk)


def embody_to_private_capture(
    actors: tuple[EmbodyActorArrays, ...],
    *,
    evaluator: SMPLXBodyJointEvaluator,
    evaluator_provenance_sha256: str | None = None,
    max_batch_frames: int = DEFAULT_EVALUATOR_BATCH_FRAMES,
) -> PrivateCaptureArrays:
    """Convert a synchronized native Embody group to the PhaseSet numeric seam.

    The evaluator is invoked only with finite arrays and never with more than
    ``max_batch_frames`` frames.  Corrupted frames remain present with a false
    joint mask and exact +0 coordinates.  No window is accepted or rejected
    here; the existing pipeline applies that frozen policy later.
    """

    if type(actors) is not tuple or len(actors) < 2:
        raise EmbodyAdapterError("actors must be an exact tuple containing at least two actors")
    if any(type(actor) is not EmbodyActorArrays for actor in actors):
        raise TypeError("every actor must be exactly EmbodyActorArrays")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    observed_evaluator_digest = getattr(evaluator, "provenance_sha256", None)
    if type(evaluator) is LicensedSMPLXEvaluator:
        evaluator_digest = _sha256_hex(
            observed_evaluator_digest, "LicensedSMPLXEvaluator provenance_sha256"
        )
        if (
            evaluator_provenance_sha256 is not None
            and evaluator_provenance_sha256 != evaluator_digest
        ):
            raise EmbodyAdapterError(
                "supplied evaluator provenance differs from the observed model binding"
            )
    else:
        evaluator_digest = _sha256_hex(
            evaluator_provenance_sha256, "evaluator_provenance_sha256"
        )
    if type(max_batch_frames) is not int or not (
        1 <= max_batch_frames <= MAX_EVALUATOR_BATCH_FRAMES
    ):
        raise EmbodyAdapterError("max_batch_frames is outside the public safety bound")
    frame_count = actors[0].frame_count
    if any(actor.frame_count != frame_count for actor in actors):
        raise EmbodyAdapterError("all group actors must have the same synchronized frame count")
    commitments = tuple(actor.actor_commitment for actor in actors)
    if len(set(commitments)) != len(commitments):
        raise EmbodyAdapterError("actor commitments must be distinct within a group")

    joints = np.zeros((frame_count, len(actors), 22, 3), dtype=np.float32)
    track_mask = np.zeros((frame_count, len(actors), 22), dtype=np.bool_)
    for actor_index, actor in enumerate(actors):
        observed = actor.observed_mask
        track_mask[:, actor_index] = observed[:, None]
        for start in range(0, frame_count, max_batch_frames):
            stop = min(start + max_batch_frames, frame_count)
            result = evaluator(
                body_pose=_safe_chunk(actor.body_pose, observed, start, stop),
                global_orient=_safe_chunk(actor.global_orient, observed, start, stop),
                transl=_safe_chunk(actor.transl, observed, start, stop),
                betas=_safe_chunk(actor.betas, observed, start, stop),
                left_hand_pose=_safe_chunk(actor.left_hand_pose, observed, start, stop),
                right_hand_pose=_safe_chunk(actor.right_hand_pose, observed, start, stop),
            )
            if type(result) is not np.ndarray:
                raise EmbodyAdapterError("SMPL-X evaluator must return an exact numpy.ndarray")
            if result.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise EmbodyAdapterError("SMPL-X evaluator output must be float32 or float64")
            if (
                result.ndim != 3
                or result.shape[0] != stop - start
                or result.shape[1] < 22
                or result.shape[2] != 3
            ):
                raise EmbodyAdapterError(
                    "SMPL-X evaluator output must have shape [chunk,J>=22,3]"
                )
            if not bool(np.isfinite(result).all()):
                raise EmbodyAdapterError("SMPL-X evaluator output contains non-finite coordinates")
            with np.errstate(over="ignore", invalid="ignore"):
                body22 = np.array(result[:, :22], dtype=np.float32, copy=True, order="C")
            if not bool(np.isfinite(body22).all()):
                raise EmbodyAdapterError(
                    "SMPL-X evaluator output is not representable as finite float32"
                )
            body22[~observed[start:stop]] = 0.0
            joints[start:stop, actor_index] = body22

    ordered = tuple(sorted(actors, key=lambda actor: actor.actor_commitment))
    source = hashlib.sha256(b"phaseset-embody-group-source-v1\x00")
    source.update(evaluator_digest.encode("ascii"))
    for actor in ordered:
        source.update(actor.actor_commitment)
        source.update(_actor_source_digest(actor))
    return PrivateCaptureArrays(
        joints=np.ascontiguousarray(joints),
        track_mask=np.ascontiguousarray(track_mask),
        actor_commitments=commitments,
        source_sha256=source.hexdigest(),
    )


__all__ = [
    "ABSOLUTE_MAX_ACTOR_FILE_BYTES",
    "BETAS",
    "BODY_POSE",
    "DEFAULT_EVALUATOR_BATCH_FRAMES",
    "DEFAULT_MAX_ACTOR_FILE_BYTES",
    "EMBODY_FEATURE_KEYS",
    "EmbodyActorArrays",
    "EmbodyAdapterError",
    "GLOBAL_ORIENT",
    "LEFT_HAND_POSE",
    "LICENSED_SMPLX_CONFIG",
    "LicensedSMPLXEvaluator",
    "MAX_ACTOR_FRAMES",
    "MAX_EVALUATOR_BATCH_FRAMES",
    "MAX_SMPLX_MODEL_BYTES",
    "MISSING",
    "RIGHT_HAND_POSE",
    "SMPLX_BODY22_NAMES",
    "SMPLXBodyJointEvaluator",
    "TRANSL",
    "embody_to_private_capture",
    "load_embody_actor_npy",
]
