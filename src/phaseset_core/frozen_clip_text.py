"""Frozen, offline CLIP text projection adapter for PhaseSet.

The adapter owns no optimizer, dataset discovery, download, split, or training
surface.  It turns an ordered tuple of captions into the official pretrained
CLIP text projection, returned as finite contiguous CPU float32 ``[Q, 512]``.
Caption commitments are carried only as lineage and never enter tokenization or
the model.  All public errors are closed codes so private paths or text cannot be
accidentally serialized by a caller.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import http.client
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import stat
import sys
import tempfile
import threading
from typing import Any, Callable, Final, Iterator
import urllib.request
import warnings


MODEL_ID: Final = "openai/clip-vit-base-patch32"
REVISION: Final = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
METHOD_ID: Final = "phaseset-clip-native-truncate-77-pretrained-projection-v1"
EMBEDDING_DIM: Final = 512
TOKEN_LENGTH: Final = 77
TOKEN_VOCAB_SIZE: Final = 49_408
TOKEN_BOS_ID: Final = 49_406
TOKEN_EOS_ID: Final = 49_407
TOKEN_PAD_ID: Final = TOKEN_EOS_ID
MAX_BATCH_SIZE: Final = 256
MAX_CAPTIONS: Final = 16_384
MAX_CAPTION_UTF8_BYTES: Final = 16_384
MAX_TOTAL_CAPTION_UTF8_BYTES: Final = 16 * 1024 * 1024
_MAX_REHYDRATION_RECEIPT_BYTES: Final = 32 * 1024 * 1024
AUTHORITY: Final = 0
PRODUCTION: Final = False
TRAINING_AUTHORIZED: Final = False

PINNED_FILES: Final = (
    (
        "pytorch_model.bin",
        605_247_071,
        "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    ),
    (
        "config.json",
        4_186,
        "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    ),
    (
        "merges.txt",
        524_657,
        "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
    ),
    (
        "preprocessor_config.json",
        316,
        "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    ),
    (
        "special_tokens_map.json",
        389,
        "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
    ),
    (
        "tokenizer_config.json",
        592,
        "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
    ),
    (
        "tokenizer.json",
        2_224_041,
        "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
    ),
    (
        "vocab.json",
        862_328,
        "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
    ),
)
_OFFICIAL_PINNED_FILES: Final = PINNED_FILES
TOTAL_SNAPSHOT_BYTES: Final = 608_863_580

_EXPECTED_RUNTIME: Final = (
    ("cuda_available", "false"),
    ("cuda_visible_devices", "hidden"),
    ("deterministic_algorithms", "true"),
    ("device", "cpu"),
    ("float32_matmul_precision", "highest"),
    ("huggingface_hub", "0.36.2"),
    ("interop_threads", "1"),
    ("intraop_threads", "1"),
    ("machine", "x86_64"),
    ("mha_fastpath_enabled", "false"),
    ("numpy", "2.4.6"),
    ("python", "3.12.12"),
    ("python_implementation", "CPython"),
    ("sys_byteorder", "little"),
    ("system", "Linux"),
    ("tokenizers", "0.22.2"),
    ("torch", "2.12.0+cu126"),
    ("torch_cuda_build", "12.6"),
    ("transformers", "4.57.3"),
)
_EXPECTED_CONFIG: Final = (
    ("attention_dropout", 0.0),
    ("bos_token_id", 0),
    ("eos_token_id", 2),
    ("hidden_act", "quick_gelu"),
    ("hidden_size", 512),
    ("intermediate_size", 2048),
    ("layer_norm_eps", 1e-05),
    ("max_position_embeddings", 77),
    ("model_type", "clip_text_model"),
    ("num_attention_heads", 8),
    ("num_hidden_layers", 12),
    ("pad_token_id", 1),
    ("projection_dim", 512),
    ("vocab_size", 49_408),
)
_EXPECTED_TOKENIZER: Final = (
    ("bos_token_id", TOKEN_BOS_ID),
    ("eos_token_id", TOKEN_EOS_ID),
    ("model_max_length", TOKEN_LENGTH),
    ("pad_token_id", TOKEN_PAD_ID),
    ("padding_side", "right"),
    ("truncation_side", "right"),
    ("unk_token_id", TOKEN_EOS_ID),
)
_EXPECTED_UNEXPECTED_COUNT: Final = 202
_EXPECTED_UNEXPECTED_SHA256: Final = (
    "354d27b4b81e1f6e9fc9e908b04fefd182847eeb35c980176b2260a8b572f597"
)
_EXPECTED_PARAMETER_COUNT: Final = 197
_EXPECTED_BUFFER_COUNT: Final = 1
_EXPECTED_SOURCE_FILES: Final = (
    (
        "transformers.modeling_clip",
        50_315,
        "3e779a99827618d6e6d57583ec4f5c0d765321dfa5054a4ffca6a56bb004d499",
    ),
    (
        "transformers.tokenization_clip_fast",
        6_766,
        "67dff0f21a56d0dc18a45f8764f3fafa02ee2b0aeafeb70761c9623f9114dd53",
    ),
    (
        "phaseset_core.training",
        175_962,
        "168d94a87ba328ddccbb86595e62fdbdbe7910d59cd4f1c10b6089ccab54827c",
    ),
)

_CONSTRUCTION_SEAL = object()
_GLOBAL_LOAD_LOCK = threading.Lock()


class FrozenClipTextAdapterError(ValueError):
    """Closed, path-free failure from the adapter boundary."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or not code.startswith("HOLD_"):
            raise TypeError("adapter error code must be a closed HOLD code")
        self.code = code
        super().__init__(code)


def _hold(code: str) -> None:
    raise FrozenClipTextAdapterError(code)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError:
        _hold("HOLD_FILE_UNREADABLE")
    return digest.hexdigest()


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & flag)


def _safe_absolute_directory(value: object) -> Path:
    try:
        supplied = Path(os.fspath(value))
        resolved = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except (TypeError, ValueError, OSError, RuntimeError):
        _hold("HOLD_SNAPSHOT_ROOT")
    if (
        not supplied.is_absolute()
        or os.path.normcase(os.path.abspath(os.fspath(supplied)))
        != os.path.normcase(os.fspath(resolved))
        or not stat.S_ISDIR(metadata.st_mode)
        or _is_reparse(metadata)
    ):
        _hold("HOLD_SNAPSHOT_ROOT")
    return resolved


def _scan_snapshot(
    root: Path,
    pins: tuple[tuple[str, int, str], ...] = PINNED_FILES,
) -> tuple[tuple[str, int, str], ...]:
    expected = {name: (size, digest) for name, size, digest in pins}
    if len(expected) != len(pins) or len({name.casefold() for name in expected}) != len(pins):
        _hold("HOLD_INTERNAL_PIN_SCHEMA")
    try:
        children = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    except OSError:
        _hold("HOLD_SNAPSHOT_CENSUS")
    names = tuple(path.name for path in children)
    if names != tuple(sorted(expected)) or len({name.casefold() for name in names}) != len(names):
        _hold("HOLD_SNAPSHOT_CENSUS")
    rows: list[tuple[str, int, str]] = []
    for child in children:
        try:
            metadata = child.lstat()
            expected_size, expected_digest = expected[child.name]
            resolved = child.resolve(strict=True)
        except (KeyError, OSError, RuntimeError):
            _hold("HOLD_SNAPSHOT_FILE_IDENTITY")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size != expected_size
            or resolved.parent != root
        ):
            _hold("HOLD_SNAPSHOT_FILE_IDENTITY")
        digest = _sha256_file(child)
        if digest != expected_digest:
            _hold("HOLD_SNAPSHOT_FILE_BYTES")
        rows.append((child.name, expected_size, digest))
    if sum(size for _name, size, _digest in rows) != sum(size for _name, size, _digest in pins):
        _hold("HOLD_SNAPSHOT_TOTAL_BYTES")
    return tuple(rows)


def _snapshot_manifest_sha256(rows: tuple[tuple[str, int, str], ...]) -> str:
    payload = {
        "files": [{"bytes": size, "name": name, "sha256": digest} for name, size, digest in rows],
        "model_id": MODEL_ID,
        "revision": REVISION,
        "schema": "phaseset-frozen-clip-snapshot-v1",
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _source_row(
    label: str,
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, int, str]:
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except (OSError, RuntimeError):
        _hold("HOLD_SOURCE_IDENTITY")
    if (
        resolved != path
        or not stat.S_ISREG(metadata.st_mode)
        or _is_reparse(metadata)
        or metadata.st_nlink != 1
    ):
        _hold("HOLD_SOURCE_IDENTITY")
    digest = _sha256_file(path)
    if (expected_bytes is not None and metadata.st_size != expected_bytes) or (
        expected_sha256 is not None and digest != expected_sha256
    ):
        _hold("HOLD_SOURCE_IDENTITY")
    return label, metadata.st_size, digest


def _adapter_source_row() -> tuple[str, int, str]:
    try:
        path = Path(__file__)
        if not path.is_absolute():
            _hold("HOLD_ADAPTER_SOURCE_IDENTITY")
    except (TypeError, ValueError, OSError, RuntimeError):
        _hold("HOLD_ADAPTER_SOURCE_IDENTITY")
    return _source_row("phaseset_core.frozen_clip_text", path)


@dataclass(frozen=True, slots=True)
class _Backend:
    torch: Any
    tokenizer_type: type[Any]
    model_type: type[Any]
    batch_encoding_type: type[Any]
    runtime_identity: tuple[tuple[str, str], ...]
    source_rows: tuple[tuple[str, int, str], ...]
    expected_config: tuple[tuple[str, object], ...]
    unexpected_count: int
    unexpected_sha256: str
    expected_parameter_count: int
    expected_buffer_count: int
    validate_training_boundary: Callable[[Any], None]
    validate_runtime: Callable[[], None]


def _runtime_facts(
    *,
    huggingface_hub: Any,
    numpy: Any,
    tokenizers: Any,
    torch: Any,
    transformers: Any,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "cuda_available": str(bool(torch.cuda.is_available())).lower(),
                "cuda_visible_devices": (
                    "hidden" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else "not-hidden"
                ),
                "deterministic_algorithms": str(
                    bool(torch.are_deterministic_algorithms_enabled())
                ).lower(),
                "device": "cpu",
                "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
                "huggingface_hub": str(huggingface_hub.__version__),
                "interop_threads": str(torch.get_num_interop_threads()),
                "intraop_threads": str(torch.get_num_threads()),
                "machine": platform.machine(),
                "mha_fastpath_enabled": str(
                    bool(torch.backends.mha.get_fastpath_enabled())
                ).lower(),
                "numpy": str(numpy.__version__),
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "sys_byteorder": sys.byteorder,
                "system": platform.system(),
                "tokenizers": str(tokenizers.__version__),
                "torch": str(torch.__version__),
                "torch_cuda_build": str(torch.version.cuda),
                "transformers": str(transformers.__version__),
            }.items()
        )
    )


def _real_backend() -> _Backend:
    try:
        import huggingface_hub
        import numpy
        import tokenizers
        import torch
        import transformers
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
        from transformers.tokenization_utils_base import BatchEncoding
        from phaseset_core import training
    except Exception:
        _hold("HOLD_RUNTIME_IMPORT")

    def validate_runtime() -> None:
        observed = _runtime_facts(
            huggingface_hub=huggingface_hub,
            numpy=numpy,
            tokenizers=tokenizers,
            torch=torch,
            transformers=transformers,
        )
        if (
            observed != _EXPECTED_RUNTIME
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or torch.cuda.is_available() is not False
            or torch.get_num_threads() != 1
            or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
            or torch.get_float32_matmul_precision() != "highest"
            or torch.backends.mha.get_fastpath_enabled() is not False
        ):
            _hold("HOLD_RUNTIME_IDENTITY")

    validate_runtime()
    try:
        modeling_path = Path(inspect.getsourcefile(CLIPTextModelWithProjection) or "")
        tokenizer_path = Path(inspect.getsourcefile(CLIPTokenizerFast) or "")
        training_path = Path(str(training.__file__))
    except (TypeError, ValueError):
        _hold("HOLD_SOURCE_IDENTITY")
    paths = (modeling_path, tokenizer_path, training_path)
    source_rows = tuple(
        _source_row(label, path, expected_bytes=size, expected_sha256=digest)
        for (label, size, digest), path in zip(_EXPECTED_SOURCE_FILES, paths, strict=True)
    )
    if training.REGISTERED_EMBEDDING_DIM != EMBEDDING_DIM:
        _hold("HOLD_TRAINING_BOUNDARY")

    def validate_training_boundary(value: Any) -> None:
        training._validate_embedding_pair(value, value, EMBEDDING_DIM)

    return _Backend(
        torch=torch,
        tokenizer_type=CLIPTokenizerFast,
        model_type=CLIPTextModelWithProjection,
        batch_encoding_type=BatchEncoding,
        runtime_identity=_EXPECTED_RUNTIME,
        source_rows=source_rows,
        expected_config=_EXPECTED_CONFIG,
        unexpected_count=_EXPECTED_UNEXPECTED_COUNT,
        unexpected_sha256=_EXPECTED_UNEXPECTED_SHA256,
        expected_parameter_count=_EXPECTED_PARAMETER_COUNT,
        expected_buffer_count=_EXPECTED_BUFFER_COUNT,
        validate_training_boundary=validate_training_boundary,
        validate_runtime=validate_runtime,
    )


@contextmanager
def _offline_loader_environment() -> Iterator[tuple[Path, list[str]]]:
    names = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
    )
    previous = {name: os.environ.get(name) for name in names}
    attempts: list[str] = []
    old_create_connection = socket.create_connection
    old_urlopen = urllib.request.urlopen
    old_http_connect = http.client.HTTPConnection.connect
    old_https_connect = http.client.HTTPSConnection.connect

    def blocked(*_args: object, **_kwargs: object) -> None:
        attempts.append("blocked")
        raise OSError("offline")

    with tempfile.TemporaryDirectory(prefix="phaseset-frozen-clip-") as cache_text:
        cache_root = Path(cache_text)
        try:
            os.environ.update(
                {
                    "HF_HOME": cache_text,
                    "HF_HUB_CACHE": cache_text,
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
            socket.create_connection = blocked
            urllib.request.urlopen = blocked
            http.client.HTTPConnection.connect = blocked
            http.client.HTTPSConnection.connect = blocked
            yield cache_root, attempts
        finally:
            socket.create_connection = old_create_connection
            urllib.request.urlopen = old_urlopen
            http.client.HTTPConnection.connect = old_http_connect
            http.client.HTTPSConnection.connect = old_https_connect
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _loading_info_digest(unexpected: tuple[str, ...]) -> str:
    return hashlib.sha256("".join(key + "\n" for key in unexpected).encode("ascii")).hexdigest()


def _validate_loading_info(info: object, backend: _Backend) -> None:
    if type(info) is not dict:
        _hold("HOLD_LOADING_INFO")
    try:
        unexpected = tuple(sorted(info.get("unexpected_keys", ())))
        missing = tuple(info.get("missing_keys", ()))
        mismatched = tuple(info.get("mismatched_keys", ()))
        errors = tuple(info.get("error_msgs", ()))
    except (AttributeError, TypeError):
        _hold("HOLD_LOADING_INFO")
    if (
        any(type(value) is not str for value in unexpected)
        or missing
        or mismatched
        or errors
        or len(unexpected) != backend.unexpected_count
        or _loading_info_digest(unexpected) != backend.unexpected_sha256
        or any(
            key != "logit_scale"
            and not key.startswith("vision_model.")
            and not key.startswith("visual_projection.")
            for key in unexpected
        )
    ):
        _hold("HOLD_LOADING_INFO")


def _config_facts(model: object) -> tuple[tuple[str, object], ...]:
    try:
        config = model.config
        facts = {
            "attention_dropout": float(config.attention_dropout),
            "bos_token_id": int(config.bos_token_id),
            "eos_token_id": int(config.eos_token_id),
            "hidden_act": str(config.hidden_act),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "layer_norm_eps": float(config.layer_norm_eps),
            "max_position_embeddings": int(config.max_position_embeddings),
            "model_type": str(config.model_type),
            "num_attention_heads": int(config.num_attention_heads),
            "num_hidden_layers": int(config.num_hidden_layers),
            "pad_token_id": int(config.pad_token_id),
            "projection_dim": int(config.projection_dim),
            "vocab_size": int(config.vocab_size),
        }
    except (AttributeError, TypeError, ValueError, OverflowError):
        _hold("HOLD_MODEL_CONFIG")
    return tuple(sorted(facts.items()))


def _tokenizer_facts(tokenizer: object) -> tuple[tuple[str, object], ...]:
    try:
        facts = {
            "bos_token_id": int(tokenizer.bos_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
            "model_max_length": int(tokenizer.model_max_length),
            "pad_token_id": int(tokenizer.pad_token_id),
            "padding_side": str(tokenizer.padding_side),
            "truncation_side": str(tokenizer.truncation_side),
            "unk_token_id": int(tokenizer.unk_token_id),
        }
    except (AttributeError, TypeError, ValueError, OverflowError):
        _hold("HOLD_TOKENIZER_CONFIG")
    return tuple(sorted(facts.items()))


def _tensor_bytes(tensor: Any) -> bytes:
    try:
        return tensor.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
    except Exception:
        _hold("HOLD_TENSOR_BYTES")


def _cpu_rng_state(torch: Any) -> Any:
    try:
        return torch.get_rng_state().clone()
    except Exception:
        _hold("HOLD_RUNTIME_RNG")


@contextmanager
def _preserve_cpu_rng(torch: Any) -> Iterator[None]:
    """Detect CPU RNG use while restoring caller state on every exit path."""

    caller_state = _cpu_rng_state(torch)
    restored_manually = False
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            guarded_state = _cpu_rng_state(torch)
            yield
            try:
                unchanged = bool(torch.equal(guarded_state, torch.get_rng_state()))
            except Exception:
                _hold("HOLD_RUNTIME_RNG")
            if not unchanged:
                _hold("HOLD_RUNTIME_RNG")
    finally:
        try:
            restored = bool(torch.equal(caller_state, torch.get_rng_state()))
            if not restored:
                torch.set_rng_state(caller_state)
                restored_manually = True
        except Exception:
            raise FrozenClipTextAdapterError("HOLD_RUNTIME_RNG") from None
        if restored_manually:
            _hold("HOLD_RUNTIME_RNG")


def _live_model_manifest(model: object, backend: _Backend) -> str:
    torch = backend.torch
    try:
        parameters = tuple(model.named_parameters())
        buffers = tuple(model.named_buffers())
        modules = tuple(model.named_modules())
    except Exception:
        _hold("HOLD_MODEL_REGISTRY")
    if (
        len(parameters) != backend.expected_parameter_count
        or len(buffers) != backend.expected_buffer_count
        or any("vision" in name.casefold() for name, _module in modules)
    ):
        _hold("HOLD_MODEL_REGISTRY")
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for kind, values in (("parameter", parameters), ("buffer", buffers)):
        for index, (name, tensor) in enumerate(values):
            if (
                type(name) is not str
                or not name
                or id(tensor) in seen
                or type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                or tensor.device.type != "cpu"
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all().item())
            ):
                _hold("HOLD_MODEL_REGISTRY")
            seen.add(id(tensor))
            if kind == "parameter" and (
                type(tensor) is not torch.nn.Parameter
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad is not None
            ):
                _hold("HOLD_MODEL_NOT_FROZEN")
            rows.append(
                {
                    "dtype": str(tensor.dtype),
                    "index": index,
                    "kind": kind,
                    "name": name,
                    "numel": int(tensor.numel()),
                    "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                }
            )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _source_manifest_sha256(rows: tuple[tuple[str, int, str], ...]) -> str:
    return hashlib.sha256(
        _canonical_json(
            [{"bytes": size, "label": label, "sha256": digest} for label, size, digest in rows]
        )
    ).hexdigest()


def _runtime_manifest_sha256(rows: tuple[tuple[str, str], ...]) -> str:
    return hashlib.sha256(_canonical_json(dict(rows))).hexdigest()


def _load_components(root: Path, backend: _Backend) -> tuple[object, object]:
    torch = backend.torch
    try:
        rng_before = torch.get_rng_state().clone()
    except Exception:
        _hold("HOLD_RUNTIME_RNG")
    with _GLOBAL_LOAD_LOCK:
        with _offline_loader_environment() as (cache_root, attempts):
            try:
                with torch.random.fork_rng(devices=[], enabled=True):
                    torch.manual_seed(0)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        tokenizer = backend.tokenizer_type.from_pretrained(
                            str(root),
                            cache_dir=str(cache_root),
                            local_files_only=True,
                            trust_remote_code=False,
                        )
                        model, loading_info = backend.model_type.from_pretrained(
                            str(root),
                            cache_dir=str(cache_root),
                            local_files_only=True,
                            trust_remote_code=False,
                            low_cpu_mem_usage=False,
                            output_loading_info=True,
                        )
            except FrozenClipTextAdapterError:
                raise
            except Exception:
                _hold("HOLD_MODEL_LOAD")
            if attempts or tuple(cache_root.rglob("*")):
                _hold("HOLD_NETWORK_OR_CACHE_ACTIVITY")
            if caught:
                _hold("HOLD_MODEL_LOAD_WARNING")
    if not bool(torch.equal(rng_before, torch.get_rng_state())):
        _hold("HOLD_RUNTIME_RNG")
    if type(tokenizer) is not backend.tokenizer_type or type(model) is not backend.model_type:
        _hold("HOLD_MODEL_TYPE")
    _validate_loading_info(loading_info, backend)
    if _config_facts(model) != backend.expected_config:
        _hold("HOLD_MODEL_CONFIG")
    if _tokenizer_facts(tokenizer) != _EXPECTED_TOKENIZER:
        _hold("HOLD_TOKENIZER_CONFIG")
    return tokenizer, model


def _freeze_model(model: object, backend: _Backend) -> None:
    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except Exception:
        _hold("HOLD_MODEL_NOT_FROZEN")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        _hold("HOLD_MODEL_NOT_FROZEN")


def _validate_captions_and_lineage(
    captions: object,
    commitments: object,
) -> tuple[tuple[str, ...], tuple[bytes, ...], tuple[str, ...]]:
    if type(captions) is not tuple or not 1 <= len(captions) <= MAX_CAPTIONS:
        _hold("HOLD_CAPTION_BATCH")
    if type(commitments) is not tuple or len(commitments) != len(captions):
        _hold("HOLD_CAPTION_LINEAGE")
    encoded_rows: list[bytes] = []
    checked_commitments: list[bytes] = []
    for caption, commitment in zip(captions, commitments, strict=True):
        if type(caption) is not str or not caption or "\x00" in caption:
            _hold("HOLD_CAPTION_VALUE")
        try:
            encoded = caption.encode("utf-8", errors="strict")
        except UnicodeError:
            _hold("HOLD_CAPTION_VALUE")
        if not 1 <= len(encoded) <= MAX_CAPTION_UTF8_BYTES:
            _hold("HOLD_CAPTION_VALUE")
        if type(commitment) is not bytes or len(commitment) != 32:
            _hold("HOLD_CAPTION_LINEAGE")
        encoded_rows.append(encoded)
        checked_commitments.append(commitment)
    if sum(map(len, encoded_rows)) > MAX_TOTAL_CAPTION_UTF8_BYTES:
        _hold("HOLD_CAPTION_BATCH")
    checked_lineage = tuple(checked_commitments)
    if len(set(checked_lineage)) != len(checked_lineage):
        _hold("HOLD_CAPTION_LINEAGE")
    text_digests = tuple(hashlib.sha256(value).hexdigest() for value in encoded_rows)
    return captions, checked_lineage, text_digests


def _validate_token_batch(value: object, count: int, backend: _Backend) -> None:
    torch = backend.torch
    if type(value) is not backend.batch_encoding_type or set(value) != {
        "input_ids",
        "attention_mask",
    }:
        _hold("HOLD_TOKEN_BATCH")
    for name in ("input_ids", "attention_mask"):
        tensor = value[name]
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.int64
            or tuple(tensor.shape) != (count, TOKEN_LENGTH)
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
        ):
            _hold("HOLD_TOKEN_BATCH")
    input_ids = value["input_ids"]
    attention_mask = value["attention_mask"]
    if not bool(((input_ids >= 0) & (input_ids < TOKEN_VOCAB_SIZE)).all().item()) or not bool(
        ((attention_mask == 0) | (attention_mask == 1)).all().item()
    ):
        _hold("HOLD_TOKEN_BATCH")


def _untruncated_token_rows(
    value: object,
    count: int,
    backend: _Backend,
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not backend.batch_encoding_type or set(value) != {"input_ids"}:
        _hold("HOLD_TOKEN_LENGTH_PROBE")
    rows = value["input_ids"]
    if type(rows) is not list or len(rows) != count:
        _hold("HOLD_TOKEN_LENGTH_PROBE")
    checked_rows: list[tuple[int, ...]] = []
    for row in rows:
        if (
            type(row) is not list
            or not row
            or row[0] != TOKEN_BOS_ID
            or row[-1] != TOKEN_EOS_ID
            or any(type(token) is not int or not 0 <= token < TOKEN_VOCAB_SIZE for token in row)
        ):
            _hold("HOLD_TOKEN_LENGTH_PROBE")
        checked_rows.append(tuple(row))
    return tuple(checked_rows)


def _validate_embedding(value: object, count: int, backend: _Backend) -> Any:
    torch = backend.torch
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or tuple(value.shape) != (count, EMBEDDING_DIM)
        or value.device.type != "cpu"
        or value.requires_grad
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        _hold("HOLD_EMBEDDING_BOUNDARY")
    try:
        backend.validate_training_boundary(value)
    except FrozenClipTextAdapterError:
        raise
    except Exception:
        _hold("HOLD_TRAINING_BOUNDARY")
    return value


@dataclass(frozen=True, slots=True)
class FrozenClipTextReceipt:
    """Path-free, text-free provenance for one ordered embedding batch."""

    batch_size: int
    caption_rows: tuple[tuple[int, str, str, int, int, bool, str, str, str], ...]
    chunk_ranges: tuple[tuple[int, int], ...]
    frozen_embedding_cache_key_sha256: str
    live_model_manifest_sha256: str
    output_bytes: int
    output_sha256: str
    output_shape: tuple[int, int]
    output_stride: tuple[int, int]
    runtime_identity: tuple[tuple[str, str], ...]
    runtime_manifest_sha256: str
    snapshot_files: tuple[tuple[str, int, str], ...]
    snapshot_manifest_sha256: str
    source_files: tuple[tuple[str, int, str], ...]
    source_manifest_sha256: str
    status: str = "ENCODED_AUTHORITY0"
    schema: str = "phaseset-frozen-clip-text-receipt-v1"

    def to_public_dict(self) -> dict[str, object]:
        return {
            "authority": AUTHORITY,
            "batch_size": self.batch_size,
            "caption_count": len(self.caption_rows),
            "caption_rows": [
                {
                    "index": index,
                    "caption_commitment": commitment,
                    "caption_utf8_sha256": text_digest,
                    "original_token_count_with_special_tokens": original_token_count,
                    "encoded_token_count_with_special_tokens": encoded_token_count,
                    "truncated": truncated,
                    "input_ids_int64_sha256": input_ids_digest,
                    "attention_mask_int64_sha256": attention_mask_digest,
                    "embedding_float32_sha256": embedding_digest,
                }
                for (
                    index,
                    commitment,
                    text_digest,
                    original_token_count,
                    encoded_token_count,
                    truncated,
                    input_ids_digest,
                    attention_mask_digest,
                    embedding_digest,
                ) in self.caption_rows
            ],
            "chunk_ranges": [list(value) for value in self.chunk_ranges],
            "embedding_selection": "official_pretrained_text_projection_raw_float32",
            "frozen_embedding_cache_key_sha256": self.frozen_embedding_cache_key_sha256,
            "live_model_manifest_sha256": self.live_model_manifest_sha256,
            "model_id": MODEL_ID,
            "method_id": METHOD_ID,
            "output": {
                "bytes": self.output_bytes,
                "device": "cpu",
                "dtype": "torch.float32",
                "finite": True,
                "requires_grad": False,
                "sha256": self.output_sha256,
                "shape": list(self.output_shape),
                "stride": list(self.output_stride),
            },
            "production": PRODUCTION,
            "revision": REVISION,
            "runtime": dict(self.runtime_identity),
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "schema": self.schema,
            "snapshot": {
                "file_count": len(self.snapshot_files),
                "files": [
                    {"bytes": size, "name": name, "sha256": digest}
                    for name, size, digest in self.snapshot_files
                ],
                "manifest_sha256": self.snapshot_manifest_sha256,
            },
            "source": {
                "files": [
                    {"bytes": size, "label": label, "sha256": digest}
                    for label, size, digest in self.source_files
                ],
                "manifest_sha256": self.source_manifest_sha256,
            },
            "status": self.status,
            "training_authorized": TRAINING_AUTHORIZED,
            "tokenization": {
                "add_special_tokens": True,
                "long_caption_policy": "clip_native_truncate_77",
                "max_tokens_including_special_tokens": TOKEN_LENGTH,
                "padding": "max_length",
                "preserve_terminal_eos": True,
                "truncation": True,
            },
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.to_public_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


class FrozenClipTextBatch:
    """Snapshot-owned frozen embeddings plus immutable lineage and receipt."""

    __slots__ = ("__embeddings", "__lineage", "__receipt")

    def __init__(
        self,
        embeddings: Any,
        lineage: tuple[bytes, ...],
        receipt: FrozenClipTextReceipt
        | FrozenClipTextRowSelectionReceipt
        | FrozenClipTextRowPoolReceipt,
        *,
        _seal: object,
    ) -> None:
        if _seal is not _CONSTRUCTION_SEAL:
            raise TypeError("FrozenClipTextBatch construction is internal")
        self.__embeddings = embeddings.detach().clone().contiguous()
        self.__lineage = lineage
        self.__receipt = receipt

    @property
    def embeddings(self) -> Any:
        return self.__embeddings.detach().clone().contiguous()

    @property
    def caption_commitments(self) -> tuple[bytes, ...]:
        return self.__lineage

    @property
    def receipt(
        self,
    ) -> FrozenClipTextReceipt | FrozenClipTextRowSelectionReceipt | FrozenClipTextRowPoolReceipt:
        return self.__receipt

    def __repr__(self) -> str:
        return (
            f"FrozenClipTextBatch(Q={len(self.__lineage)}, D={EMBEDDING_DIM}, "
            f"receipt_sha256={self.__receipt.sha256!r})"
        )


@dataclass(frozen=True, slots=True)
class FrozenClipTextRowSelectionReceipt:
    """Derived cached rows, explicitly NOT a new encoder invocation.

    The immutable original receipt remains available in memory. Its encoding
    chunks/runtime/model identities describe preparation, not the runtime or
    batching of this selection. Persist the original batch and select again;
    the original receipt rehydrator deliberately rejects this derived schema.
    """

    origin_receipt: FrozenClipTextReceipt
    selected_indices: tuple[int, ...]
    caption_rows: tuple[tuple[int, str, str, int, int, bool, str, str, str], ...]
    output_bytes: int
    output_sha256: str
    output_shape: tuple[int, int]
    output_stride: tuple[int, int]
    schema: str = "phaseset-frozen-clip-row-selection-v1"
    status: str = "ROW_SELECTION_AUTHORITY0"

    def to_public_dict(self) -> dict[str, object]:
        origin = self.origin_receipt
        return {
            "authority": AUTHORITY,
            "production": PRODUCTION,
            "training_authorized": TRAINING_AUTHORIZED,
            "schema": self.schema,
            "status": self.status,
            "operation": "ordered_exact_cached_row_selection_no_encoder_call",
            "origin_receipt_sha256": origin.sha256,
            "origin_frozen_embedding_cache_key_sha256": origin.frozen_embedding_cache_key_sha256,
            "origin_source_manifest_sha256": origin.source_manifest_sha256,
            "origin_runtime_manifest_sha256": origin.runtime_manifest_sha256,
            "origin_snapshot_manifest_sha256": origin.snapshot_manifest_sha256,
            "origin_live_model_manifest_sha256": origin.live_model_manifest_sha256,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "method_id": METHOD_ID,
            "selected_indices": list(self.selected_indices),
            "caption_rows": [list(row) for row in self.caption_rows],
            "output": {
                "bytes": self.output_bytes,
                "sha256": self.output_sha256,
                "shape": list(self.output_shape),
                "stride": list(self.output_stride),
                "dtype": "torch.float32",
                "device": "cpu",
                "requires_grad": False,
            },
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.to_public_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenClipTextRowPoolReceipt:
    """Two original cached row selections, never a new encoder invocation.

    The prefix stays bitwise unchanged. Persist both originals and reselect;
    the original-batch rehydrator deliberately rejects this derived schema.
    Prefix/extra roles are supplied by the caller, not ground-truth authority.
    """

    origin_receipts: tuple[FrozenClipTextReceipt, FrozenClipTextReceipt]
    selected_indices: tuple[tuple[int, ...], tuple[int, ...]]
    prefix_count: int
    caption_rows: tuple[tuple[int, str, str, int, int, bool, str, str, str], ...]
    output_bytes: int
    output_sha256: str
    output_shape: tuple[int, int]
    output_stride: tuple[int, int]
    schema: str = "phaseset-frozen-clip-row-pool-v1"
    status: str = "ROW_POOL_AUTHORITY0"

    def to_public_dict(self) -> dict[str, object]:
        return {
            "authority": AUTHORITY,
            "production": PRODUCTION,
            "training_authorized": TRAINING_AUTHORIZED,
            "schema": self.schema,
            "status": self.status,
            "operation": "ordered_cached_rows_from_two_originals_no_encoder_call",
            "origin_receipt_sha256": [origin.sha256 for origin in self.origin_receipts],
            "selected_indices": [list(indices) for indices in self.selected_indices],
            "prefix_count": self.prefix_count,
            "caption_rows": [list(row) for row in self.caption_rows],
            "model_id": MODEL_ID,
            "revision": REVISION,
            "method_id": METHOD_ID,
            "output": {
                "bytes": self.output_bytes,
                "sha256": self.output_sha256,
                "shape": list(self.output_shape),
                "stride": list(self.output_stride),
                "dtype": "torch.float32",
                "device": "cpu",
                "requires_grad": False,
            },
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.to_public_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


def pool_frozen_clip_text_rows(
    prefix_original: FrozenClipTextBatch,
    extra_original: FrozenClipTextBatch,
    *,
    prefix_indices: tuple[int, ...],
    extra_indices: tuple[int, ...],
) -> FrozenClipTextBatch:
    """Append authenticated cached rows without re-encoding the frozen prefix.

    Both inputs must be original encoded/rehydrated batches of the same live
    model, snapshot and numerical runtime. Their separate source/encoding
    receipts are retained; no synthetic combined encoder receipt is invented.
    """
    import torch

    prefix = select_frozen_clip_text_rows(prefix_original, prefix_indices)
    extra = select_frozen_clip_text_rows(extra_original, extra_indices)
    origins = (prefix_original.receipt, extra_original.receipt)
    for identity in (
        "snapshot_manifest_sha256",
        "live_model_manifest_sha256",
        "runtime_manifest_sha256",
    ):
        if getattr(origins[0], identity) != getattr(origins[1], identity):
            raise ValueError("cached text pool requires the same model, snapshot and runtime")
    lineage = prefix.caption_commitments + extra.caption_commitments
    if len(set(lineage)) != len(lineage):
        raise ValueError("cached text pool cannot repeat a caption commitment")
    embeddings = torch.cat((prefix.embeddings, extra.embeddings)).contiguous()
    raw = _tensor_bytes(embeddings)
    receipt = FrozenClipTextRowPoolReceipt(
        origin_receipts=origins,
        selected_indices=(prefix_indices, extra_indices),
        prefix_count=len(prefix_indices),
        caption_rows=tuple(
            (index, *row[1:])
            for index, row in enumerate(prefix.receipt.caption_rows + extra.receipt.caption_rows)
        ),
        output_bytes=len(raw),
        output_sha256=hashlib.sha256(raw).hexdigest(),
        output_shape=tuple(embeddings.shape),
        output_stride=tuple(embeddings.stride()),
    )
    return FrozenClipTextBatch(embeddings, lineage, receipt, _seal=_CONSTRUCTION_SEAL)


def select_frozen_clip_text_rows(
    original: FrozenClipTextBatch, indices: tuple[int, ...]
) -> FrozenClipTextBatch:
    """Gather from one genuine encoded/rehydrated batch without relabeling.

    Captions and commitments cannot be supplied or rebound. No model, runtime
    probe or RNG call occurs. Nested selections and repeated commitments are
    rejected: callers always address the saved original preparation census.
    This supplies cached numbers, never rights, CF truth or training authority.
    """
    from .capture_validation import _validate_text_batch

    if (
        type(original) is not FrozenClipTextBatch
        or type(original.receipt) is not FrozenClipTextReceipt
    ):
        raise ValueError("row selection requires the original sealed encoded batch")
    embeddings, commitments, _ = _validate_text_batch(original)
    if (
        type(indices) is not tuple
        or not indices
        or any(type(index) is not int or not 0 <= index < len(commitments) for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("selection requires unique in-range exact integer row indices")
    selected = embeddings[list(indices)].contiguous()
    raw = _tensor_bytes(selected)
    receipt = FrozenClipTextRowSelectionReceipt(
        origin_receipt=original.receipt,
        selected_indices=indices,
        caption_rows=tuple(
            (position, *original.receipt.caption_rows[index][1:])
            for position, index in enumerate(indices)
        ),
        output_bytes=len(raw),
        output_sha256=hashlib.sha256(raw).hexdigest(),
        output_shape=tuple(selected.shape),
        output_stride=tuple(selected.stride()),
    )
    return FrozenClipTextBatch(
        selected, tuple(commitments[index] for index in indices), receipt, _seal=_CONSTRUCTION_SEAL
    )


def _rehydration_unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if type(key) is not str or key in value:
            _hold("HOLD_REHYDRATION_RECEIPT_JSON")
        value[key] = item
    return value


def _rehydration_reject_constant(_value: str) -> object:
    _hold("HOLD_REHYDRATION_RECEIPT_JSON")


def _rehydration_closed_dict(
    value: object,
    keys: frozenset[str],
    code: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        _hold(code)
    return value


def _rehydration_sha256(value: object, code: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _hold(code)
    return value


def _rehydration_positive_int(value: object, maximum: int, code: str) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _hold(code)
    return value


def _rehydration_exact_literals(
    value: object,
    expected: dict[str, object],
    code: str,
) -> dict[str, object]:
    checked = _rehydration_closed_dict(value, frozenset(expected), code)
    if any(
        type(checked[key]) is not type(item) or checked[key] != item
        for key, item in expected.items()
    ):
        _hold(code)
    return checked


def _rehydration_caption_rows(
    value: object,
    *,
    count: int,
) -> tuple[
    tuple[tuple[int, str, str, int, int, bool, str, str, str], ...],
    tuple[bytes, ...],
]:
    if type(value) is not list or len(value) != count:
        _hold("HOLD_REHYDRATION_CAPTION_ROWS")
    keys = frozenset(
        {
            "attention_mask_int64_sha256",
            "caption_commitment",
            "caption_utf8_sha256",
            "embedding_float32_sha256",
            "encoded_token_count_with_special_tokens",
            "index",
            "input_ids_int64_sha256",
            "original_token_count_with_special_tokens",
            "truncated",
        }
    )
    rows: list[tuple[int, str, str, int, int, bool, str, str, str]] = []
    lineage: list[bytes] = []
    for index, item in enumerate(value):
        row = _rehydration_closed_dict(
            item,
            keys,
            "HOLD_REHYDRATION_CAPTION_ROWS",
        )
        if type(row["index"]) is not int or row["index"] != index:
            _hold("HOLD_REHYDRATION_CAPTION_ROWS")
        commitment = _rehydration_sha256(
            row["caption_commitment"],
            "HOLD_REHYDRATION_LINEAGE",
        )
        text_digest = _rehydration_sha256(
            row["caption_utf8_sha256"],
            "HOLD_REHYDRATION_CAPTION_ROWS",
        )
        input_digest = _rehydration_sha256(
            row["input_ids_int64_sha256"],
            "HOLD_REHYDRATION_CAPTION_ROWS",
        )
        mask_digest = _rehydration_sha256(
            row["attention_mask_int64_sha256"],
            "HOLD_REHYDRATION_CAPTION_ROWS",
        )
        output_digest = _rehydration_sha256(
            row["embedding_float32_sha256"],
            "HOLD_REHYDRATION_OUTPUT",
        )
        original_count = _rehydration_positive_int(
            row["original_token_count_with_special_tokens"],
            MAX_CAPTION_UTF8_BYTES + 2,
            "HOLD_REHYDRATION_CAPTION_ROWS",
        )
        encoded_count = row["encoded_token_count_with_special_tokens"]
        truncated = row["truncated"]
        if (
            original_count < 2
            or type(encoded_count) is not int
            or encoded_count != min(original_count, TOKEN_LENGTH)
            or type(truncated) is not bool
            or truncated != (original_count > TOKEN_LENGTH)
        ):
            _hold("HOLD_REHYDRATION_CAPTION_ROWS")
        rows.append(
            (
                index,
                commitment,
                text_digest,
                original_count,
                encoded_count,
                truncated,
                input_digest,
                mask_digest,
                output_digest,
            )
        )
        lineage.append(bytes.fromhex(commitment))
    checked_lineage = tuple(lineage)
    if len(set(checked_lineage)) != count:
        _hold("HOLD_REHYDRATION_LINEAGE")
    return tuple(rows), checked_lineage


def _rehydration_chunk_ranges(
    value: object,
    *,
    count: int,
    batch_size: int,
) -> tuple[tuple[int, int], ...]:
    if type(value) is not list:
        _hold("HOLD_REHYDRATION_CHUNKS")
    expected = tuple(
        (start, min(start + batch_size, count)) for start in range(0, count, batch_size)
    )
    checked: list[tuple[int, int]] = []
    for item in value:
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not int
            or type(item[1]) is not int
        ):
            _hold("HOLD_REHYDRATION_CHUNKS")
        checked.append((item[0], item[1]))
    if tuple(checked) != expected:
        _hold("HOLD_REHYDRATION_CHUNKS")
    return tuple(checked)


def _rehydration_snapshot(
    value: object,
) -> tuple[tuple[str, int, str], ...]:
    snapshot = _rehydration_closed_dict(
        value,
        frozenset({"file_count", "files", "manifest_sha256"}),
        "HOLD_REHYDRATION_SNAPSHOT",
    )
    files = snapshot["files"]
    if type(files) is not list or len(files) != len(_OFFICIAL_PINNED_FILES):
        _hold("HOLD_REHYDRATION_SNAPSHOT")
    rows: list[tuple[str, int, str]] = []
    for item in files:
        row = _rehydration_closed_dict(
            item,
            frozenset({"bytes", "name", "sha256"}),
            "HOLD_REHYDRATION_SNAPSHOT",
        )
        name = row["name"]
        size = row["bytes"]
        if type(name) is not str or not name or type(size) is not int or size < 1:
            _hold("HOLD_REHYDRATION_SNAPSHOT")
        rows.append(
            (
                name,
                size,
                _rehydration_sha256(
                    row["sha256"],
                    "HOLD_REHYDRATION_SNAPSHOT",
                ),
            )
        )
    checked = tuple(rows)
    if (
        type(snapshot["file_count"]) is not int
        or snapshot["file_count"] != len(checked)
        or checked != tuple(sorted(_OFFICIAL_PINNED_FILES))
        or _rehydration_sha256(
            snapshot["manifest_sha256"],
            "HOLD_REHYDRATION_SNAPSHOT",
        )
        != _snapshot_manifest_sha256(checked)
    ):
        _hold("HOLD_REHYDRATION_SNAPSHOT")
    return checked


def _rehydration_source(
    value: object,
) -> tuple[tuple[str, int, str], ...]:
    source = _rehydration_closed_dict(
        value,
        frozenset({"files", "manifest_sha256"}),
        "HOLD_REHYDRATION_SOURCE",
    )
    files = source["files"]
    if type(files) is not list or len(files) != 4:
        _hold("HOLD_REHYDRATION_SOURCE")
    rows: list[tuple[str, int, str]] = []
    for item in files:
        row = _rehydration_closed_dict(
            item,
            frozenset({"bytes", "label", "sha256"}),
            "HOLD_REHYDRATION_SOURCE",
        )
        label = row["label"]
        size = row["bytes"]
        if type(label) is not str or not label or type(size) is not int or size < 1:
            _hold("HOLD_REHYDRATION_SOURCE")
        rows.append(
            (
                label,
                size,
                _rehydration_sha256(
                    row["sha256"],
                    "HOLD_REHYDRATION_SOURCE",
                ),
            )
        )
    checked = tuple(rows)
    if (
        checked[:2] != _EXPECTED_SOURCE_FILES[:2]
        or checked[2][0] != "phaseset_core.training"
        or checked[3][0] != "phaseset_core.frozen_clip_text"
        or _rehydration_sha256(
            source["manifest_sha256"],
            "HOLD_REHYDRATION_SOURCE",
        )
        != _source_manifest_sha256(checked)
    ):
        _hold("HOLD_REHYDRATION_SOURCE")
    return checked


def _rehydration_runtime(value: object) -> tuple[tuple[str, str], ...]:
    runtime = _rehydration_closed_dict(
        value,
        frozenset(key for key, _item in _EXPECTED_RUNTIME),
        "HOLD_REHYDRATION_RUNTIME",
    )
    if any(type(item) is not str for item in runtime.values()):
        _hold("HOLD_REHYDRATION_RUNTIME")
    checked = tuple(sorted(runtime.items()))
    if checked != _EXPECTED_RUNTIME:
        _hold("HOLD_REHYDRATION_RUNTIME")
    return checked


def _rehydration_cache_key(
    *,
    batch_size: int,
    caption_rows: tuple[tuple[int, str, str, int, int, bool, str, str, str], ...],
    chunk_ranges: tuple[tuple[int, int], ...],
    runtime_manifest_sha256: str,
    snapshot_manifest_sha256: str,
    source_manifest_sha256: str,
) -> str:
    payload = {
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
        "chunk_ranges": [list(value) for value in chunk_ranges],
        "method_id": METHOD_ID,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "token_length": TOKEN_LENGTH,
        "truncation": True,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _rehydrate_frozen_clip_text_batch(
    embeddings: object,
    *,
    receipt_json_bytes: object,
    expected_receipt_sha256: object,
) -> FrozenClipTextBatch:
    if (
        type(receipt_json_bytes) is not bytes
        or not 1 <= len(receipt_json_bytes) <= _MAX_REHYDRATION_RECEIPT_BYTES
    ):
        _hold("HOLD_REHYDRATION_RECEIPT_BYTES")
    expected_digest = _rehydration_sha256(
        expected_receipt_sha256,
        "HOLD_REHYDRATION_EXPECTED_RECEIPT",
    )
    if hashlib.sha256(receipt_json_bytes).hexdigest() != expected_digest:
        _hold("HOLD_REHYDRATION_EXPECTED_RECEIPT")
    try:
        parsed = json.loads(
            receipt_json_bytes.decode("ascii"),
            object_pairs_hook=_rehydration_unique_object,
            parse_constant=_rehydration_reject_constant,
        )
    except FrozenClipTextAdapterError:
        raise
    except (UnicodeError, TypeError, ValueError, OverflowError):
        _hold("HOLD_REHYDRATION_RECEIPT_JSON")
    if receipt_json_bytes != _canonical_json(parsed) + b"\n":
        _hold("HOLD_REHYDRATION_RECEIPT_JSON")
    receipt_value = _rehydration_closed_dict(
        parsed,
        frozenset(
            {
                "authority",
                "batch_size",
                "caption_count",
                "caption_rows",
                "chunk_ranges",
                "embedding_selection",
                "frozen_embedding_cache_key_sha256",
                "live_model_manifest_sha256",
                "method_id",
                "model_id",
                "output",
                "production",
                "revision",
                "runtime",
                "runtime_manifest_sha256",
                "schema",
                "snapshot",
                "source",
                "status",
                "tokenization",
                "training_authorized",
            }
        ),
        "HOLD_REHYDRATION_RECEIPT_SCHEMA",
    )
    literals = {
        "authority": AUTHORITY,
        "embedding_selection": "official_pretrained_text_projection_raw_float32",
        "method_id": METHOD_ID,
        "model_id": MODEL_ID,
        "production": PRODUCTION,
        "revision": REVISION,
        "schema": "phaseset-frozen-clip-text-receipt-v1",
        "status": "ENCODED_AUTHORITY0",
        "training_authorized": TRAINING_AUTHORIZED,
    }
    if any(
        type(receipt_value[key]) is not type(item) or receipt_value[key] != item
        for key, item in literals.items()
    ):
        _hold("HOLD_REHYDRATION_RECEIPT_SCHEMA")
    _rehydration_exact_literals(
        receipt_value["tokenization"],
        {
            "add_special_tokens": True,
            "long_caption_policy": "clip_native_truncate_77",
            "max_tokens_including_special_tokens": TOKEN_LENGTH,
            "padding": "max_length",
            "preserve_terminal_eos": True,
            "truncation": True,
        },
        "HOLD_REHYDRATION_TOKENIZATION",
    )
    caption_rows_value = receipt_value["caption_rows"]
    if type(caption_rows_value) is not list:
        _hold("HOLD_REHYDRATION_CAPTION_ROWS")
    count = len(caption_rows_value)
    if not 1 <= count <= MAX_CAPTIONS:
        _hold("HOLD_REHYDRATION_CAPTION_ROWS")
    if type(receipt_value["caption_count"]) is not int or receipt_value["caption_count"] != count:
        _hold("HOLD_REHYDRATION_CAPTION_ROWS")
    batch_size = _rehydration_positive_int(
        receipt_value["batch_size"],
        MAX_BATCH_SIZE,
        "HOLD_REHYDRATION_CHUNKS",
    )
    caption_rows, lineage = _rehydration_caption_rows(
        caption_rows_value,
        count=count,
    )
    chunk_ranges = _rehydration_chunk_ranges(
        receipt_value["chunk_ranges"],
        count=count,
        batch_size=batch_size,
    )
    output = _rehydration_closed_dict(
        receipt_value["output"],
        frozenset(
            {
                "bytes",
                "device",
                "dtype",
                "finite",
                "requires_grad",
                "sha256",
                "shape",
                "stride",
            }
        ),
        "HOLD_REHYDRATION_OUTPUT",
    )
    if (
        type(output["bytes"]) is not int
        or output["bytes"] != count * EMBEDDING_DIM * 4
        or type(output["device"]) is not str
        or output["device"] != "cpu"
        or type(output["dtype"]) is not str
        or output["dtype"] != "torch.float32"
        or type(output["finite"]) is not bool
        or output["finite"] is not True
        or type(output["requires_grad"]) is not bool
        or output["requires_grad"] is not False
        or type(output["shape"]) is not list
        or output["shape"] != [count, EMBEDDING_DIM]
        or any(type(item) is not int for item in output["shape"])
        or type(output["stride"]) is not list
        or output["stride"] != [EMBEDDING_DIM, 1]
        or any(type(item) is not int for item in output["stride"])
    ):
        _hold("HOLD_REHYDRATION_OUTPUT")
    output_sha256 = _rehydration_sha256(
        output["sha256"],
        "HOLD_REHYDRATION_OUTPUT",
    )
    try:
        import torch
    except Exception:
        _hold("HOLD_REHYDRATION_RUNTIME_IMPORT")
    if (
        type(embeddings) is not torch.Tensor
        or embeddings.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or tuple(embeddings.shape) != (count, EMBEDDING_DIM)
        or tuple(embeddings.stride()) != (EMBEDDING_DIM, 1)
        or embeddings.requires_grad
        or not embeddings.is_contiguous()
    ):
        _hold("HOLD_REHYDRATION_OUTPUT")
    try:
        frozen_embeddings = embeddings.detach().clone().contiguous()
    except Exception:
        _hold("HOLD_REHYDRATION_OUTPUT")
    if not bool(torch.isfinite(frozen_embeddings).all().item()):
        _hold("HOLD_REHYDRATION_OUTPUT")
    output_bytes = _tensor_bytes(frozen_embeddings)
    if (
        len(output_bytes) != output["bytes"]
        or hashlib.sha256(output_bytes).hexdigest() != output_sha256
    ):
        _hold("HOLD_REHYDRATION_OUTPUT")
    row_bytes = EMBEDDING_DIM * 4
    if any(
        hashlib.sha256(output_bytes[index * row_bytes : (index + 1) * row_bytes]).hexdigest()
        != row[8]
        for index, row in enumerate(caption_rows)
    ):
        _hold("HOLD_REHYDRATION_OUTPUT")
    snapshot_files = _rehydration_snapshot(receipt_value["snapshot"])
    source_files = _rehydration_source(receipt_value["source"])
    runtime_identity = _rehydration_runtime(receipt_value["runtime"])
    snapshot_manifest = _snapshot_manifest_sha256(snapshot_files)
    source_manifest = _source_manifest_sha256(source_files)
    runtime_manifest = _runtime_manifest_sha256(runtime_identity)
    if (
        _rehydration_sha256(
            receipt_value["snapshot"]["manifest_sha256"],
            "HOLD_REHYDRATION_SNAPSHOT",
        )
        != snapshot_manifest
        or _rehydration_sha256(
            receipt_value["source"]["manifest_sha256"],
            "HOLD_REHYDRATION_SOURCE",
        )
        != source_manifest
        or _rehydration_sha256(
            receipt_value["runtime_manifest_sha256"],
            "HOLD_REHYDRATION_RUNTIME",
        )
        != runtime_manifest
    ):
        _hold("HOLD_REHYDRATION_PROVENANCE")
    cache_key = _rehydration_sha256(
        receipt_value["frozen_embedding_cache_key_sha256"],
        "HOLD_REHYDRATION_CACHE_KEY",
    )
    if cache_key != _rehydration_cache_key(
        batch_size=batch_size,
        caption_rows=caption_rows,
        chunk_ranges=chunk_ranges,
        runtime_manifest_sha256=runtime_manifest,
        snapshot_manifest_sha256=snapshot_manifest,
        source_manifest_sha256=source_manifest,
    ):
        _hold("HOLD_REHYDRATION_CACHE_KEY")
    live_model_manifest = _rehydration_sha256(
        receipt_value["live_model_manifest_sha256"],
        "HOLD_REHYDRATION_PROVENANCE",
    )
    receipt = FrozenClipTextReceipt(
        batch_size=batch_size,
        caption_rows=caption_rows,
        chunk_ranges=chunk_ranges,
        frozen_embedding_cache_key_sha256=cache_key,
        live_model_manifest_sha256=live_model_manifest,
        output_bytes=len(output_bytes),
        output_sha256=output_sha256,
        output_shape=(count, EMBEDDING_DIM),
        output_stride=(EMBEDDING_DIM, 1),
        runtime_identity=runtime_identity,
        runtime_manifest_sha256=runtime_manifest,
        snapshot_files=snapshot_files,
        snapshot_manifest_sha256=snapshot_manifest,
        source_files=source_files,
        source_manifest_sha256=source_manifest,
    )
    if receipt.canonical_json_bytes() != receipt_json_bytes or receipt.sha256 != expected_digest:
        _hold("HOLD_REHYDRATION_RECEIPT_REBUILD")
    return FrozenClipTextBatch(
        frozen_embeddings,
        lineage,
        receipt,
        _seal=_CONSTRUCTION_SEAL,
    )


def rehydrate_frozen_clip_text_batch(
    embeddings: Any,
    *,
    receipt_json_bytes: bytes,
    expected_receipt_sha256: str,
) -> FrozenClipTextBatch:
    """Restore one exact cached batch without re-running the CLIP text tower.

    ``expected_receipt_sha256`` must come from an independently authenticated
    prepared-data manifest. Rehydration preserves the original encoding
    source/runtime identities and grants no data, training, or result authority.
    """

    try:
        return _rehydrate_frozen_clip_text_batch(
            embeddings,
            receipt_json_bytes=receipt_json_bytes,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    except FrozenClipTextAdapterError:
        raise
    except Exception:
        raise FrozenClipTextAdapterError("HOLD_UNEXPECTED_REHYDRATION") from None


class FrozenClipTextAdapter:
    """Opaque frozen CLIP text tower loaded only from the pinned local snapshot."""

    __slots__ = (
        "_backend",
        "_lock",
        "_max_batch_size",
        "_model",
        "_model_manifest_sha256",
        "_pins",
        "_snapshot_root",
        "_snapshot_rows",
        "_source_rows",
        "_tokenizer",
    )

    def __init__(
        self,
        *,
        backend: _Backend,
        max_batch_size: int,
        model: object,
        model_manifest_sha256: str,
        pins: tuple[tuple[str, int, str], ...],
        snapshot_root: Path,
        snapshot_rows: tuple[tuple[str, int, str], ...],
        source_rows: tuple[tuple[str, int, str], ...],
        tokenizer: object,
        _seal: object,
    ) -> None:
        if _seal is not _CONSTRUCTION_SEAL:
            raise TypeError("FrozenClipTextAdapter construction is internal")
        self._backend = backend
        self._lock = threading.Lock()
        self._max_batch_size = max_batch_size
        self._model = model
        self._model_manifest_sha256 = model_manifest_sha256
        self._pins = pins
        self._snapshot_root = snapshot_root
        self._snapshot_rows = snapshot_rows
        self._source_rows = source_rows
        self._tokenizer = tokenizer

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def __repr__(self) -> str:
        return (
            f"FrozenClipTextAdapter(model_id={MODEL_ID!r}, revision={REVISION!r}, "
            f"max_batch_size={self._max_batch_size}, authority=0)"
        )

    def encode(
        self,
        captions: tuple[str, ...],
        caption_commitments: tuple[bytes, ...],
        *,
        batch_size: int,
    ) -> FrozenClipTextBatch:
        try:
            return self._encode(captions, caption_commitments, batch_size=batch_size)
        except FrozenClipTextAdapterError:
            raise
        except Exception:
            raise FrozenClipTextAdapterError("HOLD_UNEXPECTED_ENCODE") from None

    def _encode(
        self,
        captions: object,
        caption_commitments: object,
        *,
        batch_size: object,
    ) -> FrozenClipTextBatch:
        checked_captions, lineage, text_digests = _validate_captions_and_lineage(
            captions, caption_commitments
        )
        if (
            type(batch_size) is not int
            or not 1 <= batch_size <= self._max_batch_size
            or batch_size > MAX_BATCH_SIZE
        ):
            _hold("HOLD_BATCH_SIZE")
        torch = self._backend.torch
        with self._lock, _preserve_cpu_rng(torch):
            self._backend.validate_runtime()
            if _adapter_source_row() != self._source_rows[-1]:
                _hold("HOLD_ADAPTER_SOURCE_CHANGED")
            before = _scan_snapshot(self._snapshot_root, self._pins)
            if before != self._snapshot_rows:
                _hold("HOLD_SNAPSHOT_CHANGED")
            if _live_model_manifest(self._model, self._backend) != self._model_manifest_sha256:
                _hold("HOLD_MODEL_CHANGED")
            _freeze_model(self._model, self._backend)
            chunks: list[Any] = []
            token_rows: list[tuple[int, int, bool, str, str]] = []
            chunk_ranges: list[tuple[int, int]] = []
            for start in range(0, len(checked_captions), batch_size):
                stop = min(start + batch_size, len(checked_captions))
                chunk = list(checked_captions[start:stop])
                chunk_ranges.append((start, stop))
                try:
                    untruncated = self._tokenizer(
                        chunk,
                        add_special_tokens=True,
                        padding=False,
                        truncation=False,
                        return_attention_mask=False,
                        verbose=False,
                    )
                    original_token_rows = _untruncated_token_rows(
                        untruncated, stop - start, self._backend
                    )
                    encoded = self._tokenizer(
                        chunk,
                        add_special_tokens=True,
                        max_length=TOKEN_LENGTH,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                except FrozenClipTextAdapterError:
                    raise
                except Exception:
                    _hold("HOLD_TOKENIZE")
                _validate_token_batch(encoded, stop - start, self._backend)
                if _tokenizer_facts(self._tokenizer) != _EXPECTED_TOKENIZER:
                    _hold("HOLD_TOKENIZER_CONFIG")
                eos_token_id = int(self._tokenizer.eos_token_id)
                for local_index, original_tokens in enumerate(original_token_rows):
                    original_token_count = len(original_tokens)
                    input_row = encoded["input_ids"][local_index : local_index + 1].contiguous()
                    mask_row = encoded["attention_mask"][local_index : local_index + 1].contiguous()
                    encoded_token_count = int(mask_row.sum().item())
                    truncated = original_token_count > TOKEN_LENGTH
                    if truncated:
                        expected_tokens = (
                            *original_tokens[: TOKEN_LENGTH - 1],
                            eos_token_id,
                        )
                    else:
                        expected_tokens = (
                            *original_tokens,
                            *(TOKEN_PAD_ID,) * (TOKEN_LENGTH - original_token_count),
                        )
                    expected_input = torch.tensor(
                        expected_tokens,
                        dtype=torch.int64,
                        device="cpu",
                    ).reshape(1, TOKEN_LENGTH)
                    expected_mask = torch.zeros_like(mask_row)
                    expected_mask[0, :encoded_token_count] = 1
                    if (
                        encoded_token_count != min(original_token_count, TOKEN_LENGTH)
                        or not bool(torch.equal(input_row, expected_input))
                        or not bool(torch.equal(mask_row, expected_mask))
                    ):
                        _hold("HOLD_TOKEN_LENGTH_PROBE")
                    token_rows.append(
                        (
                            original_token_count,
                            encoded_token_count,
                            truncated,
                            hashlib.sha256(_tensor_bytes(input_row)).hexdigest(),
                            hashlib.sha256(_tensor_bytes(mask_row)).hexdigest(),
                        )
                    )
                try:
                    with torch.inference_mode():
                        text_output = self._model.text_model(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                        )
                        pooled = text_output.pooler_output
                        projected = self._model.text_projection(pooled).contiguous()
                except Exception:
                    _hold("HOLD_TEXT_FORWARD")
                chunks.append(_validate_embedding(projected, stop - start, self._backend))
            embeddings = torch.cat(chunks, dim=0).contiguous().detach().clone()
            _validate_embedding(embeddings, len(checked_captions), self._backend)
            if _live_model_manifest(self._model, self._backend) != self._model_manifest_sha256:
                _hold("HOLD_MODEL_CHANGED")
            after = _scan_snapshot(self._snapshot_root, self._pins)
            if after != before:
                _hold("HOLD_SNAPSHOT_CHANGED")
            if _adapter_source_row() != self._source_rows[-1]:
                _hold("HOLD_ADAPTER_SOURCE_CHANGED")

        output_bytes = _tensor_bytes(embeddings)
        row_digests = tuple(
            hashlib.sha256(_tensor_bytes(embeddings[index : index + 1].contiguous())).hexdigest()
            for index in range(len(checked_captions))
        )
        caption_rows = tuple(
            (
                index,
                commitment.hex(),
                text_digest,
                original_token_count,
                encoded_token_count,
                truncated,
                input_ids_digest,
                attention_mask_digest,
                output_digest,
            )
            for index, (
                commitment,
                text_digest,
                (
                    original_token_count,
                    encoded_token_count,
                    truncated,
                    input_ids_digest,
                    attention_mask_digest,
                ),
                output_digest,
            ) in enumerate(zip(lineage, text_digests, token_rows, row_digests, strict=True))
        )
        snapshot_manifest = _snapshot_manifest_sha256(self._snapshot_rows)
        source_manifest = _source_manifest_sha256(self._source_rows)
        runtime_manifest = _runtime_manifest_sha256(self._backend.runtime_identity)
        cache_key_payload = {
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
            "chunk_ranges": [list(value) for value in chunk_ranges],
            "method_id": METHOD_ID,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "runtime_manifest_sha256": runtime_manifest,
            "snapshot_manifest_sha256": snapshot_manifest,
            "source_manifest_sha256": source_manifest,
            "token_length": TOKEN_LENGTH,
            "truncation": True,
        }
        cache_key = hashlib.sha256(_canonical_json(cache_key_payload)).hexdigest()
        receipt = FrozenClipTextReceipt(
            batch_size=batch_size,
            caption_rows=caption_rows,
            chunk_ranges=tuple(chunk_ranges),
            frozen_embedding_cache_key_sha256=cache_key,
            live_model_manifest_sha256=self._model_manifest_sha256,
            output_bytes=len(output_bytes),
            output_sha256=hashlib.sha256(output_bytes).hexdigest(),
            output_shape=tuple(embeddings.shape),
            output_stride=tuple(embeddings.stride()),
            runtime_identity=self._backend.runtime_identity,
            runtime_manifest_sha256=runtime_manifest,
            snapshot_files=self._snapshot_rows,
            snapshot_manifest_sha256=snapshot_manifest,
            source_files=self._source_rows,
            source_manifest_sha256=source_manifest,
        )
        return FrozenClipTextBatch(
            embeddings,
            lineage,
            receipt,
            _seal=_CONSTRUCTION_SEAL,
        )


def _load_frozen_clip_text_adapter(
    snapshot_root: object,
    *,
    max_batch_size: object,
    backend: _Backend,
    pins: tuple[tuple[str, int, str], ...],
) -> FrozenClipTextAdapter:
    if type(max_batch_size) is not int or not 1 <= max_batch_size <= MAX_BATCH_SIZE:
        _hold("HOLD_BATCH_SIZE")
    backend.validate_runtime()
    root = _safe_absolute_directory(snapshot_root)
    before = _scan_snapshot(root, pins)
    adapter_source = _adapter_source_row()
    tokenizer, model = _load_components(root, backend)
    _freeze_model(model, backend)
    model_manifest = _live_model_manifest(model, backend)
    after = _scan_snapshot(root, pins)
    if before != after:
        _hold("HOLD_SNAPSHOT_CHANGED")
    if _adapter_source_row() != adapter_source:
        _hold("HOLD_ADAPTER_SOURCE_CHANGED")
    source_rows = backend.source_rows + (adapter_source,)
    return FrozenClipTextAdapter(
        backend=backend,
        max_batch_size=max_batch_size,
        model=model,
        model_manifest_sha256=model_manifest,
        pins=pins,
        snapshot_root=root,
        snapshot_rows=before,
        source_rows=source_rows,
        tokenizer=tokenizer,
        _seal=_CONSTRUCTION_SEAL,
    )


def load_frozen_clip_text_adapter(
    snapshot_root: str | os.PathLike[str],
    *,
    max_batch_size: int,
) -> FrozenClipTextAdapter:
    """Load the fixed text-only CLIP projection from an exact local snapshot.

    The caller must explicitly configure the qualified CPU runtime, including
    hidden CUDA and one deterministic Torch thread.  This function performs no
    download and returns no authority beyond frozen feature preparation.
    """

    try:
        backend = _real_backend()
        return _load_frozen_clip_text_adapter(
            snapshot_root,
            max_batch_size=max_batch_size,
            backend=backend,
            pins=_OFFICIAL_PINNED_FILES,
        )
    except FrozenClipTextAdapterError:
        raise
    except Exception:
        raise FrozenClipTextAdapterError("HOLD_UNEXPECTED_LOAD") from None


__all__ = [
    "AUTHORITY",
    "EMBEDDING_DIM",
    "FrozenClipTextAdapter",
    "FrozenClipTextAdapterError",
    "FrozenClipTextBatch",
    "FrozenClipTextReceipt",
    "FrozenClipTextRowSelectionReceipt",
    "FrozenClipTextRowPoolReceipt",
    "MAX_BATCH_SIZE",
    "MAX_CAPTIONS",
    "METHOD_ID",
    "MODEL_ID",
    "PRODUCTION",
    "REVISION",
    "TRAINING_AUTHORIZED",
    "load_frozen_clip_text_adapter",
    "rehydrate_frozen_clip_text_batch",
    "select_frozen_clip_text_rows",
    "pool_frozen_clip_text_rows",
]
