"""Mock-only tests for the proposed public PhaseSet frozen CLIP adapter.

These tests use Torch tensors but never import Transformers, load a checkpoint,
open a network connection, or run the real CLIP model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from phaseset_core import frozen_clip_text as adapter_module


MODULE_PATH = Path(str(adapter_module.__file__)).resolve(strict=True)


class _Config:
    attention_dropout = 0.0
    bos_token_id = 0
    eos_token_id = 2
    hidden_act = "quick_gelu"
    hidden_size = 512
    intermediate_size = 2048
    layer_norm_eps = 1e-05
    max_position_embeddings = 77
    model_type = "clip_text_model"
    num_attention_heads = 8
    num_hidden_layers = 12
    pad_token_id = 1
    projection_dim = 512
    vocab_size = 49_408


class _Output:
    def __init__(self, pooler_output: torch.Tensor) -> None:
        self.pooler_output = pooler_output


class _TextModel(torch.nn.Module):
    def __init__(self, owner: "_Model") -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> _Output:
        del attention_mask
        self._owner.forward_count += 1
        if self._owner.advance_rng:
            _ = torch.rand(())
        base = input_ids[:, 1:2].to(dtype=torch.float32)
        axis = torch.arange(512, dtype=torch.float32).unsqueeze(0) / 1024.0
        output = base + axis
        if self._owner.return_nonfinite:
            output[0, 0] = torch.nan
        return _Output(output)


class _Model(torch.nn.Module):
    last_model: "_Model | None" = None
    load_calls: list[dict[str, object]] = []

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.advance_rng = False
        self.forward_count = 0
        self.return_nonfinite = False
        self.text_model = _TextModel(self)
        self.text_projection = torch.nn.Linear(512, 512, bias=False)
        with torch.no_grad():
            self.text_projection.weight.copy_(torch.eye(512, dtype=torch.float32))
        self.register_buffer("position_ids", torch.arange(77, dtype=torch.int64))

    @classmethod
    def from_pretrained(cls, _root: str, **kwargs: object) -> tuple["_Model", dict[str, object]]:
        cls.load_calls.append(dict(kwargs))
        model = cls()
        cls.last_model = model
        return model, {
            "unexpected_keys": [],
            "missing_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }


class _Tokenizer:
    load_calls: list[dict[str, object]] = []
    scramble_formal_middle = False
    wrong_terminal_eos = False
    bos_token_id = 49_406
    eos_token_id = 49_407
    model_max_length = 77
    pad_token_id = 49_407
    padding_side = "right"
    truncation_side = "right"
    unk_token_id = 49_407

    @classmethod
    def from_pretrained(cls, _root: str, **kwargs: object) -> "_Tokenizer":
        cls.load_calls.append(dict(kwargs))
        return cls()

    def __call__(self, captions: list[str], **kwargs: object) -> object:
        token_rows = []
        for caption in captions:
            body = [
                int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "little")
                % 49_000
                for word in caption.split()
            ]
            token_rows.append([49_406, *body, 49_407])
        if "return_tensors" not in kwargs:
            assert kwargs == {
                "add_special_tokens": True,
                "padding": False,
                "truncation": False,
                "return_attention_mask": False,
                "verbose": False,
            }
            return {"input_ids": token_rows}
        assert kwargs == {
            "add_special_tokens": True,
            "max_length": 77,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        rows = torch.full((len(captions), 77), 49_407, dtype=torch.int64)
        masks = torch.zeros_like(rows)
        for index, tokens in enumerate(token_rows):
            if len(tokens) > 77:
                tokens = [*tokens[:76], 49_407]
            if self.scramble_formal_middle and len(tokens) > 2:
                tokens[1] = (tokens[1] + 1) % 49_000
            if self.wrong_terminal_eos:
                tokens[-1] = 2
            rows[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.int64)
            masks[index, : len(tokens)] = 1
        return {"input_ids": rows, "attention_mask": masks}


def _commitment(label: str) -> bytes:
    return hashlib.sha256(("lineage:" + label).encode("ascii")).digest()


def _fixture(tmp_path: Path, *, max_batch_size: int = 3):
    _Model.last_model = None
    _Model.load_calls.clear()
    _Tokenizer.load_calls.clear()
    _Tokenizer.scramble_formal_middle = False
    _Tokenizer.wrong_terminal_eos = False
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    contents = {"a.bin": b"official-a", "b.json": b"{}\n"}
    pins = []
    for name, raw in contents.items():
        path = snapshot / name
        path.write_bytes(raw)
        pins.append((name, len(raw), hashlib.sha256(raw).hexdigest()))
    boundary_calls: list[tuple[int, int]] = []

    def validate_boundary(value: torch.Tensor) -> None:
        assert type(value) is torch.Tensor
        assert value.dtype == torch.float32
        assert value.device.type == "cpu"
        assert value.is_contiguous()
        assert not value.requires_grad
        assert value.shape[1] == 512
        boundary_calls.append(tuple(value.shape))

    backend = adapter_module._Backend(
        torch=torch,
        tokenizer_type=_Tokenizer,
        model_type=_Model,
        batch_encoding_type=dict,
        runtime_identity=(("runtime", "mock-only"),),
        source_rows=(("mock.backend", 1, "0" * 64),),
        expected_config=adapter_module._EXPECTED_CONFIG,
        unexpected_count=0,
        unexpected_sha256=hashlib.sha256(b"").hexdigest(),
        expected_parameter_count=1,
        expected_buffer_count=1,
        validate_training_boundary=validate_boundary,
        validate_runtime=lambda: None,
    )
    adapter = adapter_module._load_frozen_clip_text_adapter(
        snapshot.resolve(),
        max_batch_size=max_batch_size,
        backend=backend,
        pins=tuple(pins),
    )
    assert _Model.last_model is not None
    return adapter, snapshot, boundary_calls, _Model.last_model


def test_ordered_variable_q_and_explicit_chunk_bound(tmp_path: Path) -> None:
    adapter, _snapshot, boundary_calls, model = _fixture(tmp_path, max_batch_size=3)
    captions = ("first private row", "second private row", "third private row", "fourth")
    lineage = tuple(_commitment(str(index)) for index in range(len(captions)))

    result = adapter.encode(captions, lineage, batch_size=2)

    assert tuple(result.embeddings.shape) == (4, 512)
    assert result.embeddings.dtype == torch.float32
    assert result.embeddings.device.type == "cpu"
    assert result.embeddings.is_contiguous()
    assert not result.embeddings.requires_grad
    assert model.forward_count == 2
    assert boundary_calls == [(2, 512), (2, 512), (4, 512)]
    assert result.caption_commitments == lineage
    assert result.receipt.output_shape == (4, 512)
    assert result.receipt.output_stride == (512, 1)


def test_lineage_never_changes_embeddings_and_order_is_preserved(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, _model = _fixture(tmp_path)
    captions = ("alpha private text", "beta private text", "gamma private text")
    first_lineage = tuple(_commitment("a" + str(index)) for index in range(3))
    second_lineage = tuple(_commitment("b" + str(index)) for index in range(3))

    first = adapter.encode(captions, first_lineage, batch_size=2)
    second = adapter.encode(captions, second_lineage, batch_size=2)
    different_batch = adapter.encode(captions, first_lineage, batch_size=3)
    reversed_rows = adapter.encode(
        tuple(reversed(captions)), tuple(reversed(first_lineage)), batch_size=2
    )

    assert torch.equal(first.embeddings, second.embeddings)
    assert first.receipt.output_sha256 == second.receipt.output_sha256
    assert (
        first.receipt.frozen_embedding_cache_key_sha256
        == second.receipt.frozen_embedding_cache_key_sha256
    )
    assert first.receipt.sha256 != second.receipt.sha256
    assert (
        first.receipt.frozen_embedding_cache_key_sha256
        != different_batch.receipt.frozen_embedding_cache_key_sha256
    )
    assert torch.equal(first.embeddings.flip(0), reversed_rows.embeddings)


def test_receipt_is_path_free_text_free_and_authority_zero(tmp_path: Path) -> None:
    adapter, snapshot, _boundary_calls, _model = _fixture(tmp_path)
    caption = "never serialize this private sentence"
    result = adapter.encode((caption,), (_commitment("one"),), batch_size=1)
    raw = result.receipt.canonical_json_bytes()
    decoded = json.loads(raw)

    assert caption.encode("utf-8") not in raw
    assert str(snapshot).encode("utf-8") not in raw
    assert decoded["authority"] == 0
    assert decoded["production"] is False
    assert decoded["training_authorized"] is False
    assert decoded["caption_rows"][0]["caption_utf8_sha256"] == hashlib.sha256(
        caption.encode("utf-8")
    ).hexdigest()
    assert decoded["embedding_selection"] == "official_pretrained_text_projection_raw_float32"


def test_loader_is_literal_offline_text_only(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, model = _fixture(tmp_path)
    assert adapter.max_batch_size == 3
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    for call in (_Tokenizer.load_calls[0], _Model.load_calls[0]):
        assert call["local_files_only"] is True
        assert call["trust_remote_code"] is False
        assert isinstance(call["cache_dir"], str)
    assert _Model.load_calls[0]["low_cpu_mem_usage"] is False
    assert _Model.load_calls[0]["output_loading_info"] is True


def test_runtime_identity_binds_mha_fastpath_and_restores_caller_flag() -> None:
    original = bool(torch.backends.mha.get_fastpath_enabled())
    versions = SimpleNamespace(__version__="mock-only")
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        disabled = dict(
            adapter_module._runtime_facts(
                huggingface_hub=versions,
                numpy=versions,
                tokenizers=versions,
                torch=torch,
                transformers=versions,
            )
        )
        assert disabled["mha_fastpath_enabled"] == "false"
        assert dict(adapter_module._EXPECTED_RUNTIME)["mha_fastpath_enabled"] == "false"

        torch.backends.mha.set_fastpath_enabled(True)
        enabled = dict(
            adapter_module._runtime_facts(
                huggingface_hub=versions,
                numpy=versions,
                tokenizers=versions,
                torch=torch,
                transformers=versions,
            )
        )
        assert enabled["mha_fastpath_enabled"] == "true"
        assert enabled != disabled
    finally:
        torch.backends.mha.set_fastpath_enabled(original)
    assert bool(torch.backends.mha.get_fastpath_enabled()) is original


def test_extra_file_and_snapshot_mutation_fail_without_path_leak(tmp_path: Path) -> None:
    snapshot = tmp_path / "bad-snapshot"
    snapshot.mkdir()
    raw = b"bound"
    expected = snapshot / "one.bin"
    expected.write_bytes(raw)
    (snapshot / "extra.bin").write_bytes(b"extra")
    pins = (("one.bin", len(raw), hashlib.sha256(raw).hexdigest()),)
    backend = adapter_module._Backend(
        torch=torch,
        tokenizer_type=_Tokenizer,
        model_type=_Model,
        batch_encoding_type=dict,
        runtime_identity=(("runtime", "mock-only"),),
        source_rows=(("mock.backend", 1, "0" * 64),),
        expected_config=adapter_module._EXPECTED_CONFIG,
        unexpected_count=0,
        unexpected_sha256=hashlib.sha256(b"").hexdigest(),
        expected_parameter_count=1,
        expected_buffer_count=1,
        validate_training_boundary=lambda _value: None,
        validate_runtime=lambda: None,
    )
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as caught:
        adapter_module._load_frozen_clip_text_adapter(
            snapshot.resolve(), max_batch_size=1, backend=backend, pins=pins
        )
    assert caught.value.code == "HOLD_SNAPSHOT_CENSUS"
    assert str(snapshot) not in str(caught.value)

    (snapshot / "extra.bin").unlink()
    adapter = adapter_module._load_frozen_clip_text_adapter(
        snapshot.resolve(), max_batch_size=1, backend=backend, pins=pins
    )
    expected.write_bytes(b"xxxxx")
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as changed:
        adapter.encode(("row",), (_commitment("row"),), batch_size=1)
    assert changed.value.code == "HOLD_SNAPSHOT_FILE_BYTES"
    assert str(snapshot) not in str(changed.value)


def test_nonfinite_forward_and_live_parameter_mutation_are_rejected(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, model = _fixture(tmp_path)
    model.return_nonfinite = True
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as nonfinite:
        adapter.encode(("row",), (_commitment("row"),), batch_size=1)
    assert nonfinite.value.code == "HOLD_EMBEDDING_BOUNDARY"

    model.return_nonfinite = False
    with torch.no_grad():
        model.text_projection.weight[0, 0] += 1.0
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as mutated:
        adapter.encode(("row",), (_commitment("row"),), batch_size=1)
    assert mutated.value.code == "HOLD_MODEL_CHANGED"


@pytest.mark.parametrize("batch_size", [0, 4, True])
def test_batch_size_is_exact_and_bounded(tmp_path: Path, batch_size: object) -> None:
    adapter, _snapshot, _boundary_calls, _model = _fixture(tmp_path, max_batch_size=3)
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as caught:
        adapter.encode(
            ("row",), (_commitment("row"),), batch_size=batch_size  # type: ignore[arg-type]
        )
    assert caught.value.code == "HOLD_BATCH_SIZE"


def test_batch_owns_immutable_snapshot_of_returned_tensor(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, _model = _fixture(tmp_path)
    result = adapter.encode(("row",), (_commitment("row"),), batch_size=1)
    first = result.embeddings
    first.zero_()
    assert not torch.equal(first, result.embeddings)


def test_long_caption_uses_visible_native_truncation(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, model = _fixture(tmp_path)
    caption = " ".join(f"word{index}" for index in range(76))
    result = adapter.encode((caption,), (_commitment("long"),), batch_size=1)
    row = result.receipt.to_public_dict()["caption_rows"][0]
    assert row["original_token_count_with_special_tokens"] == 78
    assert row["encoded_token_count_with_special_tokens"] == 77
    assert row["truncated"] is True
    assert result.receipt.to_public_dict()["tokenization"] == {
        "add_special_tokens": True,
        "long_caption_policy": "clip_native_truncate_77",
        "max_tokens_including_special_tokens": 77,
        "padding": "max_length",
        "preserve_terminal_eos": True,
        "truncation": True,
    }
    assert model.forward_count == 1


def test_wrong_native_terminal_eos_is_rejected_before_forward(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, model = _fixture(tmp_path)
    _Tokenizer.wrong_terminal_eos = True
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as caught:
        adapter.encode(("one row",), (_commitment("one"),), batch_size=1)
    assert caught.value.code == "HOLD_TOKEN_LENGTH_PROBE"
    assert model.forward_count == 0


def test_formal_token_body_must_equal_untruncated_native_prefix(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, model = _fixture(tmp_path)
    _Tokenizer.scramble_formal_middle = True
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as caught:
        adapter.encode(("one row",), (_commitment("one"),), batch_size=1)
    assert caught.value.code == "HOLD_TOKEN_LENGTH_PROBE"
    assert model.forward_count == 0


def test_encode_rng_advance_fails_and_restores_caller_state(tmp_path: Path) -> None:
    adapter, _snapshot, _boundary_calls, model = _fixture(tmp_path)
    caller_state = torch.get_rng_state().clone()
    model.advance_rng = True
    with pytest.raises(adapter_module.FrozenClipTextAdapterError) as caught:
        adapter.encode(("one row",), (_commitment("one"),), batch_size=1)
    assert caught.value.code == "HOLD_RUNTIME_RNG"
    assert torch.equal(caller_state, torch.get_rng_state())
    assert model.forward_count == 1


def test_planned_public_package_import_is_lazy_in_subprocess(tmp_path: Path) -> None:
    package = tmp_path / "src" / "phaseset_core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    shutil.copyfile(MODULE_PATH, package / "frozen_clip_text.py")
    source_root = str(tmp_path / "src")
    program = (
        "import json,sys;"
        f"sys.path.insert(0,{source_root!r});"
        "from phaseset_core.frozen_clip_text import "
        "EMBEDDING_DIM,METHOD_ID,load_frozen_clip_text_adapter;"
        "print(json.dumps({'dim':EMBEDDING_DIM,'method':METHOD_ID,"
        "'torch_eager':'torch' in sys.modules,"
        "'transformers_eager':'transformers' in sys.modules,"
        "'callable':callable(load_frozen_clip_text_adapter)},sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "callable": True,
        "dim": 512,
        "method": "phaseset-clip-native-truncate-77-pretrained-projection-v1",
        "torch_eager": False,
        "transformers_eager": False,
    }
