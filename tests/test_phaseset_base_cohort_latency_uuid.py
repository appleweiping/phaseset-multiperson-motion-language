from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from phaseset_core import base_cohort_latency as latency


REGISTERED_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
BARE_UUID = REGISTERED_UUID[4:]
CHANGED_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeef"
TOTAL_BYTES = 50_807_570_432


@dataclass
class _Properties:
    uuid: object
    name: str = "NVIDIA RTX A6000"
    total_memory: int = TOTAL_BYTES


class _CudaUuid:
    """CPU-only stand-in for the observed torch._C._CUuuid string surface."""

    def __init__(self, rendered: str) -> None:
        self._rendered = rendered

    def __str__(self) -> str:
        return self._rendered


def _admission(uuid: str = REGISTERED_UUID) -> latency.SharedCudaObservation:
    return latency.SharedCudaObservation(
        captured_unix_ns=1,
        device_uuid=uuid,
        memory_total_mib=49_140,
        memory_free_mib=48_429,
        memory_used_mib=23,
        gpu_utilization_percent=0,
        memory_utilization_percent=0,
        compute_mode="Default",
        driver_version="550.54.14",
    )


def _install_preallocation_cuda_surface(
    monkeypatch: pytest.MonkeyPatch,
    observed_uuid: object,
) -> list[str]:
    forbidden_calls: list[str] = []

    def forbidden(label: str):
        def call(*args: object, **kwargs: object) -> object:
            del args, kwargs
            forbidden_calls.append(label)
            raise AssertionError(f"{label} ran before UUID rejection")

        return call

    monkeypatch.setattr(latency.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(latency.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        latency.torch.cuda,
        "get_device_properties",
        lambda _index: _Properties(observed_uuid),
    )
    monkeypatch.setattr(
        latency.torch.cuda,
        "set_per_process_memory_fraction",
        forbidden("allocator cap"),
    )
    monkeypatch.setattr(latency.torch, "empty", forbidden("tensor allocation"))
    return forbidden_calls


@pytest.mark.parametrize(
    "observed_uuid",
    [
        None,
        "",
        "aaaaaaaa",
        f" {BARE_UUID}",
        f"{BARE_UUID} ",
        f"MIG-{BARE_UUID}",
        CHANGED_UUID[4:],
        BARE_UUID.upper(),
        f"GPU-{BARE_UUID.upper()}",
        f"GPU-GPU-{BARE_UUID}",
    ],
)
def test_observed_uuid_mutants_fail_before_allocator_or_tensor(
    monkeypatch: pytest.MonkeyPatch,
    observed_uuid: object,
) -> None:
    forbidden_calls = _install_preallocation_cuda_surface(monkeypatch, observed_uuid)
    runtime = latency._NvidiaSmiTorchRuntime(REGISTERED_UUID)
    with pytest.raises(latency.BaseCohortLatencyError, match="UUID"):
        runtime.initialize(
            device=torch.device("cuda"),
            precision_mode="FP32",
            admission_observation=_admission(),
        )
    assert forbidden_calls == []


def test_changed_valid_admission_uuid_fails_before_allocator_or_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_calls = _install_preallocation_cuda_surface(
        monkeypatch,
        _CudaUuid(BARE_UUID),
    )
    changed = CHANGED_UUID
    runtime = latency._NvidiaSmiTorchRuntime(REGISTERED_UUID)
    with pytest.raises(latency.BaseCohortLatencyError, match="differ"):
        runtime.initialize(
            device=torch.device("cuda"),
            precision_mode="FP32",
            admission_observation=_admission(changed),
        )
    assert forbidden_calls == []


@pytest.mark.parametrize(
    "observed_uuid",
    [_CudaUuid(BARE_UUID), BARE_UUID, REGISTERED_UUID],
)
def test_canonical_bare_or_prefixed_uuid_preserves_allocator_cap_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    observed_uuid: object,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(latency.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(latency.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        latency.torch.cuda,
        "get_device_properties",
        lambda _index: _Properties(observed_uuid),
    )

    def set_fraction(value: float, *, device: int) -> None:
        calls.append(("allocator", (value, device)))

    def empty(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object:
        calls.append(("empty", (shape, dtype, device)))
        return object()

    monkeypatch.setattr(
        latency.torch.cuda,
        "set_per_process_memory_fraction",
        set_fraction,
    )
    monkeypatch.setattr(latency.torch, "empty", empty)
    monkeypatch.setattr(latency.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(latency.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(latency.torch.cuda, "memory_allocated", lambda index: 0)
    monkeypatch.setattr(latency.torch.cuda, "memory_reserved", lambda index: 0)
    monkeypatch.setattr(latency.torch.backends.cudnn, "version", lambda: 90_100)
    monkeypatch.setattr(latency.torch.version, "cuda", "12.6")
    monkeypatch.setattr(
        latency.training_module,
        "training_environment_sha256",
        lambda: "a" * 64,
    )

    runtime = latency._NvidiaSmiTorchRuntime(REGISTERED_UUID).initialize(
        device=torch.device("cuda"),
        precision_mode="FP32",
        admission_observation=_admission(),
    )
    assert runtime.device == "cuda:0"
    assert runtime.device_uuid == REGISTERED_UUID
    assert runtime.allocator_limit_bytes == latency.CUDA_ALLOCATOR_LIMIT_BYTES
    assert runtime.torch_total_memory_bytes == TOTAL_BYTES
    assert calls == [
        (
            "allocator",
            (latency.CUDA_ALLOCATOR_LIMIT_BYTES / TOTAL_BYTES, 0),
        ),
        ("empty", ((1,), torch.float32, torch.device("cuda"))),
    ]


@pytest.mark.parametrize(
    "registered_uuid",
    [
        BARE_UUID,
        "",
        f" {REGISTERED_UUID}",
        f"{REGISTERED_UUID} ",
        f"MIG-{BARE_UUID}",
        CHANGED_UUID.upper(),
    ],
)
def test_runtime_constructor_requires_full_registered_uuid(
    registered_uuid: str,
) -> None:
    with pytest.raises(latency.BaseCohortLatencyError, match="UUID"):
        latency._NvidiaSmiTorchRuntime(registered_uuid)
