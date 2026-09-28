"""CPU-mock contracts for explicit cuBLAS workspace release."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from phaseset_core import base_cohort_latency as latency


_WORKSPACE_BYTES = 32 * 1024 * 1024


class _System:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def to(self, device: torch.device) -> _System:
        self._events.append(f"system.to:{device}")
        return self


def test_release_clears_observed_workspace_then_preserves_zero_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    state = {"allocated": _WORKSPACE_BYTES, "reserved": _WORKSPACE_BYTES}

    def synchronize(device: torch.device) -> None:
        events.append(f"synchronize:{device}")

    def clear_cublas_workspaces() -> None:
        events.append("clear_cublas_workspaces")
        assert state["allocated"] == _WORKSPACE_BYTES
        state["allocated"] = 0

    def empty_cache() -> None:
        events.append("empty_cache")
        assert state["allocated"] == 0
        state["reserved"] = 0

    def memory_allocated(device: torch.device) -> int:
        events.append(f"memory_allocated:{device}")
        return state["allocated"]

    def memory_reserved(device: torch.device) -> int:
        events.append(f"memory_reserved:{device}")
        return state["reserved"]

    monkeypatch.setattr(
        latency.torch._C,
        "_cuda_clearCublasWorkspaces",
        clear_cublas_workspaces,
        raising=False,
    )
    monkeypatch.setattr(latency.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(latency.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(latency.torch.cuda, "memory_allocated", memory_allocated)
    monkeypatch.setattr(latency.torch.cuda, "memory_reserved", memory_reserved)

    device = torch.device("cuda:0")
    latency._NvidiaSmiTorchRuntime("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee").release(
        _System(events),
        device,
    )

    assert state == {"allocated": 0, "reserved": 0}
    assert events == [
        "system.to:cpu",
        "synchronize:cuda:0",
        "clear_cublas_workspaces",
        "synchronize:cuda:0",
        "empty_cache",
        "memory_allocated:cuda:0",
        "memory_reserved:cuda:0",
    ]


@pytest.mark.parametrize(
    ("allocated", "reserved"),
    [
        (1, 0),
        (0, 1),
        (_WORKSPACE_BYTES, _WORKSPACE_BYTES),
    ],
)
def test_release_still_rejects_any_real_remaining_cuda_allocation(
    monkeypatch: pytest.MonkeyPatch,
    allocated: int,
    reserved: int,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        latency.torch._C,
        "_cuda_clearCublasWorkspaces",
        lambda: events.append("clear"),
        raising=False,
    )
    monkeypatch.setattr(
        latency.torch.cuda,
        "synchronize",
        lambda device: events.append(f"sync:{device}"),
    )
    monkeypatch.setattr(latency.torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(latency.torch.cuda, "memory_allocated", lambda _device: allocated)
    monkeypatch.setattr(latency.torch.cuda, "memory_reserved", lambda _device: reserved)

    with pytest.raises(latency.BaseCohortLatencyError, match="allocation remained"):
        latency._NvidiaSmiTorchRuntime("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee").release(
            _System(events), torch.device("cuda:0")
        )
    assert events[:5] == [
        "system.to:cpu",
        "sync:cuda:0",
        "clear",
        "sync:cuda:0",
        "empty",
    ]


def test_release_fails_closed_when_workspace_api_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(latency.torch, "_C", SimpleNamespace())
    monkeypatch.setattr(
        latency.torch.cuda,
        "synchronize",
        lambda device: events.append(f"sync:{device}"),
    )
    monkeypatch.setattr(latency.torch.cuda, "empty_cache", lambda: events.append("empty"))

    with pytest.raises(latency.BaseCohortLatencyError, match="API is unavailable") as captured:
        latency._NvidiaSmiTorchRuntime("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee").release(
            _System(events), torch.device("cuda:0")
        )
    assert isinstance(captured.value.__cause__, AttributeError)
    assert events == ["system.to:cpu", "sync:cuda:0"]


def test_release_fails_closed_when_workspace_clear_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fail_clear() -> None:
        events.append("clear")
        raise RuntimeError("private fixture failure")

    monkeypatch.setattr(
        latency.torch._C,
        "_cuda_clearCublasWorkspaces",
        fail_clear,
        raising=False,
    )
    monkeypatch.setattr(
        latency.torch.cuda,
        "synchronize",
        lambda device: events.append(f"sync:{device}"),
    )
    monkeypatch.setattr(latency.torch.cuda, "empty_cache", lambda: events.append("empty"))

    with pytest.raises(latency.BaseCohortLatencyError, match="workspace clear failed") as captured:
        latency._NvidiaSmiTorchRuntime("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee").release(
            _System(events), torch.device("cuda:0")
        )
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "private fixture failure"
    assert events == ["system.to:cpu", "sync:cuda:0", "clear"]


def test_workspace_delta_does_not_change_frozen_resource_workload_or_schedule() -> None:
    assert latency.CUDA_ALLOCATOR_LIMIT_BYTES == 2 * 1024**3
    assert latency.MINIMUM_FREE_MIB == 8192
    assert (
        latency.FIXTURE_BATCH,
        latency.FIXTURE_ACTORS,
        latency.FIXTURE_FRAMES,
        latency.FIXTURE_JOINTS,
        latency.FIXTURE_COORDINATES,
        latency.FIXTURE_EMBEDDING_DIM,
    ) == (1, 32, 200, 22, 3, 512)
    assert (
        latency.ROUND_COUNT,
        latency.ROWS_PER_ROUND,
        latency.WARMUPS_PER_VISIT,
        latency.TIMED_PER_VISIT,
        latency.SAMPLES_PER_ROW,
    ) == (9, 9, 5, 11, 99)
