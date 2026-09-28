"""POSIX ownership regression; no model or training data is involved."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from phasepair_core import checkpoint
from test_phasepair_core_checkpoint import _bindings, _sections


@pytest.mark.skipif(os.name != "posix", reason="POSIX inode lifetime semantics")
@pytest.mark.parametrize("replace_target", [False, True])
def test_original_inode_is_pinned_and_descriptor_always_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_target: bool
) -> None:
    value = checkpoint.build_cpu_synthetic_checkpoint(_bindings(), _sections())
    target = (tmp_path / "owned.ppckpt").absolute()
    original_lstat = type(target).lstat
    original_open = type(target).open
    proxy = checkpoint.write_checkpoint_atomic.__globals__["os"]
    original_dup = proxy.dup
    pins: list[int] = []
    target_lstats = 0
    outsider = b"replacement owned by another writer"

    def pin(descriptor: int) -> int:
        duplicate = original_dup(descriptor)
        pins.append(duplicate)
        return duplicate

    def replacing_lstat(self: Path, *args: object, **kwargs: object):
        nonlocal target_lstats
        if self == target and target.exists():
            target_lstats += 1
            if target_lstats == 2 and replace_target:
                pinned = os.fstat(pins[0])
                target.unlink()
                with original_open(target, "wb") as handle:
                    handle.write(outsider)
                replacement = original_lstat(target)
                assert (replacement.st_dev, replacement.st_ino) != (
                    pinned.st_dev, pinned.st_ino
                )
        return original_lstat(self, *args, **kwargs)

    monkeypatch.setattr(proxy, "dup", pin)
    monkeypatch.setattr(type(target), "lstat", replacing_lstat)
    if replace_target:
        with pytest.raises(checkpoint.CheckpointContractError):
            checkpoint.write_checkpoint_atomic(value, target)
        assert target.read_bytes() == outsider
    else:
        checkpoint.write_checkpoint_atomic(value, target)
        assert target.is_file()
    assert len(pins) == 1
    with pytest.raises(OSError):
        os.fstat(pins[0])
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob("*.committed-retirement"))
