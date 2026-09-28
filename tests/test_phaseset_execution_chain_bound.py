from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from phaseset_core import execution


def test_attempt_chain_depth_is_rejected_before_opening_past_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []

    def fake_verify(_path: Path, attempt_id: str) -> object:
        opened.append(attempt_id)
        ordinal = int(attempt_id.removeprefix("attempt-"))
        return SimpleNamespace(
            resume=SimpleNamespace(predecessor_attempt_id=f"attempt-{ordinal + 1}"),
        )

    monkeypatch.setattr(execution, "_verify_local_attempt", fake_verify)
    with pytest.raises(execution.ExecutionContractError, match="closed attempt bound"):
        execution._verify_attempt_chain(
            tmp_path,
            "attempt-0",
            remaining_attempts=2,
        )
    assert opened == ["attempt-0", "attempt-1"]


@pytest.mark.parametrize("value", [True, 0, 65])
def test_attempt_chain_depth_budget_is_closed(value: object, tmp_path: Path) -> None:
    with pytest.raises(execution.ExecutionContractError, match="closed attempt bound"):
        execution._verify_attempt_chain(
            tmp_path,
            "attempt-0",
            remaining_attempts=value,  # type: ignore[arg-type]
        )
