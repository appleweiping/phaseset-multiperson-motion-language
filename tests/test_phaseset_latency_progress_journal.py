from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from phaseset_core import base_cohort_latency as latency
from phaseset_core import latency_progress_journal as journal


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX journal contract")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _bindings() -> journal.LatencyJournalBindings:
    return journal.LatencyJournalBindings(
        resolved_cohort_sha256=_sha("resolved"),
        scored_cohort_sha256=_sha("scored"),
        source_tree_sha256=_sha("source-tree"),
        latency_protocol_sha256=_sha("protocol"),
        wrapper_sha256=_sha("wrapper"),
        registered_cuda_uuid="GPU-11111111-2222-3333-4444-555555555555",
    )


def _new_writer(tmp_path: Path, session_id: str = "session-001"):
    return journal.create_latency_progress_journal(
        tmp_path,
        session_id=session_id,
        bindings=_bindings(),
        _clock_ns=lambda: 1_800_000_000_000_000_000,
    )


def _start(writer: journal.LatencyProgressJournalWriter) -> None:
    writer.session_start(
        admission_observation_sha256=_sha("admission-observation"),
        started_unix_ns=1_800_000_000_000_000_000,
    )


def _context(writer: journal.LatencyProgressJournalWriter) -> None:
    writer.context_ready(
        runtime_sha256=_sha("runtime"),
        post_context_observation_sha256=_sha("post-context"),
    )


def _write_complete_visit(
    writer: journal.LatencyProgressJournalWriter,
    *,
    round_index: int,
    ordinal: int,
    run_id: str,
) -> None:
    writer.visit_started(
        round_index=round_index,
        ordinal_in_round=ordinal,
        run_id=run_id,
        before_observation_sha256=_sha(f"before-{round_index}-{ordinal}"),
    )
    for warmup in range(5):
        writer.warmup_completed(
            round_index=round_index,
            ordinal_in_round=ordinal,
            run_id=run_id,
            warmup_index=warmup,
        )
    for sample in range(11):
        writer.timed_sample_completed(
            round_index=round_index,
            ordinal_in_round=ordinal,
            run_id=run_id,
            sample_index=sample,
            sample_ns=100_000 + round_index * 100 + ordinal * 11 + sample,
        )
    writer.visit_ended(
        round_index=round_index,
        ordinal_in_round=ordinal,
        run_id=run_id,
        outcome="COMPLETED",
        after_observation_sha256=_sha(f"after-{round_index}-{ordinal}"),
        peak_allocated_bytes=512 * 1024**2,
        peak_reserved_bytes=768 * 1024**2,
        failure_code=None,
    )


def test_complete_exact_session_is_terminal_but_never_formal(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    _context(writer)
    for round_index, round_rows in enumerate(latency.cyclic_visit_schedule()):
        for ordinal, run_id in enumerate(round_rows):
            _write_complete_visit(
                writer,
                round_index=round_index,
                ordinal=ordinal,
                run_id=run_id,
            )
    writer.terminal(
        outcome="COMPLETED",
        latency_session_sha256=_sha("actual-latency-session"),
        failure_code=None,
    )
    path = writer.path
    writer.close()

    result = journal.read_latency_progress_journal(
        path,
        expected_bindings=_bindings(),
    )
    assert result.status == journal.COMPLETED_STATUS
    assert result.terminal_outcome == "COMPLETED"
    assert len(result.records) == 2 + 81 * 18 + 1 == 1461
    assert result.records[-1].event_type == "TERMINAL"
    assert result.uncommitted_tail_files == ()
    assert result.formal_latency is False
    assert result.authority == 0 and result.production is False


def test_handled_failure_closes_partial_visit_and_terminal_hold(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    _context(writer)
    run_id = latency.cyclic_visit_schedule()[0][0]
    writer.visit_started(
        round_index=0,
        ordinal_in_round=0,
        run_id=run_id,
        before_observation_sha256=_sha("before"),
    )
    writer.warmup_completed(
        round_index=0,
        ordinal_in_round=0,
        run_id=run_id,
        warmup_index=0,
    )
    writer.visit_ended(
        round_index=0,
        ordinal_in_round=0,
        run_id=run_id,
        outcome="HELD",
        after_observation_sha256=None,
        peak_allocated_bytes=32,
        peak_reserved_bytes=64,
        failure_code="CUDA_OUT_OF_MEMORY",
    )
    writer.terminal(
        outcome="HELD",
        latency_session_sha256=_sha("held-session"),
        failure_code="CUDA_OUT_OF_MEMORY",
    )
    path = writer.path
    writer.close()

    result = journal.read_latency_progress_journal(
        path,
        expected_bindings=_bindings(),
    )
    assert result.status == journal.HELD_STATUS
    assert result.terminal_outcome == "HELD"
    assert tuple(record.event_type for record in result.records[-2:]) == (
        "VISIT_END",
        "TERMINAL",
    )
    assert result.formal_latency is False


def test_held_terminal_failure_must_match_active_held_visit(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    _context(writer)
    run_id = latency.cyclic_visit_schedule()[0][0]
    writer.visit_started(
        round_index=0,
        ordinal_in_round=0,
        run_id=run_id,
        before_observation_sha256=_sha("before"),
    )
    writer.visit_ended(
        round_index=0,
        ordinal_in_round=0,
        run_id=run_id,
        outcome="HELD",
        after_observation_sha256=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        failure_code="CUDA_OUT_OF_MEMORY",
    )

    with pytest.raises(journal.LatencyJournalError, match="differs from held visit"):
        writer.terminal(
            outcome="HELD",
            latency_session_sha256=_sha("held-session"),
            failure_code="RESOURCE_LIMIT",
        )
    writer.close()


def test_directory_enumeration_stops_at_fixed_entry_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yielded = 0

    class Entry:
        name = "event-00000000.json"

    class Entries:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def __iter__(self):
            nonlocal yielded
            for _index in range(2 * journal.MAX_EVENTS + 1):
                yielded += 1
                yield Entry()
            raise AssertionError("bounded reader requested an entry beyond its limit")

    monkeypatch.setattr(journal.os, "scandir", lambda _fd: Entries())
    with pytest.raises(journal.LatencyJournalError, match="fixed event bound"):
        journal._bounded_directory_names(123)
    assert yielded == 2 * journal.MAX_EVENTS + 1


def test_sample_cannot_be_written_before_five_completed_warmups(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    _context(writer)
    run_id = latency.cyclic_visit_schedule()[0][0]
    writer.visit_started(
        round_index=0,
        ordinal_in_round=0,
        run_id=run_id,
        before_observation_sha256=_sha("before"),
    )
    with pytest.raises(journal.LatencyJournalError, match="out of order"):
        writer.timed_sample_completed(
            round_index=0,
            ordinal_in_round=0,
            run_id=run_id,
            sample_index=0,
            sample_ns=123,
        )
    writer.close()


def test_tail_event_without_marker_is_preserved_as_incomplete(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    path = writer.path
    writer.close()
    tail = path / "event-00000001.json"
    tail.write_bytes(b'{"binding_sha256":"truncated')
    tail.chmod(0o600)

    result = journal.read_latency_progress_journal(
        path,
        expected_bindings=_bindings(),
    )
    assert result.status == journal.INCOMPLETE_STATUS
    assert len(result.records) == 1
    assert result.uncommitted_tail_files == ((tail.name, tail.read_bytes()),)
    assert result.terminal_outcome is None
    assert result.formal_latency is False


def test_truncated_final_marker_does_not_commit_its_event(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    path = writer.path
    writer.close()
    marker = path / "event-00000000.complete"
    marker.write_bytes(marker.read_bytes()[:13])
    marker.chmod(0o600)

    result = journal.read_latency_progress_journal(
        path,
        expected_bindings=_bindings(),
    )
    assert result.status == journal.INCOMPLETE_STATUS
    assert result.records == ()
    assert tuple(name for name, _raw in result.uncommitted_tail_files) == (
        "event-00000000.json",
        "event-00000000.complete",
    )


def test_committed_event_tamper_is_rejected_even_at_tail(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    path = writer.path
    writer.close()
    event = path / "event-00000000.json"
    raw = event.read_bytes()
    event.write_bytes(raw.replace(b"SESSION_START", b"SESSION_STARK", 1))
    event.chmod(0o600)

    with pytest.raises(journal.LatencyJournalError, match="differs from event bytes"):
        journal.read_latency_progress_journal(
            path,
            expected_bindings=_bindings(),
        )


def test_appended_marker_bytes_and_nonfinal_truncation_are_rejected(
    tmp_path: Path,
) -> None:
    appended_writer = _new_writer(tmp_path, "appended-marker")
    _start(appended_writer)
    appended_path = appended_writer.path
    appended_writer.close()
    appended_marker = appended_path / "event-00000000.complete"
    appended_marker.write_bytes(appended_marker.read_bytes() + b"x")
    appended_marker.chmod(0o600)
    with pytest.raises(journal.LatencyJournalError, match="differs from event bytes"):
        journal.read_latency_progress_journal(
            appended_path,
            expected_bindings=_bindings(),
        )

    nonfinal_writer = _new_writer(tmp_path, "nonfinal-truncation")
    _start(nonfinal_writer)
    _context(nonfinal_writer)
    nonfinal_path = nonfinal_writer.path
    nonfinal_writer.close()
    nonfinal_marker = nonfinal_path / "event-00000000.complete"
    nonfinal_marker.write_bytes(nonfinal_marker.read_bytes()[:7])
    nonfinal_marker.chmod(0o600)
    with pytest.raises(journal.LatencyJournalError, match="not the sole final tail"):
        journal.read_latency_progress_journal(
            nonfinal_path,
            expected_bindings=_bindings(),
        )


def test_gap_nonfinal_damage_and_external_binding_drift_are_rejected(
    tmp_path: Path,
) -> None:
    writer = _new_writer(tmp_path, "gap-session")
    _start(writer)
    path = writer.path
    writer.close()
    gap = path / "event-00000002.json"
    gap.write_bytes(b"partial")
    gap.chmod(0o600)
    with pytest.raises(journal.LatencyJournalError, match="gap"):
        journal.read_latency_progress_journal(path, expected_bindings=_bindings())

    other = _new_writer(tmp_path, "binding-session")
    _start(other)
    other_path = other.path
    other.close()
    drifted = replace(_bindings(), source_tree_sha256=_sha("other-source"))
    with pytest.raises(journal.LatencyJournalError, match="header or chain"):
        journal.read_latency_progress_journal(
            other_path,
            expected_bindings=drifted,
        )


def test_added_tail_after_terminal_is_rejected(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    writer.terminal(
        outcome="HELD",
        latency_session_sha256=_sha("held"),
        failure_code="RESOURCE_LIMIT",
    )
    path = writer.path
    writer.close()
    added = path / "event-00000002.json"
    added.write_bytes(b"uncommitted-added-tail")
    added.chmod(0o600)

    with pytest.raises(journal.LatencyJournalError, match="sole final tail"):
        journal.read_latency_progress_journal(path, expected_bindings=_bindings())


def test_existing_next_event_causes_exclusive_writer_failure(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    occupied = writer.path / "event-00000001.json"
    occupied.write_bytes(b"occupied")
    occupied.chmod(0o600)

    with pytest.raises(journal.LatencyJournalWriteError, match="sequence 1"):
        _context(writer)
    with pytest.raises(journal.LatencyJournalWriteError, match="closed or failed"):
        _context(writer)
    writer.close()


def test_reader_rejects_symlinked_event_leaf(tmp_path: Path) -> None:
    writer = _new_writer(tmp_path)
    _start(writer)
    path = writer.path
    writer.close()
    outside = tmp_path / "outside.json"
    outside.write_bytes((path / "event-00000000.json").read_bytes())
    outside.chmod(0o600)
    event = path / "event-00000000.json"
    event.unlink()
    event.symlink_to(outside)

    with pytest.raises(journal.LatencyJournalError, match="opened safely"):
        journal.read_latency_progress_journal(path, expected_bindings=_bindings())


def test_disk_full_write_retains_only_uncommitted_physical_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _new_writer(tmp_path)
    real_write = os.write
    calls = 0

    def fail_after_prefix(fd: int, raw) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, raw[:9])
        raise OSError(errno.ENOSPC, "synthetic disk full")

    monkeypatch.setattr(os, "write", fail_after_prefix)
    with pytest.raises(journal.LatencyJournalWriteError, match="sequence 0"):
        _start(writer)
    path = writer.path
    writer.close()
    monkeypatch.setattr(os, "write", real_write)

    result = journal.read_latency_progress_journal(
        path,
        expected_bindings=_bindings(),
    )
    assert result.status == journal.INCOMPLETE_STATUS
    assert result.records == ()
    assert result.uncommitted_tail_files[0][0] == "event-00000000.json"
    assert len(result.uncommitted_tail_files[0][1]) == 9


def test_real_subprocess_hard_kill_leaves_readable_incomplete_prefix(
    tmp_path: Path,
) -> None:
    package_src = str(Path(journal.__file__).resolve().parents[1])
    payload = json.dumps(
        {
            "package_src": package_src,
            "parent": str(tmp_path),
            "bindings": _bindings().payload(),
        }
    )
    script = textwrap.dedent(
        """
        import hashlib, json, pathlib, sys, time
        cfg = json.loads(sys.argv[1])
        sys.path.insert(0, cfg['package_src'])
        from phaseset_core import latency_progress_journal as journal
        bindings = journal.LatencyJournalBindings(**{
            key: value for key, value in cfg['bindings'].items() if key != 'schema'
        })
        writer = journal.create_latency_progress_journal(
            pathlib.Path(cfg['parent']), session_id='killed-session', bindings=bindings
        )
        writer.session_start(
            admission_observation_sha256=hashlib.sha256(b'admission').hexdigest(),
            started_unix_ns=time.time_ns(),
        )
        writer.context_ready(
            runtime_sha256=hashlib.sha256(b'runtime').hexdigest(),
            post_context_observation_sha256=hashlib.sha256(b'context').hexdigest(),
        )
        print('READY', flush=True)
        time.sleep(300)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", script, payload],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)

    result = journal.read_latency_progress_journal(
        tmp_path / "killed-session",
        expected_bindings=_bindings(),
    )
    assert result.status == journal.INCOMPLETE_STATUS
    assert tuple(record.event_type for record in result.records) == (
        "SESSION_START",
        "CONTEXT_READY",
    )
    assert result.uncommitted_tail_files == ()
    assert result.formal_latency is False
