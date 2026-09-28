"""Lifetime lease, operation heartbeat and bounded checkpoint retention.

This is execution evidence, not a study launch grant. It deliberately does not
route the new parent study through the legacy 33-run AttemptStore parser.
"""

from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

from .execution import _write_once, artifact_sha256
from .host import _AttemptLease
from .training import CheckpointArtifact, TrainingCheckpointError


def parent_failure_code(error: BaseException | None) -> str | None:
    """Descriptive failure classes; never an automatic restart decision."""
    if error is None:
        return None
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "DISK_FULL"
    if isinstance(error, TrainingCheckpointError):
        return "CHECKPOINT_INVALID"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "PROCESS_INTERRUPTED"
    if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError":
        return "RESOURCE_LIMIT"
    if isinstance(error, (ValueError, TypeError)):
        return "INPUT_OR_CONTRACT"
    return "UNCLASSIFIED"


def _record(path: Path, payload: dict) -> str:
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_once(path, raw)
    return artifact_sha256(raw)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ParentRunMonitor:
    """One fresh host attempt; heartbeats continue inside long native calls.

    Retention is explicit: None preserves every payload, integer >=2 keeps the
    newest N plus their selected-best dependencies and the current best. Only
    payloads produced by this monitor in this directory can be retired. Resume
    predecessors, failed attempts, events and receipts are never deleted.
    The worker reads scalar snapshots only, not tensors or model/RNG state.
    """

    HEARTBEAT_SECONDS = 30.0

    def __init__(self, root: Path, *, keep_recent: int | None = None):
        if keep_recent is not None and (type(keep_recent) is not int or keep_recent < 2):
            raise ValueError("checkpoint retention must keep at least two recent payloads")
        self.root = root.resolve()
        self.keep_recent = keep_recent
        self._lease = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._error: BaseException | None = None
        self._started = None
        self._elapsed = 0.0
        self._heartbeat_sequence = 0
        self._last_heartbeat_sha256 = None
        self._latest_checkpoint_sha256 = None
        self._snapshot = {"phase": "STARTING", "global_step": 0}
        self._checkpoints: list[tuple[CheckpointArtifact, CheckpointArtifact | None]] = []
        self._retired: set[Path] = set()

    def start(self):
        if self._started is not None:
            raise RuntimeError("monitor may start only once")
        self._lease = _AttemptLease.acquire(self.root)
        # After acquisition, even initialization failure belongs to the host's
        # terminal/finally path. Do not make an unfinished attempt acquirable.
        for name in ("heartbeats", "checkpoint-receipts", "retention"):
            (self.root / name).mkdir(mode=0o700)
        self._started = time.monotonic()
        with self._lock:
            self._heartbeat()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _heartbeat(self):
        assert self._started is not None
        self._elapsed = time.monotonic() - self._started
        self._last_heartbeat_sha256 = _record(
            self.root / "heartbeats" / f"heartbeat-{self._heartbeat_sequence:08d}.json",
            {
                "schema": "phaseset-parent-operation-heartbeat-v1",
                "sequence": self._heartbeat_sequence,
                "process_elapsed_seconds": self._elapsed,
                "previous_heartbeat_sha256": self._last_heartbeat_sha256,
                "latest_checkpoint_sha256": self._latest_checkpoint_sha256,
                **self._snapshot,
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "formal_authority": False,
            },
        )
        self._heartbeat_sequence += 1

    def _worker(self):
        while not self._stop.wait(self.HEARTBEAT_SECONDS):
            try:
                with self._lock:
                    self._heartbeat()
            except BaseException as error:
                self._error = error
                self._stop.set()

    def check(self):
        if self._error is not None:
            raise self._error

    def observe(self, event: dict):
        self.check()
        with self._lock:
            # An event is a freshly constructed scalar host record. Make our
            # own snapshot so the background writer never observes mutation.
            self._snapshot = dict(event)

    def checkpoint(self, artifact: CheckpointArtifact, *, best: CheckpointArtifact | None):
        self.check()
        if artifact.path.parent != self.root or artifact.path.is_symlink():
            raise ValueError("retention accepts only this attempt's own checkpoint payloads")
        index = len(self._checkpoints)
        _record(
            self.root / "checkpoint-receipts" / f"checkpoint-{index:08d}.json",
            {
                "schema": "phaseset-parent-checkpoint-receipt-v1",
                "sequence": index,
                "global_step": artifact.global_step,
                "payload": artifact.path.name,
                "payload_sha256": artifact.sha256,
                "payload_bytes": artifact.path.stat().st_size,
                "best_path": None if best is None else str(best.path),
                "best_sha256": None if best is None else best.sha256,
            },
        )
        self._checkpoints.append((artifact, best))
        with self._lock:
            self._latest_checkpoint_sha256 = artifact.sha256
        if self.keep_recent is None:
            return
        recent = self._checkpoints[-self.keep_recent :]
        protected = {item.path for item, _ in recent}
        protected.update(selected.path for _, selected in recent if selected is not None)
        if best is not None:
            protected.add(best.path)
        for old_index, (old, _) in enumerate(self._checkpoints):
            if old.path in protected or old.path in self._retired:
                continue
            if old.path.parent != self.root or old.path.is_symlink() or not old.path.is_file():
                raise TrainingCheckpointError(
                    "retention payload is no longer an owned regular file"
                )
            if _file_digest(old.path) != old.sha256:
                raise TrainingCheckpointError("retention payload digest changed; preserve evidence")
            receipt = {
                "payload": old.path.name,
                "payload_sha256": old.sha256,
                "payload_bytes": old.path.stat().st_size,
                "protected_paths": sorted(str(path) for path in protected),
                "reason": "superseded_unreferenced_payload_keep_recent_and_best",
            }
            # A crash between these records is explicitly a pending retirement,
            # not a claim that deletion finished. No recursive/glob deletion.
            _record(self.root / "retention" / f"planned-{old_index:08d}.json", receipt)
            old.path.unlink()
            _record(self.root / "retention" / f"retired-{old_index:08d}.json", receipt)
            self._retired.add(old.path)

    def finish(
        self,
        *,
        outcome: str,
        failure: BaseException | None,
        terminal_snapshot: dict | None = None,
    ):
        """Stop heartbeat before writing host terminal; lease remains held."""
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=self.HEARTBEAT_SECONDS + 5)
            if self._thread.is_alive():
                raise RuntimeError("heartbeat worker did not terminate")
        self.check()
        if self._started is not None:
            with self._lock:
                if terminal_snapshot is not None:
                    self._snapshot.update(terminal_snapshot)
                self._snapshot.update(
                    phase="TERMINAL", outcome=outcome, failure_code=parent_failure_code(failure)
                )
                self._heartbeat()

    def release(self):
        """Release only after the host's terminal write (including its failure)."""
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    def summary(self) -> dict:
        return {
            "latest_heartbeat_sha256": self._last_heartbeat_sha256,
            "heartbeat_count": self._heartbeat_sequence,
            "process_elapsed_seconds": self._elapsed,
            "checkpoint_keep_recent": self.keep_recent,
            "retired_payload_count": len(self._retired),
            "monitor_failure_class": None if self._error is None else type(self._error).__name__,
            "budget_or_study_authority": False,
        }
