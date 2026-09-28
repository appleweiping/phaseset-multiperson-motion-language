from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

STAGED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGED_ROOT / "scripts"))

import release_manifest  # noqa: E402
import release_tree  # noqa: E402

OBJECT_ID = "1" * 40


def _entry(path: str = "payload.bin") -> release_tree.GitIndexEntry:
    return release_tree.GitIndexEntry("100644", OBJECT_ID, PurePosixPath(path))


def _batch(object_type: bytes, payload: bytes, *, size: bytes | None = None) -> bytes:
    raw_size = str(len(payload)).encode("ascii") if size is None else size
    return OBJECT_ID.encode("ascii") + b" " + object_type + b" " + raw_size + b"\n" + payload + b"\n"


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (OBJECT_ID.encode("ascii") + b" blob\n", "header is malformed"),
        (b"2" * 40 + b" blob 0\n\n", "object ID mismatch"),
        (_batch(b"blob", b"", size=b"00"), "byte count is malformed"),
        (_batch(b"blob", b"") + b"unexpected", "unexpected trailing bytes"),
    ),
)
def test_batch_parser_rejects_malformed_output(raw: bytes, message: str) -> None:
    with pytest.raises(release_tree.ReleaseTreeError, match=message):
        release_tree._parse_git_batch_output(raw, (_entry(),))


@pytest.mark.parametrize(
    "raw",
    (
        OBJECT_ID.encode("ascii") + b" blob 4\nab",
        OBJECT_ID.encode("ascii") + b" blob 3\nabc",
    ),
)
def test_batch_parser_rejects_truncated_payload_or_delimiter(raw: bytes) -> None:
    with pytest.raises(release_tree.ReleaseTreeError, match="payload is truncated"):
        release_tree._parse_git_batch_output(raw, (_entry(),))


def test_batch_parser_rejects_missing_object() -> None:
    raw = OBJECT_ID.encode("ascii") + b" missing\n"
    with pytest.raises(release_tree.ReleaseTreeError, match="object is missing"):
        release_tree._parse_git_batch_output(raw, (_entry(),))


def test_batch_parser_rejects_wrong_object_type() -> None:
    with pytest.raises(release_tree.ReleaseTreeError, match="object is not a blob"):
        release_tree._parse_git_batch_output(_batch(b"tree", b"abc"), (_entry(),))


def _git(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout


def test_batch_snapshot_and_manifest_match_individual_index_blob_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "RELEASE_FILES.sha256").write_bytes(release_tree.MANIFEST_HEADER)
    (tmp_path / "alpha.txt").write_bytes(b"alpha\n")
    (tmp_path / "empty.bin").write_bytes(b"")
    (tmp_path / "nested" / "binary.bin").write_bytes(b"\x00payload\n\xff")
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "add", "--all")

    entries = release_tree.git_index_entries(tmp_path)
    expected = {
        entry.path: _git(tmp_path, "cat-file", "blob", entry.object_id, binary=True)
        for entry in entries
    }
    assert release_tree.git_index_blobs(tmp_path, entries) == expected

    calls: list[tuple[str, ...]] = []
    original_run = subprocess.run

    def recording_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        command = tuple(args[0])  # type: ignore[arg-type]
        calls.append(command)
        return original_run(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(release_tree.subprocess, "run", recording_run)
    monkeypatch.setattr(release_manifest, "ROOT", tmp_path)
    rendered = release_manifest.render_manifest()
    tracked = tuple(entry for entry in entries if entry.path != release_tree.MANIFEST_RELATIVE)
    rows = "\n".join(
        f"{hashlib.sha256(expected[entry.path]).hexdigest()}\t"
        f"{len(expected[entry.path])}\t{entry.path.as_posix()}"
        for entry in tracked
    )
    assert rendered == (
        release_tree.MANIFEST_HEADER.decode("ascii") + rows + "\n"
    ).encode("utf-8")
    assert calls.count(("git", "cat-file", "--batch")) == 1

    tree = release_tree.resolve_release_tree(tmp_path)
    (tmp_path / "alpha.txt").write_bytes(b"unstaged replacement\n")
    assert tree.read_bytes(PurePosixPath("alpha.txt")) == b"alpha\n"


def test_batch_read_rejects_index_change_during_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "RELEASE_FILES.sha256").write_bytes(release_tree.MANIFEST_HEADER)
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"first index state\n")
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "add", "--all")
    entries = release_tree.git_index_entries(tmp_path)

    original_run = subprocess.run

    def changing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        command = tuple(args[0])  # type: ignore[arg-type]
        completed = original_run(*args, **kwargs)  # type: ignore[call-overload]
        if command == ("git", "cat-file", "--batch"):
            payload.write_bytes(b"second index state\n")
            original_run(
                ["git", "add", "payload.txt"],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )
        return completed

    monkeypatch.setattr(release_tree.subprocess, "run", changing_run)
    with pytest.raises(release_tree.ReleaseTreeError, match="index changed"):
        release_tree.git_index_blobs(tmp_path, entries)
