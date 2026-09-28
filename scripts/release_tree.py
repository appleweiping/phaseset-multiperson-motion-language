"""Resolve a release tree from either an exact Git checkout or an archive."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MANIFEST_RELATIVE = PurePosixPath("RELEASE_FILES.sha256")
MANIFEST_HEADER = b"# sha256\\tbytes\\tgit_index_path; self excluded\n"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReleaseTreeError(RuntimeError):
    """The checkout or archive cannot establish one exact release tree."""


@dataclass(frozen=True)
class ManifestEntry:
    """One content-addressed regular file declared by the release manifest."""

    sha256: str
    size: int
    path: PurePosixPath


@dataclass(frozen=True)
class GitIndexEntry:
    """One validated stage-zero regular file in the Git index."""

    mode: str
    object_id: str
    path: PurePosixPath


@dataclass(frozen=True)
class ReleaseTree:
    """One resolved tree whose paths and bytes share the same authority."""

    root: Path
    paths: tuple[PurePosixPath, ...]
    authority: str
    archive_content: Mapping[PurePosixPath, bytes] | None = None
    index_content: Mapping[PurePosixPath, bytes] | None = None

    def read_bytes(self, path: PurePosixPath) -> bytes:
        if path not in self.paths:
            raise ReleaseTreeError(f"path is outside the release tree: {path.as_posix()}")
        if self.authority == "git-index":
            if self.index_content is None or path not in self.index_content:
                raise ReleaseTreeError(
                    f"Git-index snapshot lacks release path: {path.as_posix()}"
                )
            return self.index_content[path]
        if self.authority == "archive-manifest":
            if self.archive_content is None or path not in self.archive_content:
                raise ReleaseTreeError(
                    f"archive snapshot lacks release path: {path.as_posix()}"
                )
            return self.archive_content[path]
        raise ReleaseTreeError(f"unknown release-tree authority: {self.authority!r}")


def clean_git_environment() -> dict[str, str]:
    """Prevent ambient Git selectors from changing the audited repository."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }


def _safe_relative_path(decoded: str) -> PurePosixPath:
    if not decoded or any(character in decoded for character in "\0\t\n\r\\:"):
        raise ReleaseTreeError(f"invalid release path: {decoded!r}")
    path = PurePosixPath(decoded)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != decoded
        or path == PurePosixPath(".")
    ):
        raise ReleaseTreeError(f"unsafe or non-canonical release path: {decoded!r}")
    return path


def is_exact_git_checkout(root: Path) -> bool:
    """Return true only when ``root`` itself, not an ancestor, is the worktree."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=False,
            env=clean_git_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    if completed.returncode != 0:
        return False
    try:
        top_level = Path(completed.stdout.decode("utf-8", errors="strict").strip())
        return top_level.resolve(strict=True) == root.resolve(strict=True)
    except (OSError, UnicodeDecodeError):
        return False


def git_index_entries(root: Path) -> tuple[GitIndexEntry, ...]:
    """Return validated regular-file entries from the stage-zero Git index."""

    if not is_exact_git_checkout(root):
        raise ReleaseTreeError("root is not an exact Git checkout")
    completed = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=root,
        check=True,
        env=clean_git_environment(),
        stdout=subprocess.PIPE,
    )
    entries: list[GitIndexEntry] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        if b"\t" not in raw:
            raise ReleaseTreeError("malformed staged release entry")
        raw_header, raw_path = raw.split(b"\t", 1)
        fields = raw_header.split(b" ")
        if len(fields) != 3:
            raise ReleaseTreeError("malformed staged release header")
        raw_mode, raw_object, raw_stage = fields
        if raw_mode not in {b"100644", b"100755"} or raw_stage != b"0":
            raise ReleaseTreeError("release index contains a non-regular or unmerged item")
        if re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", raw_object) is None:
            raise ReleaseTreeError("release index contains an invalid object id")
        entries.append(
            GitIndexEntry(
                raw_mode.decode("ascii"),
                raw_object.decode("ascii"),
                _safe_relative_path(raw_path.decode("utf-8", errors="strict")),
            )
        )
    ordered = tuple(
        sorted(entries, key=lambda value: value.path.as_posix().encode("utf-8"))
    )
    paths = tuple(entry.path for entry in ordered)
    if not ordered or MANIFEST_RELATIVE not in paths:
        raise ReleaseTreeError("Git index is empty or lacks RELEASE_FILES.sha256")
    if len(paths) != len(set(paths)):
        raise ReleaseTreeError("duplicate Git-index release path")
    return ordered


def git_tracked_paths(root: Path) -> tuple[PurePosixPath, ...]:
    """Return safe regular-file index paths in canonical byte order."""

    return tuple(entry.path for entry in git_index_entries(root))


def _parse_git_batch_output(
    raw: bytes, entries: tuple[GitIndexEntry, ...]
) -> dict[PurePosixPath, bytes]:
    """Bind one strict ``git cat-file --batch`` response to index entries."""

    content: dict[PurePosixPath, bytes] = {}
    cursor = 0
    for entry in entries:
        header_end = raw.find(b"\n", cursor)
        if header_end < 0:
            raise ReleaseTreeError(
                f"Git batch header is truncated: {entry.path.as_posix()}"
            )
        header = raw[cursor:header_end]
        cursor = header_end + 1
        expected_object = entry.object_id.encode("ascii")
        if header == expected_object + b" missing":
            raise ReleaseTreeError(
                f"Git batch object is missing: {entry.path.as_posix()}"
            )
        fields = header.split(b" ")
        if len(fields) != 3 or any(not field for field in fields):
            raise ReleaseTreeError(
                f"Git batch header is malformed: {entry.path.as_posix()}"
            )
        raw_object, raw_type, raw_size = fields
        if raw_object != expected_object:
            raise ReleaseTreeError(
                f"Git batch object ID mismatch: {entry.path.as_posix()}"
            )
        if raw_type != b"blob":
            raise ReleaseTreeError(
                f"Git index object is not a blob: {entry.path.as_posix()}"
            )
        if re.fullmatch(rb"(?:0|[1-9][0-9]*)", raw_size) is None:
            raise ReleaseTreeError(
                f"Git batch byte count is malformed: {entry.path.as_posix()}"
            )
        size = int(raw_size)
        payload_end = cursor + size
        if payload_end > len(raw) or raw[payload_end : payload_end + 1] != b"\n":
            raise ReleaseTreeError(
                f"Git batch payload is truncated: {entry.path.as_posix()}"
            )
        content[entry.path] = raw[cursor:payload_end]
        cursor = payload_end + 1
    if cursor != len(raw):
        raise ReleaseTreeError("Git batch output has unexpected trailing bytes")
    return content


def git_index_blobs(
    root: Path, entries: tuple[GitIndexEntry, ...]
) -> dict[PurePosixPath, bytes]:
    """Read all validated index blobs in one strict Git batch transaction."""

    paths = tuple(entry.path for entry in entries)
    canonical = tuple(sorted(paths, key=lambda value: value.as_posix().encode("utf-8")))
    if (
        not entries
        or paths != canonical
        or len(paths) != len(set(paths))
        or MANIFEST_RELATIVE not in paths
    ):
        raise ReleaseTreeError("Git index entries are incomplete or not canonical")
    for entry in entries:
        if entry.mode not in {"100644", "100755"}:
            raise ReleaseTreeError("release index contains a non-regular item")
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", entry.object_id) is None:
            raise ReleaseTreeError("release index contains an invalid object id")
        if _safe_relative_path(entry.path.as_posix()) != entry.path:
            raise ReleaseTreeError("release index contains an invalid path")
    requests = b"".join(entry.object_id.encode("ascii") + b"\n" for entry in entries)
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        check=False,
        env=clean_git_environment(),
        input=requests,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ReleaseTreeError("Git batch object stream failed")
    content = _parse_git_batch_output(completed.stdout, entries)
    if git_index_entries(root) != entries:
        raise ReleaseTreeError("Git index changed while reading release objects")
    return content


def _filesystem_files(root: Path) -> dict[PurePosixPath, Path]:
    files: dict[PurePosixPath, Path] = {}
    for absolute in root.rglob("*"):
        junction = getattr(absolute, "is_junction", None)
        if absolute.is_symlink() or (junction is not None and junction()):
            raise ReleaseTreeError(
                f"archive contains a link or junction: {absolute.relative_to(root)}"
            )
        if absolute.is_dir():
            continue
        if not absolute.is_file():
            raise ReleaseTreeError(
                f"archive contains a non-regular item: {absolute.relative_to(root)}"
            )
        relative = _safe_relative_path(absolute.relative_to(root).as_posix())
        if relative in files:
            raise ReleaseTreeError(f"duplicate archive path: {relative.as_posix()}")
        files[relative] = absolute
    return files


def _load_archive_snapshot(
    root: Path,
) -> tuple[tuple[ManifestEntry, ...], dict[PurePosixPath, bytes]]:
    """Parse, verify, and snapshot every byte in one archive release tree."""

    destination = root / MANIFEST_RELATIVE.as_posix()
    if destination.is_symlink() or not destination.is_file():
        raise ReleaseTreeError("archive release manifest is missing or not regular")
    raw_manifest = destination.read_bytes()
    if not raw_manifest.startswith(MANIFEST_HEADER):
        raise ReleaseTreeError("archive release manifest header is invalid")
    body = raw_manifest[len(MANIFEST_HEADER) :]
    if not body or not body.endswith(b"\n"):
        raise ReleaseTreeError("archive release manifest has no canonical rows")

    entries: list[ManifestEntry] = []
    for raw_row in body[:-1].split(b"\n"):
        try:
            row = raw_row.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ReleaseTreeError("manifest rows must be valid UTF-8") from exc
        fields = row.split("\t")
        if len(fields) != 3:
            raise ReleaseTreeError(f"malformed manifest row: {row!r}")
        digest, raw_size, decoded_path = fields
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise ReleaseTreeError(f"invalid manifest SHA-256: {digest!r}")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ReleaseTreeError(f"invalid manifest byte count: {raw_size!r}") from exc
        if size < 0 or str(size) != raw_size:
            raise ReleaseTreeError(f"non-canonical manifest byte count: {raw_size!r}")
        path = _safe_relative_path(decoded_path)
        if path == MANIFEST_RELATIVE:
            raise ReleaseTreeError("release manifest must exclude itself")
        entries.append(ManifestEntry(digest, size, path))

    ordered_paths = tuple(entry.path for entry in entries)
    canonical_paths = tuple(
        sorted(ordered_paths, key=lambda value: value.as_posix().encode("utf-8"))
    )
    if ordered_paths != canonical_paths or len(ordered_paths) != len(set(ordered_paths)):
        raise ReleaseTreeError("manifest paths are duplicate or not canonically ordered")

    files = _filesystem_files(root)
    expected_paths = set(ordered_paths) | {MANIFEST_RELATIVE}
    actual_paths = set(files)
    missing = sorted(expected_paths - actual_paths, key=lambda value: value.as_posix())
    unexpected = sorted(actual_paths - expected_paths, key=lambda value: value.as_posix())
    if missing or unexpected:
        raise ReleaseTreeError(
            "archive file set differs from manifest: "
            f"missing={[path.as_posix() for path in missing]!r}, "
            f"unexpected={[path.as_posix() for path in unexpected]!r}"
        )

    content: dict[PurePosixPath, bytes] = {MANIFEST_RELATIVE: raw_manifest}
    for entry in entries:
        raw = files[entry.path].read_bytes()
        if len(raw) != entry.size:
            raise ReleaseTreeError(f"archive byte count mismatch: {entry.path.as_posix()}")
        if hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise ReleaseTreeError(f"archive SHA-256 mismatch: {entry.path.as_posix()}")
        content[entry.path] = raw
    return tuple(entries), content


def load_archive_manifest(root: Path) -> tuple[ManifestEntry, ...]:
    """Parse the manifest and bind it to every regular file in an archive tree."""

    entries, _content = _load_archive_snapshot(root)
    return entries


def resolve_release_tree(root: Path) -> ReleaseTree:
    """Resolve paths and content reads through one consistent authority."""

    if is_exact_git_checkout(root):
        entries = git_index_entries(root)
        content = git_index_blobs(root, entries)
        return ReleaseTree(
            root,
            tuple(entry.path for entry in entries),
            "git-index",
            index_content=content,
        )
    entries, content = _load_archive_snapshot(root)
    paths = tuple(entry.path for entry in entries) + (MANIFEST_RELATIVE,)
    ordered = tuple(sorted(paths, key=lambda value: value.as_posix().encode("utf-8")))
    return ReleaseTree(root, ordered, "archive-manifest", content)


def release_paths(root: Path) -> tuple[tuple[PurePosixPath, ...], str]:
    """Return paths and authority for callers that do not read file content."""

    tree = resolve_release_tree(root)
    return tree.paths, tree.authority
