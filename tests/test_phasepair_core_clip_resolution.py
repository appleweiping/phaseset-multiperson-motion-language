from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from phasepair_core import clip_resolution as clip


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "pytorch_model.bin",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
)


def tiny_snapshot(root: Path) -> tuple[tuple[clip.ClipFilePin, ...], int]:
    pins: list[clip.ClipFilePin] = []
    for index, name in enumerate(NAMES):
        content = b"phasepair-clip-test-v1\x00" + bytes((index,)) + name.encode("ascii")
        (root / name).write_bytes(content)
        pins.append(clip.ClipFilePin(name, len(content), hashlib.sha256(content).hexdigest()))
    return tuple(pins), sum(pin.bytes for pin in pins)


def runtime_evidence() -> clip.RuntimeWheelEvidence:
    return clip.RuntimeWheelEvidence(
        (
            clip.WheelIdentity("transformers", "transformers-test.whl", 101, "1" * 64),
            clip.WheelIdentity(
                "huggingface-hub", "huggingface_hub-test.whl", 102, "2" * 64
            ),
            clip.WheelIdentity("tokenizers", "tokenizers-test.whl", 103, "3" * 64),
        )
    )


def loader_evidence() -> clip.TextLoaderEvidence:
    return clip.TextLoaderEvidence(
        "CLIPTextModelWithProjection",
        "CLIPTokenizerFast",
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        0,
        77,
    )


def trace_evidence() -> clip.OpenTraceEvidence:
    return clip.OpenTraceEvidence(
        NAMES,
        (
            "config.json",
            "pytorch_model.bin",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        (),
    )


def rights_evidence() -> clip.RightsEvidence:
    return clip.RightsEvidence(
        "PRIVATE_RESEARCH_ONLY_NO_REDISTRIBUTION",
        "4" * 64,
        True,
        False,
        False,
    )


def golden_evidence() -> clip.SyntheticCaptionGoldenEvidence:
    return clip.SyntheticCaptionGoldenEvidence(
        3,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        77,
        512,
    )


class WorkspaceTemporaryDirectory:
    """A local non-reparse temporary root on the same volume as the project."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.path = Path(self._temporary.name)

    def cleanup(self) -> None:
        self._temporary.cleanup()


class ClipResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = WorkspaceTemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def test_exact_target_eight_pins_and_total(self) -> None:
        expected = (
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
        self.assertEqual(clip.MODEL_ID, "openai/clip-vit-base-patch32")
        self.assertEqual(clip.REVISION, "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
        self.assertEqual(clip.PINNED_FILES, expected)
        self.assertEqual(sum(row[1] for row in expected), 608_863_580)
        self.assertEqual(clip.TOTAL_BYTES, 608_863_580)

    def test_public_missing_root_is_canonical_authority_zero_hold(self) -> None:
        root = self.temporary.path / "not-present"
        assessment = clip.assess_local_clip_snapshot(root)
        self.assertEqual(assessment.status, "HOLD_ROOT_MISSING")
        self.assertEqual(assessment.authority, 0)
        self.assertIs(assessment.production, False)
        self.assertIs(assessment.training_authorized, False)
        payload = json.loads(clip.canonical_assessment_bytes(assessment))
        self.assertEqual(
            set(payload),
            {
                "authority",
                "evidence",
                "extra_files",
                "files",
                "missing_files",
                "model_id",
                "production",
                "revision",
                "schema",
                "status",
                "total_bytes",
                "training_authorized",
            },
        )
        self.assertEqual(
            set(payload["evidence"]),
            {"golden", "loader", "open_trace", "rights", "runtime"},
        )
        self.assertTrue(payload["status"].startswith("HOLD_"))
        self.assertFalse(payload["production"])
        self.assertFalse(payload["training_authorized"])
        self.assertTrue(clip.canonical_assessment_bytes(assessment).endswith(b"\n"))

    def test_public_api_rejects_strings_and_identity_injection(self) -> None:
        parameters = tuple(inspect.signature(clip.assess_local_clip_snapshot).parameters)
        self.assertEqual(parameters, ("root",))
        with self.assertRaises(TypeError):
            clip.assess_local_clip_snapshot(str(self.temporary.path))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            clip.assess_local_clip_snapshot(  # type: ignore[call-arg]
                self.temporary.path,
                pins=(),
            )

    def test_tiny_private_seam_exercises_complete_tree_without_large_files(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_RUNTIME_WHEELS_ABSENT")
        self.assertEqual(tuple(item.path for item in assessment.files), NAMES)
        self.assertEqual(assessment.total_bytes, total)
        self.assertEqual(
            tuple(item.sha256 for item in assessment.files),
            tuple(pin.sha256 for pin in pins),
        )
        with self.assertRaises(FrozenInstanceError):
            assessment.status = "HOLD_ROOT_MISSING"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            assessment.files[0].path = "config.json"  # type: ignore[misc]

    def test_each_absent_evidence_has_an_explicit_hold_and_complete_still_holds(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        bundles = (
            (clip.ClipResolutionEvidence(), "HOLD_RUNTIME_WHEELS_ABSENT"),
            (
                clip.ClipResolutionEvidence(runtime=runtime_evidence()),
                "HOLD_TEXT_LOADER_EVIDENCE_ABSENT",
            ),
            (
                clip.ClipResolutionEvidence(
                    runtime=runtime_evidence(),
                    loader=loader_evidence(),
                ),
                "HOLD_OPEN_TRACE_EVIDENCE_ABSENT",
            ),
            (
                clip.ClipResolutionEvidence(
                    runtime_evidence(),
                    loader_evidence(),
                    trace_evidence(),
                ),
                "HOLD_RIGHTS_EVIDENCE_ABSENT",
            ),
            (
                clip.ClipResolutionEvidence(
                    runtime_evidence(),
                    loader_evidence(),
                    trace_evidence(),
                    rights_evidence(),
                ),
                "HOLD_SYNTHETIC_GOLDEN_ABSENT",
            ),
            (
                clip.ClipResolutionEvidence(
                    runtime_evidence(),
                    loader_evidence(),
                    trace_evidence(),
                    rights_evidence(),
                    golden_evidence(),
                ),
                "HOLD_FRESH_REVIEW_REQUIRED",
            ),
        )
        for evidence, expected_status in bundles:
            with self.subTest(expected_status=expected_status):
                assessment = clip._assess_local_clip_snapshot_for_tests(
                    self.temporary.path,
                    pins=pins,
                    expected_total_bytes=total,
                    evidence=evidence,
                )
                self.assertEqual(assessment.status, expected_status)
                self.assertTrue(assessment.status.startswith("HOLD_"))
                self.assertEqual(assessment.authority, 0)
                self.assertFalse(assessment.production)
                self.assertFalse(assessment.training_authorized)

    def test_evidence_schemas_are_closed_and_fail_closed(self) -> None:
        with self.assertRaises(clip.ClipResolutionError):
            clip.ClipResolutionEvidence(runtime={})  # type: ignore[arg-type]
        with self.assertRaises(clip.ClipResolutionError):
            clip.TextLoaderEvidence(
                "CLIPTextModelWithProjection",
                "CLIPTokenizerFast",
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                0,
                77,
            )
        with self.assertRaises(clip.ClipResolutionError):
            clip.TextLoaderEvidence(
                "CLIPTextModelWithProjection",
                "CLIPTokenizerFast",
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                1,
                77,
            )
        with self.assertRaises(clip.ClipResolutionError):
            clip.OpenTraceEvidence(NAMES, ("config.json",), ("outside.json",))
        with self.assertRaises(clip.ClipResolutionError):
            clip.RightsEvidence(
                "PRIVATE_RESEARCH_ONLY_NO_REDISTRIBUTION",
                "a" * 64,
                True,
                True,
                False,
            )
        with self.assertRaises(clip.ClipResolutionError):
            clip.SyntheticCaptionGoldenEvidence(
                2,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                77,
                512,
            )

    def test_contract_strings_reject_str_subclasses(self) -> None:
        class StringSubclass(str):
            pass

        with self.assertRaises(clip.ClipResolutionError):
            clip.TextLoaderEvidence(
                StringSubclass("CLIPTextModelWithProjection"),
                "CLIPTokenizerFast",
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                0,
                77,
            )
        with self.assertRaises(clip.ClipResolutionError):
            clip.TextLoaderEvidence(
                "CLIPTextModelWithProjection",
                StringSubclass("CLIPTokenizerFast"),
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                0,
                77,
            )
        with self.assertRaises(clip.ClipResolutionError):
            clip.RightsEvidence(
                StringSubclass("PRIVATE_RESEARCH_ONLY_NO_REDISTRIBUTION"),
                "a" * 64,
                True,
                False,
                False,
            )

    def test_missing_extra_census_and_content_tamper_hold(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        missing_path = self.temporary.path / "config.json"
        missing_path.unlink()
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_SNAPSHOT_MISSING_FILES")
        self.assertEqual(assessment.missing_files, ("config.json",))

        missing_path.write_bytes(
            b"phasepair-clip-test-v1\x00" + bytes((1,)) + b"config.json"
        )
        extra_path = self.temporary.path / "unexpected.txt"
        extra_path.write_bytes(b"unexpected")
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_SNAPSHOT_EXTRA_FILES")
        self.assertEqual(assessment.extra_files, ("unexpected.txt",))

        missing_path.unlink()
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_SNAPSHOT_CENSUS_MISMATCH")
        extra_path.unlink()
        original = b"phasepair-clip-test-v1\x00" + bytes((1,)) + b"config.json"
        missing_path.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_FILE_SHA256_MISMATCH")
        missing_path.write_bytes(original + b"x")
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_FILE_SIZE_MISMATCH")

    def test_nonregular_and_hardlink_aliases_are_rejected(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        weight = self.temporary.path / "pytorch_model.bin"
        outside_link = self.temporary.path.parent / f"{self.temporary.path.name}-hardlink"
        self.addCleanup(lambda: outside_link.exists() and outside_link.unlink())
        os.link(weight, outside_link)
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_FILE_LINK_COUNT")
        outside_link.unlink()
        weight.unlink()
        weight.mkdir()
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_FILE_NONREGULAR")

    def test_file_and_root_symlinks_or_reparse_points_are_rejected(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        target = self.temporary.path.parent / f"{self.temporary.path.name}-target"
        target.write_bytes(b"target")
        self.addCleanup(lambda: target.exists() and target.unlink())
        expected = self.temporary.path / "pytorch_model.bin"
        expected.unlink()
        try:
            os.symlink(target, expected)
        except OSError as exc:
            self.skipTest(f"file symlinks unavailable: {exc}")
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_FILE_REPARSE")

        link_root = self.temporary.path.parent / f"{self.temporary.path.name}-root-link"
        self.addCleanup(lambda: link_root.exists() and link_root.unlink())
        try:
            os.symlink(self.temporary.path, link_root, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        assessment = clip._assess_local_clip_snapshot_for_tests(
            link_root,
            pins=pins,
            expected_total_bytes=total,
        )
        self.assertEqual(assessment.status, "HOLD_ROOT_REPARSE")

    def test_reparse_metadata_is_rejected_without_symlink_privileges(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        expected = self.temporary.path / "pytorch_model.bin"

        def with_reparse_attribute(value: os.stat_result) -> SimpleNamespace:
            return SimpleNamespace(
                st_dev=value.st_dev,
                st_ino=value.st_ino,
                st_mode=value.st_mode,
                st_nlink=value.st_nlink,
                st_size=value.st_size,
                st_mtime_ns=value.st_mtime_ns,
                st_ctime_ns=value.st_ctime_ns,
                st_file_attributes=getattr(value, "st_file_attributes", 0) | 0x400,
            )

        def file_reparse_lstat(path: object) -> object:
            value = os.lstat(path)
            if Path(path) == expected:
                return with_reparse_attribute(value)
            return value

        file_operations = clip._FilesystemOps(
            file_reparse_lstat,
            os.scandir,
            os.open,
            os.fstat,
            os.read,
            os.close,
        )
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
            operations=file_operations,
        )
        self.assertEqual(assessment.status, "HOLD_FILE_REPARSE")

        resolved_root = self.temporary.path.resolve()

        def root_reparse_lstat(path: object) -> object:
            value = os.lstat(path)
            if Path(path).resolve() == resolved_root:
                return with_reparse_attribute(value)
            return value

        root_operations = clip._FilesystemOps(
            root_reparse_lstat,
            os.scandir,
            os.open,
            os.fstat,
            os.read,
            os.close,
        )
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
            operations=root_operations,
        )
        self.assertEqual(assessment.status, "HOLD_ROOT_REPARSE")

    def test_before_handle_after_identity_change_is_toctou_hold(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        fstat_calls = 0

        def changing_fstat(fd: int) -> object:
            nonlocal fstat_calls
            value = os.fstat(fd)
            fstat_calls += 1
            delta = 1 if fstat_calls == 2 else 0
            return SimpleNamespace(
                st_dev=value.st_dev,
                st_ino=value.st_ino,
                st_mode=value.st_mode,
                st_nlink=value.st_nlink,
                st_size=value.st_size,
                st_mtime_ns=value.st_mtime_ns + delta,
                st_ctime_ns=value.st_ctime_ns,
                st_file_attributes=getattr(value, "st_file_attributes", 0),
            )

        operations = clip._FilesystemOps(
            os.lstat,
            os.scandir,
            os.open,
            changing_fstat,
            os.read,
            os.close,
        )
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
            operations=operations,
        )
        self.assertEqual(assessment.status, "HOLD_FILE_TOCTOU")

    def test_second_pass_late_equal_length_mutation_of_first_file_is_rejected(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        first_file = self.temporary.path / "pytorch_model.bin"
        original = first_file.read_bytes()
        original_stat = first_file.stat()
        open_count = 0
        fd_ordinals: dict[int, int] = {}
        mutated = False

        def tracking_open(path: object, flags: int) -> int:
            nonlocal open_count
            fd = os.open(path, flags)
            open_count += 1
            fd_ordinals[fd] = open_count
            return fd

        def mutating_read(fd: int, count: int) -> bytes:
            nonlocal mutated
            # Open 10 is file 2 of pass 2: file 1 was already rehashed in that
            # pass, so only the closing all-file metadata sweep can catch this.
            if fd_ordinals[fd] == 10 and not mutated:
                replacement = bytes((original[0] ^ 1,)) + original[1:]
                self.assertEqual(len(replacement), len(original))
                first_file.write_bytes(replacement)
                # Equal-length writes may share one filesystem timestamp tick.
                # This test targets the closing METADATA sweep, so make that
                # observable premise explicit instead of depending on elapsed
                # wall time. The production scanner remains best-effort.
                os.utime(first_file, ns=(original_stat.st_atime_ns,
                                       original_stat.st_mtime_ns + 2_000_000_000))
                self.assertNotEqual(first_file.stat().st_mtime_ns, original_stat.st_mtime_ns)
                mutated = True
            return os.read(fd, count)

        def tracking_close(fd: int) -> None:
            try:
                os.close(fd)
            finally:
                fd_ordinals.pop(fd, None)

        operations = clip._FilesystemOps(
            os.lstat,
            os.scandir,
            tracking_open,
            os.fstat,
            mutating_read,
            tracking_close,
        )
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
            operations=operations,
        )
        self.assertTrue(mutated)
        self.assertEqual(open_count, 16)
        self.assertEqual(assessment.status, "HOLD_FILE_TOCTOU")

    def test_output_construction_and_canonicalization_reject_forgery(self) -> None:
        with self.assertRaises(TypeError):
            clip.ClipFileSnapshot()
        with self.assertRaises(TypeError):
            clip.ClipResolutionAssessment()
        pins, total = tiny_snapshot(self.temporary.path)
        forged = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        object.__setattr__(forged, "status", "PASS")
        with self.assertRaises(clip.ClipResolutionError):
            clip.canonical_assessment_bytes(forged)
        object.__setattr__(forged, "status", "HOLD_ROOT_MISSING")
        object.__setattr__(forged, "authority", 1)
        with self.assertRaises(clip.ClipResolutionError):
            clip.canonical_assessment_bytes(forged)

    def test_canonical_rebuild_rejects_scan_evidence_and_unissued_forgery(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)

        status_mutant = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        object.__setattr__(status_mutant, "status", "HOLD_ROOT_MISSING")
        with self.assertRaises(clip.ClipResolutionError):
            clip.canonical_assessment_bytes(status_mutant)

        file_mutant = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        object.__setattr__(file_mutant.files[0], "sha256", "f" * 64)
        with self.assertRaises(clip.ClipResolutionError):
            clip.canonical_assessment_bytes(file_mutant)

        evidence = clip.ClipResolutionEvidence(
            runtime_evidence(),
            loader_evidence(),
            trace_evidence(),
            rights_evidence(),
            golden_evidence(),
        )
        evidence_mutant = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
            evidence=evidence,
        )
        object.__setattr__(
            evidence_mutant.evidence.rights,
            "verdict",
            "PRIVATE_RESEARCH_ONLY_NO_REDISTRIBUTION_CHANGED",
        )
        with self.assertRaises(clip.ClipResolutionError):
            clip.canonical_assessment_bytes(evidence_mutant)

        valid = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        forged = object.__new__(type(valid))
        for field in (
            "status",
            "authority",
            "production",
            "training_authorized",
            "files",
            "total_bytes",
            "missing_files",
            "extra_files",
            "evidence",
        ):
            object.__setattr__(forged, field, getattr(valid, field))
        with self.assertRaises(clip.ClipResolutionError):
            clip.canonical_assessment_bytes(forged)

        equivalent_evidence_mutant = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        object.__setattr__(
            equivalent_evidence_mutant,
            "evidence",
            clip.ClipResolutionEvidence(),
        )
        with self.assertRaises(clip.ClipResolutionError):
            clip.canonical_assessment_bytes(equivalent_evidence_mutant)

    def test_canonical_rescans_and_rejects_files_changed_after_issue(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        assessment = clip._assess_local_clip_snapshot_for_tests(
            self.temporary.path,
            pins=pins,
            expected_total_bytes=total,
        )
        config = self.temporary.path / "config.json"
        original = config.read_bytes()
        config.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
        with self.assertRaisesRegex(
            clip.ClipResolutionError,
            "current double-pass snapshot differs",
        ):
            clip.canonical_assessment_bytes(assessment)

    def test_weak_lease_registry_returns_to_baseline_after_256_assessments(self) -> None:
        # Earlier tests may leave unreachable exception/closure cycles. They
        # must be collected before, not during, this relative-count assertion.
        gc.collect()
        baseline = clip._issued_lease_count_for_tests()
        missing = self.temporary.path / "lease-missing"
        assessments = [clip.assess_local_clip_snapshot(missing) for _ in range(256)]
        self.assertEqual(clip._issued_lease_count_for_tests(), baseline + 256)
        lease_refs = [weakref.ref(assessment._lease) for assessment in assessments]
        del assessments
        gc.collect()
        self.assertTrue(all(reference() is None for reference in lease_refs))
        self.assertEqual(clip._issued_lease_count_for_tests(), baseline)

    def test_public_and_private_informational_global_rebinds_have_no_authority(self) -> None:
        missing = self.temporary.path / "still-missing"
        with mock.patch.multiple(
            clip,
            MODEL_ID="attacker/model",
            REVISION="main",
            TOTAL_BYTES=1,
            PINNED_FILES=(),
            AUTHORITY=9,
            PRODUCTION=True,
            TRAINING_AUTHORIZED=True,
            _INTERNAL_PIN_CACHE=(),
            _INTERNAL_TOTAL_BYTES=1,
            _INTERNAL_MODEL_ID="attacker/model",
            _INTERNAL_REVISION="main",
        ):
            assessment = clip.assess_local_clip_snapshot(missing)
            payload = json.loads(clip.canonical_assessment_bytes(assessment))
        self.assertEqual(payload["model_id"], "openai/clip-vit-base-patch32")
        self.assertEqual(
            payload["revision"], "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        )
        self.assertEqual(payload["authority"], 0)
        self.assertFalse(payload["production"])
        self.assertFalse(payload["training_authorized"])
        self.assertEqual(payload["status"], "HOLD_ROOT_MISSING")

    def test_simultaneous_private_validator_assessor_sha_and_json_rebind_is_inert(self) -> None:
        pins, total = tiny_snapshot(self.temporary.path)
        sealed_private = clip._assess_local_clip_snapshot_for_tests
        sealed_public = clip.assess_local_clip_snapshot
        sealed_canonical = clip.canonical_assessment_bytes
        missing = self.temporary.path / "sealed-missing"

        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("rebound helper was dynamically trusted")

        with (
            mock.patch.multiple(
                clip,
                MODEL_ID="attacker/model",
                REVISION="main",
                TOTAL_BYTES=1,
                PINNED_FILES=(),
                AUTHORITY=9,
                PRODUCTION=True,
                TRAINING_AUTHORIZED=True,
                ClipFilePin=object,
                ClipResolutionAssessment=object,
                ClipResolutionError=object,
                ClipResolutionEvidence=object,
                _AssessmentLease=object,
                Path=object,
                _assess_local_clip_snapshot_for_tests=forbidden,
                _validate_assessment=forbidden,
                _make_assessment=forbidden,
                _make_file_snapshot=forbidden,
                _copy_evidence=forbidden,
                _validate_pins=forbidden,
                _scan_snapshot=forbidden,
                _scan_snapshot_once=forbidden,
                _read_directory_census=forbidden,
                _hash_one_file=forbidden,
                _resolve_root=forbidden,
                _check_root_components=forbidden,
                _component_status=forbidden,
                _stat_identity=forbidden,
                _is_reparse=forbidden,
                _same_path_and_handle_identity=forbidden,
                _evidence_hold_status=forbidden,
                _known_statuses=lambda: {"PASS"},
                _require_sha256=forbidden,
                _require_exact_str=forbidden,
            ),
            mock.patch.object(clip.hashlib, "sha256", side_effect=forbidden),
            mock.patch.object(clip.json, "dumps", side_effect=forbidden),
        ):
            private_assessment = sealed_private(
                self.temporary.path,
                pins=pins,
                expected_total_bytes=total,
            )
            private_bytes = sealed_canonical(private_assessment)
            public_assessment = sealed_public(missing)
            public_bytes = sealed_canonical(public_assessment)

        private_payload = json.loads(private_bytes)
        public_payload = json.loads(public_bytes)
        self.assertEqual(private_payload["status"], "HOLD_RUNTIME_WHEELS_ABSENT")
        self.assertEqual(len(private_payload["files"]), 8)
        self.assertEqual(public_payload["status"], "HOLD_ROOT_MISSING")
        for payload in (private_payload, public_payload):
            self.assertEqual(payload["authority"], 0)
            self.assertFalse(payload["production"])
            self.assertFalse(payload["training_authorized"])
            self.assertEqual(payload["model_id"], "openai/clip-vit-base-patch32")
            self.assertEqual(
                payload["revision"],
                "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            )

    def test_import_and_public_missing_call_need_no_ml_or_network_packages(self) -> None:
        code = r'''
import builtins
import socket
from pathlib import Path

real_import = builtins.__import__
forbidden = {"transformers", "huggingface_hub", "tokenizers", "requests", "httpx"}

def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise AssertionError("forbidden import: " + name)
    return real_import(name, *args, **kwargs)

def forbidden_socket(*args, **kwargs):
    raise AssertionError("network access attempted")

builtins.__import__ = guarded_import
socket.socket = forbidden_socket
from phasepair_core import clip_resolution
assessment = clip_resolution.assess_local_clip_snapshot(Path("definitely-not-present"))
assert assessment.status == "HOLD_ROOT_MISSING"
print(assessment.status)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "HOLD_ROOT_MISSING")


if __name__ == "__main__":
    unittest.main()
