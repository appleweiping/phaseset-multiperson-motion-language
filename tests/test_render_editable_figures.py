"""Document-only regression checks for the hardened native figure renderer."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zipfile

import pytest

pytest.importorskip("pptx", minversion="1.0.2")
SPEC = importlib.util.spec_from_file_location(
    "render_editable_figures", Path(__file__).resolve().parents[1] / "scripts" / "render_editable_figures.py"
)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)
try:
    renderer.runtime_receipt()
except RuntimeError as error:
    if str(error) == "registered canonical-build runtime identity mismatch":
        pytest.skip("canonical PPTX regeneration requires its registered document runtime", allow_module_level=True)
    raise


HERE = Path(__file__).resolve().parents[1] / "figures"
EXPECTED_SOURCES = {
    "phaseset_motivation.drawio": (
        "a0a25fbc8c55575d50c6ffa8d6add01ce780cc8842328bb3f6bba20c6509d18a"
    ),
    "phaseset_architecture.drawio": (
        "1a61a503c935a72cbfbe1ce175ab54bf975eb3e0a2164018aeda80448f472a1a"
    ),
}


class HardenedEditableFigureTests(unittest.TestCase):
    def test_sources_are_unchanged_and_pptx_is_reproducible(self) -> None:
        receipt, template = renderer.runtime_receipt()
        self.assertEqual(receipt["runtime"], renderer.REGISTERED_RUNTIME)
        self.assertEqual(renderer.REGISTERED_SOURCES, EXPECTED_SOURCES)
        self.assertEqual(
            receipt["template"]["sha256"],
            renderer.DEFAULT_TEMPLATE_SHA256,
        )
        with tempfile.TemporaryDirectory(prefix="phaseset-pptx-hardened-test-") as temporary:
            root = Path(temporary)
            for filename, expected_sha256 in EXPECTED_SOURCES.items():
                source = HERE / filename
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected_sha256)
                stem = source.stem
                first = root / f"{stem}-first.pptx"
                second = root / f"{stem}-second.pptx"
                first_row = renderer.render(source, first, template)
                second_row = renderer.render(source, second, template)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertEqual(first_row["pptx_sha256"], second_row["pptx_sha256"])
                self.assertEqual(first_row["raster_media"], 0)
                self.assertEqual(first_row["external_relationships"], 0)
                self._assert_package(first)

    def test_unregistered_runtime_is_rejected(self) -> None:
        with (
            mock.patch.object(renderer.platform, "python_version", return_value="0.0.0-mutant"),
            self.assertRaisesRegex(RuntimeError, "canonical-build runtime identity"),
        ):
            renderer.runtime_receipt()

    def test_modified_source_is_rejected(self) -> None:
        _, template = renderer.runtime_receipt()
        source = HERE / "phaseset_motivation.drawio"
        with tempfile.TemporaryDirectory(prefix="phaseset-pptx-source-mutant-") as temporary:
            root = Path(temporary)
            tree = ET.fromstring(source.read_bytes())
            vertex = next(
                item
                for item in tree.findall("./diagram/mxGraphModel/root/mxCell")
                if item.get("vertex") == "1"
            )
            original_id = vertex.attrib["id"]
            mutant_id = original_id + "-tampered"
            for item in tree.iter():
                for attribute in ("id", "parent", "source", "target"):
                    if item.get(attribute) == original_id:
                        item.set(attribute, mutant_id)
            mutant = root / source.name
            mutant.write_bytes(ET.tostring(tree, encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "figure source identity mismatch"):
                renderer.render(mutant, root / "mutant.pptx", template)
            renamed = root / "unregistered-name.drawio"
            renamed.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "figure source identity mismatch"):
                renderer.render(renamed, root / "renamed.pptx", template)

    def test_repointed_relationship_is_rejected(self) -> None:
        _, template = renderer.runtime_receipt()
        source = HERE / "phaseset_motivation.drawio"
        with tempfile.TemporaryDirectory(prefix="phaseset-pptx-opc-mutant-") as temporary:
            destination = Path(temporary) / "canonical.pptx"
            renderer.render(source, destination, template)
            with zipfile.ZipFile(destination) as archive:
                parts = {name: archive.read(name) for name in archive.namelist()}
            relationships = ET.fromstring(parts["_rels/.rels"])
            office_document = next(
                item
                for item in relationships
                if item.get("Type", "").endswith("/officeDocument")
            )
            office_document.set("Target", "docProps/app.xml")
            parts["_rels/.rels"] = renderer.xml_bytes(relationships, renderer.REL_NS)
            mutant = renderer.normalized_zip(parts)
            slide = ET.fromstring(parts["ppt/slides/slide1.xml"])
            namespace = {"p": renderer.PML_NS}
            shape_names = {
                item.get("name", "")
                for item in slide.findall(".//p:sp/p:nvSpPr/p:cNvPr", namespace)
            }
            connector_names = {
                item.get("name", "")
                for item in slide.findall(".//p:cxnSp/p:nvCxnSpPr/p:cNvPr", namespace)
            }
            with self.assertRaisesRegex(ValueError, "relationship tuple census"):
                renderer.validate_hardened_package(mutant, shape_names, connector_names)

    def _assert_package(self, path: Path) -> None:
        presentation = renderer.Presentation(str(path))
        self.assertEqual(len(presentation.slides), 1)
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.namelist(), sorted(renderer.RETAINED_PACKAGE_PARTS))
            self.assertFalse(renderer.REMOVED_TEMPLATE_PARTS & set(archive.namelist()))
            self.assertEqual(archive.comment, b"")
            for info in archive.infolist():
                self.assertEqual(info.date_time, renderer.ZIP_TIMESTAMP)
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(info.create_system, renderer.ZIP_CREATE_SYSTEM)
                self.assertEqual(info.external_attr, renderer.ZIP_EXTERNAL_ATTR)
                self.assertEqual(info.flag_bits, 0)
                self.assertEqual(info.extra, b"")
                self.assertEqual(info.comment, b"")
            parts = {name: archive.read(name) for name in archive.namelist()}
        renderer.validate_content_types(parts)
        renderer.validate_relationships(parts)
        renderer.validate_app_properties(parts)
        for part in parts.values():
            self.assertNotIn(b"<!DOCTYPE", part.upper())
            self.assertNotIn(b"<!ENTITY", part.upper())
            self.assertIsNone(renderer.PRIVATE_PATH_RE.search(part))


if __name__ == "__main__":
    unittest.main()
