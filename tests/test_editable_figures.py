"""Pure-document checks: no model, data, or numerical experiment is run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@pytest.mark.parametrize("stem", ["phaseset_motivation", "phaseset_architecture"])
def test_native_editable_figure_census_and_source_binding(stem: str) -> None:
    figures = ROOT / "figures"
    report = json.loads((figures / "EDITABLE_FIGURES.json").read_text(encoding="ascii"))
    assert report["schema"] == "phaseset-editable-figure-census-v2"
    renderer = (ROOT / "scripts" / report["renderer"]["name"]).read_bytes()
    assert len(renderer) == report["renderer"]["bytes"]
    assert hashlib.sha256(renderer).hexdigest() == report["renderer"]["sha256"]
    assert report["runtime"]["python_pptx"] == "1.0.2"
    assert report["template"]["sha256"] == "e10cc9e120961f6bd4074a373c9c80d2a06c497157e8f4972977b7bea83a8f34"
    rows = [row for row in report["figures"] if row["source"] == stem + ".drawio"]
    assert len(rows) == 1
    row = rows[0]
    source = (figures / row["source"]).read_bytes()
    pptx = figures / row["pptx"]
    assert hashlib.sha256(source).hexdigest() == row["source_sha256"]
    assert hashlib.sha256(pptx.read_bytes()).hexdigest() == row["pptx_sha256"]
    assert row["raster_media"] == 0 and row["empirical_result"] is False
    model = ET.fromstring(source).find("./diagram/mxGraphModel")
    assert model is not None
    cells = model.findall("./root/mxCell")
    vertices = [cell for cell in cells if cell.get("vertex") == "1"]
    edges = [cell for cell in cells if cell.get("edge") == "1"]
    with zipfile.ZipFile(pptx) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert len(names) == report["package_policy"]["retained_part_count"] == row["package_parts"]
        assert not any(name.startswith(("ppt/media/", "ppt/embeddings/")) for name in names)
        assert not any("vba" in name.lower() for name in names)
        assert not any("thumbnail" in name.lower() or "printersettings" in name.lower() for name in names)
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.create_system == report["package_policy"]["create_system"] == 0
            assert info.external_attr == report["package_policy"]["external_attr"]
            raw = archive.read(info)
            assert b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw
        for name in names:
            if name.endswith(".rels"):
                relationships = ET.fromstring(archive.read(name))
                assert all(item.get("TargetMode") != "External" for item in relationships)
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        shapes = slide.findall(".//p:sp", NS)
        connectors = slide.findall(".//p:cxnSp", NS)
        assert len(shapes) == row["native_shapes"] >= len(vertices)
        assert len(connectors) == row["native_connector_segments"] >= len(edges)
        assert not slide.findall(".//p:pic", NS)
        object_names = {item.get("name") for item in slide.findall(".//p:cNvPr", NS)}
        assert all("drawio-node-" + cell.attrib["id"] in object_names for cell in vertices)
        assert all(any(name and name.startswith("drawio-edge-" + cell.attrib["id"] + "-segment-")
                       for name in object_names) for cell in edges)
        assert slide.findall(".//a:t", NS), "labels must remain native text"
        app = ET.fromstring(archive.read("docProps/app.xml"))
        app_ns = {"app": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"}
        assert app.findtext("app:Slides", namespaces=app_ns) == "1"
        core = archive.read("docProps/core.xml")
        assert row["source_sha256"].encode("ascii") in core


def test_embedded_pngs_have_complete_iend() -> None:
    for stem in ("phaseset_motivation", "phaseset_architecture"):
        raw = (ROOT / "figures" / (stem + ".png")).read_bytes()
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert raw.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
