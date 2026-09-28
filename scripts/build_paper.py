"""Build the current manuscript with the hash-pinned official ICASSP 2027 kit.

This compiles existing content. It never creates scientific results or grants
submission readiness. Logs and receipts stay in a separate ignored build tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fetch_icassp2027_template import main as fetch_template


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-archive", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    paper = root / "paper"
    venue = paper / ".venue"
    template_args = ["--destination", str(venue)]
    if args.template_archive is not None:
        template_args.extend(["--archive", str(args.template_archive)])
    fetch_template(template_args)
    executables = {name: shutil.which(name) for name in ("pdflatex", "bibtex")}
    if any(value is None for value in executables.values()):
        raise RuntimeError("pdflatex and bibtex must be installed and on PATH")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_directory or paper / "build" / stamp).resolve()
    # Never mix this attempt with another build or overwrite tracked paper files.
    if output.exists() or output.is_symlink():
        raise FileExistsError("paper build output must be a new directory")
    output.mkdir(parents=True)
    env = dict(os.environ)
    env["TEXINPUTS"] = str(venue) + os.pathsep
    env["BSTINPUTS"] = str(venue) + os.pathsep
    env["BIBINPUTS"] = str(paper) + os.pathsep
    tex_command = [
        executables["pdflatex"], "-no-shell-escape", "-interaction=nonstopmode",
        "-halt-on-error", "-output-directory", str(output), "main.tex",
    ]
    commands = [tex_command, [executables["bibtex"], str(output / "main")],
                tex_command, tex_command]
    for index, command in enumerate(commands, start=1):
        with (output / f"pass-{index}.log").open("xb") as log:
            result = subprocess.run(command, cwd=paper, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180, check=False)
        if result.returncode:
            raise RuntimeError(f"paper pass {index} failed; inspect its private build log")
    pdf = output / "main.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError("paper build produced no PDF")
    log_text = (output / "main.log").read_text(encoding="utf-8", errors="replace")
    for marker in ("There were undefined references", "There were multiply-defined labels"):
        if marker in log_text:
            raise RuntimeError("paper has unresolved references or duplicate labels")
    receipt = {
        "schema": "phaseset-paper-build-v1", "created_at_utc": stamp,
        "scientific_results_created": False, "submission_ready_asserted": False,
        "source_sha256": file_sha256(paper / "main.tex"),
        "bibliography_sha256": file_sha256(paper / "references.bib"),
        "spconf_sha256": file_sha256(venue / "spconf.sty"),
        "ieeebib_sha256": file_sha256(venue / "IEEEbib.bst"),
        "pdf_sha256": file_sha256(pdf), "pdf_bytes": pdf.stat().st_size,
        "overfull_hbox_count": log_text.count("Overfull \\hbox"),
        "overfull_vbox_count": log_text.count("Overfull \\vbox"),
        "underfull_hbox_count": log_text.count("Underfull \\hbox"),
        "underfull_vbox_count": log_text.count("Underfull \\vbox"),
    }
    with (output / "build-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print("PAPER_BUILD_COMPLETED " + receipt["pdf_sha256"])
    print("Inspect page boundaries, embedded fonts, figures, and claims before publication.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"PAPER_BUILD_FAILED: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from exc
