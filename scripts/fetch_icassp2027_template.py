"""Fetch only the hash-pinned official conference style, never example papers."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


TEMPLATE_URL = (
    "https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/"
    "ICASSP2027_Paper_Templates.zip"
)
ARCHIVE_SHA256 = "fd1cc4102c4ad9a3e85eba96ee5ed261a45fab9c2630875eb2566812230c91a8"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
STYLE_SHA256 = {
    "spconf.sty": "dc5d632639040cb183f2ab62780f314845aa021be73048c8a9ec9c2072d64a86",
    "IEEEbib.bst": "7e6ca0c8b72158d504a12bb091f817c07032a021ba41ca783cca4c2dd80d570b",
}


def install_template(raw: bytes, destination: Path) -> None:
    if len(raw) > MAX_ARCHIVE_BYTES or hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("official ICASSP 2027 template archive digest mismatch")
    selected = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name, expected in STYLE_SHA256.items():
            matches = [entry for entry in archive.infolist() if entry.filename == name]
            if len(matches) != 1 or matches[0].file_size > 1024 * 1024:
                raise ValueError("official style member is absent, repeated, or oversized")
            content = archive.read(matches[0])
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("official style member digest mismatch")
            selected[name] = content
    destination.mkdir(parents=True, exist_ok=True)
    # Validate the complete target set before writing; preserve differing local files.
    for name, content in selected.items():
        path = destination / name
        if path.is_symlink() or (path.exists() and path.read_bytes() != content):
            raise ValueError("refusing to replace an existing different style file")
    for name, content in selected.items():
        path = destination / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="optional offline official template archive")
    parser.add_argument(
        "--destination", type=Path,
        default=Path(__file__).resolve().parents[1] / "paper" / ".venue",
    )
    args = parser.parse_args(argv)
    if args.archive:
        if args.archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("template archive exceeds the byte bound")
        raw = args.archive.read_bytes()
    else:
        request = Request(TEMPLATE_URL, headers={"User-Agent": "PhaseSet-paper-build/1"})
        with urlopen(request, timeout=60) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != "cmsworkshops.com":
                raise ValueError("template download left the official HTTPS host")
            raw = response.read(MAX_ARCHIVE_BYTES + 1)
    install_template(raw, args.destination)
    print("ICASSP2027_TEMPLATE_VERIFIED " + ARCHIVE_SHA256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
