# PhaseSet paper figures

These are data-free conceptual/method figures, not empirical results.

| Figure | Editable source | Publication exports | Native editable slide |
|---|---|---|---|
| Pair-bag incidence counterexample | `phaseset_motivation.drawio` | PDF, SVG, PNG | `phaseset_motivation.pptx` |
| PhaseSet architecture | `phaseset_architecture.drawio` | PDF, SVG, PNG | `phaseset_architecture.pptx` |

The draw.io files are the shared semantic/layout source. The PPTX files now
contain native PowerPoint shapes, text runs, and connector segments, with **no
slide-wide raster image**. Individual labels and nodes are editable. Regenerate
all formats from draw.io for a consistent publication update; a PowerPoint-only
edit does not update the source or other exports automatically.

The motivation figure contains 18 native shapes/text boxes and 18 connector
segments; the architecture contains 31 and 41. `EDITABLE_FIGURES.json` binds each
source and PPTX by SHA-256 and records the native-object census. The renderer
uses an allowlisted, uncompressed ZIP package with fixed metadata and records
the renderer, document runtime, and default template hashes. Repeat rendering
in that recorded runtime produced byte-identical PPTX files. Its fixed creation date is
not an assertion about when the research or figure was created.

## Rebuild

Canonical native-PPTX rebuilding requires the recorded Windows AMD64 CPython
3.12.14 document runtime, python-pptx 1.0.2, lxml 6.1.1.0 / libxml2 2.11.9 /
libxslt 1.1.45, and zlib 1.3.2. The renderer rejects other tuples and altered
source filenames/bytes rather than issuing a canonical-build receipt. Opening
or editing the provided PPTX does not require that Python runtime. The portable
artifact tests verify the delivered hashes and native objects without it.

Within the registered document runtime, run the closed project renderer:

```bash
python -m pip install 'python-pptx==1.0.2'
python scripts/render_editable_figures.py --figure-dir figures --output-dir figure-build
```

This renderer supports only the limited shapes/styles used in these two pages;
it is not a general draw.io importer. Draw.io remains the native publication
exporter. With draw.io 31.1.8, export PDF with `--export --format pdf --embed-diagram
--crop`, and SVG/PNG with `--export --format svg` or `png` plus `--embed-diagram`.
Use explicit output files in a new build directory. Check the PNG IEND trailer
after export; the known truncated-IEND exporter defect must be repaired before
strict PNG validation. Embedded diagram metadata is scanned with the rendered
content before public release.

On 2026-09-08 both final native slide files were opened in Microsoft PowerPoint
without a repair prompt and visually inspected; their hashes were unchanged
after inspection. Further renderer gate hardening reproduced the same PPTX
bytes. Both packages pass exact object-identity, relationship, XML, and ZIP
checks and reopen in python-pptx. Unused thumbnail/printer parts were removed.
This is not a claim about arbitrary future input.
PDF exports use embedded Arial TrueType fonts;
Unicode math subscripts that caused Type 3 fallback have been replaced by
native subscript text runs. The `*_preview.png` files are display conveniences,
not independent figure sources.

## Scientific correspondence

The motivation figure constructs two four-actor complete graphs with identical
actor marginals and the same six-token pair multiset, but different actor-edge
incidence. Pair-bag averaging is identical; incident moments can distinguish
the constructed structures. It is not an observed motion sample.

The architecture depicts the current implementation: fixed mask-aware body
activity feeds the physical Morlet branch, in parallel with the qualified
learned B0/B1/B2 group base. The learned base embedding does not feed Morlet.
For residual training the selected seed-specific base is frozen. Both its
global text score and the six-band text alignment contribute to retrieval.
This drawing correction does not change the model or claim encoder-to-Morlet
end-to-end gradients. Empirical annotations and paper values must come from
the same frozen aggregate when actual results exist.
