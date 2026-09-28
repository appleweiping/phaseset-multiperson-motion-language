# PhaseSet NO_RESULT technical draft

**Title:** *PhaseSet: Permutation-Invariant Periodic Relation Tokens for
Multi-Person Motion–Language Retrieval*
**State:** `DRAFT / AUTHORITY0 / NO_RESULT / NOT_FOR_SUBMISSION`

This directory contains a compact four-content-page-target manuscript plus a
references-only fifth-page target. It is a protocol-bearing technical draft,
not evidence of a completed experiment or an authorized conference submission.

## Files

- `main.tex` — PhaseSet multi-person method, frozen 572-capture protocol,
  400/96/76 participant-component split, unseen-cardinality `K=3` test,
  33-attempt ledger, H1–H8 family, one sealed test, and 100,000-draw analysis.
- `references.bib` — primary-source-checked bibliography; no placeholder author,
  title, venue, or result metadata.
- `../figures/phaseset_motivation.pdf` (PNG twin available) — data-free
  pair-bag/incidence counterexample.
- `../figures/phaseset_architecture.pdf` (PNG twin available) — data-free
  PhaseSet method diagram.
- `../docs/PHASESET_LITERATURE_REVIEW.md` — 72-page local-PDF audit and
  primary-source review of adjacent multi-person and periodic-motion work.

Every cell in the manuscript result matrix is visibly `HOLD`. Do not replace a
cell until the rights/runtime/qualification/validation-freeze chain permits the
single sealed-test execution and frozen aggregate.

## Template status

The source uses the [official ICASSP 2027 paper kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php),
unmodified `spconf.sty`, its documented 9 pt mode, and `IEEEbib.bst`. The
hash-pinned fetch script obtains only those two style files into ignored
`paper/.venue/`; no example paper or third-party illustration is republished.
The limit is four technical pages plus an optional fifth page restricted to
references, acknowledgments, and ethical-compliance statements. This draft's
fifth page contains references only.

ICASSP 2027 is non-blind. The explicit author/affiliation metadata-pending
fields must be replaced by the real author list before a submission build.
Formatting success is not submission readiness. The official full-paper
deadline is [September 16, 2026](https://2027.ieeeicassp.org/call-for-papers/).

## Local build

From the repository root, with Python, `pdflatex`, and `bibtex` installed:

```text
python scripts/build_paper.py
```

For an offline build, add `--template-archive <official-kit.zip>`. Its exact
SHA-256 is checked; a different archive is not silently accepted. Each build
gets a new ignored directory, four pass logs, and a source/style/PDF digest
receipt. The script needs no `latexmk` or Perl and disables TeX shell escape.

Verified on 2026-09-08 with MiKTeX pdfTeX/BibTeX: five pages, technical content
ends on page 4, page 5 contains only references, no undefined references or
overfull boxes, and all 18 font entries embedded with no Type 3 fallback. Two underfull horizontal
boxes, one underfull vertical box, and a nonfatal `balance` warning remain
reported in the logs. All 57 `HOLD` result cells remain visible. Inspect page
boundaries, fonts, figures, citations, and claims after every content change.
A generated PDF remains a `NO_RESULT / NOT_FOR_SUBMISSION` artifact.

`BUILD_RECEIPT.json` binds the checked source, bibliography, official styles,
and PDF by SHA-256. Both diagrams now have native editable PowerPoint objects,
alongside draw.io/PDF/SVG/PNG exports. The architecture and method text show the
implemented parallel fixed-activity periodic branch and learned group base;
they do not assert that Morlet reads learned base features.

## Non-negotiable language

- The official Embody 3D scale and the project’s 572 eligible captures are
  different quantities.
- All 27 `K=3` captures are test-only; never imply that training saw `K=3`.
- Multi-TPC is cross-domain, M3Act3D/AIOZ-GDANCE are probes, and dyadic corpora
  are backward-transfer checks—not confirmatory primary data.
- No in-house base is an official MIME, WaMo, InterEdit, or other reproduction.
- No “first,” improvement, generalization, topology, or phase-effectiveness
  statement is authorized by this draft.
