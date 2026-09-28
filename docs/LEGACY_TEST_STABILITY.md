# Legacy CLIP test determinism

A complete server run during capture-training integration reported 1000 passed
tests and two failures in the existing legacy CLIP-resolution tests. That
failed attempt is retained. The production scanner, lease implementation,
canonical model pins, and old immutable release are unchanged.

The late equal-length file-mutation test specifically exercises the closing
metadata sweep after the changed file has already been hashed in pass two.
It previously assumed that a very fast same-length write necessarily changes
the filesystem's observed timestamps. The fixture now explicitly advances its
owned temporary file's mtime and verifies that change. The real byte mutation,
16-open expectation, and required `HOLD_FILE_TOCTOU` result are unchanged. This
does not turn the production scanner's documented best-effort observations
into an atomic snapshot guarantee.

The weak-lease registry test now collects unreachable earlier test cycles
before measuring its baseline, just as it already collects them after removing
the 256 test assessments. Otherwise automatic collection during those 256
allocations can remove a pre-existing unreachable lease and invalidate the
relative-count assertion. All retention and release checks remain in place.

The corrected focused server integration passed 74 tests and 6 subtests.
The subsequent complete suite passed 1002 tests, 2 skipped, and 39 subtests
in 297.31 seconds; source bytes were unchanged across execution.
These are test-fixture changes, not relaxed model tolerances or weaker data
admission checks. Review was local-only under ARIS's unavailable-delegation
fallback; actual server outcomes remain the deterministic evidence.

## Canonical capture-index fixture newlines

The later complete-capture host integration passed its Linux tests but exposed
18 failures on each Windows CI runtime. Its test-only JSON writer used the
platform default newline translation: canonical text ending in LF became CRLF
on Windows. The production byte-equality check correctly rejected the fixture
before the intended dispatch or overlap assertion. This is distinct from the
earlier storage-stat issue, whose tests passed in that same CI run.

The fixture now supplies an explicit LF newline policy. Production code and
canonical-byte acceptance are unchanged. Two additional authenticated cases
insert actual CRLF at the outer and referenced training-index seams; both must
still be rejected even with recomputed matching digests. The corrected focused
Linux server suite passed 71 tests in 11.08 seconds with source bytes unchanged.
The failed Windows logs are retained privately. Acceptance on Windows requires
the subsequent actual CI result, not inference from this Linux observation.
