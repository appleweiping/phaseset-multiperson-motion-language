# PhaseSet: Permutation-Invariant Periodic Relation Tokens for Multi-Person Motion–Language Retrieval

PhaseSet is a data-free research implementation of permutation-invariant
periodic relation tokens for multi-person motion–language retrieval. It is the
successor to the dyadic **PhasePair** prototype, whose audited source and
annotated `v0.1.0` tag remain preserved for reproducibility.

```text
multi-person skeleton sequence
  -> shared actor motion encoder
  -> six-band Morlet decomposition
  -> all unordered actor-pair phase relations
  -> actor–relation incidence topology
  -> six group periodic tokens
  -> text encoder and band-semantic projections
  -> group motion–language retrieval representation
```

## What PhaseSet changes

- Accepts dynamically padded groups with any valid `K >= 2`; the public schema
  does not impose a semantic maximum group size.
- Uses shared actor encoders, no actor ordinal embeddings, no actor-specific
  towers, and no actor-0 coordinate reference.
- Streams unordered edges with a default runtime chunk of 256 edges and a
  fixed canonical reduction microblock of 64 edges; it never materializes a
  dense `[K,K,...]` relation tensor. B2 social attention likewise computes
  exact query chunks instead of a complete `K x K` score map.
- Aggregates incident half-edge means, population second moments, coverage,
  and missingness into a permutation-invariant topology residual.
- Hard-gates topology after the final topology MLP. At `K=2`, its value and
  parameter gradients are exact positive zero, and the full model equals the
  new pair-only control bit-for-bit under the frozen runtime.
- Retains the audited six physical Morlet center frequencies, 13D directed
  dyadic descriptor, swap algebra, and all golden tests. The legacy
  `phasepair_core` namespace keeps its immutable 30-Hz tap bytes; PhaseSet uses
  an independently hashed 20-Hz formula bank after resampling. This is a
  compatibility boundary, not a claim that the two kernel banks or full models
  are bitwise identical. See the
  [20-Hz rate contract](docs/PHASESET_20HZ_MORLET_CONTRACT.md) and
  [canonical-byte portability policy](docs/MORLET_PORTABILITY.md).

## PhaseSet-V2 research upgrade — 2026-09-26

The new [V2 amendment](docs/PHASESET_V2_AMENDMENT.md) introduces signed local
phase, time-varying actor–edge incidence, chronological capture readout, and
verified counterfactual coordination supervision. The
[87-stage matrix](configs/phaseset_v2_experiment_matrix.json) supersedes the
affected legacy 33-run plan. New modules are independent research APIs; the
production V2 training-host integration and converged literature reproductions
are not yet qualified. [TMR-Set](docs/TMR_SET_ADAPTATION.md) now implements the
complete original VAE objectives with disclosed full-parent group adaptation;
its pinned frozen language loader is not a substitute for native learning.
[WaMo-Set](docs/WAMO_SET_ADAPTATION.md) implements learnable stationary analysis
and inverse wavelets, both reconstruction branches and original/shuffled-frame
ordering, with explicit full-parent group adaptation and unpublished-detail
choices. Its software qualification is not converged baseline training.
[MIME](docs/MIME_SET_ADAPTATION.md) keeps separate role-aware original dyadic
and shared complete-parent group APIs, full-time bidirectional co-attention,
and a genuinely trainable pinned CLIP text tower. Group XYZ/long-text adapters
are disclosed; equation/software checks are not original-task convergence.
Their [full-negative parent backward seam](docs/PHASESET_V2_LITERATURE_PARENT_TRAINING.md)
replays one complete capture graph at a time without dropping original WaMo
auxiliary objectives or freezing MIME text. Dense-gradient software oracles
are not native optimizer, profile, checkpoint-host or convergence evidence.

[Server mechanism verification](docs/PHASESET_V2_MECHANISM_VERIFICATION.md)
records 117 focused CPU tests and one full-width CUDA forward/backward witness.
There are still **0/87 formal V2 stages** and no native-data retrieval result.
Approved main-data and licensed body-model assets have been obtained privately,
and the handoff fold assignments have been recovered. Whole-parent preprocessing,
run-specific admission, faithful baseline training, resource profiling, and real
human relational verification still require actual execution evidence.
The descriptions and score equation below document the preserved V1 core,
not completed V2 experiments or a submission-ready paper.

## Preserved V1 milestone

This source snapshot declares `v0.2.1`, the portable-verification patch for the
stable data-free multi-person core and execution-contract milestone. It is a
release candidate until an immutable annotated `v0.2.1` tag and its GitHub
release are observed; a package version or branch alone is not publication
evidence. The immutable `v0.2.0` release remains preserved and is not moved.
The patch changes no model or scientific contract. The milestone does not imply
that the private experiment DAG, resource study, or publication renderer has
produced a real result.
In addition to the migration contracts from `v0.2.0-alpha.1`, it contains the
executable systems 00--08, the licensed-format-independent 30-to-20 Hz group
preparation seam, deterministic base/residual optimization with immutable
checkpoint/resume, a differentiable Torch CPU skeleton-gradient oracle, and a
digest-only host-injected production adapter. Completed base qualification,
evaluation, bootstrap, and paper rendering are semantic gates rather than
generic file receipts: they rebuild canonical contents, bind exact validation
or sealed-test evidence, and require external authorization. The content-bound
renderer derives resource, claim, JSON/CSV, and LaTeX-table artifacts from one
frozen aggregate. The production periodic input
boundary remains the explicit NumPy skeleton-to-activity/Morlet descriptor
stream; the CPU oracle is a qualification path and does not imply production
skeleton autograd. Its
authority remains intentionally **data-free**: it contains no real result,
private dataset, model weight, participant identifier, credential, or private
endpoint.

Base qualification accepts only a complete nine-row census. Every row binds an
exact validation R@1 fraction, parameter count, integer latency in nanoseconds,
terminal digest, validation-selected checkpoint digest, validation-manifest
digest, query-census digest, evaluator/selector-code digest, and a unique score-
artifact digest. The production bridge recomputes the canonical qualification
and also requires an externally authenticated nine-row authorization. Each base's
resource latency is the middle value after sorting its three seed latencies;
the winning base's three seed-specific checkpoints freeze both its group
encoder and learned base logit-scale (inverse-temperature) scalar for systems
00--08. Formal consumers must also receive the qualification artifact's
externally trusted SHA-256; recomputing a digest from a caller-supplied object
does not establish trust. A selected checkpoint proves that it was the
validation best when that checkpoint was written. The immutable terminal and
qualification receipts remain necessary to prove it was the final best after
the full run. The residual score is exactly
`detach(exp(clamp(theta_base,0,ln(100))) * cosine(base,text)) +
tanh(lambda_residual) * masked_mean_band(cosine(group_band,text_band))`.

System 01 is a motion-independent capacity control: every valid actor pair
receives all six fixed band-ID tokens, without consulting activity, Morlet
support, or energy. System 03 computes cosine/sine quadrature power for each
endpoint independently, then exposes only their symmetric mean, absolute
difference, and normalized difference. It contains no cross-endpoint product,
phase, or lag, and a relation band is valid only when both endpoint powers
strictly exceed the same registered DCT floor.

System 06 keeps pair tokens unchanged and, independently for every group and
band, applies a deterministic bijection over all valid directed half-edge
slots. It preserves the valid half-edge multiset, endpoint support slots,
degree, and coverage while changing actor--edge incidence. Its routing plan is
only `O(E)` integer metadata and does not retain `O(E*D)` activations.

Formal training installs and continuously checks one numerical policy:
deterministic algorithms with errors (not warnings), highest float32 matmul
precision, disabled TF32/reduced-precision reductions, deterministic cuDNN,
and no cuDNN benchmarking. CUDA additionally requires a deterministic
`CUBLAS_WORKSPACE_CONFIG`. The environment-v2 digest binds hardware, runtime,
build, thread, and numerical-policy inventory, and the host's ambient Torch
flags are restored after the training context exits.

The registered study uses native `K>=3` Embody 3D captures, a strict
participant-disjoint `400/96/76` capture split, three base candidates across
three fixed seeds, and eight periodic residual systems across those seeds.
Test membership and restricted captions stay private; only irreversible
digests and license-safe aggregates may be released. Each of the 27
system-by-seed score tables is ranked independently; logits are never averaged
across seeds. For each capture and comparison, the three paired seed effects
are averaged in fixed seed order, then only captures are bootstrapped. Seeds are
fixed repeated blocks, not resampled units. Public summaries report the
three-seed mean and population standard deviation. H1--H8 share the same
100,000-draw capture index stream, while only H2--H8 enter Holm correction.

See:

- [migration and compatibility boundary](MIGRATION.md)
- [frozen research contract](docs/PHASESET_RESEARCH_CONTRACT.md)
- [experiment matrix](docs/PHASESET_EXPERIMENT_PLAN.md)
- [execution and recovery protocol](docs/EXECUTION_PROTOCOL.md)
- [data preparation and licensing protocol](docs/DATA_PROTOCOL.md)

## Install and verify

```bash
python -m pip install -e ".[test]"
python -m pytest
phaseset preflight
```

Real-data and accelerator commands fail closed unless the required private
manifest, access receipt, environment digest, and frozen predecessor artifacts
are present. The runner never fabricates an external receipt or silently
substitutes synthetic data for a registered experiment.

Auxiliary caption fusion is also fail closed. A frozen backend manifest binds
the exact model revision, backend implementation, inference runtime, decoding
parameters, and response schema. Test captions additionally require a trusted
host gate that verifies and consumes an external grant and returns an admission
bound to the sealed manifest/caption provenance. Three actor-order calls test
only normalized surface stability; same-family semantic review remains
explicitly provisional.

## Public/private boundary

This repository may contain source, synthetic fixtures, public-safe metadata,
paper sources, editable diagrams, aggregate results, and cryptographic
receipts. It must not contain raw or reversible human data, SMPL-X assets,
download URLs, machine-fused sealed-test captions, private endpoints,
credentials, restricted checkpoints, or participant identifiers. Refer to
[SECURITY.md](SECURITY.md) before contributing or publishing artifacts.

PhaseSet is research software and does not currently assert a real-data
performance result. Negative or inconclusive registered outcomes remain valid
deliverables and will not be replaced by unregistered seed or test selection.
