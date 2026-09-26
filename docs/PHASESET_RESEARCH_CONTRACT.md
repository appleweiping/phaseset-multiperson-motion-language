# PhaseSet frozen research contract

> Historical V1 contract. Signed local phase, temporal incidence, calibrated
> scoring, counterfactual supervision, and the formal census are amended by
> [PhaseSet-V2](PHASESET_V2_AMENDMENT.md). The original contract and code remain
> preserved; V2 has no real-data result or scientific acceptance yet.

**Contract ID:** `phaseset-multiperson-20260825`
**Frozen at:** 2026-08-25 20:43:38 +08:00
**Target:** ICASSP 2027, Multimedia Signal Processing primary topic
**State:** `FROZEN / AUTHORITY0 / NO_REAL_DATA_RESULT / NO_CLAIM`

This document is the public-safe scientific contract for PhaseSet. It
supersedes the PhasePair dyadic experiment contract for all claims about
groups of three or more people. PhasePair `v0.1.0` remains an immutable,
data-free compatibility release. This contract authorizes implementation and
synthetic verification; it does not itself grant dataset access, GPU execution
authority, an empirical result, or a conference claim.

## 1. Research question and registered architecture

The registered question is whether motion-derived, frequency-specific
actor–relation incidence structure improves multi-person motion–language
retrieval beyond a strong permutation-invariant group encoder and beyond
equally sized non-topological periodic controls.

The architecture is fixed as:

```text
multi-person 22-joint skeleton sequence
  -> shared actor motion encoder
  -> fixed six-band Morlet decomposition
  -> all unordered actor-pair directed phase relations
  -> actor–relation half-edge incidence moments
  -> six group periodic tokens
  -> frozen CLIP text encoder and shared band-semantic projection
  -> group motion–language retrieval score
```

**Rate clarification frozen before data execution.** PhasePair v0.1.0 keeps its
immutable 30-Hz kernel bytes and oracle. Because PhaseSet consumes 20-fps
activity, it derives an independent zero-DC/unit-energy three-cycle bank at
20 Hz while retaining the same six physical center frequencies. Its schema,
digest, compatibility boundary, and known-frequency/delay probes are frozen in
[`PHASESET_20HZ_MORLET_CONTRACT.md`](PHASESET_20HZ_MORLET_CONTRACT.md). PhaseSet
does not claim that its kernel bytes equal PhasePair's.

The group API accepts every sample with `K>=2` through dynamic batch padding
and has no semantic maximum K. Pair processing is `O(K^2)` and streamed; no
dense `[B,K,K,...]` neural or bookkeeping tensor is permitted. If an exact
all-edge computation does not fit the registered resource budget, execution
returns `RESOURCE_LIMIT`. It must never drop actors, sample edges, or silently
replace the graph with a nearest-neighbor approximation.

## 2. Invariants that precede all empirical claims

The following are hard software gates:

1. Actor permutation leaves group tokens, retrieval embeddings, and scores
   invariant under the frozen runtime. Input gradients are equivariant.
2. Invalid actors, joints, and frames are exact positive zero; non-contiguous
   valid actors and `K_pad=K`, `K+5`, and `2K` produce identical outputs.
3. At `K=2`, the topology residual has positive-zero floating-point bits and
   every topology parameter receives a zero gradient. Full PhaseSet and the
   new pair-only control are bitwise identical under the frozen runtime.
4. The six PhasePair Morlet bands, directed 13D descriptor, endpoint-swap
   algebra, and `v0.1.0` golden tests do not regress. This does not imply whole
   model or old-checkpoint equivalence.
5. Edge order is the lexicographic order of private actor commitments. A
   64-edge canonical microblock uses float64 fixed-tree reduction; microblocks
   use compensated accumulation; incident moments use fixed-order Chan merge.
   Commitments are lineage only and never enter a neural tensor.
6. Runtime chunks are positive multiples of 64. Results for 64, 128, 256, 320,
   and the exact full edge set are bitwise identical on one frozen runtime.
7. Profiling must demonstrate that no allocation has a pair of actor axes and
   that training does not retain `O(K^2 D)` edge activations.

## 3. Fixed model definition

For every band and unordered pair `{i,j}`, shared half-edge and pair networks
compute

```text
u_ij = HalfEdgeMLP(d(i->j))
u_ji = HalfEdgeMLP(d(j->i))
p_ij = PairMLP(u_ij + u_ji, abs(u_ij - u_ji), u_ij * u_ji)
```

Ordered concatenation of `u_ij` and `u_ji` is prohibited. The pair component is
the canonical masked mean of all valid `p_ij`.

For actor `i`, its incident half-edges produce degree, masked mean, population
second central moment `M2`, coverage `degree/(K-1)`, missingness
`1-coverage`, and mean offset from the band-global half-edge mean. The topology
node input is exactly `[delta_mean, M2, coverage, missing]`; raw degree and
`log(K)` are prohibited. A node is valid only when the actor is valid and its
degree is at least two. The final NodeMLP output is multiplied by this mask;
no biased layer follows that hard gate.

```text
group_token_b = band_mask_b * CommonPostprocess(
    pair_component_b + tanh(lambda_b) * topology_delta_b
)
```

The pair-only control passes an exact positive-zero topology residual through
the same CommonPostprocess and shares the same head capacity.

The frozen base retrieval score is augmented by a bounded learned scale times
the masked mean cosine similarity between six motion tokens and six text-band
tokens. Text bands are generated by a shared `TextBandMLP(text_embedding,
band_id)`. Text cannot generate, edit, select, or supervise physical phase,
lag, power, or coherence fields. Training uses variable-positive symmetric
InfoNCE and never assumes a fixed caption count.

## 4. Data, preprocessing, and leakage boundary

The confirmatory dataset is native multi-person Embody 3D. Access requires a
real applicant to complete the official release process. Raw data, SMPL-X
assets, download URLs, participant identifiers, machine-fused sealed-test
captions, and reversible memberships remain outside this repository.

The registered public-safe census contains 572 eligible native `K>=3`
captures, 69 participants, and 20.673 scene hours: 27 captures have `K=3` and
545 have `K=4`. The participant co-occurrence graph has 16 disconnected
components. The split is fixed at 400 train, 96 validation, and 76 test
captures, with 48/10/11 participants respectively and no participant or
capture overlap. All 27 `K=3` captures are in test, so K=3 is explicitly an
unseen-participant and unseen-cardinality evaluation; the paper may not imply
that training observed K=3.

Motion preprocessing uses 22 SMPL-X body joints, antialiased 30-to-20 fps
resampling, and 200-frame windows. The first valid frame's all-group pelvis
centroid is the common origin. It is forbidden to center or orient around an
indexed actor. Training yaw augmentation rotates the whole group jointly.
Missing spans no longer than 0.25 s may be interpolated while retaining their
original mask. A window is rejected when any actor is below 95% valid or has a
longer contiguous gap. Coherence remains a numerical input and is not used as
an edge-deletion threshold.

The primary task uses only human holistic capture descriptions at test time.
All non-overlapping 10 s windows in a capture are motion-encoded and aggregated
by a fixed masked mean. Missing holistic captions are removed before any model
score is computed and do not trigger split reassignment.

The auxiliary 10 s task may machine-fuse the K human actor descriptions and
available holistic context into one group caption plus one paraphrase intended
to preserve meaning. That request is not proof of semantic equivalence. The
frozen semantic model may not read skeletons, spectra, phase, model outputs, or
split statistics. Prompts, model identity, digests, retries, and actor-order
perturbation audits are retained privately. Such text is always labelled
machine-fused and never described as human group annotation. A same-model-
family semantic review has status
`PROVISIONAL_SAME_MODEL_FAMILY_REVIEW_NOT_GATE_CLOSING`; it cannot close any
data, execution, sealed-test, or scientific-claim gate.

InterHuman/Inter-X are K=2 backward-transfer datasets only. Multi-TPC is a
separate cross-domain task. M3Act3D and AIOZ-GDANCE are engineering/probe data,
not language-benchmark evidence. No synthetic concatenation of dyadic samples
may appear as a real multi-person primary result.

## 5. Registered selection and hypotheses

Three 512-dimensional, 8-head, 2048-FFN, dropout-0.1 base encoders compete:
ActorMean, SetPMA, and SocialTemporal. Each is trained for all three fixed
seeds. The winner maximizes validation bidirectional R@1 mean across the three
seeds; exact ties choose fewer parameters, then lower latency, then smaller
system ID. Each winning seed checkpoint is frozen and shared by every residual
system for that seed. The nine-row qualification retains the original score,
terminal, checkpoint, validation-manifest, query-census, evaluator, and unique
score-artifact evidence. Production recomputes it and requires an independently
authenticated authorization over the exact nine-row census; it must also match
an externally frozen qualification digest.
Checkpoint-local metadata establishes validation-best-at-write only; the
completed terminal and trusted qualification receipt establish final
selection. The complete systems 01--08 capacity audit is independently
digest-pinned and freshly rebuilt before formal residual construction.

The generic-token control exposes fixed band identity for all six bands on
every valid actor pair and has no motion/Morlet/energy path. The marginal
Morlet control pools each actor/band once from only that actor's response and
observation mask, then combines two cached self-marginals per pair. Its support
is length-only: partner-mask intersection, energy gates, phase, coherence, and
cross-person products are prohibited. The DCT control computes each endpoint's
self-power independently and exposes only symmetric mean and difference
statistics; it has no cross-endpoint product, phase, or lag, and requires both
endpoint powers to pass the same per-band floor. The coverage/missing-only
control derives pair/band support solely from observation-window overlap and
band length. With observation masks fixed, motion values, amplitude, energy,
phase, and coherence must leave its support, counts, and output tokens bitwise
unchanged; an unsupported band is exact positive zero after the hard band gate.

H1, the sole primary hypothesis, is `PhaseSet full > qualified base` on the
registered primary metric. H2–H8 compare full against generic tokens, marginal
Morlet power, mean/difference DCT, pair-bag, coverage/missing-only,
incidence-shuffled, and phase-stripped controls respectively. Statistical
units are captures or participant components, never captions or windows. Each
system and training seed is ranked independently; logits are never averaged
across seeds. For a capture, the three paired seed effects are averaged in
fixed order and only then receive 100,000 capture-cluster bootstrap draws.
Seeds are not resampled or treated as independent observations. H1--H8 use one
shared resampling stream; H2–H8 alone use Holm correction. The frozen aggregate
binds all 27 checkpoint/score/evaluation identities and reports the three-seed
mean plus population standard deviation.

The claim ladder is fail-closed:

- topology understanding requires stable superiority to both pair-bag and
  incidence-shuffled controls;
- phase effectiveness requires superiority to phase-stripped full;
- structural contribution requires superiority to generic six tokens;
- otherwise only the strongest supported weaker claim, including a null or
  negative result, is reportable.

No test score may select architecture, checkpoint, seed, hyperparameter,
caption policy, or gallery. After validation freezes, sealed test evaluation is
run once. A failed hypothesis remains in the final paper and release.

## 6. Execution authority and immutable evidence

Every attempt binds code, data-manifest, configuration, environment, RNG,
predecessor, and checkpoint digests. It owns an immutable manifest, lock,
heartbeat, checkpoint chain, metrics stream, terminal record, and failure
classification. Resume must validate the entire predecessor chain. Duplicate
or contradictory terminal records, truncated checkpoints, environment drift,
manifest drift, and disk exhaustion fail closed.

Real GPU admission requires all of the following: dataset-rights receipt,
private split/caption digests, deterministic core tests, Linux/CUDA Morlet and
CLIP qualification, BF16 or FP32 precision qualification, synthetic
forward/backward/checkpoint/resume, and a disposable 64-capture overfit. Until
these exist, the scientific authority remains zero.

Formal execution enables deterministic algorithms in error mode, uses highest
float32 matmul precision, disables TF32 and reduced-precision reductions,
enables deterministic cuDNN with benchmarking off, and requires a deterministic
cuBLAS workspace setting on CUDA. Hardware, runtime, build, thread counts, and
this numerical policy are bound in the environment digest and checked for live
drift.

The registered server is operational evidence only. Connection must target
the registered non-default port, verify the host key, and begin with a
read-only inventory. A refused connection is not a training result and may not
be replaced by another port or host without a versioned contract amendment.

## 7. Publication boundary

Paper tables, CSV/JSON, figures, and slides are rendered from one frozen
aggregate. The ICASSP manuscript has at most four content pages and, if used, a
fifth references-only page. Public releases undergo staged-blob, all-ref,
wheel, anonymous-codeload, release-asset, secret, private-path, restricted-data,
and rights scans. Git tags are immutable; no force-push or historical tag move
is permitted.

This contract can be superseded only by a timestamped amendment written before
the affected test observation. Amendments cannot retroactively authorize a
result, conceal a failure, or alter PhasePair `v0.1.0`.
