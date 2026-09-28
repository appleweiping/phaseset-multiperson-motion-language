# PhaseSet-V2 mechanism verification — 2026-09-26

This is implementation evidence, not a retrieval benchmark, convergence claim,
or submission-ready paper. The scientific upgrade is described in
[the V2 amendment](PHASESET_V2_AMENDMENT.md); its
[87-stage matrix](../configs/phaseset_v2_experiment_matrix.json) supersedes the
affected legacy 33-run plan.

## What was implemented

- Signed world-coordinate pelvis velocity and pelvis-relative joint velocity;
  six unchanged 20-Hz Morlet kernels; local complex cross-response, endpoint
  energies, coherence, support, and explicit phase observability.
- Shared actor and edge temporal encoders with time-varying incidence moments,
  fixed 64-edge checkpoint/recompute blocks, and an ordered capture readout.
- A common-scale cosine mixture, complete endpoint-bound relation matching,
  variable-positive symmetric InfoNCE, and verified-negative-only softplus loss.
- A true mean/difference signal DCT control, preserving its cross terms.
- A portable structural test journal using the real state machine. Production
  POSIX journal durability and runtime admission were not relaxed.

These are independent research APIs, not yet a formal V2 host execution path.
No TMR/WaMo/MIME reproduction, V2 text annotation, training-only energy-floor
fit, or native-data training is claimed here.

## Observed server checks

The existing isolated environment was reused: Python 3.12.12, NumPy 2.4.6,
Torch 2.12.0+cu126, CUDA 12.6. Numerical checks ran on the registered experiment
server, not on the local workstation.

| Check | Observed result |
|---|---|
| Focused V2, Morlet, legacy signal, host controller, and journal regressions | 117 passed, 1 CUDA-only test skipped; 32.47 s |
| Width-512 CUDA forward/backward, K=3 | 1 passed; 7.43 s |
| CUDA device | NVIDIA RTX A6000 |
| Peak allocated / reserved memory in that witness | 143,108,608 / 165,675,008 bytes |
| Source changes during either attempt | none; exact digest comparison passed |

Mechanism tests include speed-only in-phase/antiphase collision versus signed
phase separation; endpoint conjugation; observed stationary support with
unobservable phase; no fabricated missing-joint velocity; non-incident edge
stability; actor permutations and padding; exact K=2 positive-zero topology
outputs and gradients; pair-only equivalence; checkpoint/eager forward and
parameter gradients at K=3 and K=13 (78 edges, crossing the 64-edge boundary);
chronological capture order; calibration; and verified-counterfactual exclusion.

The width-512 GPU witness is **not** an arbitrary-K stress study, a complete
end-to-end training/resume run, or a resource comparison with literature methods.
The same-family GPT-5.6-Sol code review remains provisional scientific evidence.
Both execution receipts and raw logs stay private; their closed receipt digests
are:

- CPU: `e1a9a5efa7089c0a5ae8263a712492ba57032dc6b2d440794ee25d73dbefc42f`
- CUDA: `2952fdbe06d81afb23ae89389a52bbe747c1cb37ca58d11cb2580fd7dc895e1c`

Windows CI is required for the portable structural journal fix. A Linux server
check alone is not evidence of Windows execution; use the commit-bound CI result.

## Scientific execution state

Formal V2 stages completed: **0 / 87**. Sealed test remains closed.

The project-owned data directory was inspected read-only on this date: only
the supplemental Multi-TPC data was observed. Licensed Embody main motion data,
neutral SMPL-X model assets, and a corresponding access approval were not found
in the checked project locations. This is not a claim that approval could not
exist elsewhere.

The referenced chat's ZIP/Markdown handoff was not attached to the accessible
conversation. Exact pilot and three-fold component assignments remain null;
they must be recovered, not invented. Two blinded human reviewers and the real
relational challenge are still outstanding. No synthetic result is promoted
to main-data evidence, and no auxiliary dataset silently replaces Embody.

Current execution state is **BLOCKED_DATA**, with rights, handoff, and human
relational-evidence dependencies recorded separately. Once those inputs exist,
the next step is the bounded native-data audit/pilot, then the frozen formal
matrix—not additional data-free audit scaffolding.
