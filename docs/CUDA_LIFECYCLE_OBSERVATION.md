# Bounded full-width CUDA lifecycle observation

This is engineering evidence for public commit
`e3c73c031c0034701277870ca99fbbcbdd037294`, not a real-data result or completed
formal training qualification. No dataset, external checkpoint or licensed
body-model asset was used. The four lanes ran on an NVIDIA RTX A6000 using
Python3.12.12, NumPy2.4.6, Torch2.12.0+cu126 and CUDA12.6, with FP32,
deterministic backend settings and the frozen disabled-MHA-fastpath policy.

Each lane used the actual registered architecture: width512,8 attention heads,
FFN2048, full200-frame body22 windows, analytic K2/K3 groups, and frozen analytic
text embeddings. The effective batch was2 for this engineering fixture, not
the formal128-window batch. System08 retained its six-band periodic encoder,
incidence topology and text-band head. No reduced-width substitute was used.

The lifecycle was:3 uninterrupted optimizer updates; an independent2-update
run; a third newly constructed runtime restoring its update2 checkpoint and
performing update3. Three updates allow initially zero residual/topology gates
to begin passing nonzero topology gradients. Every lane used fresh work and
checkpoint directories. No epoch was completed and no validation metric or
selected checkpoint was produced.

| Observation | B0 | B1 | B2 | System08 |
|---|---:|---:|---:|---:|
| Parameter elements, including frozen base | 6351361 | 9504257 | 20014593 | 8197639 |
| Peak allocated CUDA bytes | 259429888 | 326669824 | 641133568 | 138454528 |
| Checkpoint files retained | 4 | 4 | 4 | 4 |
| Checkpoint bytes retained | 305078852 | 456479908 | 961194980 | 190464852 |
| Uninterrupted vs resumed decoded payload/model/optimizer/scheduler/RNG/cursor/manifests | exact | exact | exact | exact |

For B0/B1/B2, the group encoder and logit scale had finite nonzero gradients and
changed state. For08, the topology encoder, six topology-gate elements,
text-band MLP and residual score gate had finite nonzero gradients and changed
state. Its frozen base retained exactly the same state with no gradients.
Frozen fixture text bytes and non-gradient state were preserved in every lane.
All source files remained unchanged, and all canonical observation validators
and all local receipt-member checks passed.

Private receipt SHA-256 commitments (operational files and checkpoints are not
public release assets):

- B0: `6517734b854e74ea42982a9055de08256d14fe4876cdd91b38c091a8a92346b0`
- B1: `a0df9f67b5bb47fa957f36b8adc3ea68c412a5dae2a5c950daeaa2decd7f5e8f`
- B2: `fe0914cac54e6ae88f01f104489c44086d3d6f2cde8dd281cc1afe17f0fea671`
- System08: `4815471fc85c3dbbbf8afd2798295fee5327b437a82f6fadfa6cb0c259b70d1d`

B1/B2 were a distinct reviewed extension with explicit PMA and four-layer
social-temporal parameter censuses. Their wrapper revalidated the successful
B0/08 receipts before execution. Both preserved the full200-frame fixture and
the same2GiB cap; no architecture was reduced to fit. All60 local receipt
members across the four lanes were checked after their transfers completed.

Raw checkpoint archive bytes also happened to match for the uninterrupted and
resumed runs in each lane. Only decoded-state equality was required; this does
not promise equal serialization bytes across independent hardware or runtimes.

These figures are not throughput, latency or memory-efficiency benchmarks:
the device was shared, a2GiB allocator cap was imposed, and only this bounded
fixture was observed. BF16, formal initialization/base
qualification, real-data overfit, convergence, native-data scores and empirical
paper readiness remain separate requirements. The synthetic losses are not
reported as retrieval quality.
