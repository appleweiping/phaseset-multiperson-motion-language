# Full-parent TMR-Set adaptation

[The implementation](../src/phaseset_core/tmr_set.py) retains the
[qualified TMR components](TMR_COMPONENTS.md), including the actual six-layer
ACTOR VAE, two reconstructions, four KL terms, sampled-latent alignment and
semantic-negative-filtered contrastive training. It is explicitly a group
adaptation, not the original single-person TMR or a converged replication.

Each actor's body22 coordinates use the existing capture-wide shared origin.
A shared ACTOR encoder processes each accepted window; unordered PMA pools its
actor states. A second ACTOR encoder reads every window summary in chronological
order, with its absolute start time. Real gaps remain gaps: later windows are
not reassigned earlier timestamps. Text uses the original 768D DistilBERT token
features and separate normalized MPNet sentence features, not sentence CLIP.
Group, text and decoder defaults are 256D, FFN 1024, six layers, four heads,
dropout 0.1. The capture hierarchy/PMA are additional disclosed capacity.

The declared optimizer starting point is lr1e-4 from the
[pinned training YAML](https://github.com/Mathux/TMR/blob/6d74688730d15d43b0a755ce2b0e1f2d76138fc1/configs/model/tmr.yaml),
with AdamW from the
[pinned optimizer implementation](https://github.com/Mathux/TMR/blob/6d74688730d15d43b0a755ce2b0e1f2d76138fc1/src/model/temos.py).
That call leaves weight decay at the PyTorch AdamW default, made explicit as
0.01 in this host. `ParentHostConfig.for_literature` exposes these values;
bounded pilot overrides must be declared. The common group-study warmup/cosine
schedule is an adaptation, not the upstream training loop.

[The language loader](../src/phaseset_core/tmr_language.py) checks every
tokenizer/config/checkpoint file against pinned public digests before loading
from private directories, with local-only safetensors and no remote code.
DistilBERT is fixed at `12040accade4e8a0f71eabdb258fecc2e7e948be`; MPNet at
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. Both are Apache-2.0 pretrained
assets. The latter uses the upstream attention-masked mean plus L2 normalization.
Language models stay frozen/eval even when the trainable group model is trained.
Any caption exceeding the actual token bound fails explicitly instead of being
silently truncated. No dataset, model weights, download receipts or private
asset paths are packaged. Asset download alone is not runtime qualification.

Both sampled motion and text latents reconstruct the entire observed group.
The original six-layer decoder is reused over bounded windows, conditioned on
the global latent, absolute window time, and analytic output-slot codes. These
are **prediction slots**, not embeddings of input actor ordinal/identity.
There is no learned finite actor-slot bank or maximum semantic K. Known K sets
the auxiliary output cardinality only; K is not appended to retrieval features.

For each reconstruction, SmoothL1 costs are accumulated over every window and
observed body coordinate. A single minimum-cost slot-to-track assignment is
then solved over the entire parent. This one assignment prevents a decoder from
changing a slot's actor identity at each window to lower its reconstruction
loss. The selected summed loss is divided by the parent's total observed
coordinate count, retaining both reconstruction branches and ignoring gaps.
No first actor is chosen as the target. Matching uses a K-by-K scalar table and
O(K^3) Hungarian solver; no dense K-by-K-by-time feature tensor is allocated.
The matching cost computation is O(K^2 T); these costs must be profiled.

Known captions remain variable in number. Reconstruction/KL/latent terms are
averaged within each parent's captions, then across parents, so caption-rich
parents do not multiply auxiliary weight. The full rectangular contrastive
gallery preserves all known positives; sentence similarity only filters
training negatives against that run's learning captions, not evaluation labels.
Retrieval uses distribution means and the common unfiltered human-only gallery.

Actor commitment bytes canonicalize calculation and tie-breaking order only;
they are never embedded. Shared input encoders and invariant set pooling do
not use actor ordinal. Whole-window encoder and cost checkpointing preserve
dropout/sampling RNG and pass explicit on-device tensors for CUDA discovery.
Capture inputs themselves are not O(1) memory; no such claim is made.

`TMRGroupCapture.from_prepared` includes every accepted window of the supplied
admitted complete capture. A hand-built window tuple is a software seam, not
proof of native provenance/completeness. Runtime caller responsibilities still
include fixed asset hashes, correct body/text provenance, split/budget/seed
admission, full-parent optimizer/checkpoint/resume host, and actual hardware
qualification. GPU/BF16, native learning, convergence, and fair bounded pilot
evidence are not granted by component tests. Temporal decoder conditioning and
whole-track assignment are adaptations, not a claim of upstream bitwise identity.
