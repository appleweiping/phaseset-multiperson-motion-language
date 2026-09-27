# MIME: original dyadic and complete-parent group implementations

Independent implementation of [MIME v1](https://arxiv.org/html/2607.22702v1),
not released author code or a claim of converged original-task reproduction.
The original dyadic API and the group adapter are distinct; neither converts
XYZ into fictitious SMPL rotations or silently truncates a capture.

`OriginalMIMEMotion` takes true 135D root-displacement/22-joint 6D-rotation
features plus common-frame root positions. It retains distinct role projections,
normalizations and type embeddings, root distance/signed displacement,
relative configuration and root-motion magnitude ratio, early relation injection,
four layers of pre-LN independent self-attention, simultaneous bidirectional
full-time cross-attention and GELU FFNs, ordered frame fusion and learned-query
temporal pooling. The original Inter-X host must enforce its 30Hz/300-frame
protocol separately and supply real rotation inputs; body22 is not a substitute.

`MIMESetMotion` explicitly adapts these interaction mechanisms to shared body22
XYZ inputs. All actor weights and endpoint block weights are shared; input-role
embeddings are absent. Each endpoint receives its own directed relation encoding.
Sum/absolute-difference/product fusion makes the pair readout symmetric.
Every unordered pair is visited, checkpointed independently, and accumulated
in canonical commitment order as float64 before the window mean. No actor
ordinal or commitment is a neural feature; no dense K-by-K-by-time activation
is constructed and no edge is silently sampled. All accepted windows enter an
extra four-layer temporal hierarchy with actual absolute starts, including gaps,
and learned-query full-parent pooling. The per-window hierarchy and 20Hz XYZ
feature adapter differ from the original single 30Hz dyadic sequence.

Missing pelvis gates root-relative features, rather than inventing zero distance.
Pair pooling uses jointly observed pelvis frames. Per-actor self-attention sees
valid observed frames; root displacement does not bridge a missing frame and is
zero at each window start. These mask/boundary conventions are disclosed choices.

Both use *fine-tuned* local CLIP text, a learned projection and positive learned
logit scale, then bidirectional-average variable-known-positive InfoNCE. This
is not the immutable frozen CLIP feature adapter; that adapter remains unchanged.
The existing official asset revision/hashes are verified before local-only loading.
No vision module is instantiated or optimized. The original dyadic text API
rejects captions exceeding 77 tokens. The complete-parent group adaptation
preserves every content BPE token in consecutive 75-content-token segments,
adds BOS/EOS to each, and averages trainable pooled segment features per caption.
This long-text adaptation is disclosed, not purported original MIME behavior.

Paper settings are width 512, four layers, four heads, dropout 0.1, AdamW 1e-4,
weight decay 1e-4 and batch 128; curriculum is the separately implemented
[training-only parent sampler](PHASESET_V2_MIME_CURRICULUM.md). FFN 2048,
two-layer GELU frame fusion, ratio epsilon 1e-8 and initial logit scale 1/0.07
are disclosed implementation choices where author code is not available.
The existing three MIME short-pilot slots govern eligible choices, not added HPO.

Tests include original-input dimensions, independent full-time attention
equations, all parameter families, endpoint swap, all K=3 permutations and
input-gradient equivariance, dropout/checkpoint/RNG equality, missing root steps,
full timeline and complete text segmentation. Tiny language seams are labeled
software fixtures; only a real pinned CLIP/native attempt proves that execution.
When Transformers is installed, a random small actual CLIP component also
checks the extracted-transformer API and checkpoint/direct gradient/RNG equality;
its outer model constructor initializes the attention backend, as in asset loading.
That component is not a pretrained-language or real-data qualification.
Neither software nor native backward qualification proves original-task learning,
production-host readiness, retrieval performance, scientific claims or release.
