# PhaseSet-V2 A2 order-free control

A2 is the registered `remove_local_and_capture_order` control. It receives
the same physical six-band phase field, frozen B2 anchor, text embeddings,
losses, complete captures and actor–edge incidence as the full system. It
removes the *learned temporal order* in the actor, incident-node, directed-edge
and group-readout paths. It does not strip phase, subsample frames, shuffle
actors, or change the physical Morlet input; A1 and A5 test other questions.

Each ordered two-layer GRU stack is replaced by a shared pointwise
`width → 6 × width → width` MLP with GELU and a masked mean. The MLP is
independent at every patch and has no state, timestamp, position embedding or
convolution. The final mean is over all supported capture patches. Its
trainable parameter count is `12w² + 7w`, versus `12w² + 12w` for the two
GRU cells; at the fixed `w=512`, this is approximately 0.081% smaller per
replacement and introduces no inactive capacity padding.

All invalid patch outputs are hard-gated to exact zero after the MLP. A2
cannot be combined with the separate A3/A4/A5/A6 controls, and its model
checkpoint records the order-free identity. The same tensor-only checkpoint
identity also distinguishes A3 pair-bag, the legacy topology-off switch and A5
phase-stripped switches from the full model. The full model's previous state
payload remains unchanged.

This is a mechanism control, not evidence that time order helps retrieval.
That conclusion requires the frozen three-seed comparison on the official
human-only task and the registered statistics; software tests alone are not
the paper result.
