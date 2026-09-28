# PhaseSet-V2 A4 incidence-shuffle control

A4 is the registered `shuffle_explicit_incidence` control in the frozen V2
matrix. It tests whether the assignment of physical pair relations to actors
contributes beyond the multiset of pair packets. It is not a claim that every
form of actor topology has been erased.

The control uses the same trainable `ContinuousRetrievalSystem` and
`TemporalIncidenceEncoder` as the full phase system. For each capture, local
patch and canonical 64-edge block, it permutes only supported half-edge values
entering `_node_block`. The endpoint slots, support mask, actor degree,
coverage, physical pair-token stream, directed text packets and all parameters
remain unchanged. A fixed nonnegative `incidence_shuffle_seed` determines the
reassignment; private actor/group commitments do not enter its hash or neural
features. This makes the ablation independent of participant identity and
keeps the same data and capacity as the full arm. The seed is part of
checkpoint extra state and mismatched checkpoint loads fail closed.

The assignment is deterministic under canonical actor/edge order. Invalid
half-edge positions are untouched; supported values are bijectively gathered,
so their gradients remain connected. The control is phase-only and cannot be
combined with the separately calibrated A6 DCT comparator. No training or
paper result follows from implementation or software tests alone; formal
A4 runs still require the frozen pilot, server GPU admission and their own
immutable run receipts.
