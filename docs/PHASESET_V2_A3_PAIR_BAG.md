# PhaseSet-V2 A3 capacity-matched pair-bag control

A3 is the registered `pair_bag` mechanism control. It uses the same physical
phase field, frozen B2 anchor, text adapter, training captions, objective,
complete-capture timeline and parameterized head as V2 full. It removes the
explicit assignment of relation half-edges to actor nodes. It does **not**
discard relations, phase, temporal order or the shared incident-node and
edge-update modules.

The pair-bag streams all valid unordered pairs in 64-edge blocks. Each block
contributes the sum, square sum and count of both directed half-edge packets
to global per-patch moments. A symmetric masked mean of actor histories and
those global packet moments form one context vector per patch, broadcast to
identical node streams. The existing node MLP, two-layer node GRU,
node-conditioned edge update, two-layer edge GRU and ordered group GRU remain
active. The same pair packets then enter the normal unordered readout and
directed text score; only the explicit actor–edge incidence map is absent.
No dense `K × K` adjacency is built.

Consequently, rerouting an unchanged multiset of pair packets between actor
endpoint slots leaves A3 unchanged while it can change V2 full. A3 and full
have exactly the same trainable parameter count. `use_topology=False` remains
the distinct legacy pair-only/K=2 diagnostic; it is **not** the registered A3
because it leaves node and edge-update capacity inactive. A3 is mutually
exclusive with A2/A4/A5/A6. Its identity is bound in tensor-only checkpoints.

With exactly two valid actors, the context gate remains off, the topology
values and gradients are positive zero, and A3 is bitwise equal to V2 full on
the same weights. This implementation and its tests are software qualification,
not a claim of retrieval superiority. The registered three-seed, human-only
comparison and parent-cluster statistics are still required.
