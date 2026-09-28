# PhaseSet-V2 matrix-bound mechanism construction

`build_v2_mechanism` reads the exact registered 87-row matrix bytes and one
run ID, an admitted training-source digest, and a validation-selected complete-parent
B2 checkpoint with its exact file digest. It constructs the currently implemented complete-parent residual
systems only: PhaseSet-V2 full and A1/A2/A3/A4/A5/A6/A8. It verifies the row's
fixed seed, split and same-seed B2 predecessor, and refuses A7/A9, the old
head, base and literature rows until their distinct implementations exist.
There is no fallback to the full model for an unsupported system.

| Row | Constructor difference |
|---|---|
| PhaseSet-V2 | full signed-vector phase and incidence model |
| A1 | speed-magnitude input before the same six Morlet convolutions, with an independent typed training-floor receipt and physical cache |
| A2 | order-free temporal MLP/mean paths |
| A3 | capacity-matched pair-bag context |
| A4 | explicit incidence shuffle seeded by the registered training seed |
| A5 | explicit phase fields stripped; energy/support retained |
| A6 | true mean/difference-signal DCT with a typed, main-population floor receipt |
| A8 | full model; `bind_v2_parent_host_config` sets only the CF loss weight to zero |

For main rows, host config binding also checks the exact twelve development
training components, C00 validation, fixed seed and residual stage. A8 is
derived from the same nonzero-CF candidate, changing only its loss weight;
the caller must still present the same admitted weak training text pool as
full. Residual initialization uses a forked CPU RNG seeded by the registered
row, so same-seed full/A8 begin from the same trainable weights without
mutating the caller's RNG. The A1/A6 receipts must match the admitted
training-source manifest digest in addition to their physical config and
main population. A1 also requires a speed-only source and checks its receipt
against the model at host binding; it cannot silently use signed-vector cached
responses. The periodic transform and its claim boundary are specified in
[A1 speed control](PHASESET_V2_A1_SPEED_CONTROL.md).

`make_v2_b2_base_host` constructs a main B2 host that writes the registered
base row identity into its checkpoint manifest. It creates the registered
SocialTemporal B2 architecture under the row's forked CPU seed and requires
the 30-epoch registered schedule; a short pilot cannot carry a formal row ID.
The residual builder verifies that row identity, stage, seed, population,
checkpoint file/state digests, a completed 30-epoch terminal and its final
validation-selected checkpoint digest, then loads strict B2 weights into the
registered architecture. `make_v2_parent_host`
is the single row-bound residual host path: it compares checkpoint/source
bindings, applies A8's loss transformation and persists the residual row,
system and predecessor in every checkpoint manifest. It also requires the
registered 20-epoch residual schedule. A direct host carrying
registered A8 identity rejects a positive CF weight.

This factory does not independently prove that the private nine-run B2
qualification selected the supplied checkpoint, authenticate private assets,
freeze pilot hyperparameters or max steps, claim storage/GPU capacity, or
launch training. That selection and the admitted source digest remain private
admission responsibilities. Fold model identity can be constructed only with
its registered fold-specific B2 training/validation population; its residual
host binding is still held. Inter-X remains unimplemented and fails closed.
The returned `launch_authority` is always false.
The public CLI likewise remains fail-closed until a qualified private
production adapter is attached. The host factories are executable row-bound
construction, but no formal run is admitted by calling them. This is executable mechanism selection,
not a completed 87-stage runner or a retrieval result.
