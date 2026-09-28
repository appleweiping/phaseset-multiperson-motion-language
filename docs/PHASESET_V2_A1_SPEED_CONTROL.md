# A1: speed-only periodic-input control

A1 is the matrix-registered speed-only ablation, not a replacement for the
full PhaseSet-V2 model or a retrieval result. The common B2 checkpoint,
captions, weak counterfactual branch, text adapter, six Morlet filters,
relation encoder, scoring module, run seed and trainable capacity are retained.
Only the periodic frontend changes. For each observed pelvis or pelvis-relative
joint velocity vector `v`, A1 broadcasts `||v||₂ / sqrt(3)` into its three XYZ
channels *before* convolution. This removes sign and direction while
preserving each vector's squared energy. Taking the magnitude of a complex
Morlet response would be a different control; removing its phase fields is A5.

The full model's geometric root branch and frozen B2 global branch still see
the original group geometry. Therefore full-versus-A1 can only support a claim
about signed direction in the **periodic input**, not an assertion that the
entire A1 system is spatially direction-blind.

Speed responses have their own cache schema, and cache loading declares the
expected frontend. Main, pilot and each external fold need separate floors
fitted from their complete training actor/patch/band population. The fit
rejects mixed frontends or physical configurations. A typed speed-floor
receipt binds population, training-source manifest, actual fit config and six
floor values; source loading, model scoring and checkpoint extra-state reject
identity drift. Signed-vector floors or caches are not interchangeable.

The software constructor does not authorize a formal run. It does not fit
real-data floors, prove storage/GPU capacity, run a pilot, touch sealed test,
or establish an ablation effect. Those steps require private admission and
immutable server receipts. The official human-only evaluation gallery is
unchanged by the user's [relation-supervision amendment](PHASESET_V2_RELATION_SUPERVISION_AMENDMENT.md).
