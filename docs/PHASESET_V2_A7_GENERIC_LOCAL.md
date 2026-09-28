# A7: generic local relation control

A7 is the registered `V2-043`–`V2-045` residual control. It keeps the same
frozen B2 global anchor, adapted whole-sentence text vectors, complete-capture
timeline, actor-incidence topology, directed local-packet matching, calibrated
score and weak counterfactual training objective as PhaseSet-V2. The only
mechanism change is the input to each directed half-edge: shared per-actor
signed-velocity patch mean/RMS, the endpoint actor histories and their relative
pelvis position. There is no six-band relation descriptor, Morlet response,
phase mask or energy-floor input to the A7 neural path. A valid local packet
requires both actor tracks, not physical-band support.

Each edge is still generated in canonical 64-edge blocks; the half-edge input
width is `2 * width + 3`. At width 512, the A7 encoder has 21,504 fewer
active parameters than the full physical encoder, approximately 0.13% of the
full encoder. No dummy trainable parameters are added. A7 cannot be combined
with topology-off, A2/A3/A4/A5, DCT or speed-only controls. Checkpoint extra
state records `generic_local=true`, so a full-model checkpoint cannot resume
as A7 or vice versa. The matrix-bound host retains the same nonzero CF weight
and training-text eligibility; unlike A8, it does not suppress CF loss.

This is a *phase-independent score* control, not yet a phase-free end-to-end
data pipeline. The current common `ContinuousTrainingInput` transport loads
and verifies a shared physical cache for all residual rows and recomputes it
after eligible yaw augmentation. A7 ignores its physical response arrays when
scoring, but cache generation, storage and yaw recomputation still incur that
cost. Any later runtime/FLOPs table must report (i) active A7 neural work and
(ii) the shared phase-cache frontend separately, or qualify a genuinely
phase-free input route before making an end-to-end cost claim. A7 must not be
credited with a lower data-preparation cost merely because its score is
independent of phase.

Construction and passing software tests are not training results. The public
factory has `launch_authority=false`; pilot, formal runs and sealed-test
evaluation require independent private admissions and terminal receipts.
