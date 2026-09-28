# Complete-parent disk source

`DevelopmentParentDiskSource` consumes caller-admitted `ParentRetrievalTask`
and `ParentDiskRecord` entries. Every family must have exactly one complete
source and an explicit caption-yaw eligibility bit. Unknown or changed human
records fail before loading. The development source rejects final-test rows;
it does not discover datasets or infer rights from matching files.

The literature interface supplies complete body timelines and fixed TMR/WaMo
human-language rows. The body view loads verified prepared arrays without
reading the physical cache. The phase view additionally loads the entire
verified physical cache with the supplied training-only floors. It recomputes
all motion-derived phase, actor and root fields for a nonzero rotation; it
does not approximate a rotation of cached RMS vectors. Accepted windows,
absolute positions on the timeline, rejected intervals, masks and trailing
frame declarations stay intact. There is no dataset-sized in-memory cache.

One shared group yaw is determined solely by source, one of the three fixed
seeds, and epoch. A fixed SHA-to-angle mapping makes replay independent of
model dropout, VAE sampling and global NumPy/Python/Torch RNG state. Every
reload of that source/seed/epoch receives the same physical arrays. Validation
always uses zero yaw. Disallowed captions also receive zero yaw at training;
the private caller must bind a real, frozen text-eligibility rule and its
population, not assume all world-direction captions allow arbitrary rotation.

The physical floors must come from the correct run's learning population.
This source does not fit them or authorize reusing main-experiment floors in
a held-out fold. Source, annotation, floor, feature, augmentation-rule and
runtime identities belong in the private execution inputs.

For A1, the source requires a separate speed-only cache and a typed floor
receipt. It checks the loaded physical config and floor values against that
receipt before yielding a view. Shared-yaw recomputation preserves the
selected speed-only frontend rather than falling back to signed vectors.
See the [A1 control contract](PHASESET_V2_A1_SPEED_CONTROL.md).

The optional [frozen human CLIP rows](PHASESET_V2_FROZEN_CLIP_ROWS.md) connect
the base/V2 human-only text interface without loading a tower at training.
The legacy verified-CF branch still needs real human provenance. Under the
user's [supervision amendment](PHASESET_V2_RELATION_SUPERVISION_AMENDMENT.md),
the separate [weak CLIP pool](PHASESET_V2_WEAK_CLIP_POOL.md) supplies disclosed
machine-caption training targets without granting human verification. Neither
branch changes the human-only primary gallery. The source supplies no
production lock, heartbeat, budget, retention, hardware,
pilot, formal-stage or final-test authority.

Analytic tests exercise full timelines with a rejected gap, RNG-independent
replay, shared rotation, actual phase recomputation, exact-zero validation,
caption multiplicity, task drift and final-test exclusion. Native validation
must separately load the complete admitted development population and its
closed fixed text features. An evaluation-only source qualification does not
verify a training yaw policy or demonstrate retrieval performance.

See the [fixed language rows](PHASESET_V2_FROZEN_LANGUAGE_ROWS.md),
[literature host](PHASESET_V2_LITERATURE_PARENT_HOST.md) and
[physical cache and calibration](PHASESET_V2_CALIBRATION_AND_CACHE.md).
