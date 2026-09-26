# Embody SMPL-X adapter integration

Status: implemented integration seam. A caller-licensed official locked-head
neutral model passed a server loader/batch qualification on 2026-09-26.
This is not native-capture training or retrieval evidence. Contract fixtures
remain nonempirical and are not experimental results.

## Actual asset qualification, 2026-09-26

The official Embody example uses locked-head SMPL-X. The SMPL-X download page
warns that its shape space differs from v1.1; do not silently substitute that
model. The observed `lockedhead_20230207` neutral NPZ matched its official
archive's checksum. With the existing evaluator and isolated server runtime,
the `3 → 1 → 3` frame-batch loader check passed in 4.865 seconds, returning
finite float32 `[3,22,3]` joints and repeatable outputs. This qualifies loading
and short-tail compatibility, not a learned model or a dataset benchmark.

The downloaded human-text archives were independently matched to the existing
participant-disjoint component manifest: 572 eligible captures, 16 components,
and 400/96/76 captures with no participant/capture overlap. Native holistic
file presence is **398/96/76**: two training captures lack holistic text.
As preregistered, those are excluded from the holistic task before any model
scores, with no split reassignment. The 400-capture training universe remains
unchanged for tasks whose required annotations are present. Test caption files
were sealed by digest without decoding their semantic contents.

Dataset archives, body models, identifiers, signed URLs, private receipts and
paths remain private. These observations do not grant access to another user;
each reproducer must obtain their own official dataset/model authorization.
The next server check decoded one actual four-actor pilot-training window:
native tracks had 3600 frames per actor, the first accepted 300-frame source
window converted to finite float32 `[4,200,22,3]` in 11.048 seconds with zero
interpolated actor frames. This uses real Embody parameters and the official
body model, not fixture arrays. It does not read caption semantics, create
retrieval scores, persist a prepared training cache, or train a model.
Full-corpus motion auditing and the formal training matrix remain separate gates.

## Native boundary

For every synchronized actor, supply exactly these local `.npy` tracks and an
expected SHA-256 for each file:

- `smplx_mesh_body_pose`: float `[T,63]`
- `smplx_mesh_global_orient`: float `[T,3]`
- `smplx_mesh_transl`: float `[T,3]`
- `smplx_mesh_betas`: float `[T,300]`
- `smplx_mesh_left_hand_pose`: float `[T,45]`
- `smplx_mesh_right_hand_pose`: float `[T,45]`
- `missing`: binary `[T]` or `[T,1]`; upstream defines 0 as corrupted and 1 as
  good tracking

The adapter retains corrupted frames in place, marks all 22 body joints
untracked, and uses exact positive zero for their coordinates. It rejects a
missing track, incomplete file provenance, non-finite observed parameters,
inconsistent actor lengths, duplicate actor commitments, and invalid evaluator
output. It has no dataset discovery, download, metadata-ID, caption, or audio
surface.

`LicensedSMPLXEvaluator(model_path, device=..., expected_model_sha256=...)` is the concrete conversion
path. It lazy-imports the caller-installed official `smplx` package and calls
`smplx.create` with Embody's fixed configuration: neutral SMPL-X,
`flat_hand_mean=True`, `num_betas=300`, `num_expression_coeffs=100`, and
`use_pca=False`. It loads only a local `SMPLX_NEUTRAL.npz`, observes that exact
file's SHA-256 before and after model construction, and derives its evaluator
provenance from the observed digest and frozen configuration. It neither
downloads the model nor represents that the caller has a license.

The expected model digest is mandatory and must be obtained from the caller's
verified licensed-asset manifest. A mismatch stops before optional imports or
model construction. This is important because the upstream loader enables NPZ
pickle support. Hashing an arbitrary local file and supplying that same hash
does not establish trust, provenance, or a license; do not use that shortcut.

Install the optional software dependency with `pip install '.[embody]'`;
it pins `smplx==0.1.28`. Model assets are not included in this extra. The
environment lock and the actual licensed-model witness must also be recorded
before treating any conversion as qualified. Unit fixtures are not that witness.

The evaluator disables constructor-owned pose, shape, translation, hand, jaw,
eye, and expression parameters. Every call supplies the six Embody tensors and
explicit zero jaw/eye/expression tensors at the actual chunk length. It also
sets the owned model's `batch_size` under a lock for each chunk because the
official regular `SMPLX.forward` uses that member when repeating landmark
barycentric coordinates. This prevents a short final chunk from inheriting a
fixed constructor length. It runs in evaluation mode under `torch.no_grad`,
does not return vertices or shaped vertices, and exports the first 22 official
joints (pelvis through right wrist) as finite float32 NumPy data.

A custom evaluator remains possible for integration testing, but then the
caller must explicitly supply its provenance SHA-256. For the exact concrete
evaluator, `embody_to_private_capture` reads the observed model binding itself
and rejects any conflicting supplied value. All float64 input and output is
checked again after float32 conversion so overflow cannot enter the PhaseSet
seam.

Pass the resulting `PrivateCaptureArrays` to the existing
`prepare_capture(...)`. That existing path, rather than this adapter, owns
body-22 extraction defense in depth, 300-frame windows, 30-to-20 Hz resampling,
missing-run rejection/interpolation, group canonicalization, augmentation, and
dynamic collation.

## Access and license gates not implemented in code

- Embody data access is not established by this adapter. The official repository says a user
  must complete the release form before downloads work, that approved access
  supplies 21 links, and that the dataset is governed by the XRCIA license.
  This adapter neither requests nor proves that approval.
- The official Meta release-form page did not render in the documentation
  fetch used for this review, so its current identity fields and review timing
  are not asserted here.
- SMPL-X model assets are a separate prerequisite. The official SMPL-X site
  offers independent account registration (email/password) and requires
  acceptance of its privacy statement, Model license, and Body license for
  noncommercial research. SMPL-X access does not grant Embody access.
- No licensed dataset array, SMPL-X model asset, credential, signed link,
  private path, participant identifier, or generated output may be committed
  to the public repository.

## Official sources checked 2026-09-07

- Embody README (native folders, 30 fps, missing convention, access and XRCIA):
  https://github.com/facebookresearch/embody-3d/blob/main/README.md
- Embody feature names:
  https://github.com/facebookresearch/embody-3d/blob/main/src/enums.py
- Embody official SMPL-X invocation:
  https://github.com/facebookresearch/embody-3d/blob/main/src/run.py
- Embody local-array loading behavior:
  https://github.com/facebookresearch/embody-3d/blob/main/src/dataloader.py
- Official SMPL-X model API:
  https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py
- Official SMPL-X joint ordering:
  https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
- Official SMPL-X registration and licensing:
  https://smpl-x.is.tue.mpg.de/register.php
