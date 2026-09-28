# Frozen attention runtime policy

PhaseSet's numerical context disables PyTorch's native MHA fastpath on both
CPU and CUDA. It captures the caller's boolean, installs and checks `False`,
and restores the original value on normal and exceptional exit. The training
environment digest includes this policy; older code/environment-bound
checkpoints and qualifications are not evidence for the new runtime.

This is a runtime change, not a change to model layers, weights, masks, loss,
optimizer, or the registered CPU/CUDA tolerance (`rtol=1e-4, atol=1e-5`). It is
applied around the actual training/validation context, not just a test branch.

## Observed reason

A preserved same-weight B0 inference diagnostic found 336 group-embedding
elements outside that tolerance with native fastpath enabled (maximum absolute
difference about 2.26e-4). Disabling it reduced the exceedance count to zero
(maximum about 7.15e-7). The direct temporal comparison exhibited the same
pattern with both an all-false padding mask and no mask. This establishes the
observed runtime dependency; it is not a claim about all hardware or a general
vendor defect.

The original residual-retrieval failure remains a failed historical attempt.
The diagnostic did not overwrite it or change its acceptance threshold.

## Subsequent registered-server observations

- Focused CPU regression: 33 tests passed. A separate full suite passed 949
  tests, 2 skipped, and 39 subtests, with unchanged source bytes.
- Actual official CLIP CPU requalification: three forwards passed. The old
  two-row output golden and separate long-caption output were unchanged,
  and caller RNG was preserved. Source/runtime/receipt identities changed
  correctly; complete old and new receipts are not byte-identical.
- A new actual-width FP32 CUDA residual-retrieval engineering observation
  passed. Frozen-base embedding maximum CPU/CUDA difference was 7.153e-7;
  score difference was 4.471e-8; loss difference was zero. The zero-initialized
  residual equaled the frozen base and the exercised residual text-head
  parameters received finite nonzero gradients. The frozen text fixture was
  unchanged. Peak allocated memory was 108,373,504 bytes under a 2 GiB cap.

The last observation exercises the closed retrieval-head seam with analytic
features, not the periodic core or an optimizer. It does not qualify full
training, BF16, performance, data access, or a winning base checkpoint. A
subsequent B0 requalification stopped before model construction because no GPU
met the fixed 8 GiB free-memory requirement. Once memory became available, a
separate resource-retry chain repeated retrieval and passed B0, B1, and B2 with
the same reviewed source and unchanged numerical checks. That chain used new
attempt directories; the failed resource receipt was not overwritten. These
are bounded FP32 forward/backward observations, not optimizer, BF16, throughput,
or full-training qualification. Existing workloads were not interrupted.

## CLIP and empirical boundaries

The frozen text adapter pins the entire installed training module, records
`mha_fastpath_enabled=false`, and rejects a live true flag. Its caller must
configure the validated CPU runtime before loading it; the adapter does not
silently alter global flags. See [the text adapter](FROZEN_CLIP_TEXT_ADAPTER.md).

The [capture evaluator](CAPTURE_VALIDATION.md) now aggregates every admitted
window into one capture embedding and uses true capture identities in the
training API. Its private-host disk composition remains unfinished. Window
scores and repeating actor-set lineage IDs are not a substitute. This runtime
repair is not a completed empirical qualification or a scientific result.
