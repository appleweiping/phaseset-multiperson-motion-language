# Frozen Morlet bytes and native formula verification

The physical 30 Hz PhasePair and 20 Hz PhaseSet kernels are immutable serialized
coefficients, not learned weights. Their original SHA-256 values remain:

- PhasePair: `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`.
- PhaseSet: `633d754bba12dab0555722f4b8963409a110c8681796cc907d163096e2e42655`.

The generated Python byte modules contain the original big-endian float64
serialization. The loader validates provenance, digest, support lengths,
finiteness and trailing bytes, then returns read-only canonical kernels.
The build-only exporter preserves the original CPython 3.14.5 / NumPy 2.4.6 /
Windows provenance. It neither trains a model nor creates experimental results.

## Runtime policy

The package pins NumPy 2.4.6. Within the advertised CPython 3.12--3.14 range,
x86-64 Windows and Linux select different strict native-formula checks:

- Windows must reproduce the original exact serialized digest and original
  residual checks. Even a one-ULP within-tolerance mutation is rejected.
- Linux must reproduce the observed native-formula digest below and remain
  within `2**-52` maximum complex-tap distance from the canonical bank.
- Other interpreter families, platforms, architectures, or NumPy versions are
  not qualified by this implementation. Unrecognized formula bytes fail closed.

Matching the exact formula fingerprint is required; an arbitrary Python patch
number is not a rejection reason. This does not declare every accepted machine
fully qualified for training. Each actual run still needs its own frozen
environment and numerical/model/CUDA qualification records.

Measured verifier lanes so far are Windows Python 3.12.0 and 3.14.5, and the
Linux Python 3.12.12 server, all with NumPy 2.4.6. Simulated identity tests only
check dispatch and do not count as execution on other interpreters.

Linux native-formula fingerprints:

- PhasePair: `3b4cec820d8202d4576d829d37626acc2414f0aeecf430e17c751d9bd1731f6c`.
- PhaseSet: `926c06aa57a27139d1ed1a9e0d9c3da191717b8a9495d6fff900bda44459d9a3`.

The measured maximum complex-tap differences are respectively
`6.206335383118183e-17` and `3.1031676915590914e-17`, both below `2**-52`.
The native diagnostic's DC/energy rounding residuals need not equal the
Windows construction receipt. Physical output is always the original canonical
bank; its original DC and unit-energy guarantees are preserved and tested.

## Regression boundaries

The unchanged five-reduction construction spy, 13D relation descriptor,
endpoint-swap algebra and known-delay checks remain. Separate tests reject
artifact mutation, Windows one-ULP drift, unqualified hardware and the native
left-to-right reduction mutant. The latter must fail the Linux formula digest
gate even when its tap distance alone is within tolerance.

These checks qualify kernel software behavior, not empirical retrieval quality,
BF16 precision, GPU performance, licensed data, or a completed scientific study.
