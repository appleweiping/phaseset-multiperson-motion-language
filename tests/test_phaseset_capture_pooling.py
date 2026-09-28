"""Numeric-only tests of complete capture pooling, not model evaluation."""

import hashlib
from dataclasses import replace
from itertools import permutations
import struct

import numpy as np
import pytest
from phaseset_core.capture_pooling import (
    CapturePoolingError,
    CapturePoolingResourceLimit,
    CaptureWindowPlan,
    pool_capture_windows,
)


def _identity(value: str) -> bytes:
    return hashlib.sha256(value.encode("ascii")).digest()


def _fixture(count=3):
    ids = tuple(_identity(f"capture/window/{index}") for index in range(count))
    plan = CaptureWindowPlan(_identity("capture"), ids, tuple(300 * index for index in range(count)))
    base = np.broadcast_to(np.arange(1, count + 1, dtype=np.float32)[:, None], (count, 512)).copy()
    tokens = np.broadcast_to(base[:, None, :], (count, 6, 512)).copy()
    mask = np.ones((count, 6), dtype=np.bool_)
    mask[:, 5] = False
    mask[0, 1] = False
    tokens[~mask] = 0.0
    return plan, ids, base, tokens, mask


def test_base_and_band_means_precede_normalization():
    plan, ids, base, tokens, mask = _fixture()
    output = pool_capture_windows(plan, window_commitments=ids, base_embeddings=base, tokens=tokens, band_mask=mask)
    np.testing.assert_array_equal(output.base_embedding, np.full(512, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(output.tokens[0], np.full(512, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(output.tokens[1], np.full(512, 2.5, dtype=np.float32))
    np.testing.assert_array_equal(output.valid_window_count, [3, 2, 3, 3, 3, 0])
    assert output.band_mask.tolist() == [True, True, True, True, True, False]
    assert output.tokens[5].tobytes() == bytes(512 * 4)
    assert output.plan_sha256 == plan.sha256
    assert output.capture_commitment == plan.capture_commitment
    assert output.window_count == 3
    for array in (output.base_embedding, output.tokens, output.band_mask, output.valid_window_count):
        assert not array.flags.writeable


def test_every_input_permutation_is_bitwise_equal():
    plan, ids, base, tokens, mask = _fixture(5)
    # Cancellation makes a canonical reduction order material.
    base[:, 0] = np.array([1e20, 1.0, -1e20, 3.0, 2.0], dtype=np.float32)
    baseline = pool_capture_windows(plan, window_commitments=ids, base_embeddings=base, tokens=tokens, band_mask=mask)
    for permutation in permutations(range(5)):
        indices = list(permutation)
        observed = pool_capture_windows(
            plan,
            window_commitments=tuple(ids[index] for index in indices),
            base_embeddings=base[indices],
            tokens=tokens[indices],
            band_mask=mask[indices],
        )
        for name in ("base_embedding", "tokens", "band_mask", "valid_window_count"):
            assert getattr(observed, name).tobytes() == getattr(baseline, name).tobytes()


def test_literal_binary64_tree_and_chronology_have_independent_bit_oracle():
    ids = tuple(bytes([value]) * 32 for value in (3, 1, 4, 0, 2))
    plan = CaptureWindowPlan(bytes(32), ids, (0, 300, 600, 900, 1200))
    base = np.zeros((5, 512), dtype=np.float32)
    base[:, 0] = [2**54, 1, -(2**54), 1, 1]
    base[:, 1] = [2**24, 1, -(2**24), 1, 1]
    output = pool_capture_windows(plan, window_commitments=ids, base_embeddings=base)
    # Hand-expanded adjacent binary64 tree, not the implementation's helper.
    expected_zero = ((float(2**54) + 1.0) + (float(-(2**54)) + 1.0) + 1.0) / 5.0
    expected_one = ((float(2**24) + 1.0) + (float(-(2**24)) + 1.0) + 1.0) / 5.0
    assert struct.pack("<f", expected_zero) == bytes.fromhex("cdcc4c3e")  # 0.2
    assert struct.pack("<f", expected_one) == bytes.fromhex("9a99193f")  # 0.6
    assert output.base_embedding[:2].tobytes() == bytes.fromhex("cdcc4c3e9a99193f")
    sequential = 0.0
    for value in base[:, 0]:
        sequential += float(value)
    assert struct.pack("<f", sequential / 5.0) != bytes.fromhex("cdcc4c3e")
    assert struct.pack("<f", float(np.sum(base[:, 0].astype(np.float64))) / 5.0) != bytes.fromhex("cdcc4c3e")
    a, b, c, d, e = base[:, 1]
    wrong_float32_tree = np.float32(np.float32(np.float32(a + b) + np.float32(c + d)) + e)
    assert struct.pack("<f", float(wrong_float32_tree) / 5.0) != bytes.fromhex("9a99193f")
    # Sorting by private window digest instead of plan chronology changes the tree.
    by_digest = sorted(range(5), key=lambda index: ids[index])
    a, b, c, d, e = (float(base[index, 0]) for index in by_digest)
    wrong_order_tree = ((a + b) + (c + d) + e) / 5.0
    assert struct.pack("<f", wrong_order_tree) != bytes.fromhex("cdcc4c3e")


def test_plan_digest_literal_golden_binds_all_identity_fields_and_schema():
    first, second = bytes.fromhex("11" * 32), bytes.fromhex("22" * 32)
    plan = CaptureWindowPlan(bytes(32), (first, second), (0, 600))
    payload = (
        b'{"capture_commitment":"' + b"00" * 32
        + b'","schema":"phaseset-capture-window-plan-v1","source_start_frames":[0,600],'
        + b'"window_commitments":["' + b"11" * 32 + b'","' + b"22" * 32 + b'"]}'
    )
    expected = "8db60469ea75a3e4d1cbf7f7e64000322dcdb4ef8b85b82d4c12e022bf203f3e"
    assert hashlib.sha256(payload).hexdigest() == expected
    assert plan.sha256 == expected
    mutants = (
        replace(plan, capture_commitment=bytes.fromhex("33" * 32)),
        replace(plan, window_commitments=(bytes.fromhex("33" * 32), second)),
        replace(plan, window_commitments=(second, first)),
        replace(plan, source_start_frames=(300, 600)),
        replace(plan, source_start_frames=(0, 900)),
    )
    assert all(mutant.sha256 != expected for mutant in mutants)
    assert len({mutant.sha256 for mutant in mutants}) == len(mutants)
    wrong_schema = payload.replace(b"phaseset-capture-window-plan-v1", b"phaseset-capture-window-plan-v2")
    assert hashlib.sha256(wrong_schema).hexdigest() != expected


@pytest.mark.parametrize("mode", ["missing", "duplicate", "foreign"])
def test_missing_duplicate_or_foreign_window_rejected(mode):
    plan, ids, base, _, _ = _fixture()
    if mode == "missing":
        supplied, arrays = ids[:-1], base[:-1]
    elif mode == "duplicate":
        supplied, arrays = (ids[0], ids[0], ids[2]), base
    else:
        supplied, arrays = (*ids[:-1], _identity("foreign")), base
    with pytest.raises(CapturePoolingError, match="complete capture census"):
        pool_capture_windows(plan, window_commitments=supplied, base_embeddings=arrays)


def test_single_window_base_only_and_distinct_capture_identity():
    plan, ids, base, _, _ = _fixture(1)
    output = pool_capture_windows(plan, window_commitments=ids, base_embeddings=base)
    assert output.base_embedding.tobytes() == base[0].tobytes()
    assert output.tokens is None and output.band_mask is None and output.valid_window_count is None
    # There is deliberately no actor-set identity in this interface.
    other = replace(plan, capture_commitment=_identity("other-capture-same-actors"))
    assert other.sha256 != plan.sha256
    assert pool_capture_windows(other, window_commitments=ids, base_embeddings=base).capture_commitment != output.capture_commitment


def test_all_invalid_bands_are_positive_zero_without_mutating_inputs():
    plan, ids, base, tokens, mask = _fixture()
    tokens.fill(0.0)
    mask.fill(False)
    before = tuple(value.tobytes() for value in (base, tokens, mask))
    output = pool_capture_windows(plan, window_commitments=ids, base_embeddings=base, tokens=tokens, band_mask=mask)
    assert output.tokens.tobytes() == bytes(6 * 512 * 4)
    assert not output.band_mask.any()
    assert not output.valid_window_count.any()
    assert before == tuple(value.tobytes() for value in (base, tokens, mask))


@pytest.mark.parametrize("value", [-0.0, 1.0, np.nan])
def test_invalid_band_payload_cannot_be_hidden(value):
    plan, ids, base, tokens, mask = _fixture()
    tokens[0, 5, 0] = value
    with pytest.raises(CapturePoolingError):
        pool_capture_windows(plan, window_commitments=ids, base_embeddings=base, tokens=tokens, band_mask=mask)


def test_explicit_resource_limit_does_not_drop_windows():
    plan, ids, base, tokens, mask = _fixture()
    size = base.nbytes + tokens.nbytes + mask.nbytes
    with pytest.raises(CapturePoolingResourceLimit, match="RESOURCE_LIMIT"):
        pool_capture_windows(plan, window_commitments=ids, base_embeddings=base, tokens=tokens, band_mask=mask, max_encoded_bytes=size - 1)
    assert pool_capture_windows(plan, window_commitments=ids, base_embeddings=base, tokens=tokens, band_mask=mask, max_encoded_bytes=size).window_count == 3


def test_plan_rejects_empty_repeated_offgrid_and_reordered_starts():
    plan, ids, _, _, _ = _fixture()
    for starts in ((0, 0, 600), (0, 301, 600), (600, 300, 0), (0, True, 600)):
        with pytest.raises(CapturePoolingError):
            CaptureWindowPlan(plan.capture_commitment, ids, starts)
    with pytest.raises(CapturePoolingError):
        CaptureWindowPlan(plan.capture_commitment, (), ())


def test_input_types_shapes_finiteness_and_optional_fields():
    plan, ids, base, tokens, mask = _fixture()
    for array in (base.astype(np.float64), base[:, ::2], base.copy(order="F")):
        with pytest.raises(CapturePoolingError):
            pool_capture_windows(plan, window_commitments=ids, base_embeddings=array)
    broken = base.copy()
    broken[0, 0] = np.inf
    with pytest.raises(CapturePoolingError, match="finite"):
        pool_capture_windows(plan, window_commitments=ids, base_embeddings=broken)
    for partial in ({"tokens": tokens}, {"band_mask": mask}):
        with pytest.raises(CapturePoolingError, match="together"):
            pool_capture_windows(plan, window_commitments=ids, base_embeddings=base, **partial)


def test_plan_and_output_repr_do_not_expose_lineage_or_features():
    plan, ids, base, _, _ = _fixture()
    output = pool_capture_windows(plan, window_commitments=ids, base_embeddings=base)
    for rendered in (repr(plan), repr(output)):
        assert plan.capture_commitment.hex() not in rendered
        assert ids[0].hex() not in rendered
        assert "array(" not in rendered
