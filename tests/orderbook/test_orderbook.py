# tests/orderbook/test_orderbook.py

from __future__ import annotations

import pickle
from typing import Dict, Tuple

import pytest
from graphblas import Vector, dtypes

from toolbox.orderbook import Orderbook


def v_from_dict(size: int, data: Dict[int, float]) -> Vector:
    """Build a FP64 Vector from an index->value mapping (sparse)."""
    if not data:
        return Vector.from_coo([], [], size=size, dtype=dtypes.FP64)
    idx = list(data.keys())
    val = list(data.values())
    return Vector.from_coo(idx, val, size=size, dtype=dtypes.FP64)


def v_to_dict(v: Vector) -> Dict[int, float]:
    """Convert Vector to a dict[int,float] via COO."""
    idx, val = v.to_coo()
    return {int(i): float(x) for i, x in zip(idx.tolist(), val.tolist())}


def assert_vec_eq(actual: Vector, expected: Dict[int, float]) -> None:
    assert v_to_dict(actual) == {int(k): float(v) for k, v in expected.items()}


def test_empty_orderbook_properties() -> None:
    ob = Orderbook()
    assert ob.size == 0
    assert ob.length == 0


def test_append_sets_length_and_does_not_mutate_input_vector() -> None:
    ob = Orderbook()
    size = 10

    key = {"id": 1}
    arr = v_from_dict(size, {2: 5.0, 7: 3.0})
    before = v_to_dict(arr)

    ob.append(key, arr)

    assert ob.size == 1
    assert ob.length == size

    # Input vector must not be mutated by append
    assert v_to_dict(arr) == before


def test_append_length_mismatch_raises() -> None:
    ob = Orderbook()

    ob.append({"id": 1}, v_from_dict(5, {0: 1.0}))
    with pytest.raises(TypeError, match="Length mismatch"):
        ob.append({"id": 2}, v_from_dict(6, {0: 1.0}))


def test_get_at_index_returns_key_reference_and_vector_copy_semantics() -> None:
    ob = Orderbook()
    size = 8

    key = {"id": 123, "name": "A"}
    arr = v_from_dict(size, {1: 2.0, 3: 4.0})
    ob.append(key, arr)

    out_key, out_vec = ob.get_at_index(0)

    # Key is stored by reference: identity should be preserved
    assert out_key is key

    # Vector contents should match appended order
    assert_vec_eq(out_vec, {1: 2.0, 3: 4.0})

    # Mutating key after append should reflect when fetched again (reference semantics)
    key["name"] = "B"
    out_key2, _ = ob.get_at_index(0)
    assert out_key2["name"] == "B"


def test_remove_updates_size_and_bounds_check() -> None:
    ob = Orderbook()
    size = 5

    ob.append({"id": 1}, v_from_dict(size, {0: 1.0}))
    ob.append({"id": 2}, v_from_dict(size, {1: 2.0}))
    ob.append({"id": 3}, v_from_dict(size, {2: 3.0}))
    assert ob.size == 3

    with pytest.raises(IndexError, match="Index out of range"):
        ob.remove(-1)
    with pytest.raises(IndexError, match="Index out of range"):
        ob.remove(3)

    ob.remove(1)
    assert ob.size == 2

    k0, _ = ob.get_at_index(0)
    k1, _ = ob.get_at_index(1)
    assert k0["id"] == 1
    assert k1["id"] == 3


def test_allocate_greedy_allocates_and_mutates_values_and_removes_fulfilled_orders() -> None:
    ob = Orderbook()
    n = 10

    key1 = {"id": "o1"}
    key2 = {"id": "o2"}

    # Orders (sparse)
    # o1: idx2=5, idx7=3
    # o2: idx2=4, idx8=6
    ob.append(key1, v_from_dict(n, {2: 5.0, 7: 3.0}))
    ob.append(key2, v_from_dict(n, {2: 4.0, 8: 6.0}))

    # Available values: idx2=6, idx7=1, idx8=10
    values = v_from_dict(n, {2: 6.0, 7: 1.0, 8: 10.0})

    res = ob.allocate_greedy(values)

    # Return type: list[(dict, Vector)]
    assert isinstance(res, list)
    assert len(res) == 2

    (rk1, alloc1), (rk2, alloc2) = res
    assert rk1 is key1
    assert rk2 is key2

    # Greedy allocation on intersection:
    # For o1:
    #  idx2 min(5,6)=5, idx7 min(3,1)=1 => alloc1 {2:5,7:1}
    #  values after o1: idx2=1, idx7=0 removed, idx8=10
    # For o2:
    #  idx2 min(4,1)=1, idx8 min(6,10)=6 => alloc2 {2:1,8:6}
    #  values after o2: idx2=0 removed, idx8=4
    assert_vec_eq(alloc1, {2: 5.0, 7: 1.0})
    assert_vec_eq(alloc2, {2: 1.0, 8: 6.0})

    # values must be mutated in-place:
    # remaining values: idx8=4
    assert_vec_eq(values, {8: 4.0})

    # Orders should be updated/removed:
    # o1 remaining: idx7 had 3-1=2, idx2 becomes 0 => {7:2}
    # o2 remaining: idx2 had 4-1=3, idx8 becomes 0 => {2:3}
    assert ob.size == 2

    k0, v0 = ob.get_at_index(0)
    k1, v1 = ob.get_at_index(1)
    assert k0 is key1
    assert k1 is key2
    assert_vec_eq(v0, {7: 2.0})
    assert_vec_eq(v1, {2: 3.0})

    # Next allocation with values that fulfill everything
    values2 = v_from_dict(n, {2: 10.0, 7: 10.0})
    res2 = ob.allocate_greedy(values2)

    assert len(res2) == 2
    # both orders should be fulfilled and removed
    assert ob.size == 0
    assert ob.length == 0

    # values2 after fulfilling: idx2 left 7 (10-3), idx7 left 8 (10-2)
    assert_vec_eq(values2, {2: 7.0, 7: 8.0})


def test_allocate_greedy_length_mismatch_raises() -> None:
    ob = Orderbook()
    ob.append({"id": 1}, v_from_dict(5, {0: 1.0}))

    values = v_from_dict(6, {0: 1.0})
    with pytest.raises(TypeError, match="Length mismatch"):
        ob.allocate_greedy(values)


def test_pickle_roundtrip_preserves_entries_and_reference_keys() -> None:
    ob = Orderbook()
    n = 12
    k1 = {"id": 1}
    k2 = {"id": 2}

    ob.append(k1, v_from_dict(n, {1: 2.0, 10: 5.0}))
    ob.append(k2, v_from_dict(n, {3: 4.0}))

    blob = pickle.dumps(ob)
    ob2: Orderbook = pickle.loads(blob)

    assert ob2.size == 2
    assert ob2.length == n

    # Note: keys are pickled/unpickled dicts -> they are equal but not identical objects
    k1b, v1b = ob2.get_at_index(0)
    k2b, v2b = ob2.get_at_index(1)

    assert dict(k1b) == dict(k1)
    assert dict(k2b) == dict(k2)
    assert_vec_eq(v1b, {1: 2.0, 10: 5.0})
    assert_vec_eq(v2b, {3: 4.0})
