import numpy as np
import pytest
import pickle

from toolbox.orderbook import Orderbook


@pytest.fixture
def ob001():
    o = Orderbook()
    o.append(b"abc", np.array([1, 2, 3], dtype=np.double))
    o.append(b"def", np.array([4, 5, 6], dtype=np.double))
    o.append(b"ghi", np.array([7, 8, 9], dtype=np.double))
    return o


def test_orderbook_basic_and_allocate(ob001):
    assert ob001.size == 3
    assert ob001.length == 3

    key, arr = ob001.get_at_index(0)
    assert key == b"abc"
    assert np.array_equal(arr, np.array([1, 2, 3], dtype=np.double))

    stock = np.array([5, 5, 5], dtype=np.double)
    allocations = ob001.allocate_greedy(stock)

    # allocate_greedy must mutate stock in-place
    assert np.array_equal(stock, np.array([0, 0, 0], dtype=np.double))

    assert len(allocations) == 2
    assert allocations[0][0] == b"abc"
    assert np.array_equal(allocations[0][1], np.array([1, 2, 3], dtype=np.double))
    assert allocations[1][0] == b"def"
    assert np.array_equal(allocations[1][1], np.array([4, 3, 2], dtype=np.double))

    # first entry fully fulfilled => removed
    assert ob001.size == 2
    assert ob001.get_at_index(0)[0] == b"def"


def test_pickle_roundtrip_preserves_entries(ob001):
    pickled = pickle.dumps(ob001)
    unpickled = pickle.loads(pickled)

    assert unpickled.size == ob001.size
    assert unpickled.length == ob001.length

    for i in range(ob001.size):
        k1, a1 = ob001.get_at_index(i)
        k2, a2 = unpickled.get_at_index(i)
        assert k2 == k1
        assert np.array_equal(a2, a1)


def test_append_length_mismatch_raises(ob001):
    with pytest.raises(TypeError, match="Length mismatch"):
        ob001.append(b"zzz", np.array([1, 2], dtype=np.double))


def test_get_at_index_out_of_range_raises(ob001):
    with pytest.raises(IndexError):
        ob001.get_at_index(-1)
    with pytest.raises(IndexError):
        ob001.get_at_index(ob001.size)


def test_remove_updates_head_tail_and_resets_empty():
    ob = Orderbook()
    ob.append(b"a", np.array([1, 0, 0], dtype=np.double))
    ob.append(b"b", np.array([0, 1, 0], dtype=np.double))

    ob.remove(0)  # remove head
    assert ob.size == 1
    assert ob.get_at_index(0)[0] == b"b"

    ob.remove(0)  # now remove last remaining
    assert ob.size == 0
    assert ob.length == 0  # per your design: empty resets length


def test_allocate_greedy_length_mismatch_raises(ob001):
    stock = np.array([1, 2], dtype=np.double)
    with pytest.raises(TypeError, match="Length mismatch"):
        ob001.allocate_greedy(stock)


def test_allocate_greedy_zero_stock_no_changes(ob001):
    stock = np.zeros(ob001.length, dtype=np.double)
    before = pickle.loads(pickle.dumps(ob001))  # cheap deep copy for comparison

    allocations = ob001.allocate_greedy(stock)

    assert allocations == []
    assert np.array_equal(stock, np.zeros(ob001.length, dtype=np.double))
    assert ob001.size == before.size
    for i in range(ob001.size):
        k, a = ob001.get_at_index(i)
        k0, a0 = before.get_at_index(i)
        assert k == k0
        assert np.array_equal(a, a0)


def test_allocate_greedy_partial_fulfillment_keeps_entry():
    ob = Orderbook()
    ob.append(b"x", np.array([10, 0, 0], dtype=np.double))

    stock = np.array([3, 0, 0], dtype=np.double)
    allocs = ob.allocate_greedy(stock)

    assert len(allocs) == 1
    assert allocs[0][0] == b"x"
    assert np.array_equal(allocs[0][1], np.array([3, 0, 0], dtype=np.double))

    # entry should remain with remaining demand
    assert ob.size == 1
    k, a = ob.get_at_index(0)
    assert k == b"x"
    assert np.array_equal(a, np.array([7, 0, 0], dtype=np.double))
