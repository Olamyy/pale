import numpy as np
import pytest
from pale.serialization import tensor_to_bytes, bytes_to_tensor


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_round_trip(dtype):
    rng = np.random.default_rng(42)
    if dtype == "bool":
        arr = rng.integers(0, 2, size=(8, 16)).astype(np.dtype(dtype))
    else:
        arr = rng.standard_normal((8, 16)).astype(np.dtype(dtype))

    raw_bytes, dtype_str, shape = tensor_to_bytes(arr)
    recovered = bytes_to_tensor(raw_bytes, dtype_str, shape)

    assert np.array_equal(arr, recovered)
    assert recovered.dtype == arr.dtype
    assert recovered.shape == arr.shape


def test_dtype_str_preserved():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    _, dtype_str, _ = tensor_to_bytes(arr)
    assert dtype_str == "float32"


def test_shape_preserved():
    arr = np.zeros((3, 4, 5), dtype=np.float64)
    _, _, shape = tensor_to_bytes(arr)
    assert shape == [3, 4, 5]


def test_1d_array():
    arr = np.arange(100, dtype=np.int64)
    raw, dtype_str, shape = tensor_to_bytes(arr)
    recovered = bytes_to_tensor(raw, dtype_str, shape)
    assert np.array_equal(arr, recovered)


def test_returned_array_is_writeable():
    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    raw, dtype_str, shape = tensor_to_bytes(arr)
    recovered = bytes_to_tensor(raw, dtype_str, shape)
    assert recovered.flags["WRITEABLE"]
    recovered[0] = 99  # must not raise


def test_noncontiguous_array_round_trips(tmp_path):
    """Transposed / non-contiguous arrays must round-trip to equal values."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4).T
    raw, dtype_str, shape = tensor_to_bytes(arr)
    recovered = bytes_to_tensor(raw, dtype_str, shape)
    assert np.array_equal(arr, recovered)
    assert list(recovered.shape) == shape
