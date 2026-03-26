import numpy as np
from tensorcas.serialization import tensor_to_bytes, bytes_to_tensor


def test_returned_array_is_writeable():
    # frombuffer returns read-only by default; we must copy
    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    raw, dtype_str, shape = tensor_to_bytes(arr)
    recovered = bytes_to_tensor(raw, dtype_str, shape)
    assert recovered.flags["WRITEABLE"]
    recovered[0] = 99


def test_noncontiguous_array_round_trips():
    # Transposed arrays have non-contiguous memory; must be handled before serialization
    arr = np.arange(12, dtype=np.float32).reshape(3, 4).T
    raw, dtype_str, shape = tensor_to_bytes(arr)
    recovered = bytes_to_tensor(raw, dtype_str, shape)
    assert np.array_equal(arr, recovered)
    assert list(recovered.shape) == shape
