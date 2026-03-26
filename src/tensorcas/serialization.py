from typing import List, Tuple

import numpy as np


def tensor_to_bytes(arr: np.ndarray) -> Tuple[bytes, str, List[int]]:
    """Serialize a numpy array to raw bytes.

    Returns:
        (raw_bytes, dtype_str, shape)
        raw_bytes: C-contiguous byte representation of arr
        dtype_str: numpy dtype string, e.g. "float32"
        shape: list of dimension sizes
    """
    arr = np.ascontiguousarray(arr)
    return arr.tobytes(), str(arr.dtype), list(arr.shape)


def bytes_to_tensor(raw_bytes: bytes, dtype: str, shape: List[int]) -> np.ndarray:
    """Deserialize raw bytes back to a numpy array.

    Returns a writable C-contiguous array.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.dtype(dtype)).copy()
    return arr.reshape(shape)
