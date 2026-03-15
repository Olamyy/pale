from typing import Any, Dict

import numpy as np

from pale.errors import AdapterError


class XGBoostAdapter:
    """Extract and reconstruct XGBoost Booster parameters as numpy arrays.

    Uses save_raw("ubj") to capture the full binary model state as a uint8
    array.  This is the only approach that guarantees an exact round-trip,
    since XGBoost's internal tree representation cannot be reconstructed from
    the human-readable trees_to_dataframe() columns.

    Tensors:
        model_bytes: uint8 array of the raw UBJ-encoded model binary.
    """

    @staticmethod
    def extract(model: Any) -> Dict[str, np.ndarray]:
        """Extract booster as raw bytes.  Returns {"model_bytes": uint8 array}."""
        try:
            import xgboost as xgb
        except ImportError:
            raise AdapterError("xgboost is not installed")

        if not isinstance(model, xgb.Booster):
            raise AdapterError(
                f"XGBoostAdapter expects xgb.Booster, got {type(model).__name__}"
            )

        try:
            raw: bytes = model.save_raw("ubj")
        except Exception as e:
            raise AdapterError(f"save_raw('ubj') failed: {e}") from e

        return {"model_bytes": np.frombuffer(raw, dtype=np.uint8).copy()}

    @staticmethod
    def reconstruct(tensors: Dict[str, np.ndarray], original: Any) -> Any:
        """Reconstruct booster from stored model_bytes tensor.

        Ignores `original` — the full model state is encoded in tensors["model_bytes"].
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise AdapterError("xgboost is not installed")

        if "model_bytes" not in tensors:
            raise AdapterError("tensors must contain 'model_bytes' key")

        booster = xgb.Booster()
        try:
            booster.load_model(bytearray(tensors["model_bytes"]))
        except Exception as e:
            raise AdapterError(f"load_model from model_bytes failed: {e}") from e

        return booster
