"""Utilities for checkpoint serialization, deserialization, and measurement."""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def load_sklearn_checkpoint(path: Path) -> Any:
    """Load sklearn model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_xgboost_checkpoint(path: Path) -> Any:
    """Load XGBoost booster from disk."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def load_pytorch_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """Load PyTorch model state_dict from disk."""
    return torch.load(path, weights_only=True, map_location=torch.device("cpu"))


def extract_tensors_sklearn(model: Any) -> Dict[str, np.ndarray]:
    """Extract weight tensors from sklearn model as numpy arrays.

    For tree-based ensemble models, aggregates tree arrays across all estimators
    into single tensors. For linear models, extracts coefficients and intercept.
    """
    tensors = {}

    if hasattr(model, "coef_"):
        tensors["coef"] = np.asarray(model.coef_, dtype=np.float64)

    if hasattr(model, "intercept_"):
        tensors["intercept"] = np.asarray(model.intercept_, dtype=np.float64)

    if hasattr(model, "estimators_"):
        features, thresholds, values = [], [], []
        for estimator_row in np.array(model.estimators_).ravel():
            if hasattr(estimator_row, "tree_"):
                t = estimator_row.tree_
                features.append(t.feature)
                thresholds.append(t.threshold)
                values.append(t.value.ravel())

        if features:
            tensors["all_tree_features"] = np.concatenate(features).astype(np.float64)
            tensors["all_tree_thresholds"] = np.concatenate(thresholds).astype(np.float64)
            tensors["all_tree_values"] = np.concatenate(values).astype(np.float64)

    elif hasattr(model, "tree_"):
        tree = model.tree_
        tensors["tree_feature"] = tree.feature.astype(np.float64)
        tensors["tree_threshold"] = tree.threshold.astype(np.float64)
        tensors["tree_value"] = tree.value.ravel().astype(np.float64)

    return tensors


def extract_tensors_xgboost(booster: Any) -> Dict[str, np.ndarray]:
    """Extract weight tensors from XGBoost booster as numpy arrays.

    Extracts split thresholds, gains, and leaf values from the tree structure
    using trees_to_dataframe(). Preserves float32 dtype.
    """
    try:
        df = booster.trees_to_dataframe()

        tensors = {
            "split_gain": df["Gain"].to_numpy(dtype=np.float32),
            "split_cover": df["Cover"].to_numpy(dtype=np.float32),
            "leaf_values": df[df["Feature"] == "Leaf"]["Gain"].to_numpy(
                dtype=np.float32
            ),
        }

        threshold_data = df[df["Feature"] != "Leaf"]["Split"].to_numpy(
            dtype=np.float32
        )
        if len(threshold_data) > 0:
            tensors["split_threshold"] = threshold_data

        return tensors

    except Exception:
        return {}


def extract_tensors_pytorch(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert PyTorch state_dict to numpy tensors.

    Preserves original dtype (typically float32). Does not cast to float64
    as that doubles memory and distorts compression ratios.
    """
    tensors = {}
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            tensors[name] = param.cpu().detach().numpy()

    return tensors


def chunk_tensors(
    tensor: np.ndarray, chunk_size: int
) -> Tuple[List[bytes], np.dtype]:
    """Split a numpy array into fixed-size chunks, preserving dtype.

    Returns (chunks, dtype) so dtype info is available during delta computation.
    Preserves original dtype — does not cast to float64.
    """
    flat = tensor.ravel()
    byte_data = flat.tobytes()

    chunks = []
    for i in range(0, len(byte_data), chunk_size):
        chunks.append(byte_data[i : i + chunk_size])

    return chunks, flat.dtype


class Timer:
    """Context manager for timing operations in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
