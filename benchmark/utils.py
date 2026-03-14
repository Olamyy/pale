"""Utilities for checkpoint serialization, deserialization, and measurement."""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def save_sklearn_checkpoint(model: Any, path: Path) -> None:
    """Save sklearn model to disk using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_sklearn_checkpoint(path: Path) -> Any:
    """Load sklearn model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_xgboost_checkpoint(booster: Any, path: Path) -> None:
    """Save XGBoost booster to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load_xgboost_checkpoint(path: Path) -> Any:
    """Load XGBoost booster from disk."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def save_pytorch_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save PyTorch model state_dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_pytorch_checkpoint(path: Path) -> Dict[str, Any]:
    """Load PyTorch model state_dict from disk."""
    return torch.load(path, weights_only=True)


def checkpoint_to_bytes(path: Path) -> bytes:
    """Read checkpoint file as bytes."""
    with open(path, "rb") as f:
        return f.read()


def extract_tensors_sklearn(model: Any) -> Dict[str, np.ndarray]:
    """Extract weight tensors from sklearn model as numpy arrays."""
    tensors = {}

    if hasattr(model, "coef_"):
        tensors["coef"] = np.asarray(model.coef_, dtype=np.float64)

    if hasattr(model, "intercept_"):
        tensors["intercept"] = np.asarray(model.intercept_, dtype=np.float64)

    if hasattr(model, "estimators_"):
        for i, estimator in enumerate(model.estimators_):
            sub = extract_tensors_sklearn(estimator)
            for key, val in sub.items():
                tensors[f"estimator_{i}_{key}"] = val

    if hasattr(model, "tree_"):
        tree = model.tree_
        tensors["tree_feature"] = np.asarray(tree.feature, dtype=np.int64)
        tensors["tree_threshold"] = np.asarray(tree.threshold, dtype=np.float64)
        tensors["tree_value"] = np.asarray(tree.value, dtype=np.float64)

    return tensors


def extract_tensors_xgboost(booster: Any) -> Dict[str, np.ndarray]:
    """Extract weight tensors from XGBoost booster as numpy arrays."""
    tensors = {}

    for tree_idx in range(booster.num_boosted_rounds()):
        tree = booster.get_booster().trees[tree_idx]
        tree_bytes = tree.serialize()
        tensors[f"tree_{tree_idx}"] = np.frombuffer(tree_bytes, dtype=np.uint8)

    return tensors


def extract_tensors_pytorch(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert PyTorch state_dict to numpy tensors."""
    tensors = {}
    for name, param in state_dict.items():
        tensors[name] = param.cpu().numpy().astype(np.float64)
    return tensors


def chunk_tensors(tensor: np.ndarray, chunk_size: int) -> list:
    """Split a numpy array into fixed-size chunks."""
    flat = tensor.ravel()
    byte_data = flat.astype(np.float64).tobytes()

    chunks = []
    for i in range(0, len(byte_data), chunk_size):
        chunks.append(byte_data[i : i + chunk_size])

    return chunks


class Timer:
    """Context manager for timing operations."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
