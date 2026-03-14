"""Measure cross-run checkpoint similarity for Phase 4 viability."""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from benchmark.utils import (
    extract_tensors_pytorch,
    extract_tensors_sklearn,
    extract_tensors_xgboost,
    load_pytorch_checkpoint,
    load_sklearn_checkpoint,
    load_xgboost_checkpoint,
)


class CrossRunSimilarityMeasurer:
    """Measure similarity between checkpoints from different runs."""

    def measure_pair(
        self,
        ckpt_a_path: Path,
        ckpt_b_path: Path,
        framework: str = "sklearn",
    ) -> Dict[str, Any]:
        """Measure byte-level similarity between two checkpoints.

        Args:
            ckpt_a_path: Path to first checkpoint.
            ckpt_b_path: Path to second checkpoint.
            framework: "sklearn", "xgboost", or "pytorch".

        Returns:
            Dictionary with similarity metrics.
        """
        if framework == "sklearn":
            tensors_a = self._extract_sklearn(ckpt_a_path)
            tensors_b = self._extract_sklearn(ckpt_b_path)
        elif framework == "xgboost":
            tensors_a = self._extract_xgboost(ckpt_a_path)
            tensors_b = self._extract_xgboost(ckpt_b_path)
        elif framework == "pytorch":
            tensors_a = self._extract_pytorch(ckpt_a_path)
            tensors_b = self._extract_pytorch(ckpt_b_path)
        else:
            raise ValueError(f"Unknown framework: {framework}")

        return self._compute_similarity(tensors_a, tensors_b)

    def _extract_sklearn(self, path: Path) -> Dict[str, np.ndarray]:
        """Extract tensors from sklearn checkpoint."""
        model = load_sklearn_checkpoint(path)
        return extract_tensors_sklearn(model)

    def _extract_xgboost(self, path: Path) -> Dict[str, np.ndarray]:
        """Extract tensors from XGBoost checkpoint."""
        booster = load_xgboost_checkpoint(path)
        return extract_tensors_xgboost(booster)

    def _extract_pytorch(self, path: Path) -> Dict[str, np.ndarray]:
        """Extract tensors from PyTorch checkpoint."""
        state_dict = load_pytorch_checkpoint(None, path)
        return extract_tensors_pytorch(state_dict)

    def _compute_similarity(
        self, tensors_a: Dict[str, np.ndarray], tensors_b: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute byte-level similarity between tensor dictionaries."""
        total_bytes = 0
        identical_bytes = 0
        per_tensor_stats = {}

        for tensor_name in tensors_a:
            if tensor_name not in tensors_b:
                continue

            data_a = tensors_a[tensor_name].tobytes()
            data_b = tensors_b[tensor_name].tobytes()

            if len(data_a) != len(data_b):
                continue

            tensor_identical = sum(a == b for a, b in zip(data_a, data_b))
            tensor_total = len(data_b)

            per_tensor_stats[tensor_name] = {
                "total_bytes": tensor_total,
                "identical_bytes": tensor_identical,
                "similarity": tensor_identical / tensor_total if tensor_total > 0 else 0.0,
            }

            total_bytes += tensor_total
            identical_bytes += tensor_identical

        overall_similarity = (
            identical_bytes / total_bytes if total_bytes > 0 else 0.0
        )

        return {
            "overall_similarity": overall_similarity,
            "total_bytes": total_bytes,
            "identical_bytes": identical_bytes,
            "per_tensor_stats": per_tensor_stats,
        }


if __name__ == "__main__":
    measurer = CrossRunSimilarityMeasurer()

    print("Cross-run similarity measurement ready.")
    print("Use CrossRunSimilarityMeasurer to analyze checkpoint pairs from different runs.")
