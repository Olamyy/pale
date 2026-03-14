"""Measure compression effectiveness of checkpoint pairs."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import zstandard as zstd
from tqdm import tqdm

from benchmark.utils import (
    Timer,
    chunk_tensors,
    checkpoint_to_bytes,
    extract_tensors_pytorch,
    extract_tensors_sklearn,
    extract_tensors_xgboost,
    load_pytorch_checkpoint,
    load_sklearn_checkpoint,
    load_xgboost_checkpoint,
)


class CheckpointPairMeasurer:
    """Measure compression effectiveness for checkpoint pairs."""

    def __init__(
        self,
        chunk_sizes: List[int] = None,
        tau_values: List[float] = None,
    ):
        self.chunk_sizes = chunk_sizes or [
            256 * 1024,
            1024 * 1024,
            4 * 1024 * 1024,
        ]
        self.tau_values = tau_values or [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]

    def measure_pair(
        self,
        ckpt_a_path: Path,
        ckpt_b_path: Path,
        framework: str = "sklearn",
    ) -> Dict[str, Any]:
        """Measure compression metrics for a checkpoint pair.

        Args:
            ckpt_a_path: Path to first checkpoint.
            ckpt_b_path: Path to second checkpoint.
            framework: "sklearn", "xgboost", or "pytorch".

        Returns:
            Dictionary with measurements for each chunk_size and tau combination.
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

        return self._measure_tensors(tensors_a, tensors_b)

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

    def _measure_tensors(
        self, tensors_a: Dict[str, np.ndarray], tensors_b: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Measure compression for tensor pairs across all chunk sizes and thresholds."""
        results = {
            "pairs_measured": 0,
            "total_original_bytes": 0,
            "by_chunk_size": {},
            "restore_times_ms": [],
        }

        for chunk_size in self.chunk_sizes:
            results["by_chunk_size"][chunk_size] = {
                "measurements": [],
                "tau_stats": {tau: {"accepted": 0, "total": 0} for tau in self.tau_values},
            }

            for tensor_name, data_b in tensors_b.items():
                if tensor_name not in tensors_a:
                    continue

                data_a = tensors_a[tensor_name]

                if np.array_equal(data_a, data_b):
                    continue

                for chunk_a, chunk_b in zip(
                    chunk_tensors(data_a, chunk_size),
                    chunk_tensors(data_b, chunk_size),
                ):
                    delta = (
                        np.frombuffer(chunk_b, dtype=np.float64)
                        - np.frombuffer(chunk_a, dtype=np.float64)
                    )

                    compressed_delta = zstd.ZstdCompressor(level=3).compress(
                        delta.tobytes()
                    )
                    compressed_full = zstd.ZstdCompressor(level=3).compress(chunk_b)

                    delta_ratio = len(compressed_delta) / len(compressed_full)

                    with Timer() as timer:
                        reconstructed = np.frombuffer(chunk_a, dtype=np.float64) + delta

                    results["restore_times_ms"].append(timer.elapsed_ms)

                    for tau in self.tau_values:
                        accepted = delta_ratio < tau
                        results["by_chunk_size"][chunk_size]["tau_stats"][tau][
                            "total"
                        ] += 1
                        if accepted:
                            results["by_chunk_size"][chunk_size]["tau_stats"][tau][
                                "accepted"
                            ] += 1

                    results["by_chunk_size"][chunk_size]["measurements"].append(
                        {
                            "tensor_name": tensor_name,
                            "delta_ratio": delta_ratio,
                            "original_size": len(chunk_b),
                            "compressed_delta_size": len(compressed_delta),
                            "compressed_full_size": len(compressed_full),
                        }
                    )

                    results["pairs_measured"] += 1
                    results["total_original_bytes"] += len(chunk_b)

        return results


def classify_training_phase(step: int, total_steps: int) -> str:
    """Classify training step into phase."""
    progress = step / total_steps
    if progress < 0.2:
        return "early"
    elif progress < 0.6:
        return "mid"
    else:
        return "late"


if __name__ == "__main__":
    from pathlib import Path

    measurer = CheckpointPairMeasurer()
    checkpoint_dir = Path("benchmark/data/checkpoints")

    print("Measuring sklearn checkpoints...")
    sklearn_files = sorted(checkpoint_dir.glob("sklearn_logistic_epoch_*.pkl"))
    for i in range(len(sklearn_files) - 1):
        result = measurer.measure_pair(sklearn_files[i], sklearn_files[i + 1], "sklearn")
        print(f"sklearn pair {i}: {result['pairs_measured']} chunks measured")

    print("Measuring XGBoost checkpoints...")
    xgb_files = sorted(checkpoint_dir.glob("xgboost_classifier_round_*.pkl"))
    for i in range(len(xgb_files) - 1):
        result = measurer.measure_pair(xgb_files[i], xgb_files[i + 1], "xgboost")
        print(f"xgboost pair {i}: {result['pairs_measured']} chunks measured")

    print("Measuring PyTorch checkpoints...")
    torch_files = sorted(checkpoint_dir.glob("pytorch_resnet18_epoch_*.pt"))
    for i in range(len(torch_files) - 1):
        result = measurer.measure_pair(torch_files[i], torch_files[i + 1], "pytorch")
        print(f"pytorch pair {i}: {result['pairs_measured']} chunks measured")
