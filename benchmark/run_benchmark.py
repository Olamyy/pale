from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmark.analyze_results import BenchmarkAnalyzer, generate_phase_0_report
from benchmark.cross_run_similarity import CrossRunSimilarityMeasurer
from benchmark.measure_pairs import CheckpointPairMeasurer, classify_training_phase
from benchmark.train_models import (
    generate_pytorch_checkpoints,
    generate_sklearn_logistic_checkpoints,
    generate_xgboost_checkpoints,
)


def run_within_run_benchmark(checkpoint_dir: Path) -> list:
    """Run within-run checkpoint pair measurements."""
    measurer = CheckpointPairMeasurer()
    all_measurements = []

    print("\nMeasuring sklearn checkpoints...")
    sklearn_files = sorted(checkpoint_dir.glob("sklearn_logistic_epoch_*.pkl"))
    for i in range(len(sklearn_files) - 1):
        result = measurer.measure_pair(sklearn_files[i], sklearn_files[i + 1], "sklearn")
        if result["pairs_measured"] > 0:
            result["framework"] = "sklearn"
            result["pair_idx"] = i
            all_measurements.append(result)
            print(f"  sklearn pair {i}: {result['pairs_measured']} chunks")

    print("\nMeasuring XGBoost checkpoints...")
    xgb_files = sorted(checkpoint_dir.glob("xgboost_classifier_round_*.pkl"))
    for i in range(len(xgb_files) - 1):
        result = measurer.measure_pair(xgb_files[i], xgb_files[i + 1], "xgboost")
        if result["pairs_measured"] > 0:
            result["framework"] = "xgboost"
            result["pair_idx"] = i
            all_measurements.append(result)
            print(f"  xgboost pair {i}: {result['pairs_measured']} chunks")

    print("\nMeasuring PyTorch checkpoints...")
    torch_files = sorted(checkpoint_dir.glob("pytorch_resnet18_epoch_*.pt"))
    for i in range(len(torch_files) - 1):
        result = measurer.measure_pair(torch_files[i], torch_files[i + 1], "pytorch")
        if result["pairs_measured"] > 0:
            result["framework"] = "pytorch"
            result["pair_idx"] = i
            all_measurements.append(result)
            print(f"  pytorch pair {i}: {result['pairs_measured']} chunks")

    return all_measurements


def run_cross_run_benchmark(checkpoint_dir: Path) -> dict:
    """Run cross-run similarity measurements."""
    measurer = CrossRunSimilarityMeasurer()
    results = {}

    sklearn_files = sorted(checkpoint_dir.glob("sklearn_logistic_epoch_*.pkl"))
    if len(sklearn_files) >= 2:
        result = measurer.measure_pair(sklearn_files[0], sklearn_files[-1], "sklearn")
        results["sklearn_endpoints"] = result["overall_similarity"]
        print(f"sklearn (early vs late): {result['overall_similarity']:.1%}")

    return results


def main():
    """Run complete Phase 0 benchmark."""
    checkpoint_dir = Path("benchmark/data/checkpoints")
    results_dir = Path("benchmark/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 0: CHECKPOINT PAIR BENCHMARKING")
    print("=" * 60)

    print("\n[1/3] Generating checkpoint sequences...")
    generate_sklearn_logistic_checkpoints(checkpoint_dir)
    generate_xgboost_checkpoints(checkpoint_dir)
    generate_pytorch_checkpoints(checkpoint_dir)

    print("\n[2/3] Measuring within-run checkpoint pairs...")
    within_run_results = run_within_run_benchmark(checkpoint_dir)

    print("\n[3/3] Measuring cross-run similarity...")
    cross_run_results = run_cross_run_benchmark(checkpoint_dir)

    print("\nGenerating Phase 0 report...")
    report_path = results_dir / "phase_0_report.md"
    generate_phase_0_report(within_run_results, cross_run_results, report_path)

    print("\n" + "=" * 60)
    print("Phase 0 benchmark complete!")
    print(f"Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
