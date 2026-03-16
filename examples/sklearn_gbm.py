import argparse
import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pale.adapters.sklearn import SklearnAdapter
from pale.store import PaleStore


def _default_run_id() -> str:
    return f"sklearn-gbm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass(frozen=True)
class ExperimentConfig:
    store_root: Path = Path(".")
    run_id: str = field(default_factory=_default_run_id)
    n_samples: int = 5_000
    n_features: int = 20
    n_informative: int = 10
    validation_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    learning_rate: float = 0.1
    max_depth: int = 3
    trees_per_step: int = 50
    n_steps: int = 15
    restore_step: int = 10


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class StepMetrics:
    step: int
    total_trees: int
    train_accuracy: float
    valid_accuracy: float
    test_accuracy: float


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed sklearn GBM experiment with Pale."
    )
    parser.add_argument("--store-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", type=str, default=_default_run_id())
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--trees-per-step", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--restore-step", type=int, default=10)
    args = parser.parse_args()
    return ExperimentConfig(
        store_root=args.store_root,
        run_id=args.run_id,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        trees_per_step=args.trees_per_step,
        n_steps=args.n_steps,
        restore_step=args.restore_step,
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.n_informative <= 0 or config.n_informative > config.n_features:
        raise ValueError("n_informative must be between 1 and n_features")
    if not 0.0 < config.validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1")
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if config.validation_size + config.test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1")
    if config.trees_per_step <= 0:
        raise ValueError("trees_per_step must be positive")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not 1 <= config.restore_step <= config.n_steps:
        raise ValueError("restore_step must be between 1 and n_steps")


def prepare_data(config: ExperimentConfig) -> DatasetSplit:
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=0,
        random_state=config.random_state,
    )

    holdout_size = config.validation_size + config.test_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        stratify=y,
        random_state=config.random_state,
    )
    test_fraction_of_holdout = config.test_size / holdout_size
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_of_holdout,
        stratify=y_holdout,
        random_state=config.random_state,
    )

    return DatasetSplit(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )


def build_model(config: ExperimentConfig) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=config.trees_per_step,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        warm_start=True,
        random_state=config.random_state,
    )


def evaluate_model(
    model: GradientBoostingClassifier, data: DatasetSplit
) -> tuple[float, float, float]:
    train_accuracy = accuracy_score(data.y_train, model.predict(data.X_train))
    valid_accuracy = accuracy_score(data.y_valid, model.predict(data.X_valid))
    test_accuracy = accuracy_score(data.y_test, model.predict(data.X_test))
    return float(train_accuracy), float(valid_accuracy), float(test_accuracy)


def print_config(config: ExperimentConfig) -> None:
    print("Experiment configuration")
    print("-" * 80)
    print(f"run_id           : {config.run_id}")
    print(f"store_root       : {config.store_root}")
    print(f"n_samples        : {config.n_samples}")
    print(f"n_features       : {config.n_features}")
    print(f"trees_per_step   : {config.trees_per_step}")
    print(f"n_steps          : {config.n_steps}")
    print(f"restore_step     : {config.restore_step}")
    print()


def print_step_table(step_metrics: list[StepMetrics]) -> None:
    print("Checkpoint metrics")
    print("-" * 80)
    print(
        f"{'step':>4}  {'trees':>6}  {'train_acc':>10}  {'valid_acc':>10}  {'test_acc':>10}"
    )
    for metrics in step_metrics:
        print(
            f"{metrics.step:>4}  {metrics.total_trees:>6}  "
            f"{metrics.train_accuracy:>10.4f}  {metrics.valid_accuracy:>10.4f}  "
            f"{metrics.test_accuracy:>10.4f}"
        )
    print()


def print_storage_summary(stats: dict[str, object]) -> None:
    print("Pale storage summary")
    print("-" * 80)
    print(f"checkpoints      : {stats['checkpoints']}")
    print(f"total_chunks     : {stats['total_chunks']}")
    print(f"unique_chunks    : {stats['unique_chunks']}")
    print(f"dedup_ratio      : {stats['dedup_ratio']}")
    print(f"total_bytes      : {stats['total_bytes']}")
    print()


def train_experiment(config: ExperimentConfig) -> None:
    validate_config(config)
    data = prepare_data(config)
    step_metrics: list[StepMetrics] = []
    step_templates: dict[int, GradientBoostingClassifier] = {}

    print_config(config)

    model = build_model(config)
    with PaleStore(
        root=config.store_root, run_id=config.run_id, adapter=SklearnAdapter()
    ) as store:
        for step in range(1, config.n_steps + 1):
            total_trees = step * config.trees_per_step
            model.set_params(**{"n_estimators": total_trees})
            model.fit(data.X_train, data.y_train)

            train_accuracy, valid_accuracy, test_accuracy = evaluate_model(model, data)
            step_metrics.append(
                StepMetrics(
                    step=step,
                    total_trees=total_trees,
                    train_accuracy=train_accuracy,
                    valid_accuracy=valid_accuracy,
                    test_accuracy=test_accuracy,
                )
            )
            step_templates[step] = copy.deepcopy(model)
            parent_step = step - 1 if step > 1 else None
            store.save(model, step=step, parent_step=parent_step)

        print_step_table(step_metrics)

        restored = store.load(
            step=config.restore_step,
            original=step_templates[config.restore_step],
        )
        restored_test_accuracy = float(
            accuracy_score(data.y_test, restored.predict(data.X_test))
        )
        expected = step_metrics[config.restore_step - 1].test_accuracy
        print(
            f"Restored checkpoint step={config.restore_step}  "
            f"test_accuracy={restored_test_accuracy:.4f}  "
            f"expected={expected:.4f}"
        )
        if not np.isclose(restored_test_accuracy, expected, atol=1e-12):
            raise RuntimeError(
                "Restored checkpoint metric mismatch; checkpoint restore verification failed"
            )
        print()

        stats = store.stats()
        print_storage_summary(stats)


def main() -> None:
    config = parse_args()
    train_experiment(config)


if __name__ == "__main__":
    main()
