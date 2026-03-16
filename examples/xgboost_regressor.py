import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from pale.adapters.xgboost import XGBoostAdapter
from pale.store import PaleStore


def _default_run_id() -> str:
    return f"xgboost-regressor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass(frozen=True)
class ExperimentConfig:
    store_root: Path = Path(".")
    run_id: str = _default_run_id()
    n_samples: int = 15_000
    n_features: int = 100
    noise: float = 0.5
    validation_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int = 10
    learning_rate: float = 0.1
    rounds_per_step: int = 50
    n_steps: int = 5
    restore_step: int = 3
    max_workers: int = 8


@dataclass(frozen=True)
class DatasetSplit:
    dtrain: xgb.DMatrix
    dvalid: xgb.DMatrix
    dtest: xgb.DMatrix
    y_valid: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class StepMetrics:
    step: int
    total_rounds: int
    valid_rmse: float
    test_rmse: float


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed XGBoost regression experiment with Pale."
    )
    parser.add_argument("--store-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", type=str, default=_default_run_id())
    parser.add_argument("--n-samples", type=int, default=15_000)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--rounds-per-step", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--restore-step", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    return ExperimentConfig(
        store_root=args.store_root,
        run_id=args.run_id,
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        rounds_per_step=args.rounds_per_step,
        n_steps=args.n_steps,
        restore_step=args.restore_step,
        max_workers=args.max_workers,
    )


def validate_config(config: ExperimentConfig) -> None:
    if not 0.0 < config.validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1")
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if config.validation_size + config.test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1")
    if config.rounds_per_step <= 0:
        raise ValueError("rounds_per_step must be positive")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not 1 <= config.restore_step <= config.n_steps:
        raise ValueError("restore_step must be between 1 and n_steps")


def prepare_data(config: ExperimentConfig) -> DatasetSplit:
    X, y = make_regression(
        n_samples=config.n_samples,
        n_features=config.n_features,
        noise=config.noise,
        random_state=config.random_state,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    holdout_size = config.validation_size + config.test_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=config.random_state,
    )
    test_fraction_of_holdout = config.test_size / holdout_size
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_of_holdout,
        random_state=config.random_state,
    )

    return DatasetSplit(
        dtrain=xgb.DMatrix(X_train, label=y_train),
        dvalid=xgb.DMatrix(X_valid, label=y_valid),
        dtest=xgb.DMatrix(X_test, label=y_test),
        y_valid=y_valid,
        y_test=y_test,
    )


def build_params(config: ExperimentConfig) -> dict[str, float | int | str]:
    return {
        "max_depth": config.max_depth,
        "objective": "reg:squarederror",
        "learning_rate": config.learning_rate,
        "seed": config.random_state,
        "verbosity": 0,
    }


def rmse(booster: xgb.Booster, dmatrix: xgb.DMatrix, targets: np.ndarray) -> float:
    predictions = booster.predict(dmatrix)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def print_config(config: ExperimentConfig) -> None:
    print("Experiment configuration")
    print("-" * 80)
    print(f"run_id           : {config.run_id}")
    print(f"store_root       : {config.store_root}")
    print(f"n_samples        : {config.n_samples}")
    print(f"n_features       : {config.n_features}")
    print(f"rounds_per_step  : {config.rounds_per_step}")
    print(f"n_steps          : {config.n_steps}")
    print(f"restore_step     : {config.restore_step}")
    print()


def print_step_table(step_metrics: list[StepMetrics]) -> None:
    print("Checkpoint metrics")
    print("-" * 80)
    print(f"{'step':>4}  {'rounds':>6}  {'valid_rmse':>12}  {'test_rmse':>12}")
    for metrics in step_metrics:
        print(
            f"{metrics.step:>4}  {metrics.total_rounds:>6}  "
            f"{metrics.valid_rmse:>12.4f}  {metrics.test_rmse:>12.4f}"
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
    params = build_params(config)
    step_metrics: list[StepMetrics] = []

    print_config(config)

    with PaleStore(
        root=config.store_root,
        run_id=config.run_id,
        adapter=XGBoostAdapter(),
        max_workers=config.max_workers,
    ) as store:
        booster: xgb.Booster | None = None
        for step in range(1, config.n_steps + 1):
            booster = xgb.train(
                params,
                data.dtrain,
                num_boost_round=config.rounds_per_step,
                evals=[(data.dvalid, "valid")],
                xgb_model=booster,
                verbose_eval=False,
            )

            metrics = StepMetrics(
                step=step,
                total_rounds=step * config.rounds_per_step,
                valid_rmse=rmse(booster, data.dvalid, data.y_valid),
                test_rmse=rmse(booster, data.dtest, data.y_test),
            )
            step_metrics.append(metrics)
            parent_step = step - 1 if step > 1 else None
            store.save(booster, step=step, parent_step=parent_step)

        print_step_table(step_metrics)

        restored = store.load(step=config.restore_step)
        restored_test_rmse = rmse(restored, data.dtest, data.y_test)
        expected = step_metrics[config.restore_step - 1].test_rmse
        print(
            f"Restored checkpoint step={config.restore_step}  "
            f"test_rmse={restored_test_rmse:.4f}  "
            f"expected={expected:.4f}"
        )
        if not np.isclose(restored_test_rmse, expected, atol=1e-9):
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
