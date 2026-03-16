import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pale.adapters.pytorch import PyTorchAdapter
from pale.store import PaleStore



def _default_run_id() -> str:
    return f"pytorch-mlp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass(frozen=True)
class ExperimentConfig:
    store_root: Path = Path(".")
    run_id: str = field(default_factory=_default_run_id)
    n_samples: int = 25_000
    n_features: int = 20
    n_informative: int = 10
    validation_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    hidden_dim: int = 64
    batch_size: int = 256
    learning_rate: float = 1e-2
    n_epochs: int = 200
    restore_step: int = 15
    scheduler_milestones: tuple[int, int] = (10, 15)
    scheduler_gamma: float = 0.1


@dataclass(frozen=True)
class DatasetSplit:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    test_features: torch.Tensor
    test_targets: torch.Tensor
    input_dim: int


@dataclass(frozen=True)
class StepMetrics:
    step: int
    learning_rate: float
    train_loss: float
    valid_accuracy: float
    test_accuracy: float


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)



def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed PyTorch MLP classification experiment with Pale."
    )
    parser.add_argument("--store-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", type=str, default=_default_run_id())
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--restore-step", type=int, default=15)
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
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
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
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if not 1 <= config.restore_step <= config.n_epochs:
        raise ValueError("restore_step must be between 1 and n_epochs")
    if any(m <= 0 for m in config.scheduler_milestones):
        raise ValueError("scheduler milestones must be positive")



def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)



def prepare_data(config: ExperimentConfig) -> DatasetSplit:
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=0,
        random_state=config.random_state,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

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

    train_features = torch.from_numpy(X_train)
    valid_features = torch.from_numpy(X_valid)
    test_features = torch.from_numpy(X_test)
    train_targets = torch.from_numpy(y_train)
    valid_targets = torch.from_numpy(y_valid)
    test_targets = torch.from_numpy(y_test)

    generator = torch.Generator().manual_seed(config.random_state)
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        TensorDataset(valid_features, valid_targets),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_targets),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return DatasetSplit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        test_features=test_features,
        test_targets=test_targets,
        input_dim=X.shape[1],
    )



def build_model(config: ExperimentConfig, input_dim: int) -> MLPClassifier:
    return MLPClassifier(input_dim=input_dim, hidden_dim=config.hidden_dim)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for features, targets in loader:
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / total_examples



def predict_logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(features).detach().cpu()



def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features)
            predictions = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return correct / total



def print_config(config: ExperimentConfig) -> None:
    print("Experiment configuration")
    print("-" * 80)
    print(f"run_id           : {config.run_id}")
    print(f"store_root       : {config.store_root}")
    print(f"n_samples        : {config.n_samples}")
    print(f"n_features       : {config.n_features}")
    print(f"hidden_dim       : {config.hidden_dim}")
    print(f"batch_size       : {config.batch_size}")
    print(f"learning_rate    : {config.learning_rate}")
    print(f"n_epochs         : {config.n_epochs}")
    print(f"restore_step     : {config.restore_step}")
    print(
        f"scheduler        : milestones={config.scheduler_milestones}, gamma={config.scheduler_gamma}"
    )
    print()



def print_step_table(step_metrics: list[StepMetrics]) -> None:
    print("Checkpoint metrics")
    print("-" * 80)
    print(
        f"{'step':>4}  {'lr':>8}  {'train_loss':>12}  {'valid_acc':>10}  {'test_acc':>10}"
    )
    for metrics in step_metrics:
        print(
            f"{metrics.step:>4}  {metrics.learning_rate:>8.5f}  "
            f"{metrics.train_loss:>12.4f}  {metrics.valid_accuracy:>10.4f}  "
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
    set_random_seeds(config.random_state)
    data = prepare_data(config)
    model = build_model(config, data.input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(config.scheduler_milestones),
        gamma=config.scheduler_gamma,
    )
    step_metrics: list[StepMetrics] = []
    expected_restore_logits: torch.Tensor | None = None

    print_config(config)

    with PaleStore(root=config.store_root, run_id=config.run_id, adapter=PyTorchAdapter()) as store:
        for epoch in range(1, config.n_epochs + 1):
            train_loss = train_one_epoch(model, data.train_loader, criterion, optimizer)
            valid_accuracy = evaluate_accuracy(model, data.valid_loader)
            test_accuracy = evaluate_accuracy(model, data.test_loader)
            learning_rate = float(optimizer.param_groups[0]["lr"])

            step_metrics.append(
                StepMetrics(
                    step=epoch,
                    learning_rate=learning_rate,
                    train_loss=float(train_loss),
                    valid_accuracy=float(valid_accuracy),
                    test_accuracy=float(test_accuracy),
                )
            )
            if epoch == config.restore_step:
                expected_restore_logits = predict_logits(model, data.test_features)

            parent_step = epoch - 1 if epoch > 1 else None
            store.save(model, step=epoch, parent_step=parent_step)
            scheduler.step()

        print_step_table(step_metrics)

        restored = store.load(
            step=config.restore_step,
            original=build_model(config, data.input_dim),
        )
        restored.eval()
        restored_logits = predict_logits(restored, data.test_features)
        restored_predictions = (torch.sigmoid(restored_logits) >= 0.5).to(data.test_targets.dtype)
        restored_test_accuracy = float(
            (restored_predictions == data.test_targets).float().mean().item()
        )
        expected_accuracy = step_metrics[config.restore_step - 1].test_accuracy
        print(
            f"Restored checkpoint step={config.restore_step}  "
            f"test_accuracy={restored_test_accuracy:.4f}  "
            f"expected={expected_accuracy:.4f}"
        )
        if not np.isclose(restored_test_accuracy, expected_accuracy, atol=1e-12):
            raise RuntimeError(
                "Restored checkpoint metric mismatch; checkpoint restore verification failed"
            )
        if expected_restore_logits is None or not torch.allclose(
            restored_logits,
            expected_restore_logits,
            atol=1e-7,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "Restored checkpoint logits do not match the saved checkpoint predictions"
            )
        print()

        stats = store.stats()
        print_storage_summary(stats)



def main() -> None:
    config = parse_args()
    train_experiment(config)


if __name__ == "__main__":
    main()


