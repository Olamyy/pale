"""Generate checkpoint sequences for sklearn, XGBoost, and PyTorch models.

Each generator produces a monotonically evolving checkpoint sequence that
captures early, mid, and late training phases. Checkpoints are named with
the actual training step count (not 0-indexed loop counters) so the measurer
can infer phase from the filename alone.
"""

import json
import os
import pickle
import platform
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def generate_sklearn_checkpoints(
        output_dir: Path,
        n_checkpoints: int = 20,
        trees_per_checkpoint: int = 10,
) -> None:
    """Train a GradientBoostingClassifier on Wine and checkpoint incrementally.

    Uses warm_start=True and increments n_estimators each step so the model
    genuinely evolves — each checkpoint has more trees than the last.

    Produces checkpoints named sklearn_gb_step_{total_trees:06d}.pkl
    so the measurer can read step count directly from the filename.

    Args:
        output_dir: Base checkpoint directory (sklearn/ subdirectory will be created).
        n_checkpoints: How many checkpoints to save.
        trees_per_checkpoint: Trees added per checkpoint step.
            Total trees = n_checkpoints * trees_per_checkpoint.
    """
    sklearn_dir = output_dir / "sklearn"
    sklearn_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=trees_per_checkpoint,
        warm_start=True,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    total_trees = 0
    for step in tqdm(range(1, n_checkpoints + 1), desc="sklearn"):
        total_trees = step * trees_per_checkpoint
        model.set_params(n_estimators=total_trees)
        model.fit(X, y)

        train_acc = model.score(X, y)

        ckpt_path = sklearn_dir / f"sklearn_gb_step_{total_trees:06d}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(model, f)

        meta = {
            "step": total_trees,
            "n_estimators": total_trees,
            "train_accuracy": train_acc,
            "framework": "sklearn",
        }
        with open(ckpt_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    written = sorted(sklearn_dir.glob("sklearn_gb_step_*.pkl"))
    print(f"sklearn: wrote {len(written)} checkpoints → {sklearn_dir}")
    print(f"  total trees at final checkpoint: {total_trees}")


def generate_xgboost_checkpoints(
        output_dir: Path,
        total_rounds: int = 1000,
        checkpoint_interval: int = 100,
) -> None:
    """Train an XGBoost regressor on California Housing and checkpoint incrementally.

    Uses xgb.train with xgb_model= for true warm-start continuation — each
    checkpoint genuinely extends the previous booster rather than retraining.

    Produces checkpoints named xgboost_rounds_{round:06d}.xgb.

    Args:
        output_dir: Base checkpoint directory (xgboost/ subdirectory will be created).
        total_rounds: Total boosting rounds to train.
        checkpoint_interval: Rounds between checkpoints.
            Also the number of rounds added per xgb.train call.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed — skipping XGBoost checkpoints")
        return

    xgboost_dir = output_dir / "xgboost"
    xgboost_dir.mkdir(parents=True, exist_ok=True)

    X, y = fetch_california_housing(return_X_y=True)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
    }

    booster: Optional[xgb.Booster] = None
    evals_result: dict = {}

    rounds_done = 0
    checkpoints = range(checkpoint_interval, total_rounds + 1, checkpoint_interval)

    for target_round in tqdm(checkpoints, desc="xgboost"):
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=checkpoint_interval,
            xgb_model=booster,
            evals=[(dtrain, "train")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        rounds_done += checkpoint_interval

        ckpt_path = xgboost_dir / f"xgboost_rounds_{rounds_done:06d}.xgb"
        booster.save_model(str(ckpt_path))

        last_rmse = evals_result["train"]["rmse"][-1]

        meta = {
            "step": rounds_done,
            "total_rounds": total_rounds,
            "train_rmse": last_rmse,
            "framework": "xgboost",
        }
        with open(ckpt_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    written = sorted(xgboost_dir.glob("xgboost_rounds_*.xgb"))
    print(f"xgboost: wrote {len(written)} checkpoints → {xgboost_dir}")
    print(f"  final train RMSE: {last_rmse:.4f}")


def generate_pytorch_checkpoints(
        output_dir: Path,
        n_epochs: int = 20,
        checkpoint_interval: int = 1,
) -> None:
    """Train ResNet-18 on CIFAR-10 and checkpoint at each epoch.

    Uses a MultiStepLR scheduler with decay at epochs 10 and 15, which
    produces three distinct training phases (high-LR, mid-LR, low-LR)
    within a 20-epoch run — important for observing delta compressibility
    across phases.

    Checkpoints are named pytorch_resnet18_epoch_{epoch:06d}.pt using
    1-indexed epoch numbers (epoch 1 = after first training epoch).

    Args:
        output_dir: Base checkpoint directory (pytorch/ subdirectory will be created).
        n_epochs: Total epochs to train.
        checkpoint_interval: Save a checkpoint every N epochs.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        from torchvision.datasets import CIFAR10
        from torchvision.models import resnet18
    except ImportError:
        print("torch/torchvision not installed — skipping PyTorch checkpoints")
        return

    pytorch_dir = output_dir / "pytorch"
    pytorch_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pytorch] using device={device}")

    num_workers = int(os.environ.get("PALE_DATALOADER_WORKERS", 0))
    if num_workers > 0 and platform.system() in ("Darwin", "Windows"):
        print(
            f"[pytorch] PALE_DATALOADER_WORKERS={num_workers} requested but "
            f"platform is {platform.system()} — defaulting to 0 to avoid "
            "multiprocessing deadlocks. Set explicitly to override."
        )
        num_workers = 0

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = resnet18(num_classes=10).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 15], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, n_epochs + 1), desc="pytorch_resnet18"):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        if epoch % checkpoint_interval == 0:
            train_loss = epoch_loss / total
            train_acc = correct / total
            current_lr = scheduler.get_last_lr()[0]

            ckpt_path = pytorch_dir / f"pytorch_resnet18_epoch_{epoch:06d}.pt"
            torch.save(model.state_dict(), ckpt_path)

            meta = {
                "step": epoch,
                "total_epochs": n_epochs,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "learning_rate": current_lr,
                "framework": "pytorch",
            }
            with open(ckpt_path.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)

    written = sorted(pytorch_dir.glob("pytorch_resnet18_epoch_*.pt"))
    print(f"pytorch: wrote {len(written)} checkpoints → {pytorch_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate checkpoint sequences for Strata Phase 0 benchmarks"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["sklearn", "xgboost", "pytorch", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("benchmark/data/checkpoints"),
    )
    parser.add_argument("--sklearn-checkpoints", type=int, default=20)
    parser.add_argument("--sklearn-trees-per-step", type=int, default=10)
    parser.add_argument("--xgboost-rounds", type=int, default=1000)
    parser.add_argument("--xgboost-interval", type=int, default=100)
    parser.add_argument("--pytorch-epochs", type=int, default=20)
    parser.add_argument("--pytorch-interval", type=int, default=1)

    args = parser.parse_args()

    frameworks = (
        ["sklearn", "xgboost", "pytorch"]
        if "all" in args.frameworks
        else args.frameworks
    )

    if "sklearn" in frameworks:
        generate_sklearn_checkpoints(
            args.checkpoint_dir,
            n_checkpoints=args.sklearn_checkpoints,
            trees_per_checkpoint=args.sklearn_trees_per_step,
        )

    if "xgboost" in frameworks:
        generate_xgboost_checkpoints(
            args.checkpoint_dir,
            total_rounds=args.xgboost_rounds,
            checkpoint_interval=args.xgboost_interval,
        )

    if "pytorch" in frameworks:
        generate_pytorch_checkpoints(
            args.checkpoint_dir,
            n_epochs=args.pytorch_epochs,
            checkpoint_interval=args.pytorch_interval,
        )

    print(f"\nAll checkpoints saved to {args.checkpoint_dir}")