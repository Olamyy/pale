"""Generate checkpoint sequences for sklearn, XGBoost, and PyTorch models."""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from benchmark.utils import save_sklearn_checkpoint


def generate_sklearn_logistic_checkpoints(
    output_dir: Path, n_epochs: int = 100, checkpoint_interval: int = 10
) -> None:
    """Generate checkpoint sequence for sklearn LogisticRegression.

    Trains on MNIST digit classification with warm_start to enable
    progressive training and checkpointing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_digits(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1, warm_start=True, random_state=42, n_jobs=1
    )

    for epoch in tqdm(range(0, n_epochs, checkpoint_interval), desc="sklearn_logistic"):
        model.fit(X, y)

        ckpt_path = output_dir / f"sklearn_logistic_epoch_{epoch:06d}.pkl"
        save_sklearn_checkpoint(model, ckpt_path)


def generate_xgboost_checkpoints(
    output_dir: Path, n_rounds: int = 500, checkpoint_interval: int = 50
) -> None:
    """Generate checkpoint sequence for XGBoost classifier.

    Trains on Titanic dataset with progressive boosting rounds and saves
    intermediate models.
    """
    try:
        import xgboost as xgb
        from sklearn.datasets import make_classification

        output_dir.mkdir(parents=True, exist_ok=True)

        X, y = make_classification(n_samples=500, n_features=20, random_state=42)

        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            tree_method="hist",
        )
        model.fit(X, y)

        for round_num in tqdm(
            range(checkpoint_interval, n_rounds + 1, checkpoint_interval),
            desc="xgboost",
        ):
            model.set_params(n_estimators=round_num)
            model.fit(X, y)

            ckpt_path = output_dir / f"xgboost_classifier_round_{round_num:06d}.pkl"
            save_sklearn_checkpoint(model, ckpt_path)

    except ImportError:
        print("XGBoost not installed, skipping XGBoost checkpoints")


def generate_pytorch_checkpoints(
    output_dir: Path, n_epochs: int = 100, checkpoint_interval: int = 10
) -> None:
    """Generate checkpoint sequence for PyTorch ResNet-18.

    Trains a small ResNet-18 on CIFAR-10 and saves state_dicts at regular
    intervals.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision.transforms as transforms
        from torchvision.datasets import CIFAR10
        from torchvision.models import resnet18

        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=0
        )

        model = resnet18(num_classes=10)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in tqdm(range(n_epochs), desc="pytorch_resnet18"):
            model.train()
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % checkpoint_interval == 0:
                from benchmark.utils import save_pytorch_checkpoint

                ckpt_path = output_dir / f"pytorch_resnet18_epoch_{epoch:06d}.pt"
                save_pytorch_checkpoint(model, ckpt_path)

    except ImportError:
        print("PyTorch not installed, skipping PyTorch checkpoints")


if __name__ == "__main__":
    checkpoint_dir = Path("benchmark/data/checkpoints")

    generate_sklearn_logistic_checkpoints(checkpoint_dir)
    generate_xgboost_checkpoints(checkpoint_dir)
    generate_pytorch_checkpoints(checkpoint_dir)

    print(f"Checkpoints saved to {checkpoint_dir}")
