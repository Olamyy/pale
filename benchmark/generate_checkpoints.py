import argparse
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

CHECKPOINT_DIR = Path(__file__).parent / "data" / "checkpoints"

_RNG = np.random.default_rng(42)
_X, _Y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, random_state=42
)
_X32 = _X.astype(np.float32)
_Y32 = _Y.astype(np.float32)


def generate_sklearn_warmstart(
    out_dir: Path, n_checkpoints: int, trees_per_step: int
) -> None:
    """Train GBM with warm_start=True, saving a checkpoint every trees_per_step trees.

    Each checkpoint adds trees_per_step new trees on top of the previous ones.
    Frozen trees from earlier steps are byte-identical across checkpoints —
    this is the case where tensor-level chunking should win over file-level hashing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = GradientBoostingClassifier(
        n_estimators=trees_per_step,
        warm_start=True,
        random_state=0,
        max_depth=4,
    )
    for i in range(n_checkpoints):
        total_trees = (i + 1) * trees_per_step
        model.set_params(n_estimators=total_trees)
        model.fit(_X, _Y)
        path = out_dir / f"step_{total_trees:06d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  sklearn step {total_trees:4d} → {path.name}")


def generate_xgboost_warmstart(
    out_dir: Path, n_checkpoints: int, rounds_per_step: int
) -> None:
    """Train XGBoost with warm-start via xgb_model, saving every rounds_per_step rounds.

    Each checkpoint adds rounds_per_step new boosting rounds. Earlier rounds are
    frozen — their split values are encoded in the binary model bytes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dtrain = xgb.DMatrix(_X32, label=_Y32)
    params = {"max_depth": 4, "objective": "binary:logistic", "seed": 0, "verbosity": 0}
    booster = None

    for i in range(n_checkpoints):
        total_rounds = (i + 1) * rounds_per_step
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds_per_step,
            xgb_model=booster,
            verbose_eval=False,
        )
        path = out_dir / f"step_{total_rounds:06d}.ubj"
        booster.save_model(str(path))
        print(f"  xgboost step {total_rounds:4d} → {path.name}")


def generate_pytorch_finetune(out_dir: Path, n_epochs: int) -> None:
    """Train ResNet-18 on CIFAR-10, saving a checkpoint every epoch.

    Two phases:
      Epochs 1 to n_epochs//2:   full model training (all weights changing)
      Epochs n_epochs//2+1 to N: freeze backbone, train head only

    The frozen backbone phase is where tensor-level chunking should win.
    """
    try:
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("  Skipping PyTorch — torch/torchvision not installed")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=str(Path(__file__).parent / "data" / "cifar10"),
        train=True,
        download=True,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, num_workers=0
    )

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    freeze_epoch = n_epochs // 2
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        if epoch == freeze_epoch + 1:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
            )
            print(f"  Epoch {epoch}: freezing backbone, fine-tuning head only")

        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(inputs), labels).backward()
            optimizer.step()

        path = out_dir / f"epoch_{epoch:06d}.pt"
        torch.save(model.state_dict(), path)
        print(f"  pytorch epoch {epoch:3d} → {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate checkpoint sequences for dedup benchmark"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["sklearn", "xgboost", "pytorch"],
        default=["sklearn"],
    )
    parser.add_argument("--sklearn-checkpoints", type=int, default=20)
    parser.add_argument("--sklearn-trees-per-step", type=int, default=10)
    parser.add_argument("--xgboost-checkpoints", type=int, default=20)
    parser.add_argument("--xgboost-rounds-per-step", type=int, default=10)
    parser.add_argument("--pytorch-epochs", type=int, default=20)
    args = parser.parse_args()

    if "sklearn" in args.frameworks:
        print(
            f"\nGenerating sklearn ({args.sklearn_checkpoints} checkpoints, {args.sklearn_trees_per_step} trees/step)..."
        )
        generate_sklearn_warmstart(
            CHECKPOINT_DIR / "sklearn",
            n_checkpoints=args.sklearn_checkpoints,
            trees_per_step=args.sklearn_trees_per_step,
        )

    if "xgboost" in args.frameworks:
        print(
            f"\nGenerating XGBoost ({args.xgboost_checkpoints} checkpoints, {args.xgboost_rounds_per_step} rounds/step)..."
        )
        generate_xgboost_warmstart(
            CHECKPOINT_DIR / "xgboost",
            n_checkpoints=args.xgboost_checkpoints,
            rounds_per_step=args.xgboost_rounds_per_step,
        )

    if "pytorch" in args.frameworks:
        print(f"\nGenerating PyTorch ({args.pytorch_epochs} epochs)...")
        generate_pytorch_finetune(
            CHECKPOINT_DIR / "pytorch",
            n_epochs=args.pytorch_epochs,
        )

    print("\nDone.")
