import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CHECKPOINT_DIR = Path(__file__).parent / "data" / "checkpoints" / "pytorch"


def _make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def generate(out_dir: Path, n_epochs: int, freeze_epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    X = torch.from_numpy(rng.standard_normal((1000, 64)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 10, size=1000).astype(np.int64))

    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    frozen = False
    for epoch in range(1, n_epochs + 1):
        if epoch == freeze_epoch + 1 and not frozen:
            for name, param in model.named_parameters():
                if not name.startswith("6"):
                    param.requires_grad_(False)
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            frozen = True
            print(f"  [epoch {epoch}] backbone frozen, fine-tuning head only")

        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            criterion(model(inputs), labels).backward()
            optimizer.step()

        path = out_dir / f"epoch_{epoch:06d}.pt"
        torch.save(model.state_dict(), str(path))
        phase = "frozen" if frozen else "full"
        print(f"  pytorch epoch {epoch:3d} [{phase}] → {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epoch", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    print(
        f"\nGenerating PyTorch ({args.epochs} epochs, freeze after epoch {args.freeze_epoch})..."
    )
    generate(args.out_dir, args.epochs, args.freeze_epoch)
    print("Done.")
