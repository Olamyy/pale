import argparse
import hashlib
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zstandard as zstd

from pale.adapters.sklearn import SklearnAdapter
from pale.adapters.xgboost import XGBoostAdapter
from pale.hashing import hash_chunk
from pale.chunking import chunk_bytes
from pale.serialization import tensor_to_bytes

CHECKPOINT_DIR = Path(__file__).parent / "data" / "checkpoints"
_cctx = zstd.ZstdCompressor(level=3)


# ---------------------------------------------------------------------------
# Measurement 1: No-op fast path effectiveness
# ---------------------------------------------------------------------------


def measure_noop_fastpath(
    checkpoint_paths: List[Path],
    adapter,
) -> Dict:
    """Per-tensor analysis of byte-identical fractions and compressed sizes across consecutive steps."""
    sequences = [_extract(path, adapter) for path in checkpoint_paths]
    n = len(sequences)
    if n < 2:
        return {}

    tensor_names = sorted(sequences[0].keys())
    stats: Dict[str, Dict] = {
        name: {"identical": 0, "changed": 0, "compressed_bytes": []}
        for name in tensor_names
    }

    for prev, curr in zip(sequences, sequences[1:]):
        for name in tensor_names:
            if name not in curr:
                continue
            raw_prev, _, _ = tensor_to_bytes(prev[name])
            raw_curr, _, _ = tensor_to_bytes(curr[name])
            if hash_chunk(raw_prev) == hash_chunk(raw_curr):
                stats[name]["identical"] += 1
            else:
                stats[name]["changed"] += 1
                stats[name]["compressed_bytes"].append(len(_cctx.compress(raw_curr)))

    # Collect compressed size for tensors that never changed (sample first occurrence)
    first = sequences[0]
    for name in tensor_names:
        if not stats[name]["compressed_bytes"]:
            raw, _, _ = tensor_to_bytes(first[name])
            stats[name]["compressed_bytes"].append(len(_cctx.compress(raw)))

    total_comparisons = n - 1
    result = {}
    for name, s in stats.items():
        identical = s["identical"]
        avg_compressed = (
            sum(s["compressed_bytes"]) / len(s["compressed_bytes"])
            if s["compressed_bytes"]
            else 0
        )
        result[name] = {
            "identical": identical,
            "changed": s["changed"],
            "identical_pct": identical / total_comparisons * 100,
            "avg_compressed_bytes": avg_compressed,
        }

    all_identical = sum(s["identical"] for s in stats.values())
    all_total = total_comparisons * len(tensor_names)
    result["__summary__"] = {
        "total_tensor_steps": all_total,
        "identical": all_identical,
        "changed": all_total - all_identical,
        "identical_pct": all_identical / all_total * 100 if all_total else 0.0,
    }
    return result


# ---------------------------------------------------------------------------
# Measurement 2: Chunk-level dedup for changed tensors
# ---------------------------------------------------------------------------


def measure_chunk_dedup(
    checkpoint_paths: List[Path],
    adapter,
    chunk_size: int,
) -> Dict:
    """For tensors that change between steps, measure chunk-level reuse.

    Bypasses the no-op fast path entirely. Chunks every tensor at every step
    and checks what fraction of chunks from step N already appeared in step N-1.
    This isolates the CAS chunking contribution independent of full-tensor identity.
    """
    sequences = [_extract(path, adapter) for path in checkpoint_paths]
    n = len(sequences)
    if n < 2:
        return {}

    tensor_names = sorted(sequences[0].keys())
    changed_only: Dict[str, Dict] = defaultdict(
        lambda: {"reused_chunks": 0, "total_chunks": 0}
    )

    for prev, curr in zip(sequences, sequences[1:]):
        for name in tensor_names:
            if name not in curr:
                continue
            raw_prev, _, _ = tensor_to_bytes(prev[name])
            raw_curr, _, _ = tensor_to_bytes(curr[name])
            if hash_chunk(raw_prev) == hash_chunk(raw_curr):
                continue  # skip identical tensors — not relevant to this measurement

            chunks_prev = set(hash_chunk(c) for c in chunk_bytes(raw_prev, chunk_size))
            chunks_curr = [hash_chunk(c) for c in chunk_bytes(raw_curr, chunk_size)]

            for ch in chunks_curr:
                changed_only[name]["total_chunks"] += 1
                if ch in chunks_prev:
                    changed_only[name]["reused_chunks"] += 1

    result = {}
    total_reused = 0
    total_chunks = 0
    for name, s in changed_only.items():
        tc = s["total_chunks"]
        rc = s["reused_chunks"]
        total_reused += rc
        total_chunks += tc
        result[name] = {
            "total_chunks": tc,
            "reused_chunks": rc,
            "reuse_pct": rc / tc * 100 if tc else 0.0,
        }

    result["__summary__"] = {
        "total_chunks_in_changed_tensors": total_chunks,
        "reused_chunks": total_reused,
        "reuse_pct": total_reused / total_chunks * 100 if total_chunks else 0.0,
    }
    return result


# ---------------------------------------------------------------------------
# Measurement 3: Genuine cross-run dedup
# ---------------------------------------------------------------------------


def measure_crossrun_dedup(
    paths_run1: List[Path],
    paths_run2: List[Path],
    adapter,
    chunk_size: int,
) -> Dict:
    """Two genuinely different checkpoint sequences sharing a base model.

    Measures: of all unique chunks in run 2, what fraction already exist in
    run 1's CAS? This answers: if run 1 was already stored, how many new writes
    does run 2 require?

    Both runs are deduplicated within themselves (only unique hashes counted)
    so the comparison is apples-to-apples: unique chunks in run 1 vs unique
    chunks in run 2 that are new.
    """
    hashes_run1: set = set()
    for path in paths_run1:
        tensors = _extract(path, adapter)
        for arr in tensors.values():
            raw, _, _ = tensor_to_bytes(arr)
            for chunk in chunk_bytes(raw, chunk_size):
                hashes_run1.add(hash_chunk(chunk))

    hashes_run2: set = set()
    for path in paths_run2:
        tensors = _extract(path, adapter)
        for arr in tensors.values():
            raw, _, _ = tensor_to_bytes(arr)
            for chunk in chunk_bytes(raw, chunk_size):
                hashes_run2.add(hash_chunk(chunk))

    shared = len(hashes_run1 & hashes_run2)
    return {
        "run1_unique_chunks": len(hashes_run1),
        "run2_unique_chunks": len(hashes_run2),
        "run2_shared_with_run1": shared,
        "run2_new_chunks": len(hashes_run2) - shared,
        "cross_run_reuse_pct": shared / len(hashes_run2) * 100 if hashes_run2 else 0.0,
    }


# ---------------------------------------------------------------------------
# DVC baseline (unchanged)
# ---------------------------------------------------------------------------


def dvc_simulated_bytes(checkpoint_paths: List[Path]) -> int:
    seen: Dict[str, int] = {}
    total = 0
    for path in checkpoint_paths:
        data = path.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        if h not in seen:
            seen[h] = len(_cctx.compress(data))
            total += seen[h]
    return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract(path: Path, adapter) -> Dict[str, np.ndarray]:
    if isinstance(adapter, XGBoostAdapter):
        import xgboost as xgb

        b = xgb.Booster()
        b.load_model(str(path))
        return adapter.extract(b)
    if path.suffix == ".pt":
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=True)
        return adapter.extract(state)
    with open(path, "rb") as f:
        return adapter.extract(pickle.load(f))


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _fmt_pct(v: float, width: int = 6) -> str:
    return f"{v:>{width}.1f}%"


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_noop_report(name: str, stats: Dict) -> None:
    summary = stats.pop("__summary__", {})
    print(f"\n[{name}] No-op fast path — tensor identity across consecutive steps")
    print(
        f"  Overall: {summary.get('identical_pct', 0):.1f}% of tensor-steps are byte-identical"
    )
    print(
        f"  {'Tensor':<40} {'Identical':>10} {'Changed':>8} {'Identical%':>11} {'AvgCompressed':>14}"
    )
    print(f"  {'-' * 40} {'-' * 10} {'-' * 8} {'-' * 11} {'-' * 14}")
    for tensor_name, s in sorted(stats.items()):
        compressed_label = _fmt_bytes(int(s.get("avg_compressed_bytes", 0)))
        print(
            f"  {tensor_name:<40} {s['identical']:>10} {s['changed']:>8} "
            f"{_fmt_pct(s['identical_pct'], 10)} {compressed_label:>14}"
        )


def print_chunk_dedup_report(name: str, stats: Dict, chunk_size: int) -> None:
    label = (
        f"{chunk_size // 1024}KB"
        if chunk_size < 1024 * 1024
        else f"{chunk_size // (1024 * 1024)}MB"
    )
    summary = stats.pop("__summary__", {})
    print(f"\n[{name}] Chunk-level reuse in changed tensors (chunk={label})")
    print(
        f"  Overall: {summary.get('reuse_pct', 0):.1f}% of chunks in changed tensors already existed"
    )
    if not stats:
        print("  (no changed tensors found)")
        return
    print(f"  {'Tensor':<40} {'Total chunks':>13} {'Reused':>8} {'Reuse%':>8}")
    print(f"  {'-' * 40} {'-' * 13} {'-' * 8} {'-' * 8}")
    for tensor_name, s in sorted(stats.items()):
        print(
            f"  {tensor_name:<40} {s['total_chunks']:>13} {s['reused_chunks']:>8} "
            f"{_fmt_pct(s['reuse_pct'], 7)}"
        )


def print_crossrun_report(name: str, stats: Dict) -> None:
    print(f"\n[{name}] Cross-run chunk sharing (two genuinely different runs)")
    print(f"  Run 1 unique chunks : {stats['run1_unique_chunks']:,}")
    print(f"  Run 2 unique chunks : {stats['run2_unique_chunks']:,}")
    print(
        f"  Shared (R1 ∩ R2)    : {stats['run2_shared_with_run1']:,} ({stats['cross_run_reuse_pct']:.1f}%)"
    )
    print(f"  New in run 2        : {stats['run2_new_chunks']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _sklearn_paths(checkpoint_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Return (run1_paths, run2_paths). For sklearn we generate two seeds inline."""
    import pickle as pkl
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier

    paths_r1 = sorted(checkpoint_dir.glob("*.pkl"))
    if not paths_r1:
        print("  No sklearn checkpoints. Run generate_checkpoints.py first.")
        return [], []

    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10, random_state=99
    )

    tmp = checkpoint_dir / "_run2_tmp"
    tmp.mkdir(exist_ok=True)
    model = GradientBoostingClassifier(
        n_estimators=10, warm_start=True, random_state=99, max_depth=4
    )
    paths_r2 = []
    for i in range(len(paths_r1)):
        total_trees = (i + 1) * 10
        model.set_params(n_estimators=total_trees)
        model.fit(X, y)
        p = tmp / f"step_{total_trees:06d}.pkl"
        with open(p, "wb") as f:
            pkl.dump(model, f)
        paths_r2.append(p)

    return paths_r1, paths_r2


def _xgboost_paths(checkpoint_dir: Path) -> Tuple[List[Path], List[Path]]:
    import xgboost as xgb
    from sklearn.datasets import make_classification

    paths_r1 = sorted(checkpoint_dir.glob("*.ubj"))
    if not paths_r1:
        print("  No XGBoost checkpoints. Run generate_checkpoints.py first.")
        return [], []

    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10, random_state=99
    )
    X32, y32 = X.astype(np.float32), y.astype(np.float32)
    dtrain = xgb.DMatrix(X32, label=y32)
    params = {
        "max_depth": 4,
        "objective": "binary:logistic",
        "seed": 99,
        "verbosity": 0,
    }

    tmp = checkpoint_dir / "_run2_tmp"
    tmp.mkdir(exist_ok=True)
    booster = None
    paths_r2 = []
    for i in range(len(paths_r1)):
        booster = xgb.train(
            params, dtrain, num_boost_round=10, xgb_model=booster, verbose_eval=False
        )
        total = (i + 1) * 10
        p = tmp / f"step_{total:06d}.ubj"
        booster.save_model(str(p))
        paths_r2.append(p)

    return paths_r1, paths_r2


def _pytorch_crossrun_paths(
    tmp_dir: Path, n_finetune_epochs: int = 5
) -> Tuple[List[Path], List[Path]]:
    """Train a shared base model then fine-tune two variants with different seeds.

    Scenario: a pretrained base model is fine-tuned in two separate runs (e.g. different
    hyperparameters, different tasks). Both runs start from identical base weights.

    Returns (paths_run1, paths_run2) where:
      - run1 = [base_checkpoint] + [finetune epoch 1..N with seed=42]
      - run2 = [base_checkpoint] + [finetune epoch 1..N with seed=99]

    The base checkpoint is shared between both paths. Cross-run measurement:
    after storing run1, what fraction of run2's unique chunks already exist?
    The base model tensors (identical across both runs) should all be present;
    fine-tuned layers diverge immediately.
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((1000, 64)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 10, size=1000).astype(np.int64))
    dataset = torch.utils.data.TensorDataset(X, y)

    def _make_model() -> nn.Module:
        return nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    # Phase 1: train shared base for 5 epochs (deterministic, seed=0)
    base = _make_model()
    base_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        num_workers=0,
    )
    optimizer = torch.optim.SGD(base.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    base.train()
    for _ in range(5):
        for inputs, labels in base_loader:
            optimizer.zero_grad()
            criterion(base(inputs), labels).backward()
            optimizer.step()

    # Save the shared base checkpoint
    base_path = tmp_dir / "base.pt"
    torch.save(base.state_dict(), str(base_path))
    base_state = {k: v.clone() for k, v in base.state_dict().items()}

    # Phase 2: fine-tune two variants from the same base, different random seeds
    seeds = [42, 99]
    path_lists = []

    for i, seed in enumerate(seeds):
        run_dir = tmp_dir / f"run{i + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = _make_model()
        model.load_state_dict(base_state)

        g = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, generator=g, num_workers=0
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Each run includes the base as epoch_0 so both runs share those chunks
        paths = [base_path]
        model.train()
        for epoch in range(1, n_finetune_epochs + 1):
            for inputs, labels in loader:
                opt.zero_grad()
                criterion(model(inputs), labels).backward()
                opt.step()
            p = run_dir / f"epoch_{epoch:06d}.pt"
            torch.save(model.state_dict(), str(p))
            paths.append(p)

        path_lists.append(paths)

    return path_lists[0], path_lists[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pale dedup analysis — three separate measurements"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["sklearn", "xgboost", "pytorch"],
        default=["sklearn"],
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--chunk-size", type=int, default=256 * 1024)
    args = parser.parse_args()

    def _timed(label: str):
        class _T:
            def __enter__(self):
                self._t = time.time()
                return self

            def __exit__(self, *_):
                print(f"  [{label}: {time.time() - self._t:.1f}s]")

        return _T()

    if "sklearn" in args.frameworks:
        print("\n=== sklearn ===")
        paths = sorted((args.checkpoint_dir / "sklearn").glob("*.pkl"))
        if paths:
            adapter = SklearnAdapter()

            with _timed("noop"):
                noop = measure_noop_fastpath(paths, adapter)
            print_noop_report("sklearn", noop)

            with _timed("chunk dedup"):
                chunk = measure_chunk_dedup(paths, adapter, args.chunk_size)
            print_chunk_dedup_report("sklearn", chunk, args.chunk_size)

            with _timed("cross-run gen+measure"):
                paths_r1, paths_r2 = _sklearn_paths(args.checkpoint_dir / "sklearn")
                if paths_r1 and paths_r2:
                    cross = measure_crossrun_dedup(
                        paths_r1, paths_r2, adapter, args.chunk_size
                    )
                    print_crossrun_report("sklearn", cross)

    if "xgboost" in args.frameworks:
        print("\n=== xgboost ===")
        paths = sorted((args.checkpoint_dir / "xgboost").glob("*.ubj"))
        if paths:
            adapter = XGBoostAdapter()

            with _timed("noop"):
                noop = measure_noop_fastpath(paths, adapter)
            print_noop_report("xgboost", noop)

            with _timed("chunk dedup"):
                chunk = measure_chunk_dedup(paths, adapter, args.chunk_size)
            print_chunk_dedup_report("xgboost", chunk, args.chunk_size)

            with _timed("cross-run gen+measure"):
                paths_r1, paths_r2 = _xgboost_paths(args.checkpoint_dir / "xgboost")
                if paths_r1 and paths_r2:
                    cross = measure_crossrun_dedup(
                        paths_r1, paths_r2, adapter, args.chunk_size
                    )
                    print_crossrun_report("xgboost", cross)

    if "pytorch" in args.frameworks:
        print("\n=== pytorch ===")
        paths = sorted((args.checkpoint_dir / "pytorch").glob("*.pt"))
        if paths:
            try:
                from pale.adapters.pytorch import PyTorchAdapter

                adapter = PyTorchAdapter()

                with _timed("noop"):
                    noop = measure_noop_fastpath(paths, adapter)
                print_noop_report("pytorch", noop)

                with _timed("chunk dedup"):
                    chunk = measure_chunk_dedup(paths, adapter, args.chunk_size)
                print_chunk_dedup_report("pytorch", chunk, args.chunk_size)

                print(
                    "\n[pytorch] Cross-run dedup: generating shared-base fine-tuning runs..."
                )
                with _timed("cross-run gen+measure"):
                    import tempfile as _tmpmod

                    with _tmpmod.TemporaryDirectory(prefix="pale_crossrun_") as _tmp:
                        paths_r1, paths_r2 = _pytorch_crossrun_paths(Path(_tmp))
                        cross = measure_crossrun_dedup(
                            paths_r1, paths_r2, adapter, args.chunk_size
                        )
                        print_crossrun_report(
                            "pytorch (shared-base fine-tuning)", cross
                        )
            except ImportError:
                print("  Skipping — PyTorchAdapter not available")
