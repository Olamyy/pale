
import argparse
import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List

import zstandard as zstd

from pale.adapters.sklearn import SklearnAdapter
from pale.adapters.xgboost import XGBoostAdapter
from pale.store import PaleStore

CHECKPOINT_DIR = Path(__file__).parent / "data" / "checkpoints"
CHUNK_SIZES = [256 * 1024, 1 * 1024 * 1024, 4 * 1024 * 1024]

_cctx = zstd.ZstdCompressor(level=3)


def dvc_simulated_bytes(checkpoint_paths: List[Path]) -> int:
    """Bytes DVC would store: compress each unique file once."""
    seen: Dict[str, int] = {}
    total = 0
    for path in checkpoint_paths:
        data = path.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        if h not in seen:
            seen[h] = len(_cctx.compress(data))
            total += seen[h]
    return total


def pale_bytes_within_run(
    checkpoint_paths: List[Path], adapter, chunk_size: int, max_workers: int = 8
) -> int:
    """Compressed bytes Pale stores for a single run over all checkpoints."""
    with tempfile.TemporaryDirectory(prefix="pale_bench_") as tmp:
        root = Path(tmp)
        with PaleStore(
            root=root,
            run_id="bench",
            adapter=adapter,
            chunk_size=chunk_size,
            max_workers=max_workers,
        ) as store:
            for i, path in enumerate(checkpoint_paths):
                store.save(_load(path, adapter), step=i + 1)
        objects_dir = root / "objects"
        if not objects_dir.exists():
            return 0
        return sum(f.stat().st_size for f in objects_dir.rglob("*.chunk"))


def pale_bytes_cross_run(
    checkpoint_paths: List[Path], adapter, chunk_size: int, max_workers: int = 8
) -> Dict:
    """Two independent runs over the same checkpoints sharing one CAS store.

    Measures how much the second run saves purely from CAS dedup —
    no prev_manifest, no fast path, just content-addressed storage.
    """
    with tempfile.TemporaryDirectory(prefix="pale_cross_") as tmp:
        root = Path(tmp)
        with PaleStore(
            root=root,
            run_id="run1",
            adapter=adapter,
            chunk_size=chunk_size,
            max_workers=max_workers,
        ) as store:
            for i, path in enumerate(checkpoint_paths):
                store.save(_load(path, adapter), step=i + 1)
        bytes_run1 = sum(f.stat().st_size for f in (root / "objects").rglob("*.chunk"))

        with PaleStore(
            root=root,
            run_id="run2",
            adapter=adapter,
            chunk_size=chunk_size,
            max_workers=max_workers,
        ) as store:
            for i, path in enumerate(checkpoint_paths):
                store.save(_load(path, adapter), step=i + 1)
        bytes_run2_total = sum(
            f.stat().st_size for f in (root / "objects").rglob("*.chunk")
        )

    additional = bytes_run2_total - bytes_run1
    savings_pct = (1 - additional / bytes_run1) * 100 if bytes_run1 > 0 else 0.0
    return {
        "bytes_run1": bytes_run1,
        "bytes_run2_additional": additional,
        "cross_run_savings_pct": savings_pct,
    }


def _load(path: Path, adapter):
    if isinstance(adapter, XGBoostAdapter):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for XGBoostAdapter — uv add xgboost")
        b = xgb.Booster()
        b.load_model(str(path))
        return b
    if path.suffix == ".pt":
        import torch

        return torch.load(str(path), map_location="cpu", weights_only=True)
    with open(path, "rb") as f:
        return pickle.load(f)


def _chunk_label(chunk_size: int) -> str:
    if chunk_size >= 1024 * 1024:
        return f"{chunk_size // (1024 * 1024)}MB"
    return f"{chunk_size // 1024}KB"


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def run_benchmark(
    name: str, checkpoint_dir: Path, adapter, ext: str, max_workers: int = 8
) -> List[Dict]:
    paths = sorted(checkpoint_dir.glob(f"*.{ext}"))
    if not paths:
        print(f"  No {name} checkpoints found. Run generate_checkpoints.py first.")
        return []

    print(f"  {len(paths)} checkpoints")
    raw_total = sum(p.stat().st_size for p in paths)
    dvc_bytes = dvc_simulated_bytes(paths)
    print(f"  Raw total (uncompressed): {_fmt_bytes(raw_total)}")
    print(f"  DVC (compressed, file-level): {_fmt_bytes(dvc_bytes)}")

    cross = pale_bytes_cross_run(
        paths, adapter, chunk_size=CHUNK_SIZES[0], max_workers=max_workers
    )
    if cross["bytes_run2_additional"] == 0:
        print(f"  Cross-run: perfect dedup — second run wrote zero new bytes")
    else:
        print(
            f"  Cross-run: second run wrote {_fmt_bytes(cross['bytes_run2_additional'])} new bytes ({cross['cross_run_savings_pct']:+.1f}%)"
        )

    rows = []
    for chunk_size in CHUNK_SIZES:
        pale = pale_bytes_within_run(
            paths, adapter, chunk_size, max_workers=max_workers
        )
        label = _chunk_label(chunk_size)
        ratio = pale / dvc_bytes if dvc_bytes > 0 else 1.0
        winner = "PALE" if pale < dvc_bytes else ("TIE" if pale == dvc_bytes else "DVC")
        savings_pct = (1 - ratio) * 100
        rows.append(
            {
                "framework": name,
                "chunk_size": label,
                "dvc_bytes": dvc_bytes,
                "pale_bytes": pale,
                "ratio": ratio,
                "winner": winner,
                "savings_pct": savings_pct,
                "cross_run_savings_pct": cross["cross_run_savings_pct"],
                "cross_run_additional": cross["bytes_run2_additional"],
            }
        )
        print(
            f"  [{label}] Pale within-run: {_fmt_bytes(pale)} (ratio={ratio:.3f}, {savings_pct:+.1f}% vs DVC)"
        )

    return rows


def print_table(rows: List[Dict]) -> None:
    if not rows:
        return
    print(
        f"\n{'Framework':<12} {'Chunk':<6} {'DVC':<10} {'Pale':<10} {'Ratio':<7} {'vs DVC':<9} {'Cross-run 2nd':<14} {'Cross savings'}"
    )
    print("=" * 83)
    for r in rows:
        print(
            f"{r['framework']:<12} {r['chunk_size']:<6} "
            f"{_fmt_bytes(r['dvc_bytes']):<10} {_fmt_bytes(r['pale_bytes']):<10} "
            f"{r['ratio']:<7.3f} {r['savings_pct']:>+6.1f}%   "
            f"{_fmt_bytes(r['cross_run_additional']):<14} {r['cross_run_savings_pct']:>+.1f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedup comparison: DVC-style vs Pale")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["sklearn", "xgboost", "pytorch"],
        default=["sklearn"],
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    all_rows = []

    if "sklearn" in args.frameworks:
        print("\n--- sklearn warm-start ---")
        all_rows += run_benchmark(
            "sklearn", args.checkpoint_dir / "sklearn", SklearnAdapter(), "pkl"
        )

    if "xgboost" in args.frameworks:
        print("\n--- XGBoost ---")
        all_rows += run_benchmark(
            "xgboost",
            args.checkpoint_dir / "xgboost",
            XGBoostAdapter(),
            "ubj",
            max_workers=1,
        )

    if "pytorch" in args.frameworks:
        print("\n--- PyTorch fine-tune ---")
        try:
            from pale.adapters.pytorch import PyTorchAdapter

            all_rows += run_benchmark(
                "pytorch", args.checkpoint_dir / "pytorch", PyTorchAdapter(), "pt"
            )
        except ImportError:
            print("  Skipping — PyTorchAdapter not available")

    print_table(all_rows)
