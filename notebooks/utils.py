import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import zstandard as zstd

from tensorcas.hashing import hash_chunk as _hash
from tensorcas.chunking import chunk_bytes as _chunk
from tensorcas.serialization import tensor_to_bytes as _tensor_to_bytes


CHUNK_SIZE = 256 * 1024  # 256 KB

def _to_bytes(arr: np.ndarray) -> bytes:
    raw, _, _ = _tensor_to_bytes(arr)
    return raw


def _fmt_bytes(n: float) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"

def extract_sklearn(model) -> Dict[str, np.ndarray]:
    """Extract per-tree arrays from a fitted GradientBoosting model."""
    tensors = {}
    for i, col in enumerate(model.estimators_):
        for j, tree in enumerate(col):
            idx = i * len(col) + j
            t = tree.tree_
            tensors[f"tree_{idx:06d}_features"] = t.feature.astype(np.int32)
            tensors[f"tree_{idx:06d}_thresholds"] = t.threshold.astype(np.float64)
            tensors[f"tree_{idx:06d}_values"] = t.value.squeeze().astype(np.float64)
    return dict(sorted(tensors.items()))


def extract_xgboost(booster) -> Dict[str, np.ndarray]:
    """Extract per-tree bytes from an XGBoost Booster via JSON dump."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        booster.save_model(tmp)
        with open(tmp) as f:
            model_json = json.load(f)
    finally:
        os.unlink(tmp)
    trees = model_json["learner"]["gradient_booster"]["model"].pop("trees")
    skeleton_bytes = json.dumps(model_json).encode()
    tensors = {"__skeleton__": np.frombuffer(skeleton_bytes, dtype=np.uint8).copy()}
    for i, tree in enumerate(trees):
        tb = json.dumps(tree).encode()
        tensors[f"tree_{i:06d}"] = np.frombuffer(tb, dtype=np.uint8).copy()
    return dict(sorted(tensors.items()))


def extract_pytorch(state_dict) -> Dict[str, np.ndarray]:
    """Convert a PyTorch state_dict to numpy arrays."""
    tensors = {}
    for k, v in state_dict.items():
        arr = v.detach().cpu().numpy()
        tensors[k] = np.ascontiguousarray(arr)
    return dict(sorted(tensors.items()))

def measure_noop(sequences: List[Dict[str, np.ndarray]]) -> Dict:
    """Fraction of tensor-steps that are byte-identical to the previous checkpoint."""
    tensor_names = sorted(sequences[0].keys())
    stats = {n: {"identical": 0, "changed": 0, "compressed_bytes": []} for n in tensor_names}
    cctx = zstd.ZstdCompressor(level=3)
    for prev, curr in zip(sequences, sequences[1:]):
        for name in tensor_names:
            if name not in curr:
                continue
            raw_p, raw_c = _to_bytes(prev[name]), _to_bytes(curr[name])
            if _hash(raw_p) == _hash(raw_c):
                stats[name]["identical"] += 1
            else:
                stats[name]["changed"] += 1
                stats[name]["compressed_bytes"].append(len(cctx.compress(raw_c)))
    for name in tensor_names:
        if not stats[name]["compressed_bytes"]:
            stats[name]["compressed_bytes"].append(len(cctx.compress(_to_bytes(sequences[0][name]))))
    n_pairs = len(sequences) - 1
    result = {}
    for name, s in stats.items():
        result[name] = {
            "identical": s["identical"],
            "changed": s["changed"],
            "identical_pct": s["identical"] / n_pairs * 100,
            "avg_compressed_bytes": sum(s["compressed_bytes"]) / len(s["compressed_bytes"]),
        }
    all_id = sum(s["identical"] for s in stats.values())
    all_tot = n_pairs * len(tensor_names)
    result["__summary__"] = {"identical_pct": all_id / all_tot * 100 if all_tot else 0.0}
    return result


def measure_chunk_dedup(sequences: List[Dict[str, np.ndarray]], chunk_size: int = CHUNK_SIZE) -> Dict:
    """For changed tensors, fraction of chunks already in the cumulative CAS store."""
    tensor_names = sorted(sequences[0].keys())
    changed: Dict[str, Dict] = defaultdict(lambda: {"reused": 0, "total": 0})
    cumulative_hashes: Dict[str, Set[str]] = defaultdict(set)
    for prev, curr in zip(sequences, sequences[1:]):
        for name in tensor_names:
            if name not in curr:
                continue
            raw_p, raw_c = _to_bytes(prev[name]), _to_bytes(curr[name])
            for c in _chunk(raw_p, chunk_size):
                cumulative_hashes[name].add(_hash(c))
            if _hash(raw_p) == _hash(raw_c):
                continue
            for c in _chunk(raw_c, chunk_size):
                changed[name]["total"] += 1
                if _hash(c) in cumulative_hashes[name]:
                    changed[name]["reused"] += 1
    result = {}
    tot_r, tot_t = 0, 0
    for name, s in changed.items():
        result[name] = {
            "total": s["total"],
            "reused": s["reused"],
            "reuse_pct": s["reused"] / s["total"] * 100 if s["total"] else 0.0,
        }
        tot_r += s["reused"]
        tot_t += s["total"]
    result["__summary__"] = {
        "reuse_pct": tot_r / tot_t * 100 if tot_t else 0.0,
        "no_changed_tensors": len(changed) == 0,
    }
    return result


def measure_crossrun(
    seqs_r1: List[Dict[str, np.ndarray]],
    seqs_r2: List[Dict[str, np.ndarray]],
    chunk_size: int = CHUNK_SIZE,
) -> Dict:
    """Chunk overlap between two independent training runs."""
    def _unique_hashes(seqs) -> Set[str]:
        s: Set[str] = set()
        for tensors in seqs:
            for arr in tensors.values():
                for c in _chunk(_to_bytes(arr), chunk_size):
                    s.add(_hash(c))
        return s

    h1, h2 = _unique_hashes(seqs_r1), _unique_hashes(seqs_r2)
    shared = len(h1 & h2)
    return {
        "run1_unique": len(h1),
        "run2_unique": len(h2),
        "shared": shared,
        "new_in_run2": len(h2) - shared,
        "reuse_pct": shared / len(h2) * 100 if h2 else 0.0,
    }

def dvc_bytes(files: List[Path]) -> int:
    """Simulate DVC file-level dedup: zstd each file, count unique compressed blobs."""
    seen: dict[str, int] = {}
    cctx = zstd.ZstdCompressor(level=3)
    for path in files:
        raw = path.read_bytes()
        compressed = cctx.compress(raw)
        h = _hash(compressed)
        if h not in seen:
            seen[h] = len(compressed)
    return sum(seen.values())


def tensorcas_bytes(root: Path) -> int:
    """Sum of all .chunk file sizes under {root}/objects/."""
    return sum(p.stat().st_size for p in (root / "objects").rglob("*.chunk"))


def run_dvc_comparison(
    label: str,
    models: List[Any],
    adapter: Any,
    checkpoint_dir: Path,
) -> Dict:
    """Compare DVC-simulated vs tensorcas storage for a sequence of model checkpoints."""
    from tensorcas.store import tensorcasStore

    checkpoint_files = sorted(checkpoint_dir.glob("*"))
    dvc = dvc_bytes(checkpoint_files)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tensorcasStore(root=root, run_id="bench", adapter=adapter) as store:
            for step, model in enumerate(models, 1):
                store.save(model, step=step)
        tensorcas = tensorcas_bytes(root)

    raw_total = sum(p.stat().st_size for p in checkpoint_files)
    savings_vs_dvc = (dvc - tensorcas) / dvc * 100 if dvc else 0.0
    return {
        "label": label,
        "checkpoints": len(models),
        "raw_total": raw_total,
        "dvc_bytes": dvc,
        "tensorcas_bytes": tensorcas,
        "ratio": tensorcas / dvc if dvc else 0.0,
        "savings_vs_dvc": savings_vs_dvc,
    }

def print_noop(label: str, stats: Dict, max_tensors: int = 30) -> None:
    summary = stats.get("__summary__", {})
    print(f"\n[{label}] No-op fast path — tensor identity across steps")
    print(f"  Overall: {summary['identical_pct']:.1f}% of tensor-steps byte-identical")
    rows = sorted((k, v) for k, v in stats.items() if k != "__summary__")
    if len(rows) > max_tensors:
        print(f"  (showing {max_tensors} of {len(rows)} tensors)")
        rows = rows[:max_tensors]
    print(f"  {'Tensor':<40} {'Identical':>10} {'Changed':>8} {'Identical%':>11} {'AvgCompressed':>14}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*11} {'-'*14}")
    for name, s in rows:
        print(f"  {name:<40} {s['identical']:>10} {s['changed']:>8} "
              f"{s['identical_pct']:>10.1f}% {_fmt_bytes(s['avg_compressed_bytes']):>14}")


def print_chunk(label: str, stats: Dict, chunk_size: int = CHUNK_SIZE, max_tensors: int = 20) -> None:
    cs = f"{chunk_size // 1024}KB" if chunk_size < 1024**2 else f"{chunk_size // 1024**2}MB"
    summary = stats.get("__summary__", {})
    print(f"\n[{label}] Chunk-level reuse in changed tensors (chunk={cs})")
    if summary.get("no_changed_tensors"):
        print("  Overall: N/A — no changed tensors")
        return
    print(f"  Overall: {summary['reuse_pct']:.1f}% of chunks in changed tensors already in store")
    rows = sorted((k, v) for k, v in stats.items() if k != "__summary__")
    if len(rows) > max_tensors:
        print(f"  (showing {max_tensors} of {len(rows)} tensors)")
        rows = rows[:max_tensors]
    print(f"  {'Tensor':<40} {'Total':>8} {'Reused':>8} {'Reuse%':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for name, s in rows:
        print(f"  {name:<40} {s['total']:>8} {s['reused']:>8} {s['reuse_pct']:>7.1f}%")


def print_crossrun(label: str, stats: Dict) -> None:
    print(f"\n[{label}] Cross-run chunk sharing")
    print(f"  Run 1 unique chunks : {stats['run1_unique']:,}")
    print(f"  Run 2 unique chunks : {stats['run2_unique']:,}")
    print(f"  Shared (R1 ∩ R2)    : {stats['shared']:,}  ({stats['reuse_pct']:.1f}%)")
    print(f"  New in run 2        : {stats['new_in_run2']:,}")


def print_dvc_comparison(results: List[Dict]) -> None:
    print(f"\n{'Framework':<12} {'Steps':>6} {'Raw total':>10} {'DVC bytes':>10} {'tensorcas bytes':>10} {'Ratio':>7} {'vs DVC':>8}")
    print("=" * 70)
    for r in results:
        print(
            f"{r['label']:<12} {r['checkpoints']:>6} "
            f"{_fmt_bytes(r['raw_total']):>10} {_fmt_bytes(r['dvc_bytes']):>10} "
            f"{_fmt_bytes(r['tensorcas_bytes']):>10} {r['ratio']:>7.3f} "
            f"{r['savings_vs_dvc']:>7.1f}%"
        )
