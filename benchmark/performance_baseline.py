import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pale.store import PaleStore
from pale.adapters.pytorch import PyTorchAdapter
from pale.adapters.sklearn import SklearnAdapter
from pale.adapters.xgboost import XGBoostAdapter
import torch.nn as nn


def _median_ms(fn: Callable, n: int = 10) -> float:
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _sklearn_model(n_estimators: int):
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    m = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
    m.fit(X, y)
    return m


def _xgboost_model(n_rounds: int):
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(
        {"max_depth": 3, "objective": "binary:logistic", "seed": 0, "verbosity": 0},
        dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False,
    )


def _pytorch_state_dict(hidden: int, n_layers: int):
    layers = []
    in_features = 64
    for _ in range(n_layers):
        layers += [nn.Linear(in_features, hidden), nn.ReLU()]
        in_features = hidden
    layers.append(nn.Linear(hidden, 10))
    model = nn.Sequential(*layers)
    return model.state_dict()


def _model_size_bytes(model: Any, adapter) -> int:
    tensors = adapter.extract(model)
    return sum(t.nbytes for t in tensors.values())


def _fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def _save_fresh(adapter, model: Any, max_workers: int = 8) -> None:
    """Cold save: fresh store each rep so no-op fast path never fires."""
    with tempfile.TemporaryDirectory(prefix="pale_rep_") as rep_tmp:
        with PaleStore(
            root=Path(rep_tmp), run_id="r", adapter=adapter, max_workers=max_workers
        ) as store:
            store.save(model, step=1)


def measure_absolute(
    name: str, model: Any, adapter, n_reps: int = 10, max_workers: int = 8
) -> None:
    size_str = _fmt_size(_model_size_bytes(model, adapter))
    print(f"\n  {name}  ({size_str} tensors)")

    save_ms = _median_ms(lambda: _save_fresh(adapter, model, max_workers), n_reps)
    save_verdict = "OK" if save_ms < 500 else "SLOW"

    with tempfile.TemporaryDirectory(prefix="pale_load_") as tmp:
        with PaleStore(
            root=Path(tmp), run_id="r", adapter=adapter, max_workers=max_workers
        ) as store:
            store.save(model, step=1)
            load_ms = _median_ms(lambda: store.load(step=1), n_reps)

    load_verdict = "OK" if load_ms < 200 else "SLOW"

    print(f"    save: {save_ms:7.1f}ms  (target <500ms)  [{save_verdict}]")
    print(f"    load: {load_ms:7.1f}ms  (target <200ms)  [{load_verdict}]")


def measure_scaling(
    name: str,
    models_by_size: list,
    size_labels: list,
    adapter,
    n_reps: int = 5,
    max_workers: int = 8,
) -> None:
    print(f"\n  {name}  — save time vs model size")
    print(f"    {'Size':>12}  {'Tensor size':>12}  {'Save (ms)':>10}")
    print(f"    {'-' * 12}  {'-' * 12}  {'-' * 10}")

    prev_save_ms = None
    for model, label in zip(models_by_size, size_labels):
        size_str = _fmt_size(_model_size_bytes(model, adapter))
        try:
            save_ms = _median_ms(
                lambda m=model: _save_fresh(adapter, m, max_workers), n_reps
            )
        except Exception as exc:
            print(f"    {label:>12}  {size_str:>12}  {'ERROR':>10}  ({exc})")
            continue
        flag = ""
        if prev_save_ms is not None and save_ms > prev_save_ms * 3:
            flag = "  ← superlinear"
        print(f"    {label:>12}  {size_str:>12}  {save_ms:>10.1f}{flag}")
        prev_save_ms = save_ms


def measure_noop(
    name: str, model: Any, adapter, n_reps: int = 10, max_workers: int = 8
) -> None:
    print(f"\n  {name}  — no-op fast path (unchanged model saved twice)")

    with tempfile.TemporaryDirectory(prefix="pale_noop_") as tmp:
        with PaleStore(
            root=Path(tmp), run_id="r", adapter=adapter, max_workers=max_workers
        ) as store:
            store.save(model, step=1)
            noop_ms = _median_ms(
                lambda: store.save(model, step=_next_step(store)), n_reps
            )

    print(
        f"    noop save: {noop_ms:.1f}ms  (should be near-zero — hashing only, no CAS writes)"
    )


def _next_step(store: PaleStore) -> int:
    steps = store.list_checkpoints()
    return (max(steps) + 1) if steps else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pale performance baseline")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["sklearn", "xgboost", "pytorch"],
        default=["sklearn", "xgboost", "pytorch"],
    )
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    print(f"Reps: {args.reps} timed + 1 warmup each. Metric: median ms.")

    if "sklearn" in args.frameworks:

        print("\n" + "=" * 60)
        print("SKLEARN")
        print("=" * 60)

        sk_adapter = SklearnAdapter()
        sk_model = _sklearn_model(50)

        print("\n[1] Absolute overhead")
        measure_absolute("GBM 50 trees", sk_model, sk_adapter, args.reps)

        print("\n[2] Scaling")
        measure_scaling(
            "GBM",
            [_sklearn_model(n) for n in [10, 25, 50, 100]],
            ["10 trees", "25 trees", "50 trees", "100 trees"],
            sk_adapter,
            n_reps=max(3, args.reps // 2),
        )

        print("\n[3] No-op fast path")
        measure_noop("GBM 50 trees", sk_model, sk_adapter, args.reps)

    if "xgboost" in args.frameworks:

        print("\n" + "=" * 60)
        print("XGBOOST")
        print("=" * 60)

        xgb_adapter = XGBoostAdapter()
        xgb_model = _xgboost_model(50)

        print("\n[1] Absolute overhead")
        measure_absolute(
            "Booster 50 rounds", xgb_model, xgb_adapter, args.reps, max_workers=1
        )

        print("\n[2] Scaling")
        measure_scaling(
            "Booster",
            [_xgboost_model(n) for n in [10, 25, 50, 100]],
            ["10 rounds", "25 rounds", "50 rounds", "100 rounds"],
            xgb_adapter,
            n_reps=max(3, args.reps // 2),
            max_workers=1,
        )

        print("\n[3] No-op fast path")
        measure_noop(
            "Booster 50 rounds", xgb_model, xgb_adapter, args.reps, max_workers=1
        )

    if "pytorch" in args.frameworks:

        print("\n" + "=" * 60)
        print("PYTORCH")
        print("=" * 60)

        pt_adapter = PyTorchAdapter()
        pt_state = _pytorch_state_dict(hidden=256, n_layers=2)

        print("\n[1] Absolute overhead")
        measure_absolute("MLP 256×2", pt_state, pt_adapter, args.reps, max_workers=1)

        print("\n[2] Scaling")
        measure_scaling(
            "MLP",
            [
                _pytorch_state_dict(hidden=h, n_layers=l)
                for h, l in [(64, 2), (256, 2), (512, 4), (1024, 4)]
            ],
            ["64×2", "256×2", "512×4", "1024×4"],
            pt_adapter,
            n_reps=max(3, args.reps // 2),
            max_workers=1,
        )

        print("\n[3] No-op fast path")
        measure_noop("MLP 256×2", pt_state, pt_adapter, args.reps, max_workers=1)

    print("\n=== done ===")
