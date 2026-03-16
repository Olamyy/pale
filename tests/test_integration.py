import numpy as np
import pytest
import torch
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

from pale.adapters.pytorch import PyTorchAdapter
from pale.adapters.sklearn import SklearnAdapter
from pale.adapters.xgboost import XGBoostAdapter
from pale.errors import CorruptChunkError
from pale.store import PaleStore

_RNG = np.random.default_rng(42)
_X = _RNG.standard_normal((200, 8))
_Y = (_X[:, 0] > 0).astype(int)
_X32 = _X.astype(np.float32)
_Y32 = _Y.astype(np.float32)


def _sklearn_model(n_estimators: int = 20) -> GradientBoostingClassifier:
    m = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
    m.fit(_X, _Y)
    return m


def _xgboost_model(n_rounds: int = 10) -> xgb.Booster:
    dtrain = xgb.DMatrix(_X32, label=_Y32)
    return xgb.train(
        {"max_depth": 3, "objective": "binary:logistic", "seed": 0, "verbosity": 0},
        dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False,
    )


def _pytorch_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )


def test_sklearn_full_round_trip(tmp_path):
    model = _sklearn_model()
    preds_before = model.predict(_X)
    with PaleStore(root=tmp_path, run_id="r", adapter=SklearnAdapter()) as store:
        store.save(model, step=1)
        restored = store.load(step=1, original=model)
    np.testing.assert_array_equal(preds_before, restored.predict(_X))


def test_sklearn_warm_start_dedup(tmp_path):
    model = GradientBoostingClassifier(n_estimators=10, warm_start=True, random_state=0)
    model.fit(_X, _Y)
    with PaleStore(root=tmp_path, run_id="r", adapter=SklearnAdapter()) as store:
        store.save(model, step=1)
        model.set_params(n_estimators=20)
        model.fit(_X, _Y)
        store.save(model, step=2)
        s = store.stats()
    assert s["unique_chunks"] < s["total_chunks"]
    assert s["dedup_ratio"] < 1.0


def test_corrupt_chunk_detected_on_load(tmp_path):
    with PaleStore(root=tmp_path, run_id="r", adapter=SklearnAdapter()) as store:
        store.save(_sklearn_model(), step=1)
        chunk = next((tmp_path / "objects").rglob("*.chunk"))
        chunk.write_bytes(b"corrupt")
        with pytest.raises(CorruptChunkError):
            store.load(step=1, original=_sklearn_model())


def test_xgboost_full_round_trip(tmp_path):
    model = _xgboost_model()
    dtest = xgb.DMatrix(_X32)
    preds_before = model.predict(dtest)
    with PaleStore(
        root=tmp_path, run_id="r", adapter=XGBoostAdapter(), max_workers=1
    ) as store:
        store.save(model, step=1)
        restored = store.load(step=1)
    np.testing.assert_array_equal(preds_before, restored.predict(dtest))


def test_pytorch_full_round_trip(tmp_path):
    model = _pytorch_model()
    x = torch.from_numpy(_X32)
    with torch.no_grad():
        preds_before = model(x).numpy()
    with PaleStore(root=tmp_path, run_id="r", adapter=PyTorchAdapter()) as store:
        store.save(model.state_dict(), step=1)
        state = store.load(step=1)
    model.load_state_dict(state)
    with torch.no_grad():
        preds_after = model(x).numpy()
    np.testing.assert_array_equal(preds_before, preds_after)


def test_cross_framework_store_stats(tmp_path):
    with PaleStore(root=tmp_path, run_id="sk", adapter=SklearnAdapter()) as store:
        store.save(_sklearn_model(), step=1)
        store.save(_sklearn_model(), step=2)
    with PaleStore(
        root=tmp_path, run_id="xg", adapter=XGBoostAdapter(), max_workers=1
    ) as store:
        store.save(_xgboost_model(), step=1)
    stats = PaleStore.store_stats(tmp_path)
    assert stats["runs"] == 2
    assert stats["checkpoints"] == 3
    assert stats["unique_chunks"] < stats["total_chunks"]
    assert stats["dedup_ratio"] < 1.0
