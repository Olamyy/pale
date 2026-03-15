import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from pale.adapters.xgboost import XGBoostAdapter
from pale.errors import AdapterError


def _trained_booster(n_rounds: int = 10) -> xgb.Booster:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"max_depth": 3, "objective": "binary:logistic", "seed": 0}
    return xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)


def _predict(booster: xgb.Booster) -> np.ndarray:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    return booster.predict(xgb.DMatrix(X))


def test_extract_keys():
    tensors = XGBoostAdapter.extract(_trained_booster())
    assert set(tensors.keys()) == {"model_bytes"}


def test_extract_dtype_uint8():
    tensors = XGBoostAdapter.extract(_trained_booster())
    assert tensors["model_bytes"].dtype == np.uint8


def test_extract_nonempty():
    tensors = XGBoostAdapter.extract(_trained_booster())
    assert tensors["model_bytes"].size > 0


def test_extract_returns_writable_array():
    tensors = XGBoostAdapter.extract(_trained_booster())
    # Should be writable (not a read-only frombuffer view)
    tensors["model_bytes"][0] = tensors["model_bytes"][0]


def test_reconstruct_round_trip_predictions():
    """Reconstructed booster must produce identical predictions."""
    booster = _trained_booster()
    preds_before = _predict(booster)

    tensors = XGBoostAdapter.extract(booster)
    reconstructed = XGBoostAdapter.reconstruct(tensors, original=None)
    preds_after = _predict(reconstructed)

    np.testing.assert_array_equal(preds_before, preds_after)


def test_reconstruct_ignores_original():
    """reconstruct uses tensors, not the original booster."""
    booster = _trained_booster(n_rounds=10)
    tensors = XGBoostAdapter.extract(booster)

    # Pass a different booster as original — result must match the tensors, not original
    other_booster = _trained_booster(n_rounds=5)
    reconstructed = XGBoostAdapter.reconstruct(tensors, original=other_booster)

    np.testing.assert_array_equal(_predict(booster), _predict(reconstructed))


def test_reconstruct_missing_key_raises():
    with pytest.raises(AdapterError, match="model_bytes"):
        XGBoostAdapter.reconstruct({}, original=None)


def test_wrong_type_raises():
    with pytest.raises(AdapterError, match="expects xgb.Booster"):
        XGBoostAdapter.extract("not a booster")
