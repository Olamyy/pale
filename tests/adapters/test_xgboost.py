import numpy as np
import pytest
import xgboost as xgb

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


def test_extract_contains_skeleton_and_trees():
    tensors = XGBoostAdapter().extract(_trained_booster(5))
    assert "__skeleton__" in tensors
    assert len([k for k in tensors if k.startswith("tree_")]) == 5


def test_extract_dtype_uint8():
    for arr in XGBoostAdapter().extract(_trained_booster(3)).values():
        assert arr.dtype == np.uint8


def test_frozen_trees_identical_across_warmstart():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"max_depth": 3, "objective": "binary:logistic", "seed": 0}

    b5 = xgb.train(params, dtrain, num_boost_round=5, verbose_eval=False)
    b8 = xgb.train(params, dtrain, num_boost_round=3, xgb_model=b5, verbose_eval=False)

    adapter = XGBoostAdapter()
    t5 = adapter.extract(b5)
    t8 = adapter.extract(b8)

    for i in range(5):
        key = f"tree_{i:06d}"
        np.testing.assert_array_equal(t5[key], t8[key], err_msg=f"{key} not identical")


def test_reconstruct_round_trip_predictions():
    booster = _trained_booster()
    preds_before = _predict(booster)
    tensors = XGBoostAdapter().extract(booster)
    reconstructed = XGBoostAdapter.reconstruct(tensors, original=None)
    np.testing.assert_array_equal(preds_before, _predict(reconstructed))


def test_reconstruct_missing_model_key_raises():
    with pytest.raises(AdapterError, match="__skeleton__"):
        XGBoostAdapter.reconstruct(
            {"tree_000000": np.zeros(1, dtype=np.uint8)}, original=None
        )


def test_categorical_splits_raise():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, (200, 2)).astype(np.float32)
    y = (rng.standard_normal(200) > 0).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y, feature_types=["c", "c"], enable_categorical=True)
    params = {
        "max_depth": 3,
        "objective": "binary:logistic",
        "seed": 0,
        "tree_method": "hist",
    }
    booster = xgb.train(params, dtrain, num_boost_round=2, verbose_eval=False)
    with pytest.raises(AdapterError, match="categorical splits"):
        XGBoostAdapter().extract(booster)
