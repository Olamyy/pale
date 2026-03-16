import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from pale.adapters.sklearn import SklearnAdapter
from pale.errors import AdapterError
from pale.hashing import hash_chunk
from pale.serialization import tensor_to_bytes


def _trained_gbm(n_estimators: int = 10, n_classes: int = 2) -> GradientBoostingClassifier:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 4))
    y = (X[:, 0] * n_classes).astype(int).clip(0, n_classes - 1) if n_classes > 2 else (X[:, 0] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
    model.fit(X, y)
    return model


def test_thresholds_dtype_is_float64():
    tensors = SklearnAdapter().extract(_trained_gbm())
    for k in [k for k in tensors if k.endswith("_thresholds")]:
        assert tensors[k].dtype == np.float64, k


def test_no_numeric_arrays_raises():
    with pytest.raises(AdapterError, match="No extractable"):
        SklearnAdapter().extract(type("EmptyModel", (), {})())


def test_gbm_multiclass_round_trip():
    adapter = SklearnAdapter()
    model = _trained_gbm(n_estimators=10, n_classes=3)
    tensors = adapter.extract(model)
    reconstructed = adapter.reconstruct(tensors, model)
    for key in tensors:
        assert np.array_equal(tensors[key], adapter.extract(reconstructed)[key]), key


def test_warm_start_frozen_trees_identical():
    adapter = SklearnAdapter()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 4))
    y = (X[:, 0] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=10, warm_start=True, random_state=0)
    model.fit(X, y)
    tensors_10 = adapter.extract(model)
    model.set_params(n_estimators=20)
    model.fit(X, y)
    tensors_20 = adapter.extract(model)
    for i in range(10):
        for suffix in ("features", "thresholds", "values"):
            key = f"tree_{i:06d}_{suffix}"
            raw_10, _, _ = tensor_to_bytes(tensors_10[key])
            raw_20, _, _ = tensor_to_bytes(tensors_20[key])
            assert hash_chunk(raw_10) == hash_chunk(raw_20), key


def test_reconstruct_missing_key_raises():
    adapter = SklearnAdapter()
    model = _trained_gbm(n_estimators=5)
    tensors = adapter.extract(model)
    del tensors["tree_000000_features"]
    with pytest.raises(AdapterError, match="Missing tensor key"):
        adapter.reconstruct(tensors, model)


def test_reconstruct_does_not_mutate_original():
    adapter = SklearnAdapter()
    model = _trained_gbm(n_estimators=10)
    original_threshold = model.estimators_[0][0].tree_.threshold[0]
    tensors = dict(adapter.extract(model))
    tensors["tree_000000_thresholds"] = tensors["tree_000000_thresholds"] + 999.0
    adapter.reconstruct(tensors, model)
    assert model.estimators_[0][0].tree_.threshold[0] == original_threshold
