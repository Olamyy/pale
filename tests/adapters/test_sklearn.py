import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression

from pale.adapters.sklearn import SklearnAdapter
from pale.errors import AdapterError
from pale.hashing import hash_chunk
from pale.serialization import tensor_to_bytes


def _trained_gbm(
    n_estimators: int = 10, n_classes: int = 2
) -> GradientBoostingClassifier:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 4))
    if n_classes == 2:
        y = (X[:, 0] > 0).astype(int)
    else:
        y = (X[:, 0] * n_classes).astype(int).clip(0, n_classes - 1)
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
    model.fit(X, y)
    return model


def _trained_linear() -> LinearRegression:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    model = LinearRegression()
    model.fit(X, y)
    return model


def test_gbm_extract_per_tree_keys():
    """Each tree should have three keys: features, thresholds, values."""
    model = _trained_gbm(n_estimators=5)
    tensors = SklearnAdapter().extract(model)
    tree_count = model.estimators_.size
    for i in range(tree_count):
        assert f"tree_{i:06d}_features" in tensors
        assert f"tree_{i:06d}_thresholds" in tensors
        assert f"tree_{i:06d}_values" in tensors


def test_extract_keys_sorted():
    keys = list(SklearnAdapter().extract(_trained_gbm()).keys())
    assert keys == sorted(keys)


def test_features_dtype_is_integer():
    tensors = SklearnAdapter().extract(_trained_gbm())
    feature_keys = [k for k in tensors if k.endswith("_features")]
    for k in feature_keys:
        assert np.issubdtype(tensors[k].dtype, np.integer), k


def test_thresholds_dtype_is_float64():
    tensors = SklearnAdapter().extract(_trained_gbm())
    for k in [k for k in tensors if k.endswith("_thresholds")]:
        assert tensors[k].dtype == np.float64, k


def test_values_dtype_is_float64():
    tensors = SklearnAdapter().extract(_trained_gbm())
    for k in [k for k in tensors if k.endswith("_values")]:
        assert tensors[k].dtype == np.float64, k


def test_linear_extract_keys():
    tensors = SklearnAdapter().extract(_trained_linear())
    assert "coef_" in tensors
    assert "intercept_" in tensors


def test_linear_coef_native_dtype():
    model = _trained_linear()
    tensors = SklearnAdapter().extract(model)
    assert tensors["coef_"].dtype == model.coef_.dtype


def test_no_numeric_arrays_raises():
    class EmptyModel:
        pass

    with pytest.raises(AdapterError, match="No extractable"):
        SklearnAdapter().extract(EmptyModel())


def test_gbm_binary_round_trip():
    adapter = SklearnAdapter()
    model = _trained_gbm(n_estimators=10, n_classes=2)
    tensors = adapter.extract(model)
    reconstructed = adapter.reconstruct(tensors, model)
    for key in tensors:
        assert np.array_equal(tensors[key], adapter.extract(reconstructed)[key]), key


def test_gbm_multiclass_round_trip():
    """estimators_ is shape (n_estimators, n_classes) — must iterate correctly."""
    adapter = SklearnAdapter()
    model = _trained_gbm(n_estimators=10, n_classes=3)
    assert model.estimators_.shape[1] == 3

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
            assert hash_chunk(raw_10) == hash_chunk(raw_20), (
                f"Frozen tree {i} {suffix} hash changed after warm-start continuation"
            )


def test_reconstruct_missing_key_raises():
    adapter = SklearnAdapter()
    model = _trained_gbm(n_estimators=5)
    tensors = adapter.extract(model)

    del tensors["tree_000000_features"]
    with pytest.raises(AdapterError, match="Missing tensor key"):
        adapter.reconstruct(tensors, model)


def test_linear_round_trip():
    adapter = SklearnAdapter()
    model = _trained_linear()
    tensors = adapter.extract(model)
    reconstructed = adapter.reconstruct(tensors, model)
    assert np.array_equal(model.coef_, reconstructed.coef_)
    assert np.array_equal(model.intercept_, reconstructed.intercept_)


def test_reconstruct_does_not_mutate_original():
    adapter = SklearnAdapter()
    model = _trained_gbm(n_estimators=10)
    original_threshold = model.estimators_[0][0].tree_.threshold[0]

    tensors = dict(adapter.extract(model))
    tensors["tree_000000_thresholds"] = tensors["tree_000000_thresholds"] + 999.0
    adapter.reconstruct(tensors, model)

    assert model.estimators_[0][0].tree_.threshold[0] == original_threshold
