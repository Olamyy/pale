import copy
from typing import Any, Dict, Iterator, Tuple

import numpy as np

from pale.errors import AdapterError


class SklearnAdapter:
    """Extract and reconstruct sklearn model parameters as numpy arrays.

    Supported model types:
    - GradientBoostingClassifier / GradientBoostingRegressor (warm_start tree ensembles)
    - Linear models with coef_ / intercept_ (LogisticRegression, LinearRegression, etc.)

    GBM trees are stored per-tree rather than concatenated:
        tree_{i:06d}_features    — int32 feature indices for tree i
        tree_{i:06d}_thresholds  — float64 split thresholds for tree i
        tree_{i:06d}_values      — float64 leaf values for tree i

    This ensures that across warm_start checkpoints, frozen trees produce
    byte-identical arrays → identical BLAKE3 hashes → CAS dedup fires for free.

    NOTE: reconstruct() requires original to be a fitted model with the same tree
    count as the model extract() was called on. sklearn's DecisionTree internal
    structure (tree_) is a Cython struct that cannot be constructed from scratch.
    This is a known v0.1.0 limitation.
    """

    def extract(self, model: Any) -> Dict[str, np.ndarray]:
        """Extract model parameters. Returns dict with sorted keys."""
        tensors: Dict[str, np.ndarray] = {}

        if hasattr(model, "estimators_") and len(model.estimators_) > 0:
            tensors.update(self._extract_gbm(model))

        if hasattr(model, "coef_"):
            tensors["coef_"] = np.asarray(model.coef_)
        if hasattr(model, "intercept_"):
            tensors["intercept_"] = np.asarray(model.intercept_)

        if not tensors:
            raise AdapterError(
                f"No extractable numeric arrays found in {type(model).__name__}. "
                "Model must have estimators_, coef_, or intercept_."
            )

        return dict(sorted(tensors.items()))

    def reconstruct(self, tensors: Dict[str, np.ndarray], original: Any) -> Any:
        """Reconstruct model by writing tensors back onto a copy of original.

        original must have the same tree count as the model extract() was called on.
        Raises AdapterError if any expected tree key is missing from tensors.
        """
        model = copy.deepcopy(original)

        if hasattr(model, "estimators_") and len(model.estimators_) > 0:
            self._reconstruct_gbm(tensors, model)

        if "coef_" in tensors:
            model.coef_ = tensors["coef_"]
        if "intercept_" in tensors:
            model.intercept_ = tensors["intercept_"]

        return model

    # ------------------------------------------------------------------
    # GBM helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_trees(model: Any) -> Iterator[Tuple[int, Any]]:
        """Yield (index, tree) for each tree in estimators_.

        estimators_ is shape (n_estimators, n_classes) for multiclass,
        (n_estimators, 1) for binary/regression — iterate uniformly.
        """
        for i, stage in enumerate(np.asarray(model.estimators_).flat):
            yield i, stage

    @staticmethod
    def _tree_key(i: int, suffix: str) -> str:
        return f"tree_{i:06d}_{suffix}"

    def _extract_gbm(self, model: Any) -> Dict[str, np.ndarray]:
        """Store per-tree arrays so frozen warm_start trees produce identical hashes."""
        tensors: Dict[str, np.ndarray] = {}
        for i, tree in self._iter_trees(model):
            t = tree.tree_
            tensors[self._tree_key(i, "features")] = t.feature.copy()
            tensors[self._tree_key(i, "thresholds")] = t.threshold.copy()  # float64
            tensors[self._tree_key(i, "values")] = t.value.flatten().copy()  # float64
        return tensors

    def _reconstruct_gbm(self, tensors: Dict[str, np.ndarray], model: Any) -> None:
        """Write per-tree arrays back into model.estimators_ in-place."""
        for i, tree in self._iter_trees(model):
            feat_key = self._tree_key(i, "features")
            thresh_key = self._tree_key(i, "thresholds")
            val_key = self._tree_key(i, "values")

            for key in (feat_key, thresh_key, val_key):
                if key not in tensors:
                    raise AdapterError(
                        f"Missing tensor key '{key}' — was the correct base model "
                        "passed to reconstruct()?"
                    )

            t = tree.tree_
            t.feature[:] = tensors[feat_key].astype(t.feature.dtype)
            t.threshold[:] = tensors[thresh_key].astype(t.threshold.dtype)
            t.value[:] = tensors[val_key].reshape(t.value.shape).astype(t.value.dtype)
