import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tensorcas.adapters.deps import require
from tensorcas.errors import AdapterError


class XGBoostAdapter:
    """Extract and reconstruct XGBoost Booster parameters as numpy arrays.

    Tensors per checkpoint:

    __skeleton__: uint8 array of UTF-8 JSON bytes for the full model minus the
                  trees array. Contains objective, hyperparameters, and all
                  metadata. Identical across warm-start steps for the same
                  training configuration.

    tree_NNNNNN: one uint8 array per tree, containing the UTF-8 JSON bytes of
                 that tree's structured node data. Frozen trees across warm-start
                 steps produce byte-identical arrays — their chunks are reused
                 from previous checkpoints. Only new trees require CAS writes.

    Reconstruction: trees are reinserted into the skeleton JSON and the
    booster is loaded via load_model() from the reassembled JSON file.

    Performance: extract() uses save_raw('json') to avoid a temp-file
    round-trip, and caches per-tree JSON bytes by index so only newly added
    trees are serialized on each call.
    """

    def __init__(self) -> None:
        # Cache of tree index → serialized JSON bytes. Populated on each
        # extract() call; existing entries are never re-serialized.
        self._tree_cache: List[bytes] = []

    def extract(self, model: Any) -> Dict[str, np.ndarray]:
        """Extract skeleton + per-tree tensors from a trained Booster."""
        xgb = require("xgboost")

        if not isinstance(model, xgb.Booster):
            raise AdapterError(
                f"XGBoostAdapter expects xgb.Booster, got {type(model).__name__}"
            )

        try:
            raw = bytes(model.save_raw("json"))
            model_json = json.loads(raw)
        except Exception as e:
            raise AdapterError(f"Failed to serialize booster to JSON: {e}") from e

        trees = model_json["learner"]["gradient_booster"]["model"].pop("trees")

        for i, tree in enumerate(trees):
            if 1 in tree.get("split_type", []):
                raise AdapterError(
                    f"XGBoostAdapter does not support categorical splits "
                    f"(tree {i} contains categorical nodes). "
                    "Train with enable_categorical=False or use only numerical features."
                )

        skeleton_bytes = json.dumps(model_json).encode()

        # Extend cache for any newly added trees; existing entries are reused.
        for i in range(len(self._tree_cache), len(trees)):
            self._tree_cache.append(json.dumps(trees[i]).encode())

        tensors: Dict[str, np.ndarray] = {
            "__skeleton__": np.frombuffer(skeleton_bytes, dtype=np.uint8).copy()
        }
        for i, tree_bytes in enumerate(self._tree_cache):
            tensors[f"tree_{i:06d}"] = np.frombuffer(tree_bytes, dtype=np.uint8).copy()

        return dict(sorted(tensors.items()))

    @staticmethod
    def reconstruct(tensors: Dict[str, np.ndarray], original: Any) -> Any:
        """Reconstruct booster by reinserting trees into the skeleton JSON."""
        xgb = require("xgboost")

        if "__skeleton__" not in tensors:
            raise AdapterError("tensors must contain '__skeleton__' key")

        tree_keys = sorted(k for k in tensors if k.startswith("tree_"))

        try:
            model_json = json.loads(tensors["__skeleton__"].tobytes().decode())
            trees = [json.loads(tensors[k].tobytes().decode()) for k in tree_keys]
            model_json["learner"]["gradient_booster"]["model"]["trees"] = trees
        except Exception as e:
            raise AdapterError(f"Failed to reassemble model JSON: {e}") from e

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
            json.dump(model_json, f)

        try:
            booster = xgb.Booster()
            booster.load_model(str(tmp_path))
        except Exception as e:
            raise AdapterError(f"load_model from JSON failed: {e}") from e
        finally:
            tmp_path.unlink(missing_ok=True)

        return booster
