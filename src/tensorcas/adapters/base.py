from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelAdapter(Protocol):
    def extract(self, model: Any) -> Dict[str, np.ndarray]:
        """Extract model parameters as a dict of named numpy arrays.

        Keys must be returned in deterministic sorted order.
        Callers rely on stable ordering for manifest construction and no-op fast path.
        """
        ...

    def reconstruct(self, tensors: Dict[str, np.ndarray], original: Any) -> Any:
        """Reconstruct a model from tensors, using original as a template.

        original provides structure (class, hyperparameters) but its weights
        are replaced by those in tensors.
        """
        ...
