from typing import Any, Dict

import numpy as np

from pale.adapters.deps import require
from pale.errors import AdapterError


class PyTorchAdapter:
    """Extract and reconstruct PyTorch model parameters as numpy arrays.

    Tensors per checkpoint: one entry per key in state_dict(), named exactly
    as the state_dict key (e.g. "layer1.0.weight", "bn1.bias").

    Frozen layers across fine-tuning steps produce byte-identical arrays —
    their chunks are reused from previous checkpoints via CAS dedup.

    Reconstruction: tensors are loaded directly into a new state_dict and
    applied via load_state_dict(). original is ignored.
    """

    def extract(self, model: Any) -> Dict[str, np.ndarray]:
        torch = require("torch")

        if not isinstance(model, (torch.nn.Module, dict)):
            raise AdapterError(
                f"PyTorchAdapter expects nn.Module or state_dict, got {type(model).__name__}"
            )

        state = model.state_dict() if isinstance(model, torch.nn.Module) else model

        tensors: Dict[str, np.ndarray] = {}
        for key, tensor in state.items():
            arr = tensor.detach().cpu().numpy()
            tensors[key] = np.ascontiguousarray(arr)

        return dict(sorted(tensors.items()))

    def reconstruct(self, tensors: Dict[str, np.ndarray], original: Any) -> Any:
        torch = require("torch")

        if not tensors:
            raise AdapterError("tensors dict is empty")

        state = {k: torch.from_numpy(v.copy()) for k, v in tensors.items()}

        if original is not None and isinstance(original, torch.nn.Module):
            original.load_state_dict(state)
            return original

        return state
