# Pale

Pale is a framework-agnostic content-addressed checkpoint storage for machine learning models.

---

Saving a model checkpoint stores the full weights every time. For tree-based models trained with warm-start, or neural networks fine-tuned from a base, most of those weights are identical to the previous checkpoint. Pale stores only what changed.

Most checkpoint tools operate on files. Pale operates on tensors. By extracting individual trees, layers, and weight matrices directly, it can identify what actually changed between checkpoints and skip everything that didn't.

Pale replaces your checkpoint save and load calls. Each adapter handles framework-specific serialization internally. 

---

## Installation

```bash
# pip
pip install pale

# uv
uv add pale
```

Framework adapters are included but their dependencies are optional:

```bash
# pip
pip install "pale[sklearn]"            # scikit-learn
pip install "pale[xgboost]"            # XGBoost
pip install "pale[torch]"            # torch
pip install "pale[sklearn,xgboost,torch]"  # all of the above

# uv
uv add "pale[sklearn]"
uv add "pale[xgboost]"
uv add "pale[torch]"
```
---

## Quick start

```python
from pathlib import Path
import torch
import torch.nn as nn
from pale.store import PaleStore
from pale.adapters.pytorch import PyTorchAdapter

model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

store = PaleStore(
    root=Path("./checkpoints"),
    run_id="mlp-run-001",
    adapter=PyTorchAdapter(),
)

# Save a checkpoint at each epoch
for epoch in range(1, 21):
    # ... training loop ...
    store.save(model, step=epoch)

# Load any checkpoint
store.load(step=10, original=model)

# Inspect what was saved
print(store.stats())
# {'run_id': 'mlp-run-001', 'checkpoints': 20, 'total_chunks': 60,
#  'unique_chunks': 8, 'dedup_ratio': 0.1333, 'total_bytes': 245760}
# dedup_ratio = unique/total chunks — lower means more reuse (0.13 = 87% reused)
```

---

## Supported frameworks

| Framework | Adapter class | Supported model types |
|-----------|--------------|----------------------|
| scikit-learn | `pale.adapters.sklearn.SklearnAdapter` | `GradientBoostingClassifier`, `GradientBoostingRegressor`, `LogisticRegression`, any model with `coef_` / `intercept_` |
| XGBoost | `pale.adapters.xgboost.XGBoostAdapter` | `xgb.Booster` |
| PyTorch | `pale.adapters.pytorch.PyTorchAdapter` | Any model with a `state_dict()` |
| Custom | Implement `ModelAdapter` | `extract(model) -> Dict[str, ndarray]` and `reconstruct(tensors, original) -> model` |

---

## Benchmark results

### Storage savings vs DVC

Measured over 20-step training runs. DVC stores full checkpoints on every save; Pale deduplicates at the tensor level.

| Framework | Scenario | DVC stores | Pale stores | Savings |
|-----------|----------|------------|-------------|---------|
| sklearn | 20 warm-start steps | 1.4 MB | 80 KB | **94%** |
| XGBoost | 20 warm-start steps | 721 KB | 150 KB | **79%** |
| PyTorch | 20 epochs, frozen backbone | 10.6 MB | 5.4 MB | **49%** |

### Overhead

Measured on Apple Silicon (macOS), Python 3.12. Save uses a cold store per rep; load reuses the same store (warm page cache). Full methodology in [`benchmark/results/`](benchmark/results/).

| Framework | Model | Save | Load | No-op save |
|-----------|-------|------|------|------------|
| sklearn | GBM, 50 trees, 4 KB | 38 ms | 9 ms | **1.4 ms** |
| XGBoost | Booster, 50 rounds, 25 KB | 29 ms | 5 ms | **1.9 ms** |
| PyTorch | MLP 256×2, 332 KB | 7 ms | 1 ms | **0.5 ms** |
| PyTorch | MLP 1024×4, 12 MB | 36 ms | 16 ms | — |

The no-op fast path (tensor hash matches previous checkpoint → zero writes) is the dominant case for warm-start and fine-tuning workflows. At 0.5–1.9 ms per step it is effectively free.

---

## Core concepts

**Run and step.** A run is a training experiment, identified by a string `run_id`. A step is a checkpoint within that run, identified by an integer. Multiple runs can share the same store root — deduplication works across runs.

**Content-addressable storage.** Every tensor is split into fixed-size chunks (default 1 MiB). Each chunk is stored once, keyed by its BLAKE3 hash, under `{root}/objects/`. If two checkpoints — in the same run or different runs — contain identical chunks, only one copy exists on disk. The SQLite registry at `{root}/registry.db` tracks which chunks belong to which checkpoints, and a grace-period GC sweeps orphaned chunks after deletion.

**No-op fast path.** Before splitting a tensor into chunks, Pale hashes the full tensor and compares it against the previous checkpoint's manifest. If the hash matches, the entire tensor is skipped — no chunking, no dedup check, no CAS writes. This is where most of the savings come from: a warm-start GBM that adds trees each step has all existing trees frozen; only the new trees produce writes.

---

## CLI

```bash
pale --root ./checkpoints list
pale --root ./checkpoints list --run mlp-run-001
pale --root ./checkpoints stats
pale --root ./checkpoints stats --run mlp-run-001 --format json
pale --root ./checkpoints gc --grace 24 --yes
pale --root ./checkpoints delete --run mlp-run-001 --step 3 --yes
```

| Command | What it does |
|---------|-------------|
| `list` | List all runs and their steps. `--run` scopes to one run. |
| `stats` | Dedup statistics: chunk counts, unique chunks, dedup ratio, total bytes. |
| `gc` | Sweep orphaned blobs older than `--grace` hours (default 24). Prompts unless `--yes`. |
| `delete` | Delete a single checkpoint. Blob files are cleaned by the next `gc`. Prompts unless `--yes`. |

---

## Architecture

```
User / Framework
      │ model object
      ▼
PaleStore                          (store.py)
  save / load / gc / stats
      │ Dict[str, ndarray]              │ registry queries
      ▼                                 ▼
ModelAdapter                       Registry              (registry/registry.py)
  extract(model)                     register_checkpoint
  reconstruct(tensors, original)     list_runs / list_checkpoints
                                     delete_checkpoint / gc / stats
      │                              Backed by SQLite (registry.db)
      ▼
StorageEngine                      (storage.py)
  Per-tensor no-op fast path
  (shape/dtype pre-check → full_hash comparison)
  Tensors fanned out via ThreadPoolExecutor
      │ TensorArrayRecord               │ manifest JSON
      ▼                                 ▼
CASEngine                          ManifestWriter/Reader  (manifest.py)
  chunk → hash → batch_has            Atomic write (temp + rename)
  parallel put / get                  Path: manifests/{run_id}/step_{n:06d}.json
  (ThreadPoolExecutor)
      │ bytes
      ▼
FilesystemBackend                  (cas/filesystem.py)
  objects/{h[:2]}/{h[2:4]}/{h[4:]}.chunk
  zstd compression, atomic writes
```

---

## Development

**Run tests**

```bash
uv run pytest tests/ -q
```

The test suite requires `xgboost` and `torch` (installed as dev dependencies via `uv`). All 53 tests run without skips.

**Add an adapter**

Implement the `ModelAdapter` protocol from `pale.adapters.base`:

```python
class ModelAdapter(Protocol):
    def extract(self, model: Any) -> Dict[str, np.ndarray]:
        """Extract tensors from a model. Return a flat dict of named arrays."""
        ...

    def reconstruct(self, tensors: Dict[str, np.ndarray], original: Any) -> Any:
        """Reconstruct a model from tensors. original is the template model, if needed."""
        ...
```

`extract` should return the minimal set of arrays needed to restore model state — not raw serialization bytes. This is what makes tensor-level deduplication effective: weight updates between steps, not serialization artifacts.

`original` is required for adapters that cannot reconstruct a model from tensors alone — for example, sklearn's internal Cython tree structures cannot be built from scratch and need a fitted model as a template. PyTorch adapters typically don't need it.

See `src/pale/adapters/sklearn.py` for a reference implementation.
