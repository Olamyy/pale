# Pale

[![CI](https://github.com/Olamyy/pale/actions/workflows/ci.yml/badge.svg)](https://github.com/Olamyy/pale/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/Olamyy/pale/blob/main/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/Olamyy/pale?style=flat&sort=semver&color=blue)](https://github.com/Olamyy/pale/releases)


Pale is a checkpoint storage library for machine learning models that eliminates redundant storage at the tensor level rather than the file level. It uses [content-addressable storage](https://en.wikipedia.org/wiki/Content-addressable_storage) (CAS) as its storage layer.

---

ML frameworks store checkpoints as complete snapshots — every weight, every parameter, every time you call `.save`. For warm-start tree models or fine-tuned neural networks, most of those parameters haven't changed since the last checkpoint. You pay full storage cost to write the same bytes repeatedly.

Pale solves this by operating on tensors instead of files. Each adapter extracts the model's individual components — trees, layers, weight matrices — as named numpy arrays. Pale hashes each array and compares it against the previous checkpoint. Arrays that haven't changed are skipped entirely; only new or updated arrays are written to disk.

For example, a warm-start GBM that adds 10 trees per step writes only those 10 new trees. The previous 90 are already on disk and cost nothing to "save" again.

Pale replaces your `save` and `load` calls. Each adapter handles framework-specific serialization internally.

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

## Core concepts

**Run and step.** A run is a training experiment, identified by a string `run_id`. A step is a checkpoint within that run, identified by an integer. Multiple runs can share the same store root — deduplication works across runs.

**Content-addressable storage.** Every tensor is split into fixed-size chunks (default 256 KB) and each chunk is stored once, keyed by its BLAKE3 hash, under `{root}/objects/`. The SQLite registry at `{root}/registry.db` tracks which checkpoints reference which chunks. A grace-period GC sweeps unreferenced chunks after checkpoint deletion.

**No-op fast path.** Before writing anything, Pale hashes the full tensor and compares it against the previous checkpoint's manifest. If the hash matches, the tensor is skipped entirely — no chunking, no CAS write, no disk I/O. This is the primary source of storage savings: frozen trees, frozen layers, and unchanged weight matrices are identified in memory and never written again. At 0.5–1.9 ms per step for typical models, the no-op path is effectively free.

---

## Known limitations

**PyTorch: BatchNorm running statistics update in train mode.**
Freezing layers via `requires_grad_(False)` does not prevent BatchNorm buffers (`running_mean`, `running_var`, `num_batches_tracked`) from updating — they are updated on every forward pass in `model.train()` mode regardless. For ResNet-18 this means 80 of 122 state dict tensors change every epoch even when the entire backbone is frozen, capping the no-op rate at ~48% instead of the theoretical ~99%.

If you want these buffers to stop changing, call `model.eval()` before saving:

```python
model.eval()
store.save(model, step=epoch)
model.train()
```

This freezes running stats to their current values. The trade-off is that saved checkpoints reflect eval-mode statistics, not the running averages from the most recent training batches.

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
