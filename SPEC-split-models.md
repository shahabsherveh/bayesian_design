# Spec Delta: Split `_models.py` into family modules

Status: proposed

## Objective

Replace the 1297-line `src/bed/_models.py` monolith with per-family modules and
delete the monolith, while keeping every existing import path working. This
implements deferred item 1 from `docs/STATUS.md` ("`_models.py` (1297 lines) →
split beside its shims, delete the monolith").

The public surface (`bed.models`, the three family shims, and `bed.__init__`) is
deliberately **unchanged**: callers continue to import the same names from the
same paths.

## Current Behavior

`src/bed/_models.py` holds 13 classes and 1 function across three families, plus a
shared `Model` base:

| Family | Members |
|---|---|
| classical | `LinearGaussianModel`, `LinearModel` (inherits `Model`) |
| gaussian process | `GP`, `GaussianProcessModel`, `generate_full_design_matrix` |
| neural | `FlaxModel`, `DenseNN`, `NNFLax`, `LinearNN`, `Sinus`, `SinusInverse`, `CNN`, `NeuralNetworkBase`, `NeuralNetworkClassifier`, `NeuralNetworkRegressor` |
| shared | `Model` |

`bed/models/__init__.py` and the three shims (`classical.py`, `gaussian_process.py`,
`neural.py`) are thin re-exports of everything from `bed._models`. Two non-shim
modules also import `bed._models` directly:

- `src/bed/mechanistic.py` → `from bed._models import Model`
- `src/bed/ukf.py` → `from bed._models import DenseNN, Model, NeuralNetworkRegressor`

## New Behavior

`src/bed/_models.py` is deleted. Its contents move into a new
`src/bed/models/` module tree, so each family lives beside its existing shim, and
the shims become the implementation rather than re-exporters:

```text
src/bed/models/
├── __init__.py          # re-exports the full public set (unchanged __all__)
├── _base.py             # Model (shared base)
├── classical.py         # LinearGaussianModel, LinearModel   (now holds the code)
├── gaussian_process.py  # GP, GaussianProcessModel, generate_full_design_matrix
└── neural.py            # FlaxModel, DenseNN, NNFLax, LinearNN, Sinus, SinusInverse,
                         #   CNN, NeuralNetworkBase, NeuralNetworkClassifier,
                         #   NeuralNetworkRegressor
```

`Model` lives in `_base.py` because both `LinearModel` (classical) and
`NeuralNetworkBase` (neural) inherit from it; `classical.py` and `neural.py`
import it from `._base`.

`bed/models/__init__.py` re-exports the same names, and — to remain a drop-in for
the two direct `bed._models` importers — `src/bed/_models.py` becomes a thin
compatibility shim:

```python
from .models import (
    CNN, GP, DenseNN, FlaxModel, GaussianProcessModel, LinearGaussianModel,
    LinearModel, LinearNN, Model, NeuralNetworkBase, NeuralNetworkClassifier,
    NeuralNetworkRegressor, NNFLax, Sinus, SinusInverse,
    generate_full_design_matrix,
)
```

Actually, the cleaner end-state is to update the two direct importers
(`mechanistic.py`, `ukf.py`) to import from `bed.models` and **keep** a minimal
`_models.py` re-export shim for any external/undiscovered callers, deleting only
the 1297 lines of implementation. This is the decision below.

## Boundary decisions

1. **Keep a thin `bed/_models.py` re-export shim** (a few lines), do not delete the
   file outright. This preserves the two in-repo direct importers *and* any
   external code doing `from bed._models import X`, which `_models` was until now
   the real location for. Update `mechanistic.py` and `ukf.py` to import from
   `bed.models` (the public path) so the library itself stops reaching into the
   private name. The `# refactor` staging note in `bed/models/__init__.py` is
   updated to reflect the new reality.

2. **Shared `Model` goes in `models/_base.py`**, imported by `classical.py` and
   `neural.py` via `from ._base import Model`. `models/__init__.py` still exports
   `Model` (and does not add `_base` to `__all__`).

3. **No public API change.** `bed.models.__all__`, `bed.__all__`, and every import
   in the test/experiment suite must resolve to the same objects.

4. **Imports are split to their owning module.** `cvxpy`, `optax`, `flax.nnx`,
   `tqdm` follow their users (`cvxpy` → classical + gaussian; `optax`/`nnx`/`tqdm`
   → neural; `multivariate_normal` → gaussian_process). No `import *`.

## Success Criteria

- `python -c "import bed; from bed.models import *"` succeeds and exposes the
  same names as before (diff `dir(bed.models)` against a pre-change snapshot).
- `python -c "from bed._models import Model, DenseNN"` still works (re-export shim).
- `src/bed/_models.py` is a thin re-export (no class/function definitions).
- The 13 classes + 1 function live in `models/classical.py`,
  `models/gaussian_process.py`, `models/neural.py`, `models/_base.py`.
- `ruff check src/ tests/ experiments/` clean; this includes `I001` import sorting,
  so each module's imports are ordered.
- `ty check src/ --exit-zero-on-warning` exits 0 with no new error-level
  diagnostics (the split may reclassify some warnings; none may become errors).
- Full suite `pytest tests/ -q -n auto` → 139 passed, 3 skipped (unchanged counts).
- No behaviour change: every test that imports a model still passes unchanged.

## Boundaries

Inherits the project's working boundaries. Delta only:

- **Always:** preserve the public import surface exactly; keep `_models.py` as a
  compatibility re-export; run the full gate set before commit.
- **Ask first:** none — no new dependencies, no schema/config change.
- **Never:** rename any public class/function; change `__all__` contents;
  reorder or rewrite method bodies during the split (pure move, import-path only).

## Out of Scope (Deferred)

- The three remaining deferred items (`optimize()` return shape, `_rng` reseeding,
  `trace=True` memory).
- Any behavioural refactor of the moved code (dead-code removal, typing pass to
  clear the 70 `ty` warnings, de-duplicating the `LinearGaussianModel`/`GP`
  near-identical methods). Move only.

## Open Questions

1. **Delete `_models.py` vs. keep a shim.** I recommend keeping a thin re-export
   shim (decision 1) because two in-repo modules and possibly out-of-repo code
   import `bed._models` directly. Deleting outright is cleaner but a hard break.
   Confirm you want the shim (recommended) rather than a straight delete.

2. **`NNFLax` subclasses `DenseNN`, and `LinearNN`/`Sinus`/`SinusInverse`/`CNN`
   subclass `FlaxModel`** — all in the same `neural.py`, so no cross-module import
   within the neural family. Cross-family, `LinearModel` (classical) and
   `NeuralNetworkBase` (neural) both need `Model` from `_base`. No circularity.
   No action needed, noted for review.

3. **`generate_full_design_matrix`** is a module-level function used by both
   `LinearGaussianModel` and `GP` via the same static helper; I'll place the
   module function where it's most cohesive (suggest `classical.py`, as the two
   classes call it via `_generate_full_design_matrix`). Alternatively it could
   live in `_base.py`. Either way it stays exported from `bed.models`. Flag if you
   have a preference.
