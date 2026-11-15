# Implementation Plan: Memoised-filter `trace` without copying the model

## Overview

Add a shared `FilterSnapshotMixin` (in `bed/base.py`) providing `snapshot()`,
which returns a new filter sharing `model` and copying only the belief arrays.
`EKF` and `UKF` inherit it and declare which attributes to copy. `run()` uses
`filtr.snapshot()` instead of `deepcopy(filtr)`.

Spec: [docs/specs/SPEC-trace-snapshot.md](../docs/specs/SPEC-trace-snapshot.md) (delta, approved: shared mixin).

## Design

`src/bed/base.py`:

```python
import copy as _copy

class FilterSnapshotMixin:
    """Provide a cheap belief-only copy of a state-estimation filter.

    `snapshot()` returns a new instance that *shares* ``model`` (it is immutable
    during a sequential run) and copies the mutable belief attributes listed in
    ``_snapshot_copied_attrs``. Recorded traces therefore do not alias the live
    filter, but also do not duplicate the model graph.
    """

    # Subclasses set these.
    _snapshot_copied_attrs = ("state_prior", "measurement_error")

    def snapshot(self):
        clone = type(self).__new__(type(self))
        clone.model = self.model                      # shared, immutable during a run
        for name in self._snapshot_copied_attrs:
            setattr(clone, name, _copy.copy(getattr(self, name)))
        return clone
```

`EKF(FilterSnapshotMixin, ...)` — copied attrs: `state_prior`, `measurement_error`.
`UKF(FilterSnapshotMixin, ...)` — adds `state_innovation` (and `alpha`/`beta`/`kappa`
are immutable scalars, harmless to carry; include them for completeness).

`copy.copy` on a tuple `(mean, cov)` is a shallow tuple copy — it would *share*
the arrays. So use a deep copy of just the belief:

```python
setattr(clone, name, _copy.deepcopy(getattr(self, name)))
```

`copy.deepcopy` on `(mean, cov)` (two jnp arrays) copies only those arrays — small,
and does not touch `model` because we assign it separately.

## Changes

- `src/bed/base.py`: add `FilterSnapshotMixin`.
- `src/bed/ekf.py`: `class EKF(FilterSnapshotMixin):`, import the mixin,
  set `_snapshot_copied_attrs`.
- `src/bed/ukf.py`: same; copied attrs include `state_innovation` and the scalars.
- `src/bed/_experiments.py`: `filters.append(filtr.snapshot())` with a
  `getattr(filtr, "snapshot", None)` fallback to `deepcopy`.

## Task list

### Phase 1: mixin

- [ ] Task 1: add `FilterSnapshotMixin` to `base.py`
- [ ] Task 2: EKF/UKF inherit it + declare copied attrs

### Checkpoint

- [ ] `filters[i].model is filtr.model`; `filters[i].state_prior` distinct arrays

### Phase 2: wire + tests

- [ ] Task 3: `run()` uses `snapshot()` (with deepcopy fallback)
- [ ] Task 4: tests — shared model, independent belief, snapshot supports get_state_posterior/measurement_prior/deepcopy

### Phase 3: verify

- [ ] Task 5: full gate set (149 + new; ruff/ty/pymarkdown)

## Risks

| Risk | Mitigation |
|---|---|
| Shallow copy of `state_prior` aliases arrays | Use `deepcopy` of the attr (arrays only, model excluded) |
| `type(self).__new__` skips `__init__` invariants | Snapshot only needs the attributes listed; tests exercise get_state_posterior/measurement_prior on it |
| Circular import via `filters/` | Mixin lives in `base.py`, which imports nothing from ekf/ukf |
