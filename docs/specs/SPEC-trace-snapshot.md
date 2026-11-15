# Spec Delta: Memoised-filter `trace` without copying the model

Status: implemented

## Objective

Fix deferred item 4 (the last one) from `docs/STATUS.md`: "`trace=True`
deep-copies a full latent covariance per epoch." When `run(trace=True)` records
the filter each epoch, `deepcopy(filtr)` clones the **entire filter object graph
including the neural-network model**. The model is identical across all epochs and
never mutated during a run, so copying it every epoch is pure waste —
41 MB and ~22 ms per epoch for a modest `DenseNN`, i.e. ~4 GB retained over 100
epochs.

This replaces the full-object `deepcopy` with a lightweight filter **snapshot**
that shares the model and copies only the belief arrays.

## Current Behavior

`run()` per epoch:

```python
if trace:
    filters.append(deepcopy(filtr))
```

Measured cost (one `deepcopy`):

| model | weight_size | latent cov size | pickle size | time |
|---|---|---|---|---|
| `LinearNN` (test fixture) | 3 | 9 elements | ~2.5 KB | ~0.4 ms |
| `DenseNN([32,32,32])` | 2305 | 42.5 MB | ~41 MB | ~22 ms |

`deepcopy` copies `model` (the `NeuralNetworkRegressor`, holding the full Flax
graph) **and** `state_prior` (mean + full `d×d` covariance) every epoch. The
model reference is identical every epoch and immutable during the run.

Consumers of `ExperimentResults.filters`:

- `tests/empirical_bayes_test.py` reads `filters[0].state_prior`,
  `.measurement_error`.
- `experiments/falsify_pkpd.py` does `deepcopy(r.filters[-1])` then calls
  `.get_state_posterior(...)`, and reads `.state_prior`, `.measurement_prior(...)`.
- `experiments/sequential_pkpd.py` similar.

So a snapshot must be a **fully usable filter** (supporting `get_state_posterior`,
`measurement_prior`, and `deepcopy`-able again by callers), not just a state tuple.

## New Behavior

Add `snapshot()` to `EKF` and `UKF` that returns a new filter instance sharing
`model` and copying only the mutable belief:

```python
# EKF
def snapshot(self):
    clone = EKF.__new__(EKF)
    clone.model = self.model                      # shared, immutable during a run
    clone.state_prior = (copy(self.state_prior[0]), copy(self.state_prior[1]))
    clone.measurement_error = copy(self.measurement_error)
    return clone

# UKF additionally copies alpha/beta/kappa (scalars) and state_innovation
```

`run()` uses `filters.append(filtr.snapshot())` instead of `deepcopy(filtr)`.

## Boundary decisions

1. **Snapshot shares the model.** It is provably immutable during `run()`
   (`self.model` is set once in `Experiment.__init__`, never reassigned in the
   run loop). Sharing is safe and is the entire win. The snapshot still exposes
   `.model`, so `get_state_posterior`/`measurement_prior` work unchanged.
2. **Snapshot copies the belief arrays.** `state_prior` mean/cov and
   `measurement_error` are copied so later in-place updates to the live filter
   do not mutate already-recorded snapshots. This preserves the current
   "independent trace" contract.
3. **Snapshots remain `deepcopy`-able by callers.** `falsify_pkpd.py` and
   `sequential_pkpd.py` do `deepcopy(r.filters[-1])` and mutate the copy; a
   snapshot is an ordinary EKF/UKF instance, so that keeps working.
4. **`EKF`/`UKF` gain one method each; no signature changes elsewhere.** The
   `deepcopy(filtr)` call site is the only change in `run()`.
5. **Fallback for objects without `snapshot()`.** `run()` currently accepts any
   `filter_instance`. To stay safe, use `getattr(filtr, "snapshot", None)` and
   fall back to `deepcopy` if absent — but since both in-tree filters implement
   it, the fallback is only a safety net.

## Success Criteria

- `run(trace=True)` produces `filters` whose entries are **independent** of the
  live filter: mutating `filtr.state_prior` after the loop does not change any
  recorded snapshot (the existing trace contract).
- A snapshot supports `get_state_posterior`, `measurement_prior`, and
  `deepcopy`, so `empirical_bayes_test.py` and the PK/PD scripts keep working.
  (`measurement_posterior_cov_estimate` is a **method** on both filters, so it
  comes along automatically — nothing to snapshot.)
- The model is **not** cloned per epoch: `filters[i].model is filtr.model`
  (shared), while `filters[i].state_prior` is a distinct array from
  `filtr.state_prior`.
- Snapshotting is **faster than `deepcopy`**: measured ~10.6 ms vs ~16.4 ms per
  epoch on a `DenseNN([32,32,32])` because `deepcopy` traverses the whole object
  graph (model included) while `snapshot` copies only the declared arrays.
- **Honest scope of the win:** the dominant per-epoch cost is the latent
  covariance itself (`d²·8` bytes — e.g. 40.5 MB for `d=2305`), which must be
  retained per epoch because it *is* the recorded belief. This change removes the
  model clone and the graph traversal, it does **not** (and cannot, without a
  lazy/compressed trace design) reduce the covariance memory. The review's
  "copies the model every epoch" is real but small in absolute bytes; the win is
  the traversal time and the avoided clone.
- Full suite `pytest tests/ -q -n auto` → 156 passed, 3 skipped (149 + 7 new).
- `ruff` / `ty` / `pymarkdown` green.

## Boundaries

Inherits the project's working boundaries. Delta only:

- **Always:** snapshots independent of the live filter; model shared (proven
  immutable); snapshots remain usable filters and `deepcopy`-able.
- **Ask first:** none.
- **Never:** change the `filters` semantics (`filters[i]` is the belief *before*
  epoch `i`'s update), change `EKF`/`UKF` public signatures, or make snapshots
  alias live beliefs.

## Out of Scope (Deferred)

- The now-empty deferred backlog (this is the last item).
- Any change to how `filters` is consumed by tests/scripts.
- Storing traces on disk / memory-mapping (a much larger design).

## Open Questions

1. **Where `snapshot()` lives.** Add it to `EKF` and `UKF` (each ~8 lines) rather
   than a shared mixin, to keep the change local and the two classes' attribute
   sets explicit. Confirm you prefer per-class methods over a mixin.
