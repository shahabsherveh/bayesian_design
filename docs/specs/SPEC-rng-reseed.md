# Spec Delta: Stop reseeding the RNG inside `run()`

Status: implemented

## Objective

Fix deferred item 3 from `docs/STATUS.md`: "`_rng` reseeded inside `run()` (so
per-epoch randomness restarts from the same value)." Make each `run()` call take
its random state as a **local** generator, so repeated calls on the same
`Experiment` advance rather than replay, while keeping runs reproducible when a
seed (or an explicit `rng`) is supplied.

## Current Behavior

`Experiment.__init__` sets `self._rng = np.random.default_rng(seed)` (line 139).

`run()` begins with:

```python
self._rng = rng if rng is not None else np.random.default_rng(self.seed)
```

So every `run()` call **rebinds the instance attribute** `self._rng`. Four
methods read `self._rng`: `optimize()` (criterion keys), `_criterion_keys()`,
`_sample_latent()`, and `calculate_epig_mc()` (outer/inner samples). Consequences:

1. **Repeated `run()` calls with `rng=None` replay the same stream.** Calling
   `exp.run(..., rng=None)` twice produces the identical sequence of random
   designs/keys both times, because each call resets `self._rng` from
   `self.seed`. Verified: two consecutive reseeds yield identical `integers`.
2. **`run()` mutates shared instance state as a side effect**, which is why
   `run_experiment()` must thread explicit `rng=np.random.default_rng(self.seed)`
   / `self.seed + 1` values to get distinct streams per criterion. A caller using
   `run()` directly — as `tests/runner_test.py` does — gets replay semantics it
   did not ask for, and there is no way to get independent streams without
   constructing generators by hand.

The "per-epoch restarts from the same value" phrasing describes symptom 1 at the
call granularity: because the generator is reset at the top of each `run()`, a
caller that invokes `run()` once per epoch (or per criterion without an explicit
`rng`) sees every invocation start from the same value.

## New Behavior

`run()` uses a **local** generator and does not rebind `self._rng`:

```python
local_rng = rng if rng is not None else self._rng
```

- The four consumers (`optimize`, `_criterion_keys`, `_sample_latent`,
  `calculate_epig_mc`) receive this generator for the duration of the call
  instead of reading the mutable attribute.
- `Experiment.__init__` keeps `self._rng` as the "default stream" created from
  `seed`, so behaviour is unchanged when no explicit `rng` is passed and `run()`
  is called once — the same reproducible sequence.
- Repeated `run()` calls with `rng=None` now **advance** the shared default
  generator (they consume from it), so two bare calls produce different streams,
  matching what a caller expects from a stateful RNG. Reproducibility is
  preserved by passing an explicit `rng=np.random.default_rng(seed)` (which
  `run_experiment()` already does), or by re-creating the `Experiment`.

### How to thread it without mutating instance state

Two implementation choices; pick one in the plan:

- **(A) Scoped attribute with save/restore** — `run()` saves the previous
  `self._rng`, sets `self._rng = local_rng` for the duration, and restores it in a
  `finally`. Minimal diff (the four readers are untouched), but still a temporary
  mutation.
- **(B) Pass the generator explicitly** — add an `rng` parameter to `optimize()`
  and to the private helpers (`_criterion_keys`, `_sample_latent`,
  `calculate_epig_mc` already has access via `self`), threading `local_rng` down.
  Cleanest (no shared-state mutation) but touches more call sites, including the
  criterion-dispatch functions whose signature `run()` builds as
  `criterion_dict`.

**Recommendation: (B)**, because the whole point is that `run()` stops mutating
shared state. `calculate_epig_mc`/`calculate_random`/`calculate_epig`/`calculate_eig`
are all dispatched through `criterion_func(x, filtr=..., key=...)`; they draw
latent samples / keys via `self._rng`. Threading a single `rng` object into those
that need it (only `calculate_epig_mc` and `_sample_latent`) is a small change and
leaves `self._rng` as a pure default.

## Boundary decisions

1. **`run()` must not permanently rebind `self._rng`.** Whether via (A) or (B),
   the observable post-condition is: after `run()` returns, `self._rng` is the same
   object/state it had before (for (A)), or was only advanced if the default
   stream was used (acceptable). The chosen design is (B): no mutation at all when
   an explicit `rng` is passed; the default stream is simply consumed.
2. **Reproducibility is preserved.** Same `Experiment` + same explicit
   `rng=np.random.default_rng(s)` → identical run. Same `Experiment` + one `run()`
   call with `rng=None` → identical to today.
3. **`run_experiment()` behaviour is unchanged.** It already passes explicit
   generators (`self.seed` for warm-start, `self.seed + 1` per criterion), so the
   only change it sees is that `self._rng` is no longer rebound — and nothing
   outside `run()` reads `self._rng` between calls in that path.
4. **No public API break.** If (B) adds an `rng` keyword to `optimize()`, it must
   have a default (`rng=None` → fall back to `self._rng`) so existing callers and
   tests keep working.

## Success Criteria

- `run()` no longer executes `self._rng = ...`; `self._rng` is unchanged after a
  call that passes an explicit `rng`, and no `run()` path rebinds it.
- Two consecutive `run(..., rng=None)` calls on the same `Experiment` with the
  same settings produce **different** `selected_indices` (when randomness is
  exercised, e.g. the `RANDOM` criterion), proving they advance rather than replay.
- One `run(..., rng=np.random.default_rng(s))` call is reproducible: repeating it
  with a fresh generator seeded `s` gives identical `selected_indices` and
  criteria.
- `run_experiment()` results are unchanged (existing runner/mechanistic/EB tests
  still pass with the same assertions).
- Full suite `pytest tests/ -q -n auto` → 146 passed, 3 skipped (same as now) plus
  any new test; `ruff`/`ty`/`pymarkdown` green.

## Boundaries

Inherits the project's working boundaries. Delta only:

- **Always:** no permanent `self._rng` rebinding in `run()`; keep explicit-`rng`
  reproducibility; keep `rng=None` single-call behaviour identical to today.
- **Ask first:** none.
- **Never:** change `run_experiment()`'s seeding contract; break the
  `criterion_func(x, filtr=, key=)` dispatch signature without a defaulted
  `rng`.

## Out of Scope (Deferred)

- Deferred item 4 (`trace=True` deep-copies full covariance per epoch).
- Any change to how `run_experiment()` derives distinct seeds per criterion.
- Migrating the whole codebase off `self._rng` (only the `run()` path is fixed).

## Open Questions

1. **(A) scoped save/restore vs (B) explicit threading.** I recommend (B). Confirm
   you want the extra call-site threading over the smaller-diff save/restore.
2. **Bare double-`run()` semantics.** Confirm you want two `rng=None` calls to
   *advance* (different results) rather than replay — this is the behaviour change
   that makes the fix meaningful.
