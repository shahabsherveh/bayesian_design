# Spec Delta: Pool-aligned `grid_search` and consistent `optimize()` return

Status: implemented

## Objective

Fix the deferred item-2 defect — `optimize()` returns an int pool index for
`brute_force` but a design vector for `gradient_ascent`/`grid_search`, so
`used_indices` (re-selection avoidance) and `selected_indices` are silently inert
for the latter two. The chosen fix is to **make the continuous methods select
from the same pool as `brute_force`**, so a pool index exists natively and the
return shape is uniform — rather than bolting snap-to-nearest onto off-pool
designs.

This implements deferred item 2 from `docs/STATUS.md`.

## Current Behavior

`optimize()` returns:

| method | `best_index` | `x` |
|---|---|---|
| `brute_force` | integer pool index into `design_space` | `design_space[best_index]` |
| `gradient_ascent` | **the design vector itself** | same |
| `grid_search` | **the design vector itself** | same |

`design_space` is `data.x_train` (the training inputs) — the same array
`empirical_bayes_init` indexes (`x_train[idx]`) and the same set `exclude` masks.

`grid_search` draws `num_samples` **random points inside the bounding box** of
`design_space` (not pool points), takes the argmax, and returns that off-pool
design. Consequences:

1. `run()`'s guard (`jnp.ndim == 0 and integer dtype`) is false for both
   continuous methods → `used_indices` never grows, `selected_indices` stays empty.
2. `exclude` cannot prevent re-selection for them (same design picked repeatedly).
3. `ExperimentResults.selected_indices` is silently empty for those runs.
4. `observe()` can't resolve the off-pool design without an `underlying_model`,
   so the two methods only "work" when a truth model happens to exist.
5. `grid_search` is a *presentation* path (used by the `experiment_0` /
   `experiment_2` demo configs with `plot_inter_results = true`), but it invents
   off-pool data that the measurement pipeline cannot consume.

## New Behavior

`optimize()` returns `(best_index, x, crit_value)` with **`best_index` always an
integer index into `design_space`** and `x = design_space[best_index]` for all
three methods:

- **`brute_force`** — unchanged.
- **`grid_search`** — instead of sampling a random box, draw `num_samples`
  candidate positions **from the pool with replacement** (independent uniform
  indices into `design_space`), evaluate the criterion on those pool points, and
  return the argmax's **pool index**. `best_index` is the index actually
  evaluated, `x = design_space[best_index]`. With `exclude`, mask excluded indices
  to `-inf` before argmax (repeats of allowed indices are harmless); if every
  drawn candidate is excluded, fall back to evaluating the full non-excluded pool,
  and raise the existing "Every design in the pool has been observed." error only
  when the entire pool is excluded.
- **`gradient_ascent`** — remains a continuous optimization (that is its purpose:
  exploring the design box), but **snaps once at the end** to the nearest pool
  point: `best_index = argmin ||x - design_space||` (with excluded indices masked
  to `+inf`), `x = design_space[best_index]`, and `crit_value` recomputed at the
  snapped design. This is the single place a snap is needed.

`run()` then drops the defensive type guard and unconditionally records
`used_indices.add(int(best_index))` and `selected_indices.append(int(best_index))`.

### Why pool-sampling rather than snap-to-nearest for `grid_search`

`grid_search` is a *search* over a reduced evaluation budget. If its candidate
set is a subset of the pool, then:

- its result is a real, observable design (index exists natively — nothing to snap),
- `exclude`, `selected_indices`, `observe(index)`, and the EB path all work with
  no new conversion,
- the presentation configs still show design selection (now over real designs).

`gradient_ascent` is genuinely continuous (it follows gradients through the box),
so it cannot avoid an end-snap; that is handled explicitly and documented.

## Boundary decisions

1. **`best_index` is always a `design_space` index.** `x` is always
   `design_space[best_index]`. The two cannot disagree.
2. **`grid_search` samples candidates from the pool, with replacement.** Its
   `num_samples` parameter is the number of draws; sampling uses a fixed key for
   reproducibility (keep `jax.random.key(0)` to preserve determinism of the demo).
   Repeats are allowed; excluded indices are masked before argmax.
3. **`gradient_ascent` snaps once at the end** (nearest non-excluded pool point)
   and recomputes `crit_value` there so `crit_values` match the returned design.
4. **`exclude` applies to both** after index resolution (mask before argmin).
5. **No change to `observe()`, `calculate_*` criteria, or `ExperimentResults`.**
   `selected_indices` simply becomes non-empty for the continuous methods, which
   matches its documented meaning.

## Success Criteria

- `optimize()` returns an integer `best_index` and `design_space[best_index] == x`
  for all three methods (one focused test per method).
- `run()` records `len(selected_indices) == epochs` for each method (new test on a
  2–3 epoch run with `optimizer="grid_search"`).
- `exclude` prevents re-selection: with all-but-one index excluded, both
  `brute_force` and `grid_search` return the remaining index; `gradient_ascent`
  returns a non-excluded index.
- `grid_search`'s returned design is a member of `design_space` (assert via
  exact match against `design_space[best_index]`).
- Existing demo behaviour for `experiment_0`/`experiment_2` still produces plots
  (the config smoke tests in `config_test.py` still pass).
- Full suite `pytest tests/ -q -n auto` → 139 passed, 3 skipped **plus** new
  tests; existing tests unchanged and passing.
- `ruff` / `ty` / `pymarkdown` green.

## Boundaries

Inherits the project's working boundaries. Delta only:

- **Always:** return pool indices; sample `grid_search` candidates from the pool;
  snap `gradient_ascent` once; keep the demo configs working.
- **Ask first:** none.
- **Never:** return off-pool designs; change `observe()`; make `best_index`
  sometimes an array.

## Out of Scope (Deferred)

- Deferred items 3 (`_rng` reseeding in `run()`) and 4 (`trace=True` deep-copy).
- Making `grid_search` a high-resolution optimizer or changing its role beyond
  pool sampling.
- Any change to `calculate_epig*` / `calculate_eig` / `calculate_random`.

## Open Questions

1. **`grid_search` sampling with replacement.** Draw `num_samples` pool indices
   uniformly **with replacement** (independent uniform integers into
   `[0, len(design_space))`), evaluate the criterion on those points, and take the
   argmax. This is the confirmed default.

   Consequence for `exclude`: with replacement, a sampled index may repeat, and some
   of the drawn indices may be excluded. The rule is: mask excluded indices to
   `-inf` **before** the argmax, so repeats of an allowed index are harmless and
   excluded points can never win. If *every* drawn candidate is excluded (possible
   only when the pool is almost fully excluded and the draw happened to hit only
   excluded indices), fall back to evaluating the full non-excluded pool so the
   method still returns a valid selection; raise "Every design in the pool has
   been observed." only when the entire pool is excluded.

2. **`gradient_ascent` leaving the box** — it already prints "Design out of
   bounds, stopping optimization." and breaks; snapping to the nearest pool point
   of the last in-bounds iterate is the natural behaviour. No action needed.
