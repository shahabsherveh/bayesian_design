# Spec Delta: Fix stale docstrings in `src/bed/_experiments.py`

Status: implemented

## Objective

Correct the `__init__` (and a few adjacent method) docstrings in
`src/bed/_experiments.py` so they describe the parameters that actually exist,
rather than a set that no longer does. This is **documentation-only** — no
signature, body, or behaviour changes.

This implements deferred item 1 from `docs/STATUS.md` ("Stale `Experiment` /
`ExperimentResults` docstrings listing non-existent parameters").

## Current Behavior

`Experiment.__init__` takes, in order: `latent_cov`, `latent_innovation`,
`measurement_error`, `data`, `model`, `latent_var`, `warm_start`,
`pre_train_model`, `training_kwargs`, `plot_inter_results`, `seed`, `init`,
`empirical_bayes_kwargs`, `latent_mean`.

Its docstring instead documents a nonexistent, older API: `latent_dim`,
`latent_true`, `design_cov`, `design_mean`, `design_pool_num`,
`plot_results`, `filter_type` — none of which are parameters (or even
attributes) of `Experiment`. The class docstring's **Attributes** block has the
same problem: it lists `design_dist`, `latent_true`, `ekf`, and `plot_results`,
none of which exist on the instance.

`ExperimentResults.__init__` takes, in order: `sd_glob`, `sd_pool`,
`rmse_global`, `rmse_pool`, `selected_designs`, `crit_values`, `data`,
`crit_label="EPIG"`, `filter_type="ekf"`, `design_space=None`, `filters=[]`,
`selected_indices=None`, `rmse_pool_outputs=None`, `rmse_glob_outputs=None`.

Its docstring instead documents `rmse_params_values`, `rmse_values`, `designs`,
`grad_lists` — none of which are parameters. The class docstring's
**Attributes** block lists `rmse_params_values`, `rmse_values`, `designs`,
`crit_values`, `grad_lists`; only `crit_values` matches a stored attribute.

`optimize()`'s docstring documents `criterion_label`, `max_iters`,
`learning_rate`, `x_init`, `tol`, `stochastic`, `x_init_type`, and describes a
return of `(x_opt, crit_value, grads)` — none of which match the real signature
`(criterion_func, method, filtr, params=None, exclude=None)` and its actual
three-value return `(best_index, x, crit_value)`.

## New Behavior

Each docstring is rewritten to describe the parameters and attributes that
exist, in signature order, with accurate types and one-line descriptions. Where
a method already has a correct docstring (e.g. `calculate_rmse` was fixed in
the prior pass, `run_experiment`), it is left untouched.

Specifically:

1. `Experiment.__init__` docstring — documented parameters are the actual
   signature, in order. Note the `init` values (`"filter"` / `"empirical_bayes"`)
   and that `latent_cov`/`latent_var` are mutually exclusive fallbacks for the
   prior covariance.
2. `Experiment` class **Attributes** block — lists only real instance
   attributes set in `__init__` (e.g. `model`, `data`, `design_pool`,
   `design_space`, `true_measurements`, `measurement_error`,
   `latent_innovation`, `warm_start`, `seed`, `init`, `empirical_bayes_kwargs`).
3. `ExperimentResults.__init__` docstring — documented parameters are the
   actual signature, in order.
4. `ExperimentResults` class **Attributes** block — lists only real stored
   attributes.
5. `optimize()` docstring — documents the real signature and the real return
   `(best_index, x, crit_value)`, including the per-method meaning of
   `best_index` (an integer pool index for `brute_force`; the design vector
   itself for `gradient_ascent` / `grid_search`).

## Success Criteria

- `Experiment.__init__` docstring names exactly the signature parameters, in
  order, and no others.
- `Experiment` Attributes block names no attribute that is not set in
  `__init__`.
- `ExperimentResults.__init__` docstring names exactly the signature
  parameters, in order, and no others.
- `ExperimentResults` Attributes block names no attribute that is not stored
  in its `__init__`.
- `optimize()` docstring names the real five parameters and the real return
  tuple.
- **No code changes:** `git diff --stat` shows only comment/docstring lines
  added/removed in `src/bed/_experiments.py`; every signature, default, and
  body line is byte-identical.
- All existing gates still pass (see Verification) — the result must be
  byte-identical to the current green state, since this is documentation-only.

## Boundaries

Inherits all boundaries from the project's working conventions (ruff `E4/E7/E9/F/W6/PLE`
and its mechanical rules, ty scoped to `src/` with `--exit-zero-on-warning`, pymarkdown
per `.pymarkdown.json`). The only delta:

- **Always:** documentation must match the real signature, in order, without
  inventing parameters.
- **Ask first:** none — no new dependencies, no schema/config changes.
- **Never:** change a signature, default value, or any executable line to make
  a stale docstring "true" — the docstring yields to the code, not the reverse.
  Also never rename a parameter to match a stale docstring.

## Out of Scope (Deferred)

- All four remaining deferred items from the review (`_models.py` split,
  `optimize()` return-shape inconsistency, `_rng` reseeding, `trace=True`
  memory).
- Fixing the `calculate_epig` / `calculate_eig` / `build_design_space` /
  `build_prior` docstrings, which also use stale parameter names (`x_0`,
  `design_num`, `latent_dim`, `latent_variance`) but whose *signatures/locals*
  are small enough to already be unambiguous where they are used. Only the
  three `__init__`s and `optimize()` are named in the deferred item; the
  helpers can be a follow-up if desired.

## Verification

- [ ] `python -m pytest tests/ -q -n auto` → 139 passed, 3 skipped (unchanged)
- [ ] `ruff check src/ tests/ experiments/` → clean
- [ ] `ty check src/ --exit-zero-on-warning` → exit 0, no new warnings
- [ ] `pymarkdown --config .pymarkdown.json scan $(git ls-files '*.md')` → clean
- [ ] `git diff --stat` shows only `src/bed/_experiments.py` docstring lines;
      no signature/body line changed (inspect the diff directly)
