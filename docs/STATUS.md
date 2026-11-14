# Repository status

Snapshot taken at `b2dd7cd` ("Removes mergetool artifacts", 2026-10-26), on `main`,
65 commits, clean working tree apart from untracked tooling output. Follow-up work was
done in `4b2cbd8` (deleted `copilot-instructions.md`) and uncommitted changes to
`src/bed/_experiments.py`; see "Verified state".

This document records **what is verifiable** and what could not be verified. Anything
marked *unverified* was not executed in the environment where this snapshot was taken
(see "Verification limits").

## Summary

`bed` is an active research library for Bayesian experimental design. The library surface
is plausible and well structured; the code is organized around a layered public API over
private implementation modules, and it has no import cycles. The test suite passes
(138 passed, 3 skipped) after two release-blocking defects found and fixed during this
review — the package did not import at all, and Monte Carlo EPIG crashed for multi-output
models. The dominant remaining risk is **reproducibility**: the experiment layer and one
test file are organized around a planning document (`PROGRESS.md`) that does not exist
anywhere in the repository or its history.

## What is in place

### Library (`src/bed`, ~4.1k lines excluding vendored compat modules)

| Area | Modules | Notes |
| --- | --- | --- |
| State estimation | `ekf.py`, `ukf.py`, `filters/` | EKF + UKF; `filter_type` switch on `Experiment` |
| Design criteria | `gaussian_design.py`, `quadrature.py`, `reference.py` | Closed-form Gaussian EIG/EPIG, set EIG, mode-centred quadrature, MCMC/IS reference posterior |
| Sequential design | `sequential.py`, `adaptive.py`, `toy.py` | Greedy/random policies, value-of-adaptation toy |
| Models | `_models.py` (1298), `models/` | Classical, GP, Flax/NNX neural models |
| Mechanistic | `mechanistic.py` | PK/PD and DIVIDE models |
| Inference helpers | `empirical_bayes.py`, `stats.py`, `diagnostics.py` | EB init/MAP, bootstrap CIs, filter moment checks |
| Operations | `cli/main.py`, `provenance.py` | Config-driven CLI; run manifests with source revision and config hash |
| Experiments | `_experiments.py` (843), `experiments/` | Runner and result containers |

Public namespaces (`bed.models`, `bed.experiments`, `bed.filters`) are thin re-export
shims over `_models` / `_experiments`. This is a deliberate, documented staging step
(`# refactor` comments in the shims) and imports remain stable for callers.

### Tests

131 test functions across 21 test modules, including what look like hypothesis-tracking
names (`test_h02_monte_carlo_epig_bias_shrinks_with_the_inner_sample_size`) and
"corrections" regressions (`corrections_test.py`), suggesting prior bugs were pinned
with tests rather than fixed silently.

### Experiments

6 TOML configs (`experiment_0`–`experiment_5`) plus a separate, script-driven
investigation layer (PK/PD candidate A, DIVIDE candidate D) with pre-flight gates,
falsification runs, fidelity checks, and analysis/aggregation scripts. The investigation
layer writes versioned manifests and uses atomic, idempotent result writes
(`provenance.py`) — a notably careful design for research code.

## Issues found

Ordered by impact on the ability of an outside reader to trust or reproduce results.

1. **`PROGRESS.md` is missing but load-bearing.** 11 files reference it, including
   `tests/diagnostics_test.py` and every script in the investigation layer. It defines
   hypotheses (H0.2, H0.3, H1.1), pre-flight gates (G1–G4), registered predictions, and
   candidates A/D. It was **never committed on any branch** (`git log --all` is empty for
   it), so it was kept outside version control. As a result: pre-registered predictions
   cannot be checked against reality, "calibrated against PROGRESS.md A-5" comments are
   unverifiable, and a reader cannot tell what was predicted before a run versus after.

2. **No environment in this workspace can run the suite.** `.venv/` exists but is
   empty (0 entries in `site-packages`, no `pip`). Neither `numpy` nor `pytest` is
   importable from any interpreter on the path, so **no test, import, or experiment was
   executed** for this snapshot. Correctness of the 131 tests is therefore *unverified*
   here — not known-failing, just not demonstrated.

3. **`.github/copilot-instructions.md` documents a test file that does not exist.**
   It tells contributors to run `pytest tests/experiment_test.py::TestExperiment1::test_run`.
   There is no `tests/experiment_test.py`; it was removed in an earlier refactor while the
   instructions were left behind.

4. ~~**Test-time debugger configuration.**~~ **Fixed.** `pyproject.toml` had
   `addopts = "--pdb --pdbcls=IPython.terminal.debugger:TerminalPdb -s"`, so any failing test
   dropped into an interactive debugger instead of exiting non-zero, making the suite
   unusable in CI. Reduced to `addopts = "-s"`; the unused module-scope `import ipdb` was
   dropped from `tests/linear_test.py`. Verified: a failing test now exits `1` in 0.04s, and
   the passing suite exits `0`.

5. **`results/` is referenced by 10 scripts but absent, and its raw output is ignored.**
   Scripts default to `--out results/...`, and the `.gitignore` excludes `results/**/raw/`
   with a rationale that summaries and figures are kept. No `results/` tree is present in
   the checkout and none is tracked, so the claim that summaries/figures are kept is not
   currently true in the repository.

6. **Two configs have no wrapper.** `experiment_4` and `experiment_5` exist in
   `experiments/configs/` but no `experiments/experiment_4.py` / `_5.py` wrapper, unlike
   0–3. They are reachable only via `bed experiment --config experiment_4`. The README's
   experiment table does not mention configs 4/5 at all.

7. **README overstates the config-driven layer.** The README states the `experiments/`
   wrappers "now delegate to the same CLI config execution". This is true for
   `experiment_0`–`experiment_3`, but the other ~14 scripts in `experiments/`
   (`experiment_2_1`…`_2_4`, `experiment_2_skew`, `experiment_mc`, `computation_cost`,
   and the whole PK/PD + DIVIDE investigation layer) construct models directly in Python
   and are not config-driven. The README's variant table lists only a subset.

8. ~~**`LICENSE` file is missing.**~~ **Fixed.** Added an MIT `LICENSE`. The build no longer
   emits `Pattern 'LICEN[CS]E*' did not match any files`, and the wheel now bundles the text
   at `bed-0.0.1.dist-info/licenses/LICENSE`. The copyright line names both authors with
   commits in the history (Shahab Sherveh and Jens Sjölund, 21 commits).

9. **No CI.** `.github/` contains no Actions workflow (the `copilot-instructions.md` file was
   deleted in `4b2cbd8`). Nothing currently prevents a red suite from
   landing, though with `--pdb` removed a workflow can now detect it.

11. **`matplotlib` is an undeclared dependency.** It is imported lazily inside
    `_models.py` (plotting methods) but is absent from `requirements.txt`, so 4 tests fail
    with `ModuleNotFoundError` on a clean install. Either declare it or guard the imports
    like the plotting code does elsewhere.

## Verified state

A full suite run was completed after fixing the issues below. Environment: Python 3.12.16,
all of `requirements.txt` except `tensorflow`/`tensorflow-datasets` (572 MB, only needed by the
lazily-imported MNIST loader), plus `matplotlib`.

```
138 passed, 3 skipped in 808s
```

The 3 skips are the `pytest.importorskip("tensorflow_datasets")` guards in `data_test.py`
and `config_test.py`, i.e. expected in this environment and not failures.

### Fixed during this review

- **`src/bed/_experiments.py` was unimportable.** A leftover mergetool marker on line 296
  (`>>>>>> > 538fd2e(Improves MC method code)`) was a `SyntaxError`, so `import bed` failed
  at `HEAD`. Removed.
- **Monte Carlo EPIG crashed for multi-output models.** `calculate_epig_mc` computed a dead
  block of samples using `scale=np.sqrt(self.measurement_error)` — the full 2x2 covariance
  matrix broadcast against a 6-D `size=`, which raises `ValueError` for any model with more
  than one output. The same merge had left these arrays written but never read; the live
  computation starts below and already uses the diagonal correctly. Removed the dead block.
  `test_monte_carlo_epig_converges_to_the_closed_form_with_two_outputs` and
  `test_monte_carlo_epig_runs_and_tracks_the_closed_form` now pass.

The empty `.venv/` was bootstrapped with `python -m ensurepip --upgrade` and
`pip install --no-deps -e .` (no-deps avoids pulling TensorFlow).

## Verification limits

Still not executed in this snapshot:

- `tensorflow`-dependent paths (`get_mnist_data`, `tests/data_test.py::test_mnist_data`)
- any `bed` CLI invocation or experiment script end-to-end (the config tests do run a short
  `experiment_0`)

The suite now runs non-interactively with plain `pytest tests/`; the
`--override-ini addopts=""` workaround is no longer needed.

## Suggested next steps

1. ~~Provision the environment and record a baseline.~~ Done: 138 passed, 3 skipped.
2. ~~Move `--pdb` out of `addopts`; drop the module-scope `import ipdb`.~~ Done.
3. Recover or rewrite `PROGRESS.md` and commit it. If the pre-registered predictions
   still exist somewhere outside git, commit them unchanged with their original dates so
   the prediction/proof ordering stays auditable.
4. Add wrappers for configs 4 and 5, or drop them.
5. ~~Add `LICENSE` (MIT).~~ Done. Remaining: add a minimal GitHub Actions workflow running
   the suite; it can now rely on pytest's exit code.
6. Declare or guard `matplotlib` in `requirements.txt`.
7. Decide whether `results/` summaries should be tracked; if so, commit them, otherwise
   adjust the `.gitignore` comment so it does not claim they are kept.
