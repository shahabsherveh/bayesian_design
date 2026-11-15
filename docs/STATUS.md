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
models. The investigation layer's outputs are exploratory rather than confirmatory, because
the planning document its registered predictions lived in was never committed; the dangling
references to it have since been removed (see issue 1).

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

1. ~~**`PROGRESS.md` is missing but load-bearing.**~~ **Resolved by removing the references.**
   11 files referenced it, including `tests/diagnostics_test.py` and every script in the
   investigation layer, citing hypotheses (H0.2, H0.3, H1.1), pre-flight gates (G1–G4),
   registered predictions, and candidates A/D. It was never committed on any branch
   (`git log --all` is empty for it, and it was never added-then-deleted), so the references
   were dangling. The dangling identifiers were stripped and the surrounding technical
   content kept; the scripts still carry their labels (A-2, A-4, D-2, S1–S5, P1–P3,
   G1–G4, H0.2/H0.3/H1.1) but no longer point at a document that does not exist.

   **Consequence for readers:** the investigation layer's outputs can no longer be
   presented as pre-registered. Whatever predictions were written down elsewhere, nothing in
   this repository establishes that they predate the runs, so results from
   `sequential_pkpd.py`, `sequential_divide.py` and their analysis scripts should be read as
   exploratory, not confirmatory. If the original document is ever recovered, commit it
   unchanged with its original dates to restore that property.

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

9. ~~**No CI.**~~ **Fixed.** `.github/workflows/tests.yml` runs on push to `main`, on pull
   requests, and on manual dispatch. It has two jobs so that a lint failure reports in
   seconds rather than waiting behind the five-minute suite:
   - **`lint`** — installs only ruff and ty, then runs `ruff check src/ tests/ experiments/`
     and `ty check src/ --exit-zero-on-warning`. Takes about a second locally.
   - **`test`** — Python 3.12, CPU-only JAX, `pytest tests/ -q -n auto`.

   Verified locally against the exact commands (both exits 0), and each gate was confirmed
   to exit 1 on a deliberate violation. Not yet observed running on GitHub's runners.

10. **The fast gates also run in a pre-commit hook.** `.pre-commit-config.yaml` runs ruff, ty
    and pymarkdown before every commit, which is a better place for them than CI: they take
    about a second combined, while the test suite takes about five minutes even parallelised
    and is deliberately left to CI.

    Install once per clone: `python -m pip install pre-commit && pre-commit install`.
    `pre-commit` is in `requirements.txt`, so `pip install -e .` already provides it.

    `ruff` and `ty` were previously missing from `requirements.txt` (CI pip-installed ruff
    inline and ty was only present as a global binary), so a fresh clone could not run
    either check. They are now declared, along with `pymarkdownlnt`, which is what lets the
    hook and CI use the same versions as each other.

    Verified the hook blocks a commit: staging a file with an undefined name fails both
    hooks and `git commit` exits 1 with no commit created. `pre-commit run --all-files`
    passes on the current tree.

11. ~~**`matplotlib` is an undeclared dependency.**~~ **Fixed.** Declared in
    `requirements.txt`, alongside `pytest-xdist`. A clean install now resolves the four
    tests that previously failed with `ModuleNotFoundError`.

12. **Linting was absent; now gated, with a deliberately narrow rule set.** `pyproject.toml`
    had no `[tool.ruff]` section, so nothing was checked. A ruff gate was added selecting
    correctness rules (`E4`, `E7`, `E9`, `F`, `W6`, `PLE`) plus the mechanical cleanups that
    carry no behaviour change (`I001`, `RUF022`, `PIE790`, `RUF046`). The stylistic rules are
    ignored on purpose: even at the configured `line-length = 120` the codebase yields 290
    `E501` (line-too-long) and 528 `E702` (semicolon) violations — 1415 `E501` at ruff's
    default width — i.e. it has never been formatted by a tool and uses a dense
    research style. Enforcing those would mean rewriting the analysis code and would bury
    real defects under noise. Turning them on would be a separate, deliberate reformatting
    decision.

    Applied so far: 40 dead imports and 5 unused variables, then 62 more mechanical fixes
    (import splitting, `__all__` ordering, redundant `int()`/`pass`). All were verified
    non-behavioural: no `__init__` re-exports or side-effect imports were touched, the three
    unused variables ruff classes as unsafe to autofix (`adaptive.py` `ksum`, `ekf.py`
    `H_pred_T`, `stats.py` `n`) were reviewed by hand, and the full suite was re-run after
    each stage (138 passed, 3 skipped).

    Two rules are deliberately **not** enforced:
    - `W293` (blank line contains whitespace). The remaining 24 hits are all inside
      docstrings, where ruff classifies the fix as unsafe because it changes the literal.
      Not worth rewriting docstring bytes for invisible characters.
    - The `--unsafe-fixes` set (`SIM115`, `B905`, `B007`, `RUF059`, 24 remaining). These
      change runtime behaviour — `zip(strict=True)` raises on a length mismatch, for
      instance — so they need review rather than a blanket `--fix`.

    Two `noqa` comments were added where fixing would have broken something:
    `analyze_sequential_pkpd.py` sets `matplotlib.use("Agg")` before importing pyplot, and
    `laplace_refit_pkpd.py` imports after a `sys.path` insert. Both are order-dependent;
    the backend was verified to still resolve to `Agg`.

13. **Type checking was absent; now gated on `src/` with the codebase's real error count at
    zero.** `ty` 0.0.104 reports **201 diagnostics** across the repository. Gating on all of
    them would block every commit, so the gate is scoped to the library and runs with
    `--exit-zero-on-warning`:

    | Path | errors | warnings | gated? |
    | --- | --- | --- | --- |
    | `src/` | 0 *(2 before `import-not-found` was downgraded)* | 70 | yes |
    | `tests/` | 8 | 8 | no |
    | `experiments/` | 25 | 33 | no |

    The `tests/` and `experiments/` errors are not defects: 4 are `import-not-found` for
    sibling scripts imported after a `sys.path` insert, and the rest are Optional-narrowing
    patterns (`assert x is not None` then `x[...]`) that ty cannot follow. Annotating those
    away would add noise to research scripts for no safety gain.

    The 70 remaining `src/` warnings are real typing gaps, not bugs: `ISResult` is a
    dataclass whose attributes (`method`, `rhat_max`, `ess_bulk_min`) are attached after
    construction, and several classes leave `self.model` unannotated so it infers as
    `object`. Fixing them means introducing annotations and possibly a protocol — a
    deliberate typing pass, not a CI change. They are downgraded to warnings so the gate
    catches **new** errors (undefined names, bad argument types that are not already
    known) without demanding that rewrite.

    Verified the gate can fail: injecting `return undefined_name` into `src/` produces
    `error[unresolved-reference]` and exit 1; the clean tree exits 0.

14. **Markdown was unchecked; now linted.** `pymarkdownlnt` 0.9.41 is wired into both the
    `lint` CI job and the pre-commit hook. The initial scan found 92 problems in the three
    markdown files, of which 90 were `MD013` (line length, default 80) — the same judgment
    call as ruff's `E501`, so `MD013` is disabled in `.pymarkdown.json` rather than wrapping
    every paragraph. Note that pymarkdown does **not** auto-discover that file, so both the
    hook and CI pass `--config .pymarkdown.json` explicitly.

    The two real findings were both in this document, introduced by earlier edits here: an
    ordered-list prefix out of sequence (`14.` sitting between `9.` and `11.`, now
    renumbered 1-13 with no gaps) and a fenced code block with no language (`MD040`). Both
    are fixed. The remaining rules do catch real problems — a bare fence, list numbering,
    missing space after `#` — verified by feeding pymarkdown a deliberately malformed file.

15. **Autofix runs only on demand, and only for the tools that can actually fix.**
    `.github/workflows/autofix.yml` is `workflow_dispatch` only — deliberately not
    push-triggered, since a workflow that writes to the repository and is triggered by
    pushes can loop. It runs `ruff check --fix` and `pymarkdown fix`, then **re-runs the
    full gate set** on the result before pushing, so a fixer that breaks something cannot
    land. It defaults to opening a PR from an `autofix/<branch>-<timestamp>` branch;
    unchecking `open_pr` commits straight to the branch instead.

    `ty` is deliberately **not** in the autofix workflow. Its `--fix` cannot rephrase type
    errors: verified on this codebase it reports `0 fixed, 72 remaining`, and on a minimal
    case (`x: int = "not an int"`) likewise `0 fixed`. Including it would imply a repair
    that does not happen.

    Also verified: an `F841` unused local (which ruff classes unsafe) leaves the tree
    non-clean after `--fix`, so the workflow stops at the verify step and pushes nothing.
    Both paths were exercised locally — fixed-and-verified, and fixed-but-still-dirty.

    Note it needs `AUTOFIX_TOKEN` (or the default token) to push; the built-in
    `GITHUB_TOKEN` cannot push to a protected `main`, so the PR path is the default.

## Verified state

A full suite run was completed after fixing the issues below. Environment: Python 3.12.16,
all of `requirements.txt` except `tensorflow`/`tensorflow-datasets` (572 MB, only needed by the
lazily-imported MNIST loader), plus `matplotlib`.

```text
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

## Test suite runtime

A serial run takes about **13 minutes**; five tests account for roughly 365s of it:

| Test | Serial | Cause |
| --- | --- | --- |
| `corrections_test.py::test_divide_truth_distribution_matches_the_belief_prior_across_seeds` | 182s | 600 `divide_problem` constructions, ~220ms each |
| `adaptive_test.py::test_the_rectangle_expands_until_the_frame_is_empty` | 83s | dense adaptive quadrature |
| `adaptive_test.py::test_the_order_gate_refuses_to_certify_a_density_it_cannot_resolve` | 58s | same |
| `adaptive_test.py::test_a_low_order_reports_convergence_at_a_grossly_wrong_value` | 42s | same |
| `mechanistic_test.py::test_runner_runs_the_pkpd_problem_with_both_filters_and_reports_per_output_rmse` | 39s | sequential runner |

The dominant cost is **JAX recompilation**, not the statistics: `make_divide_problem` derives its
key with `jax.random.PRNGKey(seed)`, which is a distinct traced constant per seed, so each call is
a cold XLA compile (~130ms of the ~220ms). `cProfile` shows `backend_compile_and_load` taking
roughly half the wall time even after warmup.

**Use `pytest tests/ -n auto` (or `-n 4`).** Measured on 8 cores: 253s versus 800s serial, same
138 passed / 3 skipped, with no change to what is asserted. Reducing the seed count was tested and
rejected: at 50 seeds the statistic still passes but `max|sd - 1| = 0.1229` against a `0.15`
tolerance, leaving only 18% headroom, which would be flaky across machines.

A structural fix (cache the compilation via `fold_in`-based keys and static shapes) would help
further but touches core model code and was not attempted.

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
3. ~~Recover or rewrite `PROGRESS.md` and commit it.~~ Resolved by stripping the dangling
   references instead; see issue 1 for what that costs.
4. Add wrappers for configs 4 and 5, or drop them.
5. ~~Add `LICENSE` (MIT) and a GitHub Actions workflow.~~ Done; the workflow splits a fast
   `lint` job (ruff + ty) from the `test` job.
6. ~~Declare or guard `matplotlib` in `requirements.txt`.~~ Done, along with `pytest-xdist`,
   `pre-commit`, `ruff` and `ty`.
7. ~~Run the lint gates before commit rather than only in CI.~~ Done via
   `.pre-commit-config.yaml`; see issue 14.
8. Decide whether `results/` summaries should be tracked; if so, commit them, otherwise
   adjust the `.gitignore` comment so it does not claim they are kept.
9. Optionally enable the stylistic ruff rules (`E5`, `E7`, `I`, `UP`) after a deliberate
   `ruff format` pass. Not done here because it rewrites most of the analysis code.
10. Optionally do a typing pass on `src/` to clear the 70 `ty` warnings, then flip the gate
    to `--error-on-warning`. The main items are annotating `self.model` on the
    `ISResult`/`Posterior` paths and declaring the attributes set after construction.

## Follow-up: post-review fixes

A whole-repository review at `4cfb16f` re-ran the gates and found three defects the gates
did not catch. Fixed on branch `fix/review-critical-findings`:

| Defect | Fix |
| --- | --- |
| `run_experiment()` default `criteria` named `"RAND"`, which is not a key of `run()`'s criterion dict, so a bare call raised `ValueError` before doing any work | Default is now `EPIG`, `EIG`, `RANDOM`; new `test_run_experiment_with_default_arguments` covers the bare call. `test_unknown_criterion_raises` still asserts `"RAND"` is rejected, so the bad name cannot return. `EPIG-MC` is deliberately excluded from the defaults because its nested Monte Carlo estimate costs `num_latent_samples` evaluations per pool point; it is still available via `run()` and `--strategies` |
| `bed.models` re-exported `scipy.stats.multivariate_normal` and listed it in `__all__`, making a third-party symbol part of the public model namespace | Import and `__all__` entry removed; `tests/gp_test.py` imports scipy directly |
| `calculate_rmse`'s docstring promised "covariance matrix or batch" but the body averaged over the batch, so the documented single-matrix contract was undocumented behaviour | Body unchanged (the `.mean()` is load-bearing: `measurement_prior` returns a batch `(n, d_y, d_y)`); docstring now states the batch contract and the mean-over-batch semantics |

Mutable default arguments in `run()`, `optimize()` and `run_experiment()` were also
converted to `None` with in-body defaults.

Verification after the fixes: **139 passed, 3 skipped** (256s with `-n auto`), `ruff`
clean, `ty` exit 0 with the same 72 warnings (no new ones), `pymarkdown` clean. The
predicted count in the task plan was 139.

Two review claims were wrong and are corrected here rather than acted on:

- "`multivariate_normal` has zero references" was wrong: `tests/gp_test.py` imported it
  from `bed.models`. The grep used a single-line pattern and missed a multi-line import.
  The leak was still a real defect; only the supporting evidence was overstated.
- "`calculate_rmse`'s `.mean()` is a no-op" was wrong. Both filters return a batched
  `measurement_prior` covariance `(n, d_y, d_y)`, so `trace` yields a vector and `.mean()`
  collapses it. Removing it broke 7 runner tests. It is a documentation defect, not a
  logic defect.

Still open from the review (not done here): the `optimize()` return-shape
inconsistency (int index vs design vector) that makes `used_indices` inert for
`gradient_ascent`/`grid_search`, `_rng` being reseeded inside `run()`, and
`trace=True` deep-copying a full covariance per epoch. Two items were resolved in
follow-up deltas: the stale `Experiment`/`ExperimentResults` docstrings
(`SPEC-stale-docstrings.md`) and splitting `_models.py` into per-family modules
(`SPEC-split-models.md`) — both documentation/structure-only, suite unchanged at
139 passed, 3 skipped. See `tasks/plan.md`.
