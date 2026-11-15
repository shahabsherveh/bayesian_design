# Project: bed — Bayesian Experimental Design

Research-oriented Python library for Bayesian experimental design: classical linear and
Gaussian-process methods, neural measurement models, and sequential design driven by an
EKF/UKF.

## Tech stack

- Python **3.12+** (`requires-python = ">=3.12"`), packaged as `bed` (v0.0.1, MIT)
- **JAX** (CPU-only is fine), **Flax**/NNX neural models, NumPy, SciPy, scikit-learn,
  cvxpy, optax, typer (CLI), matplotlib, tqdm
- **TensorFlow** + `tensorflow-datasets` only for the MNIST loader — lazily imported,
  optional for synthetic work (do not require them to import or run `bed`)

## Commands

Setup:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt   # pytest, linters, pre-commit
```

- **Test (full):** `pytest tests/ -q -n auto` — parallel; ~256s. Serial is ~13 min
  (JAX recompilation dominates; do not reduce seed/sample counts to speed it up — flaky).
- **Test (focused):** `pytest tests/<file>.py::<test>`
- **Lint:** `ruff check src/ tests/ experiments/`
- **Type:** `ty check src/ --exit-zero-on-warning`
- **Markdown:** `pymarkdown --config .pymarkdown.json scan`
- **All fast gates:** `pre-commit run --all-files`
- **CLI:** `bed list-configs` · `bed experiment --config experiment_0 [--no-plot | --dry-run]`

## Code conventions

- **Dense research style**: long lines, semicolons, compact statements. `line-length = 120`.
- Ruff lint is deliberately **narrow correctness-only**: `E4 E7 E9 F W6 PLE PIE790 RUF046
  RUF022 I001`. Stylistic rules (`E401 E402 E501 E701 E702 E731 E741`) are ignored on
  purpose — the codebase was never formatter-normalized and enforcing them would mean a
  mass rewrite.
- Public namespaces (`bed.models`, `bed.experiments`, `bed.filters`) are **thin re-export
  shims** over `_models`/`_experiments` (staging step marked `# refactor`). Keep imports
  stable for callers; do not remove the shims casually.
- Config-driven experiments: `experiments/configs/*.toml` → `bed.cli.main`. Some scripts in
  `experiments/` still build models directly in Python (not config-driven).

## Boundaries (do not cross without an explicit decision)

- **Never mass-reformat.** The tree has ~1400 `E501` (line-too-long) and hundreds of `E702`
  (semicolon) violations; enabling stylistic ruff rules would bury real defects. A
  deliberate `ruff format` pass is a separate decision, not an incidental change.
- **Do not weaken the quality bar.** `ruff` clean, `ty` exit 0 (scoped to `src/`; the
  `tests/`/`experiments/` diagnostics are intentionally not gated), `pymarkdown` clean,
  full `pytest` green. The `ty` gate catches *new* errors; its ~70 `src/` warnings are known
  typing gaps (attributes attached to dataclasses post-construction, unannotated
  `self.model`) and must not be papered over with `noqa`/`type: ignore`.
- **`main` is protected.** Direct commits are blocked by the `no-commit-to-branch` hook.
  Work on a short-lived branch (`feat/`, `fix/`, `refactor/`, `chore/`) and merge via PR.
- **Don't commit raw run output.** `results/**/raw/` is gitignored; the `results/` tree is
  absent from the checkout.
- **TensorFlow paths are optional** and not exercised in the standard environment; treat
  them as lazy and guard/skip where possible.

## Known gotchas

- `.venv/` may be empty/broken in a fresh checkout. Bootstrap: `python -m ensurepip
  --upgrade` then `pip install --no-deps -e .`.
- README's "Run tests" section is **stale**: pytest `addopts` is now just `-s` (no IPython
  debugger; the module-scope `import ipdb` was removed). Trust `pyproject.toml`, not README.
- `experiment_4`/`experiment_5` configs exist but have **no wrapper script** (unlike 0–3);
  reachable only via `bed experiment --config experiment_4`.
- `calculate_rmse` averages over the batch (docstring states the batch contract).
- `ISResult` sets some attributes after construction; several classes leave `self.model`
  unannotated — this is the source of the 70 `ty` warnings, not bugs.

## Orientation & handoff

- `docs/STATUS.md` — repository status, issues found, verified state, suggested next steps.
- `tasks/plan.md` + `tasks/todo.md` — current task plan and checklists (deferred backlog).
- `SPEC-*.md` — approved spec deltas (one per shipped change).
- `graphify-out/graph.json` — project knowledge graph; `graphify query "<question>"` for
  file/relation lookups (refresh with `graphify update` when it is stale).
