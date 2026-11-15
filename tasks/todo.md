# Task list

Plan: [tasks/plan.md](plan.md). Spec: [docs/specs/SPEC-typing-cleanup.md](../docs/specs/SPEC-typing-cleanup.md).
Tasks tracked here — no external tracker is configured for this repo.

Baseline: 74 `ty` diagnostics in `src/`, `--error-on-warning` exits 1.
Target: 0 diagnostics, `--error-on-warning` exits 0, suite 156 passed / 3 skipped.

## Phase 1: self-contained annotations

- [x] **Task 1** — `callable` → `Callable` (`sequential.py:42`, `toy.py:40,41`)
  - verify: `ty check src/ --output-format=concise | grep invalid-type-form` empty
- [x] **Task 2** — `ISResult` optional diagnostics as fields (B)
  - verify: `ty check src/ --output-format=concise | grep "Unresolved attribute"` drops to 0 for `reference.py`
- [x] **Task 3** — `Moments` cross/prime fields `| None`
  - verify: `diagnostics.py` `invalid-argument-type` count drops by 9

### Checkpoint

- [x] `ty check src/ --output-format=concise | grep -c '^src/'` ≈ 50

## Phase 2: contracts and narrowing

- [x] **Task 4** — `Model` base declares `weight_size`/`output_dim`/`weight_mapping`; overrides compatible
  - verify: `incompatible-method-override` count is 0
- [x] **Task 5** — `FilterSnapshotMixin` declares `model` so `base.py:142` resolves
  - verify: `ty check src/bed/base.py` clean
- [x] **Task 6** — remaining `None`-narrowing sites (`empirical_bayes`, `classical`,
  `gaussian_process`, `sequential`, `reference`, `_experiments`)
  - verify: `missing-union-attribute` count is 0
- [x] **Task 7** — overload/builtin picks (`adaptive.py`, `toy.py:92`, `gaussian_design.py:269`)
  - verify: `no-matching-overload` count is 0
- [x] **Task 8** — optional-import + `possibly-missing-submodule` (`data.py`, `neural.py:100`)
  - verify: only the documented config-override diagnostics remain

### Checkpoint (phase 2)

- [x] `ty check src/ --error-on-warning` exits 0

## Phase 3: flip

- [x] **Task 9** — drop `--exit-zero-on-warning` from `pyproject.toml`,
  `.pre-commit-config.yaml`, `.github/workflows/tests.yml`; prune now-unused
  `[tool.ty.rules]` downgrades; update `AGENTS.md` + spec status
  - verify: `grep -rn "exit-zero-on-warning" . --exclude-dir=.venv` → none
- [x] **Task 10** — full gate set
  - `ruff check src/ tests/ experiments/`
  - `ty check src/ --error-on-warning`
  - `pymarkdown --config .pymarkdown.json scan`
  - `pytest tests/ -q -n auto` (expect 156 passed, 3 skipped)

### Checkpoint: Complete

- [x] all gates green
- [x] no new `noqa`/`# type: ignore` in `src/` (grep-verified)
- [x] diff reviewed as annotation-only

---

## Deferred backlog

Ordered backlog for the `ship-loop` skill's "next item" trigger (first unchecked wins).
Each item becomes a delta spec (`docs/specs/SPEC-<change>.md`) and runs through the
spec-approval gate.

### Shipped

- [x] 1. ~~Split `_models.py`~~ (merged in #12)
- [x] 2. ~~Unify `optimize()` return~~ (merged in #13)
- [x] 3. ~~Stop reseeding `_rng` in `run()`~~ (merged in #14)
- [x] 4. `trace=True` deep-copies full latent covariance (#15)
- [x] 5. Add wrappers for `experiment_4`/`experiment_5` configs (#19)
- [x] 6. `results/` summaries untracked; `.gitignore` comment corrected (#20)

### Still open

- [ ] 7. (Optional) Enable stylistic ruff rules after a deliberate `ruff format` pass.
  **Not started** — would mass-rewrite ~every file (526 `E702`, 290 `E501`, 957
  findings); needs its own spec and an explicit reformat decision.
- [x] 8. Typing pass on `src/` cleared all 74 `ty` diagnostics and flipped the gate to
  `--error-on-warning` (#21).
