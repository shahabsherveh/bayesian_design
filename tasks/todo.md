# Task list

Plan: [tasks/plan.md](plan.md). Tasks tracked here — no external tracker is configured for
this repo.

## Phase 1: mixin

- [x] **Task 1** — add `FilterSnapshotMixin` to `base.py`
  - verify: `from bed.base import FilterSnapshotMixin`
- [x] **Task 2** — EKF/UKF inherit it + declare `_snapshot_copied_attrs`
  - verify: `issubclass(EKF, FilterSnapshotMixin)` and `UKF` likewise

### Checkpoint

- [x] `filters[i].model is filtr.model`; `filters[i].state_prior` distinct arrays

## Phase 2: wire + tests

- [x] **Task 3** — `run()` uses `filtr.snapshot()` (deepcopy fallback if absent)
  - verify: `grep "deepcopy(filtr)"` → gone from run(); `snapshot()` present
- [x] **Task 4** — tests: shared model; independent belief; snapshot supports get_state_posterior / measurement_prior / deepcopy
  - verify: `pytest tests/runner_test.py -q -k snapshot`

### Checkpoint (phase 2)

- [x] new tests pass; empirical_bayes_test still passes

## Phase 3: verify

- [x] **Task 5** — full gate set
  - `pytest tests/ -q -n auto`
  - `ruff check src/ tests/ experiments/` clean
  - `ty check src/ --exit-zero-on-warning` exit 0
  - `pymarkdown` clean

### Checkpoint: Complete

- [x] all gates green
- [x] diff reviewed

---

## Deferred backlog

Ordered backlog for the `ship-loop` skill's "next item" trigger (first unchecked wins).
Each item here was carried over from `docs/STATUS.md` when that file was retired; each
still becomes a delta spec (`SPEC-<change>.md`) and runs through the spec-approval gate.

### Shipped (was the "Deferred" backlog in the trace-snapshot plan)

- [x] 1. ~~Split `_models.py`~~ (merged in #12)
- [x] 2. ~~Unify `optimize()` return~~ (merged in #13)
- [x] 3. ~~Stop reseeding `_rng` in `run()`~~ (merged in #14)
- [x] 4. `trace=True` deep-copies full latent covariance (this plan)

### Still open (from `docs/STATUS.md` "Suggested next steps")

- [ ] 5. Add wrappers for `experiment_4`/`experiment_5` configs, or drop them.
- [ ] 6. Decide whether `results/` summaries should be tracked; if so, commit them,
  otherwise fix the `.gitignore` comment so it does not claim they are kept.
- [ ] 7. (Optional) Enable stylistic ruff rules after a deliberate `ruff format` pass.
- [ ] 8. (Optional) Typing pass on `src/` to clear the ~70 `ty` warnings, then flip the
  gate to `--error-on-warning` (annotate `self.model` on the `ISResult`/`Posterior`
  paths; declare attributes set after construction).
