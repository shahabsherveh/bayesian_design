# Implementation Plan: Type `src/` cleanly and gate on warnings

## Overview

Drive the 74 `ty` diagnostics in `src/` to zero with real annotations, then flip
the gate to `--error-on-warning`. No `noqa`, no `# type: ignore`.

Spec: [docs/specs/SPEC-typing-cleanup.md](../docs/specs/SPEC-typing-cleanup.md)
(approved; Q1 = declare on the base class, Q2 = plain optional fields,
Q3 = zero first then flip in the same PR).

## Design

Diagnostics group into independent piles, each confined to one or two files. Order
by (risk asc, diagnostic count desc) so the tree is green-ish between commits.

**A. `callable` → `Callable`** (`sequential.py:42`, `toy.py:40,41`)
Pure annotation typo. `callable` is the builtin function, so the annotation is a
value, not a type. Import `Callable` from `collections.abc` and annotate the
actual shape, e.g. `Callable[[np.ndarray, np.ndarray], np.ndarray]`.

**B. `ISResult` optional diagnostics** (`reference.py`, 20 diagnostics)
Declare the post-construction attributes as dataclass fields with `None` defaults:

```python
method: str | None = None
rhat_max: float | None = None
ess_bulk_min: float | None = None
ess_tail_min: float | None = None
accept: float | None = None
mcmc_steps: int | None = None
mcmc_chains: int | None = None
reliable: bool | None = None
stages: list[dict] | None = None
thresholds: dict | None = None
dropped_mass: float | None = None
```

Assignment sites then check. `diagnostics()`'s `getattr`/`hasattr` can stay (it is
still the API-boundary defence), but it stops being the only reason the code works.
`is_reliable` and the mcmc branch may need `assert ... is not None` style
narrowing after a field is set in the same function — prefer restructuring the
comparison to a local variable over `assert`, to avoid `-O` sensitivity.

**C. `Model` attribute contract** (`base.py`, `models/neural.py`,
`models/classical.py`, `models/gaussian_process.py`, `mechanistic.py`,
`_experiments.py`; ~14 diagnostics)
Declare on the base `Model` the members the library relies on:
`weight_size` / `output_dim` / `weight_mapping` as documented class-level
attributes, and widen `__call__`/`train` signatures to what the subclasses
actually implement (variance in the parameter types, not a behavioural change).
`base.snapshot`'s `Self` needs the mixin to declare `model`.

**D. `None`-narrowing sites** (`empirical_bayes`, `classical`, `gaussian_process`,
`sequential`, `diagnostics`, `reference`, `_experiments`; ~13)
Most are `X | None` where `__post_init__` or a guard establishes non-`None`. Fix by
annotating the attribute's post-init invariant, not by re-checking at every use.
`diagnostics.py`'s `Moments(cross=None)` sites are genuinely `None`-carrying, so
the `Moments` fields need `| None`.

**E. Overload / builtin picks** (`adaptive.py` 3, `toy.py` 1,
`gaussian_design.py` 1; 5)
`min`/`max`/`sum` over mixed or dict-view inputs: give the builtin an explicit
`key=`/`default=`, or materialise to a typed list first.

**F. Optional-import + submodule** (`data.py` TF, `neural.py:100`; 3)
`# noqa` is not allowed. `import-not-found` is already downgraded to a warning by
config; to reach zero under `--error-on-warning` either keep those two rules at
`warn` in `[tool.ty.rules]` (documented) or add an `[tool.ty.overrides]` for the
lazy-import sites. **Decide at implementation time; prefer a documented config
override over touching the import.**

**G. Flip the gate** (`pyproject.toml`, `.pre-commit-config.yaml`,
`.github/workflows/tests.yml`, `docs/specs/*`, `AGENTS.md`)
Only once the count is 0: drop `--exit-zero-on-warning`, and remove the now-unused
`[tool.ty.rules]` downgrades *except* the ones F keeps.

## Task list

### Phase 1: self-contained annotations

- [ ] Task 1: `callable` → `Callable` in `sequential.py`, `toy.py` (A)
- [ ] Task 2: `ISResult` optional fields (B)
- [ ] Task 3: `Moments` cross fields `| None` (D, part)

### Phase 2: contracts and narrowing

- [ ] Task 4: `Model` base declarations + override compat (C)
- [ ] Task 5: `Self`/mixin `model` declaration in `base.py` (C)
- [ ] Task 6: `None`-narrowing sites (D)
- [ ] Task 7: overload/builtin picks (E)
- [ ] Task 8: optional-import + submodule resolution (F)

### Phase 3: flip

- [ ] Task 9: gate flip + config cleanup (G)
- [ ] Task 10: full gate set + suite

### Checkpoints

- After Phase 1: `ty check src/ --output-format=concise | wc -l` drops by ~24.
- After Phase 2: count is 0.
- After Phase 3: `ty check src/ --error-on-warning` exits 0; suite unchanged.

## Risks

| Risk | Mitigation |
|---|---|
| Widening a signature to satisfy the checker changes dispatch | Only widen *parameter* types where every subclass already accepts them; never add/remove params |
| Declaring `ISResult` fields as defaults changes `hasattr` behaviour | `diagnostics()` keeps its `getattr` default-to-value semantics; verify the `"method"` default is still `"is"` |
| `--error-on-warning` reveals diagnostics `src/` only sees with the flag | Flip in the last commit; CI runs the exact command |
| Chasing 74 diagnostics drifts into behaviour edits | Each change must be annotation-only; if a fix needs a code change, stop and surface it |
