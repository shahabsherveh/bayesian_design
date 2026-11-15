# Spec Delta: Type `src/` cleanly and gate on warnings

Status: implemented

## Objective

Resolve the 74 `ty` diagnostics in `src/` by **adding real annotations and
declarations** — not suppressions — so the type gate can be flipped from
`--exit-zero-on-warning` to `--error-on-warning`.

This is backlog item 8. It is a *typing* change: no runtime behaviour is
intended to change, and every fix is either an annotation, a field declaration,
or a corrected type construct that the checker was right to reject.

## Current Behavior

`ty check src/ --exit-zero-on-warning` exits 0 despite **74 warnings**
(`ty check src/ --error-on-warning` exits 1). They cluster into five kinds:

| Kind | Count | Cause |
|---|---|---|
| `missing-attribute` | 33 | attributes set after construction (`ISResult.method`), or `self.x` read inside a method the checker cannot see was initialised (`Sequential._offset`, `FlaxModel.weight_size`) |
| `invalid-argument-type` | 16 | `None` passed where an array is expected (`diagnostics.py` `Moments` cross terms), `Unknown \| float \| Rngs` |
| `missing-union-attribute` | 6 | `.round` / `.x` on `X \| None` where the code knows it is non-`None` |
| `no-matching-overload` | 4 | `min`/`max`/`sum` over a dict view or generator with mixed types |
| other | 15 | `callable` used as a type (3), invalid override (2), `not-callable` on `object` (2), invalid-parameter-default (2), import-not-found for optional TF (2), invalid-assignment (3), possibly-missing-submodule (1) |

Three of these are **not merely annotation gaps** and are worth calling out:

1. **`callable` used as a type expression** — `sequential.py:42` (`DesignProblem.target_fn`)
   and `toy.py:40-41` (`GridModel.f`, `GridModel.g`). `callable` is the builtin
   function, never a type; the annotation reads as a value. Must be
   `collections.abc.Callable`.
2. **`ISResult` attributes set after construction** — `method`, `rhat_max`,
   `ess_bulk_min`, `ess_tail_min`, `accept`, `mcmc_steps`, `mcmc_chains`,
   `reliable`, `stages`, `thresholds`, `dropped_mass`. `diagnostics()` currently
   survives this only because it uses `getattr(self, k, default)`/`hasattr`.
   Every path in `robust_posterior` does set them, so there is **no live bug**,
   but the contract is undocumented and unenforced. The fix is to declare the
   optional ones as dataclass fields with `None` defaults, which *also* makes
   `diagnostics()`'s defensive `getattr` redundant rather than load-bearing.
3. **`Models` protocol mismatch** — `classical.py:342` and `neural.py:316`
   override `Model.__call__`/`train` with incompatible signatures, and
   `FlaxModel` is used where `weight_size`/`output_dim` is expected
   (`_experiments.py:93`, `neural.py:306`) but only some subclasses define them.
   This is the unannotated `self.model` gap the STATUS snapshot described.

## New Behavior

- `ISResult` declares its optional diagnostics as fields with `None` defaults.
- `DesignProblem.target_fn`, `GridModel.f`, `GridModel.g` annotate `Callable`.
- A `Model`-shaped protocol (or explicit base-class declarations of
  `weight_size` / `output_dim` / `__call__` / `train`) makes the subclass
  overrides checkable, without changing what the classes do.
- `None | unknown` narrowing sites either annotate the invariant
  (`self.X` is non-`None` after `__post_init__`) or restructure minimally so the
  checker can follow it.
- `pyproject.toml` **and** `.pre-commit-config.yaml` **and** CI change the ty
  invocation to `--error-on-warning`.

No `# type: ignore`, no `noqa`, no blanket `cast` to silence a diagnostic.

## Success Criteria

- `ty check src/ --error-on-warning` exits 0 with **0 diagnostics**.
- The three gates stay green: `ruff check src/ tests/ experiments/`,
  `pymarkdown --config .pymarkdown.json scan`, full `pytest tests/ -q -n auto`
  (baseline 156 passed, 3 skipped — count must not drop).
- No new `# type: ignore` / `noqa` in `src/` (grep-verified).
- Runtime behaviour unchanged: the suite is the check, plus a focused sanity
  call on `robust_posterior` and one `DesignProblem`/`GridModel` construction.

## Boundaries

- **`src/` only.** `tests/` and `experiments/` diagnostics stay ungated: they are
  research scripts, and 4 of the errors are `import-not-found` for sibling
  scripts imported after a `sys.path` insert.
- **No behavioural change.** If a fix would change results, stop and surface it
  rather than "fixing" the type by changing the code's meaning.
- **No mass reformat.** This spec must not touch formatting (that is the separate
  `#7` decision); diffs stay annotation-sized.
- **Do not paper over.** `cast` only where the invariant is genuinely
  established elsewhere and cannot be expressed; each use needs a one-line
  comment saying why.

## Out of Scope (Deferred)

- `#7` — enabling stylistic ruff rules / a `ruff format` pass.
- Gating `tests/` or `experiments/` under ty.
- Annotating the `experiments/` investigation layer.

## Open Questions

1. **Protocol vs base class.** `Model` is currently a concrete base with some
   subclasses defining `weight_size`/`output_dim` and others not. Options:
   (a) declare them on `Model` as `ClassVar`/property stubs that subclasses must
   provide; (b) keep the base and add a `Protocol` for the subset the library
   actually needs. (a) is smaller and matches the existing class hierarchy; (b)
   is more faithful if some models genuinely lack the attributes.
   **Recommended: (a)**, falling back to (b) only for members not shared by all
   models.
2. **How to model the optional `ISResult` diagnostics.** Options: (a) plain
   dataclass fields defaulting to `None`; (b) a separate `ISResultDiagnostics`
   sub-object. (a) keeps `r.method = ...` working unchanged. **Recommended: (a).**
3. **Should `--error-on-warning` flip in one commit or after the count is zero?**
   Recommended: zero the diagnostics first, then flip the flag in the same PR as
   the final commit, so the gate never lands red.
