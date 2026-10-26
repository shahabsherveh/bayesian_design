"""Exact small-instance instrument for the value of adaptation (brief §6.1).

A scalar or two-dimensional parameter ``θ`` on a quadrature grid with prior weights, a few
candidate designs, Gaussian observation noise and a horizon of two or three observations. The
posterior is exact on the grid (Bayes' rule on the weights), expectations over hypothetical
outcomes are integrals over a fine outcome grid under the exact predictive density, and every
policy below is evaluated with the same terminal utility and the same integration, so the
differences between them are the quantities of interest and not numerical error:

- ``fixed_best``: the best fixed sequence, exhaustive over all sequences of the horizon;
- ``fixed_greedy``: the sequence built step by step by the exact expected terminal utility of the
  prefix (planning without observing);
- ``adaptive_greedy``: one-step expected utility recomputed after every observation;
- ``adaptive_optimal``: dynamic programming over the horizon (the Bayes-optimal adaptive policy).

The terminal utility is one of: the expected posterior variance of a target functional ``g(θ)``
(``"loss"``; lower is better); the expected posterior entropy of the grid weights over the *full*
parameter grid (``"param_entropy"``, alias ``"entropy"``: information about every parameter, not
about the target); or the expected posterior entropy of the target's marginal on the grid
(``"target_entropy"``: the weights are pooled over grid points sharing the same target value, so
for a target that is a nuisance-free coordinate it is the entropy of that coordinate's marginal
and for an injective target it coincides with the full entropy). All three are outcome-dependent
for a nonlinear model, which is exactly what the linear-Gaussian anchor in
:mod:`bed.gaussian_design` is not. Grid entropies are entropies of discrete weight vectors: their
values, and percentage reductions of them, depend on the grid resolution; only *differences*
between policies at a fixed grid converge as the grid is refined, and :func:`convergence_check`
reports policy values across both the outcome-grid and the parameter-grid resolution.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class GridModel:
    grid: np.ndarray          # (G, p) parameter grid points
    prior: np.ndarray         # (G,) prior weights, sum 1
    f: callable               # f(grid (G, p), x) -> (G,) mean observation at design x
    g: callable               # g(grid) -> (G,) target functional
    designs: np.ndarray       # (n, ...) candidate designs
    sigma: float              # observation noise sd
    ny: int = 121             # outcome-grid points per observation
    span: float = 5.0         # outcome grid spans mean ± span·sd of the predictive

    def __post_init__(self):
        self.prior = np.asarray(self.prior, dtype=float); self.prior = self.prior / self.prior.sum()
        self.F = np.stack([np.asarray(self.f(self.grid, x), dtype=float) for x in self.designs])   # (n, G)
        self.G = np.asarray(self.g(self.grid), dtype=float)                                        # (G,)
        self.n = len(self.designs)
        # pooling matrix for the target's marginal: grid points with the same target value (to 12 digits) share a bin
        _, inv = np.unique(np.round(self.G, 12), return_inverse=True)
        self.target_bins = np.zeros((self.G.size, inv.max() + 1)); self.target_bins[np.arange(self.G.size), inv] = 1.0

    # --- one observation: all outcomes at once -------------------------------------------------
    def branch(self, w, i):
        """Posteriors after design ``i`` for every outcome on the grid: ``(py (ny,), W (ny, G))``.
        The predictive is the mixture ``Σ_g w_g N(y; F[i, g], σ²)`` on a trapezoid grid of ``ny``
        points spanning mean ± span·sd, renormalised."""
        mu = self.F[i]; m = w @ mu; sd = np.sqrt(w @ (mu - m) ** 2 + self.sigma ** 2)
        y = np.linspace(m - self.span * sd, m + self.span * sd, self.ny)
        lik = np.exp(-0.5 * ((y[:, None] - mu[None, :]) / self.sigma) ** 2)      # (ny, G), constant dropped
        W = lik * w[None, :]; py = W.sum(axis=1); W = W / py[:, None]
        py = py / py.sum()
        return py, W

    def utility_rows(self, W, kind):
        if kind == "loss":
            m = W @ self.G; return W @ self.G ** 2 - m ** 2
        if kind == "target_entropy":
            W = W @ self.target_bins
        elif kind not in ("entropy", "param_entropy"):
            raise ValueError(kind)
        Wp = np.where(W > 0, W, 1.0); return -(W * np.log(Wp)).sum(axis=1)

    def utility(self, w, kind):
        return float(self.utility_rows(w[None, :], kind)[0])

    # --- expected terminal utility of a fixed sequence, exact over outcomes ----------------
    def expected_terminal(self, w, seq, kind):
        if not seq:
            return self.utility(w, kind)
        py, W = self.branch(w, seq[0])
        if len(seq) == 1:
            return float(py @ self.utility_rows(W, kind))
        return float(sum(py[a] * self.expected_terminal(W[a], seq[1:], kind) for a in range(self.ny) if py[a] > 1e-14))

    # --- policies ------------------------------------------------------------------------------
    def fixed_best(self, H, kind):
        vals = {seq: self.expected_terminal(self.prior, list(seq), kind) for seq in product(range(self.n), repeat=H)}
        best = min(vals, key=vals.get)
        return list(best), vals[best], vals

    def fixed_greedy(self, H, kind):
        seq = []
        for _ in range(H):
            vals = [self.expected_terminal(self.prior, seq + [i], kind) for i in range(self.n)]
            seq.append(int(np.argmin(vals)))
        return seq, self.expected_terminal(self.prior, seq, kind)

    def one_step_scores(self, w, kind):
        return np.array([self.expected_terminal(w, [i], kind) for i in range(self.n)])

    def adaptive_greedy_value(self, w, H, kind):
        """Expected terminal utility of one-step greedy selection recomputed after each outcome."""
        if H == 0:
            return self.utility(w, kind)
        i = int(np.argmin(self.one_step_scores(w, kind)))
        py, W = self.branch(w, i)
        if H == 1:
            return float(py @ self.utility_rows(W, kind))
        return float(sum(py[a] * self.adaptive_greedy_value(W[a], H - 1, kind) for a in range(self.ny) if py[a] > 1e-14))

    def optimal_value(self, w, H, kind):
        """Bellman recursion ``V_H(w) = min_i E_y V_{H-1}(w | i, y)``, ``V_0 = utility``."""
        if H == 0:
            return self.utility(w, kind)
        best = np.inf
        for i in range(self.n):
            py, W = self.branch(w, i)
            if H == 1:
                v = float(py @ self.utility_rows(W, kind))
            else:
                v = float(sum(py[a] * self.optimal_value(W[a], H - 1, kind) for a in range(self.ny) if py[a] > 1e-14))
            best = min(best, v)
        return best

    def evaluate_all(self, H, kind):
        fb, fb_v, _ = self.fixed_best(H, kind); fg, fg_v = self.fixed_greedy(H, kind)
        ag_v = self.adaptive_greedy_value(self.prior, H, kind); ao_v = self.optimal_value(self.prior, H, kind)
        return {"fixed_best": fb_v, "fixed_best_seq": fb, "fixed_greedy": fg_v, "fixed_greedy_seq": fg,
                "adaptive_greedy": ag_v, "adaptive_optimal": ao_v, "prior_utility": self.utility(self.prior, kind),
                "value_of_planning": fg_v - fb_v, "value_of_adaptation": fb_v - ao_v, "value_of_lookahead": ag_v - ao_v}


def convergence_check(make_model, H, kind, ny_values=(61, 121, 241), G_values=None):
    """Policy values as a function of the outcome-grid resolution ``ny`` (and, if ``G_values`` is
    given, of the parameter-grid resolution ``G``; ``make_model`` then takes ``(ny, G)``)."""
    if G_values is None:
        return {ny: make_model(ny).evaluate_all(H, kind) for ny in ny_values}
    return {(ny, G): make_model(ny, G).evaluate_all(H, kind) for ny in ny_values for G in G_values}


# ---------------------------------------------------------------------------------------------
# concrete toys
# ---------------------------------------------------------------------------------------------

def sigmoid_location_model(prior_sd=1.5, width=0.4, sigma=0.2, designs=np.linspace(-2, 2, 6), G=241, ny=121, target="theta"):
    """``y = tanh((x − θ) / width) + ε``: the design that is informative about ``θ`` is the one
    near ``θ``, so a wide prior makes locating ``θ`` first valuable — a genuinely outcome-dependent
    problem. Target ``g(θ) = θ`` or the prediction ``tanh((x* − θ)/width)`` at ``x* = 0``."""
    th = np.linspace(-4 * prior_sd, 4 * prior_sd, G)[:, None]
    prior = np.exp(-0.5 * (th[:, 0] / prior_sd) ** 2)
    f = lambda grid, x: np.tanh((x - grid[:, 0]) / width)
    g = (lambda grid: grid[:, 0]) if target == "theta" else (lambda grid: np.tanh((0.0 - grid[:, 0]) / width))
    return GridModel(th, prior, f, g, np.asarray(designs, dtype=float), sigma, ny=ny)


def linear_model(prior_sd=1.0, sigma=0.3, designs=np.linspace(-2, 2, 5), G=241, ny=121):
    """``y = x θ + ε`` with target ``θ``: the linear-Gaussian anchor on the grid, where the value of
    adaptation is zero up to integration error."""
    th = np.linspace(-5 * prior_sd, 5 * prior_sd, G)[:, None]
    prior = np.exp(-0.5 * (th[:, 0] / prior_sd) ** 2)
    return GridModel(th, prior, lambda grid, x: x * grid[:, 0], lambda grid: grid[:, 0], np.asarray(designs, dtype=float), sigma, ny=ny)


def amplitude_nuisance_model(prior_sd_theta=1.0, prior_sd_a=0.5, width=0.5, sigma=0.15, designs=np.linspace(-2, 2, 5), G=61, ny=81):
    """``y = a · sigmoid((x − θ)/width) + ε`` with target ``g = a``: the amplitude is the quantity
    of interest and the location ``θ`` a nuisance that decides which designs are informative about
    ``a``; two-dimensional grid."""
    th = np.linspace(-3 * prior_sd_theta, 3 * prior_sd_theta, G); a = 1.0 + np.linspace(-3 * prior_sd_a, 3 * prior_sd_a, G)
    TH, A = np.meshgrid(th, a, indexing="ij"); grid = np.stack([TH.ravel(), A.ravel()], axis=1)
    prior = np.exp(-0.5 * (grid[:, 0] / prior_sd_theta) ** 2 - 0.5 * ((grid[:, 1] - 1.0) / prior_sd_a) ** 2)
    f = lambda grid, x: grid[:, 1] / (1.0 + np.exp(-(x - grid[:, 0]) / width))
    g = lambda grid: grid[:, 1]
    return GridModel(grid, prior, f, g, np.asarray(designs, dtype=float), sigma, ny=ny)
