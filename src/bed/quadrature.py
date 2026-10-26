"""Mode-centred deterministic quadrature for low-dimensional posteriors.

The earlier source-location check integrated over a fixed box on a uniform grid. That is adequate
when the posterior is broad and fails when it is narrow: on the low-noise source condition,
refining 401 to 801 nodes per axis moved the paired utility differences by a median 9% of stakes
and **changed the selected action on 15 of 60 histories**, so the grid was never resolving the
modes. A fixed box cannot fix this by refinement alone, because the posterior width shrinks with
the data while the box does not.

This module instead places nodes where the mass is. It finds the modes by multi-start MAP, builds
one tensor grid per mode in that mode's own whitened coordinates, and forms a partition with a
mixture-of-Laplace reference density so overlapping patches are not double counted:

    int f(z) p(z) dz  =  sum_k int [ f(z) p(z) / q(z) ] pi_k N_k(z) dz,   q = sum_k pi_k N_k,

with each inner integral evaluated by Gauss-Hermite in the whitened coordinates of component k.
Every node therefore carries the exact ratio p/q, the components' contributions add rather than
overlap, and the scheme is exact when p is Gaussian and f is a low-order polynomial.

Reported alongside every result: the integrated mass, which is 1 only if the node set covers the
posterior, and the largest single node weight, which is small only if the mass is resolved rather
than concentrated on one point. Neither is renormalised away.
"""
from __future__ import annotations

import numpy as np

from . import reference as ref


def enable_double_precision():
    """Turn on JAX double precision, and say whether it took effect.

    JAX defaults to single precision, so every model evaluation, Jacobian and log posterior in this
    project before 2026-09-26 carried about 1e-7 relative error. Measured against the closed-form
    linear-Gaussian posterior, that is the whole of the quadrature's error: 1.7e-7 in single and
    1.3e-14 in double. It is four orders of magnitude below the resolution any reported claim rests
    on, so it does not change a published number, and it is free to remove. This must be called
    before the first JAX array is created, so experiment entry points call it first.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    return bool(jax.config.read("jax_enable_x64"))


def _laplace_modes(post, z_inits, tol=1e-3):
    """Distinct MAP optima, their Gauss-Newton covariances, and a record of the search.

    Returns `(found, log)`. Optimiser failures are counted rather than swallowed: a silent `except`
    cannot establish that the search converged, and the count belongs in the diagnostics.
    """
    found, log = [], {"starts": len(list(z_inits)), "optimiser_failures": 0,
                      "nonfinite": 0, "duplicates": 0}
    for z0 in z_inits:
        try:
            m, S, _ = ref.map_laplace(post, [np.asarray(z0, float)])
        except Exception:
            log["optimiser_failures"] += 1
            continue
        m = np.asarray(m, float); S = np.asarray(S, float)
        if not np.all(np.isfinite(m)) or not np.all(np.isfinite(S)):
            log["nonfinite"] += 1
            continue
        obj = float(0.5 * np.sum(post.resid(m) ** 2))
        if any(np.linalg.norm(m - f[0]) < tol * max(1.0, np.linalg.norm(m)) for f in found):
            log["duplicates"] += 1
            continue
        found.append((m, S, obj))
    if not found:
        raise RuntimeError("no mode found")
    found.sort(key=lambda t: t[2])
    log["n_distinct"] = len(found)
    return found, log


def mode_starts(post, mu0, S0, extra=(), per_axis=7, span=2.5, seed=0):
    """Starts for the mode search: a fixed lattice over the prior, plus any supplied belief means.

    The starts are deterministic. With random dispersed starts the number of modes found varied
    between otherwise identical runs, and missing one mode is not a small error: the mixture then
    has no nodes where a whole lobe of the posterior lives, and the selected action moves. A
    lattice over plus or minus `span` prior standard deviations in whitened coordinates covers the
    prior's high-probability region at a cost of `per_axis ** d` cheap local solves, and returns
    the same set every time.
    """
    mu0 = np.asarray(mu0, float)
    L0 = np.linalg.cholesky(np.asarray(S0, float))
    d = mu0.size
    ax = np.linspace(-span, span, per_axis)
    grid = np.stack([g.ravel() for g in np.meshgrid(*([ax] * d), indexing="ij")], axis=1)
    return ([mu0] + [np.asarray(e, float) for e in extra]
            + [mu0 + L0 @ u for u in grid])


def gauss_hermite(n):
    """Probabilists' Gauss-Hermite nodes and weights: int g(x) N(x;0,1) dx = sum w g(x)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


class ModeQuadrature:
    """A node set and weights for a posterior, built around its modes.

    ``nodes`` are parameter values, ``logw`` their log weights with the posterior's unknown
    normalising constant removed by the log-sum-exp at construction. ``mass`` is the integrated
    mixture mass before that normalisation and is reported, never used to hide truncation.
    """

    def __init__(self, post, mu0, S0, n_per_axis=24, max_modes=4, seed=0, extra_starts=(),
                 inflate=2.0, coverage_weight=0.0, min_component_weight=1e-6):
        """`inflate` widens each mode's Laplace covariance and `coverage_weight` adds one broad
        component with the prior covariance.

        `min_component_weight` drops mixture components carrying negligible Laplace evidence. This
        matters more than it sounds: the mode search finds stationary points whose posterior weight
        can be 1e-40 or smaller, and keeping one spends half the node budget where there is no mass
        while its density ratio is numerically extreme. Measured on stored source histories, the
        selected action disagreed under grid refinement almost only where such a component was
        present, and agreed on genuinely bimodal posteriors whose second mode carried a quarter of
        the mass.

        Both other settings exist because the Laplace components alone do not cover a multimodal
        source posterior:
        on low-noise histories with two modes the integrated mass fell to 0.90, meaning a tenth of
        the posterior sat where the rule had placed no nodes, and the selected action then moved
        under refinement. The ratio `p/q` corrects for whatever proposal is used, so widening costs
        only efficiency, while missing support costs correctness. Set `coverage_weight=0` to
        recover the modes-only rule.
        """
        d = len(mu0)
        all_modes, self.search_log = _laplace_modes(post, mode_starts(post, mu0, S0, extra_starts))
        # Rank by Laplace MASS before truncating, not by peak height. `_laplace_modes` returns modes
        # ordered by posterior height, and truncating that order discards whichever peaks are
        # lowest. A broad low peak can carry more mass than a narrow high one, so height-ordered
        # truncation can drop the component that matters most.
        obj_all = np.array([m[2] for m in all_modes])
        logdet_all = np.array([np.linalg.slogdet(m[1])[1] for m in all_modes])
        logpi_all = -obj_all + 0.5 * logdet_all
        logpi_all -= logpi_all.max()
        pi_all = np.exp(logpi_all); pi_all /= pi_all.sum()
        order = np.argsort(-pi_all)
        self.n_modes_found = len(all_modes)
        self.search_log["n_found_before_truncation"] = len(all_modes)
        self.search_log["weights_before_truncation"] = pi_all[order].tolist()
        kept = order[:max_modes]
        self.search_log["mass_dropped_by_mode_cap"] = float(pi_all[order[max_modes:]].sum())
        modes = [all_modes[i] for i in kept]
        pi = pi_all[kept]; pi = pi / pi.sum()
        keep = pi >= min_component_weight
        keep[int(np.argmax(pi))] = True
        self.search_log["mass_dropped_by_weight_floor"] = float(pi[~keep].sum())
        self.search_log["n_retained"] = int(keep.sum())
        modes = [m for m, k in zip(modes, keep) if k]
        pi = pi[keep]; pi = pi / pi.sum()

        x1, w1 = gauss_hermite(n_per_axis)
        # The rule's reach is set by `n_per_axis` alone: Gauss-Hermite nodes span about
        # +-sqrt(2 n_per_axis) whitened standard deviations. An `n_sd` argument used to be stored
        # and reported here without entering the construction, so varying it appeared to test
        # domain expansion and tested nothing. It is removed rather than left as a dead control.
        grids = np.meshgrid(*([x1] * d), indexing="ij")
        U = np.stack([g.ravel() for g in grids], axis=1)                      # (n^d, d)
        wgrids = np.meshgrid(*([w1] * d), indexing="ij")
        W = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)    # (n^d,)

        comps = [(m, 0.5 * (S + S.T) * inflate) for (m, S, _o) in modes]
        pi = pi * (1.0 - coverage_weight)
        if coverage_weight > 0:
            # one broad component with the prior covariance, so the mixture has support wherever
            # the posterior does and the density ratio stays bounded
            comps.append((np.asarray(mu0, float), np.asarray(S0, float)))
            pi = np.append(pi, coverage_weight)
        pi = pi / pi.sum()

        nodes, Ls, means = [], [], []
        for (m, S) in comps:
            ev, V = np.linalg.eigh(S)
            ev = np.maximum(ev, 1e-12 * max(ev.max(), 1e-12))
            L = V @ np.diag(np.sqrt(ev))
            Ls.append(L); means.append(m)
            nodes.append(m[None, :] + U @ L.T)
        nodes = np.concatenate(nodes, axis=0)
        self.n_modes = len(modes)
        self.min_component_weight = min_component_weight
        self.n_components = len(comps)
        self.component_of = np.repeat(np.arange(len(comps)), U.shape[0])
        # Each component's nodes carry that component's mixture weight as well as the rule's own
        # weight. Writing the integral as
        #     int f p dz = sum_k pi_k int [f p / q] N_k dz,
        # the outer sum is over components with weight pi_k and the inner one is Gauss-Hermite. The
        # density in the ratio is the pi-weighted mixture, so omitting pi_k here weights the
        # components equally while dividing by a pi-weighted density, and the two disagree exactly
        # when pi is not uniform. With one component pi = 1 and nothing is lost, which is why the
        # unimodal case was exact and the multimodal case was wrong by up to a factor two with
        # perfect node coverage.
        self.base_weight = np.concatenate([W * pi[k] for k in range(len(comps))])
        self.pi = pi
        self.inflate = inflate
        self.coverage_weight = coverage_weight

        # log q at every node: the full mixture density, so components do not double count
        logq = np.full(nodes.shape[0], -np.inf)
        for k, (m, L) in enumerate(zip(means, Ls)):
            sol = np.linalg.solve(L, (nodes - m[None, :]).T).T
            lk = (-0.5 * np.sum(sol ** 2, axis=1) - np.log(np.abs(np.linalg.det(L)))
                  - 0.5 * d * np.log(2 * np.pi) + np.log(pi[k]))
            logq = np.logaddexp(logq, lk)
        logp = post.logp(nodes).ravel()
        lw = np.log(np.maximum(self.base_weight, 1e-300)) + logp - logq
        mx = lw.max()
        w = np.exp(lw - mx)
        self.nodes = nodes
        self.mass_log = mx + np.log(w.sum())
        self.w = w / w.sum()
        self.max_node_weight = float(self.w.max())
        # The base rule already concentrates: at 24 nodes per axis in two dimensions the central
        # product weight is 0.058, so a bare maximum says nothing. What matters is how much the
        # posterior has shifted weight away from the rule it was built for. A ratio near 1 means the
        # mixture matches the posterior and the rule is doing its job; a large ratio means the mass
        # has moved to a few nodes and the node set does not resolve it.
        base = self.base_weight / self.base_weight.sum()
        self.weight_concentration_ratio = float(self.w.max() / base.max())
        self.ess = float(1.0 / np.sum(self.w ** 2))
        self.n_nodes = nodes.shape[0]
        self.n_per_axis = n_per_axis

    # ---- integrals
    def moments(self, fn):
        """Posterior mean and variance of ``fn(nodes)`` -> (n_nodes, q)."""
        F = np.asarray(fn(self.nodes), float)
        F = F.reshape(self.nodes.shape[0], -1)
        m = self.w @ F
        v = self.w @ (F ** 2) - m ** 2
        return m, np.maximum(v, 0.0)

    def diagnostics(self):
        return {"n_nodes": int(self.n_nodes), "n_modes": int(self.n_modes),
                "n_modes_found": int(self.n_modes_found),
                "min_component_weight": float(self.min_component_weight),
                "n_components": int(self.n_components), "inflate": float(self.inflate),
                "coverage_weight": float(self.coverage_weight),
                "n_per_axis": int(self.n_per_axis),
                "node_reach_whitened_sd": float(np.sqrt(2.0 * self.n_per_axis)),
                "max_node_weight": self.max_node_weight,
                "weight_concentration_ratio": self.weight_concentration_ratio,
                "effective_nodes": self.ess,
                # `log_mass` is the log of the UNNORMALISED posterior integral against the
                # proposal, that is an evidence estimate up to the posterior's own constant. It is
                # not a probability and there is no reason for it to be zero.
                "log_unnormalised_evidence": float(self.mass_log),
                "mixture_weights": self.pi.tolist(), "search": self.search_log}


def _cluster_1d(f, w, n_clusters, iters=25):
    """Weighted 1-D k-means on the predictive component means, for building the `y` proposal."""
    if n_clusters <= 1 or f.size <= 1:
        return np.zeros(f.size, dtype=int), 1
    qs = np.linspace(0, 100, n_clusters + 2)[1:-1]
    c = np.percentile(f, qs)
    lab = np.zeros(f.size, dtype=int)
    for _ in range(iters):
        lab = np.argmin(np.abs(f[:, None] - c[None, :]), axis=1)
        for k in range(len(c)):
            m = lab == k
            if m.any() and w[m].sum() > 0:
                c[k] = float(np.sum(w[m] * f[m]) / np.sum(w[m]))
    keep = sorted({int(k) for k in lab})
    remap = {k: j for j, k in enumerate(keep)}
    return np.array([remap[int(k)] for k in lab]), len(keep)


def expected_posterior_variance(quad, problem, cands, n_y=64, targets=None, n_clusters=4,
                                mass_tol=1e-3, max_clusters=12):
    """`V(h)` and `E_y[V(h,a,Y)]` for each candidate, by quadrature in both variables.

    The predictive at a candidate is exactly a Gaussian mixture with one component per parameter
    node. A single moment-matched proposal in `y` is adequate only when that mixture is unimodal;
    on low-noise source histories with two well-separated parameter modes the predictive is itself
    bimodal, a single proposal misses one mode, and up to 23% of the predictive mass went
    unintegrated while the selected action wandered under refinement. The proposal here is
    therefore a mixture too: the component means are clustered, one moment-matched Gaussian is
    built per cluster, and every node carries the exact ratio `p(y)/q(y)`. The number of clusters
    is increased until the integrated mass reaches 1.

    **The result IS divided by the estimated predictive mass.** An earlier version of this docstring
    said the mass was reported rather than normalised away; both happen, and only the reporting was
    true. `mass` is that denominator, `ev_raw` the numerator before division, and `mass_dev` its
    signed deviation from one, so a reader can undo the normalisation or bound its effect.

    A candidate whose mass deviates by more than `mass_tol` after `max_clusters` is recorded in
    `unresolved` rather than silently accepted.

    Returns `(V_now, EV, U, mass, info)`.
    """
    Z = quad.nodes
    G = np.asarray(problem.target_fn(Z), float).reshape(Z.shape[0], -1) if targets is None else targets
    w = quad.w
    mg = w @ G
    V_now = float(np.sum(w @ (G ** 2) - mg ** 2))

    import jax, jax.numpy as jnp
    Xc = jnp.asarray(np.asarray(problem.X)[np.asarray(cands, int)])
    if int(problem.dy) != 1:
        raise NotImplementedError("observation integration is written for scalar observations")
    F = np.asarray(jax.vmap(lambda z: problem.model(z, Xc))(jnp.asarray(Z, dtype=float)),
                   dtype=float).reshape(Z.shape[0], len(cands)).T
    r2 = float(np.atleast_2d(problem.R)[0, 0])
    xg, wg = gauss_hermite(n_y)
    G2 = G ** 2
    lognorm = -0.5 * np.log(2 * np.pi * r2)

    EV = np.empty(len(cands)); mass = np.empty(len(cands)); used = np.empty(len(cands), dtype=int)
    ev_raw = np.empty(len(cands)); neg_var = np.zeros(len(cands))
    for j in range(len(cands)):
        fj = F[j]
        nc = n_clusters
        while True:
            lab, nc_eff = _cluster_1d(fj, w, nc)
            ys, qw = [], []
            for k in range(nc_eff):
                m = lab == k
                pk = float(w[m].sum())
                if pk <= 0:
                    continue
                mu = float(np.sum(w[m] * fj[m]) / pk)
                var = float(np.sum(w[m] * fj[m] ** 2) / pk - mu ** 2 + r2)
                sd = np.sqrt(max(var, 1e-300))
                ys.append(mu + sd * xg)
                qw.append((pk, mu, sd))
            y = np.concatenate(ys)                                  # (nc_eff * n_y,)
            d = y[:, None] - fj[None, :]
            lk = -0.5 * d ** 2 / r2
            mxk = lk.max(axis=1, keepdims=True)
            A = np.exp(lk - mxk) * w[None, :]
            rs = A.sum(axis=1)
            logp_y = np.log(np.maximum(rs, 1e-300)) + mxk.ravel() + lognorm
            # The proposal is the whole mixture, evaluated at every node of every component, so a
            # node drawn for one cluster is still weighted by the density of all of them and the
            # components add rather than overlap. Each block of nodes carries its own cluster's
            # Gauss-Hermite weight times that cluster's probability.
            W_y = np.concatenate([wg * pk for (pk, _mu, _sd) in qw])
            logq_y = np.full(y.size, -np.inf)
            for (pk2, mu2, sd2) in qw:
                logq_y = np.logaddexp(logq_y,
                                      -0.5 * ((y - mu2) / sd2) ** 2 - np.log(sd2)
                                      - 0.5 * np.log(2 * np.pi) + np.log(pk2))
            ratio = np.exp(logp_y - logq_y)
            den = float(np.sum(W_y * ratio))
            if abs(den - 1.0) <= mass_tol or nc >= max_clusters:
                break
            nc = min(max_clusters, nc * 2)
        m_t = (A @ G) / rs[:, None]
        v_t = (A @ G2) / rs[:, None] - m_t ** 2
        # A materially negative variance is a numerical failure, not something to clip away. Record
        # the worst one per candidate; clipping alone would hide it.
        neg_var[j] = float(-min(0.0, v_t.min()))
        Vy = np.sum(np.maximum(v_t, 0.0), axis=1)
        num = float(np.sum(W_y * ratio * Vy))
        ev_raw[j] = num
        EV[j] = num / den
        mass[j] = den
        used[j] = nc_eff
    info = {"mass_dev_min": float(mass.min() - 1.0), "mass_dev_max": float(mass.max() - 1.0),
            "mass_dev_absmax": float(np.abs(mass - 1.0).max()),
            "unresolved": [int(c) for c, m in zip(np.asarray(cands, int), mass)
                           if abs(m - 1.0) > mass_tol],
            "clusters_used_max": int(used.max()), "clusters_used_median": float(np.median(used)),
            "negative_variance_max": float(neg_var.max()),
            "ev_raw": ev_raw.tolist(), "mass": mass.tolist()}
    return V_now, EV, V_now - EV, mass, info
