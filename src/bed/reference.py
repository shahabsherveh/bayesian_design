"""Reference posterior for a closed-form model with a Gaussian prior and Gaussian noise.

The posterior over ``z`` given observations ``(X, Y)`` is, up to a constant,
``log p(z | Y) = -½ Σ_i (f(z, x_i) - y_i)^T R^{-1} (f(z, x_i) - y_i) - ½ (z - μ)^T S^{-1} (z - μ)``.
Three tools, in increasing cost: the MAP with its Laplace covariance (Levenberg–Marquardt on the
whitened residuals, with restarts), importance sampling with a multivariate-t proposal around the
MAP, and an adaptive random-walk Metropolis sampler with parallel chains for cross-checking the
importance-sampling reference on selected cases.

Importance sampling is a *numerical reference approximation*, not an exact posterior. Every
:class:`ISResult` carries the diagnostics needed to judge it: effective sample size, the largest
normalised weight, and the Pareto shape estimate of the weight tail (Vehtari et al., "Pareto
smoothed importance sampling"; k̂ below 0.7 is the usual reliability threshold). Use
:func:`repeat_agreement` to compare independent draws and :func:`mcmc_posterior` on hard cases.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import multivariate_t


def whitening(R):
    """``W`` with ``r^T R^{-1} r = ||W r||²`` (``W = chol(R^{-1})^T``)."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    return np.linalg.cholesky(np.linalg.inv(R)).T


@dataclass
class Posterior:
    """Everything needed to evaluate the unnormalised log posterior and its whitened residuals."""
    model: object
    X: np.ndarray            # (k, ..., d_x) designs, as the model expects
    Y: np.ndarray            # (k, d_y)
    mu0: np.ndarray          # (d,)
    S0: np.ndarray           # (d, d)
    R: np.ndarray            # (d_y, d_y)
    d: int = field(init=False)
    _fns: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.mu0 = np.asarray(self.mu0, dtype=float).ravel(); self.S0 = np.asarray(self.S0, dtype=float)
        self.R = np.atleast_2d(np.asarray(self.R, dtype=float)); self.d = self.mu0.size
        self.Y = np.asarray(self.Y, dtype=float).reshape(-1, self.R.shape[0]); self.k = self.Y.shape[0]
        self.W = whitening(self.R); self.L0inv = np.linalg.inv(np.linalg.cholesky(self.S0))
        Xj = jnp.asarray(self.X); Yj = jnp.asarray(self.Y); Wj = jnp.asarray(self.W); L0 = jnp.asarray(self.L0inv); mu = jnp.asarray(self.mu0)
        model, k, dy, d = self.model, self.k, self.R.shape[0], self.d

        def resid(z):
            f = model(z, Xj).reshape(k, dy)
            return jnp.concatenate([((f - Yj) @ Wj.T).ravel(), L0 @ (z - mu)])

        def jac(z):
            J = model.jacobian(z, Xj).reshape(k, dy, d)
            return jnp.concatenate([jnp.einsum("ab,kbd->kad", Wj, J).reshape(k * dy, d), L0])

        self._fns["resid"] = jax.jit(resid); self._fns["jac"] = jax.jit(jac)
        self._fns["logp"] = jax.jit(jax.vmap(lambda z: -0.5 * jnp.sum(resid(z) ** 2)))

    def resid(self, z):
        return np.asarray(self._fns["resid"](jnp.asarray(z, dtype=float)))

    def jac(self, z):
        return np.asarray(self._fns["jac"](jnp.asarray(z, dtype=float)))

    def logp(self, Z):
        """Unnormalised log posterior at the rows of ``Z`` ``(N, d)``."""
        return np.asarray(self._fns["logp"](jnp.asarray(np.atleast_2d(Z), dtype=float)))


def map_laplace(post, z_inits, ftol=1e-10, xtol=1e-10, max_nfev=5000):
    """MAP by Levenberg–Marquardt from every start in ``z_inits``; returns the best ``z_map``, the
    Gauss–Newton (Laplace) covariance ``(J^T J)^{-1}`` of the whitened residuals, and a record of the
    starts (cost, success)."""
    best, record = None, []
    for z0 in z_inits:
        r = least_squares(post.resid, np.asarray(z0, dtype=float).ravel(), jac=post.jac, method="lm", ftol=ftol, xtol=xtol, max_nfev=max_nfev)
        record.append({"cost": float(r.cost), "success": bool(r.success), "nfev": int(r.nfev)})
        if best is None or r.cost < best.cost:
            best = r
    Jw = post.jac(best.x); cov = np.linalg.inv(Jw.T @ Jw); cov = (cov + cov.T) / 2
    w, V = np.linalg.eigh(cov)
    if w.min() <= 1e-9 * w.max():                     # an ill-conditioned Gauss-Newton matrix: project onto the PD cone and record it
        cov = (V * np.maximum(w, 1e-8 * w.max())) @ V.T; record.append({"psd_repair": True, "min_eig_rel": float(w.min() / w.max())})
    return best.x, cov, record


def pareto_khat(logw, tail_frac=0.2):
    """Generalised-Pareto shape estimate of the upper tail of the importance ratios (Zhang & Stephens
    2009 estimator as used by PSIS). Returns ``nan`` for fewer than 5 tail points."""
    w = np.exp(logw - logw.max()); w = np.sort(w)
    M = max(int(tail_frac * w.size), 5)
    if w.size < 10:
        return float("nan")
    tail = w[-M:]; u = w[-M - 1]; x = tail - u
    x = x[x > 0]; n = x.size
    if n < 5:
        return float("nan")
    m = 20 + int(np.sqrt(n)); xs = np.sort(x)
    theta = 1.0 / xs[-1] + (1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3 * xs[int(n / 4 + 0.5) - 1])
    k = np.array([np.mean(np.log1p(-t * x)) for t in theta])
    lw = n * (np.log(-theta / k) - k - 1); lw = lw - lw.max(); wts = np.exp(lw) / np.sum(np.exp(lw))
    theta_hat = float(np.sum(theta * wts))
    return float(np.mean(np.log1p(-theta_hat * x)))


@dataclass
class ISResult:
    samples: np.ndarray      # (N, d)
    logw: np.ndarray         # unnormalised log importance weights
    w: np.ndarray            # normalised weights
    z_map: np.ndarray
    cov_laplace: np.ndarray
    ess: float
    max_weight: float
    khat: float
    map_record: list

    def moments(self, fn):
        """Weighted mean and variance of ``fn(samples)`` (``fn`` maps ``(N, d)`` to ``(N, q)``)."""
        G = np.asarray(fn(self.samples), dtype=float).reshape(self.samples.shape[0], -1)
        ok = np.all(np.isfinite(G), axis=1) & (self.w > 0)     # a far-tail proposal sample can make fn non-finite at zero weight
        w = self.w[ok] / self.w[ok].sum(); G = G[ok]
        self.dropped_mass = float(1.0 - self.w[ok].sum())
        mean = w @ G; var = w @ (G - mean) ** 2
        return mean, var

    def mc_error(self, fn):
        """Approximate Monte Carlo standard error of the posterior mean of ``fn`` (per component):
        ``sqrt(var / ess)`` with the importance-sampling ESS, or the bulk ESS of the MCMC chains."""
        _, var = self.moments(fn)
        return np.sqrt(var / max(self.ess, 1.0))

    def diagnostics(self):
        """Everything an aggregator needs to decide whether this readout is usable. ``reliable`` is
        the prespecified verdict: for importance sampling ``ess >= ess_min`` and ``khat < khat_max``;
        for MCMC rank-normalised split-R̂ ≤ ``rhat_max`` on every dimension, bulk ESS ≥ ``ess_min``
        and tail ESS ≥ ``ess_tail_min``. Nothing is accepted because of the method that produced it."""
        d = {"ess": self.ess, "max_weight": self.max_weight, "khat": self.khat, "n": int(self.samples.shape[0]),
             "method": getattr(self, "method", "is"), "rhat_max": getattr(self, "rhat_max", None), "reliable": bool(getattr(self, "reliable", True))}
        for k in ("ess_bulk_min", "ess_tail_min", "accept", "mcmc_steps", "mcmc_chains", "thresholds", "stages"):
            if hasattr(self, k):
                d[k] = getattr(self, k)
        return d


def importance_posterior(post, num_samples, seed, z_inits=None, df=5, inflate=1.5, z_map=None, cov=None):
    """Importance sampling from a multivariate-t proposal ``t_df(z_map, inflate · cov_laplace)``.
    ``z_inits`` default to the prior mean and a perturbed copy; pass ``z_map, cov`` to reuse a fit."""
    if z_map is None:
        if z_inits is None:
            rng0 = np.random.default_rng(seed)
            z_inits = [post.mu0, post.mu0 + 0.3 * np.linalg.cholesky(post.S0) @ rng0.normal(size=post.d)]
        z_map, cov, record = map_laplace(post, z_inits)
    else:
        record = []
    cov = np.asarray(cov, dtype=float); cov = (cov + cov.T) / 2
    w, V = np.linalg.eigh(cov)
    if not np.all(np.isfinite(w)) or w.min() <= 1e-9 * max(w.max(), 1e-300):   # proposal shape must be PD: clip, never let scipy raise
        w = np.where(np.isfinite(w), w, 0.0); cov = (V * np.maximum(w, 1e-8 * max(w.max(), 1e-12))) @ V.T   # floor above scipy's PSD tolerance (about 2e-10 relative)
    prop = multivariate_t(loc=z_map, shape=inflate * cov, df=df, seed=int(seed))
    Z = np.atleast_2d(prop.rvs(size=num_samples)); logq = prop.logpdf(Z)
    logw = post.logp(Z) - logq; logw = np.where(np.isfinite(logw), logw, -np.inf)   # a non-finite log density gets zero weight
    w = np.exp(logw - logw.max()); w = w / w.sum()
    return ISResult(Z, logw, w, np.asarray(z_map), np.asarray(cov), float(1.0 / np.sum(w ** 2)), float(w.max()), pareto_khat(logw), record)


def is_reliable(r, ess_min, khat_max):
    return bool(r.ess >= ess_min and r.khat < khat_max)


def robust_posterior(post, num_samples, seed, z_inits=None, ess_min=500, khat_max=0.7, rounds=2, mcmc_steps=12000, mcmc_chains=16,
                     rhat_max=1.01, ess_tail_min=200, mcmc_retry=3, init_spread=2.0):
    """Reference posterior with escalation and a diagnostics-based verdict. (i) Laplace-proposal
    importance sampling; if the ESS is below ``ess_min`` or k̂ above ``khat_max``, (ii) up to
    ``rounds`` rounds of adaptive importance sampling with a heavier-tailed multivariate-t (df 3)
    proposal moment-matched to the previous weighted sample and inflated by 2; if still below the
    thresholds, (iii) adaptive Metropolis with ``mcmc_chains`` chains started ``init_spread``
    Laplace standard deviations apart, for ``mcmc_steps`` steps (half burn-in), retried once with
    ``mcmc_retry`` times the steps if the chains do not pass. The MCMC result is accepted only if
    the rank-normalised split-R̂ is at most ``rhat_max`` on every dimension, the bulk ESS at least
    ``ess_min`` and the tail ESS at least ``ess_tail_min``; ``r.ess`` is then the minimum bulk ESS
    over dimensions (a chain ESS, not the number of retained draws). Every stage records its
    diagnostics in ``r.stages``; ``r.reliable`` is the verdict and is never set by the method used.
    An unreliable readout is returned, flagged, never dropped here."""
    thresholds = {"ess_min": ess_min, "khat_max": khat_max, "rhat_max": rhat_max, "ess_tail_min": ess_tail_min}
    r = importance_posterior(post, num_samples, seed, z_inits=z_inits); r.method = "is"; r.rhat_max = None
    stages = [{"method": "is", "ess": r.ess, "khat": r.khat}]
    stage = 0
    while not is_reliable(r, ess_min, khat_max) and stage < rounds:
        stage += 1
        mean = r.w @ r.samples; dev = r.samples - mean; cov = (r.w[:, None] * dev).T @ dev
        cov = (cov + cov.T) / 2 + 1e-9 * np.eye(post.d)
        if r.ess < 20 or not np.all(np.isfinite(cov)) or not np.all(np.isfinite(mean)):   # collapsed or degenerate weighted moments: widen from the Laplace fit
            mean = r.z_map if not np.all(np.isfinite(mean)) else mean
            cov = (np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0) if np.all(np.isfinite(cov)) else 0.0) + 2.0 * r.cov_laplace
        r2 = importance_posterior(post, num_samples, seed + 1000 * stage, z_map=mean, cov=cov, df=3, inflate=2.0)
        r2.z_map, r2.cov_laplace, r2.map_record = r.z_map, r.cov_laplace, r.map_record
        r2.method = f"ais{stage}"; r2.rhat_max = None
        r = r2; stages.append({"method": r.method, "ess": r.ess, "khat": r.khat})
    r.reliable = is_reliable(r, ess_min, khat_max)
    if not r.reliable:
        steps = mcmc_steps
        for attempt in range(2):
            mc = mcmc_posterior(post, r.z_map, r.cov_laplace, num_steps=steps, seed=seed + 7 * attempt, num_chains=mcmc_chains, init_spread=init_spread)
            S = mc["samples"]; n = S.shape[0]
            rm = ISResult(S, np.zeros(n), np.full(n, 1.0 / n), r.z_map, r.cov_laplace, float(mc["ess_bulk"].min()), 1.0 / n, float("nan"), r.map_record)
            rm.method = "mcmc"; rm.rhat_max = float(mc["rhat"].max()); rm.ess_bulk_min = float(mc["ess_bulk"].min()); rm.ess_tail_min = float(mc["ess_tail"].min())
            rm.accept = mc["accept"]; rm.mcmc_steps = steps; rm.mcmc_chains = mcmc_chains
            rm.reliable = bool(rm.rhat_max <= rhat_max and rm.ess_bulk_min >= ess_min and rm.ess_tail_min >= ess_tail_min)
            stages.append({"method": "mcmc", "steps": steps, "rhat_max": rm.rhat_max, "ess_bulk_min": rm.ess_bulk_min, "ess_tail_min": rm.ess_tail_min, "accept": mc["accept"], "reliable": rm.reliable})
            r = rm
            if rm.reliable:
                break
            steps = mcmc_steps * mcmc_retry
    r.stages = stages; r.thresholds = thresholds
    return r


def repeat_agreement(post, fn, num_samples, seeds, **kw):
    """Independent importance-sampling runs; returns the per-run means of ``fn`` and their spread
    (max absolute pairwise difference), plus the runs' diagnostics — the check that the reference
    is stable across draws."""
    runs = [importance_posterior(post, num_samples, s, **kw) for s in seeds]
    means = np.array([r.moments(fn)[0] for r in runs]); vars_ = np.array([r.moments(fn)[1] for r in runs])
    spread = float(np.max(np.abs(means[:, None] - means[None, :])))
    return {"means": means, "vars": vars_, "spread": spread, "diagnostics": [r.diagnostics() for r in runs]}


def _split_chains(kept):
    """``(T, C, d)`` post-burn draws → ``(T//2, 2C, d)`` split chains."""
    T = kept.shape[0]; half = T // 2
    return np.concatenate([kept[:half], kept[half:2 * half]], axis=1)


def _rhat(chains):
    """Classical split-R̂ per dimension of ``(T, C, d)`` chains."""
    T = chains.shape[0]; means = chains.mean(axis=0); varw = chains.var(axis=0, ddof=1)
    B = T * means.var(axis=0, ddof=1); Wv = varw.mean(axis=0)
    return np.sqrt(((T - 1) / T * Wv + B / T) / np.maximum(Wv, 1e-300))


def _rank_normalise(chains):
    """Rank-normalise the pooled draws of every dimension (Vehtari et al. 2021, eq. 14)."""
    from scipy.stats import norm, rankdata
    T, C, d = chains.shape; out = np.empty_like(chains, dtype=float)
    for j in range(d):
        rk = rankdata(chains[:, :, j].reshape(-1)); out[:, :, j] = norm.ppf((rk - 3.0 / 8) / (T * C + 1.0 / 4)).reshape(T, C)
    return out


def rhat_rank(chains):
    """Rank-normalised split-R̂ per dimension: the maximum of the bulk (rank-normalised) and the
    folded (rank-normalised absolute deviation from the median) statistics of Vehtari, Gelman,
    Simpson, Carpenter and Bürkner (2021). ``chains`` is ``(T, C, d)`` of post-burn-in draws."""
    sp = _split_chains(np.asarray(chains, dtype=float))
    bulk = _rhat(_rank_normalise(sp))
    folded = _rhat(_rank_normalise(np.abs(sp - np.median(sp.reshape(-1, sp.shape[2]), axis=0))))
    return np.maximum(bulk, folded)


def _ess_geyer(chains):
    """ESS per dimension of ``(T, C, d)`` chains from the autocorrelation with Geyer's initial
    monotone sequence truncation, combined across chains as in Vehtari et al. (2021)."""
    T, C, d = chains.shape; ess = np.zeros(d)
    for j in range(d):
        x = chains[:, :, j]; mean_c = x.mean(axis=0); var_c = x.var(axis=0, ddof=1)
        W = var_c.mean(); B = T * mean_c.var(ddof=1) if C > 1 else 0.0; var_plus = (T - 1) / T * W + B / T
        if var_plus <= 0 or not np.isfinite(var_plus):
            ess[j] = float("nan"); continue
        xc = x - mean_c; n2 = 1 << (2 * T - 1).bit_length()
        f = np.fft.rfft(xc, n=n2, axis=0); acov = np.fft.irfft(np.abs(f) ** 2, n=n2, axis=0)[:T] / T   # biased autocovariance per chain
        rho = 1.0 - (W - acov.mean(axis=1)) / var_plus
        # Geyer: sum of adjacent pairs positive and monotone
        pair = rho[: (T // 2) * 2].reshape(-1, 2).sum(axis=1)
        k = 0
        while k + 1 < pair.size and pair[k + 1] > 0:
            k += 1
        pair = np.minimum.accumulate(pair[: k + 1])
        tau = -1.0 + 2.0 * pair.sum()
        ess[j] = C * T / max(tau, 1e-12)
    return ess


def ess_bulk_tail(chains):
    """Bulk ESS (on rank-normalised split chains) and tail ESS (minimum over the 5% and 95%
    quantile indicators) per dimension, as in Vehtari et al. (2021)."""
    sp = _split_chains(np.asarray(chains, dtype=float)); d = sp.shape[2]
    bulk = _ess_geyer(_rank_normalise(sp))
    tail = np.zeros(d)
    for j in range(d):
        x = sp[:, :, j]; q5, q95 = np.quantile(x, [0.05, 0.95])
        tail[j] = min(_ess_geyer((x <= q5).astype(float)[:, :, None])[0], _ess_geyer((x <= q95).astype(float)[:, :, None])[0])
    return bulk, tail


def mcmc_posterior(post, z0, cov0, num_steps, seed, num_chains=16, burn_frac=0.5, target_accept=0.3, adapt_every=100, init_spread=2.0):
    """Adaptive random-walk Metropolis with ``num_chains`` parallel chains started around ``z0``
    (dispersed by ``init_spread`` times the Laplace standard deviation ``cov0``), proposal
    ``N(0, s² cov0)`` with ``s`` adapted during burn-in to the target acceptance rate. Returns the
    post-burn samples ``(num_chains·steps, d)``, the chains ``(steps, num_chains, d)``, the
    acceptance rate, the classical split-R̂, the rank-normalised split-R̂ (``rhat``), and the bulk
    and tail ESS per dimension."""
    rng = np.random.default_rng(seed); d = post.d
    L = np.linalg.cholesky(cov0 + 1e-12 * np.eye(d))
    z = np.asarray(z0, dtype=float)[None, :] + init_spread * (rng.normal(size=(num_chains, d)) @ L.T)
    lp = post.logp(z); s = 2.38 / np.sqrt(d); burn = int(burn_frac * num_steps)
    acc = np.zeros(num_chains); kept = []; acc_window = 0.0; win = 0
    for t in range(num_steps):
        prop = z + s * (rng.normal(size=(num_chains, d)) @ L.T)
        lp_prop = post.logp(prop)
        u = np.log(rng.uniform(size=num_chains)) < lp_prop - lp
        z = np.where(u[:, None], prop, z); lp = np.where(u, lp_prop, lp)
        acc += u; acc_window += u.mean(); win += 1
        if t < burn and (t + 1) % adapt_every == 0:
            rate = acc_window / win; s *= np.exp(rate - target_accept); acc_window = 0.0; win = 0
        if t >= burn:
            kept.append(z.copy())
    kept = np.array(kept)                                  # (T, chains, d)
    bulk, tail = ess_bulk_tail(kept)
    return {"samples": kept.reshape(-1, d), "chains": kept, "accept": float(acc.mean() / num_steps), "rhat_classical": _rhat(_split_chains(kept)),
            "rhat": rhat_rank(kept), "ess_bulk": bulk, "ess_tail": tail, "scale": float(s)}


def predictive(model, samples, X, weights=None):
    """Mean and variance of ``f(z, X)`` under weighted samples; returns ``(mean (n, d_y), var (n, d_y))``."""
    F = np.asarray(jax.vmap(lambda z: model(z, jnp.asarray(X)))(jnp.asarray(samples, dtype=float)))
    F = F.reshape(F.shape[0], -1, F.shape[-1]); w = np.full(F.shape[0], 1.0 / F.shape[0]) if weights is None else np.asarray(weights)
    mean = np.einsum("s,snk->nk", w, F); var = np.einsum("s,snk->nk", w, (F - mean) ** 2)
    return mean, var
