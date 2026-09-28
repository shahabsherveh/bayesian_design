"""Sequential design loop shared by the investigations, for the two applications.

A :class:`DesignProblem` wraps a mechanistic problem (candidates, targets, prior, noise, truth).
A :class:`NoiseTable` gives every (candidate, visit) pair its own standard-normal draw, so repeated
acquisitions carry fresh noise and paired policies see common random numbers. A belief is one of
:class:`EKFBelief` (sequential Kalman update relinearised at the current mean), :class:`LaplaceBelief`
(MAP + Gauss–Newton covariance refitted on all data after every observation) or :class:`UKFBelief`
(the package's unscented filter); a policy maps the belief and the history to the next candidate.
:func:`run_episode` records the full history (designs, outcomes, visit counts, belief snapshots and
the scores that decided each step) and :func:`evaluate_history` scores it with the common reference
posterior of :mod:`bed.reference`.

Nothing in a policy sees the truth, the future, or the reference posterior.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from bed import gaussian_design as gd
from bed import reference as ref
from bed.mechanistic import DivideModel, divide_biomarkers, make_divide_problem, make_pkpd_problem


# ---------------------------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------------------------

@dataclass
class DesignProblem:
    name: str
    model: object
    X: np.ndarray             # (n, d_x) candidates
    Xt: np.ndarray            # (m, d_x) targets as virtual designs (functionals) or observations
    mu0: np.ndarray           # (d,)
    S0: np.ndarray            # (d, d)
    R: np.ndarray             # (d_y, d_y)
    Rt: object                # target-side noise: 0.0 (noise-free functional) or (d_t, d_t)
    z_true: np.ndarray        # (d,)
    replace: bool             # repeated acquisitions allowed
    target_fn: callable       # samples (N, d) -> target values (N, q) used for the terminal loss
    target_names: tuple
    truth_targets: np.ndarray = field(init=False)   # (q,)
    meta: dict = field(default_factory=dict)
    target_scale: np.ndarray = None   # (d_t,) per-output divisor applied to the target model outputs, their Jacobians and ``Rt`` (None: 1)

    def __post_init__(self):
        self.n = self.X.shape[0]; self.d = self.mu0.size; self.dy = self.R.shape[0]
        self.truth_targets = np.asarray(self.target_fn(self.z_true[None, :])).reshape(-1)
        self._f = jax.jit(lambda z, x: self.model(z, x)); self._jac = jax.jit(lambda z, x: self.model.jacobian(z, x))
        self.f_true = np.asarray(self._f(jnp.asarray(self.z_true), jnp.asarray(self.X))).reshape(self.n, self.dy)
        if self.target_scale is not None:
            self.target_scale = np.asarray(self.target_scale, dtype=float).reshape(-1)
            if np.ndim(self.Rt):
                D = np.diag(1.0 / self.target_scale); self.Rt = D @ np.asarray(self.Rt, dtype=float) @ D

    def f(self, z, X):
        return np.asarray(self._f(jnp.asarray(z, dtype=float), jnp.asarray(X))).reshape(np.asarray(X).shape[0], self.dy)

    def jac(self, z, X):
        return np.asarray(self._jac(jnp.asarray(z, dtype=float), jnp.asarray(X)))

    def jac_targets(self, z):
        """Jacobians of the (scaled) target outputs, ``(m, d_t, d)``."""
        Jt = np.asarray(self._jac(jnp.asarray(z, dtype=float), jnp.asarray(self.Xt)))
        return Jt if self.target_scale is None else Jt / self.target_scale[None, :, None]

    def feasible(self, used):
        """Boolean mask ``(n,)`` of the candidates a policy may select next given the acquisitions
        ``used`` so far: every candidate when repeats are allowed, otherwise the unused ones. This is
        the single definition of the feasible set; every selection goes through :func:`mask_infeasible`."""
        m = np.ones(self.n, dtype=bool)
        if not self.replace and len(used):
            m[np.asarray(list(used), dtype=int)] = False
        return m


def mask_infeasible(problem, scores, used):
    """Copy of ``scores`` with the infeasible candidates set to ``-inf``; ``argmax`` of the result
    is always a feasible choice (raises if nothing is feasible)."""
    m = problem.feasible(used)
    if not m.any():
        raise ValueError("no feasible candidate left")
    sc = np.array(scores, dtype=float, copy=True); sc[~m] = -np.inf
    return sc


_PKPD_SCALE_CACHE = {}


def pkpd_prior_predictive_scale(prior_scale=1.0, num_samples=4000, seed=1, **kw):
    """Root-mean-square prior-predictive standard deviation of the two PK/PD outputs over the
    steady-state pool, from ``num_samples`` draws of the prior (fixed seed, independent of any
    subject): the per-output divisor of ``standardise="per-output"``. Cached per prior."""
    key = (prior_scale, num_samples, seed, tuple(sorted((k, str(v)) for k, v in kw.items())))
    if key not in _PKPD_SCALE_CACHE:
        p = make_pkpd_problem(seed=0, prior_scale=prior_scale, **kw)
        Xt = jnp.asarray(np.asarray(p.data.x_test_pool).reshape(-1, 4)); L = np.linalg.cholesky(np.asarray(p.prior_cov))
        Z = np.asarray(p.prior_mean).ravel()[None, :] + np.random.default_rng(seed).normal(size=(num_samples, L.shape[0])) @ L.T
        F = np.asarray(jax.vmap(lambda z: p.model(z, Xt))(jnp.asarray(Z))).reshape(num_samples, -1, 2)
        _PKPD_SCALE_CACHE[key] = np.sqrt(F.var(axis=0).mean(axis=0))
    return _PKPD_SCALE_CACHE[key].copy()


def pkpd_problem(seed, prior_scale=1.0, standardise=None, **kw):
    """PK/PD as in the paper: candidates single-dose occasions, targets the 24 steady-state
    observations with the observation noise ``R`` on the target side, selection without
    replacement. Terminal targets: the noise-free steady-state predictions ``(log C, E)`` at the pool
    (loss per output). With ``standardise="per-output"`` every target output is divided by its
    prior-predictive RMS standard deviation (:func:`pkpd_prior_predictive_scale`), so that the
    acquisition (``var``), the reference utility and the terminal loss weight the two outputs the
    same way; ``None`` keeps the paper's raw units, in which the effect dominates."""
    p = make_pkpd_problem(seed=seed, prior_scale=prior_scale, **kw)
    X = np.asarray(p.data.x_train).reshape(-1, 4); Xt = np.asarray(p.data.x_test_pool).reshape(-1, 4); M = Xt.shape[0]
    model = p.model
    if standardise is None:
        scale = None; sc = np.ones(2)
    elif standardise == "per-output":
        scale = pkpd_prior_predictive_scale(prior_scale=prior_scale, **kw); sc = scale
    else:
        raise ValueError(standardise)
    target_fn = lambda Z: (np.asarray(jax.vmap(lambda z: model(z, jnp.asarray(Xt)))(jnp.asarray(Z, dtype=float))).reshape(np.asarray(Z).shape[0], M, 2) / sc).reshape(np.asarray(Z).shape[0], M * 2)
    names = tuple(f"{o}@{k}" for k in range(M) for o in ("logC", "E"))
    return DesignProblem("pkpd", model, X, Xt, np.asarray(p.prior_mean, dtype=float).ravel(), np.asarray(p.prior_cov, dtype=float), np.asarray(p.measurement_error, dtype=float),
                         np.asarray(p.measurement_error, dtype=float), np.asarray(p.z_true, dtype=float).ravel(), False, target_fn, names,
                         meta={"seed": seed, "prior_scale": prior_scale, "outputs": ("logC", "E"), "targets_per_output": M, "standardise": standardise,
                               "target_scale": None if scale is None else scale.tolist(), **kw}, target_scale=scale)


def divide_problem(seed, snr=40.0, prior_scale=1.0, nuisance_sd=None, truth_nuisance_sd="belief", **kw):
    """DIVIDE single voxel as in the paper: 126 candidates, targets the three standardised
    log-biomarkers (noise-free functionals, ``Rt = 0``), repeats allowed. Terminal targets: the
    standardised log-biomarkers.

    ``nuisance_sd`` sets the *belief* prior sd of ``(log S_0, log T_2)``; the package default is
    ``(0.1, 0.2)`` (times ``sqrt(prior_scale)``). The truth's nuisance coordinates are drawn from the
    same sd (``truth_nuisance_sd="belief"``: the matched-prior experiment, in which the truth is a
    draw from the prior the belief uses) unless ``truth_nuisance_sd`` names another pair, which is a
    labelled misspecification experiment: ``"default"`` draws the truth from the package prior while
    the belief uses ``nuisance_sd`` — the construction the 2026-09-24 ``divide_nuis`` runs used by
    mistake (the override was applied after the truth had been drawn). The underlying standard-normal
    draw is the same in every case (``z_true = mu0 + sd * eps``), so the truth of a seed is a
    deterministic function of ``(seed, truth sd)``. Both sds are recorded in ``meta`` together with
    ``meta["misspecified"]``."""
    p = make_divide_problem(seed=seed, num_voxels=1, snr=snr, prior_scale=prior_scale, **kw)
    S0 = np.asarray(p.prior_cov, dtype=float).copy(); mu0 = np.asarray(p.prior_mean, dtype=float).ravel(); z_true = np.asarray(p.z_true, dtype=float).reshape(-1).copy()   # float64 whatever the JAX default
    sd_default = np.sqrt(np.diag(S0)); eps = (z_true - mu0) / sd_default            # exact: make_divide_problem draws mu0 + sd * N(0, 1)
    belief_sd = sd_default[:2] if nuisance_sd is None else np.asarray(nuisance_sd, dtype=float)
    if isinstance(truth_nuisance_sd, str) and truth_nuisance_sd == "belief":
        truth_sd = belief_sd
    elif isinstance(truth_nuisance_sd, str) and truth_nuisance_sd == "default":
        truth_sd = sd_default[:2]
    else:
        truth_sd = np.asarray(truth_nuisance_sd, dtype=float)
    if nuisance_sd is not None or not np.allclose(truth_sd, sd_default[:2]):
        S0[0, 0], S0[1, 1] = belief_sd[0] ** 2, belief_sd[1] ** 2
        z_true[:2] = mu0[:2] + truth_sd * eps[:2]
    ls = np.asarray(p.log_scale)
    target_fn = lambda Z: np.log(np.asarray(jax.vmap(divide_biomarkers)(jnp.asarray(Z, dtype=float)))) / ls
    return DesignProblem("divide", p.model, np.asarray(p.candidates), np.asarray(p.log_targets), mu0, S0, np.asarray(p.measurement_error), 0.0,
                         z_true, True, target_fn, ("logMD", "logMK_I", "logMK_A"),
                         meta={"seed": seed, "snr": snr, "prior_scale": prior_scale, "nuisance_sd": None if nuisance_sd is None else list(nuisance_sd),
                               "prior_sd_belief": np.sqrt(np.diag(S0)).tolist(), "prior_sd_truth": np.concatenate([truth_sd, sd_default[2:]]).tolist(),
                               "misspecified": bool(not np.allclose(truth_sd, belief_sd)), **kw})


# ---------------------------------------------------------------------------------------------
# noise and beliefs
# ---------------------------------------------------------------------------------------------

class NoiseTable:
    """Standard-normal draws ``(n_candidates, max_visits, d_y)`` indexed by candidate and visit."""

    def __init__(self, n, max_visits, dy, seed):
        self.table = np.random.default_rng(seed).normal(size=(n, max_visits, dy)); self.counts = np.zeros(n, dtype=int)

    def draw(self, i):
        if self.counts[i] >= self.table.shape[1]:
            raise IndexError(f"candidate {i} visited more than {self.table.shape[1]} times")
        e = self.table[i, self.counts[i]]; self.counts[i] += 1; return e


def observe(problem, i, noise, Rchol):
    """``f(z_true, x_i) + chol(R) ε`` with fresh ``ε`` from the table."""
    return problem.f_true[i] + Rchol @ noise.draw(i)


class EKFBelief:
    kind = "ekf"

    def __init__(self, problem):
        self.p = problem; self.mean = problem.mu0.copy(); self.cov = problem.S0.copy(); self.X, self.Y = [], []

    def update(self, i, y):
        x = self.p.X[i][None]; J = self.p.jac(self.mean, x).reshape(self.p.dy, -1); f = self.p.f(self.mean, x).reshape(-1)
        Sx = J @ self.cov @ J.T + self.p.R; K = np.linalg.solve(Sx, J @ self.cov).T
        self.mean = self.mean + K @ (y - f); self.cov = gd._sym(self.cov - K @ J @ self.cov); self.X.append(i); self.Y.append(np.asarray(y))

    def linearisation(self):
        return self.p.jac(self.mean, self.p.X), self.p.jac_targets(self.mean), self.cov


class LaplaceBelief(EKFBelief):
    """Refit on all data after every observation: MAP by LM from the previous mean and the prior
    mean, Gauss–Newton covariance. Falls back to the EKF update when there are no data."""
    kind = "laplace"

    def update(self, i, y):
        self.X.append(i); self.Y.append(np.asarray(y))
        post = ref.Posterior(self.p.model, self.p.X[self.X], np.array(self.Y), self.p.mu0, self.p.S0, self.p.R)
        z_map, cov, _ = ref.map_laplace(post, [self.mean, self.p.mu0])
        self.mean, self.cov = z_map, cov


class UKFBelief(EKFBelief):
    """The package's unscented filter (α = 1 by default), for Investigation 3."""
    kind = "ukf"

    def __init__(self, problem, alpha=1.0):
        super().__init__(problem)
        from bed.ukf import UKF
        self.ukf = UKF(problem.model, jnp.asarray(problem.mu0), jnp.asarray(problem.S0), 0.0, jnp.asarray(problem.R), alpha=alpha)

    def update(self, i, y):
        x = jnp.asarray(self.p.X[i])[None, None, None, :]
        m, c = self.ukf.get_state_posterior(jnp.asarray(y).reshape(1, 1, -1), x)
        self.ukf.state_prior = (m, c); self.mean, self.cov = np.asarray(m).ravel(), np.asarray(c); self.X.append(i); self.Y.append(np.asarray(y))

    def scores(self, crit):
        Xc = jnp.asarray(self.p.X)[:, None, None, :]
        if crit == "eig":
            return np.asarray(self.ukf.calculate_eig(Xc)).reshape(-1)
        Xt = jnp.asarray(self.p.Xt)[:, None, None, :]
        return np.asarray(self.ukf.calculate_epig(Xc, Xt)).reshape(-1)


def make_belief(problem, kind, **kw):
    return {"ekf": EKFBelief, "laplace": LaplaceBelief, "ukf": UKFBelief}[kind](problem, **kw)


# ---------------------------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------------------------

def psd_repair(S, rel=1e-8):
    """Project a symmetric matrix onto the positive-definite cone by clipping its eigenvalues at
    ``rel`` times the largest (1e-8, above the tolerance scipy uses for positive definiteness); returns the repaired matrix and the largest clipped magnitude
    relative to the largest eigenvalue (0 when nothing was clipped)."""
    w, V = np.linalg.eigh((S + S.T) / 2); floor = rel * w.max()
    clipped = float(max(0.0, (floor - w.min()) / w.max())) if w.min() < floor else 0.0
    return (V * np.maximum(w, floor)) @ V.T, clipped


def closed_form_scores(belief, crit):
    """Per-candidate closed-form scores from the belief's current linearisation. If the belief
    covariance is numerically indefinite (a Gauss–Newton covariance of an ill-conditioned refit),
    it is repaired by eigenvalue clipping and the repair is counted on the belief
    (``belief.psd_repairs``, ``belief.psd_worst``) so that it is reported, never silent."""
    if belief.kind == "ukf" and crit in ("epig", "eig"):
        return belief.scores(crit)
    J, Jt, S = belief.linearisation(); p = belief.p
    def scores(S):
        if crit == "epig":
            return gd.epig_scores(J, Jt, S, p.R, p.Rt)
        if crit == "eig":
            return gd.eig_scores(J, S, p.R)
        if crit == "var":
            return gd.variance_reduction_scores(J, Jt, S, p.R, p.Rt)
        raise ValueError(crit)
    try:
        return scores(S)
    except np.linalg.LinAlgError:
        S2, clipped = psd_repair(S)
        belief.psd_repairs = getattr(belief, "psd_repairs", 0) + 1; belief.psd_worst = max(getattr(belief, "psd_worst", 0.0), clipped)
        belief.cov = S2
        return scores(S2)


class GreedyPolicy:
    def __init__(self, crit):
        self.crit = crit; self.name = f"greedy-{crit}"

    def select(self, belief, used, rng):
        sc = mask_infeasible(belief.p, closed_form_scores(belief, self.crit), used)
        return int(np.argmax(sc)), sc


class FixedPolicy:
    """Outcome-independent protocol. With ``skip_used=True`` (problems without replacement whose
    warm start is drawn per subject) the protocol is executed as its first feasible entries in
    order: an entry already acquired is skipped, so the protocol must be longer than the budget by
    the warm-start size. Raises if it runs out of feasible entries."""

    def __init__(self, sequence, name="fixed", skip_used=False):
        self.sequence = [int(i) for i in sequence]; self.name = name; self.skip_used = skip_used

    def select(self, belief, used, rng):
        if not self.skip_used:
            return self.sequence[len(used) - self._offset], None
        feas = belief.p.feasible(used)
        for i in self.sequence:
            if feas[i] and i not in used[self._offset:]:
                return i, None
        raise ValueError(f"fixed protocol {self.name} exhausted after {len(used)} acquisitions")


class RandomPolicy:
    name = "random"

    def select(self, belief, used, rng):
        pool = np.arange(belief.p.n) if belief.p.replace else np.setdiff1d(np.arange(belief.p.n), used)
        return int(rng.choice(pool)), None


# ---------------------------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------------------------

def run_episode(problem, policy, belief_kind, warm, budget, noise_seed, rng_seed=0, snapshots=True, belief_kw=None):
    """Run one episode: the warm-start designs (given, outcome-independent), then ``budget``
    selections by the policy. Returns the history."""
    belief = make_belief(problem, belief_kind, **(belief_kw or {}))
    noise = NoiseTable(problem.n, budget + len(warm), problem.dy, noise_seed); Rchol = np.linalg.cholesky(problem.R)
    rng = np.random.default_rng(rng_seed); used = []; ys = []; snaps = []; score_trace = []
    if isinstance(policy, FixedPolicy):
        policy._offset = len(warm)
    for i in list(warm):
        if not problem.feasible(used)[i]:
            raise ValueError(f"warm start repeats candidate {i} in a problem without replacement")
        y = observe(problem, i, noise, Rchol); belief.update(i, y); used.append(int(i)); ys.append(y)
    if snapshots:
        snaps.append({"step": 0, "mean": belief.mean.copy(), "cov": belief.cov.copy()})
    for t in range(budget):
        i, sc = policy.select(belief, used, rng)
        if not problem.feasible(used)[i]:
            raise ValueError(f"policy {policy.name} selected infeasible candidate {i} at step {t} (used {used})")
        if sc is not None:
            top = np.argsort(-sc)[:5]
            score_trace.append({"argmax": int(i), "max": float(sc[i]), "top5": [int(k) for k in top], "top5_scores": [float(sc[k]) for k in top],
                                "margin": float(sc[i] - np.partition(sc, -2)[-2]) if sc.size > 1 else 0.0})
        y = observe(problem, i, noise, Rchol); belief.update(i, y); used.append(int(i)); ys.append(y)
        if snapshots:
            snaps.append({"step": t + 1, "mean": belief.mean.copy(), "cov": belief.cov.copy()})
    return {"problem": problem.name, "policy": policy.name, "belief": belief_kind, "warm": [int(i) for i in warm], "designs": used,
            "outcomes": np.array(ys), "noise_seed": noise_seed, "snapshots": snaps, "scores": score_trace, "final_mean": belief.mean.copy(), "final_cov": belief.cov.copy(),
            "psd_repairs": int(getattr(belief, "psd_repairs", 0)), "psd_worst": float(getattr(belief, "psd_worst", 0.0))}


def evaluate_history(problem, history, num_samples, seed, z_inits=None, robust=True):
    """Common reference readout of a history: MAP, Laplace, importance-sampled posterior moments of
    the terminal targets, squared errors against the truth, and the IS diagnostics. With
    ``robust=True`` the reference escalates to adaptive importance sampling and then MCMC when the
    Laplace-proposal sampler is unreliable (``bed.reference.robust_posterior``); the method used is
    recorded in ``ev["is"]["method"]``."""
    idx = history["designs"]; Y = np.asarray(history["outcomes"])
    post = ref.Posterior(problem.model, problem.X[idx], Y, problem.mu0, problem.S0, problem.R)
    inits = [problem.mu0, history["final_mean"]] + (list(z_inits) if z_inits else [])
    r = ref.robust_posterior(post, num_samples, seed, z_inits=inits) if robust else ref.importance_posterior(post, num_samples, seed, z_inits=inits)
    mean, var = r.moments(problem.target_fn); truth = problem.truth_targets
    map_t = np.asarray(problem.target_fn(r.z_map[None, :])).reshape(-1)
    Jt = problem.jac_targets(r.z_map).reshape(-1, problem.d)
    lap_var = np.einsum("qi,ij,qj->q", Jt, r.cov_laplace, Jt)
    plug = np.asarray(problem.target_fn(history["final_mean"][None, :])).reshape(-1)
    return {"post_mean": mean, "post_var": var, "sq_err_post": (mean - truth) ** 2, "sq_err_map": (map_t - truth) ** 2,
            "sq_err_plugin": (plug - truth) ** 2, "laplace_var": lap_var, "z_map": r.z_map, "is": r.diagnostics()}


def target_loss_summary(problem, ev):
    """Collapse the per-target squared errors to the study's terminal loss: per output (PK/PD, mean
    over the pool) or per biomarker (DIVIDE)."""
    e = np.asarray(ev["sq_err_post"]); v = np.asarray(ev["post_var"])
    if problem.name == "pkpd":
        e = e.reshape(-1, 2).mean(axis=0); v = v.reshape(-1, 2).mean(axis=0)
        plug = np.asarray(ev["sq_err_plugin"]).reshape(-1, 2).mean(axis=0); mp = np.asarray(ev["sq_err_map"]).reshape(-1, 2).mean(axis=0)
        return {"loss": e, "post_var": v, "loss_plugin": plug, "loss_map": mp, "names": ("logC", "E")}
    return {"loss": e, "post_var": v, "loss_plugin": np.asarray(ev["sq_err_plugin"]), "loss_map": np.asarray(ev["sq_err_map"]), "names": problem.target_names}
