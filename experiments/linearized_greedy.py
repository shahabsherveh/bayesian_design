"""Data-free pre-flight: greedy EPIG and EIG design under the model linearised at the prior mean.
In the linear-Gaussian approximation the posterior covariance does not depend on the observed values,
so the whole sequential design and its predictive variance at the pool are computable from the
Jacobians alone. Reports, after each step, the ratio of the pool predictive variance (noise-free,
per output) under EPIG-greedy, EIG-greedy and random designs. Calibrated on A (PROGRESS.md A-5: the
exact posterior predictive-variance ratio EPIG/EIG was 0.75 on E and 0.90 on log C at budget 15).

Usage: python experiments/linearized_greedy.py pkpd [--budget 15]   (the DIVIDE forecast lives in preflight_divide.py)
"""
import argparse, json
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from bed import mechanistic as M


def jacobians(p):
    z = p.prior_mean; X, pool = p.data.x_train, p.data.x_test_pool
    J = np.asarray(p.model.jacobian(z, X)); Jt = np.asarray(p.model.jacobian(z, pool))
    return J, Jt, np.asarray(p.prior_cov), np.asarray(p.measurement_error)


def epig_scores(J, Jt, S, R, Rt=None):
    """Closed-form EPIG per candidate (mean over pool) for belief covariance S; J (n, dy, d), Jt (m, dy, d).
    ``Rt`` is the target-side noise covariance (defaults to R); pass a tiny jitter for noise-free
    functionals such as biomarkers."""
    Rt = R if Rt is None else Rt
    Sx = np.einsum("nid,de,nje->nij", J, S, J) + R                    # (n, dy, dy)
    Sp = np.einsum("mid,de,mje->mij", Jt, S, Jt) + Rt                 # (m, dy, dy)
    C = np.einsum("mid,de,nje->mnij", Jt, S, J)                       # (m, n, dy, dy)
    Sx_inv = np.linalg.inv(Sx)
    red = np.einsum("mnij,njk,mnlk->mnil", C, Sx_inv, C)              # C Sx^-1 C^T  (m, n, dy, dy)
    post = Sp[:, None] - red
    return 0.5 * (np.linalg.slogdet(Sp)[1][:, None] - np.linalg.slogdet(post)[1]).mean(axis=0)


def eig_scores(J, S, R):
    Sx = np.einsum("nid,de,nje->nij", J, S, J) + R
    return 0.5 * (np.linalg.slogdet(Sx)[1] - np.linalg.slogdet(R)[1])


def update(S, Ji, R):
    Sx = Ji @ S @ Ji.T + R; K = S @ Ji.T @ np.linalg.inv(Sx)
    S = S - K @ Ji @ S; return (S + S.T) / 2


def pool_var(S, Jt, per_target=False):
    """Noise-free predictive variance at the pool: per output, averaged over the pool (default), or
    per target point (summed over outputs) with ``per_target=True``."""
    v = np.einsum("mid,de,mie->mi", Jt, S, Jt)
    return v.sum(axis=1) if per_target else v.mean(axis=0)


def greedy(J, Jt, S0, R, budget, warm, crit, rng, Rt=None, replace=False, per_target=False):
    S = S0.copy(); used = list(warm)
    for i in used: S = update(S, J[i], R)
    curve = [pool_var(S, Jt, per_target)]
    for _ in range(budget):
        if crit == "random": i = int(rng.choice(np.arange(J.shape[0]) if replace else np.setdiff1d(np.arange(J.shape[0]), used)))
        else:
            sc = epig_scores(J, Jt, S, R, Rt) if crit == "epig" else eig_scores(J, S, R); sc = np.array(sc, dtype=float)
            if not replace: sc[used] = -np.inf
            i = int(sc.argmax())
        used.append(i); S = update(S, J[i], R); curve.append(pool_var(S, Jt, per_target))
    return np.array(curve), used[len(warm):]


def greedy_averaged(Js, Jts, S0, R, budget, warm, crit, Rt=None, replace=True):
    """Pseudo-Bayesian greedy design: the criterion is averaged over K prior draws, each with its
    own Jacobians and its own belief covariance (a locally optimal design linearises at one point;
    this averages the closed form over the prior instead). Js: (K, n, dy, d), Jts: (K, m, dy, d)."""
    K = Js.shape[0]; Ss = [S0.copy() for _ in range(K)]; used = list(warm)
    for i in used:
        Ss = [update(Ss[k], Js[k][i], R) for k in range(K)]
    for _ in range(budget):
        sc = np.zeros(Js.shape[1])
        for k in range(K):
            sc += (epig_scores(Js[k], Jts[k], Ss[k], R, Rt) if crit == "epig" else eig_scores(Js[k], Ss[k], R)) / K
        if not replace: sc[used] = -np.inf
        i = int(np.argmax(sc)); used.append(i); Ss = [update(Ss[k], Js[k][i], R) for k in range(K)]
    return used[len(warm):]


def run(p, budget=15, seeds=10, warm_start=3, label=""):
    J, Jt, S0, R = jacobians(p); n = J.shape[0]; res = {}
    for crit in ("epig", "eig", "random"):
        curves = []; sel = []
        for s in range(seeds):
            rng = np.random.default_rng(s); warm = list(rng.choice(n, warm_start, replace=False))
            c, u = greedy(J, Jt, S0, R, budget, warm, crit, np.random.default_rng(100 + s)); curves.append(c); sel.append(u)
        res[crit] = (np.array(curves), sel)
    out = {}
    for k in sorted({k for k in (3, 5, 10, budget) if k <= budget}):
        e, i, r = (res[c][0][:, k].mean(axis=0) for c in ("epig", "eig", "random"))
        out[k] = {"epig_over_eig": (e / i).round(3).tolist(), "epig_over_random": (e / r).round(3).tolist(), "eig_over_random": (i / r).round(3).tolist()}
        print(f"  {label} step {k:2d}: predictive-variance ratio per output  EPIG/EIG {np.round(e / i, 3)}   EPIG/RAND {np.round(e / r, 3)}   EIG/RAND {np.round(i / r, 3)}")
    X = np.asarray(p.data.x_train).reshape(n, -1)
    for c in ("epig", "eig"):
        sel = np.concatenate(res[c][1]); print(f"  {label} {c.upper()} selections: median design {np.round(np.median(X[sel], axis=0), 4)}; distinct designs {len(set(sel.tolist()))}")
    return out


if __name__ == "__main__":
    # The PK/PD forecast, calibrated against the measured outcome (PROGRESS.md A-5/B-3). The DIVIDE
    # forecast, with noise-free biomarker targets and repeats, is run by experiments/preflight_divide.py.
    ap = argparse.ArgumentParser(); ap.add_argument("problem", choices=["pkpd"]); ap.add_argument("--budget", type=int, default=15)
    a = ap.parse_args()
    p = M.make_pkpd_problem(seed=0); print("A (PK/PD), linearised at the prior mean; A-5 measured exact predictive-variance ratios EPIG/EIG 0.90 (log C), 0.75 (E) and EPIG/RAND 0.72, 0.55 at budget 15:")
    run(p, budget=a.budget, label="A")
