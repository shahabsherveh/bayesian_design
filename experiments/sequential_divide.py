"""Candidate D sequential experiment (PROGRESS.md, D plan): DIVIDE biomarker protocol design.

Five arms share a warm start: EPIG (adaptive, closed form on the three standardised biomarkers with
zero target noise), EIG (adaptive, all parameters), RANDOM, FIXED-EPIG and FIXED-EIG (the protocol
chosen greedily on the prior alone, then filtered with the data). Repeated acquisitions are allowed.
Belief: EKF with the Jacobian re-linearised at the current mean. Every acquisition, including a repeat
of an already-used design, observes the true signal plus fresh Gaussian noise (a noise table shared
across arms gives common random numbers); the stored per-candidate labels are not used. Losses: squared error of the
standardised biomarkers against the truth, from the EKF plug-in estimate after every step and, at
the end, from the MAP and from the importance-sampled posterior mean; the sampled posterior sd of
each biomarker gives the calibration ratio. For V > 1 voxels the loop scores the ROI-summed
criterion and the MAP/IS are done per voxel (voxels are independent given the shared designs).

Usage: python experiments/sequential_divide.py [--draws 100] [--voxels 1] [--budget 27] [--warm 3] [--samples 20000] [--out results/sequential_divide_V1.json]
"""
import argparse, json, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from scipy.optimize import least_squares
from scipy.stats import multivariate_t
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from linearized_greedy import epig_scores, eig_scores, greedy, greedy_averaged, update
from bed.mechanistic import DivideModel, divide_biomarkers, make_divide_problem

ARMS = ("EPIG", "EIG", "RANDOM", "FIXED-EPIG", "FIXED-EIG", "FIXED-EPIG-B", "FIXED-EIG-B")


def std_biomarkers(z, scale):
    return np.asarray(divide_biomarkers(jnp.asarray(z))) / np.asarray(scale)


def ekf_step(mean, S, Ji, R, yi, fi):
    Sx = Ji @ S @ Ji.T + R; K = S @ Ji.T @ np.linalg.inv(Sx)
    mean = mean + K @ (yi - fi); S = S - K @ Ji @ S
    return mean, (S + S.T) / 2


def voxel_posterior(p1, X, y, mu0, S0, R_sd, num_samples, seed):
    """MAP, Laplace covariance and importance-sampled posterior mean / sd of the standardised
    biomarkers for ONE voxel (d = 5) given its observations y at designs X."""
    L0inv = np.linalg.inv(np.linalg.cholesky(S0)); scale = np.asarray(p1.scale)
    f = jax.jit(lambda z: p1(z, X).reshape(-1)); jac = jax.jit(lambda z: p1.jacobian(z, X).reshape(-1, 5))
    resid = lambda z: np.concatenate([(np.asarray(f(jnp.asarray(z))) - y) / R_sd, L0inv @ (z - mu0)])
    jacr = lambda z: np.concatenate([np.asarray(jac(jnp.asarray(z))) / R_sd, L0inv])
    best = None
    for z0 in (mu0, mu0 + 0.3 * np.linalg.cholesky(S0) @ np.random.default_rng(seed).normal(size=5)):
        r = least_squares(resid, z0, jac=jacr, method="lm", ftol=1e-8, xtol=1e-8, max_nfev=2000)
        if best is None or r.cost < best.cost: best = r
    Jw = jacr(best.x); cov = np.linalg.inv(Jw.T @ Jw)
    prop = multivariate_t(loc=best.x, shape=1.5 * cov, df=5, seed=seed); Z = prop.rvs(size=num_samples)
    logpost = jax.jit(jax.vmap(lambda z: -0.5 * (jnp.sum(((p1(z, X).reshape(-1) - jnp.asarray(y)) / R_sd) ** 2) + jnp.sum((jnp.asarray(L0inv) @ (z - jnp.asarray(mu0))) ** 2))))
    lw = np.asarray(logpost(jnp.asarray(Z))) - prop.logpdf(Z); w = np.exp(lw - lw.max()); w /= w.sum()
    G = np.asarray(jax.vmap(divide_biomarkers)(jnp.asarray(Z))) / scale
    mean = w @ G; sd = np.sqrt(w @ (G - mean) ** 2)
    LG = np.log(G * scale) / np.asarray(p1.log_scale); lmean = w @ LG; lsd = np.sqrt(w @ (LG - lmean) ** 2)
    return std_biomarkers(best.x, scale), mean, sd, float(1.0 / np.sum(w ** 2)), lmean, lsd


def run_arm(p, arm, warm, budget, rng, num_samples, seed, fixed=None, noise=None):
    """``noise`` is a (n_candidates, max_acquisitions, V) table of standard normal draws shared by all
    arms of one truth draw: the r-th acquisition of candidate i observes f(z_true, x_i) + sd *
    noise[i, r], so repeats carry independent noise and paired arms see common random numbers. The
    stored per-candidate label y_train is NOT used (it would return the same value on every repeat)."""
    V = p.model.V; X = np.asarray(p.candidates); R = np.asarray(p.measurement_error); sd_noise = float(np.sqrt(R[0, 0]))
    f_true = np.asarray(p.model(p.z_true.reshape(-1), p.candidates)).reshape(X.shape[0], V)
    counts = np.zeros(X.shape[0], dtype=int); ys = []
    def observe(i):
        y = f_true[i] + sd_noise * noise[i, counts[i]]; counts[i] += 1; ys.append(y); return y
    Xt = np.asarray(p.targets); Xt_log = np.asarray(p.log_targets); mu0 = np.asarray(p.prior_mean).ravel(); S0 = np.asarray(p.prior_cov)
    truth = np.stack([std_biomarkers(p.z_true[v], p.biomarker_scale) for v in range(V)])          # (V, 3)
    jac = jax.jit(lambda z, x: p.model.jacobian(z, x)); fwd = jax.jit(lambda z, x: p.model(z, x))
    mean, S = mu0.copy(), S0.copy(); used = list(warm)
    for i in used:
        Ji = np.asarray(jac(mean, X[i][None])).reshape(V, -1); mean, S = ekf_step(mean, S, Ji, R, observe(i), np.asarray(fwd(mean, X[i][None])).reshape(V))
    Rt = 1e-12 * np.eye(V); curve = []
    def plug_in(m): return np.stack([std_biomarkers(m.reshape(V, 5)[v], p.biomarker_scale) for v in range(V)])
    curve.append(np.mean((plug_in(mean) - truth) ** 2, axis=0).tolist())
    for step in range(budget):
        if arm.startswith("FIXED"): i = int(fixed[step])
        elif arm == "RANDOM": i = int(rng.integers(0, X.shape[0]))
        else:
            J = np.asarray(jac(mean, X)); Jt = np.asarray(jac(mean, Xt_log if arm == "EPIG-LOG" else Xt))
            sc = epig_scores(J, Jt, S, R, Rt) if arm.startswith("EPIG") else eig_scores(J, S, R); i = int(np.argmax(sc))
        used.append(i)
        Ji = np.asarray(jac(mean, X[i][None])).reshape(V, -1); mean, S = ekf_step(mean, S, Ji, R, observe(i), np.asarray(fwd(mean, X[i][None])).reshape(V))
        curve.append(np.mean((plug_in(mean) - truth) ** 2, axis=0).tolist())
    # exact posterior per voxel
    p1 = DivideModel(1, p.biomarker_scale, form=p.model.form, log_scale=p.log_scale); Xs = jnp.asarray(X[used]); R_sd = float(np.sqrt(R[0, 0])); Yacq = np.array(ys)   # (n_acq, V)
    S0v = np.asarray(p.prior_cov)[:5, :5]; mu0v = mu0[:5]
    maps, posts, sds, ess, lposts, lsds = [], [], [], [], [], []
    for v in range(V):
        m_, pm, sd, e, lm, ls = voxel_posterior(p1, Xs, Yacq[:, v], mu0v, S0v, R_sd, num_samples, seed=10000 * seed + v); maps.append(m_); posts.append(pm); sds.append(sd); ess.append(e); lposts.append(lm); lsds.append(ls)
    maps, posts, sds, lposts, lsds = np.array(maps), np.array(posts), np.array(sds), np.array(lposts), np.array(lsds)
    ltruth = np.log(truth * np.asarray(p.biomarker_scale)) / np.asarray(p.log_scale)
    # EKF predictive sd of the standardised biomarkers at the end (per voxel, from the linearised covariance)
    Jt_final = np.asarray(jac(mean, Xt)).reshape(3, V, -1); ekf_sd = np.sqrt(np.stack([[Jt_final[k, v] @ S @ Jt_final[k, v] for k in range(3)] for v in range(V)]))
    return {"selected": [int(i) for i in used[len(warm):]], "curve_ekf": curve,
            "sq_err_ekf": np.mean((plug_in(mean) - truth) ** 2, axis=0).tolist(), "sq_err_map": np.mean((maps - truth) ** 2, axis=0).tolist(),
            "sq_err_post": np.mean((posts - truth) ** 2, axis=0).tolist(), "post_var": np.mean(sds ** 2, axis=0).tolist(), "ekf_var": np.mean(ekf_sd ** 2, axis=0).tolist(),
            "log_sq_err_post": np.mean((lposts - ltruth) ** 2, axis=0).tolist(), "log_post_var": np.mean(lsds ** 2, axis=0).tolist(),
            "ess_min": float(min(ess))}


def main(draws, voxels, budget, warm_start, num_samples, start=0, snr=40.0):
    rows = []
    for d in range(start, start + draws):
        p = make_divide_problem(seed=100 + d, num_voxels=voxels, snr=snr); rng = np.random.default_rng(d)
        n = p.candidates.shape[0]; warm = [int(v) for v in rng.choice(n, warm_start, replace=False)]
        # fixed protocols from the prior alone (design-only, data-free), with the warm-start designs already taken
        J0 = np.asarray(p.model.jacobian(p.prior_mean, p.candidates)); Jt0 = np.asarray(p.model.jacobian(p.prior_mean, p.targets)); S0 = np.asarray(p.prior_cov); R = np.asarray(p.measurement_error)
        fixed = {c: greedy(J0, Jt0, S0, R, budget, warm, c, np.random.default_rng(0), Rt=1e-12 * np.eye(voxels), replace=True, per_target=True)[1] for c in ("epig", "eig")}
        # pseudo-Bayesian fixed protocols: the criterion averaged over K = 24 prior draws
        rngB = np.random.default_rng(5000 + d); ZB = np.asarray(p.prior_mean).ravel() + np.sqrt(np.diag(S0)) * rngB.normal(size=(24, S0.shape[0]))
        Js = np.stack([np.asarray(p.model.jacobian(z, p.candidates)) for z in ZB]); Jts = np.stack([np.asarray(p.model.jacobian(z, p.targets)) for z in ZB])
        fixed["epig_b"] = greedy_averaged(Js, Jts, S0, R, budget, warm, "epig", Rt=1e-12 * np.eye(voxels)); fixed["eig_b"] = greedy_averaged(Js, Jts, S0, R, budget, warm, "eig")
        row = {"draw": d, "warm": warm, "truth_biomarkers": [std_biomarkers(p.z_true[v], p.biomarker_scale).tolist() for v in range(voxels)]}
        noise = np.random.default_rng(777 + d).normal(size=(n, budget + warm_start, voxels))   # common random numbers across arms
        for k, arm in enumerate(ARMS):
            fx = {"FIXED-EPIG": fixed["epig"], "FIXED-EIG": fixed["eig"], "FIXED-EPIG-B": fixed["epig_b"], "FIXED-EIG-B": fixed["eig_b"]}.get(arm)
            row[arm] = run_arm(p, arm, warm, budget, np.random.default_rng(1000 * d + k), num_samples, seed=d, fixed=fx, noise=noise)
        rows.append(row)
        e = {a: sum(row[a]["sq_err_post"]) for a in ARMS}
        print(f"draw {d:3d}: summed standardised sq. error (posterior mean) " + "  ".join(f"{a} {e[a]:.3f}" for a in ARMS) + f" | ESS min {row['EPIG']['ess_min']:.0f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=100); ap.add_argument("--start", type=int, default=0); ap.add_argument("--voxels", type=int, default=1)
    ap.add_argument("--budget", type=int, default=27); ap.add_argument("--warm", type=int, default=3); ap.add_argument("--samples", type=int, default=20000); ap.add_argument("--out", default=None); ap.add_argument("--snr", type=float, default=40.0)
    a = ap.parse_args(); t0 = time.time(); out = a.out or f"results/sequential_divide_V{a.voxels}_snr{int(a.snr)}.json"
    rows = main(a.draws, a.voxels, a.budget, a.warm, a.samples, start=a.start, snr=a.snr)
    Path(out).parent.mkdir(exist_ok=True)
    json.dump({"draws": a.draws, "start": a.start, "voxels": a.voxels, "snr": a.snr, "budget": a.budget, "warm": a.warm, "arms": ARMS, "rows": rows}, open(out, "w"), indent=1)
    print(f"wrote {out} in {time.time() - t0:.0f}s")
