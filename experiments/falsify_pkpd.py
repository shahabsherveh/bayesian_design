"""Adversarial checks on candidate A (PROGRESS.md A-3). Reruns the EKF sequential experiment on
the first N draws and, on the same selected data sets, replaces the runner's plug-in EKF mean by
(i) the MAP estimate under the Gaussian prior and (ii) the posterior predictive mean by importance
sampling from a Laplace proposal. If EPIG beats EIG under (i)/(ii) but not under the EKF mean, the
A-3 conclusion is an artifact of the estimator. Also records: the Laplace predictive variance at
the pool (the quantity EPIG targets) for each criterion's final data set, the calibration of the
EKF belief (Mahalanobis distance of the MAP under the EKF posterior, EKF predictive sd at the pool
against the sampled posterior predictive sd), and the importance-sampling effective sample size.

Usage: python experiments/falsify_pkpd.py [--draws 60] [--budget 15] [--samples 40000]
"""
import argparse, json, time
import jax, jax.numpy as jnp, numpy as np
from scipy.optimize import least_squares
from scipy.stats import multivariate_t
from copy import deepcopy
jax.config.update("jax_enable_x64", True)
from bed.experiments import Experiment
from bed.mechanistic import make_pkpd_problem

CRITS = ("EPIG", "EIG", "RANDOM")


def fit_map(p, idx, z0):
    X, Y = np.asarray(p.data.x_train)[idx].reshape(len(idx), 4), np.asarray(p.data.y_train)[idx].reshape(len(idx), 2)
    Rinv_sqrt = 1.0 / np.sqrt(np.diag(np.asarray(p.measurement_error))); L0inv = 1.0 / np.sqrt(np.diag(np.asarray(p.prior_cov))); mu0 = np.asarray(p.prior_mean).ravel()
    f = jax.jit(lambda z: p.model(z, X).reshape(len(idx), 2))
    jac = jax.jit(lambda z: p.model.jacobian(z, X).reshape(len(idx), 2, 7))
    def resid(z): return np.concatenate([((np.asarray(f(z)) - Y) * Rinv_sqrt).ravel(), (z - mu0) * L0inv])
    def jacr(z): return np.concatenate([(np.asarray(jac(z)) * Rinv_sqrt[None, :, None]).reshape(-1, 7), np.diag(L0inv)])
    Yj, Rj, Lj, muj = jnp.asarray(Y), jnp.asarray(Rinv_sqrt), jnp.asarray(L0inv), jnp.asarray(mu0)
    logpost = jax.jit(jax.vmap(lambda z: -0.5 * (jnp.sum(((p.model(z, X).reshape(len(idx), 2) - Yj) * Rj) ** 2) + jnp.sum(((z - muj) * Lj) ** 2))))
    best = None
    for start in z0:
        r = least_squares(resid, np.asarray(start).ravel(), jac=jacr, method="lm", max_nfev=2000)
        if best is None or r.cost < best.cost: best = r
    J = jacr(best.x); cov = np.linalg.inv(J.T @ J)   # Laplace covariance at the MAP
    return best.x, cov, logpost


def posterior_predictive(p, idx, z_map, cov, resid, num_samples, key):
    """Importance sampling with a multivariate-t proposal around the MAP; returns (mean, sd) of the
    pool predictions under the posterior, and the effective sample size."""
    pool = p.data.x_test_pool; M = pool.shape[0]
    prop = multivariate_t(loc=z_map, shape=1.5 * cov, df=5, seed=int(key))
    Z = prop.rvs(size=num_samples)
    logq = prop.logpdf(Z)
    # log posterior up to a constant: -0.5 * ||resid||^2 (residuals already whitened, prior included)
    logp = np.asarray(resid(jnp.asarray(Z)))
    w = np.exp(logp - logq - np.max(logp - logq)); w /= w.sum(); ess = 1.0 / np.sum(w ** 2)
    F = np.asarray(jax.vmap(lambda z: p.model(z, pool).reshape(M, 2))(jnp.asarray(Z)))   # (S, M, 2)
    mean = np.einsum("s,smk->mk", w, F); sd = np.sqrt(np.einsum("s,smk->mk", w, (F - mean) ** 2))
    return mean, sd, float(ess)


def main(draws, budget, num_samples, warm_start=3, start=0):
    out = []
    for d in range(start, start + draws):
        p = make_pkpd_problem(seed=100 + d)
        exp = Experiment(model=p.model, data=p.data, latent_cov=p.prior_cov, latent_mean=p.prior_mean,
                         measurement_error=p.measurement_error, warm_start=warm_start, seed=d)
        res = exp.run_experiment(criteria=list(CRITS), filter_types=["ekf"], filter_params=[{}], iterations=budget, optimizer_method="brute_force", trace=True)
        ek = res.experiment_results_dict["ekf"]
        warm = [int(i) for i in ek["WARM-START"].selected_indices]
        y_pool = np.asarray(p.data.y_test_pool).reshape(-1, 2)
        row = {"draw": d, "warm": warm, "ess": {}}
        for c in CRITS:
            r = ek[c]; idx = warm + [int(i) for i in r.selected_indices]
            # the trace stores the belief before each update; apply the last observation to get the final posterior
            filt = deepcopy(r.filters[-1]); last = idx[-1]
            filt.state_prior = filt.get_state_posterior(p.data.y_train[last][None], p.data.x_train[last][None])
            ekf_mean = np.asarray(filt.state_prior[0]).ravel()
            r.filters.append(filt)
            z_map, cov, resid = fit_map(p, idx, [p.prior_mean] + ([ekf_mean] if ekf_mean is not None else []))
            pred_map = np.asarray(p.model(z_map, p.data.x_test_pool)).reshape(-1, 2)
            pm, psd, ess = posterior_predictive(p, idx, z_map, cov, resid, num_samples, key=1000 * d + len(c))
            Jt = np.asarray(p.model.jacobian(z_map, p.data.x_test_pool)).reshape(-1, 2, 7)
            lap_var = np.einsum("mki,ij,mkj->mk", Jt, cov, Jt)      # Laplace predictive variance at the pool
            row[c] = {"rmse_ekf": [float(v) for v in r.rmse_pool_outputs[-1]],
                      "rmse_map": [float(v) for v in np.sqrt(np.mean((pred_map - y_pool) ** 2, axis=0))],
                      "rmse_post": [float(v) for v in np.sqrt(np.mean((pm - y_pool) ** 2, axis=0))],
                      "lap_pred_sd": [float(v) for v in np.sqrt(np.mean(lap_var, axis=0))],
                      "post_pred_sd": [float(v) for v in np.sqrt(np.mean(psd ** 2, axis=0))],
                      "ess": ess, "z_map": [float(v) for v in z_map]}
            # EKF belief calibration: Mahalanobis distance of the MAP under the EKF posterior, and EKF predictive sd at the pool
            try:
                filt = r.filters[-1]; m, S = filt.state_prior; S = np.asarray(S).reshape(7, 7); dz = z_map - np.asarray(m).ravel()
                row[c]["ekf_mahal_map"] = float(np.sqrt(dz @ np.linalg.solve(S, dz)))
                _, cov_pool = filt.measurement_prior(p.data.x_test_pool)
                row[c]["ekf_pred_sd"] = [float(v) for v in np.sqrt(np.mean(np.diagonal(np.asarray(cov_pool).reshape(-1, 2, 2), axis1=1, axis2=2), axis=0))]
            except Exception as e:
                row[c]["ekf_mahal_map"] = None
        out.append(row)
        e, i = row["EPIG"], row["EIG"]
        print(f"draw {d:2d}: E pool RMSE  EKF-mean EPIG {e['rmse_ekf'][1]:5.2f} EIG {i['rmse_ekf'][1]:5.2f} | MAP {e['rmse_map'][1]:5.2f} {i['rmse_map'][1]:5.2f} | post-mean {e['rmse_post'][1]:5.2f} {i['rmse_post'][1]:5.2f} | Laplace pred sd {e['lap_pred_sd'][1]:5.2f} {i['lap_pred_sd'][1]:5.2f} | ESS {e['ess']:.0f}/{i['ess']:.0f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=60); ap.add_argument("--start", type=int, default=0, help="first draw index (shard)"); ap.add_argument("--budget", type=int, default=15); ap.add_argument("--samples", type=int, default=40000); ap.add_argument("--out", default="results/falsify_pkpd.json")
    a = ap.parse_args(); t = time.time()
    out = main(a.draws, a.budget, a.samples, start=a.start)
    json.dump({"draws": a.draws, "start": a.start, "budget": a.budget, "samples": a.samples, "rows": out}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} in {time.time() - t:.0f}s")
