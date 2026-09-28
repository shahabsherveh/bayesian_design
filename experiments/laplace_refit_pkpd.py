"""Does the sequential EKF's distorted covariance (PROGRESS.md A-4: predictive sd at the pool 3x too
wide on log C after 15 updates, marginals fine, so the correlations are wrong) cost the criterion
anything? Five arms share a warm start: EPIG and EIG scored from the sequential EKF belief (as the
runner does), EPIG and EIG scored from a Laplace belief refitted on all data after every
observation (MAP + Gauss-Newton covariance, i.e. a re-linearized EKF), and RANDOM. Losses use the
exact posterior predictive mean by importance sampling on the final data set.

Predictions registered before running (2026-09-18):
  P1  expected pool MSE on E, EPIG-Laplace / EIG-Laplace ~ 0.8, as under the EKF (the design effect
      is a property of the problem, not of the belief's distortion);
  P2  EPIG-Laplace / EPIG-EKF in expected pool MSE between 0.85 and 1.0 (a faithful belief cannot
      hurt much, and the distortion is common to both outputs' ranking only through their weighting);
  P3  Laplace predictive sd at the pool / exact within 0.9..1.1 at every step, EKF's 1.3..3.5.

Usage: python experiments/laplace_refit_pkpd.py [--draws 60] [--budget 15] [--samples 20000]
"""
import argparse, json, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from bed.ekf import EKF
from bed.mechanistic import make_pkpd_problem
sys.path.insert(0, str(Path(__file__).resolve().parent)); from falsify_pkpd import fit_map, posterior_predictive

ARMS = ("EPIG-EKF", "EIG-EKF", "EPIG-LAP", "EIG-LAP", "RANDOM")


def run_arm(p, arm, warm, budget, rng, num_samples, key):
    X, Y, pool, y_pool = p.data.x_train, p.data.y_train, p.data.x_test_pool, np.asarray(p.data.y_test_pool).reshape(-1, 2)
    n = X.shape[0]; used = list(warm)
    ekf = EKF(p.model, p.prior_mean, p.prior_cov, 0.0, p.measurement_error)
    for i in used:
        ekf.state_prior = ekf.get_state_posterior(Y[i][None], X[i][None])
    curve_map, lap_sd, ekf_sd = [], [], []
    for _ in range(budget):
        z_map, cov, _ = fit_map(p, used, [p.prior_mean, np.asarray(ekf.state_prior[0]).ravel()])
        lap = EKF(p.model, jnp.asarray(z_map).reshape(-1, 1), jnp.asarray(cov), 0.0, p.measurement_error)
        if arm == "RANDOM":
            i = int(rng.choice(np.setdiff1d(np.arange(n), used)))
        else:
            filt = ekf if arm.endswith("EKF") else lap
            score = np.array(filt.calculate_epig(X, pool) if arm.startswith("EPIG") else filt.calculate_eig(X), dtype=float)
            score[used] = -np.inf; i = int(score.argmax())
        used.append(i)
        ekf.state_prior = ekf.get_state_posterior(Y[i][None], X[i][None])
        # bookkeeping after the update: MAP-based pool RMSE, and the two beliefs' predictive sd at the pool
        z_map, cov, _ = fit_map(p, used, [p.prior_mean, np.asarray(ekf.state_prior[0]).ravel()])
        pred = np.asarray(p.model(z_map, pool)).reshape(-1, 2); curve_map.append([float(v) for v in np.sqrt(np.mean((pred - y_pool) ** 2, 0))])
        Jt = np.asarray(p.model.jacobian(z_map, pool)).reshape(-1, 2, 7); lap_sd.append([float(v) for v in np.sqrt(np.mean(np.einsum("mki,ij,mkj->mk", Jt, cov, Jt), 0))])
        _, S = ekf.measurement_prior(pool); ekf_sd.append([float(v) for v in np.sqrt(np.mean(np.diagonal(np.asarray(S).reshape(-1, 2, 2), axis1=1, axis2=2), 0))])
    z_map, cov, logpost = fit_map(p, used, [p.prior_mean, np.asarray(ekf.state_prior[0]).ravel()])
    pm, psd, ess = posterior_predictive(p, used, z_map, cov, logpost, num_samples, key)
    return {"selected": [int(v) for v in used[len(warm):]], "curve_map": curve_map, "lap_pred_sd": lap_sd, "ekf_pred_sd": ekf_sd,
            "rmse_post": [float(v) for v in np.sqrt(np.mean((pm - y_pool) ** 2, 0))], "post_pred_sd": [float(v) for v in np.sqrt(np.mean(psd ** 2, 0))],
            "rmse_ekfmean": [float(v) for v in np.sqrt(np.mean((np.asarray(p.model(ekf.state_prior[0], pool)).reshape(-1, 2) - y_pool) ** 2, 0))], "ess": ess}


def main(draws, budget, num_samples, warm_start=3, start=0):
    out = []
    for d in range(start, start + draws):
        p = make_pkpd_problem(seed=100 + d); rng = np.random.default_rng(d)
        warm = [int(v) for v in rng.choice(p.data.x_train.shape[0], warm_start, replace=False)]
        row = {"draw": d, "warm": warm}
        for k, arm in enumerate(ARMS):
            row[arm] = run_arm(p, arm, warm, budget, np.random.default_rng(1000 * d + k), num_samples, key=10000 * d + k)
        out.append(row)
        print(f"draw {d:2d}: E pool RMSE (posterior mean) " + "  ".join(f"{a} {row[a]['rmse_post'][1]:5.2f}" for a in ARMS) + f" | final pred sd at pool, E: EKF/exact {row['EPIG-EKF']['ekf_pred_sd'][-1][1]/row['EPIG-EKF']['post_pred_sd'][1]:.2f}, Laplace/exact {row['EPIG-LAP']['lap_pred_sd'][-1][1]/row['EPIG-LAP']['post_pred_sd'][1]:.2f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=60); ap.add_argument("--start", type=int, default=0, help="first draw index (shard)"); ap.add_argument("--budget", type=int, default=15); ap.add_argument("--samples", type=int, default=20000); ap.add_argument("--out", default="results/laplace_refit_pkpd.json")
    a = ap.parse_args(); t = time.time()
    out = main(a.draws, a.budget, a.samples, start=a.start)
    json.dump({"draws": a.draws, "start": a.start, "budget": a.budget, "samples": a.samples, "arms": ARMS, "rows": out}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} in {time.time() - t:.0f}s")
