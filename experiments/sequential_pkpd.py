"""Candidate A sequential experiment (PROGRESS.md Phase 2): synthetic PK/PD, truth drawn from the
prior, 3 random warm-start samples, budget 15, EKF and UKF(alpha=1), criteria EPIG / EIG / RANDOM.
Predictions registered in PROGRESS.md before the first run.

Usage: python experiments/sequential_pkpd.py [--draws 10] [--budget 15] [--prior-scale 1.0] [--out results/sequential_pkpd.json]
"""
import argparse, json, time
from pathlib import Path
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from bed.experiments import Experiment
from bed.mechanistic import make_pkpd_problem


def main(draws, budget, prior_scale, warm_start=3):
    rows = []
    for draw in range(draws):
        p = make_pkpd_problem(seed=100 + draw, prior_scale=prior_scale)
        exp = Experiment(model=p.model, data=p.data, latent_cov=p.prior_cov, latent_mean=p.prior_mean,
                         measurement_error=p.measurement_error, warm_start=warm_start, seed=draw)
        res = exp.run_experiment(criteria=["EPIG", "EIG", "RANDOM"], filter_types=["ekf", "ukf"],
                                 filter_params=[{}, {"alpha": 1.0}], iterations=budget, optimizer_method="brute_force")
        for ft, d in res.experiment_results_dict.items():
            for crit in ("EPIG", "EIG", "RANDOM"):
                r = d[crit]
                rows.append({"draw": draw, "filter": ft, "criterion": crit,
                             "pool_rmse_logC": float(r.rmse_pool_outputs[-1][0]), "pool_rmse_E": float(r.rmse_pool_outputs[-1][1]),
                             "glob_rmse_logC": float(r.rmse_glob_outputs[-1][0]), "glob_rmse_E": float(r.rmse_glob_outputs[-1][1]),
                             "pool_curve_logC": [float(v[0]) for v in r.rmse_pool_outputs], "pool_curve_E": [float(v[1]) for v in r.rmse_pool_outputs],
                             "final_crit": float(r.crit_values[-1]), "selected": [int(i) for i in r.selected_indices],
                             "selected_times": [float(v[0].ravel()[3]) for v in r.selected_designs], "selected_doses": [float(v[0].ravel()[0]) for v in r.selected_designs]})
        e = {(row["filter"], row["criterion"]): row for row in rows if row["draw"] == draw}
        print(f"draw {draw}: pool RMSE logC  EKF EPIG {e[('ekf','EPIG')]['pool_rmse_logC']:.3f} EIG {e[('ekf','EIG')]['pool_rmse_logC']:.3f} RAND {e[('ekf','RANDOM')]['pool_rmse_logC']:.3f} | UKF EPIG {e[('ukf','EPIG')]['pool_rmse_logC']:.3f} EIG {e[('ukf','EIG')]['pool_rmse_logC']:.3f} || "
              f"pool RMSE E  EKF EPIG {e[('ekf','EPIG')]['pool_rmse_E']:.2f} EIG {e[('ekf','EIG')]['pool_rmse_E']:.2f} RAND {e[('ekf','RANDOM')]['pool_rmse_E']:.2f} | UKF EPIG {e[('ukf','EPIG')]['pool_rmse_E']:.2f} EIG {e[('ukf','EIG')]['pool_rmse_E']:.2f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=10); ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--prior-scale", type=float, default=1.0); ap.add_argument("--out", default="results/sequential_pkpd.json")
    a = ap.parse_args(); t = time.time()
    rows = main(a.draws, a.budget, a.prior_scale)
    Path(a.out).parent.mkdir(exist_ok=True)
    json.dump({"draws": a.draws, "budget": a.budget, "prior_scale": a.prior_scale, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} in {time.time() - t:.0f}s")
