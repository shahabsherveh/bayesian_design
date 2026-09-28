"""Aggregate the sharded outputs of `falsify_pkpd.py` and `laplace_refit_pkpd.py` (PROGRESS.md
A-4). Reports, for each output, the ratio of mean pool MSE over draws with a paired bootstrap CI
(the Bayes-risk comparison), win rates, the exact posterior predictive variance at the pool, and
for the Laplace-refit runs the three registered predictions P1-P3.

Usage: python experiments/analyze_falsify_pkpd.py "results/pelle/falsify_300_s*.json" "results/pelle/laplace_refit_s*.json"
"""
import glob, json, sys
import numpy as np

rng = np.random.default_rng(0)


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += json.load(open(f))["rows"]
    return rows


def ratio_ci(a, b, n=4000):
    D = len(a); idx = [rng.integers(0, D, D) for _ in range(n)]; bs = [a[i].mean() / b[i].mean() for i in idx]
    return f"{a.mean() / b.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"


def winrate(a, b):
    D = len(a); w = a < b; bs = [rng.choice(w, D).mean() for _ in range(3000)]
    return f"{w.mean() * 100:.0f}% [{np.percentile(bs, 2.5) * 100:.0f}, {np.percentile(bs, 97.5) * 100:.0f}]"


def falsify(rows):
    D = len(rows); print(f"\n=== falsify_pkpd: {D} draws (draw indices {min(r['draw'] for r in rows)}..{max(r['draw'] for r in rows)}) ===")
    arr = lambda c, key, k: np.array([x[c][key][k] for x in rows])
    for k, o in ((1, "E"), (0, "logC")):
        print(f"--- {o} ---")
        for key, lab in (("rmse_ekf", "EKF plug-in mean"), ("rmse_map", "MAP"), ("rmse_post", "exact posterior mean")):
            m = {c: arr(c, key, k) ** 2 for c in ("EPIG", "EIG", "RANDOM")}
            print(f"  {lab:22s} mean-MSE ratio EPIG/EIG {ratio_ci(m['EPIG'], m['EIG'])}  EPIG/RAND {ratio_ci(m['EPIG'], m['RANDOM'])}  EIG/RAND {ratio_ci(m['EIG'], m['RANDOM'])}  | win EPIG<EIG {winrate(arr('EPIG', key, k), arr('EIG', key, k))}")
        v = {c: arr(c, "post_pred_sd", k) ** 2 for c in ("EPIG", "EIG", "RANDOM")}
        print(f"  exact predictive variance at pool: mean ratio EPIG/EIG {ratio_ci(v['EPIG'], v['EIG'])}  EPIG/RAND {ratio_ci(v['EPIG'], v['RANDOM'])}; EPIG lower than EIG in {winrate(v['EPIG'], v['EIG'])} of draws")
        print(f"  calibration: mean MSE (post-mean) / mean predictive variance, pooled over criteria: {np.mean(np.concatenate([arr(c, 'rmse_post', k) ** 2 for c in v])) / np.mean(np.concatenate(list(v.values()))):.2f}")
        print(f"  EKF predictive sd / exact, median: " + ", ".join(f"{c} {np.median(arr(c, 'ekf_pred_sd', k) / arr(c, 'post_pred_sd', k)):.2f}" for c in v))
        lr = np.log(arr("EPIG", "rmse_post", k) / arr("EIG", "rmse_post", k)); print(f"  per-draw log-ratio EPIG/EIG (post-mean): median {np.median(lr):+.3f}, mean {lr.mean():+.3f}, sd {lr.std():.2f}")
        mse = {c: arr(c, "rmse_post", k) ** 2 for c in v}; d = mse["EPIG"] - mse["EIG"]
        print("  trimming the k most extreme paired differences: " + ", ".join(f"k={kk}: {mse['EPIG'][np.argsort(-np.abs(d))[kk:]].mean() / mse['EIG'][np.argsort(-np.abs(d))[kk:]].mean():.2f}" for kk in (0, 5, 15, 30)))
    print(f"  ESS: median {np.median([x['EPIG']['ess'] for x in rows]):.0f}, min {min(x[c]['ess'] for x in rows for c in ('EPIG', 'EIG', 'RANDOM')):.0f}")


def laplace(rows):
    D = len(rows); arms = ("EPIG-EKF", "EIG-EKF", "EPIG-LAP", "EIG-LAP", "RANDOM")
    print(f"\n=== laplace_refit_pkpd: {D} draws ===")
    for k, o in ((1, "E"), (0, "logC")):
        mse = {a: np.array([x[a]["rmse_post"][k] ** 2 for x in rows]) for a in arms}
        var = {a: np.array([x[a]["post_pred_sd"][k] ** 2 for x in rows]) for a in arms}
        print(f"--- {o}: mean pool MSE (exact posterior mean) ratios ---")
        print(f"  P1  EPIG-LAP/EIG-LAP {ratio_ci(mse['EPIG-LAP'], mse['EIG-LAP'])}   (EKF beliefs: EPIG-EKF/EIG-EKF {ratio_ci(mse['EPIG-EKF'], mse['EIG-EKF'])})   predictive-variance ratios {var['EPIG-LAP'].mean() / var['EIG-LAP'].mean():.2f} / {var['EPIG-EKF'].mean() / var['EIG-EKF'].mean():.2f}")
        print(f"  P2  EPIG-LAP/EPIG-EKF {ratio_ci(mse['EPIG-LAP'], mse['EPIG-EKF'])}   EIG-LAP/EIG-EKF {ratio_ci(mse['EIG-LAP'], mse['EIG-EKF'])}   predictive-variance: {var['EPIG-LAP'].mean() / var['EPIG-EKF'].mean():.2f}, {var['EIG-LAP'].mean() / var['EIG-EKF'].mean():.2f}")
        print(f"      vs RANDOM: EPIG-LAP {ratio_ci(mse['EPIG-LAP'], mse['RANDOM'])}  EPIG-EKF {ratio_ci(mse['EPIG-EKF'], mse['RANDOM'])}  EIG-LAP {ratio_ci(mse['EIG-LAP'], mse['RANDOM'])}  EIG-EKF {ratio_ci(mse['EIG-EKF'], mse['RANDOM'])}")
        # P3: belief fidelity along the run (predictive sd at pool vs exact final; along the run vs the Laplace of that step is the fairest available)
        B = len(rows[0]["EPIG-EKF"]["ekf_pred_sd"])
        ekf_over_lap = np.array([[x["EPIG-EKF"]["ekf_pred_sd"][s][k] / x["EPIG-EKF"]["lap_pred_sd"][s][k] for s in range(B)] for x in rows])
        final_lap_over_exact = np.array([x[a]["lap_pred_sd"][-1][k] / x[a]["post_pred_sd"][k] for x in rows for a in arms]); final_ekf_over_exact = np.array([x[a]["ekf_pred_sd"][-1][k] / x[a]["post_pred_sd"][k] for x in rows for a in arms])
        print(f"  P3  final step: Laplace sd / exact median {np.median(final_lap_over_exact):.2f} [{np.percentile(final_lap_over_exact, 10):.2f}, {np.percentile(final_lap_over_exact, 90):.2f}]; EKF sd / exact median {np.median(final_ekf_over_exact):.2f}")
        print(f"      EKF sd / Laplace sd along the EPIG-EKF run, median by step: " + " ".join(f"{v:.2f}" for v in np.median(ekf_over_lap, 0)))
        # selection overlap between EKF- and Laplace-scored EPIG
        jac = [len(set(x["EPIG-EKF"]["selected"]) & set(x["EPIG-LAP"]["selected"])) / len(set(x["EPIG-EKF"]["selected"]) | set(x["EPIG-LAP"]["selected"])) for x in rows]
        if k == 1: print(f"  design overlap EPIG-EKF vs EPIG-LAP (Jaccard) median {np.median(jac):.2f}; EIG-EKF vs EIG-LAP {np.median([len(set(x['EIG-EKF']['selected']) & set(x['EIG-LAP']['selected'])) / len(set(x['EIG-EKF']['selected']) | set(x['EIG-LAP']['selected'])) for x in rows]):.2f}")


if __name__ == "__main__":
    for pat in sys.argv[1:]:
        rows = load(pat)
        if not rows: print(f"no rows for {pat}"); continue
        (laplace if "laplace" in pat else falsify)(rows)
