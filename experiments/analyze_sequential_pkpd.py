"""Aggregate a `sequential_pkpd.py` results file against the registered predictions (PROGRESS.md
A-2): paired win counts and median ratios at the final step and along the budget, per filter and
output; a learning-curve figure with interquartile bands. Numbers are printed and written to
`<results>_summary.json`; the figure goes next to it as a PDF.

Usage: python experiments/analyze_sequential_pkpd.py [results/sequential_pkpd_100.json]
"""
import json, sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sequential_pkpd_100.json")
r = json.load(open(path)); rows = r["rows"]; D = r["draws"]; B = r["budget"]
CRITS = ("EPIG", "EIG", "RANDOM"); OUTS = ("logC", "E")
curves = {(f, c, o): np.array([[x[f"pool_curve_{o}"] for x in rows if x["draw"] == d and x["filter"] == f and x["criterion"] == c][0] for d in range(D)])
          for f in ("ekf", "ukf") for c in CRITS for o in OUTS}
glob = {(f, c, o): np.array([[x[f"glob_rmse_{o}"] for x in rows if x["draw"] == d and x["filter"] == f and x["criterion"] == c][0] for d in range(D)])
        for f in ("ekf", "ukf") for c in CRITS for o in OUTS}
summary = {"draws": D, "budget": B}


def wins(a, b):
    """fraction of draws with a < b, and the median of a/b (positive means a is better)"""
    return float(np.mean(a < b)), float(np.median(a / b))


print(f"{path}: {D} draws, budget {B}\n")
for f in ("ekf", "ukf"):
    print(f"=== {f.upper()} ===")
    for o in OUTS:
        fe, fi, fr = (curves[(f, c, o)][:, -1] for c in CRITS)
        ae, ai, ar = (curves[(f, c, o)].mean(axis=1) for c in CRITS)
        ge, gi, gr = (glob[(f, c, o)] for c in CRITS)
        rec = {"final_pool": {"EPIG<EIG": wins(fe, fi), "EPIG<RANDOM": wins(fe, fr), "EIG<RANDOM": wins(fi, fr)},
               "auc_pool": {"EPIG<EIG": wins(ae, ai), "EPIG<RANDOM": wins(ae, ar), "EIG<RANDOM": wins(ai, ar)},
               "final_global": {"EIG<EPIG": wins(gi, ge), "EPIG<RANDOM": wins(ge, gr), "EIG<RANDOM": wins(gi, gr)},
               "median_final_pool": {c: float(np.median(curves[(f, c, o)][:, -1])) for c in CRITS},
               "median_final_global": {c: float(np.median(glob[(f, c, o)])) for c in CRITS}}
        summary[f"{f}_{o}"] = rec
        print(f" {o:4s} pool, final step : " + "   ".join(f"{k} {v[0]*100:3.0f}% (ratio {v[1]:.2f})" for k, v in rec["final_pool"].items()))
        print(f" {o:4s} pool, mean/steps : " + "   ".join(f"{k} {v[0]*100:3.0f}% (ratio {v[1]:.2f})" for k, v in rec["auc_pool"].items()))
        print(f" {o:4s} global, final    : " + "   ".join(f"{k} {v[0]*100:3.0f}% (ratio {v[1]:.2f})" for k, v in rec["final_global"].items()))
        print(f" {o:4s} medians at the final step: pool " + ", ".join(f"{c} {v:.3g}" for c, v in rec["median_final_pool"].items()) + " | global " + ", ".join(f"{c} {v:.3g}" for c, v in rec["median_final_global"].items()))
    # both outputs at once (registered prediction 1)
    both = np.mean([(curves[(f, "EPIG", o)][:, -1] < curves[(f, "EIG", o)][:, -1]) for o in OUTS], axis=0) == 1
    summary[f"{f}_both_outputs_EPIG<EIG_final"] = float(both.mean())
    print(f" both outputs EPIG < EIG at the final step: {both.mean()*100:.0f}% of draws")
    # step-wise win fraction against random, E
    for o in OUTS:
        w = [float(np.mean(curves[(f, "EPIG", o)][:, k] < curves[(f, "RANDOM", o)][:, k])) for k in range(B)]
        wi = [float(np.mean(curves[(f, "EPIG", o)][:, k] < curves[(f, "EIG", o)][:, k])) for k in range(B)]
        summary[f"{f}_{o}_stepwise_EPIG<RANDOM"] = w; summary[f"{f}_{o}_stepwise_EPIG<EIG"] = wi
        print(f" {o:4s} EPIG<RANDOM by step: " + " ".join(f"{v*100:3.0f}" for v in w) + "   EPIG<EIG by step: " + " ".join(f"{v*100:3.0f}" for v in wi))
    print()
# UKF vs EKF, same criterion
for c in ("EPIG", "EIG"):
    for o in OUTS:
        u, e = curves[("ukf", c, o)][:, -1], curves[("ekf", c, o)][:, -1]
        summary[f"ukf<ekf_{c}_{o}"] = wins(u, e)
        print(f" UKF < EKF under {c:4s}, {o:4s}: {np.mean(u < e)*100:3.0f}% (median ratio UKF/EKF {np.median(u / e):.2f})")
# selected sampling times (EKF), excluding the warm start
sel = {c: np.concatenate([x["selected_times"][3:] for x in rows if x["filter"] == "ekf" and x["criterion"] == c]) for c in ("EPIG", "EIG")}
dose = {c: np.concatenate([x["selected_doses"][3:] for x in rows if x["filter"] == "ekf" and x["criterion"] == c]) for c in ("EPIG", "EIG")}
bins = [0, 1, 2, 4, 8, 12, 16, 20, 24.01]
for c in ("EPIG", "EIG"):
    h, _ = np.histogram(sel[c], bins=bins); summary[f"ekf_{c}_time_hist"] = h.tolist(); summary[f"ekf_{c}_dose400_share"] = float(np.mean(dose[c] == 400))
    print(f" {c} selected times, bins {bins}: {(h / h.sum() * 100).round(0).astype(int).tolist()} %   share at 400 mg {np.mean(dose[c] == 400)*100:.0f}%")
json.dump(summary, open(path.with_name(path.stem + "_summary.json"), "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"EPIG": "#1b9e77", "EIG": "#d95f02", "RANDOM": "#7570b3"}
fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True)
for i, f in enumerate(("ekf", "ukf")):
    for j, o in enumerate(OUTS):
        ax = axes[j, i]
        for c in CRITS:
            cur = curves[(f, c, o)]; med = np.median(cur, axis=0); lo, hi = np.percentile(cur, [25, 75], axis=0); k = np.arange(1, B + 1)
            ax.plot(k, med, color=COL[c], label=c); ax.fill_between(k, lo, hi, color=COL[c], alpha=0.15, lw=0)
        ax.set_yscale("log"); ax.set_title(f"{f.upper()}, {'log C' if o == 'logC' else 'E'}", fontsize=9)
        if j == 1: ax.set_xlabel("design step")
        if i == 0: ax.set_ylabel("pool RMSE (median, IQR)")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(path.with_name(path.stem + "_curves.pdf")); print(f"\nwrote {path.with_name(path.stem + '_summary.json')} and {path.with_name(path.stem + '_curves.pdf')}")
