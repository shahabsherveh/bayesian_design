"""Aggregate `sequential_divide.py` output against the registered predictions S1-S5 (PROGRESS.md D-2):
expected-loss ratios with paired bootstrap CIs (summed and per biomarker, three estimators), fixed vs
adaptive, calibration, estimator agreement, allocations.

Usage: python experiments/analyze_divide.py results/sequential_divide_V1.json [results/sequential_divide_V16.json]
"""
import json, sys
import numpy as np
rng = np.random.default_rng(0)
BIO = ("MD", "MK_I", "MK_A")


def ratio_ci(a, b, n=4000):
    D = len(a); idx = [rng.integers(0, D, D) for _ in range(n)]; bs = [a[i].mean() / b[i].mean() for i in idx]
    return f"{a.mean() / b.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"


for f in sys.argv[1:]:
    r = json.load(open(f)); rows = r["rows"]; D = len(rows); ARMS = tuple(r["arms"])
    print(f"\n=== {f}: V = {r['voxels']}, SNR {r.get('snr', 40)}, {D} draws, budget {r['budget']} + warm {r['warm']} ===")
    for key, lab in (("sq_err_post", "posterior mean (IS), linear units"), ("log_sq_err_post", "posterior mean (IS), LOG units"), ("sq_err_map", "MAP"), ("sq_err_ekf", "EKF plug-in")):
        if key not in rows[0]["EPIG"]: continue
        L = {a: np.array([x[a][key] for x in rows]) for a in ARMS}            # (D, 3) standardised squared errors, mean over voxels
        T = {a: L[a].sum(axis=1) for a in ARMS}
        print(f" {lab}: summed loss ratios  EPIG/EIG {ratio_ci(T['EPIG'], T['EIG'])}  EPIG/RAND {ratio_ci(T['EPIG'], T['RANDOM'])}  EIG/RAND {ratio_ci(T['EIG'], T['RANDOM'])}  | FIXED-EPIG/EPIG {ratio_ci(T['FIXED-EPIG'], T['EPIG'])}  FIXED-EIG/EIG {ratio_ci(T['FIXED-EIG'], T['EIG'])}  FIXED-EPIG/FIXED-EIG {ratio_ci(T['FIXED-EPIG'], T['FIXED-EIG'])}")
        if key in ("sq_err_post", "log_sq_err_post"):
            for k, b in enumerate(BIO):
                print(f"   {b:5s}: EPIG/EIG {ratio_ci(L['EPIG'][:, k], L['EIG'][:, k])}  EPIG/RAND {ratio_ci(L['EPIG'][:, k], L['RANDOM'][:, k])}  FIXED-EPIG/EPIG {ratio_ci(L['FIXED-EPIG'][:, k], L['EPIG'][:, k])}  | mean loss EPIG {L['EPIG'][:, k].mean():.4f} EIG {L['EIG'][:, k].mean():.4f} RAND {L['RANDOM'][:, k].mean():.4f}")
    # calibration: realised MSE / sampled posterior variance, per arm (pooled over biomarkers)
    print(" calibration, linear units (mean sq. error / mean sampled posterior variance): " + "  ".join(f"{a} {np.array([x[a]['sq_err_post'] for x in rows]).mean() / np.array([x[a]['post_var'] for x in rows]).mean():.2f}" for a in ARMS))
    if "log_post_var" in rows[0]["EPIG"]: print(" calibration, log units: " + "  ".join(f"{a} {np.array([x[a]['log_sq_err_post'] for x in rows]).mean() / np.array([x[a]['log_post_var'] for x in rows]).mean():.2f}" for a in ARMS))
    print(" EKF linearised variance / sampled posterior variance (per biomarker, EPIG arm): " + " ".join(f"{np.array([x['EPIG']['ekf_var'] for x in rows]).mean(0)[k] / np.array([x['EPIG']['post_var'] for x in rows]).mean(0)[k]:.2f}" for k in range(3)) + f" | ESS min {min(x[a]['ess_min'] for x in rows for a in ARMS):.0f}")
    # forecast comparison: sampled posterior variance ratios (the quantity the forecast predicts)
    V = {a: np.array([x[a]["post_var"] for x in rows]) for a in ARMS}
    print(" posterior-variance ratios per biomarker (forecast said EPIG/EIG 0.76, 0.96, 0.73): EPIG/EIG " + " ".join(f"{V['EPIG'][:, k].mean() / V['EIG'][:, k].mean():.2f}" for k in range(3)) + " ; EPIG/RAND " + " ".join(f"{V['EPIG'][:, k].mean() / V['RANDOM'][:, k].mean():.2f}" for k in range(3)))
    # allocation
    from bed.mechanistic import make_divide_problem, divide_te_min
    import jax.numpy as jnp
    p = make_divide_problem(seed=0, num_voxels=1); cand = np.asarray(p.candidates)
    for a in [a for a in ("EPIG", "EIG", "FIXED-EPIG", "FIXED-EIG") if a in ARMS]:
        sel = np.concatenate([x[a]["selected"] for x in rows]); c = cand[sel]
        te_extra = c[:, 2] - np.asarray(divide_te_min(jnp.asarray(c[:, 0]), jnp.asarray(c[:, 1])))
        print(f" {a:10s}: b=0 {np.mean(c[:, 0] < 0.05)*100:3.0f}%, TE above min {np.mean(te_extra > 1)*100:3.0f}%, spherical {np.mean(c[:, 1] == 0)*100:3.0f}%, b>=1.5 {np.mean(c[:, 0] >= 1.5)*100:3.0f}%, median b {np.median(c[:, 0]):.1f}, distinct/draw {np.mean([len(set(x[a]['selected'])) for x in rows]):.1f}")
    # learning curves (EKF plug-in), summed loss, mean over draws at steps 5, 10, 20, 27
    for a in ARMS:
        cur = np.array([np.sum(x[a]["curve_ekf"], axis=1) for x in rows]); print(f" curve {a:10s}: " + " ".join(f"{cur[:, k].mean():.3f}" for k in (0, 5, 10, 20, r['budget'])))
