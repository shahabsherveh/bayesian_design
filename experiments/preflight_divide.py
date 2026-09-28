"""Pre-flight for candidate D (PROGRESS.md, gates G1-G4): the data-free linearised greedy forecast
of the biomarker-variance ratios EPIG/EIG and EPIG/random at N = 30 (G1), the closed-form vs
MC-moment check at the prior (G2), the prior-predictive monotonicity fraction (G3) and where each
criterion allocates acquisitions (G4). Targets are the three standardised biomarkers with zero noise.

Usage: python experiments/preflight_divide.py [--voxels 1] [--budget 27] [--warm 3] [--seeds 20]
"""
import argparse, json, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from scipy.stats import spearmanr
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from linearized_greedy import epig_scores, eig_scores, greedy, jacobians
from bed.mechanistic import DIVIDE_BIOMARKERS, DIVIDE_PRIOR_MEAN, DIVIDE_PRIOR_SD, divide_biomarkers, make_divide_problem
from bed.diagnostics import gaussian_epig, mc_moments
from bed.ekf import EKF

ap = argparse.ArgumentParser(); ap.add_argument("--voxels", type=int, default=1); ap.add_argument("--budget", type=int, default=27); ap.add_argument("--warm", type=int, default=3); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--snr", type=float, default=40.0)
a = ap.parse_args(); out = {}
p = make_divide_problem(seed=0, num_voxels=a.voxels, snr=a.snr); J, Jt, S0, R = jacobians(p); n = J.shape[0]
Rt = 1e-12 * np.eye(R.shape[0])
cand = np.asarray(p.candidates)
print(f"D pre-flight: V={a.voxels}, {n} candidates, targets = standardised {DIVIDE_BIOMARKERS}, budget {a.budget} after warm start {a.warm}, repeats allowed")
# G3: prior predictive monotonicity
rng = np.random.default_rng(1); Z = np.asarray(DIVIDE_PRIOR_MEAN) + np.asarray(DIVIDE_PRIOR_SD) * rng.normal(size=(20000, 5))
bmax = cand[:, 0].max(); m1 = type(p.model)(1, p.biomarker_scale, form=p.model.form)
Xb = jnp.asarray(np.column_stack([np.linspace(0, bmax, 21), np.ones(21), np.full(21, 80.0), np.zeros(21)]))
Sb = np.asarray(jax.vmap(lambda z: m1(z, Xb).reshape(-1))(jnp.asarray(Z[:2000])))
viol = float(np.mean(np.any(np.diff(Sb, axis=1) > 0, axis=1))); out["G3_nonmonotone_fraction"] = viol
G = np.asarray(jax.vmap(divide_biomarkers)(jnp.asarray(Z))); print(f"  G3 prior predictive ({p.model.form} form): fraction of draws with a non-monotone signal over b in [0, {bmax}] (linear): {viol*100:.2f}%  | biomarker prior: MD {G[:,0].mean():.2f}±{G[:,0].std():.2f}, MK_I {G[:,1].mean():.2f}±{G[:,1].std():.2f}, MK_A {G[:,2].mean():.2f}±{G[:,2].std():.2f}")
# G2: closed form vs Gaussian EPIG from MC moments (targets are functionals: compare on the signal-side EPIG using a pool of candidates, and on the biomarker targets via the linear-Gaussian moments)
filt = EKF(p.model, p.prior_mean, p.prior_cov, 0.0, p.measurement_error)
X, T = p.data.x_train, p.data.x_test_pool
ref = mc_moments(p.model, p.prior_mean, p.prior_cov, X, T, measurement_error=p.measurement_error, num_samples=20000, key=jax.random.PRNGKey(3))
epig_mc = np.asarray(gaussian_epig(ref.cov, ref.cov_prime, ref.cross)); epig_cf = np.asarray(filt.calculate_epig(X, T)); eig_cf = np.asarray(filt.calculate_eig(X))
out["G2_spearman_closed_vs_mc"] = float(spearmanr(epig_cf, epig_mc).correlation); out["spearman_epig_eig_prior"] = float(spearmanr(epig_cf, eig_cf).correlation)
print(f"  G2 closed form vs MC-moment EPIG (biomarker targets, runner's shared R): Spearman {out['G2_spearman_closed_vs_mc']:.3f}; Spearman(EPIG, EIG) at the prior {out['spearman_epig_eig_prior']:.3f} (for the record only)")
# G1: forecast, repeats allowed
curves = {}; sels = {}
for crit in ("epig", "eig", "random"):
    cs, ss = [], []
    for s in range(a.seeds):
        r0 = np.random.default_rng(s); warm = list(r0.choice(n, a.warm, replace=False))
        c, u = greedy(J, Jt, S0, R, a.budget, warm, crit, np.random.default_rng(100 + s), Rt=Rt, replace=True, per_target=True); cs.append(c); ss.append(u)
    curves[crit] = np.array(cs); sels[crit] = np.concatenate(ss)
out["G1"] = {}
for k in (5, 10, 20, a.budget):
    e, i, r = (curves[c][:, k].mean(axis=0) for c in ("epig", "eig", "random"))
    out["G1"][k] = {"epig_over_eig": (e / i).round(3).tolist(), "epig_over_random": (e / r).round(3).tolist(), "mean_epig_over_eig": float(np.mean(e / i))}
    print(f"  G1 forecast, step {k:2d}: biomarker variance ratio EPIG/EIG {np.round(e / i, 3)} (mean {np.mean(e / i):.3f})   EPIG/RAND {np.round(e / r, 3)}   EIG/RAND {np.round(i / r, 3)}")
# G4: allocation
def alloc(sel):
    c = cand[sel]; te_extra = c[:, 2] - np.asarray([float(v) for v in __import__("bed.mechanistic", fromlist=["divide_te_min"]).divide_te_min(jnp.asarray(c[:, 0]), jnp.asarray(c[:, 1]))])
    return {"share_TE_above_min": float(np.mean(te_extra > 1)), "share_b0": float(np.mean(c[:, 0] < 0.05)), "share_spherical_b>=1.5": float(np.mean((c[:, 1] == 0) & (c[:, 0] >= 1.5))), "share_linear_b>=1.5": float(np.mean((c[:, 1] == 1) & (c[:, 0] >= 1.5))), "median_b": float(np.median(c[:, 0])), "distinct": int(len(set(sel.tolist())))}
out["G4"] = {c: alloc(sels[c]) for c in ("epig", "eig")}
for c in ("epig", "eig"): print(f"  G4 {c.upper()} allocation: {out['G4'][c]}")
Path("results").mkdir(exist_ok=True); json.dump(out, open(f"results/preflight_divide_V{a.voxels}_snr{int(a.snr)}.json", "w"), indent=1); print(f"wrote results/preflight_divide_V{a.voxels}_snr{int(a.snr)}.json")
