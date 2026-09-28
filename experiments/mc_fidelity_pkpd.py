"""Does the Gaussian assumption behind the closed-form EPIG hold on the PK/PD prior? Compares the
EKF closed-form EPIG at the prior with the nested Monte Carlo EPIG estimate (no Gaussian
assumption) on a subset of candidates, averaged over repeats to beat the small outer sample.
The A pre-flight only compared the closed form with a *Gaussian* EPIG built from MC moments,
which cannot detect a non-Gaussian joint.

Usage: python experiments/mc_fidelity_pkpd.py [--stride 12] [--inner 32000] [--repeats 8]
"""
import argparse, json, time
import jax, jax.numpy as jnp, numpy as np
from scipy.stats import spearmanr, pearsonr
jax.config.update("jax_enable_x64", True)
from bed.ekf import EKF
from bed.experiments import Experiment
from bed.mechanistic import make_pkpd_problem

ap = argparse.ArgumentParser(); ap.add_argument("--stride", type=int, default=12); ap.add_argument("--inner", type=int, default=32000); ap.add_argument("--repeats", type=int, default=8)
a = ap.parse_args(); t0 = time.time()
out = {}
for prior_scale in (1.0, 0.25):
    p = make_pkpd_problem(seed=0, prior_scale=prior_scale)
    sub = p.data.x_train[::a.stride]; pool = p.data.x_test_pool
    filt = EKF(p.model, p.prior_mean, p.prior_cov, 0.0, p.measurement_error)
    closed = np.asarray(filt.calculate_epig(sub, pool)); eig = np.asarray(filt.calculate_eig(sub))
    mcs = []
    for rep in range(a.repeats):
        exp = Experiment(model=p.model, data=p.data, latent_cov=p.prior_cov, latent_mean=p.prior_mean, measurement_error=p.measurement_error, warm_start=0, seed=rep)
        mcs.append(np.asarray(exp.calculate_epig_mc(sub, filt, pool, num_latent_samples=a.inner)))
    mcs = np.array(mcs); mc = mcs.mean(0); se = mcs.std(0) / np.sqrt(a.repeats)
    rho, r = spearmanr(closed, mc).correlation, pearsonr(closed, mc)[0]
    rep_rho = np.mean([spearmanr(mcs[i], mcs[j]).correlation for i in range(a.repeats) for j in range(i)])
    print(f"prior_scale {prior_scale}: {len(sub)} candidates, inner K={a.inner}, {a.repeats} repeats ({time.time()-t0:.0f}s)")
    print(f"  Spearman(closed, MC) = {rho:.3f}   Pearson = {r:.3f}   mean repeat-to-repeat Spearman of MC = {rep_rho:.3f}   Spearman(EIG, MC) = {spearmanr(eig, mc).correlation:.3f}")
    print(f"  closed range {closed.min():.3f}..{closed.max():.3f}; MC range {mc.min():.3f}..{mc.max():.3f} (median se {np.median(se):.3f}); argmax agree: {bool(closed.argmax() == mc.argmax())}; closed argmax design {np.round(np.asarray(sub[closed.argmax()]).ravel(),2)}, MC argmax {np.round(np.asarray(sub[mc.argmax()]).ravel(),2)}")
    # where do they disagree? by time bin
    times = np.asarray(sub).reshape(-1, 4)[:, 3]
    for lo, hi in ((0, 2), (2, 8), (8, 16), (16, 24.1)):
        m = (times >= lo) & (times < hi)
        if m.sum() > 2: print(f"    t in [{lo},{hi}): n={m.sum()}, mean closed {closed[m].mean():.3f}, mean MC {mc[m].mean():.3f}, ratio {closed[m].mean()/mc[m].mean():.2f}")
    out[str(prior_scale)] = {"spearman": float(rho), "pearson": float(r), "repeat_spearman": float(rep_rho), "closed": closed.tolist(), "mc": mc.tolist(), "mc_se": se.tolist(), "designs": np.asarray(sub).reshape(-1, 4).tolist()}
json.dump(out, open("results/mc_fidelity_pkpd.json", "w"), indent=1); print("wrote results/mc_fidelity_pkpd.json")
