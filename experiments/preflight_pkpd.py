"""Pre-flight gate for candidate A (PROGRESS.md, Phase 2): at the prior,
(1) Spearman(EPIG, EIG) over the candidates -- must be < 0.9 for EPIG to offer anything;
(2) Spearman(closed-form EPIG, Gaussian EPIG from MC moments of the same belief) -- must be > 0.9.
Also reports the singular-value spectra of the target and design Jacobians and where EPIG's argmax lies."""
import json
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from scipy.stats import spearmanr
jax.config.update("jax_enable_x64", True)
from bed.diagnostics import filter_moments, gaussian_epig, mc_moments
from bed.ekf import EKF
from bed.mechanistic import make_pkpd_problem, PKPD_NAMES
from bed.ukf import UKF

out = {}
for prior_scale in (1.0, 0.25):
    p = make_pkpd_problem(seed=0, prior_scale=prior_scale)
    X, pool = p.data.x_train, p.data.x_test_pool
    print(f"\nprior_scale = {prior_scale}: {X.shape[0]} candidates, {pool.shape[0]} pool points")
    ref = mc_moments(p.model, p.prior_mean, p.prior_cov, X, pool, measurement_error=p.measurement_error, num_samples=20000, key=jax.random.PRNGKey(3))
    ref_b = mc_moments(p.model, p.prior_mean, p.prior_cov, X, pool, measurement_error=p.measurement_error, num_samples=20000, key=jax.random.PRNGKey(4))
    epig_ref = np.asarray(gaussian_epig(ref.cov, ref.cov_prime, ref.cross)); epig_ref_b = np.asarray(gaussian_epig(ref_b.cov, ref_b.cov_prime, ref_b.cross))
    print(f"  MC-moment EPIG ceiling (two references): Spearman {spearmanr(epig_ref, epig_ref_b).correlation:.3f}; range {epig_ref.min():.3f}..{epig_ref.max():.3f}")
    res = {}
    for name, filt in (("EKF", EKF(p.model, p.prior_mean, p.prior_cov, 0.0, p.measurement_error)),
                       ("UKF a=1", UKF(p.model, p.prior_mean, p.prior_cov, 0.0, p.measurement_error, alpha=1.0))):
        epig = np.asarray(filt.calculate_epig(X, pool)); eig = np.asarray(filt.calculate_eig(X))
        rho_eig = spearmanr(epig, eig).correlation; rho_ref = spearmanr(epig, epig_ref).correlation
        i_epig, i_eig = int(epig.argmax()), int(eig.argmax())
        m = filter_moments(filt, X, pool)
        var_err = float(jnp.median(jnp.abs(jnp.diagonal(m.cov, axis1=-2, axis2=-1) - jnp.diagonal(ref.cov, axis1=-2, axis2=-1)) / jnp.diagonal(ref.cov, axis1=-2, axis2=-1)))
        res[name] = {"spearman_epig_eig": float(rho_eig), "spearman_closed_vs_mc_moments": float(rho_ref), "argmax_agrees_with_mc_moments": bool(i_epig == int(epig_ref.argmax())),
                     "epig_argmax_design": [float(v) for v in X[i_epig].ravel()], "eig_argmax_design": [float(v) for v in X[i_eig].ravel()], "median_rel_var_err": var_err}
        print(f"  {name:8s} gate 1 Spearman(EPIG, EIG) = {rho_eig:.3f}   gate 2 Spearman(closed, MC-moment EPIG) = {rho_ref:.3f}   median rel. variance error {var_err:.3f}")
        print(f"           EPIG argmax (D, tau, n, t) = {np.round(np.asarray(X[i_epig]).ravel(), 2)}   EIG argmax = {np.round(np.asarray(X[i_eig]).ravel(), 2)}   EPIG range {epig.min():.3f}..{epig.max():.3f}")
    # Jacobian spectra at the prior mean (EKF)
    ekf = EKF(p.model, p.prior_mean, p.prior_cov, 0.0, p.measurement_error)
    Ls = jnp.sqrt(jnp.diag(p.prior_cov))
    Jd = (p.model.jacobian(p.prior_mean, X) * Ls).reshape(-1, 7); Jt = (p.model.jacobian(p.prior_mean, pool) * Ls).reshape(-1, 7)
    sd_, st_ = jnp.linalg.svd(Jd, compute_uv=False), jnp.linalg.svd(Jt, compute_uv=False)
    print(f"  prior-scaled Jacobian singular values (rel.): design {np.round(np.asarray(sd_/sd_[0]), 3)}  target {np.round(np.asarray(st_/st_[0]), 3)}")
    sens_t = jnp.linalg.norm(Jt, axis=0) / jnp.linalg.norm(Jt); sens_d = jnp.linalg.norm(Jd, axis=0) / jnp.linalg.norm(Jd)
    print("  per-parameter prior-scaled sensitivity share, target vs design: " + ", ".join(f"{n} {float(t):.2f}/{float(d):.2f}" for n, t, d in zip(PKPD_NAMES, sens_t, sens_d)))
    out[str(prior_scale)] = res
Path("results").mkdir(exist_ok=True); json.dump(out, open("results/preflight_pkpd.json", "w"), indent=1); print("\nwrote results/preflight_pkpd.json")
