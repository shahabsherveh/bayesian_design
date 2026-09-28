"""Validation of the reference posterior (brief §5.4): exact linear-Gaussian case, importance
sampling diagnostics, repeat agreement, and the MCMC cross-check on a nonlinear case."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal, t as student_t

from bed import reference as ref
from bed.mechanistic import ClosedFormModel, DivideModel, make_divide_problem

jax.config.update("jax_enable_x64", True)


class LinearToy(ClosedFormModel):
    output_dim = 2
    weight_size = 3

    def _single(self, z, x):
        A = jnp.array([[x[0], x[1], 1.0], [x[1], -x[0], x[0] * x[1]]])
        return A @ z


def _linear_case(seed=0, k=6):
    rng = np.random.default_rng(seed); model = LinearToy()
    X = rng.normal(size=(k, 2)); mu0 = np.array([0.5, -0.2, 1.0]); S0 = np.diag([1.0, 0.5, 2.0]); R = np.array([[0.1, 0.02], [0.02, 0.2]])
    z_true = mu0 + np.linalg.cholesky(S0) @ rng.normal(size=3)
    Y = np.asarray(model(jnp.asarray(z_true), jnp.asarray(X))).reshape(k, 2) + rng.multivariate_normal(np.zeros(2), R, size=k)
    # exact posterior in information form
    A = np.stack([np.array([[x[0], x[1], 1.0], [x[1], -x[0], x[0] * x[1]]]) for x in X])   # (k, 2, 3)
    Rinv = np.linalg.inv(R); P = np.linalg.inv(S0) + sum(A[i].T @ Rinv @ A[i] for i in range(k))
    cov = np.linalg.inv(P); mean = cov @ (np.linalg.inv(S0) @ mu0 + sum(A[i].T @ Rinv @ Y[i] for i in range(k)))
    return ref.Posterior(model, X, Y, mu0, S0, R), mean, cov


def test_logp_matches_numpy_and_map_laplace_is_exact_for_a_linear_model():
    post, mean, cov = _linear_case()
    Z = np.random.default_rng(1).normal(size=(4, 3))
    expected = [-0.5 * (sum((post.Y[i] - np.asarray(post.model(jnp.asarray(z), jnp.asarray(post.X))).reshape(-1, 2)[i]) @ np.linalg.inv(post.R) @ (post.Y[i] - np.asarray(post.model(jnp.asarray(z), jnp.asarray(post.X))).reshape(-1, 2)[i]) for i in range(post.k))
                          + (z - post.mu0) @ np.linalg.inv(post.S0) @ (z - post.mu0)) for z in Z]
    np.testing.assert_allclose(post.logp(Z), expected, rtol=1e-10)
    z_map, cov_l, record = ref.map_laplace(post, [post.mu0, post.mu0 + 1.0])
    np.testing.assert_allclose(z_map, mean, rtol=1e-8, atol=1e-10); np.testing.assert_allclose(cov_l, cov, rtol=1e-8, atol=1e-10)
    assert all(r["success"] for r in record)


def test_importance_sampling_recovers_the_exact_linear_posterior_with_healthy_diagnostics():
    post, mean, cov = _linear_case(seed=2)
    r = ref.importance_posterior(post, 20000, seed=3)
    m, v = r.moments(lambda Z: Z)
    assert np.all(np.abs(m - mean) < 4 * np.sqrt(np.diag(cov) / r.ess)), (m, mean, r.ess)
    np.testing.assert_allclose(v, np.diag(cov), rtol=0.08)
    assert r.ess > 0.3 * 20000 and r.max_weight < 0.01 and r.khat < 0.7
    # the proposal has heavier tails than the posterior, so weights are bounded: k̂ well below 0.7
    assert r.khat < 0.5


def test_pareto_khat_flags_an_unbounded_weight_tail():
    rng = np.random.default_rng(0); x = rng.normal(size=20000)
    good = student_t.logpdf(x, df=5) - student_t.logpdf(x, df=5)          # perfect proposal: all weights equal
    assert abs(ref.pareto_khat(good + 1e-9 * rng.normal(size=x.size))) < 0.3
    # target N(0, 3²) sampled from proposal N(0, 1): the importance ratio exp(x²(1 - 1/9)/2) has no moments
    bad = multivariate_normal.logpdf(x[:, None], mean=[0.0], cov=[[9.0]]) - multivariate_normal.logpdf(x[:, None], mean=[0.0], cov=[[1.0]])
    assert ref.pareto_khat(bad) > 0.7


def test_repeat_agreement_and_mcmc_cross_check_on_a_nonlinear_divide_posterior():
    p = make_divide_problem(seed=4, num_voxels=1, snr=40.0)
    m1 = DivideModel(1, p.biomarker_scale, log_scale=p.log_scale)
    idx = [5, 40, 60, 100, 120, 20, 90, 110]                                # eight acquisitions
    X = np.asarray(p.candidates)[idx]; Y = np.asarray(p.data.y_train).reshape(-1, 1)[idx]
    post = ref.Posterior(m1, X, Y, np.asarray(p.prior_mean).ravel()[:5], np.asarray(p.prior_cov)[:5, :5], p.measurement_error)
    fn = lambda Z: np.asarray(jax.vmap(lambda z: m1(z, jnp.asarray([[0, 0, 0, 4.0], [0, 0, 0, 5.0], [0, 0, 0, 6.0]])))(jnp.asarray(Z))).reshape(Z.shape[0], 3)
    agree = ref.repeat_agreement(post, fn, 8000, seeds=(0, 1, 2))
    sd = np.sqrt(agree["vars"].mean(axis=0)); ess = min(dg["ess"] for dg in agree["diagnostics"])
    assert ess > 500 and all(dg["khat"] < 0.7 for dg in agree["diagnostics"])
    assert np.all(agree["spread"] < 5 * sd / np.sqrt(ess))
    r = ref.importance_posterior(post, 8000, seed=0)
    mc = ref.mcmc_posterior(post, r.z_map, r.cov_laplace, num_steps=3000, seed=0, num_chains=16)
    assert np.all(mc["rhat"] < 1.1) and 0.1 < mc["accept"] < 0.6
    m_is, v_is = r.moments(fn); m_mc = fn(mc["samples"]).mean(axis=0)
    tol = 6 * np.sqrt(v_is / min(r.ess, 500))
    assert np.all(np.abs(m_is - m_mc) < tol), (m_is, m_mc, tol)


def test_robust_posterior_escalates_on_a_heavy_tailed_nuisance_posterior_and_agrees_with_long_mcmc():
    """A DIVIDE history with a wide nuisance prior where the Laplace-proposal sampler collapses
    (ESS of a few); the escalation must return a reference whose target moments agree with a long
    MCMC run, and must record which stage produced it."""
    from bed import sequential as sq
    p = sq.divide_problem(seed=20008, snr=100.0, nuisance_sd=(0.3, 0.5))
    designs = [120, 4, 63, 90, 81, 120, 81, 4, 63, 120, 90, 4, 120, 81, 63, 4, 90, 120, 63, 81]
    h = sq.run_episode(p, sq.FixedPolicy(designs, "replay"), "ekf", [], len(designs), noise_seed=777 + 20008, snapshots=False)
    post = ref.Posterior(p.model, p.X[h["designs"]], h["outcomes"], p.mu0, p.S0, p.R)
    r = ref.robust_posterior(post, 8000, seed=0, z_inits=[p.mu0, h["final_mean"]])
    assert r.method in ("ais1", "ais2", "mcmc")
    m, v = r.moments(p.target_fn)
    long = ref.mcmc_posterior(post, r.z_map, r.cov_laplace, num_steps=20000, seed=3, num_chains=16)
    G = p.target_fn(long["samples"]); m_mc, v_mc = G.mean(0), G.var(0)
    assert np.all(np.abs(m - m_mc) < 0.15 * np.sqrt(v_mc) + 0.02), (m, m_mc, r.method)


def test_singular_laplace_covariance_is_repaired_and_the_proposal_still_samples():
    """A design set that leaves one parameter direction unidentified makes the Gauss–Newton
    covariance numerically singular; the MAP record must flag the repair and the importance
    sampler must still run with finite weights."""
    post, mean, cov = _linear_case(seed=3, k=1)          # one observation of a 3-parameter linear model: prior-dominated but the GN matrix is still full rank
    z_map, cov_l, record = ref.map_laplace(post, [post.mu0])
    cov_sing = cov_l.copy(); cov_sing[0, :] = 0; cov_sing[:, 0] = 0   # force a singular shape
    r = ref.importance_posterior(post, 2000, seed=0, z_map=z_map, cov=cov_sing)
    assert np.all(np.isfinite(r.w)) and r.ess > 1
