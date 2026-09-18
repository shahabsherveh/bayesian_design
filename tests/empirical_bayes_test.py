"""Tests for bed.empirical_bayes: exactness on the linear model, hyperparameter recovery, agreement of
the Laplace evidence with the true marginal likelihood on a nonlinear model, the Flax network path,
and the runner integration."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy.stats import multivariate_normal, multivariate_t

from bed.data import create_synthetic_data
from bed.ekf import EKF
from bed.empirical_bayes import empirical_bayes_init, log_marginal_likelihood, map_estimate
from bed.experiments import Experiment
from bed.models import LinearModel, LinearNN, Model, NeuralNetworkRegressor
from bed.ukf import UKF

jax.config.update("jax_enable_x64", True)


def _linear_problem(n=12, d=3, noise_sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(n, 1, d))); z = jnp.asarray(rng.normal(size=(d, 1)))
    m = LinearModel(); y = m(z, X).reshape(-1) + noise_sd * jnp.asarray(rng.normal(size=n))
    return m, X, y, z


class Tanh(Model):
    """y = z_0 tanh(z_1 x) + z_2, a two-parameter-nonlinear scalar model for the Laplace checks."""
    def _single(self, z, x):
        return z[0] * jnp.tanh(z[1] * x[0]) + z[2]

    def __call__(self, z, x, **kwargs):
        z = jnp.asarray(z).reshape(-1); x = jnp.asarray(x).reshape(-1, 1)
        return jax.vmap(lambda xi: self._single(z, xi))(x).reshape(-1, 1, 1, 1)

    def jacobian(self, z, x):
        z = jnp.asarray(z).reshape(-1); x = jnp.asarray(x).reshape(-1, 1)
        return jax.vmap(lambda xi: jax.jacobian(lambda zz: self._single(zz, xi))(z))(x).reshape(-1, 1, 3)


# --- linear model: exactness -------------------------------------------------------------------

def test_linear_evidence_equals_the_direct_gaussian_density_and_does_not_depend_on_the_linearisation_point():
    m, X, y, _ = _linear_problem(); n, d = 12, 3
    S0 = 0.7 * jnp.eye(d) + 0.1; R = jnp.array([[0.01]]); mu0 = 0.3 * jnp.ones((d, 1))
    Xf = np.asarray(X).reshape(n, d)
    direct = multivariate_normal(mean=Xf @ np.asarray(mu0).ravel(), cov=Xf @ np.asarray(S0) @ Xf.T + 0.01 * np.eye(n)).logpdf(np.asarray(y))
    for z_lin in (jnp.zeros(d), jnp.ones(d), -2.0 * jnp.ones(d)):
        np.testing.assert_allclose(float(log_marginal_likelihood(m, z_lin, X, y, mu0, S0, R)), direct, rtol=1e-10)


def test_linear_map_and_laplace_equal_the_exact_bayesian_regression_posterior():
    m, X, y, _ = _linear_problem(); n, d = 12, 3
    S0 = 0.5 * jnp.eye(d); R = jnp.array([[0.04]]); mu0 = jnp.zeros((d, 1))
    z_map, cov = map_estimate(m, X, y, mu0, S0, R)
    Xf = np.asarray(X).reshape(n, d); A = np.linalg.inv(np.asarray(S0)) + Xf.T @ Xf / 0.04
    exact_cov = np.linalg.inv(A); exact_mean = exact_cov @ (Xf.T @ np.asarray(y) / 0.04)
    np.testing.assert_allclose(z_map, exact_mean, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(cov, exact_cov, rtol=1e-7, atol=1e-10)


def test_linear_type_ii_recovers_the_generating_scales_at_large_n():
    rng = np.random.default_rng(3); n, d, tau_true, s_true = 600, 4, 2.5, 0.09
    X = jnp.asarray(rng.normal(size=(n, 1, d))); z = jnp.asarray(np.sqrt(tau_true) * rng.normal(size=(d, 1)) )
    m = LinearModel(); y = m(z, X).reshape(-1) + np.sqrt(s_true) * jnp.asarray(rng.normal(size=n))
    # data from several draws of z are needed to pin tau; concatenate five independent problems
    Xs, ys = [X], [y]
    for k in range(4):
        zk = jnp.asarray(np.sqrt(tau_true) * rng.normal(size=(d, 1))); Xk = jnp.asarray(rng.normal(size=(n, 1, d)))
        Xs.append(Xk); ys.append(m(zk, Xk).reshape(-1) + np.sqrt(s_true) * jnp.asarray(rng.normal(size=n)))
    # a single shared z is what the model assumes, so test on the single problem: tau is then only weakly
    # identified (d = 4 draws) but s is well identified
    eb = empirical_bayes_init(m, X, y, jnp.zeros((d, 1)), jnp.eye(d), jnp.array([[1.0]]), fit_noise=True, iterations=1)
    assert abs(eb.noise_scale - s_true) / s_true < 0.15, eb.noise_scale
    assert 0.1 * tau_true < eb.prior_scale < 10 * tau_true, eb.prior_scale
    # the fitted prior scale equals the ML-II value for a single z draw: ||z||^2 / d up to shrinkage
    assert eb.log_marginal_likelihood > float(log_marginal_likelihood(m, jnp.zeros(d), X, y, jnp.zeros((d, 1)), 0.01 * jnp.eye(d), 0.09 * jnp.eye(1)))
    # the returned covariances carry the fitted scales, and the returned belief is the exact posterior under them
    np.testing.assert_allclose(np.asarray(eb.prior_cov), eb.prior_scale * np.eye(d)); np.testing.assert_allclose(np.asarray(eb.noise_cov), [[eb.noise_scale]])
    Xf = np.asarray(X).reshape(n, d); A = np.eye(d) / eb.prior_scale + Xf.T @ Xf / eb.noise_scale
    np.testing.assert_allclose(np.asarray(eb.cov), np.linalg.inv(A), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(np.asarray(eb.mean).ravel(), np.linalg.solve(A, Xf.T @ np.asarray(y) / eb.noise_scale), rtol=1e-6, atol=1e-9)


def test_evidence_gradient_matches_finite_differences():
    m, X, y, _ = _linear_problem(seed=5); d = 3; mu0 = jnp.zeros((d, 1))
    f = lambda th: log_marginal_likelihood(m, jnp.zeros(d), X, y, mu0, jnp.exp(th[0]) * jnp.eye(d), jnp.exp(th[1]) * jnp.eye(1))
    th = jnp.array([0.3, -3.0]); g = jax.grad(f)(th)
    for k in range(2):
        e = jnp.zeros(2).at[k].set(1e-6); fd = (f(th + e) - f(th - e)) / 2e-6
        np.testing.assert_allclose(float(g[k]), float(fd), rtol=1e-5, atol=1e-6)


def test_evidence_has_an_interior_maximum_in_the_noise_scale_on_well_specified_data():
    m, X, y, _ = _linear_problem(n=40, noise_sd=0.1, seed=7); d = 3; mu0 = jnp.zeros((d, 1))
    grid = np.logspace(-4, 1, 60)
    vals = [float(log_marginal_likelihood(m, jnp.zeros(d), X, y, mu0, jnp.eye(d), s * jnp.eye(1))) for s in grid]
    k = int(np.argmax(vals)); assert 0 < k < len(grid) - 1
    assert 0.003 < grid[k] < 0.03, grid[k]     # around the true 0.01


# --- nonlinear model: the Laplace evidence and belief -----------------------------------------

def _tanh_problem(n=30, noise_sd=0.1, seed=11):
    rng = np.random.default_rng(seed); m = Tanh()
    X = jnp.asarray(rng.uniform(-2, 2, size=(n, 1, 1))); z_true = jnp.array([1.5, 1.2, -0.3])
    y = m(z_true, X).reshape(-1) + noise_sd * jnp.asarray(rng.normal(size=n))
    return m, X, y, z_true


def test_nonlinear_map_is_stationary_and_laplace_cov_is_the_inverse_gauss_newton_hessian():
    m, X, y, _ = _tanh_problem(); S0 = 4.0 * jnp.eye(3); R = jnp.array([[0.01]]); mu0 = jnp.zeros((3, 1))
    z_map, cov = map_estimate(m, X, y, mu0, S0, R)
    logpost = lambda z: -0.5 * jnp.sum((m(z, X).reshape(-1) - y) ** 2) / 0.01 - 0.5 * jnp.sum(z ** 2) / 4.0
    g = jax.grad(logpost)(jnp.asarray(z_map)); assert float(jnp.max(jnp.abs(g))) < 1e-5, g
    J = np.asarray(m.jacobian(z_map, X)).reshape(-1, 3); A = np.linalg.inv(np.asarray(S0)) + J.T @ J / 0.01
    np.testing.assert_allclose(cov, np.linalg.inv(A), rtol=1e-7, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(cov) > 0) and np.trace(cov) < np.trace(np.asarray(S0))


def test_laplace_evidence_at_the_map_matches_the_true_marginal_likelihood_by_importance_sampling():
    m, X, y, _ = _tanh_problem(); S0 = 4.0 * jnp.eye(3); R = jnp.array([[0.01]]); mu0 = jnp.zeros((3, 1))
    z_map, cov = map_estimate(m, X, y, mu0, S0, R)
    lap = float(log_marginal_likelihood(m, z_map, X, y, mu0, S0, R))
    prop = multivariate_t(loc=z_map, shape=2.0 * cov, df=5, seed=0); Z = prop.rvs(size=200000)
    F = np.asarray(jax.vmap(lambda z: m(z, X).reshape(-1))(jnp.asarray(Z)))
    loglik = -0.5 * np.sum((F - np.asarray(y)) ** 2, axis=1) / 0.01 - 0.5 * len(y) * np.log(2 * np.pi * 0.01)
    logprior = multivariate_normal(mean=np.zeros(3), cov=np.asarray(S0)).logpdf(Z)
    lw = loglik + logprior - prop.logpdf(Z); true = np.log(np.mean(np.exp(lw - lw.max()))) + lw.max()
    assert abs(lap - true) < 0.25, (lap, true)


def test_nonlinear_empirical_bayes_converges_and_fits_a_sensible_prior_scale():
    m, X, y, z_true = _tanh_problem(n=60)
    eb = empirical_bayes_init(m, X, y, jnp.zeros((3, 1)), jnp.eye(3), jnp.array([[0.01]]), fit_noise=False, iterations=4)
    taus = [h[0] for h in eb.history]
    assert abs(np.log(taus[-1] / taus[-2])) < 0.05, taus          # alternation has settled
    assert 0.2 < eb.prior_scale < 20, eb.prior_scale              # ||z_true||^2 / 3 = 1.26
    np.testing.assert_allclose(np.asarray(eb.mean).ravel(), np.asarray(z_true), atol=0.15)
    assert np.allclose(np.asarray(eb.cov), np.asarray(eb.cov).T) and np.all(np.linalg.eigvalsh(np.asarray(eb.cov)) > 0)


# --- Flax network path and runner integration --------------------------------------------------

INPUT_DIM, NUM_TRAIN, NUM_POOL, NOISE_VAR = 2, 40, 12, 0.05


def _nn_experiment(init="filter", seed=0, warm_start=4, **eb):
    flax_model = LinearNN(input_dim=INPUT_DIM, rngs=nnx.Rngs(0)); truth = deepcopy(flax_model)
    nnx.update(truth, truth.weights_to_state(jax.random.normal(jax.random.PRNGKey(1234), (flax_model.weight_size, 1))))
    data = create_synthetic_data(truth, num_train=NUM_TRAIN, num_test=NUM_POOL, input_dim=INPUT_DIM, embedding_dim=1,
                                 embedding_noise_std=0.1, measurement_noise_std=NOISE_VAR ** 0.5, key=jax.random.PRNGKey(0))
    return Experiment(model=NeuralNetworkRegressor(flax_model), data=data, latent_cov=0.5 * jnp.eye(flax_model.weight_size),
                      latent_innovation=0.0, measurement_error=NOISE_VAR * jnp.eye(1), warm_start=warm_start, seed=seed,
                      init=init, empirical_bayes_kwargs=eb)


def test_empirical_bayes_runs_on_a_flax_network_and_the_belief_feeds_both_filters():
    exp = _nn_experiment(); model, data = exp.model, exp.data; d = model.flax_model.weight_size
    idx = jnp.arange(6)
    eb = empirical_bayes_init(model, data.x_train[idx], data.y_train[idx], exp.state_init_prior[0], exp.state_init_prior[1],
                              exp.measurement_error, fit_noise=True, iterations=2)
    assert eb.mean.shape == (d, 1) and eb.cov.shape == (d, d) and np.isfinite(eb.log_marginal_likelihood)
    assert eb.prior_scale > 0 and eb.noise_scale > 0 and np.all(np.linalg.eigvalsh(np.asarray(eb.cov)) > 0)
    np.testing.assert_allclose(np.asarray(eb.noise_cov), eb.noise_scale * np.asarray(exp.measurement_error))
    np.testing.assert_allclose(np.asarray(eb.prior_cov), eb.prior_scale * np.asarray(exp.state_init_prior[1]))
    for F in (EKF, UKF):
        filt = F(model, eb.mean, eb.cov, 0.0, eb.noise_cov)
        assert np.all(np.isfinite(np.asarray(filt.calculate_epig(data.x_train[6:12], data.x_test_pool))))


def test_runner_with_empirical_bayes_init_starts_every_criterion_from_the_laplace_belief():
    exp = _nn_experiment(init="empirical_bayes", fit_noise=False, iterations=2)
    res = exp.run_experiment(criteria=["EPIG", "EIG"], filter_types=["ekf", "ukf"], filter_params=[{}, {}], iterations=3,
                             optimizer_method="brute_force", trace=True)
    for ft in ("ekf", "ukf"):
        d = res.experiment_results_dict[ft]; eb = d["WARM-START"].init_info
        assert eb is not None and eb.prior_scale > 0
        # the first traced filter state of every criterion is the empirical Bayes belief, not the warm-start filter state
        for c in ("EPIG", "EIG"):
            f0 = d[c].filters[0]
            np.testing.assert_allclose(np.asarray(f0.state_prior[0]).ravel(), np.asarray(eb.mean).ravel(), atol=1e-10)
            np.testing.assert_allclose(np.asarray(f0.state_prior[1]), np.asarray(eb.cov), atol=1e-10)
            assert np.allclose(np.asarray(f0.measurement_error), np.asarray(eb.noise_cov))
            assert len(set(d[c].selected_indices)) == 3 and set(d[c].selected_indices).isdisjoint(d["WARM-START"].selected_indices)


def test_default_init_is_unchanged_and_unknown_init_raises():
    a = _nn_experiment().run_experiment(criteria=["EPIG"], filter_types=["ekf"], filter_params=[{}], iterations=3, optimizer_method="brute_force")
    b = _nn_experiment(init="filter").run_experiment(criteria=["EPIG"], filter_types=["ekf"], filter_params=[{}], iterations=3, optimizer_method="brute_force")
    assert a.experiment_results_dict["ekf"]["EPIG"].selected_indices == b.experiment_results_dict["ekf"]["EPIG"].selected_indices
    assert a.experiment_results_dict["ekf"]["WARM-START"].init_info is None
    with pytest.raises(ValueError):
        _nn_experiment(init="laplace")


def test_cli_validates_the_empirical_bayes_table():
    from bed.cli.main import _build_empirical_bayes
    assert _build_empirical_bayes({"fit_noise": True, "iterations": 2}) == {"fit_noise": True, "iterations": 2}
    with pytest.raises(ValueError):
        _build_empirical_bayes({"fit_noise": True, "iters": 2})
