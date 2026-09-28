"""PK/PD forward model, vector observations through both filters, and the general Monte Carlo EPIG."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bed.data import create_synthetic_data
from bed.ekf import EKF
from bed.experiments import Experiment
from bed.mechanistic import PKPDModel, STEADY_STATE_DOSES, make_pkpd_problem
from bed.models import DenseNN, NeuralNetworkRegressor
from bed.ukf import UKF

jax.config.update("jax_enable_x64", True)


def test_pkpd_shapes_and_jacobian_against_finite_differences():
    model = PKPDModel()
    z = jnp.log(jnp.array([1.5, 0.09, 35.0, 0.5, 100.0, 8.0, 10.0]))
    x = jnp.array([[200.0, 24.0, 1.0, 2.0], [400.0, 8.0, STEADY_STATE_DOSES, 8.0], [100.0, 12.0, 3.0, 5.0]])[:, None, None, :]
    y = model(z, x)
    assert y.shape == (3, 1, 1, 2)
    jac = model.jacobian(z, x)
    assert jac.shape == (3, 2, 7)
    h = 1e-6
    fd = jnp.stack([(model(z + h * jnp.eye(7)[i], x) - model(z - h * jnp.eye(7)[i], x)).reshape(3, 2) / (2 * h) for i in range(7)], axis=-1)
    assert jnp.allclose(jac, fd, rtol=1e-5, atol=1e-6)


def test_pkpd_steady_state_is_the_limit_of_repeated_dosing_and_exceeds_a_single_dose():
    model = PKPDModel()
    z = jnp.log(jnp.array([1.5, 0.09, 35.0, 0.5, 100.0, 8.0, 10.0]))
    x_ss = jnp.array([[300.0, 8.0, STEADY_STATE_DOSES, 4.0]])[:, None, None, :]
    x_many = jnp.array([[300.0, 8.0, 200.0, 4.0]])[:, None, None, :]
    x_one = jnp.array([[300.0, 8.0, 1.0, 4.0]])[:, None, None, :]
    assert jnp.allclose(model(z, x_ss), model(z, x_many), rtol=1e-8)
    assert float(model(z, x_ss)[0, 0, 0, 0]) > float(model(z, x_one)[0, 0, 0, 0])   # log C
    assert float(model(z, x_ss)[0, 0, 0, 1]) > float(model(z, x_one)[0, 0, 0, 1])   # effect


def test_pkpd_problem_builder_shapes_and_noise_free_targets():
    p = make_pkpd_problem(seed=1, num_times=10, num_global=7)
    assert p.data.x_train.shape == (30, 1, 1, 4) and p.data.y_train.shape == (30, 1, 1, 2)
    assert p.data.x_test_pool.shape == (24, 1, 1, 4) and p.data.x_test_glob.shape == (7, 1, 1, 4)
    assert jnp.allclose(p.data.y_test_pool, p.model(p.z_true, p.data.x_test_pool))
    assert p.prior_mean.shape == (7, 1) and p.prior_cov.shape == (7, 7) and p.measurement_error.shape == (2, 2)


def test_filters_agree_on_a_linear_two_output_model_and_ukf_update_handles_vector_observations():
    flax_model = DenseNN(input_dim=2, hidden_dims=[], output_dim=2, rngs=nnx.Rngs(0))
    model = NeuralNetworkRegressor(flax_model)
    z = flax_model.state_to_weights(nnx.state(flax_model)); cov = 0.4 * jnp.eye(flax_model.weight_size)
    R = jnp.array([[0.05, 0.01], [0.01, 0.08]])
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 1, 1, 2)); pool = jax.random.normal(jax.random.PRNGKey(2), (4, 1, 1, 2))
    ekf, ukf = EKF(model, z, cov, 0.0, R), UKF(model, z, cov, 0.0, R)
    assert jnp.allclose(ekf.calculate_eig(x), ukf.calculate_eig(x), rtol=1e-9)
    assert jnp.allclose(ekf.calculate_epig(x, pool), ukf.calculate_epig(x, pool), rtol=1e-9)
    y = jnp.array([[[0.3, -0.2]]])
    me, ce = ekf.get_state_posterior(y, x[:1]); mu, cu = ukf.get_state_posterior(y, x[:1])
    assert jnp.allclose(me.reshape(-1), mu.reshape(-1), rtol=1e-9, atol=1e-12)
    assert jnp.allclose(ce, cu, rtol=1e-9, atol=1e-12)


def test_monte_carlo_epig_converges_to_the_closed_form_with_two_outputs():
    # Nested Monte Carlo is biased downward at finite inner sample size, and the bias grows with
    # the output dimension; the check is convergence, not agreement at one K. Outer samples are
    # one per pool point, so the pool is tiled to give the estimator 60 of them.
    flax_model = DenseNN(input_dim=2, hidden_dims=[], output_dim=2, rngs=nnx.Rngs(0))
    data = create_synthetic_data(flax_model, num_train=20, num_test=6, input_dim=2, embedding_dim=1, output_dim=2,
                                 measurement_noise_std=0.2, key=jax.random.PRNGKey(0))
    R = jnp.diag(jnp.array([0.04, 0.09]))
    exp = Experiment(model=NeuralNetworkRegressor(flax_model), data=data, latent_cov=0.5 * jnp.eye(flax_model.weight_size),
                     measurement_error=R, warm_start=0)
    filt = EKF(exp.model, exp.state_init_prior[0], exp.state_init_prior[1], 0.0, R)
    x = exp.design_space[:5]
    pool = jnp.tile(exp.design_pool, (10, 1, 1, 1))
    closed = np.asarray(exp.calculate_epig(x, filt), dtype=float)
    bias = {}
    for k in (2000, 16000):
        est = np.stack([np.asarray(exp.calculate_epig_mc(x, filt, x_1=pool, num_latent_samples=k), dtype=float) for _ in range(4)])
        assert np.all(np.isfinite(est))
        bias[k] = float(np.abs(est.mean(axis=0) - closed).mean())
        if k == 16000:
            assert np.corrcoef(est.mean(axis=0), closed)[0, 1] > 0.9
    assert bias[16000] < bias[2000], bias
    assert bias[16000] < 0.2, bias


def test_runner_runs_the_pkpd_problem_with_both_filters_and_reports_per_output_rmse():
    p = make_pkpd_problem(seed=0, num_times=20, num_global=10)
    exp = Experiment(model=p.model, data=p.data, latent_cov=p.prior_cov, latent_mean=p.prior_mean,
                     measurement_error=p.measurement_error, warm_start=2, seed=0)
    res = exp.run_experiment(criteria=["EPIG", "EIG", "RANDOM"], filter_types=["ekf", "ukf"], filter_params=[{}, {}],
                             iterations=3, optimizer_method="brute_force")
    for ft in ("ekf", "ukf"):
        r = res.experiment_results_dict[ft]["EPIG"]
        assert len(r.rmse_pool) == 3 and len(r.rmse_pool_outputs) == 3
        assert jnp.asarray(r.rmse_pool_outputs[-1]).shape == (2,)
        assert len(set(r.selected_indices)) == 3
        assert np.all(np.isfinite([float(v) for v in r.crit_values]))
