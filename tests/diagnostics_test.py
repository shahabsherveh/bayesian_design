"""Level 0 instruments (PROGRESS.md, H0.2, H0.3) and Level 1 regression (H1.1)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bed.diagnostics import filter_moments, gaussian_epig, mc_moments, moment_errors
from bed.ekf import EKF
from bed.models import DenseNN, NeuralNetworkRegressor
from bed.ukf import UKF

jax.config.update("jax_enable_x64", True)
R = 0.05 * jnp.eye(1)


def _setup(hidden_dims, prior_var=0.5, seed=0):
    flax_model = DenseNN(input_dim=2, hidden_dims=hidden_dims, output_dim=1, rngs=nnx.Rngs(seed), activation="tanh")
    model = NeuralNetworkRegressor(flax_model)
    z = flax_model.state_to_weights(nnx.state(flax_model))
    cov = prior_var * jnp.eye(flax_model.weight_size)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 1, 1, 2))
    pool = jax.random.normal(jax.random.PRNGKey(2), (5, 1, 1, 2))
    return model, z, cov, x, pool


def test_h03_mc_moments_are_exact_for_a_linear_model_up_to_sampling_error():
    model, z, cov, x, pool = _setup(hidden_dims=[])
    ekf = EKF(model, z, cov, 0.0, R)
    exact = filter_moments(ekf, x, pool)                  # exact for a linear model
    errors = {}
    for n in (1_000, 16_000):
        mc = mc_moments(model, z, cov, x, pool, measurement_error=R, num_samples=n, key=jax.random.PRNGKey(3))
        errors[n] = moment_errors(mc, exact)
    # sampling error shrinks like 1/sqrt(N): a 16x larger sample should cut it by about 4x
    for key in ("mean", "var", "cross"):
        assert errors[16_000][key] < errors[1_000][key] / 2, (key, errors)
    # and the large-sample estimate is within a few standard errors of the exact moments
    var_scale = float(jnp.diagonal(exact.cov, axis1=-2, axis2=-1).mean())
    assert errors[16_000]["mean"] < 4 * jnp.sqrt(var_scale / 16_000)
    assert errors[16_000]["var"] < 0.05 * var_scale


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_h11_ekf_and_ukf_moments_are_exact_for_a_linear_model(alpha):
    model, z, cov, x, pool = _setup(hidden_dims=[])
    ekf = EKF(model, z, cov, 0.0, R)
    ukf = UKF(model, z, cov, 0.0, R, alpha=alpha)
    e, u = filter_moments(ekf, x, pool), filter_moments(ukf, x, pool)
    for a, b in ((e.mean, u.mean), (e.cov, u.cov), (e.cov_prime, u.cov_prime), (e.cross, u.cross)):
        assert jnp.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_gaussian_epig_from_filter_moments_matches_the_filters_own_criterion():
    model, z, cov, x, pool = _setup(hidden_dims=[3])
    for filt in (EKF(model, z, cov, 0.0, R), UKF(model, z, cov, 0.0, R)):
        m = filter_moments(filt, x, pool)
        assert jnp.allclose(gaussian_epig(m.cov, m.cov_prime, m.cross), filt.calculate_epig(x, pool), rtol=1e-10)


def test_h02_monte_carlo_epig_bias_shrinks_with_the_inner_sample_size():
    from bed.data import create_synthetic_data
    from bed.experiments import Experiment
    from copy import deepcopy
    flax_model = DenseNN(input_dim=2, hidden_dims=[], output_dim=1, rngs=nnx.Rngs(0))
    truth = deepcopy(flax_model)
    data = create_synthetic_data(truth, num_train=20, num_test=6, input_dim=2, embedding_dim=1,
                                 measurement_noise_std=float(R[0, 0]) ** 0.5, key=jax.random.PRNGKey(0))
    exp = Experiment(model=NeuralNetworkRegressor(flax_model), data=data,
                     latent_cov=0.5 * jnp.eye(flax_model.weight_size), measurement_error=R, warm_start=0)
    filt = EKF(exp.model, exp.state_init_prior[0], exp.state_init_prior[1], 0.0, R)
    x = exp.design_space[:3]
    closed = np.asarray(exp.calculate_epig(x, filt), dtype=float)
    bias = {}
    for k in (100, 3000):
        est = np.stack([np.asarray(exp.calculate_epig_mc(x, filt, num_latent_samples=k), dtype=float) for _ in range(30)])
        bias[k] = float(np.abs(est.mean(axis=0) - closed).mean())
    assert bias[3000] < bias[100], bias
