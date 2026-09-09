import jax.numpy as jnp

from bed.ukf import UKF


class QuadraticModel:
    def __call__(self, state, x):
        return (state[0] ** 2 + x.reshape(-1)[0] * state[1]).reshape(1, 1)


def test_ukf_nonlinear_update_reduces_uncertainty():
    ukf = UKF(
        model=QuadraticModel(),
        state_prev=jnp.array([[1.0], [0.5]]),
        state_cov_prev=jnp.diag(jnp.array([0.2, 0.3])),
        state_innovation=0.0,
        measurement_error=jnp.array([[0.1]]),
    )

    prior_mean, prior_cov = ukf.measurement_prior(jnp.array([[[[2.0]]]]))
    posterior_mean, posterior_cov = ukf.get_state_posterior(
        jnp.array([[2.5]]), jnp.array([[[[2.0]]]])
    )

    assert prior_mean.shape == (1, 1)
    assert posterior_mean.shape == (2,)
    assert posterior_cov.shape == (2, 2)
    assert jnp.trace(posterior_cov) < jnp.trace(ukf.state_prior[1])


def test_ukf_batched_measurement_prior():
    ukf = UKF(
        model=QuadraticModel(),
        state_prev=jnp.array([1.0, 0.5]),
        state_cov_prev=jnp.eye(2),
        state_innovation=0.0,
        measurement_error=jnp.array([[0.1]]),
    )

    mean, covariance = ukf.measurement_prior(
        jnp.array([[[[1.0]]], [[[2.0]]]])
    )

    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)
