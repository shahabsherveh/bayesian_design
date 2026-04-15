import jax.numpy as jnp

from bed.ekf import EKF
from bed.models import LinearModel

from flax import nnx

from bed.utils import state_to_weights


class TestEKF:
    def test_pos(self):
        # Define a simple linear model
        model = LinearModel(input_dim=2, rngs=nnx.Rngs(0))
        model_state = nnx.state(model)
        model_params = nnx.filter_state(model_state, nnx.Param)
        state = state_to_weights(model_params)
        state_cov = jnp.eye(3)  # Initial covariance
        state_innovation_cov = 0
        measurement_cov = jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[1.0]])
        X = jnp.array([[1.0, 0.5]])
        ekf = EKF(
            model,
            state_cov,
            state_innovation_cov,
            measurement_cov,
        )
        posterior = ekf.get_state_posterior(measurement=measurement, x=X)
        cov_prior = state_cov
        mean_pior = state
        X_intercept = jnp.hstack([jnp.ones((X.shape[0], 1)), X])  # Add intercept tem
        S = (
            jnp.linalg.inv(cov_prior)
            + jnp.linalg.inv(measurement_cov)[0, 0] * X_intercept.T @ X_intercept
        )
        cov_pos = jnp.linalg.inv(S)
        mean_pos = cov_pos @ (
            jnp.linalg.inv(cov_prior) @ mean_pior + X_intercept.T * measurement[0, 0]
        )

        assert jnp.allclose(posterior[0], mean_pos)
        assert jnp.allclose(posterior[1], cov_pos)

    def test_pred_pos_estimate(self):
        # Define a simple linear model
        model = LinearModel(input_dim=2, rngs=nnx.Rngs(0))
        state_cov = jnp.eye(3)  # Initial covariance
        state_innovation_cov = 0
        measurement_cov = jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[10.0]])
        X = jnp.array([[1.0, 0.5]])
        ekf = EKF(
            model,
            state_cov,
            state_innovation_cov,
            measurement_cov,
        )

        cov_est = ekf.measurement_posterior_cov_estimate(
            X + 4,
            X,
        )

        posterior = ekf.get_state_posterior(measurement=measurement, x=X)
        ekf.update_model_state(*posterior)
        _, cov_post = ekf.measurement_prior(X + 4)
        assert jnp.allclose(cov_post, cov_est)
