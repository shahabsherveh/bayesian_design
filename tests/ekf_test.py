from flax import nnx
import jax.numpy as jnp

from bed.ekf import EKF
from bed.models import LinearModel, LinearNN, NeuralNetworkRegressor


class TestEKF:
    def test_pos(self):
        # Define a simple linear model
        model = LinearNN(input_dim=2, rngs=nnx.Rngs(0))
        state = jnp.array([[1.0], [1.0], [2.0]])  # Initial state
        state_cov = 0.3 * jnp.eye(3)  # Initial covariance
        state_innovation_cov = 0
        measurement_cov = 0.25 * jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[3]])
        X = jnp.array([10, 1.5]).reshape(1, 1, 1, 2)
        X_bias = jnp.concatenate(
            [
                jnp.ones((1, 1, 1, 1)),
                X,
            ],
            axis=-1,
        ).squeeze()  # Add bias term
        ekf = EKF(
            NeuralNetworkRegressor(model),
            state,
            state_cov,
            state_innovation_cov,
            measurement_cov,
        )
        posterior = ekf.get_state_posterior(measurement=measurement, x=X)
        cov_prior = state_cov
        mean_pior = state
        S = jnp.linalg.inv(cov_prior) + jnp.linalg.inv(measurement_cov)[
            0, 0
        ] * jnp.outer(X_bias, X_bias)
        cov_pos = jnp.linalg.inv(S)
        mean_pos = cov_pos @ (
            jnp.linalg.inv(cov_prior) @ mean_pior
            + jnp.linalg.inv(measurement_cov)[0, 0]
            * X_bias[:, None]
            * measurement[0, 0]
        )

        assert jnp.allclose(posterior[0], mean_pos, rtol=1e-4)
        assert jnp.allclose(posterior[1], cov_pos, rtol=1e-4)

    def test_pred_pos_estimate(self):
        # Define a simple linear model
        model = LinearNN(input_dim=2, rngs=nnx.Rngs(0))
        state = jnp.array([[3.0], [1.0], [0.0]])  # Initial state
        state_cov = jnp.eye(3)  # Initial covariance
        state_innovation_cov = 0
        measurement_cov = jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[10.0]])
        X = jnp.array([[1.0, 0.5]]).reshape(1, 1, 1, 2)
        ekf = EKF(
            NeuralNetworkRegressor(model),
            state,
            state_cov,
            state_innovation_cov,
            measurement_cov,
        )
        posterior = ekf.measurement_posterior(
            measurement=measurement, x_obs=X, x_pred=X + 1
        )

        cov_est = ekf.measurement_posterior_cov_estimate(
            X + 1,
            X,
        )
        assert jnp.allclose(posterior[1], cov_est)
