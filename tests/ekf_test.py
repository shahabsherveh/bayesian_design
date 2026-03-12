import jax.numpy as jnp
import matplotlib.pyplot as plt
from bed.models import EKF, LinearModel


class TestLinearModel:
    def test_jacobian(self):
        X = jnp.array([[1.0, 0.5], [0.0, 1.0]])
        z = jnp.array([[0.5], [1.0]])
        model = LinearModel()
        assert model.jacobian(z, X)


class TestEKF:
    def test_pos(self):
        # Define a simple linear model
        model = LinearModel()
        state = jnp.array([[0.0], [0.0]])  # Initial state
        state_cov = jnp.eye(2)  # Initial covariance
        state_innovation_cov = 0
        measurement_cov = jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[1.0]])
        X = jnp.array([[1.0, 0.5]])
        ekf = EKF(
            model,
            state,
            state_cov,
            state_innovation_cov,
            measurement_cov,
        )
        posterior = ekf.get_state_posterior(measurement=measurement, x=X)
        cov_prior = state_cov
        mean_pior = state
        S = jnp.linalg.inv(cov_prior) + jnp.linalg.inv(measurement_cov)[0, 0] * X.T @ X
        cov_pos = jnp.linalg.inv(S)
        mean_pos = cov_pos @ (
            jnp.linalg.inv(cov_prior) @ mean_pior
            + jnp.linalg.inv(measurement_cov)[0, 0] * X.T * measurement[0, 0]
        )

        assert jnp.allclose(posterior.mean, mean_pos.flatten())
        assert jnp.allclose(posterior.cov, cov_pos)

    def test_pred_pos_estimate(self):
        # Define a simple linear model
        model = LinearModel()
        state = jnp.array([[0.0], [0.0]])  # Initial state
        state_cov = jnp.eye(2)  # Initial covariance
        state_innovation_cov = 0
        measurement_cov = jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[10.0]])
        X = jnp.array([[1.0, 0.5]])
        ekf = EKF(
            model,
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
        assert jnp.allclose(posterior.cov, cov_est)
        breakpoint()
