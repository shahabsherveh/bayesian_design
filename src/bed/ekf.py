from bed.utils import state_to_weights, weights_to_state
from flax import nnx
from .models import Model
from scipy.stats import multivariate_normal
import jax.numpy as jnp
import numpy as np


class EKF:
    """
    Extended Kalman Filter for state estimation and experimental design.

    Implements EKF updates for nonlinear state-space models. Used for
    sequential experimental design where the goal is to estimate hidden
    states based on noisy measurements.

    Model form:
        x_t = f(x_{t-1}) + w_t,  w_t ~ N(0, Q)  (state dynamics)
        y_t = h(x_t, d_t) + v_t, v_t ~ N(0, R)  (measurement model)

    where d_t is the experimental design at time t.

    Attributes:
        model: Model defining measurement function h(x, d)
        state_prior: Tuple (mean, cov) of prior state distribution
        measurement_error: Measurement noise covariance R
    """

    def __init__(
        self,
        model,
        state_cov,
        state_innovation,
        measurement_error,
    ):
        """
        Initialize Extended Kalman Filter.

        Args:
            model: Model object with __call__ and jacobian methods
            state_prev: Prior state mean (from previous time step)
            state_cov_prev: Prior state covariance
            state_innovation: Process noise covariance Q
            measurement_error: Measurement noise covariance R
        """
        self.model = model
        self.state_mean_prior = self.get_prior_weights()
        self.state_cov_prior = state_cov + state_innovation
        self.measurement_error = measurement_error

    def get_prior_weights(self):
        model_state = nnx.state(self.model)
        model_params = nnx.filter_state(model_state, nnx.Param)
        model_weights = state_to_weights(model_params)
        return model_weights

    def get_state_posterior(self, measurement, x):
        """
        Update state estimate given a measurement (correction step).

        Applies EKF update equations using linearization around current
        state estimate.

        Args:
            measurement: Observed measurement value y
            x: Design/input at which measurement was taken

        Returns:
            tuple: (posterior_mean, posterior_cov) after incorporating measurement
        """
        prior_mean = self.state_mean_prior
        prior_cov = self.state_cov_prior
        H = self.model.gradient(x)
        S = H @ prior_cov @ H.T + self.measurement_error
        K = prior_cov @ H.T @ jnp.linalg.inv(S)
        mean_post = prior_mean + K @ (measurement - self.model(x))
        cov_post = (np.eye(len(prior_mean)) - K @ H) @ prior_cov @ (
            np.eye(len(prior_mean)) - K @ H
        ).T + K @ self.measurement_error @ K.T
        return mean_post, cov_post

    def update_model_state(self, mean_post, cov_post):
        """
        Update the model's internal state with the new posterior distribution.

        This method should be called after get_state_posterior to ensure that
        the model's parameters are updated to reflect the new state estimate.

        Args:
            mean_post: Posterior mean of the state after measurement update
            cov_post: Posterior covariance of the state after measurement update
        """
        self.state_mean_prior = mean_post
        self.state_cov_prior = cov_post
        model_state_new = weights_to_state(mean_post, self.model)
        nnx.update(self.model, model_state_new)

    def measurement_prior(self, x):
        """
        Get predictive distribution of measurement before observing.

        Args:
            x: Design/input for hypothetical measurement

        Returns:
            tuple: (mean, cov) of predicted measurement distribution
        """
        H = self.model.gradient(x)
        mean_meas = self.model(x)
        cov_meas = H @ self.state_cov_prior @ H.T + self.measurement_error
        return mean_meas, cov_meas

    def measurement_posterior_cov_estimate(self, x_pred, x_obs):
        """
        Estimate posterior measurement covariance without actual observation.

        Useful for design optimization - predicts uncertainty reduction
        for hypothetical measurement at x_obs.

        Args:
            x_pred: Design point for prediction
            x_obs: Hypothetical observation point

        Returns:
            Estimated posterior covariance of measurement at x_pred
        """
        meas_prior_pred = self.measurement_prior(x_pred)
        meas_prior_obs = self.measurement_prior(x_obs)
        cov_cross = (
            self.model.gradient(x_pred)
            @ self.state_cov_prior
            @ self.model.gradient(x_obs).T
        )
        K = cov_cross @ jnp.linalg.inv(meas_prior_obs[1])
        cov_meas_pos = meas_prior_pred[1] - K @ cov_cross.T
        return cov_meas_pos
