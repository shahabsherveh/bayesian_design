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
        model: Model,
        state_prev,
        state_cov_prev,
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
        self.state_prior = self._get_state_prior(
            state_prev, state_cov_prev, state_innovation
        )
        self.measurement_error = measurement_error

    @staticmethod
    def _get_state_prior(state_prev, state_cov_prev, state_innovation):
        """
        Propagate state distribution through dynamics (prediction step).

        For linear dynamics f(x) = x, this simply adds process noise.

        Args:
            state_prev: Previous state estimate
            state_cov_prev: Previous state covariance
            state_innovation: Process noise covariance Q

        Returns:
            tuple: (prior_mean, prior_cov)
        """
        mean = state_prev
        cov = state_cov_prev + state_innovation * \
            jnp.eye(state_cov_prev.shape[0])
        return mean, cov

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
        prior_mean = self.state_prior[0]
        prior_cov = self.state_prior[1]
        H = self.model.jacobian(prior_mean, x)
        HT = jnp.matrix_transpose(H)
        S = H @ prior_cov @ HT + self.measurement_error
        K = prior_cov @ HT @ jnp.linalg.inv(S)
        KT = jnp.matrix_transpose(K)
        predict_meas = self.model(prior_mean, x)
        epsilon = jnp.matrix_transpose(measurement - predict_meas)
        mean_post = (prior_mean + K @ epsilon).squeeze([0, 1])
        F = np.eye(len(self.state_prior[0])) - K @ H
        FT = jnp.matrix_transpose(F)
        # cov_post = (F @ self.state_prior[1]).squeeze(0)
        cov_post = (
            F @ self.state_prior[1] @ FT + K @ self.measurement_error @ KT
        ).squeeze(0)
        return mean_post, cov_post

    def measurement_prior(self, x):
        """
        Get predictive distribution of measurement before observing.

        Args:
            x: Design/input for hypothetical measurement

        Returns:
            tuple: (mean, cov) of predicted measurement distribution
        """
        H = self.model.jacobian(self.state_prior[0], x)
        HT = jnp.matrix_transpose(H)
        mean_meas = self.model(self.state_prior[0], x)
        cov_meas = H @ self.state_prior[1] @ HT + self.measurement_error
        return mean_meas, cov_meas

    def calculate_epig(self, x, x_1):
        state_prev = self.state_prior[0]
        j_1 = self.model.jacobian(state_prev.reshape(-1, 1), x_1)[None, ...]
        j_1_T = jnp.matrix_transpose(j_1)
        j_0 = self.model.jacobian(state_prev.reshape(-1, 1), x)[:, None, ...]
        j_0_T = jnp.matrix_transpose(j_0)
        sigma = self.state_prior[1]
        _, s_x = self.measurement_prior(x)
        s_x_inv = jnp.linalg.inv(s_x[:, None, ...])
        posterior_covs_deficit = j_1 @ sigma @ (
            j_0_T @ s_x_inv @ j_0) @ sigma @ j_1_T
        cov_0 = j_1 @ sigma @ j_1_T + self.measurement_error

        # epig = -jnp.log(1 - (posterior_covs_deficit / cov_0)) / 2
        epig = (
            -jnp.log(
                jnp.linalg.det(
                    jnp.eye(self.measurement_error.shape[0])
                    - posterior_covs_deficit @ jnp.linalg.inv(cov_0)
                )
            )
            / 2
        )
        # Get the diagonal to ignore the cross-covariance of the design pool_values
        # Makes sense since in classical case the trace where calculated for the information matrix
        # epig = posterior_covs_deficit.diagonal() / cov_0.diagonal()
        return epig.mean(axis=1)

    def measurement_posterior(self, x_pred, x_obs, measurement):
        """
        Get measurement distribution at x_pred after observing at x_obs.

        Args:
            x_pred: Design point for prediction
            x_obs: Design point where measurement was taken
            measurement: Observed value at x_obs

        Returns:
            tuple: (mean, cov) of measurement distribution at x_pred
        """
        state_post = self.get_state_posterior(measurement, x_obs)
        H = self.model.jacobian(state_post[0], x_pred)
        HT = jnp.matrix_transpose(H)
        mean_meas_post = self.model(state_post[0], x_pred)
        cov_meas_post = H @ state_post[1] @ HT + self.measurement_error
        return mean_meas_post, cov_meas_post

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
        H_obs = self.model.jacobian(self.state_prior[0], x_obs)
        H_obs_T = jnp.matrix_transpose(H_obs)
        H_pred = self.model.jacobian(self.state_prior[0], x_pred)
        H_pred_T = jnp.matrix_transpose(H_pred)
        cov_cross = H_pred @ self.state_prior[1] @ H_obs_T
        cov_cross_T = cov_cross.T if cov_cross.ndim == 2 else cov_cross.swapaxes(
            1, 2)
        K = cov_cross @ jnp.linalg.inv(meas_prior_obs[1])
        cov_meas_pos = meas_prior_pred[1] - K @ cov_cross_T
        return cov_meas_pos

    def calculate_mutual_information(self, x_pred, x_obs):
        """
        Calculate mutual information between measurements at x_pred and x_obs.

        Quantifies how much observing at x_obs reduces uncertainty about
        potential measurement at x_pred.

        Args:
            x_pred: Prediction design point
            x_obs: Observation design point

        Returns:
            Mutual information in nats
        """
        cov_pos_estimate = self.measurement_posterior_cov_estimate(
            x_pred, x_obs)
        cov_prior = self.measurement_prior(x_pred)[1]
        prior = multivariate_normal(cov=cov_prior)
        posterior = multivariate_normal(cov=cov_pos_estimate)
        return prior.entropy() - posterior.entropy()

    def measurement_preditive_prior(self, x):
        """
        Compute prior predictive distribution of measurement.

        Alias/duplicate of measurement_prior(). Returns the distribution of
        measurements before making observations.

        Args:
            x: Design vector for measurement prediction

        Returns:
            Tuple (mean, cov) of prior predictive distribution

        Note:
            This is functionally identical to measurement_prior().
            Consider using measurement_prior() instead to avoid duplication.
        """
        state_prior = self.state_prior
        H = self.model.jacobian(state_prior[0], x)
        HT = jnp.matrix_transpose(H)
        mean_meas_prior = self.model(state_prior[0], x)
        cov_meas_prior = H @ state_prior[1] @ HT + self.measurement_error
        return mean_meas_prior, cov_meas_prior

    def calculate_eig(self, x, *arg, **kwargs):
        """
        Calculate Expected Information Gain (EIG) about parameters.

        EIG measures how much information observing at x_1 provides about
        the latent parameters themselves (not predictions). For linear models,
        this equals x_1^T @ Cov_prior @ x_1, which is the prior variance
        of measurements at x_1.

        Args:
            x_0: Proposed observation design of shape (d, 1)
            x_1: Not used (kept for interface compatibility with calculate_epig)

        Returns:
            EIG value (scalar). Higher values indicate more informative designs.

        Note:
            For linear Gaussian models, EIG has a closed form and doesn't require
            Monte Carlo estimation. The optimal EIG design is proportional to
            the eigenvector with largest eigenvalue of the prior covariance.
        """
        state_prior_mean = self.state_prior[0]
        state_prior_cov = self.state_prior[1]
        measurement_error = self.measurement_error

        H = self.model.jacobian(state_prior_mean.reshape(-1, 1), x)
        H_T = H.T if H.ndim == 2 else H.swapaxes(1, 2)

        eig = jnp.log((H @ state_prior_cov @ H_T / measurement_error) + 1) / 2
        return jnp.atleast_1d(eig.squeeze())
