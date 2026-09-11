"""Unscented Kalman filtering for nonlinear measurement models."""

from os import environ

from flax.nnx import Rngs
import jax
from scipy.stats import multivariate_normal
import jax.numpy as jnp

from bed._models import DenseNN, NeuralNetworkRegressor, Model


class UKF:
    """Unscented Kalman Filter with the same interface as :class:`EKF`.

    The state transition is the identity, matching the random-walk model used
    by the sequential experiment runner.
    """

    def __init__(
        self,
        model: Model,
        state_prev,
        state_cov_prev,
        state_innovation,
        measurement_error,
        alpha=0.1,
        beta=2.0,
        kappa=0.0,
    ):
        self.model = model
        self.measurement_error = jnp.asarray(measurement_error)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._state_prior = None
        self.state_innovation = (
            state_innovation * jnp.eye(state_cov_prev.shape[0])
            if jnp.ndim(state_innovation) == 0
            else jnp.asarray(state_innovation)
        )
        self.state_prior = self._get_state_prior(
            state_prev, state_cov_prev, self.state_innovation
        )

    @staticmethod
    def _get_state_prior(state_prev, state_cov_prev, state_innovation):
        state_cov_prev = jnp.asarray(state_cov_prev)
        return jnp.asarray(state_prev).reshape(-1), state_cov_prev + state_innovation

    def _get_sigma_points(self, mean, covariance):
        n = mean.size
        lam = self.alpha**2 * (n + self.kappa) - n
        scale = n + lam
        covariance = (covariance + covariance.T) / 2
        root = jnp.linalg.cholesky(scale * covariance)
        points = jnp.concatenate(
            [mean[None], mean[None] + root.T, mean[None] - root.T], axis=0
        )
        weights_mean = jnp.full(2 * n + 1, 1 / (2 * scale))
        weights_cov = weights_mean.at[0].set(
            1 - n / scale + (1 - self.alpha**2 + self.beta)
        )
        weights_mean = weights_mean.at[0].set(1 - n / scale)
        return points, jnp.expand_dims(weights_mean, axis=[0, 2, 3, 4]), jnp.expand_dims(weights_cov, axis=[0, 2, 3, 4])

    @property
    def state_prior(self):
        return self._state_prior

    @state_prior.setter
    def state_prior(self, value):
        mean, cov = value
        self.sigma_points = self._get_sigma_points(mean, cov)
        self._state_prior = mean, cov

    def _measurements(self, points, x):
        values = jax.vmap(lambda theta: self.model(
            theta.T, x), out_axes=1)(points)
        return values

    def _measurement_statistics(self, x):
        if jnp.ndim(x) == 3:
            x = x[None, ...]
        sigma_points, sigma_weights_m, sigma_weights_c = self.sigma_points
        latent_mean = sigma_points[:1]
        values = self._measurements(sigma_points, x)
        mean = jnp.sum(values * sigma_weights_m, axis=1)
        deviations = values - jnp.expand_dims(mean, axis=1)
        covariance = (
            jnp.sum(sigma_weights_c *
                    deviations * deviations, axis=1)
            + self.measurement_error
        )
        cross_covariance = jnp.sum(
            jnp.expand_dims(sigma_points - latent_mean, axis=[
                            0, 2, 4]) @ (sigma_weights_c * deviations),
            axis=1
        )
        return mean, covariance, cross_covariance

    def get_state_posterior(self, measurement, x):
        """Update the state using an unscented measurement transform."""
        measurement = jnp.asarray(measurement).reshape(-1)
        mean_meas, S_x, P_x = (
            self._measurement_statistics(x)
        )
        gain = (
            P_x) @ jnp.linalg.inv(S_x
                                  )
        mean = self.state_prior[0] + \
            jnp.squeeze(gain @ (measurement - mean_meas))
        covariance = jnp.squeeze(
            self.state_prior[1] - gain @ S_x @ jnp.matrix_transpose(gain))
        covariance = (covariance + covariance.T) / 2
        return mean, covariance

    def measurement_prior(self, x):
        """Return the unscented predictive distribution at a design point."""
        mean, covariance, _ = self._measurement_statistics(x)
        return mean, covariance

    def measurement_posterior(self, x_pred, x_obs, measurement):
        """Return the predictive distribution after an observed measurement."""
        state_post = self.get_state_posterior(measurement, x_obs)
        old_state = self.state_prior
        self.state_prior = state_post
        try:
            mean, covariance = self.measurement_prior(x_pred)
        finally:
            self.state_prior = old_state
        return mean, covariance

    def measurement_posterior_cov_estimate(self, x_pred, x_obs):
        """Estimate predictive covariance after observing at ``x_obs``."""
        if jnp.ndim(x_pred) == 3:
            x_pred = x_pred[None, ...]
        if jnp.ndim(x_obs) == 3:
            x_obs = x_obs[None, ...]
        points, weights_mean, weights_cov = self.sigma_points
        pred_values = self._measurements(points, x_pred)
        obs_values = self._measurements(points, x_obs)
        pred_mean = jnp.sum(pred_values * weights_mean, axis=1)
        obs_mean = jnp.sum(obs_values * weights_mean, axis=1)
        pred_dev = pred_values - jnp.expand_dims(pred_mean, axis=1)
        obs_dev = obs_values - jnp.expand_dims(obs_mean, axis=1)
        cov_pred = jnp.sum(weights_cov * pred_dev * pred_dev, axis=1) + self.measurement_error
        cov_obs = jnp.sum(weights_cov * obs_dev * obs_dev, axis=1) + self.measurement_error
        cross = jnp.sum(weights_cov * pred_dev * obs_dev, axis=1)
        return cov_pred - cross * cross / cov_obs

    def calculate_mutual_information(self, x_pred, x_obs):
        """Calculate mutual information in nats."""
        posterior_cov = self.measurement_posterior_cov_estimate(x_pred, x_obs)
        prior_cov = self.measurement_prior(x_pred)[1]
        return multivariate_normal(cov=prior_cov).entropy() - multivariate_normal(
            cov=posterior_cov
        ).entropy()

    def measurement_preditive_prior(self, x):
        """Backward-compatible alias for :meth:`measurement_prior`."""
        return self.measurement_prior(x)

    def _sigma_deviations(self, x):
        """Sigma-point measurement statistics at the designs ``x``.

        Args:
            x: Designs of shape ``(B, 1, 1, d)`` (a single ``(1, 1, d)`` design is
                promoted to a batch of one).

        Returns:
            tuple: ``(mean, deviations)`` with ``mean`` of shape ``(B, d_y)`` and
            ``deviations`` of shape ``(B, K, d_y)`` holding ``f(sigma_k, x_b) - mean_b``
            for the ``K = 2 n + 1`` sigma points of the current state prior (``n`` the
            state dimension).
        """
        if jnp.ndim(x) == 3:
            x = x[None, ...]
        points, weights_mean, _ = self.sigma_points
        values = self._measurements(points, x)
        values = values.reshape(values.shape[0], values.shape[1], -1)
        mean = jnp.sum(values * weights_mean.reshape(1, -1, 1), axis=1)
        return mean, values - mean[:, None, :]

    def calculate_epig(self, x, x_1):
        """Closed-form EPIG of the candidate designs ``x`` for the test pool ``x_1``.

        All covariances come from one sigma-point transform of the current state
        prior. With ``dev`` the sigma-point deviations of ``_sigma_deviations`` and
        ``w`` the covariance weights,

            S_x   = sum_k w_k dev_x[k] dev_x[k]^T + R          (candidate)
            S'_j  = sum_k w_k dev_j[k] dev_j[k]^T + R          (test point j)
            C_j   = sum_k w_k dev_j[k] dev_x[k]^T              (cross-covariance)

        and ``EPIG(x) = mean_j 1/2 [log det S'_j - log det (S'_j - C_j S_x^{-1} C_j^T)]``.
        The Schur complement is the predictive covariance at ``x'_j`` conditional on
        ``y`` under the Gaussian approximation of the joint ``(y, y'_j)``. For a
        forward model linear in the state the sigma-point moments are exact, so
        ``C_j = J'_j Sigma_t J_x^T`` and the result equals the EKF closed form.
        No inverse of the state covariance is needed.

        Args:
            x: Candidate designs, shape ``(B, 1, 1, d)``.
            x_1: Test pool, shape ``(M, 1, 1, d)``.

        Returns:
            EPIG per candidate, shape ``(B,)``, in nats.
        """
        weights_cov = self.sigma_points[2].reshape(-1)
        _, dev_x = self._sigma_deviations(x)
        _, dev_1 = self._sigma_deviations(x_1)
        R = self.measurement_error
        S_x = jnp.einsum("k,bka,bkc->bac", weights_cov, dev_x, dev_x) + R
        S_1 = jnp.einsum("k,mka,mkc->mac", weights_cov, dev_1, dev_1) + R
        C = jnp.einsum("k,mka,bkc->bmac", weights_cov, dev_1, dev_x)
        S_plus = S_1[None] - C @ jnp.linalg.inv(S_x)[:, None] @ jnp.matrix_transpose(C)
        logdet_1 = jnp.linalg.slogdet(S_1).logabsdet
        logdet_plus = jnp.linalg.slogdet(S_plus).logabsdet
        epig = 0.5 * (logdet_1[None, :] - logdet_plus)
        return epig.mean(axis=1)

    def calculate_eig(self, x, *args, **kwargs):
        mean, S_x, P_x = self._measurement_statistics(x)
        _, state_cov = self.state_prior
        state_cov_post = state_cov - \
            P_x @ jnp.linalg.inv(S_x) @ jnp.matrix_transpose(P_x)
        eig = 0.5 * (jnp.linalg.slogdet(state_cov).logabsdet -
                     jnp.linalg.slogdet(state_cov_post).logabsdet)
        return eig.squeeze()


if __name__ == "__main__":
    environ["JAX_TRACEBACK_FILTERING"] = "off"
    rngs = Rngs(0)
    key = jax.random.key(0)
    designs = jax.random.normal(shape=(10, 1, 1, 5), key=key)
    x = designs[0][None, ...]
    input_dim = designs.shape[-1]
    hidden_dims = [2, 2]
    output_dim = 1
    node_dims = jnp.array([input_dim] + hidden_dims + [output_dim])
    latent_dim = jnp.sum(
        node_dims[:-1]*node_dims[1:]) + jnp.sum(node_dims[1:])
    flax_model = DenseNN(input_dim=input_dim, hidden_dims=hidden_dims,
                         output_dim=output_dim, rngs=rngs)
    model = NeuralNetworkRegressor(model=flax_model)
    state_prev_cov = jnp.eye(latent_dim)
    state_prev = jnp.zeros(latent_dim)
    ukf = UKF(model=model, state_prev=state_prev, state_cov_prev=state_prev_cov,
              state_innovation=0, measurement_error=.1)
    epig = ukf.calculate_epig(designs[0], designs)
