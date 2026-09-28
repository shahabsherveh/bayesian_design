"""Ground-truth moments and criterion values for checking the Gaussian filters.

The filters approximate the moments of ``f(z, x)`` under the Gaussian belief
``z ~ N(mean, cov)``. This module estimates those moments by Monte Carlo from the
same belief, extracts the corresponding moments from an EKF or UKF, and evaluates
the closed-form EPIG from any set of moments, so that moment-propagation error can
be measured separately from posterior-approximation error.

Shapes follow the package convention: designs ``(B, 1, 1, d)``, predictive means
``(B, d_y)``, predictive covariances ``(B, d_y, d_y)`` and cross-covariances
``Cov(y'_j, y_b)`` of shape ``(B, M, d_y, d_y)``.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bed.ekf import EKF
from bed.ukf import UKF


@dataclass
class Moments:
    """Predictive moments at candidates ``x`` and pool ``x_prime``.

    ``cov`` and ``cov_prime`` include the measurement noise when a measurement
    error was supplied; ``cross`` never does (independent noise).
    """

    mean: jnp.ndarray            # (B, d_y)
    cov: jnp.ndarray             # (B, d_y, d_y)
    mean_prime: jnp.ndarray      # (M, d_y) or None
    cov_prime: jnp.ndarray       # (M, d_y, d_y) or None
    cross: jnp.ndarray           # (B, M, d_y, d_y) or None


def _outputs(model, samples, x):
    """Evaluate ``f(z, x)`` for every sampled ``z``; returns ``(N, B, d_y)``."""
    values = jax.vmap(lambda z: model(z.reshape(-1, 1), x))(samples)
    return values.reshape(values.shape[0], values.shape[1], -1)


def mc_moments(model, mean, cov, x, x_prime=None, measurement_error=None,
               num_samples=4000, key=jax.random.PRNGKey(0)):
    """Monte Carlo moments of ``f(z, x)`` and ``f(z, x')`` under ``z ~ N(mean, cov)``.

    The same parameter samples are used at ``x`` and ``x_prime``, so the
    cross-covariance is estimated jointly. Uses the unbiased (N - 1) normalisation.
    """
    mean = jnp.asarray(mean).reshape(-1)
    cov = jnp.asarray(cov)
    if jnp.ndim(x) == 3:
        x = x[None, ...]
    root = jnp.linalg.cholesky((cov + cov.T) / 2)
    samples = mean[None, :] + jax.random.normal(key, (num_samples, mean.size)) @ root.T
    fx = _outputs(model, samples, x)
    mean_x = fx.mean(axis=0)
    dev_x = fx - mean_x[None]
    cov_x = jnp.einsum("nba,nbc->bac", dev_x, dev_x) / (num_samples - 1)
    if measurement_error is not None:
        cov_x = cov_x + jnp.asarray(measurement_error)
    if x_prime is None:
        return Moments(mean_x, cov_x, None, None, None)
    if jnp.ndim(x_prime) == 3:
        x_prime = x_prime[None, ...]
    fp = _outputs(model, samples, x_prime)
    mean_p = fp.mean(axis=0)
    dev_p = fp - mean_p[None]
    cov_p = jnp.einsum("nma,nmc->mac", dev_p, dev_p) / (num_samples - 1)
    if measurement_error is not None:
        cov_p = cov_p + jnp.asarray(measurement_error)
    cross = jnp.einsum("nma,nbc->bmac", dev_p, dev_x) / (num_samples - 1)
    return Moments(mean_x, cov_x, mean_p, cov_p, cross)


def filter_moments(filt, x, x_prime=None):
    """The moments an EKF or UKF assigns to ``f(z, x)`` and ``f(z, x')`` under its belief.

    Covariances include the filter's measurement error, as in its criteria.
    """
    if jnp.ndim(x) == 3:
        x = x[None, ...]
    if x_prime is not None and jnp.ndim(x_prime) == 3:
        x_prime = x_prime[None, ...]
    if isinstance(filt, UKF):
        w = filt.sigma_points[2].reshape(-1)
        mean_x, dev_x = filt._sigma_deviations(x)
        cov_x = jnp.einsum("k,bka,bkc->bac", w, dev_x, dev_x) + filt.measurement_error
        if x_prime is None:
            return Moments(mean_x, cov_x, None, None, None)
        mean_p, dev_p = filt._sigma_deviations(x_prime)
        cov_p = jnp.einsum("k,mka,mkc->mac", w, dev_p, dev_p) + filt.measurement_error
        cross = jnp.einsum("k,mka,bkc->bmac", w, dev_p, dev_x)
        return Moments(mean_x, cov_x, mean_p, cov_p, cross)
    if isinstance(filt, EKF):
        z, sigma = filt.state_prior
        z = jnp.asarray(z).reshape(-1, 1)
        jac_x = filt.model.jacobian(z, x)                          # (B, d_y, n)
        mean_x = filt.model(z, x).reshape(x.shape[0], -1)
        cov_x = jac_x @ sigma @ jnp.matrix_transpose(jac_x) + filt.measurement_error
        if x_prime is None:
            return Moments(mean_x, cov_x, None, None, None)
        jac_p = filt.model.jacobian(z, x_prime)                    # (M, d_y, n)
        mean_p = filt.model(z, x_prime).reshape(x_prime.shape[0], -1)
        cov_p = jac_p @ sigma @ jnp.matrix_transpose(jac_p) + filt.measurement_error
        cross = jnp.einsum("mai,ij,bcj->bmac", jac_p, sigma, jac_x)
        return Moments(mean_x, cov_x, mean_p, cov_p, cross)
    raise TypeError(f"Unsupported filter type {type(filt).__name__}")


def gaussian_epig(cov_x, cov_prime, cross):
    """Closed-form EPIG from joint Gaussian moments, in nats, shape ``(B,)``.

    ``cov_x`` (B, d_y, d_y) and ``cov_prime`` (M, d_y, d_y) must include the
    measurement noise; ``cross`` is ``Cov(y'_j, y_b)`` of shape (B, M, d_y, d_y).
    """
    schur = cov_prime[None] - cross @ jnp.linalg.inv(cov_x)[:, None] @ jnp.matrix_transpose(cross)
    logdet_prime = jnp.linalg.slogdet(cov_prime).logabsdet
    logdet_schur = jnp.linalg.slogdet(schur).logabsdet
    return 0.5 * (logdet_prime[None, :] - logdet_schur).mean(axis=1)


def gaussian_eig(cov_x, measurement_error):
    """Closed-form EIG under linearisation, ½ log det S_x − ½ log det R, shape ``(B,)``."""
    return 0.5 * (jnp.linalg.slogdet(cov_x).logabsdet - jnp.linalg.slogdet(jnp.asarray(measurement_error))[1])


def moment_errors(approx: Moments, reference: Moments):
    """Root-mean-square errors of ``approx`` against ``reference`` moments.

    Returns a dict with keys ``mean``, ``var`` (diagonal of the predictive
    covariance at the candidates), ``cov_prime`` (diagonal at the pool) and
    ``cross`` (all cross-covariances), each a scalar.
    """
    def rms(a, b):
        return float(jnp.sqrt(jnp.mean((jnp.asarray(a) - jnp.asarray(b)) ** 2)))
    out = {"mean": rms(approx.mean, reference.mean),
           "var": rms(jnp.diagonal(approx.cov, axis1=-2, axis2=-1), jnp.diagonal(reference.cov, axis1=-2, axis2=-1))}
    if approx.cross is not None and reference.cross is not None:
        out["cov_prime"] = rms(jnp.diagonal(approx.cov_prime, axis1=-2, axis2=-1),
                               jnp.diagonal(reference.cov_prime, axis1=-2, axis2=-1))
        out["cross"] = rms(approx.cross, reference.cross)
    return out
