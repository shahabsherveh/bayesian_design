"""Empirical Bayes initialisation of the filter belief.

The filters start from a Gaussian belief ``N(mu_0, Sigma_0)`` over the latent parameters and an
observation covariance ``R``. Both are usually set by hand. This module chooses their scales from
the warm-start observations by maximising the marginal likelihood of the model linearised at the
MAP (type-II maximum likelihood; the Laplace/Gauss--Newton evidence of Immer et al., 2021 and
Daxberger et al., 2021), and returns the Laplace belief at the MAP as the state to start filtering
from. For a linear model everything is exact.

Hyperparameters are two positive scales, ``Sigma_0 = tau * S`` and ``R = s * R_0``, where ``S`` and
``R_0`` are the shapes the user configured. With a handful of warm-start points the evidence
supports no more than that; ``s`` is fixed unless ``fit_noise=True``.

Works for any :class:`bed._models.Model` exposing ``__call__(z, x)`` and ``jacobian(z, x)`` with the
package's batch conventions (``x`` of shape ``(n, ...)``, outputs ``(n, ..., d_y)``, Jacobian
``(n, d_y, d)`` after flattening), so for the linear model and for the Flax networks alike.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

LOG_SCALE_BOUNDS = (np.log(1e-6), np.log(1e6))


def _flatten(model, z, x):
    """Model output and Jacobian at ``z`` for the designs ``x``, as ``(N,)`` and ``(N, d)`` with
    ``N = n * d_y``."""
    z = jnp.asarray(z).reshape(-1, 1)
    n = int(jnp.asarray(x).shape[0])
    f = jnp.asarray(model(z, x)).reshape(n, -1)
    d_y = f.shape[1]
    jac = jnp.asarray(model.jacobian(z, x)).reshape(n, d_y, z.size)
    return f.reshape(-1), jac.reshape(n * d_y, z.size), d_y


def _block_noise(noise_cov, n, d_y):
    """Cholesky factor of the full ``(n d_y, n d_y)`` observation covariance ``I_n (x) R``."""
    R = jnp.atleast_2d(jnp.asarray(noise_cov)).reshape(d_y, d_y)
    return jnp.kron(jnp.eye(n), R)


def log_marginal_likelihood(model, z_lin, x, y, prior_mean, prior_cov, noise_cov):
    """Log marginal likelihood of ``y`` under the model linearised at ``z_lin``:
    ``y ~ N(f(z_lin) + J (mu_0 - z_lin), J Sigma_0 J^T + I (x) R)``. Exact for a linear model, and
    equal to the Gauss--Newton Laplace evidence when ``z_lin`` is the MAP. Evaluated in the
    ``d``-dimensional information form, so the cost does not grow with the cube of ``n``."""
    z_lin = jnp.asarray(z_lin).reshape(-1); mu0 = jnp.asarray(prior_mean).reshape(-1)
    f, J, d_y = _flatten(model, z_lin, x)
    n = f.size // d_y
    Rf = _block_noise(noise_cov, n, d_y)
    r = jnp.asarray(y).reshape(-1) - (f + J @ (mu0 - z_lin))
    S0 = jnp.asarray(prior_cov)
    LR = jnp.linalg.cholesky(Rf); L0 = jnp.linalg.cholesky(S0)
    Rinv_r = jax.scipy.linalg.cho_solve((LR, True), r)
    Rinv_J = jax.scipy.linalg.cho_solve((LR, True), J)
    A = jnp.linalg.inv(S0) + J.T @ Rinv_J                      # posterior precision
    LA = jnp.linalg.cholesky(A)
    b = J.T @ Rinv_r
    quad = r @ Rinv_r - b @ jax.scipy.linalg.cho_solve((LA, True), b)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(LR))) + 2 * jnp.sum(jnp.log(jnp.diag(L0))) + 2 * jnp.sum(jnp.log(jnp.diag(LA)))
    return -0.5 * (quad + logdet + r.size * jnp.log(2 * jnp.pi))


def map_estimate(model, x, y, prior_mean, prior_cov, noise_cov, z_init=None, restarts=4, seed=0):
    """MAP of the latent parameters under the Gaussian prior, by Levenberg--Marquardt on the
    whitened residuals, from the prior mean, ``z_init`` and ``restarts`` draws from the prior; the
    lowest cost wins. The restarts matter: for models such as ``a tanh(b x)`` the prior mean zero
    is an exact stationary point and a single start never leaves it. Returns the MAP and the
    Laplace (Gauss--Newton) covariance at it."""
    mu0 = np.asarray(prior_mean, dtype=float).reshape(-1); d = mu0.size
    L0 = np.linalg.cholesky(np.asarray(prior_cov, dtype=float)); L0inv = np.linalg.inv(L0)
    yv = np.asarray(y, dtype=float).reshape(-1)
    n = int(jnp.asarray(x).shape[0]); d_y = yv.size // n
    LR = np.linalg.cholesky(np.asarray(_block_noise(noise_cov, n, d_y))); LRinv = np.linalg.inv(LR)

    def resid(z):
        f, _, _ = _flatten(model, z, x)
        return np.concatenate([LRinv @ (np.asarray(f) - yv), L0inv @ (z - mu0)])

    def jacr(z):
        _, J, _ = _flatten(model, z, x)
        return np.concatenate([LRinv @ np.asarray(J), L0inv])

    starts = [mu0] + ([np.asarray(z_init, dtype=float).reshape(-1)] if z_init is not None else [])
    rng = np.random.default_rng(seed)
    starts += [mu0 + L0 @ rng.normal(size=d) for _ in range(int(restarts))]
    best = None
    for z0 in starts:
        r = least_squares(resid, z0, jac=jacr, method="lm", max_nfev=5000)
        if best is None or r.cost < best.cost:
            best = r
    Jw = jacr(best.x)
    cov = np.linalg.inv(Jw.T @ Jw)
    return best.x, cov


@dataclass
class EmpiricalBayesResult:
    mean: jnp.ndarray            # (d, 1) MAP under the fitted prior
    cov: jnp.ndarray             # (d, d) Laplace covariance at the MAP
    prior_cov: jnp.ndarray       # (d, d) fitted prior covariance tau * S
    noise_cov: jnp.ndarray       # (d_y, d_y) fitted (or kept) observation covariance s * R_0
    prior_scale: float
    noise_scale: float
    log_marginal_likelihood: float
    history: list = field(default_factory=list)   # (prior_scale, noise_scale, lml) per iteration


def empirical_bayes_init(model, x, y, prior_mean, prior_cov, noise_cov, fit_noise=False, iterations=3,
                         prior_scale_init=1.0, noise_scale_init=1.0):
    """Choose the prior scale (and optionally the noise scale) by type-II maximum likelihood on the
    observations ``(x, y)``, and return the Laplace belief at the MAP to start the filter from.

    ``prior_cov`` and ``noise_cov`` are the *shapes* ``S`` and ``R_0``; the fitted covariances are
    ``tau * S`` and ``s * R_0``. Alternates ``iterations`` times between the MAP under the current
    hyperparameters and the hyperparameters maximising the evidence of the model linearised at
    that MAP. One iteration is exact for a linear model.
    """
    S = jnp.asarray(prior_cov, dtype=float); R0 = jnp.atleast_2d(jnp.asarray(noise_cov, dtype=float))
    mu0 = jnp.asarray(prior_mean, dtype=float).reshape(-1)
    tau, s = float(prior_scale_init), float(noise_scale_init)
    history = []
    z_map = np.asarray(mu0)
    for _ in range(int(iterations)):
        z_map, _ = map_estimate(model, x, y, mu0, tau * S, s * R0, z_init=z_map)
        z_lin = jnp.asarray(z_map)

        def objective(theta):
            t = jnp.exp(theta[0]); sc = jnp.exp(theta[1]) if fit_noise else s
            return -log_marginal_likelihood(model, z_lin, x, y, mu0, t * S, sc * R0)

        obj = jax.jit(jax.value_and_grad(objective))
        theta0 = np.array([np.log(tau), np.log(s)] if fit_noise else [np.log(tau), 0.0])
        bounds = [LOG_SCALE_BOUNDS] * (2 if fit_noise else 1)

        def fun(th):
            v, g = obj(jnp.asarray(np.concatenate([th, [0.0]]) if not fit_noise else th))
            g = np.asarray(g)[: len(th)]
            return float(v), g
        res = minimize(fun, theta0[: len(bounds)], jac=True, method="L-BFGS-B", bounds=bounds)
        tau = float(np.exp(res.x[0]))
        if fit_noise:
            s = float(np.exp(res.x[1]))
        history.append((tau, s, -float(res.fun)))
    z_map, cov = map_estimate(model, x, y, mu0, tau * S, s * R0, z_init=z_map)
    lml = float(log_marginal_likelihood(model, z_map, x, y, mu0, tau * S, s * R0))
    return EmpiricalBayesResult(mean=jnp.asarray(z_map).reshape(-1, 1), cov=jnp.asarray((cov + cov.T) / 2),
                                prior_cov=tau * S, noise_cov=s * R0, prior_scale=tau, noise_scale=s,
                                log_marginal_likelihood=lml, history=history)
