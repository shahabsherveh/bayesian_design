"""Correctness tests for the sigma-point closed-form EPIG of the UKF."""
import jax
import jax.numpy as jnp
from flax import nnx

from bed.ekf import EKF
from bed.models import DenseNN, NeuralNetworkRegressor
from bed.ukf import UKF

jax.config.update("jax_enable_x64", True)


class QuadraticModel:
    """y = z0^2 + x z1, nonlinear in the state; follows the (N, 1, 1, d) convention."""

    def __call__(self, state, x):
        return (state[0] ** 2 + x.reshape(-1) * state[1]).reshape(-1, 1, 1, 1)


def _designs(key, num, dim):
    return jax.random.normal(key, (num, 1, 1, dim))


def _filters(hidden_dims, prior_var, key=0):
    flax_model = DenseNN(input_dim=2, hidden_dims=hidden_dims, output_dim=1, rngs=nnx.Rngs(key))
    model = NeuralNetworkRegressor(flax_model)
    n = flax_model.weight_size
    z0 = flax_model.state_to_weights(nnx.state(flax_model))
    cov = prior_var * jnp.eye(n)
    R = 0.05 * jnp.eye(1)
    return EKF(model, z0, cov, 0.0, R), UKF(model, z0, cov, 0.0, R)


def test_epig_equals_ekf_closed_form_for_linear_model():
    # A network without hidden layers is linear in its weights, where the sigma-point
    # moments are exact and C_j = J'_j Sigma J_x^T. The two closed forms must coincide.
    ekf, ukf = _filters(hidden_dims=[], prior_var=0.5)
    x = _designs(jax.random.PRNGKey(1), 5, 2)
    pool = _designs(jax.random.PRNGKey(2), 7, 2)
    epig_ekf = ekf.calculate_epig(x, pool)
    epig_ukf = ukf.calculate_epig(x, pool)
    assert epig_ukf.shape == (5,)
    assert jnp.allclose(epig_ukf, epig_ekf, rtol=1e-10, atol=1e-12)
    assert jnp.all(epig_ukf > 0)


def test_epig_matches_gaussian_mutual_information_of_sigma_point_joint():
    # Independent computation: assemble the 2x2 joint covariance of (y, y'_j) from the
    # same sigma points and evaluate the Gaussian mutual information through the
    # joint determinant, 1/2 log(det S_x det S'_j / det Joint).
    ukf = UKF(
        model=QuadraticModel(),
        state_prev=jnp.array([0.7, -0.4]),
        state_cov_prev=jnp.diag(jnp.array([0.3, 0.6])),
        state_innovation=0.0,
        measurement_error=jnp.array([[0.1]]),
        alpha=1.0,
    )
    x = jnp.array([[[[1.5]]], [[[-0.3]]]])
    pool = jnp.array([[[[0.2]]], [[[2.0]]], [[[-1.1]]]])
    points, wm, wc = ukf.sigma_points
    wm, wc = wm.reshape(-1), wc.reshape(-1)
    R = float(ukf.measurement_error[0, 0])

    def fvals(design):
        return jnp.array([float(ukf.model(p, design).reshape(())) for p in points])

    expected = []
    for b in range(x.shape[0]):
        fx = fvals(x[b:b + 1])
        mx = jnp.sum(wm * fx)
        mi = []
        for j in range(pool.shape[0]):
            fj = fvals(pool[j:j + 1])
            mj = jnp.sum(wm * fj)
            sxx = jnp.sum(wc * (fx - mx) ** 2) + R
            sjj = jnp.sum(wc * (fj - mj) ** 2) + R
            sxj = jnp.sum(wc * (fx - mx) * (fj - mj))
            joint = jnp.array([[sxx, sxj], [sxj, sjj]])
            mi.append(0.5 * (jnp.log(sxx) + jnp.log(sjj) - jnp.log(jnp.linalg.det(joint))))
        expected.append(jnp.mean(jnp.array(mi)))
    expected = jnp.array(expected)
    assert jnp.allclose(ukf.calculate_epig(x, pool), expected, rtol=1e-10)


def test_epig_is_nonnegative_and_vanishes_for_uninformative_observations():
    # With alpha = 1 all sigma weights are positive, so every covariance estimate is
    # PSD and the mutual information cannot be negative. As the observation noise
    # grows the observation carries no information and EPIG tends to zero.
    _, ukf = _filters(hidden_dims=[4], prior_var=0.5)
    x = _designs(jax.random.PRNGKey(3), 4, 2)
    pool = _designs(jax.random.PRNGKey(4), 6, 2)
    ukf.alpha = 1.0
    ukf.state_prior = ukf.state_prior  # regenerate sigma points with the new alpha
    epig = ukf.calculate_epig(x, pool)
    assert jnp.all(epig >= 0)
    ukf.measurement_error = 1e8 * jnp.eye(1)
    assert jnp.all(ukf.calculate_epig(x, pool) < 1e-6)


def test_epig_batch_of_candidates_equals_single_candidate_calls():
    _, ukf = _filters(hidden_dims=[4], prior_var=0.5)
    x = _designs(jax.random.PRNGKey(5), 4, 2)
    pool = _designs(jax.random.PRNGKey(6), 6, 2)
    batched = ukf.calculate_epig(x, pool)
    single = jnp.concatenate([ukf.calculate_epig(x[b], pool) for b in range(x.shape[0])])
    assert batched.shape == (4,)
    assert jnp.allclose(batched, single)


def test_epig_approaches_ekf_for_tight_prior_on_nonlinear_model():
    # On a GELU network the UKF and EKF criteria differ by second-order terms in the
    # prior covariance; as the prior tightens they must agree.
    x = _designs(jax.random.PRNGKey(7), 5, 2)
    pool = _designs(jax.random.PRNGKey(8), 7, 2)
    rel_err = []
    for prior_var in (1e-1, 1e-3):
        ekf, ukf = _filters(hidden_dims=[4], prior_var=prior_var)
        e, u = ekf.calculate_epig(x, pool), ukf.calculate_epig(x, pool)
        rel_err.append(float(jnp.max(jnp.abs(u - e) / jnp.abs(e))))
    assert rel_err[1] < rel_err[0]
    assert rel_err[1] < 1e-2


# ---------------------------------------------------------------------------
# Oracles independent of bed.ukf: their own sigma points (Cholesky columns), their own
# weights, explicit loops and explicit joint covariances. They share only the forward pass.
# ---------------------------------------------------------------------------


def _oracle_epig(f, z, sigma, R, X, pool, alpha, beta=2.0, kappa=0.0):
    import numpy as np

    z = np.asarray(z).ravel()
    n = z.size
    lam = alpha**2 * (n + kappa) - n
    L = np.linalg.cholesky((n + lam) * np.asarray(sigma))
    points = np.vstack([z, z + L.T, z - L.T])
    wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + 1 - alpha**2 + beta

    def deviations(x):
        vals = np.stack([np.asarray(f(p, x)).reshape(-1) for p in points])
        return vals - wm @ vals

    out = []
    for x in X:
        dx = deviations(x)
        s_x = (dx * wc[:, None]).T @ dx + R
        mi = []
        for xp in pool:
            dp = deviations(xp)
            s_p = (dp * wc[:, None]).T @ dp + R
            c = (dp * wc[:, None]).T @ dx
            joint = np.block([[s_x, c.T], [c, s_p]])
            mi.append(0.5 * (np.log(np.linalg.det(s_x)) + np.log(np.linalg.det(s_p)) - np.log(np.linalg.det(joint))))
        out.append(np.mean(mi))
    return np.asarray(out)


def _dense(output_dim, seed=1):
    flax_model = DenseNN(input_dim=3, hidden_dims=[5], output_dim=output_dim, rngs=nnx.Rngs(seed))
    model = NeuralNetworkRegressor(flax_model)
    z = flax_model.state_to_weights(nnx.state(flax_model))
    return flax_model, model, z


def _full_rank_prior(n, seed=0):
    a = jax.random.normal(jax.random.PRNGKey(seed), (n, n))
    return 0.3 * a @ a.T / n + 0.05 * jnp.eye(n)


def test_epig_two_outputs_with_full_prior_matches_independent_oracle():
    # d_y = 2 makes C_j a genuinely asymmetric matrix, so this fails if C and C^T are swapped
    # in the Schur complement; a non-diagonal prior fails if rows and columns of the Cholesky
    # factor are confused; alpha < 1 fails if the (1 - alpha^2) term of w_0^(c) is missing.
    flax_model, model, z = _dense(output_dim=2)
    sigma = _full_rank_prior(flax_model.weight_size)
    R = jnp.array([[0.05, 0.01], [0.01, 0.08]])
    X = jax.random.normal(jax.random.PRNGKey(3), (4, 3))
    pool = jax.random.normal(jax.random.PRNGKey(4), (5, 3))

    def f(p, x):
        return model(jnp.asarray(p), jnp.asarray(x).reshape(1, 1, 1, -1))

    for alpha in (1.0, 0.3):
        ukf = UKF(model, z, sigma, 0.0, R, alpha=alpha)
        got = ukf.calculate_epig(X[:, None, None, :], pool[:, None, None, :])
        want = _oracle_epig(f, z, sigma, R, X, pool, alpha)
        assert jnp.allclose(got, want, rtol=1e-10, atol=1e-12), (alpha, got, want)


def test_epig_two_outputs_linear_model_matches_jacobian_closed_form():
    # Linear two-output network: the exact Gaussian answer follows from the Jacobians,
    # C_j = J'_j Sigma J_x^T, S = J Sigma J^T + R. Independent of any filter class.
    flax_model = DenseNN(input_dim=3, hidden_dims=[], output_dim=2, rngs=nnx.Rngs(2))
    model = NeuralNetworkRegressor(flax_model)
    z = flax_model.state_to_weights(nnx.state(flax_model))
    sigma = _full_rank_prior(flax_model.weight_size, seed=5)
    R = jnp.array([[0.05, 0.01], [0.01, 0.08]])
    X = jax.random.normal(jax.random.PRNGKey(6), (3, 1, 1, 3))
    pool = jax.random.normal(jax.random.PRNGKey(7), (4, 1, 1, 3))

    def jac(x):
        return jax.jacobian(lambda zz: model(zz, x[None]).reshape(-1))(z.ravel())

    expected = []
    for x in X:
        jx = jac(x)
        s_x = jx @ sigma @ jx.T + R
        mi = []
        for xp in pool:
            jp = jac(xp)
            s_p = jp @ sigma @ jp.T + R
            c = jp @ sigma @ jx.T
            mi.append(0.5 * (jnp.linalg.slogdet(s_p)[1] - jnp.linalg.slogdet(s_p - c @ jnp.linalg.inv(s_x) @ c.T)[1]))
        expected.append(jnp.mean(jnp.array(mi)))
    ukf = UKF(model, z, sigma, 0.0, R, alpha=0.5)
    assert jnp.allclose(ukf.calculate_epig(X, pool), jnp.array(expected), rtol=1e-10)


class _Quadratic1D:
    """y = x z^2 with a scalar state; the (N, 1, 1, d) convention."""

    def __call__(self, state, x):
        return (x.reshape(-1) * state[0] ** 2).reshape(-1, 1, 1, 1)


def test_epig_scalar_quadratic_state_matches_analytic_gaussian_moments():
    # For z ~ N(m, s^2) in one dimension the scaled unscented transform with beta = 2 matches
    # the fourth moment at every alpha, provided w_0^(c) carries the (1 - alpha^2) term. Hence
    # Var(x z^2) = x^2 v and Cov(x z^2, x' z^2) = x x' v with v = 4 m^2 s^2 + 2 s^4, and the
    # criterion has a closed form. This is the regression test for the central weight.
    m, s2, R = 0.8, 0.5, 0.05
    X = jnp.array([1.5, -0.7, 0.3, 2.0])
    pool = jnp.array([0.5, -1.2, 1.0])
    v = 4 * m**2 * s2 + 2 * s2**2

    def analytic(x):
        s_x = x**2 * v + R
        terms = [
            0.5 * jnp.log(s_x * (xp**2 * v + R) / (s_x * (xp**2 * v + R) - (x * xp * v) ** 2))
            for xp in pool
        ]
        return jnp.mean(jnp.array(terms))

    want = jnp.array([analytic(x) for x in X])
    for alpha in (1.0, 0.3, 0.05):
        ukf = UKF(_Quadratic1D(), jnp.array([m]), jnp.array([[s2]]), 0.0, jnp.array([[R]]), alpha=alpha)
        got = ukf.calculate_epig(X.reshape(-1, 1, 1, 1), pool.reshape(-1, 1, 1, 1))
        assert jnp.allclose(got, want, rtol=1e-9), (alpha, got, want)


def test_eig_equals_ekf_closed_form_for_linear_model():
    ekf, ukf = _filters(hidden_dims=[], prior_var=0.5)
    x = _designs(jax.random.PRNGKey(9), 5, 2)
    assert jnp.allclose(ukf.calculate_eig(x), ekf.calculate_eig(x), rtol=1e-10)
