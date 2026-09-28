"""Instrument checks for ``bed.gaussian_design`` (brief §5.3): posterior moments and Gaussian MI
against independent calculations, the noise-free target limit, set utilities, the linear-Gaussian
outcome-independence anchor, and regression against the script the paper results came from."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bed import gaussian_design as gd
from bed.ekf import EKF
from bed.mechanistic import make_divide_problem, make_pkpd_problem

jax.config.update("jax_enable_x64", True)


def _random_problem(seed, n=6, m=3, d=4, dy=2, dt=1):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)); S = A @ A.T / d + 0.5 * np.eye(d)
    J = rng.normal(size=(n, dy, d)); Jt = rng.normal(size=(m, dt, d))
    B = rng.normal(size=(dy, dy)); R = B @ B.T / dy + 0.1 * np.eye(dy)
    return J, Jt, S, R


def test_kalman_update_matches_information_form_for_stacked_designs():
    J, Jt, S, R = _random_problem(0)
    idx = [0, 3, 3]                                   # a repeat
    post = gd.posterior_cov(S, J, R, idx)
    info = gd.information_update(S, gd._stack(J, idx), gd._block_diag(R, 3))
    np.testing.assert_allclose(post, info, rtol=1e-10, atol=1e-12)
    # sequential application equals the joint one (independent noise)
    seq = S.copy()
    for i in idx:
        seq = gd.kalman_update(seq, J[i], R)
    np.testing.assert_allclose(seq, post, rtol=1e-10, atol=1e-12)


def test_epig_scores_equal_block_determinant_mutual_information_and_the_ekf():
    J, Jt, S, R = _random_problem(1, dt=2)
    Rt = 0.3 * np.eye(2)
    sc = gd.epig_scores(J, Jt, S, R, Rt)
    for i in range(J.shape[0]):
        mi = np.mean([gd.gaussian_mi(J[i] @ S @ J[i].T + R, Jt[j] @ S @ Jt[j].T + Rt, Jt[j] @ S @ J[i].T) for j in range(Jt.shape[0])])
        np.testing.assert_allclose(sc[i], mi, rtol=1e-12)
    # against the package's EKF on the PK/PD model at the prior (targets carry R there)
    p = make_pkpd_problem(seed=0, num_times=5)
    X, pool = p.data.x_train, p.data.x_test_pool
    ekf = EKF(p.model, p.prior_mean, p.prior_cov, 0.0, p.measurement_error)
    Jn = np.asarray(p.model.jacobian(p.prior_mean, X)); Jtn = np.asarray(p.model.jacobian(p.prior_mean, pool))
    np.testing.assert_allclose(gd.epig_scores(Jn, Jtn, np.asarray(p.prior_cov), np.asarray(p.measurement_error), np.asarray(p.measurement_error)),
                               np.asarray(ekf.calculate_epig(X, pool)), rtol=1e-9)
    np.testing.assert_allclose(gd.eig_scores(Jn, np.asarray(p.prior_cov), np.asarray(p.measurement_error)), np.asarray(ekf.calculate_eig(X)), rtol=1e-9)


def test_noise_free_target_limit_is_continuous():
    """The ``1e-12`` jitter used for the DIVIDE biomarkers is the ``R_t → 0`` limit, not a modelling
    change: scores at R_t = 0, 1e-12 and 1e-8 agree to 1e-6 relative."""
    p = make_divide_problem(seed=0, num_voxels=1)
    J = np.asarray(p.model.jacobian(p.prior_mean, p.candidates)); Jt = np.asarray(p.model.jacobian(p.prior_mean, p.targets))
    S, R = np.asarray(p.prior_cov), np.asarray(p.measurement_error)
    s0 = gd.epig_scores(J, Jt, S, R, 0.0)
    for eps in (1e-12, 1e-8):
        np.testing.assert_allclose(gd.epig_scores(J, Jt, S, R, eps * np.eye(1)), s0, rtol=1e-6)
    assert np.all(np.isfinite(s0)) and np.all(s0 >= 0)


def test_set_utilities_reduce_to_scores_and_are_monotone_and_order_free():
    J, Jt, S, R = _random_problem(2)
    for i in range(J.shape[0]):
        np.testing.assert_allclose(gd.set_epig([i], J, Jt, S, R), gd.epig_scores(J, Jt, S, R)[i], rtol=1e-12)
        np.testing.assert_allclose(gd.set_eig([i], J, S, R), gd.eig_scores(J, S, R)[i], rtol=1e-12)
        np.testing.assert_allclose(gd.set_variance_reduction([i], J, Jt, S, R), gd.variance_reduction_scores(J, Jt, S, R)[i], rtol=1e-10)
    for crit in ("epig", "epig_joint", "eig", "var"):
        a = gd.set_utility([1, 4], crit, J, Jt, S, R)
        assert gd.set_utility([1, 4, 2], crit, J, Jt, S, R) >= a - 1e-12            # adding an observation never hurts
        np.testing.assert_allclose(gd.set_utility([4, 1], crit, J, Jt, S, R), a, rtol=1e-12)   # order-free
    # joint vector-target information is at least the mean-marginal one for a single target
    Jt1 = Jt[:1]
    np.testing.assert_allclose(gd.set_epig([0, 2], J, Jt1, S, R, joint=True), gd.set_epig([0, 2], J, Jt1, S, R), rtol=1e-12)


def test_complementarity_example_of_the_memo():
    """Memo §4: Y1 = N + e1 (calibration), Y2 = T + N + e2. I(T; Y2) alone is small when Var(N) is
    large; jointly the pair recovers T. The set utility must show the gain; per-candidate sums miss
    it. Formulas: I(T;Y2) = ½ log(1 + 1/(τ² + r2)), I(T;Y1,Y2) = ½ log(1 + 1/(v_N + r2)) with
    v_N = τ² r1/(τ² + r1)."""
    tau2, r1, r2 = 25.0, 0.01, 0.1
    S = np.diag([1.0, tau2])                          # z = (T, N)
    J = np.array([[[0.0, 1.0]], [[1.0, 1.0]]])        # Y1, Y2
    Jt = np.array([[[1.0, 0.0]]])                     # target T
    R = np.array([[1.0]])                             # per-design noise handled by scaling rows below
    # unequal noises: scale the Jacobians so that noise 1 corresponds to r1, r2
    Js = J / np.sqrt(np.array([r1, r2]))[:, None, None]
    i2 = gd.set_epig([1], Js, Jt, S, R); i12 = gd.set_epig([0, 1], Js, Jt, S, R); i1 = gd.set_epig([0], Js, Jt, S, R)
    vN = tau2 * r1 / (tau2 + r1)
    np.testing.assert_allclose(i2, 0.5 * np.log(1 + 1 / (tau2 + r2)), rtol=1e-12)
    np.testing.assert_allclose(i12, 0.5 * np.log(1 + 1 / (vN + r2)), rtol=1e-12)
    assert i1 == pytest.approx(0.0, abs=1e-12)
    assert gd.complementarity(0, 1, "epig", Js, Jt, S, R) > 1.0


def test_linear_gaussian_posterior_covariance_is_outcome_independent():
    """Anchor for Investigation 1 (memo §3): with a linear model, known noise and a linear target, the
    filter's posterior covariance after a sequence of designs does not depend on the observed values,
    so an adaptive rule cannot beat the best fixed sequence of the same length."""
    rng = np.random.default_rng(5); d, n = 3, 5
    A = rng.normal(size=(n, 1, d)); S0 = np.eye(d); R = np.array([[0.2]]); z_true = rng.normal(size=d)
    covs = []
    for rep in range(3):
        S = S0.copy(); m = np.zeros(d)
        for i in (2, 0, 2):
            y = A[i][0] @ z_true + np.sqrt(R[0, 0]) * rng.normal()
            Sx = A[i] @ S @ A[i].T + R; K = S @ A[i].T @ np.linalg.inv(Sx)
            m = m + (K @ (y - A[i] @ m)).ravel(); S = S - K @ A[i] @ S
        covs.append(S)
    np.testing.assert_allclose(covs[0], covs[1], atol=1e-12); np.testing.assert_allclose(covs[0], covs[2], atol=1e-12)
    np.testing.assert_allclose(covs[0], gd.posterior_cov(S0, A, R, [2, 0, 2]), atol=1e-12)


def test_greedy_reproduces_the_paper_script_on_divide():
    """Regression: the greedy selection of ``experiments/linearized_greedy.py`` (source of the
    paper's fixed protocols) is reproduced by the module on the DIVIDE problem."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import linearized_greedy as lg
    p = make_divide_problem(seed=0, num_voxels=1)
    J = np.asarray(p.model.jacobian(p.prior_mean, p.candidates)); Jt = np.asarray(p.model.jacobian(p.prior_mean, p.targets))
    S, R = np.asarray(p.prior_cov), np.asarray(p.measurement_error); warm = [3, 40, 77]
    old_e = lg.greedy(J, Jt, S, R, 8, warm, "epig", np.random.default_rng(0), Rt=1e-12 * np.eye(1), replace=True, per_target=True)[1]
    old_i = lg.greedy(J, Jt, S, R, 8, warm, "eig", np.random.default_rng(0), Rt=1e-12 * np.eye(1), replace=True, per_target=True)[1]
    b = gd.LinearBeliefs([J], [Jt], S, R, Rt=0.0)
    assert gd.greedy(b, "epig", 8, warm)[0] == old_e
    assert gd.greedy(b, "eig", 8, warm)[0] == old_i


def test_exchange_and_exhaustive_search_improve_on_greedy():
    J, Jt, S, R = _random_problem(7, n=7, m=2, d=5, dy=1, dt=1)
    b = gd.LinearBeliefs([J], [Jt], S, R)
    g, _ = gd.greedy(b, "epig", 3)
    ex, ex_val, traces = gd.exchange_search(b, "epig", g, n_random_starts=3, rng=np.random.default_rng(1))
    best, best_val, top = gd.exhaustive_best(b, "epig", 3)
    g_val = b.utility(g, "epig")
    assert ex_val >= g_val - 1e-12 and best_val >= ex_val - 1e-12
    assert all(np.all(np.diff(t) >= -1e-12) for t in traces)          # sweeps never decrease the utility
    assert sorted(top, reverse=True) == top


def test_prior_averaged_beliefs_reduce_to_the_single_draw_case():
    p = make_divide_problem(seed=0, num_voxels=1)
    J = np.asarray(p.model.jacobian(p.prior_mean, p.candidates)); Jt = np.asarray(p.model.jacobian(p.prior_mean, p.targets))
    S, R = np.asarray(p.prior_cov), np.asarray(p.measurement_error)
    one = gd.LinearBeliefs([J], [Jt], S, R); three = gd.LinearBeliefs([J, J, J], [Jt, Jt, Jt], S, R)
    np.testing.assert_allclose(three.scores("epig", [S] * 3), one.scores("epig", [S]), rtol=1e-12)
    np.testing.assert_allclose(three.utility([1, 5, 5], "epig"), one.utility([1, 5, 5], "epig"), rtol=1e-12)
    np.testing.assert_allclose(three.target_variance([1, 5], per_target=True), one.target_variance([1, 5], per_target=True), rtol=1e-12)
