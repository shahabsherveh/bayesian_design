"""The independent review's counterexample: rescaling the belief covariance is not a uniform
rescaling of the variance-reduction utilities and can reverse a decision; rescaling the
utilities cannot (Proposition 1, corollary (ii))."""
import numpy as np

from bed import gaussian_design as gd


def test_covariance_rescaling_can_reverse_the_choice_but_utility_rescaling_cannot():
    S = np.eye(2); J = np.array([[[1.0, 0.0]], [[2.0, 2.0]]]); Jt = np.array([[[1.0, 0.0]]]); R = np.array([[1.0]])
    u_half = gd.variance_reduction_scores(J, Jt, 0.5 * S, R, 0.0); u_two = gd.variance_reduction_scores(J, Jt, 2.0 * S, R, 0.0)
    np.testing.assert_allclose(u_half, [1 / 6, 1 / 5]); np.testing.assert_allclose(u_two, [4 / 3, 16 / 17])
    assert int(np.argmax(u_half)) == 1 and int(np.argmax(u_two)) == 0
    u = gd.variance_reduction_scores(J, Jt, S, R, 0.0)
    for lam in (0.1, 3.0):
        assert int(np.argmax(lam * u)) == int(np.argmax(u))
        e = lam * u - u; assert abs((e.max() - e.min()) - abs(lam - 1) * (u.max() - u.min())) < 1e-12
