"""The exact small-instance instrument: policy orderings that must hold by construction, the
linear-Gaussian anchor (no value of adaptation), and outcome-integration convergence."""
import numpy as np

from bed import toy


def test_policy_values_are_ordered_by_construction():
    m = toy.sigmoid_location_model(G=121, ny=61)
    for kind in ("loss", "entropy"):
        r = m.evaluate_all(2, kind)
        tol = 1e-9
        assert r["adaptive_optimal"] <= r["fixed_best"] + tol            # the optimal adaptive policy contains every fixed sequence
        assert r["adaptive_optimal"] <= r["adaptive_greedy"] + tol       # and every adaptive greedy policy
        assert r["fixed_best"] <= r["fixed_greedy"] + tol                # exhaustive beats greedy planning
        assert r["fixed_best"] <= r["prior_utility"] + tol               # observing helps


def test_linear_gaussian_grid_has_no_value_of_adaptation():
    m = toy.linear_model(G=241, ny=121)
    r = m.evaluate_all(2, "loss")
    # exact: posterior variance after designs x1, x2 is (1/s² + (x1² + x2²)/σ²)^{-1}; the best pair is the two largest |x|
    s2, sig2 = 1.0, 0.3 ** 2; xs = np.linspace(-2, 2, 5)
    exact = 1.0 / (1 / s2 + (2 * 4.0) / sig2)
    assert abs(r["fixed_best"] - exact) < 2e-3 * exact
    assert abs(r["value_of_adaptation"]) < 2e-3 * exact and abs(r["value_of_lookahead"]) < 2e-3 * exact


def test_sigmoid_model_has_positive_value_of_adaptation_that_converges_with_the_outcome_grid():
    make = lambda ny: toy.sigmoid_location_model(G=161, ny=ny)
    conv = toy.convergence_check(make, 2, "loss", ny_values=(61, 121))
    v61, v121 = conv[61]["value_of_adaptation"], conv[121]["value_of_adaptation"]
    assert v121 > 0.0
    assert abs(v61 - v121) < 0.25 * v121, (v61, v121)
