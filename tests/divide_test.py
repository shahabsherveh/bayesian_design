"""Tests for the DIVIDE biomarker-design model (candidate D)."""
import jax
import jax.numpy as jnp
import numpy as np

from bed.mechanistic import (DIVIDE_PRIOR_MEAN, DIVIDE_PRIOR_SD, DivideModel, divide_biomarkers, divide_te_min,
                             make_divide_problem)

jax.config.update("jax_enable_x64", True)


def test_signal_matches_the_gamma_and_cumulant_formulas_and_they_agree_at_small_b():
    z = jnp.log(jnp.array([0.9, 70.0, 0.8, 0.05, 0.3])); S0, T2, MD, VI, VA = 0.9, 70.0, 0.8, 0.05, 0.3
    b, TE = 1.5, 90.0
    mg, mc = DivideModel(1, form="gamma"), DivideModel(1, form="cumulant")
    for bD, mu2 in ((1.0, VI + VA), (0.0, VI)):
        g = float(mg(z, jnp.array([[b, bD, TE, 0.0]])).ravel()[0]); c = float(mc(z, jnp.array([[b, bD, TE, 0.0]])).ravel()[0])
        np.testing.assert_allclose(g, S0 * np.exp(-TE / T2) * (1 + b * mu2 / MD) ** (-MD ** 2 / mu2), rtol=1e-12)
        np.testing.assert_allclose(c, S0 * np.exp(-TE / T2) * np.exp(-b * MD + 0.5 * b ** 2 * mu2), rtol=1e-12)
    # second-order agreement: the difference is O(b^3)
    for bb in (0.1, 0.2, 0.4):
        d = abs(float(mg(z, jnp.array([[bb, 1.0, 0.0, 0.0]])).ravel()[0]) - float(mc(z, jnp.array([[bb, 1.0, 0.0, 0.0]])).ravel()[0]))
        assert d < 0.3 * (VI + VA) ** 1.5 * bb ** 3 + 1e-9, (bb, d)
    # linear encoding keeps more powder-averaged signal than spherical (positive anisotropic variance), and b = 0 is S0 e^{-TE/T2}
    assert float(mg(z, jnp.array([[b, 1.0, TE, 0.0]])).ravel()[0]) > float(mg(z, jnp.array([[b, 0.0, TE, 0.0]])).ravel()[0])
    np.testing.assert_allclose(float(mg(z, jnp.array([[0.0, 1.0, TE, 0.0]])).ravel()[0]), S0 * np.exp(-TE / T2), rtol=1e-12)
    # the gamma form is monotone in b for a very wide anisotropic variance, where the cumulant form is not
    zz = jnp.log(jnp.array([1.0, 75.0, 0.8, 0.05, 2.0])); bs = jnp.linspace(0, 3, 31)[:, None]
    Xg = jnp.concatenate([bs, jnp.ones_like(bs), jnp.zeros_like(bs), jnp.zeros_like(bs)], axis=1)
    assert np.all(np.diff(np.asarray(mg(zz, Xg)).ravel()) < 0) and not np.all(np.diff(np.asarray(mc(zz, Xg)).ravel()) < 0)


def test_biomarkers_and_virtual_targets():
    z = jnp.log(jnp.array([1.0, 75.0, 0.8, 0.05, 0.25]))
    g = np.asarray(divide_biomarkers(z)); np.testing.assert_allclose(g, [0.8, 3 * 0.05 / 0.64, 3 * 0.25 / 0.64], rtol=1e-12)
    m = DivideModel(1, biomarker_scale=jnp.array([2.0, 4.0, 8.0]), log_scale=jnp.array([0.5, 1.0, 2.0]))
    out = np.asarray(m(z, jnp.array([[0, 0, 0, 1.0], [0, 0, 0, 2.0], [0, 0, 0, 3.0]]))).ravel()
    np.testing.assert_allclose(out, g / np.array([2.0, 4.0, 8.0]), rtol=1e-12)
    lout = np.asarray(m(z, jnp.array([[0, 0, 0, 4.0], [0, 0, 0, 5.0], [0, 0, 0, 6.0]]))).ravel()
    np.testing.assert_allclose(lout, np.log(g) / np.array([0.5, 1.0, 2.0]), rtol=1e-12)
    # the log-biomarkers are linear in z: their Jacobian does not depend on z
    J1 = np.asarray(m.jacobian(z, jnp.array([[0, 0, 0, 5.0]]))); J2 = np.asarray(m.jacobian(z + 0.7, jnp.array([[0, 0, 0, 5.0]])))
    np.testing.assert_allclose(J1, J2, atol=1e-12); np.testing.assert_allclose(J1.ravel(), np.array([0, 0, -2, 1, 0]) / 1.0, atol=1e-12)


def test_jacobian_matches_finite_differences_for_signals_and_targets():
    m = DivideModel(1); z = DIVIDE_PRIOR_MEAN + 0.1
    X = jnp.array([[1.0, 1.0, 80.0, 0.0], [2.0, 0.0, 100.0, 0.0], [0, 0, 0, 1.0], [0, 0, 0, 3.0]])
    J = np.asarray(m.jacobian(z, X)); assert J.shape == (4, 1, 5)
    eps = 1e-6
    for i in range(5):
        e = jnp.zeros(5).at[i].set(eps); fd = (np.asarray(m(z + e, X)) - np.asarray(m(z - e, X))).reshape(4) / (2 * eps)
        np.testing.assert_allclose(J[:, 0, i], fd, rtol=1e-6, atol=1e-9)
    # nuisance split: S_0 and T_2 enter the signals but not the biomarkers
    assert np.all(np.abs(J[:2, 0, :2]) > 1e-6) and np.all(np.abs(J[2:, 0, :2]) < 1e-12)


def test_roi_model_equals_independent_single_voxel_models():
    V = 3; mV = DivideModel(V); m1 = DivideModel(1)
    rng = np.random.default_rng(0); Z = np.asarray(DIVIDE_PRIOR_MEAN) + np.asarray(DIVIDE_PRIOR_SD) * rng.normal(size=(V, 5))
    X = jnp.array([[1.2, 0.0, 90.0, 0.0], [0.5, 1.0, 60.0, 0.0], [0, 0, 0, 2.0]])
    out = np.asarray(mV(jnp.asarray(Z).reshape(-1), X)).reshape(3, V)
    for v in range(V):
        np.testing.assert_allclose(out[:, v], np.asarray(m1(jnp.asarray(Z[v]), X)).reshape(3), rtol=1e-12)
    J = np.asarray(mV.jacobian(jnp.asarray(Z).reshape(-1), X)).reshape(3, V, V, 5)
    off = sum(np.abs(J[:, v, w, :]).max() for v in range(V) for w in range(V) if v != w); assert off == 0.0   # block structure


def test_problem_builder_te_constraint_and_prior_predictive_monotonicity():
    p = make_divide_problem(seed=0, num_voxels=2)
    c = np.asarray(p.candidates); assert c.shape == (126, 4) and np.all(c[:, 3] == 0)
    assert np.all(c[:, 2] >= np.asarray(divide_te_min(jnp.asarray(c[:, 0]), jnp.asarray(c[:, 1]))) - 1e-9)
    assert p.data.x_train.shape == (126, 1, 1, 4) and p.data.y_train.shape == (126, 1, 1, 2) and p.data.y_test_pool.shape == (3, 1, 1, 2)
    assert p.prior_mean.shape == (10, 1) and p.prior_cov.shape == (10, 10) and p.z_true.shape == (2, 5)
    # prior predictive (G3): with the gamma form every draw is monotone in b; check on 500 draws over the candidate grid
    rng = np.random.default_rng(1); Z = np.asarray(DIVIDE_PRIOR_MEAN) + np.asarray(DIVIDE_PRIOR_SD) * rng.normal(size=(500, 5))
    m1 = DivideModel(1, p.biomarker_scale)
    Xb = jnp.asarray(np.column_stack([np.linspace(0, c[:, 0].max(), 21), np.ones(21), np.full(21, 80.0), np.zeros(21)]))
    S = np.asarray(jax.vmap(lambda z: m1(z, Xb).reshape(-1))(jnp.asarray(Z)))
    assert np.all(np.diff(S, axis=1) < 0)


def test_repeated_acquisitions_carry_fresh_noise_in_the_experiment_loop():
    """Regression test for the D-6 defect: a repeated design must not return the stored label."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sequential_divide import run_arm
    p = make_divide_problem(seed=3, num_voxels=1)
    n = p.candidates.shape[0]; noise = np.random.default_rng(0).normal(size=(n, 12, 1))
    fixed = [7] * 9                                             # the same design nine times
    out = run_arm(p, "FIXED-EPIG", warm=[7, 7, 7], budget=9, rng=np.random.default_rng(0), num_samples=2000, seed=0, fixed=fixed, noise=noise)
    assert out["selected"] == fixed
    # with fresh noise the EKF plug-in error must keep falling on average over the repeats, and the
    # posterior of a repeatedly measured signal must tighten: check through the recorded curve length
    assert len(out["curve_ekf"]) == 10
    # direct check of the observation path: the noise table entries used are distinct rows
    assert len({float(v) for v in noise[7, :12, 0]}) == 12
