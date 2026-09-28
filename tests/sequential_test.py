"""The shared sequential loop: regression against the script behind the paper's DIVIDE numbers,
exactness of the refitted belief on a linear model, fresh noise on repeats, and the reference
readout."""
import sys
from pathlib import Path

import jax
import numpy as np

from bed import sequential as sq

jax.config.update("jax_enable_x64", True)


def test_ekf_greedy_epig_reproduces_the_paper_script_on_divide():
    """Same truth, warm start and noise table as ``experiments/sequential_divide.py`` (draw d):
    the new loop must select the same designs and end at the same EKF mean."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sequential_divide import run_arm
    from bed.mechanistic import make_divide_problem
    d, budget, warm_start = 2, 6, 3
    p_old = make_divide_problem(seed=100 + d, num_voxels=1, snr=40.0); rng = np.random.default_rng(d)
    warm = [int(v) for v in rng.choice(p_old.candidates.shape[0], warm_start, replace=False)]
    noise = np.random.default_rng(777 + d).normal(size=(p_old.candidates.shape[0], budget + warm_start, 1))
    old = run_arm(p_old, "EPIG", warm, budget, np.random.default_rng(0), 2000, seed=d, noise=noise)
    p = sq.divide_problem(seed=100 + d, snr=40.0)
    h = sq.run_episode(p, sq.GreedyPolicy("epig"), "ekf", warm, budget, noise_seed=777 + d)
    assert h["designs"][warm_start:] == old["selected"]
    # the old script's EKF plug-in loss at the end equals ours from the recorded final mean
    from bed.mechanistic import divide_biomarkers
    plug_old = np.asarray(old["sq_err_ekf"])
    g = np.asarray(divide_biomarkers(h["final_mean"])) / np.asarray(p_old.biomarker_scale)
    truth = np.asarray(divide_biomarkers(p_old.z_true[0])) / np.asarray(p_old.biomarker_scale)
    np.testing.assert_allclose((g - truth) ** 2, plug_old, rtol=1e-6, atol=1e-12)   # solve vs inv: 4e-8 relative


def test_laplace_belief_is_exact_on_a_linear_model_and_ekf_equals_it_there():
    from bed.mechanistic import ClosedFormModel
    import jax.numpy as jnp

    class Lin(ClosedFormModel):
        output_dim = 1; weight_size = 3

        def _single(self, z, x):
            return jnp.array([x[0] * z[0] + x[1] * z[1] + z[2]])

    rng = np.random.default_rng(0); X = rng.normal(size=(8, 2)); mu0 = np.zeros(3); S0 = np.diag([1.0, 2.0, 0.5]); R = np.array([[0.1]])
    z_true = np.array([0.3, -0.7, 1.1])
    tf = lambda Z: np.asarray(Z)[:, :1]
    p = sq.DesignProblem("lin", Lin(), X, X[:1], mu0, S0, R, R, z_true, False, tf, ("z0",))
    hs = {k: sq.run_episode(p, sq.FixedPolicy([0, 1, 2, 3, 4], "f"), k, [5, 6], 5, noise_seed=1) for k in ("ekf", "laplace")}
    idx = hs["ekf"]["designs"]; Y = hs["ekf"]["outcomes"].reshape(-1)
    A = np.column_stack([X[idx], np.ones(len(idx))]); P = np.linalg.inv(S0) + A.T @ A / R[0, 0]
    cov = np.linalg.inv(P); mean = cov @ (A.T @ Y / R[0, 0])
    for k in ("ekf", "laplace"):
        np.testing.assert_allclose(hs[k]["final_mean"], mean, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(hs[k]["final_cov"], cov, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(hs["ekf"]["outcomes"], hs["laplace"]["outcomes"])     # common random numbers


def test_repeats_get_fresh_noise_and_random_policy_respects_replacement():
    p = sq.divide_problem(seed=0)
    h = sq.run_episode(p, sq.FixedPolicy([9] * 6, "rep"), "ekf", [9, 9], 6, noise_seed=0)
    ys = h["outcomes"].reshape(-1)
    assert len(set(np.round(ys, 12))) == 8
    pk = sq.pkpd_problem(seed=0, num_times=10)
    hr = sq.run_episode(pk, sq.RandomPolicy(), "ekf", [0, 1], 8, noise_seed=0, rng_seed=3)
    assert len(set(hr["designs"])) == 10                                    # without replacement in PK/PD


def test_evaluate_history_returns_a_calibrated_looking_readout():
    p = sq.divide_problem(seed=1, snr=100.0)
    h = sq.run_episode(p, sq.GreedyPolicy("epig"), "ekf", [4, 50, 90], 10, noise_seed=1)
    ev = sq.evaluate_history(p, h, num_samples=4000, seed=0)
    s = sq.target_loss_summary(p, ev)
    assert s["loss"].shape == (3,) and np.all(np.isfinite(s["loss"])) and ev["is"]["ess"] > 200 and ev["is"]["khat"] < 0.7
    assert np.all(s["post_var"] > 0)


def test_indefinite_belief_covariance_is_repaired_and_counted():
    """A covariance made negative along a target direction makes the closed form raise; the
    scores must then be computed on the projected covariance and the repair counted."""
    from bed import gaussian_design as gd
    p = sq.divide_problem(seed=0)
    b = sq.EKFBelief(p); v = p.jac_targets(b.mean)[0].reshape(-1); v = v / np.linalg.norm(v)
    b.cov = b.cov - 1.05 * float(v @ b.cov @ v) * np.outer(v, v)      # indefinite along the first target
    with np.testing.assert_raises(np.linalg.LinAlgError):
        gd.epig_scores(p.jac(b.mean, p.X), p.jac_targets(b.mean), b.cov, p.R, p.Rt)
    sc = sq.closed_form_scores(b, "epig")
    assert np.all(np.isfinite(sc)) and b.psd_repairs == 1 and b.psd_worst > 0
    assert np.linalg.eigvalsh(b.cov).min() > 0
