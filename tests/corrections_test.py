"""Regression checks for the correction register of the Gaussian-sufficiency round
(``docs/investigations/sufficiency/CORRECTIONS.md``): truth drawn from the belief prior, one
feasible set everywhere, standardised PK/PD targets, MCMC accepted by diagnostics only, target
entropy in the toy, and subject-clustered intervals."""
import jax
import numpy as np

from bed import reference as ref
from bed import sequential as sq
from bed import stats, toy

jax.config.update("jax_enable_x64", True)


# ---- C1: truth versus assumed prior --------------------------------------------------------

def test_divide_truth_is_drawn_from_the_belief_prior_by_default_and_the_old_construction_is_labelled():
    base = sq.divide_problem(30000, snr=100.0)
    same = sq.divide_problem(30000, snr=100.0, nuisance_sd=(0.1, 0.2))
    np.testing.assert_allclose(same.z_true, base.z_true, atol=1e-12)
    assert not same.meta["misspecified"]
    wide = sq.divide_problem(30000, snr=100.0, nuisance_sd=(0.3, 0.5))
    eps = (base.z_true - base.mu0) / np.sqrt(np.diag(base.S0))
    np.testing.assert_allclose(wide.z_true[:2], base.mu0[:2] + np.array([0.3, 0.5]) * eps[:2], atol=1e-12)
    np.testing.assert_allclose(wide.z_true[2:], base.z_true[2:])
    np.testing.assert_allclose(np.sqrt(np.diag(wide.S0))[:2], [0.3, 0.5]); assert not wide.meta["misspecified"]
    assert wide.meta["prior_sd_belief"][:2] == [0.3, 0.5] and np.allclose(wide.meta["prior_sd_truth"][:2], [0.3, 0.5])
    old = sq.divide_problem(30000, snr=100.0, nuisance_sd=(0.3, 0.5), truth_nuisance_sd="default")
    np.testing.assert_allclose(old.z_true, base.z_true, atol=1e-12)          # the 2026-09-24 construction: narrow truth, wide belief
    np.testing.assert_allclose(np.sqrt(np.diag(old.S0))[:2], [0.3, 0.5]); assert old.meta["misspecified"]
    assert np.allclose(old.meta["prior_sd_truth"][:2], [0.1, 0.2])


def test_divide_truth_distribution_matches_the_belief_prior_across_seeds():
    """Standardised truths ``(z_true - mu0) / sd_belief`` have unit sample variance in every
    coordinate under the matched construction and variance (0.1/0.3)², (0.2/0.5)² under the
    labelled misspecification."""
    Z = []; Zm = []
    for seed in range(30000, 30300):
        p = sq.divide_problem(seed, snr=100.0, nuisance_sd=(0.3, 0.5)); Z.append((p.z_true - p.mu0) / np.sqrt(np.diag(p.S0)))
        q = sq.divide_problem(seed, snr=100.0, nuisance_sd=(0.3, 0.5), truth_nuisance_sd="default"); Zm.append((q.z_true - q.mu0) / np.sqrt(np.diag(q.S0)))
    sd = np.std(np.array(Z), axis=0); sdm = np.std(np.array(Zm), axis=0)
    assert np.all(np.abs(sd - 1.0) < 0.15), sd
    assert abs(sdm[0] - 0.1 / 0.3) < 0.06 and abs(sdm[1] - 0.2 / 0.5) < 0.08 and np.all(np.abs(sdm[2:] - 1.0) < 0.15), sdm


# ---- C2: feasible candidates everywhere ----------------------------------------------------

def test_feasible_mask_is_the_single_definition_and_infeasible_selections_raise():
    pk = sq.pkpd_problem(0, num_times=10); dv = sq.divide_problem(0)
    assert dv.feasible([3, 3, 5]).all()
    m = pk.feasible([0, 4]); assert not m[0] and not m[4] and m.sum() == pk.n - 2
    sc = sq.mask_infeasible(pk, np.ones(pk.n), [0, 4]); assert np.isneginf(sc[[0, 4]]).all() and np.isfinite(sc).sum() == pk.n - 2
    h = sq.run_episode(pk, sq.GreedyPolicy("var"), "ekf", [0, 1], 12, noise_seed=0)
    assert len(set(h["designs"])) == 14
    class Repeat:
        name = "repeat"
        def select(self, belief, used, rng):
            return int(used[-1]), None
    with np.testing.assert_raises(ValueError):
        sq.run_episode(pk, Repeat(), "ekf", [0, 1], 2, noise_seed=0)
    with np.testing.assert_raises(ValueError):
        sq.run_episode(pk, sq.RandomPolicy(), "ekf", [2, 2], 1, noise_seed=0)
    hr = sq.run_episode(dv, sq.FixedPolicy([7] * 4, "rep"), "ekf", [7], 4, noise_seed=0)          # repeats with fresh noise where allowed
    assert len(set(np.round(hr["outcomes"].reshape(-1), 12))) == 5
    with np.testing.assert_raises(ValueError):                                                       # a protocol colliding with the warm start is caught
        sq.run_episode(pk, sq.FixedPolicy([0, 5, 6], "f"), "ekf", [0, 1], 3, noise_seed=0)
    hf = sq.run_episode(pk, sq.FixedPolicy([0, 5, 1, 6, 7], "f", skip_used=True), "ekf", [0, 1], 3, noise_seed=0)   # executed as its first feasible entries
    assert hf["designs"] == [0, 1, 5, 6, 7]


# ---- C4: objective consistency, PK/PD standardised targets ----------------------------------

def test_pkpd_standardised_targets_scale_values_jacobians_and_target_noise_consistently():
    raw = sq.pkpd_problem(3, num_times=10); std = sq.pkpd_problem(3, num_times=10, standardise="per-output")
    sc = std.target_scale; assert sc.shape == (2,) and np.all(sc > 0)
    z = std.mu0 + 0.1 * np.sqrt(np.diag(std.S0))
    np.testing.assert_allclose(std.target_fn(z[None]).reshape(-1, 2), raw.target_fn(z[None]).reshape(-1, 2) / sc, rtol=1e-12)
    np.testing.assert_allclose(std.jac_targets(z), raw.jac_targets(z) / sc[None, :, None], rtol=1e-12)
    D = np.diag(1 / sc); np.testing.assert_allclose(std.Rt, D @ raw.Rt @ D, rtol=1e-12)
    # finite-difference check that the scaled Jacobian differentiates the scaled target function
    e = np.zeros(std.d); e[2] = 1e-6
    fd = (std.target_fn((z + e)[None]) - std.target_fn((z - e)[None])).reshape(-1, 2) / 2e-6
    np.testing.assert_allclose(std.jac_targets(z)[:, :, 2], fd, rtol=1e-4, atol=1e-8)
    # the acquisition on standardised targets equals the loss it is matched to: sum of scaled target variances
    from bed import gaussian_design as gd
    Jt = std.jac_targets(z); v = gd.target_variance(std.S0, Jt, std.Rt)
    assert abs(v - np.sum(np.einsum("mti,ij,mtj->mt", Jt, std.S0, Jt) + np.diag(std.Rt)[None, :])) < 1e-10
    assert std.meta["standardise"] == "per-output" and raw.meta["standardise"] is None


# ---- C3: reference validation ---------------------------------------------------------------

def test_rank_normalised_rhat_and_ess_behave_on_known_chains():
    rng = np.random.default_rng(0); T, C = 2000, 4
    iid = rng.normal(size=(T, C, 2)); r = ref.rhat_rank(iid); assert np.all(r < 1.01), r
    bulk, tail = ref.ess_bulk_tail(iid); assert np.all(bulk > 0.7 * T * C) and np.all(tail > 0.5 * T * C), (bulk, tail)
    shifted = iid.copy(); shifted[:, 0, 0] += 1.5; assert ref.rhat_rank(shifted)[0] > 1.1
    rho = 0.9; ar = np.zeros((T, C, 1))
    for t in range(1, T):
        ar[t] = rho * ar[t - 1] + rng.normal(size=(C, 1))
    b, _ = ref.ess_bulk_tail(ar); expected = T * C * (1 - rho) / (1 + rho)
    assert 0.5 * expected < b[0] < 2.0 * expected, (b, expected)


def test_mcmc_stage_is_accepted_only_by_its_diagnostics_and_reports_a_chain_ess():
    from tests.reference_test import _linear_case
    post, mean, cov = _linear_case()
    r = ref.robust_posterior(post, 2000, seed=0, khat_max=-10.0, ess_min=300, mcmc_steps=6000, mcmc_chains=8)   # IS can never pass: forced escalation
    assert r.method == "mcmc" and r.reliable and r.rhat_max <= 1.01 and r.ess_bulk_min >= 300
    assert r.ess == r.ess_bulk_min and r.ess < r.samples.shape[0]                    # a chain ESS, not the retained draw count
    m, v = r.moments(lambda Z: Z); err = r.mc_error(lambda Z: Z)
    assert np.all(np.abs(m - mean) < 5 * err + 1e-6) and np.all(np.abs(v / np.diag(cov) - 1) < 0.2)
    d = r.diagnostics(); assert d["reliable"] and d["method"] == "mcmc" and d["ess_tail_min"] >= 200 and len(d["stages"]) >= 2
    r2 = ref.robust_posterior(post, 2000, seed=0, khat_max=-10.0, ess_min=10 ** 7, mcmc_steps=800, mcmc_chains=4, mcmc_retry=1)
    assert r2.method == "mcmc" and not r2.reliable and not r2.diagnostics()["reliable"]    # kept, flagged, not dropped


# ---- C6: toy validity ------------------------------------------------------------------------

def test_target_entropy_equals_full_entropy_for_an_injective_target_and_the_marginal_otherwise():
    m = toy.sigmoid_location_model(G=61, ny=41)
    for kind_a, kind_b in (("target_entropy", "param_entropy"), ("entropy", "param_entropy")):
        assert abs(m.expected_terminal(m.prior, [1, 3], kind_a) - m.expected_terminal(m.prior, [1, 3], kind_b)) < 1e-12
    a = toy.amplitude_nuisance_model(G=21, ny=41)
    W = a.prior[None, :]; full = a.utility_rows(W, "param_entropy")[0]; tgt = a.utility_rows(W, "target_entropy")[0]
    marg = a.prior.reshape(21, 21).sum(axis=0)                   # grid is (theta, a) with a on the second axis
    assert abs(tgt - (-(marg * np.log(marg)).sum())) < 1e-12 and tgt < full


# ---- C5: independent statistical unit -----------------------------------------------------

def test_cluster_bootstrap_reduces_to_the_paired_bootstrap_and_widens_under_duplicated_rows():
    rng = np.random.default_rng(1); n = 200
    a = rng.lognormal(size=n); b = a * rng.lognormal(sigma=0.3, size=n)
    plain = stats.ratio_ci(a, b, np.arange(n), n_boot=3000, seed=0)
    dup = stats.ratio_ci(np.repeat(a, 4), np.repeat(b, 4), np.repeat(np.arange(n), 4), n_boot=3000, seed=0)
    naive = stats.ratio_ci(np.repeat(a, 4), np.repeat(b, 4), np.arange(4 * n), n_boot=3000, seed=0)
    assert abs(plain[0] - a.mean() / b.mean()) < 1e-12 and abs(dup[0] - plain[0]) < 1e-12
    assert abs((dup[2] - dup[1]) - (plain[2] - plain[1])) < 0.3 * (plain[2] - plain[1])     # duplicating rows must not narrow the interval
    assert (naive[2] - naive[1]) < 0.7 * (dup[2] - dup[1])                                  # the naive unit does narrow it
    assert stats.noninferiority([1.02, 0.97, 1.04], 0.05) == "noninferior" and stats.noninferiority([1.1, 1.02, 1.2], 0.05) == "inferior" and stats.noninferiority([1.0, 0.9, 1.1], 0.05) == "inconclusive"
