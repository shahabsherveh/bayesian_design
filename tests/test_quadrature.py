"""The mode-centred quadrature against closed forms and against a deliberately hard posterior."""
import numpy as np
import pytest

from bed.quadrature import enable_double_precision

# The exactness claims below are at double precision. In JAX's single-precision default the same
# quantities carry about 1e-7 relative error, which is the floor these tests would otherwise be
# measuring rather than the quadrature.
assert enable_double_precision()

from bed import reference as ref  # noqa: E402
from bed.quadrature import (ModeQuadrature, gauss_hermite, _laplace_modes,  # noqa: E402
                            mode_starts)


class LinearModel:
    """y = A z, so the posterior is Gaussian and every moment is available in closed form."""

    def __init__(self, A):
        self.A = np.asarray(A, float)

    def __call__(self, z, X):
        import jax.numpy as jnp
        z = jnp.asarray(z); X = jnp.asarray(X)
        return jnp.einsum("ij,j->i", jnp.asarray(self.A), z)[None, :] * jnp.ones((X.shape[0], 1))

    def jacobian(self, z, X):
        import jax.numpy as jnp
        return jnp.broadcast_to(jnp.asarray(self.A)[None], (jnp.asarray(X).shape[0],) + self.A.shape)


def gauss_hermite_is_exact():
    x, w = gauss_hermite(20)
    for k, exact in ((0, 1.0), (2, 1.0), (4, 3.0), (6, 15.0)):
        assert abs((w * x ** k).sum() - exact) < 1e-10


def test_gauss_hermite_exact():
    gauss_hermite_is_exact()


@pytest.mark.parametrize("d,k", [(2, 3), (2, 8), (3, 5)])
def test_linear_gaussian_posterior_exact(d, k):
    """Against the Kalman closed form: mean, covariance and target moments to 1e-9."""
    rng = np.random.default_rng(11 + d * 17 + k)
    A = rng.normal(size=(1, d))
    mu0 = rng.normal(size=d); S0 = np.eye(d) * 0.7 + 0.1
    S0 = S0 @ S0.T
    R = np.array([[0.35 ** 2]])
    X = rng.normal(size=(k, d))
    ztrue = mu0 + np.linalg.cholesky(S0) @ rng.normal(size=d)
    Y = (A @ ztrue)[None, :].repeat(k, 0) + rng.normal(size=(k, 1)) * 0.35

    post = ref.Posterior(LinearModel(A), X, Y, mu0, S0, R)
    # closed form: stacking k identical rows of A
    H = np.repeat(A, k, axis=0)
    Rb = np.eye(k) * R[0, 0]
    S0i = np.linalg.inv(S0)
    Sn = np.linalg.inv(S0i + H.T @ np.linalg.inv(Rb) @ H)
    mn = Sn @ (S0i @ mu0 + H.T @ np.linalg.inv(Rb) @ Y.ravel())

    q = ModeQuadrature(post, mu0, S0, n_per_axis=24)
    m, v = q.moments(lambda Z: Z)
    assert np.allclose(m, mn, atol=1e-9, rtol=1e-7), (m, mn)
    assert np.allclose(v, np.diag(Sn), atol=1e-9, rtol=1e-7), (v, np.diag(Sn))
    # one mode, and the weight moves off the base rule by exactly what the inflation implies: for a
    # linear-Gaussian model the Laplace component IS the posterior, so with the proposal widened by
    # `inflate` the density ratio peaks at inflate**(d/2) relative to the uninflated rule.
    assert q.n_components == 1
    assert abs(q.weight_concentration_ratio - 2.0 ** (d / 2)) < 0.25 * 2.0 ** (d / 2), \
        (q.weight_concentration_ratio, 2.0 ** (d / 2))
    # and with no inflation it is exactly the base rule
    q1 = ModeQuadrature(post, mu0, S0, n_per_axis=24, inflate=1.0)
    m1, v1 = q1.moments(lambda Z: Z)
    assert abs(q1.weight_concentration_ratio - 1.0) < 1e-6, q1.weight_concentration_ratio
    assert np.allclose(m1, mn, atol=1e-9, rtol=1e-7)


def test_reports_concentration_rather_than_hiding_it():
    """A posterior far narrower than the prior is still resolved, and says so."""
    rng = np.random.default_rng(3)
    d = 2
    A = np.array([[1.0, 0.4]])
    mu0 = np.zeros(d); S0 = np.eye(d) * 4.0
    R = np.array([[0.02 ** 2]])                      # very sharp likelihood
    X = rng.normal(size=(30, d))
    Y = np.full((30, 1), 0.3) + rng.normal(size=(30, 1)) * 0.02
    post = ref.Posterior(LinearModel(A), X, Y, mu0, S0, R)
    q = ModeQuadrature(post, mu0, S0, n_per_axis=24)
    m, v = q.moments(lambda Z: Z)
    H = np.repeat(A, 30, axis=0); Rb = np.eye(30) * R[0, 0]
    Sn = np.linalg.inv(np.linalg.inv(S0) + H.T @ np.linalg.inv(Rb) @ H)
    mn = Sn @ (np.linalg.inv(S0) @ mu0 + H.T @ np.linalg.inv(Rb) @ Y.ravel())
    assert np.allclose(m, mn, atol=1e-8, rtol=1e-6)
    assert np.allclose(v, np.diag(Sn), atol=0, rtol=1e-6)
    assert q.n_modes == 1 and q.weight_concentration_ratio < 4.0, q.diagnostics()


def test_mixture_components_carry_their_weights():
    """A two-component posterior, against a converged brute-force grid.

    This is the case the earlier implementation got wrong: each component's nodes carried the
    Gauss-Hermite weight but not the component's mixture weight, while the density in the ratio was
    mixture-weighted. With one component the mixture weight is 1 and nothing is lost, so every
    unimodal test passed while a strongly unequal two-component posterior was wrong by up to a
    factor two with perfect node coverage.
    """
    import jax.numpy as jnp

    class Quadratic:
        """y = (z0^2 + z1) gives a posterior symmetric in the sign of z0, hence two modes."""

        def __call__(self, z, X):
            z = jnp.asarray(z)
            return jnp.full((jnp.asarray(X).shape[0], 1), z[0] ** 2 + z[1])

        def jacobian(self, z, X):
            z = jnp.asarray(z)
            return jnp.broadcast_to(jnp.array([[2.0 * z[0], 1.0]])[None],
                                    (jnp.asarray(X).shape[0], 1, 2))

    # The second coordinate is given a tight prior so it cannot absorb the shift; the likelihood
    # then pins z0^2, leaving two well-separated modes near +-1 rather than one curved ridge. The
    # prior mean is offset so the two modes carry **unequal** mass: with equal mixture weights the
    # omitted factor is a constant and the bug is invisible, which is exactly what a symmetric
    # version of this fixture showed when the defect was reintroduced deliberately.
    mu0 = np.array([0.75, 0.0]); S0 = np.diag([0.55, 0.01])
    R = np.array([[0.15 ** 2]])
    X = np.zeros((4, 1)); Y = np.full((4, 1), 1.0)
    post = ref.Posterior(Quadratic(), X, Y, mu0, S0, R)

    ax = np.linspace(-8, 8, 1201)
    Z = np.stack([g.ravel() for g in np.meshgrid(ax, ax, indexing="ij")], axis=1)
    lp = np.concatenate([post.logp(Z[s:s + 200000]).ravel() for s in range(0, Z.shape[0], 200000)])
    w = np.exp(lp - lp.max()); w /= w.sum()
    m_grid = w @ Z
    v_grid = w @ (Z ** 2) - m_grid ** 2

    q = ModeQuadrature(post, mu0, S0, n_per_axis=48, max_modes=8)
    assert q.n_modes >= 2, q.diagnostics()
    m, v = q.moments(lambda Zn: Zn)
    assert np.allclose(m, m_grid, atol=2e-3), (m, m_grid)
    assert np.allclose(v, v_grid, rtol=2e-2), (v, v_grid)


def test_broad_low_peak_survives_the_mode_cap():
    """A broad low peak carries more mass than a narrow high one and must not be truncated away.

    The mode search returns modes ordered by posterior height. Truncating that order at `max_modes`
    drops the lowest peaks, which is the wrong criterion, because mass is height times width.

    The fixture makes the two orders genuinely disagree, which an earlier version did not: the
    likelihood's slope in the second coordinate differs by well, so the taller mode is also much
    narrower. Measured here the taller peak carries 0.336 of the mass and the broader one 0.664, so
    ranking by height retains the wrong component. Without that separation the test passes with the
    defect reintroduced, which is what the first version of it did.
    """
    import jax.numpy as jnp

    class Wells:
        def __init__(self, chi, clo):
            self.chi, self.clo = chi, clo

        def __call__(self, z, X):
            z = jnp.asarray(z)
            c = jnp.where(z[0] > 0, self.chi, self.clo)
            return jnp.full((jnp.asarray(X).shape[0], 1), z[0] ** 2 + c * z[1])

        def jacobian(self, z, X):
            z = jnp.asarray(z)
            c = jnp.where(z[0] > 0, self.chi, self.clo)
            return jnp.broadcast_to(jnp.array([[2.0 * z[0], c]])[None],
                                    (jnp.asarray(X).shape[0], 1, 2))

    mu0 = np.array([0.2, 0.0]); S0 = np.diag([1.0, 1.0]); R = np.array([[0.2 ** 2]])
    post = ref.Posterior(Wells(8.0, 0.08), np.zeros((4, 1)), np.full((4, 1), 1.0), mu0, S0, R)

    q_all = ModeQuadrature(post, mu0, S0, n_per_axis=40, max_modes=8)
    assert q_all.n_modes == 2, q_all.diagnostics()
    w = q_all.search_log["weights_before_truncation"]
    # the recorded weights must be in mass order, heaviest first
    assert w == sorted(w, reverse=True), w
    assert w[0] > 0.6 and w[1] < 0.4, w

    # with one component allowed, the retained mode must be the heavier, which is NOT the taller
    q_one = ModeQuadrature(post, mu0, S0, n_per_axis=40, max_modes=1)
    kept = q_one.nodes.mean(axis=0)
    taller = min(_laplace_modes(post, mode_starts(post, mu0, S0))[0], key=lambda t: t[2])[0]
    assert np.linalg.norm(kept - taller) > 0.5, (kept, taller)
    assert q_one.search_log["mass_dropped_by_mode_cap"] == pytest.approx(w[1], rel=1e-6)


def test_search_log_records_what_was_found_and_why_it_went():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(1, 2))
    mu0 = np.zeros(2); S0 = np.eye(2); R = np.array([[0.3 ** 2]])
    post = ref.Posterior(LinearModel(A), rng.normal(size=(3, 2)), np.zeros((3, 1)), mu0, S0, R)
    q = ModeQuadrature(post, mu0, S0, n_per_axis=20)
    d = q.diagnostics()
    for k in ("n_found_before_truncation", "weights_before_truncation", "mass_dropped_by_mode_cap",
              "mass_dropped_by_weight_floor", "n_retained", "optimiser_failures", "starts"):
        assert k in d["search"], (k, d["search"].keys())
    assert d["search"]["starts"] > 0
    assert "log_unnormalised_evidence" in d and "n_sd" not in d


def test_predictive_normalisation_is_reported_not_hidden():
    """The expected variance IS divided by the estimated predictive mass; both parts are returned.

    This one needs a concrete problem to integrate over, and the only one to hand lives in the
    experiment harness rather than the library. It therefore skips where the harness is absent,
    which is the case in a checkout that has the library alone.
    """
    import sys
    sys.path.insert(0, "experiments/suff")
    pytest.importorskip("suff_common",
                        reason="needs experiments/suff, which ships separately from the library")
    from suff_common import CONFIGS, make_problem
    from bed.quadrature import expected_posterior_variance

    p = make_problem(CONFIGS["source_v2"], 50000)
    post = ref.Posterior(p.model, p.X[[10, 40, 90]], np.zeros((3, 1)), p.mu0, p.S0, p.R)
    q = ModeQuadrature(post, p.mu0, p.S0, n_per_axis=24)
    V, EV, U, mass, info = expected_posterior_variance(q, p, np.arange(p.n), n_y=48)
    assert np.allclose(EV, np.asarray(info["ev_raw"]) / np.asarray(info["mass"]), rtol=1e-12)
    assert abs(info["mass_dev_absmax"]) < 1e-3, info["mass_dev_absmax"]
    assert info["negative_variance_max"] >= 0.0
    assert isinstance(info["unresolved"], list)


def test_generated_tex_control_words_are_letters_only(tmp_path):
    """The manuscript macro file must compile. TeX ends a control word at the first non-letter.

    Thirty-six of sixty-five generated names once contained digits, so `\\newcommand{\\FhContrastK16}`
    defined nothing and the file failed with "Missing begin document". The transfer artifact was
    unusable and nothing in the pipeline noticed.
    """
    import re
    import shutil
    import subprocess
    from pathlib import Path

    tex = Path("results/suff2/summaries/manuscript_numbers.tex")
    if not tex.exists():
        pytest.skip("macro file not generated in this checkout")
    names = re.findall(r"\\newcommand\{\\([^}]+)\}", tex.read_text())
    assert names, "no macros found"
    bad = [n for n in names if not n.isalpha()]
    assert not bad, f"control words must be letters only: {bad[:5]}"

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex unavailable; the name check above still ran")
    d = tmp_path
    shutil.copy(tex, d / "macros.tex")
    # Expand EVERY macro, not one of them. A definition can be well formed while its body is not,
    # and a document that uses a single macro compiles cleanly with sixty-four broken ones behind it.
    body = "\n".join("\\%s\\par" % n for n in names)
    (d / "t.tex").write_text("\\documentclass{article}\n\\input{macros.tex}\n"
                             "\\begin{document}\n" + body + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"],
                       cwd=d, capture_output=True, text=True)
    log = (d / "t.log").read_text() if (d / "t.log").exists() else ""
    errors = [ln for ln in log.splitlines() if ln.startswith("!")]
    assert r.returncode == 0 and not errors, \
        f"{len(errors)} TeX errors expanding {len(names)} macros: {errors[:5]}"
