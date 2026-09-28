"""Correctness tests for the adaptive partition reference.

`bed.adaptive` is what adjudicated whether the campaign's mode-centred rule is right, so it needs
oracles that do not come from this package. Every expected value below is either a closed form
written out by hand or a brute-force grid sum in NumPy.

The module has two failure modes of its own and both are exercised here, because a reference whose
failures are untested cannot adjudicate anything:

* a rectangle that is too small, where a coarse frame rule reports an empty frame while a thin
  ridge leaves the box;
* a mesh that is too coarse, where the embedded error estimate is small because the cells are badly
  resolved rather than well resolved, so the rule reports convergence at the wrong value.
"""
import numpy as np
import pytest

from bed.adaptive import AdaptiveQuadrature
from bed.quadrature import enable_double_precision

enable_double_precision()


class Gaussian:
    """Unnormalised Gaussian log density; the moments are known in closed form."""

    def __init__(self, mean, cov):
        self.m = np.asarray(mean, float)
        self.S = np.asarray(cov, float)
        self.P = np.linalg.inv(self.S)

    def logp(self, z):
        d = np.asarray(z, float) - self.m
        return -0.5 * np.einsum("ni,ij,nj->n", d, self.P, d)


class AsymmetricBimodal:
    """Two well-separated Gaussians with UNEQUAL weights.

    Equal weights would make the component weighting invisible, which is exactly how a missing
    mixture weight once survived a regression test elsewhere in this project.
    """

    w0, w1 = 0.336, 0.664
    m0, m1 = np.array([-2.0, 0.0]), np.array([2.5, 0.0])
    v0, v1 = 0.25, 1.00

    def logp(self, z):
        z = np.asarray(z, float)
        a, b = z - self.m0, z - self.m1
        la = (np.log(self.w0) - 0.5 * (a[:, 0] ** 2 + a[:, 1] ** 2) / self.v0
              - np.log(self.v0 * 2 * np.pi))
        lb = (np.log(self.w1) - 0.5 * (b[:, 0] ** 2 + b[:, 1] ** 2) / self.v1
              - np.log(self.v1 * 2 * np.pi))
        return np.logaddexp(la, lb)


class Banana:
    """A thin curved ridge that leaves any modestly sized rectangle."""

    def logp(self, z):
        z = np.asarray(z, float)
        return -0.5 * (z[:, 0] ** 2 / 4.0 + (z[:, 1] - 0.6 * (z[:, 0] ** 2 - 4.0)) ** 2 / 0.09)


def brute(post, fn, lo, hi, n):
    """Independent oracle: a uniform midpoint sum, with the boundary mass reported."""
    g0 = np.linspace(lo[0], hi[0], n)
    g1 = np.linspace(lo[1], hi[1], n)
    X, Y = np.meshgrid(g0, g1, indexing="ij")
    Z = np.stack([X.ravel(), Y.ravel()], axis=1)
    lp = np.asarray(post.logp(Z), float).ravel()
    w = np.exp(lp - lp.max())
    w /= w.sum()
    F = np.asarray(fn(Z), float).reshape(Z.shape[0], -1)
    edge = w[(np.abs(Z[:, 0] - lo[0]) < 1e-12) | (np.abs(Z[:, 0] - hi[0]) < 1e-12)
             | (np.abs(Z[:, 1] - lo[1]) < 1e-12) | (np.abs(Z[:, 1] - hi[1]) < 1e-12)].sum()
    return w @ F, w @ (F ** 2) - (w @ F) ** 2, float(edge)


def test_exact_on_a_gaussian():
    """Closed-form mean, cross moment and variance, to better than 1e-9."""
    S = np.array([[1.7, 0.6], [0.6, 0.9]])
    m = np.array([0.4, -0.8])
    q = AdaptiveQuadrature(Gaussian(m, S), np.zeros(2), 4.0 * np.eye(2))
    fn = lambda z: np.stack([z[:, 0], z[:, 1], z[:, 0] * z[:, 1]], axis=1)
    mu, var = q.moments(fn)
    assert np.allclose(mu[:2], m, atol=1e-9), mu[:2]
    assert np.isclose(mu[2], S[0, 1] + m[0] * m[1], atol=1e-9)
    assert np.allclose(var[:2], np.diag(S), atol=1e-8), var[:2]


def test_asymmetric_bimodal_matches_the_analytic_mixture():
    """A mixture whose components carry unequal weight, against the analytic moments."""
    p = AsymmetricBimodal()
    q = AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2))
    mu, _ = q.moments(lambda z: np.stack([z[:, 0], z[:, 0] ** 2], axis=1))
    e0 = p.w0 * p.m0[0] + p.w1 * p.m1[0]
    e1 = p.w0 * (p.m0[0] ** 2 + p.v0) + p.w1 * (p.m1[0] ** 2 + p.v1)
    assert np.isclose(mu[0], e0, atol=1e-9), (mu[0], e0)
    assert np.isclose(mu[1], e1, atol=1e-8), (mu[1], e1)


def test_the_rectangle_expands_until_the_frame_is_empty():
    """On a ridge that leaves the initial box, the domain must grow and reach the brute answer.

    This is the control for the tail check. An earlier version integrated the frame with a single
    coarse rule, which steps over a ridge this thin and reports an empty frame precisely when the
    frame carries most of the mass; it reported a relative tail of 0 at the initial span.
    """
    p = Banana()
    fn = lambda z: np.stack([z[:, 1], z[:, 1] ** 2], axis=1)
    q = AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2), refine_fn=fn, max_cells=40000)
    mu, _ = q.moments(fn)

    d = q.diagnostics()
    assert d["span_sd"] > 7.0, "the rectangle never grew, so the tail check did not fire"
    assert d["expansions"][0]["frame_relative"] > 0.1, \
        "the first frame should carry a large fraction; a coarse rule would report ~0 here"
    assert d["tail_relative"] <= 1e-8

    ref, _, edge = brute(p, fn, (-16.0, -10.0), (16.0, 100.0), 4001)
    assert edge < 1e-12
    assert np.isclose(mu[1], ref[1], rtol=2e-3), (mu[1], ref[1])


def test_a_low_order_reports_convergence_at_a_grossly_wrong_value():
    """The rule's own failure mode, asserted rather than assumed.

    At six Gauss-Legendre points per axis the embedded error estimate is small because the cells are
    badly resolved, so the rule stops early and reports `mesh_converged` at a value that is wrong by
    a factor of three. This is why the reference is the agreement of several orders and never one
    setting.
    """
    p = Banana()
    fn = lambda z: np.stack([z[:, 1], z[:, 1] ** 2], axis=1)
    # Budget chosen by measurement, not by caution: at rtol 1e-10 and 60000 cells these two
    # tests took 160 s between them, and every conclusion below is identical at 1e-7 and 8000
    # cells, which takes 10 s. The low order is wrong by 70% and the order spread exceeds the
    # threshold at both settings, so the cheaper one loses nothing.
    kw = dict(refine_fn=fn, rtol=1e-7, max_cells=8000)
    lo = AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2), n_gl=6, n_gl_low=3, **kw)
    hi = AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2), n_gl=14, n_gl_low=7, **kw)
    v_lo, v_hi = float(lo.moments(fn)[0][1]), float(hi.moments(fn)[0][1])

    assert lo.diagnostics()["mesh_converged"], "the low order is supposed to CLAIM convergence"
    assert abs(v_lo - v_hi) > 0.5 * abs(v_hi), (v_lo, v_hi)


def test_the_order_gate_refuses_to_certify_a_density_it_cannot_resolve():
    """On the banana even orders 10, 12 and 14 disagree, and the gate must say so.

    The disagreement is not in the mesh but in the domain: the expansion stops at a span of 73.4
    standard deviations at order 10 and 45.9 at orders 12 and 14, because the frame integral that
    decides when to stop is itself computed at the order under test. On a density whose support runs
    far beyond the initial rectangle, the stopping span can therefore depend on the order.

    This is the gate working. The campaign posteriors are nothing like this -- there the three-order
    agreement is 5.5e-8 across all 72 development histories -- and the point of the test is that a
    density the rule cannot resolve is detected rather than certified.
    """
    p = Banana()
    fn = lambda z: np.stack([z[:, 1], z[:, 1] ** 2], axis=1)
    kw = dict(refine_fn=fn, rtol=1e-7, max_cells=8000)
    v = [float(AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2),
                                  n_gl=n, n_gl_low=n // 2, **kw).moments(fn)[0][1])
         for n in (10, 12, 14)]
    spread = max(v) - min(v)
    assert spread / max(abs(x) for x in v) > 5e-3, \
        "the orders agree here, so this density no longer exercises the gate"


def test_the_order_gate_certifies_a_density_it_can_resolve():
    """The other side of the gate: on a well-behaved mixture the orders must agree closely."""
    p = AsymmetricBimodal()
    g = lambda z: np.stack([z[:, 0], z[:, 1]], axis=1)
    rf = lambda z: (lambda G: np.concatenate([G, G ** 2], axis=1))(g(z))
    v = [float(AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2), refine_fn=rf,
                                  n_gl=n, n_gl_low=n // 2, rtol=1e-10,
                                  max_cells=40000).moments(g)[1].sum())
         for n in (10, 12, 14)]
    spread = (max(v) - min(v)) / max(abs(x) for x in v)
    assert spread < 1e-8, (v, spread)


def test_refinement_on_the_first_moment_alone_under_resolves_the_second():
    """`V` is a second moment, so the refinement must be driven by `g` and `g**2`.

    Driving it by `g` alone leaves the variance under-resolved, which is the defect this test
    exists to keep out. The comparison is against a brute-force grid, not against another setting.
    """
    p = AsymmetricBimodal()
    g = lambda z: np.stack([z[:, 0], z[:, 1]], axis=1)
    g_and_g2 = lambda z: (lambda G: np.concatenate([G, G ** 2], axis=1))(g(z))

    box_lo, box_hi = (-8.0, -8.0), (8.0, 8.0)
    _, ref_var, edge = brute(p, g, box_lo, box_hi, 3001)
    assert edge < 1e-8, edge          # the grid's own boundary ring, not the rule's tail

    kw = dict(n_gl=8, n_gl_low=4, rtol=1e-6, max_cells=4000)
    v_first = AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2), refine_fn=g, **kw).moments(g)[1]
    v_both = AdaptiveQuadrature(p, np.zeros(2), 9.0 * np.eye(2),
                                refine_fn=g_and_g2, **kw).moments(g)[1]
    err_first = float(np.max(np.abs(v_first - ref_var) / np.abs(ref_var)))
    err_both = float(np.max(np.abs(v_both - ref_var) / np.abs(ref_var)))
    assert err_both <= err_first + 1e-12, (err_both, err_first)
    assert err_both < 1e-6, err_both


def test_diagnostics_separate_tail_truncation_from_mesh_resolution():
    """The two error sources are reported apart, because they are fixed by different actions."""
    q = AdaptiveQuadrature(Gaussian([0.0, 0.0], np.eye(2)), np.zeros(2), 4.0 * np.eye(2))
    d = q.diagnostics()
    for k in ("tail_relative", "tail_unresolved_relative", "mesh_outstanding_relative",
              "mesh_converged", "span_sd", "n_cells", "refinements"):
        assert k in d, k
    assert d["tail_relative"] <= 1e-8
    assert d["mesh_converged"]
    assert np.isclose(float(np.sum(q.w)), 1.0, atol=1e-12)


@pytest.mark.parametrize("n_gl", [8, 10, 12])
def test_the_weights_are_a_probability_distribution_at_every_order(n_gl):
    q = AdaptiveQuadrature(AsymmetricBimodal(), np.zeros(2), 9.0 * np.eye(2),
                           n_gl=n_gl, n_gl_low=n_gl // 2)
    assert np.all(q.w >= 0)
    assert np.isclose(float(q.w.sum()), 1.0, atol=1e-12)
    assert q.ess > 1.0
