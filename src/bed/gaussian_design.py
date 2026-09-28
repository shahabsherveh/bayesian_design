"""Design utilities under a frozen Gaussian belief (numpy, data-free).

Belief ``z ~ N(m, S)`` over ``d`` parameters. A design ``i`` observes ``y_i = f_i(z) + e_i`` with
``e_i ~ N(0, R)``; linearised, ``f_i(z) ≈ f_i(m) + J_i (z - m)`` with ``J_i`` of shape ``(d_y, d)``.
A target ``j`` is a (linearised) functional ``g_j(z)`` with Jacobian ``Jt_j`` of shape ``(d_t, d)``,
observed with noise covariance ``Rt`` (``Rt = 0`` for a noise-free functional such as a biomarker).

Every quantity here is exact for the linear-Gaussian model and is the local Gaussian surrogate
otherwise. None of it depends on observed values, so design sequences can be planned before any
data are seen; that is the property Investigation 1 tests the applications against.

Conventions. ``J`` is ``(n, d_y, d)`` for ``n`` candidates, ``Jt`` is ``(m, d_t, d)`` for ``m``
targets. Scores are per candidate, ``(n,)``. Set utilities take a list of candidate indices
(repeats allowed) and return a float. "EPIG" means the *mean over targets* of the Gaussian mutual
information between the (set of) observation(s) and each target separately, which is the paper's
definition; the joint vector-target information is available as ``joint=True`` and is never mixed
with the mean-marginal one inside a comparison.
"""

from itertools import combinations, combinations_with_replacement

import numpy as np


# ---------------------------------------------------------------------------------------------
# covariance algebra
# ---------------------------------------------------------------------------------------------

def _sym(S):
    return (S + S.T) / 2


def _logdet(A):
    """log det of a (stack of) symmetric positive definite matrices via ``slogdet``; raises if the
    sign is not positive, so an invalid covariance is never hidden by an absolute value."""
    sign, ld = np.linalg.slogdet(A)
    if np.any(sign <= 0):
        raise np.linalg.LinAlgError("non-positive determinant in a covariance matrix")
    return ld


def _block_diag(R, k):
    """``k`` copies of ``R`` on the diagonal."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    return np.kron(np.eye(k), R)


def _stack(J, idx):
    """Stack the Jacobians of the designs ``idx`` (repeats allowed) into ``(k d_y, d)``."""
    J = np.asarray(J)
    return J[list(idx)].reshape(-1, J.shape[-1])


def kalman_update(S, J, R):
    """Posterior covariance after observing the stacked design(s) with Jacobian ``J`` ``(q, d)`` and
    noise covariance ``R`` ``(q, q)``: ``S - S J^T (J S J^T + R)^{-1} J S``, symmetrised."""
    J = np.asarray(J, dtype=float).reshape(-1, S.shape[0]); R = np.atleast_2d(np.asarray(R, dtype=float))
    Sx = J @ S @ J.T + R
    K = np.linalg.solve(Sx, J @ S).T          # S J^T Sx^{-1}
    return _sym(S - K @ J @ S)


def information_update(S, J, R):
    """The same posterior covariance in information form, ``(S^{-1} + J^T R^{-1} J)^{-1}``; used
    to cross-check :func:`kalman_update`."""
    J = np.asarray(J, dtype=float).reshape(-1, S.shape[0]); R = np.atleast_2d(np.asarray(R, dtype=float))
    return _sym(np.linalg.inv(np.linalg.inv(S) + J.T @ np.linalg.solve(R, J)))


def posterior_cov(S0, J, R, idx):
    """Posterior covariance after observing the designs ``idx`` (repeats allowed) once each."""
    idx = list(idx)
    if not idx:
        return np.asarray(S0, dtype=float).copy()
    return kalman_update(np.asarray(S0, dtype=float), _stack(J, idx), _block_diag(R, len(idx)))


def gaussian_mi(Sxx, Syy, Cyx):
    """Mutual information of jointly Gaussian ``x`` and ``y`` from the blocks ``Cov(x) = Sxx``,
    ``Cov(y) = Syy`` and ``Cov(y, x) = Cyx``, as ``½[log det Sxx + log det Syy − log det joint]``.
    An independent formula to the Schur-complement one used in the scores."""
    joint = np.block([[Sxx, Cyx.T], [Cyx, Syy]])
    return 0.5 * (_logdet(Sxx) + _logdet(Syy) - _logdet(joint))


# ---------------------------------------------------------------------------------------------
# per-candidate scores
# ---------------------------------------------------------------------------------------------

def predictive_cov(J, S, R):
    """``(n, d_y, d_y)`` predictive covariances ``J_i S J_i^T + R``."""
    return np.einsum("nid,de,nje->nij", J, S, J) + np.atleast_2d(np.asarray(R, dtype=float))


def epig_scores(J, Jt, S, R, Rt=0.0):
    """Closed-form EPIG per candidate: ``mean_j ½[log det S'_j − log det(S'_j − C_j S_x^{-1} C_j^T)]``
    with ``S_x = J_i S J_i^T + R``, ``S'_j = Jt_j S Jt_j^T + Rt`` and ``C_j = Jt_j S J_i^T``. ``Rt``
    may be ``0`` (noise-free targets): the Schur complement stays positive definite as long as the
    prior target covariance is, because ``S_x ≻ 0``."""
    J = np.asarray(J, dtype=float); Jt = np.asarray(Jt, dtype=float); S = np.asarray(S, dtype=float)
    Sx = predictive_cov(J, S, R)                                       # (n, dy, dy)
    Sp = np.einsum("mid,de,mje->mij", Jt, S, Jt)                       # (m, dt, dt)
    if np.ndim(Rt) or Rt != 0.0:
        Sp = Sp + np.atleast_2d(np.asarray(Rt, dtype=float))
    C = np.einsum("mid,de,nje->mnij", Jt, S, J)                        # (m, n, dt, dy)
    Sx_inv = np.linalg.inv(Sx)
    red = np.einsum("mnij,njk,mnlk->mnil", C, Sx_inv, C)               # C Sx^{-1} C^T
    post = Sp[:, None] - red
    return 0.5 * (_logdet(Sp)[:, None] - _logdet(post)).mean(axis=0)


def eig_scores(J, S, R):
    """Closed-form EIG per candidate under linearisation, ``½[log det S_x − log det R]``."""
    Sx = predictive_cov(np.asarray(J, dtype=float), np.asarray(S, dtype=float), R)
    return 0.5 * (_logdet(Sx) - _logdet(np.atleast_2d(np.asarray(R, dtype=float))))


def target_variance(S, Jt, Rt=0.0, per_target=False):
    """Predictive variance of the targets under ``S``: the trace of ``Jt_j S Jt_j^T + Rt`` per
    target, summed over targets (default) or per target (``per_target=True``). This is the loss-based
    objective (expected squared error of the posterior mean equals the posterior variance under a
    calibrated Gaussian belief)."""
    Jt = np.asarray(Jt, dtype=float)
    v = np.einsum("mid,de,mie->mi", Jt, S, Jt).sum(axis=1)
    if np.ndim(Rt) or Rt != 0.0:
        v = v + np.trace(np.atleast_2d(np.asarray(Rt, dtype=float)))
    return v if per_target else float(v.sum())


def variance_reduction_scores(J, Jt, S, R, Rt=0.0):
    """Loss-based score per candidate: the reduction in :func:`target_variance` (summed over
    targets) that observing the candidate once would give. The prediction-loss-matched
    variance-reduction baseline of the brief."""
    J = np.asarray(J, dtype=float); base = target_variance(S, Jt, Rt)
    return np.array([base - target_variance(kalman_update(S, J[i], R), Jt, Rt) for i in range(J.shape[0])])


# ---------------------------------------------------------------------------------------------
# set (batch) utilities: the whole set observed jointly
# ---------------------------------------------------------------------------------------------

def set_epig(idx, J, Jt, S, R, Rt=0.0, joint=False):
    """EPIG of the set ``idx`` observed jointly: ``mean_j I(Y_idx; y'_j)`` (``joint=False``, the
    mean-marginal target information) or ``I(Y_idx; Y'_all)`` (``joint=True``, the vector-target
    information). Both use the full joint covariance of the stacked observations; summing
    per-candidate scores would miss complementarity."""
    idx = list(idx)
    if not idx:
        return 0.0
    S = np.asarray(S, dtype=float); Jt = np.asarray(Jt, dtype=float)
    Jset = _stack(J, idx); Rset = _block_diag(R, len(idx))
    Sxx = Jset @ S @ Jset.T + Rset
    Rt2 = np.atleast_2d(np.asarray(Rt, dtype=float)) if (np.ndim(Rt) or Rt != 0.0) else None
    if joint:
        Jt_all = Jt.reshape(-1, Jt.shape[-1])
        Syy = Jt_all @ S @ Jt_all.T
        if Rt2 is not None:
            Syy = Syy + _block_diag(Rt2, Jt.shape[0])
        return gaussian_mi(Sxx, Syy, Jt_all @ S @ Jset.T)
    vals = []
    for j in range(Jt.shape[0]):
        Syy = Jt[j] @ S @ Jt[j].T
        if Rt2 is not None:
            Syy = Syy + Rt2
        vals.append(gaussian_mi(Sxx, Syy, Jt[j] @ S @ Jset.T))
    return float(np.mean(vals))


def set_eig(idx, J, S, R):
    """EIG of the set observed jointly, ``½[log det S_xx − log det R_set]``."""
    idx = list(idx)
    if not idx:
        return 0.0
    Jset = _stack(J, idx); Rset = _block_diag(R, len(idx)); S = np.asarray(S, dtype=float)
    return 0.5 * float(_logdet(Jset @ S @ Jset.T + Rset) - _logdet(Rset))


def set_variance_reduction(idx, J, Jt, S, R, Rt=0.0):
    """Reduction of the summed target variance by observing the set ``idx`` jointly."""
    return target_variance(S, Jt, Rt) - target_variance(posterior_cov(S, J, R, idx), Jt, Rt)


def set_utility(idx, crit, J, Jt, S, R, Rt=0.0):
    """Dispatch: ``crit`` in {"epig", "epig_joint", "eig", "var"}."""
    if crit == "epig":
        return set_epig(idx, J, Jt, S, R, Rt)
    if crit == "epig_joint":
        return set_epig(idx, J, Jt, S, R, Rt, joint=True)
    if crit == "eig":
        return set_eig(idx, J, S, R)
    if crit == "var":
        return set_variance_reduction(idx, J, Jt, S, R, Rt)
    raise ValueError(crit)


def complementarity(a, b, crit, J, Jt, S, R, Rt=0.0, base=()):
    """``F(base ∪ {a, b}) − F(base ∪ {a}) − F(base ∪ {b}) + F(base)``: positive means the pair is
    worth more together than apart (a supermodular pair, the calibration mechanism of
    Investigation 2); ``base`` is an already selected set."""
    base = list(base)
    F = lambda s: set_utility(s, crit, J, Jt, S, R, Rt)
    return F(base + [a, b]) - F(base + [a]) - F(base + [b]) + F(base)


# ---------------------------------------------------------------------------------------------
# beliefs for pseudo-Bayesian (prior-averaged) design
# ---------------------------------------------------------------------------------------------

class LinearBeliefs:
    """A list of linearisation points sharing the prior covariance: ``Js[k]`` ``(n, d_y, d)`` and
    ``Jts[k]`` ``(m, d_t, d)`` at draw ``k``. Scores and set utilities are averaged over draws
    (each draw keeps its own posterior covariance along a greedy path). One draw at the prior mean
    is the plain local design."""

    def __init__(self, Js, Jts, S0, R, Rt=0.0, weights=None):
        self.Js = [np.asarray(J, dtype=float) for J in Js]; self.Jts = [np.asarray(J, dtype=float) for J in Jts]
        self.S0 = np.asarray(S0, dtype=float); self.R = np.atleast_2d(np.asarray(R, dtype=float)); self.Rt = Rt
        self.K = len(self.Js); self.n = self.Js[0].shape[0]
        self.w = np.full(self.K, 1.0 / self.K) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)

    def scores(self, crit, Ss):
        """Averaged per-candidate scores given the per-draw current covariances ``Ss``."""
        out = np.zeros(self.n)
        for k in range(self.K):
            if crit == "epig":
                sc = epig_scores(self.Js[k], self.Jts[k], Ss[k], self.R, self.Rt)
            elif crit == "eig":
                sc = eig_scores(self.Js[k], Ss[k], self.R)
            elif crit == "var":
                sc = variance_reduction_scores(self.Js[k], self.Jts[k], Ss[k], self.R, self.Rt)
            else:
                raise ValueError(crit)
            out += self.w[k] * sc
        return out

    def update(self, Ss, i):
        return [kalman_update(Ss[k], self.Js[k][i], self.R) for k in range(self.K)]

    def utility(self, idx, crit):
        """Averaged set utility of ``idx`` from the prior."""
        return float(sum(self.w[k] * set_utility(idx, crit, self.Js[k], self.Jts[k], self.S0, self.R, self.Rt) for k in range(self.K)))

    def target_variance(self, idx, per_target=False):
        """Averaged target variance after observing ``idx`` (the loss-based terminal objective)."""
        vals = [target_variance(posterior_cov(self.S0, self.Js[k], self.R, idx), self.Jts[k], self.Rt, per_target) for k in range(self.K)]
        return np.tensordot(self.w, np.asarray(vals, dtype=float), axes=1)


# ---------------------------------------------------------------------------------------------
# design search
# ---------------------------------------------------------------------------------------------

def greedy(beliefs, crit, budget, warm=(), replace=True, rng=None):
    """One pass of greedy selection on the averaged score, starting after the (data-free) warm-start
    designs. Returns the selected indices (excluding the warm start) and the trace of the summed
    target variance (averaged over draws) after each selection, warm start included as entry 0."""
    Ss = [beliefs.S0.copy() for _ in range(beliefs.K)]; used = list(warm)
    for i in used:
        Ss = beliefs.update(Ss, i)
    trace = [float(np.sum(beliefs.target_variance(used)))]
    for _ in range(budget):
        if crit == "random":
            pool = np.arange(beliefs.n) if replace else np.setdiff1d(np.arange(beliefs.n), used)
            i = int((rng or np.random.default_rng(0)).choice(pool))
        else:
            sc = beliefs.scores(crit, Ss)
            if not replace:
                sc = sc.copy(); sc[used] = -np.inf
            i = int(np.argmax(sc))
        used.append(i); Ss = beliefs.update(Ss, i); trace.append(float(np.sum(beliefs.target_variance(used))))
    return used[len(warm):], np.array(trace)


def exchange_search(beliefs, crit, init, warm=(), replace=True, max_sweeps=20, rng=None, n_random_starts=0,
                    budget=None, tol=1e-10):
    """Coordinate-wise exchange (local search) on the averaged *set* utility ``F(warm ∪ set)``:
    every position of the set is in turn replaced by the candidate maximising ``F``, until a full
    sweep improves nothing. Starts from ``init`` (e.g. the greedy solution) and, optionally, from
    ``n_random_starts`` random sets of size ``budget``; returns the best set found, its utility and
    the utility after each sweep of each start (so the plateau is documented). Every returned
    utility is evaluated from the prior with the warm start included."""
    warm = list(warm); rng = rng or np.random.default_rng(0)
    F = lambda s: beliefs.utility(warm + list(s), crit)
    starts = [list(init)]
    for _ in range(n_random_starts):
        k = len(init) if budget is None else budget
        pool = np.arange(beliefs.n)
        starts.append([int(v) for v in (rng.choice(pool, k, replace=True) if replace else rng.choice(pool, k, replace=False))])
    best, best_val, traces = None, -np.inf, []
    for s in starts:
        cur = list(s); val = F(cur); trace = [val]
        for _ in range(max_sweeps):
            improved = False
            for pos in range(len(cur)):
                cand_best, cand_val = cur[pos], val
                for i in range(beliefs.n):
                    if i == cur[pos] or (not replace and i in cur):
                        continue
                    trial = cur.copy(); trial[pos] = i; v = F(trial)
                    if v > cand_val + tol:
                        cand_best, cand_val = i, v
                if cand_best != cur[pos]:
                    cur[pos] = cand_best; val = cand_val; improved = True
            trace.append(val)
            if not improved:
                break
        traces.append(trace)
        if val > best_val:
            best, best_val = cur, val
    return best, float(best_val), traces


def enumerate_sets(n, k, replace=True):
    """All candidate sets of size ``k`` (multisets if ``replace``)."""
    return combinations_with_replacement(range(n), k) if replace else combinations(range(n), k)


def exhaustive_best(beliefs, crit, k, warm=(), replace=True, top=5):
    """Exhaustive optimum of the averaged set utility over all sets of size ``k``; returns the best
    set, its utility and the ``top`` best (set, utility) pairs. Only for small ``n`` and ``k``."""
    warm = list(warm); vals = []
    for s in enumerate_sets(beliefs.n, k, replace):
        vals.append((beliefs.utility(warm + list(s), crit), tuple(s)))
    vals.sort(reverse=True)
    return list(vals[0][1]), float(vals[0][0]), vals[:top]
