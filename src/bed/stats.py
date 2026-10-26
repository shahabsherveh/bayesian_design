"""Paired statistics with the simulated subject as the independent unit.

Every terminal or common-history comparison in the investigations is paired: the same subject
(truth draw, warm start, noise table) is run under every arm, and a subject may contribute several
histories (steps, policies). The resampling unit is therefore the subject, never the history: a
bootstrap replicate draws subjects with replacement and keeps every row of a drawn subject. With
one row per subject this is the ordinary paired bootstrap.
"""
import numpy as np


def _groups(clusters):
    clusters = np.asarray(clusters); _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable"); counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return order, starts, counts


def cluster_bootstrap(stat, arrays, clusters, n_boot=4000, seed=0, alpha=0.05):
    """``stat(*resampled_arrays) -> float`` evaluated on ``n_boot`` cluster-bootstrap replicates of
    the row arrays ``arrays`` (all of length ``n``) with cluster labels ``clusters``; returns
    ``[point, lower, upper]`` with the point estimate on the full data and percentile limits."""
    arrays = [np.asarray(a) for a in arrays]; n = arrays[0].shape[0]
    order, starts, counts = _groups(clusters); C = counts.size; rng = np.random.default_rng(seed)
    point = float(stat(*arrays)); reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, C, C)
        idx = np.concatenate([order[starts[c]:starts[c] + counts[c]] for c in pick])
        reps[b] = stat(*[a[idx] for a in arrays])
    return [point, float(np.percentile(reps, 100 * alpha / 2)), float(np.percentile(reps, 100 * (1 - alpha / 2)))]


def mean_ci(x, clusters, **kw):
    return cluster_bootstrap(lambda a: a.mean(), [x], clusters, **kw)


def ratio_ci(a, b, clusters, **kw):
    """Ratio of means ``mean(a) / mean(b)`` (e.g. Bayes-risk ratio of two arms), paired by row."""
    return cluster_bootstrap(lambda u, v: u.mean() / v.mean(), [a, b], clusters, **kw)


def diff_ci(a, b, clusters, **kw):
    """Difference of means ``mean(a) - mean(b)`` (absolute paired excess), paired by row."""
    return cluster_bootstrap(lambda u, v: u.mean() - v.mean(), [a, b], clusters, **kw)


def noninferiority(ratio, margin):
    """Verdict of a noninferiority comparison from a ratio interval ``[point, lo, hi]`` of the
    cheaper arm's risk over the comparator's: ``"noninferior"`` if the upper limit is below
    ``1 + margin``, ``"inferior"`` if the lower limit is above 1, otherwise ``"inconclusive"``. An
    interval that crosses one is inconclusive, not evidence of sufficiency."""
    _, lo, hi = ratio
    if hi < 1.0 + margin:
        return "noninferior"
    if lo > 1.0:
        return "inferior"
    return "inconclusive"
