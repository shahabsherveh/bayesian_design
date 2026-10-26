"""Adaptive partition integration in two dimensions, as an independent reference.

This exists to check `bed.quadrature`, so it is built to fail differently. `ModeQuadrature` finds
modes, places Gaussian components on them and carries an importance ratio `p/q`; it fails when the
mode search misses a peak, when a component is misweighted, or when a curved ridge defeats a
Gaussian proposal. Nothing here does any of that. There is no mode search, no Gaussian proposal and
no density ratio: a rectangle is partitioned into disjoint cells, each cell is integrated by a
tensor Gauss-Legendre rule, and a cell is subdivided when its own embedded error estimate says
subdividing would move the answer. Its failure modes are a rectangle that is too small and a mesh
that is too coarse, and both are measured rather than assumed away.

The two error sources are reported apart from each other.

* **Tail truncation** is what lies outside the rectangle. It is bounded by enlarging the rectangle
  and integrating the frame that is added, so the reported number is the measured contribution of
  the region the previous rectangle omitted, not a Gaussian tail bound.
* **Mesh resolution** is how much one more refinement moves the answer. It is the sum of the
  embedded error estimates over the leaf cells, and the refinement history records it as it falls.

A target-weighted integrand can need resolution where the posterior mass does not, since a sharp
source kernel makes a small region matter for the signal moments while carrying little mass.
Refinement is therefore driven by a vector of integrands supplied by the caller, not by the density
alone.

**Known limitation, found by `tests/adaptive_test.py`.** The frame integral that decides when to
stop enlarging the rectangle is itself computed at the order under test, so on a density whose
support runs far beyond the initial rectangle the stopping span can depend on that order. On a thin
curved ridge the expansion halts at 73.4 prior standard deviations at order 10 and at 45.9 at
orders 12 and 14, and the resulting second moments differ by about 1%. This is why a single setting
is never the answer: the reference is the agreement of three orders, which detects such a density
rather than certifying it. The posteriors in this study are nothing like that -- the three-order
agreement is 5.5e-8 across all 72 development histories -- but a caller integrating something
heavier-tailed should read the gate rather than assume it passes.
"""
from __future__ import annotations

import heapq

import numpy as np

_GL_CACHE = {}


def _gl(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def _tensor(lo, hi, n):
    """Tensor Gauss-Legendre nodes and weights on the rectangle [lo, hi] in two dimensions."""
    x, w = _gl(n)
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
    p0, p1 = mid[0] + half[0] * x, mid[1] + half[1] * x
    G0, G1 = np.meshgrid(p0, p1, indexing="ij")
    W = np.outer(w, w).ravel() * half[0] * half[1]
    return np.stack([G0.ravel(), G1.ravel()], axis=1), W


class _Cell:
    __slots__ = ("lo", "hi", "pts", "W", "val", "err", "depth")

    def __init__(self, lo, hi, pts, W, val, err, depth):
        self.lo, self.hi, self.pts, self.W = lo, hi, pts, W
        self.val, self.err, self.depth = val, err, depth

    def __lt__(self, other):      # heapq is a min-heap and we want the worst cell first
        return self.err > other.err


class AdaptiveQuadrature:
    """A leaf-cell node set and weights for a posterior, refined where refinement changes it.

    Exposes `nodes`, `w` and `moments(fn)` with the same meaning as `bed.quadrature.ModeQuadrature`,
    so the two can be swapped in any evaluator, but shares no machinery with it.
    """

    def __init__(self, post, mu0, S0, refine_fn=None, n_gl=10, n_gl_low=5, rtol=1e-7,
                 max_cells=12000, init_span=7.0, tail_tol=1e-8, max_expansions=6, min_depth=2):
        self.post = post
        mu0 = np.asarray(mu0, float)
        S0 = np.asarray(S0, float)
        sd = np.sqrt(np.maximum(np.diag(S0), 1e-30))
        self.n_gl, self.n_gl_low = n_gl, n_gl_low

        # A stable offset for exp(logp), from a coarse probe of the widest rectangle considered.
        probe, _ = _tensor(mu0 - init_span * sd, mu0 + init_span * sd, 32)
        off = float(np.max(np.asarray(post.logp(probe), float)))
        self.log_offset = off

        span = init_span
        self.expansion_log = []
        for it in range(max_expansions):
            lo, hi = mu0 - span * sd, mu0 + span * sd
            leaves, info = self._refine_box(post, refine_fn, lo, hi, rtol, max_cells,
                                            min_depth, off)
            # Tail check: integrate the frame that the NEXT larger rectangle would add, and report
            # its contribution relative to what is already inside. This measures the omitted region
            # instead of bounding it by an assumed shape.
            frame, frame_unresolved = self._frame_mass(post, refine_fn, lo, hi, mu0, sd,
                                                      span, 1.6, off)
            inside = info["totals"]
            scale = np.maximum(np.abs(inside), 1e-300)
            rel = float(np.max(np.abs(frame) / scale))
            # An unresolved frame is not an empty frame: carry its own outstanding error into the
            # decision, so a frame the refinement failed to resolve forces expansion rather than
            # passing as negligible.
            rel_unres = float(frame_unresolved / float(scale.max()))
            self.expansion_log.append({"span": float(span), "frame_relative": rel,
                                       "frame_unresolved_relative": rel_unres,
                                       "n_cells": info["n_cells"]})
            if max(rel, rel_unres) <= tail_tol:
                break
            span *= 1.6
        self.span = float(span)
        self.box = (lo, hi)
        self.tail_relative = rel
        self.tail_unresolved_relative = rel_unres
        self.refine_info = info

        nodes = np.concatenate([c.pts for c in leaves], axis=0)
        Wgeom = np.concatenate([c.W for c in leaves], axis=0)
        logp = np.asarray(post.logp(nodes), float).ravel()
        lw = np.log(np.maximum(Wgeom, 1e-300)) + logp
        mx = lw.max()
        w = np.exp(lw - mx)
        self.nodes = nodes
        self.mass_log = float(mx + np.log(w.sum()))
        self.w = w / w.sum()
        self.n_nodes = int(nodes.shape[0])
        self.n_cells = int(len(leaves))
        self.ess = float(1.0 / np.sum(self.w ** 2))
        self.max_node_weight = float(self.w.max())

    # ---- mesh construction
    @staticmethod
    def _eval(post, refine_fn, lo, hi, n_gl, n_gl_low, off):
        """Integrate [p, p*refine] over one cell at two rule orders; return (value, error, nodes)."""
        out = []
        for n in (n_gl, n_gl_low):
            pts, W = _tensor(lo, hi, n)
            p = np.exp(np.asarray(post.logp(pts), float).ravel() - off)
            if refine_fn is None:
                F = np.ones((pts.shape[0], 1))
            else:
                F = np.asarray(refine_fn(pts), float).reshape(pts.shape[0], -1)
            v = np.concatenate([[W @ p], W @ (p[:, None] * F)])
            out.append((v, pts, W))
        err = np.abs(out[0][0] - out[1][0])
        return out[0][0], err, out[0][1], out[0][2]

    def _refine_box(self, post, refine_fn, lo, hi, rtol, max_cells, min_depth, off):
        heap, totals, n_cells = [], None, 0
        m = 2 ** min_depth
        for i in range(m):
            for j in range(m):
                clo = np.array([lo[0] + (hi[0] - lo[0]) * i / m, lo[1] + (hi[1] - lo[1]) * j / m])
                chi = np.array([lo[0] + (hi[0] - lo[0]) * (i + 1) / m,
                                lo[1] + (hi[1] - lo[1]) * (j + 1) / m])
                v, e, pts, W = self._eval(post, refine_fn, clo, chi, self.n_gl, self.n_gl_low, off)
                totals = v if totals is None else totals + v
                heapq.heappush(heap, _Cell(clo, chi, pts, W, v, float(e.max()), min_depth))
                n_cells += 1

        history = []
        while n_cells < max_cells:
            # scale each component by the running total so a small component cannot dominate
            scale = np.maximum(np.abs(totals), 1e-300)
            err_tot = sum(c.err for c in heap)
            if err_tot <= rtol * float(scale.max()):
                break
            c = heapq.heappop(heap)
            mid = 0.5 * (c.lo + c.hi)
            kids, ksum = [], 0.0
            for i in range(2):
                for j in range(2):
                    klo = np.array([c.lo[0] if i == 0 else mid[0], c.lo[1] if j == 0 else mid[1]])
                    khi = np.array([mid[0] if i == 0 else c.hi[0], mid[1] if j == 0 else c.hi[1]])
                    v, e, pts, W = self._eval(post, refine_fn, klo, khi,
                                              self.n_gl, self.n_gl_low, off)
                    kids.append(_Cell(klo, khi, pts, W, v, float(e.max()), c.depth + 1))
            child_sum = sum(k.val for k in kids)
            totals = totals - c.val + child_sum
            for k in kids:
                heapq.heappush(heap, k)
            n_cells += 3
            history.append({"n_cells": n_cells,
                            "mass": float(totals[0]),
                            "step_change_rel": float(np.max(np.abs(child_sum - c.val)
                                                            / np.maximum(np.abs(totals), 1e-300))),
                            "outstanding_rel": float(sum(k.err for k in heap)
                                                     / max(float(np.max(np.abs(totals))), 1e-300))})
        outstanding = float(sum(c.err for c in heap))
        return list(heap), {"n_cells": n_cells, "totals": totals, "log_offset": off,
                            "outstanding_error": outstanding,
                            "outstanding_relative": outstanding
                            / max(float(np.max(np.abs(totals))), 1e-300),
                            "converged": bool(outstanding <= rtol
                                              * max(float(np.max(np.abs(totals))), 1e-300)),
                            "refinements": len(history), "history": history[-60:]}

    def _frame_mass(self, post, refine_fn, lo, hi, mu0, sd, span, grow, off, frame_cells=1200):
        """Integrate the frame between this rectangle and one `grow` times wider, adaptively.

        A single coarse rule here is worse than no check at all. On a thin curved ridge leaving the
        rectangle, a 16-point rule over a frame many standard deviations across steps over the
        ridge entirely and returns zero, so the tail diagnostic reports that nothing was omitted
        precisely when a lot was. The frame therefore gets the same embedded-error refinement as
        the interior, and the refinement's own outstanding error is returned so a frame that failed
        to resolve cannot be read as an empty frame.
        """
        LO, HI = mu0 - span * grow * sd, mu0 + span * grow * sd
        total, unresolved = None, 0.0
        boxes = [(np.array([LO[0], LO[1]]), np.array([HI[0], lo[1]])),
                 (np.array([LO[0], hi[1]]), np.array([HI[0], HI[1]])),
                 (np.array([LO[0], lo[1]]), np.array([lo[0], hi[1]])),
                 (np.array([hi[0], lo[1]]), np.array([HI[0], hi[1]]))]
        for blo, bhi in boxes:
            if np.any(bhi <= blo):
                continue
            _leaves, info = self._refine_box(post, refine_fn, blo, bhi, 1e-6,
                                             frame_cells, 3, off)
            total = info["totals"] if total is None else total + info["totals"]
            unresolved = max(unresolved, float(info["outstanding_error"]))
        if total is None:
            return np.zeros(1), 0.0
        return total, unresolved

    # ---- integrals
    def moments(self, fn):
        F = np.asarray(fn(self.nodes), float).reshape(self.nodes.shape[0], -1)
        m = self.w @ F
        v = self.w @ (F ** 2) - m ** 2
        return m, np.maximum(v, 0.0)

    def diagnostics(self):
        return {"reference": "adaptive_partition",
                "n_nodes": self.n_nodes, "n_cells": self.n_cells,
                "n_gl": self.n_gl, "n_gl_low": self.n_gl_low,
                "span_sd": self.span, "expansions": self.expansion_log,
                "tail_relative": float(self.tail_relative),
                "tail_unresolved_relative": float(self.tail_unresolved_relative),
                "mesh_outstanding_relative": float(self.refine_info["outstanding_relative"]),
                "mesh_converged": bool(self.refine_info["converged"]),
                "refinements": int(self.refine_info["refinements"]),
                "refinement_history": self.refine_info["history"],
                "effective_nodes": self.ess, "max_node_weight": self.max_node_weight,
                "log_unnormalised_evidence": self.mass_log}
