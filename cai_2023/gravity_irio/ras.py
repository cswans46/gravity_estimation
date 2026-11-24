"""Biproportional (RAS) balancing, i.e. the doubly constrained gravity model solver."""
from __future__ import annotations

import numpy as np


def ras(seed, row_totals, col_totals, tol: float = 1e-10, maxit: int = 10_000,
        rescale: bool = True):
    """Find X = diag(r) S diag(s) with X 1 = row_totals and 1'X = col_totals.

    Rows/columns with zero target are zeroed. If totals are inconsistent and
    ``rescale`` is True, col_totals are scaled to sum to sum(row_totals).
    Returns (X, info) where info has 'iterations', 'converged', 'max_rel_err',
    'scale_factor' (col total rescaling applied).
    """
    S = np.array(seed, dtype=float)
    r = np.asarray(row_totals, dtype=float).copy()
    c = np.asarray(col_totals, dtype=float).copy()
    if np.any(S < 0) or np.any(r < 0) or np.any(c < 0):
        raise ValueError("RAS requires non-negative seed and totals")
    scale = 1.0
    if rescale and c.sum() > 0:
        scale = r.sum() / c.sum()
        c = c * scale
    S[r == 0, :] = 0.0
    S[:, c == 0] = 0.0
    X = S
    converged = False
    it = 0
    for it in range(1, maxit + 1):
        rs = X.sum(axis=1)
        X = X * np.divide(r, rs, out=np.zeros_like(r), where=rs > 0)[:, None]
        cs = X.sum(axis=0)
        X = X * np.divide(c, cs, out=np.zeros_like(c), where=cs > 0)[None, :]
        err = np.max(np.abs(X.sum(axis=1) - r)) / max(r.max(), 1e-300)
        if err < tol:
            converged = True
            break
    info = {"iterations": it, "converged": converged, "scale_factor": scale,
            "max_rel_err": float(np.max(np.abs(X.sum(axis=1) - r)) / max(r.max(), 1e-300)),
            "infeasible_rows": int(np.sum((r > 0) & (S.sum(axis=1) == 0))),
            "infeasible_cols": int(np.sum((c > 0) & (S.sum(axis=0) == 0)))}
    return X, info
