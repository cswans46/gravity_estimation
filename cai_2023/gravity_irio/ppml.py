"""Poisson pseudo-maximum likelihood (PPML) with high-dimensional fixed effects.

Implements the IRLS algorithm of Correia, Guimarães & Zylkin (2020, "ppmlhdfe"):
each IRLS step is a weighted least-squares regression of the working variable on
the regressors and the fixed effects, and the fixed effects are partialled out with
weighted alternating projections (method of alternating projections, MAP).

Only numpy is required. Typical use in this project is a gravity regression

    E[F_{ij,t}] = exp(alpha_{i,t} + gamma_{j,t} - delta * ln d_ij + controls_ij * b)

with origin-year and destination-year fixed effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PPMLResult:
    coef: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    names: list[str]
    n_obs: int
    n_dropped: int
    n_clusters: int | None
    converged: bool
    iterations: int
    deviance: float
    mu: np.ndarray = field(repr=False)
    keep: np.ndarray = field(repr=False)  # boolean mask of obs used (relative to input)

    def summary(self) -> str:
        lines = [f"{'':<16}{'coef':>12}{'se':>12}{'z':>10}"]
        for n, b, s in zip(self.names, self.coef, self.se):
            lines.append(f"{n:<16}{b:>12.4f}{s:>12.4f}{b / s if s > 0 else np.nan:>10.2f}")
        lines.append(f"N={self.n_obs} (dropped {self.n_dropped}), clusters={self.n_clusters}, "
                     f"converged={self.converged} in {self.iterations} it")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"coef": dict(zip(self.names, self.coef)), "se": dict(zip(self.names, self.se)),
                "n_obs": self.n_obs, "converged": self.converged}


def _factorize(g) -> tuple[np.ndarray, int]:
    codes, uniques = np.unique(np.asarray(g), return_inverse=True)
    return uniques.astype(np.int64), len(codes)


def _drop_all_zero_groups(y: np.ndarray, fes: list[np.ndarray]) -> np.ndarray:
    """Iteratively drop observations in FE groups whose y are all zero.

    Such groups push the FE to -inf (perfect prediction of zeros) and contribute
    nothing to the slope estimates (see Correia et al. 2019, separation)."""
    keep = np.ones(len(y), dtype=bool)
    while True:
        changed = False
        for g in fes:
            sums = np.bincount(g[keep], weights=y[keep], minlength=g.max() + 1)
            bad = sums[g] <= 0
            new_keep = keep & ~bad
            if new_keep.sum() != keep.sum():
                keep, changed = new_keep, True
        if not changed:
            return keep


def _demean_two_way(M: np.ndarray, g: np.ndarray, h: np.ndarray, ng: int, nh: int,
                    w: np.ndarray) -> np.ndarray:
    """Exact weighted projection on two FE sets (a_g + b_h) via the Schur complement.

    Normal equations  [Da N; N' Db] [a; b] = [ra; rb],  N_gh = sum of w in cell (g, h).
    Eliminating a:    (Db - N' Da^-1 N) b = rb - N' Da^-1 ra.
    The system is singular by one dimension per connected component (a+c, b-c); a tiny
    ridge selects one solution without affecting fitted values. For origin-year x
    destination-year FE the matrix is block diagonal by year, so the sparse LU is cheap.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    Da = np.bincount(g, weights=w, minlength=ng)
    Db = np.bincount(h, weights=w, minlength=nh)
    N = sparse.csr_matrix((w, (g, h)), shape=(ng, nh))
    Dainv = sparse.diags(1.0 / np.where(Da > 0, Da, 1.0))
    S = sparse.diags(Db) - (N.T @ Dainv @ N)
    ridge = 1e-11 * max(Db.mean(), 1e-300)
    lu = splu((S + ridge * sparse.eye(nh)).tocsc())
    WM = M * w[:, None]
    ra = np.stack([np.bincount(g, weights=WM[:, c], minlength=ng) for c in range(M.shape[1])], 1)
    rb = np.stack([np.bincount(h, weights=WM[:, c], minlength=nh) for c in range(M.shape[1])], 1)
    b = lu.solve(rb - N.T @ (Dainv @ ra))
    a = Dainv @ (ra - N @ b)
    return M - a[g] - b[h]


def _demean(M: np.ndarray, fes: list[np.ndarray], ngroups: list[int], w: np.ndarray,
            tol: float = 1e-10, maxit: int = 10_000) -> np.ndarray:
    """Residualise columns of M on all FE sets (exact for two FE sets, MAP otherwise)."""
    M = M.copy()
    if not fes:
        return M
    if len(fes) == 2 and min(ngroups) > 0:
        return _demean_two_way(M, fes[0], fes[1], ngroups[0], ngroups[1], w)
    wsum = [np.bincount(g, weights=w, minlength=n) for g, n in zip(fes, ngroups)]
    for _ in range(maxit):
        M_old = M.copy()
        for g, n, ws in zip(fes, ngroups, wsum):
            for c in range(M.shape[1]):
                means = np.bincount(g, weights=w * M[:, c], minlength=n) / np.where(ws > 0, ws, 1.0)
                M[:, c] -= means[g]
        if len(fes) == 1:
            break
        diff = np.max(np.abs(M - M_old)) / (1.0 + np.max(np.abs(M_old)))
        if diff < tol:
            break
    return M


def ppml_hdfe(y, X, fe=(), names=None, cluster=None, tol: float = 1e-9, maxit: int = 200,
              demean_tol: float = 1e-10, verbose: bool = False) -> PPMLResult:
    """PPML with any number of (absorbed) fixed-effect dimensions.

    Parameters
    ----------
    y : (n,) non-negative outcome (levels, zeros allowed).
    X : (n, p) regressors (no constant: it is absorbed by the FE; if no FE are given a
        constant is added automatically).
    fe : sequence of (n,) group identifiers (any hashable dtype).
    cluster : (n,) cluster identifiers for cluster-robust SEs (e.g. country pair).
              If None, heteroskedasticity-robust (Eicker-Huber-White) SEs are used.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    names = list(names) if names is not None else [f"x{i}" for i in range(p)]
    fes_raw = [np.asarray(f) for f in fe]

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y >= 0)
    fes_codes = [_factorize(f)[0] for f in fes_raw]
    keep = ok.copy()
    if fes_codes:
        sub = _drop_all_zero_groups(y[ok], [f[ok] for f in fes_codes])
        keep[np.flatnonzero(ok)[~sub]] = False

    yk, Xk = y[keep], X[keep]
    fes = []
    ngroups = []
    for f in fes_raw:
        c, ng = _factorize(f[keep])
        fes.append(c)
        ngroups.append(ng)
    if not fes:
        Xk = np.column_stack([np.ones(len(yk)), Xk])
        names = ["const"] + names

    # Rescale y for numerical stability (PPML slopes are invariant to scaling y).
    scale = yk.mean() if yk.mean() > 0 else 1.0
    ys = yk / scale

    mu = 0.5 * (ys + ys.mean())
    eta = np.log(mu)
    beta = np.zeros(Xk.shape[1])
    dev_old = np.inf
    converged = False
    for it in range(1, maxit + 1):
        z = eta + (ys - mu) / mu
        w = mu
        M = _demean(np.column_stack([z, Xk]), fes, ngroups, w, tol=demean_tol)
        zt, Xt = M[:, 0], M[:, 1:]
        XtW = Xt.T * w
        beta = np.linalg.solve(XtW @ Xt, XtW @ zt)
        resid = zt - Xt @ beta
        eta = z - resid
        eta = np.clip(eta, -700, 700)
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2.0 * np.sum(np.where(ys > 0, ys * np.log(ys / mu), 0.0) - (ys - mu))
        if verbose:
            print(f"it {it}: deviance {dev:.6g}")
        if np.abs(dev - dev_old) / (0.1 + np.abs(dev)) < tol:
            converged = True
            break
        dev_old = dev

    # Sandwich variance using FE-partialled regressors at the solution.
    w = mu
    Xt = _demean(Xk, fes, ngroups, w, tol=demean_tol)
    H = (Xt.T * w) @ Xt
    Hinv = np.linalg.pinv(H)
    score = Xt * (ys - mu)[:, None]
    n_clusters = None
    if cluster is not None:
        cl, G = _factorize(np.asarray(cluster)[keep])
        S = np.zeros((G, Xt.shape[1]))
        np.add.at(S, cl, score)
        meat = S.T @ S * (G / max(G - 1, 1))
        n_clusters = G
    else:
        meat = score.T @ score
    V = Hinv @ meat @ Hinv
    return PPMLResult(coef=beta, se=np.sqrt(np.clip(np.diag(V), 0, None)), vcov=V, names=names,
                      n_obs=int(keep.sum()), n_dropped=int(n - keep.sum()), n_clusters=n_clusters,
                      converged=converged, iterations=it, deviance=float(dev),
                      mu=mu * scale, keep=keep)
