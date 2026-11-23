"""Exact replication of Cai (2023), "A calibrated gravity model of interregional trade".

Everything here follows Sections 2-3 of the paper (equation numbers refer to it).

Gravity (AvW 2004), eq. (1)-(3):   x_ijk = (y_ik e_jk / y_k) (t_ijk / (Pi_ik P_jk))^(1-s_k)
Trade costs, eq. (8):              t_ijk = g_k * tau_k^{b_ij} * lambda_k^{l_ij} * d_ij^{delta_k}
Estimating equation, eq. (9):
    log z_ijk = xi_k + theta_k log d_ij + phi_k b_ij + c_k l_ij + mu_ik + nu_jk + eps_ijk
    z_ijk = x_ijk / (y_ik e_jk),  y_ik = sum_j x_ijk,  e_jk = sum_i x_ijk  (row/column totals)
    b_ij = 1 if i != j (border), l_ij = 1 if common official language
    Estimator: "fixed effect estimator" (OLS with exporter and importer FE), product by
    product, zero flows dropped, heteroskedasticity-robust SEs; distance = CEPII distwces
    (internal distance on the diagonal, so d_ii > 0).
    Data in the paper: WIOD 2009 (2013 release), 27 EU countries, flows summed over all users.
Calibration, eq. (4)-(6):          X_k = <a_k> F_k <b_k>,  F_k = [d_ij^theta_k]
    i.e. RAS of the seed F_k to the regional row totals y_k and column totals e_k.
    Border and language terms are constant within a country and drop out (Sect. 3.4).
Accuracy (Sect. 3.5): STPE = sum|xhat-x| / sum x * 100;  DTPE = (sum_i xhat_ii - sum_i x_ii) / sum_i x_ii * 100.

The panel extension (this project) replaces mu_ik, nu_jk with mu_ik,t and nu_jk,t and pools
the years with a common theta_k. ``year=...`` reproduces Cai's single cross-section.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ppml import _demean, _factorize, ppml_hdfe
from .ras import ras

EU27_PRE_BREXIT_EX_HRV = [  # Cai's sample: pre-Brexit EU member states excluding Croatia
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HUN",
    "IRL", "ITA", "LTU", "LUX", "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE"]


@dataclass
class OLSResult:
    coef: np.ndarray
    se: np.ndarray
    names: list
    n_obs: int
    r2_within: float
    vcov: np.ndarray

    def get(self, name):
        i = self.names.index(name)
        return self.coef[i], self.se[i]


def ols_hdfe(y, X, fe, names, robust: str = "HC1", cluster=None) -> OLSResult:
    """OLS with absorbed fixed effects (exact two-way projection) and robust SEs.

    HC1 uses the reghdfe/areg small-sample factor n / (n - p - sum(G_f) + (#FE - 1)).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    codes, ng = [], []
    for f in fe:
        c, g = _factorize(f)
        codes.append(c)
        ng.append(g)
    w = np.ones(len(y))
    M = _demean(np.column_stack([y, X]), codes, ng, w)
    yt, Xt = M[:, 0], M[:, 1:]
    XtX = Xt.T @ Xt
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ Xt.T @ yt
    e = yt - Xt @ b
    n, p = Xt.shape
    if cluster is not None:
        cl, G = _factorize(np.asarray(cluster))
        S = np.zeros((G, p))
        np.add.at(S, cl, Xt * e[:, None])
        V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - p - sum(ng) + len(ng) - 1))
    else:
        meat = (Xt * e[:, None] ** 2).T @ Xt
        dof = n - p - sum(ng) + max(len(ng) - 1, 0)
        V = XtXi @ meat @ XtXi * (n / max(dof, 1) if robust == "HC1" else 1.0)
    r2w = 1 - (e @ e) / max(yt @ yt, 1e-300)
    return OLSResult(b, np.sqrt(np.clip(np.diag(V), 0, None)), list(names), n, float(r2w), V)


def cai_regressors(bil, a, b):
    """Cai eq. (9) regressors for origin index a and destination index b arrays."""
    lnd = np.log(bil["dist"][a, b])
    border = 1.0 - bil["home"][a, b]
    lang = bil["comlang_off"][a, b].copy() if "comlang_off" in bil else np.zeros(len(a))
    # CEPII gives comlang_off = 0 on the diagonal blocks after alignment; a country shares its
    # own official language with itself.
    lang = np.where(bil["home"][a, b] == 1, 1.0, lang)
    return np.column_stack([lnd, border, lang]), ["lndist", "border", "comlang_off"]


def cai_estimate(F_tij, bil, countries, sample=None, years=None, estimator: str = "ols",
                 cluster_pairs: bool = False):
    """Estimate Cai's eq. (9) for one product on a (t, i, j) array of flows x_ij,t.

    sample  : list of region codes to keep (Cai used EU27_PRE_BREXIT_EX_HRV); None = all valid
    years   : list of year indices; one year = Cai's cross-section, several = panel with
              origin-year and destination-year FE
    estimator : 'ols' (Cai: log z, zeros dropped, robust SE) or 'ppml' (robustness)
    Returns dict with theta (the distance coefficient, negative), se, border, language, n.
    """
    F = np.asarray(F_tij, float)
    if years is not None:
        F = F[list(years)]
    T, C, _ = F.shape
    keep = np.asarray(bil["valid"], bool).copy()
    if sample is not None:
        keep &= np.isin(np.asarray(countries), sample)
    idx = np.flatnonzero(keep)
    t, a, b = (v.ravel() for v in np.meshgrid(np.arange(T), idx, idx, indexing="ij"))
    x = F[t, a, b]
    # y_ik = row totals, e_jk = column totals of the (sample) trade matrix, by year
    sub = F[:, idx][:, :, idx]
    y = sub.sum(axis=2)          # (t, i_sub)
    e = sub.sum(axis=1)          # (t, j_sub)
    pos = {c: n for n, c in enumerate(idx)}
    ai = np.array([pos[v] for v in a])
    bi = np.array([pos[v] for v in b])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = x / (y[t, ai] * e[t, bi])
    X, names = cai_regressors(bil, a, b)
    fe = [a * T + t, b * T + t]
    cl = a * C + b if cluster_pairs else None
    if estimator == "ols":
        ok = (x > 0) & np.isfinite(z) & np.all(np.isfinite(X), axis=1)
        res = ols_hdfe(np.log(z[ok]), X[ok], [f[ok] for f in fe], names,
                       cluster=None if cl is None else cl[ok])
        out = dict(theta=res.coef[0], se=res.se[0], border=res.coef[1], se_border=res.se[1],
                   language=res.coef[2], se_language=res.se[2], n_obs=res.n_obs,
                   n_zero_dropped=int((~ok).sum()), r2_within=res.r2_within)
    else:
        ok = np.isfinite(z) & np.all(np.isfinite(X), axis=1)
        r = ppml_hdfe(np.where(z[ok] > 0, z[ok], 0.0), X[ok], fe=[f[ok] for f in fe], names=names,
                      cluster=(cl[ok] if cl is not None else a[ok] * C + b[ok]))
        out = dict(theta=r.coef[0], se=r.se[0], border=r.coef[1], se_border=r.se[1],
                   language=r.coef[2], se_language=r.se[2], n_obs=r.n_obs, n_zero_dropped=0,
                   r2_within=np.nan)
    out["delta"] = -out["theta"]  # positive decay convention used elsewhere in the package
    return out


def cai_by_product(F_tikj, bil, countries, products, **kw) -> pd.DataFrame:
    """Run ``cai_estimate`` for each product k of a (t, i, k, j) array (Cai Table 1 layout)."""
    rows = []
    for k, name in enumerate(products):
        r = cai_estimate(F_tikj[:, :, k, :], bil, countries, **kw)
        rows.append(dict(product=name, **r))
    return pd.DataFrame(rows)


def cai_calibrate(y, e, dist, theta, **ras_kw):
    """Cai eq. (4)-(6): X = RAS(seed d^theta; rows y, cols e). theta is negative (Cai's sign)."""
    seed = np.asarray(dist, float) ** float(theta)
    return ras(seed, y, e, **ras_kw)


def stpe(xhat, x):
    x = np.asarray(x, float)
    return float(np.abs(np.asarray(xhat) - x).sum() / x.sum() * 100)


def dtpe(xhat, x):
    dx, dh = np.trace(np.asarray(x, float)), np.trace(np.asarray(xhat, float))
    return float((dh - dx) / dx * 100)
