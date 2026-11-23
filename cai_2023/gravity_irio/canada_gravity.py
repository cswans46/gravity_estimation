"""Distance elasticities estimated directly from Statistics Canada's interprovincial flows.

Same specification as Cai (2023) eq. (9), applied to provinces instead of countries:

    log z_{ib,k,t} = theta_k log d^{(k)}_{ib} + phi_k b_ib + c_k l_ib + mu_{ik,t} + nu_{bk,t} + eps

* z = T / (y e) with y, e the row/column totals of the (domestic) shipment matrix of
  sector k in year t (own-province supply on the diagonal, as in Cai).
* b_ib = 1 if i != b: the *provincial* border effect (within-country analogue of Cai's
  border dummy; interprovincial trade barriers).
* l_ib: common official language. Quebec's official language is French, New Brunswick is
  bilingual, every other province/territory English -> l = 0 only for Quebec with a
  province other than Quebec and New Brunswick (``canada_language_matrix``).
* d^{(k)}_{ib}: activity-weighted CES distance from ``business_locations`` (sector k's
  establishments in i to all users' establishments in b, or a user-share-weighted
  aggregate), or plain population-weighted distances.
* ``slope_by='origin'`` estimates theta_{k,i} (one elasticity per NAICS sector x origin
  province); 'destination' gives theta_{k,b}. Identification comes from the 12 other
  provinces each year x the panel years; expect wide intervals for small provinces.

Flows by commodity (SUPC) are mapped to producing sectors with the supply table
(``commodity_flows_to_sectors``) before estimation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cai import ols_hdfe
from .ppml import ppml_hdfe


def canada_language_matrix(provinces) -> np.ndarray:
    lang = {p: ({"fr"} if p == "QC" else {"en", "fr"} if p == "NB" else {"en"}) for p in provinces}
    return np.array([[float(bool(lang[a] & lang[b])) for b in provinces] for a in provinces])


def commodity_flows_to_sectors(T_icbt, comm_to_sector):
    """T[i, c, b, t] by commodity -> T[i, k, b, t] by producing sector, using commodity->sector
    weights (rows sum to 1), e.g. ``concordance.commodity_to_industry`` from the (provincial
    or national) supply table composed with the industry concordance."""
    return np.einsum("icbt,ck->ikbt", np.asarray(T_icbt, float), np.asarray(comm_to_sector, float))


def _design(T_ibt, dist_ib, lang, slope_by, R, provinces):
    Tn = T_ibt.shape[-1]
    i, b, t = (v.ravel() for v in np.meshgrid(np.arange(R), np.arange(R), np.arange(Tn), indexing="ij"))
    x = T_ibt[i, b, t]
    y = T_ibt.sum(axis=1)          # (i, t)
    e = T_ibt.sum(axis=0)          # (b, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = x / (y[i, t] * e[b, t])
    lnd = np.log(dist_ib[i, b])
    border = (i != b).astype(float)
    cols, names = [], []
    if slope_by is None:
        cols.append(lnd); names.append("lndist")
    else:
        g = i if slope_by == "origin" else b
        for r in range(R):
            cols.append(lnd * (g == r)); names.append(f"lndist_{provinces[r]}")
    cols += [border, lang[i, b]]
    names += ["border", "comlang_off"]
    X = np.column_stack(cols)
    fe = [i * Tn + t, b * Tn + t]
    return x, z, X, names, fe, i * R + b


def estimate_ipt_elasticities(T_ikbt, dist, provinces, sectors, estimator: str = "ols",
                              slope_by: str | None = None, drop=("NU",), min_positive: int = 30):
    """theta for every sector k from T[i, k, b, t].

    dist : (R, R) common distances, or (R, K, R) sector-specific (origin side),
           or (R, K, R, K) -> collapsed with equal user weights (pass a collapsed
           (R, K, R) from ``aggregate_user_distances`` to use SUT use shares).
    Returns a long DataFrame (sector, [region], theta, se, delta, border, language, n_obs).
    """
    provinces = list(provinces)
    keep = [n for n, p in enumerate(provinces) if p not in set(drop or ())]
    provs = [provinces[n] for n in keep]
    T = np.asarray(T_ikbt, float)[np.ix_(keep, range(T_ikbt.shape[1]), keep, range(T_ikbt.shape[3]))]
    D = np.asarray(dist, float)
    if D.ndim == 4:
        with np.errstate(invalid="ignore"):
            D = np.nanmean(D ** -1.0, axis=3) ** -1.0
    if D.ndim == 2:
        D = np.repeat(D[:, None, :], T.shape[1], axis=1)
    D = D[np.ix_(keep, range(D.shape[1]), keep)]
    R = len(provs)
    lang = canada_language_matrix(provs)
    rows = []
    for k, name in enumerate(sectors):
        Tk = T[:, k, :, :]
        if (Tk > 0).sum() < min_positive or not np.all(np.isfinite(D[:, k, :])):
            continue
        x, z, X, names, fe, cl = _design(Tk, D[:, k, :], lang, slope_by, R, provs)
        if estimator == "ols":
            ok = (x > 0) & np.isfinite(z)
            r = ols_hdfe(np.log(z[ok]), X[ok], [f[ok] for f in fe], names)
            coef, se, n = r.coef, r.se, r.n_obs
        else:
            ok = np.isfinite(z)
            r = ppml_hdfe(np.where(z[ok] > 0, z[ok], 0), X[ok], fe=[f[ok] for f in fe], names=names,
                          cluster=cl[ok])
            coef, se, n = r.coef, r.se, r.n_obs
        nd = len(names) - 2
        for q in range(nd):
            reg = None if slope_by is None else provs[q]
            rows.append(dict(sector=name, region=reg, theta=coef[q], se=se[q], delta=-coef[q],
                             border=coef[nd], se_border=se[nd], language=coef[nd + 1],
                             se_language=se[nd + 1], n_obs=n, estimator=estimator))
    return pd.DataFrame(rows)
