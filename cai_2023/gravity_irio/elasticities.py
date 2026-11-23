"""Panel estimation of distance elasticities from the ICIO.

Estimating equation (structural gravity, PPML, Santos Silva & Tenreyro 2006):

    E[y_{ij,t}] = exp( alpha_{i,t} + gamma_{j,t} - delta * ln d_ij + b_home * home_ij
                       + sum_c b_c * control_{c,ij} )

y is either the flow level F_{ij,t} or Cai's normalised flow z_{ij,t} = F/(Y_i E_j)
(``dep="z"``). Because Y_{i,t} and E_{j,t} are absorbed by the origin-year and
destination-year FE, both give consistent estimates of delta; they differ only in the
implicit observation weights (PPML on z down-weights flows of large countries).

Two levels of disaggregation:
  * by supplying industry k           -> delta_k        (Cai 2023, 50 regressions)
  * by supplier k x user m (+ FD)     -> delta_{k,m}    (50 x 51 regressions)

delta_{k,m} is what the Canadian disaggregation needs: with delta_k only, the
gravity-RAS split across *demanding* sectors collapses to import proportionality
(proof in docs/METHODS.md, section 3.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .icio import pair_flow, sector_flows
from .ppml import ppml_hdfe


def _long_panel(Y_tij, bil, use, controls):
    nT, C, _ = Y_tij.shape
    idx = np.flatnonzero(use)
    t, a, b = np.meshgrid(np.arange(nT), idx, idx, indexing="ij")
    t, a, b = t.ravel(), a.ravel(), b.ravel()
    y = Y_tij[t, a, b].astype(float)
    cols = [np.log(bil["dist"][a, b])]
    names = ["lndist"]
    if "home" in controls:
        cols.append(bil["home"][a, b]); names.append("home")
    for c in controls:
        if c != "home":
            cols.append(bil[c][a, b]); names.append(c)
    X = np.column_stack(cols)
    fe = [a * nT + t, b * nT + t]
    cluster = a * C + b
    return y, X, names, fe, cluster


def gravity_panel(Y_tij, bil, controls=("home", "contig", "comlang_off", "colony"),
                  use=None, years=None, cluster=True, **ppml_kw):
    """One PPML gravity regression on a (t, i, j) panel. Returns PPMLResult.

    The distance elasticity is ``-coef['lndist']``.
    """
    Y = np.asarray(Y_tij, dtype=float)
    if years is not None:
        Y = Y[list(years)]
    if use is None:
        use = bil["valid"] & np.all(np.isfinite(bil["dist"]) | ~bil["valid"][None, :], axis=1)
    controls = [c for c in controls if c == "home" or c in bil]
    y, X, names, fe, cl = _long_panel(Y, bil, use, controls)
    n_neg = int(np.sum(y < 0))
    y = np.where(y < 0, 0.0, y)  # e.g. negative inventory changes in FD
    res = ppml_hdfe(y, X, fe=fe, names=names, cluster=cl if cluster else None, **ppml_kw)
    res.n_negative_clipped = n_neg
    return res


def _row(res, **labels):
    d = dict(labels)
    for n, b, s in zip(res.names, res.coef, res.se):
        d[n] = b
        d[f"se_{n}"] = s
    d["delta"] = -d["lndist"]
    d["se_delta"] = d["se_lndist"]
    d.update(n_obs=res.n_obs, n_pos=int((res.mu > 0).sum()), converged=res.converged,
              n_neg_clipped=getattr(res, "n_negative_clipped", 0))
    return d


def estimate_sector_elasticities(Z5, FD4, X3, bil, industries, dep: str = "F",
                                 include_final_demand: bool = True, by_year: bool = False,
                                 years=None, **kw) -> pd.DataFrame:
    """Cai (2023)-style delta_k for each supplying industry, pooled over the panel."""
    s = sector_flows(Z5, FD4, X3, include_final_demand=include_final_demand)
    Yall = s["z"] if dep == "z" else s["F"]          # (t,i,k,j)
    rows = []
    nT = Yall.shape[0]
    for k, name in enumerate(industries):
        Ytij = Yall[:, :, k, :]
        if by_year:
            for t in range(nT):
                r = gravity_panel(Ytij[t:t + 1], bil, **kw)
                rows.append(_row(r, industry=name, year=(years[t] if years else t)))
        else:
            r = gravity_panel(Ytij, bil, **kw)
            rows.append(_row(r, industry=name))
    return pd.DataFrame(rows)


def _one_pair(Z5, FD4, X3, bil, k, m, dep, min_positive, kw):
    F, z = pair_flow(Z5, FD4, X3, k, m)
    Y = z if dep == "z" else F
    n_pos = int(np.sum(np.nan_to_num(F) > 0))
    if n_pos < min_positive:
        return None
    try:
        return gravity_panel(Y, bil, **kw)
    except np.linalg.LinAlgError:
        return None


def estimate_pair_elasticities(Z5, FD4, X3, bil, industries, dep: str = "F",
                               include_fd_user: bool = True, min_positive: int = 500,
                               n_jobs: int = 1, supply_subset=None, user_subset=None,
                               **kw) -> pd.DataFrame:
    """delta_{k,m}: supplying industry k x using industry m (and final demand 'FD').

    Pairs with fewer than ``min_positive`` positive flows in the panel are skipped
    (delta set to NaN; ``fill_pair_elasticities`` can fill them from delta_k).
    """
    K = len(industries)
    ks = range(K) if supply_subset is None else supply_subset
    users = list(range(K)) if user_subset is None else list(user_subset)
    if include_fd_user:
        users = users + [None]
    jobs = [(k, m) for k in ks for m in users]

    def run(k, m):
        r = _one_pair(Z5, FD4, X3, bil, k, m, dep, min_positive, kw)
        lab = dict(supplier=industries[k], user=("FD" if m is None else industries[m]))
        if r is None:
            return dict(lab, delta=np.nan, se_delta=np.nan, converged=False)
        return _row(r, **lab)

    if n_jobs == 1:
        rows = [run(k, m) for k, m in jobs]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run)(k, m) for k, m in jobs)
    return pd.DataFrame(rows)


def fill_pair_elasticities(pair_df: pd.DataFrame, sector_df: pd.DataFrame,
                           shrink: bool = True, max_se: float | None = None) -> pd.DataFrame:
    """Fill / shrink delta_{k,m} toward delta_k.

    * Missing or non-converged pairs (or se > max_se) are replaced by delta_k.
    * If ``shrink``: empirical-Bayes (James-Stein-type) shrinkage within each supplier k:
        delta_km_shr = w * delta_km + (1 - w) * delta_k,   w = tau2_k / (tau2_k + se_km^2)
      with tau2_k = max(0, var_m(delta_km) - mean_m(se_km^2)) (method of moments).
    This is a design choice of this implementation (not in Cai 2023); see METHODS.md.
    """
    df = pair_df.merge(sector_df[["industry", "delta"]].rename(
        columns={"industry": "supplier", "delta": "delta_k"}), on="supplier", how="left")
    bad = ~df["converged"].astype(bool) | df["delta"].isna()
    if max_se is not None:
        bad |= df["se_delta"] > max_se
    df["delta_raw"] = df["delta"]
    df.loc[bad, "delta"] = np.nan
    df["filled"] = bad
    if shrink:
        out = []
        for _, g in df.groupby("supplier", sort=False):
            ok = g["delta"].notna()
            if ok.sum() >= 3:
                tau2 = max(0.0, g.loc[ok, "delta"].var() - (g.loc[ok, "se_delta"] ** 2).mean())
                w = tau2 / (tau2 + g["se_delta"] ** 2)
                g = g.copy()
                g.loc[ok, "delta"] = (w * g["delta"] + (1 - w) * g["delta_k"])[ok]
                g["shrink_weight"] = np.where(ok, w, 0.0)
            out.append(g)
        df = pd.concat(out)
    df["delta"] = df["delta"].fillna(df["delta_k"])
    return df


def pair_matrix(df: pd.DataFrame, suppliers, users, col: str = "delta") -> np.ndarray:
    """Pivot a long (supplier, user) table into a (K_supply, K_user) array."""
    p = df.pivot(index="supplier", columns="user", values=col)
    return p.reindex(index=list(suppliers), columns=list(users)).to_numpy()
