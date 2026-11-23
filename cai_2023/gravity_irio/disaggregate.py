"""Disaggregate Canadian interprovincial trade by demanding sector: Z[i, j, b, u].

    i = origin province, j = commodity, b = destination province,
    u = user in b (intermediate industries, optionally followed by final-demand categories)

Observed inputs (Statistics Canada provincial SUT + interprovincial trade flows):
    T[i, j, b]  domestic-origin shipments of commodity j from province i to province b
                (i == b: own-province supply), all users combined
    U[j, b, u]  use of commodity j by user u in province b, all origins (incl. imports)
    M[j, b]     international imports of j into b (optional)

Two estimators of Z[i,j,b,u] with the same margins
    sum_u Z[i,j,b,u] = T[i,j,b]           (row constraint)
    sum_i Z[i,j,b,u] = Udom[j,b,u]        (column constraint)

1. Import proportionality (IPA) -- every user in b buys j from origin i in the same
   proportion:  Z = T[i,j,b] * Udom[j,b,u] / sum_u Udom[j,b,u].
2. Gravity-RAS (Cai 2023 extended to users): for each (j, b) solve the doubly
   constrained gravity model over (i, u) with seed
        S[i, u] = d_ib ** (-delta[j, u]) * exp(kappa * home[j, u] * 1[i == b])
   where delta[j,u] are ICIO supplier x user distance elasticities mapped to the
   Canadian classification. If delta[j, u] does not vary over u (and kappa = 0) the
   seed is rank one and RAS returns exactly the IPA solution.
"""
from __future__ import annotations

import numpy as np

from .ras import ras


def domestic_use(U, M=None, dom_share=None):
    """Domestic-origin use Udom[j,b,u].

    * dom_share given (j,b,u) or (j,u): Udom = U * dom_share (e.g. user-specific
      domestic shares for Canada from the ICIO CAN block).
    * else if M given: IPA for international imports, Udom = U * (1 - M / sum_u U).
    * else: Udom = U (all use assumed domestic; RAS rescales to T).
    """
    U = np.asarray(U, dtype=float)
    if dom_share is not None:
        s = np.asarray(dom_share, dtype=float)
        if s.ndim == 2:
            s = s[:, None, :]
        return U * s
    if M is not None:
        tot = U.sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(tot > 0, 1.0 - np.asarray(M, float) / tot, 0.0)
        return U * np.clip(share, 0.0, 1.0)[:, :, None]
    return U


def ipa_split(T, Udom):
    """Import-proportionality split: Z[i,j,b,u] = T[i,j,b] * Udom[j,b,u] / sum_u Udom."""
    T = np.asarray(T, float)
    tot = Udom.sum(axis=2, keepdims=True)
    share = np.divide(Udom, tot, out=np.zeros_like(Udom), where=tot > 0)  # (j,b,u)
    return T[:, :, :, None] * share[None, :, :, :]


def gravity_split(T, Udom, dist, delta, kappa: float = 0.0, home=None,
                  tol: float = 1e-10, maxit: int = 10_000):
    """Gravity-RAS split. Returns (Z, diagnostics).

    dist  : (R, R) distances, *positive* diagonal (internal distance) required.
    delta : (J, U) distance elasticities (positive numbers, decay = d^-delta).
    home  : (J, U) home-bias coefficients (e.g. ICIO 'home' PPML coefficient by pair);
            multiplied by ``kappa`` (0 = off, 1 = full international home bias).
    """
    T = np.asarray(T, float)
    R, J, _ = T.shape
    Uu = Udom.shape[2]
    dist = np.asarray(dist, float)
    if np.any(np.diag(dist) <= 0):
        raise ValueError("dist must have positive internal distances on the diagonal")
    delta = np.asarray(delta, float)
    lnd = np.log(dist)
    Z = np.zeros((R, J, R, Uu))
    scale = np.ones((J, R))
    nonconv = []
    for j in range(J):
        for b in range(R):
            r = T[:, j, b]
            c = Udom[j, b, :]
            if r.sum() <= 0 or c.sum() <= 0:
                continue
            logS = -lnd[:, b][:, None] * delta[j][None, :]
            if home is not None and kappa != 0:
                logS = logS + kappa * np.outer(np.arange(R) == b, home[j])
            logS -= logS.max()
            X, info = ras(np.exp(logS), r, c, tol=tol, maxit=maxit)
            Z[:, j, b, :] = X
            scale[j, b] = info["scale_factor"]
            if not info["converged"]:
                nonconv.append((j, b))
    return Z, {"col_scale": scale, "nonconverged": nonconv}


def map_pair_elasticities(delta_km, supply_conc, user_conc):
    """Map ICIO delta (K_s, K_u) to the Canadian classification (J, U).

    supply_conc : (J, K_s) weights, rows sum to 1 (Canadian commodity -> ICIO industry)
    user_conc   : (U, K_u) weights, rows sum to 1 (Canadian user -> ICIO user incl. 'FD')
    Weighted average of elasticities. (Averaging elasticities rather than decay
    functions is an approximation; see METHODS.md.)
    """
    Ws = np.asarray(supply_conc, float)
    Wu = np.asarray(user_conc, float)
    return Ws @ np.asarray(delta_km, float) @ Wu.T


def compare_splits(Z_grav, Z_ipa):
    """Summary of how far the gravity split moves away from IPA."""
    diff = np.abs(Z_grav - Z_ipa).sum()
    tot = Z_ipa.sum()
    R = Z_ipa.shape[0]
    # own-province sourcing share by user u (pooled over j, b): IPA vs gravity
    own_g = sum(Z_grav[b, :, b, :].sum(axis=0) for b in range(R))
    own_i = sum(Z_ipa[b, :, b, :].sum(axis=0) for b in range(R))
    use = Z_ipa.sum(axis=(0, 1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"total": float(tot), "share_reallocated": float(0.5 * diff / tot) if tot else 0.0,
                "own_province_share_by_user_gravity": own_g / use,
                "own_province_share_by_user_ipa": own_i / use}
