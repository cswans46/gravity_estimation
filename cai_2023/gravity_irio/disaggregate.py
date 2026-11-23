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


def _seed_logs(lnd_ib_u, delta_iu):
    return -lnd_ib_u * delta_iu


def gravity_split(T, Udom, dist, delta, kappa: float = 0.0, home=None, delta_by: str | None = None,
                  tol: float = 1e-10, maxit: int = 10_000):
    """Gravity-RAS split. Returns (Z, diagnostics).

    dist  : (R, R) one distance per province pair, or (R, J, R, U) user-specific
            distances d[i, j, b, u] (e.g. ``business_locations.sector_pair_distances``
            mapped to the commodity/user classification). Diagonal (internal) > 0.
    delta : distance elasticities (positive, decay d^-delta), any of
            (J,)        supplier only           (Cai 2023 / IPT-ODBus estimates)
            (J, U)      supplier x user         (ICIO delta_{k,m})
            (J, R)      supplier x origin       (IPT estimates with slope_by='origin')
    home  : (J, U) home-bias coefficients, scaled by ``kappa`` (0 = off).
    delta_by : 'supplier' | 'user' | 'origin'; inferred from delta's shape when unambiguous.

    For each (j, b) the seed S[i, u] = d[i,j,b,u]^(-delta) * exp(kappa h 1[i=b]) is RAS-ed
    to rows T[., j, b] and columns Udom[j, b, .]. With (R, R) distances and (J,) or (J, R)
    deltas the seed is rank one and the result equals IPA.
    """
    T = np.asarray(T, float)
    R, J, _ = T.shape
    Uu = Udom.shape[2]
    dist = np.asarray(dist, float)
    user_specific = dist.ndim == 4
    if user_specific:
        if np.any(~np.isfinite(dist)):
            raise ValueError("user-specific distances contain NaN; fill (e.g. with province-pair distances)")
        lnd4 = np.log(dist)
    else:
        if np.any(np.diag(dist) <= 0):
            raise ValueError("dist must have positive internal distances on the diagonal")
        lnd2 = np.log(dist)
    delta = np.asarray(delta, float)
    if delta_by is None:
        if delta.ndim == 1:
            delta_by = "supplier"
        elif delta.shape == (J, Uu) and Uu != R:
            delta_by = "user"
        elif delta.shape == (J, R) and Uu != R:
            delta_by = "origin"
        else:
            raise ValueError("ambiguous delta shape; pass delta_by='user' or 'origin'")
    Z = np.zeros((R, J, R, Uu))
    scale = np.ones((J, R))
    nonconv = []
    for j in range(J):
        if delta_by == "supplier":
            dl = np.full((R, Uu), float(delta[j]))
        elif delta_by == "user":
            dl = np.repeat(delta[j][None, :], R, axis=0)
        else:  # origin-specific delta[j, i]
            dl = np.repeat(delta[j][:, None], Uu, axis=1)
        for b in range(R):
            r = T[:, j, b]
            c = Udom[j, b, :]
            if r.sum() <= 0 or c.sum() <= 0:
                continue
            lnd = lnd4[:, j, b, :] if user_specific else np.repeat(lnd2[:, b][:, None], Uu, axis=1)
            logS = -lnd * dl
            if home is not None and kappa != 0:
                logS = logS + kappa * np.outer(np.arange(R) == b, home[j])
            logS -= logS.max()
            X, info = ras(np.exp(logS), r, c, tol=tol, maxit=maxit)
            Z[:, j, b, :] = X
            scale[j, b] = info["scale_factor"]
            if not info["converged"]:
                nonconv.append((j, b))
    return Z, {"col_scale": scale, "nonconverged": nonconv}


def cai_interprovincial_trade(supply_dom, use_dom, dist, delta, tol=1e-10):
    """Cai (2023) calibration when only provincial tables are available (no IPT flows).

    supply_dom : (R, J) domestic shipments of j from province i to all Canadian destinations
                 (output - international exports - inventory change of domestic goods)
    use_dom    : (J, R) domestic-origin absorption of j in province b (total use - international
                 imports)
    dist       : (R, R) or (R, J, R) distances (diag > 0); delta : (J,) positive elasticities.
    Returns T_hat (R, J, R) with T_hat[., j, .] = RAS(d^-delta_j; rows supply, cols use).
    """
    S = np.asarray(supply_dom, float)
    U = np.asarray(use_dom, float)
    R, J = S.shape
    D = np.asarray(dist, float)
    out = np.zeros((R, J, R))
    for j in range(J):
        dj = D if D.ndim == 2 else D[:, j, :]
        X, _ = ras(dj ** (-float(np.ravel(delta)[j])), S[:, j], U[j], tol=tol)
        out[:, j, :] = X
    return out


def schulte_allocation(s, u, rng):
    """Schulte, Jakobs & Pauliuk (2021) block-wise random allocation of supply s (origins) to
    use u (target sectors) for one (commodity, destination) matrix: origins in order, target
    sectors in random order, north-west-corner filling. Margins are preserved exactly."""
    s = np.asarray(s, float).copy()
    u = np.asarray(u, float).copy()
    if u.sum() > 0:
        u = u * s.sum() / u.sum()
    order = rng.permutation(len(u))
    X = np.zeros((len(s), len(u)))
    i, q = 0, 0
    while i < len(s) and q < len(order):
        j = order[q]
        a = min(s[i], u[j])
        X[i, j] += a
        s[i] -= a
        u[j] -= a
        if s[i] <= 1e-12 * max(1.0, a):
            i += 1
        if u[j] <= 1e-12 * max(1.0, a):
            q += 1
    return X


def schulte_envelope(T, Udom, n_draws: int = 200, seed: int = 0, stat=None):
    """Monte Carlo envelope of the proportionality uncertainty (Schulte et al. 2021).

    Each draw re-allocates every (j, b) block with ``schulte_allocation``. Returns running
    min, max, mean and sd of Z[i,j,b,u] (or of ``stat(Z)`` if given, e.g. own-province share
    by user), which bound how much any user-level split can differ from IPA/gravity given
    the observed margins.
    """
    rng = np.random.default_rng(seed)
    T = np.asarray(T, float)
    R, J, _ = T.shape
    mn = mx = mean = m2 = None
    for n in range(1, n_draws + 1):
        Z = np.zeros((R, J, R, Udom.shape[2]))
        for j in range(J):
            for b in range(R):
                if T[:, j, b].sum() > 0 and Udom[j, b].sum() > 0:
                    Z[:, j, b, :] = schulte_allocation(T[:, j, b], Udom[j, b], rng)
        v = Z if stat is None else np.asarray(stat(Z), float)
        if mn is None:
            mn, mx, mean, m2 = v.copy(), v.copy(), np.zeros_like(v), np.zeros_like(v)
        mn, mx = np.minimum(mn, v), np.maximum(mx, v)
        d = v - mean
        mean += d / n
        m2 += d * (v - mean)
    return {"min": mn, "max": mx, "mean": mean, "sd": np.sqrt(m2 / max(n_draws - 1, 1))}


def own_province_share_by_user(Z):
    """Share of user u's domestic purchases sourced in its own province (pooled over j, b)."""
    R = Z.shape[0]
    own = sum(Z[b, :, b, :].sum(axis=0) for b in range(R))
    tot = Z.sum(axis=(0, 1, 2))
    return np.divide(own, tot, out=np.full_like(own, np.nan), where=tot > 0)


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


def map_distances(D4, supply_conc, user_conc, theta: float = -1.0):
    """Sector-pair distances D4[i, k, b, m] (e.g. ICIO or NAICS sectors from business
    locations) -> commodity x user distances d[i, j, b, u] by CES aggregation:
        d[i,j,b,u]^theta = sum_k sum_m C^s[j,k] C^u[u,m] D4[i,k,b,m]^theta.
    NaN cells (sector absent in a province) get zero weight (weights renormalised)."""
    D = np.asarray(D4, float)
    ok = np.isfinite(D)
    Dt = np.where(ok, D, 1.0) ** theta
    Cs, Cu = np.asarray(supply_conc, float), np.asarray(user_conc, float)
    num = np.einsum("jk,ikbm,um->ijbu", Cs, Dt * ok, Cu, optimize=True)
    den = np.einsum("jk,ikbm,um->ijbu", Cs, ok.astype(float), Cu, optimize=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (num / den) ** (1.0 / theta), np.nan)


def frechet_own_share_bounds(T, Udom):
    """Exact bounds on each user's own-province sourcing share given only the margins.

    For one (j, b) block with origin supplies r_i and user totals c_u (sum r = sum c), the cell
    (i=b, u) lies in [max(0, c_u - (sum r - r_b)), min(r_b, c_u)] (Frechet bounds); for a given
    user u the blocks are independent, so summing the per-block bounds gives the attainable
    range of u's own-province purchases. Any split consistent with the observed IPT and use
    tables (IPA, gravity, SDM, Schulte draws) lies inside.
    """
    T = np.asarray(T, float)
    R, J, _ = T.shape
    U = Udom.shape[2]
    lo, hi, tot = np.zeros(U), np.zeros(U), np.zeros(U)
    for j in range(J):
        for b in range(R):
            r = T[:, j, b]
            c = Udom[j, b]
            if r.sum() <= 0 or c.sum() <= 0:
                continue
            c = c * r.sum() / c.sum()
            lo += np.maximum(0.0, c - (r.sum() - r[b]))
            hi += np.minimum(r[b], c)
            tot += c
    with np.errstate(divide="ignore", invalid="ignore"):
        return lo / tot, hi / tot
