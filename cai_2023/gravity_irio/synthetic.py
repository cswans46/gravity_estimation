"""Synthetic data generators with known parameters (for tests and demos)."""
from __future__ import annotations

import numpy as np

from .sdm import SDMData, weight_matrix


def synthetic_icio(C=30, K=4, T=5, seed=0, delta_km=None, home=1.5, noise="poisson"):
    """ICIO-shaped arrays whose supplier x user flows follow PPML gravity.

    Returns dict(Z5, FD4, X3, bil, delta_km (K, K+1), industries, countries).
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10_000, (C, 2))
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, rng.uniform(100, 600, C))
    if delta_km is None:
        delta_km = rng.uniform(0.6, 1.8, (K, K + 1))
    homem = np.eye(C)
    size_o = rng.lognormal(0, 1, (T, C, K))
    size_d = rng.lognormal(0, 1, (T, C, K + 1))
    Z5 = np.zeros((T, C, K, C, K))
    FD4 = np.zeros((T, C, K, C))
    for k in range(K):
        for m in range(K + 1):
            mu = (size_o[:, :, k][:, :, None] * size_d[:, :, m][:, None, :]
                  * d[None] ** (-delta_km[k, m]) * np.exp(home * homem)[None]) * 1e4
            F = rng.poisson(mu).astype(float) if noise == "poisson" else mu
            if m < K:
                Z5[:, :, k, :, m] = F
            else:
                FD4[:, :, k, :] = F
    X3 = Z5.sum(axis=(3, 4)) + FD4.sum(axis=3) + 1.0
    bil = {"dist": d, "home": homem, "valid": np.ones(C, bool), "controls": []}
    return dict(Z5=Z5, FD4=FD4, X3=X3, bil=bil, delta_km=delta_km,
                industries=[f"k{i}" for i in range(K)], countries=[f"c{i}" for i in range(C)])


def synthetic_sdm(S=4, R=8, T=5, seed=0, shape=None, scale=None, noise_sd=0.02, kind="sf"):
    """Data generated from Torój's SDM with known gamma decay parameters.

    The supply mass is exogenous (population-like) so the DGP is exact.
    Returns (SDMData, truth dict).
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, (R, 2))
    dist = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    shape = rng.uniform(0.8, 2.5, S) if shape is None else np.asarray(shape, float)
    scale = rng.uniform(80, 400, S) if scale is None else np.asarray(scale, float)
    X = rng.uniform(100, 200, S)
    Zn = rng.uniform(0, 1, (S, S))
    Zn = Zn / Zn.sum(axis=0, keepdims=True) * X[None, :] * rng.uniform(0.3, 0.5, S)[None, :]
    VA = X - Zn.sum(axis=0)
    Ynat = X - Zn.sum(axis=1)
    beta0 = VA / X
    beta = beta0[:, None] * Zn / VA[None, :]
    pop = rng.lognormal(0, 0.7, R)
    va = np.zeros((T, S, R)); y = np.zeros((T, S, R)); mass = np.zeros((T, S, R))
    for t in range(T):
        popt = pop * np.exp(0.05 * rng.standard_normal(R))
        ysh = popt / popt.sum()
        m = np.stack([popt * rng.lognormal(0, 0.5, R) for _ in range(S)])
        msh = m / m.sum(axis=1, keepdims=True)
        W = np.stack([weight_matrix(dist, shape[s], scale[s], msh[s], 1.0, kind) for s in range(S)])
        A = np.eye(S * R) - (beta[:, :, None, None] * W[:, None]).transpose(0, 2, 1, 3).reshape(S * R, S * R)
        y[t] = Ynat[:, None] * ysh[None, :]
        b = (beta0[:, None] * np.einsum("srp,sp->sr", W, y[t])).reshape(-1)
        eps = rng.normal(0, noise_sd, (S, R)) * VA[:, None] / R
        eps -= eps.mean(axis=1, keepdims=True)
        va[t] = np.linalg.solve(A, b + eps.reshape(-1)).reshape(S, R)
        mass[t] = m
    data = SDMData(va=va, y=y, Znat=np.repeat(Zn[None], T, 0), Xnat=np.repeat(X[None], T, 0),
                   dist=dist, mass=mass)
    return data, {"shape": shape, "scale": scale, "dist": dist}


def synthetic_canada(R=5, J=3, U=4, seed=0, delta_ju=None):
    """Canadian-style arrays generated from the gravity-RAS DGP."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 4000, (R, 2))
    dist = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(dist, rng.uniform(50, 300, R))
    delta = rng.uniform(0.3, 2.0, (J, U)) if delta_ju is None else delta_ju
    a = rng.lognormal(0, 1, (R, J))
    c = rng.lognormal(0, 1, (J, R, U))
    Z = np.zeros((R, J, R, U))
    for j in range(J):
        for b in range(R):
            Z[:, j, b, :] = a[:, j][:, None] * c[j, b][None, :] * dist[:, b][:, None] ** (-delta[j][None, :])
    T = Z.sum(axis=3)
    U_dom = Z.sum(axis=0)
    return dict(Z=Z, T=T, Udom=U_dom, dist=dist, delta=delta)
