import numpy as np
import pandas as pd
import pytest

from gravity_irio.disaggregate import compare_splits, gravity_split, ipa_split, map_pair_elasticities
from gravity_irio.elasticities import (estimate_pair_elasticities, estimate_sector_elasticities,
                                       fill_pair_elasticities, gravity_panel, pair_matrix)
from gravity_irio.cepii import bilateral_arrays
from gravity_irio.distances import ces_distance
from gravity_irio.ppml import ppml_hdfe
from gravity_irio.ras import ras
from gravity_irio.sdm import (SpatialIRIO, fit_decay_to_power_law, fit_decay_to_shares,
                              origin_shares_from_flows, prior_from_gravity, weight_matrix)
from gravity_irio.synthetic import synthetic_canada, synthetic_icio, synthetic_sdm


# ------------------------------------------------------------------ PPML
def test_ppml_recovers_parameters():
    rng = np.random.default_rng(0)
    C, T = 30, 6
    i, j, t = (a.ravel() for a in np.meshgrid(np.arange(C), np.arange(C), np.arange(T), indexing="ij"))
    xy = rng.uniform(0, 10, (C, 2))
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)) + 0.5
    eta = rng.normal(size=(C, T))[i, t] + rng.normal(size=(C, T))[j, t] - 1.1 * np.log(d[i, j]) + 1.7 * (i == j)
    y = rng.poisson(np.exp(eta) * 30).astype(float)
    r = ppml_hdfe(y, np.column_stack([np.log(d[i, j]), (i == j) * 1.0]), fe=[i * T + t, j * T + t],
                  names=["lndist", "home"], cluster=i * C + j)
    assert r.converged
    assert abs(r.coef[0] + 1.1) < 4 * r.se[0] + 0.02
    assert abs(r.coef[1] - 1.7) < 4 * r.se[1] + 0.03


def test_ppml_drops_all_zero_groups():
    y = np.array([0, 0, 1, 2, 3, 4.0])
    x = np.array([1, 2, 1, 2, 3, 1.0])
    g = np.array([0, 0, 1, 1, 1, 1])
    r = ppml_hdfe(y, x, fe=[g])
    assert r.n_dropped == 2


# ------------------------------------------------------------------ elasticities
def test_pair_and_sector_elasticities():
    syn = synthetic_icio(C=25, K=3, T=4, seed=3)
    pair = estimate_pair_elasticities(syn["Z5"], syn["FD4"], syn["X3"], syn["bil"], syn["industries"],
                                      controls=("home",), min_positive=10)
    D = pair_matrix(pair, syn["industries"], syn["industries"] + ["FD"])
    SE = pair_matrix(pair, syn["industries"], syn["industries"] + ["FD"], "se_delta")
    assert np.max(np.abs(D - syn["delta_km"]) / SE) < 4.0  # cluster-robust SEs are honest
    sec = estimate_sector_elasticities(syn["Z5"], syn["FD4"], syn["X3"], syn["bil"], syn["industries"],
                                       controls=("home",))
    assert sec["delta"].between(0.5, 2.0).all()
    filled = fill_pair_elasticities(pair, sec, shrink=True)
    assert filled["delta"].notna().all()


def test_dep_z_and_F_agree_roughly():
    syn = synthetic_icio(C=25, K=1, T=3, seed=5, noise="none")
    from gravity_irio.icio import pair_flow
    F, z = pair_flow(syn["Z5"], syn["FD4"], syn["X3"], 0, 0)
    rF = gravity_panel(F, syn["bil"], controls=("home",))
    rz = gravity_panel(z, syn["bil"], controls=("home",))
    # noiseless DGP: both exact
    assert abs(rF.coef[0] + syn["delta_km"][0, 0]) < 1e-5
    assert abs(rz.coef[0] + syn["delta_km"][0, 0]) < 1e-5


def test_cepii_alignment():
    df = pd.DataFrame({
        "iso_o": ["CHN", "CHN", "MEX", "MEX"], "iso_d": ["CHN", "MEX", "CHN", "MEX"],
        "distwces": [1000.0, 13000.0, 13000.0, 600.0], "dist": [900.0, 12000, 12000, 500],
        "contig": [0, 0, 0, 0], "comlang_off": [0, 0, 0, 0]})
    b = bilateral_arrays(["CN1", "CN2", "MX1", "ROW"], df)
    assert b["dist"][0, 1] == 1000.0 and b["home"][0, 1] == 1.0
    assert b["dist"][0, 2] == 13000.0 and b["home"][0, 2] == 0.0
    assert not b["valid"][3]


# ------------------------------------------------------------------ RAS / disaggregation
def test_ras_margins():
    rng = np.random.default_rng(1)
    S = rng.uniform(size=(5, 7))
    r, c = rng.uniform(1, 2, 5), rng.uniform(1, 2, 7)
    X, info = ras(S, r, c)
    assert info["converged"]
    np.testing.assert_allclose(X.sum(1), r, rtol=1e-8)
    np.testing.assert_allclose(X.sum(0), c * r.sum() / c.sum(), rtol=1e-8)


def test_gravity_split_nests_ipa_when_delta_constant_over_users():
    syn = synthetic_canada(R=5, J=3, U=4, seed=2)
    delta_const = np.repeat(syn["delta"][:, :1], 4, axis=1)
    Zg, _ = gravity_split(syn["T"], syn["Udom"], syn["dist"], delta_const)
    Zi = ipa_split(syn["T"], syn["Udom"])
    np.testing.assert_allclose(Zg, Zi, rtol=1e-7, atol=1e-10)


def test_gravity_split_recovers_dgp():
    syn = synthetic_canada(R=6, J=3, U=5, seed=4)
    Zg, diag = gravity_split(syn["T"], syn["Udom"], syn["dist"], syn["delta"])
    np.testing.assert_allclose(Zg, syn["Z"], rtol=1e-6)
    Zi = ipa_split(syn["T"], syn["Udom"])
    assert np.abs(Zi - syn["Z"]).sum() > 1e-3 * syn["Z"].sum()  # IPA is wrong here
    np.testing.assert_allclose(Zg.sum(axis=3), syn["T"], rtol=1e-8)
    np.testing.assert_allclose(Zg.sum(axis=0), syn["Udom"], rtol=1e-6)
    assert compare_splits(Zg, Zi)["share_reallocated"] > 0


def test_map_pair_elasticities():
    d = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = map_pair_elasticities(d, [[1, 0], [0.5, 0.5], [0, 1]], [[1, 0]])
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 3.0])


def test_ces_distance_internal_smaller():
    rng = np.random.default_rng(0)
    reg = np.repeat(["A", "B"], 20)
    lat = np.r_[rng.normal(45, 0.5, 20), rng.normal(50, 0.5, 20)]
    lon = np.r_[rng.normal(-75, 0.5, 20), rng.normal(-100, 0.5, 20)]
    D, regs = ces_distance(reg, lat, lon, rng.uniform(1, 10, 40), area_km2=rng.uniform(10, 100, 40))
    assert D[0, 0] < D[0, 1] and np.isclose(D[0, 1], D[1, 0])


# ------------------------------------------------------------------ SDM
def test_sdm_incremental_determinant_exact():
    data, _ = synthetic_sdm(S=3, R=6, T=2, seed=0)
    m = SpatialIRIO(data)
    phi = m.initial_phi()
    st = m._init_state(phi)
    new = phi[1] + np.array([0.4, -0.3])
    dl, payload = m._propose_sector(st, 1, new)
    p2 = phi.copy(); p2[1] = new
    assert np.isclose(dl, m.log_post(p2) - m.log_post(phi), atol=1e-8)
    m._accept(st, 1, new, payload)
    np.testing.assert_allclose(st.Ainv[0], np.linalg.inv(m.A(0, st.W[:, 0])), atol=1e-10)


def test_sdm_recovers_weights_and_flows_consistent():
    data, tr = synthetic_sdm(S=3, R=8, T=5, seed=1)
    m = SpatialIRIO(data)
    mode = m.fit_mode(sweeps=8)
    for s in range(3):
        Wt = weight_matrix(tr["dist"], tr["shape"][s], tr["scale"][s], m.mshare[0, s])
        assert np.abs(Wt - m.W(s, mode[s], 0)).max() < 0.05
    res = m.sample(n_iter=600, burn=300, thin=2, chains=2, init=mode, jitter=0.1, seed=0)
    Z = m.posterior_flows(res, n_draws=30, quantiles=None)
    # flows aggregate back to the national table
    np.testing.assert_allclose(Z["mean"].sum(axis=(0, 2)), data.Znat[-1], rtol=1e-8)


def test_sdm_estimated_supply_exponent_runs():
    data, _ = synthetic_sdm(S=2, R=6, T=3, seed=2)
    m = SpatialIRIO(data, supply_exponent="estimate")
    res = m.sample(n_iter=200, burn=100, chains=1, seed=0)
    assert res.draws.shape[-1] == 3
    assert (res.natural()[..., 2] > 0).all() and (res.natural()[..., 2] < 1).all()


def test_priors_from_gravity_and_shares():
    data, tr = synthetic_sdm(S=2, R=8, T=1, seed=3)
    shape, scale = fit_decay_to_power_law(1.2, tr["dist"])
    assert shape > 0 and scale > 0
    pr = prior_from_gravity([1.0, 1.5], tr["dist"])
    assert pr.mu_logscale.shape == (2,)
    W = weight_matrix(tr["dist"], 1.5, 200.0, None, 0.0)
    sh, sc, loss = fit_decay_to_shares(W, tr["dist"], None, 0.0)
    assert np.abs(weight_matrix(tr["dist"], sh, sc, None, 0.0) - W).max() < 1e-3
    T = np.random.default_rng(0).uniform(size=(4, 2, 4))
    assert np.allclose(origin_shares_from_flows(T).sum(axis=1), 1)


def test_balance_flows():
    from gravity_irio.sdm import balance_flows
    rng = np.random.default_rng(0)
    Z = rng.uniform(size=(3, 2, 3, 2))
    r, c = rng.uniform(1, 2, (3, 2)), rng.uniform(1, 2, (3, 2))
    Zb, info = balance_flows(Z, r, c)
    assert info["converged"]
    np.testing.assert_allclose(Zb.sum(axis=(2, 3)), r, rtol=1e-8)
