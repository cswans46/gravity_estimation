import numpy as np
import pandas as pd
import pytest

from gravity_irio.business_locations import (aggregate_grid, aggregate_user_distances, benchmark_weights,
                                             impute_employees, sector_pair_distances)
from gravity_irio.cai import cai_calibrate, cai_estimate, dtpe, ols_hdfe, stpe
from gravity_irio.canada_gravity import canada_language_matrix, estimate_ipt_elasticities
from gravity_irio.concordance import (ICIO2025, canonical_icio, commodity_concordance, industry_concordance,
                                      isic_to_icio, load_bridge, naics_candidates, naics_to_icio)
from gravity_irio.disaggregate import (cai_interprovincial_trade, gravity_split, ipa_split, map_distances,
                                       own_province_share_by_user, schulte_allocation, schulte_envelope)
from gravity_irio.final_demand import from_provincial_sut, share_out_like_canada
from gravity_irio.icio import aggregate_regions
from gravity_irio.sdm import (SpatialIRIO, TorojPrior, balance_three_way, canada_r_home, home_share_stats,
                              weight_matrix)
from gravity_irio.sdm_sensitivity import run_sensitivity
from gravity_irio.synthetic import synthetic_ipt, synthetic_locations, synthetic_sdm

PROV10 = ["NL", "PE", "NS", "NB", "QC", "ON", "MB", "SK", "AB", "BC"]


# ---------------------------------------------------------------- Cai replication
def _cai_world(C=27, T=1, theta=-1.5, border=-1.2, lang=0.4, seed=0, sigma=0.3):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 3000, (C, 2))
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, rng.uniform(50, 200, C))
    L = (rng.uniform(size=(C, C)) < 0.15).astype(float); L = np.maximum(L, L.T)
    F = np.zeros((T, C, C))
    for t in range(T):
        F[t] = (np.outer(rng.lognormal(0, 1, C), rng.lognormal(0, 1, C)) * d ** theta
                * np.exp(border * (1 - np.eye(C)) + lang * L) * np.exp(sigma * rng.standard_normal((C, C))))
    bil = {"dist": d, "home": np.eye(C), "comlang_off": L * (1 - np.eye(C)), "valid": np.ones(C, bool)}
    return F, bil, [f"c{i}" for i in range(C)]


def test_cai_ols_recovers_eq9_cross_section_and_panel():
    F, bil, cs = _cai_world(T=1)
    r = cai_estimate(F, bil, cs)
    assert abs(r["theta"] + 1.5) < 4 * r["se"] + 0.02
    assert abs(r["border"] + 1.2) < 4 * r["se_border"] + 0.05
    assert abs(r["language"] - 0.4) < 4 * r["se_language"] + 0.05
    Fp, bilp, _ = _cai_world(T=5, seed=1)
    rp = cai_estimate(Fp, bilp, cs)
    assert rp["se"] < r["se"] and abs(rp["theta"] + 1.5) < 4 * rp["se"] + 0.02
    rpp = cai_estimate(Fp, bilp, cs, estimator="ppml")
    assert abs(rpp["theta"] + 1.5) < 0.1


def test_cai_sample_restriction_and_zero_drop():
    F, bil, cs = _cai_world(T=1)
    F[0, 0, 1] = 0.0
    r = cai_estimate(F, bil, cs, sample=cs[:10])
    assert r["n_obs"] == 99 and r["n_zero_dropped"] == 1


def test_ols_hdfe_matches_dummy_regression():
    rng = np.random.default_rng(0)
    n = 300
    g, h = rng.integers(0, 8, n), rng.integers(0, 6, n)
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(size=8)[g] + rng.normal(size=6)[h] + rng.normal(size=n) * 0.1
    r = ols_hdfe(y, x, [g, h], ["x"])
    D = np.column_stack([x, np.eye(8)[g], np.eye(6)[h][:, 1:]])
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    assert np.isclose(r.coef[0], b[0], atol=1e-8)


def test_cai_calibration_and_metrics():
    rng = np.random.default_rng(0)
    d = rng.uniform(100, 1000, (4, 4)); np.fill_diagonal(d, 50)
    X = np.outer(rng.uniform(1, 2, 4), rng.uniform(1, 2, 4)) * d ** -1.3
    Xh, _ = cai_calibrate(X.sum(1), X.sum(0), d, -1.3)
    assert stpe(Xh, X) < 1e-6 and abs(dtpe(Xh, X)) < 1e-6
    Xw, _ = cai_calibrate(X.sum(1), X.sum(0), d, -2.0)
    assert dtpe(Xw, X) > 0  # too steep decay overstates intra-regional trade (Cai Sect. 4.2)


def test_aggregate_split_regions():
    Z = np.random.rand(1, 4, 2, 4, 2); FD = np.random.rand(1, 4, 2, 4); X = np.random.rand(1, 4, 2)
    Zn, FDn, Xn, c = aggregate_regions(Z, FD, X, ["CAN", "CN1", "CN2", "MX1"])
    assert c == ["CAN", "CHN", "MEX"] and np.isclose(Zn.sum(), Z.sum()) and np.isclose(FDn.sum(), FD.sum())
    Z2 = np.random.rand(1, 3, 2, 3, 2)
    with pytest.raises(ValueError):
        aggregate_regions(Z2, np.random.rand(1, 3, 2, 3), np.random.rand(1, 3, 2), ["CHN", "CN1", "CN2"])


# ---------------------------------------------------------------- concordances
def test_icio_labels_and_isic():
    assert len(ICIO2025) == 50
    assert canonical_icio("CAN_24A") == "C24A" and canonical_icio("62T63") == "J62_63"
    assert canonical_icio("97T98") == "T" and canonical_icio("36T39") == "E"
    assert isic_to_icio("2431") == "C24A" and isic_to_icio("2432") == "C24B" and isic_to_icio("3011") == "C301"


def test_naics_bridge_covers_all_icio_and_is_unique_at_6_digits():
    b = load_bridge()
    assert set(b["icio2025"]) == set(ICIO2025)
    assert naics_to_icio("325410", b) == "C21" and naics_to_icio("325110", b) == "C20"
    assert naics_to_icio("541510", b) == "J62_63" and naics_to_icio("541110", b) == "M"
    assert naics_to_icio("336611", b) == "C301" and naics_to_icio("331511", b) == "C24A"
    assert set(naics_candidates("21", b)) == {"B05", "B06", "B07", "B08", "B09"}


def test_industry_and_commodity_concordance():
    rep = []
    M = industry_concordance(["BS111A00", "BS21E000", "GS611100"], report=rep,
                             split_weights={"B05": 1, "B06": 0, "B07": 2, "B08": 1, "B09": 0})
    assert np.allclose(M.sum(1), 1) and len(rep) == 1
    V = np.array([[10.0, 0, 0], [2.0, 8.0, 0]])
    C = commodity_concordance(V, ["BS111A00", "BS324000", "BS541500"])
    assert np.allclose(C.sum(1), 1)


# ---------------------------------------------------------------- locations / IPT
def test_sector_pair_distances_vary_by_user():
    df, provs = synthetic_locations(R=4, K=3)
    d = impute_employees(df)
    assert d["weight"].gt(0).all() and d["employees_imputed"].any()
    tot = d.groupby(["prov", "sector"])["weight"].sum().reset_index().rename(columns={"weight": "value"})
    tot["value"] *= 2
    d2 = benchmark_weights(d, tot)
    assert np.isclose(d2["weight"].sum(), 2 * d["weight"].sum())
    D4, p, s = sector_pair_distances(aggregate_grid(d, 25), provs)
    assert D4.shape == (4, 3, 4, 3)
    assert np.all(D4[0, 0, 0, :] < D4[0, 0, 1, :])          # internal < interprovincial
    assert np.ptp(D4[0, 0, 1, :]) > 1.0                       # depends on the user sector
    sh = np.full((3, 4, 3), 1 / 3)
    assert aggregate_user_distances(D4, sh).shape == (4, 3, 4)


def test_ipt_elasticities_cai_spec_and_by_origin():
    Tm, dist, delta = synthetic_ipt(R=10, K=2, T=6, seed=3)
    r = estimate_ipt_elasticities(Tm, dist, PROV10, ["a", "b"], drop=())
    assert np.allclose(r["delta"], delta, atol=0.05)
    assert np.allclose(r["border"], -1.0, atol=0.1)
    ro = estimate_ipt_elasticities(Tm, dist, PROV10, ["a", "b"], drop=(), slope_by="origin")
    assert len(ro) == 20
    rp = estimate_ipt_elasticities(Tm, dist, PROV10, ["a", "b"], drop=(), estimator="ppml")
    assert np.allclose(rp["delta"], delta, atol=0.08)
    L = canada_language_matrix(["QC", "NB", "ON"])
    assert L[0, 1] == 1 and L[0, 2] == 0 and L[1, 2] == 1


# ---------------------------------------------------------------- 4-D construction
def _canada_case(seed=0, R=5, J=2, U=3):
    rng = np.random.default_rng(seed)
    D = rng.uniform(200, 3000, (R, J, R, U))
    for b in range(R):
        D[b, :, b, :] = rng.uniform(30, 150, (J, U))
    delta = rng.uniform(0.5, 1.5, J)
    a, c = rng.lognormal(0, 1, (R, J)), rng.lognormal(0, 1, (J, R, U))
    Z = np.zeros((R, J, R, U))
    for j in range(J):
        for b in range(R):
            Z[:, j, b, :] = a[:, j][:, None] * c[j, b][None, :] * D[:, j, b, :] ** (-delta[j])
    return Z, D, delta


def test_user_specific_distances_break_ipa_with_supplier_delta():
    Z, D, delta = _canada_case()
    T, U = Z.sum(axis=3), Z.sum(axis=0)
    Zg, _ = gravity_split(T, U, D, delta)
    np.testing.assert_allclose(Zg, Z, rtol=1e-6)
    assert np.abs(ipa_split(T, U) - Z).sum() > 1e-3 * Z.sum()
    Zo, _ = gravity_split(T, U, D, np.repeat(delta[:, None], 5, axis=1), delta_by="origin")
    np.testing.assert_allclose(Zo, Z, rtol=1e-6)


def test_map_distances_identity_concordance():
    Z, D, delta = _canada_case(J=2, U=2)
    out = map_distances(D, np.eye(2), np.eye(2))
    np.testing.assert_allclose(out, D)


def test_cai_interprovincial_from_provincial_tables():
    rng = np.random.default_rng(1)
    R, J = 5, 2
    d = rng.uniform(200, 3000, (R, R)); d = (d + d.T) / 2; np.fill_diagonal(d, 80)
    T = np.zeros((R, J, R))
    for j, dl in enumerate([0.8, 1.4]):
        T[:, j, :] = np.outer(rng.lognormal(0, 1, R), rng.lognormal(0, 1, R)) * d ** -dl
    Th = cai_interprovincial_trade(T.sum(axis=2), T.sum(axis=0), d, [0.8, 1.4])
    np.testing.assert_allclose(Th, T, rtol=1e-6)


def test_schulte_allocation_preserves_margins_and_envelope_brackets():
    rng = np.random.default_rng(0)
    s, u = rng.uniform(1, 5, 4), rng.uniform(1, 5, 6)
    X = schulte_allocation(s, u, rng)
    assert np.allclose(X.sum(1), s) and np.allclose(X.sum(0), u * s.sum() / u.sum())
    Z, D, delta = _canada_case()
    T, U = Z.sum(axis=3), Z.sum(axis=0)
    from gravity_irio.disaggregate import frechet_own_share_bounds
    env = schulte_envelope(T, U, n_draws=30, stat=own_province_share_by_user)
    lo, hi = frechet_own_share_bounds(T, U)
    for Z_ in (ipa_split(T, U), Z, gravity_split(T, U, D, delta)[0]):
        v = own_province_share_by_user(Z_)
        assert np.all(v >= lo - 1e-9) and np.all(v <= hi + 1e-9)
    assert np.all(env["min"] >= lo - 1e-9) and np.all(env["max"] <= hi + 1e-9)


# ---------------------------------------------------------------- SDM upgrades
def test_toroj_prior_and_sampler():
    data, tr = synthetic_sdm(S=2, R=7, T=3, seed=4)
    d = tr["dist"]
    hm = [np.diag(weight_matrix(d, tr["shape"][s], tr["scale"][s], None, 0)).mean() for s in range(2)]
    off = d[~np.eye(7, dtype=bool)]
    pr = TorojPrior(hm, [0.1, 0.1], r_home=np.percentile(off, 5), d_max=off.max(), a_min=0.05, a_max=20)
    assert np.isfinite(pr.logpdf(0, *pr.initial(0)))
    assert pr.logpdf(0, np.log(100.0), 0.0) == -np.inf  # outside shape bounds
    m = SpatialIRIO(data, prior=pr, supply_exponent="fixed1")
    res = m.sample(n_iter=200, burn=100, chains=1, seed=0)
    assert np.isfinite(res.logpost).all()
    assert res.summary()["corr_logshape_logscale"].notna().all()


def test_distance_time_mixture_and_three_way_balance():
    data, tr = synthetic_sdm(S=2, R=6, T=2, seed=5)
    data.dist_time = tr["dist"] * np.random.default_rng(0).uniform(0.8, 1.3, tr["dist"].shape)
    data.__post_init__()
    m = SpatialIRIO(data)
    assert m.param_names == ["shape", "scale", "gamma", "psi"]
    phi = m.initial_phi(); st = m._init_state(phi)
    new = phi[0].copy(); new += 0.2
    dl, _ = m._propose_sector(st, 0, new)
    p2 = phi.copy(); p2[0] = new
    assert np.isclose(dl, m.log_post(p2) - m.log_post(phi), atol=1e-8)
    Z = m.flows(phi)
    R, S = 6, 2
    rng = np.random.default_rng(1)
    row = Z.sum(axis=(2, 3)) * rng.uniform(0.9, 1.1, (R, S))
    row *= Z.sum() / row.sum()
    Zb, info = balance_three_way(Z, row=row, col=Z.sum(axis=(0, 1)), block=Z.sum(axis=(0, 2)))
    assert info["converged"]
    np.testing.assert_allclose(Zb.sum(axis=(0, 2)), Z.sum(axis=(0, 2)), rtol=1e-6)


def test_final_demand_consistency_and_identity():
    rng = np.random.default_rng(0)
    T, S, R, C = 2, 3, 4, 3
    FU = rng.uniform(1, 2, (T, S, R, C)); m = rng.uniform(0, .3, (T, S, R)); EX = rng.uniform(0, 1, (T, S, R))
    f = rng.uniform(5, 10, (T, S))
    yd, yx, k = from_provincial_sut(FU, m, EX, f)
    np.testing.assert_allclose(yd.sum(2) + yx.sum(2), f)
    yd2, yx2, _ = share_out_like_canada({"household": f * .6, "exports": f * .4},
                                        {"household": rng.uniform(1, 2, (T, R)),
                                         "exports": rng.uniform(1, 2, (T, S, R))}, f)
    np.testing.assert_allclose(yd2.sum(2) + yx2.sum(2), f)


def test_sensitivity_runner_and_helpers():
    data, _ = synthetic_sdm(S=2, R=6, T=2, seed=6)
    df, Zs = run_sensitivity(data, exogenous_mass=data.mass * 1.1 + 0.1)
    assert set(df["scenario"]) >= {"baseline", "n_R-1", "psi_0", "mass_exogenous", "diag_cov"}
    assert canada_r_home() > 100
    Tr = np.random.default_rng(0).uniform(size=(4, 2, 4))
    mu, sd = home_share_stats(Tr)
    assert mu.shape == (2,) and np.all((mu > 0) & (mu < 1))
