"""Step 3 (separate routine): Bayesian spatial Durbin IRIO following Torój (2024), with the
supply-side term of Torój (2016), giving Z[r, i, s, j] for Canadian provinces.

Inputs (tidy CSV in --sdm-dir):
    regional_va.csv        year, region, sector, value     (Table 36-10-0402-01 / provincial SUT)
    regional_fd.csv        year, region, sector, value     domestic-origin final use located in r
                           (provincial SUT 36-10-0478-01, see gravity_irio/final_demand.py)
    national_Z.csv         year, supplier, user, value     national domestic intermediate flows
    national_X.csv         year, sector, value
    distances.csv          origin, destination, km
Optional:
    driving_time.csv       origin, destination, km          second distance measure (Torój 2024 eq. 5)
    exports_direct.csv     year, region, sector, value      international exports by origin (y_direct)
    supply_mass.csv        year, region, sector, value      exogenous mass (e.g. LFS 14-10-0023 employment)
    column_totals.csv      supplier, region, user, value    provincial domestic use (else Torój eq. 1)
    row_totals.csv         region, sector, value            regional supply potential (extended RAS)
    home_shares.csv        sector, mean, sd                 Torój prior calibration (else from ipt_flows.csv)
    ipt_flows.csv          year, origin, destination, commodity(=sector), value
Usage:
    python spatial_irio.py --demo
    python spatial_irio.py --sdm-dir data/canada_sdm --prior toroj --chains 4 --iter 6000 --sensitivity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_irio.canada_io import to_array
from gravity_irio.sdm import (LogNormalPrior, SDMData, SpatialIRIO, TorojPrior, balance_three_way, canada_r_home,
                              home_share_stats, prior_from_gravity)
from gravity_irio.sdm_sensitivity import run_sensitivity

HERE = Path(__file__).resolve().parent


def _yrs(f, L, name):
    p = f / name
    if not p.exists():
        return None
    arr, _ = to_array(pd.read_csv(p), ["year", "sector", "region"], labels=L)
    return arr


def load_sdm(folder):
    f = Path(folder)
    va_df = pd.read_csv(f / "regional_va.csv")
    years = sorted(va_df["year"].unique())
    sectors = list(dict.fromkeys(va_df["sector"]))
    regions = list(dict.fromkeys(va_df["region"]))
    L = {"year": years, "sector": sectors, "region": regions}
    va = _yrs(f, L, "regional_va.csv")
    y = _yrs(f, L, "regional_fd.csv")
    Zn, _ = to_array(pd.read_csv(f / "national_Z.csv"), ["year", "supplier", "user"],
                     labels={"year": years, "supplier": sectors, "user": sectors})
    Xn, _ = to_array(pd.read_csv(f / "national_X.csv"), ["year", "sector"], labels=L)
    LL = {"origin": regions, "destination": regions}
    d, _ = to_array(pd.read_csv(f / "distances.csv"), ["origin", "destination"], "km", labels=LL)
    dt = None
    if (f / "driving_time.csv").exists():
        dt, _ = to_array(pd.read_csv(f / "driving_time.csv"), ["origin", "destination"], "km", labels=LL)
    data = SDMData(va, y, Zn, Xn, d, _yrs(f, L, "supply_mass.csv"), sectors, regions, dist_time=dt,
                   y_direct=_yrs(f, L, "exports_direct.csv"))
    extra = {}
    if (f / "supply_mass_exogenous.csv").exists():   # e.g. lagged VA or employment
        extra["mass_exo"] = _yrs(f, L, "supply_mass_exogenous.csv")
    if (f / "column_totals.csv").exists():
        extra["col"], _ = to_array(pd.read_csv(f / "column_totals.csv"), ["supplier", "region", "user"],
                                   labels={"supplier": sectors, "region": regions, "user": sectors})
    if (f / "row_totals.csv").exists():
        extra["row"], _ = to_array(pd.read_csv(f / "row_totals.csv"), ["region", "sector"],
                                   labels={"region": regions, "sector": sectors})
    if (f / "home_shares.csv").exists():
        hs = pd.read_csv(f / "home_shares.csv").set_index("sector").loc[sectors]
        extra["home"] = (hs["mean"].to_numpy(), hs["sd"].to_numpy())
    elif (f / "ipt_flows.csv").exists():
        ipt = pd.read_csv(f / "ipt_flows.csv")
        T, _ = to_array(ipt, ["origin", "commodity", "destination"],
                        labels={"origin": regions, "destination": regions, "commodity": sectors})
        extra["home"] = home_share_stats(T)
    return data, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdm-dir", default=str(HERE / "data" / "canada_sdm"))
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--supply-exponent", default="estimate", help="estimate | fixed0 | fixed1 | <float>")
    ap.add_argument("--covariance", choices=["iw", "diag"], default="iw")
    ap.add_argument("--n-eff", choices=["R", "R-1"], default="R")
    ap.add_argument("--prior", choices=["toroj", "gravity", "weak"], default="toroj")
    ap.add_argument("--r-home", type=float, default=None, help="km; overrides the computed Torój radius")
    ap.add_argument("--r-home-rule", choices=["median", "first_exit"], default="median",
                    help="water gaps on the capital-to-neighbour line: median line (default) or shoreline")
    ap.add_argument("--shape-min", type=float, default=0.01, help="Torój eq. 7 lower bound (his: ~0.01)")
    ap.add_argument("--shape-max", type=float, default=10.0)
    ap.add_argument("--shape-prior", choices=["uniform", "loguniform"], default="uniform",
                    help="uniform = Torój eq. 7; loguniform = uniform in log shape (sensitivity)")
    ap.add_argument("--boundaries", default=None,
                    help="StatCan provincial boundary shapefile (.shp, EPSG:3347) for the exact Torój radius")
    ap.add_argument("--elasticities", default=str(HERE / "results" / "canada_ipt_elasticities_none.csv"))
    ap.add_argument("--proposal", choices=["adaptive", "toroj"], default="adaptive")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--iter", type=int, default=6000)
    ap.add_argument("--burn", type=int, default=3000)
    ap.add_argument("--thin", type=int, default=3)
    ap.add_argument("--draws-for-flows", type=int, default=200)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if a.demo:
        from gravity_irio.synthetic import synthetic_sdm
        data, truth = synthetic_sdm(S=4, R=8, T=5, seed=1)
        off = data.dist[~np.eye(data.R, dtype=bool)]
        Wd = [np.diag(__import__("gravity_irio.sdm", fromlist=["weight_matrix"]).weight_matrix(
            data.dist, truth["shape"][s], truth["scale"][s], None, 0)) for s in range(4)]
        extra = {"home": (np.array([w.mean() for w in Wd]), np.array([w.std() for w in Wd]))}
        a.r_home, a.chains, a.iter, a.burn = float(np.percentile(off, 5)), 2, 1500, 500
    else:
        data, extra = load_sdm(a.sdm_dir)
    print(f"national identity gap (should be ~0): {data.national_identity_gap():.2e}")

    se = a.supply_exponent if a.supply_exponent in ("estimate", "fixed0", "fixed1") else float(a.supply_exponent)
    off = data.dist[~np.eye(data.R, dtype=bool)]
    if a.prior == "toroj":
        if "home" not in extra:
            raise SystemExit("toroj prior needs home_shares.csv or ipt_flows.csv")
        from gravity_irio.sdm import CANADA_CAPITALS
        provs = [r for r in (data.regions or []) if r in CANADA_CAPITALS]
        polys = None
        if a.boundaries:
            from gravity_irio.business_locations import normalize_province
            from gravity_irio.geography import read_shapefile
            polys = {normalize_province(k): v for k, v in read_shapefile(a.boundaries, "PRUID").items()}
        if a.r_home:
            r_home, how = a.r_home, "user value"
        else:
            r_home, pairs = canada_r_home(provs, polygons=polys, return_pairs=True, rule=a.r_home_rule)
            how = "exact boundaries" if polys else "approximation: border half-way between capitals"
            pd.DataFrame(pairs, columns=["region", "neighbour", "km"]).to_csv(out / "r_home_pairs.csv", index=False)
        pkw = dict(a_min=a.shape_min, a_max=a.shape_max, shape_prior=a.shape_prior)
        prior = TorojPrior(*extra["home"], r_home=r_home, d_max=float(off.max()), **pkw)
        print(f"Torój prior: shape ~ {a.shape_prior}({a.shape_min}, {a.shape_max}); r_home = {r_home:.1f} km ({how}), d_max = {off.max():.0f} km")
    elif a.prior == "gravity":
        el = pd.read_csv(a.elasticities).set_index("sector")["delta"].reindex(data.sectors)
        prior = prior_from_gravity(el.fillna(el.mean()).to_numpy(), data.dist)
    else:
        prior = LogNormalPrior.weak(data.S, data.dist)

    model = SpatialIRIO(data, prior=prior, supply_exponent=se, covariance=a.covariance, n_eff=a.n_eff)
    mode = model.fit_mode(verbose=True)
    res = model.sample(n_iter=a.iter, burn=a.burn, thin=a.thin, chains=a.chains, init=mode,
                       proposal=a.proposal, verbose=True)
    summ = res.summary()
    summ.to_csv(out / "sdm_posterior_summary.csv", index=False)
    print(summ.round(3).to_string())
    print("acceptance by sector:", res.accept.mean(axis=0).round(2))
    if a.prior == "toroj":   # does the posterior pile up at a shape bound (Torój's own does at 0.01)?
        sh = summ[summ["param"] == "shape"].set_index("sector")
        lo_hit = sh.index[sh["q05"] < a.shape_min * 1.1].tolist()
        hi_hit = sh.index[sh["q95"] > a.shape_max / 1.1].tolist()
        if lo_hit or hi_hit:
            print(f"NOTE: posterior shape near the prior bound for {lo_hit} (lower) / {hi_hit} (upper). "
                  "The decay profile is usually still identified along the shape-scale ridge; compare "
                  "the 'shape_wide' sensitivity run.")
    bad = summ[(summ["rhat"] > 1.05) & (summ["corr_logshape_logscale"] > -0.9)]
    if len(bad):
        print("WARNING: R-hat > 1.05 without the near -1 shape/scale correlation that Torój (2025) "
              "treats as an identified profile:", sorted(set(bad["sector"])))
    balance = None
    if "row" in extra:
        colZ = extra.get("col")
        block = data.Znat[-1]
        balance = dict(row=extra["row"], col=None if colZ is None else colZ.sum(axis=0), block=block)
    Z = model.posterior_flows(res, t=-1, col_totals=extra.get("col"), n_draws=a.draws_for_flows, balance=balance)
    np.save(out / "sdm_Z4_mean.npy", Z["mean"])
    np.save(out / "sdm_Z4_sd.npy", Z["sd"])
    for q, arr in Z.get("quantiles", {}).items():
        np.save(out / f"sdm_Z4_q{int(q * 100):02d}.npy", arr)
    np.save(out / "sdm_draws.npy", res.natural())
    if a.sensitivity:
        df, Zs = run_sensitivity(data, prior=prior, exogenous_mass=extra.get("mass_exo"),
                                 col_totals=extra.get("col"))
        if a.prior == "toroj":  # radius sensitivity: half and double the Torój radius
            Zb = Zs["baseline"]
            alt = [(f"r_home_x{fct}", dict(pkw, r_home=r_home * fct)) for fct in (0.5, 2.0)]
            alt += [("shape_wide", dict(pkw, r_home=r_home, a_min=a.shape_min / 10, a_max=a.shape_max * 10)),
                    ("shape_loguniform", dict(pkw, r_home=r_home, shape_prior="loguniform"))]
            for name, kw in alt:
                pr = TorojPrior(*extra["home"], d_max=float(off.max()), **kw)
                d2, Z2 = run_sensitivity(data, prior=pr, scenarios={name: {}}, col_totals=extra.get("col"))
                Zv = next(iter(Z2.values()))
                d2["share_reallocated_vs_baseline"] = 0.5 * np.abs(Zv - Zb).sum() / Zb.sum()
                df = pd.concat([df, d2])
        df.to_csv(out / "sdm_sensitivity.csv", index=False)
        print(df.groupby("scenario")[["own_share", "share_reallocated_vs_baseline"]].mean().round(3))


if __name__ == "__main__":
    main()
