"""Step 3 (separate routine): Bayesian spatial Durbin IRIO (Torój) with gamma distance decay.

Inputs (tidy CSV in --sdm-dir):
    regional_va.csv     year, region, sector, value
    regional_fd.csv     year, region, sector, value    (regional net final demand by commodity)
    national_Z.csv      year, supplier, user, value    (domestic intermediate flows)
    national_X.csv      year, sector, value
    distances.csv       origin, destination, km        (diag = internal distance or 0)
Optional:
    supply_mass.csv     year, region, sector, value    (W supply term; default = va)
    column_totals.csv   supplier, region, user, value  (e.g. provincial domestic use; else Torój eq. 4)
    gravity elasticities by supplier (results/elasticities_sector.csv) -> informative prior

Usage:
    python spatial_irio.py --demo
    python spatial_irio.py --sdm-dir data/canada_sdm --prior gravity --chains 4 --iter 4000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_irio.canada_io import to_array
from gravity_irio.sdm import SDMData, SDMPrior, SpatialIRIO, prior_from_gravity

HERE = Path(__file__).resolve().parent


def load_sdm(folder):
    f = Path(folder)
    va_df = pd.read_csv(f / "regional_va.csv")
    years = sorted(va_df["year"].unique())
    sectors = list(dict.fromkeys(va_df["sector"]))
    regions = list(dict.fromkeys(va_df["region"]))
    L = {"year": years, "sector": sectors, "region": regions}
    va, _ = to_array(va_df, ["year", "sector", "region"], labels=L)
    y, _ = to_array(pd.read_csv(f / "regional_fd.csv"), ["year", "sector", "region"], labels=L)
    Zn, _ = to_array(pd.read_csv(f / "national_Z.csv"), ["year", "supplier", "user"],
                     labels={"year": years, "supplier": sectors, "user": sectors})
    Xn, _ = to_array(pd.read_csv(f / "national_X.csv"), ["year", "sector"], labels=L)
    d, _ = to_array(pd.read_csv(f / "distances.csv"), ["origin", "destination"], "km",
                    labels={"origin": regions, "destination": regions})
    mass = None
    if (f / "supply_mass.csv").exists():
        mass, _ = to_array(pd.read_csv(f / "supply_mass.csv"), ["year", "sector", "region"], labels=L)
    col = None
    if (f / "column_totals.csv").exists():
        col, _ = to_array(pd.read_csv(f / "column_totals.csv"), ["supplier", "region", "user"],
                          labels={"supplier": sectors, "region": regions, "user": sectors})
    return SDMData(va, y, Zn, Xn, d, mass, sectors, regions), col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdm-dir", default=str(HERE / "data" / "canada_sdm"))
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--kind", choices=["sf", "pdf"], default="sf")
    ap.add_argument("--supply-exponent", default="fixed1", help="fixed1 | fixed0 | estimate | <float>")
    ap.add_argument("--prior", choices=["weak", "gravity"], default="weak")
    ap.add_argument("--elasticities", default=str(HERE / "results" / "elasticities_sector.csv"))
    ap.add_argument("--sector-to-icio", default=None,
                    help="CSV sector,icio_industry,weight mapping SDM sectors to ICIO (for --prior gravity)")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--iter", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=2)
    ap.add_argument("--draws-for-flows", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if a.demo:
        from gravity_irio.synthetic import synthetic_sdm
        data, truth = synthetic_sdm(S=4, R=8, T=5, seed=1)
        col = None
        a.chains, a.iter, a.burn = 2, 1500, 500
    else:
        data, col = load_sdm(a.sdm_dir)
    print(f"national identity gap (should be ~0): {data.national_identity_gap():.2e}")

    se = a.supply_exponent
    se = float(se) if se not in ("fixed1", "fixed0", "estimate") else se
    prior = None
    if a.prior == "gravity":
        el = pd.read_csv(a.elasticities).set_index("industry")["delta"]
        if a.sector_to_icio:
            from gravity_irio.canada_io import concordance_matrix
            C = concordance_matrix(pd.read_csv(a.sector_to_icio), "sector", "icio_industry",
                                   data.sectors, list(el.index))
            delta_s = C @ el.to_numpy()
        else:
            delta_s = el.reindex(data.sectors).to_numpy()
        prior = prior_from_gravity(delta_s, data.dist, kind=a.kind)
    model = SpatialIRIO(data, prior=prior, kind=a.kind, supply_exponent=se)
    mode = model.fit_mode(sweeps=10, verbose=True)
    res = model.sample(n_iter=a.iter, burn=a.burn, thin=a.thin, chains=a.chains, init=mode, verbose=True)
    summ = res.summary()
    summ.to_csv(out / "sdm_posterior_summary.csv", index=False)
    print(summ.round(3).to_string())
    print("acceptance rates by sector:", res.accept.mean(axis=0).round(2))
    if (summ["rhat"] > 1.05).any():
        print("WARNING: some R-hat > 1.05 -- shape/scale may be weakly identified; run longer or "
              "use an informative prior (--prior gravity).")
    Z = model.posterior_flows(res, t=-1, col_totals=col, n_draws=a.draws_for_flows)
    np.save(out / "sdm_Z4_mean.npy", Z["mean"])
    np.save(out / "sdm_Z4_sd.npy", Z["sd"])
    for q, arr in Z.get("quantiles", {}).items():
        np.save(out / f"sdm_Z4_q{int(q * 100):02d}.npy", arr)
    np.save(out / "sdm_draws.npy", res.natural())


if __name__ == "__main__":
    main()
