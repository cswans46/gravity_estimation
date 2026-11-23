"""Step 1b: distance elasticities from Statistics Canada interprovincial flows + business locations.

1. Business locations (ODBus and/or Business Register counts by CSD) -> employment weights
   (observed, imputed, benchmarked to province x sector employment, e.g. LFS 14-10-0023 or
   SEPH 14-10-0202) -> 25 km grid -> CES distances by (origin province, supplying sector) x
   (destination province, using sector): results/canada_sector_pair_distances.npy
2. Interprovincial flows T[i, commodity, b, year] (Table 12-10-0101 / 12-10-0088) -> producing
   sectors via the supply table -> Cai (2023) eq. (9) within Canada (OLS log z, origin-year and
   destination-year FE, provincial border dummy, common official language) for each sector,
   optionally one elasticity per sector x origin province (--slope-by origin).

Distances (--distances):
    population  (default) province_distances.csv: population-weighted CES distances from the
                2021 GAF (or the capitals/land-area fallback), the same concept as CEPII distwces
    odbus       ODBus activity-weighted (origin sector x destination) distances where ODBus has
                coverage (in practice ON, BC, AB), population distances elsewhere -- a sensitivity

Inputs (tidy CSVs in --canada-dir, written by prepare_data.py):
    province_distances.csv        origin, destination, km
    --odbus                       ODBus v1.0 csv as downloaded (only for --distances odbus)
    br_csd.csv + csd_centroids.csv  optional Business Register counts by CSD and CSD centroids
    employment_benchmark.csv      optional prov, sector, value
    ipt_flows.csv                 year, origin, destination, commodity, value
    sut_supply.csv + sut_member_roles.csv  (all years pooled; else supply_national.csv)
Usage:
    python canada_elasticities.py --canada-dir data/canada --slope-by none
    python canada_elasticities.py --canada-dir data/canada --distances odbus --odbus ODBus_v1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_irio.business_locations import (PROVINCES, aggregate_grid, benchmark_weights, expand_coarse,
                                             impute_employees, load_br_csd, load_odbus, sector_pair_distances)
from gravity_irio.canada_gravity import commodity_flows_to_sectors, estimate_ipt_elasticities
from gravity_irio.canada_io import to_array
from gravity_irio.canada_sut import NAICS2_SECTORS, industry_sector_matrix, load_overrides
from gravity_irio.concordance import commodity_to_industry

HERE = Path(__file__).resolve().parent


def _supply(f):
    """National supply V (commodity x IOIC industry), pooled over years, and industry labels."""
    labels = {}
    if (f / "sut_supply.csv").exists() and (f / "sut_member_roles.csv").exists():
        R = pd.read_csv(f / "sut_member_roles.csv", dtype=str)
        ind = R[R["role"] == "industry"]
        code = dict(zip(ind["member"], ind["code"]))
        labels = dict(zip(ind["code"], ind["label"]))
        S = pd.read_csv(f / "sut_supply.csv", dtype={"year": str})
        S = S[S["member"].isin(code) & (S["province"] != "CA")]
        sup = S.assign(industry=S["member"].map(code)).groupby(["commodity", "industry"], as_index=False)["value"].sum()
    else:
        sup = pd.read_csv(f / "supply_national.csv")
    return sup, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canada-dir", default=str(HERE / "data" / "canada"))
    ap.add_argument("--distances", choices=["population", "odbus"], default="population")
    ap.add_argument("--odbus", default=None, help="ODBus_v1.csv (default: <canada-dir>/odbus.csv)")
    ap.add_argument("--cell-km", type=float, default=25.0)
    ap.add_argument("--impute", choices=["median", "count"], default="median")
    ap.add_argument("--slope-by", choices=["none", "origin", "destination"], default="none")
    ap.add_argument("--estimator", choices=["ols", "ppml"], default="ols")
    ap.add_argument("--drop", default="NU", help="comma-separated provinces to exclude from estimation")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    f, out = Path(a.canada_dir), Path(a.out)
    out.mkdir(exist_ok=True)
    sectors = list(NAICS2_SECTORS)

    # ---- 1. distances
    dp = pd.read_csv(f / "province_distances.csv")
    provs = [p for p in PROVINCES if p in set(dp["origin"])]
    Dp, _ = to_array(dp, ["origin", "destination"], "km", labels={"origin": provs, "destination": provs})
    dist = Dp
    tag = a.slope_by
    if a.distances == "odbus":
        tag = f"{a.slope_by}_odbus"
        pts = []
        odp = Path(a.odbus) if a.odbus else f / "odbus.csv"
        od = load_odbus(odp, sector_level="naics2")
        print(f"ODBus: kept {od.attrs['n_kept']} of {od.attrs['n_raw']} records; NAICS source "
              f"{od.attrs['naics_source']}; {od.attrs['n_no_sector']} without a sector")
        pts.append(impute_employees(od, a.impute))
        if (f / "br_csd.csv").exists():
            br = load_br_csd(f / "br_csd.csv", pd.read_csv(f / "csd_centroids.csv"), sector_level="naics2")
            pts.append(br.assign(weight=br["employees"]))
        loc = pd.concat(pts, ignore_index=True)
        if (f / "employment_benchmark.csv").exists():
            loc = benchmark_weights(loc, pd.read_csv(f / "employment_benchmark.csv"))
        cov = loc.groupby("prov").size().reindex(provs).fillna(0).astype(int)
        print("ODBus records by province:", cov.to_dict())
        D4, dprov, dsec = sector_pair_distances(aggregate_grid(loc, a.cell_km), provs, sectors)
        np.save(out / "canada_sector_pair_distances.npy", D4)
        pd.Series(dprov).to_csv(out / "canada_distance_provinces.csv", index=False, header=["prov"])
        pd.Series(dsec).to_csv(out / "canada_distance_sectors.csv", index=False, header=["sector"])
        with np.errstate(divide="ignore", invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            D3 = np.nanmean(np.where(np.isfinite(D4), D4, np.nan) ** -1.0, axis=3) ** -1.0   # (R, K, R)
        # pairs where either province lacks ODBus coverage in that sector -> population distances
        thin = cov.to_numpy() < 500
        bad = ~np.isfinite(D3) | thin[:, None, None] | thin[None, None, :]
        dist = np.where(bad, Dp[:, None, :], D3)
        print(f"ODBus distances used for {1 - bad.mean():.1%} of (origin, sector, destination) cells; "
              f"population distances elsewhere")

    # ---- 2. IPT flows -> sector flows -> elasticities
    ipt = pd.read_csv(f / "ipt_flows.csv")
    years = sorted(ipt["year"].unique())
    comms = list(dict.fromkeys(ipt["commodity"]))
    T, _ = to_array(ipt, ["origin", "commodity", "destination", "year"],
                    labels={"origin": provs, "destination": provs, "commodity": comms, "year": years})
    sup, labels = _supply(f)
    inds = list(dict.fromkeys(sup["industry"]))
    V, _ = to_array(sup, ["commodity", "industry"], labels={"commodity": comms, "industry": inds})
    C = commodity_to_industry(V) @ industry_sector_matrix(inds, sectors, labels, load_overrides(f))
    unm = T.sum(axis=(0, 2, 3))[C.sum(axis=1) == 0].sum() / T.sum()
    print(f"IPT value without a supplying industry in the SUT (dropped): {unm:.2%}")
    Tk = commodity_flows_to_sectors(T, C)
    res = estimate_ipt_elasticities(Tk, dist, provs, sectors, estimator=a.estimator,
                                    slope_by=None if a.slope_by == "none" else a.slope_by,
                                    drop=tuple(x for x in a.drop.split(",") if x))
    res.to_csv(out / f"canada_ipt_elasticities_{tag}.csv", index=False)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
