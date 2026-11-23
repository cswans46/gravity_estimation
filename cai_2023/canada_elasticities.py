"""Step 1b: distance elasticities from Statistics Canada interprovincial flows + business locations.

1. Business locations (ODBus and/or Business Register counts by CSD) -> employment weights
   (observed, imputed, benchmarked to province x sector employment, e.g. LFS 14-10-0023 or
   SEPH 14-10-0202) -> 25 km grid -> CES distances by (origin province, supplying sector) x
   (destination province, using sector): results/canada_sector_pair_distances.npy
2. Interprovincial flows T[i, commodity, b, year] (Table 12-10-0101 / 12-10-0088) -> producing
   sectors via the supply table -> Cai (2023) eq. (9) within Canada (OLS log z, origin-year and
   destination-year FE, provincial border dummy, common official language) for each sector,
   optionally one elasticity per sector x origin province (--slope-by origin).

Inputs (tidy CSVs in --canada-dir):
    odbus.csv                     ODBus v1.0 as downloaded (columns detected loosely)
    br_csd.csv + csd_centroids.csv  optional Business Register counts by CSD and CSD centroids
    employment_benchmark.csv      optional prov, sector, value
    ipt_flows.csv                 year, origin, destination, commodity, value
    supply_national.csv           commodity, industry (IOIC), value   (commodity -> industry)
Usage:
    python canada_elasticities.py --canada-dir data/canada --sector-level naics2 --slope-by origin
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
from gravity_irio.concordance import commodity_concordance, commodity_to_industry, naics_level_concordance

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canada-dir", default=str(HERE / "data" / "canada"))
    ap.add_argument("--sector-level", default="naics2",
                    help="naics2 (ODBus derived_NAICS sectors, default) | naics3 | naics4 ... | icio")
    ap.add_argument("--cell-km", type=float, default=25.0)
    ap.add_argument("--impute", choices=["median", "count"], default="median")
    ap.add_argument("--slope-by", choices=["none", "origin", "destination"], default="none")
    ap.add_argument("--estimator", choices=["ols", "ppml"], default="ols")
    ap.add_argument("--drop", default="NU", help="comma-separated provinces to exclude from estimation")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    f, out = Path(a.canada_dir), Path(a.out)
    out.mkdir(exist_ok=True)

    # ---- 1. locations -> distances
    pts = []
    if (f / "odbus.csv").exists():
        od = load_odbus(f / "odbus.csv", sector_level=a.sector_level)
        print(f"ODBus: kept {od.attrs['n_kept']} of {od.attrs['n_raw']} records; NAICS source "
              f"{od.attrs['naics_source']}; {od.attrs['n_no_sector']} without a code at the requested level")
        od = impute_employees(od, a.impute)
        if a.sector_level.startswith("naics") and a.sector_level != "naics2":
            od = expand_coarse(od, int(a.sector_level[-1]))
        pts.append(od)
    if (f / "br_csd.csv").exists():
        br = load_br_csd(f / "br_csd.csv", pd.read_csv(f / "csd_centroids.csv"), sector_level=a.sector_level)
        pts.append(br.assign(weight=br["employees"]))
    if not pts:
        raise SystemExit("need odbus.csv and/or br_csd.csv")
    loc = pd.concat(pts, ignore_index=True)
    if (f / "employment_benchmark.csv").exists():
        loc = benchmark_weights(loc, pd.read_csv(f / "employment_benchmark.csv"))
    D4, provs, sectors = sector_pair_distances(aggregate_grid(loc, a.cell_km))
    np.save(out / "canada_sector_pair_distances.npy", D4)
    pd.Series(provs).to_csv(out / "canada_distance_provinces.csv", index=False, header=["prov"])
    pd.Series(sectors).to_csv(out / "canada_distance_sectors.csv", index=False, header=["sector"])

    # ---- 2. IPT flows -> sector flows -> elasticities
    ipt = pd.read_csv(f / "ipt_flows.csv")
    years = sorted(ipt["year"].unique())
    comms = list(dict.fromkeys(ipt["commodity"]))
    T, _ = to_array(ipt, ["origin", "commodity", "destination", "year"],
                    labels={"origin": provs, "destination": provs, "commodity": comms, "year": years})
    sup = pd.read_csv(f / "supply_national.csv")
    inds = list(dict.fromkeys(sup["industry"]))
    V, _ = to_array(sup, ["commodity", "industry"], labels={"commodity": comms, "industry": inds})
    if a.sector_level == "icio":
        C = commodity_concordance(V, inds, icio_codes=sectors)
    else:
        # commodity -> producing IOIC industry (supply table) -> NAICS sector at the ODBus level;
        # IOIC aggregates coarser than the level are split by national activity in the location data
        act = loc.groupby("sector")["weight"].sum().to_dict()
        rep = []
        C = commodity_to_industry(V) @ naics_level_concordance(inds, sectors, int(a.sector_level[-1]),
                                                               split_weights=act, report=rep)
        for code, cands in dict(rep).items():
            print(f"  IOIC {code} split over {cands} by activity weights")
    Tk = commodity_flows_to_sectors(T, C)
    res = estimate_ipt_elasticities(Tk, D4, provs, sectors, estimator=a.estimator,
                                    slope_by=None if a.slope_by == "none" else a.slope_by,
                                    drop=tuple(x for x in a.drop.split(",") if x))
    res.to_csv(out / f"canada_ipt_elasticities_{a.slope_by}.csv", index=False)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
