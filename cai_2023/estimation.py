"""Step 1: distance elasticities from the OECD ICIO 2025, following Cai (2023) exactly.

Cai (2023) eq. (9), product by product:
    log z_ijk = xi_k + theta_k log d_ij + phi_k b_ij + c_k l_ij + mu_ik + nu_jk + eps_ijk
  OLS with exporter and importer fixed effects, zero flows dropped, robust SEs,
  d = CEPII distwces, b = border dummy (i != j), l = common official language,
  flows summed over all users (intermediate + final).
Panel extension (default): mu_ik,t and nu_jk,t over all 28 years with a common theta_k.
Cross-section exactly as Cai: --year 2009 (and --sample eu27 for his 27-country sample).

Outputs (./results):
    cai_sector.csv   theta_k by ICIO industry (Cai Table 1 layout)
    cai_pair.csv     theta_{k,m} by supplying x using industry (+ final demand), same spec
Usage:
    python estimation.py --data-dir "D:/Models/OECD Research" --cepii-dir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gravity_irio.cai import EU27_PRE_BREXIT_EX_HRV, cai_by_product, cai_estimate
from gravity_irio.cepii import bilateral_arrays, load_dist_cepii
from gravity_irio.icio import aggregate_regions, icio_to_numpy, load_icio_rdata, sector_flows

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"D:\Models\OECD Research")
    ap.add_argument("--stem", default="ICIO2025econ")
    ap.add_argument("--cepii-dir", default=str(HERE))
    ap.add_argument("--dist-col", default="distwces")
    ap.add_argument("--estimator", choices=["ols", "ppml"], default="ols", help="ols = Cai; ppml = robustness")
    ap.add_argument("--sample", choices=["all", "eu27"], default="all")
    ap.add_argument("--year", type=int, default=None, help="single cross-section (Cai used 2009)")
    ap.add_argument("--keep-splits", action="store_true", help="keep CN1/CN2, MX1/MX2 instead of CHN/MEX")
    ap.add_argument("--level", choices=["sector", "pair", "both"], default="both")
    ap.add_argument("--min-positive", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    b = icio_to_numpy(load_icio_rdata(a.data_dir, a.stem))
    Z5, FD4, X3, countries = b["Z5"], b["FD4"], b["X3"], b["countries"]
    if not a.keep_splits:
        Z5, FD4, X3, countries = aggregate_regions(Z5, FD4, X3, countries)
    years = b["years"]
    yidx = None if a.year is None else [list(map(int, years)).index(a.year)]
    print(f"ICIO: {len(years)} years, {len(countries)} regions, {len(b['industries'])} industries")

    bil = bilateral_arrays(countries, load_dist_cepii(Path(a.cepii_dir) / "dist_cepii.xls"), dist_col=a.dist_col)
    print("dropped (no CEPII match):", [c for c, v in zip(countries, bil["valid"]) if not v])
    sample = EU27_PRE_BREXIT_EX_HRV if a.sample == "eu27" else None
    kw = dict(sample=sample, years=yidx, estimator=a.estimator)

    if a.level in ("sector", "both"):
        F = sector_flows(Z5, FD4, X3, include_final_demand=True)["F"]   # summed over all users
        sec = cai_by_product(F, bil, countries, b["industries"], **kw)
        sec.to_csv(out / "cai_sector.csv", index=False)
        print(sec[["product", "theta", "se", "border", "language", "n_obs"]].round(3).to_string())

    if a.level in ("pair", "both"):
        rows = []
        users = list(range(len(b["industries"]))) + [None]
        for k, kn in enumerate(b["industries"]):
            for m in users:
                F = FD4[:, :, k, :] if m is None else Z5[:, :, k, :, m]
                lab = dict(supplier=kn, user="FD" if m is None else b["industries"][m])
                if (F > 0).sum() < a.min_positive:
                    rows.append(dict(lab, theta=float("nan")))
                    continue
                rows.append(dict(lab, **cai_estimate(F, bil, countries, **kw)))
        pd.DataFrame(rows).to_csv(out / "cai_pair.csv", index=False)


if __name__ == "__main__":
    main()
