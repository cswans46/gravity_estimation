"""Step 2: Canadian 4-D intermediate/final flows Z[i, j, b, u] (origin province, commodity,
destination province, user), by gravity-RAS, compared with import proportionality (IPA) and
bounded with the Schulte et al. (2021) block-randomisation envelope.

Trade margins (--trade-source)
  ipt                : observed interprovincial flows T[i,j,b] (Table 12-10-0101 / 12-10-0088)
  provincial-tables  : only provincial SUT/IO tables: T is first estimated by Cai's (2023)
                       calibration, RAS of d^-delta_j to domestic supply by origin and domestic
                       absorption by destination (Cai eq. 4-6), then split by user.
Elasticities (--delta-source)
  icio-pair    results/cai_pair.csv         theta_{k,m}  (supplier x user, ICIO)
  icio-sector  results/cai_sector.csv       theta_k      (Cai 2023 exactly)
  ipt          results/canada_ipt_elasticities_none.csv / _origin.csv  (StatCan IPT + ODBus)
Distances (--distances)
  province       results of distances.ces_distance or province_distances.csv (R x R)
  user-specific  results/canada_sector_pair_distances.npy (ODBus/BR activity-weighted)

Note (METHODS Sect. 3.2): province-pair distances with supplier-only elasticities reproduce IPA
exactly; departures from IPA come from user-varying elasticities and/or user-specific distances.

Usage:
    python canada_disaggregation.py --demo
    python canada_disaggregation.py --canada-dir data/canada --delta-source ipt --distances user-specific
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_irio.canada_io import concordance_matrix, load_canada, to_array
from gravity_irio.concordance import canonical_icio
from gravity_irio.disaggregate import (cai_interprovincial_trade, compare_splits, domestic_use,
                                       frechet_own_share_bounds, gravity_split, ipa_split, map_distances,
                                       map_pair_elasticities, own_province_share_by_user, schulte_envelope)
from gravity_irio.elasticities import pair_matrix

HERE = Path(__file__).resolve().parent


def to_long(Z, provinces, commodities, users):
    i, j, b, u = np.nonzero(Z)
    return pd.DataFrame({"origin": np.array(provinces)[i], "commodity": np.array(commodities)[j],
                         "destination": np.array(provinces)[b], "user": np.array(users)[u],
                         "value": Z[i, j, b, u]})


def _conc(cd, side, sectors):
    """concordance_<side>.csv (ICIO 2025) or concordance_<side>_naics2.csv (NAICS 2-digit)."""
    from gravity_irio.concordance import ICIO2025
    icio = all(str(s) in ICIO2025 or s == "FD" for s in sectors)
    return pd.read_csv(cd / (f"concordance_{side}.csv" if icio else f"concordance_{side}_naics2.csv"), dtype=str).astype(
        {"weight": float})


def _delta(a, res, comms, users, provs, cd):
    """Return (delta, delta_by) in the Canadian classification."""
    if a.delta_source in ("icio-pair", "icio-sector"):
        fn = "cai_pair.csv" if a.delta_source == "icio-pair" else "cai_sector.csv"
        el = pd.read_csv(res / fn)
        el["delta"] = -el["theta"]
        if a.delta_source == "icio-sector":
            el = el.rename(columns={"product": "supplier"})
            sup = [canonical_icio(x) for x in el["supplier"]]
            csup = concordance_matrix(pd.read_csv(cd / "concordance_supply.csv"), "commodity", "icio_industry",
                                      comms, list(el["supplier"]))
            return csup @ el["delta"].to_numpy(), "supplier"
        sup, usr = list(dict.fromkeys(el["supplier"])), list(dict.fromkeys(el["user"]))
        D = pair_matrix(el, sup, usr, "delta")
        rowmean = np.nanmean(D, axis=1, keepdims=True)
        D = np.where(np.isfinite(D), D, rowmean)  # thin pairs -> supplier mean
        csup = concordance_matrix(pd.read_csv(cd / "concordance_supply.csv"), "commodity", "icio_industry", comms, sup)
        cusr = concordance_matrix(pd.read_csv(cd / "concordance_user.csv"), "user", "icio_user", users, usr)
        return map_pair_elasticities(D, csup, cusr), "user"
    by = a.ipt_slope
    el = pd.read_csv(res / f"canada_ipt_elasticities_{by}.csv", dtype={"sector": str})
    sectors = list(dict.fromkeys(el["sector"]))
    csup = concordance_matrix(_conc(cd, "supply", sectors), "commodity", "icio_industry", comms, sectors)
    if by.startswith("none"):
        return csup @ el.set_index("sector").loc[sectors, "delta"].to_numpy(), "supplier"
    piv = el.pivot(index="sector", columns="region", values="delta").reindex(index=sectors, columns=provs)
    piv = piv.apply(lambda r: r.fillna(r.mean()), axis=1)
    return csup @ piv.to_numpy(), "origin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canada-dir", default=str(HERE / "data" / "canada"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--trade-source", choices=["ipt", "provincial-tables"], default="ipt")
    ap.add_argument("--delta-source", choices=["icio-pair", "icio-sector", "ipt"], default="icio-pair")
    ap.add_argument("--ipt-slope", choices=["none", "origin", "none_odbus", "origin_odbus"], default="none",
                    help="which canada_ipt_elasticities_<x>.csv to use")
    ap.add_argument("--distances", choices=["province", "user-specific"], default="province")
    ap.add_argument("--schulte-draws", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    out, res, cd = Path(a.out), Path(a.results), Path(a.canada_dir)
    out.mkdir(exist_ok=True)

    if a.demo:
        from gravity_irio.synthetic import synthetic_canada
        syn = synthetic_canada(R=6, J=4, U=5, seed=0)
        T, Udom, dist, delta, by = syn["T"], syn["Udom"], syn["dist"], syn["delta"], "user"
        provs = [f"P{i}" for i in range(6)]; comms = [f"c{j}" for j in range(4)]; users = [f"u{u}" for u in range(5)]
    else:
        c = load_canada(cd)
        T, dist, provs, comms, users = c["T"], c["dist"], c["provinces"], c["commodities"], c["users"]
        Udom = domestic_use(c["U"], c["M"])
        delta, by = _delta(a, res, comms, users, provs, cd)
        if a.distances == "user-specific":
            D4 = np.load(res / "canada_sector_pair_distances.npy")
            dprov = pd.read_csv(res / "canada_distance_provinces.csv")["prov"].tolist()
            dsec = pd.read_csv(res / "canada_distance_sectors.csv", dtype=str)["sector"].tolist()
            order = [dprov.index(p) for p in provs]
            D4 = D4[np.ix_(order, range(len(dsec)), order, range(len(dsec)))]
            csup = concordance_matrix(_conc(cd, "supply", dsec), "commodity", "icio_industry", comms, dsec)
            cu = _conc(cd, "user", dsec)
            # final-demand users: use all business activity (equal weights) unless an 'FD'
            # (population) layer was included in the location data
            if "FD" not in dsec:
                fd = cu[cu["icio_user"] == "FD"]
                cu = pd.concat([cu[cu["icio_user"] != "FD"],
                                pd.DataFrame([(u, s, w / len(dsec)) for u, w in zip(fd["user"], fd["weight"])
                                              for s in dsec], columns=["user", "icio_user", "weight"])])
            cusr = concordance_matrix(cu, "user", "icio_user", users, dsec)
            dist = map_distances(D4, csup, cusr)
            # cells without activity data fall back to province-pair distances
            dist = np.where(np.isfinite(dist), dist, c["dist"][:, None, :, None])
        if a.trade_source == "provincial-tables":
            sd = pd.read_csv(cd / "supply_domestic.csv")   # province, commodity, value
            S_dom, _ = to_array(sd, ["province", "commodity"], labels={"province": provs, "commodity": comms})
            dj = delta if by == "supplier" else (np.nanmean(delta, axis=1))
            d2 = c["dist"]
            T = cai_interprovincial_trade(S_dom, Udom.sum(axis=2), d2, dj)
            np.save(out / "canada_T_cai_calibrated.npy", T)

    Zi = ipa_split(T, Udom)
    Zg, diag = gravity_split(T, Udom, dist, delta, delta_by=by)
    cmp = compare_splits(Zg, Zi)
    print(f"share of flows reallocated by gravity vs IPA: {cmp['share_reallocated']:.3%}")
    print(f"RAS non-converged cells: {len(diag['nonconverged'])}; column rescale range "
          f"{np.nanmin(diag['col_scale']):.3f}-{np.nanmax(diag['col_scale']):.3f}")
    for name, Z in (("gravity", Zg), ("ipa", Zi)):
        np.save(out / f"canada_Z4_{name}.npy", Z)
        to_long(Z, provs, comms, users).to_csv(out / f"canada_Z4_{name}.csv", index=False)
    summ = pd.DataFrame({"user": users, "own_share_gravity": own_province_share_by_user(Zg),
                         "own_share_ipa": own_province_share_by_user(Zi)})
    summ["own_share_frechet_min"], summ["own_share_frechet_max"] = frechet_own_share_bounds(T, Udom)
    if a.schulte_draws:
        env = schulte_envelope(T, Udom, n_draws=a.schulte_draws, stat=own_province_share_by_user)
        summ["own_share_schulte_min"], summ["own_share_schulte_max"] = env["min"], env["max"]
    summ.to_csv(out / "canada_own_province_share_by_user.csv", index=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
