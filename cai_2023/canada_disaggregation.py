"""Step 2: split Canadian interprovincial flows by demanding sector, Z[i, j, b, u].

Compares import proportionality (IPA) with gravity-RAS using ICIO supplier x user
distance elasticities (output of estimation.py).

Usage:
    python canada_disaggregation.py --canada-dir data/canada --elasticities results/elasticities_pair_filled.csv
    python canada_disaggregation.py --demo        # synthetic data, no inputs needed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_irio.canada_io import concordance_matrix, load_canada
from gravity_irio.disaggregate import (compare_splits, domestic_use, gravity_split, ipa_split,
                                       map_pair_elasticities)
from gravity_irio.elasticities import pair_matrix

HERE = Path(__file__).resolve().parent


def to_long(Z, provinces, commodities, users):
    i, j, b, u = np.nonzero(Z)
    return pd.DataFrame({"origin": np.array(provinces)[i], "commodity": np.array(commodities)[j],
                         "destination": np.array(provinces)[b], "user": np.array(users)[u],
                         "value": Z[i, j, b, u]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canada-dir", default=str(HERE / "data" / "canada"))
    ap.add_argument("--elasticities", default=str(HERE / "results" / "elasticities_pair_filled.csv"))
    ap.add_argument("--kappa", type=float, default=0.0, help="weight on ICIO home-bias coefficients (0=off)")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if a.demo:
        from gravity_irio.synthetic import synthetic_canada
        syn = synthetic_canada(R=6, J=4, U=5, seed=0)
        T, Udom, dist, delta = syn["T"], syn["Udom"], syn["dist"], syn["delta"]
        provs = [f"P{i}" for i in range(6)]; comms = [f"c{j}" for j in range(4)]; users = [f"u{u}" for u in range(5)]
        home = None
    else:
        cd = load_canada(a.canada_dir)
        T, dist = cd["T"], cd["dist"]
        provs, comms, users = cd["provinces"], cd["commodities"], cd["users"]
        Udom = domestic_use(cd["U"], cd["M"])
        el = pd.read_csv(a.elasticities)
        sup = list(dict.fromkeys(el["supplier"])); usr = list(dict.fromkeys(el["user"]))
        D = pair_matrix(el, sup, usr, "delta")
        csup = concordance_matrix(pd.read_csv(Path(a.canada_dir) / "concordance_supply.csv"),
                                  "commodity", "icio_industry", comms, sup)
        cusr = concordance_matrix(pd.read_csv(Path(a.canada_dir) / "concordance_user.csv"),
                                  "user", "icio_user", users, usr)
        delta = map_pair_elasticities(D, csup, cusr)
        home = map_pair_elasticities(pair_matrix(el, sup, usr, "home"), csup, cusr) if "home" in el else None

    Zi = ipa_split(T, Udom)
    Zg, diag = gravity_split(T, Udom, dist, delta, kappa=a.kappa, home=home)
    cmp = compare_splits(Zg, Zi)
    print(f"share of flows reallocated by gravity vs IPA: {cmp['share_reallocated']:.3%}")
    print(f"RAS non-converged (j,b) cells: {len(diag['nonconverged'])}; "
          f"column-total rescale range: {diag['col_scale'].min():.3f}-{diag['col_scale'].max():.3f}")
    to_long(Zg, provs, comms, users).to_csv(out / "canada_Z4_gravity.csv", index=False)
    to_long(Zi, provs, comms, users).to_csv(out / "canada_Z4_ipa.csv", index=False)
    np.save(out / "canada_Z4_gravity.npy", Zg)
    np.save(out / "canada_Z4_ipa.npy", Zi)
    pd.DataFrame({"user": users, "own_share_gravity": cmp["own_province_share_by_user_gravity"],
                  "own_share_ipa": cmp["own_province_share_by_user_ipa"]}).to_csv(
        out / "canada_own_province_share_by_user.csv", index=False)


if __name__ == "__main__":
    main()
