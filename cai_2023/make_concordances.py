"""Write concordance_supply.csv and concordance_user.csv (Canadian SUT -> ICIO 2025).

Inputs in --canada-dir:
    supply_national.csv   commodity, industry, value   (national supply table, IOIC industries)
    provincial_use.csv    province, commodity, user, value   (users = IOIC industries + final demand)
Optional:
    ioic_overrides.csv    ioic, icio2025, weight       (resolve aggregates such as BS21E000)
    icio_canada_output.csv icio2025, value             (ICIO CAN gross output, splits ambiguous codes)
Final-demand users (codes not starting with BS/NP/GS) map to 'FD'.
Usage:
    python make_concordances.py --canada-dir data/canada
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_irio.canada_io import to_array
from gravity_irio.concordance import ICIO2025, commodity_concordance, industry_concordance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canada-dir", default="data/canada")
    a = ap.parse_args()
    f = Path(a.canada_dir)
    sup = pd.read_csv(f / "supply_national.csv")
    comms, inds = list(dict.fromkeys(sup["commodity"])), list(dict.fromkeys(sup["industry"]))
    V, _ = to_array(sup, ["commodity", "industry"], labels={"commodity": comms, "industry": inds})
    overrides = None
    if (f / "ioic_overrides.csv").exists():
        o = pd.read_csv(f / "ioic_overrides.csv")
        overrides = {k: list(g["icio2025"]) for k, g in o.groupby("ioic")}
    w = None
    if (f / "icio_canada_output.csv").exists():
        w = pd.read_csv(f / "icio_canada_output.csv").set_index("icio2025")["value"].to_dict()
    icio = list(ICIO2025)
    rep = []
    Cs = commodity_concordance(V, inds, icio, overrides=overrides, split_weights=w, report=rep)
    rows = [(c, icio[k], Cs[n, k]) for n, c in enumerate(comms) for k in np.flatnonzero(Cs[n] > 1e-12)]
    pd.DataFrame(rows, columns=["commodity", "icio_industry", "weight"]).to_csv(f / "concordance_supply.csv", index=False)

    users = list(dict.fromkeys(pd.read_csv(f / "provincial_use.csv")["user"]))
    is_ind = [str(u).upper()[:2] in ("BS", "NP", "GS") for u in users]
    Cu = industry_concordance([u for u, i in zip(users, is_ind) if i], icio, overrides=overrides,
                              split_weights=w, report=rep)
    rows, n = [], 0
    for u, i in zip(users, is_ind):
        if i:
            rows += [(u, icio[k], Cu[n, k]) for k in np.flatnonzero(Cu[n] > 1e-12)]
            n += 1
        else:
            rows.append((u, "FD", 1.0))
    pd.DataFrame(rows, columns=["user", "icio_user", "weight"]).to_csv(f / "concordance_user.csv", index=False)
    if rep:
        print("ambiguous IOIC codes split by weights (add ioic_overrides.csv to fix):")
        for c, cands in dict(rep).items():
            print(f"  {c}: {cands}")


if __name__ == "__main__":
    main()
