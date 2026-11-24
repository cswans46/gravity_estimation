"""Provincial supply-use tables (36-10-0478) -> NAICS 2-digit sectors and the inputs of the
spatial Durbin routine (spatial_irio.py).

Sector level: NAICS 2-digit sectors (31-33, 44-45 and 48-49 combined), i.e. the ODBus
``derived_NAICS`` level, so that the SDM, the IPT elasticities and the business-location
distances share one classification. An IOIC code carries its NAICS prefix
('BS111A00' -> 11, 'GS611100' -> 61); codes whose prefix is a single digit (aggregates) are
resolved from the label ('Retail trade' -> 44-45) or, failing that, must be listed in
``ioic_sector_overrides.csv`` (ioic, sector).

Commodities -> supplying sectors: market shares of the supply table, D[s, c] = V[c, s] / q_c
(fixed product-sales structure, Eurostat model D). Provincial shares are used for flows that
originate in a province (exports), national shares otherwise.

SDM inputs (one row per year, sector, region unless stated):
  regional_va.csv        VA = output - intermediate use (basic prices, all origins)
  regional_fd.csv        domestic-origin final demand located in the region (households, NPISH,
                         government, GFCF, inventories), net of international imports
                         (import proportionality within province x commodity), rescaled so the
                         national identity X = Z 1 + F holds (residual absorbs discrepancies)
  exports_direct.csv     international exports supplied by the origin province (12-10-0101)
  national_Z.csv         year, supplier, user: domestic intermediate flows, D @ U_dom
  national_X.csv         national gross output
  distances.csv          origin, destination, km
  supply_mass_exogenous.csv  lagged VA (t-1; first year: own year) for the exogenous-mass run
  ipt_flows.csv          IPT flows by sector (for the Torój home-share prior)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .business_locations import PROVINCES
from .concordance import ioic_to_naics_prefix, naics_level

NAICS2_SECTORS = ["11", "21", "22", "23", "31-33", "41", "44-45", "48-49", "51", "52", "53", "54",
                  "55", "56", "61", "62", "71", "72", "81", "91"]
LABEL_SECTORS = [  # for IOIC aggregates without a 2-digit prefix
    (r"agricultur|crop|animal|forestry|fishing", "11"), (r"mining|oil and gas|quarrying", "21"),
    (r"utilit", "22"), (r"construction", "23"), (r"manufactur", "31-33"), (r"wholesale", "41"),
    (r"retail", "44-45"), (r"transportation|warehousing", "48-49"), (r"information|cultural", "51"),
    (r"finance|insurance", "52"), (r"real estate|rental|leasing|owner-occupied", "53"),
    (r"professional|scientific|technical", "54"), (r"management of companies", "55"),
    (r"administrative|waste management", "56"), (r"education", "61"), (r"health|social assistance", "62"),
    (r"arts|entertainment|recreation", "71"), (r"accommodation|food services", "72"),
    (r"other services|repair|personal|religious|civic", "81"), (r"public administration|government", "91"),
]


def ioic_sector(code: str, label: str = "", overrides: dict | None = None) -> str:
    if overrides and code in overrides:
        return overrides[code]
    p = ioic_to_naics_prefix(code)
    if len(p) >= 2:
        s = naics_level(p[:2], 2)
        return "91" if s == "92" else s
    for rx, s in LABEL_SECTORS:
        if re.search(rx, str(label), re.I):
            return s
    raise KeyError(f"IOIC {code} ({label}): add it to ioic_sector_overrides.csv")


def load_overrides(folder) -> dict:
    f = Path(folder) / "ioic_sector_overrides.csv"
    if not f.exists():
        return {}
    o = pd.read_csv(f, dtype=str)
    return dict(zip(o["ioic"], o["sector"]))


def member_sectors(roles: pd.DataFrame, overrides=None) -> dict:
    """member -> sector for every member with role 'industry'."""
    out = {}
    for r in roles[roles["role"] == "industry"].itertuples():
        code = r.code if isinstance(r.code, str) else r.member
        out[r.member] = ioic_sector(code, r.label, overrides)
    return out


def industry_sector_matrix(industries, sectors=NAICS2_SECTORS, labels=None, overrides=None):
    labels = labels or {}
    M = np.zeros((len(industries), len(sectors)))
    for n, c in enumerate(industries):
        M[n, list(sectors).index(ioic_sector(c, labels.get(c, ""), overrides))] = 1.0
    return M


def _pivot(df, index, columns, rows, cols):
    p = df.pivot_table(index=index, columns=columns, values="value", aggfunc="sum")
    return p.reindex(index=rows, columns=cols).fillna(0.0).to_numpy()


def build_sdm_inputs(canada_dir, sdm_dir, distances=None, years=None, provinces=PROVINCES,
                     sectors=NAICS2_SECTORS, verbose=True) -> dict:
    f, o = Path(canada_dir), Path(sdm_dir)
    o.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(f / "sut_member_roles.csv", dtype=str)
    msec = member_sectors(roles, load_overrides(f))
    role = dict(zip(roles["member"], roles["role"]))
    U = pd.read_csv(f / "sut_use.csv", dtype={"year": str})
    S = pd.read_csv(f / "sut_supply.csv", dtype={"year": str})
    M = pd.read_csv(f / "international_imports_panel.csv", dtype={"year": str})
    X = pd.read_csv(f / "international_exports_panel.csv", dtype={"year": str})
    ipt = pd.read_csv(f / "ipt_flows.csv", dtype={"year": str})
    for d in (U, S):
        d["role"] = d["member"].map(role)
        d["sector"] = d["member"].map(msec)
    yrs = sorted(set(U["year"]) & set(S["year"]) & set(M["year"]) & set(X["year"]))
    if years:
        yrs = [y for y in yrs if int(y) in {int(v) for v in years}]
    if not yrs:
        raise RuntimeError("no common years between the SUT (36-10-0478) and the IPT table (12-10-0101)")
    provs = [p for p in provinces if p in set(U["province"])]
    sectors = all_sectors = list(sectors)
    R, K = len(provs), len(sectors)
    rows = {k: [] for k in ("va", "fd", "xd", "Z", "X")}
    report, Dnat_last = [], None
    ipt_sec = []
    for y in yrs:
        u, s = U[U["year"] == y], S[S["year"] == y]
        s_ind = s[s["role"] == "industry"]
        s_ind = s_ind[s_ind["province"] != "CA"]
        comms = sorted(set(u["commodity"]) | set(s_ind["commodity"]))
        C = len(comms)
        # provincial supply by sector V[r, c, k] and market shares
        V = np.stack([_pivot(s_ind[s_ind["province"] == p], "commodity", "sector", comms, sectors) for p in provs])
        Vn = V.sum(axis=0)
        q = Vn.sum(axis=1, keepdims=True)
        Dn = np.divide(Vn, q, out=np.zeros_like(Vn), where=q > 0).T                      # (K, C)
        qr = V.sum(axis=2, keepdims=True)
        Dr = np.where(qr > 0, V / np.where(qr > 0, qr, 1), Dn.T[None]).transpose(0, 2, 1)  # (R, K, C)
        x = V.sum(axis=1)                                                                   # (R, K)
        # use: intermediate by user sector, final demand total, international import shares
        ui = u[(u["role"] == "industry") & (u["province"] != "CA")]
        uf = u[u["role"].str.startswith("final_", na=False) & (u["province"] != "CA")]
        Ui = np.stack([_pivot(ui[ui["province"] == p], "commodity", "sector", comms, sectors) for p in provs])
        Fi = np.stack([uf[uf["province"] == p].groupby("commodity")["value"].sum().reindex(comms).fillna(0).to_numpy()
                       for p in provs])                                                     # (R, C)
        Mi = M[M["year"] == y].pivot_table(index="province", columns="commodity", values="value", aggfunc="sum")
        Mi = Mi.reindex(index=provs, columns=comms).fillna(0).to_numpy()
        tot = Ui.sum(axis=2) + np.clip(Fi, 0, None)
        m = np.clip(np.divide(Mi, tot, out=np.zeros_like(Mi), where=tot > 0), 0, 1)
        Udom = Ui * (1 - m)[:, :, None]
        Fdom = Fi * (1 - m)
        Z = np.einsum("kc,rcu->ku", Dn, Udom)
        va = x - Ui.sum(axis=1)
        if (va < 0).any():
            report.append(f"{y}: {int((va < 0).sum())} negative VA cells set to 0")
            va = np.clip(va, 0, None)
        yfd = np.clip(np.einsum("kc,rc->rk", Dn, Fdom), 0, None)
        Xe = X[X["year"] == y].pivot_table(index="province", columns="commodity", values="value", aggfunc="sum")
        Xe = Xe.reindex(index=provs, columns=comms).fillna(0).to_numpy()
        ydir = np.einsum("rkc,rc->rk", Dr, Xe)
        # national identity X = Z 1 + F: residual F, regional FD rescaled (exports kept as observed)
        Xn = x.sum(axis=0)
        Fres = Xn - Z.sum(axis=1)
        target = Fres - ydir.sum(axis=0)
        for k in range(K):
            if Xn[k] <= 0:
                continue
            if target[k] > 0:
                base = yfd[:, k] if yfd[:, k].sum() > 0 else va[:, k]
                yfd[:, k] = target[k] * base / max(base.sum(), 1e-12)
            else:
                yfd[:, k] = 0.0
                if ydir[:, k].sum() > 0 and Fres[k] > 0:
                    ydir[:, k] *= Fres[k] / ydir[:, k].sum()
                report.append(f"{y} {sectors[k]}: residual final demand <= exports; exports rescaled")
        for r, p in enumerate(provs):
            for k, sname in enumerate(sectors):
                rows["va"].append((y, sname, p, va[r, k]))
                rows["fd"].append((y, sname, p, yfd[r, k]))
                rows["xd"].append((y, sname, p, ydir[r, k]))
        for k, sname in enumerate(sectors):
            rows["X"].append((y, sname, Xn[k]))
            for v, uname in enumerate(sectors):
                rows["Z"].append((y, sname, uname, Z[k, v]))
        # IPT flows by supplying sector
        it = ipt[ipt["year"] == y]
        it = it[it["commodity"].isin(comms)]
        cidx = pd.Index(comms).get_indexer(it["commodity"])
        w = Dn[:, cidx] * it["value"].to_numpy()[None]
        for k, sname in enumerate(sectors):
            ipt_sec.append(pd.DataFrame({"year": y, "origin": it["origin"].values, "destination": it["destination"].values,
                                         "commodity": sname, "value": w[k]}))
        Dnat_last = (comms, Dn)
    cols = ["year", "sector", "region", "value"]
    Xdf = pd.DataFrame(rows["X"], columns=["year", "sector", "value"])
    live = set(Xdf.groupby("sector")["value"].sum().loc[lambda v: v > 0].index)
    dropped = [s_ for s_ in sectors if s_ not in live]
    if dropped:
        report.append(f"sectors without output dropped: {dropped}")
    keep = lambda r: r[1] in live and (len(r) < 4 or not isinstance(r[2], str) or r[2] in live or r[2] in provs)
    rows = {k: [r for r in v if keep(r)] for k, v in rows.items()}
    sectors = [s_ for s_ in sectors if s_ in live]
    va_df = pd.DataFrame(rows["va"], columns=cols)
    va_df.to_csv(o / "regional_va.csv", index=False)
    pd.DataFrame(rows["fd"], columns=cols).to_csv(o / "regional_fd.csv", index=False)
    pd.DataFrame(rows["xd"], columns=cols).to_csv(o / "exports_direct.csv", index=False)
    pd.DataFrame(rows["Z"], columns=["year", "supplier", "user", "value"]).to_csv(o / "national_Z.csv", index=False)
    pd.DataFrame(rows["X"], columns=["year", "sector", "value"]).to_csv(o / "national_X.csv", index=False)
    lag = va_df.sort_values("year").copy()
    lag["value"] = lag.groupby(["sector", "region"])["value"].shift(1).fillna(lag["value"])
    lag.to_csv(o / "supply_mass_exogenous.csv", index=False)
    ipt_all = pd.concat(ipt_sec)
    (ipt_all[ipt_all["commodity"].isin(live)].groupby(["year", "origin", "destination", "commodity"], as_index=False)["value"].sum()
     .to_csv(o / "ipt_flows.csv", index=False))
    if distances is not None:
        dd = pd.read_csv(distances)
        dd[dd["origin"].isin(provs) & dd["destination"].isin(provs)].to_csv(o / "distances.csv", index=False)
    comms, Dn = Dnat_last
    pd.DataFrame([(c, all_sectors[k], Dn[k, n]) for n, c in enumerate(comms) for k in np.flatnonzero(Dn[:, n] > 0)],
                 columns=["commodity", "sector", "weight"]).to_csv(o / "commodity_to_sector.csv", index=False)
    (o / "build_report.txt").write_text("\n".join(report) or "no adjustments")
    if verbose:
        print(f"SDM inputs: years {yrs[0]}-{yrs[-1]}, {R} regions, {len(sectors)} sectors -> {o}")
        for line in report[:15]:
            print("  " + line)
        if len(report) > 15:
            print(f"  ... {len(report) - 15} more in build_report.txt")
    return {"years": yrs, "regions": provs, "sectors": list(sectors), "report": report}
