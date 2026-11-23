"""Converters from Statistics Canada table downloads (the "-eng.zip" full-table CSVs) to the
tidy CSV inputs of this package.

Table 12-10-0101-01  Interprovincial and international trade flows, basic prices, detail level
  columns: REF_DATE, GEO (= origin province), Trade flow detail, Product ("Label [CODE]"), VALUE
  'To <province>' rows are shipments from GEO to that province, *including* GEO itself
  (own-province supply); the other 'To' rows sum exactly to 'Interprovincial exports'.
  -> ipt_flows.csv (year, origin, destination, commodity, value)        [$ thousands]
     international_exports.csv / international_imports.csv (year, province, commodity, value)
     products.csv (commodity, label)

Table 36-10-0478-01  Supply and use tables, detail level, provincial and territorial
  The layout is detected (``describe_table`` writes every dimension and its members), and the
  roles of the supply/use columns (industry, final demand category, imports, exports, ignore)
  are guessed into an editable CSV (sut_member_roles.csv) that the conversion then uses.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .business_locations import normalize_province

STD_COLS = {"REF_DATE", "GEO", "DGUID", "UOM", "UOM_ID", "SCALAR_FACTOR", "SCALAR_ID", "VECTOR",
            "COORDINATE", "VALUE", "STATUS", "SYMBOL", "TERMINATED", "DECIMALS"}
CODE_RE = re.compile(r"\[([A-Za-z0-9_.\-]+)\]\s*$")


def _csv_in(path):
    """Open the data CSV of a StatCan table (zip or extracted csv). Returns a binary handle."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        z = zipfile.ZipFile(p)
        name = next(n for n in z.namelist() if n.lower().endswith(".csv") and "metadata" not in n.lower())
        return z.open(name)
    return open(p, "rb")


def read_table(path, usecols=None, chunksize=None):
    return pd.read_csv(_csv_in(path), dtype=str, usecols=usecols, chunksize=chunksize, encoding="utf-8-sig")


def code_of(member: str):
    m = CODE_RE.search(str(member))
    return m.group(1) if m else None


def label_of(member: str):
    return CODE_RE.sub("", str(member)).strip()


def describe_table(path, out_txt, max_members: int = 2000, chunksize: int = 500_000):
    """Write every non-standard column of a StatCan table with its members (and years/GEO)."""
    members = {}
    n = 0
    for ch in read_table(path, chunksize=chunksize):
        n += len(ch)
        for c in ch.columns:
            if c in STD_COLS - {"REF_DATE", "GEO", "SCALAR_FACTOR", "UOM"}:
                continue
            s = members.setdefault(c, {})
            for v in ch[c].dropna().unique():
                if len(s) < max_members:
                    s[v] = s.get(v, 0) + 1
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"rows: {n}\ncolumns: {list(members)}\n\n")
        for c, s in members.items():
            f.write(f"== {c} ({len(s)} members{'+' if len(s) >= max_members else ''})\n")
            for v in s:
                f.write(f"   {v}\n")
    return members


# ----------------------------------------------------------------------------- 12-10-0101
def convert_12100101(path, out_dir, chunksize: int = 500_000):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flows, intl_x, intl_rx, intl_m, prods = [], [], [], [], {}
    n_missing = 0
    cols = ["REF_DATE", "GEO", "Trade flow detail", "Product", "VALUE", "SCALAR_FACTOR"]
    for ch in read_table(path, usecols=cols, chunksize=chunksize):
        sf = set(ch["SCALAR_FACTOR"].dropna())
        if sf - {"thousands"}:
            raise ValueError(f"unexpected scalar factor {sf}")
        ch = ch.assign(commodity=ch["Product"].map(code_of))
        ch = ch[ch["commodity"].notna()]                      # drop Total products/goods/services
        for p in ch.drop_duplicates("commodity").itertuples():
            prods[p.commodity] = label_of(p.Product)
        ch = ch.assign(origin=ch["GEO"].map(normalize_province),
                       value=pd.to_numeric(ch["VALUE"], errors="coerce"))
        ch = ch[ch["origin"].notna()]                         # drop Canada, territorial enclaves abroad
        n_missing += int(ch["value"].isna().sum())
        ch["value"] = ch["value"].fillna(0.0)
        tf = ch["Trade flow detail"]
        to = ch[tf.str.startswith("To ")].copy()
        to["destination"] = to["Trade flow detail"].str[3:].map(normalize_province)
        to = to[to["destination"].notna()]
        flows.append(to[["REF_DATE", "origin", "destination", "commodity", "value"]])
        intl_x.append(ch[tf == "International exports"][["REF_DATE", "origin", "commodity", "value"]])
        intl_rx.append(ch[tf == "International re-exports"][["REF_DATE", "origin", "commodity", "value"]])
        intl_m.append(ch[tf == "International imports"][["REF_DATE", "origin", "commodity", "value"]])
    ren = {"REF_DATE": "year", "origin": "province"}
    F = pd.concat(flows).rename(columns={"REF_DATE": "year"})
    F.to_csv(out / "ipt_flows.csv", index=False)
    X = pd.concat(intl_x).rename(columns=ren).groupby(["year", "province", "commodity"], as_index=False)["value"].sum()
    X.to_csv(out / "international_exports_panel.csv", index=False)
    if intl_rx:
        pd.concat(intl_rx).rename(columns=ren).to_csv(out / "international_reexports_panel.csv", index=False)
    M = pd.concat(intl_m).rename(columns=ren)
    M.to_csv(out / "international_imports_panel.csv", index=False)
    pd.Series(prods, name="label").rename_axis("commodity").reset_index().to_csv(out / "products.csv", index=False)
    return {"years": sorted(F["year"].unique()), "n_flows": len(F), "n_products": len(prods),
            "n_missing_values_set_to_0": n_missing}


def ipt_year_files(out_dir, year):
    """Single-year files for the 4-D split: interprovincial_flows.csv, international_imports.csv."""
    out = Path(out_dir)
    F = pd.read_csv(out / "ipt_flows.csv", dtype={"year": str})
    F[F["year"] == str(year)].drop(columns="year").to_csv(out / "interprovincial_flows.csv", index=False)
    M = pd.read_csv(out / "international_imports_panel.csv", dtype={"year": str})
    M[M["year"] == str(year)].drop(columns="year").to_csv(out / "international_imports.csv", index=False)


# ----------------------------------------------------------------------------- 36-10-0478
ROLE_RULES = [  # (role, regex on the member label, applied in order)
    ("ignore", r"^total|total (supply|use|demand|intermediate|final)|^sum"),
    ("international_imports", r"international imports|imports from abroad|^imports$"),
    ("interprovincial_imports", r"interprovincial imports|imports from other provinces"),
    ("international_exports", r"international exports|re-exports|exports to abroad|^exports$"),
    ("interprovincial_exports", r"interprovincial exports|exports to other provinces"),
    ("ignore", r"tax|subsid|margin|commission|wholesale|retail|transportation margins|duties"),
    ("final_inventories", r"inventor"),
    ("final_household", r"household|personal expenditure|consumption expenditure of households"),
    ("final_npish", r"non-profit institutions serving|npish"),
    ("final_government", r"government.*(consumption|expenditure)|general government"),
    ("final_gfcf", r"capital formation|investment|machinery and equipment|construction (?!.*industr)|intellectual property"),
]


SUPPLY_LABEL = re.compile(r"^supply\b|\bsupply\b.*industr|^output\b|^total supply|^production by", re.I)


def guess_role(member: str) -> str:
    code = code_of(member) or ""
    lab = label_of(member).lower()
    if re.search(ROLE_RULES[0][1], lab):
        return "ignore"                                  # totals
    if re.match(r"^(BS|NP|GS)\d", code):
        return "industry"                                # IOIC industry codes before label rules
    for role, rx in ROLE_RULES[1:]:
        if re.search(rx, lab):
            return role
    return "review"


def detect_sut_layout(path, nrows: int = 300_000):
    """Find the product column, the supplier/user column, the supply-vs-use column and the
    valuation column of table 36-10-0478 by their members."""
    d = next(iter(read_table(path, chunksize=nrows)))
    dims = [c for c in d.columns if c not in STD_COLS]
    info = {"dims": dims, "members": {}}
    for c in dims:
        vals = pd.Series(d[c].dropna().astype(str).unique())
        info["members"][c] = len(vals)
        codes = vals.map(code_of).dropna()
        if len(codes) and codes.str.match(r"^M[A-Z]{1,3}\d").mean() > 0.5:
            info.setdefault("product", c)
        elif len(codes) and codes.str.match(r"^(BS|NP|GS)").any():
            info.setdefault("member", c)
        elif vals.str.contains("price", case=False).any():
            info.setdefault("valuation", c)
        elif vals.str.contains(r"\bsupply\b|\buse\b", case=False, regex=True).any():
            info.setdefault("table", c)
    if "member" not in info:   # fallback: the non-product dimension with the most members
        rest = [c for c in dims if c not in (info.get("product"), info.get("valuation"), info.get("table"))]
        if rest:
            info["member"] = max(rest, key=lambda c: info["members"][c])
    return info


def convert_36100478(path, out_dir, roles_csv=None, valuation_pattern="basic", years=None,
                     chunksize: int = 1_000_000):
    """Extract use and supply blocks at basic prices into tidy files.

    Writes sut_use.csv / sut_supply.csv (year, province, commodity, member, value) and, if
    ``roles_csv`` does not exist, sut_member_roles.csv (member, code, label, table, role) with
    guessed roles for review. Returns a summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lay = detect_sut_layout(path)
    need = {"product", "member"}
    if not need <= set(lay):
        raise RuntimeError(f"could not detect SUT layout {lay}; run describe and send sut_describe.txt")
    pc, mc, tc, vc = lay["product"], lay["member"], lay.get("table"), lay.get("valuation")
    cols = ["REF_DATE", "GEO", "VALUE", pc, mc] + [c for c in (tc, vc) if c]
    use, sup, roles = [], [], {}
    for ch in read_table(path, usecols=cols, chunksize=chunksize):
        if years:
            ch = ch[ch["REF_DATE"].isin([str(y) for y in years])]
        if vc:
            ch = ch[ch[vc].str.contains(valuation_pattern, case=False, na=False)]
        ch = ch.assign(province=ch["GEO"].map(normalize_province),
                       commodity=ch[pc].map(code_of), value=pd.to_numeric(ch["VALUE"], errors="coerce").fillna(0.0))
        ch = ch[ch["commodity"].notna() & (ch["value"] != 0)]          # zeros dropped (file size)
        geo_ca = ch["GEO"].str.strip().eq("Canada")
        ch.loc[geo_ca, "province"] = "CA"
        ch = ch[ch["province"].notna()]
        if tc:
            kind = ch[tc].str.lower()
        else:  # no separate supply/use dimension: infer from the member label
            kind = ch[mc].map(lambda m: "supply" if SUPPLY_LABEL.search(label_of(m)) else "use")
        for mbr, k in ch[[mc]].assign(k=kind).drop_duplicates().itertuples(index=False):
            roles.setdefault(mbr, ("supply" if "supply" in str(k) else "use"))
        base = ch.assign(member=ch[mc], year=ch["REF_DATE"])[["year", "province", "commodity", "member", "value"]]
        is_sup = kind.str.contains("supply", na=False)
        sup.append(base[is_sup.values])
        use.append(base[~is_sup.values])
    U, S = pd.concat(use), pd.concat(sup)
    U.to_csv(out / "sut_use.csv", index=False)
    S.to_csv(out / "sut_supply.csv", index=False)
    rfile = Path(roles_csv) if roles_csv else out / "sut_member_roles.csv"
    if not rfile.exists():
        pd.DataFrame([(m, code_of(m), label_of(m), t, guess_role(m)) for m, t in roles.items()],
                     columns=["member", "code", "label", "table", "role"]).to_csv(rfile, index=False)
    return {"layout": lay, "years": sorted(U["year"].unique()), "n_use": len(U), "n_supply": len(S),
            "roles_file": str(rfile)}


def sut_year_files(out_dir, year, roles_csv=None):
    """From sut_use/sut_supply + roles: the single-year inputs of the 4-D split and concordances.

    provincial_use.csv     province, commodity, user, value   (industries + final-demand categories)
    supply_national.csv    commodity, industry, value          (Canada supply table, industries)
    supply_domestic.csv    province, commodity, value          (output - intl exports, for the
                                                               provincial-tables-only route)
    Final-demand members are kept individually (user = member code) so that the user concordance
    maps them to 'FD'; their role is recorded in final_demand_categories.csv.
    """
    out = Path(out_dir)
    R = pd.read_csv(roles_csv or out / "sut_member_roles.csv")
    if (R["role"] == "review").any():
        print(f"WARNING: {int((R['role'] == 'review').sum())} SUT members still have role 'review' in "
              f"{roles_csv or out / 'sut_member_roles.csv'} -- they are ignored until you set a role.")
    role = dict(zip(R["member"], R["role"]))
    code = dict(zip(R["member"], R["code"].fillna(R["member"])))
    U = pd.read_csv(out / "sut_use.csv", dtype={"year": str})
    U = U[U["year"] == str(year)]
    U["role"] = U["member"].map(role)
    keep = U["role"].eq("industry") | U["role"].str.startswith("final_", na=False)
    pu = U[keep & (U["province"] != "CA")].assign(user=lambda d: d["member"].map(code))
    pu[["province", "commodity", "user", "value"]].to_csv(out / "provincial_use.csv", index=False)
    fd = U[U["role"].str.startswith("final_", na=False)].drop_duplicates("member")
    pd.DataFrame({"user": fd["member"].map(code), "category": fd["role"].str[6:]}).to_csv(
        out / "final_demand_categories.csv", index=False)
    S = pd.read_csv(out / "sut_supply.csv", dtype={"year": str})
    S = S[S["year"] == str(year)]
    S["role"] = S["member"].map(role)
    ind = S[S["role"].eq("industry")]
    nat = ind[ind["province"] == "CA"]
    if nat.empty:  # no Canada rows: sum provinces
        nat = ind.groupby(["commodity", "member"], as_index=False)["value"].sum()
    nat.assign(industry=nat["member"].map(code))[["commodity", "industry", "value"]].to_csv(
        out / "supply_national.csv", index=False)
    prov_out = ind[ind["province"] != "CA"].groupby(["province", "commodity"])["value"].sum()
    X = pd.read_csv(out / "international_exports_panel.csv", dtype={"year": str})  # excl. re-exports
    X = X[X["year"] == str(year)].set_index(["province", "commodity"])["value"]
    dom = (prov_out - X.reindex(prov_out.index).fillna(0)).clip(lower=0).rename("value").reset_index()
    dom.to_csv(out / "supply_domestic.csv", index=False)
    return {"users": pu["user"].nunique(), "commodities": pu["commodity"].nunique(),
            "final_demand_members": len(fd)}
