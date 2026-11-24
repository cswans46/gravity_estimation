"""Checks on the converted 36-10-0478 files. Run:  python diagnose_sut.py > sut_diagnosis.txt"""
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data" / "canada")
txt = (d / "sut_describe.txt").read_text(encoding="utf-8").split("\n== ")
for block in txt:
    head = block.splitlines()[0] if block else ""
    if not head.startswith(("Industry", "Product")):          # print the short dimensions in full
        print("== " + block[:3000])
roles = pd.read_csv(d / "sut_member_roles.csv", dtype=str)
role = dict(zip(roles["member"], roles["role"]))
U = pd.read_csv(d / "sut_use.csv", dtype={"year": str})
S = pd.read_csv(d / "sut_supply.csv", dtype={"year": str})
y = sorted(U["year"].unique())[-3]
for geo in ("CA", "ON"):
    u = U[(U["year"] == y) & (U["province"] == geo)].assign(role=lambda x: x["member"].map(role))
    s = S[(S["year"] == y) & (S["province"] == geo)].assign(role=lambda x: x["member"].map(role))
    print(f"\n### {geo} {y}: totals by role (thousands $)")
    print("use:   ", u.groupby("role")["value"].sum().round(0).to_dict())
    print("supply:", s.groupby("role")["value"].sum().round(0).to_dict())
    out = s[s["role"] == "industry"].groupby("member")["value"].sum()
    inp = u[u["role"] == "industry"].groupby("member")["value"].sum()
    va = (out - inp.reindex(out.index).fillna(0)).sort_values()
    print(f"industries: {len(out)}, output {out.sum():,.0f}, intermediate input {inp.sum():,.0f}, "
          f"negative VA in {int((va < 0).sum())}")
    print(va.head(8).round(0).to_string())
    cs = s.groupby("commodity")["value"].sum()
    cu = u[~u["role"].isin(["ignore"])].groupby("commodity")["value"].sum()
    r = (cu / cs).replace([float("inf")], float("nan")).dropna()
    print(f"use/supply by commodity: median {r.median():.3f}, p10 {r.quantile(.1):.3f}, p90 {r.quantile(.9):.3f}")
