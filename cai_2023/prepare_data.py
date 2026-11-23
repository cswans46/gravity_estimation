"""Step 0: Statistics Canada downloads -> the tidy inputs of every other script.

    python prepare_data.py --ipt-zip 12100101-eng.zip --sut-zip 36100478-eng.zip \
        --boundaries lpr_000b21a_e.zip [--gaf 2021_92-151_X.zip] [--year 2021]

Writes to --out (default data/canada):
  from 12-10-0101  ipt_flows.csv, international_{exports,reexports,imports}_panel.csv, products.csv
  from 36-10-0478  sut_describe.txt, sut_use.csv, sut_supply.csv, sut_member_roles.csv (EDIT ME)
  distances        province_distances.csv (GAF population-weighted CES; else capitals + land area)
  year files       interprovincial_flows.csv, international_imports.csv, provincial_use.csv,
                   final_demand_categories.csv, supply_national.csv, supply_domestic.csv
and to --sdm-out (default data/canada_sdm) the inputs of spatial_irio.py.

sut_member_roles.csv is written once with guessed roles (industry, final_household, final_npish,
final_government, final_gfcf, final_inventories, international_exports, ..., ignore, review);
it is never overwritten, so your edits persist. Members left as 'review' are ignored.
Exit code 3 = roles need review (run.bat stops there the first time).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gravity_irio import statcan_tables as st
from gravity_irio.canada_sut import build_sdm_inputs
from gravity_irio.province_distances import (fallback_distances, load_gaf_points, population_distances,
                                             write_distances)

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipt-zip", required=True)
    ap.add_argument("--sut-zip", default=None)
    ap.add_argument("--boundaries", default=None, help="lpr_000b21a_e.zip (fallback distances)")
    ap.add_argument("--gaf", default=None, help="2021 Geographic Attribute File (zip or csv)")
    ap.add_argument("--year", type=int, default=None, help="year of the 4-D split (default: latest common)")
    ap.add_argument("--out", default=str(HERE / "data" / "canada"))
    ap.add_argument("--sdm-out", default=str(HERE / "data" / "canada_sdm"))
    ap.add_argument("--force", action="store_true", help="redo conversions even if outputs exist")
    ap.add_argument("--continue-with-review", action="store_true",
                    help="do not stop when roles still contain 'review'")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 12-10-0101
    if a.force or not (out / "ipt_flows.csv").exists():
        r = st.convert_12100101(a.ipt_zip, out)
        print(f"12-10-0101: years {r['years'][0]}-{r['years'][-1]}, {r['n_flows']:,} flows, "
              f"{r['n_products']} products, {r['n_missing_values_set_to_0']:,} suppressed cells set to 0")
    else:
        print("12-10-0101: already converted (use --force to redo)")

    # ---- distances
    if a.gaf:
        D, provs = population_distances(load_gaf_points(a.gaf))
        how = "GAF population-weighted CES"
    elif a.boundaries:
        D, provs = fallback_distances(a.boundaries)
        how = "FALLBACK capitals + 0.67*sqrt(area/pi); download the GAF for the preferred distances"
    else:
        sys.exit("need --gaf or --boundaries for province distances")
    write_distances(D, provs, out / "province_distances.csv")
    print(f"province_distances.csv: {how}")

    if not a.sut_zip:
        print("no --sut-zip: stopping after the IPT conversion")
        return
    # ---- 36-10-0478
    if a.force or not (out / "sut_use.csv").exists():
        st.describe_table(a.sut_zip, out / "sut_describe.txt")
        print(f"36-10-0478: layout written to {out / 'sut_describe.txt'}")
        try:
            r = st.convert_36100478(a.sut_zip, out)
        except RuntimeError as e:
            sys.exit(f"{e}\nPlease send {out / 'sut_describe.txt'}.")
        lay = {k: v for k, v in r["layout"].items() if k != "members"}
        print(f"36-10-0478: layout {lay}; years {r['years'][0]}-{r['years'][-1]}; "
              f"{r['n_use']:,} use and {r['n_supply']:,} supply rows")
        if r["n_supply"] == 0:
            sys.exit("no supply rows detected; please send sut_describe.txt")
    roles = pd.read_csv(out / "sut_member_roles.csv")
    print("SUT member roles:", roles["role"].value_counts().to_dict())
    rev = roles[roles["role"] == "review"]
    if len(rev) and not a.continue_with_review:
        print(f"\n{len(rev)} members need a role in {out / 'sut_member_roles.csv'}:")
        print(rev[["code", "label"]].to_string(index=False))
        print("\nSet their 'role' column, then run again (or pass --continue-with-review).")
        sys.exit(3)

    # ---- year files and SDM inputs
    sut_years = sorted(pd.read_csv(out / "sut_use.csv", usecols=["year"])["year"].astype(int).unique())
    ipt_years = sorted(pd.read_csv(out / "ipt_flows.csv", usecols=["year"])["year"].astype(int).unique())
    common = sorted(set(sut_years) & set(ipt_years))
    if not common:
        sys.exit(f"no common years: SUT {sut_years}, IPT {ipt_years}")
    year = a.year or common[-1]
    st.ipt_year_files(out, year)
    r = st.sut_year_files(out, year)
    print(f"year files for {year}: {r}")
    build_sdm_inputs(out, a.sdm_out, distances=out / "province_distances.csv", years=common)


if __name__ == "__main__":
    main()
