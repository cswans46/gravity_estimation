"""Step 1: panel distance elasticities from the OECD ICIO (Cai 2023, extended).

Outputs (in ./results):
    elasticities_sector.csv   delta_k            by supplying industry (Cai 2023)
    elasticities_pair.csv     delta_{k,m}        by supplying x using industry (+ FD)
    elasticities_pair_filled.csv                 filled / shrunk toward delta_k

Usage:
    python estimation.py --data-dir "D:/Models/OECD Research" --cepii-dir . [--n-jobs 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gravity_irio.cepii import bilateral_arrays, load_dist_cepii
from gravity_irio.elasticities import (estimate_pair_elasticities, estimate_sector_elasticities,
                                       fill_pair_elasticities)
from gravity_irio.icio import icio_to_numpy, load_icio_rdata

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"D:\Models\OECD Research")
    ap.add_argument("--stem", default="ICIO2025econ")
    ap.add_argument("--cepii-dir", default=str(HERE))
    ap.add_argument("--dist-col", default="distwces")
    ap.add_argument("--dep", choices=["F", "z"], default="z",
                    help="z: Cai-normalised F/(Y E) (your original set-up); F: flow levels (standard PPML)")
    ap.add_argument("--controls", default="home,contig,comlang_off,colony")
    ap.add_argument("--no-final-demand", action="store_true",
                    help="sector-level F excludes final demand (the original code's behaviour)")
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--min-positive", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    blocks = icio_to_numpy(load_icio_rdata(a.data_dir, a.stem))
    print(f"ICIO: {len(blocks['years'])} years, {len(blocks['countries'])} regions, "
          f"{len(blocks['industries'])} industries")

    dist_df = load_dist_cepii(Path(a.cepii_dir) / "dist_cepii.xls")
    bil = bilateral_arrays(blocks["countries"], dist_df, dist_col=a.dist_col)
    dropped = [c for c, v in zip(blocks["countries"], bil["valid"]) if not v]
    print(f"regions without CEPII match (dropped): {dropped}; distwces filled from dist: {bil['n_filled']}")

    controls = tuple(a.controls.split(",")) if a.controls else ()
    args = (blocks["Z5"], blocks["FD4"], blocks["X3"], bil, blocks["industries"])
    sec = estimate_sector_elasticities(*args, dep=a.dep, include_final_demand=not a.no_final_demand,
                                       controls=controls)
    sec.to_csv(out / "elasticities_sector.csv", index=False)
    print(sec[["industry", "delta", "se_delta", "home", "converged"]].to_string())

    if not a.skip_pairs:
        pair = estimate_pair_elasticities(*args, dep=a.dep, controls=controls,
                                          min_positive=a.min_positive, n_jobs=a.n_jobs)
        pair.to_csv(out / "elasticities_pair.csv", index=False)
        fill_pair_elasticities(pair, sec, shrink=True).to_csv(out / "elasticities_pair_filled.csv", index=False)


if __name__ == "__main__":
    main()
