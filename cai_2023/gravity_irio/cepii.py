"""CEPII GeoDist distances and bilateral controls, aligned to ICIO region codes.

``dist_cepii.xls`` (Mayer & Zignago 2011) has one row per (iso_o, iso_d), *including*
iso_o == iso_d rows that carry internal distances, and columns
    contig, comlang_off, comlang_ethno, colony, comcol, curcol, col45, smctry,
    dist, distcap, distw, distwces
``distwces`` is the population-weighted distance with CES aggregation, theta = -1.

ICIO regions that are not CEPII countries:
    * ROW (rest of world)            -> dropped (no meaningful distance)
    * CN1/CN2, MX1/MX2 (extended ICIO split of China/Mexico by processing-trade status)
      -> mapped to the parent country (CHN/MEX). Pairs between the two halves get
         the parent's *internal* distance and home = 1. See docs/METHODS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# CEPII uses some legacy ISO3 codes
CEPII_TO_ISO = {"ROM": "ROU", "ZAR": "COD", "TMP": "TLS", "YUG": "SRB", "PAL": "PSE"}

DEFAULT_PARENT = {"CN1": "CHN", "CN2": "CHN", "MX1": "MEX", "MX2": "MEX"}

BILATERAL_DUMMIES = ["contig", "comlang_off", "comlang_ethno", "colony", "comcol", "smctry"]


def parent_code(region: str, parent_map: dict | None = None) -> str:
    pm = DEFAULT_PARENT if parent_map is None else parent_map
    return pm.get(region, region)


def load_dist_cepii(path) -> pd.DataFrame:
    """Read dist_cepii (.xls needs xlrd; .csv/.dta also accepted)."""
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".dta"):
        df = pd.read_stata(path)
    else:
        df = pd.read_excel(path)
    df["iso_o"] = df["iso_o"].replace(CEPII_TO_ISO)
    df["iso_d"] = df["iso_d"].replace(CEPII_TO_ISO)
    for c in ("dist", "distcap", "distw", "distwces"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # CEPII uses '.' for missing
    return df


def bilateral_arrays(regions, dist_df: pd.DataFrame, dist_col: str = "distwces",
                     controls=BILATERAL_DUMMIES, parent_map: dict | None = None,
                     fallback_col: str = "dist"):
    """Build (C, C) distance and control matrices for a list of ICIO regions.

    Returns dict with
        'dist'      (C,C) float, NaN where unavailable
        'home'      (C,C) 1 if same parent country (incl. CN1->CN2), else 0
        <control>   (C,C) for each available bilateral dummy (0 on the diagonal blocks)
        'valid'     (C,) bool: region has a CEPII match
    Missing ``dist_col`` values are filled from ``fallback_col`` (distwces is missing for
    a few small countries in GeoDist); the number filled is reported in 'n_filled'.
    """
    regions = list(regions)
    parents = [parent_code(r, parent_map) for r in regions]
    idx = dist_df.set_index(["iso_o", "iso_d"])
    C = len(regions)
    out = {"dist": np.full((C, C), np.nan), "home": np.zeros((C, C))}
    ctrl = [c for c in controls if c in dist_df.columns]
    for c in ctrl:
        out[c] = np.zeros((C, C))
    n_filled = 0
    have = set(dist_df["iso_o"])
    for a, pa in enumerate(parents):
        for b, pb in enumerate(parents):
            out["home"][a, b] = float(pa == pb)
            if pa not in have or pb not in have or (pa, pb) not in idx.index:
                continue
            row = idx.loc[(pa, pb)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            d = row.get(dist_col, np.nan)
            if not np.isfinite(d) and fallback_col in row:
                d = row[fallback_col]
                n_filled += 1
            out["dist"][a, b] = d
            if pa != pb:
                for c in ctrl:
                    out[c][a, b] = float(row[c]) if np.isfinite(row[c]) else 0.0
    out["valid"] = np.array([p in have for p in parents])
    out["controls"] = ctrl
    out["n_filled"] = n_filled
    return out
