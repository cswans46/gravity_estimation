"""Interprovincial distance matrices (km), with internal distances on the diagonal.

1. Population-weighted CES distances (preferred; the CEPII ``distwces`` concept the ICIO
   elasticities are estimated with): 2021 Census Geographic Attribute File (GAF, cat. 92-151-X)
   -- dissemination-block population aggregated to dissemination-area representative points
   (DARPLAT/DARPLONG), then 10 km cells, then CES (theta = -1) between and within provinces.
2. Fallback without the GAF: great-circle distances between provincial capitals, and internal
   distance 0.67 * sqrt(land area / pi) from the boundary file's LANDAREA (CEPII's formula for
   countries lacking city data). Crude for large provinces: use only until the GAF is available.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .business_locations import PROVINCES, _PROV_SGC, aggregate_grid, sector_pair_distances
from .distances import haversine


def _find(cols, prefix):
    for c in cols:
        if c.upper().startswith(prefix):
            return c
    raise KeyError(f"no column starting with {prefix} in {list(cols)[:20]}...")


def load_gaf_points(path) -> pd.DataFrame:
    """GAF (zip or csv) -> population points: prov, lat, lon, weight (one per dissemination area).

    Columns are matched by prefix (PRUID, DAUID, DARPLAT, DARPLONG, DBPOP), so the bilingual
    2021 names (PRUID_PRIDU, DARPLAT_ADLAT, DBPOP2021_IDPOP2021, ...) are found."""
    p = Path(path)
    want = ("PRUID", "DAUID", "DARPLAT", "DARPLONG", "DBPOP")

    def _open():
        if p.suffix.lower() == ".zip":
            z = zipfile.ZipFile(p)
            return z.open(next(n for n in z.namelist() if n.lower().endswith(".csv")))
        return open(p, "rb")

    for enc in ("utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(_open(), dtype=str, encoding=enc,
                            usecols=lambda c: c.strip().upper().startswith(want))
            break
        except UnicodeDecodeError:
            continue
    ren = {}
    for c in d.columns:
        for w, k in zip(want, ("pr", "da", "lat", "lon", "pop")):
            if c.strip().upper().startswith(w) and k not in ren.values():
                ren[c] = k
    d = d.rename(columns=ren)
    missing = {"pr", "da", "lat", "lon", "pop"} - set(d.columns)
    if missing:
        raise KeyError(f"GAF columns not found: {missing}; columns are {list(d.columns)}")
    d["pop"] = pd.to_numeric(d["pop"], errors="coerce").fillna(0)
    g = d.groupby("da").agg(pr=("pr", "first"), lat=("lat", "first"), lon=("lon", "first"), weight=("pop", "sum"))
    g["prov"] = g["pr"].str[:2].map(_PROV_SGC)
    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g = g.dropna(subset=["lat", "lon", "prov"])
    return g[g["weight"] > 0].reset_index(drop=True)[["prov", "lat", "lon", "weight"]].assign(sector="POP")


def population_distances(points: pd.DataFrame, cell_km: float = 10.0, provinces=None):
    provinces = provinces or [p for p in PROVINCES if p in set(points["prov"])]
    D4, provs, _ = sector_pair_distances(aggregate_grid(points.assign(sector="POP"), cell_km), provinces, ["POP"])
    return D4[:, 0, :, 0], provs


def fallback_distances(boundary_zip, provinces=PROVINCES):
    from .geography import _parse_dbf_field
    from .sdm import CANADA_CAPITALS
    b = zipfile.ZipFile(boundary_zip)
    dbf = b.read(next(n for n in b.namelist() if n.lower().endswith(".dbf")))
    area = dict(zip([_PROV_SGC[c] for c in _parse_dbf_field(dbf, "PRUID")],
                    map(float, _parse_dbf_field(dbf, "LANDAREA"))))
    R = len(provinces)
    D = np.zeros((R, R))
    for a, pa in enumerate(provinces):
        for c, pc in enumerate(provinces):
            D[a, c] = (0.67 * np.sqrt(area[pa] / np.pi) if a == c
                       else float(haversine(*CANADA_CAPITALS[pa], *CANADA_CAPITALS[pc])))
    return D, list(provinces)


def write_distances(D, provinces, path):
    rows = [(a, b, float(D[i, j])) for i, a in enumerate(provinces) for j, b in enumerate(provinces)]
    pd.DataFrame(rows, columns=["origin", "destination", "km"]).to_csv(path, index=False)
