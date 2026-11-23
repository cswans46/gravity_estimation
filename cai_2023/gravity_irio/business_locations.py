"""Business locations (ODBus / Business Register) -> activity-weighted sector-pair distances.

Why: CEPII's distwces weights city pairs by population. For interprovincial flows of
commodity j bought by user u, the relevant geography is where *j's producers* are in the
origin province and where *u's establishments* are in the destination province. Using
business locations weighted by economic activity gives

    d^{(j,u)}_{ib} = ( sum_{k in i} sum_{l in b} w^j_{ik} w^u_{bl} d_kl^theta )^(1/theta),  theta = -1

(Head & Mayer 2010; the same CES form as distwces). Because this distance depends on the
*user* u, a gravity-RAS seed built from it is no longer rank one within (j, b), so the split
across demanding sectors departs from import proportionality even with supplier-only
elasticities (see docs/METHODS.md, Sect. 4).

Sources and weights
-------------------
* ODBus (Statistics Canada, Open Database of Businesses v1.0, ~450k records; fields include
  NAICS code, number of employees, latitude/longitude, census subdivision, province).
  Coverage comes from municipal/provincial open data and is uneven; employee counts are
  often missing.
* Business Register "Businesses by Census Subdivision" (AAFC/StatCan): counts by CSD x
  NAICS x employment size class, complete coverage. Needs CSD centroids (Census geography).

Pipeline: load -> sector codes -> employment weights (observed, imputed, benchmarked to
province x sector totals) -> aggregate to grid cells (or CSDs) -> pairwise CES distances.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .concordance import NAICS2_COMBOS, load_bridge, naics_digits, naics_level, naics_to_icio
from .distances import haversine

PROVINCES = ["NL", "PE", "NS", "NB", "QC", "ON", "MB", "SK", "AB", "BC", "YT", "NT", "NU"]
_PROV_NAMES = {
    "newfoundland and labrador": "NL", "newfoundland": "NL", "prince edward island": "PE",
    "nova scotia": "NS", "new brunswick": "NB", "quebec": "QC", "québec": "QC", "ontario": "ON",
    "manitoba": "MB", "saskatchewan": "SK", "alberta": "AB", "british columbia": "BC",
    "yukon": "YT", "northwest territories": "NT", "nunavut": "NU",
}
_PROV_SGC = {"10": "NL", "11": "PE", "12": "NS", "13": "NB", "24": "QC", "35": "ON", "46": "MB",
             "47": "SK", "48": "AB", "59": "BC", "60": "YT", "61": "NT", "62": "NU"}

# Business Register employment size classes -> midpoint employees (upper class: assumed mean)
BR_SIZE_MIDPOINTS = {"1-4": 2.5, "5-9": 7, "10-19": 14.5, "20-49": 34.5, "50-99": 74.5,
                     "100-199": 149.5, "200-499": 349.5, "500+": 800.0}


def normalize_province(x) -> str | None:
    s = str(x).strip()
    if s.upper() in PROVINCES:
        return s.upper()
    if s[:2] in _PROV_SGC:
        return _PROV_SGC[s[:2]]
    return _PROV_NAMES.get(s.lower())


def _find(cols, *keys):
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keys):
            return c
    return None


# ODBus v1.0 column names (as distributed). Missing values are coded "..".
ODBUS_COLUMNS = {
    "naics": "derived_NAICS", "naics_src1": "source_NAICS_primary", "naics_src2": "source_NAICS_secondary",
    "sector_text": "business_sector", "subsector_text": "business_subsector", "naics_descr": "NAICS_descr",
    "employees": "total_no_employees", "lat": "latitude", "lon": "longitude", "prov": "prov_terr",
    "csduid": "CSDUID", "pruid": "PRUID", "provider": "provider", "status": "status",
}
def _norm_text(x) -> str:
    s = "" if x is None else str(x).strip().lower()
    return "" if s in ("..", "nan", "none") else re.sub(r"\s+", " ", s)


def load_odbus(path, sector_level: str = "naics2", min_n: int = 5, min_purity: float = 0.6,
               keep_closed: bool = False) -> pd.DataFrame:
    """Read the ODBus v1.0 CSV (exact column names, '..' = missing).

    NAICS assignment, in order (column 'naics_source' records which step applied):
      1. 'source'  : source_NAICS_primary (then _secondary) when it has >= 3 digits and agrees
                     with derived_NAICS at the 2-digit sector level;
      2. 'derived' : derived_NAICS (the harmonised code; 2-digit in most records);
      3. 'text'    : records whose code is missing or coarser than needed get the modal code of
                     records with the same business_sector (+ subsector) text, learned from the
                     records that have a code at that level -- only if the text group has
                     >= min_n coded records and the modal code's share is >= min_purity, and
                     the learned code agrees with any 2-digit code the record already has.
    sector_level : 'naics2' (ODBus derived sectors), 'naics3', 'naics4', ... or 'icio'.
    Records at a coarser level than requested are kept with a coarser code; use
    ``expand_coarse`` to distribute their weight over the finer sectors.
    """
    df = pd.read_csv(path, dtype=str, na_values=[".."], keep_default_na=True, encoding_errors="replace",
                     low_memory=False)
    C = ODBUS_COLUMNS
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[C["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[C["lon"]], errors="coerce"),
        "prov": df[C["prov"]].map(normalize_province),
        "csduid": df.get(C["csduid"]),
        "employees": pd.to_numeric(df.get(C["employees"]), errors="coerce"),
        "text": (df.get(C["sector_text"], pd.Series("", index=df.index)).map(_norm_text) + " | "
                 + df.get(C["subsector_text"], pd.Series("", index=df.index)).map(_norm_text)),
        "provider": df.get(C["provider"]),
    })
    if not keep_closed and C["status"] in df:
        st = df[C["status"]].fillna("").str.lower()
        out = out[~st.str.contains("closed|inactive|not active|cancel", regex=True)]
        df = df.loc[out.index]
    # derived_NAICS uses the US wholesale code 42 for a few records (Canada: 41); 99 = unclassified
    der = df[C["naics"]].map(naics_digits).replace({"42": "41", "99": ""})
    src = df.get(C["naics_src1"], pd.Series("", index=df.index)).map(naics_digits)
    src2 = df.get(C["naics_src2"], pd.Series("", index=df.index)).map(naics_digits)
    src = src.where(src.str.len() >= 3, src2)
    agree = (der == "") | (src.map(lambda c: naics_level(c, 2)) == der.map(lambda c: naics_level(c, 2)))
    use_src = (src.str.len() >= 3) & agree
    out["naics"] = np.where(use_src, src, der)
    out["naics_source"] = np.where(use_src, "source", np.where(der != "", "derived", "none"))

    want = int(re.match(r"naics(\d)", sector_level).group(1)) if sector_level.startswith("naics") else 6
    for lev in range(2, want + 1):
        need = out["naics"].str.len() < lev
        coded = out[~need & (out["text"] != " | ")]
        if coded.empty:
            continue
        lab = coded["naics"].map(lambda c: naics_level(c, lev))
        g = pd.DataFrame({"text": coded["text"], "code": lab}).groupby(["text", "code"]).size()
        tot = g.groupby(level=0).sum()
        best = g.groupby(level=0).idxmax().map(lambda t: t[1])
        purity = g.groupby(level=0).max() / tot
        ok = (tot >= min_n) & (purity >= min_purity)
        table = best[ok]
        cand = out.loc[need, "text"].map(table)
        cur2 = out.loc[need, "naics"].map(lambda c: naics_level(c, 2))
        cand2 = cand.map(lambda c: naics_level(naics_digits(c), 2) if isinstance(c, str) else None)
        fill = cand.notna() & (cur2.isna() | (cur2 == cand2))
        idx = cand[fill].index
        out.loc[idx, "naics"] = cand[fill].map(naics_digits)
        out.loc[idx, "naics_source"] = "text"
    if sector_level == "icio":
        out["sector"] = assign_sector(out["naics"], "icio")
    else:
        out["sector"] = out["naics"].map(lambda c: naics_level(c, want))
    out["sector2"] = out["naics"].map(lambda c: naics_level(c, 2))
    n0 = len(df)
    out = out.dropna(subset=["lat", "lon", "prov"])
    out = out[(out["lat"].between(41, 84)) & (out["lon"].between(-142, -52))]
    out.attrs.update(n_raw=n0, n_kept=len(out), n_no_sector=int(out["sector"].isna().sum()),
                     naics_source=out["naics_source"].value_counts().to_dict())
    return out


def expand_coarse(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """Distribute records coded only at the 2-digit sector over the finer ``level``-digit sectors
    of the same parent, in proportion to the province's weight shares of those finer sectors
    (same location, fractional weight). Keeps totals by province x 2-digit sector."""
    d = df.copy()
    if "weight" not in d:
        raise ValueError("compute weights first (impute_employees)")
    fine = d[d["sector"].notna()]
    coarse = d[d["sector"].isna() & d["sector2"].notna()]
    sh = fine.groupby(["prov", "sector2", "sector"])["weight"].sum()
    sh = sh / sh.groupby(level=[0, 1]).transform("sum")
    sh = sh.rename("share").reset_index()
    ex = coarse.drop(columns="sector").merge(sh, on=["prov", "sector2"], how="inner")
    ex["weight"] = ex["weight"] * ex["share"]
    return pd.concat([fine, ex.drop(columns="share")], ignore_index=True)


def assign_sector(naics: pd.Series, level: str = "icio") -> pd.Series:
    if level == "icio":
        b = load_bridge()
        cache = {}
        def f(code):
            if not code:
                return None
            if code not in cache:
                cache[code] = naics_to_icio(code, b)
            return cache[code]
        return naics.map(f)
    m = re.match(r"naics(\d)", level)
    if not m:
        raise ValueError(level)
    k = int(m.group(1))
    return naics.map(lambda s: naics_level(s, k) if isinstance(s, str) else None)


def load_br_csd(path, csd_centroids: pd.DataFrame, sector_level: str = "icio",
                size_midpoints=BR_SIZE_MIDPOINTS) -> pd.DataFrame:
    """Business Register counts by CSD x NAICS x size class -> weighted points at CSD centroids.

    csd_centroids : DataFrame with columns csduid, lat, lon (e.g. population-weighted
                    centroids from the Census geography files).
    Weight = count x size-class midpoint employees. Column names are detected loosely.
    """
    df = pd.read_csv(path, dtype=str)
    cols = list(df.columns)
    c_csd = _find(cols, "csd") or _find(cols, "sdr")
    c_naics = _find(cols, "naics") or _find(cols, "scian")
    c_size = _find(cols, "size") or _find(cols, "employ") or _find(cols, "taille")
    c_cnt = _find(cols, "count") or _find(cols, "number") or _find(cols, "nombre")
    d = pd.DataFrame({"csduid": df[c_csd].str.strip(),
                      "naics": df[c_naics].str.replace(r"[^0-9]", "", regex=True),
                      "size": df[c_size].str.replace(" ", ""),
                      "count": pd.to_numeric(df[c_cnt], errors="coerce").fillna(0)})
    mids = {k.replace(" ", ""): v for k, v in size_midpoints.items()}
    d["employees"] = d["count"] * d["size"].map(mids).fillna(1.0)
    d = d.merge(csd_centroids.assign(csduid=csd_centroids["csduid"].astype(str)), on="csduid", how="inner")
    d["prov"] = d["csduid"].str[:2].map(_PROV_SGC)
    d["sector"] = assign_sector(d["naics"], sector_level)
    return d[["prov", "sector", "naics", "lat", "lon", "employees"]]


def impute_employees(df: pd.DataFrame, method: str = "median", by=("prov", "sector")) -> pd.DataFrame:
    """Fill missing/zero employee counts.

    'median' : median observed employees in the same province x sector (then sector, then 1)
    'count'  : every establishment counts 1 (pure location counts)
    The column 'employees_imputed' flags imputed rows.
    """
    d = df.copy()
    if "sector2" in d:  # records coded only at the 2-digit level impute within that sector
        d["_sector_orig"] = d["sector"]
        d["sector"] = d["sector"].fillna(d["sector2"])
    miss = ~(d["employees"] > 0)
    d["employees_imputed"] = miss
    if method == "count":
        d["weight"] = 1.0
        return d
    obs = d[~miss]
    med = obs.groupby(list(by))["employees"].median()
    med_s = obs.groupby("sector")["employees"].median()
    fill = d.loc[miss, list(by)].apply(lambda r: med.get(tuple(r), np.nan), axis=1)
    fill = fill.fillna(d.loc[miss, "sector"].map(med_s)).fillna(1.0)
    d["weight"] = d["employees"].where(~miss, fill).astype(float)
    if "_sector_orig" in d:
        d["sector"] = d.pop("_sector_orig")
    return d


def benchmark_weights(df: pd.DataFrame, totals: pd.DataFrame, value: str = "value") -> pd.DataFrame:
    """Rescale weights so province x sector sums equal external totals (e.g. LFS/SEPH
    employment or provincial value added by industry). Cells absent from ``totals`` keep
    their weights. This corrects ODBus's uneven coverage *between* provinces/sectors; it
    cannot fix uneven coverage *within* a province (use BR CSD counts for that)."""
    d = df.copy()
    cur = d.groupby(["prov", "sector"])["weight"].sum().rename("cur")
    t = totals.set_index(["prov", "sector"])[value].rename("target")
    f = (t / cur).replace([np.inf, -np.inf], np.nan)
    d = d.join(f.rename("factor"), on=["prov", "sector"])
    d["weight"] = d["weight"] * d["factor"].fillna(1.0)
    return d.drop(columns="factor")


def aggregate_grid(df: pd.DataFrame, cell_km: float = 25.0) -> pd.DataFrame:
    """Aggregate weighted points to ~cell_km x cell_km cells within each province.

    Returns one row per (prov, cell, sector) with the activity-weighted centroid and the
    cell internal distance 0.67 sqrt(area / pi) (CEPII convention).
    """
    d = df.dropna(subset=["sector"]).copy()
    dlat = cell_km / 111.2
    d["iy"] = np.floor(d["lat"] / dlat).astype(int)
    latc = (d["iy"] + 0.5) * dlat
    dlon = cell_km / (111.2 * np.cos(np.radians(latc)))
    d["ix"] = np.floor(d["lon"] / dlon).astype(int)
    d["cell"] = d["prov"] + ":" + d["iy"].astype(str) + ":" + d["ix"].astype(str)
    d["wlat"] = d["lat"] * d["weight"]
    d["wlon"] = d["lon"] * d["weight"]
    g = d.groupby(["prov", "cell", "sector"], as_index=False)[["weight", "wlat", "wlon"]].sum()
    # one centroid per cell (all sectors), so that sectors share cell locations
    c = d.groupby("cell")[["weight", "wlat", "wlon"]].sum()
    cent = pd.DataFrame({"lat": c["wlat"] / c["weight"], "lon": c["wlon"] / c["weight"]})
    g = g.drop(columns=["wlat", "wlon"]).join(cent, on="cell")
    g["self_km"] = 0.67 * np.sqrt(cell_km ** 2 / np.pi)
    return g


def sector_pair_distances(cells: pd.DataFrame, provinces=None, sectors=None, theta: float = -1.0,
                          chunk: int = 2000):
    """CES distances d[i, j, b, u] between (origin province i, sector j) and (destination b, sector u).

    cells : output of ``aggregate_grid`` (columns prov, cell, sector, weight, lat, lon, self_km).
    Returns (D4, provinces, sectors) with D4 shape (R, K, R, K); NaN where a province has no
    activity in a sector.
    """
    provinces = list(provinces or [p for p in PROVINCES if p in set(cells["prov"])])
    sectors = list(sectors or sorted(cells["sector"].dropna().unique()))
    loc = cells.drop_duplicates("cell")[["cell", "lat", "lon", "self_km"]].reset_index(drop=True)
    cidx = {c: n for n, c in enumerate(loc["cell"])}
    R, K, n = len(provinces), len(sectors), len(loc)
    W = np.zeros((n, R * K))
    for row in cells.itertuples(index=False):
        if row.prov in provinces and row.sector in sectors:
            W[cidx[row.cell], provinces.index(row.prov) * K + sectors.index(row.sector)] += row.weight
    tot = W.sum(axis=0)
    Wn = np.divide(W, tot, out=np.zeros_like(W), where=tot > 0)
    lat, lon, selfd = loc["lat"].to_numpy(), loc["lon"].to_numpy(), loc["self_km"].to_numpy()
    P = np.zeros((R * K, R * K))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Dc = haversine(lat[s:e, None], lon[s:e, None], lat[None, :], lon[None, :])
        ii = np.arange(s, e)
        Dc[ii - s, ii] = selfd[s:e]
        Dc = np.maximum(Dc, 1e-3)
        P += Wn[s:e].T @ (Dc ** theta @ Wn)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(P > 0, P ** (1.0 / theta), np.nan)
    D4 = D.reshape(R, K, R, K)
    missing = tot.reshape(R, K) == 0
    D4[missing, :, :] = np.nan
    D4[:, :, missing] = np.nan
    return D4, provinces, sectors


def aggregate_user_distances(D4, user_shares, theta: float = -1.0):
    """Collapse user-specific distances to one distance per (i, j, b) for estimating delta_j
    from flows summed over users:  d_ijb = ( sum_u s[j,b,u] d[i,j,b,u]^theta )^(1/theta),
    s[j,b,u] = user u's share of province b's use of j (rows sum to 1). NaN distances are
    dropped and the shares renormalised."""
    D = np.asarray(D4, float)
    s = np.asarray(user_shares, float)
    ok = np.isfinite(D)
    Dt = np.where(ok, D, 1.0) ** theta
    w = s[None, :, :, :] * ok
    wsum = w.sum(axis=3)
    num = (w * Dt).sum(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(wsum > 0, (num / wsum) ** (1.0 / theta), np.nan)
