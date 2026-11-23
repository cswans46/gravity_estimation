"""Concordances between Statistics Canada classifications and the OECD ICIO 2025 industries.

Chain:  SUPC commodity --(national supply table)--> IOIC industry --(code structure / official
        IOIC-NAICS table)--> NAICS 2017 --(bridge)--> ISIC Rev.4 --> ICIO 2025 (50 industries)

* ``ICIO2025`` lists the 50 ICIO 2025 industries (OECD 2026 ICIO documentation annex) with the
  ISIC Rev.4 divisions/groups they cover. ``canonical_icio`` accepts both the letter style
  (``A01``, ``C10T12``, ``C24A``, ``D``) and the numeric style of the R objects (``01``,
  ``10T12``, ``24A``, ``35``, ``97T98``).
* ``data/naics2022_to_icio2025_official.csv``: derived from Statistics Canada's official
  NAICS Canada 2022 V1.0 -> ISIC Rev.4 concordance (921 classes, 1,750 links); used first for
  6-digit codes and, averaged, for shorter codes.
* ``data/naics_to_icio2025_bridge.csv``: hand-built NAICS -> ICIO bridge (2- to 6-digit
  prefixes, NAICS 2017 and 2022 codes, longest-prefix match), fixed against the official
  concordance (``validate_bridge``: 0 disagreements); used for codes absent from NAICS 2022.
* IOIC codes (``BS111A00``, ``NP813000``, ``GS611100`` ...) embed a NAICS prefix up to the
  first letter; ``ioic_to_naics_prefix`` extracts it. Letter-suffixed aggregates are handled
  by the prefix (e.g. ``BS111A00`` -> ``111``). An official IOIC-NAICS table can override.
* Commodities have no NAICS code: their mapping is taken from the supply (make) table as the
  share of each commodity produced by each industry (``commodity_to_industry``).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

# code: (description, ISIC Rev.4 divisions [2-digit] or groups/classes [3-4 digit])
ICIO2025 = {
    "A01": ("Agriculture and hunting", ["01"]),
    "A02": ("Forestry and logging", ["02"]),
    "A03": ("Fishing and aquaculture", ["03"]),
    "B05": ("Mining of coal and lignite", ["05"]),
    "B06": ("Crude petroleum and natural gas", ["06"]),
    "B07": ("Mining of metal ores", ["07"]),
    "B08": ("Other mining and quarrying", ["08"]),
    "B09": ("Mining support service activities", ["09"]),
    "C10T12": ("Food, beverages, tobacco", ["10", "11", "12"]),
    "C13T15": ("Textiles, apparel, leather", ["13", "14", "15"]),
    "C16": ("Wood and cork", ["16"]),
    "C17_18": ("Paper; printing", ["17", "18"]),
    "C19": ("Coke and refined petroleum", ["19"]),
    "C20": ("Chemicals", ["20"]),
    "C21": ("Pharmaceuticals", ["21"]),
    "C22": ("Rubber and plastics", ["22"]),
    "C23": ("Other non-metallic minerals", ["23"]),
    "C24A": ("Basic iron and steel", ["241", "2431"]),
    "C24B": ("Basic precious and non-ferrous metals", ["242", "2432"]),
    "C25": ("Fabricated metal products", ["25"]),
    "C26": ("Computer, electronic, optical", ["26"]),
    "C27": ("Electrical equipment", ["27"]),
    "C28": ("Machinery n.e.c.", ["28"]),
    "C29": ("Motor vehicles", ["29"]),
    "C301": ("Ships and boats", ["301"]),
    "C302T309": ("Other transport equipment", ["302", "303", "304", "309"]),
    "C31T33": ("Furniture, other manufacturing, repair", ["31", "32", "33"]),
    "D": ("Electricity, gas, steam", ["35"]),
    "E": ("Water, sewerage, waste", ["36", "37", "38", "39"]),
    "F": ("Construction", ["41", "42", "43"]),
    "G": ("Wholesale and retail trade", ["45", "46", "47"]),
    "H49": ("Land transport and pipelines", ["49"]),
    "H50": ("Water transport", ["50"]),
    "H51": ("Air transport", ["51"]),
    "H52": ("Warehousing and support for transport", ["52"]),
    "H53": ("Postal and courier", ["53"]),
    "I": ("Accommodation and food services", ["55", "56"]),
    "J58T60": ("Publishing, audiovisual, broadcasting", ["58", "59", "60"]),
    "J61": ("Telecommunications", ["61"]),
    "J62_63": ("IT and information services", ["62", "63"]),
    "K": ("Financial and insurance", ["64", "65", "66"]),
    "L": ("Real estate", ["68"]),
    "M": ("Professional, scientific, technical", ["69", "70", "71", "72", "73", "74", "75"]),
    "N": ("Administrative and support", ["77", "78", "79", "80", "81", "82"]),
    "O": ("Public administration and defence", ["84"]),
    "P": ("Education", ["85"]),
    "Q": ("Health and social work", ["86", "87", "88"]),
    "R": ("Arts, entertainment, recreation", ["90", "91", "92", "93"]),
    "S": ("Other service activities", ["94", "95", "96"]),
    "T": ("Households as employers", ["97", "98"]),
}
_SECTION_NUMERIC = {"D": "35", "E": "36T39", "F": "41T43", "G": "45T47", "I": "55T56", "K": "64T66",
                    "L": "68", "M": "69T75", "N": "77T82", "O": "84", "P": "85", "Q": "86T88",
                    "R": "90T93", "S": "94T96", "T": "97T98"}
_NUMERIC_TO_CANON = {}
for _c in ICIO2025:
    num = _SECTION_NUMERIC.get(_c, re.sub(r"^[A-Z]", "", _c))
    _NUMERIC_TO_CANON[num] = _c
    _NUMERIC_TO_CANON[num.replace("_", "T")] = _c
    _NUMERIC_TO_CANON[_c] = _c
    _NUMERIC_TO_CANON[_c.replace("_", "T")] = _c
# common older/alternative spellings
_NUMERIC_TO_CANON.update({"01T02": None, "05T06": None, "07T08": None})


def canonical_icio(label: str) -> str:
    """Map an ICIO industry label in either style to the canonical 2025 code."""
    lab = str(label).strip().upper()
    lab = lab.split("_", 1)[1] if re.match(r"^[A-Z]{3}_", lab) else lab  # strip 'CAN_'
    c = _NUMERIC_TO_CANON.get(lab)
    if c is None:
        raise KeyError(f"unknown or pre-2025 ICIO industry label: {label!r}")
    return c


def isic_to_icio(isic: str) -> str:
    """ISIC Rev.4 code (2-4 digits, letters stripped) -> ICIO 2025 code (most specific match)."""
    s = re.sub(r"[^0-9]", "", str(isic))
    best, blen = None, -1
    for code, (_, parts) in ICIO2025.items():
        for p in parts:
            if s.startswith(p) and len(p) > blen:
                best, blen = code, len(p)
    if best is None:
        raise KeyError(f"ISIC {isic} not covered")
    return best


def load_bridge(path=None) -> pd.DataFrame:
    return pd.read_csv(path or DATA / "naics_to_icio2025_bridge.csv", dtype=str)


def naics_candidates(code, bridge: pd.DataFrame | None = None) -> list:
    """ICIO codes a NAICS code (any length) can map to, using the bridge.

    * If a bridge entry is a prefix of the code (or equal to it), the longest such entry wins
      and that single ICIO code is returned. More specific bridge entries *below* the code
      (exceptions such as 3254 pharmaceuticals inside 325 chemicals) do not make the code
      ambiguous: they cover minor activities, and size-weighted splits are available through
      ``industry_concordance(class_weights=...)``.
    * If the code is shorter than every matching entry (e.g. '21', or an IOIC aggregate like
      'BS4A0000' -> '4'), all ICIO codes of the entries that extend it are returned.
    """
    b = load_bridge() if bridge is None else bridge
    s = re.sub(r"[^0-9]", "", str(code))
    if not s:
        return []
    pref = b[[s.startswith(p) for p in b["naics_prefix"]]]
    if not pref.empty:
        return [pref.loc[pref["naics_prefix"].str.len().idxmax(), "icio2025"]]
    ext = b[[p.startswith(s) for p in b["naics_prefix"]]]
    return list(dict.fromkeys(ext["icio2025"]))


_OFFICIAL = None


def load_official_table() -> pd.DataFrame:
    """NAICS (6-digit) -> ICIO 2025 links derived from Statistics Canada's official
    NAICS Canada 2022 V1.0 and 2017 V3.0 -> ISIC Rev.4 concordances
    (data/naics2022_to_icio2025_official.csv, data/naics2017_to_icio2025_official.csv;
    column 'naics2022' holds the code of either version; 'version' says which;
    weight = share of the class's ISIC links in that ICIO industry; primary = main ICIO
    industry, see METHODS 7.2)."""
    global _OFFICIAL
    if _OFFICIAL is None:
        parts = []
        for ver in ("2022", "2017"):  # 2022 first: it wins for the 831 codes present in both
            f = DATA / f"naics{ver}_to_icio2025_official.csv"
            if f.exists():
                t = pd.read_csv(f, dtype={f"naics{ver}": str}).rename(columns={f"naics{ver}": "naics2022"})
                parts.append(t.assign(version=ver))
        _OFFICIAL = (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["naics2022", "icio2025", "n_isic_links", "weight", "primary", "version"]))
        # a code present in both versions keeps only its 2022 links (identical ICIO mapping, tested)
        v22 = set(_OFFICIAL.loc[_OFFICIAL["version"] == "2022", "naics2022"])
        _OFFICIAL = _OFFICIAL[(_OFFICIAL["version"] == "2022") | ~_OFFICIAL["naics2022"].isin(v22)]
    return _OFFICIAL


def naics_to_icio_weights(code, class_weights: dict, links: str = "primary") -> dict:
    """{ICIO code: weight} for a NAICS code of any length, from the official concordance,
    weighting the 6-digit NAICS 2022 classes it contains by ``class_weights`` (e.g. Business
    Register employment by 6-digit NAICS; classes missing from the dict get 0).

    links='primary': each class goes wholly to its primary ICIO industry (recommended);
    links='all'    : each class is split over all its ISIC links equally (secondary, '*'
                     links are usually minor activities, so this over-weights them).
    Returns {} if no weighted class is found (callers then fall back to the bridge).
    """
    s = re.sub(r"[^0-9]", "", str(code))
    off = load_official_table()
    sub = off[off["naics2022"].str.startswith(s)] if s else off.iloc[0:0]
    if links == "primary":
        sub = sub[sub["primary"].astype(bool)].assign(weight=1.0)
    if not len(sub):
        return {}
    cw = sub["naics2022"].map(lambda c: float(class_weights.get(c, 0.0)))
    w = (sub["weight"] * cw).groupby(sub["icio2025"]).sum()
    return (w[w > 0] / w.sum()).to_dict() if w.sum() > 0 else {}


def naics_to_icio(code, bridge: pd.DataFrame | None = None) -> str | None:
    """Unique ICIO code for a NAICS code: the official primary link for 6-digit NAICS 2022 or
    2017 codes, otherwise the bridge (None if unmatched or ambiguous)."""
    s = re.sub(r"[^0-9]", "", str(code))
    if len(s) == 6:
        off = load_official_table()
        hit = off[(off["naics2022"] == s) & off["primary"].astype(bool)]
        if len(hit):
            return hit["icio2025"].iloc[0]
    c = naics_candidates(code, bridge)
    return c[0] if len(c) == 1 else None


def load_official_naics_isic(path) -> pd.DataFrame:
    """Read Statistics Canada's NAICS 2017 V3.0 -> ISIC Rev.4 concordance (CSV export).

    Column names vary across releases; the first column containing 'NAICS' and 'Code' and
    the first containing 'ISIC' and 'Code' are used. '*' (partial) flags are kept in 'partial'.
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # the NAICS 2017 V3.0 file is Latin-1
        text = raw.decode("latin-1")
    lines = text.splitlines()
    # skip title lines above the header (the 2017 file starts with a one-line title)
    hdr = next(i for i, l in enumerate(lines) if "NAICS" in l.upper() and "CODE" in l.upper() and "ISIC" in l.upper())
    import io
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), dtype=str)
    ncol = next(c for c in df.columns if "NAICS" in c.upper() and "CODE" in c.upper())
    icol = next(c for c in df.columns if "ISIC" in c.upper() and "CODE" in c.upper())
    tcol = next((c for c in df.columns if "NAICS" in c.upper() and "TITLE" in c.upper()), None)
    out = pd.DataFrame({"naics": df[ncol].str.strip(), "isic": df[icol].str.strip(),
                        "title": df[tcol] if tcol else ""})
    # ISIC codes look like 'A0111', 'A0111*' (partial link) or 'A011X' (aggregate of a group)
    out["partial"] = out["isic"].str.contains(r"\*", regex=True)
    out["isic_digits"] = out["isic"].str.replace(r"[^0-9]", "", regex=True)
    out = out[out["isic_digits"].str.len() >= 2].copy()

    def safe(code):
        try:
            return isic_to_icio(code)
        except KeyError:  # ISIC 99 (extraterritorial organisations) has no ICIO industry
            return None
    out["icio2025"] = out["isic_digits"].map(safe)
    out.attrs["unmapped"] = out.loc[out["icio2025"].isna(), ["naics", "isic"]].values.tolist()
    return out.dropna(subset=["icio2025"])


def official_naics_to_icio_weights(off: pd.DataFrame) -> pd.DataFrame:
    """NAICS 6-digit -> ICIO weights from the official table.

    The concordance has no value data, so a NAICS class is split equally over the ISIC links
    it has ('n' = number of links into that ICIO industry). 'primary' marks the ICIO industry
    with the most links (ties: the first listed, which StatCan lists for the main activity in
    most cases). Partial links ('*') typically cover minor activities of the class.
    """
    off = off.reset_index(drop=True)
    first = off.groupby("naics").head(1).set_index("naics")["icio2025"]
    g = off.groupby(["naics", "icio2025"]).size().rename("n").reset_index()
    g["weight"] = g["n"] / g.groupby("naics")["n"].transform("sum")
    g["is_first"] = g["icio2025"] == g["naics"].map(first)
    g = g.sort_values(["naics", "n", "is_first"], ascending=[True, False, False])
    g["primary"] = ~g.duplicated("naics")
    return g[["naics", "icio2025", "n", "weight", "primary"]].reset_index(drop=True)


def validate_bridge(off: pd.DataFrame, bridge: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare the bridge with the official concordance, NAICS class by class.

    Returns one row per NAICS class with columns: bridge (ICIO code from the bridge), primary
    (official primary ICIO code), all_official (all ICIO codes it links to), status in
    {'agree_primary', 'agree_secondary' (bridge = a non-primary official link),
    'disagree', 'missing' (bridge has no entry)}.
    """
    w = official_naics_to_icio_weights(off)
    b = load_bridge() if bridge is None else bridge
    rows = []
    for n, g in w.groupby("naics"):
        br = naics_to_icio(n, b)
        prim = g.loc[g["primary"], "icio2025"].iloc[0]
        allo = list(g["icio2025"])
        st = ("missing" if br is None else "agree_primary" if br == prim
              else "agree_secondary" if br in allo else "disagree")
        rows.append(dict(naics=n, bridge=br, primary=prim, all_official="|".join(allo),
                         primary_weight=float(g.loc[g["primary"], "weight"].iloc[0]), status=st))
    return pd.DataFrame(rows)


def naics2017_to_2022(path=None) -> pd.DataFrame:
    """Statistics Canada's NAICS 2017 V3.0 -> 2022 V1.0 correspondence (changed codes only).
    Use it to recode 2017-coded data (e.g. ODBus, older IOIC tables) before NAICS-level work."""
    f = path or DATA / "statcan_naics2017v3_to_2022v1.csv"
    d = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    return d.rename(columns={d.columns[0]: "naics2017", d.columns[3]: "naics2022", d.columns[2]: "change"})[
        ["naics2017", "naics2022", "change"]]


def ioic_to_naics_prefix(ioic: str) -> str:
    """'BS111A00' -> '111', 'BS21E000' -> '21', 'GS611100' -> '6111', 'NP813000' -> '813'."""
    s = str(ioic).strip().upper()
    s = re.sub(r"^(BS|NP|GS|HH)", "", s)
    m = re.match(r"^([0-9]+)", s)
    if not m:
        raise ValueError(f"no NAICS prefix in IOIC code {ioic!r}")
    return m.group(1).rstrip("0") or m.group(1)


def ioic_to_icio(ioic: str, bridge=None, overrides: dict | None = None) -> list:
    """Candidate ICIO codes for an IOIC industry (override > NAICS prefix in the code)."""
    if overrides and ioic in overrides:
        v = overrides[ioic]
        return [v] if isinstance(v, str) else list(v)
    p = ioic_to_naics_prefix(ioic)
    c = naics_candidates(p, bridge)
    if not c:
        raise KeyError(f"IOIC {ioic} (NAICS prefix {p}) not in bridge; supply an override")
    return c


def industry_concordance(ioic_codes, icio_codes=None, overrides=None, split_weights=None,
                         report: list | None = None, class_weights: dict | None = None) -> np.ndarray:
    """(n_ioic, n_icio) weights mapping SUT industries to ICIO 2025 industries (rows sum to 1).

    Mapping, in order of preference:
      1. ``overrides`` (exact, user-supplied);
      2. ``class_weights`` given: the official NAICS 2022 concordance, with the 6-digit classes
         inside the IOIC prefix weighted by those sizes (``naics_to_icio_weights``);
      3. otherwise the bridge (6-digit codes: official primary link). IOIC aggregates spanning
         several ICIO industries (e.g. 'BS21E000' = mining except oil and gas) are split with
         ``split_weights`` (ICIO code -> weight, e.g. ICIO Canada gross output) or equally.
    Ambiguous codes are appended to ``report``.
    """
    icio = list(ICIO2025) if icio_codes is None else [canonical_icio(c) for c in icio_codes]
    b = load_bridge()
    M = np.zeros((len(ioic_codes), len(icio)))
    for n, c in enumerate(ioic_codes):
        base = {}
        if class_weights is not None and not (overrides and c in overrides):
            base = {k: v for k, v in naics_to_icio_weights(ioic_to_naics_prefix(c), class_weights).items()
                    if k in icio}
        if base:
            cands = list(base)
            w = np.array([base[x] for x in cands])
        else:
            cands = [x for x in ioic_to_icio(c, b, overrides) if x in icio]
            w = np.array([1.0 if split_weights is None else float(split_weights.get(x, 0.0)) for x in cands])
            if w.sum() <= 0:
                w = np.ones(len(cands))
        for x, wx in zip(cands, w / w.sum()):
            M[n, icio.index(x)] = wx
        if len(cands) > 1 and report is not None:
            report.append((c, cands))
    return M


def commodity_to_industry(supply, normalize: bool = True) -> np.ndarray:
    """Commodity -> industry weights from a supply (make) table V (commodities x industries):
    w[c, n] = V[c, n] / sum_n V[c, n] (share of commodity c produced by industry n)."""
    V = np.clip(np.asarray(supply, float), 0, None)
    if not normalize:
        return V
    rs = V.sum(axis=1, keepdims=True)
    return np.divide(V, rs, out=np.zeros_like(V), where=rs > 0)


def commodity_concordance(supply, ioic_codes, icio_codes=None, **kw) -> np.ndarray:
    """(n_commodity, n_icio): commodity -> ICIO via the supply table and the IOIC mapping."""
    return commodity_to_industry(supply) @ industry_concordance(ioic_codes, icio_codes, **kw)


# ------------------------------------------------------------------ NAICS-level sectors
# NAICS 2-digit sectors that span several 2-digit codes (official NAICS sector definitions)
NAICS2_COMBOS = {"31": "31-33", "32": "31-33", "33": "31-33", "44": "44-45", "45": "44-45",
                 "48": "48-49", "49": "48-49"}


def naics_digits(x) -> str:
    """'..' / NaN -> ''; '44-45' -> '44'; '446110' -> '446110'."""
    s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip()
    if s in ("..", "nan", "None", ""):
        return ""
    return re.sub(r"[^0-9]", "", s.split("-")[0])


def naics_level(code, level: int):
    """NAICS code truncated to ``level`` digits; 2-digit uses the sector combos (31-33 ...)."""
    if not isinstance(code, str) or len(code) < level:
        return None
    c = code[:level]
    return NAICS2_COMBOS.get(c, c) if level == 2 else c


def naics_level_concordance(ioic_codes, sectors, level: int, split_weights=None, overrides=None,
                            report: list | None = None) -> np.ndarray:
    """(n_ioic, n_sectors) weights mapping SUT industries (IOIC) to NAICS sectors at ``level``
    digits (e.g. the ODBus derived_NAICS sectors with level=2, or finer ODBus-based sectors).

    If the IOIC prefix is at least ``level`` digits the mapping is unique; if it is shorter
    (aggregates such as 'BS21E000' at level 3) the industry is split over the listed sectors
    that extend the prefix, by ``split_weights`` (e.g. province-wide employment from ODBus/BR)
    or equally. Government ('GS') and non-profit ('NP') codes use the NAICS digits they carry.
    """
    sectors = list(sectors)
    M = np.zeros((len(ioic_codes), len(sectors)))
    for n, code in enumerate(ioic_codes):
        if overrides and code in overrides:
            cands = [overrides[code]] if isinstance(overrides[code], str) else list(overrides[code])
        else:
            p = ioic_to_naics_prefix(code)
            if len(p) >= level:
                cands = [naics_level(p, level)]
            else:
                cands = [s for s in sectors if re.sub(r"-.*", "", s).startswith(p)
                         or (level == 2 and naics_level(p.ljust(2, "0")[:2], 2) == s and len(p) == 2)]
        cands = [c for c in cands if c in sectors]
        if not cands:
            raise KeyError(f"IOIC {code}: no matching sector at level {level}")
        w = np.array([1.0 if split_weights is None else float(split_weights.get(c, 0.0)) for c in cands])
        w = w if w.sum() > 0 else np.ones(len(cands))
        for c, wc in zip(cands, w / w.sum()):
            M[n, sectors.index(c)] = wc
        if len(cands) > 1 and report is not None:
            report.append((code, cands))
    return M


# Final-demand categories in the provincial SUT map to the ICIO user 'FD'.
FINAL_DEMAND_PREFIXES = ("PEC", "PE", "CE", "ME", "IN", "EX", "FD", "HH", "NP", "GOV", "GFCF", "INV")
