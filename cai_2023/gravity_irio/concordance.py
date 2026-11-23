"""Concordances between Statistics Canada classifications and the OECD ICIO 2025 industries.

Chain:  SUPC commodity --(national supply table)--> IOIC industry --(code structure / official
        IOIC-NAICS table)--> NAICS 2017 --(bridge)--> ISIC Rev.4 --> ICIO 2025 (50 industries)

* ``ICIO2025`` lists the 50 ICIO 2025 industries (OECD 2026 ICIO documentation annex) with the
  ISIC Rev.4 divisions/groups they cover. ``canonical_icio`` accepts both the letter style
  (``A01``, ``C10T12``, ``C24A``, ``D``) and the numeric style of the R objects (``01``,
  ``10T12``, ``24A``, ``35``, ``97T98``).
* ``data/naics2017_to_icio2025.csv`` is a hand-built NAICS -> ICIO bridge at the 2- to
  5-digit level (longest-prefix match). It was written from the NAICS 2017 and ISIC Rev.4
  structures; ``validate_bridge`` checks it against Statistics Canada's official
  NAICS 2017 V3.0 -> ISIC Rev.4 concordance (download from
  https://www.statcan.gc.ca/en/subjects/standard/naics/2017/v3/concordance-rev4).
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
    return pd.read_csv(path or DATA / "naics2017_to_icio2025.csv", dtype=str)


def naics_candidates(code, bridge: pd.DataFrame | None = None) -> list:
    """ICIO codes a NAICS code (any length) can map to.

    Longest-prefix match first (a 6-digit code gets exactly one answer). If the code is
    *shorter* than the bridge entries (e.g. '21' or an IOIC aggregate), all ICIO codes of
    bridge entries that extend it are returned.
    """
    b = load_bridge() if bridge is None else bridge
    s = re.sub(r"[^0-9]", "", str(code))
    if not s:
        return []
    m = b[[s.startswith(p) for p in b["naics_prefix"]]]
    longest = None
    if not m.empty:
        longest = m.loc[m["naics_prefix"].str.len().idxmax()]
    ext = b[[p.startswith(s) and len(p) > len(s) for p in b["naics_prefix"]]]
    if longest is not None and (ext.empty or len(longest["naics_prefix"]) >= len(s)):
        # children of s that map elsewhere (e.g. 3254 inside 325) make s itself ambiguous
        cands = [longest["icio2025"]] + list(ext["icio2025"])
        return list(dict.fromkeys(cands))
    return list(dict.fromkeys(ext["icio2025"]))


def naics_to_icio(code, bridge: pd.DataFrame | None = None) -> str | None:
    """Unique ICIO code for a NAICS code, or None if unmatched / ambiguous."""
    c = naics_candidates(code, bridge)
    return c[0] if len(c) == 1 else (c[0] if c and len(re.sub(r"[^0-9]", "", str(code))) >= 6 else None)


def load_official_naics_isic(path) -> pd.DataFrame:
    """Read Statistics Canada's NAICS 2017 V3.0 -> ISIC Rev.4 concordance (CSV export).

    Column names vary across releases; the first column containing 'NAICS' and 'Code' and
    the first containing 'ISIC' and 'Code' are used. '*' (partial) flags are kept in 'partial'.
    """
    df = pd.read_csv(path, dtype=str, encoding_errors="replace")
    ncol = next(c for c in df.columns if "NAICS" in c.upper() and "CODE" in c.upper())
    icol = next(c for c in df.columns if "ISIC" in c.upper() and "CODE" in c.upper())
    out = pd.DataFrame({"naics": df[ncol].str.strip(), "isic": df[icol].str.strip()})
    out["partial"] = out["isic"].str.contains(r"\*", regex=True)
    out["isic_digits"] = out["isic"].str.replace(r"[^0-9]", "", regex=True)
    out = out[out["isic_digits"].str.len() >= 2]
    out["icio2025"] = out["isic_digits"].map(isic_to_icio)
    return out


def official_naics_to_icio_weights(off: pd.DataFrame) -> pd.DataFrame:
    """NAICS 6-digit -> ICIO weights from the official table (equal split over the ISIC
    classes a NAICS class links to; no value data are available to weight them)."""
    g = off.groupby(["naics", "icio2025"]).size().rename("n").reset_index()
    g["weight"] = g["n"] / g.groupby("naics")["n"].transform("sum")
    return g[["naics", "icio2025", "weight"]]


def validate_bridge(off: pd.DataFrame, bridge: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare the hand-built bridge with the official concordance, NAICS class by class.
    Returns rows where the bridge's ICIO code is not among the official ones."""
    w = official_naics_to_icio_weights(off)
    b = load_bridge() if bridge is None else bridge
    w["bridge"] = [naics_to_icio(n, b) for n in w["naics"]]
    agree = w.groupby("naics").apply(lambda g: g["bridge"].iloc[0] in set(g["icio2025"]))
    bad = agree[~agree].index
    return w[w["naics"].isin(bad)]


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
                         report: list | None = None) -> np.ndarray:
    """(n_ioic, n_icio) weights mapping SUT industries to ICIO 2025 industries (rows sum to 1).

    Ambiguous IOIC aggregates (e.g. 'BS21E000' = mining except oil and gas -> B05/B07/B08)
    are split with ``split_weights`` (dict ICIO code -> weight, e.g. Canada's ICIO gross
    output by industry) or equally if None. Ambiguous codes are appended to ``report``.
    """
    icio = list(ICIO2025) if icio_codes is None else [canonical_icio(c) for c in icio_codes]
    b = load_bridge()
    M = np.zeros((len(ioic_codes), len(icio)))
    for n, c in enumerate(ioic_codes):
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
