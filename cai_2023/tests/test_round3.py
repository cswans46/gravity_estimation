import struct
from pathlib import Path

import numpy as np
import pytest

from gravity_irio.business_locations import expand_coarse, impute_employees, load_odbus
from gravity_irio.concordance import naics_level, naics_level_concordance
from gravity_irio.final_demand import fit_fd_keys_on_canada
from gravity_irio.geography import first_boundary_crossing, lcc3347, read_shapefile, toroj_r_home
from gravity_irio.sdm import CANADA_CAPITALS, canada_r_home
from gravity_irio.distances import haversine

FIX = Path(__file__).parent / "fixtures" / "odbus_sample.csv"


def test_odbus_exact_columns_and_naics_refinement():
    d2 = load_odbus(FIX, "naics2")
    # the ten rows supplied by the user: pharmacies derived_NAICS 44 -> sector 44-45
    moncton = d2[d2["prov"] == "NB"]
    assert len(moncton) == 10
    assert (moncton.loc[moncton["text"].str.startswith("community pharmacy"), "sector"] == "44-45").all()
    # 'Limited Groceries' has no derived code: learned from coded records with the same text
    lim = moncton[moncton["text"].str.startswith("limited groceries")]
    assert (lim["sector"] == "44-45").all() and (lim["naics_source"] == "text").all()
    # 'General Groceries' has no coded peers -> stays unassigned
    assert moncton.loc[moncton["text"].str.startswith("general groceries"), "sector"].isna().all()
    d4 = load_odbus(FIX, "naics4")
    assert set(d4.loc[d4["prov"] == "NB", "sector"].dropna()) <= {"4461", "4451"}
    w = impute_employees(d4)
    e = expand_coarse(w, 4)
    assert e["sector"].notna().all()


def test_naics_level_helpers():
    assert naics_level("446110", 2) == "44-45" and naics_level("332710", 3) == "332"
    secs = ["11", "21", "31-33", "44-45", "48-49", "54"]
    M = naics_level_concordance(["BS111A00", "BS324000", "BS484000", "BS541500"], secs, 2)
    assert [secs[i] for i in M.argmax(1)] == ["11", "31-33", "48-49", "54"]


def _write_square_shapefile(path, squares):
    """Tiny polygon shapefile writer (test helper) -- squares: {code: (x0, y0, side)}."""
    recs = []
    for n, (code, (x0, y0, a)) in enumerate(squares.items(), 1):
        pts = np.array([[x0, y0], [x0, y0 + a], [x0 + a, y0 + a], [x0 + a, y0], [x0, y0]], float)
        body = struct.pack("<i4d", 5, x0, y0, x0 + a, y0 + a) + struct.pack("<2i", 1, 5) + struct.pack("<i", 0)
        body += pts.astype("<f8").tobytes()
        recs.append(struct.pack(">2i", n, len(body) // 2) + body)
    content = b"".join(recs)
    hdr = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(content)) // 2) + struct.pack("<2i4d4d", 1000, 5, *([0.0] * 8))
    Path(path).write_bytes(hdr + content)
    # dbf with one character field PRUID
    names = list(squares)
    fld = b"PRUID".ljust(11, b"\x00") + b"C" + b"\x00" * 4 + bytes([2]) + b"\x00" * 15
    hlen, rlen = 32 + 32 + 1, 1 + 2
    dbf = struct.pack("<BBBBIHH", 3, 26, 1, 1, len(names), hlen, rlen) + b"\x00" * 20 + fld + b"\x0D"
    dbf += b"".join(b" " + nm.encode().ljust(2) for nm in names) + b"\x1A"
    Path(path).with_suffix(".dbf").write_bytes(dbf)


def test_shapefile_reader_and_exact_toroj_radius(tmp_path):
    # two adjacent 200 km squares in EPSG:3347 metres around Ottawa; capitals at their centres
    x0, y0 = lcc3347(45.0, -76.0)
    shp = tmp_path / "prov.shp"
    _write_square_shapefile(shp, {"35": (x0 - 200e3, y0 - 100e3, 200e3), "24": (x0, y0 - 100e3, 200e3)})
    polys = read_shapefile(shp, "PRUID")
    assert set(polys) == {"35", "24"} and polys["35"][0].shape == (5, 2)
    # invert approximately: place capitals 100 km either side of x0 along the projected x axis
    from scipy.optimize import minimize
    def inv(x, y):
        f = lambda p: np.hypot(*(np.array(lcc3347(*p)) - [x, y]))
        return tuple(minimize(f, [45.0, -76.0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-6}).x)
    caps = {"ON": inv(x0 - 100e3, y0), "QC": inv(x0 + 100e3, y0)}
    r, rows = toroj_r_home({"ON": polys["35"], "QC": polys["24"]}, caps, {"ON": ["QC"], "QC": ["ON"]})
    assert abs(r - 100.0) < 0.5 and len(rows) == 2


def test_canada_r_home_is_pooled_mean_of_half_distances():
    r, rows = canada_r_home(["ON", "QC", "MB"], return_pairs=True)
    expect = np.mean([0.5 * haversine(*CANADA_CAPITALS[a], *CANADA_CAPITALS[b]) for a, b, _ in rows])
    assert np.isclose(r, expect) and len(rows) == 4  # ON-QC, ON-MB, QC-ON, MB-ON


def test_first_boundary_crossing_concave():
    L = [np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10], [0, 0]], float)]
    assert np.isclose(first_boundary_crossing([2, 2], [20, 2], L), 8.0)
    assert np.isclose(first_boundary_crossing([2, 2], [2, 20], L), 8.0)
    assert np.isclose(first_boundary_crossing([2, 8], [20, 8], L), 2.0)


def test_fd_keys_learned_on_canada_recover_structure():
    rng = np.random.default_rng(0)
    T, S, R = 4, 5, 10
    pop = rng.lognormal(0, 1, (T, R)); inc = pop * rng.lognormal(0, .4, (T, R))
    va = inc[:, None, :] * rng.lognormal(0, .6, (T, S, R))
    nat = rng.uniform(50, 100, (T, S))[:, :, None]

    def alloc(x):
        x = x / x.sum(2, keepdims=True)
        return nat * x * rng.lognormal(0, .05, x.shape)
    FU = {"household": alloc(np.broadcast_to(inc[:, None, :] ** 0.9, (T, S, R))),
          "government": alloc(np.broadcast_to(pop[:, None, :], (T, S, R))), "exports": alloc(va)}
    m = fit_fd_keys_on_canada(FU, {"population": pop, "income": inc, "va_sector": va})
    assert m.keys["household"][0] == ["income"] and abs(m.keys["household"][1][0] - 0.9) < 0.05
    assert m.keys["government"][0] == ["population"]
    assert m.keys["exports"][0] == ["va_sector"]
    assert m.report["chosen"].sum() == 3


def test_bridge_consistent_with_official_naics2022_concordance():
    from gravity_irio.concordance import DATA, load_official_naics_isic, validate_bridge, naics_to_icio
    off = load_official_naics_isic(DATA / "statcan_naics2022_v1_isic4.csv")
    assert off["naics"].nunique() == 921
    v = validate_bridge(off)
    assert (v["status"] == "disagree").sum() == 0 and (v["status"] == "missing").sum() == 0
    # NAICS 2022 renumbering handled; 2017-only codes still map through the bridge
    assert naics_to_icio("513110") == "J58T60" and naics_to_icio("519211") == "R"
    assert naics_to_icio("456110") == "G" and naics_to_icio("511110") == "J58T60"
    assert naics_to_icio("324121") == "C23" and naics_to_icio("324110") == "C19"


def test_size_weighted_official_mapping():
    from gravity_irio.concordance import ICIO2025, industry_concordance
    ic = list(ICIO2025)
    M = industry_concordance(["BS324000"], class_weights={"324110": 900, "324121": 40, "324122": 30, "324190": 30})
    assert np.isclose(M[0, ic.index("C23")], 0.07) and np.isclose(M[0].sum(), 1)
    M0 = industry_concordance(["BS324000"])
    assert M0[0, ic.index("C19")] == 1.0


def test_shapefile_reader_accepts_zip(tmp_path):
    import zipfile
    shp = tmp_path / "lpr.shp"
    _write_square_shapefile(shp, {"35": (0.0, 0.0, 10.0)})
    z = tmp_path / "lpr_000b21a_e.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(shp, "lpr_000b21a_e.shp")
        f.write(shp.with_suffix(".dbf"), "lpr_000b21a_e.dbf")
    polys = read_shapefile(z, "PRUID")
    assert list(polys) == ["35"] and polys["35"][0].shape == (5, 2)
