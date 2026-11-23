"""StatCan table conversion, SUT -> SDM inputs, province distances (synthetic files)."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from gravity_irio import statcan_tables as st
from gravity_irio.canada_sut import build_sdm_inputs, ioic_sector
from gravity_irio.province_distances import load_gaf_points, population_distances

PROVS = ["Ontario", "Quebec", "Manitoba"]
INDS = ["Crop production [BS111A00]", "Retail trade [BS4A0000]", "Food manufacturing [BS311000]"]
FD = ["Household final consumption expenditure [PEC00000]", "Total use [TU]"]
PRODS = ["Wheat [MPG111A03]", "Meat [MPG311A01]", "Retail services [MPS44A000]"]


def _zip(path, df, name):
    b = io.BytesIO()
    df.to_csv(b, index=False)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, b.getvalue())
    return path


@pytest.fixture
def canada(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for g in PROVS + ["Canada"]:
        for pi, p in enumerate(PRODS):
            for n, i in enumerate(INDS):
                rows.append((2021, g, "Basic price", "Supply", i, p, 1000.0 if n == pi else 0.0))
                rows.append((2021, g, "Basic price", "Use", i, p, float(rng.integers(10, 100))))
            for m in FD:
                rows.append((2021, g, "Basic price", "Use", m, p, 300.0))
    sut = pd.DataFrame(rows, columns=["REF_DATE", "GEO", "Valuation", "Supply and use", "Industry", "Product", "VALUE"])
    sut["SCALAR_FACTOR"] = "thousands"
    _zip(tmp_path / "sut.zip", sut, "36100478.csv")
    rows = []
    for g in PROVS:
        for p in PRODS:
            for h in PROVS:
                rows.append((2021, g, f"To {h}", p, 50.0))
            rows += [(2021, g, "International exports", p, 100.0), (2021, g, "International imports", p, 20.0),
                     (2021, g, "International re-exports", p, 1.0)]
    ipt = pd.DataFrame(rows, columns=["REF_DATE", "GEO", "Trade flow detail", "Product", "VALUE"])
    ipt["SCALAR_FACTOR"] = "thousands"
    _zip(tmp_path / "ipt.zip", ipt, "12100101.csv")
    out = tmp_path / "canada"
    st.convert_12100101(tmp_path / "ipt.zip", out)
    st.convert_36100478(tmp_path / "sut.zip", out)
    return tmp_path, out


def test_roles_and_layout(canada):
    _, out = canada
    r = pd.read_csv(out / "sut_member_roles.csv")
    role = dict(zip(r["code"], r["role"]))
    assert role["BS4A0000"] == "industry"          # 'retail' must not be read as a margin
    assert role["PEC00000"] == "final_household" and role["TU"] == "ignore"
    x = pd.read_csv(out / "international_exports_panel.csv")
    assert x["value"].eq(100.0).all()               # re-exports kept separately


def test_sdm_inputs_identity(canada):
    tmp, out = canada
    pd.DataFrame([(a, b, 100.0 if a != b else 30.0) for a in ["ON", "QC", "MB"] for b in ["ON", "QC", "MB"]],
                 columns=["origin", "destination", "km"]).to_csv(out / "province_distances.csv", index=False)
    r = build_sdm_inputs(out, tmp / "sdm", distances=out / "province_distances.csv", verbose=False)
    assert set(r["sectors"]) == {"11", "31-33", "44-45"}
    import spatial_irio
    data, extra = spatial_irio.load_sdm(tmp / "sdm")
    assert data.national_identity_gap() < 1e-8
    assert "mass_exo" in extra and "home" in extra


def test_ioic_sector():
    assert ioic_sector("BS111A00") == "11" and ioic_sector("GS611100") == "61"
    assert ioic_sector("BS4A0000", "Retail trade") == "44-45"
    with pytest.raises(KeyError):
        ioic_sector("BS9Z0000", "Something")


def test_gaf_population_distances(tmp_path):
    rng = np.random.default_rng(1)
    rows = [("35", f"35{n:04d}", 43.7 + rng.normal(0, .3), -79.4 + rng.normal(0, .3), 100) for n in range(50)]
    rows += [("24", f"24{n:04d}", 46.8 + rng.normal(0, .3), -71.2 + rng.normal(0, .3), 100) for n in range(50)]
    d = pd.DataFrame(rows, columns=["PRUID_PRIDU", "DAUID_ADIDU", "DARPLAT_ADLAT", "DARPLONG_ADLONG",
                                    "DBPOP2021_IDPOP2021"])
    D, p = population_distances(load_gaf_points(_zip(tmp_path / "gaf.zip", d, "gaf.csv")))
    assert p == ["QC", "ON"]
    assert 500 < D[0, 1] < 800 and D[0, 0] < 100


def test_default_fd_keys_priority():
    from gravity_irio.final_demand import default_keys
    av = {"population": np.ones((1, 3)), "hh_disposable_income": np.ones((1, 3)) * 2,
          "va": np.ones((1, 3)), "va_by_sector": np.ones((1, 2, 3))}
    k = default_keys(av)
    assert default_keys.chosen["household"] == "hh_disposable_income"
    assert default_keys.chosen["gfcf"] == "va" and default_keys.chosen["exports"] == "va_by_sector"
    with pytest.raises(KeyError):
        default_keys({"population": np.ones((1, 3))})


def test_loguniform_shape_prior():
    from gravity_irio.sdm import TorojPrior
    kw = dict(home_mean=np.array([0.7]), home_sd=np.array([0.1]), r_home=400.0, d_max=4000.0, n_grid=20)
    pu, pl = TorojPrior(**kw), TorojPrior(**kw, shape_prior="loguniform")
    la = np.log(0.05)
    lc = np.log(pu._interp(pu.mu[0], la))
    # same conditional scale prior; the log-uniform shape density is flat in log shape
    assert np.isfinite(pu.logpdf(0, la, lc)) and np.isfinite(pl.logpdf(0, la, lc))
    d_u = pu.logpdf(0, la + 1, lc) - pu.logpdf(0, la, lc)
    d_l = pl.logpdf(0, la + 1, lc) - pl.logpdf(0, la, lc)
    assert d_u - d_l == pytest.approx(1.0, abs=1e-9)
    assert pu.logpdf(0, np.log(0.005), lc) == -np.inf
