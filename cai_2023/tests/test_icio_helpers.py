import numpy as np
import pytest
import xarray as xr
from gravity_icio.icio import _country_industry, build_bilateral_flows


def test_country_industry_parser():
    assert _country_industry(["CAN_01", "USA_02"]) == [("CAN", "01"), ("USA", "02")]


def test_country_industry_parser_rejects_bad_label():
    with pytest.raises(ValueError):
        _country_industry(["CAN01"])


def test_bilateral_flow_normalization_and_absorption():
    years = [2000]
    rows = ["AAA_01", "AAA_02", "BBB_01", "BBB_02"]
    countries = ["AAA", "BBB"]
    Z = xr.DataArray(
        np.ones((1, 4, 4)),
        dims=("dim_0", "dim_1", "dim_2"),
        coords={"dim_0": years, "dim_1": rows, "dim_2": rows},
    )
    X = xr.DataArray(
        np.full((1, 4), 10.0),
        dims=("dim_0", "dim_1"),
        coords={"dim_0": years, "dim_1": rows},
    )
    FD = xr.DataArray(
        np.ones((1, 4, 2)),
        dims=("dim_0", "dim_1", "dim_2"),
        coords={"dim_0": years, "dim_1": rows, "dim_2": countries},
    )
    cfg = {"icio": {"exclude_countries": [], "years": [2000, 2000]}}
    flow, output, absorption = build_bilateral_flows({"Z": Z, "X": X, "FD": FD}, cfg)

    assert len(flow) == 8
    assert np.allclose(flow["F"], 3.0)
    assert np.allclose(flow["z"], 0.05)
    assert np.allclose(output["output"], 10.0)
    assert np.allclose(absorption["absorption"], 6.0)
