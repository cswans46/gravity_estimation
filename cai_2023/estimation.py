from pathlib import Path
import numpy as np
import pandas as pd
import pyreadr
import xarray as xr
import os

def prepare_ICIO_data(data):
    """Build Cai-style normalized bilateral trade flows z_{ijk,t} from the ICIO data.

    i = origin country, j = destination country, k = supplying industry,
    m = purchasing industry, t = year.
    """
    Z, X, FD = data["Z"], data["X"], data["FD"]

    years = list(np.asarray(Z.coords["dim_0"].values, dtype=object))
    row_labels = list(np.asarray(Z.coords["dim_1"].values, dtype=object))
    countries = list(dict.fromkeys(l.split("_", 1)[0] for l in row_labels))
    industries = list(dict.fromkeys(l.split("_", 1)[1] for l in row_labels))
    nT, nC, nK = len(years), len(countries), len(industries)

    # Reshaping below assumes country-major ordering (AGO_01..AGO_97T98, ARE_01, ...).
    expected = [f"{c}_{k}" for c in countries for k in industries]
    assert row_labels == expected, "Z rows are not country-major ordered"
    assert list(np.asarray(Z.coords["dim_2"].values, dtype=object)) == expected
    assert list(np.asarray(FD.coords["dim_2"].values, dtype=object)) == countries

    # Z_{ik,jm,t} as a 5-D array: (t, i, k, j, m)
    Z5 = Z.values.reshape(nT, nC, nK, nC, nK)

    # F_{ijk,t} = sum_m Z_{ik,jm,t}  -> bilateral trade by supplying industry.
    # Diagonal (i == j) entries are retained: domestic transactions stay in.
    F = Z5.sum(axis=4)                                # (t, i, k, j)

    # Y_{ik,t} = X_{ik,t}: gross output of industry k in country i.
    Y = X.to_numpy().reshape(nT, nC, nK)              # (t, i, k)

    # E_{jk,t} = sum_i sum_m Z_{ik,jm,t} + sum_i FD_{ik,j,t}:
    # destination j's total (intermediate + final) expenditure on industry-k output.
    FD4 = FD.values.reshape(nT, nC, nK, nC)           # (t, i, k, j)
    E = Z5.sum(axis=(1, 4)) + FD4.sum(axis=1)         # (t, k, j)

    # z_{ijk,t} = F_{ijk,t} / (Y_{ik,t} * E_{jk,t})
    with np.errstate(divide="ignore", invalid="ignore"):
        z = F / (Y[:, :, :, None] * E[:, None, :, :])  # (t, i, k, j)

    dims = ("year", "origin", "industry", "destination")
    coords = {"year": years, "origin": countries,
              "industry": industries, "destination": countries}
    return {
        "F": xr.DataArray(F, dims=dims, coords=coords),
        "Y": xr.DataArray(Y, dims=dims[:3], coords={d: coords[d] for d in dims[:3]}),
        "E": xr.DataArray(E, dims=("year", "industry", "destination"),
                          coords={d: coords[d] for d in ("year", "industry", "destination")}),
        "z": xr.DataArray(z, dims=dims, coords=coords),
    }


def main():
    # Bring in ICIO data
    data_dir = r"D:\Models\OECD Research"
    data_path = Path(data_dir)
    output_path = os.path.dirname(os.path.abspath(__file__))
    # Intermediate flow matrix, output, and final demand files required.
    files = ["Z","X","FD"]
    data = {}
    
    for f in files:
        file_path = data_path / f"ICIO2025econ{f}.Rdata"
        result = pyreadr.read_r(str(file_path))
        # grab the first (usually only) object in the .Rdata file
        data[f] = next(iter(result.values()))
        
    ICIO_data_prepared = prepare_ICIO_data(data)
    
    # Bring in CEPII data (distwces is the population weighted distance between countries).
    # It also features a \theta = -1 for CES distance transformation.
    distwces_cepii_file = output_path / "dist_cepii.xls"
    # Bring in language etc. control by bilateral country pair.
    controls_cepii_file = output_path / "geo_cepii.xlsx"
    distwces_cepii = pd.read_excel(distwces_cepii_file)
    controls_cepii = pd.read_excel(controls_cepii_file)
    
    
    
if __name__ == "__main__":
    main()
    