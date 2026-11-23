"""OECD ICIO data preparation (Cai 2023 normalisation, extended to supplier x user pairs).

Notation (follows the original ``estimation.py``):
    i = origin country, j = destination country, k = supplying industry,
    m = purchasing (using) industry, t = year.

Two flow concepts are built:

* Sector level (Cai 2023):        F_{ijk,t} = sum_m Z_{ik,jm,t} [+ FD_{ik,j,t}]  -- see note
* Supplier x user pair level:     F_{ij,km,t} = Z_{ik,jm,t}     (m = 1..K)
                                  F_{ij,k,FD,t} = FD_{ik,j,t}   (final demand as user "FD")

NOTE on the sector-level flow. The original code used intermediate flows only
(F = sum_m Z) for the numerator, but total (intermediate + final) expenditure for E.
Cai (2023) models *total* bilateral trade of product k. ``include_final_demand``
controls whether FD is added to F (default True, consistent with E). Set it to False
to reproduce the earlier behaviour. This is flagged in docs/METHODS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def split_labels(row_labels):
    countries = list(dict.fromkeys(l.split("_", 1)[0] for l in row_labels))
    industries = list(dict.fromkeys(l.split("_", 1)[1] for l in row_labels))
    expected = [f"{c}_{k}" for c in countries for k in industries]
    if list(row_labels) != expected:
        raise ValueError("Z rows are not country-major ordered (expected CCC_ind with countries outermost)")
    return countries, industries


def load_icio_rdata(data_dir, stem: str = "ICIO2025econ", files=("Z", "X", "FD")):
    """Read the ICIO .Rdata objects (requires pyreadr). Returns dict of xarray objects."""
    import pyreadr  # local import so the rest of the package works without it

    data = {}
    for f in files:
        res = pyreadr.read_r(str(Path(data_dir) / f"{stem}{f}.Rdata"))
        data[f] = next(iter(res.values()))
    return data


def icio_to_numpy(data):
    """Convert the xarray objects produced by ``load_icio_rdata`` to numpy blocks.

    Returns dict with Z5 (t,i,k,j,m), FD4 (t,i,k,j), X3 (t,i,k) and the label lists.
    """
    Z, X, FD = data["Z"], data["X"], data["FD"]
    years = list(np.asarray(Z.coords["dim_0"].values, dtype=object))
    row_labels = list(np.asarray(Z.coords["dim_1"].values, dtype=object))
    countries, industries = split_labels(row_labels)
    if list(np.asarray(Z.coords["dim_2"].values, dtype=object)) != row_labels:
        raise ValueError("Z columns do not match Z rows")
    if list(np.asarray(FD.coords["dim_2"].values, dtype=object)) != countries:
        raise ValueError("FD columns are not one per country in the same order as Z")
    nT, nC, nK = len(years), len(countries), len(industries)
    return {
        "Z5": np.asarray(Z.values).reshape(nT, nC, nK, nC, nK),
        "FD4": np.asarray(FD.values).reshape(nT, nC, nK, nC),
        "X3": np.asarray(X.to_numpy()).reshape(nT, nC, nK),
        "years": years, "countries": countries, "industries": industries,
    }


def sector_flows(Z5, FD4, X3, include_final_demand: bool = True):
    """Cai (2023) sector-level quantities.

    Returns F (t,i,k,j), Y (t,i,k), E (t,k,j) and z = F / (Y E).
    Domestic (i == j) flows are retained.
    """
    F = Z5.sum(axis=4)
    if include_final_demand:
        F = F + FD4
    Y = X3
    E = Z5.sum(axis=(1, 4)) + FD4.sum(axis=1)  # (t,k,j)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = F / (Y[:, :, :, None] * E[:, None, :, :])
    return {"F": F, "Y": Y, "E": E, "z": z}


def pair_flow(Z5, FD4, X3, k: int, m: int | None):
    """Supplier-k x user-m flows F_{ij,t} (t,i,j); m=None gives final demand as user.

    Also returns the normalised z_{ij,t} = F / (Y_{ik,t} E_{jm,t}) where
    E_{jm,t} = sum_i F_{ij,t} is user m's total purchases of k in country j.
    """
    F = FD4[:, :, k, :] if m is None else Z5[:, :, k, :, m]
    Y = X3[:, :, k]                      # (t,i)
    E = F.sum(axis=1)                    # (t,j)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = F / (Y[:, :, None] * E[:, None, :])
    return F, z


def prepare_ICIO_data(data, include_final_demand: bool = True):
    """Backwards-compatible wrapper around the original function: returns xarray objects."""
    import xarray as xr

    b = icio_to_numpy(data)
    s = sector_flows(b["Z5"], b["FD4"], b["X3"], include_final_demand=include_final_demand)
    dims = ("year", "origin", "industry", "destination")
    coords = {"year": b["years"], "origin": b["countries"], "industry": b["industries"],
              "destination": b["countries"]}
    return {
        "F": xr.DataArray(s["F"], dims=dims, coords=coords),
        "Y": xr.DataArray(s["Y"], dims=dims[:3], coords={d: coords[d] for d in dims[:3]}),
        "E": xr.DataArray(s["E"], dims=("year", "industry", "destination"),
                          coords={d: coords[d] for d in ("year", "industry", "destination")}),
        "z": xr.DataArray(s["z"], dims=dims, coords=coords),
    }
