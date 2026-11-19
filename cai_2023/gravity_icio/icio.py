"""OECD ICIO loading and bilateral flow construction."""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import pyreadr

LOGGER = logging.getLogger(__name__)


def _read_rdata(path, object_name=None):
    result = pyreadr.read_r(str(path))
    if object_name:
        if object_name not in result:
            raise KeyError(f"Object {object_name!r} not found in {path}; found {list(result)}")
        return result[object_name]
    if not result:
        raise ValueError(f"No objects found in {path}")
    return next(iter(result.values()))


def _country_industry(labels, separator="_"):
    parsed = []
    for label in labels:
        if separator not in str(label):
            raise ValueError(f"Cannot parse ICIO row label {label!r} using separator {separator!r}")
        country, industry = str(label).split(separator, 1)
        parsed.append((country, industry))
    return parsed


def load_icio(config):
    paths = config["paths"]
    icfg = config["icio"]
    root = Path(paths["icio_dir"]).expanduser()
    objects = icfg.get("objects", {})
    pattern = icfg.get("file_pattern", "ICIO2025econ{table}.Rdata")
    tables = {key: _read_rdata(root / pattern.format(table=key), objects.get(key))
              for key in ("Z", "X", "FD")}
    return tables


def build_bilateral_flows(tables, config):
    """Return F(i,j,k,t), output Y(i,k,t), absorption E(j,k,t), and normalized z.

    Z is assumed to have dimensions (year, supplying industry/country, purchasing
    industry/country). FD is assumed (year, supplying industry/country, destination).
    The code checks coordinate labels and matrix shape before reshaping.
    """
    Z, X, FD = (tables[k] for k in ("Z", "X", "FD"))
    years = list(Z.index) if isinstance(Z, pd.DataFrame) else list(Z.coords[Z.dims[0]].values)
    zrows = list(Z.columns) if isinstance(Z, pd.DataFrame) else list(Z.coords[Z.dims[1]].values)
    zcols = list(Z.columns) if isinstance(Z, pd.DataFrame) else list(Z.coords[Z.dims[2]].values)
    separator = config["icio"].get("country_industry_separator", "_")
    row_pairs, col_pairs = _country_industry(zrows, separator), _country_industry(zcols, separator)
    if row_pairs != col_pairs:
        raise ValueError("ICIO Z row and column labels differ or are not identically ordered.")
    countries = list(dict.fromkeys(c for c, _ in row_pairs))
    industries = list(dict.fromkeys(k for _, k in row_pairs))
    expected = [(c, k) for c in countries for k in industries]
    if row_pairs != expected:
        raise ValueError("ICIO labels are not country-major with a common industry ordering.")
    nt, nc, nk = len(years), len(countries), len(industries)
    expected_industries = config["icio"].get("expected_industries")
    if expected_industries is not None and nk != int(expected_industries):
        raise ValueError(
            f"Found {nk} ICIO industries; configuration expects {expected_industries}. "
            "Check the row labels and whether aggregate rows are included."
        )
    if not isinstance(X, pd.DataFrame):
        x_years = list(X.coords[X.dims[0]].values)
        if x_years != years:
            raise ValueError("X and Z year coordinates do not match.")
        if len(X.dims) == 2:
            x_labels = list(X.coords[X.dims[1]].values)
            if x_labels != zrows:
                raise ValueError("X country-industry labels do not match Z row labels.")
    if not isinstance(FD, pd.DataFrame):
        fd_years = list(FD.coords[FD.dims[0]].values)
        fd_rows = list(FD.coords[FD.dims[1]].values)
        if fd_years != years or fd_rows != zrows:
            raise ValueError("FD year or country-industry coordinates do not match Z.")
    zvalues = Z.to_numpy() if isinstance(Z, pd.DataFrame) else np.asarray(Z.values)
    if zvalues.shape != (nt, nc * nk, nc * nk):
        raise ValueError(f"Unexpected Z shape {zvalues.shape}; expected {(nt, nc*nk, nc*nk)}")
    Z5 = zvalues.reshape(nt, nc, nk, nc, nk)

    xvalues = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X.values)
    if xvalues.shape == (nt, nc * nk):
        output = xvalues.reshape(nt, nc, nk)
    elif xvalues.shape == (nt, nc, nk):
        output = xvalues
    else:
        raise ValueError(f"Unexpected X shape {xvalues.shape}; expected {(nt,nc*nk)} or {(nt,nc,nk)}")

    fdvalues = FD.to_numpy() if hasattr(FD, "to_numpy") else np.asarray(FD.values)
    # OECD final demand may be category-resolved (country x final-demand category)
    # or already aggregated to destination country. Refuse to infer category layout.
    fdcols = list(FD.columns) if isinstance(FD, pd.DataFrame) else list(FD.coords[FD.dims[2]].values)\n    if fdcols != countries:\n        raise ValueError("FD destination-country coordinates do not match the Z country order.")
    if len(fdcols) == nc:
        if fdvalues.shape != (nt, nc * nk, nc):
            raise ValueError(f"FD shape {fdvalues.shape} inconsistent with destination-country columns")
        FD4 = fdvalues.reshape(nt, nc, nk, nc)
        final_sales = FD4  # (t, origin, industry, destination)
    else:
        raise ValueError(
            "FD columns are not one-per-country. The OECD final-demand category layout must be "
            "mapped to destination countries explicitly; see ICIO schema output."
        )

    intermediate_sales = Z5.sum(axis=4)
    F = intermediate_sales + final_sales
    exclude = set(config["icio"].get("exclude_countries", ["ROW"]))
    keep = [i for i, c in enumerate(countries) if c not in exclude]
    if len(keep) != nc:
        F, output = F[:, keep, :, :][:, :, :, keep], output[:, keep, :]
        countries = [countries[i] for i in keep]
    absorption = F.sum(axis=1)  # (t, industry, destination)
    year_range = config["icio"].get("years")
    if year_range:
        start_year, end_year = year_range
        year_mask = np.array([start_year <= int(y) <= end_year for y in years])
        F, output, absorption = F[year_mask], output[year_mask], absorption[year_mask]
        years = [y for y, keep_year in zip(years, year_mask) if keep_year]
        nt = len(years)
    denom = output[:, :, :, None] * absorption[:, None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(F, denom, out=np.full_like(F, np.nan, dtype=float), where=denom > 0)

    idx = pd.MultiIndex.from_product([years, countries, industries, countries],
                                     names=["year", "origin", "industry", "destination"])
    flow = pd.DataFrame({"F": F.reshape(-1), "z": normalized.reshape(-1)}, index=idx).reset_index()
    out_idx = pd.MultiIndex.from_product([years, countries, industries],
                                         names=["year", "origin", "industry"])
    output_df = pd.DataFrame({"output": output.reshape(-1)}, index=out_idx).reset_index()
    abs_idx = pd.MultiIndex.from_product([years, industries, countries],
                                         names=["year", "industry", "destination"])
    abs_df = pd.DataFrame({"absorption": absorption.reshape(-1)}, index=abs_idx).reset_index()
    return flow, output_df, abs_df
