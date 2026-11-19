"""Chunked loading and pair-year or pair-year-industry aggregation of RTA/tariff data."""
import logging

import pandas as pd
import pyreadstat

LOGGER = logging.getLogger(__name__)


def load_agreements(path, settings, chunk_size=500_000):
    required = {
        key: settings.get(f"{key}_variable")
        for key in ("origin", "destination", "year", "industry", "rta", "tariff")
    }
    selected = [value for value in required.values() if value]
    if not all(required[key] for key in ("origin", "destination", "year")):
        raise ValueError(
            "Configure origin_variable, destination_variable, and year_variable after schema inspection."
        )
    if not (required["rta"] or required["tariff"]):
        raise ValueError("Configure at least one RTA or tariff variable after inspecting the Stata schema.")

    chunks = []
    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta, str(path), chunksize=chunk_size, usecols=selected
    )
    for chunk, _meta in reader:
        rename = {value: key for key, value in required.items() if value}
        chunk = chunk.rename(columns=rename)
        for column in ("origin", "destination"):
            chunk[column] = chunk[column].astype("string").str.upper().str.strip()
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce").astype("Int64")
        if "industry" in chunk:
            chunk["industry"] = chunk["industry"].astype("string").str.strip()
        chunks.append(chunk)

    if not chunks:
        raise ValueError("No observations read from the Stata file.")
    raw = pd.concat(chunks, ignore_index=True)

    if "rta" in raw:
        numeric_rta = pd.to_numeric(raw["rta"], errors="coerce")
        if numeric_rta.notna().sum() == raw["rta"].notna().sum():
            raw["rta"] = numeric_rta
        else:
            labels = raw["rta"].astype("string").str.strip().str.lower()
            mapping = {
                "1": 1, "true": 1, "yes": 1, "y": 1, "in force": 1,
                "0": 0, "false": 0, "no": 0, "n": 0, "not in force": 0,
            }
            raw["rta"] = labels.map(mapping)
            if raw["rta"].isna().any() and labels.notna().any():
                unknown = sorted(labels[raw["rta"].isna()].dropna().unique().tolist())
                raise ValueError(
                    f"Unrecognized RTA indicator labels {unknown}; recode the variable explicitly."
                )
    if "tariff" in raw:
        raw["tariff"] = pd.to_numeric(raw["tariff"], errors="coerce")

    aggregations = {}
    if "rta" in raw:
        aggregations["rta"] = settings.get("rta_aggregation", "max")
    if "tariff" in raw:
        aggregations["tariff"] = settings.get("tariff_aggregation", "mean")
    group_columns = ["origin", "destination", "year"]
    if "industry" in raw:
        group_columns.append("industry")
    pair = raw.groupby(group_columns, dropna=False).agg(aggregations).reset_index()
    pair["pair"] = pair["origin"] + "_" + pair["destination"]
    return pair
