"""Chunked loading and pair-year aggregation of RTA/tariff data."""
import logging
import pandas as pd
import pyreadstat

LOGGER = logging.getLogger(__name__)


def load_agreements(path, settings, chunk_size=500_000):
    required = {key: settings.get(f"{key}_variable") for key in
                ("origin", "destination", "year", "rta", "tariff")}
    selected = [v for v in required.values() if v]
    if not all(required[k] for k in ("origin", "destination", "year")):
        raise ValueError("Configure origin_variable, destination_variable, and year_variable after schema inspection.")
    if not selected:
        raise ValueError("No agreement columns configured.")
    chunks = []
    reader = pyreadstat.read_file_in_chunks(pyreadstat.read_dta, str(path),
                                             chunksize=chunk_size, usecols=selected)
    for chunk, _meta in reader:
        rename = {v: k for k, v in required.items() if v}
        chunk = chunk.rename(columns=rename)
        for col in ("origin", "destination"):
            chunk[col] = chunk[col].astype("string").str.upper().str.strip()
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce").astype("Int64")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("No observations read from the Stata file.")
    raw = pd.concat(chunks, ignore_index=True)
    aggregations = {}
    if "rta" in raw:
        aggregations["rta"] = settings.get("rta_aggregation", "max")
    if "tariff" in raw:
        aggregations["tariff"] = settings.get("tariff_aggregation", "mean")
    pair = raw.groupby(["origin", "destination", "year"], dropna=False).agg(aggregations).reset_index()
    pair["pair"] = pair["origin"] + "_" + pair["destination"]
    return pair
