"""Merge CEPII bilateral controls and agreement/tariff data."""
import numpy as np
import pandas as pd


def _read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".xls":
        return pd.read_excel(path, engine="xlrd")
    return pd.read_excel(path, engine="openpyxl")


def merge_controls(flow, config, paths, agreements=None):
    c = config["cepii"]
    distance = _read_table(paths["distance"])
    geography = _read_table(paths["geography"])
    origin, dest = c.get("origin_variable", "iso_o"), c.get("destination_variable", "iso_d")
    for name, frame in (("distance", distance), ("geography", geography)):
        if origin not in frame or dest not in frame:
            raise KeyError(f"{name} file must contain {origin!r} and {dest!r}; found {list(frame.columns)}")
    for label, frame in (("distance", distance), ("geography", geography)):
        duplicated = frame.duplicated([origin, dest], keep=False)
        if duplicated.any():
            examples = frame.loc[duplicated, [origin, dest]].head(5).to_dict("records")
            raise ValueError(f"CEPII {label} file has duplicate country pairs; examples: {examples}")
    dcol = c.get("distance_variable", "distwces")
    if dcol not in distance:
        raise KeyError(f"Distance variable {dcol!r} missing from CEPII distance file.")
    distance = distance[[origin, dest, dcol]].rename(columns={origin:"origin", dest:"destination", dcol:"distance"})
    geography = geography.rename(columns={origin:"origin", dest:"destination"})
    controls = distance.merge(geography, on=["origin","destination"], how="outer", suffixes=("", "_geo"), validate="one_to_one")
    controls["origin"] = controls["origin"].astype(str).str.upper()
    controls["destination"] = controls["destination"].astype(str).str.upper()
    panel = flow.merge(controls, on=["origin","destination"], how="left", indicator="_cepii_merge", validate="many_to_one")
    lang = c.get("language_variable", "comlang_off")
    if lang in panel:
        panel["language"] = pd.to_numeric(panel[lang], errors="coerce")
    else:
        panel["language"] = np.nan
    panel["border"] = (panel["origin"] != panel["destination"]).astype(int)
    panel["log_distance"] = np.where(panel["distance"] > 0, np.log(panel["distance"]), np.nan)
    if agreements is not None:
        agreement_keys = ["origin", "destination", "year"]
        if "industry" in agreements.columns:
            agreement_keys.append("industry")
        panel = panel.merge(agreements, on=agreement_keys, how="left", indicator="_agreement_merge", validate="many_to_one")
    return panel
