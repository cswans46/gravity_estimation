"""Tidy-CSV loaders for the Canadian inputs -> dense arrays.

The Statistics Canada tables come in several layouts; rather than hard-code one, the
pipeline expects the tidy long files below (one value column). See docs/METHODS.md
section 5 for the suggested source tables and the required pre-processing.

interprovincial_flows.csv   origin, destination, commodity, value   (origin == destination
                            is own-province supply of domestic origin)
provincial_use.csv          province, commodity, user, value         (users = industries,
                            then final-demand categories; all origins incl. imports)
international_imports.csv   province, commodity, value               (optional)
province_distances.csv      origin, destination, km                   (diag = internal dist.)
concordance_supply.csv      commodity, icio_industry, weight
concordance_user.csv        user, icio_user, weight                   (icio_user may be 'FD')
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_array(df: pd.DataFrame, dims: list[str], value: str = "value", labels: dict | None = None):
    labels = dict(labels or {})
    for d in dims:
        labels.setdefault(d, list(dict.fromkeys(df[d])))
    arr = np.zeros(tuple(len(labels[d]) for d in dims))
    idx = tuple(pd.Categorical(df[d], categories=labels[d]).codes for d in dims)
    ok = np.all([i >= 0 for i in idx], axis=0)
    np.add.at(arr, tuple(i[ok] for i in idx), df[value].to_numpy(float)[ok])
    return arr, labels


def concordance_matrix(df: pd.DataFrame, src: str, dst: str, src_labels, dst_labels, w: str = "weight"):
    M, _ = to_array(df, [src, dst], w, {src: list(src_labels), dst: list(dst_labels)})
    rs = M.sum(axis=1, keepdims=True)
    if np.any(rs == 0):
        missing = [s for s, r in zip(src_labels, rs[:, 0]) if r == 0]
        raise ValueError(f"concordance has no mapping for: {missing[:10]}")
    return M / rs


def load_canada(folder, provinces=None):
    from pathlib import Path
    f = Path(folder)
    flows = pd.read_csv(f / "interprovincial_flows.csv")
    use = pd.read_csv(f / "provincial_use.csv")
    provinces = provinces or list(dict.fromkeys(flows["origin"]))
    commodities = list(dict.fromkeys(use["commodity"]))
    users = list(dict.fromkeys(use["user"]))
    T, _ = to_array(flows, ["origin", "commodity", "destination"],
                    labels={"origin": provinces, "destination": provinces, "commodity": commodities})
    U, _ = to_array(use, ["commodity", "province", "user"],
                    labels={"commodity": commodities, "province": provinces, "user": users})
    M = None
    if (f / "international_imports.csv").exists():
        M, _ = to_array(pd.read_csv(f / "international_imports.csv"), ["commodity", "province"],
                        labels={"commodity": commodities, "province": provinces})
    d, _ = to_array(pd.read_csv(f / "province_distances.csv"), ["origin", "destination"], "km",
                    labels={"origin": provinces, "destination": provinces})
    return dict(T=T, U=U, M=M, dist=d, provinces=provinces, commodities=commodities, users=users)
