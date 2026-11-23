"""Distances between sub-national regions, built to match CEPII ``distwces``.

For the ICIO elasticities to transfer, Canadian interprovincial distances should be
constructed the same way as CEPII's distwces (Head & Mayer 2010):

    d_ab = ( sum_{k in a} sum_{l in b} (pop_k / pop_a) (pop_l / pop_b) d_kl^theta )^(1/theta),
    theta = -1,

with k, l populated places (e.g. census subdivisions / population centres) and
d_kl the great-circle distance. Internal distances (a == b) come out of the same
formula with d_kk replaced by an own-place internal distance 0.67 sqrt(area_k / pi).
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def ces_distance(region, lat, lon, pop, area_km2=None, theta: float = -1.0, regions=None,
                 min_internal_km: float = 1.0):
    """Population-weighted CES distance matrix between regions (km).

    region, lat, lon, pop : arrays over places (e.g. census subdivisions).
    area_km2              : place areas; used for place self-distance 0.67 sqrt(area/pi).
    """
    region = np.asarray(region)
    lat, lon, pop = (np.asarray(v, float) for v in (lat, lon, pop))
    regs = list(dict.fromkeys(region)) if regions is None else list(regions)
    D = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    self_d = (0.67 * np.sqrt(np.asarray(area_km2, float) / np.pi) if area_km2 is not None
              else np.full(len(lat), min_internal_km))
    np.fill_diagonal(D, np.maximum(self_d, min_internal_km))
    D = np.maximum(D, min_internal_km)
    out = np.zeros((len(regs), len(regs)))
    for a, ra in enumerate(regs):
        ia = region == ra
        wa = pop[ia] / pop[ia].sum()
        for b, rb in enumerate(regs):
            ib = region == rb
            wb = pop[ib] / pop[ib].sum()
            out[a, b] = (wa @ (D[np.ix_(ia, ib)] ** theta) @ wb) ** (1.0 / theta)
    return out, regs


def internal_distance_area(area_km2):
    """CEPII-style internal distance 0.67 * sqrt(area / pi) (fallback when no point data)."""
    return 0.67 * np.sqrt(np.asarray(area_km2, float) / np.pi)
