"""Torój's home-region radius computed from real boundaries (no GIS libraries required).

Torój (2024, Sect. 3): "Within-region supplies are interpreted as up to 51.2 km. This value has
been derived as an average of all the distances from the regional capitals to the region's
border, measured on the straight lines to the capitals of the neighboring regions." (Polish
NUTS-3 regions.) ``toroj_r_home`` reproduces that definition for any set of regions:

  for every region r and every neighbour q of r:
      draw the straight segment from capital(r) to capital(q) (in a conformal map projection);
      record the distance from capital(r) to the first crossing of r's boundary;
  r_home = mean of all recorded distances (pooled over all region-neighbour pairs).

Inputs: province polygons from Statistics Canada's boundary files (shapefile, EPSG:3347
"NAD83 / Statistics Canada Lambert") or any GeoJSON in lon/lat. Pure-numpy readers are
included; coordinates of capitals are projected with the EPSG:3347 Lambert Conformal Conic.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# EPSG:3347 NAD83 / Statistics Canada Lambert (LCC 2SP, GRS80)
_A, _F = 6378137.0, 1 / 298.257222101
_E = np.sqrt(2 * _F - _F ** 2)
_LAT1, _LAT2, _LAT0, _LON0 = np.radians([49.0, 77.0, 63.390675, -91.86666666666666])
_X0, _Y0 = 6200000.0, 3000000.0


def _m(phi):
    return np.cos(phi) / np.sqrt(1 - _E ** 2 * np.sin(phi) ** 2)


def _t(phi):
    es = _E * np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - es) / (1 + es)) ** (_E / 2)


_N = (np.log(_m(_LAT1)) - np.log(_m(_LAT2))) / (np.log(_t(_LAT1)) - np.log(_t(_LAT2)))
_FF = _m(_LAT1) / (_N * _t(_LAT1) ** _N)
_RHO0 = _A * _FF * _t(_LAT0) ** _N


def lcc3347(lat, lon):
    """Forward EPSG:3347 projection (Snyder 1987, eqs. 15-1 to 15-10). Returns metres (x, y)."""
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    rho = _A * _FF * _t(phi) ** _N
    th = _N * (lam - _LON0)
    return _X0 + rho * np.sin(th), _Y0 + _RHO0 - rho * np.cos(th)


# ----------------------------------------------------------------------------- readers
def read_shapefile(shp_path, name_field="PRUID"):
    """Minimal ESRI shapefile reader for (Multi)Polygon layers (shape types 5, 15, 25).

    Returns {name: [ring arrays (n,2)]} in the file's coordinates; names from the .dbf.
    """
    shp = Path(shp_path)
    data = shp.read_bytes()
    rings_by_rec = []
    pos = 100
    while pos < len(data):
        _, clen = struct.unpack(">2i", data[pos:pos + 8])
        rec = data[pos + 8:pos + 8 + 2 * clen]
        pos += 8 + 2 * clen
        stype = struct.unpack("<i", rec[:4])[0]
        if stype == 0:
            rings_by_rec.append([])
            continue
        if stype not in (5, 15, 25):
            raise ValueError(f"shape type {stype} is not a polygon")
        nparts, npts = struct.unpack("<2i", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts])) + [npts]
        off = 44 + 4 * nparts
        pts = np.frombuffer(rec[off:off + 16 * npts], dtype="<f8").reshape(npts, 2)
        rings_by_rec.append([pts[parts[k]:parts[k + 1]].copy() for k in range(nparts)])
    names = _read_dbf_field(shp.with_suffix(".dbf"), name_field)
    out = {}
    for nm, rings in zip(names, rings_by_rec):
        out.setdefault(nm, []).extend(rings)
    return out


def _read_dbf_field(dbf_path, field):
    b = Path(dbf_path).read_bytes()
    nrec, hlen, rlen = struct.unpack("<IHH", b[4:12])
    fields, p, off = [], 32, 1
    while b[p] != 0x0D:
        nm = b[p:p + 11].split(b"\x00")[0].decode("latin-1")
        ln = b[p + 16]
        fields.append((nm, off, ln))
        off += ln
        p += 32
    f = next((x for x in fields if x[0].upper() == field.upper()), None)
    if f is None:
        raise KeyError(f"{field} not in dbf fields {[x[0] for x in fields]}")
    return [b[hlen + i * rlen + f[1]: hlen + i * rlen + f[1] + f[2]].decode("latin-1").strip()
            for i in range(nrec)]


def read_geojson(path, name_field):
    """GeoJSON (lon/lat) -> {name: [rings]} projected to EPSG:3347 metres."""
    gj = json.loads(Path(path).read_text())
    out = {}
    for ft in gj["features"]:
        nm = str(ft["properties"][name_field])
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                r = np.asarray(ring, float)
                x, y = lcc3347(r[:, 1], r[:, 0])
                out.setdefault(nm, []).append(np.column_stack([x, y]))
    return out


# --------------------------------------------------------------------------- geometry
def first_boundary_crossing(p0, p1, rings):
    """Distance from p0 to the first crossing of any ring edge along segment p0->p1 (None if none)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    r = p1 - p0
    best = np.inf
    for ring in rings:
        a = ring[:-1]
        s = ring[1:] - ring[:-1]
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        ok = np.abs(den) > 1e-12
        qp = a - p0
        t = np.where(ok, (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / np.where(ok, den, 1), np.inf)
        u = np.where(ok, (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / np.where(ok, den, 1), np.inf)
        hit = ok & (t > 1e-9) & (t <= 1) & (u >= 0) & (u <= 1)
        if hit.any():
            best = min(best, float(t[hit].min()))
    return None if not np.isfinite(best) else best * float(np.hypot(*r))


def toroj_r_home(polygons, capitals, neighbours, regions=None):
    """Torój's within-region radius (km) and the per-pair table.

    polygons   : {region: [rings in EPSG:3347 metres]}
    capitals   : {region: (lat, lon)}
    neighbours : {region: [neighbouring regions]}
    """
    regions = list(regions or capitals)
    rows = []
    for r in regions:
        if r not in polygons:
            continue
        c0 = np.array(lcc3347(*capitals[r]))
        for q in neighbours.get(r, []):
            if q not in regions:
                continue
            c1 = np.array(lcc3347(*capitals[q]))
            d = first_boundary_crossing(c0, c1, polygons[r])
            rows.append((r, q, None if d is None else d / 1000.0))
    vals = [d for _, _, d in rows if d is not None]
    return float(np.mean(vals)), rows


def neighbours_from_polygons(polygons, tol_m=2000.0):
    """Regions whose boundaries come within tol_m of each other (land adjacency)."""
    from sklearn.neighbors import NearestNeighbors
    names = list(polygons)
    pts = {n: np.vstack(polygons[n]) for n in names}
    out = {n: [] for n in names}
    for i, a in enumerate(names):
        nn = NearestNeighbors(n_neighbors=1).fit(pts[a])
        for b in names[i + 1:]:
            d, _ = nn.kneighbors(pts[b][:: max(1, len(pts[b]) // 20000)])
            if d.min() <= tol_m:
                out[a].append(b)
                out[b].append(a)
    return out
