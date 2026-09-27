"""Peta batas provinsi untuk dashboard.

Sumber: geoBoundaries gbOpen IDN ADM1 (34 provinsi, © OpenStreetMap contributors),
lisensi ODbL 1.0. File hasil olahan sudah disertakan di data/external/indonesia_provinces.geojson;
fungsi di bawah hanya diperlukan kalau ingin membuat ulang.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

GEOBOUNDARIES_URL = (
    "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/releaseData/"
    "gbOpen/IDN/ADM1/geoBoundaries-IDN-ADM1_simplified.geojson"
)
ISO_TO_PIHPS = {
    "ID-AC": 1, "ID-SU": 2, "ID-SB": 3, "ID-RI": 4, "ID-KR": 5, "ID-JA": 6, "ID-BE": 7, "ID-SS": 8,
    "ID-BB": 9, "ID-LA": 10, "ID-BT": 11, "ID-JB": 12, "ID-JK": 13, "ID-JT": 14, "ID-YO": 15,
    "ID-JI": 16, "ID-BA": 17, "ID-NB": 18, "ID-NT": 19, "ID-KB": 20, "ID-KS": 21, "ID-KT": 22,
    "ID-KI": 23, "ID-KU": 24, "ID-GO": 25, "ID-SN": 26, "ID-SG": 27, "ID-ST": 28, "ID-SA": 29,
    "ID-SR": 30, "ID-MA": 31, "ID-MU": 32, "ID-PA": 33, "ID-PB": 34,
}


def _round(o, nd=3):
    if isinstance(o, (list, tuple)):
        return [_round(x, nd) for x in o]
    return round(o, nd)


def prepare_province_geojson(out_path: Path, tolerance: float = 0.01, min_area: float = 0.002) -> Path:
    """Unduh geoBoundaries, sederhanakan geometri, pasang province_id PIHPS."""
    from shapely.geometry import MultiPolygon, mapping, shape

    from .reference import province_reference

    names = province_reference().set_index("province_id")["province"].to_dict()
    g = requests.get(GEOBOUNDARIES_URL, timeout=120).json()
    feats = []
    for f in g["features"]:
        iso = f["properties"]["shapeISO"]
        pid = ISO_TO_PIHPS[iso]
        geom = shape(f["geometry"]).simplify(tolerance, preserve_topology=True)
        if geom.geom_type == "MultiPolygon":
            parts = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
            keep = [p for p in parts if p.area > min_area] or parts[:1]
            geom = MultiPolygon(keep) if len(keep) > 1 else keep[0]
        gm = mapping(geom)
        feats.append({"type": "Feature", "id": pid,
                      "properties": {"province_id": pid, "province": names[pid], "iso": iso},
                      "geometry": {"type": gm["type"], "coordinates": _round(gm["coordinates"])}})
    out = {"type": "FeatureCollection",
           "metadata": {"source": "geoBoundaries gbOpen IDN ADM1", "license": "ODbL 1.0",
                        "attribution": "© OpenStreetMap contributors; geoBoundaries"},
           "features": sorted(feats, key=lambda x: x["id"])}
    out = orient_for_d3(out)
    out_path = Path(out_path)
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out_path


def orient_for_d3(geojson: dict) -> dict:
    """d3-geo/Plotly butuh ring luar SEARAH jarum jam (kebalikan RFC 7946).
    Kalau terbalik, poligon dianggap "seluruh dunia kecuali provinsi" -> peta jadi kotak penuh warna."""
    from shapely.geometry import MultiPolygon, Polygon, mapping, shape
    from shapely.geometry.polygon import orient

    for f in geojson["features"]:
        geom = shape(f["geometry"])
        if isinstance(geom, Polygon):
            geom = orient(geom, sign=-1.0)
        elif isinstance(geom, MultiPolygon):
            geom = MultiPolygon([orient(g, sign=-1.0) for g in geom.geoms])
        gm = mapping(geom)
        f["geometry"] = {"type": gm["type"], "coordinates": _round(gm["coordinates"])}
    return geojson


def load_geojson(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
