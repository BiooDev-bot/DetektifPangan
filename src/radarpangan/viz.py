"""Helper visualisasi (matplotlib) — peta provinsi & peta rambatan, siap untuk proposal/poster."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

PALETTE = {
    "ink": "#16202a", "muted": "#5b6875", "line": "#e3e7ec", "accent": "#0f6e5c",
    "low": "#cfe8dc", "watch": "#f4b942", "high": "#d9480f", "spike": "#7a1f5c", "nodata": "#d8dde3",
}
MODEL_COLORS = {"naive": "#9aa7b4", "seasonal": "#b8a07e", "sarima": "#7b8fd1", "logreg": "#5fa8a0", "lgbm": "#d9480f"}


def _rings(geom: dict):
    if geom["type"] == "Polygon":
        yield geom["coordinates"][0]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly[0]


def plot_choropleth(ax, geojson: dict, values: dict, cmap="YlOrRd", vmin=None, vmax=None,
                    colors: dict | None = None, edgecolor="white", nodata=PALETTE["nodata"],
                    colorbar_label: str | None = None, title: str | None = None):
    """values: {province_id: angka}; atau colors: {province_id: warna} untuk kategori."""
    patches, facecolors = [], []
    vals = [v for v in values.values() if v is not None and np.isfinite(v)] if values else []
    vmin = np.nanmin(vals) if vmin is None and vals else vmin
    vmax = np.nanmax(vals) if vmax is None and vals else vmax
    cm = plt.get_cmap(cmap)
    for f in geojson["features"]:
        pid = f["properties"]["province_id"]
        if colors is not None:
            fc = colors.get(pid, nodata)
        else:
            v = values.get(pid)
            fc = nodata if v is None or not np.isfinite(v) else cm((v - vmin) / (vmax - vmin + 1e-12))
        for ring in _rings(f["geometry"]):
            patches.append(Polygon(np.asarray(ring), closed=True))
            facecolors.append(fc)
    pc = PatchCollection(patches, facecolor=facecolors, edgecolor=edgecolor, linewidth=0.4)
    ax.add_collection(pc)
    ax.set_xlim(94.5, 141.5)
    ax.set_ylim(-11.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    if colors is None and colorbar_label:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(colorbar_label)
    return ax


def plot_network_map(ax, geojson: dict, ref, edges, leader_scores=None, max_edges: int = 25,
                     title: str | None = None, color=PALETTE["high"]):
    """Panah dari provinsi pemimpin ke provinsi pengikut (edge signifikan)."""
    plot_choropleth(ax, geojson, {}, colors={}, nodata="#eef1f4")
    pos = {int(r.province_id): (r.lon, r.lat) for r in ref.itertuples()}
    e = edges.copy()
    e["strength"] = -np.log10(e["q_value"].clip(lower=1e-12))
    e = e.sort_values("strength", ascending=False).head(max_edges)
    smax = e["strength"].max() if len(e) else 1
    for r in e.itertuples():
        a, b = pos[int(r.source)], pos[int(r.target)]
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.6 + 2.2 * r.strength / smax,
                                    alpha=0.35 + 0.55 * r.strength / smax, shrinkA=3, shrinkB=4,
                                    connectionstyle="arc3,rad=0.15"))
    if leader_scores is not None and len(leader_scores):
        top = leader_scores[leader_scores["leader_score"] > 0].head(6)
        for r in top.itertuples():
            x, y = pos[int(r.province_id)]
            ax.scatter([x], [y], s=40 + 25 * r.leader_score, color=color, zorder=5, edgecolor="white")
            ax.annotate(ref.set_index("province_id").at[int(r.province_id), "province"], (x, y),
                        xytext=(4, 4), textcoords="offset points", fontsize=8, zorder=6)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return ax


def savefig(fig, path, dpi: int = 160):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
