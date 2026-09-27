"""DEPLOYMENT — ekspor dashboard statis (HTML + JSON) ke folder site/.

Folder site/ bisa langsung di-hosting gratis (GitHub Pages / Vercel / Netlify) dan
dijadikan QR code poster. Tidak perlu server: halaman membaca data/*.json.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES_HTML = (
    'Sumber data: <a href="https://www.bi.go.id/hargapangan" target="_blank" rel="noopener">PIHPS Nasional – Bank Indonesia</a>; '
    'curah hujan <a href="https://power.larc.nasa.gov" target="_blank" rel="noopener">NASA POWER</a>; '
    'batas provinsi © OpenStreetMap contributors via geoBoundaries (ODbL); '
    'tanggal hari besar: hasil sidang isbat/SKB pemerintah. '
    'Prediksi bersifat probabilistik dan dapat keliru.'
)


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_clean), encoding="utf-8")


def export_site(cfg, P, risk: pd.DataFrame, model_meta: dict, refs: dict, panels: dict,
                events: pd.DataFrame, edges: pd.DataFrame, leader_scores: pd.DataFrame,
                leaders: dict, eval_summary: dict | None = None, default_commodity: str = "com_16",
                history_days: int | None = None, max_edges: int = 30) -> Path:
    site = P.site
    template = P.root / "deployment" / "site_template"
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(template, site)
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(P.external / "indonesia_provinces.geojson", data_dir / "provinces.geojson")

    asof = pd.Timestamp(risk["asof"].iloc[0])
    h = cfg["target"]["horizon_days"]
    thr_by_com = {}
    for c in refs["com_name"]:
        cat = refs["c2cat"][c]
        over = cfg["target"].get("threshold_overrides") or {}
        thr_by_com[c] = over.get(c, over.get(cat, cfg["target"]["default_threshold"]))
    commodities = [{"id": c, "name": refs["com_name"][c], "cat_id": refs["c2cat"][c],
                    "category": refs["cat_name"][refs["c2cat"][c]], "threshold": thr_by_com[c]}
                   for c in sorted(refs["com_name"], key=lambda x: int(x.split("_")[1]))
                   if c in set(risk["commodity_id"])]
    ref = refs["prov_ref"]
    metrics = {}
    if eval_summary:
        metrics = {k: eval_summary.get(k) for k in
                   ["pr_auc_lgbm", "pr_auc_naive", "event_detection_rate", "median_lead_days",
                    "far", "false_episodes_per_series_year"]}
    meta = {
        "asof": f"{asof:%Y-%m-%d}",
        "window": {"start": f"{asof + pd.Timedelta(days=1):%Y-%m-%d}", "end": f"{asof + pd.Timedelta(days=h):%Y-%m-%d}"},
        "horizon_days": h,
        "alarm_threshold": model_meta["alarm_threshold"],
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "commodities": commodities,
        "default_commodity": default_commodity,
        "metrics": metrics,
        "centroids": [{"p": int(r.province_id), "lat": r.lat, "lon": r.lon} for r in ref.itertuples()],
        "sources": SOURCES_HTML,
    }
    _dump(meta, data_dir / "meta.json")

    # harga terkini per seri untuk ditampilkan
    Pc = panels[cfg["target"]["price_type"]]
    last_price = Pc.loc[:asof].ffill().iloc[-1]
    rows = []
    for r in risk.itertuples(index=False):
        reasons = [x["text"] for x in json.loads(r.reasons_json)] if r.level != "Sedang lonjak" else []
        rows.append({"c": r.commodity_id, "p": int(r.province_id), "prob": round(float(r.prob), 4),
                     "level": r.level, "r14": _clean(r.r14_now), "ret7": _clean(r.ret_7),
                     "price": _clean(last_price.get((r.commodity_id, int(r.province_id)), np.nan)),
                     "reasons": reasons})
    _dump(rows, data_dir / "risk_latest.json")

    # riwayat harga per komoditas
    days = history_days or cfg["deployment"]["history_days"]
    start = asof - pd.Timedelta(days=days)
    Pp = panels.get(4)
    wk = Pc.loc[start:asof]
    wk = wk[wk.index.dayofweek < 5]
    dates = [f"{d:%Y-%m-%d}" for d in wk.index]
    ev = events[(events["date"] >= start) & (events["date"] <= asof)]
    for c in {x["id"] for x in commodities}:
        out = {"dates": dates, "consumer": {}, "producer": {}, "onsets": {}}
        for p in ref["province_id"]:
            col = (c, int(p))
            if col in wk.columns:
                out["consumer"][str(p)] = [None if not np.isfinite(v) else round(float(v)) for v in wk[col].to_numpy()]
            if Pp is not None and col in Pp.columns:
                pv = Pp.loc[wk.index, col].to_numpy()
                if np.isfinite(pv).any():
                    out["producer"][str(p)] = [None if not np.isfinite(v) else round(float(v)) for v in pv]
        if (c, 0) in wk.columns:
            out["national"] = [None if not np.isfinite(v) else round(float(v)) for v in wk[(c, 0)].to_numpy()]
        for p, g in ev[ev["commodity_id"] == c].groupby("province_id"):
            out["onsets"][str(int(p))] = [f"{d:%Y-%m-%d}" for d in g["date"]]
        _dump(out, data_dir / "history" / f"{c}.json")

    # jaringan rambatan
    net = {}
    e = edges[edges["significant"]].copy()
    e["strength"] = -np.log10(e["q_value"].clip(lower=1e-12))
    for c in {x["id"] for x in commodities}:
        ec = e[e["commodity_id"] == c].sort_values("strength", ascending=False).head(max_edges)
        ls = leader_scores[leader_scores["commodity_id"] == c].sort_values("leader_score", ascending=False)
        lo = {}
        for (cc, tgt), srcs in leaders.items():
            if cc == c:
                lo[str(tgt)] = srcs
        net[c] = {
            "edges": [[int(r.source), int(r.target), round(float(r.strength), 3),
                       _clean(getattr(r, "lift", np.nan))] for r in ec.itertuples()],
            "leaders": [[int(r.province_id), int(r.leader_score), int(r.out_degree), int(r.in_degree)]
                        for r in ls.itertuples()],
            "leaders_of": lo,
        }
    _dump(net, data_dir / "network.json")
    return site
