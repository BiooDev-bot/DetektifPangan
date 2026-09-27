"""Orkestrasi langkah-langkah pipeline (dipakai notebook & run_pipeline.py).

Notebook memanggil fungsi tingkat-rendah satu per satu supaya prosesnya terlihat;
modul ini merangkai fungsi yang sama untuk otomatisasi (update harian).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import build_price_panels
from .features import CATEGORICAL, ID_COLS, LABEL_COLS, build_features
from .labeling import events_table, make_labels, threshold_per_column
from .network import build_network, select_leaders
from .reference import calendar_features, holiday_table, nearest_neighbors, province_reference
from .scraper import PIHPSClient, load_raw_cache, save_reference_tables, scrape_pihps
from .weather import fetch_all_provinces, rainfall_anomalies

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Referensi
# -----------------------------------------------------------------------------
def load_refs(P) -> dict:
    provs = pd.read_csv(P.external / "pihps_provinces.csv")
    coms = pd.read_csv(P.external / "pihps_commodities.csv")
    coms["name"] = coms["name"].str.strip()
    variants = coms[coms["id"].str.startswith("com_")]
    cats = coms[coms["id"].str.startswith("cat_")]
    return {
        "provinces": provs,
        "commodities": coms,
        "c2cat": dict(zip(variants["id"], variants["cat_id"])),
        "com_name": dict(zip(variants["id"], variants["name"])),
        "cat_name": dict(zip(cats["id"], cats["name"])),
        "prov_ref": province_reference(),
        "prov_name": dict(zip(province_reference()["province_id"], province_reference()["province"])),
    }


def client_from_cfg(cfg) -> PIHPSClient:
    sc = cfg["scraping"]
    return PIHPSClient(sc["base_url"], sc["timeout_seconds"], sc["max_retries"], sc["sleep_seconds"])


# -----------------------------------------------------------------------------
# DATA UNDERSTANDING — pengumpulan
# -----------------------------------------------------------------------------
def collect(cfg, P, price_types=None, commodities=None, start=None, end=None, workers=None,
            refresh_recent_months=None, weather=True) -> pd.DataFrame:
    sc = cfg["scraping"]
    client = client_from_cfg(cfg)
    refs = save_reference_tables(client, P.external)
    all_coms = [c for c in refs["pihps_commodities"]["id"] if c.startswith("com_")]
    coms = commodities or (all_coms if sc["commodities"] == "all" else sc["commodities"])
    summary = scrape_pihps(
        client, P.raw_pihps, price_types or sc["price_types"], coms,
        start or sc["start_date"], end or sc["end_date"],
        workers=workers or sc["workers"],
        refresh_recent_months=sc["refresh_recent_months"] if refresh_recent_months is None else refresh_recent_months,
    )
    build_long(P)
    if weather:
        update_weather(cfg, P)
    return summary


def build_long(P) -> pd.DataFrame:
    refs = load_refs(P)
    long = load_raw_cache(P.raw_pihps, refs["provinces"], refs["commodities"])
    long.to_parquet(P.interim / "pihps_long.parquet", index=False)
    return long


def update_weather(cfg, P, end=None) -> pd.DataFrame:
    w = cfg["weather"]
    pr = fetch_all_provinces(province_reference(), P.raw_weather, w["climatology_start"], end)
    pr.to_parquet(P.interim / "rainfall_daily.parquet", index=False)
    return pr


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------
def save_panel(panel: pd.DataFrame, path: Path) -> None:
    flat = panel.copy()
    flat.columns = [f"{c}__{p}" for c, p in flat.columns]
    flat.index.name = "date"
    flat.to_parquet(path)


def load_panel(path: Path) -> pd.DataFrame:
    flat = pd.read_parquet(path)
    flat.columns = pd.MultiIndex.from_tuples([(c.split("__")[0], int(c.split("__")[1])) for c in flat.columns],
                                             names=["commodity_id", "province_id"])
    return flat


def prepare(cfg, P, long: pd.DataFrame | None = None, leaders_mode: str = "evaluation",
            save: bool = True) -> dict:
    """Panel bersih -> label -> jaringan lead-lag -> fitur.

    leaders_mode:
      "evaluation": provinsi pemimpin dipelajari HANYA dari data sebelum periode uji pertama
                    (dipakai untuk walk-forward, bebas leakage)
      "final"     : dipelajari dari seluruh data (dipakai model produksi/dashboard)
    """
    refs = load_refs(P)
    long = pd.read_parquet(P.interim / "pihps_long.parquet") if long is None else long
    panels, clean_log = build_price_panels(long, cfg)
    Pc = panels[cfg["target"]["price_type"]]
    thr = threshold_per_column(Pc.columns, refs["c2cat"], cfg)
    labels = make_labels(Pc, thr, cfg)
    events = events_table(labels, Pc)
    events = events[events["province_id"] != 0].reset_index(drop=True)

    first_test = pd.Timestamp(cfg["validation"]["first_test_start"])
    leaders_end = (first_test - pd.Timedelta(days=cfg["validation"]["purge_days"] + 1)
                   if leaders_mode == "evaluation" else None)
    edges, lscores = build_network(Pc, labels["onset"], p=cfg["features"]["leader_lag_weeks"], end=leaders_end)
    leaders = select_leaders(edges, cfg["features"]["n_leaders"])

    cal = calendar_features(Pc.index, holiday_table(Pc.index.min().year - 1, Pc.index.max().year + 2),
                            cfg["features"]["calendar_cap_days"])
    rain = None
    rp = P.interim / "rainfall_daily.parquet"
    if rp.exists():
        w = cfg["weather"]
        rain = rainfall_anomalies(pd.read_parquet(rp), w["climatology_start"], w["climatology_end"])
    feats, groups = build_features(panels, labels, thr, cal, rain, nearest_neighbors(cfg["features"]["n_neighbors"]),
                                   leaders, refs["c2cat"], cfg, province_names=refs["prov_name"],
                                   weekdays_only=cfg["validation"]["weekdays_only"])
    out = {"panels": panels, "clean_log": clean_log, "thresholds": thr, "labels": labels, "events": events,
           "edges": edges, "leader_scores": lscores, "leaders": leaders, "calendar": cal, "rain": rain,
           "features": feats, "groups": groups}
    if save:
        suffix = "" if leaders_mode == "evaluation" else "_final"
        for pt, pan in panels.items():
            save_panel(pan, P.processed / f"panel_pt{pt}.parquet")
        events.to_parquet(P.processed / "events.parquet", index=False)
        edges.to_parquet(P.processed / f"network_edges{suffix}.parquet", index=False)
        lscores.to_parquet(P.processed / f"leader_scores{suffix}.parquet", index=False)
        feats.to_parquet(P.processed / f"features{suffix}.parquet", index=False)
        (P.processed / "feature_groups.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")
        (P.processed / f"leaders{suffix}.json").write_text(
            json.dumps({f"{c}|{p}": v for (c, p), v in leaders.items()}), encoding="utf-8")
    return out


def lgbm_params(cfg, P) -> dict:
    """Parameter LightGBM: hasil tuning (models/lgbm_best_params.json) jika ada, selain itu dari config."""
    params = dict(cfg["model"]["lgbm"])
    f = P.models / "lgbm_best_params.json"
    if f.exists():
        params.update(json.loads(f.read_text(encoding="utf-8")))
    return params


def feature_columns(df: pd.DataFrame, groups: dict[str, str], exclude_groups: tuple[str, ...] = ()) -> list[str]:
    return [c for c in df.columns
            if c not in ID_COLS + LABEL_COLS and groups.get(c) not in exclude_groups]


# -----------------------------------------------------------------------------
# DEPLOYMENT — model final + prediksi terbaru
# -----------------------------------------------------------------------------
def train_final_model(cfg, P, feats: pd.DataFrame, groups: dict[str, str]):
    """Latih LightGBM pada SELURUH data berlabel. 120 hari terakhir berlabel dipakai
    untuk early stopping & memilih ambang alarm, lalu model dilatih ulang dengan
    jumlah pohon terbaik pada seluruh data."""
    from . import models as M
    from .metrics import best_threshold, pr_auc

    mcfg, vcfg = cfg["model"], cfg["validation"]
    fcols = feature_columns(feats, groups)
    data = feats[(feats["eligible"] == 1) & feats["y"].notna()]
    last = data["date"].max()
    val_start = last - pd.Timedelta(days=vcfg["val_days"] - 1)
    fit = data[data["date"] < val_start - pd.Timedelta(days=vcfg["purge_days"])]
    val = data[data["date"] >= val_start]
    seed = cfg["project"]["random_seed"]
    params = lgbm_params(cfg, P)
    b1 = M.train_lgbm(fit[fcols], fit["y"].to_numpy(), val[fcols], val["y"].to_numpy(), params,
                      CATEGORICAL, mcfg["neg_sample_rate"], seed)
    s_val = M.predict_lgbm(b1, val[fcols])
    thr = best_threshold(val["y"].to_numpy(), s_val, mcfg["threshold_metric"], mcfg.get("precision_target", 0.5))
    n_trees = max(int((b1.best_iteration or 100) * 1.1), 20)
    final = M.train_lgbm(data[fcols], data["y"].to_numpy(), None, None, params, CATEGORICAL,
                         mcfg["neg_sample_rate"], seed, n_estimators=n_trees)
    meta = {"trained_until": str(last.date()), "n_rows": int(len(data)), "n_trees": n_trees,
            "alarm_threshold": float(thr), "val_pr_auc": pr_auc(val["y"], s_val),
            "val_period": [str(val_start.date()), str(last.date())], "features": fcols}
    final.save_model(str(P.models / "lgbm_final.txt"))
    (P.models / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return final, meta


def risk_level(prob: np.ndarray, spike_now: np.ndarray, thr: float) -> np.ndarray:
    lvl = np.where(prob >= thr, "Tinggi", np.where(prob >= 0.5 * thr, "Waspada", "Rendah"))
    return np.where(spike_now == 1, "Sedang lonjak", lvl)


def predict_latest(cfg, P, booster, meta: dict, feats: pd.DataFrame, refs: dict, k_reasons: int = 3) -> pd.DataFrame:
    from .explain import lgbm_shap, top_reasons
    from . import models as M

    fcols = meta["features"]
    asof = feats["date"].max()
    cur = feats[feats["date"] == asof].reset_index(drop=True)
    prob = M.predict_lgbm(booster, cur[fcols])
    sv, _ = lgbm_shap(booster, cur[fcols])
    reasons = [top_reasons(sv[i], cur.loc[i, fcols], fcols, k_reasons) for i in range(len(cur))]
    ref = refs["prov_ref"].set_index("province_id")
    out = pd.DataFrame({
        "asof": asof, "commodity_id": cur["commodity_id"], "province_id": cur["province_id"].astype(int),
        "prob": prob, "spike_now": cur["spike_now"].fillna(0).astype(int),
        "r14_now": cur["r14_now"], "ret_7": cur["ret_7"],
    })
    out["commodity"] = out["commodity_id"].map(refs["com_name"])
    out["category"] = out["commodity_id"].map(refs["c2cat"]).map(refs["cat_name"])
    out["province"] = out["province_id"].map(ref["province"])
    out["lat"] = out["province_id"].map(ref["lat"])
    out["lon"] = out["province_id"].map(ref["lon"])
    out["level"] = risk_level(out["prob"].to_numpy(), out["spike_now"].to_numpy(), meta["alarm_threshold"])
    out["reasons"] = [" | ".join(r["text"] for r in rs) for rs in reasons]
    out["reasons_json"] = [json.dumps(rs, ensure_ascii=False) for rs in reasons]
    return out
