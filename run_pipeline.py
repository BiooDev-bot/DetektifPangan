"""RadarPangan — menjalankan pipeline tanpa notebook (untuk otomatisasi / update harian).

Contoh:
    python run_pipeline.py scrape            # ambil/lengkapi data PIHPS + curah hujan (resume dari cache)
    python run_pipeline.py prepare           # panel bersih, label, jaringan lead-lag, fitur
    python run_pipeline.py evaluate          # walk-forward: baseline vs LightGBM (+ ringkasan metrik)
    python run_pipeline.py train             # latih model final di seluruh data
    python run_pipeline.py predict           # prediksi 14 hari ke depan + bangun dashboard statis (site/)
    python run_pipeline.py update            # rutinitas harian: scrape 2 bulan terakhir -> fitur -> prediksi -> site
    python run_pipeline.py update --retrain  # sama, tapi latih ulang model (mis. seminggu/sebulan sekali)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from radarpangan import pipeline as PL  # noqa: E402
from radarpangan.config import load_project  # noqa: E402

log = logging.getLogger("radarpangan")


def cmd_scrape(cfg, P, args):
    s = PL.collect(cfg, P, refresh_recent_months=args.refresh_months)
    print(s["status"].value_counts().to_string())


def cmd_prepare(cfg, P, args):
    for mode in ("evaluation", "final"):
        t = time.time()
        out = PL.prepare(cfg, P, leaders_mode=mode)
        print(f"prepare[{mode}] {out['features'].shape} dalam {time.time() - t:.0f} dtk")


def cmd_evaluate(cfg, P, args):
    from radarpangan.evaluation import compare_models, site_summary
    from radarpangan.validation import modeling_frame, run_walk_forward

    feats = pd.read_parquet(P.processed / "features.parquet")
    groups = json.loads((P.processed / "feature_groups.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(P.processed / "events.parquet")
    fcols = PL.feature_columns(feats, groups)
    pred, info = run_walk_forward(modeling_frame(feats), fcols, cfg, lgbm_params=PL.lgbm_params(cfg, P), tag="main")
    pred.to_parquet(P.processed / "oof_predictions.parquet", index=False)
    info.to_csv(P.tables / "walk_forward_folds.csv", index=False)
    table = compare_models(pred, events, cfg["target"]["horizon_days"], ["naive", "seasonal", "logreg", "lgbm"])
    table.to_csv(P.tables / "model_comparison.csv", index=False)
    (P.tables / "eval_summary.json").write_text(json.dumps(site_summary(table), indent=1), encoding="utf-8")
    print(table.drop(columns=["key"]).round(3).to_string(index=False))


def cmd_train(cfg, P, args):
    feats = pd.read_parquet(P.processed / "features_final.parquet")
    groups = json.loads((P.processed / "feature_groups.json").read_text(encoding="utf-8"))
    _, meta = PL.train_final_model(cfg, P, feats, groups)
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=1))


def cmd_predict(cfg, P, args):
    import lightgbm as lgb

    from radarpangan.site import export_site

    refs = PL.load_refs(P)
    feats = pd.read_parquet(P.processed / "features_final.parquet")
    booster = lgb.Booster(model_file=str(P.models / "lgbm_final.txt"))
    meta = json.loads((P.models / "model_meta.json").read_text(encoding="utf-8"))
    risk = PL.predict_latest(cfg, P, booster, meta, feats, refs)
    risk.to_csv(P.output / "risk_latest.csv", index=False)
    panels = {pt: PL.load_panel(P.processed / f"panel_pt{pt}.parquet") for pt in cfg["scraping"]["price_types"]
              if (P.processed / f"panel_pt{pt}.parquet").exists()}
    leaders = {(k.split("|")[0], int(k.split("|")[1])): v
               for k, v in json.loads((P.processed / "leaders_final.json").read_text(encoding="utf-8")).items()}
    ev_path = P.tables / "eval_summary.json"
    site = export_site(cfg, P, risk, meta, refs, panels, pd.read_parquet(P.processed / "events.parquet"),
                       pd.read_parquet(P.processed / "network_edges_final.parquet"),
                       pd.read_parquet(P.processed / "leader_scores_final.parquet"), leaders,
                       json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None)
    print(f"Prediksi per {risk['asof'].iloc[0]:%Y-%m-%d}: {risk['level'].value_counts().to_dict()}")
    print("Dashboard statis:", site / "index.html")


def cmd_update(cfg, P, args):
    PL.collect(cfg, P, refresh_recent_months=cfg["scraping"]["refresh_recent_months"])
    PL.prepare(cfg, P, leaders_mode="final")
    if args.retrain or not (P.models / "lgbm_final.txt").exists():
        cmd_train(cfg, P, args)
    cmd_predict(cfg, P, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scrape", "prepare", "evaluate", "train", "predict", "update", "all"])
    ap.add_argument("--retrain", action="store_true", help="latih ulang model saat update")
    ap.add_argument("--refresh-months", type=int, default=None, help="jumlah bulan terakhir yang diambil ulang")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, P = load_project(ROOT)
    steps = {"scrape": [cmd_scrape], "prepare": [cmd_prepare], "evaluate": [cmd_evaluate], "train": [cmd_train],
             "predict": [cmd_predict], "update": [cmd_update],
             "all": [cmd_scrape, cmd_prepare, cmd_evaluate, cmd_train, cmd_predict]}
    for fn in steps[args.command]:
        t = time.time()
        fn(cfg, P, args)
        log.info("%s selesai dalam %.0f dtk", fn.__name__, time.time() - t)


if __name__ == "__main__":
    main()
