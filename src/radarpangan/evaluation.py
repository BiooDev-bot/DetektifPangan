"""EVALUATION — merangkum prediksi out-of-sample walk-forward menjadi tabel metrik."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import pr_auc, summarize

MODEL_LABELS = {
    "naive": 'Naive "harga minggu lalu"',
    "seasonal": "Seasonal naive (tahun lalu)",
    "sarima": "SARIMA (ARIMA + Fourier)",
    "logreg": "Regresi logistik",
    "lgbm": "LightGBM",
}


def compare_models(pred: pd.DataFrame, events: pd.DataFrame, horizon: int,
                   models: list[str] | None = None) -> pd.DataFrame:
    models = models or [c.replace("score_", "") for c in pred.columns if c.startswith("score_")]
    rows = []
    for m in models:
        s = summarize(pred, events, horizon, f"score_{m}", f"thr_{m}", prob=m in ("lgbm", "logreg", "sarima"))
        s["model"] = MODEL_LABELS.get(m, m)
        s["key"] = m
        rows.append(s)
    cols = ["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "far", "alarm_rate",
            "event_detection_rate", "median_lead_days", "share_lead_ge_7",
            "false_episodes_per_series_year", "episode_precision", "n_events", "n_rows", "pos_rate", "key"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]


def pr_auc_by(pred: pd.DataFrame, by: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(by, observed=True):
        r = {by: key, "n": len(g), "pos_rate": g["y"].mean()}
        for m in models:
            r[m] = pr_auc(g["y"], g[f"score_{m}"])
        rows.append(r)
    return pd.DataFrame(rows)


def site_summary(table: pd.DataFrame) -> dict:
    t = table.set_index("key")
    out = {
        "pr_auc_lgbm": float(t.at["lgbm", "pr_auc"]),
        "pr_auc_naive": float(t.at["naive", "pr_auc"]),
        "event_detection_rate": float(t.at["lgbm", "event_detection_rate"]),
        "median_lead_days": float(t.at["lgbm", "median_lead_days"]),
        "far": float(t.at["lgbm", "far"]),
        "false_episodes_per_series_year": float(t.at["lgbm", "false_episodes_per_series_year"]),
    }
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in out.items()}
