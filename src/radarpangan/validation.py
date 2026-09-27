"""MODELING/EVALUATION — validasi walk-forward (expanding window) dengan purging.

Untuk tiap fold k dengan blok uji [T_k, T_k + 3 bulan):

   |<----------- fit ----------->|purge|<-- val (120 hr) -->|purge|<-- test -->|
                                 14 hr                       14 hr  T_k

  - fit   : melatih model
  - val   : early stopping LightGBM + memilih ambang alarm (tanpa menyentuh data uji)
  - purge : 14 hari (= horizon). Label y(t) melihat harga sampai t+14, jadi baris latih
            terakhir harus berakhir 14 hari sebelum periode berikutnya -> tidak ada kebocoran.
  - test  : prediksi out-of-sample, lalu jendela maju 3 bulan (seperti pemakaian nyata).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import models as M
from .metrics import best_threshold

log = logging.getLogger(__name__)


def make_folds(dates: pd.Series | pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    v = cfg["validation"]
    purge = pd.Timedelta(days=v["purge_days"])
    one = pd.Timedelta(days=1)
    last = pd.Timestamp(pd.Series(dates).max())
    starts = pd.date_range(v["first_test_start"], last, freq=f"{v['test_block_months']}MS")
    rows = []
    for k, t0 in enumerate(starts):
        t1 = min(t0 + pd.DateOffset(months=v["test_block_months"]) - one, last)
        val_end = t0 - purge - one
        val_start = val_end - pd.Timedelta(days=v["val_days"]) + one
        fit_end = val_start - purge - one
        rows.append({"fold": k, "fit_end": fit_end, "val_start": val_start, "val_end": val_end,
                     "test_start": t0, "test_end": t1})
    return pd.DataFrame(rows)


def modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Baris yang valid untuk latih/uji: tidak sedang lonjak & label masa depan lengkap."""
    return df[(df["eligible"] == 1) & df["y"].notna()].reset_index(drop=True)


def run_walk_forward(
    data: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    folds: pd.DataFrame | None = None,
    use_models: tuple[str, ...] = ("naive", "seasonal", "logreg", "lgbm"),
    categorical: list[str] | None = None,
    lgbm_params: dict | None = None,
    tag: str = "",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan walk-forward. Return (prediksi out-of-sample per baris uji, info per fold)."""
    mcfg = cfg["model"]
    seed = cfg["project"]["random_seed"]
    params = dict(lgbm_params or mcfg["lgbm"])
    categorical = categorical if categorical is not None else ["commodity_code", "category_code", "province_code"]
    folds = make_folds(data["date"], cfg) if folds is None else folds
    thr_metric, prec_t = mcfg["threshold_metric"], mcfg.get("precision_target", 0.5)

    preds, infos = [], []
    for f in folds.itertuples(index=False):
        t_start = time.time()
        d = data["date"]
        fit = data[d <= f.fit_end]
        val = data[(d >= f.val_start) & (d <= f.val_end)]
        test = data[(d >= f.test_start) & (d <= f.test_end)]
        if len(test) == 0 or fit["y"].sum() == 0:
            continue
        out = test[["date", "commodity_id", "province_id", "y"]].copy()
        out["fold"] = f.fold
        info = {"fold": f.fold, "test_start": f.test_start, "test_end": f.test_end,
                "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
                "pos_fit": float(fit["y"].mean()), "pos_test": float(test["y"].mean())}

        def add(name, s_val, s_test):
            thr = best_threshold(val["y"].to_numpy(), s_val, thr_metric, prec_t)
            out[f"score_{name}"] = s_test
            out[f"thr_{name}"] = thr
            info[f"thr_{name}"] = thr

        if "naive" in use_models:
            add("naive", M.score_naive_last_week(val), M.score_naive_last_week(test))
        if "seasonal" in use_models:
            add("seasonal", M.score_seasonal_naive(val), M.score_seasonal_naive(test))
        if "logreg" in use_models:
            num_cols = [c for c in feature_cols if c not in categorical]
            lr = M.train_logreg(fit[num_cols], fit["y"].to_numpy(), mcfg["neg_sample_rate"], seed)
            add("logreg", M.predict_logreg(lr, val[num_cols]), M.predict_logreg(lr, test[num_cols]))
        if "lgbm" in use_models:
            booster = M.train_lgbm(fit[feature_cols], fit["y"].to_numpy(), val[feature_cols],
                                   val["y"].to_numpy(), params, categorical,
                                   mcfg["neg_sample_rate"], seed)
            info["lgbm_best_iter"] = booster.best_iteration
            add("lgbm", M.predict_lgbm(booster, val[feature_cols]), M.predict_lgbm(booster, test[feature_cols]))
        info["seconds"] = round(time.time() - t_start, 1)
        infos.append(info)
        preds.append(out)
        if verbose:
            msg = " | ".join(f"{k}={v:.3f}" for k, v in info.items() if k.startswith("thr_"))
            print(f"[{tag}] fold {f.fold:2d} uji {f.test_start:%Y-%m}..{f.test_end:%Y-%m} "
                  f"n_fit={len(fit):,} pos_uji={info['pos_test']:.3f} {msg} ({info['seconds']} dtk)")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(infos)
