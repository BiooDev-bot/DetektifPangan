"""MODELING — baseline & model utama.

  Naive "harga minggu lalu" : skor = kenaikan 7 hari terakhir (tren minggu lalu diteruskan)
  Seasonal naive            : skor = kenaikan 14 hari yang terjadi tahun lalu pada periode yang sama
  SARIMA                    : ARIMA(1,1,1) + suku Fourier musiman tahunan pada harga mingguan,
                              peluang lonjakan dari distribusi ramalan 1-2 minggu ke depan
  Regresi logistik          : model linear dengan fitur yang sama (pembanding "fitur vs model")
  LightGBM                  : gradient boosting trees, model utama
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Baseline sederhana (tanpa training)
# -----------------------------------------------------------------------------
def score_naive_last_week(df: pd.DataFrame) -> np.ndarray:
    return df["ret_7"].to_numpy(dtype="float64")


def score_seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    return df["seas_ret_14_ly"].to_numpy(dtype="float64")


# -----------------------------------------------------------------------------
# Subsampling negatif (mempercepat training; bobot dikoreksi supaya probabilitas tetap wajar)
# -----------------------------------------------------------------------------
def negative_subsample(y: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if rate >= 1.0:
        return np.arange(len(y)), np.ones(len(y))
    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(y)) < rate)
    idx = np.nonzero(keep)[0]
    w = np.where(y[idx] == 1, 1.0, 1.0 / rate)
    return idx, w


# -----------------------------------------------------------------------------
# LightGBM
# -----------------------------------------------------------------------------
def train_lgbm(X_fit: pd.DataFrame, y_fit: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
               params: dict, categorical: list[str], neg_sample_rate: float = 1.0, seed: int = 42,
               n_estimators: int | None = None):
    import lightgbm as lgb

    p = dict(params)
    n_est = n_estimators or p.pop("n_estimators", 2000)
    p.pop("n_estimators", None)
    es = p.pop("early_stopping_rounds", 100)
    p.update({"seed": seed, "feature_pre_filter": False})
    idx, w = negative_subsample(np.asarray(y_fit), neg_sample_rate, seed)
    cats = [c for c in categorical if c in X_fit.columns]
    dtrain = lgb.Dataset(X_fit.iloc[idx], label=np.asarray(y_fit)[idx], weight=w,
                         categorical_feature=cats, free_raw_data=True)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = []
    if X_val is not None and len(X_val) and np.asarray(y_val).sum() > 0:
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, reference=dtrain)
        valid_sets = [dval]
        callbacks.append(lgb.early_stopping(es, verbose=False))
    p.setdefault("metric", "average_precision")
    booster = lgb.train(p, dtrain, num_boost_round=n_est, valid_sets=valid_sets, callbacks=callbacks)
    return booster


def predict_lgbm(booster, X: pd.DataFrame) -> np.ndarray:
    it = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else None
    return booster.predict(X, num_iteration=it)


# -----------------------------------------------------------------------------
# Regresi logistik
# -----------------------------------------------------------------------------
def train_logreg(X_fit: pd.DataFrame, y_fit: np.ndarray, neg_sample_rate: float = 1.0, seed: int = 42):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx, w = negative_subsample(np.asarray(y_fit), neg_sample_rate, seed)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=500))
    X = X_fit.iloc[idx].replace([np.inf, -np.inf], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, np.asarray(y_fit)[idx], logisticregression__sample_weight=w)
    return model


def predict_logreg(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.replace([np.inf, -np.inf], np.nan))[:, 1]


# -----------------------------------------------------------------------------
# SARIMA (ARIMA + Fourier) pada harga mingguan
# -----------------------------------------------------------------------------
def fourier_terms(index: pd.DatetimeIndex, period_days: float = 365.25, k: int = 2) -> pd.DataFrame:
    t = (index - pd.Timestamp("2000-01-01")).days.to_numpy() / period_days
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(2 * np.pi * i * t)
        cols[f"cos{i}"] = np.cos(2 * np.pi * i * t)
    return pd.DataFrame(cols, index=index)


def sarima_series_scores(price: pd.Series, fit_end, score_start, score_end, threshold: float,
                         order=(1, 1, 1), k: int = 2, min_weeks: int = 104) -> pd.Series:
    """Skor peluang lonjakan pada origin mingguan (Jumat) dalam [score_start, score_end].

    Model dilatih hanya dengan data <= fit_end. Pada tiap origin o (data s.d. o), ramal log harga
    rata-rata mingguan 1 & 2 minggu ke depan (mu_h, sigma_h), lalu
        skor = max_h P( y_{o+h} - y_o > ln(1+ambang) ) = max_h [1 - Phi((ln(1+ambang) - (mu_h - y_o)) / sigma_h)]
    """
    from scipy.stats import norm
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.log(price).resample("W-FRI").mean()
    y = y.loc[: pd.Timestamp(score_end) + pd.Timedelta(days=21)]
    first_valid = y.first_valid_index()
    if first_valid is None:
        return pd.Series(dtype="float64")
    y = y.loc[first_valid:]
    X = fourier_terms(y.index, k=k).to_numpy()
    yv = y.to_numpy(dtype="float64")                 # array numpy: hindari masalah indeks tanggal statsmodels
    train_mask = (y.index <= pd.Timestamp(fit_end))
    n_train = int(train_mask.sum())
    if np.isfinite(yv[:n_train]).sum() < min_weeks:
        return pd.Series(dtype="float64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(yv[:n_train], exog=X[:n_train], order=order, trend="n")
        res = mod.fit(disp=False, maxiter=100)
        res_all = res.apply(yv, exog=X)          # parameter tetap, filter dijalankan di seluruh data
    log_thr = np.log1p(threshold)
    idx = np.nonzero((y.index >= pd.Timestamp(score_start)) & (y.index <= pd.Timestamp(score_end)))[0]
    out = {}
    for i in idx:
        if i + 2 >= len(yv) or not np.isfinite(yv[i]):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # prediksi dinamis dari origin i: langkah i+1 & i+2 hanya memakai data s.d. i
            pr = res_all.get_prediction(start=i + 1, end=i + 2, dynamic=0)
        mu = np.asarray(pr.predicted_mean, dtype="float64").ravel()
        sd = np.sqrt(np.asarray(pr.var_pred_mean, dtype="float64").ravel())
        z = (log_thr - (mu - yv[i])) / np.where(sd > 1e-9, sd, np.nan)
        out[y.index[i]] = float(np.nanmax(1 - norm.cdf(z)))
    return pd.Series(out, dtype="float64")


def sarima_fold_scores(P: pd.DataFrame, series: list[tuple[str, int]], thresholds: pd.Series,
                       fit_end, score_start, score_end, order=(1, 1, 1), k: int = 2,
                       n_jobs: int = 1) -> pd.DataFrame:
    def one(col):
        s = sarima_series_scores(P[col], fit_end, score_start, score_end,
                                 float(thresholds.get(col, 0.15)), order, k)
        return pd.DataFrame({"date": s.index, "commodity_id": col[0], "province_id": col[1],
                             "score_sarima": s.to_numpy()})

    if n_jobs == 1:
        parts = [one(c) for c in series]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs)(delayed(one)(c) for c in series)
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["date", "commodity_id", "province_id", "score_sarima"])
    return pd.concat(parts, ignore_index=True)
