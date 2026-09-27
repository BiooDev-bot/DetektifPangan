"""EVALUATION — metrik untuk kejadian langka & peringatan dini.

  PR-AUC (average precision) : ringkasan presisi-recall, cocok untuk kelas langka
                               (baseline acak = proporsi positif, bukan 0.5 seperti ROC-AUC)
  FAR (false alarm ratio)    : dari semua alarm, berapa persen yang TIDAK diikuti lonjakan
  Episode alarm palsu/thn    : rangkaian alarm berturut-turut yang tidak diikuti lonjakan,
                               per seri (provinsi x komoditas) per tahun -> beban operasional
  Deteksi kejadian & lead time: berapa persen kejadian lonjakan yang didahului alarm,
                               dan berapa hari sebelumnya alarm pertama berbunyi
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def best_threshold(y, s, metric: str = "f1", precision_target: float = 0.5) -> float:
    """Pilih ambang alarm pada data validasi."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or y.sum() == 0:
        return float(np.nanmax(s)) + 1e-9 if len(s) else np.inf
    prec, rec, thr = precision_recall_curve(y, s)
    prec, rec = prec[:-1], rec[:-1]
    if metric == "precision_target":
        ok = np.where(prec >= precision_target)[0]
        return float(thr[ok[0]]) if len(ok) else float(thr[-1])
    beta = 2.0 if metric == "f2" else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
    f = np.nan_to_num(f)
    return float(thr[int(np.argmax(f))])


def at_threshold(y, s, thr) -> dict:
    y = np.asarray(y).astype(int)
    a = np.asarray(s, dtype=float) >= np.asarray(thr, dtype=float)
    tp = int((a & (y == 1)).sum())
    fp = int((a & (y == 0)).sum())
    fn = int((~a & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else np.nan
    return {"precision": prec, "recall": rec, "f1": f1,
            "far": (1 - prec) if prec == prec else np.nan,
            "alarm_rate": float(a.mean()) if len(a) else np.nan, "tp": tp, "fp": fp, "fn": fn}


# -----------------------------------------------------------------------------
# Metrik berbasis kejadian
# -----------------------------------------------------------------------------
def event_detection(pred: pd.DataFrame, events: pd.DataFrame, horizon: int, score_col: str,
                    thr_col: str = "threshold") -> pd.DataFrame:
    """Untuk setiap kejadian (onset) di periode prediksi: terdeteksi? lead time berapa hari?

    pred  : [date, commodity_id, province_id, score_col, thr_col] (baris origin yang dievaluasi)
    events: [date, commodity_id, province_id, ...] tanggal onset
    Hanya kejadian yang jendela peringatannya [onset-H, onset-1] berada penuh di periode prediksi.
    """
    p = pred[["date", "commodity_id", "province_id", score_col, thr_col]].copy()
    p["alarm"] = p[score_col] >= p[thr_col]
    t0, t1 = p["date"].min(), p["date"].max()
    ev = events[(events["date"] - pd.Timedelta(days=horizon) >= t0) & (events["date"] - pd.Timedelta(days=1) <= t1)].copy()
    alarms = p[p["alarm"]].groupby(["commodity_id", "province_id"])["date"].apply(lambda s: np.sort(s.to_numpy()))
    covered = p.groupby(["commodity_id", "province_id"])["date"].agg(["min", "max"])
    lead, detected, in_scope = [], [], []
    for d, c, pr in ev[["date", "commodity_id", "province_id"]].itertuples(index=False):
        key = (c, pr)
        ok = key in covered.index and covered.at[key, "min"] <= d - pd.Timedelta(days=horizon)
        in_scope.append(ok)
        arr = alarms.get(key)
        if arr is None or len(arr) == 0:
            detected.append(False)
            lead.append(np.nan)
            continue
        lo = np.datetime64(d - pd.Timedelta(days=horizon))
        hi = np.datetime64(d - pd.Timedelta(days=1))
        i = np.searchsorted(arr, lo, side="left")
        if i < len(arr) and arr[i] <= hi:
            detected.append(True)
            lead.append((np.datetime64(d) - arr[i]) / np.timedelta64(1, "D"))
        else:
            detected.append(False)
            lead.append(np.nan)
    ev["in_scope"] = in_scope
    ev["detected"] = detected
    ev["lead_days"] = lead
    return ev[ev["in_scope"]].drop(columns="in_scope")


def alarm_episodes(pred: pd.DataFrame, events: pd.DataFrame, horizon: int, score_col: str,
                   thr_col: str = "threshold", max_gap_days: int = 4) -> pd.DataFrame:
    """Gabungkan alarm berturut-turut (jeda <= 4 hari, akhir pekan) menjadi episode.
    Episode 'palsu' = tidak ada onset dalam (awal episode, akhir episode + H]."""
    p = pred.loc[pred[score_col] >= pred[thr_col], ["date", "commodity_id", "province_id"]]
    p = p.sort_values(["commodity_id", "province_id", "date"])
    if p.empty:
        return pd.DataFrame(columns=["commodity_id", "province_id", "start", "end", "n_days", "is_false"])
    gap = p.groupby(["commodity_id", "province_id"])["date"].diff().dt.days
    new = gap.isna() | (gap > max_gap_days)
    p["ep"] = new.cumsum()
    ep = p.groupby("ep").agg(commodity_id=("commodity_id", "first"), province_id=("province_id", "first"),
                             start=("date", "min"), end=("date", "max"), n_days=("date", "size"))
    ev = events.groupby(["commodity_id", "province_id"])["date"].apply(lambda s: np.sort(s.to_numpy()))
    is_false = []
    for c, pr, s, e in ep[["commodity_id", "province_id", "start", "end"]].itertuples(index=False):
        arr = ev.get((c, pr))
        if arr is None:
            is_false.append(True)
            continue
        lo, hi = np.datetime64(s), np.datetime64(e + pd.Timedelta(days=horizon))
        i = np.searchsorted(arr, lo, side="right")
        is_false.append(not (i < len(arr) and arr[i] <= hi))
    ep["is_false"] = is_false
    return ep.reset_index(drop=True)


def summarize(pred: pd.DataFrame, events: pd.DataFrame, horizon: int, score_col: str,
              thr_col: str = "threshold", prob: bool = False) -> dict:
    """Ringkasan lengkap satu model pada prediksi out-of-sample."""
    y = pred["y"].to_numpy()
    s = pred[score_col].to_numpy()
    out = {"n_rows": len(pred), "pos_rate": float(np.mean(y)),
           "pr_auc": pr_auc(y, s), "roc_auc": roc_auc(y, s)}
    if prob:
        out["brier"] = float(np.mean((np.clip(s, 0, 1) - y) ** 2))
    out.update(at_threshold(y, s, pred[thr_col].to_numpy()))
    ed = event_detection(pred, events, horizon, score_col, thr_col)
    out["n_events"] = len(ed)
    out["event_detection_rate"] = float(ed["detected"].mean()) if len(ed) else np.nan
    out["median_lead_days"] = float(ed["lead_days"].median()) if ed["detected"].any() else np.nan
    out["mean_lead_days"] = float(ed["lead_days"].mean()) if ed["detected"].any() else np.nan
    out["share_lead_ge_7"] = float((ed["lead_days"] >= 7).mean()) if len(ed) else np.nan
    ep = alarm_episodes(pred, events, horizon, score_col, thr_col)
    n_series = pred.groupby(["commodity_id", "province_id"]).ngroups
    years = (pred["date"].max() - pred["date"].min()).days / 365.25
    out["false_episodes_per_series_year"] = float(ep["is_false"].sum() / max(n_series * years, 1e-9))
    out["episode_precision"] = float(1 - ep["is_false"].mean()) if len(ep) else np.nan
    return out
