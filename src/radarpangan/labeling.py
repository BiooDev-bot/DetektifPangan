"""BUSINESS -> DATA MINING GOAL: definisi kejadian lonjakan & label peringatan dini.

Definisi (default config):
  r14(t)      = P(t) / P(t-14 hari) - 1                     P = harga konsumen yang sudah dihaluskan
  lonjak(t)   = r14(t) > ambang (15%)                        -> status "sedang lonjak"
  onset(t)    = lonjak(t) dan 14 hari sebelumnya tidak lonjak -> awal sebuah KEJADIAN lonjakan
  y(t)        = 1 jika ada onset di (t, t+14]                 -> target peringatan dini
  eligible(t) = tidak sedang lonjak di t (alarm hanya berguna SEBELUM lonjakan terjadi)

Lead time sebuah kejadian = tanggal onset - tanggal alarm pertama di jendela [onset-14, onset-1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_per_column(columns: pd.MultiIndex, commodity_to_cat: dict, cfg: dict) -> pd.Series:
    """Ambang lonjakan per seri. Bisa dibedakan per kategori/varian lewat target.threshold_overrides."""
    tcfg = cfg["target"]
    over = tcfg.get("threshold_overrides") or {}
    vals = [over.get(com, over.get(commodity_to_cat.get(com), tcfg["default_threshold"]))
            for com, _prov in columns]
    return pd.Series(vals, index=columns, dtype="float64")


def future_window_max(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """max(df[t+1 .. t+horizon]) untuk tiap t (index = kalender harian)."""
    return df[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)


def remaining_run_length(a: np.ndarray) -> np.ndarray:
    """Untuk matriks 0/1 (waktu x seri): panjang run 1 berturut-turut mulai dari t ke depan."""
    out = np.zeros_like(a, dtype="float64")
    run = np.zeros(a.shape[1])
    for t in range(a.shape[0] - 1, -1, -1):
        run = (run + 1.0) * a[t]
        out[t] = run
    return out


def days_since(a: np.ndarray, cap: int = 365) -> np.ndarray:
    """Jumlah hari sejak terakhir kali a == 1 (0 jika hari ini 1). Belum pernah -> cap."""
    out = np.zeros_like(a, dtype="float64")
    since = np.full(a.shape[1], float(cap))
    for t in range(a.shape[0]):
        since = np.where(a[t] == 1, 0.0, np.minimum(since + 1.0, cap))
        out[t] = since
    return out


def make_labels(P: pd.DataFrame, thresholds: pd.Series | float, cfg: dict) -> dict[str, pd.DataFrame]:
    """P: panel harga konsumen harian (kalender), kolom MultiIndex [commodity_id, province_id]."""
    t = cfg["target"]
    w, h, cool = t["change_window_days"], t["horizon_days"], t["cooldown_days"]

    r = P / P.shift(w) - 1
    if isinstance(thresholds, pd.Series):
        above = r.gt(thresholds.reindex(P.columns), axis=1)
    else:
        above = r > thresholds
    spike = above.astype("float64").where(r.notna())               # 1 / 0 / NaN
    s0 = spike.fillna(0.0)
    prior = s0.shift(1).rolling(cool, min_periods=1).max().fillna(0.0)
    onset = ((s0 == 1) & (prior == 0)).astype("float64")

    y = future_window_max(onset, h)
    n_valid = r.notna().astype("float64")[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)
    y = y.where(n_valid >= h / 2)                                     # masa depan terlalu bolong -> tak berlabel
    y.loc[y.index > P.index.max() - pd.Timedelta(days=h)] = np.nan    # jendela masa depan belum lengkap

    eligible = (spike == 0) & P.notna()
    return {"r14": r, "spike": spike, "onset": onset, "y": y, "eligible": eligible}


def events_table(labels: dict[str, pd.DataFrame], P: pd.DataFrame, peak_days: int = 30,
                 window: int = 14) -> pd.DataFrame:
    """Daftar kejadian lonjakan (1 baris = 1 onset) beserta besar dan durasinya."""
    onset = labels["onset"]
    spike = labels["spike"].fillna(0.0)
    run = pd.DataFrame(remaining_run_length(spike.to_numpy()), index=spike.index, columns=spike.columns)
    base = P.shift(window)
    peak = P[::-1].rolling(peak_days, min_periods=1).max()[::-1]

    o = onset.to_numpy() == 1
    ti, ci = np.nonzero(o)
    cols = onset.columns
    ev = pd.DataFrame({
        "date": onset.index[ti],
        "commodity_id": [cols[j][0] for j in ci],
        "province_id": [int(cols[j][1]) for j in ci],
        "rise_at_onset": labels["r14"].to_numpy()[ti, ci],
        "peak_rise_30d": peak.to_numpy()[ti, ci] / base.to_numpy()[ti, ci] - 1,
        "spike_days": run.to_numpy()[ti, ci],
        "price_before": base.to_numpy()[ti, ci],
        "price_at_onset": P.to_numpy()[ti, ci],
    })
    return ev.sort_values(["date", "commodity_id", "province_id"]).reset_index(drop=True)
