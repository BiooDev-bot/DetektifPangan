"""DATA PREPARATION — membersihkan & menyusun panel harga harian.

Alur untuk tiap jenis pasar:
  long -> wide (tanggal observasi x [komoditas, provinsi])
       -> buang nilai ekstrem / salah input (blip 1 hari, salah skala x10)
       -> smoothing median bergulir 3 observasi (trailing, tanpa melihat masa depan)
       -> reindex ke kalender harian + isi maju (ffill) maks 7 hari
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def long_to_wide(long_df: pd.DataFrame, price_type_id: int) -> pd.DataFrame:
    d = long_df[long_df["price_type_id"] == price_type_id]
    wide = d.pivot_table(index="date", columns=["commodity_id", "province_id"], values="price", aggfunc="mean")
    return wide.sort_index().sort_index(axis=1)


def detect_bad_values(wide: pd.DataFrame, blip_log_threshold: float = np.log(1.5),
                      scale_log_threshold: float = np.log(4.0), window: int = 31) -> pd.DataFrame:
    """Tandai nilai yang hampir pasti salah input (True = buang).

    1) Blip: 1 observasi melonjak/anjlok > 50% terhadap observasi sebelum DAN sesudahnya
       lalu langsung kembali -> ciri salah ketik, bukan lonjakan harga sungguhan.
    2) Salah skala: menyimpang > 4x dari median 31 observasi di sekitarnya (mis. kelebihan/
       kekurangan satu angka nol).
    Catatan: deteksi ini memakai observasi SESUDAHNYA, jadi hanya dipakai untuk membersihkan
    data historis. Di deployment, titik terbaru ditandai 'belum terverifikasi'.
    """
    x = np.log(wide)
    prev = x.ffill().shift(1)
    nxt = x.bfill().shift(-1)
    up = (x - prev > blip_log_threshold) & (x - nxt > blip_log_threshold)
    down = (prev - x > blip_log_threshold) & (nxt - x > blip_log_threshold)
    med = x.rolling(window, center=True, min_periods=5).median()
    scale = (x - med).abs() > scale_log_threshold
    return (up | down | scale) & x.notna()


def smooth_trailing(wide: pd.DataFrame, obs: int = 3) -> pd.DataFrame:
    """Median bergulir `obs` observasi terakhir (hanya masa lalu -> aman untuk real-time)."""
    if obs <= 1:
        return wide
    return wide.rolling(obs, min_periods=1).median()


def to_calendar(wide: pd.DataFrame, start=None, end=None, ffill_limit: int = 7) -> pd.DataFrame:
    idx = pd.date_range(start or wide.index.min(), end or wide.index.max(), freq="D")
    return wide.reindex(idx).ffill(limit=ffill_limit)


def build_price_panels(long_df: pd.DataFrame, cfg: dict, start=None, end=None
                       ) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Panel harga bersih per jenis pasar (index kalender harian, kolom [komoditas, provinsi]).

    Return: (panels, cleaning_log) — cleaning_log = jumlah nilai dibuang per jenis pasar/komoditas.
    """
    tcfg = cfg["target"]
    start = pd.Timestamp(start or long_df["date"].min())
    end = pd.Timestamp(end or long_df["date"].max())
    panels, logs = {}, []
    for pt in sorted(long_df["price_type_id"].unique()):
        wide = long_to_wide(long_df, int(pt))
        bad = detect_bad_values(wide, tcfg["blip_log_threshold"])
        n_bad = bad.sum()
        logs.append(pd.DataFrame({"price_type_id": int(pt), "n_obs": wide.notna().sum(), "n_removed": n_bad}))
        clean = wide.mask(bad)
        clean = smooth_trailing(clean, tcfg["smoothing_obs"])
        panels[int(pt)] = to_calendar(clean, start, end, tcfg["ffill_limit_days"])
    log_df = pd.concat(logs).reset_index()
    return panels, log_df


def quality_report(long_df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan kualitas per (jenis pasar, komoditas, provinsi)."""
    bdays = pd.bdate_range(long_df["date"].min(), long_df["date"].max())
    g = long_df.groupby(["price_type", "commodity", "province"], observed=True)
    rep = g.agg(first_date=("date", "min"), last_date=("date", "max"),
                n_obs=("price", "count"), n_rows=("price", "size"),
                median_price=("price", "median"))
    rep["coverage_vs_all_bdays"] = rep["n_obs"] / len(bdays)
    return rep.reset_index()
