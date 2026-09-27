"""Curah hujan harian per provinsi dari NASA POWER + fitur anomali.

Sumber : https://power.larc.nasa.gov (gratis, tanpa API key)
Titik  : koordinat ibu kota / kota pasar utama tiap provinsi (lihat reference.py)
Anomali: jumlah hujan w hari terakhir dibandingkan klimatologi (1991-2017) pada
         hari-dalam-tahun yang sama -> z-score. Klimatologi diambil dari periode
         SEBELUM data harga dimulai, jadi fitur ini bebas kebocoran (leakage).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_power_daily(lat: float, lon: float, start: str, end: str,
                      parameter: str = "PRECTOTCORR", retries: int = 4) -> pd.Series:
    params = {
        "parameters": parameter, "community": "AG", "latitude": lat, "longitude": lon,
        "start": pd.Timestamp(start).strftime("%Y%m%d"), "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "format": "JSON",
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(POWER_URL, params=params, timeout=120)
            r.raise_for_status()
            js = r.json()
            fill = js.get("header", {}).get("fill_value", -999.0)
            data = js["properties"]["parameter"][parameter]
            s = pd.Series(data, dtype="float64")
            s.index = pd.to_datetime(s.index, format="%Y%m%d")
            return s.where(s != fill).sort_index()
        except Exception as e:  # noqa: BLE001
            log.warning("NASA POWER gagal (%d/%d): %s", attempt, retries, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"NASA POWER gagal untuk ({lat}, {lon})")


def fetch_all_provinces(ref: pd.DataFrame, raw_dir: Path, start: str, end: str | None = None,
                        parameter: str = "PRECTOTCORR", refresh_days: int = 45) -> pd.DataFrame:
    """Ambil & cache curah hujan semua provinsi. Cache per provinsi di raw_dir/*.csv.

    Hanya `refresh_days` terakhir yang diambil ulang bila cache sudah ada.
    Return: DataFrame long [date, province_id, precip_mm].
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    frames = []
    for row in ref.itertuples(index=False):
        path = raw_dir / f"nasa_power_{row.province_id:02d}.csv"
        old = None
        fetch_start = pd.Timestamp(start)
        if path.exists():
            old = pd.read_csv(path, parse_dates=["date"]).set_index("date")["precip_mm"]
            fetch_start = max(pd.Timestamp(start), old.dropna().index.max() - pd.Timedelta(days=refresh_days))
        if fetch_start <= end:
            new = fetch_power_daily(row.lat, row.lon, fetch_start, end, parameter)
            s = new if old is None else pd.concat([old[old.index < fetch_start], new])
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.rename("precip_mm").rename_axis("date").reset_index().to_csv(path, index=False)
        else:
            s = old
        frames.append(pd.DataFrame({"date": s.index, "province_id": row.province_id, "precip_mm": s.to_numpy()}))
        log.info("Hujan %s: %d hari", row.province, len(s))
    return pd.concat(frames, ignore_index=True)


def rainfall_anomalies(precip: pd.DataFrame, clim_start: str, clim_end: str,
                       windows=(7, 30, 90), doy_halfwidth: int = 15) -> dict[str, pd.DataFrame]:
    """Anomali (z-score) jumlah hujan bergulir w hari vs klimatologi hari-dalam-tahun.

    precip: long [date, province_id, precip_mm]
    Return: {f"rain_{w}_anom": wide DataFrame (index=tanggal kalender, kolom=province_id)}
    """
    wide = precip.pivot_table(index="date", columns="province_id", values="precip_mm")
    wide = wide.asfreq("D")
    # data NASA POWER ~3 hari terakhir kosong (latensi) -> isi maju maksimal 5 hari
    wide = wide.ffill(limit=5)
    out = {}
    for w in windows:
        roll = wide.rolling(w, min_periods=int(w * 0.8)).sum()
        clim = roll.loc[clim_start:clim_end]
        doy = clim.index.dayofyear.to_numpy()
        doy = np.where(doy == 366, 365, doy)
        mu = np.full((366, roll.shape[1]), np.nan)
        sd = np.full((366, roll.shape[1]), np.nan)
        vals = clim.to_numpy()
        for d in range(1, 366):
            # jendela +-15 hari (melingkar) supaya klimatologi halus
            dist = np.abs(((doy - d) + 182) % 365 - 182)
            sel = vals[dist <= doy_halfwidth]
            mu[d] = np.nanmean(sel, axis=0)
            sd[d] = np.nanstd(sel, axis=0)
        mu[0], sd[0] = mu[1], sd[1]
        all_doy = roll.index.dayofyear.to_numpy()
        all_doy = np.where(all_doy == 366, 365, all_doy)
        # batas bawah SD (1 mm/hari x w) supaya musim kemarau (SD ~0) tidak menghasilkan z ekstrem
        sd_floor = np.maximum(sd[all_doy], 1.0 * w)
        z = (roll.to_numpy() - mu[all_doy]) / sd_floor
        out[f"rain_{w}_anom"] = pd.DataFrame(np.clip(z, -5, 5), index=roll.index, columns=roll.columns)
    return out
