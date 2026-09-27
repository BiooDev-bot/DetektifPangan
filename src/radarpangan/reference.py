"""Data referensi: provinsi (koordinat ibu kota, pulau), tetangga geografis, kalender hari besar."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Provinsi — id mengikuti PIHPS (GetRefProvince). Koordinat = ibu kota / kota pasar utama.
# -----------------------------------------------------------------------------
_PROVINCES = [
    # id, nama PIHPS, kota acuan, lat, lon, pulau
    (1, "Aceh", "Banda Aceh", 5.5483, 95.3238, "Sumatera"),
    (2, "Sumatera Utara", "Medan", 3.5952, 98.6722, "Sumatera"),
    (3, "Sumatera Barat", "Padang", -0.9471, 100.4172, "Sumatera"),
    (4, "Riau", "Pekanbaru", 0.5071, 101.4478, "Sumatera"),
    (5, "Kepulauan Riau", "Tanjung Pinang", 0.9186, 104.4665, "Sumatera"),
    (6, "Jambi", "Jambi", -1.6101, 103.6131, "Sumatera"),
    (7, "Bengkulu", "Bengkulu", -3.7928, 102.2608, "Sumatera"),
    (8, "Sumatera Selatan", "Palembang", -2.9761, 104.7754, "Sumatera"),
    (9, "Kepulauan Bangka Belitung", "Pangkal Pinang", -2.1316, 106.1169, "Sumatera"),
    (10, "Lampung", "Bandar Lampung", -5.3971, 105.2668, "Sumatera"),
    (11, "Banten", "Serang", -6.1104, 106.1640, "Jawa"),
    (12, "Jawa Barat", "Bandung", -6.9175, 107.6191, "Jawa"),
    (13, "DKI Jakarta", "Jakarta", -6.2088, 106.8456, "Jawa"),
    (14, "Jawa Tengah", "Semarang", -6.9667, 110.4167, "Jawa"),
    (15, "DI Yogyakarta", "Yogyakarta", -7.7956, 110.3695, "Jawa"),
    (16, "Jawa Timur", "Surabaya", -7.2575, 112.7521, "Jawa"),
    (17, "Bali", "Denpasar", -8.6705, 115.2126, "Bali-Nusa Tenggara"),
    (18, "Nusa Tenggara Barat", "Mataram", -8.5833, 116.1167, "Bali-Nusa Tenggara"),
    (19, "Nusa Tenggara Timur", "Kupang", -10.1772, 123.6070, "Bali-Nusa Tenggara"),
    (20, "Kalimantan Barat", "Pontianak", -0.0263, 109.3425, "Kalimantan"),
    (21, "Kalimantan Selatan", "Banjarmasin", -3.3186, 114.5944, "Kalimantan"),
    (22, "Kalimantan Tengah", "Palangka Raya", -2.2161, 113.9135, "Kalimantan"),
    (23, "Kalimantan Timur", "Samarinda", -0.5022, 117.1536, "Kalimantan"),
    (24, "Kalimantan Utara", "Tanjung Selor", 2.8375, 117.3653, "Kalimantan"),
    (25, "Gorontalo", "Gorontalo", 0.5435, 123.0568, "Sulawesi"),
    (26, "Sulawesi Selatan", "Makassar", -5.1477, 119.4327, "Sulawesi"),
    (27, "Sulawesi Tenggara", "Kendari", -3.9985, 122.5130, "Sulawesi"),
    (28, "Sulawesi Tengah", "Palu", -0.8917, 119.8707, "Sulawesi"),
    (29, "Sulawesi Utara", "Manado", 1.4748, 124.8421, "Sulawesi"),
    (30, "Sulawesi Barat", "Mamuju", -2.6748, 118.8885, "Sulawesi"),
    (31, "Maluku", "Ambon", -3.6954, 128.1814, "Maluku"),
    (32, "Maluku Utara", "Ternate", 0.7893, 127.3842, "Maluku"),
    (33, "Papua", "Jayapura", -2.5337, 140.7181, "Papua"),
    (34, "Papua Barat", "Manokwari", -0.8615, 134.0620, "Papua"),
]


def province_reference() -> pd.DataFrame:
    return pd.DataFrame(_PROVINCES, columns=["province_id", "province", "city", "lat", "lon", "island"])


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def distance_matrix(ref: pd.DataFrame | None = None) -> pd.DataFrame:
    ref = province_reference() if ref is None else ref
    lat, lon = ref["lat"].to_numpy(), ref["lon"].to_numpy()
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return pd.DataFrame(d, index=ref["province_id"], columns=ref["province_id"])


def nearest_neighbors(k: int = 3, ref: pd.DataFrame | None = None) -> dict[int, list[int]]:
    """k provinsi terdekat (jarak antar ibu kota) — dipakai sebagai 'tetangga' pasar."""
    d = distance_matrix(ref)
    out = {}
    for pid in d.index:
        order = d.loc[pid].drop(pid).sort_values()
        out[int(pid)] = [int(x) for x in order.index[:k]]
    return out


# -----------------------------------------------------------------------------
# Kalender hari besar (tanggal resmi pemerintah RI / hasil sidang isbat).
# 2027+ memakai estimasi kalender Umm al-Qura (paket hijridate), selisih maks ±1 hari.
# -----------------------------------------------------------------------------
_OFFICIAL = {
    # tahun: (1 Ramadan, Idulfitri / 1 Syawal, Iduladha / 10 Zulhijah)
    2017: ("2017-05-27", "2017-06-25", "2017-09-01"),
    2018: ("2018-05-17", "2018-06-15", "2018-08-22"),
    2019: ("2019-05-06", "2019-06-05", "2019-08-11"),
    2020: ("2020-04-24", "2020-05-24", "2020-07-31"),
    2021: ("2021-04-13", "2021-05-13", "2021-07-20"),
    2022: ("2022-04-03", "2022-05-02", "2022-07-10"),
    2023: ("2023-03-23", "2023-04-22", "2023-06-29"),
    2024: ("2024-03-12", "2024-04-10", "2024-06-17"),
    2025: ("2025-03-01", "2025-03-31", "2025-06-06"),
    2026: ("2026-02-19", "2026-03-21", "2026-05-27"),
}


def _hijri_estimate(year: int) -> tuple[str, str, str] | None:
    try:
        from hijridate import Gregorian, Hijri
    except ImportError:  # paket opsional
        return None
    out = {}
    h_year = Gregorian(year, 6, 1).to_hijri().year
    for hy in (h_year - 1, h_year, h_year + 1):
        for key, (m, d) in {"ramadan": (9, 1), "idulfitri": (10, 1), "iduladha": (12, 10)}.items():
            g = Hijri(hy, m, d).to_gregorian()
            if g.year == year:
                out[key] = g.isoformat()
    if len(out) == 3:
        return out["ramadan"], out["idulfitri"], out["iduladha"]
    return None


def holiday_table(start_year: int = 2017, end_year: int = 2028) -> pd.DataFrame:
    rows = []
    for y in range(start_year, end_year + 1):
        src = "resmi"
        dates = _OFFICIAL.get(y)
        if dates is None:
            dates, src = _hijri_estimate(y), "estimasi-hijri"
        if dates is not None:
            rows += [
                (pd.Timestamp(dates[0]), "awal_ramadan", src),
                (pd.Timestamp(dates[1]), "idulfitri", src),
                (pd.Timestamp(dates[2]), "iduladha", src),
            ]
        rows += [(pd.Timestamp(date(y, 12, 25)), "natal", "tetap"),
                 (pd.Timestamp(date(y, 1, 1)), "tahun_baru", "tetap")]
    return pd.DataFrame(rows, columns=["date", "event", "source"]).sort_values("date").reset_index(drop=True)


def calendar_features(dates: pd.DatetimeIndex, holidays: pd.DataFrame, cap_days: int = 120) -> pd.DataFrame:
    """Fitur kalender per tanggal (tidak memakai info masa depan yang tak diketahui:
    tanggal hari besar sudah diumumkan jauh hari)."""
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=dates)
    for ev in ["awal_ramadan", "idulfitri", "iduladha", "natal", "tahun_baru"]:
        ev_dates = np.sort(holidays.loc[holidays["event"] == ev, "date"].to_numpy(dtype="datetime64[ns]"))
        d64 = dates.to_numpy(dtype="datetime64[ns]")
        i_next = np.searchsorted(ev_dates, d64, side="left")    # hari-H dihitung "0 hari lagi"
        i_prev = np.searchsorted(ev_dates, d64, side="right")   # hari-H dihitung "0 hari sejak"
        nxt = np.where(i_next < len(ev_dates), ev_dates[np.minimum(i_next, len(ev_dates) - 1)], np.datetime64("NaT"))
        prv = np.where(i_prev > 0, ev_dates[np.maximum(i_prev - 1, 0)], np.datetime64("NaT"))
        to_next = (nxt - d64) / np.timedelta64(1, "D")
        since_prev = (d64 - prv) / np.timedelta64(1, "D")
        out[f"days_to_{ev}"] = np.clip(to_next, 0, cap_days)
        out[f"days_since_{ev}"] = np.clip(since_prev, 0, cap_days)
    # Ramadan = dari awal_ramadan sampai H-1 Idulfitri
    out["is_ramadan"] = ((out["days_since_awal_ramadan"] < 30) & (out["days_to_idulfitri"] <= 30)
                         & (out["days_to_idulfitri"] > 0)).astype("int8")
    out["is_nataru"] = (((dates.month == 12) & (dates.day >= 15)) | ((dates.month == 1) & (dates.day <= 7))).astype("int8")
    doy = dates.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["dow"] = dates.dayofweek.astype("int8")
    return out.drop(columns=["days_since_natal", "days_since_tahun_baru"])
