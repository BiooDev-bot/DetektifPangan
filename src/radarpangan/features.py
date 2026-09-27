"""DATA PREPARATION — rekayasa fitur untuk model peringatan dini.

Semua fitur pada tanggal t HANYA memakai informasi yang sudah tersedia pada t
(jendela bergulir ke belakang, shift positif). Kelompok fitur:

  harga        : kenaikan 1-28 hari, volatilitas, z-score, jarak ke puncak/lembah 90 hari,
                 kenaikan tahunan, pola musiman tahun lalu, riwayat lonjakan
  nasional     : kenaikan harga nasional, posisi harga provinsi vs nasional
  rantai_pasok : harga produsen / pedagang besar / pasar modern dan selisihnya (margin)
  tetangga     : kenaikan & status lonjak di 3 provinsi terdekat
  pemimpin     : kenaikan di provinsi "pemimpin" hasil jaringan lead-lag (Granger)
  kalender     : jarak ke Ramadan, Idulfitri, Iduladha, Natal, Tahun Baru
  cuaca        : anomali curah hujan 7/30/90 hari (provinsi sendiri & sentra produksi)
  identitas    : komoditas, kategori, provinsi (kategori untuk LightGBM)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labeling import days_since

LABEL_COLS = ["y", "eligible", "spike_now", "onset_now"]
ID_COLS = ["date", "commodity_id", "province_id"]
CATEGORICAL = ["commodity_code", "category_code", "province_code"]


class FeatureAssembler:
    """Kumpulkan fitur dari berbagai granularitas langsung ke baris terpilih (hemat memori)."""

    def __init__(self, dates: pd.DatetimeIndex, columns: pd.MultiIndex, row_mask: np.ndarray):
        self.dates, self.columns = dates, columns
        self.ti, self.ci = np.nonzero(row_mask)
        self.com_codes, self.com_levels = pd.factorize(columns.get_level_values(0))
        self.prov_codes, self.prov_levels = pd.factorize(columns.get_level_values(1))
        self.data: dict[str, np.ndarray] = {}
        self.groups: dict[str, str] = {}

    def _put(self, name, values, group):
        self.data[name] = values.astype("float32", copy=False)
        self.groups[name] = group

    def series(self, name: str, frame: pd.DataFrame, group: str):
        arr = frame.reindex(index=self.dates, columns=self.columns).to_numpy(dtype="float32")
        self._put(name, arr[self.ti, self.ci], group)

    def array(self, name: str, arr: np.ndarray, group: str):
        self._put(name, np.asarray(arr, dtype="float32")[self.ti, self.ci], group)

    def by_date(self, name: str, values: pd.Series, group: str):
        v = values.reindex(self.dates).to_numpy(dtype="float32")
        self._put(name, v[self.ti], group)

    def by_date_province(self, name: str, frame: pd.DataFrame, group: str):
        arr = frame.reindex(index=self.dates, columns=self.prov_levels).to_numpy(dtype="float32")
        self._put(name, arr[self.ti, self.prov_codes[self.ci]], group)

    def by_date_commodity(self, name: str, frame: pd.DataFrame, group: str):
        arr = frame.reindex(index=self.dates, columns=self.com_levels).to_numpy(dtype="float32")
        self._put(name, arr[self.ti, self.com_codes[self.ci]], group)

    def frame(self) -> pd.DataFrame:
        out = pd.DataFrame({
            "date": self.dates[self.ti],
            "commodity_id": np.asarray(self.columns.get_level_values(0))[self.ci],
            "province_id": np.asarray(self.columns.get_level_values(1)).astype("int16")[self.ci],
        })
        return pd.concat([out, pd.DataFrame(self.data)], axis=1)


def _broadcast(nat: pd.DataFrame, cols: pd.MultiIndex) -> pd.DataFrame:
    """Sebarkan panel (tanggal x komoditas) ke semua kolom [komoditas, provinsi]."""
    b = nat.reindex(columns=cols.get_level_values(0))
    b.columns = cols
    return b


def _to3d(frame: pd.DataFrame, coms, provs) -> np.ndarray:
    return frame.to_numpy(dtype="float64").reshape(len(frame), len(coms), len(provs))


def _nbr_mean(X3: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Rata-rata tetangga (abaikan NaN). A[p, q] = bobot tetangga q untuk provinsi p."""
    M = np.isfinite(X3).astype("float64")
    num = np.einsum("tcq,pq->tcp", np.nan_to_num(X3), A)
    den = np.einsum("tcq,pq->tcp", M, A)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _nbr_mean_by_com(X3: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Seperti _nbr_mean tapi bobot berbeda per komoditas: B[c, p, q]."""
    M = np.isfinite(X3).astype("float64")
    num = np.einsum("tcq,cpq->tcp", np.nan_to_num(X3), B)
    den = np.einsum("tcq,cpq->tcp", M, B)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def build_features(
    panels: dict[int, pd.DataFrame],
    labels: dict[str, pd.DataFrame],
    thresholds: pd.Series,
    calendar: pd.DataFrame,
    rain: dict[str, pd.DataFrame] | None,
    neighbors: dict[int, list[int]],
    leaders: dict[tuple[str, int], list[int]] | None,
    commodity_to_cat: dict[str, str],
    cfg: dict,
    province_names: dict[int, str] | None = None,
    min_date=None,
    weekdays_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (tabel fitur long [date, commodity_id, province_id, fitur..., label...], {fitur: kelompok})."""
    fcfg = cfg["features"]
    P = panels[cfg["target"]["price_type"]]
    dates = P.index
    coms = sorted(P.columns.get_level_values(0).unique(), key=lambda c: int(c.split("_")[1]))
    provs = sorted(p for p in P.columns.get_level_values(1).unique() if p != 0)
    cols = pd.MultiIndex.from_product([coms, provs], names=["commodity_id", "province_id"])

    L_all = np.log(P)
    L = L_all.reindex(columns=cols)
    nat = L_all.xs(0, axis=1, level=1).reindex(columns=coms) if 0 in P.columns.get_level_values(1) else None

    # --- baris yang dipakai: hari kerja, lewat masa warm-up, harga tersedia ---------------------
    keep = L.notna().to_numpy()
    if weekdays_only:
        keep &= (dates.dayofweek < 5)[:, None]
    md = pd.Timestamp(min_date or fcfg["min_date"])
    keep &= (dates >= md)[:, None]
    A = FeatureAssembler(dates, cols, keep)

    # --- 1. dinamika harga -------------------------------------------------------------------
    for w in fcfg["return_windows"]:
        A.series(f"ret_{w}", L - L.shift(w), "harga")
    A.series("ret_7_prev", L.shift(7) - L.shift(14), "harga")
    d1 = L.diff()
    for w in fcfg["vol_windows"]:
        A.series(f"vol_{w}", d1.rolling(w, min_periods=max(3, w // 2)).std(), "harga")
    for w in fcfg["zscore_windows"]:
        m = L.rolling(w, min_periods=w // 2).mean()
        s = L.rolling(w, min_periods=w // 2).std()
        A.series(f"z_{w}", (L - m) / s.where(s > 1e-9), "harga")
    A.series("dist_max_90", L - L.rolling(90, min_periods=30).max(), "harga")
    A.series("dist_min_90", L - L.rolling(90, min_periods=30).min(), "harga")
    A.series("rel_365", L - L.rolling(365, min_periods=180).mean(), "harga")
    A.series("ret_yoy", L - L.shift(364), "harga")
    # pola musiman: kenaikan 14 hari yang terjadi TAHUN LALU pada jendela ke depan yang sama
    A.series("seas_ret_14_ly", L.shift(364 - 14) - L.shift(364), "harga")
    r14 = labels["r14"].reindex(columns=cols)
    A.series("r14_now", r14, "harga")
    A.series("gap_to_threshold", (-r14).add(thresholds.reindex(cols), axis=1), "harga")
    spike0 = labels["spike"].reindex(columns=cols).fillna(0.0)
    onset = labels["onset"].reindex(columns=cols).fillna(0.0)
    A.array("days_since_spike", days_since(spike0.to_numpy(), 365), "harga")
    A.series("n_onsets_365", onset.rolling(365, min_periods=1).sum(), "harga")

    # --- 2. nasional -------------------------------------------------------------------------
    if nat is not None:
        A.by_date_commodity("nat_ret_7", nat - nat.shift(7), "nasional")
        A.by_date_commodity("nat_ret_14", nat - nat.shift(14), "nasional")
        rel = L - _broadcast(nat, cols)
        A.series("rel_nat", rel, "nasional")
        A.series("rel_nat_chg14", rel - rel.shift(14), "nasional")
    spike_share = spike0.T.groupby(level=0).mean().T                   # porsi provinsi yg sedang lonjak
    A.by_date_commodity("nat_spike_share", spike_share, "nasional")
    A.by_date_commodity("nat_spike_share_chg7", spike_share - spike_share.shift(7), "nasional")

    # --- 3. rantai pasok ---------------------------------------------------------------------
    logs = {}
    for pt, tag in [(4, "prod"), (3, "whole"), (2, "modern")]:
        if pt not in panels:
            continue
        Lx_all = np.log(panels[pt]).reindex(index=dates)
        Lx = Lx_all.reindex(columns=cols)
        logs[tag] = Lx
        A.series(f"ret_7_{tag}", Lx - Lx.shift(7), "rantai_pasok")
        A.series(f"ret_14_{tag}", Lx - Lx.shift(14), "rantai_pasok")
        if 0 in Lx_all.columns.get_level_values(1):
            nx = Lx_all.xs(0, axis=1, level=1).reindex(columns=coms)
            A.by_date_commodity(f"nat_ret_14_{tag}", nx - nx.shift(14), "rantai_pasok")
            if tag == "prod" and nat is not None:
                nm = nat - nx
                A.by_date_commodity("nat_margin_cons_prod", nm, "rantai_pasok")
                A.by_date_commodity("nat_margin_cons_prod_chg14", nm - nm.shift(14), "rantai_pasok")
    if "prod" in logs:
        m = L - logs["prod"]
        A.series("margin_cons_prod", m, "rantai_pasok")
        A.series("margin_cons_prod_chg14", m - m.shift(14), "rantai_pasok")
        mm, ms = m.rolling(90, min_periods=30).mean(), m.rolling(90, min_periods=30).std()
        A.series("margin_cons_prod_z90", (m - mm) / ms.where(ms > 1e-9), "rantai_pasok")
    if "whole" in logs:
        m = L - logs["whole"]
        A.series("margin_cons_whole", m, "rantai_pasok")
        A.series("margin_cons_whole_chg14", m - m.shift(14), "rantai_pasok")
    if "whole" in logs and "prod" in logs:
        A.series("margin_whole_prod", logs["whole"] - logs["prod"], "rantai_pasok")
    if "modern" in logs:
        A.series("gap_modern", logs["modern"] - L, "rantai_pasok")

    # --- 4. tetangga geografis ---------------------------------------------------------------
    pidx = {p: i for i, p in enumerate(provs)}
    Adj = np.zeros((len(provs), len(provs)))
    for p, nb in neighbors.items():
        if p in pidx:
            for q in nb:
                if q in pidx:
                    Adj[pidx[p], pidx[q]] = 1.0
    R7 = _to3d(L - L.shift(7), coms, provs)
    R14 = _to3d(L - L.shift(14), coms, provs)
    S3 = _to3d(spike0, coms, provs)
    L3 = _to3d(L, coms, provs)
    T = len(dates)
    flat = lambda x: x.reshape(T, -1)                                     # noqa: E731
    A.array("nbr_ret_7_mean", flat(_nbr_mean(R7, Adj)), "tetangga")
    A.array("nbr_ret_14_mean", flat(_nbr_mean(R14, Adj)), "tetangga")
    nbr_idx = [[pidx[q] for q in neighbors.get(p, []) if q in pidx] for p in provs]
    kmax = max(len(x) for x in nbr_idx)
    stack = np.full((T, len(coms), len(provs), kmax), np.nan)
    for i, nb in enumerate(nbr_idx):
        for k, q in enumerate(nb):
            stack[:, :, i, k] = R7[:, :, q]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            A.array("nbr_ret_7_max", flat(np.nanmax(stack, axis=3)), "tetangga")
    A.array("nbr_spike_share", flat(_nbr_mean(S3, Adj)), "tetangga")
    A.array("nbr_rel_price", flat(L3 - _nbr_mean(L3, Adj)), "tetangga")
    del stack

    # --- 5. provinsi pemimpin (jaringan lead-lag) --------------------------------------------
    if leaders:
        B = np.zeros((len(coms), len(provs), len(provs)))
        cidx = {c: i for i, c in enumerate(coms)}
        for (c, p), srcs in leaders.items():
            if c in cidx and p in pidx:
                for q in srcs:
                    if q in pidx:
                        B[cidx[c], pidx[p], pidx[q]] = 1.0
        A.array("lead_ret_7_mean", flat(_nbr_mean_by_com(R7, B)), "pemimpin")
        A.array("lead_ret_14_mean", flat(_nbr_mean_by_com(R14, B)), "pemimpin")
        A.array("lead_spike_share", flat(_nbr_mean_by_com(S3, B)), "pemimpin")
    del R7, R14, S3, L3

    # --- 6. kalender -------------------------------------------------------------------------
    for c in calendar.columns:
        A.by_date(c, calendar[c], "kalender")

    # --- 7. cuaca ----------------------------------------------------------------------------
    if rain:
        sentra = fcfg.get("sentra_provinces") or []
        name_to_id = {v: k for k, v in (province_names or {}).items()}
        sentra_ids = [name_to_id[n] for n in sentra if n in name_to_id]
        for name, frame in rain.items():
            A.by_date_province(name, frame, "cuaca")
            if sentra_ids:
                A.by_date(f"{name}_sentra", frame.reindex(columns=sentra_ids).mean(axis=1), "cuaca")
            if name == "rain_30_anom":
                # pengaruh hujan tertunda (siklus tanam-panen): hujan 30-60 & 60-90 hari lalu
                for lag in (30, 60):
                    A.by_date_province(f"{name}_lag{lag}", frame.shift(lag, freq="D"), "cuaca")
                    if sentra_ids:
                        A.by_date(f"{name}_lag{lag}_sentra",
                                  frame.reindex(columns=sentra_ids).mean(axis=1).shift(lag, freq="D"), "cuaca")

    # --- 8. identitas (kategori) -------------------------------------------------------------
    com_code = np.array([int(c.split("_")[1]) for c in cols.get_level_values(0)])
    cat_code = np.array([int(str(commodity_to_cat.get(c, "cat_0")).split("_")[1]) for c in cols.get_level_values(0)])
    prov_code = np.asarray(cols.get_level_values(1)).astype(int)
    A.array("commodity_code", np.broadcast_to(com_code, (T, len(cols))), "identitas")
    A.array("category_code", np.broadcast_to(cat_code, (T, len(cols))), "identitas")
    A.array("province_code", np.broadcast_to(prov_code, (T, len(cols))), "identitas")

    groups = dict(A.groups)

    # --- label (tidak dipakai sebagai fitur!) --------------------------------------------------
    A.series("y", labels["y"].reindex(columns=cols), "label")
    A.series("eligible", labels["eligible"].reindex(columns=cols).astype("float64"), "label")
    A.series("spike_now", labels["spike"].reindex(columns=cols), "label")
    A.series("onset_now", onset, "label")

    df = A.frame()
    for c in CATEGORICAL:
        df[c] = df[c].astype("int16")
    return df, groups


# -----------------------------------------------------------------------------
# Kamus fitur (untuk dokumentasi / data dictionary)
# -----------------------------------------------------------------------------
_DOCS = {
    "ret_7_prev": "Perubahan log harga konsumen minggu sebelumnya (t-14 → t-7)",
    "dist_max_90": "Jarak log harga ke harga tertinggi 90 hari terakhir",
    "dist_min_90": "Jarak log harga ke harga terendah 90 hari terakhir",
    "rel_365": "Log harga relatif terhadap rata-rata 365 hari",
    "ret_yoy": "Perubahan log harga dibanding 364 hari lalu",
    "seas_ret_14_ly": "Perubahan harga 14 hari yang terjadi tahun lalu pada jendela ke depan yang sama (pola musiman)",
    "r14_now": "Kenaikan harga 14 hari saat ini (P_t/P_t-14 - 1)",
    "gap_to_threshold": "Selisih ambang lonjakan dengan kenaikan 14 hari saat ini",
    "days_since_spike": "Hari sejak terakhir berstatus lonjak (maks 365)",
    "n_onsets_365": "Jumlah kejadian lonjakan dalam 365 hari terakhir",
    "nat_ret_7": "Perubahan log harga nasional 7 hari", "nat_ret_14": "Perubahan log harga nasional 14 hari",
    "rel_nat": "Log harga provinsi relatif terhadap harga nasional",
    "rel_nat_chg14": "Perubahan 14 hari dari selisih harga provinsi vs nasional",
    "nat_spike_share": "Porsi provinsi yang sedang lonjak (komoditas sama)",
    "nat_spike_share_chg7": "Perubahan 7 hari porsi provinsi yang sedang lonjak",
    "nat_margin_cons_prod": "Margin log harga konsumen - produsen tingkat nasional",
    "nat_margin_cons_prod_chg14": "Perubahan 14 hari margin konsumen-produsen nasional",
    "margin_cons_prod": "Margin log harga konsumen - produsen (provinsi)",
    "margin_cons_prod_chg14": "Perubahan 14 hari margin konsumen-produsen",
    "margin_cons_prod_z90": "Z-score margin konsumen-produsen terhadap 90 hari terakhir",
    "margin_cons_whole": "Margin log harga konsumen - pedagang besar",
    "margin_cons_whole_chg14": "Perubahan 14 hari margin konsumen-pedagang besar",
    "margin_whole_prod": "Margin log harga pedagang besar - produsen",
    "gap_modern": "Log harga pasar modern - pasar tradisional",
    "nbr_ret_7_mean": "Rata-rata perubahan 7 hari di 3 provinsi terdekat",
    "nbr_ret_14_mean": "Rata-rata perubahan 14 hari di 3 provinsi terdekat",
    "nbr_ret_7_max": "Perubahan 7 hari tertinggi di 3 provinsi terdekat",
    "nbr_spike_share": "Porsi provinsi tetangga yang sedang lonjak",
    "nbr_rel_price": "Log harga relatif terhadap rata-rata tetangga",
    "lead_ret_7_mean": "Rata-rata perubahan 7 hari di provinsi pemimpin (jaringan Granger)",
    "lead_ret_14_mean": "Rata-rata perubahan 14 hari di provinsi pemimpin",
    "lead_spike_share": "Porsi provinsi pemimpin yang sedang lonjak",
    "is_ramadan": "1 jika bulan Ramadan", "is_nataru": "1 jika 15 Des - 7 Jan",
    "doy_sin": "Musiman tahunan (sin hari ke-n)", "doy_cos": "Musiman tahunan (cos hari ke-n)",
    "dow": "Hari dalam minggu (0=Senin)",
    "commodity_code": "Kode varian komoditas (kategori LightGBM)",
    "category_code": "Kode kategori komoditas (kategori LightGBM)",
    "province_code": "Kode provinsi (kategori LightGBM)",
}


def feature_dictionary(groups: dict[str, str]) -> pd.DataFrame:
    rows = []
    for f, g in groups.items():
        d = _DOCS.get(f)
        if d is None:
            parts = f.split("_")
            if f.startswith("ret_") and parts[1].isdigit():
                who = {"prod": "produsen", "whole": "pedagang besar", "modern": "pasar modern"}.get(parts[2] if len(parts) > 2 else "", "konsumen")
                d = f"Perubahan log harga {who} {parts[1]} hari"
            elif f.startswith("nat_ret_14_"):
                d = "Perubahan log harga nasional 14 hari (" + {"prod": "produsen", "whole": "pedagang besar", "modern": "pasar modern"}[parts[3]] + ")"
            elif f.startswith("vol_"):
                d = f"Volatilitas: simpangan baku perubahan log harian {parts[1]} hari"
            elif f.startswith("z_"):
                d = f"Z-score log harga terhadap {parts[1]} hari terakhir"
            elif f.startswith("days_to_"):
                d = f"Hari menuju {f[8:].replace('_', ' ')} berikutnya (maks 120)"
            elif f.startswith("days_since_"):
                d = f"Hari sejak {f[11:].replace('_', ' ')} terakhir (maks 120)"
            elif f.startswith("rain_"):
                lag = next((p[3:] for p in parts if p.startswith("lag")), None)
                d = (f"Anomali (z) curah hujan {parts[1]} hari" + (f", digeser {lag} hari" if lag else "")
                     + (" — rata-rata sentra produksi" if f.endswith("_sentra") else " — provinsi sendiri"))
            else:
                d = f
        rows.append({"fitur": f, "kelompok": g, "deskripsi": d})
    return pd.DataFrame(rows)
