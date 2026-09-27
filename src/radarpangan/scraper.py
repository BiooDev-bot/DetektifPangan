"""Scraper PIHPS Bank Indonesia — https://www.bi.go.id/hargapangan

Cara kerja (hasil inspeksi halaman "Tabel Harga Berdasarkan Komoditas"):
  - Halaman web memanggil endpoint JSON di  {base_url}/GetGridDataKomoditas
    dengan parameter: price_type_id, comcat_id, province_id, regency_id,
    showKota, showPasar, tipe_laporan, start_date, end_date (format YYYY-MM-DD).
  - Respons: {"data": [{"no": "I", "name": "Semua Provinsi", "level": 0,
                        "15/09/2026": "67,000", ...}, {"name": "Aceh", "level": 1, ...}]}
      * level 0 = rata-rata nasional ("Semua Provinsi"), level 1 = provinsi
      * kunci tanggal format DD/MM/YYYY, hanya hari kerja
      * harga berupa string dengan pemisah ribuan koma; "-" = tidak ada data
  - Referensi: GetRefPriceType, GetRefCommodityAndCategory, GetRefProvince, GetRefRegency.

Strategi scraping:
  - 1 request = 1 jenis pasar x 1 varian komoditas x 1 bulan x semua provinsi.
    (Rentang panjang membuat server sangat lambat: 1 bulan ~2 dtk, 1 tahun ~45 dtk.)
  - Tiap chunk disimpan apa adanya (JSON.gz) di data/raw/pihps -> bisa di-resume,
    dan N bulan terakhir selalu diambil ulang karena data terbaru bisa direvisi.
"""
from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga"
PRICE_TYPES = {1: "Pasar Tradisional", 2: "Pasar Modern", 3: "Pedagang Besar", 4: "Produsen"}
NATIONAL_NAME = "Semua Provinsi"
NATIONAL_ID = 0
_META_KEYS = {"no", "name", "level"}


# -----------------------------------------------------------------------------
# HTTP client
# -----------------------------------------------------------------------------
class PIHPSClient:
    """Client sopan untuk API PIHPS (retry + backoff + jeda antar-request)."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = 180,
        max_retries: int = 5,
        sleep_seconds: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        # satu Session per thread (requests.Session tidak dijamin thread-safe)
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (RadarPangan research scraper; USB 2026 Data Mining)",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": "https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalKomoditas",
                }
            )
            self._local.session = s
        return self._local.session

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{endpoint}"
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    payload = r.json()
                    if self.sleep_seconds:
                        time.sleep(self.sleep_seconds)
                    return payload
                last_err = f"HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as e:  # ValueError = JSON rusak
                last_err = repr(e)
            wait = min(60, 2**attempt)
            log.warning("Percobaan %d/%d gagal (%s) -> tunggu %ds | %s %s",
                        attempt, self.max_retries, last_err, wait, endpoint, params)
            time.sleep(wait)
        raise RuntimeError(f"Gagal mengambil {url} params={params}: {last_err}")

    # --- referensi -----------------------------------------------------------
    def price_types(self) -> pd.DataFrame:
        return pd.DataFrame(self._get("GetRefPriceType")["data"])

    def commodities(self) -> pd.DataFrame:
        """Kategori (cat_1..cat_10) dan varian (com_1..com_21) dalam satu tabel pohon."""
        df = pd.DataFrame(self._get("GetRefCommodityAndCategory")["data"])
        df["name"] = df["name"].str.strip()
        return df

    def provinces(self) -> pd.DataFrame:
        df = pd.DataFrame(self._get("GetRefProvince")["data"])
        df["name"] = df["name"].str.strip()
        return df.sort_values("id").reset_index(drop=True)

    def regencies(self, province_id: int, price_type_id: int = 1) -> pd.DataFrame:
        payload = self._get("GetRefRegency", {"price_type_id": price_type_id, "ref_prov_id": province_id})
        return pd.DataFrame(payload.get("data", []))

    # --- data harga ----------------------------------------------------------
    def grid_komoditas(
        self,
        price_type_id: int,
        comcat_id: str,
        start: date | str,
        end: date | str,
        province_id: str = "",
        regency_id: str = "",
        show_kota: bool = False,
        show_pasar: bool = False,
        tipe_laporan: int = 1,
    ) -> list[dict]:
        """Tabel harga 1 komoditas: baris = nasional + provinsi, kolom = tanggal.

        tipe_laporan: 1 = harian, 4 = mingguan, 5 = bulanan.
        """
        params = {
            "price_type_id": price_type_id,
            "comcat_id": comcat_id,
            "province_id": province_id,
            "regency_id": regency_id,
            "showKota": str(show_kota).lower(),
            "showPasar": str(show_pasar).lower(),
            "tipe_laporan": tipe_laporan,
            "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
        }
        return self._get("GetGridDataKomoditas", params).get("data", [])


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
def parse_price(value) -> float:
    """'67,000' -> 67000.0 ; '-' / '' / None / 0 -> NaN."""
    if value is None:
        return np.nan
    s = str(value).strip().replace(",", "")
    if s in ("", "-"):
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return v if v > 0 else np.nan


def parse_grid(records: list[dict], price_type_id: int, commodity_id: str) -> pd.DataFrame:
    """Ubah respons grid (wide) menjadi format panjang (long)."""
    rows = []
    for rec in records:
        name = str(rec.get("name", "")).strip()
        level = rec.get("level")
        for key, val in rec.items():
            if key in _META_KEYS:
                continue
            rows.append((key, name, level, val))
    df = pd.DataFrame(rows, columns=["date_str", "region_name", "level", "value_raw"])
    df["date"] = pd.to_datetime(df["date_str"], format="%d/%m/%Y", errors="coerce")
    df["price"] = df["value_raw"].map(parse_price).astype("float64")
    df["price_type_id"] = int(price_type_id)
    df["commodity_id"] = commodity_id
    return df[["date", "price_type_id", "commodity_id", "region_name", "level", "price"]]


# -----------------------------------------------------------------------------
# Scraping dengan cache per bulan
# -----------------------------------------------------------------------------
def month_chunks(start: date | str, end: date | str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    chunks = []
    cur = start.replace(day=1)
    while cur <= end:
        nxt = cur + pd.offsets.MonthBegin(1)
        chunks.append((max(cur, start), min(nxt - pd.Timedelta(days=1), end)))
        cur = nxt
    return chunks


def chunk_path(raw_dir: Path, price_type_id: int, commodity_id: str, month: pd.Timestamp) -> Path:
    return Path(raw_dir) / f"pt{price_type_id}" / commodity_id / f"{month:%Y-%m}.json.gz"


def _write_json_gz(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)  # atomic: file tidak pernah setengah jadi


def _read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_reference_tables(client: PIHPSClient, external_dir: Path) -> dict[str, pd.DataFrame]:
    """Simpan tabel referensi PIHPS (jenis pasar, komoditas, provinsi) ke data/external."""
    external_dir = Path(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    refs = {
        "pihps_price_types": client.price_types(),
        "pihps_commodities": client.commodities(),
        "pihps_provinces": client.provinces(),
    }
    for name, df in refs.items():
        df.to_csv(external_dir / f"{name}.csv", index=False)
    return refs


def scrape_pihps(
    client: PIHPSClient,
    raw_dir: Path,
    price_types: list[int],
    commodity_ids: list[str],
    start: date | str,
    end: date | str | None = None,
    workers: int = 2,
    refresh_recent_months: int = 2,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Ambil semua kombinasi (jenis pasar x varian x bulan) yang belum ada di cache.

    Return: tabel ringkasan status per chunk (ok / cached / error).
    """
    raw_dir = Path(raw_dir)
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    chunks = month_chunks(start, end)
    recent_cutoff = (end.replace(day=1) - pd.DateOffset(months=max(refresh_recent_months - 1, 0)))

    tasks, summary = [], []
    for pt in price_types:
        for com in commodity_ids:
            for c_start, c_end in chunks:
                month = c_start.replace(day=1)
                path = chunk_path(raw_dir, pt, com, month)
                if path.exists() and month < recent_cutoff:
                    summary.append({"price_type_id": pt, "commodity_id": com, "month": month, "status": "cached"})
                    continue
                tasks.append((pt, com, c_start, c_end, month, path))

    log.info("Chunk total=%d | dari cache=%d | perlu diambil=%d", len(tasks) + len(summary), len(summary), len(tasks))

    def _run(task):
        pt, com, c_start, c_end, month, path = task
        t0 = time.time()
        records = client.grid_komoditas(pt, com, c_start, c_end)
        _write_json_gz(
            path,
            {
                "meta": {
                    "price_type_id": pt, "commodity_id": com,
                    "start_date": f"{c_start:%Y-%m-%d}", "end_date": f"{c_end:%Y-%m-%d}",
                    "scraped_at": datetime.now().isoformat(timespec="seconds"),
                    "source": f"{client.base_url}/GetGridDataKomoditas",
                },
                "data": records,
            },
        )
        return {"price_type_id": pt, "commodity_id": com, "month": month, "status": "ok",
                "n_rows": len(records), "seconds": round(time.time() - t0, 2)}

    bar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(tasks), desc="Scraping PIHPS", unit="chunk")
        except ImportError:
            bar = None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_run, t): t for t in tasks}
        for fut in as_completed(futures):
            pt, com, _, _, month, _ = futures[fut]
            try:
                summary.append(fut.result())
            except Exception as e:  # jangan hentikan seluruh proses karena 1 chunk gagal
                log.error("Chunk gagal pt=%s %s %s: %s", pt, com, f"{month:%Y-%m}", e)
                summary.append({"price_type_id": pt, "commodity_id": com, "month": month,
                                "status": "error", "error": str(e)})
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    return pd.DataFrame(summary)


def load_raw_cache(
    raw_dir: Path,
    provinces: pd.DataFrame,
    commodities: pd.DataFrame,
    price_types: list[int] | None = None,
) -> pd.DataFrame:
    """Gabungkan semua chunk JSON.gz jadi satu tabel panjang yang rapi.

    Kolom: date, price_type_id, price_type, commodity_id, commodity, cat_id, category,
           province_id, province, price
    """
    raw_dir = Path(raw_dir)
    frames = []
    for path in sorted(raw_dir.glob("pt*/*/*.json.gz")):
        pt = int(path.parts[-3][2:])
        if price_types is not None and pt not in price_types:
            continue
        payload = _read_json_gz(path)
        if payload.get("data"):
            frames.append(parse_grid(payload["data"], pt, path.parts[-2]))
    if not frames:
        raise FileNotFoundError(f"Tidak ada cache di {raw_dir}. Jalankan scraping dulu.")
    df = pd.concat(frames, ignore_index=True)

    prov_map = dict(zip(provinces["name"].str.strip(), provinces["id"].astype(int)))
    prov_map[NATIONAL_NAME] = NATIONAL_ID
    df["province_id"] = df["region_name"].map(prov_map)
    unknown = df.loc[df["province_id"].isna(), "region_name"].unique()
    if len(unknown):
        log.warning("Nama wilayah tidak dikenal (dibuang): %s", list(unknown)[:10])
    df = df.dropna(subset=["province_id", "date"])
    df["province_id"] = df["province_id"].astype(int)
    df["province"] = df["region_name"].replace({NATIONAL_NAME: "Nasional"})

    var = commodities[commodities["id"].str.startswith("com_")][["id", "name", "cat_id"]]
    cat = commodities[commodities["id"].str.startswith("cat_")][["id", "name"]]
    var = var.merge(cat.rename(columns={"id": "cat_id", "name": "category"}), on="cat_id", how="left")
    var = var.rename(columns={"id": "commodity_id", "name": "commodity"})
    df = df.merge(var, on="commodity_id", how="left")
    df["price_type"] = df["price_type_id"].map(PRICE_TYPES)

    df = (df.sort_values(["price_type_id", "commodity_id", "province_id", "date"])
            .drop_duplicates(["price_type_id", "commodity_id", "province_id", "date"], keep="last"))
    cols = ["date", "price_type_id", "price_type", "commodity_id", "commodity", "cat_id",
            "category", "province_id", "province", "price"]
    return df[cols].reset_index(drop=True)
