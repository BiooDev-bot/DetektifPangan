"""EVALUATION/DEPLOYMENT — penjelasan model dengan SHAP.

Nilai SHAP dihitung dengan TreeSHAP bawaan LightGBM (booster.predict(pred_contrib=True)),
algoritma yang sama dengan shap.TreeExplainer tapi jauh lebih cepat dan tanpa masalah versi.
Satuan SHAP = log-odds: positif -> mendorong risiko naik, negatif -> menurunkan risiko.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

GROUP_LABELS = {
    "harga": "Dinamika harga sendiri",
    "nasional": "Kondisi nasional",
    "rantai_pasok": "Rantai pasok (produsen/pedagang besar)",
    "tetangga": "Provinsi tetangga",
    "pemimpin": "Provinsi pemimpin (lead-lag)",
    "kalender": "Kalender hari besar",
    "cuaca": "Anomali curah hujan",
    "identitas": "Identitas komoditas/provinsi",
}

_EVENT = {"awal_ramadan": "awal Ramadan", "idulfitri": "Idulfitri", "iduladha": "Iduladha",
          "natal": "Natal", "tahun_baru": "Tahun Baru"}


def _num(x: float, fmt: str = "+.1f") -> str:
    return format(x, fmt).replace(".", ",")


def _pct(v: float) -> str:
    """Log-return -> persen bertanda, desimal koma (mis. +9,7%)."""
    return _num(np.expm1(v) * 100) + "%"


def _vs(v: float, below: str, above: str) -> str:
    """Log-selisih -> 'x% <below>' bila negatif, 'x% <above>' bila positif."""
    x = _num(abs(np.expm1(v)) * 100, ".1f") + "%"
    return f"{x} {below}" if v < 0 else f"{x} {above}"


def _pp(v: float) -> str:
    """Selisih proporsi -> poin persen bertanda."""
    return _num(v * 100) + " poin"


def describe_feature(name: str, v: float) -> str:
    """Kalimat singkat (bahasa awam) untuk satu fitur & nilainya."""
    if v is None or not np.isfinite(v):
        return f"{name}: data tidak tersedia"
    n = name
    if n.startswith("ret_") and n.split("_")[1].isdigit():
        w = n.split("_")[1]
        tag = n.split("_")[2] if len(n.split("_")) > 2 else ""
        who = {"prod": "Harga produsen", "whole": "Harga pedagang besar", "modern": "Harga pasar modern"}.get(tag, "Harga")
        return f"{who} {_pct(v)} dalam {w} hari terakhir"
    table = {
        "ret_7_prev": lambda: f"Minggu sebelumnya harga {_pct(v)}",
        "dist_max_90": lambda: ("Harga sedang di puncak 90 hari terakhir" if v > -0.005 else
                                "Harga " + _num(abs(np.expm1(v)) * 100, ".1f") + "% di bawah puncak 90 hari terakhir"),
        "dist_min_90": lambda: "Harga " + _num(np.expm1(v) * 100, ".1f") + "% di atas titik terendah 90 hari terakhir",
        "rel_365": lambda: "Harga " + _vs(v, "di bawah rata-rata setahun terakhir", "di atas rata-rata setahun terakhir"),
        "ret_yoy": lambda: "Harga " + _vs(v, "lebih rendah dari tahun lalu", "lebih tinggi dari tahun lalu"),
        "seas_ret_14_ly": lambda: f"Tahun lalu pada periode ini harga {_pct(v)} dalam 14 hari (pola musiman)",
        "r14_now": lambda: f"Harga sudah berubah {_num(v * 100)}% dalam 14 hari terakhir",
        "gap_to_threshold": lambda: f"Tinggal {_num(v * 100, '.1f')} poin persen lagi menuju ambang lonjakan",
        "days_since_spike": lambda: f"{v:.0f} hari sejak lonjakan terakhir",
        "n_onsets_365": lambda: f"{v:.0f} kali lonjakan dalam setahun terakhir",
        "nat_ret_7": lambda: f"Harga nasional {_pct(v)} (7 hari)",
        "nat_ret_14": lambda: f"Harga nasional {_pct(v)} (14 hari)",
        "rel_nat": lambda: (f"Harga {_num(abs(np.expm1(v)) * 100, '.1f')}% di bawah rata-rata nasional (berpotensi menyusul naik)"
                            if v < 0 else f"Harga {_num(np.expm1(v) * 100, '.1f')}% di atas rata-rata nasional"),
        "rel_nat_chg14": lambda: f"Selisih dengan harga nasional berubah {_pct(v)} (14 hari)",
        "nat_spike_share": lambda: f"{v * 100:.0f}% provinsi lain sedang mengalami lonjakan komoditas ini",
        "nat_spike_share_chg7": lambda: f"Porsi provinsi yang melonjak berubah {_pp(v)} dalam 7 hari",
        "nat_ret_14_prod": lambda: f"Harga produsen nasional {_pct(v)} (14 hari)",
        "nat_ret_14_whole": lambda: f"Harga pedagang besar nasional {_pct(v)} (14 hari)",
        "nat_ret_14_modern": lambda: f"Harga pasar modern nasional {_pct(v)} (14 hari)",
        "nat_margin_cons_prod": lambda: "Secara nasional harga konsumen " + _vs(v, "di bawah harga produsen", "di atas harga produsen"),
        "nat_margin_cons_prod_chg14": lambda: f"Margin konsumen-produsen nasional berubah {_pp(v)} (14 hari)",
        "margin_cons_prod": lambda: "Harga konsumen " + _vs(v, "di bawah harga produsen", "di atas harga produsen"),
        "margin_cons_prod_chg14": lambda: f"Margin konsumen-produsen berubah {_pp(v)} (14 hari)",
        "margin_cons_prod_z90": lambda: f"Margin konsumen-produsen {_num(v)} SD dari normalnya",
        "margin_cons_whole": lambda: "Harga konsumen " + _vs(v, "di bawah harga pedagang besar", "di atas harga pedagang besar"),
        "margin_cons_whole_chg14": lambda: f"Margin konsumen-pedagang besar berubah {_pp(v)} (14 hari)",
        "margin_whole_prod": lambda: "Harga pedagang besar " + _vs(v, "di bawah harga produsen", "di atas harga produsen"),
        "gap_modern": lambda: "Harga pasar modern " + _vs(v, "lebih murah dari pasar tradisional", "lebih mahal dari pasar tradisional"),
        "nbr_ret_7_mean": lambda: f"Harga di provinsi tetangga berubah rata-rata {_pct(v)} dalam 7 hari",
        "nbr_ret_14_mean": lambda: f"Harga di provinsi tetangga berubah rata-rata {_pct(v)} dalam 14 hari",
        "nbr_ret_7_max": lambda: f"Salah satu provinsi tetangga berubah {_pct(v)} dalam 7 hari",
        "nbr_spike_share": lambda: f"{v * 100:.0f}% provinsi tetangga sedang mengalami lonjakan",
        "nbr_rel_price": lambda: "Harga " + _vs(v, "lebih murah dari provinsi tetangga (berpotensi menyusul naik)", "lebih mahal dari provinsi tetangga"),
        "lead_ret_7_mean": lambda: f"Provinsi \"pemimpin\" (biasanya naik duluan) berubah rata-rata {_pct(v)} dalam 7 hari",
        "lead_ret_14_mean": lambda: f"Provinsi \"pemimpin\" (biasanya naik duluan) berubah rata-rata {_pct(v)} dalam 14 hari",
        "lead_spike_share": lambda: f"{v * 100:.0f}% provinsi \"pemimpin\" sedang mengalami lonjakan",
        "is_ramadan": lambda: "Sedang bulan Ramadan" if v >= 0.5 else "Di luar bulan Ramadan",
        "is_nataru": lambda: "Periode Natal & Tahun Baru" if v >= 0.5 else "Di luar periode Nataru",
        "doy_sin": lambda: "Pola musiman (waktu dalam setahun)",
        "doy_cos": lambda: "Pola musiman (waktu dalam setahun)",
        "dow": lambda: "Hari dalam seminggu",
        "commodity_code": lambda: "Karakter komoditas (riwayat volatilitas)",
        "category_code": lambda: "Karakter kelompok komoditas",
        "province_code": lambda: "Karakter provinsi (riwayat volatilitas)",
    }
    if n in table:
        return table[n]()
    if n.startswith("vol_"):
        return f"Harga sedang bergejolak: rata-rata ±{_num(v * 100, '.1f')}% per hari ({n.split('_')[1]} hari terakhir)"
    if n.startswith("z_"):
        return f"Harga {_num(v)} SD dari rata-rata {n.split('_')[1]} hari terakhir"
    if n.startswith("days_to_"):
        ev = _EVENT.get(n[len("days_to_"):], n)
        return f"H-{v:.0f} menuju {ev}" if v < 120 else f"Masih lebih dari 4 bulan menuju {ev} (pola musiman)"
    if n.startswith("days_since_"):
        ev = _EVENT.get(n[len("days_since_"):], n)
        return f"{v:.0f} hari setelah {ev}" if v < 120 else f"Sudah lebih dari 4 bulan setelah {ev} (pola musiman)"
    if n.startswith("rain_"):
        w = n.split("_")[1]
        where = " di sentra produksi" if n.endswith("_sentra") else ""
        lag = next((p[3:] for p in n.split("_") if p.startswith("lag")), None)
        when = f" ({lag}–{int(lag) + int(w)} hari lalu)" if lag else ""
        return f"Curah hujan {w} hari{where}{when} {_num(v)} SD dari normal"
    return f"{n} = {v:.3g}"


def lgbm_shap(booster, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (shap_values [n x p], expected_value [n])."""
    it = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else None
    contrib = booster.predict(X, num_iteration=it, pred_contrib=True)
    return contrib[:, :-1], contrib[:, -1]


def importance_table(shap_values: np.ndarray, feature_names: list[str], groups: dict[str, str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    imp["group"] = imp["feature"].map(groups).fillna("lainnya")
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_importance(imp: pd.DataFrame) -> pd.DataFrame:
    g = imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
    out = g.rename("mean_abs_shap").reset_index()
    out["share"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    out["label"] = out["group"].map(GROUP_LABELS).fillna(out["group"])
    return out


def top_reasons(shap_row: np.ndarray, x_row: pd.Series, feature_names: list[str], k: int = 3,
                skip: tuple[str, ...] = ("commodity_code", "category_code", "province_code", "dow")) -> list[dict]:
    """k fitur yang paling MENAIKKAN risiko untuk satu baris prediksi."""
    order = np.argsort(-shap_row)
    out, seen = [], set()
    for j in order:
        f = feature_names[j]
        if shap_row[j] <= 0:
            break
        v = float(x_row[f])
        if f in skip or f.startswith("doy_") or not np.isfinite(v):
            continue                                    # nilai kosong bukan alasan yang bisa dijelaskan
        fam = re.sub(r"_\d+", "", f)                     # satu alasan per jenis sinyal (vol_14 ~ vol_30)
        if fam in seen:
            continue
        seen.add(fam)
        out.append({"feature": f, "shap": float(shap_row[j]), "value": v, "text": describe_feature(f, v)})
        if len(out) == k:
            break
    return out
