"""Jaringan rambatan harga antarprovinsi: "provinsi mana yang biasanya naik duluan?"

Dua bukti yang saling melengkapi (per varian komoditas):
  1) Granger causality bersyarat (data mingguan):
       r_B(t) ~ r_B(t-1..t-p) + r_Nasional(t-1..t-p)            (model terbatas)
       r_B(t) ~ ... + r_A(t-1..t-p)                             (model lengkap)
     Uji F: apakah kenaikan A minggu-minggu sebelumnya membantu memprediksi B,
     setelah dikontrol tren nasional (supaya gelombang nasional seperti Lebaran tidak
     dianggap rambatan). p-value dikoreksi Benjamini-Hochberg (FDR) per komoditas.
  2) Presedensi kejadian (data harian): dari semua onset lonjakan di A, berapa persen
     diikuti onset di B dalam 1-14 hari? Dibandingkan peluang dasar B -> "lift".

Edge A -> B dianggap signifikan bila q < alpha dan jumlah koefisien lag A > 0.
Skor pemimpin provinsi = jumlah edge keluar - jumlah edge masuk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .labeling import future_window_max


def weekly_returns(P: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """Return log mingguan dari panel harga harian (rata-rata mingguan log harga)."""
    return np.log(P).resample(freq).mean().diff()


def _lags(x: np.ndarray, p: int) -> np.ndarray:
    """Matriks lag (T-p) x p: kolom k = x(t-k-1)."""
    T = len(x)
    return np.column_stack([x[p - k - 1: T - k - 1] for k in range(p)])


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-value."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def granger_pairwise(R: pd.DataFrame, control: pd.Series | None = None, p: int = 2,
                     min_obs: int = 60) -> pd.DataFrame:
    """Uji Granger bersyarat untuk semua pasangan kolom R (minggu x provinsi)."""
    arr = R.to_numpy(dtype=float)
    names = list(R.columns)
    T, N = arr.shape
    ctl = None
    if control is not None:
        ctl = _lags(control.reindex(R.index).to_numpy(dtype=float), p)
    rows = []
    for j in range(N):
        Y = arr[p:, j]
        own = _lags(arr[:, j], p)
        base = np.column_stack([np.ones(T - p), own] + ([ctl] if ctl is not None else []))
        for i in range(N):
            if i == j:
                continue
            xl = _lags(arr[:, i], p)
            Xu = np.column_stack([base, xl])
            m = np.isfinite(Y) & np.isfinite(Xu).all(axis=1)
            n = int(m.sum())
            if n < min_obs:
                continue
            yu, Xum, Xrm = Y[m], Xu[m], base[m]
            bu, *_ = np.linalg.lstsq(Xum, yu, rcond=None)
            br, *_ = np.linalg.lstsq(Xrm, yu, rcond=None)
            rss_u = float(((yu - Xum @ bu) ** 2).sum())
            rss_r = float(((yu - Xrm @ br) ** 2).sum())
            df2 = n - Xum.shape[1]
            if rss_u <= 0 or df2 <= 0:
                continue
            F = ((rss_r - rss_u) / p) / (rss_u / df2)
            rows.append((names[i], names[j], F, stats.f.sf(F, p, df2), float(bu[-p:].sum()), n))
    return pd.DataFrame(rows, columns=["source", "target", "F", "p_value", "coef_sum", "n_weeks"])


def event_precedence(onset: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """onset: harian x provinsi (0/1). Hitung A->B: onset A diikuti onset B dalam 1..window hari."""
    O = onset.fillna(0.0).to_numpy()
    F = future_window_max(onset.fillna(0.0), window).fillna(0.0).to_numpy()
    n_src = O.sum(axis=0)
    follow = O.T @ F                        # [i, j] = jumlah onset i yang diikuti onset j
    base = F.mean(axis=0)                   # peluang dasar j punya onset dalam jendela acak
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (follow / n_src[:, None]) / base[None, :]
    names = list(onset.columns)
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                rows.append((a, b, n_src[i], O[:, j].sum(), follow[i, j], lift[i, j]))
    return pd.DataFrame(rows, columns=["source", "target", "n_onset_source", "n_onset_target",
                                       "n_followed", "lift"])


def build_network(P: pd.DataFrame, onset: pd.DataFrame | None = None, p: int = 2, alpha: float = 0.05,
                  window: int = 14, start=None, end=None, national_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bangun edge list & skor pemimpin untuk semua varian komoditas.

    P      : panel harga konsumen harian, kolom [commodity_id, province_id] (termasuk nasional = 0)
    onset  : panel onset harian (opsional, untuk presedensi kejadian)
    start/end : batasi periode data yang dipakai (PENTING untuk mencegah leakage saat evaluasi)
    """
    P = P.loc[start:end]
    R_all = weekly_returns(P)
    edges = []
    for com in P.columns.get_level_values(0).unique():
        R = R_all[com]
        ctl = R[national_id] if national_id in R.columns else None
        R = R.drop(columns=[national_id], errors="ignore")
        g = granger_pairwise(R, ctl, p=p)
        if g.empty:
            continue
        g["q_value"] = bh_fdr(g["p_value"].to_numpy())
        if onset is not None and com in onset.columns.get_level_values(0):
            oc = onset[com].loc[start:end].drop(columns=[national_id], errors="ignore")
            ep = event_precedence(oc, window)
            g = g.merge(ep, on=["source", "target"], how="left")
        g.insert(0, "commodity_id", com)
        edges.append(g)
    edges = pd.concat(edges, ignore_index=True)
    edges["significant"] = (edges["q_value"] < alpha) & (edges["coef_sum"] > 0)
    return edges, leader_scores(edges)


def leader_scores(edges: pd.DataFrame) -> pd.DataFrame:
    sig = edges[edges["significant"]]
    out_deg = sig.groupby(["commodity_id", "source"]).size().rename("out_degree")
    in_deg = sig.groupby(["commodity_id", "target"]).size().rename("in_degree")
    out_deg.index.names = in_deg.index.names = ["commodity_id", "province_id"]
    allp = pd.MultiIndex.from_frame(
        pd.concat([edges[["commodity_id", "source"]].set_axis(["commodity_id", "province_id"], axis=1),
                   edges[["commodity_id", "target"]].set_axis(["commodity_id", "province_id"], axis=1)])
        .drop_duplicates())
    df = pd.concat([out_deg, in_deg], axis=1).reindex(allp).fillna(0).astype(int)
    df["leader_score"] = df["out_degree"] - df["in_degree"]
    return df.reset_index().sort_values(["commodity_id", "leader_score"], ascending=[True, False])


def select_leaders(edges: pd.DataFrame, k: int = 3) -> dict[tuple[str, int], list[int]]:
    """Untuk tiap (komoditas, provinsi target): k provinsi sumber dengan bukti Granger terkuat
    (koefisien positif). Dipakai sebagai fitur 'harga provinsi pemimpin'."""
    e = edges[edges["coef_sum"] > 0].copy()
    e["strength"] = -np.log10(e["p_value"].clip(lower=1e-300))
    e = e.sort_values("strength", ascending=False)
    leaders = {}
    for (com, tgt), grp in e.groupby(["commodity_id", "target"], sort=False):
        leaders[(com, int(tgt))] = [int(s) for s in grp["source"].head(k)]
    return leaders
