"""Buat QR code untuk footer poster (wajib: link prototipe + link video YouTube unlisted).

Pakai:
    python deployment/make_qr.py --url https://username.github.io/radarpangan --name qr_prototipe
    python deployment/make_qr.py --url https://youtu.be/XXXX --name qr_video

Hasil: reports/figures/<name>.png (resolusi tinggi, aman untuk cetak A3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_Q


def make_qr(url: str, out: Path, box_size: int = 20, border: int = 2) -> Path:
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_Q, box_size=box_size, border=border)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", default="qr_prototipe")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = make_qr(a.url, root / "reports" / "figures" / f"{a.name}.png")
    print("QR tersimpan:", path)
