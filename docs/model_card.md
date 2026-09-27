# Model Card — RadarPangan LightGBM

| | |
|---|---|
| Versi | 1.0 — dilatih s.d. 2026-09-11 (1,451,057 baris, 401 pohon) |
| Tugas | Klasifikasi biner: peluang onset lonjakan harga konsumen dalam 14 hari ke depan per provinsi × varian |
| Algoritma | LightGBM (gradient boosting trees), satu model global, subsampling negatif berbobot |
| Parameter | `models/lgbm_best_params.json` (dipilih dengan walk-forward semu Jul 2020 – Des 2021) |
| Ambang alarm | 0,217 (maks. F1 pada 120 hari validasi terakhir) |
| Fitur | 78 fitur — lihat `docs/data_dictionary.md` |
| Data | PIHPS Bank Indonesia (4 jenis pasar), NASA POWER, kalender hari besar, Jan 2018 – 25 Sep 2026 |

## Penggunaan yang dimaksud
- Peringatan dini untuk TPID/Pemda/Bapanas/BI, pedagang, dan masyarakat: **"provinsi & komoditas mana yang perlu diwaspadai 2 minggu ke depan"**.
- Bukan untuk: spekulasi/penimbunan, penetapan harga, atau keputusan otomatis tanpa verifikasi lapangan.

## Performa (out-of-sample walk-forward Jan 2022 – Sep 2026)
| Model | PR-AUC | ROC-AUC | Presisi | Recall | FAR | Deteksi kejadian | Median lead | Episode alarm palsu/seri/thn |
|---|---|---|---|---|---|---|---|---|
| Naive "harga minggu lalu" | 0,256 | 0,615 | 0,308 | 0,352 | 69% | 85% | 7 hari | 2,68 |
| Seasonal naive (tahun lalu) | 0,114 | 0,507 | 0,176 | 0,277 | 82% | 50% | 13 hari | 2,00 |
| Regresi logistik | 0,398 | 0,878 | 0,377 | 0,517 | 62% | 79% | 11 hari | 1,64 |
| **LightGBM** | 0,501 | 0,920 | 0,452 | 0,540 | 55% | 85% | 11 hari | 1,30 |

## Faktor & keterbatasan
- Harga = rata-rata provinsi PIHPS; lonjakan lokal kab/kota bisa tersamarkan. 34 provinsi (provinsi pemekaran Papua 2022 belum terpisah).
- Kejadian untuk beras/gula/minyak/daging sapi jarang → probabilitas pada kategori ini kurang stabil.
- Data 2–3 hari terakhir NASA POWER belum tersedia (diisi nilai terakhir).
- Pola dapat berubah karena kebijakan (HET, impor, operasi pasar) → pantau drift & latih ulang bulanan.

## Pertimbangan etis
- Data publik agregat, tanpa data pribadi.
- Risiko salah tafsir: alarm adalah **peluang**, bukan kepastian; dashboard menampilkan alasan (SHAP) dan disclaimer.
- Risiko penyalahgunaan (spekulasi) dimitigasi dengan fokus pada tingkat provinsi & komunikasi kepada pembuat kebijakan.
