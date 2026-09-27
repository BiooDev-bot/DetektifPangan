# Ringkasan Hasil — bahan Proposal, Poster, PPT & Video

Semua angka berasal dari run pipeline pada data asli (data s.d. **25 Sep 2026**) dan dihasilkan otomatis dari `reports/tables/`.
Jika notebook dijalankan ulang dengan data lebih baru, angka bisa sedikit berubah — salin angka terbaru dari `reports/tables/06_*.csv`.

## 1. Angka kunci
- **Data**: 5.450.980 baris harga harian PIHPS (21 varian × 4 jenis pasar × 34 provinsi + nasional, Jan 2018 – 25 Sep 2026) + curah hujan NASA POWER; **8.457 kejadian lonjakan**.
- **Uji**: walk-forward **19 fold** out-of-sample (Jan 2022 – Sep 2026), *purge* 14 hari, 78 fitur.
- **LightGBM PR-AUC 0,501** vs naive "harga minggu lalu" 0,256 (**1,96×**), seasonal naive 0,114, regresi logistik 0,398; ROC-AUC 0,920. Pada subset 5 varian volatil (origin mingguan), LightGBM PR-AUC 0,579 vs SARIMA 0,319 dan naive 0,317.
- **85% kejadian lonjakan didahului alarm**, median **lead time 11 hari** (naive: 7 hari); 67% kejadian mendapat alarm ≥ 7 hari sebelumnya.
- **Episode alarm palsu 1,30 per seri per tahun** (naive: 2,68) — ±separuh beban alarm palsu naive.
- LightGBM lebih baik dari naive di **8 dari 10 kategori** (belum unggul di: Gula Pasir, Telur Ayam).
- Jaringan rambatan: **4,411 edge signifikan** (Granger bersyarat, FDR 5%); provinsi "radar" teratas: **Jawa Timur, Jawa Tengah, Jawa Barat, Banten, DI Yogyakarta**; edge yang dipelajari dari 2018–2021 tetap lebih prediktif (lift lebih tinggi) pada 2022–2026 di **13 dari 21** varian.
- Kelompok yang paling menurunkan PR-AUC bila dihapus: hanya harga sendiri + identitas (−0,0649); tanpa kalender (−0,0297); tanpa tetangga & pemimpin (tanpa sinyal spasial) (−0,0244); tanpa pemimpin (−0,0060). Kelompok yang tidak menambah PR-AUC: tanpa rantai_pasok.
- Fitur terpenting (SHAP): `vol_30`, `commodity_code`, `vol_14`, `lead_ret_14_mean`, `province_code`, `lead_ret_7_mean`, `nbr_rel_price`, `ret_7`.

## 2. Pemetaan ke bab proposal (sesuai ketentuan USB 2026)
| Bab proposal | Ambil dari | Gambar / tabel |
|---|---|---|
| 5. Pendahuluan (latar belakang, batasan) | NB01 §1.1, §1.3; batasan: 34 provinsi, harga rata-rata provinsi, hari kerja, 2018–2026 | `fig_03_04`, `fig_03_06` |
| 6. Tujuan, hasil, solusi ↔ tema | NB01 §1.2, §1.4 (tujuan bisnis & data mining, SDGs 1/2/12) | `fig_01_crisp_dm` |
| 7. Kajian pustaka | Rujukan di bagian 6 dokumen ini | – |
| 8. Solusi (dataset, metode, metrik) | NB02 (dataset & scraping), NB04 (label, jaringan, fitur), NB05 (model & walk-forward), NB06 §6.1 (metrik) | `fig_04_02`, `fig_04_03`, `fig_05_01`, `docs/data_dictionary.md` |
| 9. Hasil pengujian | NB06 §6.1–6.5, §6.9 | tabel di bagian 3; `fig_06_01`–`fig_06_04` |
| 10. Analisis hasil | NB06 §6.3, §6.6–6.8, §6.10 + poin di bagian 4 | `fig_06_05`, `fig_06_07`, `fig_06_08`, `fig_06_09` |
| 11. Kesimpulan | NB06 §6.9–6.10, NB07 §7.7 | – |
| Prototipe / poster | NB07, dashboard `site/`, QR | `fig_07_01`, `qr_prototipe.png` |

## 3. Tabel hasil (siap tempel)

### 3.1 Perbandingan model — seluruh periode uji
| Model | PR-AUC | ROC-AUC | Presisi | Recall | FAR | Deteksi kejadian | Median lead | Episode alarm palsu/seri/thn |
|---|---|---|---|---|---|---|---|---|
| Naive "harga minggu lalu" | 0,256 | 0,615 | 0,308 | 0,352 | 69% | 85% | 7 hari | 2,68 |
| Seasonal naive (tahun lalu) | 0,114 | 0,507 | 0,176 | 0,277 | 82% | 50% | 13 hari | 2,00 |
| Regresi logistik | 0,398 | 0,878 | 0,377 | 0,517 | 62% | 79% | 11 hari | 1,64 |
| **LightGBM** | 0,501 | 0,920 | 0,452 | 0,540 | 55% | 85% | 11 hari | 1,30 |

### 3.2 Pembanding SARIMA (subset 5 varian volatil × 34 provinsi, origin tiap Jumat)
| Model | PR-AUC | Deteksi kejadian |
|---|---|---|
| Naive "harga minggu lalu" | 0,317 | 74% |
| Seasonal naive (tahun lalu) | 0,170 | 45% |
| SARIMA (ARIMA + Fourier) | 0,319 | 79% |
| Regresi logistik | 0,466 | 84% |
| LightGBM | 0,579 | 86% |

### 3.3 Per kategori komoditas (LightGBM)
| Kategori | Proporsi positif | PR-AUC naive | PR-AUC LightGBM | Kejadian uji | Terdeteksi | Median lead |
|---|---|---|---|---|---|---|
| Cabai Merah | 25,6% | 0,427 | 0,600 | 1538 | 98% | 13 hari |
| Cabai Rawit | 20,9% | 0,403 | 0,587 | 1306 | 97% | 11 hari |
| Bawang Merah | 11,2% | 0,357 | 0,496 | 425 | 96% | 8 hari |
| Daging Ayam | 3,5% | 0,204 | 0,348 | 142 | 89% | 7 hari |
| Beras | 1,9% | 0,145 | 0,212 | 448 | 25% | 6 hari |
| Bawang Putih | 1,7% | 0,138 | 0,206 | 72 | 75% | 5 hari |
| Daging Sapi | 1,1% | 0,076 | 0,184 | 90 | 47% | 9 hari |
| Minyak Goreng | 1,7% | 0,106 | 0,163 | 203 | 41% | 4 hari |
| Gula Pasir | 0,8% | 0,113 | 0,092 | 59 | 44% | 5 hari |
| Telur Ayam | 0,7% | 0,085 | 0,077 | 30 | 63% | 5 hari |

### 3.4 Ablation kelompok sinyal
| Varian | PR-AUC | Δ PR-AUC vs lengkap | Fold model lengkap lebih baik |
|---|---|---|---|
| hanya harga sendiri + identitas | 0,4363 | +0,0649 | 19/19 |
| tanpa kalender | 0,4715 | +0,0297 | 17/19 |
| tanpa tetangga & pemimpin (tanpa sinyal spasial) | 0,4767 | +0,0244 | 18/19 |
| tanpa pemimpin | 0,4952 | +0,0060 | 14/19 |
| tanpa tetangga | 0,4970 | +0,0041 | 13/19 |
| tanpa cuaca | 0,5002 | +0,0010 | 10/19 |
| tanpa rantai_pasok | 0,5045 | -0,0034 | 6/19 |

### 3.5 Kriteria sukses
| Kriteria | Target | Hasil | Status |
|---|---|---|---|
| PR-AUC LightGBM ≥ 1,5× naive | ≥ 1,5× | 1.96× | ✅ tercapai |
| LightGBM > semua baseline (termasuk SARIMA di subset) | ya | ya | ✅ tercapai |
| Deteksi kejadian | ≥ 60% | 85% | ✅ tercapai |
| Median lead time | ≥ 7 hari | 11 hari | ✅ tercapai |
| FAR & episode alarm palsu dilaporkan + dapat diatur | ya | FAR 55%; 1.30 episode palsu/seri/thn | ✅ tercapai |
| Validasi tanpa kebocoran | ya | walk-forward + purge 14 hr + uji leakage (NB04) lulus | ✅ tercapai |

## 4. Poin analisis (bahan bab 10)
1. **Model vs baseline** — LightGBM menggandakan PR-AUC baseline naive dan unggul di setiap fold uji. Kekuatan utamanya bukan menangkap *lebih banyak* kejadian (naive juga tinggi karena alarm ketika harga sudah naik), tetapi **alarm lebih awal** (11 vs 7 hari) dengan **alarm palsu jauh lebih sedikit**.
2. **Linear vs non-linear** — regresi logistik dengan fitur yang sama sudah jauh di atas naive, jadi fitur rekayasa (tetangga, pemimpin, kalender, rantai pasok) membawa informasi; LightGBM menambah lagi karena menangkap interaksi (mis. kenaikan tetangga × menjelang Idulfitri).
3. **Per komoditas** — performa terbaik di cabai & bawang merah (kejadian sering, sinyal kuat; deteksi 98% kejadian cabai merah, 96% bawang merah). Untuk komoditas yang harganya "lengket", deteksi jauh lebih rendah — beras 25%, minyak goreng 41%, gula 44%, daging sapi 47% — karena lonjakannya kecil, jarang, dan sering berupa "loncatan" harga tanpa pemanasan. **Angka deteksi 85% keseluruhan didominasi cabai & bawang** (kejadiannya paling banyak) — sampaikan ini terbuka saat presentasi.
4. **Sinyal rambatan** — uji Granger bersyarat menemukan provinsi di Jawa (terutama Jawa Timur) sebagai "pemimpin"; edge dari 2018–2021 tetap berlaku pada 2022+ sehingga peta rambatan layak dipakai sebagai radar pemantauan.
5. **Sinyal mana yang benar-benar membantu (ablation)** — kalender paling berharga, disusul sinyal spasial (tetangga + provinsi pemimpin, saling tumpang tindih); curah hujan kecil & tidak konsisten. **Margin produsen–konsumen tidak menambah performa** (hanya sedikit membantu daging ayam & minyak goreng) → hipotesis awal tidak terbukti pada data PIHPS tingkat provinsi (harga produsen tidak lengkap & jarang diperbarui). Laporkan sebagai temuan jujur, bukan kegagalan.
6. **Keterbatasan** — harga rata-rata provinsi, data produsen tidak lengkap, tidak ada data stok/produksi harian, Granger ≠ kausalitas fisik, FAR harian masih tinggi (alarm = "waspada", bukan kepastian).

## 5. Draf abstrak (±190 kata — sesuaikan gaya bahasa tim)
> Lonjakan harga pangan bergejolak seperti cabai, bawang, dan daging ayam sering datang tiba-tiba, berbeda antarprovinsi, dan merambat antarwilayah, sementara informasi harga yang tersedia bersifat deskriptif. Penelitian ini membangun RadarPangan, sistem peringatan dini lonjakan harga pangan 14 hari ke depan beserta peta rambatannya antarprovinsi, mengikuti metodologi CRISP-DM. Data harian Pusat Informasi Harga Pangan Strategis (PIHPS) Bank Indonesia untuk 21 varian komoditas, empat jenis pasar, dan 34 provinsi periode 2018–2026 (5.450.980 baris) dikumpulkan melalui *web scraping* dan diperkaya curah hujan NASA POWER serta kalender hari besar. Lonjakan didefinisikan sebagai kenaikan harga konsumen melebihi ambang yang disesuaikan volatilitas komoditas dalam 14 hari. Sebanyak 78 fitur disusun dari dinamika harga, margin rantai pasok, provinsi tetangga, provinsi pemimpin hasil uji kausalitas Granger bersyarat, kalender, dan anomali hujan, lalu dimodelkan dengan LightGBM dan dibandingkan dengan baseline naive, seasonal naive, SARIMA, dan regresi logistik menggunakan validasi *walk-forward*. Pada pengujian *out-of-sample* Jan 2022–Sep 2026, LightGBM mencapai PR-AUC 0,501 (1,96× baseline naive), mendeteksi 85% kejadian lonjakan dengan median waktu peringatan 11 hari. Alasan setiap alarm dijelaskan menggunakan SHAP dan disajikan dalam dasbor publik yang diperbarui harian.

## 6. Rujukan metode (sudah dicek; sesuaikan gaya sitasi)
- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proc. 4th Int. Conf. on the Practical Applications of Knowledge Discovery and Data Mining*, 29–39.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*. https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*, 56–67. https://doi.org/10.1038/s42256-019-0138-9
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica, 37*(3), 424–438. https://doi.org/10.2307/1912791
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting, 16*(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/ (ARIMA + suku Fourier untuk musiman panjang)
- Baquedano, F. G. (2015). *Developing an indicator of price anomalies as an early warning tool: A compound growth approach*. FAO. https://www.fao.org/fileadmin/user_upload/foodprice/docs/resources/a-i7550e.pdf
- Runfola, D., et al. (2020). geoBoundaries: A global database of political administrative boundaries. *PLOS ONE, 15*(4), e0231866. https://doi.org/10.1371/journal.pone.0231866
- Sumber data: Bank Indonesia — PIHPS Nasional (https://www.bi.go.id/hargapangan); NASA Langley Research Center — POWER Project (https://power.larc.nasa.gov).

## 7. Daftar gambar (`reports/figures/`)
| File | Kegunaan |
|---|---|
| `fig_01_crisp_dm.png` | Alur CRISP-DM proyek (bab Metodologi / poster) |
| `fig_03_01_kelengkapan.png` | Kelengkapan data per jenis pasar & tahun |
| `fig_03_02_contoh_blip.png` |  |
| `fig_03_03_ketersediaan_produsen.png` |  |
| `fig_03_04_tren_nasional.png` | Tren harga nasional 10 kategori 2018–2026 (latar belakang) |
| `fig_03_05_volatilitas.png` | Sebaran kenaikan 14 hari per kategori → alasan ambang per komoditas |
| `fig_03_06_pola_idulfitri.png` | Pola harga di sekitar Idulfitri (musiman) |
| `fig_03_07_musim_lonjakan.png` | Frekuensi lonjakan per bulan |
| `fig_03_08_peta_frekuensi.png` | Peta frekuensi lonjakan cabai rawit & bawang merah |
| `fig_03_09_lead_produsen.png` | Uji lead-lag harga produsen vs konsumen |
| `fig_03_10_rantai_pasok.png` | Harga konsumen vs pedagang besar vs produsen |
| `fig_03_11_jarak_vs_korelasi.png` | Makin dekat provinsi, makin serempak harganya |
| `fig_03_12_hujan_lag.png` | Pengaruh hujan tertunda 1–3 bulan |
| `fig_04_01_pembersihan.png` | Data mentah vs bersih |
| `fig_04_02_ilustrasi_label.png` | Ilustrasi label peringatan dini (bab Solusi) |
| `fig_04_03_jaringan_rambatan.png` | Peta rambatan harga antarprovinsi (poster!) |
| `fig_05_01_walk_forward.png` | Desain validasi walk-forward (bab Solusi/Metode) |
| `fig_05_02_prauc_per_fold.png` | PR-AUC per fold (stabilitas) |
| `fig_06_01_perbandingan_model.png` | Perbandingan model + kurva PR (bab Hasil, poster) |
| `fig_06_02_lead_time.png` | Lead time & deteksi per kategori |
| `fig_06_03_tradeoff_ambang.png` | Trade-off ambang alarm |
| `fig_06_04_ablation.png` | Ablation kelompok sinyal |
| `fig_06_05_shap_importance.png` | Fitur & kelompok sinyal terpenting (SHAP) |
| `fig_06_06_shap_beeswarm.png` | SHAP beeswarm |
| `fig_06_07_shap_dependence.png` | SHAP dependence fitur kunci |
| `fig_06_08_studi_kasus.png` | Studi kasus kejadian nyata: alarm H-x sebelum lonjakan (poster!) |
| `fig_06_09_provinsi_radar.png` | Provinsi 'radar' yang paling sering naik duluan |
| `fig_07_01_peta_risiko.png` | Peta risiko 14 hari ke depan (poster/prototipe) |
| `fig_07_02_monitoring.png` | Pemantauan performa per bulan |
| `qr_prototipe.png` | QR prototipe (ganti URL di config.yaml lalu buat ulang) |
