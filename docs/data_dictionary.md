# Kamus Data RadarPangan

## Tabel mentah (`data/interim/pihps_long.parquet`)

| Kolom | Tipe | Keterangan |
|---|---|---|
| date | tanggal | Tanggal harga (hanya hari kerja) |
| price_type_id / price_type | int / teks | 1 Pasar Tradisional, 2 Pasar Modern, 3 Pedagang Besar, 4 Produsen |
| commodity_id / commodity | teks | Varian `com_1`–`com_21` (mis. `com_16` Cabai Rawit Merah) |
| cat_id / category | teks | Kategori `cat_1`–`cat_10` (Beras, Daging Ayam, …) |
| province_id / province | int / teks | 1–34 sesuai PIHPS; 0 = Nasional ("Semua Provinsi") |
| price | float | Harga Rp/kg; kosong bila PIHPS menulis "-" |

## Label (`data/processed/features.parquet`)

| Kolom | Keterangan |
|---|---|
| y | 1 jika ada onset lonjakan dalam (t, t+14 hari]; kosong bila 14 hari ke depan belum lengkap |
| eligible | 1 jika tidak sedang lonjak pada t (baris yang dipakai melatih/menguji alarm) |
| spike_now | 1 jika r14(t) > ambang (status 'Sedang lonjak') |
| onset_now | 1 jika t adalah hari pertama kejadian lonjakan |

## Fitur (78 kolom)

| Fitur | Kelompok | Deskripsi | % kosong |
|---|---|---|---|
| `ret_1` | Dinamika harga sendiri | Perubahan log harga konsumen 1 hari | 0.0% |
| `ret_3` | Dinamika harga sendiri | Perubahan log harga konsumen 3 hari | 0.0% |
| `ret_7` | Dinamika harga sendiri | Perubahan log harga konsumen 7 hari | 0.0% |
| `ret_14` | Dinamika harga sendiri | Perubahan log harga konsumen 14 hari | 0.0% |
| `ret_28` | Dinamika harga sendiri | Perubahan log harga konsumen 28 hari | 0.0% |
| `ret_7_prev` | Dinamika harga sendiri | Perubahan log harga konsumen minggu sebelumnya (t-14 → t-7) | 0.0% |
| `vol_7` | Dinamika harga sendiri | Volatilitas: simpangan baku perubahan log harian 7 hari | 0.0% |
| `vol_14` | Dinamika harga sendiri | Volatilitas: simpangan baku perubahan log harian 14 hari | 0.0% |
| `vol_30` | Dinamika harga sendiri | Volatilitas: simpangan baku perubahan log harian 30 hari | 0.0% |
| `z_30` | Dinamika harga sendiri | Z-score log harga terhadap 30 hari terakhir | 7.2% |
| `z_90` | Dinamika harga sendiri | Z-score log harga terhadap 90 hari terakhir | 4.0% |
| `dist_max_90` | Dinamika harga sendiri | Jarak log harga ke harga tertinggi 90 hari terakhir | 0.0% |
| `dist_min_90` | Dinamika harga sendiri | Jarak log harga ke harga terendah 90 hari terakhir | 0.0% |
| `rel_365` | Dinamika harga sendiri | Log harga relatif terhadap rata-rata 365 hari | 3.1% |
| `ret_yoy` | Dinamika harga sendiri | Perubahan log harga dibanding 364 hari lalu | 9.1% |
| `seas_ret_14_ly` | Dinamika harga sendiri | Perubahan harga 14 hari yang terjadi tahun lalu pada jendela ke depan yang sama (pola musiman) | 9.1% |
| `r14_now` | Dinamika harga sendiri | Kenaikan harga 14 hari saat ini (P_t/P_t-14 - 1) | 0.0% |
| `gap_to_threshold` | Dinamika harga sendiri | Selisih ambang lonjakan dengan kenaikan 14 hari saat ini | 0.0% |
| `days_since_spike` | Dinamika harga sendiri | Hari sejak terakhir berstatus lonjak (maks 365) | 0.0% |
| `n_onsets_365` | Dinamika harga sendiri | Jumlah kejadian lonjakan dalam 365 hari terakhir | 0.0% |
| `nat_ret_7` | Kondisi nasional | Perubahan log harga nasional 7 hari | 0.0% |
| `nat_ret_14` | Kondisi nasional | Perubahan log harga nasional 14 hari | 0.0% |
| `rel_nat` | Kondisi nasional | Log harga provinsi relatif terhadap harga nasional | 0.0% |
| `rel_nat_chg14` | Kondisi nasional | Perubahan 14 hari dari selisih harga provinsi vs nasional | 0.0% |
| `nat_spike_share` | Kondisi nasional | Porsi provinsi yang sedang lonjak (komoditas sama) | 0.0% |
| `nat_spike_share_chg7` | Kondisi nasional | Perubahan 7 hari porsi provinsi yang sedang lonjak | 0.0% |
| `ret_7_prod` | Rantai pasok | Perubahan log harga produsen 7 hari | 44.8% |
| `ret_14_prod` | Rantai pasok | Perubahan log harga produsen 14 hari | 45.1% |
| `nat_ret_14_prod` | Rantai pasok | Perubahan log harga nasional 14 hari (produsen) | 1.6% |
| `nat_margin_cons_prod` | Rantai pasok | Margin log harga konsumen - produsen tingkat nasional | 1.3% |
| `nat_margin_cons_prod_chg14` | Rantai pasok | Perubahan 14 hari margin konsumen-produsen nasional | 1.6% |
| `ret_7_whole` | Rantai pasok | Perubahan log harga pedagang besar 7 hari | 3.7% |
| `ret_14_whole` | Rantai pasok | Perubahan log harga pedagang besar 14 hari | 3.8% |
| `nat_ret_14_whole` | Rantai pasok | Perubahan log harga nasional 14 hari (pedagang besar) | 0.0% |
| `ret_7_modern` | Rantai pasok | Perubahan log harga pasar modern 7 hari | 20.6% |
| `ret_14_modern` | Rantai pasok | Perubahan log harga pasar modern 14 hari | 20.8% |
| `nat_ret_14_modern` | Rantai pasok | Perubahan log harga nasional 14 hari (pasar modern) | 0.0% |
| `margin_cons_prod` | Rantai pasok | Margin log harga konsumen - produsen (provinsi) | 44.4% |
| `margin_cons_prod_chg14` | Rantai pasok | Perubahan 14 hari margin konsumen-produsen | 45.1% |
| `margin_cons_prod_z90` | Rantai pasok | Z-score margin konsumen-produsen terhadap 90 hari terakhir | 46.2% |
| `margin_cons_whole` | Rantai pasok | Margin log harga konsumen - pedagang besar | 3.7% |
| `margin_cons_whole_chg14` | Rantai pasok | Perubahan 14 hari margin konsumen-pedagang besar | 3.8% |
| `margin_whole_prod` | Rantai pasok | Margin log harga pedagang besar - produsen | 45.0% |
| `gap_modern` | Rantai pasok | Log harga pasar modern - pasar tradisional | 20.3% |
| `nbr_ret_7_mean` | Provinsi tetangga | Rata-rata perubahan 7 hari di 3 provinsi terdekat | 0.0% |
| `nbr_ret_14_mean` | Provinsi tetangga | Rata-rata perubahan 14 hari di 3 provinsi terdekat | 0.0% |
| `nbr_ret_7_max` | Provinsi tetangga | Perubahan 7 hari tertinggi di 3 provinsi terdekat | 0.0% |
| `nbr_spike_share` | Provinsi tetangga | Porsi provinsi tetangga yang sedang lonjak | 0.0% |
| `nbr_rel_price` | Provinsi tetangga | Log harga relatif terhadap rata-rata tetangga | 0.0% |
| `lead_ret_7_mean` | Provinsi pemimpin | Rata-rata perubahan 7 hari di provinsi pemimpin (jaringan Granger) | 0.3% |
| `lead_ret_14_mean` | Provinsi pemimpin | Rata-rata perubahan 14 hari di provinsi pemimpin | 0.3% |
| `lead_spike_share` | Provinsi pemimpin | Porsi provinsi pemimpin yang sedang lonjak | 0.3% |
| `days_to_awal_ramadan` | Kalender | Hari menuju awal ramadan berikutnya (maks 120) | 0.0% |
| `days_since_awal_ramadan` | Kalender | Hari sejak awal ramadan terakhir (maks 120) | 0.0% |
| `days_to_idulfitri` | Kalender | Hari menuju idulfitri berikutnya (maks 120) | 0.0% |
| `days_since_idulfitri` | Kalender | Hari sejak idulfitri terakhir (maks 120) | 0.0% |
| `days_to_iduladha` | Kalender | Hari menuju iduladha berikutnya (maks 120) | 0.0% |
| `days_since_iduladha` | Kalender | Hari sejak iduladha terakhir (maks 120) | 0.0% |
| `days_to_natal` | Kalender | Hari menuju natal berikutnya (maks 120) | 0.0% |
| `days_to_tahun_baru` | Kalender | Hari menuju tahun baru berikutnya (maks 120) | 0.0% |
| `is_ramadan` | Kalender | 1 jika bulan Ramadan | 0.0% |
| `is_nataru` | Kalender | 1 jika 15 Des - 7 Jan | 0.0% |
| `doy_sin` | Kalender | Musiman tahunan (sin hari ke-n) | 0.0% |
| `doy_cos` | Kalender | Musiman tahunan (cos hari ke-n) | 0.0% |
| `dow` | Kalender | Hari dalam minggu (0=Senin) | 0.0% |
| `rain_7_anom` | Curah hujan | Anomali (z) curah hujan 7 hari — provinsi sendiri | 0.0% |
| `rain_7_anom_sentra` | Curah hujan | Anomali (z) curah hujan 7 hari — rata-rata sentra produksi | 0.0% |
| `rain_30_anom` | Curah hujan | Anomali (z) curah hujan 30 hari — provinsi sendiri | 0.0% |
| `rain_30_anom_sentra` | Curah hujan | Anomali (z) curah hujan 30 hari — rata-rata sentra produksi | 0.0% |
| `rain_30_anom_lag30` | Curah hujan | Anomali (z) curah hujan 30 hari, digeser 30 hari — provinsi sendiri | 0.0% |
| `rain_30_anom_lag30_sentra` | Curah hujan | Anomali (z) curah hujan 30 hari, digeser 30 hari — rata-rata sentra produksi | 0.0% |
| `rain_30_anom_lag60` | Curah hujan | Anomali (z) curah hujan 30 hari, digeser 60 hari — provinsi sendiri | 0.0% |
| `rain_30_anom_lag60_sentra` | Curah hujan | Anomali (z) curah hujan 30 hari, digeser 60 hari — rata-rata sentra produksi | 0.0% |
| `rain_90_anom` | Curah hujan | Anomali (z) curah hujan 90 hari — provinsi sendiri | 0.0% |
| `rain_90_anom_sentra` | Curah hujan | Anomali (z) curah hujan 90 hari — rata-rata sentra produksi | 0.0% |
| `commodity_code` | Identitas | Kode varian komoditas (kategori LightGBM) | 0.0% |
| `category_code` | Identitas | Kode kategori komoditas (kategori LightGBM) | 0.0% |
| `province_code` | Identitas | Kode provinsi (kategori LightGBM) | 0.0% |
