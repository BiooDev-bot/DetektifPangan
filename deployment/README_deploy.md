# Deployment RadarPangan — dari laptop ke QR code poster

Dashboard RadarPangan adalah **situs statis** (folder `site/`: HTML + JS + JSON). Tidak perlu server/backend,
jadi bisa di-hosting **gratis** dan tidak "tidur" seperti aplikasi Streamlit/Heroku — penting karena juri
akan membuka QR poster dari HP kapan saja.

```
python run_pipeline.py update      # scrape data terbaru -> fitur -> prediksi -> tulis ulang site/
```

---

## 1. Lihat dashboard di laptop

```bash
python -m http.server 8000 --directory site
```
Buka `http://localhost:8000`. (Klik dua kali `index.html` tidak bisa: browser memblokir `fetch` ke file lokal.)

Tambahkan `?k=com_14` di URL untuk langsung membuka komoditas tertentu, mis. `http://localhost:8000/?k=com_11` (bawang merah).

---

## 2. Hosting (pilih salah satu)

### Opsi A — GitHub Actions → Vercel + update otomatis (direkomendasikan)
Dashboard production: **<https://radarpangan.vercel.app>** (project Vercel `radarpangan`, scope `bioodev-bots-projects`).
Workflow membangun `site/` di runner GitHub lalu men-deploy-nya lewat Vercel CLI. Project Vercel **sengaja tidak
disambungkan ke GitHub** (Git integration): kalau disambungkan, setiap push dibangun dari root repo yang tidak berisi
`site/` (folder itu ada di `.gitignore`) sehingga link production bisa rusak.

1. Repo GitHub `BiooDev-bot/DetektifPangan` berisi folder proyek ini sebagai root repo.
   `data/raw` (cache ±35 MB) ikut di-push supaya update harian tidak perlu scraping ulang dari 2018.
2. Buat token: <https://vercel.com/account/tokens> → **Create Token** → Scope: `bioodev-bots-projects` →
   Expiration minimal sampai akhir November 2026 (final USB 19 Nov 2026).
3. Simpan sebagai secret GitHub `VERCEL_TOKEN`: repo → **Settings → Secrets and variables → Actions → New repository secret**
   (atau di terminal: `gh secret set VERCEL_TOKEN --repo BiooDev-bot/DetektifPangan`). Jangan tempel token di file, commit, atau chat.
4. Tab **Actions → "Update dashboard RadarPangan" → Run workflow** (jalankan pertama kali secara manual).
   Urutan step: update data → simpan cache (commit) → deploy Vercel → verifikasi `generated_at` dashboard live.
5. Workflow `.github/workflows/update-dashboard.yml` berjalan otomatis **setiap hari kerja pukul 17.00 WIB**.
   Centang `retrain` saat menjalankan manual untuk melatih ulang model (disarankan sebulan sekali).

> `VERCEL_ORG_ID` & `VERCEL_PROJECT_ID` sudah ada di `env` workflow, jadi tidak perlu `vercel link` (`site/` dibuat ulang
> dari nol setiap update, sehingga folder `.vercel` di dalamnya selalu hilang). Deploy gagal 401/403 → token salah scope
> atau kedaluwarsa: buat token baru lalu perbarui secret `VERCEL_TOKEN`.
>
> Jika server GitHub (luar negeri) diblokir/timeout oleh `bi.go.id`, pakai Opsi C (update dari laptop) lalu deploy
> `site/` dengan `npx vercel deploy site --prod --yes --project radarpangan`.

### Opsi B — Vercel (drag & drop / CLI)
- Dashboard Vercel → **Add New → Project → Deploy from folder**: pilih folder `site/` (Framework: *Other*, tanpa build command).
- Atau CLI: `npx vercel deploy site --prod`.
- Update: jalankan `python run_pipeline.py update`, lalu deploy ulang folder `site/`.

### Opsi C — Update dari laptop Windows (Task Scheduler)
1. Buka **Task Scheduler → Create Basic Task** → *Daily* (Senin–Jumat) pukul 16.30.
2. Action: *Start a program* → `C:\Users\Bio\Documents\USB-DataMining\MiningProcess\deployment\update_harian.bat`.
3. Setelah update, unggah `site/` ke hosting (git push / Vercel CLI).

---

## 3. QR code untuk poster (wajib 2 QR)

1. Isi `deployment.dashboard_url` di `config.yaml` dengan URL hosting.
2. Jalankan:
   ```bash
   python deployment/make_qr.py --url https://<username>.github.io/radarpangan/ --name qr_prototipe
   python deployment/make_qr.py --url https://youtu.be/<id-video-unlisted> --name qr_video
   ```
3. Hasil: `reports/figures/qr_prototipe.png` dan `qr_video.png` (resolusi tinggi, aman dicetak A3).
   Uji scan dari 2–3 HP berbeda sebelum poster dicetak.

---

## 4. Checklist sebelum pengumpulan
- [ ] `python run_pipeline.py update` sukses dan tanggal "Data s.d." di dashboard = hari kerja terakhir
- [ ] Dashboard terbuka dari HP (data seluler, bukan Wi-Fi kampus) < 5 detik
- [ ] Link prototipe bisa dibuka tanpa login (*view only*)
- [ ] QR prototipe & QR video di footer poster sudah diuji scan
- [ ] Angka di proposal/poster/PPT sama dengan `reports/tables/06_perbandingan_model.csv`

## 5. Pemantauan & pemeliharaan
| Aspek | Cara | Tindakan |
|---|---|---|
| Kesegaran data | Tanggal "Data s.d." di dashboard | > 3 hari kerja tertinggal → cek log Actions / koneksi ke bi.go.id |
| Scraper | Status chunk di log `run_pipeline.py` | API berubah → sesuaikan `src/radarpangan/scraper.py` |
| Drift fitur | Notebook 07 bagian 7.5 (PSI) | PSI > 0,25 → `update --retrain` |
| Performa nyata | Bandingkan alarm vs lonjakan yang terjadi 14 hari kemudian | PR-AUC bulanan < baseline 2 bulan berturut-turut → evaluasi ulang |
