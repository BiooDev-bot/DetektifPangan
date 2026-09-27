@echo off
REM ==========================================================================
REM  RadarPangan - update harian (jadwalkan lewat Windows Task Scheduler)
REM  Scrape 2 bulan terakhir -> fitur -> prediksi 14 hari -> dashboard site\
REM  Tambahkan argumen --retrain untuk melatih ulang model (mis. sebulan sekali)
REM ==========================================================================
cd /d "%~dp0\.."

REM Cari python: venv proyek -> env conda "radarpangan" (anaconda3/miniconda3) -> python di PATH
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\radarpangan\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\radarpangan\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\radarpangan\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\radarpangan\python.exe"

echo [RadarPangan] memakai %PY%
"%PY%" run_pipeline.py update %*
if errorlevel 1 (
    echo [RadarPangan] Update GAGAL - cek koneksi ke bi.go.id / log di atas.
    exit /b 1
)
echo [RadarPangan] Update selesai. Folder site\ siap diunggah ke hosting.
