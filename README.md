# Student Risk Level C4.5

Proyek ini mengimplementasikan algoritma C4.5 (custom) untuk klasifikasi `tingkat_risiko` kelulusan berdasarkan dataset mahasiswa.

## Prasyarat

- Python 3.10+
- `pip`
- `git`

## Clone Repository

```bash
git clone https://github.com/andinoferdi/student-risk-level-c45.git
cd student-risk-level-c45
```

## Install Dependency

Disarankan pakai virtual environment.

```bash
python -m venv .venv
```

Aktifkan virtual environment:

- Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

- Mac/Linux:

```bash
source .venv/bin/activate
```

Install requirement:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Menjalankan Project

Jalankan dengan konfigurasi default:

```bash
python app.py
```

Atau tentukan file data secara eksplisit:

```bash
python app.py --data dataset/data_manajemen_perguruan_tinggi.csv
```

## Contoh Opsi Umum

```bash
python app.py --n_repeats 30 --save_plots true --output_dir outputs
```

Tes cepat:

```bash
python app.py --n_repeats 1 --save_plots false --output_dir outputs_test
```

## Output

Secara default hasil disimpan ke folder `outputs/`, termasuk:

- `metrics_per_run.csv`
- `metrics_summary.csv`
- `representative_selection.csv`
- File visualisasi (`.png`) jika `--save_plots true`

## Troubleshooting Singkat

- `ModuleNotFoundError`: jalankan kembali `python -m pip install -r requirements.txt`
- `Dataset tidak ditemukan`: pastikan path benar, misalnya `--data dataset/data_manajemen_perguruan_tinggi.csv`
