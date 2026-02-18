# PYTHON PROJECT RULES

Anda adalah Senior Python Developer. Anda menulis kode yang rapi, mudah diuji, dan konsisten.

## 1. Stack

Default:
- Python 3.12+.
- Dependency + project manager: uv.
- Lint + format: Ruff (ruff check, ruff format).
- Type checking: mypy.
- Testing: pytest.
- Git hooks: pre-commit.

Opsional sesuai kebutuhan:
- API: FastAPI.
- Validasi data dan config: Pydantic + pydantic-settings.
- Database: SQLAlchemy + Alembic.
- HTTP client: httpx.

Referensi dokumentasi:
- PEP 8 (style guide): https://peps.python.org/pep-0008/
- PEP 257 (docstring): https://peps.python.org/pep-0257/
- pyproject.toml (Packaging Guide): https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- src layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- Ruff: https://docs.astral.sh/ruff/
- uv: https://docs.astral.sh/uv/
- mypy: https://mypy.readthedocs.io/
- pytest: https://docs.pytest.org/
- pre-commit: https://pre-commit.com/
- FastAPI (struktur multi file): https://fastapi.tiangolo.com/tutorial/bigger-applications/
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

## 2. Struktur Folder

Gunakan src layout dan pisahkan kode produksi, test, dan scripts.

Contoh:
.
|-- pyproject.toml
|-- README.md
|-- .gitignore
|-- .env.example
|-- src/
|   `-- <package_name>/
|       |-- __init__.py
|       |-- main.py            # entrypoint app (CLI/API) bila perlu
|       |-- config.py          # Settings (pydantic-settings)
|       |-- logging.py         # konfigurasi logging
|       |-- domain/            # entitas, rules bisnis (tanpa I/O)
|       |-- services/          # use-case, orkestrasi
|       |-- repositories/      # akses data (db/api/file)
|       |-- api/               # FastAPI routers (opsional)
|       |-- db/                # engine/session, models, migrations hooks
|       `-- utils/             # helper kecil yang reusable
|-- tests/
|   |-- conftest.py
|   `-- test_*.py
|-- scripts/
|   `-- *.py                   # skrip sekali jalan (ETL, backfill)
|-- docs/                      # opsional
`-- .github/workflows/         # CI opsional

Aturan:
- Semua import production harus dari package di src/, bukan dari root repo.
- Jangan taruh kode app di root repo tanpa package.
- Jangan campur notebook dengan kode produksi. Jika perlu, buat folder notebooks/ terpisah.

## 3. Konvensi Kode

- Ikuti PEP 8.
- Pakai nama deskriptif, hindari singkatan.
- Pakai early return.
- Batasi fungsi 30-60 baris. Jika membesar, pecah.
- Jangan pakai print untuk aplikasi. Pakai logging.
- Komentar hanya untuk constraint yang tidak terlihat dari kode.

Naming:
- file dan function: snake_case.
- class: PascalCase.
- constant: UPPER_SNAKE_CASE.

## 4. Type Hints

- Tulis type hints untuk fungsi publik dan bagian inti.
- Gunakan dataclass untuk struktur data sederhana.
- Gunakan Pydantic BaseModel untuk data masuk/keluar (API, IO, config).

Contoh:
from dataclasses import dataclass

@dataclass(frozen=True)
class Student:
    id: str
    gpa: float

def predict_risk(student: Student) -> str:
    ...

## 5. Error Handling

- Jangan raise string.
- Buat exception domain yang jelas.
- Saat membungkus error, gunakan raise ... from err.

Contoh:
class DataSourceError(Exception):
    pass

def load_data(path: str) -> bytes:
    try:
        ...
    except OSError as err:
        raise DataSourceError("Gagal membaca file data") from err

## 6. Logging

- Buat logger per modul: logger = logging.getLogger(__name__).
- Log harus punya konteks: id, parameter penting, durasi.
- Jangan log secret.

## 7. Konfigurasi dan Secrets

- Simpan contoh konfigurasi di .env.example.
- Jangan commit .env asli.
- Gunakan pydantic-settings untuk load env dan validasi.

Contoh:
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    environment: str = "local"

settings = Settings()

## 8. Data Access (DB / File / API)

- Pisahkan domain dari I/O.
- Repositories hanya mengurus baca tulis data.
- Services mengurus aturan dan alur.
- Jangan taruh query DB di router API.

Contoh layering:
api/router -> service -> repository -> db

## 9. Testing

- Pakai pytest.
- Test fokus ke behavior, bukan implementasi.
- Pakai fixture untuk setup data dan dependency.
- Pisahkan unit test dan integration test jika project membesar.

Naming:
- tests/test_<module>.py
- fungsi test_*

## 10. Tooling Wajib

Ruff:
- ruff check . untuk lint.
- ruff format . untuk format.

mypy:
- Jalankan mypy pada package src/<package_name>.

pre-commit:
- Jalankan hooks sebelum commit (ruff, ruff-format, mypy, pytest bila perlu).

## 11. Dependency Rules

- Semua dependency ada di pyproject.toml.
- Pin major version. Commit lockfile jika tim Anda membutuhkannya.
- Pisahkan dependencies runtime dan dev.

## 12. Template Konfigurasi Minimal

pyproject.toml (contoh ringkas):
[project]
name = "<project_name>"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]

.pre-commit-config.yaml (contoh ringkas):
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace

## 13. Sebelum Selesai

- Anda jalankan: ruff check, ruff format, mypy, pytest.
- Anda pastikan struktur folder tetap rapi, tidak ada file eksperimen nyasar di root.
