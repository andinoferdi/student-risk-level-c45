Implementasi Machine Learning untuk Klasifikasi Tingkat Risiko Mahasiswa Menggunakan Algoritma Decision Tree C4.5 pada College Student Management Dataset

Dosen Pengampu:
INDAH WERDININGSIH S.Si., M.Kom.

Disusun oleh:
Kelompok 9

Andino Ferdiansah 434231065
Sayu Damar Yunan 434231079
M. Catra Hanif 'Azmi 434231104

D4 TEKNIK INFORMATIKA
FAKULTAS VOKASI
UNIVERSITAS AIRLANGGA
2026

BAB I
PENDAHULUAN

1.1 Latar Belakang
Perguruan tinggi menyimpan banyak data akademik dan data aktivitas belajar, misalnya nilai, beban mata kuliah, kehadiran, dan aktivitas pada Learning Management System. Data ini dapat dipakai untuk mendukung keputusan berbasis bukti, termasuk mendeteksi mahasiswa yang berisiko mengalami masalah akademik lebih awal, sehingga tindak lanjut dapat dilakukan lebih cepat.

Laporan praktikum ini membahas implementasi klasifikasi untuk memprediksi tingkat risiko mahasiswa menggunakan College Student Management Dataset. Dataset tersebut memuat catatan mahasiswa yang mencakup performa akademik, metrik keterlibatan pada LMS, dan label tingkat risiko.

Metode yang dipakai adalah algoritma pohon keputusan C4.5 karena menghasilkan model yang mudah dijelaskan melalui aturan if-then, sehingga dosen atau pihak kampus dapat membaca alasan prediksi secara langsung. C4.5 juga mendukung atribut kategorikal dan numerik, serta membentuk pengujian pada atribut kontinu dengan penentuan batas nilai.

Dalam pelaksanaannya, proses pengolahan mengikuti alur pembersihan atau prapemrosesan, transformasi, pemodelan, lalu evaluasi hasil.

1.2 Rumusan Masalah
1. Bagaimana menyiapkan data College Student Management Dataset agar siap digunakan untuk klasifikasi tingkat risiko mahasiswa.
2. Bagaimana membangun model klasifikasi berbasis C4.5 untuk memprediksi tingkat risiko mahasiswa.
3. Bagaimana mengevaluasi kinerja model klasifikasi yang dibangun menggunakan data uji.
4. Bagaimana menampilkan aturan keputusan dari model agar hasil klasifikasi dapat dijelaskan.

1.3 Tujuan Penelitian
1. Menyiapkan dan membersihkan data sehingga fitur dan target siap dipakai untuk proses pelatihan model.
2. Mengimplementasikan algoritma C4.5 untuk melakukan klasifikasi tingkat risiko mahasiswa.
3. Mengukur kinerja model menggunakan evaluasi pada data uji.
4. Menghasilkan aturan keputusan dari pohon keputusan sebagai bentuk interpretasi model.

1.4 Manfaat Penelitian
Penelitian ini memberi manfaat praktis berupa contoh implementasi klasifikasi yang runtut, mulai dari persiapan data sampai evaluasi hasil, sehingga dapat digunakan sebagai pola kerja untuk praktikum berikutnya.

Penelitian ini memberi manfaat akademik karena menunjukkan penerapan C4.5 pada kasus pendidikan dan menekankan interpretabilitas model melalui aturan keputusan, yang relevan untuk pelaporan dan diskusi hasil.

Penelitian ini memberi manfaat bagi konteks pengelolaan pendidikan karena prediksi tingkat risiko dapat dipakai sebagai dasar prioritas pemantauan atau pendampingan.

BAB II
LANDASAN TEORI

2.1 Data Mining dan KDD
Data mining adalah proses menemukan pola atau pengetahuan yang berguna dari data. Kerangka kerja yang umum dipakai adalah Knowledge Discovery in Databases, yang terdiri dari Data Selection, Preprocessing, Transformation, Data Mining, dan Evaluation.

2.2 Algoritma Decision Tree dan C4.5
Decision tree membagi data secara rekursif berdasarkan atribut tertentu hingga menghasilkan struktur pohon. Algoritma C4.5 merupakan pengembangan dari ID3 dan memilih atribut pemecah berdasarkan gain ratio, serta mampu menangani atribut kategorik dan numerik melalui pemilihan nilai ambang. C4.5 juga menerapkan pruning untuk mengurangi overfitting.

2.3 Evaluasi Menggunakan Confusion Matrix
Confusion matrix merangkum perbandingan antara label aktual dan label prediksi. Dari confusion matrix dapat dihitung metrik akurasi, presisi, recall, dan F1-score untuk menilai performa model klasifikasi.

BAB III
IMPLEMENTASI DAN HASIL

3.1 Deskripsi Dataset
Dataset yang digunakan adalah College Student Management Dataset dari Kaggle. Dataset berisi 1.545 baris dan 15 kolom. Target yang diprediksi adalah tingkat_risiko dengan tiga kelas, yaitu Tinggi, Sedang, dan Rendah.

Link dataset:
https://www.kaggle.com/datasets/ziya07/college-student-management-dataset

Code:
```python
print("--- 1. Memuat Dataset ---")
data_path = resolve_data_path(args.data)
df_raw = pd.read_csv(data_path)
print(f"Total data awal: {len(df_raw)} baris, {df_raw.shape[1]} kolom")
```

Hasil:
```text
--- 1. Memuat Dataset ---
Total data awal: 1545 baris, 15 kolom
```

3.2 Data Selection (Pemilihan Fitur dan Target)
Pada tahap ini, kolom id_siswa tidak digunakan sebagai fitur karena hanya bersifat identitas. Fitur yang digunakan meliputi usia, jenis_kelamin, jurusan, ipk, beban_mata_kuliah, rata_rata_nilai_mata_kuliah, tingkat_kehadiran, status_studi, login_lms_bulan_lalu, rata_rata_durasi_sesi_menit, tingkat_pengumpulan_tugas, jumlah_partisipasi_forum, dan tingkat_penyelesaian_video. Kolom target adalah tingkat_risiko.

Code:
```python
print("\n--- 2. Seleksi Data (Memilih Fitur dan Target) ---")
drop_cols = ["id_siswa"]
df = clean_and_prepare(df_raw, target_col=args.target, drop_cols=drop_cols)

print(f"Kolom setelah selection: {list(df.columns)}")
print("Distribusi target:")
print(df[args.target].value_counts())
```

Hasil:
```text
--- 2. Seleksi Data (Memilih Fitur dan Target) ---
Kolom setelah selection: ['usia', 'jenis_kelamin', 'jurusan', 'ipk', 'beban_mata_kuliah', 'rata_rata_nilai_mata_kuliah', 'tingkat_kehadiran', 'status_studi', 'login_lms_bulan_lalu', 'rata_rata_durasi_sesi_menit', 'tingkat_pengumpulan_tugas', 'jumlah_partisipasi_forum', 'tingkat_penyelesaian_video', 'tingkat_risiko']
Distribusi target:
tingkat_risiko
Tinggi    805
Sedang    456
Rendah    284
Name: count, dtype: int64
```

3.3 Preprocessing dan Transformation
Pemeriksaan kualitas data menunjukkan tidak ada nilai kosong pada seluruh kolom yang digunakan, serta tidak ditemukan baris duplikat. Fitur kategorik diproses sebagai atribut kategorik oleh algoritma C4.5, sedangkan fitur numerik diproses sebagai atribut kontinu.

Code:
```python
def clean_and_prepare(df: pd.DataFrame, target_col: str, drop_cols: list):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan. Kolom yang ada: {list(df.columns)}")

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.dropna(subset=[target_col])

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().mean() >= 0.95:
                df[col] = num
            else:
                df[col] = df[col].astype(str).str.strip()

    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown").astype(str)

    df[target_col] = df[target_col].astype(str).str.strip()
    return df
```

Hasil:
```text
--- 3. Pemahaman Data (Info & Statistik Singkat) ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1545 entries, 0 to 1544
Data columns (total 14 columns):
 #   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
 0   usia                         1545 non-null   int64
 1   jenis_kelamin                1545 non-null   object
 2   jurusan                      1545 non-null   object
 3   ipk                          1545 non-null   float64
 4   beban_mata_kuliah            1545 non-null   int64
 5   rata_rata_nilai_mata_kuliah  1545 non-null   float64
 6   tingkat_kehadiran            1545 non-null   float64
 7   status_studi                 1545 non-null   object
 8   login_lms_bulan_lalu         1545 non-null   int64
 9   rata_rata_durasi_sesi_menit  1545 non-null   int64
10   tingkat_pengumpulan_tugas    1545 non-null   float64
11   jumlah_partisipasi_forum     1545 non-null   int64
12   tingkat_penyelesaian_video   1545 non-null   float64
13   tingkat_risiko               1545 non-null   object
dtypes: float64(5), int64(5), object(4)
```

3.4 Data Splitting (Train, Validation, Test)
Data dibagi menjadi data latih dan data uji dengan rasio 80:20. Dari 1.545 baris, diperoleh 1.236 baris untuk data latih dan 309 baris untuk data uji. Selanjutnya, data latih dipecah lagi menjadi train internal dan validation untuk pruning sederhana.

Code:
```python
print("\n--- 4. Pemisahan Data (Latih/Uji) ---")
X = df.drop(columns=[args.target])
y = df[args.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=args.test_size,
    random_state=42,
    stratify=y
)
print(f"Jumlah data latih: {len(X_train)} baris")
print(f"Jumlah data uji  : {len(X_test)} baris")

print("\n--- 5. Pemisahan Internal Latih/Validasi (untuk pruning sederhana) ---")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=args.val_size,
    random_state=42,
    stratify=y_train
)
print(f"Train untuk bangun pohon: {len(X_tr)} baris")
print(f"Validation untuk pruning: {len(X_val)} baris")
```

Hasil:
```text
--- 4. Pemisahan Data (Latih/Uji) ---
Jumlah data latih: 1236 baris
Jumlah data uji  : 309 baris

--- 5. Pemisahan Internal Latih/Validasi (untuk pruning sederhana) ---
Train untuk bangun pohon: 927 baris
Validation untuk pruning: 309 baris
```

3.5 Data Mining (Training C4.5)
Model C4.5 dibangun menggunakan kriteria gain ratio untuk memilih pemecah terbaik pada setiap node. Untuk fitur numerik, algoritma mencari ambang batas.

Code:
```python
print("\n--- 7. Penambangan Data (Pelatihan C4.5) ---")
model = C45Classifier(
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    min_gain_ratio=args.min_gain_ratio
)
model.fit(X_tr, y_tr, feature_types=feature_types)
```

Hasil:
```text
--- 6. Transformasi (Tipe Fitur) ---
Numerik      (10): ['usia', 'ipk', 'beban_mata_kuliah', 'rata_rata_nilai_mata_kuliah', 'tingkat_kehadiran', 'login_lms_bulan_lalu', 'rata_rata_durasi_sesi_menit', 'tingkat_pengumpulan_tugas', 'jumlah_partisipasi_forum', 'tingkat_penyelesaian_video']
Kategorikal  (3): ['jenis_kelamin', 'jurusan', 'status_studi']

--- 7. Penambangan Data (Pelatihan C4.5) ---
```

3.6 Evaluation (Confusion Matrix)
Evaluasi dilakukan pada data uji setelah proses pruning menggunakan validation. Hasil evaluasi menunjukkan akurasi 1.0000, presisi makro 1.0000, recall makro 1.0000, dan F1-score makro 1.0000.

Code:
```python
print("\n--- 8. Evaluasi (Pruning + Matriks Kebingungan) ---")
model.prune(X_val, y_val)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

print(f"Akurasi : {acc:.4f}")
print(f"Presisi : {pr:.4f} (macro)")
print(f"Recall  : {rc:.4f} (macro)")
print(f"F1-score: {f1:.4f} (macro)")

labels = sorted(y.unique().tolist())
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nMatriks Kebingungan (baris=aktual, kolom=prediksi):")
cm_df = pd.DataFrame(cm, index=[f"aktual_{l}" for l in labels], columns=[f"prediksi_{l}" for l in labels])
print(cm_df)
```

Hasil:
```text
--- 8. Evaluasi (Pruning + Matriks Kebingungan) ---
Akurasi : 1.0000
Presisi : 1.0000 (macro)
Recall  : 1.0000 (macro)
F1-score: 1.0000 (macro)

Matriks Kebingungan (baris=aktual, kolom=prediksi):
               prediksi_Rendah  prediksi_Sedang  prediksi_Tinggi
aktual_Rendah               57                0                0
aktual_Sedang                0               91                0
aktual_Tinggi                0                0              161
```

3.7 Aturan Pohon Keputusan (Ringkas)
Aturan ringkas dari pohon keputusan yang terbentuk adalah sebagai berikut.
1. Jika tingkat_kehadiran <= 0.745 maka tingkat_risiko = Tinggi.
2. Jika tingkat_kehadiran > 0.745 dan ipk <= 2.495 maka tingkat_risiko = Tinggi.
3. Jika tingkat_kehadiran > 0.745 dan ipk > 2.495 dan tingkat_kehadiran <= 0.845 maka tingkat_risiko = Sedang.
4. Jika tingkat_kehadiran > 0.845 dan ipk <= 2.995 maka tingkat_risiko = Sedang.
5. Jika tingkat_kehadiran > 0.845 dan ipk > 2.995 maka tingkat_risiko = Rendah.

Code:
```python
print("\n--- 9. Aturan Pohon (Ringkas) ---")
print(model.root.to_rules())
```

Hasil:
```text
--- 9. Aturan Pohon (Ringkas) ---
IF tingkat_kehadiran <= 0.745
  THEN class = Tinggi (n=341)
ELSE  # tingkat_kehadiran > 0.745
  IF ipk <= 2.495
    THEN class = Tinggi (n=142)
  ELSE  # ipk > 2.495
    IF tingkat_kehadiran <= 0.845
      THEN class = Sedang (n=187)
    ELSE  # tingkat_kehadiran > 0.845
      IF ipk <= 2.995
        THEN class = Sedang (n=87)
      ELSE  # ipk > 2.995
        THEN class = Rendah (n=170)
```

BAB IV
KESIMPULAN
Tahapan KDD telah berhasil diterapkan pada College Student Management Dataset. Dataset tidak memiliki missing value dan siap diproses. Model C4.5 menghasilkan aturan yang mudah diinterpretasi dan menunjukkan bahwa tingkat_kehadiran serta ipk menjadi atribut utama dalam pemisahan kelas tingkat_risiko.
