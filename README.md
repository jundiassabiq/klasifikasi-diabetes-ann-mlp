# Klasifikasi Status Diabetes dengan ANN-MLP

Tugas Besar Kecerdasan Buatan (IF21308). Proyek ini mengklasifikasikan status diabetes (tidak diabetes, prediabetes, diabetes) dari indikator kesehatan responden menggunakan Artificial Neural Network berarsitektur Multilayer Perceptron (MLP).

Dokumen ini menjelaskan keadaan proyek saat ini, yaitu **tahap persiapan data (preprocessing) yang sudah selesai**, dan cara memakai hasilnya untuk tahap pemodelan.

## Status

| Tahap | Status |
|---|---|
| Eksplorasi dan audit data | Selesai |
| Pra-pemrosesan data (dedup, split, scaling, class weight) | Selesai |
| Baseline dan pelatihan model MLP | Belum dimulai |
| Evaluasi, analisis pola faktor kesehatan | Belum dimulai |

## Struktur repositori

```
.
├── README.md
├── 01_preprocessing.ipynb        Notebook Colab: audit data dan preprocessing
├── dataset/                      Data mentah (3 file CSV, BRFSS 2015)
├── data/
│   └── processed/                Data siap latih (lihat bagian "Keluaran preprocessing")
└── reports/                      Tabel dan gambar hasil eksplorasi (CSV dan PNG)
```

Catatan: pada versi folder saat ini, `processed` dan `reports` masih berada di dalam folder hasil unduhan Google Drive (`processed-...` dan `reports-...`). Pindahkan isinya ke `data/processed/` dan `reports/` agar sesuai struktur di atas.

## Dataset

- **Nama:** CDC Diabetes Health Indicators (survei BRFSS 2015, Amerika Serikat), diambil dari Kaggle.
- **File utama:** `diabetes_012_health_indicators_BRFSS2015.csv`
- **Ukuran:** 253.680 baris, 22 kolom (21 fitur input dan 1 target `Diabetes_012`).
- **Kualitas:** tidak ada missing value. Semua nilai berada dalam rentang yang valid.
- **File lain di `dataset/`:** `diabetes_binary_health_indicators_BRFSS2015.csv` (target biner) dan `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (biner, seimbang 50/50, 70.692 baris). Keduanya belum dipakai dan disiapkan untuk eksperimen pembanding.

### Target

| Nilai | Kategori | Jumlah (asli) | Persen |
|---|---|---|---|
| 0 | Tidak diabetes | 213.703 | 84,24% |
| 1 | Prediabetes | 4.631 | 1,83% |
| 2 | Diabetes | 35.346 | 13,93% |

Rasio kelas mayoritas terhadap minoritas sekitar 46 : 1.

### Fitur (21 atribut)

| Tipe | Atribut | Rentang |
|---|---|---|
| Biner (14) | HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex | 0 atau 1 |
| Numerik (3) | BMI | 12 sampai 98 |
| | MentHlth (jumlah hari kesehatan mental kurang baik) | 0 sampai 30 |
| | PhysHlth (jumlah hari kesehatan fisik kurang baik) | 0 sampai 30 |
| Ordinal (4) | GenHlth (persepsi kesehatan umum) | 1 sampai 5 |
| | Age (kelompok usia) | 1 sampai 13 |
| | Education | 1 sampai 6 |
| | Income | 1 sampai 8 |

## Pra-pemrosesan

Seluruh langkah ada di `01_preprocessing.ipynb` dan dijalankan di Google Colab. Seed acak yang dipakai adalah **42**.

| No | Langkah | Hasil |
|---|---|---|
| 1 | Audit data | Missing value 0. Duplikat penuh 23.899 baris (9,42%). Baris berfitur identik tetapi berlabel berbeda: 3.707 baris. |
| 2 | Hapus duplikat penuh | 253.680 menjadi **229.781** baris. |
| 3 | Stratified split 70/15/15 | Train 160.846, validasi 34.467, test 34.468. |
| 4 | Scaling | `StandardScaler` pada 7 kolom (BMI, MentHlth, PhysHlth, GenHlth, Age, Education, Income). Scaler dipasang **hanya pada train**. |
| 5 | Penanganan ketimpangan kelas | Class weight dari data train. Tidak ada resampling. |

### Keputusan dan alasannya

- **Duplikat dibuang sebelum split.** Baris yang sama di train dan test akan membuat skor terlalu optimis. Sekitar 99% baris yang dibuang berasal dari kelas 0, sehingga proporsi kelas bergeser (lihat tabel di bawah).
- **Baris berlabel bertentangan dipertahankan.** Ini adalah noise label bawaan survei: dua responden dengan jawaban identik tetapi status berbeda. Membuangnya akan membuat data uji lebih mudah dari kondisi nyata.
- **Scaler hanya dipasang pada train.** Rata-rata validasi dan test yang tidak persis 0 menunjukkan tidak ada kebocoran informasi.
- **Kolom biner tidak di-scale.** Nilainya sudah 0 atau 1.
- **Class weight, bukan SMOTE.** Lebih sederhana, tidak membuat sampel sintetis dengan nilai pecahan pada kolom biner, dan data tetap asli. SMOTE (atau SMOTENC) boleh dicoba kemudian sebagai pembanding, hanya pada data train.
- **Tidak ada transformasi log.** MentHlth dan PhysHlth bimodal (69,3% dan 63,1% bernilai 0, serta 4,8% dan 7,6% bernilai 30), sehingga transformasi log tidak menormalkannya.

### Distribusi kelas sebelum dan sesudah deduplikasi

| Kategori | Sebelum | Sesudah | % sebelum | % sesudah |
|---|---|---|---|---|
| Tidak diabetes | 213.703 | 190.055 | 84,24 | 82,71 |
| Prediabetes | 4.631 | 4.629 | 1,83 | 2,01 |
| Diabetes | 35.346 | 35.097 | 13,93 | 15,27 |
| Total | 253.680 | 229.781 | 100 | 100 |

### Distribusi kelas per subset

| Subset | Jumlah | Kelas 0 | Kelas 1 | Kelas 2 |
|---|---|---|---|---|
| Train | 160.846 | 133.038 (82,71%) | 3.240 (2,01%) | 24.568 (15,27%) |
| Validasi | 34.467 | 28.508 (82,71%) | 695 (2,02%) | 5.264 (15,27%) |
| Test | 34.468 | 28.509 (82,71%) | 694 (2,01%) | 5.265 (15,28%) |

## Keluaran preprocessing

Folder `data/processed/`:

| File | Isi |
|---|---|
| `train.csv` | 160.846 baris, 22 kolom, sudah di-scale |
| `val.csv` | 34.467 baris, 22 kolom, sudah di-scale |
| `test.csv` | 34.468 baris, 22 kolom, sudah di-scale |
| `scaler.joblib` | Objek `StandardScaler` yang sudah di-fit pada train |
| `scaler_params.csv` | Mean dan std per kolom yang di-scale |
| `class_weights.json` | Bobot kelas untuk training |

**Urutan kolom** pada ketiga file CSV (perhatikan: berbeda dari file mentah):

1. 14 kolom biner: `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`
2. 3 kolom numerik: `BMI`, `MentHlth`, `PhysHlth` (sudah di-scale)
3. 4 kolom ordinal: `GenHlth`, `Age`, `Education`, `Income` (sudah di-scale)
4. Kolom target `Diabetes_012` (bilangan bulat 0, 1, 2), selalu di posisi terakhir

**Class weight:**

```json
{"0": 0.403, "1": 16.5479, "2": 2.1823}
```

### Cara memuat data (Keras)

```python
import json
import pandas as pd

TARGET = "Diabetes_012"

train = pd.read_csv("data/processed/train.csv")
val   = pd.read_csv("data/processed/val.csv")
test  = pd.read_csv("data/processed/test.csv")   # hanya untuk evaluasi akhir

FEATURES = [c for c in train.columns if c != TARGET]   # 21 fitur

X_train = train[FEATURES].values.astype("float32"); y_train = train[TARGET].values
X_val   = val[FEATURES].values.astype("float32");   y_val   = val[TARGET].values
X_test  = test[FEATURES].values.astype("float32");  y_test  = test[TARGET].values

# kunci JSON bertipe string, Keras membutuhkan kunci integer
with open("data/processed/class_weights.json") as f:
    class_weight = {int(k): v for k, v in json.load(f).items()}

# contoh pemakaian
# model.fit(X_train, y_train, validation_data=(X_val, y_val),
#           class_weight=class_weight, ...)
```

Untuk memproses data baru (misalnya antarmuka prediksi), gunakan scaler yang sama dan terapkan hanya pada 7 kolom:

```python
import joblib
scaler = joblib.load("data/processed/scaler.joblib")
SCALE_COLS = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]
df_baru[SCALE_COLS] = scaler.transform(df_baru[SCALE_COLS])
```

## Aturan pemakaian data

1. **`test.csv` hanya dipakai satu kali** untuk evaluasi akhir. Pemilihan model, tuning, dan early stopping memakai `val.csv` (atau cross-validation pada train).
2. **Jangan memasang ulang scaler.** Selalu pakai `scaler.joblib`.
3. **Resampling (SMOTE atau oversampling) hanya pada data train**, tidak pernah pada validasi atau test. Jika dipakai, jangan kombinasikan dengan class weight tanpa alasan yang dicatat.
4. **Metrik utama adalah macro F1 dan recall per kelas**, bukan akurasi. Menebak "tidak diabetes" untuk semua sampel sudah memberi akurasi sekitar 82,7%.

## Keterbatasan data

- **Noise label.** Sebanyak 3.707 baris (1,61% data bersih) punya "kembaran" dengan fitur identik tetapi label berbeda. Kelas prediabetes paling terdampak: 337 baris, atau 7,28% dari kelasnya, dibandingkan 4,46% untuk diabetes dan 0,95% untuk tidak diabetes. Akibatnya recall kelas prediabetes diperkirakan rendah, dan itu berasal dari data, bukan dari model.
- **Kembaran lintas subset.** Sekitar 389 baris test dan 392 baris validasi memiliki fitur yang sama dengan baris di train tetapi label berbeda. Ini bukan kebocoran, melainkan konsekuensi dari noise label di atas, dan menjadi batas atas akurasi yang mungkin dicapai (sekitar 1,1% sampel test).
- **Jumlah sampel prediabetes terbatas.** Hanya 694 sampel di test dan 695 di validasi, sehingga metrik kelas ini lebih berisik. Laporkan simpangan baku antar fold jika memakai cross-validation.
- **Sifat data.** Data berasal dari survei swakelola (self-reported) di Amerika Serikat tahun 2015. Model ini menunjukkan pola faktor risiko, bukan alat diagnosis.
- **Pergeseran proporsi kelas.** Deduplikasi menurunkan proporsi kelas 0 dari 84,24% menjadi 82,71%, jadi angka prevalensi pada dataset bersih tidak sama dengan data asli.

## Isi folder `reports/`

| File | Isi |
|---|---|
| `01_kualitas_data.csv` | Jumlah baris, missing, duplikat, dan baris berlabel bertentangan |
| `02_rentang_nilai.csv` | Nilai unik atau rentang tiap atribut |
| `03_statistik_deskriptif.csv` | Statistik dan skewness fitur numerik dan ordinal |
| `04_distribusi_kelas.csv` | Distribusi kelas target (data asli) |
| `05_deskripsi_atribut.csv` | Tipe, min, max, dan jumlah nilai unik tiap atribut |
| `06_perbandingan_dedup.csv` | Distribusi kelas sebelum dan sesudah dedup, serta baris berlabel bertentangan |
| `07_ringkasan_dedup.csv` | Ringkasan penghapusan duplikat per kelas |
| `08_ringkasan_split.csv` | Ukuran dan proporsi kelas tiap subset |
| `09_pemeriksaan_scaling.csv` | Mean dan std hasil scaling (bukti tidak ada kebocoran) |
| `fig_distribusi_kelas.png` | Grafik distribusi kelas target |
| `fig_distribusi_numerik_ordinal.png` | Histogram fitur numerik dan ordinal |
| `fig_proporsi_biner.png` | Proporsi nilai 1 pada fitur biner |

## Cara mereproduksi

1. Unggah `dataset.zip` ke Google Drive pada `MyDrive/Klasifikasi-Diabetes/`.
2. Buka `01_preprocessing.ipynb` di Google Colab dan jalankan seluruh sel dari atas ke bawah. Izinkan akses Google Drive ketika diminta.
3. Hasil tersimpan di Drive pada `data/processed/` dan `reports/`.

Pustaka yang dipakai: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib` (semuanya sudah tersedia di Colab).

## Rencana tahap berikutnya

1. Baseline pembanding: Logistic Regression dan Random Forest dengan class weight.
2. Model MLP sesuai rancangan: 21 neuron input, hidden layer 16 dan 8 neuron (ReLU), output 3 neuron (Softmax). Optimizer Adam, loss sparse categorical crossentropy, EarlyStopping pada validation loss.
3. Eksperimen penanganan ketimpangan kelas (tanpa penanganan, class weight, SMOTE) dan tuning hyperparameter.
4. Evaluasi akhir pada test set: accuracy, precision, recall, F1 per kelas, macro F1, confusion matrix, AUC-ROC one-vs-rest, serta cross-validation pada konfigurasi terbaik.
5. Analisis pola faktor kesehatan melalui permutation importance atau SHAP.
