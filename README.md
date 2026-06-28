# Sistem Rekomendasi Produk Berbasis Collaborative Filtering
### pada Lingkungan Terdistribusi

**Tugas Akhir — ET234801**  
**Mahasiswa:** Muhammad Arsy Athallah (5027221048)  
**Dosen Pembimbing:** Hafara Firdausi, S.Kom., M.Kom.  
**Institut Teknologi Sepuluh Nopember**

---

## Deskripsi

Penelitian ini membangun dan membandingkan tiga pendekatan sistem rekomendasi produk pada dataset Amazon Product Reviews 2023 (Video Games) yang diimplementasikan di atas infrastruktur terdistribusi MongoDB:

| Model | Pendekatan | Library |
|-------|-----------|---------|
| **ALS-CF** | Collaborative Filtering (Matrix Factorization) | `implicit` |
| **PLM-CBF** | Content-Based Filtering (BERT Embeddings + VADER) | `sentence-transformers` |
| **GRAM-CF** | Graph-Based + Bayesian Personalized Ranking | `PyTorch` |

---

## Struktur Folder

```
recommendation-system/
├── src/                        # Source code utama
│   ├── config.py               # Konfigurasi global
│   ├── data/
│   │   ├── preprocessing.py    # Filtering, train-test split
│   │   ├── download_dataset.py # Download dari HuggingFace
│   │   └── mongodb_loader.py   # Load data ke MongoDB
│   ├── models/
│   │   ├── collaborative.py    # ALS-CF
│   │   ├── plm_content_based.py# PLM-CBF
│   │   ├── gram_model.py       # GRAM-CF (arsitektur PyTorch)
│   │   └── gram_trainer.py     # GRAM-CF (training loop + BPR)
│   ├── evaluation/
│   │   └── metrics.py          # Precision@K, Recall@K, NDCG@K, dll.
│   └── utils/
│       └── logger.py
│
├── notebooks/                  # Script pelatihan per model (Kaggle-ready)
│   ├── 01_als_cf.py            # Training ALS-CF
│   ├── 01b_als_cf_tuning.py    # Hyperparameter tuning ALS
│   ├── 02_plm_cbf.py           # Training PLM-CBF
│   ├── 03_gram_cf.py           # Training GRAM-CF
│   └── 04_model_comparison.py  # Perbandingan ketiga model
│
├── streamlit_app/              # Aplikasi Suggestify
│   ├── app.py                  # Entry point
│   ├── database.py             # Koneksi MongoDB
│   ├── recommender.py          # Inference pipeline
│   ├── requirements.txt
│   ├── assets/
│   │   └── logo.png
│   └── scripts/                # Setup & migrasi data
│
├── docker/                     # Konfigurasi MongoDB Sharded Cluster
│   ├── docker-compose.yml
│   ├── init-mongo-sharding.js  # Kode Sumber 3.12 (Buku TA)
│   └── init-sharding.sh
│
├── scripts/                    # Pipeline utama
│   ├── 01_download_data.py
│   └── 02_preprocess_and_load.py
│
├── results/                    # Hasil evaluasi
│   ├── all_metrics_comparison.csv
│   ├── gram_metrics.json
│   └── comparison_report.txt
│
├── data/                       # Tidak di-push (lihat .gitignore)
│   ├── raw/                    # Dataset mentah dari HuggingFace
│   └── processed/              # Train/test split hasil preprocessing
│
├── generate_figures.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/arsyathallah/Muhammad-Arsy-Athallah_Tugas-Akhir.git
cd Source-Code-Tugas-Akhir

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
cp .env.example .env
```

### 2. Jalankan MongoDB Sharded Cluster

```bash
cd docker
docker-compose up -d
# Verifikasi:
docker exec -it mongo-router mongosh --eval "sh.status()"
cd ..
```

### 3. Download & Preprocessing Dataset

```bash
python scripts/01_download_data.py
python scripts/02_preprocess_and_load.py
```

### 4. Training Model

```bash
python notebooks/01_als_cf.py           # ALS-CF
python notebooks/02_plm_cbf.py          # PLM-CBF
python notebooks/03_gram_cf.py          # GRAM-CF
python notebooks/04_model_comparison.py # Perbandingan ketiga model
```

### 5. Jalankan Aplikasi Suggestify

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## Arsitektur Sistem Terdistribusi (MongoDB)

```
         Application (Python / Streamlit)
                        │
              ┌─────────▼─────────┐
              │   mongos router   │  ← Entry point (port 27017)
              └─────────┬─────────┘
                        │ routing berdasarkan shard key
          ┌─────────────┴─────────────┐
          ▼                           ▼
  ┌──────────────┐           ┌──────────────┐
  │ Shard 1      │           │ Shard 2      │
  │ (port 27018) │           │ (port 27020) │
  │ user_id data │           │ product data │
  └──────────────┘           └──────────────┘
                        ▲
             ┌──────────┴──────────┐
             │   Config Server     │  ← Metadata cluster (port 27019)
             └─────────────────────┘
```

**Shard key strategy:**
- `reviews`, `ratings`, `recommendations`, `user_profiles` → shard key: `user_id` (hashed)
- `product_profiles` → shard key: `product_id` (hashed)

---

## Dataset

**Amazon Product Reviews 2023 — Video Games**  
Sumber: [HuggingFace — McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

| Parameter | Nilai |
|-----------|-------|
| Total interaksi (setelah filtering) | ~814.586 |
| Jumlah pengguna unik | 94.762 |
| Jumlah produk unik | 25.612 |
| Sparsitas matriks | 99,97% |

---

## Hasil Evaluasi (K=10)

| Model | NDCG@10 | Precision@10 | Recall@10 | Hit Rate@10 |
|-------|---------|--------------|-----------|-------------|
| ALS-CF | **0.0627** | **0.0122** | **0.1075** | **0.1184** |
| PLM-CBF | 0.0078 | 0.0016 | 0.0147 | 0.0162 |
| GRAM-CF | 0.0053 | 0.0013 | 0.0114 | 0.0128 |

---

## Tech Stack

- **Python 3.8+**
- **MongoDB 7.0** — Distributed database dengan sharding & replica set
- **Docker Compose** — Orkestrasi MongoDB cluster
- **implicit** — ALS dengan backend C/Cython
- **sentence-transformers** — PLM embedding (all-MiniLM-L6-v2)
- **PyTorch 2.0+** — GRAM-CF dengan BPR Loss
- **VADER** — Analisis sentimen ulasan
- **Streamlit** — Antarmuka aplikasi Suggestify

---

## Lisensi

MIT License

## Author

**Muhammad Arsy Athallah** — NRP 5027221048  
Departemen Teknik Komputer, Institut Teknologi Sepuluh Nopember
