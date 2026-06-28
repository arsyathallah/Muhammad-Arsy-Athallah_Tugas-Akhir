# Streamlit App — Sistem Rekomendasi Video Game (UAT)

## Setup (sekali saja)

### 1. Copy data files dari Kaggle output
Setelah NB2 atau NB3 selesai, download dari `/kaggle/working/` dan letakkan di `data/`:
```
streamlit_app/data/
├── item_plm_embeddings.npy   ← dari NB2 atau NB3
└── item_id_list.pkl          ← dari NB2 atau NB3
```

### 2. Download metadata game
```bash
cd recommendation-system
pip install requests tqdm
python streamlit_app/scripts/download_metadata.py
```
Script ini mendownload metadata dari McAuley Lab dan membuat `data/game_metadata.csv`.
Ukuran download: ~200-500 MB, waktu: ~5-10 menit.

### 3. Install dependencies
```bash
pip install -r streamlit_app/requirements.txt
```

### 4. Jalankan lokal
```bash
streamlit run streamlit_app/app.py
```

---

## Deploy ke Streamlit Cloud (gratis)

1. Push folder `streamlit_app/` ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set **Main file path**: `streamlit_app/app.py`
4. Upload data files via Streamlit Cloud secrets atau sertakan di repo

**Catatan**: File `item_plm_embeddings.npy` ~80MB — perlu Git LFS jika > 100MB.

---

## Struktur Hasil UAT

Setiap responden menghasilkan 1 baris di `data/uat_results.csv`:

| Kolom | Isi |
|-------|-----|
| session_id | ID unik per responden |
| responder | Nama/inisial |
| rated_count | Jumlah game yang dirating |
| relevant_count | Rekomendasi yang dianggap relevan |
| sus_score | Skor SUS (0-100) |
| sus_answers | JSON jawaban per pertanyaan |
| feedback | JSON feedback per rekomendasi |

**SUS Score interpretation:**
- ≥ 80: Excellent
- ≥ 68: Good (above average)
- < 68: Needs improvement

---

## Konfigurasi

Edit konstanta di `app.py`:
```python
MIN_RATINGS = 5    # minimum game dirating sebelum lanjut
TOP_N       = 10   # jumlah rekomendasi ditampilkan
```
