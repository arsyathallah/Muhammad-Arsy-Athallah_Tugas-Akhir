

import os, sys, json, pickle, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

os.system("pip install sentence-transformers -q")

import torch
from sentence_transformers import SentenceTransformer

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR  = "/kaggle/working"

import glob

def find_data_dir():
    matches = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    if not matches:
        raise FileNotFoundError("train.parquet tidak ditemukan di /kaggle/input/")
    return os.path.dirname(matches[0])

DATA_DIR = find_data_dir()

print(f"Device   : {DEVICE}")
print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"Data dir : {DATA_DIR}")

train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")

print(f"Train : {len(train_df):,} rows | {train_df['user_id'].nunique():,} users | {train_df['product_id'].nunique():,} products")
print(f"Test  : {len(test_df):,}  rows")

PLM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_ENC = 256
MAX_CHARS = 1536

print(f"Loading PLM: {PLM_MODEL}")
encoder = SentenceTransformer(PLM_MODEL, device=DEVICE)
print(f"PLM loaded   (embedding dim: {EMBED_DIM})")

print("Aggregating review text per product...")
product_texts = {}
product_ratings = defaultdict(list)
product_sentiments = defaultdict(list)

for row in train_df.itertuples():
    pid  = row.product_id
    text = str(row.review_text) if pd.notna(row.review_text) else ""
    product_ratings[pid].append(row.rating)
    if text.strip():
        product_texts.setdefault(pid, [])
        product_texts[pid].append(text)

product_id_list = sorted(product_texts.keys())
texts_list = [
    " ".join(product_texts[pid])[:MAX_CHARS]
    for pid in product_id_list
]

print(f"Encoding {len(texts_list):,} products with PLM (batch={BATCH_ENC})...")
t0 = time.time()

item_embeddings = encoder.encode(
    texts_list,
    batch_size=BATCH_ENC,
    show_progress_bar=True,
    device=DEVICE,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
item_embeddings = item_embeddings.astype(np.float32)
elapsed = time.time() - t0

print(f"Encoding done in {elapsed:.1f}s")
print(f"Embeddings shape: {item_embeddings.shape}")

np.save(f"{OUT_DIR}/item_plm_embeddings.npy", item_embeddings)
with open(f"{OUT_DIR}/item_id_list.pkl", "wb") as f:
    pickle.dump(product_id_list, f)
print("Saved → item_plm_embeddings.npy, item_id_list.pkl")

pid_to_idx  = {pid: i for i, pid in enumerate(product_id_list)}
pid_to_emb  = {pid: item_embeddings[i] for i, pid in enumerate(product_id_list)}

product_profiles = {}
for pid in product_id_list:
    ratings = product_ratings[pid]
    n       = len(ratings)
    avg_r   = float(np.mean(ratings))
    conf    = 1 - 1 / (1 + np.log1p(n))

    product_profiles[pid] = {
        "avg_rating"  : avg_r,
        "review_count": n,
        "review_score": conf * ((avg_r - 1) / 4.0),
    }

print(f"Product profiles built: {len(product_profiles):,}")

print("Building user profiles...")
t0 = time.time()

user_sums    = defaultdict(lambda: np.zeros(EMBED_DIM, dtype=np.float32))
user_weights = defaultdict(float)
user_products = defaultdict(set)
user_ratings  = defaultdict(list)

for row in train_df.itertuples():
    uid = row.user_id
    pid = row.product_id
    r   = row.rating
    user_products[uid].add(pid)
    user_ratings[uid].append(r)
    if pid in pid_to_emb:
        w = (r - 1) / 4.0
        user_sums[uid]    += pid_to_emb[pid] * w
        user_weights[uid] += w

user_profiles = {}
for uid in user_products:
    w   = user_weights[uid]
    emb = user_sums[uid] / w if w > 0 else np.zeros(EMBED_DIM, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    user_profiles[uid] = {
        "embedding"       : emb,
        "reviewed_products": user_products[uid],
        "avg_rating_given": float(np.mean(user_ratings[uid])),
    }

elapsed = time.time() - t0
print(f"User profiles built: {len(user_profiles):,} in {elapsed:.1f}s")

TOP_N      = 50
BATCH_SCORE = 2000

item_mat_T = item_embeddings.T

review_scores  = np.array([product_profiles[pid]["review_score"]  for pid in product_id_list], dtype=np.float32)
rating_factors = np.clip(
    np.array([(product_profiles[pid]["avg_rating"] - 1) / 4.0 for pid in product_id_list], dtype=np.float32),
    0.0, 1.0
)
static = 0.1 * review_scores + 0.1 * rating_factors

print(f"Generating CBF scores for {len(user_profiles):,} users × {len(product_id_list):,} products...")
t0 = time.time()

target_users = list(user_profiles.keys())
all_uids, all_pids, all_cbf = [], [], []

for b in range(0, len(target_users), BATCH_SCORE):
    batch_users = target_users[b:b + BATCH_SCORE]

    vecs, valid_uids = [], []
    for uid in batch_users:
        emb = user_profiles[uid]["embedding"]
        if np.any(emb != 0):
            vecs.append(emb)
            valid_uids.append(uid)

    if not vecs:
        continue

    user_mat    = np.vstack(vecs).astype(np.float32)
    sims        = user_mat @ item_mat_T
    np.clip(sims, 0, None, out=sims)
    scores_batch = 0.8 * sims + static
    np.clip(scores_batch, 0.0, 1.0, out=scores_batch)

    actual_top = min(TOP_N, len(product_id_list))
    for i, uid in enumerate(valid_uids):
        row = scores_batch[i].copy()
        for rpid in user_profiles[uid]["reviewed_products"]:
            j = pid_to_idx.get(rpid)
            if j is not None:
                row[j] = -1.0
        top_idx = np.argpartition(row, -actual_top)[-actual_top:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        for j in top_idx:
            if row[j] > 0:
                all_uids.append(uid)
                all_pids.append(product_id_list[j])
                all_cbf.append(float(row[j]))

elapsed = time.time() - t0
cbf_scores = pd.DataFrame({"user_id": all_uids, "product_id": all_pids, "cbf_score": all_cbf})
print(f"CBF scoring done in {elapsed:.1f}s — {len(cbf_scores):,} rows")

cbf_scores.to_parquet(f"{OUT_DIR}/cbf_scores.parquet", index=False)
print(f"Saved → {OUT_DIR}/cbf_scores.parquet")

def compute_ranking_metrics(rec_df, test_df, score_col, k_list, threshold=3.0):
    user_relevant = {}
    for row in test_df.itertuples():
        if row.rating >= threshold:
            user_relevant.setdefault(row.user_id, set()).add(row.product_id)

    recs_by_user = (
        rec_df.sort_values(score_col, ascending=False)
        .groupby("user_id")["product_id"]
        .apply(list)
        .to_dict()
    )
    results = {}
    for k in k_list:
        ndcg_l, prec_l, rec_l, hit_l = [], [], [], []
        for uid, relevant in user_relevant.items():
            if uid not in recs_by_user:
                continue
            top  = recs_by_user[uid][:k]
            hits = sum(1 for p in top if p in relevant)
            dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top) if p in relevant)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg_l.append(dcg / idcg if idcg > 0 else 0.0)
            prec_l.append(hits / k)
            rec_l.append(hits / len(relevant) if relevant else 0.0)
            hit_l.append(1 if hits > 0 else 0)
        results[k] = {
            f"NDCG@{k}"     : round(float(np.mean(ndcg_l)), 4) if ndcg_l else 0.0,
            f"Precision@{k}": round(float(np.mean(prec_l)), 4) if prec_l else 0.0,
            f"Recall@{k}"   : round(float(np.mean(rec_l)),  4) if rec_l  else 0.0,
            f"HitRate@{k}"  : round(float(np.mean(hit_l)),  4) if hit_l  else 0.0,
            "users_evaluated": len(ndcg_l),
        }
    return results

K_LIST    = [5, 10, 20]
THRESHOLD = 3.0

print("\nGenerating eval scores for test products...")
test_products = test_df["product_id"].unique().tolist()
test_valid    = [p for p in test_products if p in pid_to_emb]

test_item_mat = np.vstack([pid_to_emb[p] for p in test_valid]).astype(np.float32)
test_static   = np.clip(
    np.array([(product_profiles.get(p, {}).get("avg_rating", 3) - 1) / 4.0
              for p in test_valid], dtype=np.float32), 0.0, 1.0
) * 0.2

all_uids_e, all_pids_e, all_scores_e = [], [], []
for b in range(0, len(target_users), BATCH_SCORE):
    batch_users = target_users[b:b + BATCH_SCORE]
    vecs, valid_uids = [], []
    for uid in batch_users:
        emb = user_profiles[uid]["embedding"]
        if np.any(emb != 0):
            vecs.append(emb)
            valid_uids.append(uid)
    if not vecs:
        continue
    user_mat     = np.vstack(vecs).astype(np.float32)
    sims         = user_mat @ test_item_mat.T
    np.clip(sims, 0, None, out=sims)
    scores_batch = 0.8 * sims + test_static
    np.clip(scores_batch, 0.0, 1.0, out=scores_batch)
    top_n = min(100, len(test_valid))
    for i, uid in enumerate(valid_uids):
        row     = scores_batch[i]
        top_idx = np.argpartition(row, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        for j in top_idx:
            if row[j] > 0:
                all_uids_e.append(uid)
                all_pids_e.append(test_valid[j])
                all_scores_e.append(float(row[j]))

cbf_eval = pd.DataFrame({"user_id": all_uids_e, "product_id": all_pids_e, "cbf_score": all_scores_e})
cbf_eval.to_parquet(f"{OUT_DIR}/cbf_eval_scores.parquet", index=False)

metrics = compute_ranking_metrics(cbf_eval, test_df, "cbf_score", K_LIST, THRESHOLD)

print("\n" + "=" * 50)
print("PLM-CBF — Ranking Metrics")
print("=" * 50)
for k, m in metrics.items():
    print(f"\n  K = {k}:")
    for mk, mv in m.items():
        print(f"    {mk:<20} {mv}")

metrics["model"] = "PLM-CBF (all-MiniLM-L6-v2)"
with open(f"{OUT_DIR}/cbf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved → {OUT_DIR}/cbf_metrics.json")
print("\n Notebook 2 (PLM-CBF) selesai!")
print("   Output: cbf_scores.parquet, cbf_eval_scores.parquet,")
print("           item_plm_embeddings.npy, item_id_list.pkl, cbf_metrics.json")
print("   → Upload output sebagai Kaggle dataset untuk dipakai di Notebook 3 & 4")
