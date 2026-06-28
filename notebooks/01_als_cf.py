

import os, json, pickle, time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from sklearn.preprocessing import LabelEncoder

import glob

def find_data_dir():
    matches = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    if not matches:
        raise FileNotFoundError("train.parquet tidak ditemukan di /kaggle/input/")
    return os.path.dirname(matches[0])

DATA_DIR = find_data_dir()
OUT_DIR  = "/kaggle/working"

print("Libraries loaded ")
print(f"Data dir : {DATA_DIR}")
print(f"Output   : {OUT_DIR}")

train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")

print(f"Train : {len(train_df):,} rows | {train_df['user_id'].nunique():,} users | {train_df['product_id'].nunique():,} products")
print(f"Test  : {len(test_df):,}  rows | {test_df['user_id'].nunique():,} users  | {test_df['product_id'].nunique():,} products")
print(f"\nRating distribution (train):")
print(train_df['rating'].value_counts().sort_index())

user_enc = LabelEncoder()
item_enc = LabelEncoder()

all_users    = pd.concat([train_df['user_id'], test_df['user_id']]).unique()
all_products = pd.concat([train_df['product_id'], test_df['product_id']]).unique()

user_enc.fit(all_users)
item_enc.fit(all_products)

train_df['user_idx'] = user_enc.transform(train_df['user_id'])
train_df['item_idx'] = item_enc.transform(train_df['product_id'])

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)
print(f"Total unique users : {n_users:,}")
print(f"Total unique items : {n_items:,}")

alpha = 40
confidence = 1.0 + alpha * (train_df['rating'] - 1) / 4.0

item_user_matrix = csr_matrix(
    (confidence.values, (train_df['item_idx'].values, train_df['user_idx'].values)),
    shape=(n_items, n_users)
)
print(f"Sparse matrix: {item_user_matrix.shape}, nnz={item_user_matrix.nnz:,}")
sparsity = 1 - item_user_matrix.nnz / (n_users * n_items)
print(f"Sparsity: {sparsity*100:.2f}%")

ALS_CONFIG = {
    "factors"       : 128,
    "iterations"    : 75,
    "regularization": 0.001,
    "use_gpu"       : False,
}

print(f"Training ALS: factors={ALS_CONFIG['factors']}, iterations={ALS_CONFIG['iterations']}")
t0 = time.time()

model = implicit.als.AlternatingLeastSquares(**ALS_CONFIG)
model.fit(item_user_matrix)

elapsed = time.time() - t0
print(f"Training done in {elapsed:.1f}s")

TOP_N = 50

print(f"Generating top-{TOP_N} recommendations for {n_users:,} users...")
t0 = time.time()

U = np.array(model.item_factors)
V = np.array(model.user_factors)

user_history = (
    train_df.groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

all_uids, all_pids, all_scores = [], [], []
BATCH = 1024

for b_start in range(0, n_users, BATCH):
    b_end = min(b_start + BATCH, n_users)

    batch_scores = U[b_start:b_end] @ V.T

    for local_i, u_idx in enumerate(range(b_start, b_end)):
        scores = batch_scores[local_i].copy()

        for iid in user_history.get(u_idx, set()):
            scores[iid] = -np.inf

        top_idx = np.argpartition(scores, -TOP_N)[-TOP_N:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        user_str = user_enc.classes_[u_idx]
        for j in top_idx:
            if scores[j] > -np.inf:
                all_uids.append(user_str)
                all_pids.append(item_enc.classes_[j])
                all_scores.append(float(scores[j]))

elapsed = time.time() - t0
cf_scores = pd.DataFrame({"user_id": all_uids, "product_id": all_pids, "cf_score": all_scores})
print(f"Done in {elapsed:.1f}s — {len(cf_scores):,} rows for {cf_scores['user_id'].nunique():,} users")

cf_scores.to_parquet(f"{OUT_DIR}/cf_scores.parquet", index=False)
print(f"Saved → {OUT_DIR}/cf_scores.parquet")

print("Generating rating predictions on test set...")

test_df['user_idx'] = test_df['user_id'].map(
    {u: i for i, u in enumerate(user_enc.classes_)}
).fillna(-1).astype(int)
test_df['item_idx'] = test_df['product_id'].map(
    {p: i for i, p in enumerate(item_enc.classes_)}
).fillna(-1).astype(int)

valid_test = test_df[(test_df['user_idx'] >= 0) & (test_df['item_idx'] >= 0)].copy()

user_factors = model.item_factors
item_factors = model.user_factors

preds = []
BATCH = 4096
for b_start in range(0, len(valid_test), BATCH):
    batch = valid_test.iloc[b_start:b_start + BATCH]
    u_f   = user_factors[batch['user_idx'].values]
    i_f   = item_factors[batch['item_idx'].values]
    score = (u_f * i_f).sum(axis=1)
    score_norm = 1.0 + 4.0 * (score - score.min()) / (score.max() - score.min() + 1e-8)
    preds.extend(score_norm.tolist())

valid_test = valid_test.copy()
valid_test['prediction'] = preds

rmse = np.sqrt(((valid_test['rating'] - valid_test['prediction']) ** 2).mean())
mae  = (valid_test['rating'] - valid_test['prediction']).abs().mean()
print(f"ALS RMSE : {rmse:.4f}")
print(f"ALS MAE  : {mae:.4f}")

valid_test[['user_id', 'product_id', 'rating', 'prediction']].to_parquet(
    f"{OUT_DIR}/cf_predictions.parquet", index=False
)

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
            top = recs_by_user[uid][:k]
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

print("\nEvaluating CF (ALS) ranking metrics...")
metrics = compute_ranking_metrics(cf_scores, test_df, "cf_score", K_LIST, THRESHOLD)

print("\n" + "=" * 50)
print("CF (ALS) — Ranking Metrics")
print("=" * 50)
for k, m in metrics.items():
    print(f"\n  K = {k}:")
    for mk, mv in m.items():
        print(f"    {mk:<20} {mv}")

metrics["RMSE"] = round(float(rmse), 4)
metrics["MAE"]  = round(float(mae), 4)
metrics["model"] = "ALS-CF"

with open(f"{OUT_DIR}/cf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved → {OUT_DIR}/cf_metrics.json")

with open(f"{OUT_DIR}/encoders.pkl", "wb") as f:
    pickle.dump({"user_enc": user_enc, "item_enc": item_enc}, f)
print("Encoders saved → encoders.pkl")
print("\n Notebook 1 (CF-ALS) selesai!")
print(f"   Output files: cf_scores.parquet, cf_predictions.parquet, cf_metrics.json, encoders.pkl")
