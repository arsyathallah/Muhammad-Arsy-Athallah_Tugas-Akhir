

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

OUT_DIR = "/kaggle/working"

import glob

def find_data_dir():
    matches = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    if not matches:
        raise FileNotFoundError("train.parquet tidak ditemukan di /kaggle/input/")
    return os.path.dirname(matches[0])

def find_nb_output(filename):
    matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    return os.path.dirname(matches[0]) if matches else None

DATA_DIR = find_data_dir()
NB1_DIR  = find_nb_output("cf_scores.parquet")
NB2_DIR  = find_nb_output("cbf_scores.parquet")
NB3_DIR  = find_nb_output("gram_scores.parquet")

print("All imports loaded ")
print(f"Data    : {DATA_DIR}")
print(f"NB1 out : {NB1_DIR}")
print(f"NB2 out : {NB2_DIR}")
print(f"NB3 out : {NB3_DIR}")

test_df = pd.read_parquet(f"{DATA_DIR}/test.parquet")
print(f"Test set: {len(test_df):,} rows | {test_df['user_id'].nunique():,} users")

def load_scores(folder, filename):
    if folder is None:
        print(f"  SKIP: {filename} (dataset not attached)")
        return pd.DataFrame()
    path = f"{folder}/{filename}"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  Loaded: {path}  ({len(df):,} rows)")
        return df
    print(f"  NOT FOUND: {path}")
    return pd.DataFrame()

def load_metrics(folder, filename):
    if folder is None:
        return {}
    path = f"{folder}/{filename}"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

print("\nLoading scores...")
cf_scores   = load_scores(NB1_DIR, "cf_scores.parquet")
cbf_scores  = load_scores(NB2_DIR, "cbf_scores.parquet")
gram_scores = load_scores(NB3_DIR, "gram_scores.parquet")

print("\nLoading metrics...")
cf_metrics   = load_metrics(NB1_DIR, "cf_metrics.json")
cbf_metrics  = load_metrics(NB2_DIR, "cbf_metrics.json")
gram_metrics = load_metrics(NB3_DIR, "gram_metrics.json")

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
            "NDCG"     : round(float(np.mean(ndcg_l)), 4) if ndcg_l else 0.0,
            "Precision": round(float(np.mean(prec_l)), 4) if prec_l else 0.0,
            "Recall"   : round(float(np.mean(rec_l)),  4) if rec_l  else 0.0,
            "HitRate"  : round(float(np.mean(hit_l)),  4) if hit_l  else 0.0,
            "N"        : len(ndcg_l),
        }
    return results

K_LIST    = [5, 10, 20]
THRESHOLD = 3.0

models_to_eval = [
    ("ALS-CF",    cf_scores,   "cf_score"),
    ("PLM-CBF",   cbf_scores,  "cbf_score"),
    ("GRAM-CF",   gram_scores, "gram_score"),
]

all_results = []
print("\nEvaluating all models...")
print("=" * 60)

for model_name, score_df, score_col in models_to_eval:
    if score_df.empty:
        print(f"  SKIP: {model_name} (no scores)")
        continue
    if score_col not in score_df.columns:
        score_col = [c for c in score_df.columns if "score" in c.lower()]
        if not score_col:
            print(f"  SKIP: {model_name} (score column not found)")
            continue
        score_col = score_col[0]
    print(f"  → {model_name} (col: {score_col})")
    metrics = compute_ranking_metrics(score_df, test_df, score_col, K_LIST, THRESHOLD)
    for k, m in metrics.items():
        all_results.append({
            "Model"    : model_name,
            "K"        : k,
            "NDCG@K"   : m["NDCG"],
            "Prec@K"   : m["Precision"],
            "Recall@K" : m["Recall"],
            "HitRate@K": m["HitRate"],
            "Users"    : m["N"],
        })

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("FULL MODEL COMPARISON")
print("=" * 70)

for k in K_LIST:
    subset = (
        results_df[results_df["K"] == k]
        .sort_values("NDCG@K", ascending=False)
        [["Model", "NDCG@K", "Prec@K", "Recall@K", "HitRate@K", "Users"]]
    )
    print(f"\nK = {k}")
    print(subset.to_string(index=False))

print("\n" + "=" * 70)
print("HIGHLIGHT: NDCG@10")
print("=" * 70)
k10 = results_df[results_df["K"] == 10].sort_values("NDCG@K", ascending=False)
for _, row in k10.iterrows():
    bar = "█" * int(row["NDCG@K"] * 600)
    print(f"  {row['Model']:<30} {row['NDCG@K']:.4f}  {bar}")

winner = k10.iloc[0]["Model"] if not k10.empty else "-"
print(f"\n Best model (NDCG@10): {winner}")

results_df.to_csv(f"{OUT_DIR}/all_metrics_comparison.csv", index=False)
print(f"\nFull table saved → {OUT_DIR}/all_metrics_comparison.csv")

from math import pi

sns.set_style("whitegrid")

COLORS = {
    "ALS-CF"  : "#4C72B0",
    "PLM-CBF" : "#DD8452",
    "GRAM-CF" : "#55A868",
}

metrics_to_plot = ["NDCG@K", "Prec@K", "Recall@K"]
axis_titles     = ["NDCG@K", "Precision@K", "Recall@K"]
model_names     = results_df["Model"].unique().tolist()
x      = np.arange(len(K_LIST))
width  = 0.8 / len(model_names)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Perbandingan Model Rekomendasi — Amazon Reviews 2023 (Video Games)",
    fontsize=13, fontweight='bold', y=1.02
)

for ax_idx, (metric, title) in enumerate(zip(metrics_to_plot, axis_titles)):
    ax = axes[ax_idx]
    for i, mname in enumerate(model_names):
        mdata  = results_df[results_df["Model"] == mname].sort_values("K")
        vals   = mdata[metric].values
        offset = (i - len(model_names) / 2 + 0.5) * width
        color  = COLORS.get(mname, "#999999")
        bars   = ax.bar(x + offset, vals, width=width * 0.9,
                        label=mname, color=color, alpha=0.88, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.4f}", ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color='#333')
    ax.set_xlabel("K", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_LIST])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    if ax_idx == 0:
        ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Chart 1 saved → {OUT_DIR}/model_comparison.png")

k10 = results_df[results_df["K"] == 10].sort_values("NDCG@K", ascending=False).reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
bar_colors = [COLORS.get(m, "#999") for m in k10["Model"]]
bars = ax2.bar(k10["Model"], k10["NDCG@K"], color=bar_colors,
               width=0.5, alpha=0.9, edgecolor="white")

winner_idx = int(k10["NDCG@K"].idxmax())
bars[winner_idx].set_edgecolor("#FFD700")
bars[winner_idx].set_linewidth(3)

for bar, val in zip(bars, k10["NDCG@K"]):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.0003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax2.annotate(" Terbaik",
             xy=(winner_idx, k10["NDCG@K"].max()),
             xytext=(winner_idx, k10["NDCG@K"].max() * 1.14),
             ha='center', fontsize=10, color='#B8860B', fontweight='bold')
ax2.set_title("Perbandingan NDCG@10 antar Model", fontsize=13, fontweight="bold")
ax2.set_ylabel("NDCG@10", fontsize=11)
ax2.set_ylim(0, k10["NDCG@K"].max() * 1.35)
ax2.set_xlabel("")
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ndcg10_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Chart 2 saved → {OUT_DIR}/ndcg10_comparison.png")

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.suptitle("Tren Metrik seiring Penambahan K", fontsize=13, fontweight='bold')

for ax_idx, (metric, title) in enumerate(zip(
        ["NDCG@K", "Prec@K", "Recall@K"],
        ["NDCG@K", "Precision@K", "Recall@K"])):
    ax = axes3[ax_idx]
    for mname in model_names:
        mdata = results_df[results_df["Model"] == mname].sort_values("K")
        vals  = mdata[metric].values
        ks    = mdata["K"].values
        color = COLORS.get(mname, "#999")
        ax.plot(ks, vals, marker='o', color=color, label=mname,
                linewidth=2.2, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
        for k_val, v in zip(ks, vals):
            ax.annotate(f"{v:.4f}", (k_val, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha='center', fontsize=7.5, color=color, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("K", fontsize=10)
    ax.set_xticks(K_LIST)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    if ax_idx == 0:
        ax.legend(fontsize=9)
        ax.set_ylabel("Score", fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/metric_trend.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Chart 3 saved → {OUT_DIR}/metric_trend.png")

k10_heat = (results_df[results_df["K"] == 10]
            .set_index("Model")[["NDCG@K", "Prec@K", "Recall@K", "HitRate@K"]]
            .rename(columns={"NDCG@K": "NDCG@10", "Prec@K": "Precision@10",
                              "Recall@K": "Recall@10", "HitRate@K": "HitRate@10"}))

fig4, ax4 = plt.subplots(figsize=(8, 3.5))
sns.heatmap(k10_heat, annot=True, fmt=".4f", cmap="YlGn",
            linewidths=0.6, linecolor='white', ax=ax4,
            annot_kws={"size": 12, "weight": "bold"},
            cbar_kws={"shrink": 0.8, "label": "Score"})
ax4.set_title("Heatmap Metrik Evaluasi (K=10)", fontsize=13, fontweight='bold', pad=12)
ax4.set_xlabel("")
ax4.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/metrics_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Chart 4 saved → {OUT_DIR}/metrics_heatmap.png")

radar_cols   = ["NDCG@K", "Prec@K", "Recall@K", "HitRate@K"]
radar_labels = ["NDCG@10", "Precision@10", "Recall@10", "HitRate@10"]
k10_radar    = results_df[results_df["K"] == 10].set_index("Model")[radar_cols].copy()

for col in radar_cols:
    mx = k10_radar[col].max()
    k10_radar[col] = k10_radar[col] / mx if mx > 0 else 0.0

N_r    = len(radar_cols)
angles = [n / N_r * 2 * pi for n in range(N_r)] + [0]

fig5, ax5 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax5.set_facecolor('#f8f8f8')

for mname, row in k10_radar.iterrows():
    vals  = row[radar_cols].tolist() + [row[radar_cols[0]]]
    color = COLORS.get(mname, "#999")
    ax5.plot(angles, vals, linewidth=2.2, color=color, label=mname, marker='o', markersize=6)
    ax5.fill(angles, vals, alpha=0.13, color=color)

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(radar_labels, fontsize=10, fontweight='bold')
ax5.set_yticks([0.25, 0.5, 0.75, 1.0])
ax5.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color='gray')
ax5.set_ylim(0, 1)
ax5.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax5.set_title("Profil Performa Model — K=10 (Ternormalisasi)",
              fontsize=12, fontweight='bold', pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.38, 1.18), fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/radar_chart.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Chart 5 saved → {OUT_DIR}/radar_chart.png")

rmse_cf = cf_metrics.get("rmse", cf_metrics.get("RMSE", "-"))

report_lines = [
    "=" * 70,
    "LAPORAN PERBANDINGAN MODEL REKOMENDASI",
    "Dataset : Amazon Reviews 2023 (Video Games)",
    "Models  : ALS-CF | PLM-CBF | GRAM-CF",
    "=" * 70,
    "",
    "Deskripsi Model:",
    "  ALS-CF   : Alternating Least Squares — matrix factorization pada implicit feedback",
    "  PLM-CBF  : Content-Based Filtering — Sentence-BERT (all-MiniLM-L6-v2, 384-dim)",
    "  GRAM-CF  : PLM item encoder + learnable user embedding, BPR loss (GRAM NAACL 2022)",
    "",
    f"Rating Prediction (ALS-CF): RMSE = {rmse_cf}",
    "",
]

for k in [5, 10, 20]:
    subset = results_df[results_df["K"] == k].sort_values("NDCG@K", ascending=False)
    report_lines.append(f"Ranking Metrics @ K={k}:")
    report_lines.append(
        subset[["Model", "NDCG@K", "Prec@K", "Recall@K", "HitRate@K"]]
        .to_string(index=False)
    )
    report_lines.append("")

if not k10.empty:
    best = k10.iloc[0]
    report_lines += [
        "=" * 70,
        f"Best Model (NDCG@10): {best['Model']}  →  {best['NDCG@K']:.4f}",
        "=" * 70,
    ]

report_text = "\n".join(report_lines)
print(report_text)

with open(f"{OUT_DIR}/comparison_report.txt", "w") as f:
    f.write(report_text)
print(f"\nReport saved → {OUT_DIR}/comparison_report.txt")
print("\n Notebook 4 selesai — 3 model dibandingkan tanpa Hybrid.")
print("   Output visualisasi:")
print("     model_comparison.png  — grouped bar semua metrik & K")
print("     ndcg10_comparison.png — sorot model terbaik NDCG@10")
print("     metric_trend.png      — tren metrik seiring K")
print("     metrics_heatmap.png   — heatmap semua metrik K=10")
print("     radar_chart.png       — profil performa ternormalisasi")
