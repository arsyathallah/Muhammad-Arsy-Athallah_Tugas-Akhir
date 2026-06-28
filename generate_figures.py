#!/usr/bin/env python3
"""
Generate missing figures for Tugas Akhir Bab IV.
Run from inside the recommendation-system/ directory:
    python generate_figures.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent
DATA    = BASE / "data" / "processed"
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})
C = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


# ══════════════════════════════════════════════════════════════════════════════
# Gambar 4.1 & 4.2 — dataset_overview.png
# ══════════════════════════════════════════════════════════════════════════════
def fig_dataset_overview():
    print("\n[1/4] Generating dataset_overview.png ...")
    df = pd.read_parquet(DATA / "train.parquet",
                         columns=["user_id", "product_id", "rating"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Kiri: Distribusi Rating
    ax = axes[0]
    rc = df["rating"].value_counts().sort_index()
    bars = ax.bar(rc.index.astype(int), rc.values,
                  color=C[0], edgecolor="white", linewidth=0.6, width=0.7)
    ax.set_title("Distribusi Rating dalam Dataset", fontweight="bold")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah Rating")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    # Kanan: Distribusi Review per Pengguna
    ax = axes[1]
    rpu = df.groupby("user_id")["product_id"].count()
    cap = rpu.quantile(0.99)
    ax.hist(rpu.clip(upper=cap), bins=50,
            color=C[1], edgecolor="white", linewidth=0.3, log=True)
    med = rpu.median()
    ax.axvline(med, color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {int(med)}")
    ax.set_title("Distribusi Jumlah Review per Pengguna", fontweight="bold")
    ax.set_xlabel("Jumlah Review")
    ax.set_ylabel("Jumlah Pengguna (skala log)")
    ax.legend()

    plt.tight_layout()
    out = FIGURES / "dataset_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Gambar 4.3, 4.4, 4.5 — sentiment_analysis.png
# ══════════════════════════════════════════════════════════════════════════════
def fig_sentiment_analysis():
    print("\n[2/4] Generating sentiment_analysis.png ...")
    df = pd.read_parquet(DATA / "train.parquet",
                         columns=["user_id", "product_id", "rating",
                                  "rating_normalized", "review_text"])

    # ── VADER on sample ───────────────────────────────────────────────────────
    vader_ok = False
    labels = []
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        SAMPLE = 20_000
        sample = (df.dropna(subset=["review_text"])
                    .sample(n=min(SAMPLE, len(df)), random_state=42))
        sia = SentimentIntensityAnalyzer()
        print(f"  Menjalankan VADER pada {len(sample):,} review...")
        scores = [sia.polarity_scores(str(t))["compound"]
                  for t in sample["review_text"]]
        labels = ["Positif" if s >= 0.05 else "Negatif" if s <= -0.05 else "Netral"
                  for s in scores]
        vader_ok = True
    except ImportError:
        print("  ⚠ vaderSentiment tidak terinstall — panel kiri dilewati")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel kiri: Distribusi Sentimen VADER
    ax = axes[0]
    if vader_ok:
        from collections import Counter
        cnt   = Counter(labels)
        order = ["Positif", "Netral", "Negatif"]
        vals  = [cnt.get(k, 0) for k in order]
        clrs  = ["#4CAF50", "#FF9800", "#F44336"]
        bars  = ax.bar(order, vals, color=clrs, edgecolor="white", linewidth=0.6)
        total = sum(vals)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:,}\n({v / total * 100:.1f}%)",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "vaderSentiment\ntidak terinstall",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("Distribusi Sentimen Produk (VADER)", fontweight="bold")
    ax.set_xlabel("Kategori Sentimen")
    ax.set_ylabel("Jumlah Review")

    # Panel tengah: Distribusi Review Score per Produk
    ax = axes[1]
    prod_scores = df.groupby("product_id")["rating_normalized"].mean()
    ax.hist(prod_scores, bins=30, color=C[0], edgecolor="white", linewidth=0.3)
    mn = prod_scores.mean()
    ax.axvline(mn, color="red", linestyle="--", linewidth=1.5,
               label=f"Rata-rata: {mn:.2f}")
    ax.set_title("Distribusi Review Score Produk", fontweight="bold")
    ax.set_xlabel("Review Score (0–1)")
    ax.set_ylabel("Jumlah Produk")
    ax.legend()

    # Panel kanan: Distribusi Jumlah Review per Produk
    ax = axes[2]
    rpp = df.groupby("product_id")["user_id"].count()
    cap = rpp.quantile(0.99)
    ax.hist(rpp.clip(upper=cap), bins=50,
            color=C[2], edgecolor="white", linewidth=0.3, log=True)
    medp = rpp.median()
    ax.axvline(medp, color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {int(medp)}")
    ax.set_title("Distribusi Jumlah Review per Produk", fontweight="bold")
    ax.set_xlabel("Jumlah Review")
    ax.set_ylabel("Jumlah Produk (skala log)")
    ax.legend()

    plt.tight_layout()
    out = FIGURES / "sentiment_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Training curve GRAM-CF
# ══════════════════════════════════════════════════════════════════════════════
def fig_gram_training_curve():
    print("\n[3/4] Generating gram_training_curve.png ...")

    candidates = [
        RESULTS / "gram_metrics.json",
        RESULTS / "gram_cf_metrics.json",
        RESULTS / "notebook_03_metrics.json",
    ]
    history = None
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            history = data.get("training_history") or data.get("history")
            if history:
                print(f"  Loaded: {p.name}")
                break

    if not history:
        print("  ⚠ gram_metrics.json tidak ditemukan di results/")
        print("    → Download gram_metrics.json dari Kaggle output notebook_03")
        print("      lalu taruh di results/ dan jalankan ulang script ini.")
        return

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"]  for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, color=C[0], linewidth=2,
            marker="o", markersize=5, zorder=3)
    ax.fill_between(epochs, losses, alpha=0.12, color=C[0])
    ax.set_title("Kurva Training Loss GRAM-CF per Epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (BPR)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(epochs)

    plt.tight_layout()
    out = FIGURES / "gram_training_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Copy gambar perbandingan yang sudah ada
# ══════════════════════════════════════════════════════════════════════════════
def copy_comparison_figures():
    print("\n[4/4] Menyalin gambar perbandingan model ...")
    names = [
        "model_comparison.png",
        "metric_trend.png",
        "metrics_heatmap.png",
        "ndcg10_comparison.png",
        "radar_chart.png",
    ]
    for name in names:
        src = RESULTS / name
        dst = FIGURES / name
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"  ✓ Disalin: {name}")
            else:
                print(f"  — Sudah ada: {name}")
        else:
            print(f"  ⚠ Tidak ditemukan: {name}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Generate Figures — Tugas Akhir Bab IV")
    print("=" * 60)

    fig_dataset_overview()
    fig_sentiment_analysis()
    fig_gram_training_curve()
    copy_comparison_figures()

    print("\n" + "=" * 60)
    print(f"  Selesai! Semua gambar tersimpan di:")
    print(f"  {FIGURES}")
    print("=" * 60)
