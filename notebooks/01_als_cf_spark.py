import os, json, pickle, time, glob
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def find_data_dir():
    matches = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    if not matches:
        raise FileNotFoundError("train.parquet tidak ditemukan di /kaggle/input/")
    return os.path.dirname(matches[0])


DATA_DIR = find_data_dir()
OUT_DIR = "/kaggle/working"

spark = (
    SparkSession.builder
    .appName("CF-ALS-Spark")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("Spark session ready ", spark.version)
print(f"Data dir : {DATA_DIR}")
print(f"Output   : {OUT_DIR}")

train_df = (spark.read.parquet(f"{DATA_DIR}/train.parquet")
            .select("user_id", "product_id", F.col("rating").cast("float")))
test_df = (spark.read.parquet(f"{DATA_DIR}/test.parquet")
           .select("user_id", "product_id", F.col("rating").cast("float")))
train_df.cache()
test_df.cache()

n_train = train_df.count()
n_test = test_df.count()
print(f"Train : {n_train:,} rows | "
      f"{train_df.select('user_id').distinct().count():,} users | "
      f"{train_df.select('product_id').distinct().count():,} products")
print(f"Test  : {n_test:,} rows")

user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx",
                             handleInvalid="skip").fit(train_df)
item_indexer = StringIndexer(inputCol="product_id", outputCol="item_idx",
                             handleInvalid="skip").fit(train_df)

n_users = len(user_indexer.labels)
n_items = len(item_indexer.labels)
print(f"Total unique users : {n_users:,}")
print(f"Total unique items : {n_items:,}")


def encode(df):
    out = item_indexer.transform(user_indexer.transform(df))
    return (out.withColumn("user_idx", F.col("user_idx").cast("int"))
               .withColumn("item_idx", F.col("item_idx").cast("int")))


ALPHA = 40.0
train_idx = (encode(train_df)
             .withColumn("pref", (F.col("rating") - F.lit(1.0)) / F.lit(4.0))
             .select("user_idx", "item_idx", "pref")
             .cache())
train_idx.count()

ALS_CONFIG = dict(rank=128, maxIter=75, regParam=0.001)

_best_param_files = (glob.glob("/kaggle/input/**/cf_best_params.json", recursive=True)
                     + glob.glob(f"{OUT_DIR}/cf_best_params.json"))
if _best_param_files:
    with open(_best_param_files[0]) as _f:
        _tuned = json.load(_f)
    ALS_CONFIG = {"rank": int(_tuned["rank"]),
                  "maxIter": int(_tuned["maxIter"]),
                  "regParam": float(_tuned["regParam"])}
    ALPHA = float(_tuned.get("alpha", ALPHA))
    print(f"Loaded tuned params from {_best_param_files[0]}: {ALS_CONFIG}, alpha={ALPHA}")
else:
    print(f"No tuned params found, using defaults: {ALS_CONFIG}, alpha={ALPHA}")
print(f"Training ALS: rank={ALS_CONFIG['rank']}, maxIter={ALS_CONFIG['maxIter']}, "
      f"implicitPrefs=True, alpha={ALPHA}")
t0 = time.time()

als = ALS(
    rank=ALS_CONFIG["rank"],
    maxIter=ALS_CONFIG["maxIter"],
    regParam=ALS_CONFIG["regParam"],
    userCol="user_idx",
    itemCol="item_idx",
    ratingCol="pref",
    implicitPrefs=True,
    alpha=ALPHA,
    coldStartStrategy="drop",
    seed=42,
)
model = als.fit(train_idx)

print(f"Training done in {time.time() - t0:.1f}s")

TOP_N = 50
print(f"Generating top-{TOP_N} recommendations for {n_users:,} users...")
t0 = time.time()

raw_recs = model.recommendForAllUsers(TOP_N + 50)
exploded = (raw_recs
            .select("user_idx", F.explode("recommendations").alias("rec"))
            .select("user_idx",
                    F.col("rec.item_idx").alias("item_idx"),
                    F.col("rec.rating").alias("cf_score")))

seen = train_idx.select("user_idx", "item_idx")
exploded = exploded.join(seen, ["user_idx", "item_idx"], "left_anti")

rank_w = Window.partitionBy("user_idx").orderBy(F.col("cf_score").desc())
top_recs = (exploded
            .withColumn("_rk", F.row_number().over(rank_w))
            .filter(F.col("_rk") <= TOP_N)
            .drop("_rk"))

u2s = IndexToString(inputCol="user_idx", outputCol="user_id",
                    labels=user_indexer.labels)
i2s = IndexToString(inputCol="item_idx", outputCol="product_id",
                    labels=item_indexer.labels)
top_recs = i2s.transform(u2s.transform(top_recs))

cf_scores = (top_recs
             .select("user_id", "product_id", "cf_score")
             .toPandas())
print(f"Done in {time.time() - t0:.1f}s, {len(cf_scores):,} rows for "
      f"{cf_scores['user_id'].nunique():,} users")

cf_scores.to_parquet(f"{OUT_DIR}/cf_scores.parquet", index=False)
print(f"Saved -> {OUT_DIR}/cf_scores.parquet")

print("Generating rating predictions on test set...")
test_idx = encode(test_df)
pred = model.transform(test_idx).na.drop(subset=["prediction"])

stats = pred.agg(F.min("prediction").alias("lo"),
                 F.max("prediction").alias("hi")).first()
lo, hi = float(stats["lo"]), float(stats["hi"])
span = (hi - lo) or 1e-8
pred = pred.withColumn(
    "prediction",
    F.lit(1.0) + F.lit(4.0) * (F.col("prediction") - F.lit(lo)) / F.lit(span),
)

rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",
                           predictionCol="prediction").evaluate(pred)
mae = RegressionEvaluator(metricName="mae", labelCol="rating",
                          predictionCol="prediction").evaluate(pred)
print(f"ALS RMSE : {rmse:.4f}")
print(f"ALS MAE  : {mae:.4f}")

cf_predictions = pred.select("user_id", "product_id", "rating", "prediction").toPandas()
cf_predictions.to_parquet(f"{OUT_DIR}/cf_predictions.parquet", index=False)

test_pdf = test_df.select("user_id", "product_id", "rating").toPandas()


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
            dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top) if p in relevant)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg_l.append(dcg / idcg if idcg > 0 else 0.0)
            prec_l.append(hits / k)
            rec_l.append(hits / len(relevant) if relevant else 0.0)
            hit_l.append(1 if hits > 0 else 0)

        results[k] = {
            f"NDCG@{k}": round(float(np.mean(ndcg_l)), 4) if ndcg_l else 0.0,
            f"Precision@{k}": round(float(np.mean(prec_l)), 4) if prec_l else 0.0,
            f"Recall@{k}": round(float(np.mean(rec_l)), 4) if rec_l else 0.0,
            f"HitRate@{k}": round(float(np.mean(hit_l)), 4) if hit_l else 0.0,
            "users_evaluated": len(ndcg_l),
        }
    return results


K_LIST = [5, 10, 20]
THRESHOLD = 3.0

print("\nEvaluating CF (ALS) ranking metrics...")
metrics = compute_ranking_metrics(cf_scores, test_pdf, "cf_score", K_LIST, THRESHOLD)

print("\n" + "=" * 50)
print("CF (Spark ALS) Ranking Metrics")
print("=" * 50)
for k, m in metrics.items():
    print(f"\n  K = {k}:")
    for mk, mv in m.items():
        print(f"    {mk:<20} {mv}")

metrics["RMSE"] = round(float(rmse), 4)
metrics["MAE"] = round(float(mae), 4)
metrics["model"] = "ALS-CF-Spark"
metrics["engine"] = "pyspark.ml.recommendation.ALS"

with open(f"{OUT_DIR}/cf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved -> {OUT_DIR}/cf_metrics.json")

with open(f"{OUT_DIR}/encoders.pkl", "wb") as f:
    pickle.dump({"user_labels": list(user_indexer.labels),
                 "item_labels": list(item_indexer.labels)}, f)
print("Encoders (labels) saved -> encoders.pkl")

spark.stop()
print("\nNotebook 1 (CF-ALS Spark) selesai!")
print("   Output: cf_scores.parquet, cf_predictions.parquet, cf_metrics.json, encoders.pkl")
