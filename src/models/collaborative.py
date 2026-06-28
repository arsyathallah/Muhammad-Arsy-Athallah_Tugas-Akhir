import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    ALS_CONFIG, ALS_TUNING_GRID, MODELS_DIR,
    TRAIN_PARQUET, TEST_PARQUET, EVAL_CONFIG,
    SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY, SPARK_APP_NAME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CollaborativeFilteringModel:

    USER_COL = "user_index"
    ITEM_COL = "item_index"

    def __init__(self, spark=None):
        self.spark = spark
        self._owns_spark = spark is None

        self.model = None
        self.user_indexer = None
        self.item_indexer = None
        self.user_labels: List[str] = []
        self.item_labels: List[str] = []

        self.global_mean: float = 0.0
        self.max_rating = 5.0
        self.min_rating = 1.0
        self.implicit_prefs = bool(ALS_CONFIG.get("implicit_prefs", True))

        self._is_trained = False
        self._train_pairs = None

    def _get_spark(self):
        if self.spark is not None:
            return self.spark

        from pyspark.sql import SparkSession

        logger.info(f"   Starting SparkSession (master={SPARK_MASTER}, "
                    f"driver={SPARK_DRIVER_MEMORY}, executor={SPARK_EXECUTOR_MEMORY})")
        self.spark = (
            SparkSession.builder
            .appName(SPARK_APP_NAME)
            .master(SPARK_MASTER)
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")
        return self.spark

    def _to_spark(self, df: pd.DataFrame, columns: List[str]):
        spark = self._get_spark()
        cols = [c for c in columns if c in df.columns]
        return spark.createDataFrame(df[cols])

    def _fit_indexers(self, sdf):
        from pyspark.ml.feature import StringIndexer

        self.user_indexer = StringIndexer(
            inputCol="user_id", outputCol=self.USER_COL, handleInvalid="skip"
        ).fit(sdf)
        self.item_indexer = StringIndexer(
            inputCol="product_id", outputCol=self.ITEM_COL, handleInvalid="skip"
        ).fit(sdf)

        self.user_labels = list(self.user_indexer.labels)
        self.item_labels = list(self.item_indexer.labels)
        logger.info(f"   Indexers fitted: {len(self.user_labels)} users, "
                    f"{len(self.item_labels)} items")

    def _apply_indexers(self, sdf):
        from pyspark.sql import functions as F

        out = self.item_indexer.transform(self.user_indexer.transform(sdf))
        return (out
                .withColumn(self.USER_COL, F.col(self.USER_COL).cast("int"))
                .withColumn(self.ITEM_COL, F.col(self.ITEM_COL).cast("int")))

    def train(self, train_df: pd.DataFrame = None,
              factors: int = None, iterations: int = None,
              regularization: float = None) -> "CollaborativeFilteringModel":
        from pyspark.ml.recommendation import ALS
        from pyspark.sql import functions as F

        logger.info("=" * 60)
        logger.info(" TRAINING COLLABORATIVE FILTERING (Spark MLlib ALS)")
        logger.info("=" * 60)

        spark = self._get_spark()

        if train_df is None:
            sdf = spark.read.parquet(str(TRAIN_PARQUET))
        else:
            sdf = self._to_spark(train_df, ["user_id", "product_id", "rating"])
        sdf = sdf.select("user_id", "product_id", F.col("rating").cast("float")).cache()

        n_rows = sdf.count()
        logger.info(f"   Loaded training data: {n_rows} rows")

        rank = int(factors or ALS_CONFIG.get("rank", ALS_CONFIG.get("factors", 128)))
        max_iter = int(iterations or ALS_CONFIG.get("max_iter", ALS_CONFIG.get("iterations", 75)))
        reg_param = float(regularization if regularization is not None
                          else ALS_CONFIG.get("reg_param", ALS_CONFIG.get("regularization", 0.001)))
        self.implicit_prefs = bool(ALS_CONFIG.get("implicit_prefs", True))
        alpha = float(ALS_CONFIG.get("alpha", 40.0))

        logger.info(f"   ALS Config: rank={rank}, maxIter={max_iter}, regParam={reg_param}, "
                    f"implicitPrefs={self.implicit_prefs}, alpha={alpha}")

        self._fit_indexers(sdf)
        indexed = self._apply_indexers(sdf).select(self.USER_COL, self.ITEM_COL, "rating")

        if self.implicit_prefs:
            indexed = indexed.withColumn("pref", (F.col("rating") - F.lit(1.0)) / F.lit(4.0))
            rating_col = "pref"
        else:
            rating_col = "rating"
        indexed = indexed.cache()

        self.global_mean = float(indexed.agg(F.mean("rating")).first()[0])
        logger.info(f"   Global mean rating: {self.global_mean:.4f}")

        als = ALS(
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            userCol=self.USER_COL,
            itemCol=self.ITEM_COL,
            ratingCol=rating_col,
            implicitPrefs=self.implicit_prefs,
            alpha=alpha,
            nonnegative=bool(ALS_CONFIG.get("nonnegative", False)),
            coldStartStrategy=ALS_CONFIG.get("cold_start_strategy", "drop"),
            seed=int(EVAL_CONFIG.get("random_seed", 42)),
        )

        logger.info("Training ALS model on Spark...")
        self.model = als.fit(indexed)

        self._train_pairs = indexed.select(self.USER_COL, self.ITEM_COL).persist()
        self._is_trained = True
        sdf.unpersist()

        logger.info(" ALS model trained successfully!")
        self._log_training_rmse(indexed)
        return self

    def _log_training_rmse(self, indexed):
        try:
            from pyspark.ml.evaluation import RegressionEvaluator

            preds = self.model.transform(indexed).na.drop(subset=["prediction"])
            preds = self._scale_predictions(preds)
            rmse = RegressionEvaluator(
                metricName="rmse", labelCol="rating", predictionCol="prediction"
            ).evaluate(preds)
            logger.info(f"   Training RMSE: {rmse:.4f}")
        except Exception as e:
            logger.warning(f"   Could not compute training RMSE: {e}")

    def _scale_predictions(self, preds):
        from pyspark.sql import functions as F

        if self.implicit_prefs:
            stats = preds.agg(
                F.min("prediction").alias("lo"), F.max("prediction").alias("hi")
            ).first()
            lo, hi = float(stats["lo"]), float(stats["hi"])
            span = (hi - lo) or 1e-8
            return preds.withColumn(
                "prediction",
                F.lit(self.min_rating)
                + (F.lit(self.max_rating - self.min_rating))
                * (F.col("prediction") - F.lit(lo)) / F.lit(span),
            )
        return preds.withColumn(
            "prediction",
            F.greatest(F.lit(self.min_rating),
                       F.least(F.lit(self.max_rating), F.col("prediction"))),
        )

    def predict(self, test_df: pd.DataFrame = None) -> pd.DataFrame:
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        spark = self._get_spark()
        if test_df is None:
            tsdf = spark.read.parquet(str(TEST_PARQUET))
        else:
            tsdf = self._to_spark(test_df, ["user_id", "product_id", "rating"])

        n_test = tsdf.count()
        indexed = self._apply_indexers(tsdf)

        preds = self.model.transform(indexed).na.drop(subset=["prediction"])
        preds = self._scale_predictions(preds)

        result = (preds
                  .select("user_id", "product_id", "rating", "prediction")
                  .toPandas())
        skipped = n_test - len(result)
        logger.info(f"Generated {len(result)} predictions "
                    f"(skipped {skipped} cold-start pairs out of {n_test})")
        return result

    def recommend_for_all_users(self, n: int = 10,
                                exclude_rated: bool = True) -> pd.DataFrame:
        if not self._is_trained:
            raise ValueError("Model not trained.")

        from pyspark.ml.feature import IndexToString
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        logger.info(f"Generating top-{n} recommendations for all users "
                    f"(exclude_rated={exclude_rated})...")

        buffer = n + 50 if exclude_rated else n
        recs = self.model.recommendForAllUsers(buffer)

        exploded = (recs
                    .select(self.USER_COL, F.explode("recommendations").alias("rec"))
                    .select(self.USER_COL,
                            F.col(f"rec.{self.ITEM_COL}").alias(self.ITEM_COL),
                            F.col("rec.rating").alias("cf_score")))

        if exclude_rated:
            exploded = exploded.join(self._train_pairs,
                                     [self.USER_COL, self.ITEM_COL], "left_anti")

        rank_w = Window.partitionBy(self.USER_COL).orderBy(F.col("cf_score").desc())
        topn = (exploded
                .withColumn("_rk", F.row_number().over(rank_w))
                .filter(F.col("_rk") <= n)
                .drop("_rk"))

        u2s = IndexToString(inputCol=self.USER_COL, outputCol="user_id",
                            labels=self.user_labels)
        i2s = IndexToString(inputCol=self.ITEM_COL, outputCol="product_id",
                            labels=self.item_labels)
        topn = i2s.transform(u2s.transform(topn))

        stats = topn.agg(F.min("cf_score").alias("lo"),
                         F.max("cf_score").alias("hi")).first()
        lo, hi = float(stats["lo"]), float(stats["hi"])
        span = (hi - lo) or 1e-8
        topn = topn.withColumn(
            "predicted_rating",
            F.lit(self.min_rating)
            + F.lit(self.max_rating - self.min_rating)
            * (F.col("cf_score") - F.lit(lo)) / F.lit(span),
        )

        result = (topn
                  .select("user_id", "product_id", "cf_score", "predicted_rating")
                  .toPandas())
        logger.info(f" Generated recommendations for {result['user_id'].nunique()} users "
                    f"({len(result)} total entries)")
        return result

    def get_cf_scores(self, user_id: str,
                      candidate_products: List[str] = None) -> Dict[str, float]:
        if not self._is_trained:
            raise ValueError("Model not trained.")

        from pyspark.sql import functions as F

        if user_id not in self.user_labels:
            return {}
        u_idx = self.user_labels.index(user_id)

        if candidate_products:
            item_idx = [(pid, self.item_labels.index(pid))
                        for pid in candidate_products if pid in self.item_labels]
        else:
            item_idx = list(zip(self.item_labels, range(len(self.item_labels))))

        if not item_idx:
            return {}

        spark = self._get_spark()
        rows = [(u_idx, i_idx, pid) for pid, i_idx in item_idx]
        cand = spark.createDataFrame(rows, [self.USER_COL, self.ITEM_COL, "product_id"])

        preds = self.model.transform(cand).na.drop(subset=["prediction"])
        pdf = preds.select("product_id", "prediction").toPandas()
        scores = 1.0 / (1.0 + np.exp(-np.clip(pdf["prediction"].values, -10, 10)))
        return dict(zip(pdf["product_id"].values, scores.astype(float)))

    def tune_hyperparameters(self, train_df: pd.DataFrame = None,
                             test_df: pd.DataFrame = None) -> Dict:
        from pyspark.ml.evaluation import RegressionEvaluator

        logger.info(" Starting hyperparameter tuning (Spark ALS)...")

        spark = self._get_spark()
        train_sdf = (spark.read.parquet(str(TRAIN_PARQUET)) if train_df is None
                     else self._to_spark(train_df, ["user_id", "product_id", "rating"]))
        test_sdf = (spark.read.parquet(str(TEST_PARQUET)) if test_df is None
                    else self._to_spark(test_df, ["user_id", "product_id", "rating"]))
        train_pdf = train_sdf.select("user_id", "product_id", "rating").toPandas()
        test_pdf = test_sdf.select("user_id", "product_id", "rating").toPandas()

        ranks = ALS_TUNING_GRID.get("ranks", ALS_TUNING_GRID.get("factors", [64, 128]))
        regs = ALS_TUNING_GRID.get("reg_params", ALS_TUNING_GRID.get("regularizations", [0.001, 0.01]))
        iters = ALS_TUNING_GRID.get("max_iters", ALS_TUNING_GRID.get("iterations", [50, 75]))

        best_rmse = float("inf")
        best_params: Dict = {}
        all_results = []
        total = len(ranks) * len(regs) * len(iters)
        idx = 0

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                        predictionCol="prediction")

        for rank in ranks:
            for reg in regs:
                for it in iters:
                    idx += 1
                    logger.info(f"   [{idx}/{total}] rank={rank}, reg={reg}, maxIter={it}")

                    model = CollaborativeFilteringModel(spark=self.spark)
                    model.train(train_pdf, factors=rank, iterations=it, regularization=reg)
                    preds = model.predict(test_pdf)

                    if len(preds) > 0:
                        rmse = float(np.sqrt(np.mean(
                            (preds["rating"].values - preds["prediction"].values) ** 2)))
                    else:
                        rmse = float("inf")

                    all_results.append({"rank": rank, "reg_param": reg,
                                        "max_iter": it, "rmse": rmse})
                    logger.info(f"     RMSE = {rmse:.4f}")

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {"rank": rank, "reg_param": reg, "max_iter": it}

        logger.info(f"\n Best params: {best_params}, RMSE = {best_rmse:.4f}")
        self.train(train_pdf, factors=best_params["rank"],
                   iterations=best_params["max_iter"],
                   regularization=best_params["reg_param"])

        return {
            "best_factors": best_params["rank"],
            "best_regularization": best_params["reg_param"],
            "best_iterations": best_params["max_iter"],
            "best_rmse": best_rmse,
            "all_results": pd.DataFrame(all_results),
        }

    def save(self, path: Path = None):
        if not self._is_trained:
            raise ValueError("Model not trained.")

        path = Path(path or MODELS_DIR / "als")
        path.mkdir(parents=True, exist_ok=True)

        self.model.write().overwrite().save(str(path / "als_spark_model"))
        if self._train_pairs is not None:
            (self._train_pairs.write.mode("overwrite")
             .parquet(str(path / "train_pairs.parquet")))

        meta = {
            "user_labels": self.user_labels,
            "item_labels": self.item_labels,
            "global_mean": self.global_mean,
            "implicit_prefs": self.implicit_prefs,
            "rank": int(self.model.rank),
        }
        with open(path / "als_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info(f" Spark ALS model saved to {path}")

    def load(self, path: Path = None):
        from pyspark.ml.recommendation import ALSModel

        path = Path(path or MODELS_DIR / "als")
        meta_path = path / "als_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        spark = self._get_spark()
        self.model = ALSModel.load(str(path / "als_spark_model"))

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.user_labels = meta["user_labels"]
        self.item_labels = meta["item_labels"]
        self.global_mean = meta["global_mean"]
        self.implicit_prefs = meta.get("implicit_prefs", True)

        from pyspark.ml.feature import StringIndexerModel
        self.user_indexer = StringIndexerModel.from_labels(
            self.user_labels, inputCol="user_id", outputCol=self.USER_COL,
            handleInvalid="skip")
        self.item_indexer = StringIndexerModel.from_labels(
            self.item_labels, inputCol="product_id", outputCol=self.ITEM_COL,
            handleInvalid="skip")

        pairs_path = path / "train_pairs.parquet"
        if pairs_path.exists():
            self._train_pairs = spark.read.parquet(str(pairs_path)).persist()

        self._is_trained = True
        logger.info(f" Spark ALS model loaded from {path}")
        logger.info(f"   Users: {len(self.user_labels)}, Items: {len(self.item_labels)}")

    def stop_spark(self):
        if self.spark is not None and self._owns_spark:
            try:
                self.spark.stop()
                logger.info(" SparkSession stopped.")
            except Exception as e:
                logger.warning(f" Could not stop SparkSession: {e}")
            finally:
                self.spark = None
