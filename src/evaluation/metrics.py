
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from src.config import EVAL_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RecommenderEvaluator:

    def __init__(self, rating_threshold: float = None):
        self.rating_threshold = rating_threshold or EVAL_CONFIG["rating_threshold"]
        self.results = {}

    @staticmethod
    def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
        predictions = np.array(predictions, dtype=float)
        actuals = np.array(actuals, dtype=float)
        return float(np.sqrt(np.mean((predictions - actuals) ** 2)))

    @staticmethod
    def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
        predictions = np.array(predictions, dtype=float)
        actuals = np.array(actuals, dtype=float)
        return float(np.mean(np.abs(predictions - actuals)))

    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if k <= 0:
            return 0.0
        rec_at_k = recommended[:k]
        hits = len(set(rec_at_k) & relevant)
        return hits / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if not relevant:
            return 0.0
        rec_at_k = recommended[:k]
        hits = len(set(rec_at_k) & relevant)
        return hits / len(relevant)

    @staticmethod
    def f1_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        p = RecommenderEvaluator.precision_at_k(recommended, relevant, k)
        r = RecommenderEvaluator.recall_at_k(recommended, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if not relevant or k <= 0:
            return 0.0

        rec_at_k = recommended[:k]

        dcg = 0.0
        for i, item in enumerate(rec_at_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)

        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def average_precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if not relevant or k <= 0:
            return 0.0

        rec_at_k = recommended[:k]
        hits = 0
        sum_precisions = 0.0

        for i, item in enumerate(rec_at_k):
            if item in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)

        return sum_precisions / min(len(relevant), k)

    @staticmethod
    def hit_rate_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        rec_at_k = recommended[:k]
        return 1.0 if len(set(rec_at_k) & relevant) > 0 else 0.0

    def evaluate_predictions(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        if predictions_df.empty:
            return {"rmse": float("inf"), "mae": float("inf")}

        actuals = predictions_df["rating"].values
        preds = predictions_df["prediction"].values

        results = {
            "rmse": self.rmse(preds, actuals),
            "mae": self.mae(preds, actuals),
            "n_predictions": len(predictions_df),
        }

        logger.info(f"Prediction Metrics:")
        logger.info(f"   RMSE: {results['rmse']:.4f}")
        logger.info(f"   MAE:  {results['mae']:.4f}")
        logger.info(f"   N:    {results['n_predictions']}")

        return results

    def evaluate_top_k(self, recommendations: Dict[str, List[str]],
                       test_df: pd.DataFrame,
                       k_values: List[int] = None) -> pd.DataFrame:
        k_values = k_values or EVAL_CONFIG["k_values"]

        relevant_items = (
            test_df[test_df["rating"] >= self.rating_threshold]
            .groupby("user_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        results = []

        for k in k_values:
            precisions = []
            recalls = []
            f1s = []
            ndcgs = []
            aps = []
            hit_rates = []

            for user_id, relevant in relevant_items.items():
                if user_id not in recommendations:
                    continue

                rec_list = recommendations[user_id]

                precisions.append(self.precision_at_k(rec_list, relevant, k))
                recalls.append(self.recall_at_k(rec_list, relevant, k))
                f1s.append(self.f1_at_k(rec_list, relevant, k))
                ndcgs.append(self.ndcg_at_k(rec_list, relevant, k))
                aps.append(self.average_precision_at_k(rec_list, relevant, k))
                hit_rates.append(self.hit_rate_at_k(rec_list, relevant, k))

            n_users = len(precisions)
            results.append({
                "k": k,
                "precision@k": np.mean(precisions) if precisions else 0,
                "recall@k": np.mean(recalls) if recalls else 0,
                "f1@k": np.mean(f1s) if f1s else 0,
                "ndcg@k": np.mean(ndcgs) if ndcgs else 0,
                "map@k": np.mean(aps) if aps else 0,
                "hit_rate@k": np.mean(hit_rates) if hit_rates else 0,
                "n_users_evaluated": n_users,
            })

        results_df = pd.DataFrame(results)

        logger.info(f"\n Top-K Evaluation Results:")
        logger.info(f"\n{results_df.to_string(index=False)}")

        return results_df

    def compare_models(self, models_results: Dict[str, pd.DataFrame],
                       test_df: pd.DataFrame,
                       k_values: List[int] = None) -> pd.DataFrame:
        k_values = k_values or EVAL_CONFIG["k_values"]

        logger.info("=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)

        all_results = []

        for model_name, recommendations in models_results.items():
            logger.info(f"\n--- {model_name} ---")
            metrics = self.evaluate_top_k(recommendations, test_df, k_values)
            metrics["model"] = model_name
            all_results.append(metrics)

        comparison_df = pd.concat(all_results, ignore_index=True)

        comparison_pivot = comparison_df.pivot_table(
            index="model",
            columns="k",
            values=["precision@k", "recall@k", "ndcg@k", "hit_rate@k"],
        )

        logger.info(f"\nComparison Summary:")
        logger.info(f"\n{comparison_pivot.to_string()}")

        self.results["comparison"] = comparison_df
        return comparison_df

    def generate_report(self, comparison_df: pd.DataFrame = None,
                        prediction_results: Dict[str, Dict] = None,
                        output_path: str = None) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("EVALUATION REPORT — Hybrid Recommendation System")
        lines.append("=" * 70)

        if prediction_results:
            lines.append("\n## Rating Prediction Metrics")
            lines.append("-" * 40)
            lines.append(f"{'Model':<20} {'RMSE':<10} {'MAE':<10}")
            lines.append("-" * 40)
            for model_name, metrics in prediction_results.items():
                rmse = metrics.get("rmse", "N/A")
                mae = metrics.get("mae", "N/A")
                if isinstance(rmse, float):
                    lines.append(f"{model_name:<20} {rmse:<10.4f} {mae:<10.4f}")
                else:
                    lines.append(f"{model_name:<20} {rmse:<10} {mae:<10}")

        if comparison_df is not None and not comparison_df.empty:
            lines.append("\n## Top-K Recommendation Metrics")
            lines.append("-" * 70)
            lines.append(comparison_df.to_string(index=False))

        if comparison_df is not None and not comparison_df.empty:
            lines.append("\n## Best Model (by NDCG@10)")
            lines.append("-" * 40)
            k10 = comparison_df[comparison_df["k"] == 10] if 10 in comparison_df["k"].values else comparison_df
            if not k10.empty:
                best = k10.loc[k10["ndcg@k"].idxmax()]
                lines.append(f"Winner: {best['model']} with NDCG@{int(best['k'])} = {best['ndcg@k']:.4f}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report
