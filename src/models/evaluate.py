"""Evaluation metrics for candidate-level and observation-level model assessment.

Provides row-level classification metrics (Accuracy, Precision, Recall, Macro F1,
Confusion Matrix, Log Loss) and observation-level metrics (Top-1 Accuracy,
Candidate Miss Rate), with support for raw GPS observation inventories and
zero-candidate observations.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


class Timer:
    """Accurate elapsed wall-clock timer for measuring training and inference times."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start_time: Optional[float] = None

    def start(self) -> "Timer":
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer was never started.")
        self.elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


def compute_candidate_metrics(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    y_prob: Optional[Union[np.ndarray, List[float], pd.Series]] = None,
) -> Dict[str, Any]:
    """Compute candidate-level binary classification metrics."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    binary_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
        "confusion_matrix": cm,
        "support_total": int(len(y_true_arr)),
        "support_positive": int(np.sum(y_true_arr == 1)),
        "support_negative": int(np.sum(y_true_arr == 0)),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        eps = 1e-15
        y_prob_arr = np.clip(y_prob_arr, eps, 1.0 - eps)
        try:
            prob_matrix = np.column_stack([1.0 - y_prob_arr, y_prob_arr])
            loss_val = float(log_loss(y_true_arr, prob_matrix, labels=[0, 1]))
            metrics["log_loss"] = loss_val
        except Exception:
            metrics["log_loss"] = None
    else:
        metrics["log_loss"] = None

    return metrics


def compute_observation_top1_accuracy(
    df: pd.DataFrame,
    prob_col: str = "prob_matching",
    label_col: str = "is_matching_candidate",
    trip_col: str = "trip_id",
    obs_col: str = "observation_id",
    all_observations: Optional[Union[int, pd.DataFrame]] = None,
) -> float:
    """Compute Observation-level Top-1 Accuracy.

    For each GPS observation, select the candidate with highest predicted probability.
    If that top candidate has label == 1, the observation is correctly matched.

    If all_observations is provided (as a raw observation DataFrame or total count),
    observations with zero candidates are treated as incorrect (not matched).
    """
    if df.empty:
        return 0.0

    sorted_df = df.sort_values(
        by=[trip_col, obs_col, prob_col], ascending=[True, True, False]
    )
    top_candidates = sorted_df.drop_duplicates(subset=[trip_col, obs_col], keep="first")
    correct_count = int((top_candidates[label_col] == 1).sum())

    if all_observations is not None:
        if isinstance(all_observations, pd.DataFrame):
            total_obs = len(all_observations.drop_duplicates(subset=[trip_col, obs_col]))
        else:
            total_obs = int(all_observations)
    else:
        total_obs = len(top_candidates)

    return float(correct_count / total_obs) if total_obs > 0 else 0.0


def compute_candidate_miss_rate(
    df: pd.DataFrame,
    label_col: str = "is_matching_candidate",
    trip_col: str = "trip_id",
    obs_col: str = "observation_id",
    all_observations: Optional[Union[int, pd.DataFrame]] = None,
) -> float:
    """Compute Candidate Miss Rate from the candidate generation stage.

    Measures the fraction of GPS observations where NONE of the proposed
    candidates matched the true reference road arc.

    Crucially, observations with ZERO generated candidates in `df` are included
    when `all_observations` (e.g. from the raw .track inventory) is provided,
    correcting the miss-rate denominator.

    Args:
        df: Candidate rows DataFrame.
        label_col: Binary match label column.
        trip_col: Trip identifier column.
        obs_col: Observation identifier column.
        all_observations: Total observation count or DataFrame of all raw observations.

    Returns:
        Miss rate in [0.0, 1.0].
    """
    if df.empty:
        if all_observations is not None:
            return 1.0
        return 0.0

    # Observations present in df with at least one matching candidate
    obs_grouped = df.groupby([trip_col, obs_col])[label_col].max()
    observations_with_match = set(obs_grouped[obs_grouped == 1].index)

    if all_observations is not None:
        if isinstance(all_observations, pd.DataFrame):
            raw_obs_set = set(
                zip(all_observations[trip_col].values, all_observations[obs_col].values)
            )
            total_observations = len(raw_obs_set)
            # Missed observations = total raw observations - raw observations that found a match
            missed_count = len(raw_obs_set - observations_with_match)
        else:
            total_observations = int(all_observations)
            found_count = len(observations_with_match)
            missed_count = max(0, total_observations - found_count)
    else:
        # Fallback to observations present in candidate table
        total_observations = len(obs_grouped)
        missed_count = int((obs_grouped == 0).sum())

    return float(missed_count / total_observations) if total_observations > 0 else 0.0


def evaluate_model_predictions(
    df_eval: pd.DataFrame,
    pred_col: str = "pred_matching",
    prob_col: Optional[str] = "prob_matching",
    label_col: str = "is_matching_candidate",
    trip_col: str = "trip_id",
    obs_col: str = "observation_id",
    all_observations: Optional[Union[int, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Comprehensive evaluation combining candidate-level and observation-level metrics."""
    y_true = df_eval[label_col].values
    y_pred = df_eval[pred_col].values
    y_prob = df_eval[prob_col].values if prob_col and prob_col in df_eval.columns else None

    results = compute_candidate_metrics(y_true, y_pred, y_prob)

    if prob_col and prob_col in df_eval.columns:
        results["top1_accuracy"] = compute_observation_top1_accuracy(
            df_eval,
            prob_col=prob_col,
            label_col=label_col,
            trip_col=trip_col,
            obs_col=obs_col,
            all_observations=all_observations,
        )
    else:
        results["top1_accuracy"] = None

    results["candidate_miss_rate"] = compute_candidate_miss_rate(
        df_eval,
        label_col=label_col,
        trip_col=trip_col,
        obs_col=obs_col,
        all_observations=all_observations,
    )

    return results
