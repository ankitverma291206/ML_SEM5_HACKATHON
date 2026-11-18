"""Evaluation metrics for candidate-level and observation-level model assessment.

Provides both row-level classification metrics (Accuracy, Precision, Recall, Macro F1,
Confusion Matrix, Log Loss) and domain-specific observation-level metrics (Top-1 Accuracy,
Candidate Miss Rate), along with time-measurement utilities.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
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
    """Compute candidate-level binary classification metrics.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1).
        y_prob: Predicted probability for positive class (1).

    Returns:
        Dictionary of computed metrics including Macro F1 and confusion matrix.
    """
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
        # Clip probabilities to prevent log_loss numerical instability
        eps = 1e-15
        y_prob_arr = np.clip(y_prob_arr, eps, 1.0 - eps)
        try:
            # Need 2D probabilities for log_loss with classes [0, 1]
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
) -> float:
    """Compute Observation-level Top-1 Accuracy.

    For each GPS observation (grouped by trip_id and observation_id), select the
    candidate with the highest predicted positive-class probability. If that top
    candidate has label == 1, the observation is counted as correct.

    Args:
        df: DataFrame containing candidate rows with predictions and true labels.
        prob_col: Name of predicted probability column.
        label_col: Name of ground-truth candidate label column.
        trip_col: Trip identifier column.
        obs_col: Observation identifier column within the trip.

    Returns:
        Top-1 accuracy as a float in [0.0, 1.0].
    """
    if df.empty:
        return 0.0

    # Sort candidates by predicted probability descending within each observation
    sorted_df = df.sort_values(
        by=[trip_col, obs_col, prob_col], ascending=[True, True, False]
    )

    # Take the top candidate for each observation
    top_candidates = sorted_df.drop_duplicates(subset=[trip_col, obs_col], keep="first")

    # Correct if the top candidate has matching label == 1
    correct_count = int((top_candidates[label_col] == 1).sum())
    total_obs = int(len(top_candidates))

    return float(correct_count / total_obs) if total_obs > 0 else 0.0


def compute_candidate_miss_rate(
    df: pd.DataFrame,
    label_col: str = "is_matching_candidate",
    trip_col: str = "trip_id",
    obs_col: str = "observation_id",
) -> float:
    """Compute the Candidate Miss Rate from the candidate generation stage.

    Measures the fraction of observations where NONE of the proposed candidates
    matched the true reference road arc (i.e., sum of positive labels for that
    observation is 0). If a true segment was never retrieved as a candidate,
    no downstream classifier can predict it correctly.

    Args:
        df: DataFrame containing candidate rows with true labels.
        label_col: Ground truth label column.
        trip_col: Trip identifier column.
        obs_col: Observation identifier column.

    Returns:
        Fraction of observations missing their true road segment in the candidate set.
    """
    if df.empty:
        return 0.0

    obs_grouped = df.groupby([trip_col, obs_col])[label_col].max()
    total_observations = len(obs_grouped)
    missed_observations = int((obs_grouped == 0).sum())

    return float(missed_observations / total_observations) if total_observations > 0 else 0.0


def evaluate_model_predictions(
    df_eval: pd.DataFrame,
    pred_col: str = "pred_matching",
    prob_col: Optional[str] = "prob_matching",
    label_col: str = "is_matching_candidate",
    trip_col: str = "trip_id",
    obs_col: str = "observation_id",
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
        )
    else:
        results["top1_accuracy"] = None

    results["candidate_miss_rate"] = compute_candidate_miss_rate(
        df_eval,
        label_col=label_col,
        trip_col=trip_col,
        obs_col=obs_col,
    )

    return results
