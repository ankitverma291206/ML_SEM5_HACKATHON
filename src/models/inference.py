"""Inference interface for PathSnap candidate scoring.

Provides a clean, standardized prediction interface for Arya (HMM/Viterbi)
and Ankit (FastAPI backend). Accepts candidate feature rows, preserves compound
identifiers (trip_id, observation_id, arc_id), and safely extracts
positive-class probabilities without assumptions about column indices.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.config import FeatureConfig
from src.models.evaluate import Timer


def get_positive_class_index(pipeline: Pipeline, pos_label: int = 1) -> int:
    """Identify the column index for the target positive class in predict_proba.

    Safely verifies against the estimator's `classes_` array rather than assuming
    index 1, preventing inverted probability scoring.
    """
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)
        if pos_label in classes:
            return classes.index(pos_label)
    return 1


def score_candidates(
    pipeline: Pipeline,
    df_candidates: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score candidate road segments and return matching probabilities.

    Contract for Arya (HMM decoder) and Ankit (FastAPI):
    - Input: DataFrame containing candidate rows with identifiers and features.
    - Output: DataFrame preserving [trip_id, observation_id, arc_id] plus:
        - `prob_matching`: Probability that this candidate matches the road segment.
        - `pred_matching`: Binary prediction based on specified threshold.
        - `inference_time_sec`: Time taken to run prediction.

    Args:
        pipeline: Fitted scikit-learn Pipeline (imputer + scaler + classifier).
        df_candidates: Candidate features DataFrame.
        config: Feature configuration (uses default if None).
        threshold: Classification probability threshold (default: 0.5).

    Returns:
        DataFrame with compound identifiers and positive-class scores.
    """
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df_candidates, require_label=False)

    X = df_candidates[config.feature_columns]

    with Timer() as t:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            probs = pipeline.predict_proba(X)
            pos_idx = get_positive_class_index(pipeline, pos_label=1)
            prob_matching = probs[:, pos_idx]
        elif hasattr(classifier, "decision_function"):
            raw_scores = pipeline.decision_function(X)
            # Sigmoid transformation for raw decision boundary margins
            prob_matching = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            prob_matching = pipeline.predict(X).astype(float)

        pred_matching = (prob_matching >= threshold).astype(int)

    result_df = pd.DataFrame(
        {
            "trip_id": df_candidates["trip_id"].values,
            "observation_id": df_candidates["observation_id"].values,
            "arc_id": df_candidates["arc_id"].values,
            "prob_matching": prob_matching,
            "pred_matching": pred_matching,
        }
    )

    # Attach any existing candidate distance if present for convenience
    if "distance_m" in df_candidates.columns:
        result_df["distance_m"] = df_candidates["distance_m"].values

    return result_df


def save_model_artifacts(
    pipeline: Pipeline,
    config: FeatureConfig,
    metadata: Dict[str, Any],
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Dict[str, Path]:
    """Save model pipeline and accompanying configuration artifacts for deployment.

    Saves:
        - `best_model.pkl`: Serialized scikit-learn Pipeline.
        - `feature_config.json`: Feature definitions and preprocessing rules.
        - `model_config.json`: Hyperparameters, training metadata, and metrics.

    Args:
        pipeline: Fitted Pipeline.
        config: FeatureConfig.
        metadata: Dictionary containing training parameters and metrics.
        artifacts_dir: Destination directory.

    Returns:
        Dictionary mapping artifact names to their saved paths.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "best_model.pkl"
    feature_config_path = out_dir / "feature_config.json"
    model_config_path = out_dir / "model_config.json"

    # Save serialized pipeline
    joblib.dump(pipeline, model_path)

    # Save feature configuration
    config.save_json(feature_config_path)

    # Save model config metadata
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "model_path": model_path,
        "feature_config_path": feature_config_path,
        "model_config_path": model_config_path,
    }


def load_model_artifacts(
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Tuple[Pipeline, FeatureConfig, Dict[str, Any]]:
    """Load the serialized model pipeline and configuration for inference.

    Args:
        artifacts_dir: Directory containing saved artifacts.

    Returns:
        Tuple of (Pipeline, FeatureConfig, model_config dict).
    """
    out_dir = Path(artifacts_dir)
    model_path = out_dir / "best_model.pkl"
    feature_config_path = out_dir / "feature_config.json"
    model_config_path = out_dir / "model_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    pipeline = joblib.load(model_path)
    config = FeatureConfig.from_json(feature_config_path)

    with open(model_config_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, config, metadata
