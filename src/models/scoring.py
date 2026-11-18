"""Inference and scoring interface for Arya's HMM and Ankit's FastAPI.

Accepts candidate road segments, strictly preserves compound identifiers
(trip_id, observation_id, arc_id), and safely extracts positive-class
matching probabilities without making assumptions about column ordering.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.config import FeatureConfig
from src.models.evaluate import Timer
from src.models.validation import get_positive_class_index


def score_candidates(
    pipeline: Pipeline,
    df_candidates: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score candidate road segments and return positive-class probabilities.

    Contract for Arya (HMM decoder) and Ankit (FastAPI):
    - Preserves compound identifiers: [trip_id, observation_id, arc_id].
    - Outputs:
        - `prob_matching`: Probability that candidate matches true road arc.
        - `pred_matching`: Binary prediction based on threshold.
    """
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df_candidates, require_label=False)

    X = df_candidates[config.feature_columns]

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probs = pipeline.predict_proba(X)
        pos_idx = get_positive_class_index(pipeline, pos_label=1)
        prob_matching = probs[:, pos_idx]
    elif hasattr(classifier, "decision_function"):
        raw_scores = pipeline.decision_function(X)
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

    if "distance_m" in df_candidates.columns:
        result_df["distance_m"] = df_candidates["distance_m"].values

    return result_df


def save_model_artifacts(
    pipeline: Pipeline,
    config: FeatureConfig,
    metadata: Dict[str, Any],
    artifacts_dir: Union[str, Path] = "artifacts",
    model_filename: str = "best_model.pkl",
) -> Dict[str, Path]:
    """Save model pipeline and configuration for deployment."""
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / model_filename
    feature_config_path = out_dir / "feature_config.json"
    model_config_path = out_dir / "model_config.json"

    joblib.dump(pipeline, model_path)
    config.save_json(feature_config_path)

    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "model_path": model_path,
        "feature_config_path": feature_config_path,
        "model_config_path": model_config_path,
    }


def load_model_artifacts(
    artifacts_dir: Union[str, Path] = "artifacts",
    model_filename: str = "best_model.pkl",
) -> Tuple[Pipeline, FeatureConfig, Dict[str, Any]]:
    """Load serialized model and configuration for inference."""
    out_dir = Path(artifacts_dir)
    model_path = out_dir / model_filename
    feature_config_path = out_dir / "feature_config.json"
    model_config_path = out_dir / "model_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    pipeline = joblib.load(model_path)
    config = FeatureConfig.from_json(feature_config_path)

    with open(model_config_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, config, metadata
