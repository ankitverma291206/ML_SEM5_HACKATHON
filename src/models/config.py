"""Configuration and schema validation for ML candidate classification models.

Provides flexible, configurable feature definitions, circular angle encodings,
and strict leakage-prevention validation so that Atharv's ML pipeline
reliably validates incoming datasets from Aum without data corruption.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


# Default baseline proposed feature columns (subject to update when Aum finalizes feature_schema.md)
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "distance_m",
    "projection_position",
    "road_bearing",
    "gps_heading",
    "heading_difference",
    "gps_speed",
    "time_delta",
    "segment_length",
]

DEFAULT_IDENTIFIER_COLUMNS: List[str] = [
    "trip_id",
    "observation_id",
    "arc_id",
]

DEFAULT_LABEL_COLUMN: str = "is_matching_candidate"


def encode_circular_angles(
    angles: Union[np.ndarray, pd.Series, List[float]],
    period: float = 360.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode circular quantities (headings/bearings) into sine and cosine components.

    Ensures that values near 0° and 360° have minimal Euclidean distance in feature space:
        sin_comp = sin(2 * pi * angle / period)
        cos_comp = cos(2 * pi * angle / period)

    Args:
        angles: Array or Series of angles in degrees.
        period: Full circle cycle (default: 360.0 degrees).

    Returns:
        Tuple of (sin_components, cos_components) as numpy ndarrays.
    """
    arr = np.asarray(angles, dtype=float)
    radians = 2.0 * np.pi * (arr % period) / period
    return np.sin(radians), np.cos(radians)


@dataclass
class FeatureConfig:
    """Manages feature definitions and dataset validation for candidate models.

    Strictly forbids identifier columns or the target label from appearing
    inside predictive feature columns to prevent data leakage.
    """

    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    identifier_columns: List[str] = field(default_factory=lambda: list(DEFAULT_IDENTIFIER_COLUMNS))
    label_column: str = DEFAULT_LABEL_COLUMN
    scale_features: bool = True
    impute_missing: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration integrity against accidental leakage hazards."""
        # 1. Reject identifiers inside predictive feature list
        overlap = set(self.feature_columns).intersection(set(self.identifier_columns))
        if overlap:
            raise ValueError(
                f"Identifier leakage hazard: Identifier column(s) {sorted(overlap)} "
                f"cannot be used as predictive features in 'feature_columns'."
            )

        # 2. Reject target label inside predictive feature list
        if self.label_column in self.feature_columns:
            raise ValueError(
                f"Target leakage hazard: Target label column '{self.label_column}' "
                f"cannot appear inside predictive 'feature_columns'."
            )

    def to_dict(self) -> Dict:
        """Serialize configuration to dictionary."""
        return {
            "feature_columns": self.feature_columns,
            "identifier_columns": self.identifier_columns,
            "label_column": self.label_column,
            "scale_features": self.scale_features,
            "impute_missing": self.impute_missing,
            "random_seed": self.random_seed,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FeatureConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def validate_dataframe(
        self, df: pd.DataFrame, require_label: bool = False
    ) -> None:
        """Validate that a DataFrame adheres strictly to required schemas and types.

        Raises:
            ValueError: If mandatory columns are missing, contain invalid values,
                        or target labels are non-binary.
            TypeError: If predictive feature columns have non-numeric dtypes.
        """
        cols: Set[str] = set(df.columns)

        # 1. Validate required identifier columns
        missing_ids = [col for col in self.identifier_columns if col not in cols]
        if missing_ids:
            raise ValueError(f"DataFrame is missing identifier columns: {missing_ids}")

        # Identifiers must not contain null/missing values
        for col in self.identifier_columns:
            if df[col].isna().any():
                raise ValueError(
                    f"Identifier column '{col}' contains {df[col].isna().sum()} null values."
                )

        # 2. Validate required feature columns
        missing_features = [col for col in self.feature_columns if col not in cols]
        if missing_features:
            raise ValueError(
                f"DataFrame is missing configured feature columns: {missing_features}"
            )

        # 3. Check numerical feature types
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(
                    f"Predictive feature column '{col}' has non-numeric dtype '{df[col].dtype}'. "
                    f"All predictive features must be numeric (int or float)."
                )

        # 4. Validate target label if required
        if require_label:
            if self.label_column not in cols:
                raise ValueError(
                    f"DataFrame is missing required target label column: '{self.label_column}'"
                )

            non_null_labels = df[self.label_column].dropna()
            unique_labels = set(non_null_labels.unique())
            # Accept valid binary representations: {0, 1}, {0}, {1}, {False, True}
            valid_binary = {0, 1, False, True}
            if not unique_labels.issubset(valid_binary):
                raise ValueError(
                    f"Target label column '{self.label_column}' contains non-binary values: "
                    f"{unique_labels}. Expected binary labels (0 or 1)."
                )
