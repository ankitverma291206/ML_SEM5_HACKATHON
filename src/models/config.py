"""Configuration and schema validation for ML candidate classification models.

This module provides flexible, configurable feature definitions so that
Atharv's ML experimentation pipeline can easily adapt to Aum's eventual
feature schema without requiring code refactoring.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
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


@dataclass
class FeatureConfig:
    """Manages feature definitions and dataset validation for candidate models."""

    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    identifier_columns: List[str] = field(default_factory=lambda: list(DEFAULT_IDENTIFIER_COLUMNS))
    label_column: str = DEFAULT_LABEL_COLUMN
    scale_features: bool = True
    impute_missing: bool = True
    random_seed: int = 42

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

    def save_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "FeatureConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def validate_dataframe(
        self, df: pd.DataFrame, require_label: bool = False
    ) -> None:
        """Validate that a DataFrame contains all required identifiers and features.

        Raises:
            ValueError: If any mandatory column is missing or types are malformed.
        """
        cols: Set[str] = set(df.columns)

        missing_ids = [col for col in self.identifier_columns if col not in cols]
        if missing_ids:
            raise ValueError(f"DataFrame is missing identifier columns: {missing_ids}")

        missing_features = [col for col in self.feature_columns if col not in cols]
        if missing_features:
            raise ValueError(
                f"DataFrame is missing configured feature columns: {missing_features}"
            )

        if require_label and self.label_column not in cols:
            raise ValueError(
                f"DataFrame is missing required target label column: '{self.label_column}'"
            )
