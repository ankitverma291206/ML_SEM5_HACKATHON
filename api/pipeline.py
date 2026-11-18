"""Integration point for the trained model and route decoder."""

from __future__ import annotations


class PredictionPipelineUnavailable(RuntimeError):
    """Raised until the team supplies a real, loadable inference pipeline."""


def is_ready() -> bool:
    # Update this when the real preprocessing, model, and decoder are integrated.
    return False


def predict(trip_id: str) -> dict:
    """Run inference without loading or returning the reference route."""
    raise PredictionPipelineUnavailable(
        "Prediction pipeline is not configured yet. Integrate the trained model and decoder first."
    )
