"""FastAPI application for the PathSnap project."""

from __future__ import annotations

import os
import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.data import configured_trips_dir, list_trip_ids, observations
from api.pipeline import PredictionPipelineUnavailable, is_ready, predict


class PredictionRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$", description="Eight-digit trip ID")


app = FastAPI(
    title="PathSnap API",
    description="Trip access and route-prediction API for the PathSnap project.",
    version="0.1.0",
)

origins = [origin.strip() for origin in os.getenv(
    "PATHSNAP_CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000",
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def evaluator_ui():
    """Serve the evaluator interface from the same host as the API."""
    return FileResponse(Path(__file__).resolve().parents[1] / "ui" / "index.html")


@app.get("/model-results", tags=["evaluation"])
def model_results() -> dict:
    """Return recorded model results without filling gaps with estimates."""
    path = Path(__file__).resolve().parents[1] / "experiments" / "atharv_student_test_results.csv"
    rows = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({
                    "model_name": row["model_name"], "partition": "Student test",
                    "test_accuracy": row.get("test_accuracy"),
                    "test_macro_f1": row.get("test_macro_f1"),
                    "test_top1_accuracy": row.get("test_top1_accuracy"),
                    "test_log_loss": row.get("test_log_loss"),
                    "note": "Recorded student-test metrics",
                })

    # These are group-CV results, not student-test measurements.
    cv_path = Path(__file__).resolve().parents[1] / "experiments" / "xgboost_group_cv_summary.csv"
    if cv_path.is_file():
        with cv_path.open("r", newline="", encoding="utf-8") as stream:
            cv = {r["metric"]: r["mean"] for r in csv.DictReader(stream)}
        rows.append({
            "model_name": "XGBoost", "partition": "Group CV",
            "test_accuracy": cv.get("candidate_accuracy"), "test_macro_f1": cv.get("candidate_macro_f1"),
            "test_top1_accuracy": cv.get("end_to_end_top1_accuracy"),
            "test_log_loss": cv.get("candidate_log_loss"),
            "note": "30-trip group CV; no student-test result recorded",
        })

    logistic_path = Path(__file__).resolve().parents[1] / "experiments" / "logistic_regression_metrics.csv"
    if logistic_path.is_file():
        with logistic_path.open("r", newline="", encoding="utf-8") as stream:
            logistic = next(csv.DictReader(stream), {})
        rows.append({
            "model_name": "Logistic Regression", "partition": "Group CV (labeled trips)",
            "test_accuracy": logistic.get("overall_accuracy"),
            "test_macro_f1": logistic.get("macro_f1_score"),
            "test_top1_accuracy": logistic.get("top_1_accuracy"),
            "test_log_loss": logistic.get("log_loss"),
            "note": "Training-script CV metrics; not student-test results",
        })
    else:
        rows.append({
            "model_name": "Logistic Regression", "partition": "Not evaluated",
            "test_accuracy": None, "test_macro_f1": None,
            "test_top1_accuracy": None, "test_log_loss": None,
            "note": "Run the Logistic Regression training command to generate CV metrics",
        })
    return {"models": rows}


@app.get("/hmm-status", tags=["evaluation"])
def hmm_status() -> dict:
    """Expose the fitted transition model summary to the evaluator UI."""
    path = Path(__file__).resolve().parents[1] / "artifacts" / "hmm_transition_model.json"
    if not path.is_file():
        return {"trained": False}
    import json
    model = json.loads(path.read_text(encoding="utf-8"))
    transition_probabilities = model.pop("transition_probabilities", {})
    summary_path = Path(__file__).resolve().parents[1] / "experiments" / "xgboost_group_cv_summary.csv"
    cv_metrics = {}
    if summary_path.is_file():
        with summary_path.open("r", newline="", encoding="utf-8") as stream:
            cv_metrics = {r["metric"]: float(r["mean"]) for r in csv.DictReader(stream)}
    return {"trained": True, **model, "transition_probabilities": transition_probabilities, "cv_metrics": cv_metrics}


@app.get("/health", tags=["system"])
def health() -> dict:
    trips_dir = configured_trips_dir()
    return {
        "status": "ok" if trips_dir.is_dir() else "data_unavailable",
        "partition": os.getenv("PATHSNAP_PARTITION", "student_test"),
        "trips_directory_configured": trips_dir.is_dir(),
        "prediction_ready": is_ready(),
    }


@app.get("/trips", tags=["trips"])
def get_trips() -> dict:
    """List IDs from the configured partition; no labels or routes are returned."""
    ids = list_trip_ids(str(configured_trips_dir()))
    return {
        "partition": os.getenv("PATHSNAP_PARTITION", "student_test"),
        "count": len(ids),
        "trips": [{"trip_id": trip_id} for trip_id in ids],
    }


@app.get("/trips/{trip_id}/observations", tags=["trips"])
def get_observations(
    trip_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict:
    return observations(trip_id, offset, limit)


@app.post("/predict", tags=["inference"])
def predict_trip(request: PredictionRequest) -> dict:
    """Run the real model pipeline once it has been integrated."""
    # Validate the trip ID against the configured partition before inference.
    from api.data import _trip_files

    _trip_files(request.trip_id)
    try:
        return predict(request.trip_id)
    except PredictionPipelineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
