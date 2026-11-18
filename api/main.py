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
    """Return the checked-in student-test comparison, if it is available."""
    path = Path(__file__).resolve().parents[1] / "experiments" / "atharv_student_test_results.csv"
    if not path.is_file():
        return {"models": []}
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {"models": rows, "source": "experiments/atharv_student_test_results.csv"}


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
