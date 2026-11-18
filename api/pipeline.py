"""Candidate scoring and Viterbi route reconstruction for the evaluator API."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

from xgboost import XGBClassifier

from api.data import _trip_files
from src.data.load_trip import load_trip
from src.features.candidates import generate_candidates
from src.models.train_xgboost import make_model_features
from src.models.viterbi import ViterbiRouteDecoder


class PredictionPipelineUnavailable(RuntimeError):
    pass


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "xgboost_candidate_model.json"
HMM_PATH = ROOT / "artifacts" / "hmm_transition_model.json"


@lru_cache(maxsize=1)
def _model() -> XGBClassifier:
    if not MODEL_PATH.is_file():
        raise PredictionPipelineUnavailable(f"Trained model not found: {MODEL_PATH.name}")
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def is_ready() -> bool:
    return MODEL_PATH.is_file() and HMM_PATH.is_file()


def predict(trip_id: str) -> dict:
    """Predict candidate scores, Viterbi decode, and return map-ready route points."""
    files = _trip_files(trip_id)
    trip = load_trip(files["track"].parent, trip_id)
    candidates = generate_candidates(trip, trip_id=trip_id)
    if candidates.empty:
        return {"trip_id": trip_id, "route": [], "predicted_arcs": [], "observations": len(trip["track"]), "message": "No road candidates were found."}

    candidates = candidates.copy()
    candidates["prob_matching"] = _model().predict_proba(make_model_features(candidates))[:, 1]
    hmm_parameters = json.loads(HMM_PATH.read_text(encoding="utf-8")) if HMM_PATH.is_file() else {}
    decoder = ViterbiRouteDecoder(learned_transition_probs=hmm_parameters.get("transition_probabilities", {}))
    decoded = decoder.decode_trip(
        candidates, len(trip["track"]), trip["arcs"], prob_col="prob_matching"
    )

    # Return the endpoint coordinates of each decoded arc for the frontend polyline.
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = []
    for arc_id in decoded:
        if arc_id < 0 or arc_id >= len(arcs):
            continue
        row = arcs.iloc[arc_id]
        for node_id in (int(row.from_node_id), int(row.to_node_id)):
            node = nodes.iloc[node_id]
            point = {"latitude": float(node["latitude"]), "longitude": float(node["longitude"])}
            if not route or route[-1] != point:
                route.append(point)
    return {
        "trip_id": trip_id,
        "route": route,
        "predicted_arcs": decoded,
        "observations": len(trip["track"]),
        "candidate_count": len(candidates),
        "decoder": "trained HMM + Viterbi road topology",
    }
