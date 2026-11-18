"""Fit a first-order road-arc HMM transition table from labeled training trips.

Hidden states are directed road arcs. Emissions at inference come from the
candidate classifier; transitions are estimated from consecutive aligned arcs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET_DATA = ROOT / "data" / "processed" / "train_candidates.parquet"
CSV_DATA = ROOT / "data" / "processed" / "train_candidates.csv.gz"
OUT = ROOT / "artifacts" / "hmm_transition_model.json"


# Simple explanation: Run this model workflow and save its results.
def main() -> None:
    if not PARQUET_DATA.is_file() and not CSV_DATA.is_file():
        raise FileNotFoundError("Run python -m src.features.build_training_dataset first")
    df = pd.read_parquet(PARQUET_DATA) if PARQUET_DATA.is_file() else pd.read_csv(CSV_DATA, dtype={"trip_id": str})
    required = {"trip_id", "observation_id", "arc_id", "is_matching_candidate"}
    if not required.issubset(df.columns):
        raise ValueError(f"Training data missing columns: {sorted(required - set(df.columns))}")
    counts: Counter[str] = Counter()
    sequence_pairs = 0
    trips = []
    for trip_id, trip in df.groupby("trip_id", sort=True):
        trip_id = str(trip_id).zfill(8)
        trips.append(trip_id)
        positives = trip[trip.is_matching_candidate == 1].sort_values("observation_id")
        seq = positives.drop_duplicates("observation_id").arc_id.astype(int).tolist()
        arc_path = ROOT / "Student folder" / "train" / trip_id / f"{trip_id}.arcs"
        arcs = pd.read_csv(arc_path, sep=r"\s+", header=None, names=["from", "to"])
        for a, b in zip(seq, seq[1:]):
            if a == b:
                kind = "same_arc"
            elif int(arcs.iloc[a, 1]) == int(arcs.iloc[b, 0]):
                kind = "direct"
            elif int(arcs.iloc[a, 1]) == int(arcs.iloc[b, 1]) or int(arcs.iloc[a, 0]) == int(arcs.iloc[b, 0]):
                kind = "reverse_or_junction"
            else:
                kind = "disconnected"
            counts[kind] += 1
            sequence_pairs += 1
    # Smooth across the four topology transition classes.
    transitions = {kind: (counts[kind] + 1) / (sequence_pairs + 4) for kind in ("same_arc", "direct", "reverse_or_junction", "disconnected")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "model_type": "first_order_road_arc_hmm",
        "hidden_state": "directed road arc id",
        "emission_source": "candidate classifier probability",
        "transition_smoothing": "add-one",
        "training_trip_count": len(trips),
        "training_trip_ids": trips,
        "observed_transition_pairs": sequence_pairs,
        "transition_probabilities": transitions,
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT)}: {len(trips)} trips, {sequence_pairs} transitions")


if __name__ == "__main__":
    main()
