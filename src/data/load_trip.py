"""Load one PathSnap trip using the documented whitespace-separated formats."""
from pathlib import Path
import numpy as np
import pandas as pd

def load_trip(trip_dir: str | Path, trip_id: str | None = None) -> dict:
    trip_dir = Path(trip_dir)
    trip_id = trip_id or trip_dir.name

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track", sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp_s"]
    )
    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes", sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )
    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs", sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"]
    )
    route = pd.read_csv(
        trip_dir / f"{trip_id}.route", sep=r"\s+", header=None,
        names=["arc_id"]
    )

    # Row indices are IDs in this dataset, so preserve their zero-based meaning.
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    return {"track": track, "nodes": nodes, "arcs": arcs, "route": route}

def basic_validation(data: dict) -> dict:
    track, nodes, arcs, route = (
        data["track"], data["nodes"], data["arcs"], data["route"]
    )
    endpoint_ids = pd.concat(
        [arcs["from_node_id"], arcs["to_node_id"]], ignore_index=True
    )
    route_ids = route["arc_id"]

    return {
        "track_rows": len(track),
        "node_rows": len(nodes),
        "arc_rows": len(arcs),
        "route_rows": len(route),
        "track_missing_values": int(track.isna().sum().sum()),
        "node_missing_values": int(nodes.isna().sum().sum()),
        "arc_missing_values": int(arcs.isna().sum().sum()),
        "route_missing_values": int(route.isna().sum().sum()),
        "invalid_arc_endpoint_rows": int(
            ((arcs["from_node_id"] < 0) |
             (arcs["to_node_id"] < 0) |
             (arcs["from_node_id"] >= len(nodes)) |
             (arcs["to_node_id"] >= len(nodes))).sum()
        ),
        "invalid_route_arc_ids": int(
            ((route_ids < 0) | (route_ids >= len(arcs))).sum()
        ),
        "timestamps_non_decreasing": bool(
            track["timestamp_s"].is_monotonic_increasing
        ),
        "track_route_row_counts_equal": len(track) == len(route),
    }
