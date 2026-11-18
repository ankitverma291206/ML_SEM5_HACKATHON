"""Feature enrichment, label-reliability flags and candidate-miss reporting.

Pure pandas/numpy (no shapely/pyproj), so it can be unit-tested anywhere.

Column groups (import these in the model code so labels can never leak in):
    ID_COLUMNS       identifiers, never model inputs
    FEATURE_COLUMNS  model inputs
    LABEL_COLUMNS    answer / label-side columns, never model inputs
"""

import numpy as np
import pandas as pd

from src.graph.road_graph import haversine_array

HEADING_WINDOW = 5          # observations used for the smoothed heading
MIN_HEADING_DISP_M = 5.0    # below this net movement the heading is jitter -> NaN

ID_COLUMNS = ["partition", "trip_id", "observation_id", "arc_id"]

FEATURE_COLUMNS = [
    "distance_m",
    "projection_position",
    "heading_difference",        # 1-step GPS heading vs road bearing
    "heading_difference_5",      # 5-step smoothed GPS heading vs road bearing
    "gps_speed_mps",
    "time_delta_s",
    "large_or_invalid_time_gap",
    "segment_length_m",
    "n_candidates",              # candidates generated for this observation
    "distance_rank",             # 0 = nearest candidate of this observation
    "distance_minus_min_m",      # distance minus the nearest candidate's distance
]

# Kept in the table for inspection but deliberately NOT in FEATURE_COLUMNS
# (raw 0-360 angles are poor model inputs; heading_difference* carry the signal).
EXTRA_COLUMNS = ["road_bearing", "gps_heading", "gps_heading_5"]

LABEL_COLUMNS = [
    "aligned_arc_id",
    "alignment_distance_m",
    "is_matching_candidate",
    "obs_has_positive",
]


def _angle_diff(a, b):
    """Smallest absolute angular difference in degrees."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def smoothed_heading(track, window=HEADING_WINDOW, min_disp_m=MIN_HEADING_DISP_M):
    """Heading from observation i-window to i; NaN when there is no real movement."""
    lon = track["longitude"].to_numpy(float)
    lat = track["latitude"].to_numpy(float)
    n = len(lon)
    heading = np.full(n, np.nan)
    if n > window:
        dlat = lat[window:] - lat[:-window]
        mid = np.radians((lat[window:] + lat[:-window]) / 2.0)
        dlon = (lon[window:] - lon[:-window]) * np.cos(mid)
        disp_m = np.hypot(dlat, dlon) * 111_320.0
        bearing = (np.degrees(np.arctan2(dlon, dlat)) + 360.0) % 360.0
        bearing[disp_m < min_disp_m] = np.nan
        heading[window:] = bearing
    return heading


def add_features(labelled, alignment, data):
    """Add context features and label-reliability columns to one trip's table.

    labelled   output of label_candidates() (has aligned_arc_id, is_matching_candidate)
    alignment  output of align_track_to_route()
    data       output of load_trip()
    """
    df = labelled.copy()
    if df.empty:
        for col in FEATURE_COLUMNS + EXTRA_COLUMNS + LABEL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.Series(dtype="float64")
        return df

    # --- segment length (haversine between arc end nodes) --------------------
    nodes, arcs = data["nodes"], data["arcs"]
    lon = nodes["longitude"].to_numpy(float)
    lat = nodes["latitude"].to_numpy(float)
    src = arcs["from_node_id"].to_numpy(int)
    dst = arcs["to_node_id"].to_numpy(int)
    seg_len = haversine_array(lon[src], lat[src], lon[dst], lat[dst])
    df["segment_length_m"] = seg_len[df["arc_id"].to_numpy(int)]

    # --- per-observation context among its own candidates --------------------
    grp = df.groupby("observation_id")["distance_m"]
    df["n_candidates"] = grp.transform("size").astype("int16")
    df["distance_rank"] = (grp.rank(method="first") - 1).astype("int16")
    df["distance_minus_min_m"] = df["distance_m"] - grp.transform("min")

    # --- smoothed heading -----------------------------------------------------
    h5 = smoothed_heading(data["track"])
    df["gps_heading_5"] = h5[df["observation_id"].to_numpy(int)]
    df["heading_difference_5"] = _angle_diff(
        df["gps_heading_5"].to_numpy(float), df["road_bearing"].to_numpy(float)
    )

    # --- label-side reliability ----------------------------------------------
    df = df.merge(
        alignment[["observation_id", "alignment_distance_m"]],
        on="observation_id", how="left", validate="many_to_one",
    )
    df["obs_has_positive"] = (
        df.groupby("observation_id")["is_matching_candidate"].transform("max")
    ).astype("int8")

    df = df.sort_values(["observation_id", "distance_rank"]).reset_index(drop=True)
    return df


def candidate_report(df, alignment, n_obs, trip_id, partition):
    """Per-trip candidate-miss / label-reliability summary (computed BEFORE any filtering)."""
    cand_obs = int(df["observation_id"].nunique()) if len(df) else 0
    pos_obs = (
        int(df.loc[df["is_matching_candidate"] == 1, "observation_id"].nunique())
        if len(df) else 0
    )
    adist = alignment["alignment_distance_m"].to_numpy(float)
    return {
        "partition": partition,
        "trip_id": trip_id,
        "gps_observations": int(n_obs),
        "observations_with_candidates": cand_obs,
        "observations_with_positive": pos_obs,
        "candidate_miss_rate": 1.0 - pos_obs / n_obs,
        "candidate_rows": int(len(df)),
        "median_alignment_distance_m": float(np.median(adist)),
        "share_aligned_over_50m": float(np.mean(adist > 50.0)),
    }


def training_view(df):
    """Rows safe to train on: observations whose true arc is among the candidates.

    Keep the full table for evaluation / candidate-miss reporting; only filter
    when fitting. Fit-time filtering hides misses from classifier metrics, so
    always report candidate_miss_rate separately.
    """
    return df[df["obs_has_positive"] == 1]
