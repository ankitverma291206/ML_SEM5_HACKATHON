"""Read trip files without exposing reference routes to API clients."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException


REQUIRED_SUFFIXES = ("track", "nodes", "arcs", "route")


def configured_trips_dir() -> Path:
    """Resolve the trip directory from the environment or repository layout."""
    import os

    configured = os.getenv("PATHSNAP_TRIPS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[1]
    return repository_root / "Student folder" / "student_test"


def _trip_files(trip_id: str) -> dict[str, Path]:
    if len(trip_id) != 8 or not trip_id.isdecimal():
        raise HTTPException(status_code=400, detail="trip_id must be an 8-digit ID")

    trip_dir = configured_trips_dir() / trip_id
    files = {suffix: trip_dir / f"{trip_id}.{suffix}" for suffix in REQUIRED_SUFFIXES}
    if not trip_dir.is_dir() or not all(path.is_file() for path in files.values()):
        raise HTTPException(status_code=404, detail="Trip not found in the configured partition")
    return files


@lru_cache(maxsize=4)
def list_trip_ids(directory: str) -> tuple[str, ...]:
    root = Path(directory)
    if not root.is_dir():
        return ()
    ids = []
    for child in root.iterdir():
        if child.is_dir() and child.name.isdecimal() and len(child.name) == 8:
            if all((child / f"{child.name}.{suffix}").is_file() for suffix in REQUIRED_SUFFIXES):
                ids.append(child.name)
    return tuple(sorted(ids))


def observations(trip_id: str, offset: int, limit: int) -> dict:
    """Return a page of GPS observations. Reference labels are never returned."""
    track_path = _trip_files(trip_id)["track"]
    points = []
    with track_path.open("r", encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if row_index < offset:
                continue
            if len(points) >= limit:
                break
            columns = line.split()
            if len(columns) != 3:
                raise HTTPException(status_code=422, detail=f"Malformed track row {row_index}")
            try:
                longitude, latitude, timestamp = map(float, columns)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"Malformed track row {row_index}") from exc
            points.append({
                "observation_id": row_index,
                "longitude": longitude,
                "latitude": latitude,
                "timestamp": timestamp,
            })

    return {"trip_id": trip_id, "offset": offset, "limit": limit, "observations": points}
