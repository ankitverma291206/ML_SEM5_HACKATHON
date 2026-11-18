"""Inspect PathSnap trip files and write a per-trip inventory.

Expected input directory:
    Student folder/
      train/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
      student_test/<trip_id>/...
Evaluator-testing data is deliberately not scanned.
"""
from pathlib import Path
import argparse
import pandas as pd

PARTITIONS = ("train", "student_test")
EXTENSIONS = (".track", ".nodes", ".arcs", ".route")

def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

def inspect_dataset(root: Path) -> pd.DataFrame:
    rows = []
    for partition in PARTITIONS:
        partition_dir = root / partition
        if not partition_dir.exists():
            print(f"WARNING: missing partition directory: {partition_dir}")
            continue
        for trip_dir in sorted(p for p in partition_dir.iterdir() if p.is_dir()):
            trip_id = trip_dir.name
            row = {"partition": partition, "trip_id": trip_id}
            for ext in EXTENSIONS:
                file_path = trip_dir / f"{trip_id}{ext}"
                row[f"{ext[1:]}_exists"] = file_path.is_file()
                row[f"{ext[1:]}_rows"] = (
                    count_nonempty_lines(file_path) if file_path.is_file() else None
                )
                row[f"{ext[1:]}_bytes"] = (
                    file_path.stat().st_size if file_path.is_file() else None
                )
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path,
                        help="Path to extracted 'Student folder' directory")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/dataset_inventory.csv"))
    args = parser.parse_args()

    inventory = inspect_dataset(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output, index=False)

    print("\nTrip counts:")
    print(inventory.groupby("partition")["trip_id"].nunique().to_string())
    print("\nMissing files by type:")
    for ext in EXTENSIONS:
        col = f"{ext[1:]}_exists"
        print(f"{ext}: {(~inventory[col]).sum()}")
    print(f"\nSaved inventory to: {args.output}")

if __name__ == "__main__":
    main()
