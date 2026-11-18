"""Run data-quality validation across all supplied trips."""

from pathlib import Path

import pandas as pd

from load_trip import load_trip, basic_validation


DATA_ROOT = Path("data/Student folder")
PARTITIONS = ("train", "student_test")


def validate_dataset():
    results = []

    for partition in PARTITIONS:
        partition_dir = DATA_ROOT / partition

        for trip_dir in sorted(partition_dir.iterdir()):
            if not trip_dir.is_dir():
                continue

            trip_id = trip_dir.name

            try:
                data = load_trip(trip_dir, trip_id)
                result = basic_validation(data)

                result["partition"] = partition
                result["trip_id"] = trip_id
                result["validation_error"] = ""

            except Exception as error:
                result = {
                    "partition": partition,
                    "trip_id": trip_id,
                    "validation_error": str(error),
                }

            results.append(result)

    report = pd.DataFrame(results)

    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data_quality_report.csv"
    report.to_csv(output_file, index=False)

    print("\nTrips checked:")
    print(report.groupby("partition")["trip_id"].count())

    print("\nValidation errors:")
    print(report["validation_error"].ne("").sum())

    print("\nSaved report to:", output_file)


if __name__ == "__main__":
    validate_dataset()
