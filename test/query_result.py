import json
import argparse
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "output.json"

def load_output(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def find_result_by_time(results, target_time):

    for item in results:
        if item["timestamp"] == target_time:
            return item
    return None

def print_result(result):
    # Print one sensor analysis result.

    print("=" * 60)
    print("          Sensor Analysis Result")
    print("=" * 60)
    print(f"Timestamp        : {result['timestamp']}")
    print(f"Detection result : {result['detect']}")

    print("-" * 60)
    print(
        f"Anomaly Sensors  : "
        f"{', '.join(result['anomaly_sensors']) if result['anomaly_sensors'] else 'None'}"
    )
    print(
        f"Warning Sensors  : "
        f"{', '.join(result['warning_sensors']) if result['warning_sensors'] else 'None'}"
    )

    print("\nSensor Scores")
    print("-" * 60)
    for sensor, score in result["sensor_scores"].items():
        print(f"{sensor:<15}: {score}")
    print("-" * 60)
    print(f"Overall Score    : {result['score']}")

    print("\nReason")
    print("-" * 60)
    print(result["reason"])

    print("\nSuggestion")
    print("-" * 60)
    print(result["suggestion"])

    print("=" * 60)

def main():

    parser = argparse.ArgumentParser(
        description="Query sensor analysis result by timestamp."
    )

    parser.add_argument(
        "--time",
        required=True,
        help='Timestamp to query, e.g. "2026-08-16 18:31:00"'
    )

    args = parser.parse_args()

    # Load output.json
    try:
        output_data = load_output(OUTPUT_PATH)

    except FileNotFoundError:
        print(f"Error: Output file not found: {OUTPUT_PATH}")
        return

    # Find result
    result = find_result_by_time(
        output_data["results"],
        args.time
    )

    # Print result
    if result is None:
        print("=" * 60)
        print("Result Not Found")
        print("=" * 60)
        print(f"Timestamp : {args.time}")
        print("\nNo analysis result was found for this timestamp.")
        return

    print_result(result)

if __name__ == "__main__":
    main()