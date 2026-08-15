import csv
import json
from pathlib import Path
from agent import run_agent

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SENSOR_DATA_PATH = DATA_DIR / "cleaned_sensor_data.csv"
OUTPUT_PATH = DATA_DIR / "output.json"

# Sensor Rules
SENSOR_RULES = {
    "temp": {
        "normal_min": 45.0,
        "normal_max": 50.0,
        "abnormal_low": 43.0,
        "abnormal_high": 52.0
    },
    "pressure": {
        "normal_min": 1.00,
        "normal_max": 1.05,
        "abnormal_low": 0.97,
        "abnormal_high": 1.08
    },
    "vibration": {
        "min_value": 0.00,
        "normal_min": 0.02,
        "normal_max": 0.04,
        "abnormal_high": 0.07
    }
}

# Read Sensor Data
def read_sensor_data(file_path):
    """
    Read sensor_data.csv.

    Returns:
        sensor_data:
            Data that will be sent to the AI Agent.
        ground_truth:
            Original labels kept for later benchmark/evaluation.
    """

    sensor_data = []
    ground_truth = []

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):

            sensor_data.append({
                "row": row_number,
                "timestamp": row["timestamp"],
                "temp": float(row["temp"]),
                "pressure": float(row["pressure"]),
                "vibration": float(row["vibration"])
            })

            ground_truth.append({
                "row": row_number,
                "label": row["label"]
            })

    return sensor_data, ground_truth

# Build the Input JSON that will be sent to the AI Agent.
def build_input_data(sensor_data):

    input_data = {
        "rules": SENSOR_RULES,
        "data": sensor_data
    }

    return input_data



# Save Output JSON
def save_output(output_data, file_path):

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2
        )

def main():

    print("========================================")
    print(" Smart Factory Sensor Analysis Agent")
    print("========================================")


    # 1. Read sensor data
    print("\n[1/4] Reading sensor data...")
    sensor_data, ground_truth = read_sensor_data(
        SENSOR_DATA_PATH
    )
    print(f"Loaded {len(sensor_data)} sensor records.")

    # 2. Build Input JSON
    print("\n[2/4] Building Input JSON...")
    input_data = build_input_data(
        sensor_data
    )
    print("Input JSON created.")

    # 3. Run AI Agent
    print("\n[3/4] Running AI Agent...")
    output_data = run_agent(
        input_data
    )
    print("AI Agent analysis completed.")

    # 4. Save Output JSON

    print("\n[4/4] Saving Output JSON...")
    save_output(
        output_data,
        OUTPUT_PATH
    )
    print(f"Output saved to: {OUTPUT_PATH}")
    print("\n========================================")
    print(" Analysis completed successfully.")
    print("========================================")

if __name__ == "__main__":
    main()