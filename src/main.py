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

def run_agent_in_batches(sensor_data):
    """
    Split sensor data into batches and send each batch
    to the AI Agent sequentially.
    """

    BATCH_SIZE = 100
    all_results = []
    total = len(sensor_data)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_data = sensor_data[start:end]
        print(
            f"\nProcessing rows "
            f"{start + 1} ~ {end} "
            f"({end}/{total})..."
        )
        batch_input = build_input_data(
            batch_data
        )
        batch_output = run_agent(
            batch_input
        )
        all_results.extend(
            batch_output["results"]
        )
        print(
            f"Batch completed: "
            f"{len(batch_output['results'])} results"
        )
    return {
        "results": all_results
    }

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

    # 1. Read sensor data
    print("\n[1/3] Reading sensor data...")
    sensor_data, ground_truth = read_sensor_data(
        SENSOR_DATA_PATH
    )
    print(f"Dataset : {SENSOR_DATA_PATH}")
    print(f"Loaded {len(sensor_data)} sensor records.")

    # 2. Run AI Agent in batches
    print("\n[2/3] Running AI Agent...")
    print("[✓] Detect anomalies")
    print("[✓] Calculate anomaly scores")
    print("[✓] Generate AI analysis")
    output_data = run_agent_in_batches(
        sensor_data
    )
    print("\nAll batches completed.")

    # 3. Save Output JSON
    print("\n[3/3] Saving Output JSON...")
    save_output(
        output_data,
        OUTPUT_PATH
    )
    
    print("\n========================================")
    print(" Smart Factory Sensor Analysis Agent")
    print("========================================")

    print("\nInput")
    print("-" * 50)
    print(f"Dataset : {SENSOR_DATA_PATH}")
    print(f"Records : {len(sensor_data)}")
    print("\nPipeline")
    print("-" * 50)
    print("[✓] Read sensor data")
    print("[✓] Detect anomalies")
    print("[✓] Calculate anomaly scores")
    print("[✓] Generate AI analysis")

    normal_count = sum(
        1
        for item in output_data["results"]
        if item["detect"] == "normal"
    )

    abnormal_count = sum(
        1
        for item in output_data["results"]
        if item["detect"] == "abnormal"
    )
    total = normal_count + abnormal_count
    if total > 0:
        anomaly_rate = abnormal_count / total * 100
    else:
        anomaly_rate = 0.0

    print("\nResult")
    print("-" * 50)
    print(f"Normal Records   : {normal_count}")
    print(f"Abnormal Records : {abnormal_count}")
    print(f"Anomaly Rate     : {anomaly_rate:.2f}%")
    print("\nOutput")
    print("-" * 50)
    print(f"Output saved to: {OUTPUT_PATH}")
    print("\n Analysis completed successfully.")
    print("========================================")

if __name__ == "__main__":
    main()