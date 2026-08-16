import json
import math
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path Configuration
# smart_factory_agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
SENSOR_DATA_PATH = DATA_DIR / "cleaned_sensor_data.csv"
OUTPUT_PATH = DATA_DIR / "output.json"

# Allow importing config.py from src/
sys.path.insert(0, str(SRC_DIR))
from config import SENSOR_CONFIG

# Ground Truth Score Calculation
def calculate_sensor_score(sensor, value):
    """
    Calculate the ground truth score for one sensor.

    Score definition:
        Normal   -> 0
        Warning  -> 0 ~ 1
        Abnormal -> 1
    """

    config = SENSOR_CONFIG[sensor]
    normal_min = config["normal_min"]
    normal_max = config["normal_max"]
    if sensor == "vibration":
        abnormal_low = config["generation_min"]
    else:
        abnormal_low = config["abnormal_low"]
    abnormal_high = config["abnormal_high"]

    # Normal
    if normal_min <= value <= normal_max:
        return 0.0

    # Warning: value is too high
    if normal_max < value < abnormal_high:
        score = (
            (value - normal_max)
            / (abnormal_high - normal_max)
        )
        return max(0.0, min(1.0, score))
    # Warning: value is too low
    if abnormal_low < value < normal_min:
        score = (
            (normal_min - value)
            / (normal_min - abnormal_low)
        )
        return max(0.0, min(1.0, score))

    # Abnormal
    return 1.0

def calculate_ground_truth_score(row):

    scores = []
    for sensor in SENSOR_CONFIG:
        value = float(row[sensor])
        sensor_score = calculate_sensor_score(sensor, value)
        scores.append(sensor_score)

    return max(scores)

def load_sensor_data():

    if not SENSOR_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find sensor data: {SENSOR_DATA_PATH}"
        )
    return pd.read_csv(SENSOR_DATA_PATH)

def load_output():

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find output file: {OUTPUT_PATH}"
        )
    with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "results" not in data:
        raise ValueError(
            "output.json does not contain 'results'."
        )
    return data["results"]

def build_evaluation_data(sensor_data, results):
    """
    Match CSV rows with Agent results using row number.

    Returns:
        ground_truth_labels
        predicted_labels
        ground_truth_scores
        predicted_scores
    """

    ground_truth_labels = []
    predicted_labels = []
    ground_truth_scores = []
    predicted_scores = []
    errors = []

    for result in results:

        row_number = result["row"]

        # CSV row index:
        # row 1 -> index 0
        csv_index = row_number - 1

        if csv_index < 0 or csv_index >= len(sensor_data):
            raise ValueError(
                f"Invalid row number in output.json: {row_number}"
            )

        sensor_row = sensor_data.iloc[csv_index]

        # Ground Truth Label
        ground_truth_label = str(
            sensor_row["label"]
        ).strip().lower()

        # Agent Label
        predicted_label = str(
            result["detect"]
        ).strip().lower()

        ground_truth_labels.append(
            ground_truth_label
        )
        predicted_labels.append(
            predicted_label
        )

        # Ground Truth Score
        ground_truth_score = calculate_ground_truth_score(
            sensor_row
        )

        # Agent Score
        predicted_score = float(
            result["score"]
        )

        ground_truth_scores.append(
            ground_truth_score
        )
        predicted_scores.append(
            predicted_score
        )

        # Error Detection
        label_error = (
            ground_truth_label != predicted_label
        )

        score_error = not math.isclose(
            ground_truth_score,
            predicted_score,
            abs_tol=0.01
        )

        if label_error or score_error:

            error_info = {
                "row": row_number,
                "timestamp": result.get("timestamp"),
            }
            if label_error:
                error_info["label"] = {
                    "ground_truth": ground_truth_label,
                    "agent": predicted_label
                }
            if score_error:
                error_info["score"] = {
                    "ground_truth": round(
                        ground_truth_score, 4
                    ),
                    "agent": round(
                        predicted_score, 4
                    ),
                    "error": round(
                        abs(
                            ground_truth_score
                            - predicted_score
                        ),
                        4
                    )
                }
            errors.append(error_info)

    return (
        ground_truth_labels,
        predicted_labels,
        ground_truth_scores,
        predicted_scores,
        errors
    )

def evaluate_labels(ground_truth, predicted):
    """
    Calculate label classification metrics.
    """

    total_samples = len(ground_truth)

    correct = sum(
        gt == pred
        for gt, pred in zip(ground_truth, predicted)
    )

    accuracy = accuracy_score(
        ground_truth,
        predicted
    )
    precision = precision_score(
        ground_truth,
        predicted,
        pos_label="abnormal",
        zero_division=0
    )
    recall = recall_score(
        ground_truth,
        predicted,
        pos_label="abnormal",
        zero_division=0
    )
    f1 = f1_score(
        ground_truth,
        predicted,
        pos_label="abnormal",
        zero_division=0
    )

    print("========== Label Evaluation ==========")
    print(f"Total samples : {total_samples}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy * 100:.2f}%")
    print(f"Precision     : {precision * 100:.2f}%")
    print(f"Recall        : {recall * 100:.2f}%")
    print(f"F1-score      : {f1 * 100:.2f}%")
    print()

def evaluate_scores(ground_truth, predicted):
    """
    Calculate score evaluation metrics.

    Metrics:
        MAE
        RMSE
        Within ±0.01
        Within ±0.05
    """

    if len(ground_truth) == 0:
        raise ValueError("No score data available.")

    errors = [
        abs(gt - pred)
        for gt, pred in zip(ground_truth, predicted)
    ]

    # MAE
    mae = sum(errors) / len(errors)
    # RMSE
    mse = sum(
        (gt - pred) ** 2
        for gt, pred in zip(ground_truth, predicted)
    ) / len(ground_truth)

    rmse = math.sqrt(mse)

    # Within tolerance
    within_001 = sum(
        error <= 0.01
        for error in errors
    ) / len(errors)
    within_005 = sum(
        error <= 0.05
        for error in errors
    ) / len(errors)

    print("========== Score Evaluation ==========")
    print(f"MAE           : {mae:.4f}")
    print(f"RMSE          : {rmse:.4f}")
    print(f"Within ±0.01  : {within_001 * 100:.2f}%")
    print(f"Within ±0.05  : {within_005 * 100:.2f}%")
    print()

def main():

    print("Loading sensor data...")
    sensor_data = load_sensor_data()
    print("Loading AI Agent output...")
    results = load_output()

    if len(sensor_data) != len(results):
        print(
            f"Warning: sensor data has {len(sensor_data)} rows, "
            f"but output.json has {len(results)} results."
        )

    (
        ground_truth_labels,
        predicted_labels,
        ground_truth_scores,
        predicted_scores,
        errors
    ) = build_evaluation_data(
        sensor_data,
        results
    )

    print()
    # Label evaluation
    evaluate_labels(
        ground_truth_labels,
        predicted_labels
    )
    # Score evaluation
    evaluate_scores(
        ground_truth_scores,
        predicted_scores
    )
    # error cases
    if errors:
        print("========== Error Cases ==========")
        for error in errors:
            print(f"Row       : {error['row']}")
            print(f"Timestamp : {error['timestamp']}")
            if "label" in error:
                print(
                    f"Label     : "
                    f"Ground Truth = {error['label']['ground_truth']}, "
                    f"Agent = {error['label']['agent']}"
                )
            if "score" in error:
                print(
                    f"Score     : "
                    f"Ground Truth = {error['score']['ground_truth']}, "
                    f"Agent = {error['score']['agent']}, "
                    f"Error = {error['score']['error']}"
                )
            print("--------------------------------")
    else:
        print("========== Error Cases ==========")
        print("No errors found.")

if __name__ == "__main__":
    main()