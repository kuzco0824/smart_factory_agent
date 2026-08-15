import pandas as pd
import json
import math
from datetime import datetime

# Configuration
from config import SENSOR_CONFIG

MIN_ROWS = 100
MAX_ROWS = 500

EXPECTED_COLUMNS = [
    "timestamp",
    "temp",
    "pressure",
    "vibration",
    "label"
]
SENSOR_COLUMNS = [
    "temp",
    "pressure",
    "vibration"
]
VALID_LABELS = {
    "normal",
    "abnormal"
}

# Error Helper
def create_error(row, column, error, solution, solve):

    # Create a standardized validation error.
    return {
        "row": row,
        "column": column,
        "error": error,
        "solution": solution,
        "solve_type": solve,
    }

def print_error(error):

    print(
        f"[ERROR] Row {error['row']}, "
        f"Column '{error['column']}':"
    )
    print(f"  Error: {error['error']}")
    print(f"  Solution: {error['solution']}")
    print()

def validate_columns(df):
    # Check whether required columns exist.

    errors = []
    actual_columns = list(df.columns)

    # Missing columns
    for column in EXPECTED_COLUMNS:

        if column not in actual_columns:

            errors.append(
                create_error(
                    "-",
                    column,
                    f"Missing required column '{column}'.",
                    f"Add the required column '{column}' to the CSV file.",
                    0
                )
            )

    # Unexpected columns
    for column in actual_columns:

        if column not in EXPECTED_COLUMNS:

            errors.append(
                create_error(
                    "-",
                    column,
                    f"Unexpected column '{column}'.",
                    "Remove the unnecessary column from the CSV file."
                )
            )

    return errors

def validate_row_count(df):
    # Check whether dataset size is within the required range.

    errors = []

    row_count = len(df)

    if row_count < MIN_ROWS:

        errors.append(
            create_error(
                "-",
                "dataset",
                f"Dataset contains only {row_count} rows.",
                f"Generate at least {MIN_ROWS} rows.",
                0
            )
        )

    elif row_count > MAX_ROWS:

        errors.append(
            create_error(
                "-",
                "dataset",
                f"Dataset contains {row_count} rows.",
                f"Reduce the dataset to at most {MAX_ROWS} rows.",
                0
            )
        )

    return errors

def validate_data_types(row, row_number):
    errors = []

    # sensors
    for sensor in SENSOR_COLUMNS:

        value = row[sensor]

        try:
            float(value)
        except (ValueError, TypeError):
            errors.append(
                create_error(
                    row_number,
                    sensor,
                    f"Invalid float value: {value}.",
                    f"{sensor} must be a float.",
                    2
                )
            )
            continue

    # label
    if not isinstance(row["label"], str):
        errors.append(
            create_error(
                row_number,
                "label",
                "Invalid data type.",
                "Label must be a string.",
                4
            )
        )

    return errors

def validate_missing_values(row, row_number):
    # Check whether the row contains missing values.

    errors = []

    for column in SENSOR_COLUMNS:

        value = row[column]

        if pd.isna(value) or value == "":

            errors.append(
                create_error(
                    row_number,
                    column,
                    "Missing value.",
                    f"Provide a valid value for '{column}'.",
                    1
                )
            )
            continue

        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                errors.append(
                    create_error(
                        row_number,
                        column,
                        f"Invalid numeric value: {value}.",
                        "Provide a valid finite numeric value.",
                        2
                    )
                )
        except (ValueError, TypeError):
            # Invalid type is handled by validate_data_types()
            pass

    return errors

def validate_timestamps(row, row_number, previous_timestamp):
    # Check timestamp format, ordering, and duplicates.

    errors = []
    timestamp = row["timestamp"]
    # Check format
    try:
        current_timestamp = datetime.strptime(
            str(timestamp),
            "%Y-%m-%d %H:%M:%S"
        )

    except (ValueError, TypeError):

        errors.append(
            create_error(
                row_number,
                "timestamp",
                f"Invalid timestamp format: {timestamp}",
                "Use the format YYYY-MM-DD HH:MM:SS",
                3
            )
        )
        return errors, None

    # Check time interval
    if previous_timestamp is not None:

        time_difference = (
            current_timestamp - previous_timestamp
        ).total_seconds() / 60

        if time_difference not in (1, 5):
            errors.append(
                create_error(
                    row_number,
                    "timestamp",
                    f"Invalid time interval: {time_difference} minutes.",
                    "Timestamp must be exactly 1 or 5 minutes after the previous row.",
                    3
                )
            )
            return errors, None

    return errors, current_timestamp

def validate_labels(row, row_number):
    """
    Check whether label is valid and consistent with sensor values.
    """

    errors = []

    label = row["label"]
    # Check valid label
    if label not in VALID_LABELS:

        errors.append(
            create_error(
                row_number,
                "label",
                f"Invalid label: {label}",
                "Label must be either 'normal' or 'abnormal'.",
                4
            )
        )

        return errors

    # Convert sensor values
    try:
        temp = float(row["temp"])
        pressure = float(row["pressure"])
        vibration = float(row["vibration"])

    except (ValueError, TypeError):
        # Type validation will report this error.
        return errors

    # Check abnormal ranges
    temp_abnormal = (
        temp < SENSOR_CONFIG["temp"]["abnormal_low"]
        or temp > SENSOR_CONFIG["temp"]["abnormal_high"]
    )
    pressure_abnormal = (
        pressure < SENSOR_CONFIG["pressure"]["abnormal_low"]
        or pressure > SENSOR_CONFIG["pressure"]["abnormal_high"]
    )
    vibration_abnormal = (
        vibration > SENSOR_CONFIG["vibration"]["abnormal_high"]
    )

    # At least one sensor is abnormal
    expected_label = (
        "abnormal"
        if temp_abnormal
        or pressure_abnormal
        or vibration_abnormal
        else "normal"
    )
    # Compare label
    if label != expected_label:
        errors.append(
            create_error(
                row_number,
                "label",
                (
                    f"Label is '{label}', "
                    f"but sensor values indicate '{expected_label}'."
                ),
                (
                    f"Change label to '{expected_label}' "
                    "or correct the sensor value."
                ),
                4
            )
        )

    return errors

# =========================
# Main Validation
# =========================

def validate_data(df):
    # Run all validation checks.

    errors = []
    # ---------------------------------
    # Dataset-level validation
    # ---------------------------------

    errors.extend(
        validate_columns(df)
    )
    errors.extend(
        validate_row_count(df)
    )

    # If required columns are missing,
    # row-level validation cannot continue safely.
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print("========== Validation Result ==========")

        for error in errors:
            print_error(error)

        print("========================================")
        print("Validation FAILED")
        print(f"Errors: {len(errors)}")
        print("========================================")

        return {
            "valid": False,
            "errors": errors
        }

    # ---------------------------------
    # Row-level validation
    # ---------------------------------
    previous_timestamp = None

    for index, row in df.iterrows():

        # +1 because DataFrame index starts from 0
        row_number = index + 2

        # 1. Data Type
        errors.extend(
            validate_data_types(
                row,
                row_number
            )
        )
        # 2. Missing Value
        errors.extend(
            validate_missing_values(
                row,
                row_number
            )
        )
        #3. Timestamp
        timestamp_errors, previous_timestamp = validate_timestamps(
            row,
            row_number,
            previous_timestamp
        )
        errors.extend(timestamp_errors)
        # 4. Label
        errors.extend(
            validate_labels(
                row,
                row_number
            )
        )

    return errors

def save_error_report(errors, file_path):
    errors.sort(key=lambda error: error["solve_type"] == 3)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            errors,
            file,
            ensure_ascii=False,
            indent=4
        )
# Main
# =========================

def main():

    csv_path = "./data/sensor_data.csv"
    error_path = "./data/validation_errors.json"

    # Keep sensor values as strings
    # so decimal places can be checked.
    df = pd.read_csv(
        csv_path,
        dtype={
            "temp": str,
            "pressure": str,
            "vibration": str
        }
    )

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

    errors = validate_data(df)
    save_error_report(errors, error_path)

    print("========== Validation Result ==========")

    if errors:

        for error in errors:
            print_error(error)
        print("========================================")
        print("Validation FAILED")
        print(f"Errors: {len(errors)}")
        print("========================================")
        return False
    
    print("Validation PASSED")
    print("========================================")

    return True


if __name__ == "__main__":
    main()