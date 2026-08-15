import json
import math
import random
import pandas as pd

# Configuration
from config import SENSOR_CONFIG

def is_sensor_normal(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def clean_missing_value(df, row, column):
    sensor = column
    config = SENSOR_CONFIG[sensor]

    # First row
    if row == 0:
        next_value = float(df.loc[row + 1, column])

        if is_sensor_normal(next_value):
            value = next_value
        else:
            value = random.uniform(
                config["normal_min"],
                config["normal_max"]
            )
    # Last row
    elif row == len(df) - 1:
        previous_value = float(df.loc[row - 1, column])

        if is_sensor_normal(previous_value):
            value = previous_value
        else:
            value = random.uniform(
                config["normal_min"],
                config["normal_max"]
            )
    # Middle rows
    else:
        previous_value = float(df.loc[row - 1, column])
        next_value = float(df.loc[row + 1, column])
        previous_normal = is_sensor_normal(previous_value)
        next_normal = is_sensor_normal(next_value)

        if previous_normal and next_normal:
            value = (previous_value + next_value) / 2
        else:
            value = random.uniform(
                config["normal_min"],
                config["normal_max"]
            )

    decimals = SENSOR_CONFIG[sensor]["precision"]
    df[column] = df[column].astype(object)
    df.loc[row, column] = round(value, decimals)

def clean_data_type(df, row, column):
    value = df.loc[row, column]

    try:
        value = float(value)
    except (ValueError, TypeError):
        clean_missing_value(df, row, column)
        return False

    decimals = SENSOR_CONFIG[column]["precision"]
    df[column] = df[column].astype(object)
    df.loc[row, column] = round(value, decimals)

def clean_timestamp(df, row):
    df.drop(index=row, inplace=True)

def clean_label(df, row):
    sensors = ["temp", "pressure", "vibration"]
    values = {}

    # check values are legal
    try:
        for sensor in sensors:
            val = float(df.loc[row, sensor])
            if math.isnan(val):
                return
            values[sensor] = val
    except (ValueError, TypeError):
        return

    for sensor, value in values.items():
        config = SENSOR_CONFIG[sensor]
        if sensor == "vibration":
            abnormal = value > config["abnormal_high"]
        else:
            abnormal = (
                value < config["abnormal_low"]
                or value > config["abnormal_high"]
            )
        if abnormal:
            df.loc[row, "label"] = "abnormal"
            return
    df.loc[row, "label"] = "normal"


def process_error(df, error):

    row = error["row"] - 2
    column = error["column"]
    solve_type = error["solve_type"]

    if solve_type == 1:
        clean_missing_value(df, row, column)

    elif solve_type == 2:
        clean_data_type(df, row, column)

    elif solve_type == 3:
        clean_timestamp(df, row)

    elif solve_type == 4:
        clean_label(df, row)

    elif solve_type == 0:
        print(
            "This issue cannot be automatically resolved. "
            "Please confirm with the vendor."
        )

def main():

    csv_path = "./data/sensor_data.csv"
    error_path = "./data/validation_errors.json"
    output_path = "./data/cleaned_sensor_data.csv"

    df = pd.read_csv(
        csv_path,
        dtype={
            "temp": str,
            "pressure": str,
            "vibration": str
        }
    )

    with open(error_path, "r", encoding="utf-8") as file:
        errors = json.load(file)

    for error in errors:
        process_error(df, error)

    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    main()