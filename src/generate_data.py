import argparse
import csv
import random
from pathlib import Path
from datetime import datetime, timedelta


# Configuration

ANOMALY_PROBABILITY = 0.06
RECOVERY_PROBABILITY = 0.50
from config import SENSOR_CONFIG


# Initialize sensor values
def initialize_values():
    """
    Generate the initial sensor values.
    All sensors start inside their normal ranges.
    """

    values = {}

    for sensor, config in SENSOR_CONFIG.items():
        values[sensor] = round(
            random.uniform(
                config["normal_min"],
                config["normal_max"]
            ),
            config["precision"]
        )

    return values

def generate_normal_value(sensor, current_value):
    """
    Generate the next value during normal operation.
    The next value is based on the previous value and
    a random change within the sensor's step range.
    """

    config = SENSOR_CONFIG[sensor]
    change = random.uniform(
        -config["step"],
        config["step"]
    )
    new_value = current_value + change

    if new_value < config["normal_min"] - config["step"]:
        new_value = config["normal_min"]

    elif new_value > config["normal_max"] + config["step"]:
        new_value = config["normal_max"]


    return round(new_value, config["precision"])

def generate_abnormal_value(sensor, direction):
    """
    Generate a value directly inside the abnormal range.
    The abnormal direction is fixed by the anomaly event.
    """

    config = SENSOR_CONFIG[sensor]

    if direction == "high":

        # Generate a value above the abnormal threshold.
        if sensor == "temp":
            value = random.uniform(
                config["abnormal_high"] + 0.1,
                config["generation_max"]
            )
        else :
            value = random.uniform(
                config["abnormal_high"] + 0.01,
                config["generation_max"]
            )
    else:
        # Low abnormality.
        if sensor == "temp":
            value = random.uniform(
                config["generation_min"],
                config["abnormal_low"] - 0.1
            )
        else :
            value = random.uniform(
                config["generation_min"],
                config["abnormal_low"] - 0.01
            )

    return round(value, config["precision"])

def create_anomaly_event():
    """
    Randomly select 1~3 sensors and assign a fixed
    abnormal direction to each sensor.

    Example:
        {
            "temp": "high",
            "pressure": "low"
        }
    """

    sensors = list(SENSOR_CONFIG.keys())
    sensor_count = random.randint(1, len(sensors))
    selected_sensors = random.sample(
        sensors,
        sensor_count
    )

    anomaly = {}
    for sensor in selected_sensors:

        if sensor == "vibration":
            # Vibration only defines high abnormality.
            direction = "high"

        else:
            direction = random.choice([
                "low",
                "high"
            ])
        anomaly[sensor] = direction

    return anomaly

def continue_abnormal_value(sensor, current_value, direction):
    """
    Generate the next value while an anomaly continues.
    The value changes randomly but must not in the normal section.
    """

    config = SENSOR_CONFIG[sensor]

    change = random.uniform(
        -config["step"],
        config["step"]
    )
    new_value = current_value + change

    if direction == "high":

        if new_value <= config["normal_max"]:
            new_value = (
                config["normal_max"]
                + random.uniform(0, config["step"])
            )
        elif new_value > config["generation_max"]:
            new_value = config["generation_max"]

    else:

        if new_value >= config["normal_min"]:
            new_value = (
                config["normal_min"]
                - random.uniform(0, config["step"])
            )
        elif new_value < config["generation_min"]:
            new_value = config["generation_min"]

    return round(new_value, config["precision"])

def update_values(values, anomaly):
    """
    Generate the next values for all sensors.
    If no anomaly exists:
        all sensors use normal changes.
    If an anomaly exists:
        abnormal sensors remain in their fixed abnormal
        direction, while other sensors continue normally.
    """

    new_values = {}

    for sensor in SENSOR_CONFIG:

        current_value = values[sensor]

        if anomaly is not None and sensor in anomaly:

                # Continue existing anomaly
                new_values[sensor] = continue_abnormal_value(
                    sensor,
                    current_value,
                    anomaly[sensor]
                )
        else:
                # Normal operation
                new_values[sensor] = generate_normal_value(
                    sensor,
                    current_value
                )

    return new_values

# Check whether a sensor is normal
def is_sensor_abnormal(sensor, value):

    config = SENSOR_CONFIG[sensor]
    if sensor == "vibration":
        return value > config["abnormal_high"]

    return (
        value < config["abnormal_low"]
        or value > config["abnormal_high"]
    )

def determine_label(values):
    """
    Label rules:
        All three sensors normal
            -> normal

        Any sensor is not normal
            -> abnormal
    """

    for sensor, value in values.items():
        if is_sensor_abnormal(sensor, value):
            return "abnormal"
    return "normal"


# Generate dataset
def generate_data(row_count, interval_minutes):

    data = []
    timestamp = datetime.now().replace(
        second=0,
        microsecond=0
    )
    values = initialize_values()
    anomaly = None

    for _ in range(row_count):

        # 1. Check whether a new anomaly should start
        if anomaly is None:

            if random.random() < ANOMALY_PROBABILITY:

                anomaly = create_anomaly_event()
                # Sudden anomaly:
                # immediately move selected sensors
                # into their abnormal ranges.
                for sensor, direction in anomaly.items():

                    values[sensor] = generate_abnormal_value(sensor, direction)
            else:
                # Normal operation
                values = update_values(values, None)
        else:
            # 2. Existing anomaly continues
            values = update_values(values, anomaly)

        # 3. Determine label from actual sensor values
        label = determine_label(values)
        # 4. add current record
        data.append({
            "timestamp": timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "temp": values["temp"],
            "pressure": values["pressure"],
            "vibration": values["vibration"],
            "label": label,
        })

        # 5. Decide whether anomaly recovers
        if anomaly is not None:
            if random.random() < RECOVERY_PROBABILITY:
                anomaly = None

        # 6. Move to next timestamp
        timestamp += timedelta(
            minutes=interval_minutes
        )

    return data

def save_csv(data):

    # Project root:
    # smart_factory_agent/
    project_root = Path(__file__).resolve().parent.parent
    # Data directory:
    # smart_factory_agent/data/
    data_dir = project_root / "data"
    # Create data directory if it does not exist
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "sensor_data.csv"

    fieldnames = [
        "timestamp",
        "temp",
        "pressure",
        "vibration",
        "label",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved to: {output_path}")

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate sensor time-series data."
    )
    parser.add_argument(
        "--rows",
        type=int,
        required=True,
        help="Number of rows to generate (100-500)."
    )
    parser.add_argument(
        "--interval",
        type=int,
        choices=[1, 5],
        default=1,
        help="Timestamp interval in minutes (1 or 5)."
    )
    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     default="sensor_data.csv",
    #     help="Output CSV filename."
    # )
    args = parser.parse_args()
    if not 100 <= args.rows <= 500:
        parser.error(
            "--rows must be between 100 and 500."
        )
    return args

def main():

    args = parse_arguments()

    data = generate_data(
        row_count=args.rows,
        interval_minutes=args.interval
    )
    save_csv(data)

    print(
        f"Generated {args.rows} rows "
        f"with {args.interval}-minute intervals."
    )

if __name__ == "__main__":
    main()