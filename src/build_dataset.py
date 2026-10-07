"""
Build the modelling dataset for the single well-evidenced sensor/zone pair
identified in docs/data_audit.md:

    env sensor 6012002000869  <->  occupancy zone 3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8

Joins every environmental variable from that sensor with the zone's
occupancy status and headcount on a common 15-minute grid, adds basic
time-of-day / day-of-week features, and writes the result to
data/processed/room_dataset.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_env import load_env_long, resample_sensor_variable
from load_occupancy import load_occupancy_events, resample_zone_headcount

ENV_PATH = "/mnt/user-data/uploads/5EnvSensor_MayToDec2024_180kRows.csv"
OCC_PATH = "/home/claude/rawdata/occupancy/5occupancySensor_MayToDec2024_9MRows.csv"
SENSOR_ID = 6012002000869
ZONE_ID = "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"
FREQ = "15min"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "room_dataset.csv"

ENV_VARIABLES = [
    "Carbon dioxide",
    "Temperature",
    "Humidity",
    "Formaldehyde",
    "Light Volatile Organic Compounds",
    "Pressure",
    "Particulate matter 1",
    "Particulate matter 2.5",
    "Particulate matter 4",
    "Particulate matter 10",
]


def main():
    print("Loading raw data ...")
    env_long = load_env_long(ENV_PATH)
    occ_events = load_occupancy_events(OCC_PATH)

    # Restrict to the sensor's own real data window (per the audit, sensor 869's
    # readings run 2024-02-29 -> 2024-06-13; outside that window there is no
    # real reading to resample, so we don't want the grid to extend past it).
    sensor_rows = env_long[env_long["sensorid"] == SENSOR_ID]
    start, end = sensor_rows["timestamp"].min(), sensor_rows["timestamp"].max()
    print(f"Sensor {SENSOR_ID} active window: {start} -> {end}")

    frames = {}
    for var in ENV_VARIABLES:
        wide = resample_sensor_variable(env_long, var, freq=FREQ, start=start, end=end)
        if SENSOR_ID in wide.columns:
            frames[var] = wide[SENSOR_ID]
        else:
            print(f"  WARNING: sensor {SENSOR_ID} missing for variable '{var}'")

    env_df = pd.DataFrame(frames)

    headcount = resample_zone_headcount(occ_events, freq=FREQ, start=start, end=end)
    occ_series = headcount[ZONE_ID].rename("headcount")

    df = env_df.join(occ_series, how="inner")
    df.index.name = "timestamp"
    df = df.reset_index()

    # Occupied/not-occupied binary label (headcount > 0), useful for a
    # classification framing alongside the regression (headcount) framing.
    df["occupied"] = (df["headcount"] > 0).astype(int)

    # Time features
    local = df["timestamp"].dt.tz_convert("Australia/Melbourne")
    df["hour"] = local.dt.hour
    df["dayofweek"] = local.dt.dayofweek  # 0=Monday
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    before = len(df)
    df = df.dropna(subset=ENV_VARIABLES + ["headcount"])
    print(f"Rows before dropna: {before}, after: {len(df)} ({before - len(df)} dropped)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    main()
