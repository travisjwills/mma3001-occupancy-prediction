"""
Infer the correspondence between the 5 environmental sensors and the 5
occupancy zones.

Why this is needed
-------------------
`5EnvSensor_MayToDec2024_180kRows.csv` identifies locations by `sensorid`
(numeric). `5occupancySensor_MayToDec2024_9MRows.csv` identifies locations by
`floorspaceid` (GUID). Neither the "Sensor ID and Locations" lookup table nor
the "OCC" tab (which Monash itself labels "Incorrect Data") provides a usable
crosswalk between these two identifier systems for the specific 5 sensors
and 5 zones present in this data (see docs/data_audit.md for the full trail).

Method
------
CO2 concentration is a well-established physical proxy for occupancy
(exhaled CO2 raises concentration when people are present). We:
  1. Resample both CO2 (env) and headcount (occupancy) to a common 15-min
     grid over their shared time window.
  2. Compute the Pearson correlation between every (sensor, zone) pair.
  3. Solve the resulting 5x5 assignment problem (maximise total correlation,
     one-to-one) using the Hungarian algorithm.
  4. Report both the matched pairing AND the full correlation matrix, so a
     reader can judge how confident each match is (a near-diagonal matrix
     with large margins between best and second-best supports the inferred
     mapping; small margins are a limitation to flag).

This script produces: reports/sensor_zone_correlation_matrix.csv and prints
the inferred pairing with confidence margins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from load_env import load_env_long, resample_sensor_variable
from load_occupancy import load_occupancy_events, resample_zone_headcount

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ENV_PATH = RAW_DIR / "5EnvSensor_MayToDec2024_180kRows.csv"
OCC_PATH = RAW_DIR / "5occupancySensor_MayToDec2024_9MRows.csv"
FREQ = "15min"
OUT_DIR = Path(__file__).resolve().parents[1] / "reports"


def main():
    print("Loading environmental sensor data ...")
    env_long = load_env_long(ENV_PATH)
    print(f"  {len(env_long):,} readings across variables: {sorted(env_long['variable'].dropna().unique())}")

    print("Loading occupancy event log ...")
    occ_events = load_occupancy_events(OCC_PATH)
    print(f"  {len(occ_events):,} events across {occ_events['floorspaceid'].nunique()} zones")

    # Overlapping time window between the two datasets
    env_start, env_end = env_long["timestamp"].min(), env_long["timestamp"].max()
    occ_start, occ_end = occ_events["timestamp"].min(), occ_events["timestamp"].max()
    start = max(env_start, occ_start)
    end = min(env_end, occ_end)
    print(f"Env window:  {env_start} -> {env_end}")
    print(f"Occ window:  {occ_start} -> {occ_end}")
    print(f"Overlap used for matching: {start} -> {end}")

    co2 = resample_sensor_variable(env_long, "Carbon dioxide", freq=FREQ, start=start, end=end)
    headcount = resample_zone_headcount(occ_events, freq=FREQ, start=start, end=end)

    sensors = list(co2.columns)
    zones = list(headcount.columns)
    print(f"\nSensors ({len(sensors)}): {sensors}")
    print(f"Zones ({len(zones)}): {zones}")

    corr = pd.DataFrame(index=sensors, columns=zones, dtype=float)
    for s in sensors:
        for z in zones:
            paired = pd.concat([co2[s], headcount[z]], axis=1).dropna()
            if len(paired) < 20:
                corr.loc[s, z] = np.nan
                continue
            corr.loc[s, z] = paired.iloc[:, 0].corr(paired.iloc[:, 1])

    OUT_DIR.mkdir(exist_ok=True)
    corr.to_csv(OUT_DIR / "sensor_zone_correlation_matrix.csv")
    print("\nCorrelation matrix (CO2 vs headcount):")
    print(corr.round(3))

    # Hungarian assignment maximises total correlation (cost = -corr, NaN -> very low)
    cost = -corr.fillna(-1.0).to_numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    print("\nInferred sensor -> zone pairing (Hungarian assignment, max total correlation):")
    results = []
    for r, c in zip(row_ind, col_ind):
        sensor, zone = sensors[r], zones[c]
        best = corr.loc[sensor].max()
        second_best = corr.loc[sensor].drop(zone).max()
        margin = best - second_best if pd.notna(second_best) else np.nan
        print(f"  {sensor}  <->  {zone}   r={corr.loc[sensor, zone]:.3f}   margin over 2nd best={margin:.3f}")
        results.append(
            {
                "sensorid": sensor,
                "floorspaceid": zone,
                "correlation": corr.loc[sensor, zone],
                "margin_over_second_best": margin,
            }
        )

    pd.DataFrame(results).to_csv(OUT_DIR / "sensor_zone_match.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'sensor_zone_correlation_matrix.csv'}")
    print(f"Saved: {OUT_DIR / 'sensor_zone_match.csv'}")


if __name__ == "__main__":
    main()
