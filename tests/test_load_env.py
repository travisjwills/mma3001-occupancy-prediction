"""Tests for src/load_env.py.

Covers JSON-blob flattening and the sensor-variable resampling, including
the same bin-alignment regression covered for load_occupancy.py (both
loaders must produce grids that line up with each other, since
match_sensors_to_zones.py and build_dataset.py join them directly).
"""
import pandas as pd

from load_env import load_env_long, resample_sensor_variable


def test_load_env_long_flattens_json_correctly(tiny_env_csv):
    long_df = load_env_long(str(tiny_env_csv))
    assert set(long_df.columns) == {"sensorid", "timestamp", "variable", "unit", "value"}
    assert len(long_df) == 3  # one CO2 reading per row in the fixture
    assert set(long_df["sensorid"]) == {111, 222}
    assert (long_df["variable"] == "Carbon dioxide").all()
    assert (long_df["unit"] == "ppm").all()


def test_load_env_long_converts_unix_seconds_to_utc_timestamp(tiny_env_csv):
    long_df = load_env_long(str(tiny_env_csv))
    first = long_df.iloc[0]
    # 1700000000 unix seconds -> 2023-11-14 22:13:20 UTC
    assert first["timestamp"] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")


def test_resample_sensor_variable_pivots_wide_and_aligns_bins(tiny_env_csv):
    long_df = load_env_long(str(tiny_env_csv))
    wide = resample_sensor_variable(
        long_df, "Carbon dioxide", freq="15min",
        start="2023-11-14 22:07:00+00:00",  # deliberately not on a 15-min boundary
        end="2023-11-14 22:45:00+00:00",
    )
    assert set(wide.columns) == {111, 222}
    # Every bin edge must land on a clean 15-minute clock boundary, matching
    # resample_zone_headcount's grid (see test_load_occupancy.py).
    assert all(ts.minute % 15 == 0 and ts.second == 0 for ts in wide.index)
    # Sensor 111 has two readings 15 minutes apart (500 then 600); each
    # should land in its own bin rather than collapsing to one NaN column.
    assert wide[111].notna().sum() == 2
    assert wide[222].notna().sum() == 1
