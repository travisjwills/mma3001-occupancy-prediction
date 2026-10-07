"""
Load and resample the 5-sensor environmental feed.

The source file (`5EnvSensor_MayToDec2024_180kRows.csv`) stores one row per
sensor reading *event*, with all variables for that reading bundled as a JSON
blob in the `jsondata` column, e.g.:

    [{"values": [{"time": 1702273965, "value": 603}],
      "variable": {"name": "Carbon dioxide", "unit": "ppm", ...}}, ...]

Each row typically contains ~10 variables (Battery, CO2, Humidity,
Formaldehyde, Pressure, Temperature, LVOC, PM1/2.5/4/10) captured at the same
timestamp. We flatten this into a tidy long table, then pivot to wide.
"""
from __future__ import annotations

import ast
import json

import pandas as pd


def _parse_jsondata(raw: str) -> list[dict]:
    """jsondata uses doubled double-quotes (CSV-escaped JSON). json.loads
    handles it directly once pandas has already un-escaped the CSV quoting."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # fall back for any stray formatting issues
        return ast.literal_eval(raw)


def load_env_long(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw env sensor file and flatten into long format.

    Returns
    -------
    pd.DataFrame
        Columns: sensorid, timestamp (UTC), variable, unit, value
    """
    cols = ["sensorid", "jsondata"]
    df = pd.read_csv(path, usecols=cols, nrows=nrows)

    records = []
    for sensorid, raw in zip(df["sensorid"], df["jsondata"]):
        try:
            entries = _parse_jsondata(raw)
        except Exception:
            continue
        for entry in entries:
            var = entry.get("variable", {})
            for v in entry.get("values", []):
                records.append(
                    {
                        "sensorid": sensorid,
                        "timestamp": v.get("time"),
                        "variable": var.get("name"),
                        "unit": var.get("unit"),
                        "value": v.get("value"),
                    }
                )

    long_df = pd.DataFrame.from_records(records)
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], unit="s", utc=True)
    return long_df


def resample_sensor_variable(
    long_df: pd.DataFrame,
    variable: str,
    freq: str = "15min",
    start=None,
    end=None,
) -> pd.DataFrame:
    """Pivot one variable (e.g. 'Carbon dioxide') to a regular-interval wide
    table: index = timestamp, columns = sensorid, values = mean in each bin.
    """
    sub = long_df[long_df["variable"] == variable].copy()
    if start is None:
        start = sub["timestamp"].min()
    if end is None:
        end = sub["timestamp"].max()

    sub = sub.set_index("timestamp")
    wide = (
        sub.groupby("sensorid")["value"]
        .resample(freq)
        .mean()
        .unstack(level=0)
    )
    # .resample() floors timestamps onto freq-aligned bin edges (e.g. :00/:15/
    # :30/:45), so the reindex target must use the same aligned edges rather
    # than an index that starts exactly at `start` (which may fall mid-bin).
    full_index = pd.date_range(
        pd.Timestamp(start).floor(freq), pd.Timestamp(end).ceil(freq), freq=freq, tz="UTC"
    )
    wide = wide.reindex(full_index)
    return wide
