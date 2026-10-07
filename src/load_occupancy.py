"""
Load and resample the occupancy sensor log.

The source file (`5occupancySensor_MayToDec2024_9MRows.csv`, delivered as a .zip)
is an EVENT LOG, not a regular time series: a row is written only when a zone's
`occupancystatus` changes. To get a value at arbitrary time steps we must
forward-fill the last known status/headcount between recorded events.

Columns of interest:
    floorspaceid   - GUID identifying one of 5 monitored zones
    occupancystatus - CurrentlyOccupied / RecentlyOccupied / NotOccupied
    headcount      - integer people count at that event
    collecteddate  - timestamp the event was logged (timezone-aware)
"""
from __future__ import annotations

import pandas as pd


OCC_STATUS_ORDER = ["NotOccupied", "RecentlyOccupied", "CurrentlyOccupied"]


def load_occupancy_events(path: str) -> pd.DataFrame:
    """Read the raw occupancy event log into a tidy DataFrame.

    Parameters
    ----------
    path : str
        Path to the (already-unzipped) occupancy CSV.

    Returns
    -------
    pd.DataFrame
        Columns: floorspaceid, occupancystatus, headcount, timestamp (UTC).
    """
    usecols = ["floorspaceid", "occupancystatus", "headcount", "collecteddate"]
    df = pd.read_csv(path, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["collecteddate"], utc=True, format="ISO8601")
    df = df.drop(columns=["collecteddate"])
    df = df.sort_values(["floorspaceid", "timestamp"]).reset_index(drop=True)
    return df


def resample_zone_headcount(
    events: pd.DataFrame,
    freq: str = "15min",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Turn the irregular event log into a regular-interval headcount series
    per zone, by forward-filling the last known headcount into each time bin.

    Returns a wide DataFrame: index = timestamp, columns = floorspaceid.
    """
    if start is None:
        start = events["timestamp"].min()
    if end is None:
        end = events["timestamp"].max()
    # Align bin edges with pandas .resample()-style flooring so this grid
    # matches resample_sensor_variable()'s grid exactly (see load_env.py).
    full_index = pd.date_range(
        pd.Timestamp(start).floor(freq), pd.Timestamp(end).ceil(freq), freq=freq, tz="UTC"
    )

    series = {}
    for zone_id, grp in events.groupby("floorspaceid"):
        s = grp.set_index("timestamp")["headcount"]
        # de-duplicate timestamps (keep the last event if several share a timestamp)
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(s.index.union(full_index)).sort_index()
        s = s.ffill()
        s = s.reindex(full_index)
        series[zone_id] = s

    return pd.DataFrame(series, index=full_index)
