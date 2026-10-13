"""Tests for src/load_occupancy.py.

The occupancy source file is an EVENT LOG (a row only when status changes),
so the critical behaviour to test is the forward-fill resampling: a reader
must see the last known headcount at every time step, not just at the
moments an event happened to be logged.
"""
import pandas as pd

from load_occupancy import load_occupancy_events, resample_zone_headcount


def test_load_occupancy_events_parses_and_sorts(tiny_occupancy_csv):
    df = load_occupancy_events(str(tiny_occupancy_csv))
    assert list(df.columns) == ["floorspaceid", "occupancystatus", "headcount", "timestamp"]
    assert len(df) == 2
    # sorted by (floorspaceid, timestamp) ascending
    assert df["timestamp"].is_monotonic_increasing
    assert df["headcount"].tolist() == [2, 0]


def test_resample_forward_fills_between_events(tiny_occupancy_csv):
    events = load_occupancy_events(str(tiny_occupancy_csv))
    wide = resample_zone_headcount(
        events, freq="15min",
        start="2024-01-01 00:00:00+00:00", end="2024-01-01 01:00:00+00:00",
    )
    z1 = wide["Z1"]

    # Before the first event (00:00), headcount is unknown -> NaN, not 0.
    assert pd.isna(z1.loc["2024-01-01 00:00:00+00:00"])
    # At/after the "occupied" event (00:05) but before the next bin edge,
    # the 00:15 bin should carry headcount=2 forward.
    assert z1.loc["2024-01-01 00:15:00+00:00"] == 2
    assert z1.loc["2024-01-01 00:30:00+00:00"] == 2
    # After the "vacated" event (00:40), headcount should drop to 0.
    assert z1.loc["2024-01-01 00:45:00+00:00"] == 0
    assert z1.loc["2024-01-01 01:00:00+00:00"] == 0


def test_resample_grid_is_bin_aligned_regardless_of_start_time(tiny_occupancy_csv):
    """Regression test: an earlier version of this pipeline built the output
    grid starting exactly at `start` (e.g. 00:05:00) while pandas .resample()
    produces bins aligned to clock boundaries (00:00, 00:15, ...). That
    mismatch silently produced an all-NaN result. The grid must always land
    on clean `freq` boundaries no matter what `start`/`end` are passed."""
    events = load_occupancy_events(str(tiny_occupancy_csv))
    wide = resample_zone_headcount(
        events, freq="15min",
        start="2024-01-01 00:07:00+00:00",  # deliberately NOT on a 15-min boundary
        end="2024-01-01 00:50:00+00:00",
    )
    # Every timestamp in the index must fall on a 15-minute clock boundary.
    assert all(ts.minute % 15 == 0 and ts.second == 0 for ts in wide.index)
