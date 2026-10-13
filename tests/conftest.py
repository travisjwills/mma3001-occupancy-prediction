"""Shared pytest fixtures: small synthetic CSVs so tests run in milliseconds
and never need the real (multi-hundred-MB) source data files."""
import sys
from pathlib import Path

import pytest

# Make src/ importable as a package-less set of modules, same way the
# scripts in src/ import each other.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def tiny_env_csv(tmp_path):
    """A 3-row env sensor CSV with the same nested-JSON shape as the real
    5EnvSensor_MayToDec2024_180kRows.csv file, for two sensors."""
    path = tmp_path / "tiny_env.csv"
    rows = [
        '"id","sensorid","devicetype","jsondata","status","processing_errors","createdate","processdate"',
        # sensor A, t=1700000000 (2023-11-14 22:13:20 UTC): CO2=500
        '1,"111","env","[{""values"": [{""time"": 1700000000, ""value"": 500}], '
        '""variable"": {""id"": 67, ""name"": ""Carbon dioxide"", ""unit"": ""ppm"", ""source"": 0, ""structure"": 1}}]",'
        '"Processed",NULL,"2023-11-14","2023-11-14"',
        # sensor A, t=1700000900 (15 min later): CO2=600
        '2,"111","env","[{""values"": [{""time"": 1700000900, ""value"": 600}], '
        '""variable"": {""id"": 67, ""name"": ""Carbon dioxide"", ""unit"": ""ppm"", ""source"": 0, ""structure"": 1}}]",'
        '"Processed",NULL,"2023-11-14","2023-11-14"',
        # sensor B, t=1700000000: CO2=400
        '3,"222","env","[{""values"": [{""time"": 1700000000, ""value"": 400}], '
        '""variable"": {""id"": 67, ""name"": ""Carbon dioxide"", ""unit"": ""ppm"", ""source"": 0, ""structure"": 1}}]",'
        '"Processed",NULL,"2023-11-14","2023-11-14"',
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def tiny_occupancy_csv(tmp_path):
    """A small occupancy event-log CSV: one zone, status changes over a
    known window, used to check forward-fill resampling."""
    path = tmp_path / "tiny_occ.csv"
    rows = [
        '"id","deviceid","floorspaceid","occupancystatus","headcount","collecteddate","occupancystatuschangedate","previousoccupancystatus"',
        # zone Z1 becomes occupied (headcount=2) at 00:05
        '1,"dev1","Z1","CurrentlyOccupied",2,"2024-01-01 00:05:00+00","2024-01-01 00:05:00+00","NotOccupied"',
        # zone Z1 becomes empty at 00:40
        '2,"dev1","Z1","NotOccupied",0,"2024-01-01 00:40:00+00","2024-01-01 00:40:00+00","CurrentlyOccupied"',
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
