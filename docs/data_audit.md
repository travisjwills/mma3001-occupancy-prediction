# Data Audit Log

This document records the data-quality investigation carried out before any
modelling, as listed in the Dataset 2 project information ("missing, irregular or
potentially unreliable observations that require auditing"). Each entry is a
finding, its evidence, and its consequence for project scope.

## Files used

| File | Rows | Description |
|---|---|---|
| `5EnvSensor_MayToDec2024_180kRows.csv` | 179,447 | Raw env sensor feed, nested JSON per row, 5 sensors |
| `5occupancySensor_MayToDec2024_9MRows.csv` (zipped) | 9,091,732 | Occupancy status-change event log, 5 zones |
| `Sensor_ID_and_Locations.xlsx` ("ENV" tab) | 78 | Numeric sensor ID -> Building/Room lookup |
| `Sensor_ID_and_Locations.xlsx` ("OCC" tab, labelled "Incorrect Data" by source) | 89 | GUID (SpaceId/FloorId) -> Room lookup |

Two further files (`Detailed_14_07_26_-_17_07_26.csv`, `PMV_07_25_-_04_26.csv`,
`Env_data_MSI_Lab...csv`) were supplied as examples from a different dataset
and are **not used** in this project (see project chat history for the
correction).

## Finding 1 — The room-location lookup tables do not cover the sensors actually present in the data

- `5EnvSensor...csv` contains exactly 5 distinct `sensorid` values:
  `6012002000227, 6012002000241, 6012002000326, 6012002000777, 6012002000869`.
- Checked against the "ENV" lookup tab: **only 1 of 5** (`6012002000869` ->
  Building 60, Room G.38, "Compactus_upper") is present. The other 4 do not
  appear in the lookup table at all.
- `5occupancySensor...csv` contains exactly 5 distinct `floorspaceid` GUIDs.
  Checked against the "OCC" lookup tab (which the source file itself titles
  "Incorrect Data – SpaceFloorId"): **only 2 of 5** GUIDs are present
  (`cca1ac3e...` -> G.20, `227f1068...` -> G.25), and the sheet's own name
  warns this mapping may not be trustworthy even for those two.
- **Consequence:** there is no reliable, source-provided way to join the
  5 environmental sensors to the 5 occupancy zones by physical room name.
  An initial plan to scope the project to the two "confirmed" rooms
  (G.20/G.25) is not viable, because neither of their occupancy GUIDs
  corresponds to any of the 5 environmental sensors in the actual data.

## Finding 2 — 3 of the 5 environmental sensors have corrupted/stuck timestamps

While building the resampling pipeline (`src/load_env.py`), CO2 readings for
3 of the 5 sensors collapsed to only 1–2 distinct 15-minute time bins despite
having tens of thousands of rows each:

| sensorid | rows (all variables) | distinct 15-minute intervals with a CO2 reading |
|---|---|---|
| 6012002000227 | 234,993 | 1 |
| 6012002000241 | 618,422 | **2** |
| 6012002000326 | 641,163 | 20,091 |
| 6012002000777 | 120,186 | 1 |
| 6012002000869 | 322,751 | 9,650 |

Sensors 227 and 777 report *every single one* of their ~21k–120k readings
under one identical timestamp (e.g. all of sensor 227's readings are stamped
`2023-12-11 05:07:53+00:00`). Sensor 241 does the same but splits its
readings across exactly two timestamps. This is consistent with a device
clock fault or a logging bug on the sensor/gateway side (e.g. a timestamp
field that failed to update and was left at its last/default value), not
real sensor behaviour — it is physically implausible for a device to report
tens of thousands of distinct environmental readings at the exact same
instant.

**Consequence:** only sensors `6012002000326` and `6012002000869` have usable
time-series data. The project's usable environmental sample is **2 sensors**,
not 5.

## Finding 3 — Inferred sensor-to-zone correspondence via CO2/headcount correlation

Because Findings 1–2 rule out a lookup-table join, sensor-to-zone pairing was
inferred from the data itself: CO2 is a well-established physical proxy for
occupancy (CO2 rises when people are present and breathing in an enclosed
space), so each sensor's CO2 series was correlated against each zone's
headcount series on a common 15-minute grid, and a one-to-one assignment was
solved with the Hungarian algorithm (`src/match_sensors_to_zones.py`,
reproducible; outputs in `reports/sensor_zone_correlation_matrix.csv` and
`reports/sensor_zone_match.csv`).

Results (only the 2 sensors with real time series produced usable
correlations; the other 3 are NaN by construction, per Finding 2):

| sensorid | best-matching zone | r | margin over 2nd-best |
|---|---|---|---|
| 6012002000869 | `3e5a14df-...` | **0.609** | 0.063 |
| 6012002000326 | `227f1068-...` | -0.105 | 0.018 |

**Interpretation:**
- `6012002000869 <-> 3e5a14df-...` is a moderately strong, plausible match
  (r≈0.61) and is the one pairing in this dataset with reasonable evidence
  behind it.
- `6012002000326` does not correlate meaningfully (weakly negative) with any
  zone, so its physical zone cannot be inferred with confidence from this
  method. It may be sited somewhere not reflected in the occupancy zones at
  all (e.g. a corridor or external area), or the occupancy log for its true
  zone may itself be noisy.

## Scope decision

Given Findings 1–3, the dataset actually supports a **single well-evidenced
environment↔occupancy pair** (sensor 869 / zone 3e5a14df), plus one weakly-
evidenced candidate pair (sensor 326). This is materially narrower than the
original "5 rooms" framing implied by the filenames, and is itself a central
finding of the project: **the headline dataset size (9M / 180k rows) masks
a much smaller usable sample once data quality is audited.**

See the project README for how this shapes the final problem scope.
