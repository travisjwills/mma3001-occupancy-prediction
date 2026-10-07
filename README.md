# MMA3001 — Predicting Room Occupancy from Environmental Sensor Data

Individual project for MMA3001 (Numerical Methods and Machine Learning),
using Monash Smart Infrastructure's Dataset 2 (Living Lab occupancy and
environmental sensors, Building 60/69).

**Status:** data audit complete, scope confirmed, modelling in progress.
See `docs/data_audit.md` for the full data-quality investigation.

## Problem

Can indoor environmental readings (CO2, temperature, humidity, etc.) be used
to estimate room occupancy, without a dedicated occupancy sensor? CO2 is a
known physical proxy for occupancy (people raise CO2 by breathing), so this
is tested as a practical, physically-grounded estimation problem relevant to
low-cost building monitoring and HVAC/energy management.

## Repository structure

```
data/raw/         source CSVs (gitignored — too large for git; see below)
data/processed/   cleaned/resampled intermediate outputs (gitignored)
src/              reusable data-loading and analysis code
notebooks/         exploratory notebooks
reports/           generated tables/figures from src/ scripts
docs/              data audit log and other written documentation
tests/             unit tests for src/
```

## Data

Source files are not committed to this repository (too large for git /
contain a GUID-identified building's operational data). To reproduce:
1. Obtain `5EnvSensor_MayToDec2024_180kRows.csv` and
   `5occupancySensor_MayToDec2024_9MRows.csv` (zipped) from the unit's
   Dataset 2 drive folder.
2. Place them under `data/raw/`.
3. Update the `ENV_PATH` / `OCC_PATH` constants at the top of
   `src/match_sensors_to_zones.py` if your paths differ.

## Reproducing the data audit

```bash
pip install -r requirements.txt
python src/match_sensors_to_zones.py
```

This regenerates `reports/sensor_zone_correlation_matrix.csv` and
`reports/sensor_zone_match.csv`, and prints the key findings summarised in
`docs/data_audit.md`.

## AI use disclosure

This project was developed with assistance from Claude (Anthropic). See
`docs/ai_use.md` for the detailed disclosure required by the project brief
(tools used, tasks, level of contribution, verification steps, and decisions
that remained the student's own).
