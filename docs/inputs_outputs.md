# Inputs and Outputs Specification

This document defines, precisely, what the project's computational solution
accepts and produces — both at the top level (the overall pipeline) and for
the individual functions in `src/`. Per-function detail (parameters, types,
return values, exceptions) is also available as generated HTML documentation
in `docs/html/` (see `scripts/build_html_docs.sh`); this file gives the
human-readable summary the brief asks for.

## 1. Top-level inputs

| Input | Format | Units / domain | Required? |
|---|---|---|---|
| Environmental sensor feed (`5EnvSensor_MayToDec2024_180kRows.csv`) | CSV, one row per reading event, variables bundled as a JSON blob in the `jsondata` column (see §3) | CO2: ppm (0–5000 plausible); Temperature: °C (0–50 plausible); Humidity: %RH (0–100); Formaldehyde: µg/m³; LVOC: ppb; Pressure: mb; PM1/2.5/4/10: µg/m³; timestamps: Unix seconds, UTC | Yes |
| Occupancy event log (`5occupancySensor_MayToDec2024_9MRows.csv`, zipped) | CSV, one row per status-change event | `headcount`: non-negative integer; `occupancystatus`: one of `CurrentlyOccupied`/`RecentlyOccupied`/`NotOccupied`; `floorspaceid`: GUID string; `collecteddate`: ISO 8601 timestamp with UTC offset | Yes |
| Resampling frequency (`FREQ` constant, e.g. `"15min"`) | pandas frequency string | Must be a valid pandas offset alias | No — defaults to `15min` throughout |
| Target sensor/zone pair (`SENSOR_ID`, `ZONE_ID` constants in `build_dataset.py`) | integer sensor ID; GUID string | Must match an ID actually present in the data | Yes, once the audit in `docs/data_audit.md` has identified a usable pair |

## 2. Top-level outputs

| Output | Format | Meaning / intended use |
|---|---|---|
| `data/processed/room_dataset.csv` | CSV, one row per 15-minute interval | The joined modelling dataset: environmental readings + occupancy label for one room. Input to `train_model.py`. |
| `reports/sensor_zone_correlation_matrix.csv`, `sensor_zone_match.csv` | CSV | Evidence for which sensor/zone pairing was used, and how confident that inference is (see `docs/data_audit.md`) |
| `reports/model_comparison.csv`, `cv_results.csv` (F1 per fold for all four models), `feature_importance.csv`, `confusion_matrices.txt` | CSV / text | Model performance metrics, used directly in `docs/model_results.md` |
| `reports/optimisation_comparison.csv`, `hyperparam_tuning.csv`, `sensitivity_env_only.csv` | CSV | Runtime/memory/accuracy trade-off results, used in `docs/optimisation.md` |
| `reports/cv_fold_dates.csv`, `daily_occupancy.csv` | CSV | Dates and occupied share of each cross-validation fold, and of each calendar day; used to explain the weak fold in `docs/model_results.md` |
| `reports/figures/*.png` | PNG, 300dpi | Report/slide-ready charts |
| `reports/test_results.xml`, `test_report.html` | JUnit XML / HTML | Automated test results (see §5) |

**Intended use:** this pipeline is a feasibility study, not a deployed system — the outputs demonstrate whether environmental sensing can estimate occupancy for one audited room, and quantify how well, not a production inference service. A real deployment would additionally need a packaged model artifact and a live data-ingestion path, which are out of scope here (see `docs/data_audit.md` and `docs/model_results.md` for the stated limitations).

## 3. Function-level inputs/outputs (key functions)

### `load_env.load_env_long(path, nrows=None) -> pd.DataFrame`
- **Input:** `path` — string/Path to the raw env CSV. `nrows` — optional int, limits rows read (for quick testing).
- **Output:** long-format DataFrame with columns `sensorid` (int), `timestamp` (UTC `datetime64`), `variable` (str, e.g. `"Carbon dioxide"`), `unit` (str), `value` (float).
- **Invalid/missing input handling:** a row whose `jsondata` field fails to parse as JSON is silently skipped (wrapped in `try/except`) rather than raising, since a handful of malformed rows should not abort processing of 179k+ rows; this is a deliberate trade-off, documented here rather than hidden, and the row-count printed by callers makes any large-scale parse failure visible.

### `load_env.resample_sensor_variable(long_df, variable, freq="15min", start=None, end=None) -> pd.DataFrame`
- **Input:** long-format DataFrame (as above), the variable name to pivot (must match a value in the `variable` column exactly), frequency, optional explicit time window.
- **Output:** wide DataFrame, index = time bins (UTC, aligned to `freq` boundaries), columns = `sensorid`, values = mean reading in that bin. A sensor/bin with no reading is `NaN`, not zero or interpolated — missingness is preserved explicitly rather than guessed at.
- **Domain note:** `start`/`end` can be any timestamp; internally they are floored/ceiled to the nearest `freq` boundary so the output grid aligns exactly with `load_occupancy.resample_zone_headcount`'s grid (required for the two to be joined — see the regression test in `tests/test_load_env.py` and `tests/test_load_occupancy.py` covering this).

### `load_occupancy.load_occupancy_events(path) -> pd.DataFrame`
- **Input:** path to the (unzipped) occupancy CSV.
- **Output:** DataFrame with columns `floorspaceid`, `occupancystatus`, `headcount`, `timestamp` (UTC), sorted by zone then time.

### `load_occupancy.resample_zone_headcount(events, freq="15min", start=None, end=None) -> pd.DataFrame`
- **Input:** events DataFrame (as above).
- **Output:** wide DataFrame, index = time bins, columns = `floorspaceid`, values = forward-filled headcount.
- **Invalid/missing input handling:** since the source is an event log (a row only on status change), any time bin before a zone's first recorded event has **no known state** and is left as `NaN` rather than assumed to be 0 — assuming "not occupied" for unknown history would silently bias the labels. Downstream, `build_dataset.py` drops rows with NaN in any required column (see §4) rather than imputing them, so the final modelling dataset only contains time bins with a genuinely known occupancy state.

### `train_model.best_co2_threshold(co2, y) -> float`
- **Input:** `co2` — 1D array of CO2 readings; `y` — matching 0/1 occupancy labels.
- **Output:** a single float threshold (ppm) that maximises F1 on the given data, scanned over 5th–99th percentile candidates (every 2nd percentile) of the training CO2 values.
- **Domain:** both arrays must be the same length and contain no NaNs (enforced upstream by `build_dataset.py`'s `dropna()`).

### `train_model.evaluate(y_true, y_pred, y_score=None) -> dict`
- **Input:** true labels, predicted labels, optional predicted probabilities/scores.
- **Output:** dict of `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.
- **Invalid/missing input handling:** if `y_true` contains only one class (can happen in a small CV fold), ROC-AUC is mathematically undefined; the function returns `NaN` for that metric rather than raising, so a single degenerate fold doesn't crash a whole cross-validation run (covered by `tests/test_train_model.py::test_evaluate_handles_single_class_without_crashing`).

## 4. Data cleaning / row-dropping policy

`build_dataset.py` drops any row with a missing value in any of the 10
environmental variables or `headcount` (`df.dropna(subset=...)`). This is a
deliberate, conservative choice: rather than impute environmental readings
(which could fabricate a physically unrealistic value) or assume a default
occupancy state, rows without a complete, genuinely-observed record are
excluded. The console output of `build_dataset.py` reports exactly how many
rows were dropped and why (4,099 kept of 10,129 candidate rows for the
audited sensor/zone pair), so this is an auditable, not a silent, step.

## 5. Automated tests

`tests/` contains pytest unit tests for the data-loading and modelling
helper functions above, run against small synthetic fixtures (`tests/conftest.py`)
rather than the real multi-hundred-MB source files, so the suite runs in
under 2 seconds and needs no external data. Run with:

```bash
pytest tests/ -v --junitxml=reports/test_results.xml
python scripts/junit_to_html.py   # renders reports/test_report.html
```

Current status: 9/9 tests passing. Notably, two tests
(`test_resample_grid_is_bin_aligned_regardless_of_start_time` in both
`test_load_env.py` and `test_load_occupancy.py`) are regression tests for a
real bug found during development — an earlier version of the resampling
code produced an all-`NaN` result because the output time grid wasn't
aligned to the same clock boundaries pandas' own `.resample()` uses. These
tests exist specifically so that bug cannot silently reappear.
