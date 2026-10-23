# MMA3001 — Predicting Room Occupancy from Environmental Sensor Data

Individual project for MMA3001 (Numerical Methods and Machine Learning),
using Monash Smart Infrastructure: (MMA3001 Dataset 2), (Living Lab occupancy and
environmental sensors, Building 60/69).

**Status:** data audit, modelling, optimisation, figures, and repo
infrastructure (tests/docs/license) complete. Written report in progress.

## Problem

Can indoor environmental readings (CO2, temperature, humidity, etc.) be used
to estimate room occupancy, without a dedicated occupancy sensor? CO2 is a
known physical proxy for occupancy (people raise CO2 by breathing), so this
is tested as a practical, physically-grounded estimation problem relevant to
low-cost building monitoring and HVAC/energy management.

## Repository structure

```
LICENSE                 MIT license for the code (see note on data below)
README.md               this file
requirements.txt        Python dependencies

src/                     all pipeline code (run in this order — see below)
tests/                   pytest unit tests + fixtures (synthetic data, no large files needed)
scripts/                 helper scripts: build_html_docs.sh, junit_to_html.py

docs/                    written findings and specifications
  data_audit.md          data-quality investigation (lookup tables, broken sensors, correlation matching)
  model_results.md       baseline vs Random Forest results, validation, sensitivity check
  optimisation.md        runtime/memory/accuracy trade-offs, hyperparameter tuning, final model choice
  inputs_outputs.md      formal inputs/outputs specification for the pipeline and key functions
  html/                  auto-generated HTML API documentation (see scripts/build_html_docs.sh)

data/
  raw/                   (empty on GitHub) place source CSVs here to reproduce
  processed/             (empty on GitHub) cleaned dataset is written here when src/build_dataset.py runs

reports/                 all generated outputs (tables, figures, test results)
  figures/               7 report/slide-ready PNG charts
  test_results.xml, test_report.html   automated test results
  *.csv, *.txt           numeric results backing docs/ and the written report
```

## Key results

Single-room occupancy classification (sensor `6012002000869` / zone
`3e5a14df-...`, the one sensor/zone pair the data audit found to be usable —
see `docs/data_audit.md`), 4,099 samples, chronological 80/20 split. "CV" is
5-fold time-series cross-validation on the training portion.

| Model | Test F1 | Test ROC-AUC | Mean CV F1 | Fit time | Predict/row | Model size |
|---|---|---|---|---|---|---|
| Baseline: CO2 threshold rule | 0.345 | 0.568 | 0.519 | — | — | — |
| Logistic Regression | 0.756 | 0.708 | 0.536 | 11.3 ms | 1.1 µs | 1.8 KB |
| Random Forest | 0.763 | 0.736 | 0.698 | 716.0 ms | 91.2 µs | 4.2 MB |
| **Gradient Boosting (recommended final model)** | 0.761 | 0.724 | 0.702 | 432.3 ms | 1.5 µs | 135.8 KB |

All three machine-learning models are within 1 point of F1 on the held-out
test period, so that split alone cannot separate them. Cross-validation can:
Logistic Regression is much less stable over time (mean CV F1 0.536 vs about
0.70 for the tree models). Gradient Boosting matches the Random Forest on
accuracy but predicts ~59x faster and is ~31x smaller, so it is the
recommended design — see `docs/optimisation.md` for the full trade-off analysis
(including an earlier Logistic Regression recommendation that the
cross-validation check overturned).

See `docs/model_results.md` for full validation results, including why one
cross-validation fold is weak for every model and a check for whether the
model is just learning the room's schedule rather than reading the air (it
isn't, mostly).

### Headcount Regression

The same data, split and folds are also used to predict the number of people
in the room (0 to 22) with Decision Tree Regression, its depth chosen by
time-series cross-validation on the training data (`src/headcount_regression.py`).

| Model | Test MAE (people) | Test RMSE (people) | Test predictions within 1 person | Mean CV MAE (people) |
|---|---|---|---|---|
| Mean baseline (always predict the training average) | 1.23 | 1.36 | 49% | 2.27 |
| Linear Regression | 0.83 | 1.08 | 70% | 2.03 |
| **Decision Tree Regression (depth 6)** | 0.80 | 1.52 | 85% | 1.36 |

The tree has the lowest typical error in cross-validation and on the test
period, but it sometimes predicts a crowd that did not arrive, so its RMSE is
the worst. The test period was unusually quiet (mean 0.7 people, against 1.8
in training), so test R^2 is negative for every model and is not a useful
summary here; see `docs/model_results.md`.

## Reproducing the pipeline

```bash
pip install -r requirements.txt
```

1. Obtain `5EnvSensor_MayToDec2024_180kRows.csv` and
   `5occupancySensor_MayToDec2024_9MRows.csv` (zipped) from the unit's
   Dataset 2 drive folder and place them under `data/raw/`.
2. Update the path constants at the top of each `src/` script if your paths
   differ from the defaults.
3. Run in order:
   ```bash
   python src/match_sensors_to_zones.py   # data audit: infers sensor<->zone pairing
   python src/build_dataset.py            # builds data/processed/room_dataset.csv
   python src/train_model.py              # baseline vs Random Forest + validation
   python src/sensitivity_env_only.py     # time-of-day sensitivity check
   python src/cv_fold_diagnostics.py      # dates/occupancy of each CV fold (explains the weak fold)
   python src/headcount_regression.py     # extension: predict headcount with Decision Tree Regression
   python src/optimisation.py             # runtime/memory/accuracy comparison + tuning
   python src/make_figures.py             # generates reports/figures/*.png
   ```

## Tests

```bash
pytest tests/ -v --junitxml=reports/test_results.xml
python scripts/junit_to_html.py   # renders reports/test_report.html
```

11/11 tests passing. Tests run against small synthetic fixtures
(`tests/conftest.py`), not the real multi-hundred-MB source files, so the
suite runs in under 2 seconds with no external data required. Two of the
tests are regression tests for a real bug found during development (a
time-grid misalignment between the two resampling functions that silently
produced all-`NaN` output) — see `docs/inputs_outputs.md` §5 for details.

## Code documentation

HTML API documentation (generated from docstrings via Python's built-in
`pydoc`) is in `docs/html/` — open `docs/html/index.html`. Regenerate with:

```bash
bash scripts/build_html_docs.sh
```

## License

Code is MIT licensed (see `LICENSE`). The source sensor data is **not**
included in this repository and remains subject to Monash University's own
data-sharing terms for the MMA3001 unit.

## AI use disclosure
AI assistance (Claude, Anthropic) was used throughout development of this
repository; the full disclosure — what it was used for, how outputs were
verified, and which decisions were made — will be written
in the final report.
