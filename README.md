# MMA3001 — Predicting Room Occupancy from Environmental Sensor Data

Individual project for MMA3001 (Numerical Methods and Machine Learning),
using Monash Smart Infrastructure's Dataset 2 (Living Lab occupancy and
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
  figures/               6 report/slide-ready PNG charts
  test_results.xml, test_report.html   automated test results
  *.csv, *.txt           numeric results backing docs/ and the written report
```

## Key results

Single-room occupancy classification (sensor `6012002000869` / zone
`3e5a14df-...`, the one sensor/zone pair the data audit found to be usable —
see `docs/data_audit.md`), 4,099 samples, chronological 80/20 split:

| Model | F1 | ROC-AUC | Fit time | Predict/row | Model size |
|---|---|---|---|---|---|
| Baseline: CO2 threshold rule | 0.345 | 0.568 | — | — | — |
| Random Forest (env + time features) | 0.763 | 0.736 | 716.0 ms | 91.2 µs | 4.2 MB |
| **Logistic Regression (recommended final model)** | 0.756 | 0.708 | 11.3 ms | 1.1 µs | 1.8 KB |

Random Forest edges out Logistic Regression on raw accuracy, but that edge
is within the noise observed across cross-validation folds, while Logistic
Regression is ~60-80x cheaper to run and ~2,300x smaller — see
`docs/optimisation.md` for the full trade-off analysis and why Logistic
Regression, not the more "impressive" model, is the recommended design.

See `docs/model_results.md` for full validation results (including a check
for whether the model is just learning the room's schedule rather than
reading the air — it isn't, mostly).

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
   python src/optimisation.py             # runtime/memory/accuracy comparison + tuning
   python src/make_figures.py             # generates reports/figures/*.png
   ```

## Tests

```bash
pytest tests/ -v --junitxml=reports/test_results.xml
python scripts/junit_to_html.py   # renders reports/test_report.html
```

9/9 tests passing. Tests run against small synthetic fixtures
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
in `docs/ai_use.md` alongside the final report.

## Remaining work

- [ ] Written report (5-10 pages, pulling together `docs/data_audit.md`,
      `docs/model_results.md`, `docs/optimisation.md`, `docs/inputs_outputs.md`)
- [ ] `docs/ai_use.md` — full AI-use disclosure and critical reflection
- [ ] Presentation slides
