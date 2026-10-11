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

## Key results (so far)

Single-room occupancy classification (sensor 869 / zone `3e5a14df-...`,
4,099 samples, chronological 80/20 split):

| Model | F1 | ROC-AUC |
|---|---|---|
| Baseline: CO2 threshold rule | 0.345 | 0.568 |
| Random Forest (env + time features) | 0.763 | 0.736 |
| Random Forest (env features only) | 0.738 | 0.667 |

See `docs/model_results.md` for full results, cross-validation stability,
and an honest discussion of what the model is and isn't actually learning.

## Next steps

      gradient boosting) for the "credible alternatives" comparison
- [ ] Figures: time-series plot of CO2 vs headcount, feature importance chart,
      confusion matrices
- [ ] Headcount regression as a secondary framing (data already supports it)
- [ ] Write up full report (engineering context, limitations, AI use disclosure)

## Optimisation & final design choice

After comparing accuracy, runtime, memory, and hyperparameter-tuning
behaviour (`docs/optimisation.md`), **Logistic Regression is the recommended
final model**, not the Random Forest used for interpretation/feature
analysis above:

| Model | F1 | ROC-AUC | Fit time | Predict/row | Model size |
|---|---|---|---|---|---|
| Logistic Regression | 0.756 | 0.708 | 11.3 ms | 1.1 µs | 1.8 KB |
| Random Forest | 0.763 | 0.736 | 716.0 ms | 91.2 µs | 4.2 MB |

Random Forest's accuracy edge is within cross-validation noise; Logistic
Regression's efficiency advantage (60-80x faster, ~2300x smaller) is large
and consistent, which matters more for a real low-cost sensor deployment.
Hyperparameter tuning (RandomizedSearchCV) was also tried and made held-out
performance *worse* — documented as a limitation of the ~5-week training
window rather than hidden.

## Figures

Report/slide-ready figures (300 dpi PNG) are in `reports/figures/`, generated
by `src/make_figures.py`:

1. `01_co2_vs_headcount.png` — one week of CO2 vs headcount, showing the
   physical relationship the whole project rests on
2. `02_sensor_zone_correlation.png` — the 5×5 sensor/zone correlation matrix
   from the data audit (visualises why only 1 pair was usable)
3. `03_model_comparison_f1.png` — baseline vs all 3 ML models, F1 on test set
4. `04_feature_importance.png` — Random Forest feature importances
5. `05_cv_stability.png` — per-fold F1 across 5-fold TimeSeriesSplit CV
6. `06_accuracy_vs_cost.png` — accuracy vs. inference-time trade-off
