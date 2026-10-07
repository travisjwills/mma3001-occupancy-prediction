# Model Results — Single-Room Occupancy Classification

**Room:** env sensor `6012002000869` / occupancy zone `3e5a14df-...`
(see `docs/data_audit.md` for how this pairing was established).

**Task:** binary classification — is the room occupied (`headcount > 0`)?

**Data:** 4,099 fifteen-minute samples, 2024-04-30 to 2024-06-13 (after
joining sensor and occupancy data and dropping rows with missing values —
see `src/build_dataset.py`). Occupied rate: 48.2% (reasonably balanced).

**Split:** chronological 80/20 (train on the earlier ~5 weeks, test on the
final ~9 days) — not a random split, because adjacent 15-min readings are
highly autocorrelated and a random split would leak information between
train and test, inflating reported accuracy.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline: CO2 > 470 ppm (threshold fit on train) | 0.504 | 0.578 | 0.245 | 0.345 | 0.568 |
| Random Forest (all env + time features) | 0.724 | 0.703 | 0.833 | **0.763** | 0.736 |

5-fold `TimeSeriesSplit` cross-validation on the training set (checks
stability over time rather than relying on one lucky split):

| | mean F1 | std F1 |
|---|---|---|
| Baseline | 0.519 | 0.258 |
| Random Forest | 0.698 | 0.197 |

The Random Forest outperforms the single-variable baseline on every fold but
one (fold 3, where both models collapse to F1≈0.3 — worth investigating
further, e.g. a public holiday or unusual occupancy pattern in that week).
Both models' large fold-to-fold variance (std ≈ 0.2–0.26) is itself a
finding: **5 weeks of training data is not enough for a stable estimate of
real-world performance**, and this should be stated plainly as a limitation
rather than reporting only the single best test-set number.

## Feature importance (Random Forest)

| Feature | Importance |
|---|---|
| hour | 0.290 |
| Temperature | 0.180 |
| Carbon dioxide | 0.159 |
| Formaldehyde | 0.091 |
| Pressure | 0.062 |
| Humidity | 0.046 |
| LVOC | 0.044 |
| dayofweek | 0.038 |
| PM1 | 0.030 |
| PM2.5 | 0.021 |
| is_weekend | 0.015 |
| PM4 | 0.014 |
| PM10 | 0.011 |

## Important caveat: the model may be learning the schedule, not the air

`hour` (time of day) is the single most important feature — more important
than CO2, the variable the whole project is physically motivated by. This
means a meaningful part of the Random Forest's accuracy likely comes from
learning "this room tends to be occupied 9am–5pm on weekdays" rather than
purely from environmental sensing. That's not necessarily wrong — occupancy
*is* genuinely time-patterned, and a real deployed system could legitimately
use time-of-day as a feature — but it does **undercut the original framing**
of "can environmental sensors alone substitute for an occupancy sensor,"
since a schedule-based heuristic (with no sensors at all) might already get
close to this performance.

**Sensitivity check performed** (`src/sensitivity_env_only.py`): refit the
Random Forest with `hour`/`dayofweek`/`is_weekend` removed entirely.

| Model | F1 | ROC-AUC |
|---|---|---|
| Full (env + time features) | 0.763 | 0.736 |
| Env-only (no time features) | 0.738 | 0.667 |

F1 barely moves (0.763 -> 0.738, a 3% relative drop), so the environmental
variables alone retain almost all of the classification performance — the
model is **not** primarily a disguised schedule-lookup. ROC-AUC drops more
noticeably (0.736 -> 0.667), meaning time-of-day does meaningfully help rank
borderline cases (useful for setting a probability threshold in practice),
but the core result holds: **environmental sensing alone is doing most of
the work**, which supports the project's original premise.

## What this validation does and does not establish

- **Does establish:** on this one room, over this ~6-week window, a
  multivariate environmental model beats a single-variable CO2 threshold by
  a clear margin (F1 0.76 vs 0.35), and the improvement is consistent across
  most (not all) time-series CV folds.
- **Does not establish:** generalisation to other rooms (only 1 of 5 zones
  was usable — see data audit), generalisation beyond this ~6-week window,
  or that the model is using environmental signal rather than a learned
  schedule (pending the sensitivity check above).
