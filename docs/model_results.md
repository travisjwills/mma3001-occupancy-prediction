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
stability over time rather than relying on one lucky split), run for all four
models:

| Model | mean F1 | std F1 | F1 in folds 0 to 4 |
|---|---|---|---|
| Baseline (CO2 threshold) | 0.519 | 0.258 | 0.705, 0.696, 0.768, 0.307, 0.118 |
| Logistic Regression | 0.536 | 0.216 | 0.825, 0.652, 0.619, 0.219, 0.366 |
| Random Forest | 0.698 | 0.197 | 0.810, 0.760, 0.806, 0.305, 0.807 |
| Gradient Boosting | 0.702 | 0.180 | 0.806, 0.764, 0.809, 0.343, 0.786 |

Random Forest and Gradient Boosting are the most stable and are
indistinguishable from each other (0.698 vs 0.702). Logistic Regression looks
as good as them on the single held-out test period (F1 0.756) but is clearly
weaker across time: its mean fold F1 is about 0.16 lower, and it falls to
0.62 and 0.37 in folds 2 and 4 where the tree models score about 0.8. A
plausible, untested reason is that a linear model uses hour of day as a single
number, so it cannot learn a daytime-shaped occupancy pattern.

**Fold 3 is weak for every model** (F1 0.22 to 0.34). `src/cv_fold_diagnostics.py`
shows why: fold 3 covers 24-30 May (Melbourne time), when the room was
occupied about 75% of the time on Saturday 25 and Sunday 26 May (earlier
Saturdays were under 20%) and almost empty on Tuesday 28 May (1%). The usual
weekday/weekend pattern was reversed, so models trained on earlier weeks were
caught out. Why the room's use changed is not known.

The large fold-to-fold variance (std about 0.18 to 0.26 for every model) is
itself a finding: **5 weeks of training data is not enough for a stable
estimate of real-world performance**, and this should be stated plainly as a
limitation rather than reporting only the single best test-set number.

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

## Extension: predicting headcount (Decision Tree Regression)

`src/headcount_regression.py` predicts the number of people (0 to 22) rather
than occupied/not occupied, on the same chronological split and the same five
time-ordered folds. The tree's maximum depth was chosen by cross-validation on
the training data only (lowest mean CV MAE among depths 2, 3, 4, 5, 6, 8, 10
and unlimited, at least 10 intervals per leaf): depth 6, with mean CV MAE
1.36. Depths 3 to unlimited all scored between 1.36 and 1.41, so the choice
among them is within the fold noise (std about 0.6).

| Model | Test MAE | Test RMSE | Test R^2 | Within 1 person | Mean CV MAE (std) |
|---|---|---|---|---|---|
| Mean baseline | 1.231 | 1.364 | -1.418 | 48.5% | 2.267 (0.511) |
| Linear Regression | 0.828 | 1.076 | -0.505 | 69.9% | 2.031 (0.641) |
| Decision Tree Regression (depth 6) | 0.805 | 1.516 | -1.987 | 85.1% | 1.360 (0.641) |

**Interpretation:**
- Across the five training folds the tree is clearly best: its mean absolute
  error is 1.36 people, against 2.27 for the mean baseline and 2.03 for Linear
  Regression.
- On the test period it again has the lowest MAE and the most predictions
  within one person (85%), but it sometimes predicts five or six people when
  the room is nearly empty, which inflates RMSE (figure 7).
- The test period was unusually quiet: mean 0.72 people and a maximum of 8,
  against a training mean of 1.77 and maximum of 22, with only 0.2% of test
  intervals at 5 or more people (14.9% in training). With so little variation,
  R^2 is negative for every model, including the mean baseline, so MAE is the
  more meaningful test measure here.
- Counting people is clearly harder than detecting presence: the same inputs
  give a useful yes/no classifier but only a rough headcount estimate.



- **Does establish:** on this one room, over this ~6-week window, a
  multivariate environmental model beats a single-variable CO2 threshold by
  a clear margin (F1 0.76 vs 0.35), and the improvement is consistent across
  most (not all) time-series CV folds.
- **Does not establish:** generalisation to other rooms (only 1 of 5 zones
  was usable — see data audit), generalisation beyond this ~6-week window,
  or that the model is using environmental signal rather than a learned
  schedule (pending the sensitivity check above).
