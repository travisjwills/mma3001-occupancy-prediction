# Optimisation & Performance Analysis

Brief requirement #6: analyse (and where appropriate improve) accuracy,
runtime, memory, robustness, and trade-offs between performance measures,
and justify the final design against the baseline.

All numbers from `src/optimisation.py`, run on the single-room dataset
(4,099 rows, 13 features, same chronological 80/20 split as `train_model.py`).
Fit times are best-of-5 repeats; predict times are best-of-50 repeats
divided by test-set size, to remove OS scheduling noise.

## Model comparison

| Model | F1 | ROC-AUC | Fit time | Predict (per row) | Peak train memory | Model size |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.756 | 0.708 | 11.3 ms | 1.09 µs | 0.79 MB | 1.8 KB |
| Random Forest (default) | 0.763 | 0.736 | 716.0 ms | 91.2 µs | 1.07 MB | 4,175.3 KB |
| Gradient Boosting | 0.761 | 0.724 | 432.3 ms | 1.54 µs | 0.50 MB | 135.8 KB |

### Key finding: on one test split the models look alike, but cross-validation separates them

On the single held-out test period the three models are within 1 point of F1
(0.756 to 0.763), so accuracy there does not decide between them. Cost does:
relative to the Random Forest, Logistic Regression trains ~63x faster
(11 ms vs 716 ms), predicts ~84x faster per row (1.1 µs vs 91.2 µs) and is
~2,300x smaller (1.8 KB vs 4.2 MB). Gradient Boosting sits in between: ~1.7x
faster to train than the Random Forest, ~59x faster per prediction (1.5 µs)
and ~31x smaller (136 KB).

But a single 9-day test period is a weak basis for choosing a model. The
5-fold time-series cross-validation in `docs/model_results.md` (mean F1:
Gradient Boosting 0.702, Random Forest 0.698, Logistic Regression 0.536) shows
that Logistic Regression is clearly less reliable over time (it drops to 0.37
in fold 4, where the tree models score about 0.8), even though it matched them
on the test period. An earlier version of this analysis recommended Logistic
Regression on the test-split result alone; the cross-validation check
overturned that recommendation.

## Hyperparameter tuning: a cautionary result, not a success story

`RandomizedSearchCV` (20 candidate configurations x 5 `TimeSeriesSplit` folds
= 100 fits, 39.7s total) was used to tune the Random Forest's
`n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`.

| | F1 (held-out test) | ROC-AUC (held-out test) |
|---|---|---|
| Default RF | 0.763 | 0.736 |
| Tuned RF | **0.736** | **0.723** |

**Tuning made held-out performance worse, not better.** The search picked
parameters (`max_depth=None`, i.e. fully-grown trees) that scored well on
the training-set CV folds but generalised slightly worse to the untouched
test period. Combined with the large fold-to-fold F1 variance already noted
in `docs/model_results.md` (std ≈ 0.2 across folds), this points to the same
root cause: **the training window (~5 weeks) is short enough that
cross-validation scores are themselves noisy**, so optimising against them
can chase that noise rather than genuine generalisation. The honest
conclusion is to keep the untuned, moderately-regularised default
configuration (`max_depth=8`, `min_samples_leaf=5`) rather than the
CV-selected one, and to flag "more training data would likely make tuning
worthwhile" as a limitation/future-work item instead of reporting the tuned
number as an improvement it isn't.

## Rough computational cost estimate (Random Forest)

The Random Forest's cost isn't a clean closed-form FLOP count the way a
linear model's is (`#features` multiply-adds for Logistic Regression), so a
loose estimate was used: training touches up to
`n_trees x n_samples x n_features x depth` ≈ 300 x 3,279 x 13 x 8 ≈
**102 million** comparisons; a single prediction touches
`n_trees x depth` = 300 x 8 = **2,400** comparisons (one root-to-leaf walk
per tree). This matches the observed ~90µs/row prediction time and explains
why Random Forest, despite being "only" a few hundred shallow trees, is
~2–3 orders of magnitude more expensive per prediction than the linear
models.

## Final design decision

**Gradient Boosting is recommended as the final design.** It matches the
Random Forest on accuracy (test F1 0.761 vs 0.763; mean CV F1 0.702 vs 0.698,
with the smallest fold-to-fold spread of the four models, std 0.180), but is
~59x faster per prediction (1.5 µs vs 91.2 µs) and ~31x smaller (136 KB vs
4.2 MB), so on accuracy-plus-cost it dominates the Random Forest. It costs
more than Logistic Regression (~38x the training time, ~76x the model size,
1.4x the prediction time), but those costs are small in absolute terms
(432 ms to train, 136 KB to store), while Logistic Regression's mean F1 in
cross-validation is about 0.17 lower than Gradient Boosting's. Training time is not a constraint here
because the model would be retrained only occasionally.

**Logistic Regression** remains the right choice only if a few kilobytes of
memory were a hard constraint and the single-test-split result were the whole
story, which the cross-validation shows it is not. **Random Forest** remains
useful in this project as the richer model used for the feature-importance and
schedule-vs-environment sensitivity analysis (`docs/model_results.md`), even
though it is not the model recommended for deployment.

Limits of this conclusion: Gradient Boosting was used at library defaults and
not tuned; the CV gap between Gradient Boosting and Random Forest (0.004) is far
smaller than the fold-to-fold noise, so the choice between them rests on cost,
not accuracy; and timings come from one machine, so only the relative ordering
should be relied on.
