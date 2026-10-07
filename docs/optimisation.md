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

### Key finding: Logistic Regression is nearly as accurate, for a fraction of the cost

Logistic Regression reaches F1 = 0.756, within **1 percentage point** of the
Random Forest's 0.763 — but trains **~63x faster** (11ms vs 716ms), predicts
**~84x faster per row** (1.1µs vs 91.2µs), and the saved model is **~2,300x
smaller** (1.8KB vs 4.2MB). Random Forest does win more clearly on ROC-AUC
(0.736 vs 0.708), meaning it ranks borderline cases somewhat better, which
matters if the deployed system needs a tunable decision threshold rather
than a fixed one.

**Engineering interpretation:** for a real building-monitoring deployment —
where this model would run continuously on cheap edge hardware or be
retrained often as more data arrives — Logistic Regression's near-identical
accuracy at a fraction of the compute/memory footprint makes it the more
defensible production choice, despite Random Forest being the headline
"improved model" in the earlier accuracy-only comparison. This is the kind
of trade-off the brief asks for explicitly: the best model by raw accuracy
is not automatically the right final design once cost is considered.

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

Given the full picture — accuracy, robustness under CV, runtime, memory,
and the tuning result above — **Logistic Regression is recommended as the
final design**, not Random Forest. The 1-point F1 gap from Random Forest is
smaller than the fold-to-fold noise already observed in cross-validation,
so it is not a reliably real difference, while the runtime/memory/size
advantages are large, consistent, and would matter in a real low-cost
building-sensor deployment. Random Forest remains useful in this project as
the richer model used for the feature-importance and schedule-vs-environment
sensitivity analysis (`docs/model_results.md`), where its non-linearity and
built-in feature-importance output were genuinely useful tools for
*understanding* the problem, even though it isn't the model recommended for
*deployment*.
