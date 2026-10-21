"""
Generate the figures used in the written report and presentation slides.

Reads from data/processed/room_dataset.csv and the reports/*.csv files
produced by train_model.py, sensitivity_env_only.py, optimisation.py and
match_sensors_to_zones.py. Writes PNGs (300 dpi, report/slide-ready) to
reports/figures/.

Palette follows the project's data-viz conventions: blue (#2a78d6) as the
primary single-series/sequential hue, with orange/aqua/yellow as the fixed
categorical order for multi-model comparisons, muted ink for axes/gridlines,
and a consistent light chart surface throughout so every figure matches.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "room_dataset.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- palette (see dataviz skill reference) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRID,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig1_co2_vs_headcount():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    window = df[(df["timestamp"] >= "2024-05-06") & (df["timestamp"] < "2024-05-13")]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(window["timestamp"], window["Carbon dioxide"], color=BLUE, linewidth=1.6, label="CO2 (ppm)")
    ax1.set_ylabel("CO2 (ppm)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(axis="y", linewidth=0.6)

    ax2 = ax1.twinx()
    ax2.fill_between(window["timestamp"], window["headcount"], color=ORANGE, alpha=0.25, step="post")
    ax2.plot(window["timestamp"], window["headcount"], color=ORANGE, linewidth=1.2, drawstyle="steps-post",
              label="Headcount")
    ax2.set_ylabel("Headcount (people)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.spines["top"].set_visible(False)

    ax1.set_title("CO2 tracks occupancy: one week, sensor 869 / zone 3e5a14df", fontsize=12, loc="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_co2_vs_headcount.png", dpi=300)
    plt.close(fig)


def fig2_sensor_zone_correlation_heatmap():
    corr = pd.read_csv(REPORTS_DIR / "sensor_zone_correlation_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = corr.to_numpy(dtype=float)
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", BLUE, "#0d366b"])
    im = ax.imshow(np.nan_to_num(data, nan=-1), cmap=cmap, vmin=-0.2, vmax=0.65)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "n/a" if np.isnan(val) else f"{val:.2f}"
            color = "white" if (not np.isnan(val) and val > 0.3) else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels([c[:8] + "…" for c in corr.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Sensor × zone CO2/headcount correlation\n(3 of 5 sensors unusable — broken timestamps)",
                  fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_sensor_zone_correlation.png", dpi=300)
    plt.close(fig)


def fig3_model_comparison_bar():
    base = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    opt = pd.read_csv(REPORTS_DIR / "optimisation_comparison.csv")

    rows = []
    for _, r in base.iterrows():
        rows.append({"model": r["model"], "f1": r["f1"], "roc_auc": r["roc_auc"]})
    for _, r in opt.iterrows():
        if r["model"] == "Random Forest (default)":
            continue  # already in base comparison (same config)
        rows.append({"model": r["model"], "f1": r["f1"], "roc_auc": r["roc_auc"]})

    df = pd.DataFrame(rows).drop_duplicates(subset="model").sort_values("f1")
    labels = [m.replace("Baseline: ", "Baseline\n(") + (")" if "Baseline" in m else "") for m in df["model"]]
    colors = [RED if "Baseline" in m else (BLUE if "Logistic" in m else (AQUA if "Random Forest" in m else YELLOW))
              for m in df["model"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(df))
    bars = ax.barh(y, df["f1"], color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("F1 score (held-out test set)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linewidth=0.6)
    for b, val in zip(bars, df["f1"]):
        ax.text(val + 0.015, b.get_y() + b.get_height() / 2, f"{val:.3f}", va="center", fontsize=9, color=INK)
    ax.set_title("Model comparison: F1 on held-out test set", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_model_comparison_f1.png", dpi=300)
    plt.close(fig)


def fig4_feature_importance():
    imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv", index_col=0).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [ORANGE if idx == "hour" else BLUE for idx in imp.index]
    ax.barh(imp.index, imp["importance"], color=colors, height=0.6)
    ax.set_xlabel("Random Forest feature importance")
    ax.grid(axis="x", linewidth=0.6)
    ax.set_title("Feature importance — hour-of-day leads,\nbut env. variables matter too",
                  fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_feature_importance.png", dpi=300)
    plt.close(fig)


def fig5_cv_stability():
    cv = pd.read_csv(REPORTS_DIR / "cv_results.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.2
    x = cv["fold"].to_numpy()
    series = [
        ("baseline_f1", "Baseline (CO2 threshold)", RED),
        ("logreg_f1", "Logistic Regression", BLUE),
        ("gb_f1", "Gradient Boosting", YELLOW),
        ("rf_f1", "Random Forest", AQUA),
    ]
    for i, (col, label, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, cv[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linewidth=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
    ax.set_title("5-fold TimeSeriesSplit CV: performance is unstable across folds", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_cv_stability.png", dpi=300)
    plt.close(fig)


def fig6_accuracy_vs_cost():
    """Five-panel dot plot: test F1 is similar across the three models, but
    stability across time and cost are not.

    Dots (not bars) are used on the log-scale panels because a bar's length is
    not meaningful on a log axis. Panels share the model order and colours of
    figure 3. The cross-validation panel is the mean F1 over the five
    time-ordered folds from ``cv_results.csv``.
    """
    opt = pd.read_csv(REPORTS_DIR / "optimisation_comparison.csv")
    cv = pd.read_csv(REPORTS_DIR / "cv_results.csv")
    order = ["Logistic Regression", "Gradient Boosting", "Random Forest (default)"]
    opt = opt.set_index("model").loc[order].reset_index()
    labels = ["Logistic Regression", "Gradient Boosting", "Random Forest"]
    colors = [BLUE, YELLOW, AQUA]
    cv_means = np.array([cv["logreg_f1"].mean(), cv["gb_f1"].mean(), cv["rf_f1"].mean()])

    panels = [
        ("F1 on test set (higher is better)", opt["f1"].to_numpy(), False, "{:.3f}", (0, 1)),
        ("Mean F1 across 5 CV folds (higher is better)", cv_means, False, "{:.3f}", (0, 1)),
        ("Training time, ms (log scale)", opt["fit_time_s"].to_numpy() * 1000, True, "{:,.1f}", None),
        ("Prediction time per row, µs (log scale)", opt["predict_time_ms_per_row"].to_numpy() * 1000, True, "{:,.1f}", None),
        ("Saved model size, KB (log scale)", opt["model_size_kb"].to_numpy(), True, "{:,.1f}", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharey=True)
    axes.ravel()[-1].set_visible(False)
    y = np.arange(len(order))[::-1]  # first model at the top
    for ax, (title, vals, logx, fmt, xlim) in zip(axes.ravel(), panels):
        ax.scatter(vals, y, s=140, color=colors, zorder=3)
        for v, yy in zip(vals, y):
            ax.annotate(fmt.format(v), (v, yy), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color=INK)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(vals.min() / 4, vals.max() * 4)
        else:
            ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="x", linewidth=0.6, alpha=0.6)
    fig.suptitle("Similar test accuracy, but Logistic Regression is less stable over time", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "06_accuracy_vs_cost.png", dpi=300)
    plt.close(fig)


def main():
    fig1_co2_vs_headcount()
    fig2_sensor_zone_correlation_heatmap()
    fig3_model_comparison_bar()
    fig4_feature_importance()
    fig5_cv_stability()
    fig6_accuracy_vs_cost()
    print("Saved figures to", FIG_DIR)
    for f in sorted(FIG_DIR.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
