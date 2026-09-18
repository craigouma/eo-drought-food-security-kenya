"""Figures for the paper.

All figures use the portfolio palette: a single-hue navy sequential ramp for
maps and index severity, navy and blue for series, and no rainbow colormaps.
"""

import json

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from build_dataset import CLASS_NAMES, FEATURES
from config import BLUE, BLUE_LT, BORDER, MUTED, NAVY, PAPER, PROCESSED, STUDY_COUNTIES, TEXT
from fetch_boundaries import load_counties
from train_model import MODEL_DIR

RESULTS_PATH = MODEL_DIR / "model_results.json"
SHAP_SUMMARY_PATH = MODEL_DIR / "shap_summary.json"
SHAP_VALUES_PATH = MODEL_DIR / "shap_values_lead{lead}.npz"

NAVY_RAMP = LinearSegmentedColormap.from_list("navy_ramp", ["#F4F8FC", BLUE_LT, "#9EC1E8", "#3C6FA5", NAVY])
DRY_WET = LinearSegmentedColormap.from_list("dry_wet", ["#8C5A2B", "#D9C6A5", "#F4F8FC", "#8FB4D9", NAVY])

PHASE_SHADE = {1: "#FFFFFF", 2: "#E3EDF7", 3: "#C3D6E8"}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
            "font.size": 9,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.grid": True,
            "grid.color": "#E6EDF4",
            "grid.linewidth": 0.6,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def county_map() -> None:
    """Kenya's counties, shaded by long-run rainfall, study counties outlined."""
    counties = load_counties()
    chirps = pd.read_csv(PROCESSED / "chirps_monthly_county.csv", parse_dates=["date"])
    baseline = chirps[(chirps["year"] >= 1981) & (chirps["year"] <= 2010)]
    annual = baseline.groupby(["county", "year"], as_index=False)["rainfall_mm"].sum()
    mean_annual = annual.groupby("county", as_index=False)["rainfall_mm"].mean()
    merged = counties.merge(mean_annual, on="county", how="left")

    figure, axis = plt.subplots(figsize=(6.2, 6.6))
    merged.plot(
        column="rainfall_mm",
        cmap=NAVY_RAMP,
        linewidth=0.4,
        edgecolor="#FFFFFF",
        ax=axis,
        legend=True,
        legend_kwds={
            "label": "Mean annual rainfall, 1981-2010 (mm)",
            "orientation": "horizontal",
            "shrink": 0.62,
            "pad": 0.02,
            "aspect": 30,
        },
    )
    study = merged[merged["county"].isin(STUDY_COUNTIES)]
    study.boundary.plot(ax=axis, color=NAVY, linewidth=1.5)

    for _, row in study.iterrows():
        point = row["geometry"].representative_point()
        axis.annotate(
            row["county"],
            xy=(point.x, point.y),
            ha="center",
            va="center",
            fontsize=8.5,
            color="#FFFFFF" if row["rainfall_mm"] > 450 else TEXT,
            weight="bold",
        )

    axis.set_axis_off()
    axis.set_title(
        "The five arid and semi-arid study counties",
        loc="left",
        color=NAVY,
        fontsize=11,
    )
    figure.savefig(PAPER / "fig_county_map.png")
    plt.close(figure)


def rainfall_vci_timeseries() -> None:
    """SPI-3 and VCI for the study counties, with classified phases shaded."""
    spi = pd.read_csv(PROCESSED / "spi_county.csv", parse_dates=["date"])
    vci = pd.read_csv(PROCESSED / "vci_county_monthly.csv", parse_dates=["date"])
    ipc = pd.read_csv(PROCESSED / "ipc_county_phase.csv", parse_dates=["reporting_date"])

    start = pd.Timestamp("2011-01-01")
    figure, axes = plt.subplots(
        len(STUDY_COUNTIES), 1, figsize=(7.4, 9.2), sharex=True
    )

    for axis, county in zip(axes, STUDY_COUNTIES):
        county_spi = spi[(spi["county"] == county) & (spi["date"] >= start)]
        county_vci = vci[(vci["county"] == county) & (vci["date"] >= start)]
        county_ipc = ipc[
            (ipc["county"] == county) & (ipc["reporting_date"] >= start)
        ].sort_values("reporting_date")

        # Shade the classified periods by phase, so the outcome the model is
        # predicting is visible behind the indicators it predicts it from.
        dates = list(county_ipc["reporting_date"]) + [pd.Timestamp("2026-09-01")]
        for (_, row), end in zip(county_ipc.iterrows(), dates[1:]):
            phase = min(int(row["ipc_phase"]), 3)
            if phase > 1:
                axis.axvspan(
                    row["reporting_date"], end, color=PHASE_SHADE[phase], zorder=0
                )

        axis.axhline(0, color=BORDER, linewidth=0.8)
        axis.plot(
            county_spi["date"], county_spi["spi3"], color=NAVY, linewidth=1.0,
            label="SPI-3 (rainfall)",
        )
        axis.set_ylim(-3.2, 3.2)
        axis.set_ylabel("SPI-3", color=NAVY, fontsize=8.5)

        twin = axis.twinx()
        twin.plot(
            county_vci["date"], county_vci["vci"], color=BLUE, linewidth=1.0,
            alpha=0.85, label="VCI (vegetation)",
        )
        twin.axhline(35, color=BLUE, linewidth=0.7, linestyle=(0, (4, 3)), alpha=0.7)
        twin.set_ylim(-20, 160)
        twin.set_ylabel("VCI", color=BLUE, fontsize=8.5)
        twin.grid(False)

        axis.set_title(county, loc="left", color=NAVY, fontsize=10)

    handles = [
        plt.Line2D([], [], color=NAVY, linewidth=1.2, label="SPI-3, rainfall (left axis)"),
        plt.Line2D([], [], color=BLUE, linewidth=1.2, label="VCI, vegetation (right axis)"),
        plt.Line2D(
            [], [], color=BLUE, linewidth=0.8, linestyle=(0, (4, 3)),
            label="VCI 35, conventional drought threshold",
        ),
        Patch(facecolor=PHASE_SHADE[2], label="Classified Stressed"),
        Patch(facecolor=PHASE_SHADE[3], label="Classified Crisis or worse"),
    ]
    axes[-1].legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
        fontsize=8,
    )
    figure.suptitle(
        "Rainfall and vegetation indicators against classified food insecurity phases",
        x=0.02, ha="left", color=NAVY, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.98))
    figure.savefig(PAPER / "fig_rainfall_vci_timeseries.png")
    plt.close(figure)


def classification_performance() -> None:
    """Confusion matrix, alert trade-off, and performance by lead time."""
    results = json.loads(RESULTS_PATH.read_text())
    walk = results["walkforward_eo_only_lead_1"]

    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))

    matrix = np.array(walk["model"]["confusion_matrix"], dtype=float)
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    axis = axes[0]
    image = axis.imshow(normalised, cmap=NAVY_RAMP, vmin=0, vmax=1)
    labels = [CLASS_NAMES[c] for c in (1, 2, 3)]
    axis.set_xticks(range(3), labels, fontsize=8, rotation=20, ha="right")
    axis.set_yticks(range(3), labels, fontsize=8)
    for i in range(3):
        for j in range(3):
            axis.text(
                j, i, f"{normalised[i, j]:.2f}\n({int(matrix[i, j])})",
                ha="center", va="center", fontsize=8,
                color="#FFFFFF" if normalised[i, j] > 0.55 else TEXT,
            )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Classified by FEWS NET")
    axis.set_title("Confusion matrix, one month lead", loc="left", color=NAVY)
    axis.grid(False)
    figure.colorbar(image, ax=axis, shrink=0.8, label="Share of row")

    axis = axes[1]
    curve = pd.DataFrame(walk["crisis_alert_curve"]).dropna()
    axis.plot(curve["threshold"], curve["recall"], color=NAVY, marker="o", markersize=3,
              label="Recall: share of Crisis periods flagged")
    axis.plot(curve["threshold"], curve["precision"], color=BLUE, marker="s", markersize=3,
              label="Precision: share of flags that were Crisis")
    axis.set_xlabel("Alert threshold on modelled probability of Crisis")
    axis.set_ylim(0, 1.02)
    axis.set_title("Alert trade-off", loc="left", color=NAVY)
    axis.legend(fontsize=7.5, loc="lower center")

    axis = axes[2]
    leads = [1, 3, 6]
    detection = [
        results[f"walkforward_eo_only_lead_{lead}"]["transitions"][
            "model_direction_on_worsened"
        ]
        for lead in leads
    ]
    intervals = [
        results[f"walkforward_eo_only_lead_{lead}"]["transitions"][
            "model_direction_on_worsened_ci"
        ]
        for lead in leads
    ]
    lower = [d - ci[0] for d, ci in zip(detection, intervals)]
    upper = [ci[1] - d for d, ci in zip(detection, intervals)]
    axis.errorbar(
        leads, detection, yerr=[lower, upper], color=NAVY, marker="o", capsize=4,
        linewidth=1.2,
    )
    axis.axhline(0, color=BLUE, linestyle=(0, (4, 3)), linewidth=1.0)
    axis.text(
        3.1, 0.03, "Persistence catches none of these by construction",
        fontsize=7.5, color=BLUE,
    )
    axis.set_xticks(leads)
    axis.set_xlabel("Lead time (months)")
    axis.set_ylabel("Share of deteriorations anticipated")
    axis.set_ylim(-0.05, 1.0)
    axis.set_title("Deteriorations caught by lead time", loc="left", color=NAVY)

    figure.tight_layout()
    figure.savefig(PAPER / "fig_classification_performance.png")
    plt.close(figure)


def load_shap(lead: int = 1):
    payload = np.load(str(SHAP_VALUES_PATH).format(lead=lead), allow_pickle=True)
    return payload


def shap_bar() -> None:
    """Global importance, split between the Crisis class and overall."""
    summary = json.loads(SHAP_SUMMARY_PATH.read_text())["lead_1"]
    frame = pd.DataFrame(summary["global_importance"]).sort_values(
        "mean_abs_shap_crisis"
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    positions = np.arange(len(frame))
    axis.barh(positions + 0.2, frame["mean_abs_shap_crisis"], height=0.4, color=NAVY,
              label="Contribution to the Crisis class")
    axis.barh(positions - 0.2, frame["mean_abs_shap"], height=0.4, color="#9EC1E8",
              label="Contribution across all three classes")
    axis.set_yticks(positions, frame["feature"], fontsize=8.5)
    axis.set_xlabel("Mean absolute SHAP value (log-odds)")
    axis.set_title(
        "What the model reads, one month lead", loc="left", color=NAVY
    )
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(PAPER / "fig_shap_bar.png")
    plt.close(figure)


def shap_beeswarm() -> None:
    """Beeswarm for the Crisis class, drawn directly to keep the palette."""
    payload = load_shap(1)
    values = payload["values"][:, :, 2]
    feature_values = payload["feature_values"]
    features = list(payload["features"])

    order = np.argsort(np.abs(values).mean(axis=0))
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    rng = np.random.default_rng(0)

    for position, index in enumerate(order):
        x = values[:, index]
        colour_source = feature_values[:, index].astype(float)
        finite = np.isfinite(colour_source)
        jitter = rng.uniform(-0.18, 0.18, size=x.shape)
        if finite.sum() > 1:
            # Numeric feature: shade each point by where its value sits.
            low, high = np.nanpercentile(colour_source[finite], [5, 95])
            normalised = np.clip((colour_source - low) / (high - low + 1e-9), 0, 1)
            axis.scatter(
                x, position + jitter, c=normalised, cmap=NAVY_RAMP, s=6, alpha=0.75,
                linewidths=0, vmin=0, vmax=1,
            )
        else:
            # Categorical feature: there is no low-to-high order to shade by.
            axis.scatter(
                x, position + jitter, color=MUTED, s=6, alpha=0.5, linewidths=0
            )

    axis.axvline(0, color=BORDER, linewidth=0.9)
    axis.set_yticks(range(len(order)), [features[i] for i in order], fontsize=8.5)
    axis.set_xlabel("SHAP value for the Crisis class (log-odds)")
    axis.set_title(
        "Per-observation contributions to a Crisis call", loc="left", color=NAVY
    )
    mappable = plt.cm.ScalarMappable(cmap=NAVY_RAMP)
    mappable.set_array([])
    bar = figure.colorbar(mappable, ax=axis, shrink=0.75, pad=0.02)
    bar.set_ticks([0, 1])
    bar.set_ticklabels(["low", "high"])
    bar.set_label("Feature value", fontsize=8)
    figure.tight_layout()
    figure.savefig(PAPER / "fig_shap_beeswarm.png")
    plt.close(figure)


def shap_dependence() -> None:
    """Dependence plots for the leading time-varying drivers."""
    payload = load_shap(1)
    summary = json.loads(SHAP_SUMMARY_PATH.read_text())["lead_1"]
    values = payload["values"][:, :, 2]
    feature_values = payload["feature_values"]
    features = list(payload["features"])

    chosen = ["vci_recent", "spi6_recent", "vci_prior2"]
    figure, axes = plt.subplots(1, len(chosen), figsize=(10.4, 3.4))

    for axis, name in zip(axes, chosen):
        index = features.index(name)
        x = feature_values[:, index].astype(float)
        y = values[:, index]
        axis.axhline(0, color=BORDER, linewidth=0.9)
        axis.scatter(x, y, s=8, color=NAVY, alpha=0.45, linewidths=0)

        threshold = summary["crisis_thresholds"].get(name, {})
        curve = threshold.get("curve")
        if curve:
            axis.plot(curve["x"], curve["mean_shap_crisis"], color=BLUE, linewidth=1.4)
        if threshold.get("threshold") is not None:
            axis.axvline(
                threshold["threshold"], color=BLUE, linestyle=(0, (4, 3)), linewidth=1.0
            )
            axis.annotate(
                f"sign change at {threshold['threshold']:.1f}",
                xy=(threshold["threshold"], axis.get_ylim()[1]),
                xytext=(4, -10), textcoords="offset points",
                fontsize=7.5, color=BLUE,
            )
        axis.set_xlabel(name)
        axis.set_ylabel("SHAP value, Crisis class" if name == chosen[0] else "")
        axis.set_title(name, loc="left", color=NAVY)

    figure.suptitle(
        "Where each indicator starts pushing the model toward a Crisis call",
        x=0.01, ha="left", color=NAVY, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(PAPER / "fig_shap_dependence.png")
    plt.close(figure)


def main() -> None:
    apply_style()
    county_map()
    print("wrote fig_county_map.png")
    rainfall_vci_timeseries()
    print("wrote fig_rainfall_vci_timeseries.png")
    classification_performance()
    print("wrote fig_classification_performance.png")
    shap_bar()
    print("wrote fig_shap_bar.png")
    shap_beeswarm()
    print("wrote fig_shap_beeswarm.png")
    shap_dependence()
    print("wrote fig_shap_dependence.png")


if __name__ == "__main__":
    main()
