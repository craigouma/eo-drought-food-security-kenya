"""SHAP analysis of the Earth observation classifier.

Global importance, dependence structure, and the alert thresholds the
dependence structure implies. The model explained here is the
Earth-observation-only variant trained on reporting periods up to 2020, since
that is the model whose behaviour the paper interprets: what a monitoring system
would be reading if it had nothing but satellite rainfall and vegetation data.

The thresholds reported at the end are read off the SHAP dependence curve. For
each leading driver, the value at which its mean SHAP contribution to the Crisis
class crosses from negative to positive is the point at which that indicator
starts pushing the model toward a Crisis call rather than away from it, which is
the natural candidate for an escalation trigger.
"""

import argparse
import json

import numpy as np
import pandas as pd
import shap

from build_dataset import CLASS_NAMES, FEATURES
from config import ROOT
from train_model import MODEL_DIR, fit_model, load_panel

OUT_PATH = MODEL_DIR / "shap_summary.json"
VALUES_PATH = MODEL_DIR / "shap_values_lead{lead}.npz"

CRISIS_INDEX = 2  # third class: Crisis or worse

# Features whose dependence curve is read for an operational threshold.
THRESHOLD_FEATURES = [
    "spi3_recent",
    "vci_recent",
    "vci_prior1",
    "vci_prior2",
    "spi6_recent",
    "vci_trend",
]


def compute_shap(lead: int = 1):
    """SHAP values for the test block under the EO-only model."""
    panel = load_panel(lead)
    model, _, test = fit_model(panel, FEATURES)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(test[FEATURES])
    # xgboost multiclass returns (n_samples, n_features, n_classes)
    values = np.asarray(values)
    return model, test, values, explainer


def global_importance(test: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, overall and for the Crisis class."""
    overall = np.abs(values).mean(axis=(0, 2))
    crisis = np.abs(values[:, :, CRISIS_INDEX]).mean(axis=0)
    frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap": overall,
            "mean_abs_shap_crisis": crisis,
        }
    )
    return frame.sort_values("mean_abs_shap_crisis", ascending=False).reset_index(
        drop=True
    )


def crossing_threshold(
    feature_values: np.ndarray, shap_values: np.ndarray, bins: int = 12
) -> dict:
    """Where a feature's contribution to the Crisis class changes sign.

    The feature is binned, the mean SHAP contribution is taken within each bin,
    and the crossing point is interpolated between the two bin centres that
    straddle zero. Returned as None when the curve does not cross within the
    observed range.
    """
    finite = np.isfinite(feature_values) & np.isfinite(shap_values)
    x = feature_values[finite]
    y = shap_values[finite]
    if x.size < 50:
        return {"threshold": None, "reason": "too few observations"}

    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 4:
        return {"threshold": None, "reason": "feature is near constant"}

    centres, means = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (x >= low) & (x <= high)
        if mask.sum() >= 10:
            centres.append(float(np.median(x[mask])))
            means.append(float(y[mask].mean()))
    centres = np.array(centres)
    means = np.array(means)
    if centres.size < 3:
        return {"threshold": None, "reason": "too few populated bins"}

    signs = np.sign(means)
    crossings = np.where(np.diff(signs) != 0)[0]
    if crossings.size == 0:
        return {
            "threshold": None,
            "reason": "no sign change across the observed range",
            "curve": {"x": centres.tolist(), "mean_shap_crisis": means.tolist()},
        }

    index = crossings[0]
    x0, x1 = centres[index], centres[index + 1]
    y0, y1 = means[index], means[index + 1]
    threshold = float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)
    return {
        "threshold": threshold,
        "direction": "below this value the indicator pushes toward Crisis"
        if means[0] > 0
        else "above this value the indicator pushes toward Crisis",
        "curve": {"x": centres.tolist(), "mean_shap_crisis": means.tolist()},
    }


def run(lead: int = 1) -> dict:
    model, test, values, explainer = compute_shap(lead)
    importance = global_importance(test, values)

    np.savez_compressed(
        str(VALUES_PATH).format(lead=lead),
        values=values,
        features=np.array(FEATURES, dtype=object),
        feature_values=test[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(),
        y_true=test["ipc_class"].to_numpy(),
        counties=test["county"].to_numpy().astype(str),
        dates=test["reporting_date"].astype(str).to_numpy(),
    )

    thresholds = {}
    for feature in THRESHOLD_FEATURES:
        index = FEATURES.index(feature)
        thresholds[feature] = crossing_threshold(
            pd.to_numeric(test[feature], errors="coerce").to_numpy(),
            values[:, index, CRISIS_INDEX],
        )

    summary = {
        "lead_months": lead,
        "n_explained": int(len(test)),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "global_importance": importance.to_dict(orient="records"),
        "crisis_thresholds": thresholds,
        "expected_value": np.asarray(explainer.expected_value).tolist(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 3, 6])
    args = parser.parse_args()

    summaries = {}
    for lead in args.leads:
        summary = run(lead)
        summaries[f"lead_{lead}"] = summary
        print(f"\nlead {lead} month(s), {summary['n_explained']:,} explained rows")
        frame = pd.DataFrame(summary["global_importance"]).head(8)
        print(frame.round(4).to_string(index=False))
        for feature, result in summary["crisis_thresholds"].items():
            if result.get("threshold") is not None:
                print(
                    f"  {feature}: Crisis contribution changes sign at "
                    f"{result['threshold']:.2f} ({result['direction']})"
                )

    OUT_PATH.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
