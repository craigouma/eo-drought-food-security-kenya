"""Train and evaluate the XGBoost classifier on the EO panel.

The split respects time: the model trains on reporting periods up to and
including 2020 and is tested on 2021 onward, with an inner validation block
(the last two years of the training period) used for early stopping so that no
test-period information reaches the fitted model.

Two reference points are reported alongside the model, because a classifier of
food insecurity phases is easy to make look good against no baseline at all:

* the majority class, which is what a system predicting "Minimal" every time
  would achieve;
* persistence, which predicts that a county stays in the phase it was assigned
  at the previous reporting date. Persistence is the baseline an operational
  early warning system actually has to beat, since the previous classification
  is always available.

Two model variants are fitted for the same reason. The Earth-observation-only
model answers whether satellite rainfall and vegetation data alone carry the
signal. The combined model adds the previous classification to those features
and answers the question that matters operationally: whether Earth observation
data adds anything on top of what the previous bulletin already said.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from build_dataset import (
    CATEGORICAL_FEATURES,
    CLASS_NAMES,
    FEATURES,
    FEATURES_NO_COUNTY,
    FEATURES_WITH_PERSISTENCE,
    OUT_TEMPLATE,
    build_panel,
)
from config import PROCESSED, ROOT, STUDY_COUNTIES, TRAIN_END_YEAR

MODEL_DIR = ROOT / "model"
RESULTS_PATH = MODEL_DIR / "model_results.json"
PREDICTIONS_PATH = PROCESSED / "test_predictions.csv"

INNER_VALIDATION_START = TRAIN_END_YEAR - 1  # last two years of the training block

PARAMS = dict(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5.0,
    reg_lambda=1.0,
    objective="multi:softprob",
    tree_method="hist",
    enable_categorical=True,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    random_state=42,
)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a proportion.

    Several of the quantities that matter here are measured on small subsets
    (the periods in which a county's phase changed, for instance), so the point
    estimates are reported with an interval rather than on their own.
    """
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [float(max(0.0, centre - margin)), float(min(1.0, centre + margin))]


def load_panel(lead: int) -> pd.DataFrame:
    path = PROCESSED / OUT_TEMPLATE.format(lead=lead)
    if path.exists():
        panel = pd.read_csv(path, parse_dates=["reporting_date"])
    else:
        panel = build_panel(lead)
    for column in CATEGORICAL_FEATURES:
        panel[column] = panel[column].astype("category")
    return panel


def balanced_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    weights = {cls: len(y) / (len(counts) * n) for cls, n in counts.items()}
    return y.map(weights).to_numpy()


def fit_model(
    panel: pd.DataFrame, features: list[str] = FEATURES
) -> tuple[XGBClassifier, pd.DataFrame, pd.DataFrame]:
    train = panel[panel["split"] == "train"]
    test = panel[panel["split"] == "test"]

    inner_train = train[train["year"] < INNER_VALIDATION_START]
    inner_validation = train[train["year"] >= INNER_VALIDATION_START]

    x_train = inner_train[features]
    y_train = inner_train["ipc_class"] - 1
    x_validation = inner_validation[features]
    y_validation = inner_validation["ipc_class"] - 1

    model = XGBClassifier(num_class=3, **PARAMS)
    model.fit(
        x_train,
        y_train,
        sample_weight=balanced_weights(y_train),
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    return model, train, test


def persistence_baseline(panel: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Each county's class at the preceding reporting date."""
    ordered = panel.sort_values(["county", "reporting_date"]).copy()
    ordered["previous_class"] = ordered.groupby("county", observed=True)[
        "ipc_class"
    ].shift(1)
    lookup = ordered.set_index(["county", "reporting_date"])["previous_class"]
    keys = pd.MultiIndex.from_arrays([test["county"], test["reporting_date"]])
    previous = lookup.reindex(keys).to_numpy()
    return np.where(np.isnan(previous), 1.0, previous)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, probabilities=None) -> dict:
    present = sorted(set(y_true) | set(y_pred))
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci": wilson_interval(int((y_true == y_pred).sum()), len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=present)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=present)
        ),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=[1, 2, 3],
            target_names=[CLASS_NAMES[c] for c in (1, 2, 3)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 2, 3]).tolist(),
    }
    if probabilities is not None and len(set(y_true)) > 1:
        metrics["auroc_ovr_macro"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=[1, 2, 3],
            )
        )
        for index, cls in enumerate((1, 2, 3)):
            binary = (y_true == cls).astype(int)
            if binary.sum() and binary.sum() < len(binary):
                metrics[f"auroc_{CLASS_NAMES[cls].split()[0].lower()}"] = float(
                    roc_auc_score(binary, probabilities[:, index])
                )
    return metrics


def crisis_alert_curve(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    """Precision and recall for a Crisis alert as the alert threshold moves.

    An early warning system does not have to act on the model's most likely
    class. It can raise an alert whenever the modelled probability of Crisis
    passes a chosen threshold, accepting more false alarms in exchange for
    catching more real ones. This records that trade-off explicitly.
    """
    crisis_probability = probabilities[:, 2]
    actual = (y_true == 3).astype(int)
    curve = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        flagged = (crisis_probability >= threshold).astype(int)
        true_positive = int(((flagged == 1) & (actual == 1)).sum())
        flagged_total = int(flagged.sum())
        curve.append(
            {
                "threshold": float(threshold),
                "flagged": flagged_total,
                "precision": float(true_positive / flagged_total) if flagged_total else None,
                "recall": float(true_positive / actual.sum()) if actual.sum() else None,
            }
        )
    return curve


def transition_analysis(
    test: pd.DataFrame, predictions: np.ndarray, persistence: np.ndarray
) -> dict:
    """How the model behaves where the classification actually moves.

    Persistence is unbeatable while a county stays in the same phase, and wrong
    by construction whenever the phase changes. Since a change is the event an
    early warning system exists to anticipate, the periods in which the phase
    moved are scored separately, and deteriorations (a move to a worse phase)
    separately again.
    """
    previous = test["previous_class"].to_numpy()
    actual = test["ipc_class"].to_numpy()
    known = ~np.isnan(previous)

    changed = known & (previous != actual)
    worsened = known & (previous < actual)

    def share(mask: np.ndarray, condition: np.ndarray) -> float | None:
        return float(condition[mask].mean()) if mask.sum() else None

    def share_ci(mask: np.ndarray, condition: np.ndarray) -> list[float] | None:
        return wilson_interval(int(condition[mask].sum()), int(mask.sum()))

    return {
        "model_exact_on_changed_ci": share_ci(changed, predictions == actual),
        "model_exact_on_worsened_ci": share_ci(worsened, predictions == actual),
        "model_direction_on_worsened_ci": share_ci(worsened, predictions > previous),
        "n_with_previous": int(known.sum()),
        "n_changed": int(changed.sum()),
        "n_worsened": int(worsened.sum()),
        "changed_share": float(changed.sum() / known.sum()) if known.sum() else None,
        "model_exact_on_changed": share(changed, predictions == actual),
        "persistence_exact_on_changed": share(changed, persistence == actual),
        "model_exact_on_worsened": share(worsened, predictions == actual),
        "model_direction_on_worsened": share(worsened, predictions > previous),
        "model_exact_on_stable": share(known & (previous == actual), predictions == actual),
        "persistence_exact_on_stable": share(
            known & (previous == actual), persistence == actual
        ),
    }


def run(lead: int, features: list[str], tag: str, test_keys=None) -> dict:
    panel = load_panel(lead)
    model, train, test = fit_model(panel, features)

    if test_keys is not None:
        keys = list(zip(test["county"], test["reporting_date"]))
        keep = np.array([key in test_keys for key in keys])
        test = test[keep]

    probabilities = model.predict_proba(test[features])
    predictions = probabilities.argmax(axis=1) + 1
    y_true = test["ipc_class"].to_numpy()
    persistence = persistence_baseline(panel, test).astype(int)

    results = {
        "lead_months": lead,
        "feature_set": tag,
        "features": features,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_period": [int(train["year"].min()), int(train["year"].max())],
        "test_period": [int(test["year"].min()), int(test["year"].max())],
        "best_iteration": int(model.best_iteration),
        "model": evaluate(y_true, predictions, probabilities),
        "majority_baseline": evaluate(y_true, np.ones_like(y_true)),
        "persistence_baseline": evaluate(y_true, persistence),
        "crisis_alert_curve": crisis_alert_curve(y_true, probabilities),
        "transitions": transition_analysis(test, predictions, persistence),
    }

    study_mask = test["county"].isin(STUDY_COUNTIES).to_numpy()
    if study_mask.sum():
        results["study_counties"] = {
            "model": evaluate(
                y_true[study_mask], predictions[study_mask], probabilities[study_mask]
            ),
            "persistence_baseline": evaluate(y_true[study_mask], persistence[study_mask]),
            "crisis_alert_curve": crisis_alert_curve(
                y_true[study_mask], probabilities[study_mask]
            ),
        }

    output = test[
        ["county", "reporting_date", "year", "ipc_class", "is_study_county"]
    ].copy()
    output["predicted_class"] = predictions
    output["persistence_class"] = persistence
    for index, cls in enumerate((1, 2, 3)):
        output[f"prob_class{cls}"] = probabilities[:, index]
    output["lead_months"] = lead
    output["feature_set"] = tag

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / f"xgb_ipc_{tag}_lead{lead}.json")
    if tag == "eo_only":
        existing = (
            pd.read_csv(PREDICTIONS_PATH, parse_dates=["reporting_date"])
            if PREDICTIONS_PATH.exists()
            else None
        )
        if existing is not None:
            existing = existing[
                ~(
                    (existing["lead_months"] == lead)
                    & (existing["feature_set"] == tag)
                )
            ]
            output = pd.concat([existing, output], ignore_index=True)
        output.to_csv(PREDICTIONS_PATH, index=False)
    return results


def walk_forward(
    lead: int, features: list[str], first_test_year: int = 2016
) -> tuple[pd.DataFrame, dict]:
    """Expanding-window evaluation, one fold per year.

    A single split leaves few reporting periods to score, particularly once the
    five study counties are isolated. This repeatedly trains on every reporting
    period up to the end of year Y and predicts year Y+1, so each year from
    `first_test_year` onward is predicted by a model that never saw it or
    anything after it. The pooled out-of-sample predictions are what the paper
    reports for the study counties.
    """
    panel = load_panel(lead)
    years = sorted(panel["year"].unique())
    test_years = [y for y in years if y >= first_test_year]

    folds = []
    for year in test_years:
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if len(test) == 0 or (train["ipc_class"].nunique() < 3):
            continue
        inner_cut = train["year"].max()
        inner_train = train[train["year"] < inner_cut]
        inner_validation = train[train["year"] == inner_cut]
        if len(inner_validation) == 0 or inner_train["ipc_class"].nunique() < 3:
            inner_train, inner_validation = train, train

        model = XGBClassifier(num_class=3, **PARAMS)
        model.fit(
            inner_train[features],
            inner_train["ipc_class"] - 1,
            sample_weight=balanced_weights(inner_train["ipc_class"] - 1),
            eval_set=[(inner_validation[features], inner_validation["ipc_class"] - 1)],
            verbose=False,
        )
        probabilities = model.predict_proba(test[features])
        fold = test[
            [
                "county", "reporting_date", "year", "ipc_class", "previous_class",
                "is_study_county",
            ]
        ].copy()
        fold["predicted_class"] = probabilities.argmax(axis=1) + 1
        for index, cls in enumerate((1, 2, 3)):
            fold[f"prob_class{cls}"] = probabilities[:, index]
        fold["n_train"] = len(train)
        folds.append(fold)

    pooled = pd.concat(folds, ignore_index=True)
    pooled["persistence_class"] = pooled["previous_class"].fillna(1).astype(int)
    pooled["lead_months"] = lead

    probabilities = pooled[[f"prob_class{c}" for c in (1, 2, 3)]].to_numpy()
    y_true = pooled["ipc_class"].to_numpy()
    predictions = pooled["predicted_class"].to_numpy()
    persistence = pooled["persistence_class"].to_numpy()

    summary = {
        "lead_months": lead,
        "n_folds": len(folds),
        "test_years": [int(pooled["year"].min()), int(pooled["year"].max())],
        "n_test": int(len(pooled)),
        "model": evaluate(y_true, predictions, probabilities),
        "persistence_baseline": evaluate(y_true, persistence),
        "majority_baseline": evaluate(y_true, np.ones_like(y_true)),
        "transitions": transition_analysis(pooled, predictions, persistence),
        "crisis_alert_curve": crisis_alert_curve(y_true, probabilities),
    }

    study = pooled[pooled["is_study_county"]]
    if len(study):
        study_probabilities = study[[f"prob_class{c}" for c in (1, 2, 3)]].to_numpy()
        summary["study_counties"] = {
            "n_test": int(len(study)),
            "model": evaluate(
                study["ipc_class"].to_numpy(),
                study["predicted_class"].to_numpy(),
                study_probabilities,
            ),
            "persistence_baseline": evaluate(
                study["ipc_class"].to_numpy(), study["persistence_class"].to_numpy()
            ),
            "transitions": transition_analysis(
                study,
                study["predicted_class"].to_numpy(),
                study["persistence_class"].to_numpy(),
            ),
            "crisis_alert_curve": crisis_alert_curve(
                study["ipc_class"].to_numpy(), study_probabilities
            ),
        }
    return pooled, summary


def common_test_keys(leads: list[int]) -> set:
    """County-period keys present in the test block at every lead time.

    Longer leads need more months of prior data, so they lose a few rows at the
    start and end of the record. Comparing lead times on different test sets
    would confound the comparison, so all leads are scored on the intersection.
    """
    keys = None
    for lead in leads:
        panel = load_panel(lead)
        test = panel[panel["split"] == "test"]
        lead_keys = set(zip(test["county"], test["reporting_date"]))
        keys = lead_keys if keys is None else keys & lead_keys
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 3, 6])
    args = parser.parse_args()

    if PREDICTIONS_PATH.exists():
        PREDICTIONS_PATH.unlink()

    shared_keys = common_test_keys(args.leads)
    print(f"common test block: {len(shared_keys):,} county-periods\n")

    variants = [
        ("eo_only", FEATURES),
        ("eo_plus_persistence", FEATURES_WITH_PERSISTENCE),
        ("eo_no_county", FEATURES_NO_COUNTY),
    ]

    all_results = {"common_test_n": len(shared_keys)}
    for tag, features in variants:
        for lead in args.leads:
            results = run(lead, features, tag, shared_keys)
            all_results[f"{tag}_lead_{lead}"] = results
            model = results["model"]
            crisis = model["per_class"]["Crisis or worse"]
            transitions = results["transitions"]
            print(
                f"{tag:<20} lead {lead}m: accuracy={model['accuracy']:.3f} "
                f"macro_f1={model['macro_f1']:.3f} "
                f"auroc={model.get('auroc_ovr_macro', float('nan')):.3f} "
                f"crisis_recall={crisis['recall']:.3f} "
                f"crisis_precision={crisis['precision']:.3f}"
            )
            print(
                f"{'':<20}          on the {transitions['n_changed']} periods where the "
                f"phase changed: model exact={transitions['model_exact_on_changed']:.3f}, "
                f"persistence exact={transitions['persistence_exact_on_changed']:.3f}; "
                f"deterioration direction caught="
                f"{transitions['model_direction_on_worsened']:.3f} "
                f"of {transitions['n_worsened']}"
            )
        print()

    print("walk-forward evaluation (one fold per year, expanding window)")
    for tag, features in variants:
        for lead in args.leads:
            pooled, summary = walk_forward(lead, features)
            all_results[f"walkforward_{tag}_lead_{lead}"] = summary
            if tag == "eo_only":
                pooled.to_csv(
                    PROCESSED / f"walkforward_predictions_lead{lead}.csv", index=False
                )
            model = summary["model"]
            study = summary.get("study_counties", {}).get("model", {})
            transitions = summary["transitions"]
            print(
                f"  {tag:<20} lead {lead}m: n={summary['n_test']:,} "
                f"accuracy={model['accuracy']:.3f} macro_f1={model['macro_f1']:.3f} "
                f"auroc={model.get('auroc_ovr_macro', float('nan')):.3f} | "
                f"study counties n={study.get('n', 0)} "
                f"accuracy={study.get('accuracy', float('nan')):.3f} "
                f"crisis_recall="
                f"{study.get('per_class', {}).get('Crisis or worse', {}).get('recall', float('nan')):.3f}"
            )
            print(
                f"  {'':<20}          changed periods n={transitions['n_changed']}: "
                f"model exact={transitions['model_exact_on_changed']:.3f}, "
                f"deteriorations caught="
                f"{transitions['model_direction_on_worsened']:.3f} "
                f"of {transitions['n_worsened']}"
            )
    print()

    persistence = all_results[f"eo_only_lead_{args.leads[0]}"]["persistence_baseline"]
    majority = all_results[f"eo_only_lead_{args.leads[0]}"]["majority_baseline"]
    print(
        f"{'persistence baseline':<20}       accuracy={persistence['accuracy']:.3f} "
        f"macro_f1={persistence['macro_f1']:.3f} "
        f"crisis_recall={persistence['per_class']['Crisis or worse']['recall']:.3f}"
    )
    print(
        f"{'majority baseline':<20}       accuracy={majority['accuracy']:.3f} "
        f"macro_f1={majority['macro_f1']:.3f}"
    )

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
