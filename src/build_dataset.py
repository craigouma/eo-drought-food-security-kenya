"""Assemble the modelling panel: EO predictors against FEWS NET IPC outcomes.

One row is one county and one FEWS NET reporting date. The outcome is the
county's area-majority IPC phase, collapsed to three classes (Minimal,
Stressed, Crisis or worse). The predictors are Earth observation indicators
observed strictly before the classification period, so the panel poses a
forecasting question rather than a nowcasting one.

The lead time is a parameter. At a lead of `L` months, the most recent month a
feature may use is `L` months before the month the classification describes. A
lead of one month is the operational case: a monitoring system holding rainfall
and vegetation data through the end of the previous month. Longer leads answer
how much earlier the same signal is usable.
"""

import argparse

import numpy as np
import pandas as pd

from config import PROCESSED, STUDY_COUNTIES, TEST_START_YEAR, TRAIN_END_YEAR
from rainfall_anomaly import BASELINE_END, BASELINE_START

SPI_PATH = PROCESSED / "spi_county.csv"
VCI_PATH = PROCESSED / "vci_county_monthly.csv"
IPC_PATH = PROCESSED / "ipc_county_phase.csv"
CHIRPS_PATH = PROCESSED / "chirps_monthly_county.csv"
OUT_TEMPLATE = "model_panel_lead{lead}.csv"

CLASS_NAMES = {1: "Minimal", 2: "Stressed", 3: "Crisis or worse"}

# Earth observation predictors, plus the structural and seasonal context needed
# to read them (a given rainfall deficit means something different in Turkana
# than in Kisii, and something different in March than in August).
FEATURES = [
    "spi1_recent",
    "spi3_recent",
    "spi6_recent",
    "spi3_prior",
    "vci_recent",
    "vci_prior1",
    "vci_prior2",
    "vci_trend",
    "mean_annual_rain_mm",
    "month",
    "season",
    "county",
]

# The second model variant additionally sees the classification a county
# received at the previous reporting date, which is what an operational system
# would have in hand.
FEATURES_WITH_PERSISTENCE = FEATURES + ["previous_class", "months_since_previous"]

CATEGORICAL_FEATURES = ["season", "county"]

SEASONS = {
    1: "dry_jf", 2: "dry_jf",
    3: "long_rains", 4: "long_rains", 5: "long_rains",
    6: "dry_jjas", 7: "dry_jjas", 8: "dry_jjas", 9: "dry_jjas",
    10: "short_rains", 11: "short_rains", 12: "short_rains",
}


def month_index(dates: pd.Series) -> pd.Series:
    """Months since year zero, so that lags are simple integer arithmetic."""
    dates = pd.to_datetime(dates)
    return dates.dt.year * 12 + dates.dt.month


def load_monthly_eo() -> pd.DataFrame:
    """County-month table of rainfall indices and vegetation condition."""
    spi = pd.read_csv(SPI_PATH, parse_dates=["date"])
    vci = pd.read_csv(VCI_PATH, parse_dates=["date"])
    eo = spi.merge(
        vci[["county", "date", "ndvi", "vci"]], on=["county", "date"], how="left"
    )
    eo["m"] = month_index(eo["date"])
    return eo


def county_aridity() -> pd.DataFrame:
    """Long-run mean annual rainfall per county, from the 1981-2010 baseline."""
    chirps = pd.read_csv(CHIRPS_PATH, parse_dates=["date"])
    baseline = chirps[
        (chirps["year"] >= BASELINE_START) & (chirps["year"] <= BASELINE_END)
    ]
    annual = (
        baseline.groupby(["county", "year"], as_index=False)["rainfall_mm"].sum()
    )
    return (
        annual.groupby("county", as_index=False)["rainfall_mm"]
        .mean()
        .rename(columns={"rainfall_mm": "mean_annual_rain_mm"})
    )


def load_outcomes() -> pd.DataFrame:
    ipc = pd.read_csv(IPC_PATH, parse_dates=["reporting_date"])
    ipc["ipc_class"] = ipc["ipc_phase"].clip(upper=3).astype(int)
    ipc["class_name"] = ipc["ipc_class"].map(CLASS_NAMES)
    ipc["m"] = month_index(ipc["reporting_date"])
    return ipc


def build_panel(lead: int = 1) -> pd.DataFrame:
    """Join EO features at the requested lead time onto the IPC outcomes."""
    eo = load_monthly_eo()
    outcomes = load_outcomes()
    aridity = county_aridity()

    keyed = eo.set_index(["county", "m"]).sort_index()

    def lookup(column: str, offset: int) -> pd.Series:
        keys = pd.MultiIndex.from_arrays(
            [outcomes["county"], outcomes["m"] - offset]
        )
        return pd.Series(
            keyed[column].reindex(keys).to_numpy(), index=outcomes.index
        )

    panel = outcomes[
        [
            "county", "reporting_date", "year", "month", "ipc_phase", "ipc_class",
            "class_name", "share_crisis_plus", "mean_phase", "classified_share",
        ]
    ].copy()

    panel["spi1_recent"] = lookup("spi1", lead)
    panel["spi3_recent"] = lookup("spi3", lead)
    panel["spi6_recent"] = lookup("spi6", lead)
    panel["spi3_prior"] = lookup("spi3", lead + 2)
    panel["vci_recent"] = lookup("vci", lead)
    panel["vci_prior1"] = lookup("vci", lead + 1)
    panel["vci_prior2"] = lookup("vci", lead + 2)
    panel["vci_trend"] = panel["vci_recent"] - panel["vci_prior2"]
    panel["season"] = panel["month"].map(SEASONS)
    panel = panel.merge(aridity, on="county", how="left")

    # The phase assigned at the previous reporting date is information an
    # operational system always holds. It is carried here both as the
    # persistence baseline and as a feature for the model variant that is
    # allowed to use it.
    ordered = outcomes.sort_values(["county", "reporting_date"])
    ordered["previous_class"] = ordered.groupby("county", observed=True)[
        "ipc_class"
    ].shift(1)
    ordered["months_since_previous"] = ordered.groupby("county", observed=True)[
        "m"
    ].diff()
    panel = panel.merge(
        ordered[
            ["county", "reporting_date", "previous_class", "months_since_previous"]
        ],
        on=["county", "reporting_date"],
        how="left",
    )
    panel["lead_months"] = lead
    panel["split"] = np.where(
        panel["year"] <= TRAIN_END_YEAR,
        "train",
        np.where(panel["year"] >= TEST_START_YEAR, "test", "unused"),
    )
    panel["is_study_county"] = panel["county"].isin(STUDY_COUNTIES)

    required = ["spi3_recent", "vci_recent", "vci_prior2"]
    complete = panel.dropna(subset=required).reset_index(drop=True)
    return complete.sort_values(["reporting_date", "county"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 3, 6])
    args = parser.parse_args()

    for lead in args.leads:
        panel = build_panel(lead)
        path = PROCESSED / OUT_TEMPLATE.format(lead=lead)
        panel.to_csv(path, index=False)
        counts = panel.groupby("split")["ipc_class"].value_counts().unstack(fill_value=0)
        print(f"\nlead {lead} month(s): {len(panel):,} county-periods -> {path.name}")
        print(f"  reporting dates {panel['reporting_date'].min():%Y-%m} "
              f"to {panel['reporting_date'].max():%Y-%m}")
        print(counts.rename(columns=CLASS_NAMES).to_string())
        study = panel[panel["is_study_county"]]
        print(f"  study counties: {len(study):,} rows, "
              f"{(study['ipc_class'] == 3).mean():.1%} in Crisis or worse")


if __name__ == "__main__":
    main()
