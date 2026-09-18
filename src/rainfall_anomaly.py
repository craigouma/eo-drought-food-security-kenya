"""Standardized rainfall anomalies from the CHIRPS county series.

The anomaly is the simplified, SPI-style measure described in the operating
brief: for a given county and calendar month, monthly rainfall minus the
long-run mean for that county and calendar month, divided by the long-run
standard deviation.

The climatological baseline is the 1981-2010 World Meteorological Organization
standard normal period. Fixing the baseline to a period that ends before the
model's test window means no statistic computed from the test years feeds back
into the predictors.
"""

import numpy as np
import pandas as pd

from config import PROCESSED

CHIRPS_PATH = PROCESSED / "chirps_monthly_county.csv"
OUT_PATH = PROCESSED / "rainfall_anomaly_county.csv"

BASELINE_START = 1981
BASELINE_END = 2010


def load_chirps() -> pd.DataFrame:
    frame = pd.read_csv(CHIRPS_PATH, parse_dates=["date"])
    return frame.sort_values(["county", "date"]).reset_index(drop=True)


def climatology(frame: pd.DataFrame) -> pd.DataFrame:
    """Per county, per calendar month mean and standard deviation of rainfall."""
    baseline = frame[
        (frame["year"] >= BASELINE_START) & (frame["year"] <= BASELINE_END)
    ]
    stats = (
        baseline.groupby(["county", "month"])["rainfall_mm"]
        .agg(clim_mean="mean", clim_sd="std", clim_n="count")
        .reset_index()
    )
    return stats


def add_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the 1-month and 3-month standardized rainfall anomalies."""
    stats = climatology(frame)
    out = frame.merge(stats, on=["county", "month"], how="left")
    out["rain_anom"] = (out["rainfall_mm"] - out["clim_mean"]) / out["clim_sd"]

    # Three-month accumulated rainfall, standardized against its own baseline
    # distribution for the same ending calendar month. This is the conventional
    # agricultural-drought accumulation window.
    out = out.sort_values(["county", "date"]).reset_index(drop=True)
    out["rain_3m_mm"] = (
        out.groupby("county", observed=True)["rainfall_mm"]
        .transform(lambda s: s.rolling(3, min_periods=3).sum())
    )
    baseline_3m = out[
        (out["year"] >= BASELINE_START) & (out["year"] <= BASELINE_END)
    ]
    stats_3m = (
        baseline_3m.groupby(["county", "month"])["rain_3m_mm"]
        .agg(clim3_mean="mean", clim3_sd="std")
        .reset_index()
    )
    out = out.merge(stats_3m, on=["county", "month"], how="left")
    out["rain_anom_3m"] = (out["rain_3m_mm"] - out["clim3_mean"]) / out["clim3_sd"]
    return out.sort_values(["county", "date"]).reset_index(drop=True)


def main() -> None:
    frame = load_chirps()
    out = add_anomalies(frame)
    keep = [
        "date", "year", "month", "county", "rainfall_mm", "clim_mean", "clim_sd",
        "rain_anom", "rain_3m_mm", "rain_anom_3m",
    ]
    out[keep].to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} county-months)")
    print(f"climatological baseline: {BASELINE_START}-{BASELINE_END}")
    summary = (
        out.groupby("county", observed=True)["rain_anom"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
    )
    print(summary.to_string())
    driest = out.nsmallest(5, "rain_anom_3m")[
        ["date", "county", "rainfall_mm", "rain_anom_3m"]
    ]
    print("\nfive driest county-months on the 3-month anomaly:")
    print(driest.to_string(index=False))


if __name__ == "__main__":
    main()
