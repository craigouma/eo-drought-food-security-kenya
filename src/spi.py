"""Standardized Precipitation Index at 1, 3, and 6 month accumulations.

The standardized anomaly in `rainfall_anomaly.py` divides by the standard
deviation, which treats monthly rainfall as if it were symmetric. In arid
counties it is not: the distribution is strongly right-skewed, so a simple
z-score compresses the dry tail and stretches the wet one. The Standardized
Precipitation Index of McKee, Doesken and Kleist (1993) fits a gamma
distribution to the accumulated rainfall for each county and calendar month,
evaluates the cumulative probability of the observed total, and maps it through
the inverse normal distribution. Both measures are kept so that the difference
between them is visible rather than assumed.

The gamma parameters are fitted on the 1981-2010 baseline only.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED
from rainfall_anomaly import BASELINE_END, BASELINE_START, load_chirps

OUT_PATH = PROCESSED / "spi_county.csv"
WINDOWS = (1, 3, 6)


def accumulate(frame: pd.DataFrame, window: int) -> pd.Series:
    """Rolling rainfall total over `window` months, per county."""
    return (
        frame.groupby("county", observed=True)["rainfall_mm"]
        .transform(lambda s: s.rolling(window, min_periods=window).sum())
    )


def _fit_spi(baseline_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map accumulated rainfall onto the SPI scale using a fitted gamma."""
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if baseline_values.size < 10:
        return np.full(values.shape, np.nan)

    zero_fraction = float((baseline_values <= 0).mean())
    positive = baseline_values[baseline_values > 0]
    if positive.size < 5:
        return np.full(values.shape, np.nan)

    shape, _, scale = stats.gamma.fit(positive, floc=0)

    cumulative = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    positive_obs = finite & (values > 0)
    zero_obs = finite & (values <= 0)
    cumulative[positive_obs] = zero_fraction + (1 - zero_fraction) * stats.gamma.cdf(
        values[positive_obs], shape, loc=0, scale=scale
    )
    cumulative[zero_obs] = zero_fraction / 2 if zero_fraction > 0 else 1e-6

    # Keep the index finite at the extremes rather than returning infinities.
    cumulative = np.clip(cumulative, 1e-6, 1 - 1e-6)
    return stats.norm.ppf(cumulative)


def compute_spi(frame: pd.DataFrame, window: int) -> pd.Series:
    """SPI for one accumulation window, fitted per county and calendar month."""
    column = f"acc_{window}"
    work = frame.copy()
    work[column] = accumulate(work, window)
    result = pd.Series(np.nan, index=work.index, dtype="float64")

    baseline_mask = (work["year"] >= BASELINE_START) & (work["year"] <= BASELINE_END)
    for (county, month), group in work.groupby(["county", "month"], observed=True):
        baseline_values = group.loc[baseline_mask.loc[group.index], column].to_numpy()
        result.loc[group.index] = _fit_spi(baseline_values, group[column].to_numpy())
    return result


def build_spi_table() -> pd.DataFrame:
    frame = load_chirps()
    out = frame[["date", "year", "month", "county", "rainfall_mm"]].copy()
    for window in WINDOWS:
        out[f"spi{window}"] = compute_spi(frame, window)
    return out


def main() -> None:
    table = build_spi_table()
    table.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(table):,} county-months)")
    print(f"gamma fitted on the {BASELINE_START}-{BASELINE_END} baseline")
    print(
        table[[f"spi{w}" for w in WINDOWS]]
        .describe()
        .round(3)
        .to_string()
    )
    driest = table.nsmallest(8, "spi3")[["date", "county", "rainfall_mm", "spi3"]]
    print("\neight driest county-months on SPI-3:")
    print(driest.to_string(index=False))


if __name__ == "__main__":
    main()
