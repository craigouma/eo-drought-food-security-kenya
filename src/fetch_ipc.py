"""Build the county-level food insecurity outcome from FEWS NET classifications.

Source: FEWS NET Data Warehouse, Kenya acute food insecurity classifications
(current situation), distributed publicly and listed on the Humanitarian Data
Exchange as "Kenya Current Situation FEWS NET Acute Food Insecurity
Classifications Data from 2011". These are IPC-compatible phase classifications
(1 Minimal, 2 Stressed, 3 Crisis, 4 Emergency, 5 Famine) produced by FEWS NET
for sub-county food security units, on a roughly tri-annual reporting cycle.

Two public endpoints are used:

* `ipcphase` returns the tabular classification record for every unit and
  reporting date, which is what defines the set of reporting dates.
* `ipcphasemap` returns, for one reporting date, one dissolved polygon per
  phase. Intersecting those polygons with the county boundaries gives the share
  of each county's area in each phase, which is how a unit-level classification
  is turned into a county-level outcome.

The area-majority phase for a county is used as the outcome label. The share of
county area in Phase 3 or worse is retained alongside it as a continuous measure
of the severity of the classification.
"""

import json
import time

import geopandas as gpd
import pandas as pd
import requests

from config import PROCESSED, RAW
from fetch_boundaries import load_counties

PHASE_TABLE_URL = (
    "https://fdw.fews.net/api/ipcphase/"
    "?preference=best&country=KE&scenario=CS&format=csv"
)
PHASE_MAP_URL = "https://fdw.fews.net/api/ipcphasemap/"

RAW_TABLE = RAW / "fewsnet_ke_ipc_current_situation.csv"
MAP_DIR = RAW / "ipc_phase_maps"
OUT_PATH = PROCESSED / "ipc_county_phase.csv"

# Equal-area projection used only for computing polygon areas.
EQUAL_AREA_CRS = "EPSG:6933"

PHASE_NAMES = {
    1: "Minimal",
    2: "Stressed",
    3: "Crisis",
    4: "Emergency",
    5: "Famine",
}


def download_phase_table() -> pd.DataFrame:
    """Fetch the tabular FEWS NET classification record for Kenya."""
    if not RAW_TABLE.exists():
        response = requests.get(PHASE_TABLE_URL, timeout=600)
        response.raise_for_status()
        RAW_TABLE.write_bytes(response.content)
    table = pd.read_csv(RAW_TABLE, low_memory=False)
    table["reporting_date"] = pd.to_datetime(table["reporting_date"])
    return table


def reporting_dates(table: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(table["reporting_date"].dropna().unique())


def download_phase_map(date: pd.Timestamp) -> dict:
    """Fetch (and cache) the dissolved phase polygons for one reporting date."""
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(date).strftime("%Y-%m-%d")
    path = MAP_DIR / f"ke_cs_{stamp}.geojson"
    if not path.exists():
        response = requests.get(
            PHASE_MAP_URL,
            params={
                "country": "KE",
                "scenario": "CS",
                "collection_date": stamp,
                "format": "geojson",
            },
            timeout=600,
        )
        response.raise_for_status()
        path.write_bytes(response.content)
        time.sleep(1.0)
    return json.loads(path.read_text())


def _polygons_only(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop non-polygonal parts and repair invalid rings.

    Some FEWS NET phase maps arrive as geometry collections that mix polygons
    with the boundary lines of neighbouring units. Only the polygonal parts
    carry area, so only those are kept.
    """
    exploded = frame.explode(index_parts=False).reset_index(drop=True)
    exploded = exploded[exploded.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    repaired = exploded.copy()
    invalid = ~repaired.geometry.is_valid
    if invalid.any():
        repaired.loc[invalid, "geometry"] = repaired.loc[invalid, "geometry"].buffer(0)
    return repaired[~repaired.geometry.is_empty].reset_index(drop=True)


def county_phase_shares(date: pd.Timestamp, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """Share of each county's area in each IPC phase for one reporting date."""
    payload = download_phase_map(date)
    features = payload.get("features", [])
    if not features:
        return pd.DataFrame()

    phases = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    phases = phases[["value", "geometry"]].to_crs(EQUAL_AREA_CRS)
    phases = _polygons_only(phases)
    counties_ea = _polygons_only(counties.to_crs(EQUAL_AREA_CRS))
    county_area = (
        counties_ea.assign(part_area=counties_ea.geometry.area)
        .groupby("county", as_index=False)["part_area"]
        .sum()
        .rename(columns={"part_area": "county_area"})
    )

    overlay = gpd.overlay(
        counties_ea[["county", "geometry"]],
        phases,
        how="intersection",
        keep_geom_type=False,
    )
    if overlay.empty:
        return pd.DataFrame()
    overlay["piece_area"] = overlay.geometry.area

    shares = (
        overlay.groupby(["county", "value"], as_index=False)["piece_area"].sum()
        .merge(county_area, on="county", how="left")
    )
    shares["share"] = shares["piece_area"] / shares["county_area"]
    shares["reporting_date"] = pd.Timestamp(date)
    return shares.rename(columns={"value": "phase"})


def build_county_outcomes(limit: int | None = None) -> pd.DataFrame:
    table = download_phase_table()
    dates = reporting_dates(table)
    if limit:
        dates = dates[:limit]
    counties = load_counties()[["county", "geometry"]]

    frames = []
    for i, date in enumerate(dates, start=1):
        shares = county_phase_shares(date, counties)
        if not shares.empty:
            frames.append(shares)
        if i % 10 == 0:
            print(f"  {i}/{len(dates)} reporting dates processed", flush=True)

    long = pd.concat(frames, ignore_index=True)
    wide = (
        long.pivot_table(
            index=["county", "reporting_date"],
            columns="phase",
            values="share",
            fill_value=0.0,
        )
        .rename(columns=lambda p: f"share_phase{int(p)}")
        .reset_index()
    )
    for phase in range(1, 6):
        column = f"share_phase{phase}"
        if column not in wide.columns:
            wide[column] = 0.0

    share_columns = [f"share_phase{p}" for p in range(1, 6)]
    wide["classified_share"] = wide[share_columns].sum(axis=1)
    # Renormalise onto the classified part of the county, so counties that are
    # only partly covered by a classification are not scored as if the
    # unclassified remainder were Phase 1.
    normalised = wide[share_columns].div(wide["classified_share"], axis=0)
    normalised.columns = [f"norm_{c}" for c in share_columns]
    wide = pd.concat([wide, normalised], axis=1)

    norm_columns = [f"norm_{c}" for c in share_columns]
    wide["ipc_phase"] = (
        wide[norm_columns].to_numpy().argmax(axis=1) + 1
    )
    wide["ipc_phase_name"] = wide["ipc_phase"].map(PHASE_NAMES)
    wide["share_crisis_plus"] = wide[
        ["norm_share_phase3", "norm_share_phase4", "norm_share_phase5"]
    ].sum(axis=1)
    wide["mean_phase"] = sum(
        wide[f"norm_share_phase{p}"] * p for p in range(1, 6)
    )
    wide["year"] = wide["reporting_date"].dt.year
    wide["month"] = wide["reporting_date"].dt.month
    return wide.sort_values(["county", "reporting_date"]).reset_index(drop=True)


def main() -> None:
    outcomes = build_county_outcomes()
    outcomes.to_csv(OUT_PATH, index=False)
    print(
        f"wrote {OUT_PATH} ({len(outcomes):,} county-periods, "
        f"{outcomes['county'].nunique()} counties, "
        f"{outcomes['reporting_date'].nunique()} reporting dates, "
        f"{outcomes['reporting_date'].min():%Y-%m} to "
        f"{outcomes['reporting_date'].max():%Y-%m})"
    )
    print(outcomes["ipc_phase_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
