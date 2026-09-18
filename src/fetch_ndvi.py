"""County-mean MODIS NDVI and the Vegetation Condition Index.

Source: the MODIS 16-day vegetation index product at 250 m, version 6.1,
accessed as cloud-optimised GeoTIFFs through the Microsoft Planetary Computer
STAC catalogue. The catalogue's `modis-13Q1-061` collection holds both the Terra
product (MOD13Q1, from 2000) and the Aqua product (MYD13Q1, from mid-2002), each
on a 16-day cycle and offset from the other by 8 days, so the combined series has
an effective 8-day cadence once Aqua comes online. Both are used here. The
Planetary Computer is used instead of Google Earth Engine because it serves the
identical NASA products without an interactive account setup. NASA places no
restrictions on subsequent use or redistribution of MODIS data.

Each 16-day composite is read at a decimated resolution of roughly 2 km (GeoTIFF
overview level 3 of a 250 m grid) over the four MODIS sinusoidal tiles that cover
Kenya, and reduced to a mean NDVI per county. Reading a decimated grid rather
than the full 250 m grid keeps the transfer to a few gigabytes; county means over
areas of 10,000 to 70,000 square kilometres are not sensitive to that choice, and
`validate_decimation` checks it directly against a full-resolution read.

The Vegetation Condition Index follows Kogan (1995):

    VCI = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)

with the minimum and maximum taken per county and per calendar month over a
baseline period that ends before the model's test window.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.features import rasterize

from config import MODIS_END_YEAR, MODIS_START_YEAR, PROCESSED, STUDY_COUNTIES
from fetch_boundaries import load_counties

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13Q1-061"
NDVI_ASSET = "250m_16_days_NDVI"
TILES = ["h21v08", "h22v08", "h21v09", "h22v09"]
KENYA_BBOX = [33.8, -4.8, 42.1, 5.6]

# Overview index 3 of [2, 4, 8, 16] is a decimation factor of 16, i.e. about 4 km.
# Index 2 is a factor of 8, about 2 km, which is what the pipeline uses.
OVERVIEW_LEVEL = 2

NDVI_SCALE = 1e-4
NDVI_FILL = -3000
NDVI_MIN_VALID = -2000

VCI_BASELINE_START = 2001
VCI_BASELINE_END = 2020

NDVI_PATH = PROCESSED / "modis_ndvi_county_monthly.csv"
VCI_PATH = PROCESSED / "vci_county_monthly.csv"


def open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def search_items(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per MODIS tile and composite date, with the signed NDVI href."""
    catalog = open_catalog()
    search = catalog.search(
        collections=[COLLECTION],
        bbox=KENYA_BBOX,
        datetime=f"{start_year}-01-01/{end_year}-12-31",
    )
    records = []
    for item in search.items():
        tile = item.id.split(".")[2]
        if tile not in TILES:
            continue
        records.append(
            {
                "tile": tile,
                "composite_date": pd.Timestamp(item.properties["start_datetime"][:10]),
                "href": item.assets[NDVI_ASSET].href,
            }
        )
    frame = pd.DataFrame(records).drop_duplicates(["tile", "composite_date"])
    return frame.sort_values(["composite_date", "tile"]).reset_index(drop=True)


def county_label_grid(href: str, counties) -> tuple[np.ndarray, dict]:
    """Rasterise county polygons onto one tile's decimated grid.

    County boundaries do not change between composites, so this is computed once
    per tile and reused for every date.
    """
    with rasterio.open(href, overview_level=OVERVIEW_LEVEL) as src:
        shape = src.shape
        transform = src.transform
        crs = src.crs
    projected = counties.to_crs(crs)
    shapes = [
        (geom, idx + 1)
        for idx, geom in enumerate(projected.geometry)
        if geom is not None and not geom.is_empty
    ]
    labels = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    meta = {"shape": shape, "names": list(projected["county"])}
    return labels, meta


def tile_county_sums(href: str, labels: np.ndarray, n_counties: int):
    """Sum and count of valid NDVI values per county label for one tile-date."""
    with rasterio.open(href, overview_level=OVERVIEW_LEVEL) as src:
        band = src.read(1)
    valid = (band != NDVI_FILL) & (band >= NDVI_MIN_VALID) & (labels > 0)
    flat_labels = labels[valid]
    values = band[valid].astype("float64") * NDVI_SCALE
    sums = np.bincount(flat_labels, weights=values, minlength=n_counties + 1)
    counts = np.bincount(flat_labels, minlength=n_counties + 1)
    return sums, counts


def build_ndvi_table(start_year: int, end_year: int, workers: int = 8) -> pd.DataFrame:
    counties = load_counties()[["county", "geometry"]].reset_index(drop=True)
    n_counties = len(counties)
    items = search_items(start_year, end_year)
    print(f"  {len(items):,} tile-composites found", flush=True)

    label_grids = {}
    for tile in sorted(items["tile"].unique()):
        href = items[items["tile"] == tile].iloc[0]["href"]
        labels, meta = county_label_grid(href, counties)
        label_grids[tile] = labels
        print(f"  label grid for {tile}: {meta['shape']}", flush=True)

    dates = sorted(items["composite_date"].unique())
    by_date = {date: group for date, group in items.groupby("composite_date")}

    def process_date(date):
        sums = np.zeros(n_counties + 1)
        counts = np.zeros(n_counties + 1)
        for _, row in by_date[date].iterrows():
            tile_sums, tile_counts = tile_county_sums(
                row["href"], label_grids[row["tile"]], n_counties
            )
            sums += tile_sums
            counts += tile_counts
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        return [
            {
                "composite_date": pd.Timestamp(date),
                "county": counties.loc[i, "county"],
                "ndvi": means[i + 1],
                "n_pixels": int(counts[i + 1]),
            }
            for i in range(n_counties)
        ]

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, chunk in enumerate(pool.map(process_date, dates), start=1):
            records.extend(chunk)
            if i % 25 == 0:
                print(f"  {i}/{len(dates)} composites processed", flush=True)

    composites = pd.DataFrame(records)
    composites["year"] = composites["composite_date"].dt.year
    composites["month"] = composites["composite_date"].dt.month
    monthly = (
        composites.groupby(["county", "year", "month"], as_index=False)
        .agg(ndvi=("ndvi", "mean"), n_composites=("ndvi", "size"))
    )
    monthly["date"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )
    return monthly.sort_values(["county", "date"]).reset_index(drop=True)


def add_vci(monthly: pd.DataFrame) -> pd.DataFrame:
    """Kogan (1995) Vegetation Condition Index, per county and calendar month."""
    baseline = monthly[
        (monthly["year"] >= VCI_BASELINE_START) & (monthly["year"] <= VCI_BASELINE_END)
    ]
    extremes = (
        baseline.groupby(["county", "month"])["ndvi"]
        .agg(ndvi_min="min", ndvi_max="max")
        .reset_index()
    )
    out = monthly.merge(extremes, on=["county", "month"], how="left")
    out["vci"] = (
        100.0 * (out["ndvi"] - out["ndvi_min"]) / (out["ndvi_max"] - out["ndvi_min"])
    )
    return out.sort_values(["county", "date"]).reset_index(drop=True)


def validate_decimation(sample_dates: int = 2) -> pd.DataFrame:
    """Compare county-mean NDVI at the decimated and full 250 m resolutions.

    Reads one MODIS tile at both resolutions for a small number of dates and
    reports the difference in county means, so the decimation choice is checked
    rather than asserted.
    """
    counties = load_counties()[["county", "geometry"]].reset_index(drop=True)
    items = search_items(2015, 2016)
    items = items[items["tile"] == "h21v08"].head(sample_dates)
    rows = []
    for _, row in items.iterrows():
        coarse_labels, _ = county_label_grid(row["href"], counties)
        coarse_sums, coarse_counts = tile_county_sums(
            row["href"], coarse_labels, len(counties)
        )
        with rasterio.open(row["href"]) as src:
            full_shape, full_transform, crs = src.shape, src.transform, src.crs
            band = src.read(1)
        projected = counties.to_crs(crs)
        full_labels = rasterize(
            [
                (geom, idx + 1)
                for idx, geom in enumerate(projected.geometry)
                if geom is not None and not geom.is_empty
            ],
            out_shape=full_shape,
            transform=full_transform,
            fill=0,
            dtype="int32",
        )
        valid = (band != NDVI_FILL) & (band >= NDVI_MIN_VALID) & (full_labels > 0)
        full_sums = np.bincount(
            full_labels[valid],
            weights=band[valid].astype("float64") * NDVI_SCALE,
            minlength=len(counties) + 1,
        )
        full_counts = np.bincount(full_labels[valid], minlength=len(counties) + 1)
        for i, name in enumerate(counties["county"]):
            if coarse_counts[i + 1] == 0 or full_counts[i + 1] == 0:
                continue
            rows.append(
                {
                    "composite_date": row["composite_date"],
                    "county": name,
                    "ndvi_2km": coarse_sums[i + 1] / coarse_counts[i + 1],
                    "ndvi_250m": full_sums[i + 1] / full_counts[i + 1],
                }
            )
    check = pd.DataFrame(rows)
    check["difference"] = check["ndvi_2km"] - check["ndvi_250m"]
    return check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=MODIS_START_YEAR)
    parser.add_argument("--end", type=int, default=MODIS_END_YEAR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        check = validate_decimation()
        print(check.describe()[["ndvi_2km", "ndvi_250m", "difference"]].round(4).to_string())
        return

    monthly = build_ndvi_table(args.start, args.end, args.workers)
    monthly.to_csv(NDVI_PATH, index=False)
    print(f"wrote {NDVI_PATH} ({len(monthly):,} county-months)")

    with_vci = add_vci(monthly)
    with_vci.to_csv(VCI_PATH, index=False)
    print(f"wrote {VCI_PATH}")
    study = with_vci[with_vci["county"].isin(STUDY_COUNTIES)]
    print(
        study.groupby("county", observed=True)[["ndvi", "vci"]]
        .agg(["count", "mean", "min", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
