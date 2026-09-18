"""Extract county-mean monthly rainfall from CHIRPS v2.0.

Source: Climate Hazards Center, University of California Santa Barbara,
CHIRPS v2.0 Africa monthly rasters (0.05 degree, mm per month). The archive is
public domain and needs no account.

Each monthly raster is downloaded, clipped to the five study counties, reduced to
an area-weighted county mean, and then discarded unless `--keep-rasters` is set.
Only the resulting table (`data/processed/chirps_monthly_county.csv`) is kept,
which is a few tens of kilobytes rather than a few gigabytes.
"""

import argparse
import gzip
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask

from config import CHIRPS_BASE_URL, CHIRPS_END_YEAR, CHIRPS_START_YEAR, PROCESSED, RAW
from fetch_boundaries import load_target_counties

OUT_PATH = PROCESSED / "chirps_monthly_county.csv"
CHIRPS_NODATA = -9999.0


def month_url(year: int, month: int) -> str:
    return f"{CHIRPS_BASE_URL}/chirps-v2.0.{year}.{month:02d}.tif.gz"


def fetch_raster_bytes(year: int, month: int, keep: bool = False) -> bytes:
    """Return the decompressed GeoTIFF bytes for one CHIRPS month."""
    cache = RAW / f"chirps-v2.0.{year}.{month:02d}.tif"
    if cache.exists():
        return cache.read_bytes()
    response = requests.get(month_url(year, month), timeout=300)
    response.raise_for_status()
    raw = gzip.decompress(response.content)
    if keep:
        cache.write_bytes(raw)
    return raw


def county_means(raster_bytes: bytes, counties) -> dict[str, float]:
    """Area-mean rainfall in mm for each county polygon."""
    means = {}
    with rasterio.MemoryFile(raster_bytes) as memfile:
        with memfile.open() as src:
            for _, row in counties.iterrows():
                clipped, _ = mask(src, [row["geometry"]], crop=True, all_touched=False)
                values = clipped[0].astype("float64")
                values[values <= CHIRPS_NODATA] = np.nan
                means[row["county"]] = float(np.nanmean(values))
    return means


def fetch_month(args) -> dict:
    year, month, counties, keep = args
    raster_bytes = fetch_raster_bytes(year, month, keep=keep)
    record = {"year": year, "month": month}
    record.update(county_means(raster_bytes, counties))
    return record


def build_table(start_year: int, end_year: int, workers: int = 6, keep: bool = False) -> pd.DataFrame:
    counties = load_target_counties()
    jobs = [
        (year, month, counties, keep)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, record in enumerate(pool.map(fetch_month, jobs), start=1):
            records.append(record)
            if i % 60 == 0:
                print(f"  {i}/{len(jobs)} months processed", flush=True)

    wide = pd.DataFrame(records).sort_values(["year", "month"])
    long = wide.melt(
        id_vars=["year", "month"], var_name="county", value_name="rainfall_mm"
    )
    long["date"] = pd.to_datetime(
        dict(year=long["year"], month=long["month"], day=1)
    )
    return long[["date", "year", "month", "county", "rainfall_mm"]].sort_values(
        ["county", "date"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=CHIRPS_START_YEAR)
    parser.add_argument("--end", type=int, default=CHIRPS_END_YEAR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--keep-rasters", action="store_true")
    args = parser.parse_args()

    table = build_table(args.start, args.end, args.workers, args.keep_rasters)
    table.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(table):,} county-months, {args.start}-{args.end})")
    print(
        table.groupby("county", observed=True)["rainfall_mm"]
        .agg(["count", "mean", "min", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
