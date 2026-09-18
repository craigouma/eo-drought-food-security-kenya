"""Download and prepare Kenya county boundaries.

Source: Humanitarian Data Exchange, Kenya administrative boundaries (COD-AB),
county level (ADM1). The dataset is public and needs no account or token.

Running this module as a script writes `data/raw/ken_admin1.geojson` and
`data/processed/target_counties.geojson`.
"""

import zipfile

import geopandas as gpd
import requests

from config import COUNTIES, HDX_BOUNDARY_URL, PROCESSED, RAW

ZIP_PATH = RAW / "ken_adm.geojson.zip"
ADM1_PATH = RAW / "ken_admin1.geojson"
TARGETS_PATH = PROCESSED / "target_counties.geojson"


def download_boundaries() -> None:
    """Fetch the COD-AB boundary archive if it is not already cached."""
    if ZIP_PATH.exists():
        return
    response = requests.get(HDX_BOUNDARY_URL, timeout=300, allow_redirects=True)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)


def load_counties() -> gpd.GeoDataFrame:
    """Return all 47 Kenyan counties as a GeoDataFrame in EPSG:4326."""
    download_boundaries()
    if not ADM1_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extract("ken_admin1.geojson", RAW)
    counties = gpd.read_file(ADM1_PATH)
    counties = counties[["adm1_name", "adm1_pcode", "area_sqkm", "geometry"]]
    counties = counties.rename(columns={"adm1_name": "county", "adm1_pcode": "pcode"})
    return counties.to_crs("EPSG:4326")


def load_target_counties() -> gpd.GeoDataFrame:
    """Return the five ASAL study counties, in the order used throughout."""
    counties = load_counties()
    targets = counties[counties["county"].isin(COUNTIES)].copy()
    missing = set(COUNTIES) - set(targets["county"])
    if missing:
        raise ValueError(f"county names not found in COD-AB boundaries: {sorted(missing)}")
    targets["county"] = targets["county"].astype("category")
    targets["county"] = targets["county"].cat.set_categories(COUNTIES)
    return targets.sort_values("county").reset_index(drop=True)


def main() -> None:
    targets = load_target_counties()
    targets.to_file(TARGETS_PATH, driver="GeoJSON")
    total_area = targets["area_sqkm"].sum()
    print(f"wrote {TARGETS_PATH.relative_to(TARGETS_PATH.parents[2])}")
    for _, row in targets.iterrows():
        print(f"  {row['county']:<10} {row['pcode']}  {row['area_sqkm']:>10,.0f} sq km")
    print(f"  {'total':<10} {'':<5}  {total_area:>10,.0f} sq km")


if __name__ == "__main__":
    main()
