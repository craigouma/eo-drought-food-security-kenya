"""Shared configuration for the EO drought and food security pipeline.

Every path, constant, and colour used by more than one stage of the pipeline is
defined here so that the notebook and the standalone scripts cannot drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
PAPER = ROOT / "paper"

for _d in (RAW, PROCESSED, PAPER):
    _d.mkdir(parents=True, exist_ok=True)

# Five arid and semi-arid land (ASAL) counties of northern Kenya.
COUNTIES = ["Turkana", "Marsabit", "Samburu", "Baringo", "Wajir"]

# Analysis window. CHIRPS starts in 1981; MODIS Terra NDVI starts in February 2000.
# The rainfall climatology is computed over the full CHIRPS record available here,
# while the modelling panel is restricted to the period with both data sources.
CHIRPS_START_YEAR = 1981
CHIRPS_END_YEAR = 2024
MODEL_START_YEAR = 2001
MODEL_END_YEAR = 2024

# Time-respecting split: train on the earlier years, test on the later ones.
TRAIN_END_YEAR = 2018
TEST_START_YEAR = 2019

# Humanitarian Data Exchange, Kenya common operational dataset, administrative
# boundaries (COD-AB). County level is ADM1 in this dataset.
HDX_BOUNDARY_URL = (
    "https://data.humdata.org/dataset/2c0b7571-4bef-4347-9b81-b2174c13f9ef/"
    "resource/674ea496-5451-4312-bcab-8aa95fa3f36c/download/"
    "ken_admin_boundaries.geojson.zip"
)

# Climate Hazards Center, CHIRPS v2.0, Africa monthly rasters.
CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs"

# Portfolio palette (see the operating brief, section 1.3).
NAVY = "#1A3A5C"
BLUE = "#2563EB"
BLUE_LT = "#EFF6FF"
BORDER = "#CBD8E6"
TEXT = "#1E2D3D"
MUTED = "#566B7E"
BG_ALT = "#F4F8FC"

# Four-class combined drought indicator scheme.
CDI_CLASSES = ["Normal", "Watch", "Warning", "Alert"]
