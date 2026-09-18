# Decisions Log

This file records the substantive choices made while producing this repository,
so the reasoning is auditable rather than buried in code or prose. Four of these
are departures from the original analysis plan and are marked as such.

## 1. Outcome label: real FEWS NET classifications, not a constructed proxy

**Decision.** The outcome is the acute food insecurity phase published by FEWS
NET for Kenya, obtained from the FEWS NET Data Warehouse and aggregated to
county level. The fallback of constructing a Combined Drought Indicator from the
rainfall and vegetation predictors was not used.

**Why.** The plan preferred real IPC or IPC-compatible phase data and allowed a
constructed proxy only if real data could not be obtained without a lengthy
request process. Real data was obtainable. Two public endpoints were found and
tested:

- `https://fdw.fews.net/api/ipcphase/?preference=best&country=KE&scenario=CS&format=csv`
  returns 28,336 classification records for Kenya from 2011, one per food
  security unit per reporting date, with the phase value, the unit name, and the
  reporting date. The Humanitarian Data Exchange lists this exact URL as the
  distribution point for its "Kenya Current Situation FEWS NET Acute Food
  Insecurity Classifications Data from 2011" dataset.
- `https://fdw.fews.net/api/ipcphasemap/?country=KE&scenario=CS&collection_date=<date>&format=geojson`
  returns, for one reporting date, one dissolved polygon per phase.

Neither needs an account or a key. The `ipcvalue` and per-record endpoints do
require authentication, but they are not needed.

**Consequence.** This removes the circularity the plan anticipated and flagged
as a limitation. The predictors are satellite observations; the outcome is an
independent expert classification incorporating market, livelihood, and
assessment information that no satellite sees. The task is correspondingly
harder, and the reported performance is lower than a self-referential drought
index model would show. Section 7 of the paper discusses what remains: the label
is a judgement rather than a measurement.

## 2. Vegetation data: Planetary Computer instead of Google Earth Engine

**Departure from the plan.** The plan called for MODIS NDVI via Google Earth
Engine and set a rainfall-only fallback if Earth Engine access was not set up in
time.

**Decision.** MODIS data was read from the Microsoft Planetary Computer STAC
catalogue instead. The rainfall-only fallback was not needed.

**Why.** No Earth Engine credential existed on this machine
(`~/.config/earthengine/` was absent) and creating one needs an interactive
browser session. The Planetary Computer serves the same NASA products
(`modis-13Q1-061`) as cloud-optimised GeoTIFFs with anonymous, key-free access
through the `planetary-computer` and `pystac-client` packages. This is the same
data, from the same producer, by a different route, so the fallback to a
rainfall-only analysis would have discarded real vegetation data that was
available.

**Detail worth recording.** The Planetary Computer collection holds both Terra
(MOD13Q1) and Aqua (MYD13Q1). Both are used, giving an effective 8-day cadence
from mid-2002 rather than 16-day. The paper says so.

Two other access routes were tried first and are recorded so they are not
retried blindly:

- the ORNL DAAC MODIS subset REST API returned HTTP 500 on every endpoint,
  including its own product listing, across repeated attempts;
- the NOAA STAR blended Vegetation Health geoTIFF archive is reachable but its
  public geoTIFF directory holds only 2024 onward, which is too short a record
  for a VCI baseline.

## 3. Spatial scope: indicators extracted for all 47 counties

**Departure from the plan.** The plan scoped data extraction to the five study
counties.

**Decision.** Rainfall, NDVI, VCI, and IPC outcomes were extracted for all 47
Kenyan counties. The five ASAL counties remain the subject of the paper: they
are the focus of the map, the indicator time series, a dedicated results
section, and the interpretation.

**Why.** FEWS NET publishes roughly three reporting dates a year, so five
counties over 2011 to 2026 yields about 245 labelled observations in total and
about 60 in a held-out test window. That is too thin to fit a classifier on and
too thin to evaluate one with. Extending extraction to all 47 counties gives
2,350 county-periods and 1,363 pooled out-of-sample predictions, while costing
only additional compute on rasters that were being downloaded anyway.

**Check performed.** Because training nationally risks fitting relationships
that do not hold in the ASALs, study-county performance is reported separately
throughout, and a model variant that drops county identity entirely is reported
as a sensitivity check. The paper states that ASAL performance is measurably
weaker than national performance.

## 4. Evaluation: walk-forward validation as the primary scheme

**Departure from the plan.** The plan called for a single time-respecting
train/test split.

**Decision.** Both are reported. The fixed split (train up to 2020, test 2021
onward) is kept. The primary evaluation is expanding-window walk-forward
validation, one fold per year from 2016, which pools 1,363 out-of-sample
predictions.

**Why.** A single split leaves 658 test observations nationally and 60 in the
study counties, and lands the entire test window inside the 2021 to 2023 Horn of
Africa drought. Conclusions drawn from one such window would be conclusions
about that drought. Walk-forward keeps the time ordering intact (each fold
trains only on periods preceding the year it predicts) while spreading the
evaluation across ten years.

## 5. Baselines: persistence is reported even though the model loses to it

**Decision.** Persistence, meaning a rule that repeats each county's previous
published classification, is reported alongside the majority-class baseline, and
the paper states plainly that the satellite-only model does not beat it on
aggregate accuracy (0.810 against 0.850).

**Why.** Without it the model's 0.810 accuracy against a 0.607 majority-class
baseline would read as a strong result, and that reading would be wrong. Food
insecurity phases are highly autocorrelated, and the previous bulletin is always
available to an operational system. Reporting the model only against the weaker
baseline would overstate the contribution.

**What this led to.** Scoring the periods where the classification actually
changed, separately, since persistence is wrong on all of them by construction.
That is where the satellite model's contribution is real and measurable, and it
became the paper's central result.

## 6. Rainfall index: gamma-fitted SPI alongside the simple anomaly

**Decision.** Both a simple standardized anomaly (rainfall minus long-run mean,
divided by long-run standard deviation) and a properly gamma-fitted SPI at one,
three, and six month accumulations are computed. The SPI is used in the model.

**Why.** The plan specified a simplified SPI-style measure and that is retained
in `rainfall_anomaly.py`. Monthly rainfall in the ASAL counties is strongly
right-skewed, so dividing by a standard deviation compresses the dry tail, which
is the tail that matters. Fitting a gamma per county and calendar month, as in
McKee et al. (1993), is the standard correction and costs little.

**Baseline.** 1981 to 2010, the WMO standard normal period, which ends well
before the modelling window so no test-period statistic enters a predictor.

## 7. MODIS read at 2 km rather than 250 m, and validated

**Decision.** Each MODIS composite is read at GeoTIFF overview level 2, a
decimation factor of 8, giving roughly 2 km pixels.

**Why.** Reading the four Kenyan tiles at native 250 m for every 16-day
composite from 2001 to 2026 is several terabytes of transfer. County means over
areas of 10,000 to 70,000 square kilometres do not need 250 m pixels.

**Check performed.** `fetch_ndvi.py --validate` compares county means from the
decimated grid against a full-resolution read for the same tile and dates. Over
52 county-composite comparisons the mean difference is 0.002 NDVI units and the
largest absolute difference is 0.017, against county means ranging from 0.15 to
0.68. The paper reports these figures rather than asserting the choice is safe.

## 8. VCI baseline period ends before the test window

**Decision.** The per-county, per-calendar-month NDVI minimum and maximum in the
Kogan VCI formula are taken over 2001 to 2020.

**Why.** Using the full record would let post-2020 extremes define the range
that post-2020 observations are scored against, which is a subtle leak into the
test period.

**Consequence.** VCI values after 2020 can fall outside 0 to 100. They are left
unclipped, because clipping would discard the extreme values the model most
needs.

## 9. County aggregation of sub-county classifications

**Decision.** A county's phase is the phase covering the largest share of its
classified area, computed by intersecting the dissolved per-phase polygons with
the county boundary in an equal-area projection (EPSG:6933). Shares are
renormalised onto the classified part of the county.

**Why.** FEWS NET classifies livelihood and administrative units that do not
align with counties. Area-majority is the simplest defensible rule, and
renormalising prevents an unclassified remainder from being scored as Phase 1.
Classified area averages 97.1 percent of county area and never falls below 66.3
percent.

**Known weakness, stated in the paper.** Area-majority understates localised
crises. A county with a quarter of its area in Crisis and three quarters in
Stressed is labelled Stressed.

## 10. Known data gaps

- The MODIS series used here is missing July to October 2025 and June 2026.
  This removes one of the 50 FEWS NET reporting periods (October 2025) from the
  modelling panel, which therefore holds 2,303 rather than 2,350 rows at a
  one-month lead.
- FEWS NET reporting is not on a fixed cadence across the whole record: roughly
  quarterly in the earlier years and roughly tri-annual later. The panel carries
  a `months_since_previous` column so this is visible rather than assumed away.

## 11. Class collapsing

**Decision.** IPC phases 3 (Crisis), 4 (Emergency), and 5 (Famine) are merged
into one Crisis-or-worse class.

**Why.** Across 2,350 county-periods the counts are 1,401 Phase 1, 749 Phase 2,
190 Phase 3, 10 Phase 4, and no Phase 5. Ten Phase 4 observations cannot support
a separate class, and the operational distinction that matters most is whether a
county has crossed into Crisis.

## 12. Repository structure

**Departure from the plan.** The planned structure was `paper/`, `notebook/`,
`index.html`, `DECISIONS.md`, and `README.md`.

**Decision.** Three directories were added: `src/` holds the pipeline as eleven
importable modules, `data/processed/` holds the county-level series as committed
CSV files, and `model/` holds the fitted models, results, and SHAP summaries.

**Why.** A notebook that downloads several gigabytes of raster data before it
can produce a number is not reproducible in practice. Splitting the pipeline
into modules and committing the county-level intermediate series means the
modelling, SHAP, and figure stages re-run in minutes from a clean checkout,
while the fetch stages remain available and are what produced those series. The
notebook imports the same modules, so there is one implementation rather than
two that can drift apart.

## 13. Citations verified, and one corrected

Every citation was checked against Crossref or the publisher before use. One
correction resulted: the methodological template paper (Climate 14(1):14, DOI
10.3390/cli14010014) is by Ogembo, Olala, Ronoh, Mukama, and Akinyi, not the
author list assumed at the outset. The FEWS NET-linked NDVI study was identified
as Shukla et al. (2021), PLOS ONE 16(1): e0242883.
