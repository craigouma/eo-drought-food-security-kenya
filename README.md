# Satellite-Derived Drought Indicators as Early Predictors of Food Insecurity Risk in Northern Kenya's Arid and Semi-Arid Lands

Craig Carlos Ouma, Independent Researcher.
[Paper (PDF)](paper/eo_drought_food_security.pdf) ·
[Notebook](notebook/eo_drought_food_security.ipynb) ·
[Landing page](https://craigouma.github.io/eo-drought-food-security-kenya/) ·
[Decisions log](DECISIONS.md)

## What this is

Food security classifications for Kenya's arid and semi-arid counties are
published roughly three times a year and rest on field assessment. Satellite
rainfall and vegetation data arrive continuously and cost nothing. This paper
measures how much of the published classification the satellite indicators can
actually anticipate, and how far ahead.

The analysis pairs CHIRPS rainfall (1981 to 2026) and MODIS vegetation indices
(2001 to 2026) with 2,350 county-period acute food insecurity classifications
published by FEWS NET between 2011 and 2026, for all 47 Kenyan counties. The
five arid and semi-arid study counties are Turkana, Marsabit, Samburu, Baringo,
and Wajir.

## Headline results

Pooled expanding-window walk-forward validation, 2016 to 2026, 1,363
out-of-sample county-periods.

| Model | Accuracy | Macro F1 | AUROC | Deteriorations anticipated |
|---|---|---|---|---|
| Majority class | 0.607 | 0.252 | n/a | 0% |
| Persistence (repeat previous classification) | 0.850 | 0.769 | n/a | 0% |
| Satellite only, 1 month lead | 0.810 | 0.707 | 0.913 | 59.3% |
| Satellite only, 6 month lead | 0.809 | 0.716 | 0.912 | 63.3% |
| Satellite plus previous classification | 0.824 | 0.710 | 0.933 | 41.6% |
| Satellite without county identity | 0.775 | 0.664 | 0.896 | 54.0% |

Three findings carry the paper:

1. **Satellite data alone does not beat persistence.** A rule that repeats each
   county's previous published classification reaches 0.850 accuracy against the
   model's 0.810. Food insecurity phases are highly autocorrelated, and the
   paper says so plainly rather than reporting the model only against the much
   weaker majority-class baseline.
2. **Persistence is silent exactly when warning is needed.** On the 205
   county-periods where the classification actually moved, persistence is wrong
   by construction. The satellite-only model assigns the correct class in 58.5%
   of them and anticipates the direction of 59.3% of the 113 deteriorations at
   one month of lead, rising to 63.3% at six months.
3. **Feeding the previous classification back into the model suppresses
   warning.** Doing so raises accuracy to 0.824 and cuts deterioration detection
   to 41.6%. That is a design warning for any system blending the two.

The SHAP analysis places the Crisis decision boundary near a Vegetation
Condition Index of 18, inside the severe vegetation deficit band (10 to 20) that
Kenya's National Drought Management Authority already uses operationally. The
convergence is independent: the model was never shown that scheme.

## Repository layout

```
eo-drought-food-security-kenya/
├── paper/
│   ├── eo_drought_food_security.tex     LaTeX source
│   ├── eo_drought_food_security.pdf     compiled paper
│   └── fig_*.png                        six figures
├── notebook/
│   └── eo_drought_food_security.ipynb   full pipeline, executed
├── src/
│   ├── config.py                        counties, windows, palette, paths
│   ├── fetch_boundaries.py              HDX Kenya COD-AB county polygons
│   ├── fetch_chirps.py                  CHIRPS rainfall to county means
│   ├── rainfall_anomaly.py              simple standardized anomaly
│   ├── spi.py                           gamma-fitted SPI at 1, 3, 6 months
│   ├── fetch_ndvi.py                    MODIS NDVI to county means, Kogan VCI
│   ├── fetch_ipc.py                     FEWS NET classifications to county labels
│   ├── build_dataset.py                 modelling panel at a chosen lead time
│   ├── train_model.py                   XGBoost, baselines, walk-forward
│   ├── shap_analysis.py                 SHAP values and alert thresholds
│   └── figures.py                       all six figures
├── data/processed/                      county-level series, committed
├── model/                               fitted models, results, SHAP summaries
├── index.html                           GitHub Pages landing page
├── DECISIONS.md                         substantive choices and departures
└── README.md
```

Raw rasters are not committed. They are re-downloadable from the cited public
archives and amount to several gigabytes.

## Data

| Source | Variable | Resolution | Coverage used | Access |
|---|---|---|---|---|
| CHIRPS v2.0, Climate Hazards Center | Monthly rainfall | 0.05 degree | 1981-01 to 2026-08 | Direct download, public domain |
| MOD13Q1 and MYD13Q1 v6.1 (NASA) | NDVI | 250 m, 16-day | 2001-01 to 2026-08 | Planetary Computer STAC, anonymous |
| FEWS NET classifications | IPC-compatible phase | Sub-county units | 2011-01 to 2026-06 | FEWS NET Data Warehouse, public |
| HDX Kenya COD-AB | County boundaries | 47 counties | current | Direct download |

None required an account, a key, or an approval process. CHIRPS is in the public
domain with rights waived by its producer. NASA places no restriction on the
subsequent use, sale, or redistribution of MODIS data. FEWS NET distributes the
classification records used here under a public data usage policy.

## Running it

```bash
python3.12 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib scipy requests seaborn \
    "rasterio==1.4.3" "pyproj==3.7.1" "shapely==2.0.7" geopandas \
    xgboost scikit-learn shap planetary-computer pystac-client jupyter
```

With the committed county-level CSVs in `data/processed/`, the modelling stages
run in a few minutes:

```bash
cd src
../.venv/bin/python build_dataset.py      # modelling panel at 1, 3, 6 month leads
../.venv/bin/python train_model.py        # fit, evaluate, walk-forward, baselines
../.venv/bin/python shap_analysis.py      # SHAP values and alert thresholds
../.venv/bin/python figures.py            # all six figures
```

To rebuild the county-level series from the public archives, which takes about
an hour and roughly three gigabytes of transfer:

```bash
cd src
../.venv/bin/python fetch_boundaries.py
../.venv/bin/python fetch_chirps.py --start 1981 --end 2026 --workers 8
../.venv/bin/python spi.py
../.venv/bin/python fetch_ndvi.py --start 2001 --end 2026 --workers 8
../.venv/bin/python fetch_ndvi.py --validate    # decimation check against 250 m
../.venv/bin/python fetch_ipc.py
```

The notebook runs the whole pipeline end to end and reproduces every number and
figure in the paper.

Building the PDF needs a LaTeX toolchain:

```bash
cd paper && tectonic -X compile eo_drought_food_security.tex
```

## Method in brief

- **Rainfall anomaly.** Standardized Precipitation Index at 1, 3, and 6 month
  accumulations, gamma-fitted per county and calendar month on the 1981 to 2010
  World Meteorological Organization normal period.
- **Vegetation condition.** Kogan (1995) VCI, with the per-county,
  per-calendar-month NDVI minimum and maximum taken over a 2001 to 2020 baseline
  that ends before the test window.
- **Outcome.** The area-majority FEWS NET phase per county per reporting date,
  computed by intersecting the dissolved per-phase polygons with county
  boundaries in an equal-area projection. Phases 3 and above are merged into one
  Crisis-or-worse class.
- **Features.** SPI at three accumulations, VCI at three lags and its
  three-month trend, calendar month and season, county mean annual rainfall, and
  county identity. Every feature is observed strictly before the period being
  classified.
- **Model.** XGBoost, depth 4, class weights inversely proportional to class
  frequency, early stopping on an inner validation block drawn from the end of
  the training period.
- **Evaluation.** Expanding-window walk-forward validation, one fold per year
  from 2016, alongside a single fixed split (train through 2020, test from
  2021). No observation is ever shuffled across time.

## Limitations

The outcome is an expert classification rather than a measurement, so the model
predicts a considered human judgement. The county is a coarse unit and an
area-majority rule understates localised crises. Conflict, displacement, market
prices, and humanitarian assistance all enter a food insecurity classification
and none is visible to a satellite. The evaluation rests on 151 Crisis-or-worse
classifications, 205 phase changes, and 113 deteriorations, so intervals are
reported throughout. Section 7 of the paper states these in full.

## Citation

```bibtex
@misc{ouma2026eodrought,
  title   = {Satellite-Derived Drought Indicators as Early Predictors of
             Food Insecurity Risk in Northern {Kenya's} Arid and
             Semi-Arid Lands},
  author  = {Ouma, Craig Carlos},
  year    = {2026},
  month   = {September},
  note    = {Preprint},
  url     = {https://craigouma.github.io/eo-drought-food-security-kenya/}
}
```

## Note on affiliation

This is independent research. It is not produced, reviewed, commissioned, or
endorsed by FEWS NET, Kenya's National Drought Management Authority, the Kenya
Space Agency, or the European Space Agency. The five counties studied are
drought-prone arid and semi-arid counties where food security early warning is a
recognised policy priority, which is why they were chosen.
