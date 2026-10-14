# Cyclone Wind-Intensity Forecasting

A reconstructed portfolio project based on Suhana Patel's resume description. This implementation uses **NOAA IBTrACS v04r01** storm tracks to forecast **maximum sustained wind speed 6 hours ahead** using Random Forest regression. It is not a real-time weather-warning product.

## Official data

NOAA/NCEI IBTrACS: https://www.ncei.noaa.gov/products/international-best-track-archive

Download the official modern-era global CSV (`ibtracs.since1980.list.v04r01.csv`) from:

https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv

The CSV's **second row contains units** and is skipped by the loader. Raw data is intentionally excluded from Git.

## Run

```bash
python -m pip install -r requirements.txt
python -m cyclone.pipeline --input data/raw/ibtracs.since1980.list.v04r01.csv --output outputs
pytest -q
```

The pipeline outputs metrics, predictions, basin-level evaluation, feature importance, and a serialized model. It uses **storm-group holdout** to prevent observations from the same cyclone appearing in both train and test.

## Method

1. Read official IBTrACS observations and parse storm ID, UTC time, basin, location, wind, and pressure.
2. Select USA_WIND where available and WMO_WIND otherwise (mixed-source wind estimates are an important limitation).
3. Build causal features from current/prior observations: wind, 6-hour change, pressure, location, motion speed, and basin.
4. Construct a **strictly 6-hour-ahead target** only for storm observations with a matching timestamp 6 hours later.
5. Train a Random Forest regressor on storm-separated train/test groups.
6. Evaluate MAE, RMSE, R2, per-basin MAE, and feature importance.

**Important:** Forecasting future wind using *current wind* is valid as a baseline. No future observations are used as model inputs. NOAA IBTrACS data are historical best tracks, not live operational forecasts.

## Status

The code and tests are committed; actual evaluation metrics are **not claimed** until the NOAA dataset has been downloaded and the pipeline run. The original project files were unavailable; this is a reconstruction from the resume description.
