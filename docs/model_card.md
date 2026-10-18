# Model card: 24-hour carbon intensity forecaster

## Purpose

Forecast Great Britain's national grid carbon intensity (gCO2/kWh) for each half-hour, 24 hours ahead, to support scheduling of flexible electricity demand.

**Intended use:** analysis, education and as a transparent comparison point for the official forecast.
**Not intended for:** operational grid decisions, regulated emissions reporting, or as the sole basis for financial decisions. The dashboard's recommendations use NESO's official forecast, which is more accurate.

## Data

- **Source:** Carbon Intensity API (NESO), national half-hourly actual intensity and generation mix.
- **Window:** the most recent 90 days by default, refreshed daily.
- **Preparation:** loaded to a SQL warehouse, validated by five quality checks, reindexed onto a complete half-hourly time grid so that gaps appear as missing values instead of silently shifting lags.

## Model

- **Algorithm:** scikit-learn `HistGradientBoostingRegressor` (handles missing values natively, fast, strong on tabular data).
- **Features** (all known at forecast time, 24 hours before the target):
  - intensity 24 hours, 48 hours and 7 days earlier
  - mean intensity over the previous 24 hours
  - wind, solar and gas share of generation 24 hours earlier
  - hour of day (sine and cosine encoded) and day of week
- **Target:** actual national intensity.

## Evaluation

- **Split:** time-ordered; the most recent 20% of periods are held out. No shuffling, so the model never sees the future.
- **Metrics:** mean absolute error (MAE), root mean squared error (RMSE) and the share of forecasts within 10% of actual.
- **Baselines:**
  - *Naive:* same half-hour yesterday. The model must beat this to be useful.
  - *NESO:* the grid operator's published forecast, which uses weather forecasts.
- **Slicing:** MAE is reported by time-of-day band to show where the model is weakest.
- **Explainability:** permutation importance, the increase in MAE when each feature is shuffled.

Current figures are written to `reports/metrics.json` on every run and published in each daily GitHub Actions run summary.

## Limitations and risks

- **No weather inputs.** Wind and solar output depend on tomorrow's weather, which this model cannot see. Expect large errors when the weather changes sharply from one day to the next.
- **Short history.** 90 days covers one season. Patterns shift between summer and winter, so the model is retrained daily on recent data.
- **National, not local.** Intensity varies by region. A site in Scotland sees a different grid from one in London.
- **Distribution shift.** New capacity (for example, more offshore wind or battery storage) changes the grid's behaviour over time. Daily retraining partly mitigates this; drift monitoring would be the production answer.
- **Average, not marginal, emissions.** Intensity measures the average emissions of the generation mix. Shifting a large load can change which plants run (marginal emissions), so estimated CO2 savings are indicative.

## Ethical considerations

The model uses only public, aggregate, non-personal data. Its main risk is over-confidence: a user acting on an inaccurate forecast. This is mitigated by benchmarking against NESO, showing error by time of day, and using NESO's forecast for recommendations.

## Ownership

Built and maintained by Courage Labhani Paul as a portfolio project.
