"""Machine learning: forecast GB carbon intensity 24 hours ahead.

Question the model answers: "Standing here now, what will intensity be at the
same half-hour tomorrow?" That is the horizon a business needs to schedule
tomorrow's energy-heavy work.

The model is judged against two honest baselines:
- naive:  "tomorrow looks like today" (intensity 24 hours earlier)
- NESO:   the grid operator's own published forecast, which uses weather data
          we do not have. Beating naive proves skill; NESO is the bar to aim for.

Usage:
    python -m gridcarbon.forecast
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from gridcarbon.config import DEFAULT_DB_PATH, DEFAULT_METRICS_PATH, PERIODS_PER_DAY
from gridcarbon.warehouse import connect

logger = logging.getLogger("gridcarbon.forecast")

HORIZON: Final[int] = PERIODS_PER_DAY  # 48 half-hours = 24 hours ahead
WEEK: Final[int] = PERIODS_PER_DAY * 7

FEATURES: Final[tuple[str, ...]] = (
    "lag_24h",
    "lag_48h",
    "lag_7d",
    "mean_prev_24h",
    "wind_lag_24h",
    "solar_lag_24h",
    "gas_lag_24h",
    "hour_sin",
    "hour_cos",
    "day_of_week",
)

TIME_BANDS: Final[dict[str, tuple[int, int]]] = {
    "night (00-06 UTC)": (0, 6),
    "morning (06-12 UTC)": (6, 12),
    "afternoon (12-18 UTC)": (12, 18),
    "evening (18-24 UTC)": (18, 24),
}

MIN_TRAINING_ROWS: Final[int] = PERIODS_PER_DAY * 7


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    mae: float
    rmse: float
    within_10pct: float  # share of predictions within 10% of actual

    @staticmethod
    def of(actual: pd.Series, predicted: pd.Series) -> ErrorMetrics:
        mask = actual.notna() & predicted.notna()
        a = actual[mask].to_numpy(dtype=float)
        p = predicted[mask].to_numpy(dtype=float)
        if a.size == 0:
            raise ValueError("No overlapping values to score.")
        err = p - a
        return ErrorMetrics(
            mae=round(float(np.mean(np.abs(err))), 2),
            rmse=round(float(np.sqrt(np.mean(err**2))), 2),
            within_10pct=round(float(np.mean(np.abs(err) <= 0.1 * np.abs(a))), 3),
        )


@dataclass(frozen=True, slots=True)
class ForecastReport:
    train_rows: int
    test_rows: int
    test_start: str
    test_end: str
    model: ErrorMetrics
    naive: ErrorMetrics
    neso: ErrorMetrics | None
    skill_vs_naive: float  # 1 - MAE_model / MAE_naive; above 0 means better than naive
    mae_by_time_band: dict[str, float]
    feature_importance: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True, slots=True)
class TrainedForecaster:
    model: HistGradientBoostingRegressor
    report: ForecastReport


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def load_periods(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load the silver table onto a complete half-hourly index (gaps become NaN)."""
    frame = pd.read_sql_query(
        """
        SELECT period_start, actual, neso_forecast, wind_pct, solar_pct, gas_pct
        FROM clean_periods
        ORDER BY period_start
        """,
        connection,
    )
    if frame.empty:
        raise ValueError("The warehouse has no data. Run the pipeline first.")
    frame["period_start"] = pd.to_datetime(frame["period_start"], utc=True)
    frame = frame.set_index("period_start")
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="30min")
    return frame.reindex(full_index)


def build_features(periods: pd.DataFrame, extend_by: int = 0) -> pd.DataFrame:
    """Create model features. Everything is shifted so only past data is used.

    ``extend_by`` adds future rows (with no actual yet) so we can forecast ahead.
    """
    frame = periods.copy()
    if extend_by > 0:
        last = frame.index.max()
        future = pd.date_range(last + pd.Timedelta(minutes=30), periods=extend_by, freq="30min")
        frame = pd.concat([frame, pd.DataFrame(index=future, columns=frame.columns, dtype=float)])

    actual = frame["actual"].astype(float)
    features = pd.DataFrame(index=frame.index)
    features["lag_24h"] = actual.shift(HORIZON)
    features["lag_48h"] = actual.shift(HORIZON * 2)
    features["lag_7d"] = actual.shift(WEEK)
    features["mean_prev_24h"] = actual.shift(HORIZON).rolling(PERIODS_PER_DAY, min_periods=24).mean()
    features["wind_lag_24h"] = frame["wind_pct"].astype(float).shift(HORIZON)
    features["solar_lag_24h"] = frame["solar_pct"].astype(float).shift(HORIZON)
    features["gas_lag_24h"] = frame["gas_pct"].astype(float).shift(HORIZON)

    index = pd.DatetimeIndex(features.index)
    hour = index.hour + index.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_of_week"] = index.dayofweek

    features["actual"] = actual
    features["neso_forecast"] = frame["neso_forecast"].astype(float)
    return features


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )


def train_and_evaluate(periods: pd.DataFrame, test_fraction: float = 0.2) -> TrainedForecaster:
    """Time-ordered train/test split (no shuffling: the future must not leak into training)."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    data = build_features(periods)
    data = data[data["actual"].notna() & data["lag_24h"].notna()]
    if len(data) < MIN_TRAINING_ROWS:
        raise ValueError(f"Need at least {MIN_TRAINING_ROWS} usable rows, have {len(data)}.")

    split = int(len(data) * (1 - test_fraction))
    train, test = data.iloc[:split], data.iloc[split:]
    feature_list = list(FEATURES)

    model = _new_model()
    model.fit(train[feature_list], train["actual"])
    predicted = pd.Series(model.predict(test[feature_list]), index=test.index)

    model_metrics = ErrorMetrics.of(test["actual"], predicted)
    naive_metrics = ErrorMetrics.of(test["actual"], test["lag_24h"])
    neso_metrics = (
        ErrorMetrics.of(test["actual"], test["neso_forecast"])
        if test["neso_forecast"].notna().any()
        else None
    )

    test_index = pd.DatetimeIndex(test.index)
    hours = test_index.hour
    by_band: dict[str, float] = {}
    for band, (lo, hi) in TIME_BANDS.items():
        mask = (hours >= lo) & (hours < hi)
        if mask.any():
            band_err = (predicted[mask] - test["actual"][mask]).abs()
            by_band[band] = round(float(band_err.mean()), 2)

    report = ForecastReport(
        train_rows=len(train),
        test_rows=len(test),
        test_start=test_index.min().isoformat(),
        test_end=test_index.max().isoformat(),
        model=model_metrics,
        naive=naive_metrics,
        neso=neso_metrics,
        skill_vs_naive=round(1 - model_metrics.mae / naive_metrics.mae, 3)
        if naive_metrics.mae > 0
        else 0.0,
        mae_by_time_band=by_band,
        feature_importance=_permutation_importance(model, test, feature_list),
    )

    # Refit on all data so live forecasts use the most recent patterns.
    final_model = _new_model()
    final_model.fit(data[feature_list], data["actual"])
    return TrainedForecaster(model=final_model, report=report)


def _permutation_importance(
    model: HistGradientBoostingRegressor, test: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """How much MAE worsens when each feature is shuffled: a simple explainability check."""
    rng = np.random.default_rng(42)
    actual = test["actual"].to_numpy(dtype=float)
    base_mae = float(np.mean(np.abs(model.predict(test[features]) - actual)))
    importance: dict[str, float] = {}
    for feature in features:
        shuffled = test[features].copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        mae = float(np.mean(np.abs(model.predict(shuffled) - actual)))
        importance[feature] = round(mae - base_mae, 2)
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


def forecast_ahead(periods: pd.DataFrame, model: HistGradientBoostingRegressor) -> pd.Series:
    """Predict intensity for the 24 hours after the last observed actual."""
    last_actual = periods["actual"].last_valid_index()
    if last_actual is None:
        raise ValueError("No actual values to forecast from.")
    observed = periods.loc[:last_actual]  # keep the regular half-hour index intact
    features = build_features(observed, extend_by=HORIZON)
    future = features.iloc[-HORIZON:]
    values = model.predict(future[list(FEATURES)])
    return pd.Series(values, index=future.index, name="model_forecast")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the 24h forecaster.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path: Path = args.db
    metrics_path: Path = args.metrics
    with connect(db_path) as connection:
        periods = load_periods(connection)
    trained = train_and_evaluate(periods)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(trained.report.to_json(), encoding="utf-8")

    r = trained.report
    logger.info("Test MAE (gCO2/kWh): model %.1f | naive %.1f | NESO %s",
                r.model.mae, r.naive.mae, f"{r.neso.mae:.1f}" if r.neso else "n/a")
    logger.info("Skill vs naive: %.0f%%. Metrics saved to %s", r.skill_vs_naive * 100, metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
