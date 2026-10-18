from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import pandas as pd

from gridcarbon.extract import CarbonIntensityClient, IntensityRecord
from gridcarbon.forecast import FEATURES, HORIZON, build_features, forecast_ahead, load_periods, train_and_evaluate
from gridcarbon.pipeline import run_pipeline
from gridcarbon.recommend import best_window
from gridcarbon.warehouse import connect
from tests.fixtures import FakeApi, utc


def _record(hour: int, forecast: int) -> IntensityRecord:
    start = utc(2026, 10, 19) + timedelta(minutes=30 * hour)
    return IntensityRecord(start, start + timedelta(minutes=30), forecast, None, "moderate")


class RecommendTest(unittest.TestCase):
    def test_picks_lowest_window(self) -> None:
        values = [300, 300, 100, 120, 300, 300]
        forecast = [_record(i, v) for i, v in enumerate(values)]
        window = best_window(forecast, duration_hours=1, energy_kwh=1000)
        self.assertEqual(window.start, forecast[2].period_start)
        self.assertEqual(window.end, forecast[3].period_end)
        self.assertEqual(window.avg_intensity, 110.0)
        self.assertEqual(window.run_now_intensity, 300.0)
        self.assertEqual(window.co2_saved_kg, 190.0)  # (300 - 110) g/kWh * 1000 kWh
        self.assertAlmostEqual(window.saving_pct, 63.3)

    def test_running_now_can_be_best(self) -> None:
        forecast = [_record(i, v) for i, v in enumerate([50, 200, 200])]
        window = best_window(forecast, duration_hours=0.5, energy_kwh=10)
        self.assertEqual(window.co2_saved_kg, 0.0)

    def test_rejects_jobs_longer_than_forecast(self) -> None:
        with self.assertRaises(ValueError):
            best_window([_record(0, 100)], duration_hours=3, energy_kwh=1)

    def test_rejects_invalid_inputs(self) -> None:
        forecast = [_record(i, 100) for i in range(4)]
        with self.assertRaises(ValueError):
            best_window(forecast, duration_hours=0, energy_kwh=1)
        with self.assertRaises(ValueError):
            best_window(forecast, duration_hours=1, energy_kwh=-5)


class ForecastTest(unittest.TestCase):
    _tmp: ClassVar[tempfile.TemporaryDirectory[str]]
    periods: ClassVar[pd.DataFrame]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        db = Path(cls._tmp.name) / "w.db"
        client = CarbonIntensityClient(FakeApi(), pause_seconds=0)
        run_pipeline(days=45, db_path=db, client=client, now=utc(2026, 10, 18, 12))
        with connect(db) as connection:
            cls.periods = load_periods(connection)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_features_use_only_past_values(self) -> None:
        features = build_features(self.periods)
        row = HORIZON + 10
        self.assertEqual(features["lag_24h"].iloc[row], self.periods["actual"].iloc[row - HORIZON])
        self.assertTrue(features["lag_24h"].iloc[:HORIZON].isna().all())

    def test_model_beats_naive_and_reports_metrics(self) -> None:
        trained = train_and_evaluate(self.periods)
        report = trained.report
        self.assertGreater(report.test_rows, 0)
        self.assertLess(report.model.mae, report.naive.mae)
        self.assertIsNotNone(report.neso)
        self.assertEqual(set(report.feature_importance), set(FEATURES))
        self.assertEqual(len(report.mae_by_time_band), 4)
        self.assertIn('"skill_vs_naive"', report.to_json())

    def test_forecast_ahead_covers_next_24h(self) -> None:
        trained = train_and_evaluate(self.periods)
        ahead = forecast_ahead(self.periods, trained.model)
        self.assertEqual(len(ahead), HORIZON)
        self.assertTrue(ahead.notna().all())
        self.assertGreater(ahead.index[0], self.periods["actual"].last_valid_index())

    def test_refuses_tiny_datasets(self) -> None:
        with self.assertRaises(ValueError):
            train_and_evaluate(self.periods.iloc[:200])


if __name__ == "__main__":
    unittest.main()
