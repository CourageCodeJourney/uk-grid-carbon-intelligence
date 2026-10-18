from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from gridcarbon.extract import CarbonIntensityClient
from gridcarbon.pipeline import floor_to_half_hour, run_pipeline
from gridcarbon.quality import run_quality_checks
from gridcarbon.warehouse import connect, latest_successful_run
from tests.fixtures import FakeApi, utc


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "warehouse.db"
        self.client = CarbonIntensityClient(FakeApi(), pause_seconds=0)
        self.now = utc(2026, 10, 18, 22, 47)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _count(self, sql: str) -> int:
        with connect(self.db) as connection:
            row = connection.execute(sql).fetchone()
        return int(row[0])

    def test_floor_to_half_hour(self) -> None:
        self.assertEqual(floor_to_half_hour(utc(2026, 1, 1, 9, 47)), utc(2026, 1, 1, 9, 30))
        self.assertEqual(floor_to_half_hour(utc(2026, 1, 1, 9, 12)), utc(2026, 1, 1, 9, 0))

    def test_end_to_end_run_populates_every_layer(self) -> None:
        result = run_pipeline(days=10, db_path=self.db, client=self.client, now=self.now)
        self.assertEqual(result.status, "success")
        periods = self._count("SELECT COUNT(*) FROM raw_intensity")
        self.assertGreaterEqual(periods, 10 * 48)
        self.assertEqual(self._count("SELECT COUNT(*) FROM raw_generation"), periods * 9)
        self.assertEqual(self._count("SELECT COUNT(*) FROM clean_periods"), periods)
        self.assertGreaterEqual(self._count("SELECT COUNT(*) FROM analytics_daily"), 10)
        self.assertEqual(self._count("SELECT COUNT(*) FROM analytics_hourly_profile"), 24)

    def test_rerun_is_idempotent(self) -> None:
        run_pipeline(days=5, db_path=self.db, client=self.client, now=self.now)
        first = self._count("SELECT COUNT(*) FROM raw_intensity")
        run_pipeline(days=5, db_path=self.db, client=self.client, now=self.now)
        self.assertEqual(self._count("SELECT COUNT(*) FROM raw_intensity"), first)
        self.assertEqual(self._count("SELECT COUNT(*) FROM pipeline_runs"), 2)

    def test_run_is_audited(self) -> None:
        run_pipeline(days=3, db_path=self.db, client=self.client, now=self.now)
        with connect(self.db) as connection:
            last = latest_successful_run(connection)
        assert last is not None
        self.assertEqual(last.status, "success")
        self.assertGreater(last.intensity_rows, 0)

    def test_failed_run_is_recorded(self) -> None:
        def broken(path: str) -> dict[str, object]:
            raise RuntimeError("API down")

        with self.assertRaises(RuntimeError):
            run_pipeline(
                days=1, db_path=self.db, client=CarbonIntensityClient(broken, pause_seconds=0), now=self.now
            )
        with connect(self.db) as connection:
            row = connection.execute("SELECT status FROM pipeline_runs").fetchone()
        self.assertEqual(row[0], "failed")


class QualityTest(unittest.TestCase):
    def test_checks_flag_bad_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "w.db"
            client = CarbonIntensityClient(FakeApi(), pause_seconds=0)
            run_pipeline(days=2, db_path=db, client=client, now=utc(2026, 10, 18, 12))
            with connect(db) as connection:
                self.assertEqual(run_quality_checks(connection), [])
                connection.execute("UPDATE raw_intensity SET actual = 5000 WHERE rowid = 1")
                connection.execute(
                    "DELETE FROM raw_generation WHERE fuel = 'gas' AND rowid IN "
                    "(SELECT rowid FROM raw_generation WHERE fuel = 'gas' LIMIT 3)"
                )
                found = {issue.check: issue.failing_rows for issue in run_quality_checks(connection)}
            self.assertEqual(found.get("intensity_in_plausible_range"), 1)
            self.assertEqual(found.get("generation_mix_sums_to_100"), 3)

    def test_gap_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "w.db"
            client = CarbonIntensityClient(FakeApi(), pause_seconds=0)
            run_pipeline(days=2, db_path=db, client=client, now=utc(2026, 10, 18, 12))
            with connect(db) as connection:
                connection.execute(
                    "DELETE FROM raw_intensity WHERE period_start IN "
                    "(SELECT period_start FROM raw_intensity ORDER BY period_start LIMIT 2 OFFSET 10)"
                )
                found = {issue.check: issue.failing_rows for issue in run_quality_checks(connection)}
            self.assertEqual(found.get("gaps_in_time_series"), 2)


class WarehouseTest(unittest.TestCase):
    def test_connect_rolls_back_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "w.db"
            with self.assertRaises(sqlite3.OperationalError):
                with connect(db) as connection:
                    connection.execute(
                        "INSERT INTO raw_generation VALUES ('2026-01-01T00:00:00Z', 'wind', 1, 'x')"
                    )
                    connection.execute("SELECT * FROM table_that_does_not_exist")
            with connect(db) as connection:
                row = connection.execute("SELECT COUNT(*) FROM raw_generation").fetchone()
            self.assertEqual(row[0], 0)


if __name__ == "__main__":
    unittest.main()
