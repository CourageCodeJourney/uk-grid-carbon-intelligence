from __future__ import annotations

import unittest
from datetime import timedelta
from typing import Any

from gridcarbon.extract import (
    CarbonIntensityClient,
    SchemaError,
    date_chunks,
    format_api_time,
    parse_api_time,
    parse_generation,
    parse_intensity,
)
from tests.fixtures import FakeApi, utc


class TimeHelpersTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        value = utc(2026, 10, 18, 12, 30)
        self.assertEqual(format_api_time(value), "2026-10-18T12:30Z")
        self.assertEqual(parse_api_time("2026-10-18T12:30Z"), value)

    def test_naive_datetime_rejected(self) -> None:
        with self.assertRaises(ValueError):
            format_api_time(utc(2026, 1, 1).replace(tzinfo=None))

    def test_chunks_cover_range_without_gaps(self) -> None:
        start, end = utc(2026, 1, 1), utc(2026, 2, 15)
        chunks = list(date_chunks(start, end, max_days=14))
        self.assertEqual(chunks[0][0], start)
        self.assertEqual(chunks[-1][1], end)
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            self.assertEqual(prev_end, next_start)
        self.assertTrue(all(b - a <= timedelta(days=14) for a, b in chunks))

    def test_chunks_reject_empty_range(self) -> None:
        with self.assertRaises(ValueError):
            list(date_chunks(utc(2026, 1, 2), utc(2026, 1, 1)))


class ParsingTest(unittest.TestCase):
    def test_parse_intensity_handles_null_actual(self) -> None:
        payload: dict[str, Any] = {
            "data": [
                {
                    "from": "2026-10-18T12:00Z",
                    "to": "2026-10-18T12:30Z",
                    "intensity": {"forecast": 150, "actual": None, "index": "moderate"},
                }
            ]
        }
        [record] = parse_intensity(payload)
        self.assertEqual(record.forecast, 150)
        self.assertIsNone(record.actual)

    def test_parse_intensity_rejects_bad_shape(self) -> None:
        with self.assertRaises(SchemaError):
            parse_intensity({"data": [{"from": "2026-10-18T12:00Z"}]})
        with self.assertRaises(SchemaError):
            parse_intensity({"error": "boom"})

    def test_parse_intensity_rejects_non_numeric_values(self) -> None:
        bad = {
            "data": [
                {
                    "from": "2026-10-18T12:00Z",
                    "to": "2026-10-18T12:30Z",
                    "intensity": {"forecast": "high", "actual": 1, "index": "low"},
                }
            ]
        }
        with self.assertRaises(SchemaError):
            parse_intensity(bad)

    def test_parse_generation_one_record_per_fuel(self) -> None:
        payload: dict[str, Any] = {
            "data": [
                {
                    "from": "2026-10-18T12:00Z",
                    "to": "2026-10-18T12:30Z",
                    "generationmix": [{"fuel": "wind", "perc": 40}, {"fuel": "gas", "perc": 60.0}],
                }
            ]
        }
        records = parse_generation(payload)
        self.assertEqual([r.fuel for r in records], ["wind", "gas"])
        self.assertEqual(records[0].share_pct, 40.0)


class ClientTest(unittest.TestCase):
    def test_history_is_fetched_in_chunks(self) -> None:
        api = FakeApi()
        client = CarbonIntensityClient(api, pause_seconds=0)
        records = client.intensity_history(utc(2026, 1, 1), utc(2026, 1, 31))
        self.assertEqual(len(api.calls), 3)  # 14 + 14 + 2 days
        self.assertGreaterEqual(len(records), 30 * 48)

    def test_forecast_48h(self) -> None:
        client = CarbonIntensityClient(FakeApi(), pause_seconds=0)
        forecast = client.forecast_next_48h(utc(2026, 10, 18, 12))
        self.assertEqual(len(forecast), 96)
        self.assertTrue(all(r.forecast is not None for r in forecast))


if __name__ == "__main__":
    unittest.main()
