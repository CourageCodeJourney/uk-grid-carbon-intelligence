"""Synthetic API responses so the whole pipeline can be tested offline.

The shapes match the real Carbon Intensity API exactly; the numbers follow a
realistic pattern (evening peak, midday solar dip, wind-driven swings).
"""

from __future__ import annotations

import math
import random
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from gridcarbon.extract import API_TIME_FORMAT, parse_api_time

_RANGE_PATH = re.compile(r"^/(intensity|generation)/([^/]+)/([^/]+)$")
_FW48_PATH = re.compile(r"^/intensity/([^/]+)/fw48h$")


def _period_values(start: datetime, rng: random.Random, wind: float) -> tuple[int, dict[str, float]]:
    hour = start.hour + start.minute / 60
    solar = max(0.0, math.sin((hour - 6) / 12 * math.pi)) * 12  # daytime bell, 0-12%
    nuclear = 14.0
    imports = 10.0
    biomass = 6.0
    hydro = 2.0
    gas = max(5.0, 100 - wind - solar - nuclear - imports - biomass - hydro - 1.0)
    other = 100 - (wind + solar + nuclear + imports + biomass + hydro + gas)
    mix = {
        "biomass": biomass,
        "coal": 0.0,
        "gas": gas,
        "hydro": hydro,
        "imports": imports,
        "nuclear": nuclear,
        "other": round(other, 2),
        "solar": round(solar, 2),
        "wind": round(wind, 2),
    }
    evening_peak = 25 * math.exp(-((hour - 17.5) ** 2) / 4)
    intensity = 40 + gas * 3.6 + evening_peak + rng.gauss(0, 6)
    return max(10, int(intensity)), mix


def build_series(
    start: datetime, end: datetime, seed: int = 7
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (intensity_periods, generation_periods) for [start, end).

    Values depend only on the timestamp (and seed), so overlapping requests
    return identical data, just like the real API.
    """
    intensity: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    cursor = start
    while cursor < end:
        rng = random.Random(seed * 1_000_003 + int(cursor.timestamp()))
        days = cursor.timestamp() / 86_400
        wind = 30 + 18 * math.sin(2 * math.pi * days / 3.7) + 8 * math.sin(days * 1.3) + rng.gauss(0, 2)
        wind = min(60.0, max(3.0, wind))
        actual, mix = _period_values(cursor, rng, wind)
        nxt = cursor + timedelta(minutes=30)
        frm, to = cursor.strftime(API_TIME_FORMAT), nxt.strftime(API_TIME_FORMAT)
        intensity.append(
            {
                "from": frm,
                "to": to,
                "intensity": {
                    "forecast": actual + int(rng.gauss(0, 8)),
                    "actual": actual,
                    "index": "moderate",
                },
            }
        )
        generation.append(
            {
                "from": frm,
                "to": to,
                "generationmix": [{"fuel": f, "perc": p} for f, p in mix.items()],
            }
        )
        cursor = nxt
    return intensity, generation


class FakeApi:
    """Stands in for the HTTP getter. Records every path it was asked for."""

    def __init__(self, seed: int = 7) -> None:
        self.calls: list[str] = []
        self._seed = seed

    def __call__(self, path: str) -> dict[str, Any]:
        self.calls.append(path)
        forward = _FW48_PATH.match(path)
        if forward:
            start = parse_api_time(forward.group(1))
            intensity, _ = build_series(start, start + timedelta(hours=48), self._seed)
            for period in intensity:
                period["intensity"]["actual"] = None
            return {"data": intensity}
        match = _RANGE_PATH.match(path)
        if not match:
            raise AssertionError(f"Unexpected path {path}")
        kind, frm, to = match.groups()
        # The real API is end-inclusive; mimic that so de-duplication gets tested.
        start = parse_api_time(frm)
        end = parse_api_time(to) + timedelta(minutes=30)
        intensity, generation = build_series(start, end, self._seed)
        return {"data": intensity if kind == "intensity" else generation}


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)
