"""Extract layer: pull data from the GB Carbon Intensity API.

Design choices worth knowing for an interview:
- The HTTP call is injected (``JsonGetter``), so tests run offline with fake data.
- Every response is validated before it enters the warehouse. Bad data fails
  loudly here instead of silently corrupting reports later.
- Long date ranges are split into chunks because the API caps request ranges.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast

import requests

from gridcarbon.config import API_BASE_URL, MAX_RANGE_DAYS, REQUEST_PAUSE_SECONDS

API_TIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%MZ"

# A function that takes an API path (e.g. "/intensity") and returns parsed JSON.
JsonGetter = Callable[[str], dict[str, Any]]


class ApiError(RuntimeError):
    """The API could not be reached or returned an unusable response."""


class SchemaError(ValueError):
    """A response did not match the shape we expect."""


@dataclass(frozen=True, slots=True)
class IntensityRecord:
    """Carbon intensity (gCO2/kWh) for one half-hour period."""

    period_start: datetime
    period_end: datetime
    forecast: int | None
    actual: int | None
    index: str


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Share of national generation from one fuel in one half-hour period."""

    period_start: datetime
    fuel: str
    share_pct: float


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #


def parse_api_time(value: str) -> datetime:
    """Parse the API's timestamp format (e.g. 2026-10-18T12:30Z) as UTC."""
    return datetime.strptime(value, API_TIME_FORMAT).replace(tzinfo=UTC)


def format_api_time(value: datetime) -> str:
    """Format a datetime for use in an API path. Naive datetimes are rejected."""
    if value.tzinfo is None:
        raise ValueError("Datetimes must be timezone-aware (UTC).")
    return value.astimezone(UTC).strftime(API_TIME_FORMAT)


def date_chunks(
    start: datetime, end: datetime, max_days: int = MAX_RANGE_DAYS
) -> Iterator[tuple[datetime, datetime]]:
    """Split [start, end) into consecutive windows no longer than ``max_days``."""
    if end <= start:
        raise ValueError("end must be after start")
    if max_days < 1:
        raise ValueError("max_days must be at least 1")
    step = timedelta(days=max_days)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end


# --------------------------------------------------------------------------- #
# Validation: turn untyped JSON into typed records, or fail clearly
# --------------------------------------------------------------------------- #


def _data_list(payload: dict[str, Any]) -> list[Any]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise SchemaError("Response is missing a 'data' list.")
    return data


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaError(f"'{field}' must be a number or null, got {value!r}")
    return int(value)


def _require_str(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise SchemaError(f"'{key}' must be a string, got {value!r}")
    return value


def _require_dict(item: dict[str, Any], key: str) -> dict[str, Any]:
    value = item.get(key)
    if not isinstance(value, dict):
        raise SchemaError(f"'{key}' must be an object, got {value!r}")
    return cast(dict[str, Any], value)


def parse_intensity(payload: dict[str, Any]) -> list[IntensityRecord]:
    """Validate an /intensity response and convert it to records."""
    records: list[IntensityRecord] = []
    for raw in _data_list(payload):
        if not isinstance(raw, dict):
            raise SchemaError("Each intensity period must be an object.")
        item = cast(dict[str, Any], raw)
        values = _require_dict(item, "intensity")
        index = values.get("index")
        records.append(
            IntensityRecord(
                period_start=parse_api_time(_require_str(item, "from")),
                period_end=parse_api_time(_require_str(item, "to")),
                forecast=_optional_int(values.get("forecast"), "forecast"),
                actual=_optional_int(values.get("actual"), "actual"),
                index=index if isinstance(index, str) else "unknown",
            )
        )
    return records


def parse_generation(payload: dict[str, Any]) -> list[GenerationRecord]:
    """Validate a /generation response and convert it to one record per fuel."""
    records: list[GenerationRecord] = []
    for raw in _data_list(payload):
        if not isinstance(raw, dict):
            raise SchemaError("Each generation period must be an object.")
        item = cast(dict[str, Any], raw)
        start = parse_api_time(_require_str(item, "from"))
        mix = item.get("generationmix")
        if not isinstance(mix, list):
            raise SchemaError("'generationmix' must be a list.")
        for raw_fuel in mix:
            if not isinstance(raw_fuel, dict):
                raise SchemaError("Each generation mix entry must be an object.")
            fuel_item = cast(dict[str, Any], raw_fuel)
            share = fuel_item.get("perc")
            if isinstance(share, bool) or not isinstance(share, int | float):
                raise SchemaError(f"'perc' must be a number, got {share!r}")
            records.append(
                GenerationRecord(
                    period_start=start,
                    fuel=_require_str(fuel_item, "fuel"),
                    share_pct=float(share),
                )
            )
    return records


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def make_http_getter(
    session: requests.Session | None = None,
    *,
    base_url: str = API_BASE_URL,
    timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> JsonGetter:
    """Build a JSON getter that retries transient failures with backoff."""
    http = session if session is not None else requests.Session()
    http.headers.update({"Accept": "application/json"})

    def get_json(path: str) -> dict[str, Any]:
        url = f"{base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = http.get(url, timeout=timeout_seconds)
            except requests.RequestException as exc:  # network blip: retry
                last_error = exc
            else:
                transient = response.status_code == 429 or response.status_code >= 500
                if not transient:
                    if response.status_code >= 400:
                        raise ApiError(f"GET {url} returned HTTP {response.status_code}")
                    try:
                        payload: object = response.json()
                    except ValueError as exc:
                        raise ApiError(f"GET {url} did not return JSON") from exc
                    if not isinstance(payload, dict):
                        raise ApiError(f"GET {url} returned JSON that is not an object")
                    return cast(dict[str, Any], payload)
                last_error = ApiError(f"HTTP {response.status_code}")
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
        raise ApiError(f"GET {url} failed after {max_attempts} attempts") from last_error

    return get_json


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class CarbonIntensityClient:
    """Typed client for the endpoints this project uses."""

    def __init__(
        self,
        get_json: JsonGetter | None = None,
        *,
        pause_seconds: float = REQUEST_PAUSE_SECONDS,
    ) -> None:
        self._get_json: JsonGetter = get_json if get_json is not None else make_http_getter()
        self._pause_seconds = pause_seconds

    def _pause(self) -> None:
        if self._pause_seconds > 0:
            time.sleep(self._pause_seconds)

    def intensity_history(self, start: datetime, end: datetime) -> list[IntensityRecord]:
        """Forecast and actual national intensity for every period in [start, end)."""
        records: list[IntensityRecord] = []
        for chunk_start, chunk_end in date_chunks(start, end):
            path = f"/intensity/{format_api_time(chunk_start)}/{format_api_time(chunk_end)}"
            records.extend(parse_intensity(self._get_json(path)))
            self._pause()
        return records

    def generation_history(self, start: datetime, end: datetime) -> list[GenerationRecord]:
        """National generation mix for every period in [start, end)."""
        records: list[GenerationRecord] = []
        for chunk_start, chunk_end in date_chunks(start, end):
            path = f"/generation/{format_api_time(chunk_start)}/{format_api_time(chunk_end)}"
            records.extend(parse_generation(self._get_json(path)))
            self._pause()
        return records

    def forecast_next_48h(self, start: datetime | None = None) -> list[IntensityRecord]:
        """NESO's official forecast for the next 48 hours from ``start`` (default: now)."""
        origin = start if start is not None else datetime.now(UTC)
        path = f"/intensity/{format_api_time(origin)}/fw48h"
        return parse_intensity(self._get_json(path))
