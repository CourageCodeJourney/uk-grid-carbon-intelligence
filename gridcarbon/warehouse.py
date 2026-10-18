"""Load and transform layers: a small SQL warehouse in SQLite.

The warehouse follows the same layered (medallion) pattern used on platforms
such as Databricks and Snowflake, at laptop scale:

    raw_*        exactly what the API returned, upserted idempotently   (bronze)
    clean_*      typed, pivoted, joined, analysis-ready                  (silver)
    analytics_*  business-level aggregates that power the dashboard      (gold)

``pipeline_runs`` records every run for lineage and auditability.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from gridcarbon.config import ALL_FUELS, RENEWABLE_FUELS
from gridcarbon.extract import GenerationRecord, IntensityRecord

DB_TIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"

SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS raw_intensity (
    period_start     TEXT PRIMARY KEY,
    period_end       TEXT NOT NULL,
    forecast         INTEGER,
    actual           INTEGER,
    intensity_index  TEXT NOT NULL,
    loaded_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_generation (
    period_start  TEXT NOT NULL,
    fuel          TEXT NOT NULL,
    share_pct     REAL NOT NULL,
    loaded_at     TEXT NOT NULL,
    PRIMARY KEY (period_start, fuel)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    window_start     TEXT NOT NULL,
    window_end       TEXT NOT NULL,
    intensity_rows   INTEGER NOT NULL DEFAULT 0,
    generation_rows  INTEGER NOT NULL DEFAULT 0,
    quality_issues   INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL
);
"""


def _fuel_columns_sql() -> str:
    return ",\n    ".join(
        f"MAX(CASE WHEN fuel = '{fuel}' THEN share_pct END) AS {fuel}_pct" for fuel in ALL_FUELS
    )


def _renewable_sum_sql() -> str:
    return " + ".join(f"COALESCE(g.{fuel}_pct, 0)" for fuel in RENEWABLE_FUELS)


def _views_sql() -> str:
    return f"""
DROP VIEW IF EXISTS analytics_hourly_profile;
DROP VIEW IF EXISTS analytics_daily;
DROP VIEW IF EXISTS clean_periods;
DROP VIEW IF EXISTS clean_generation_wide;

-- Silver: one row per period, one column per fuel.
CREATE VIEW clean_generation_wide AS
SELECT
    period_start,
    {_fuel_columns_sql()}
FROM raw_generation
GROUP BY period_start;

-- Silver: intensity joined to the generation mix, with calendar features.
CREATE VIEW clean_periods AS
SELECT
    i.period_start,
    i.forecast            AS neso_forecast,
    i.actual,
    i.intensity_index,
    CAST(strftime('%H', i.period_start) AS INTEGER) AS hour_utc,
    CAST(strftime('%w', i.period_start) AS INTEGER) AS day_of_week,
    date(i.period_start)  AS period_date,
    {", ".join(f"g.{fuel}_pct" for fuel in ALL_FUELS)},
    {_renewable_sum_sql()} AS renewable_pct
FROM raw_intensity AS i
LEFT JOIN clean_generation_wide AS g USING (period_start)
WHERE i.actual IS NOT NULL;

-- Gold: daily trend for executives.
CREATE VIEW analytics_daily AS
SELECT
    period_date,
    ROUND(AVG(actual), 1)        AS avg_intensity,
    MIN(actual)                  AS min_intensity,
    MAX(actual)                  AS max_intensity,
    ROUND(AVG(renewable_pct), 1) AS avg_renewable_pct,
    COUNT(*)                     AS periods
FROM clean_periods
GROUP BY period_date
ORDER BY period_date;

-- Gold: average intensity by hour of day, i.e. when the grid is usually greenest.
CREATE VIEW analytics_hourly_profile AS
SELECT
    hour_utc,
    ROUND(AVG(actual), 1)        AS avg_intensity,
    ROUND(AVG(renewable_pct), 1) AS avg_renewable_pct
FROM clean_periods
GROUP BY hour_utc
ORDER BY hour_utc;
"""


def to_db_time(value: datetime) -> str:
    """Store every timestamp as UTC ISO-8601 text so SQLite can sort and filter it."""
    if value.tzinfo is None:
        raise ValueError("Datetimes must be timezone-aware (UTC).")
    return value.astimezone(UTC).strftime(DB_TIME_FORMAT)


def from_db_time(value: str) -> datetime:
    return datetime.strptime(value, DB_TIME_FORMAT).replace(tzinfo=UTC)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the warehouse, creating it and its schema if needed. Commits on success."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_SQL)
        connection.executescript(_views_sql())
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def upsert_intensity(
    connection: sqlite3.Connection, records: Iterable[IntensityRecord], loaded_at: datetime
) -> int:
    """Insert or update intensity rows. Re-running the same window is safe (idempotent)."""
    stamp = to_db_time(loaded_at)
    rows = [
        (
            to_db_time(r.period_start),
            to_db_time(r.period_end),
            r.forecast,
            r.actual,
            r.index,
            stamp,
        )
        for r in records
    ]
    connection.executemany(
        """
        INSERT INTO raw_intensity
            (period_start, period_end, forecast, actual, intensity_index, loaded_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (period_start) DO UPDATE SET
            period_end      = excluded.period_end,
            forecast        = excluded.forecast,
            actual          = COALESCE(excluded.actual, raw_intensity.actual),
            intensity_index = excluded.intensity_index,
            loaded_at       = excluded.loaded_at
        """,
        rows,
    )
    return len(rows)


def upsert_generation(
    connection: sqlite3.Connection, records: Iterable[GenerationRecord], loaded_at: datetime
) -> int:
    """Insert or update generation-mix rows (one per period and fuel)."""
    stamp = to_db_time(loaded_at)
    rows = [(to_db_time(r.period_start), r.fuel, r.share_pct, stamp) for r in records]
    connection.executemany(
        """
        INSERT INTO raw_generation (period_start, fuel, share_pct, loaded_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (period_start, fuel) DO UPDATE SET
            share_pct = excluded.share_pct,
            loaded_at = excluded.loaded_at
        """,
        rows,
    )
    return len(rows)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: int
    started_at: datetime
    finished_at: datetime | None
    intensity_rows: int
    generation_rows: int
    quality_issues: int
    status: str


def start_run(
    connection: sqlite3.Connection, started_at: datetime, window_start: datetime, window_end: datetime
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO pipeline_runs (started_at, window_start, window_end, status)
        VALUES (?, ?, ?, 'running')
        """,
        (to_db_time(started_at), to_db_time(window_start), to_db_time(window_end)),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("Could not record pipeline run.")
    return cursor.lastrowid


def finish_run(
    connection: sqlite3.Connection,
    run_id: int,
    *,
    finished_at: datetime,
    intensity_rows: int,
    generation_rows: int,
    quality_issues: int,
    status: str,
) -> None:
    connection.execute(
        """
        UPDATE pipeline_runs
        SET finished_at = ?, intensity_rows = ?, generation_rows = ?,
            quality_issues = ?, status = ?
        WHERE run_id = ?
        """,
        (to_db_time(finished_at), intensity_rows, generation_rows, quality_issues, status, run_id),
    )


def latest_successful_run(connection: sqlite3.Connection) -> RunSummary | None:
    row = connection.execute(
        """
        SELECT run_id, started_at, finished_at, intensity_rows, generation_rows,
               quality_issues, status
        FROM pipeline_runs
        WHERE status IN ('success', 'success_with_warnings')
        ORDER BY run_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    run_id, started, finished, i_rows, g_rows, issues, status = row
    return RunSummary(
        run_id=int(run_id),
        started_at=from_db_time(str(started)),
        finished_at=from_db_time(str(finished)) if finished is not None else None,
        intensity_rows=int(i_rows),
        generation_rows=int(g_rows),
        quality_issues=int(issues),
        status=str(status),
    )
