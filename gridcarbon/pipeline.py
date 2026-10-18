"""End-to-end ETL run: extract -> load -> transform -> quality checks -> audit log.

Usage:
    python -m gridcarbon.pipeline --days 90
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gridcarbon.config import DEFAULT_DB_PATH, DEFAULT_HISTORY_DAYS
from gridcarbon.extract import CarbonIntensityClient
from gridcarbon.quality import QualityIssue, run_quality_checks
from gridcarbon.warehouse import (
    connect,
    finish_run,
    start_run,
    upsert_generation,
    upsert_intensity,
)

logger = logging.getLogger("gridcarbon.pipeline")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: int
    intensity_rows: int
    generation_rows: int
    issues: tuple[QualityIssue, ...]

    @property
    def status(self) -> str:
        return "success" if not self.issues else "success_with_warnings"


def floor_to_half_hour(value: datetime) -> datetime:
    """Align a timestamp to the start of its half-hour settlement period."""
    return value.replace(minute=0 if value.minute < 30 else 30, second=0, microsecond=0)


def run_pipeline(
    *,
    days: int = DEFAULT_HISTORY_DAYS,
    db_path: Path = DEFAULT_DB_PATH,
    client: CarbonIntensityClient | None = None,
    now: datetime | None = None,
) -> PipelineResult:
    """Refresh the last ``days`` days of data into the warehouse."""
    if days < 1:
        raise ValueError("days must be at least 1")
    api = client if client is not None else CarbonIntensityClient()
    started_at = now if now is not None else datetime.now(UTC)
    window_end = floor_to_half_hour(started_at)
    window_start = window_end - timedelta(days=days)

    with connect(db_path) as connection:
        run_id = start_run(connection, started_at, window_start, window_end)
        connection.commit()  # record the attempt even if extraction fails
        try:
            logger.info("Extracting %s to %s", window_start.isoformat(), window_end.isoformat())
            intensity = api.intensity_history(window_start, window_end)
            generation = api.generation_history(window_start, window_end)

            intensity_rows = upsert_intensity(connection, intensity, started_at)
            generation_rows = upsert_generation(connection, generation, started_at)
            issues = tuple(run_quality_checks(connection))
            result = PipelineResult(run_id, intensity_rows, generation_rows, issues)

            finish_run(
                connection,
                run_id,
                finished_at=datetime.now(UTC),
                intensity_rows=intensity_rows,
                generation_rows=generation_rows,
                quality_issues=len(issues),
                status=result.status,
            )
        except Exception:
            finish_run(
                connection,
                run_id,
                finished_at=datetime.now(UTC),
                intensity_rows=0,
                generation_rows=0,
                quality_issues=0,
                status="failed",
            )
            connection.commit()
            raise

    for issue in result.issues:
        logger.warning("Quality check '%s': %d rows. %s", issue.check, issue.failing_rows, issue.description)
    logger.info(
        "Run %d finished (%s): %d intensity rows, %d generation rows",
        result.run_id,
        result.status,
        result.intensity_rows,
        result.generation_rows,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the UK grid carbon warehouse.")
    parser.add_argument("--days", type=int, default=DEFAULT_HISTORY_DAYS, help="days of history")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite warehouse path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    days: int = args.days
    db_path: Path = args.db
    run_pipeline(days=days, db_path=db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
