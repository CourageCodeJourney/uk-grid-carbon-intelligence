"""Data quality checks: the governance gate between loading and reporting.

Each check is a SQL query that returns the number of offending rows. A check
that finds problems produces a ``QualityIssue`` instead of crashing, so the
pipeline can still publish data while clearly flagging what is wrong.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from gridcarbon.config import (
    MAX_PLAUSIBLE_INTENSITY,
    MIN_PLAUSIBLE_INTENSITY,
    MIX_SUM_TOLERANCE_PCT,
)


@dataclass(frozen=True, slots=True)
class QualityCheck:
    name: str
    description: str
    sql: str


@dataclass(frozen=True, slots=True)
class QualityIssue:
    check: str
    description: str
    failing_rows: int


CHECKS: Final[tuple[QualityCheck, ...]] = (
    QualityCheck(
        name="intensity_in_plausible_range",
        description=(
            f"Actual intensity outside {MIN_PLAUSIBLE_INTENSITY}-"
            f"{MAX_PLAUSIBLE_INTENSITY} gCO2/kWh"
        ),
        sql=f"""
            SELECT COUNT(*) FROM raw_intensity
            WHERE actual IS NOT NULL
              AND (actual < {MIN_PLAUSIBLE_INTENSITY} OR actual > {MAX_PLAUSIBLE_INTENSITY})
        """,
    ),
    QualityCheck(
        name="generation_mix_sums_to_100",
        description=f"Generation mix does not sum to 100% (+/- {MIX_SUM_TOLERANCE_PCT})",
        sql=f"""
            SELECT COUNT(*) FROM (
                SELECT period_start, SUM(share_pct) AS total
                FROM raw_generation
                GROUP BY period_start
                HAVING ABS(total - 100) > {MIX_SUM_TOLERANCE_PCT}
            )
        """,
    ),
    QualityCheck(
        name="periods_missing_generation_mix",
        description="Intensity periods with no matching generation mix",
        sql="""
            SELECT COUNT(*) FROM raw_intensity AS i
            WHERE NOT EXISTS (
                SELECT 1 FROM raw_generation AS g WHERE g.period_start = i.period_start
            )
        """,
    ),
    QualityCheck(
        name="gaps_in_time_series",
        description="Half-hour periods missing between the first and last period",
        sql="""
            SELECT COALESCE(
                CAST(ROUND((julianday(MAX(period_start)) - julianday(MIN(period_start))) * 48)
                     AS INTEGER) + 1 - COUNT(*),
                0
            )
            FROM raw_intensity
        """,
    ),
    QualityCheck(
        name="past_periods_missing_actual",
        description="Periods older than 24 hours that still have no actual value",
        sql="""
            SELECT COUNT(*) FROM raw_intensity
            WHERE actual IS NULL
              AND julianday(period_start) < julianday('now') - 1
        """,
    ),
)


def run_quality_checks(connection: sqlite3.Connection) -> list[QualityIssue]:
    """Run every check and return the ones that found problems."""
    issues: list[QualityIssue] = []
    for check in CHECKS:
        row = connection.execute(check.sql).fetchone()
        failing = int(row[0]) if row is not None and row[0] is not None else 0
        if failing > 0:
            issues.append(QualityIssue(check.name, check.description, failing))
    return issues
