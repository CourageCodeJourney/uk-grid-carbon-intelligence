"""Central configuration. Every tunable value lives here, not scattered in code."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Public, keyless API run by National Energy System Operator (NESO) and partners.
API_BASE_URL: Final[str] = "https://api.carbonintensity.org.uk"

# The API rejects history requests longer than this, so we fetch in chunks.
MAX_RANGE_DAYS: Final[int] = 14

# Pause between API calls so we stay a polite consumer of a free public service.
REQUEST_PAUSE_SECONDS: Final[float] = 0.5

# The grid reports in half-hour settlement periods: 48 per day.
PERIODS_PER_DAY: Final[int] = 48

DEFAULT_DB_PATH: Final[Path] = Path("data/warehouse.db")
DEFAULT_METRICS_PATH: Final[Path] = Path("reports/metrics.json")
DEFAULT_HISTORY_DAYS: Final[int] = 90

# Fuels the API reports, grouped for analysis.
ALL_FUELS: Final[tuple[str, ...]] = (
    "biomass",
    "coal",
    "gas",
    "hydro",
    "imports",
    "nuclear",
    "other",
    "solar",
    "wind",
)
RENEWABLE_FUELS: Final[tuple[str, ...]] = ("wind", "solar", "hydro")

# Data quality thresholds (see gridcarbon/quality.py).
MIN_PLAUSIBLE_INTENSITY: Final[int] = 0
MAX_PLAUSIBLE_INTENSITY: Final[int] = 1000  # gCO2/kWh; GB has never come close
MIX_SUM_TOLERANCE_PCT: Final[float] = 2.0

UK_TIMEZONE: Final[str] = "Europe/London"
