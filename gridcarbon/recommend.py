"""Decision layer: turn a forecast into an action a business can take.

Given a flexible job (e.g. a 3-hour batch run using 500 kWh), find the start
time in the next 48 hours with the lowest average carbon intensity, and
quantify the CO2 saved versus starting immediately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from gridcarbon.extract import IntensityRecord

PERIOD_HOURS = 0.5


@dataclass(frozen=True, slots=True)
class GreenWindow:
    start: datetime
    end: datetime
    avg_intensity: float  # gCO2/kWh across the window
    run_now_intensity: float  # gCO2/kWh if started at the first available period
    worst_intensity: float  # gCO2/kWh for the dirtiest window, for context
    co2_saved_kg: float  # versus running now
    saving_pct: float  # versus running now


def best_window(
    forecast: Sequence[IntensityRecord], *, duration_hours: float, energy_kwh: float
) -> GreenWindow:
    """Find the contiguous window with the lowest mean forecast intensity."""
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if energy_kwh < 0:
        raise ValueError("energy_kwh cannot be negative")

    periods = sorted(
        (r for r in forecast if r.forecast is not None), key=lambda r: r.period_start
    )
    slots = max(1, round(duration_hours / PERIOD_HOURS))
    if len(periods) < slots:
        raise ValueError(
            f"Forecast covers {len(periods) * PERIOD_HOURS:g}h; the job needs {duration_hours:g}h."
        )

    values = [float(r.forecast) for r in periods if r.forecast is not None]
    window_sums = [sum(values[i : i + slots]) for i in range(len(values) - slots + 1)]
    best_i = min(range(len(window_sums)), key=window_sums.__getitem__)

    best_avg = window_sums[best_i] / slots
    now_avg = window_sums[0] / slots
    worst_avg = max(window_sums) / slots
    # gCO2/kWh * kWh = grams; divide by 1000 for kilograms.
    saved_kg = (now_avg - best_avg) * energy_kwh / 1000

    return GreenWindow(
        start=periods[best_i].period_start,
        end=periods[best_i + slots - 1].period_end,
        avg_intensity=round(best_avg, 1),
        run_now_intensity=round(now_avg, 1),
        worst_intensity=round(worst_avg, 1),
        co2_saved_kg=round(saved_kg, 1),
        saving_pct=round((1 - best_avg / now_avg) * 100, 1) if now_avg > 0 else 0.0,
    )
