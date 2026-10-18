"""UK Grid Carbon Intelligence: dashboard for decision-makers.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from gridcarbon.config import DEFAULT_DB_PATH, DEFAULT_HISTORY_DAYS, UK_TIMEZONE
from gridcarbon.extract import ApiError, CarbonIntensityClient, IntensityRecord
from gridcarbon.forecast import TrainedForecaster, forecast_ahead, load_periods, train_and_evaluate
from gridcarbon.pipeline import run_pipeline
from gridcarbon.quality import QualityIssue, run_quality_checks
from gridcarbon.recommend import GreenWindow, best_window
from gridcarbon.warehouse import RunSummary, connect, latest_successful_run

UK = ZoneInfo(UK_TIMEZONE)
REFRESH_AFTER = timedelta(hours=6)

# Categorical slots in fixed order (validated colourblind-safe palette).
SERIES_1, SERIES_2, SERIES_3, SERIES_4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
HIGHLIGHT = "rgba(27, 175, 122, 0.18)"

st.set_page_config(page_title="UK Grid Carbon Intelligence", page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------- #
# Data access (cached so the API and model are not hit on every click)
# --------------------------------------------------------------------------- #


@st.cache_resource(ttl=timedelta(hours=1), show_spinner="Refreshing grid data...")
def ensure_warehouse(days: int, db_path: str) -> RunSummary | None:
    """Run the pipeline if the warehouse is missing or stale. Returns the latest run."""
    path = Path(db_path)
    with connect(path) as connection:
        last = latest_successful_run(connection)
    fresh = last is not None and datetime.now(UTC) - last.started_at < REFRESH_AFTER
    if not fresh:
        run_pipeline(days=days, db_path=path)
        with connect(path) as connection:
            last = latest_successful_run(connection)
    return last


@st.cache_data(ttl=timedelta(minutes=30))
def read_table(db_path: str, sql: str) -> pd.DataFrame:
    with connect(Path(db_path)) as connection:
        return pd.read_sql_query(sql, connection)


@st.cache_data(ttl=timedelta(minutes=30))
def read_periods(db_path: str) -> pd.DataFrame:
    with connect(Path(db_path)) as connection:
        return load_periods(connection)


@st.cache_data(ttl=timedelta(minutes=30))
def read_quality(db_path: str) -> list[QualityIssue]:
    with connect(Path(db_path)) as connection:
        return run_quality_checks(connection)


@st.cache_resource(ttl=timedelta(hours=6), show_spinner="Training forecast model...")
def train_model(db_path: str) -> TrainedForecaster:
    return train_and_evaluate(read_periods(db_path))


@st.cache_data(ttl=timedelta(minutes=15), show_spinner="Fetching NESO 48-hour forecast...")
def live_forecast() -> list[IntensityRecord]:
    return CarbonIntensityClient().forecast_next_48h()


def uk_time(value: datetime) -> str:
    return value.astimezone(UK).strftime("%a %d %b, %H:%M")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def forecast_chart(
    forecast: list[IntensityRecord], window: GreenWindow, model_line: pd.Series | None
) -> alt.LayerChart:
    neso = pd.DataFrame(
        {
            "time": [r.period_start.astimezone(UK).replace(tzinfo=None) for r in forecast],
            "intensity": [r.forecast for r in forecast],
            "series": "NESO official forecast",
        }
    )
    frames = [neso]
    if model_line is not None and not model_line.empty:
        index = pd.DatetimeIndex(model_line.index).tz_convert(UK).tz_localize(None)
        frames.append(
            pd.DataFrame(
                {"time": index, "intensity": model_line.to_numpy(), "series": "This project's ML model"}
            )
        )
    lines = pd.concat(frames, ignore_index=True)
    band = pd.DataFrame(
        {
            "start": [window.start.astimezone(UK).replace(tzinfo=None)],
            "end": [window.end.astimezone(UK).replace(tzinfo=None)],
        }
    )

    highlight = (
        alt.Chart(band)
        .mark_rect(color=HIGHLIGHT)
        .encode(x="start:T", x2="end:T", tooltip=[alt.Tooltip("start:T", format="%a %H:%M", title="Best start")])
    )
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False)
    line = (
        alt.Chart(lines)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:T", title="UK time", axis=alt.Axis(format="%a %H:%M", grid=False)),
            y=alt.Y("intensity:Q", title="gCO2/kWh"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["NESO official forecast", "This project's ML model"],
                    range=[SERIES_1, SERIES_2],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(
                    domain=["NESO official forecast", "This project's ML model"],
                    range=[[1, 0], [5, 3]],
                ),
                legend=None,
            ),
        )
    )
    points = (
        line.mark_point(size=60, filled=True)
        .encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("time:T", format="%a %d %b %H:%M", title="Time"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("intensity:Q", format=".0f", title="gCO2/kWh"),
            ],
        )
        .add_params(nearest)
    )
    return alt.layer(highlight, line, points).properties(height=320)


def daily_chart(daily: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(daily)
        .mark_line(strokeWidth=2, color=SERIES_1, point=alt.OverlayMarkDef(size=30, filled=True))
        .encode(
            x=alt.X("period_date:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("avg_intensity:Q", title="Average gCO2/kWh"),
            tooltip=[
                alt.Tooltip("period_date:T", title="Date"),
                alt.Tooltip("avg_intensity:Q", title="Average", format=".0f"),
                alt.Tooltip("min_intensity:Q", title="Lowest"),
                alt.Tooltip("max_intensity:Q", title="Highest"),
                alt.Tooltip("avg_renewable_pct:Q", title="Renewable %", format=".1f"),
            ],
        )
        .properties(height=260)
    )


def mix_chart(mix: pd.DataFrame) -> alt.Chart:
    groups = ["Renewables", "Gas", "Nuclear", "Imports & other"]
    return (
        alt.Chart(mix)
        .mark_area()
        .encode(
            x=alt.X("period_date:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("share:Q", stack="normalize", title="Share of generation", axis=alt.Axis(format="%")),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=groups, range=[SERIES_1, SERIES_2, SERIES_3, SERIES_4]),
                sort=groups,
                legend=alt.Legend(title=None, orient="top"),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("period_date:T", title="Date"),
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("share:Q", title="Share %", format=".1f"),
            ],
        )
        .properties(height=260)
    )


def hourly_chart(hourly: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(hourly)
        .mark_bar(color=SERIES_1, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("hour_utc:O", title="Hour of day (UTC)"),
            y=alt.Y("avg_intensity:Q", title="Average gCO2/kWh"),
            tooltip=[
                alt.Tooltip("hour_utc:O", title="Hour (UTC)"),
                alt.Tooltip("avg_intensity:Q", title="Average", format=".0f"),
                alt.Tooltip("avg_renewable_pct:Q", title="Renewable %", format=".1f"),
            ],
        )
        .properties(height=240)
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def main() -> None:
    st.title("UK Grid Carbon Intelligence")
    st.caption(
        "Cut emissions by running flexible, energy-heavy work when the GB electricity grid is "
        "greenest. Data: [Carbon Intensity API](https://carbonintensity.org.uk) (NESO)."
    )

    with st.sidebar:
        st.header("Your workload")
        duration = st.slider("Job duration (hours)", 0.5, 12.0, 3.0, 0.5)
        energy = st.number_input("Energy used (kWh)", min_value=1.0, value=500.0, step=50.0)
        st.caption("Examples: overnight data processing, EV fleet charging, an industrial batch run.")

    db_path = str(DEFAULT_DB_PATH)
    try:
        last_run = ensure_warehouse(DEFAULT_HISTORY_DAYS, db_path)
    except (ApiError, sqlite3.Error) as exc:
        st.error(f"Could not refresh data from the Carbon Intensity API: {exc}")
        last_run = None

    # ---- 1. Recommendation ------------------------------------------------ #
    st.subheader("When should you run it?")
    model_line: pd.Series | None = None
    trained: TrainedForecaster | None = None
    try:
        trained = train_model(db_path)
        model_line = forecast_ahead(read_periods(db_path), trained.model)
    except ValueError as exc:
        st.info(f"Forecast model not available yet: {exc}")

    try:
        forecast = live_forecast()
        window = best_window(forecast, duration_hours=float(duration), energy_kwh=float(energy))
    except (ApiError, ValueError) as exc:
        st.error(f"Could not build a recommendation: {exc}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        best_start = window.start.astimezone(UK)
        c1.metric(
            "Best start (UK time)",
            best_start.strftime("%a %H:%M"),
            delta=best_start.strftime("%d %b"),
            delta_color="off",
        )
        c2.metric("CO2 saved vs starting now", f"{window.co2_saved_kg:,.1f} kg")
        c3.metric("Emissions reduction", f"{window.saving_pct:.0f}%")
        c4.metric(
            "Intensity in best window",
            f"{window.avg_intensity:.0f} g/kWh",
            delta=f"{window.avg_intensity - window.run_now_intensity:.0f} vs now",
            delta_color="inverse",
        )
        st.altair_chart(forecast_chart(forecast, window, model_line), use_container_width=True)
        st.caption(
            f"Shaded band: the greenest {duration:g}-hour window. Running in the dirtiest window "
            f"instead would average {window.worst_intensity:.0f} gCO2/kWh. Recommendations use NESO's "
            "forecast because it is the most accurate (see model performance below)."
        )

    # ---- 2. Trends -------------------------------------------------------- #
    st.subheader("How green is the grid?")
    daily = read_table(db_path, "SELECT * FROM analytics_daily WHERE periods >= 40")
    left, right = st.columns(2)
    with left:
        st.markdown("**Daily average carbon intensity**")
        if daily.empty:
            st.info("No data yet.")
        else:
            st.altair_chart(daily_chart(daily), use_container_width=True)
    with right:
        st.markdown("**Generation mix**")
        mix = read_table(
            db_path,
            """
            SELECT period_date,
                   AVG(COALESCE(wind_pct,0) + COALESCE(solar_pct,0) + COALESCE(hydro_pct,0)) AS renewables,
                   AVG(COALESCE(gas_pct,0)) AS gas,
                   AVG(COALESCE(nuclear_pct,0)) AS nuclear,
                   AVG(COALESCE(imports_pct,0) + COALESCE(biomass_pct,0) + COALESCE(coal_pct,0)
                       + COALESCE(other_pct,0)) AS other
            FROM clean_periods GROUP BY period_date ORDER BY period_date
            """,
        )
        if mix.empty:
            st.info("No data yet.")
        else:
            long = mix.melt(id_vars="period_date", var_name="key", value_name="share")
            labels = {"renewables": "Renewables", "gas": "Gas", "nuclear": "Nuclear", "other": "Imports & other"}
            long["source"] = long["key"].map(labels)
            long["order"] = long["key"].map({k: i for i, k in enumerate(labels)})
            st.altair_chart(mix_chart(long), use_container_width=True)

    hourly = read_table(db_path, "SELECT * FROM analytics_hourly_profile")
    if not hourly.empty:
        st.markdown("**Typical intensity by hour of day**")
        st.altair_chart(hourly_chart(hourly), use_container_width=True)

    # ---- 3. Model performance --------------------------------------------- #
    st.subheader("Forecast model performance")
    if trained is not None:
        r = trained.report
        rows = [
            ("This project's ML model", r.model.mae, r.model.rmse, r.model.within_10pct),
            ("Naive: same time yesterday", r.naive.mae, r.naive.rmse, r.naive.within_10pct),
        ]
        if r.neso is not None:
            rows.append(("NESO official forecast", r.neso.mae, r.neso.rmse, r.neso.within_10pct))
        table = pd.DataFrame(rows, columns=["Forecast", "MAE (gCO2/kWh)", "RMSE", "Within 10% of actual"])
        table["Within 10% of actual"] = (table["Within 10% of actual"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption(
            f"Tested on {r.test_rows:,} unseen half-hours ({r.test_start[:10]} to {r.test_end[:10]}), "
            f"24 hours ahead. Skill vs naive: {r.skill_vs_naive:.0%}. NESO uses weather forecasts this "
            "model does not have, so it is the benchmark to beat, not the baseline."
        )
        a, b = st.columns(2)
        with a:
            st.markdown("**Error by time of day (MAE)**")
            st.dataframe(
                pd.DataFrame(list(r.mae_by_time_band.items()), columns=["Time band", "MAE (gCO2/kWh)"]),
                hide_index=True,
                use_container_width=True,
            )
        with b:
            st.markdown("**What drives the model (permutation importance)**")
            st.dataframe(
                pd.DataFrame(
                    list(r.feature_importance.items()), columns=["Feature", "MAE increase if removed"]
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.caption("Full limitations and intended use: see docs/model_card.md in the repository.")

    # ---- 4. Governance ---------------------------------------------------- #
    with st.expander("Data quality and lineage"):
        if last_run is not None:
            st.write(
                f"Last successful pipeline run: **{uk_time(last_run.started_at)}** "
                f"(run #{last_run.run_id}, {last_run.intensity_rows:,} intensity rows, "
                f"{last_run.generation_rows:,} generation rows)."
            )
        issues = read_quality(db_path)
        if not issues:
            st.success("All data quality checks passed.")
        for issue in issues:
            st.warning(f"{issue.description}: {issue.failing_rows:,} rows ({issue.check})")


main()
