import os
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from cassandra_client import validate_keyspace
from data_utils import fetch_recent, fetch_all
from streamlit_autorefresh import st_autorefresh
from footer_utils import add_footer
from river.drift import ADWIN


# ==========================================================================================
# EUROPEAN COUNTRY SETTINGS
# ==========================================================================================

EUROPEAN_COUNTRIES = [
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
    'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
    'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE', 'GBR', 'CHE', 'NOR', 'ISL'
]

COUNTRY_NAME_MAP = {
    'Austria': 'AUT', 'Belgium': 'BEL', 'Bulgaria': 'BGR', 'Croatia': 'HRV',
    'Cyprus': 'CYP', 'Czech Republic': 'CZE', 'Czechia': 'CZE', 'Denmark': 'DNK',
    'Estonia': 'EST', 'Finland': 'FIN', 'France': 'FRA', 'Germany': 'DEU',
    'Greece': 'GRC', 'Hungary': 'HUN', 'Ireland': 'IRL', 'Italy': 'ITA',
    'Latvia': 'LVA', 'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Malta': 'MLT',
    'Netherlands': 'NLD', 'Poland': 'POL', 'Portugal': 'PRT', 'Romania': 'ROU',
    'Slovakia': 'SVK', 'Slovenia': 'SVN', 'Spain': 'ESP', 'Sweden': 'SWE',
    'United Kingdom': 'GBR', 'UK': 'GBR', 'Switzerland': 'CHE', 'Norway': 'NOR',
    'Iceland': 'ISL'
}

ISO_TO_NAME = {v: k for k, v in COUNTRY_NAME_MAP.items() if k not in ['UK', 'Czechia']}


def normalize_country(country):
    """Convert raw country string into ISO code, if European."""
    if pd.isna(country) or country == '':
        return None
    return COUNTRY_NAME_MAP.get(country, None)


# ==========================================================================================
# MAIN APP
# ==========================================================================================

def run():

    st.title("📡 Change Detector – Job Type Evolution (Top 3 in Real Time)")

    TABLE = os.getenv("CASSANDRA_TABLE", "jobs")
    LOOKBACK_MINUTES = 60

    # Validate database
    keyspace_exists, error_msg = validate_keyspace()
    if not keyspace_exists:
        st.error("❌ Database Error")
        st.warning(error_msg)
        st.stop()

    # ==========================================================================================
    # VIEW MODE (stored data or real-time)
    # ==========================================================================================

    st.subheader("📊 View Mode")
    mode = st.radio(
        "Select:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"],
        horizontal=True
    )

    if mode == "📁 View Existing Database":
        df_raw = fetch_all()
        if df_raw.empty:
            st.warning("No data available.")
            st.stop()
        st.success(f"Loaded {len(df_raw)} rows from Cassandra.")
    else:
        st_autorefresh(interval=3000, key="change_detector_refresh")
        df_raw = fetch_recent(LOOKBACK_MINUTES)
        if df_raw.empty:
            st.warning("Waiting for live data…")
            st.stop()
        st.success(f"🔴 LIVE: {len(df_raw)} recent rows")

    # Filter to Europe
    df_raw["country_iso"] = df_raw["country"].apply(normalize_country)
    df_raw = df_raw[df_raw["country_iso"].isin(EUROPEAN_COUNTRIES)]

    if df_raw.empty:
        st.warning("No European job data available.")
        st.stop()

    # ==========================================================================================
    # COUNTRY SELECTOR
    # ==========================================================================================

    st.divider()
    st.header("🌍 Select Country")

    countries_available = sorted(df_raw["country_iso"].dropna().unique())
    default_country = "DEU"  # Germany default

    country_map = {ISO_TO_NAME.get(c, c): c for c in countries_available}

    selected_country_name = st.selectbox(
        "Choose Country",
        options=[ISO_TO_NAME.get(c, c) for c in countries_available],
        index=countries_available.index(default_country) if default_country in countries_available else 0
    )
    selected_iso = country_map[selected_country_name]

    df_country = df_raw[df_raw["country_iso"] == selected_iso]

    if df_country.empty:
        st.warning("No job data for selected country.")
        st.stop()

    # ==========================================================================================
    # TOP 3 JOB TYPES
    # ==========================================================================================

    top3_jobs = df_country["title"].value_counts().head(3).index.tolist()

    st.subheader(f"🎯 Top 3 Job Types in {selected_country_name}")
    for i, job in enumerate(top3_jobs, start=1):
        st.write(f"**{i}. {job}**")

    # ==========================================================================================
    # RESAMPLE IN 2-SECOND BUCKETS
    # ==========================================================================================

    df_country["ts"] = pd.to_datetime(df_country["ts"], errors="coerce")
    df_country = df_country.dropna(subset=["ts"])
    df_country = df_country.set_index("ts")

    lines = []
    for job in top3_jobs:
        df_job = df_country[df_country["title"] == job]
        grouped = df_job.resample("2S").size().rename(job)
        lines.append(grouped)

    df_lines = pd.concat(lines, axis=1).fillna(0)

    # ==========================================================================================
    # ADWIN DRIFT DETECTION — SAFE + LENGTH-ALIGNED + NO NONE VALUES
    # ==========================================================================================

    drift_results = {}

    for job in top3_jobs:
        adwin = ADWIN()

        cumulative = df_lines[job].cumsum()

        cumulative = (
            cumulative
            .replace([None, "None", ""], 0)
            .fillna(0)
            .astype(float)
        )

        drift_flags = []

        for value in cumulative:
            try:
                value = float(value)
            except:
                value = 0.0
            changed = adwin.update(value)
            drift_flags.append(bool(changed))   # <-- force boolean

        # ENSURE SAME LENGTH AS df_lines
        drift_series = pd.Series(drift_flags, index=cumulative.index)

        # FORCE BOOLEAN + NO NONE
        drift_series = drift_series.fillna(False).astype(bool)

        drift_results[job] = drift_series

    # ==========================================================================================
    # 🔥 ENHANCED VISUALIZATION – CUMULATIVE EVOLUTION + DRIFT ANNOTATIONS
    # ==========================================================================================

    st.divider()
    st.header("📈 Real-Time Job Type Evolution with Drift Signals")

    fig = go.Figure()
    colors = ["#2E86C1", "#28B463", "#CA6F1E"]

    for idx, job in enumerate(top3_jobs):

        cumulative = df_lines[job].cumsum()

        # Main line
        fig.add_trace(go.Scatter(
            x=df_lines.index,
            y=cumulative,
            mode="lines",
            name=job,
            line=dict(width=3, color=colors[idx])
        ))

        # Drift markers (+ annotation)
        drift_points = drift_results[job][drift_results[job] == True]

        if not drift_points.empty:
            fig.add_trace(go.Scatter(
                x=drift_points.index,
                y=cumulative.loc[drift_points.index],
                mode="markers+text",
                name=f"{job} DRIFT",
                text=["⚠ drift"] * len(drift_points),
                textposition="top center",
                marker=dict(size=14, color="red", symbol="diamond", line=dict(width=2, color="white")),
                showlegend=True
            ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cumulative Job Count",
        height=500,
        legend=dict(title="Job Type"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


    # ==========================================================================================
    # 🔥 DRIFT HEATMAP (JOB TYPE vs TIME)
    # ==========================================================================================

    st.subheader("🔥 Drift Heatmap (Intensity Over Time)")

    heatmap_df = pd.DataFrame({
        job: drift_results[job].astype(int)
        for job in top3_jobs
    })
    heatmap_df.index = df_lines.index

    fig_h = go.Figure(data=go.Heatmap(
        z=heatmap_df.T.values,
        x=heatmap_df.index,
        y=heatmap_df.columns,
        colorscale="Reds",
        colorbar=dict(title="Drift Strength"),
    ))

    fig_h.update_layout(
        xaxis_title="Time",
        yaxis_title="Job Type",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_h, use_container_width=True)


    # ==========================================================================================
    # 🔥 Drift Timeline – Human readable event feed
    # ==========================================================================================

    st.subheader("🕒 Drift Event Timeline")

    timeline = []
    for job in top3_jobs:
        cumu = df_lines[job].cumsum()
        for ts, flag in drift_results[job].items():
            if flag:
                timeline.append({
                    "time": ts.strftime("%H:%M:%S"),
                    "job": job,
                    "value": cumu.loc[ts]
                })

    timeline_df = pd.DataFrame(timeline)

    if timeline_df.empty:
        st.info("No drift events detected yet.")
    else:
        st.dataframe(
            timeline_df.sort_values("time"),
            use_container_width=True,
            hide_index=True
        )



    # ==========================================================================================
    # DRIFT SUMMARY TABLE
    # ==========================================================================================

    st.subheader("📡 ADWIN Drift Detection Summary")

    summary_rows = []
    for job in top3_jobs:
        num_drifts = int(drift_results[job].astype(int).sum())
        summary_rows.append({"Job Type": job, "Drift Events": num_drifts})

    st.table(pd.DataFrame(summary_rows))


    # ==========================================================================================
    # SIMPLE SNAPSHOT CHANGE DETECTION (your original)
    # ==========================================================================================

    st.divider()
    st.header("🧭 Job Type Appearance / Removal Summary")

    if "previous_snapshot" not in st.session_state:
        st.session_state.previous_snapshot = set()

    current_types = set(df_country["title"].unique())
    previous_types = st.session_state.previous_snapshot

    new_types = current_types - previous_types
    removed_types = previous_types - current_types

    st.metric("Total Job Types", len(current_types))
    st.metric("New Job Types Detected", len(new_types))
    st.metric("Removed Job Types", len(removed_types))

    st.session_state.previous_snapshot = current_types

    with st.expander("📄 View Job Type Changes"):
        st.write("### Newly Appeared:")
        st.write(list(new_types) if new_types else "None")

        st.write("### Removed Job Types:")
        st.write(list(removed_types) if removed_types else "None")

    # ==========================================================================================
    # RECENT JOB LOG
    # ==========================================================================================

    st.divider()
    st.header("📋 Recent Jobs")

    with st.expander("Show recent 20 jobs"):
        cols_display = ["title", "company_name", "location", "skill", "ts"]
        st.dataframe(
            df_country.reset_index()[cols_display]
            .sort_values("ts", ascending=False)
            .head(20)
        )

    add_footer("CSOMA Team")
