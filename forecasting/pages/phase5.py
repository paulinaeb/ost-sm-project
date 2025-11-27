import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from cassandra_client import validate_keyspace
from data_utils import fetch_recent, fetch_all
from streamlit_autorefresh import st_autorefresh
from footer_utils import add_footer
from river.drift import ADWIN

# ==========================================
# CONSTANTS
# ==========================================
EUROPEAN_COUNTRIES = [
    'AUT','BEL','BGR','HRV','CYP','CZE','DNK','EST','FIN','FRA','DEU','GRC','HUN',
    'IRL','ITA','LVA','LTU','LUX','MLT','NLD','POL','PRT','ROU','SVK','SVN','ESP',
    'SWE','GBR','CHE','NOR','ISL'
]

COUNTRY_NAME_MAP = {
    'Austria':'AUT','Belgium':'BEL','Bulgaria':'BGR','Croatia':'HRV',
    'Cyprus':'CYP','Czech Republic':'CZE','Czechia':'CZE','Denmark':'DNK',
    'Estonia':'EST','Finland':'FIN','France':'FRA','Germany':'DEU',
    'Greece':'GRC','Hungary':'HUN','Ireland':'IRL','Italy':'ITA',
    'Latvia':'LVA','Lithuania':'LTU','Luxembourg':'LUX','Malta':'MLT',
    'Netherlands':'NLD','Poland':'POL','Portugal':'PRT','Romania':'ROU',
    'Slovakia':'SVK','Slovenia':'SVN','Spain':'ESP','Sweden':'SWE',
    'United Kingdom':'GBR','UK':'GBR','Switzerland':'CHE',
    'Norway':'NOR','Iceland':'ISL'
}

ISO_TO_NAME = {v: k for k, v in COUNTRY_NAME_MAP.items() if k not in ["UK", "Czechia"]}


def normalize_country(country):
    if pd.isna(country) or country == "":
        return None
    return COUNTRY_NAME_MAP.get(country, None)


# ================================================================
# MAIN STREAMLIT PAGE
# ================================================================
def run():

    st.title("📡 Change Detector – Job Type Evolution (Top 3 in Real Time)")
    LOOKBACK_MINUTES = 60

    # Validate Cassandra
    ok, err = validate_keyspace()
    if not ok:
        st.error("❌ Database Error")
        st.write(err)
        st.stop()

    # ============================================================
    # VIEW MODE
    # ============================================================
    st.subheader("📊 View Mode")
    mode = st.radio(
        "Select:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"],
        horizontal=True
    )

    if mode == "📁 View Existing Database":
        df_raw = fetch_all()
    else:
        st_autorefresh(interval=3000, key="phase5_refresh")
        df_raw = fetch_recent(LOOKBACK_MINUTES)

    if df_raw.empty:
        st.warning("No data available.")
        st.stop()

    # Ensure ingested_at exists
    if "ingested_at" not in df_raw.columns:
        st.error("❌ Column 'ingested_at' missing in table.")
        st.stop()

    # Create timestamp column
    df_raw["ts"] = pd.to_datetime(df_raw["ingested_at"], errors="coerce")
    df_raw = df_raw.dropna(subset=["ts"])

    # ============================================================
    # FILTER EUROPE
    # ============================================================
    df_raw["country_iso"] = df_raw["country"].apply(normalize_country)
    df_raw = df_raw[df_raw["country_iso"].isin(EUROPEAN_COUNTRIES)]

    if df_raw.empty:
        st.warning("No European job data available.")
        st.stop()

    # ============================================================
    # COUNTRY SELECTOR
    # ============================================================
    st.divider()

    countries_available = sorted(df_raw["country_iso"].dropna().unique())

    # default: Germany if exists, else first available
    default_country = "HUN" if "HUN" in countries_available else countries_available[0]

    country_map = {ISO_TO_NAME.get(c, c): c for c in countries_available}

    selected_country_name = st.selectbox(
        "Choose Country",
        [ISO_TO_NAME.get(c, c) for c in countries_available],
        index=countries_available.index(default_country)
    )

    selected_iso = country_map[selected_country_name]

    df_country = df_raw[df_raw["country_iso"] == selected_iso].copy()
    df_country = df_country.set_index("ts").sort_index()


    # ============================================================
    # 🌍 COUNTRY vs EUROPE — Cumulative Trend with ADWIN
    # ============================================================

    st.divider()
    st.header("🌍 Country vs Europe – Trend (with ADWIN Drift Detection)")

    # ---- Job selector ----
    job_options = df_country["title"].value_counts().index.tolist()
    selected_job = st.selectbox("Select Job Type for Comparison", job_options)

    # ============================================================
    # EUROPE = all European countries (EUROPEAN_COUNTRIES list)
    # EXCLUDING the selected country
    # ============================================================

    europe_df = df_raw[
        (df_raw["title"] == selected_job) &
        (df_raw["country_iso"].isin(EUROPEAN_COUNTRIES)) &
        (df_raw["country_iso"] != selected_iso)   # exclude selected country
    ].copy()

    europe_df = europe_df.set_index("ts").sort_index()
    europe_series = europe_df.resample("2S").size().cumsum().rename("Europe")

    # ============================================================
    # COUNTRY = selected country
    # ============================================================

    country_df = df_country[df_country["title"] == selected_job].copy()
    country_series = country_df.resample("2S").size().cumsum().rename(selected_country_name)

    # ============================================================
    # ALIGN BOTH SERIES (forward-fill missing timestamps)
    # ============================================================

    df_compare = pd.concat([country_series, europe_series], axis=1)
    df_compare = df_compare.fillna(method="ffill").fillna(0)

    # ============================================================
    # ADWIN DRIFT DETECTION
    # ============================================================

    ad_ct, ad_eu = ADWIN(delta=0.2), ADWIN(delta=0.2)
    drift_ct, drift_eu = [], []

    for a, b in zip(df_compare[selected_country_name], df_compare["Europe"]):
        drift_ct.append(ad_ct.update(float(a)))
        drift_eu.append(ad_eu.update(float(b)))

    df_compare["drift_ct"] = drift_ct
    df_compare["drift_eu"] = drift_eu

    # ============================================================
    # PLOT COUNTRY vs EUROPE
    # ============================================================

    fig_ce = go.Figure()

    # Country line
    fig_ce.add_trace(go.Scatter(
        x=df_compare.index,
        y=df_compare[selected_country_name],
        mode="lines",
        name=f"{selected_country_name} ()",
        line=dict(color="green", width=3)
    ))

    # Europe line
    fig_ce.add_trace(go.Scatter(
        x=df_compare.index,
        y=df_compare["Europe"],
        mode="lines",
        name="Europe (cumulative)",
        line=dict(color="blue", width=3),
        yaxis="y2"
    ))

    # Country drift markers
    dc = df_compare[df_compare["drift_ct"] == True]
    fig_ce.add_trace(go.Scatter(
        x=dc.index,
        y=dc[selected_country_name],
        mode="markers",
        marker=dict(size=12, color="red", symbol="diamond"),
        name=f"{selected_country_name} Drift"
    ))

    # Europe drift markers
    de = df_compare[df_compare["drift_eu"] == True]
    fig_ce.add_trace(go.Scatter(
        x=de.index,
        y=de["Europe"],
        mode="markers",
        marker=dict(size=12, color="orange", symbol="star"),
        name="Europe Drift",
        yaxis="y2"
    ))

    fig_ce.update_layout(
        title=f"{selected_job} – Trend: {selected_country_name} vs Europe (All Countries Combined)",
        height=520,
        yaxis=dict(title=f"{selected_country_name} Count"),
        yaxis2=dict(title="Europe Count", overlaying="y", side="right"),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig_ce, use_container_width=True)

    

    # ============================================================
    # 1️⃣ TOP 3 JOB TYPES
    # ============================================================
    top3 = df_country["title"].value_counts().head(3).index.tolist()

    st.subheader(f"🎯 Top 3 Job Types in {selected_country_name}")
    for i, job in enumerate(top3, start=1):
        st.write(f"**{i}. {job}**")

    # ============================================================
    # 2️⃣ BUILD df_lines (2-second buckets)
    # ============================================================
    lines = []
    for job in top3:
        s = df_country[df_country["title"] == job].resample("2S").size().rename(job)
        lines.append(s)

    df_lines = pd.concat(lines, axis=1).fillna(0)

    # ============================================================
    # 3️⃣ ADWIN – CUMULATIVE DRIFT DETECTION
    # ============================================================
    drift_results = {}

    for job in top3:
        ad = ADWIN(delta=0.2)
        cumulative = df_lines[job].cumsum().astype(float)

        flags = []
        for v in cumulative:
            flags.append(bool(ad.update(float(v))))

        drift_results[job] = pd.Series(flags, index=cumulative.index)

    # ============================================================
    # 4️⃣ CUMULATIVE EVOLUTION + DRIFT
    # ============================================================
    st.divider()
    st.header("📈 Job Evolution ")

    fig = go.Figure()
    colors = ["#2E86C1", "#28B463", "#CA6F1E"]

    for idx, job in enumerate(top3):
        cumu = df_lines[job].cumsum()

        fig.add_trace(go.Scatter(
            x=df_lines.index,
            y=cumu,
            mode="lines",
            name=job,
            line=dict(color=colors[idx], width=3)
        ))

        drift_points = drift_results[job][drift_results[job] == True]
        if not drift_points.empty:
            fig.add_trace(go.Scatter(
                x=drift_points.index,
                y=cumu.loc[drift_points.index],
                mode="markers",
                marker=dict(size=14, color="red", symbol="diamond"),
                name=f"{job} Drift"
            ))

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 5️⃣ RATE OF CHANGE (Derivative)
    # ============================================================
    st.subheader("📈 Rate of Change (Δ jobs every 2 seconds)")

    roc_df = pd.DataFrame()
    for job in top3:
        roc_df[job] = df_lines[job].diff().fillna(0)

    fig_roc = go.Figure()
    for idx, job in enumerate(top3):
        fig_roc.add_trace(go.Scatter(
            x=roc_df.index,
            y=roc_df[job],
            mode="lines+markers",
            name=f"{job} ROC",
            line=dict(width=2, color=colors[idx])
        ))

    st.plotly_chart(fig_roc, use_container_width=True)



    