import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
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

    # Validate Cassandra connection
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
        st.warning("No data found.")
        st.stop()

    # ============================================================
    # CLEAN & FILTER EUROPE
    # ============================================================
    df_raw["country_iso"] = df_raw["country"].apply(normalize_country)
    df_raw = df_raw[df_raw["country_iso"].isin(EUROPEAN_COUNTRIES)]

    if df_raw.empty:
        st.warning("No European job data.")
        st.stop()

    # ============================================================
    # COUNTRY SELECTOR (working version)
    # ============================================================
    st.divider()
    st.header("🌍 Select Country")

    countries_available = sorted(df_raw["country_iso"].dropna().unique())
    default_country = "HUN" if "HUN" in countries_available else countries_available[0]

    country_map = {ISO_TO_NAME.get(c, c): c for c in countries_available}

    selected_country_name = st.selectbox(
        "Choose Country",
        [ISO_TO_NAME.get(c, c) for c in countries_available],
        index=countries_available.index(default_country)
    )
    selected_iso = country_map[selected_country_name]

    # Country-level filtered data
    df_country = df_raw[df_raw["country_iso"] == selected_iso].copy()
    df_country["ts"] = pd.to_datetime(df_country["ts"], errors="coerce")
    df_country = df_country.dropna(subset=["ts"]).set_index("ts").sort_index()

    if df_country.empty:
        st.warning(f"No valid timestamp data for {selected_country_name}")
        st.stop()

    # ============================================================
    # 1️⃣ – TOP 3 JOB TYPES
    # ============================================================
    top3 = df_country["title"].value_counts().head(3).index.tolist()

    st.subheader(f"🎯 Top 3 Job Types in {selected_country_name}")
    for i, job in enumerate(top3, start=1):
        st.write(f"**{i}. {job}**")

    # ============================================================
    # 2️⃣ – BUILD df_lines FOR TOP 3
    # ============================================================
    lines = []
    for job in top3:
        s = df_country[df_country["title"] == job].resample("2S").size().rename(job)
        lines.append(s)

    df_lines = pd.concat(lines, axis=1).fillna(0)

    # ============================================================
    # 3️⃣ – ADWIN DRIFT PER JOB TYPE
    # ============================================================
    drift_results = {}

    for job in top3:
        ad = ADWIN()
        cumulative = df_lines[job].cumsum().astype(float)

        flags = []
        for v in cumulative:
            flags.append(bool(ad.update(float(v))))

        drift_results[job] = pd.Series(flags, index=cumulative.index)

    # ============================================================
    # 4️⃣ – CUMULATIVE CURVES + DRIFT
    # ============================================================
    st.divider()
    st.header("📈 Evolution with Drift Detection (ADWIN)")

    fig = go.Figure()
    colors = ["#2E86C1", "#28B463", "#CA6F1E"]

    for idx, job in enumerate(top3):
        cumu = df_lines[job].cumsum()

        fig.add_trace(go.Scatter(
            x=df_lines.index, y=cumu,
            mode="lines", name=job,
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
    # 5️⃣ – RATE OF CHANGE (FIRST DERIVATIVE)
    # ============================================================
    st.subheader("📈 Rate of Change (Δ jobs / 2 seconds)")

    roc_df = pd.DataFrame()
    for job in top3:
        roc_df[job] = df_lines[job].diff().fillna(0)

    fig_roc = go.Figure()
    for idx, job in enumerate(top3):
        fig_roc.add_trace(go.Scatter(
            x=roc_df.index, y=roc_df[job],
            mode="lines+markers",
            name=f"{job} ROC",
            line=dict(color=colors[idx], width=2)
        ))

    st.plotly_chart(fig_roc, use_container_width=True)

    # ============================================================
    # 6️⃣ – DRIFT HEATMAP
    # ============================================================
    st.subheader("🔥 Drift Heatmap")

    heatmap_df = pd.DataFrame({job: drift_results[job].astype(int) for job in top3})
    heatmap_df.index = df_lines.index

    fig_h = go.Figure(data=go.Heatmap(
        z=heatmap_df.T.values,
        x=heatmap_df.index,
        y=heatmap_df.columns,
        colorscale="Reds"
    ))

    st.plotly_chart(fig_h, use_container_width=True)

    # ============================================================
    # 7️⃣ – DRIFT TIMELINE
    # ============================================================
    st.subheader("🕒 Drift Event Timeline")

    timeline = []
    for job in top3:
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
        st.dataframe(timeline_df, hide_index=True)

    # ============================================================
    # 8️⃣ – DRIFT SUMMARY TABLE
    # ============================================================
    st.subheader("📡 Drift Summary (ADWIN)")
    summary_rows = [{"Job": job, "Drifts": int(drift_results[job].sum())} for job in top3]
    st.table(pd.DataFrame(summary_rows))

    # ============================================================
    # 9️⃣ – COUNTRY vs EUROPE TREND (10-SECOND BUCKETS + ADWIN)
    # ============================================================

    st.divider()
    st.header("🌍 Country vs Europe – 10-Second Job Frequency Comparison")

    # ---- Job selector ----
    job_options = df_country["title"].value_counts().index.tolist()
    selected_job = st.selectbox("Select Job Type", job_options)

    # ---- PREPARE EUROPE-LEVEL DATA ----
    df_eu = df_raw[df_raw["title"] == selected_job].copy()
    df_eu["ts"] = pd.to_datetime(df_eu["ts"], errors="coerce")
    df_eu = df_eu.dropna(subset=["ts"]).set_index("ts").sort_index()

    # 10-second buckets
    eu_bucket = df_eu.resample("2S").size().rename("Europe")

    # ---- PREPARE COUNTRY-LEVEL DATA ----
    df_ct = df_country[df_country["title"] == selected_job].copy()
    ct_bucket = df_ct.resample("2S").size().rename(selected_country_name)

    # ---- ALIGN BOTH SERIES ----
    df_compare = pd.concat([ct_bucket, eu_bucket], axis=1).fillna(0)

    # ---- APPLY ADWIN TO BUCKET VALUES (not cumulative!) ----
    ad_ct, ad_eu = ADWIN(), ADWIN()
    drift_ct, drift_eu = [], []

    for c_val, e_val in zip(df_compare[selected_country_name], df_compare["Europe"]):
        drift_ct.append(ad_ct.update(float(c_val)))
        drift_eu.append(ad_eu.update(float(e_val)))

    df_compare["drift_ct"] = drift_ct
    df_compare["drift_eu"] = drift_eu

    # ============================================================
    # PLOT COUNTRY VS EUROPE (10-second buckets)
    # ============================================================

    fig_ce = go.Figure()

    # Country
    fig_ce.add_trace(go.Scatter(
        x=df_compare.index,
        y=df_compare[selected_country_name],
        mode="lines+markers",
        name=f"{selected_country_name} (2s frequency)",
        line=dict(color="green", width=3)
    ))

    # Europe
    fig_ce.add_trace(go.Scatter(
        x=df_compare.index,
        y=df_compare["Europe"],
        mode="lines+markers",
        name="Europe (2s frequency)",
        line=dict(color="blue", width=3),
        yaxis="y2"
    ))

    # Drift: country
    dc = df_compare[df_compare["drift_ct"] == True]
    fig_ce.add_trace(go.Scatter(
        x=dc.index, y=dc[selected_country_name],
        mode="markers", marker=dict(size=14, color="red", symbol="diamond"),
        name=f"{selected_country_name} Drift"
    ))

    # Drift: Europe
    de = df_compare[df_compare["drift_eu"] == True]
    fig_ce.add_trace(go.Scatter(
        x=de.index, y=de["Europe"],
        mode="markers", marker=dict(size=14, color="orange", symbol="star"),
        name="Europe Drift",
        yaxis="y2"
    ))

    fig_ce.update_layout(
        title=f"{selected_job} – 2s Job Frequency: {selected_country_name} vs Europe",
        height=520,
        yaxis=dict(title=f"{selected_country_name} Jobs / 2s", titlefont=dict(color="green")),
        yaxis2=dict(title="Europe Jobs / 2s", overlaying="y", side="right", titlefont=dict(color="blue")),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig_ce, use_container_width=True)

    # ============================================================
    # TREND SUMMARY
    # ============================================================

    st.subheader("📘 Trend Summary (10-second bucket analysis)")

    slope_ct = df_compare[selected_country_name].mean()
    slope_eu = df_compare["Europe"].mean()

    if slope_ct > slope_eu:
        st.success(f"🚀 {selected_country_name} shows higher job frequency than Europe.")
    elif slope_ct < slope_eu:
        st.warning(f"📉 Europe shows higher job frequency.")
    else:
        st.info("⚖️ Both regions have similar frequencies.")

    # ============================================================
    # 🌟 CLEAN, EASY TREND VISUALIZATION BLOCK
    # ============================================================

    st.divider()
    st.header("📈 Country vs Europe – Clear Trend Analysis")

    # ------------------------------------------
    # 1) Validate "ts" column
    # ------------------------------------------
    df_raw["ts"] = pd.to_datetime(df_raw["ts"], errors="coerce")
    df_country["ts"] = pd.to_datetime(df_country["ts"], errors="coerce")

    df_raw = df_raw.dropna(subset=["ts"])
    df_country = df_country.dropna(subset=["ts"])

    # ------------------------------------------
    # 2) Select job type
    # ------------------------------------------
    job_options = df_country["title"].value_counts().index.tolist()

    if len(job_options) == 0:
        st.warning("No job types found for this country.")
        st.stop()

    selected_job = st.selectbox("Select Job Type", job_options)

    # ------------------------------------------
    # 3) Build Europe & Country time series (2s buckets)
    # ------------------------------------------
    df_europe_job = df_raw[df_raw["title"] == selected_job].set_index("ts").sort_index()
    df_country_job = df_country[df_country["title"] == selected_job].set_index("ts").sort_index()

    series_europe = df_europe_job.resample("2S").size().fillna(0)
    series_country = df_country_job.resample("2S").size().fillna(0)

    # ------------------------------------------
    # 4) Smooth using Moving Average (window=5)
    # ------------------------------------------
    smooth_europe = series_europe.rolling(window=5, min_periods=1).mean()
    smooth_country = series_country.rolling(window=5, min_periods=1).mean()

    # ------------------------------------------
    # 5) ALIGN both time series
    # ------------------------------------------
    df_trend = pd.concat([
        smooth_country.rename(selected_country_name),
        smooth_europe.rename("Europe")
    ], axis=1).fillna(method="ffill").fillna(0)

    # ------------------------------------------
    # 6) ADWIN drift detection (light)
    # ------------------------------------------
    ad_ct, ad_eu = ADWIN(), ADWIN()
    drift_ct, drift_eu = [], []

    for a, b in zip(df_trend[selected_country_name], df_trend["Europe"]):
        drift_ct.append(ad_ct.update(float(a)))
        drift_eu.append(ad_eu.update(float(b)))

    df_trend["drift_ct"] = drift_ct
    df_trend["drift_eu"] = drift_eu

    # ------------------------------------------
    # 7) Plot CLEAN SMOOTHED TREND LINES
    # ------------------------------------------
    st.subheader("📉 Smoothed Trend Line (5-window moving average)")

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=df_trend.index,
        y=df_trend[selected_country_name],
        mode="lines",
        name=selected_country_name,
        line=dict(color="green", width=3)
    ))

    fig1.add_trace(go.Scatter(
        x=df_trend.index,
        y=df_trend["Europe"],
        mode="lines",
        name="Europe",
        line=dict(color="blue", width=3)
    ))

    # Drift markers (minimal)
    ct_drift_points = df_trend[df_trend["drift_ct"] == True]
    eu_drift_points = df_trend[df_trend["drift_eu"] == True]

    fig1.add_trace(go.Scatter(
        x=ct_drift_points.index,
        y=ct_drift_points[selected_country_name],
        mode="markers",
        marker=dict(color="red", size=12, symbol="diamond"),
        name=f"{selected_country_name} Drift"
    ))

    fig1.add_trace(go.Scatter(
        x=eu_drift_points.index,
        y=eu_drift_points["Europe"],
        mode="markers",
        marker=dict(color="orange", size=12, symbol="star"),
        name="Europe Drift"
    ))

    fig1.update_layout(
        height=450,
        yaxis_title="Smoothed Job Frequency (2s buckets)",
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ------------------------------------------
    # 8) Trend GAP visualization
    # ------------------------------------------
    st.subheader("📊 Trend Gap (Country – Europe)")

    gap = df_trend[selected_country_name] - df_trend["Europe"]

    fig_gap = go.Figure()

    fig_gap.add_trace(go.Scatter(
        x=gap.index,
        y=gap,
        mode="lines",
        line=dict(color="#8E44AD", width=3),
        name="Trend Gap"
    ))

    # Shade above/below zero
    fig_gap.add_hrect(y0=0, y1=gap.max(), fillcolor="green", opacity=0.1, line_width=0)
    fig_gap.add_hrect(y0=gap.min(), y1=0, fillcolor="red", opacity=0.1, line_width=0)

    fig_gap.update_layout(
        height=300,
        yaxis_title="Trend Gap (positive = country ahead)",
    )

    st.plotly_chart(fig_gap, use_container_width=True)

    # ------------------------------------------
    # 9) Trend Summary
    # ------------------------------------------
    st.subheader("📘 Trend Summary")

    slope_ct = df_trend[selected_country_name].diff().mean()
    slope_eu = df_trend["Europe"].diff().mean()

    if slope_ct > slope_eu:
        st.success(f"🚀 {selected_country_name} is rising faster than Europe.")
    elif slope_ct < slope_eu:
        st.warning(f"📉 Europe is rising faster than {selected_country_name}.")
    else:
        st.info("⚖️ Both trends are similar.")

    st.write(f"Drift Events — {selected_country_name}: **{sum(drift_ct)}**,  Europe: **{sum(drift_eu)}**")
