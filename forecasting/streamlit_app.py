import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from cassandra_client import get_session
from streamlit_autorefresh import st_autorefresh


# ============== SETTINGS ==================
TABLE = os.getenv("CASSANDRA_TABLE", "jobs")
LOOKBACK_MINUTES = 60

st.set_page_config(page_title="LinkedIn Jobs – Live Stream", layout="wide")
st.title("🔴 Live LinkedIn Jobs Stream")


# ============== HELPERS ==================
def per_bucket_counts(df, bucket="1min"):
    """Aggregate timestamped jobs into buckets (per-minute or per-second)."""
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame(columns=["date", "jobs"])
    s = (df
         .set_index("ts")
         .assign(jobs=1)
         .resample(bucket)["jobs"]
         .sum()
         .rename_axis("date")
         .reset_index())
    return s


# ============== CASSANDRA FETCH ==================
@st.cache_data(ttl=3)
def fetch_recent(minutes=60):
    session = get_session()
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)

    query = f"""
    SELECT id, title, company_name, location, country, skill,
           created_at, ingested_at
    FROM {TABLE}
    WHERE ingested_at >= %s AND ingested_at < %s ALLOW FILTERING;
    """

    rows = session.execute(query, (start, now))
    df = pd.DataFrame(list(rows))

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ingested_at"], errors="coerce")

    for c in ["company_name", "title", "location", "country", "skill"]:
        if c in df:
            df[c] = df[c].fillna("").astype(str)

    return df.sort_values("ts")


# ============== UI ==================
minutes = st.slider("Lookback window (minutes)", 5, 240, LOOKBACK_MINUTES)

df = fetch_recent(minutes)

if df.empty:
    st.warning("No data found. Start producer/consumer.")
    st.stop()


# ============== METRICS ==================
col1, col2, col3 = st.columns(3)

col1.metric("Total Jobs", len(df))
col2.metric("Companies", df["company_name"].nunique())
col3.metric("Newest job", df["ts"].max().strftime("%Y-%m-%d %H:%M:%S"))

st.write("Last refresh:", datetime.utcnow().strftime("%H:%M:%S.%f"))


# ============== TABLE ==================
st.subheader("📥 Latest Jobs")
cols = ["ts", "title", "company_name", "location", "country", "skill"]
cols = [c for c in cols if c in df.columns]
st.dataframe(df[cols].tail(100), use_container_width=True)


# ============== REAL-TIME VIEW ==================
st.subheader("⚡ Real-time Streaming View")

# Auto-refresh every 2 seconds
st_autorefresh(interval=2000, key="live_refresh")

df_live = fetch_recent(60)   # last hour live

st.metric("Rows", len(df_live))

if df_live.empty:
    st.info("No live jobs yet…")
else:
    st.write("### Latest arrivals")
    st.dataframe(df_live.sort_values("ts").tail(20))

    per_min = per_bucket_counts(df_live, "1min")
    

# # ============================================
# 📈 1) TOTAL JOB COUNT OVER TIME (CUMULATIVE)
# ============================================

st.subheader("📈 Total Jobs Stored in Cassandra Over Time")

# Fetch the entire table count from Cassandra
@st.cache_data(ttl=2)
def get_total_count():
    session = get_session()
    result = session.execute(f"SELECT COUNT(*) AS count FROM {TABLE};")
    row = result.one()
    return row["count"] if row else 0


# Build a small time series from session_state
if "count_history" not in st.session_state:
    st.session_state.count_history = []

current_count = get_total_count()

# Append new count
st.session_state.count_history.append({
    "time": datetime.utcnow(),
    "count": current_count
})

df_count = pd.DataFrame(st.session_state.count_history)

# Plot
st.line_chart(df_count.set_index("time")["count"])

# ============================================
# 📊 JOBS PER TIMESTAMP (COLUMN GRAPH)
# ============================================

st.subheader("📊 Jobs per Timestamp (Batch Frequency)")

df_live = fetch_recent(6)   # last 6

if df_live.empty:
    st.info("No data found yet…")
else:
    # Convert timestamp
    df_live["ts"] = pd.to_datetime(df_live["ts"], errors="coerce")

    # Group by minute (change to "10S" if needed)
    per_ts = (
        df_live
        .set_index("ts")
        .assign(jobs=1)
        .resample("2s")["jobs"]
        .sum()
        .rename_axis("timestamp")
        .reset_index()
    )

    st.bar_chart(
        per_ts.set_index("timestamp")["jobs"],
        height=400
    )



# ============== TOP COMPANIES ==================
st.subheader("🏢 Top Companies")
top_companies = df["company_name"].value_counts().head(10)
st.bar_chart(top_companies)
