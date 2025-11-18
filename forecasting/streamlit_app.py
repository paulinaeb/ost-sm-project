import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from cassandra_client import get_session
from streamlit_autorefresh import st_autorefresh
import importlib

# ==============================================================================
# SETUP
# ==============================================================================
st.set_page_config(page_title="LinkedIn Jobs – Live Stream", layout="wide")

# ===================== FIXED HORIZONTAL NAV BAR =====================
st.markdown("""
    <style>
        .nav-container {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 20px;
            margin-bottom: 30px;
        }

        .nav-btn {
            font-size: 18px;
            font-weight: 600;
            padding: 14px 30px;
            border-radius: 40px;
            color: white !important;
            text-decoration: none;
            white-space: nowrap;
            transition: 0.25s ease;
        }

        .phase1 { background: linear-gradient(45deg, #ff4b2b, #ff416c); }
        .phase2 { background: linear-gradient(45deg, #ff9a00, #ff6a00); }
        .phase3 { background: linear-gradient(45deg, #00c6ff, #0072ff); }
        .phase4 { background: linear-gradient(45deg, #42e695, #3bb2b8); }

        .active-btn {
            border: 3px solid white;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
        }

        .nav-btn:hover {
            transform: translateY(-4px);
            box-shadow: 0px 6px 12px rgba(0,0,0,0.25);
        }
    </style>
""", unsafe_allow_html=True)


current_phase = st.query_params.get("phase", "Phase 3")

# ---- ALL BUTTONS IN ONE SINGLE BLOCK ----
st.markdown(
    f"""
    <div class="nav-container">
        <a class="nav-btn phase1 {'active-btn' if current_phase=='Phase 1' else ''}" href="?phase=Phase 1">Phase 1</a>
        <a class="nav-btn phase2 {'active-btn' if current_phase=='Phase 2' else ''}" href="?phase=Phase 2">Phase 2</a>
        <a class="nav-btn phase3 {'active-btn' if current_phase=='Phase 3' else ''}" href="?phase=Phase 3">Phase 3</a>
        <a class="nav-btn phase4 {'active-btn' if current_phase=='Phase 4' else ''}" href="?phase=Phase 4">Phase 4</a>
    </div>
    """,
    unsafe_allow_html=True
)




# ==============================================================================
# FUNCTION TO LOAD OTHER PHASES
# ==============================================================================
def load_other_phase(module_name):
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "run"):
            module.run()
        else:
            exec(open(module.__file__).read())
    except Exception as e:
        st.error(f"Failed to load {module_name}: {e}")

# ==============================================================================
# PHASE 3 = YOUR ORIGINAL DASHBOARD (UNCHANGED)
# ==============================================================================
def run_phase3():

    # ---------------- SETTINGS ----------------
    TABLE = os.getenv("CASSANDRA_TABLE", "jobs")
    LOOKBACK_MINUTES = 60

    st.title("🔴 Live LinkedIn Jobs Stream")

    # ---------------- HELPERS ----------------
    def per_bucket_counts(df, bucket="1min"):
        if df.empty or "ts" not in df.columns:
            return pd.DataFrame(columns=["date", "jobs"])
        s = (
            df.set_index("ts")
            .assign(jobs=1)
            .resample(bucket)["jobs"]
            .sum()
            .rename_axis("date")
            .reset_index()
        )
        return s

    # ---------------- FETCH RECENT ----------------
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
            df[c] = df[c].fillna("").astype(str)

        return df.sort_values("ts")

    # ---------------- FETCH ALL ----------------
    @st.cache_data(ttl=30)
    def fetch_all():
        session = get_session()
        query = f"""
        SELECT id, title, company_name, location, country, skill,
               created_at, ingested_at
        FROM {TABLE};
        """
        rows = session.execute(query)
        df = pd.DataFrame(list(rows))

        if df.empty:
            return df

        df["ts"] = pd.to_datetime(df["ingested_at"], errors="coerce")

        for c in ["company_name", "title", "location", "country", "skill"]:
            df[c] = df[c].fillna("").astype(str)

        return df.sort_values("ts")

    # ---------------- MODE SELECTOR ----------------
    st.subheader("View Mode")

    mode = st.radio(
        "Choose how to view data:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"]
    )

    minutes = LOOKBACK_MINUTES

    # ---------------- EXISTING TABLE ----------------
    if mode == "📁 View Existing Database":
        st.header("📁 Full Database View")
        df = fetch_all()

        if df.empty:
            st.warning("No data found in Cassandra table.")
            st.stop()

        st.success(f"Loaded {len(df)} rows from the full table.")

        st.dataframe(df, use_container_width=True)

        st.subheader("Top Companies")
        top_comp = df["company_name"].value_counts().head(10)
        st.bar_chart(top_comp)

        st.stop()

    # ---------------- REAL TIME ----------------
    st.header("⚡ Real-time Streaming Mode")

    df = fetch_recent(minutes)

    if df.empty:
        st.warning("No live data. Start producer/consumer first.")
        st.stop()

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Recent Jobs", len(df))
    col2.metric("Companies (Recent)", df["company_name"].nunique())
    col3.metric("Newest job", df["ts"].max().strftime("%Y-%m-%d %H:%M:%S"))

    st.write("Last refresh:", datetime.utcnow().strftime("%H:%M:%S.%f"))

    # ---------------- TABLE ----------------
    st.subheader("📥 Latest Jobs (Recent Stream)")
    cols = ["ts", "title", "company_name", "location", "country", "skill"]
    st.dataframe(df[cols].tail(100), use_container_width=True)

    # ---------------- AUTO REFRESH ----------------
    st.subheader("🔄 Live Updates (Auto Refresh Every 2s)")
    st_autorefresh(interval=2000, key="live_refresh")

    df_live = fetch_recent(60)

    if not df_live.empty:
        st.metric("Live rows last hour", len(df_live))
        st.dataframe(df_live.tail(20))
        per_min = per_bucket_counts(df_live)
    else:
        st.info("Waiting for live data…")

    # ---------------- COUNT HISTORY ----------------
    @st.cache_data(ttl=2)
    def get_total_count():
        session = get_session()
        result = session.execute(f"SELECT COUNT(*) AS count FROM {TABLE};")
        row = result.one()
        return row["count"] if row else 0

    if "count_history" not in st.session_state:
        st.session_state.count_history = []

    current_count = get_total_count()
    st.session_state.count_history.append({
        "time": datetime.utcnow(),
        "count": current_count
    })

    df_count = pd.DataFrame(st.session_state.count_history)

    st.subheader("📈 Total Jobs Over Time")
    st.line_chart(df_count.set_index("time")["count"])

    # ---------------- BUCKET FREQUENCY ----------------
    st.subheader("📊 Job Frequency (2s Buckets)")
    df_last6 = fetch_recent(6)

    if not df_last6.empty:
        df_last6["ts"] = pd.to_datetime(df_last6["ts"], errors="coerce")

        per_ts = (
            df_last6.set_index("ts")
            .assign(jobs=1)
            .resample("2s")["jobs"]
            .sum()
            .rename_axis("timestamp")
            .reset_index()
        )

        st.bar_chart(per_ts.set_index("timestamp")["jobs"])
    else:
        st.info("No 2-second frequency data yet…")

    # ---------------- TOP COMPANIES ----------------
    st.subheader("🏢 Top Companies (Recent Stream)")
    top_companies = df["company_name"].value_counts().head(10)
    st.bar_chart(top_companies)


# ==============================================================================
# ROUTING USING TOP MENU
# ==============================================================================
if current_phase == "Phase 1":
    load_other_phase("pages.phase1")

elif current_phase == "Phase 2":
    load_other_phase("pages.phase2")

elif current_phase == "Phase 3":
    run_phase3()  # 🔥 YOUR ENTIRE DASHBOARD

elif current_phase == "Phase 4":
    load_other_phase("pages.phase4")
