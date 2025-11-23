import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from footer_utils import add_footer
from data_utils import fetch_recent, fetch_all
from cassandra_client import validate_keyspace

def preprocess_data(raw_data):
    """
    Preprocess raw data to prepare it for training.
    :param raw_data: DataFrame containing raw data from Cassandra.
    :return: Processed DataFrame.
    """
    # Convert dates to datetime
    #raw_data['posting_date'] = pd.to_datetime(raw_data['posting_date'])
    #raw_data['retrieval_date'] = pd.to_datetime(raw_data['retrieval_date'])

    # Add a 'metric' column (e.g., count of jobs per week)
    raw_data['metric'] = 1  # Each row represents one job posting

    return raw_data
# ---------------- DATA PROCESSING ----------------
# def prepare_data(df):
#     """Add ISO codes and filter European countries"""
#     if df.empty:
#         return df
        
#     df['country_iso'] = df['country'].apply(normalize_country)
#     # Filter only European countries
#     df = df[df['country_iso'].isin(EUROPEAN_COUNTRIES)].copy()
#     return df
def train_weekly_country_model(data):
    """
    Train a prediction model for weekly per-country analytics.
    :param data: DataFrame containing the data.
    """
    # Ensure the 'date' column is in datetime format
    #data['date'] = pd.to_datetime(data['date'])

    # Aggregate data by week and country
    data['week'] = data['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_data = data.groupby(['week', 'country']).sum().reset_index()

    # Train a simple model (e.g., Linear Regression)
    model = LinearRegression()
    X = pd.get_dummies(weekly_data[['week', 'country']], drop_first=True)
    y = weekly_data['metric']  # Replace 'metric' with the target column in your data
    model.fit(X, y)
    return model

def run():
    st.title("📈 Predictive Insights")

    # ---------------- SETTINGS ----------------
    TABLE = os.getenv("CASSANDRA_TABLE", "jobs")
    LOOKBACK_MINUTES = 60

    st.write("Content for Predictive Insights page coming soon...")

    # ---------------- VALIDATE KEYSPACE ----------------
    keyspace_exists, error_msg = validate_keyspace()
    if not keyspace_exists:
        st.error(f"❌ **Database Connection Error**")
        st.warning(error_msg)
        st.info("""
        **To fix this:**
        1. Make sure Cassandra is running: `docker-compose up -d`
        2. Create the database by running the consumer: `python streaming\\kafka_consumer.py`
        3. Start the producer to ingest data: `python streaming\\kafka_producer.py`
        4. Refresh this page.
        """)
        st.stop()

    # ---------------- MODE SELECTOR ----------------
    st.subheader("📊 View Mode")
    
    mode = st.radio(
        "Choose how to view data:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"],
        horizontal=True
    )
    
    # ---------------- FETCH DATA BASED ON MODE ----------------
    if mode == "📁 View Existing Database":
        df_raw = fetch_all()
        
        if df_raw.empty:
            st.warning("⚠️ No data found in the database. Start the producer to ingest data.")
            st.stop()
        
         # ---------------- PREPROCESS DATA ----------------
        st.write("Preprocessing data...")
        df = preprocess_data(df_raw)
        
        # if df.empty:
        #     st.warning("⚠️ No European country data found in the database.")
        #     st.stop()
        
        # ---------------- TRAIN MODEL ----------------
        st.write("Training the weekly per-country analytics model...")
        model = train_weekly_country_model(df)
        st.success("Model training complete!")
        #st.success(f"✅ Loaded {len(df)} jobs from {df['country_iso'].nunique()} European countries")
        
    else:  # Real-time Streaming Mode
        st_autorefresh(interval=3000, key="country_radar_refresh")
        
        df_raw = fetch_recent(LOOKBACK_MINUTES)
        
        if df_raw.empty:
            st.warning("⚠️ No live data yet. Start the producer to see real-time updates.")
            st.info("Waiting for streaming data...")
            st.stop()
        
        df = prepare_data(df_raw)
        
        if df.empty:
            st.warning("⚠️ No European country data in recent stream.")
            st.stop()
        
        st.success(f"🔴 LIVE: {len(df)} jobs from {df['country_iso'].nunique()} countries")
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    st.divider()

    add_footer("Tibor Buti")