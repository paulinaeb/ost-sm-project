import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
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
    raw_data['created_at'] = pd.to_datetime(raw_data['created_at'])
    raw_data['ingested_at'] = pd.to_datetime(raw_data['ingested_at'])

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

    # Ensure the 'created_at' column is in datetime format
    data['week'] = data['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
    # Aggregate data by week and country, summing only numeric columns
    weekly_data = data.groupby(['week', 'country'], as_index=False).agg({'metric': 'sum'})
    st.write(weekly_data.head())
    # Convert 'week' to string for compatibility with pd.get_dummies()
    weekly_data['week'] = weekly_data['week'].astype(str)
    st.write(weekly_data.head())
    # Train a simple model (e.g., Linear Regression)
    model = LinearRegression()
    X = pd.get_dummies(weekly_data[['week', 'country']], drop_first=True)
    y = weekly_data['metric']  # Replace 'metric' with the target column in your data
    model.fit(X, y)
    return model

def visualize_weekly_data(data):
    """
    Visualize weekly job postings by country.
    :param data: Aggregated DataFrame with weekly job postings.
    """
    # 1. Ensure 'week' is actual datetime objects (Crucial for the x-axis to look good)
    data['week'] = pd.to_datetime(data['week'])

    # Calculate the total jobs per week across all countries
    weekly_totals = data.groupby('week')['metric'].sum().reset_index()
    
    # Find the first week where we have significant data (e.g., > 10 jobs total)
    # This automatically finds where the "spike" starts
    active_weeks = weekly_totals[weekly_totals['metric'] > 100]['week']
    
    if not active_weeks.empty:
        start_date = active_weeks.min()
        # Filter the main dataframe to only show data after that start date
        data = data[data['week'] >= start_date]

    fig = px.area(
        data,
        x="week",
        y="metric",
        color="country",
        title="Weekly Job Postings by Country",
        labels={"week": "Week", "metric": "Job Postings", "country": "Country"},
    )
    st.plotly_chart(fig, use_container_width=True)

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
        st.write(df.head())
        
        # if df.empty:
        #     st.warning("⚠️ No European country data found in the database.")
        #     st.stop()
        
        # ---------------- TRAIN MODEL ----------------
        st.write("Training the weekly per-country analytics model...")
        model = train_weekly_country_model(df)
        st.success("Model training complete!")
        #st.success(f"✅ Loaded {len(df)} jobs from {df['country_iso'].nunique()} European countries")
        
        # ---------------- VISUALIZE DATA ----------------
        st.write("Visualizing weekly job postings...")
        df['week'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_data = df.groupby(['week', 'country'], as_index=False).agg({'metric': 'sum'})

        # 2. Get list of countries & identify the Top 5 (for default selection)
        all_countries = sorted(weekly_data['country'].unique())
        
        # Calculate top 5 countries by total volume so the chart isn't empty on load
        top_countries = df.groupby('country')['metric'].sum().nlargest(5).index.tolist()

        col1, col2 = st.columns([3, 1]) # Create columns for better layout

        with col2:
            # The Checkbox (Placed to the right or top)
            select_all = st.checkbox("Select All Countries")

        with col1:
            if select_all:
                # If checked, we disable the box and select everything
                selected_countries = all_countries
                st.info(f"✅ Displaying all {len(all_countries)} countries.")
            else:
                # Otherwise, show the picker
                selected_countries = st.multiselect(
                    "Select Countries to Compare:",
                    options=all_countries,
                    default=top_countries
                )
                
        # 4. Filter the data based on selection
        if not selected_countries:
            st.warning("⚠️ Please select at least one country to view the plot.")
        else:
            filtered_data = weekly_data[weekly_data['country'].isin(selected_countries)]
            
        # Pass ONLY the filtered data to your plotting function
        visualize_weekly_data(filtered_data)


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