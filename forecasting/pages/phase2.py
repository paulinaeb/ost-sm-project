import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from footer_utils import add_footer
from data_utils import fetch_recent, fetch_all
from cassandra_client import validate_keyspace

# --- CONFIGURATION ---
# Citation: Cerqueira, V., et al. "Machine Learning vs Statistical Methods for Time Series Forecasting: Size Matters". 
# We implement a Recursive Multi-step Forecasting strategy using Random Forest.
LAG_FEATURES = [1, 2, 3, 4]  # Use past 4 weeks to predict next week
TEST_SIZE_WEEKS = 4          # Last 4 weeks for testing
VAL_SIZE_WEEKS = 4           # Previous 4 weeks for validation
FORECAST_HORIZON = 12        # Weeks to predict into the future

# def preprocess_data(raw_data):
#     """
#     Preprocess raw data to prepare it for training.
#     :param raw_data: DataFrame containing raw data from Cassandra.
#     :return: Processed DataFrame.
#     """
#     # Convert dates to datetime
#     raw_data['created_at'] = pd.to_datetime(raw_data['created_at'])
#     raw_data['ingested_at'] = pd.to_datetime(raw_data['ingested_at'])

#     # Add a 'metric' column (e.g., count of jobs per week)
#     raw_data['metric'] = 1  # Each row represents one job posting

#     return raw_data

def preprocess_weekly_data(df):
    """
    Aggregates raw job data into weekly counts per country and fills missing weeks.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Ensure datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    # Normalize to start of week (Monday)
    df['week'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
    
    # Group by Country and Week
    weekly_counts = df.groupby(['country', 'week']).size().reset_index(name='job_count')
    
    if weekly_counts.empty:
        return pd.DataFrame()

    # Fill gaps: Ensure every country has continuous weeks
    full_df = []
    all_weeks = pd.date_range(start=weekly_counts['week'].min(), end=weekly_counts['week'].max(), freq='W-MON')
    
    for country in weekly_counts['country'].unique():
        country_data = weekly_counts[weekly_counts['country'] == country].set_index('week')
        # Reindex to fill missing weeks with 0
        country_data = country_data.reindex(all_weeks, fill_value=0).reset_index().rename(columns={'index': 'week'})
        country_data['country'] = country
        full_df.append(country_data)
        
    return pd.concat(full_df, ignore_index=True)

def create_lag_features(df, lags):
    """
    Creates lag features for supervised learning.
    Input: Time series data.
    Output: DataFrame with columns like 'lag_1', 'lag_2', etc.
    """
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['job_count'].shift(lag)
    return df.dropna()

def split_data_time_series(df):
    """
    Splits data into Train, Validation, and Test sets respecting temporal order.
    """
    unique_weeks = sorted(df['week'].unique())
    # Safety check for small datasets
    if len(unique_weeks) < (TEST_SIZE_WEEKS + VAL_SIZE_WEEKS + 2):
        return df, pd.DataFrame(), pd.DataFrame()
    
    # Determine split points
    test_start = unique_weeks[-TEST_SIZE_WEEKS]
    val_start = unique_weeks[-(TEST_SIZE_WEEKS + VAL_SIZE_WEEKS)]
    
    train = df[df['week'] < val_start]
    val = df[(df['week'] >= val_start) & (df['week'] < test_start)]
    test = df[df['week'] >= test_start]

    return train, val, test

def train_model(train_df):
    """
    Trains a Random Forest Regressor.
    """
    X_train = train_df[[f'lag_{l}' for l in LAG_FEATURES]]
    y_train = train_df['job_count']
    
    # RandomForest is chosen for its robustness to non-linearities and lack of scaling requirement
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, test_df):
    """
    Calculates RMSE and MAE for the model.
    """
    if test_df.empty:
        return 0, 0, []
        
    X_test = test_df[[f'lag_{l}' for l in LAG_FEATURES]]
    y_test = test_df['job_count']
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return rmse, mae, predictions

def recursive_forecast(model, last_window, horizon):
    """
    Generates future predictions by feeding predictions back into the model.
    """
    future_predictions = []
    current_features = list(last_window) # Should be [lag_1, lag_2, ...]
    
    for _ in range(horizon):
        # Prepare features for single prediction
        # Scikit-learn expects 2D array
        features_array = np.array(current_features).reshape(1, -1)
        pred = model.predict(features_array)[0]
        future_predictions.append(max(0, pred)) # Relu: Jobs can't be negative
        
        # Update features: shift everything to right, new pred becomes lag_1
        # current: [t-1, t-2, t-3, t-4] -> new: [pred, t-1, t-2, t-3]
        current_features = [pred] + current_features[:-1]
        
    return future_predictions

def visualize_results(country, train, val, test, test_preds, future_preds, rmse):
    """
    Visualizes historical data, test performance, and future forecasts.
    """
    fig = go.Figure()

    # 1. Historical Data (Train + Val)
    history = pd.concat([train, val])
    fig.add_trace(go.Scatter(
        x=history['week'], 
        y=history['job_count'],
        mode='lines',
        name='Historical Data (Train/Val)',
        line=dict(color='gray', width=1)
    ))

    # 2. Test Data (Actual)
    fig.add_trace(go.Scatter(
        x=test['week'], 
        y=test['job_count'],
        mode='lines+markers',
        name='Actual Test Data',
        line=dict(color='blue')
    ))

    # 3. Test Predictions (Model Performance)
    fig.add_trace(go.Scatter(
        x=test['week'], 
        y=test_preds,
        mode='lines+markers',
        name=f'Model Validation (RMSE: {rmse:.2f})',
        line=dict(color='orange', dash='dot')
    ))

    # 4. Future Forecast
    last_date = test['week'].max()
    future_dates = [last_date + timedelta(weeks=i+1) for i in range(len(future_preds))]
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_preds,
        mode='lines+markers',
        name='Future Forecast (AI)',
        line=dict(color='green', width=3)
    ))

    fig.update_layout(
        title=f"Weekly Analytics & Forecast: {country}",
        xaxis_title="Week",
        yaxis_title="Job Postings",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

# def generate_predictions(df, weeks_to_predict=12):
#     """
#     Trains a model per country and generates future predictions.
#     """
#     df = df.copy()
    
#     # 1. Convert 'week' to a numerical value (Time Delta) for Linear Regression
#     # We use "days since the first date" so the math is continuous
#     min_date = df['week'].min()
#     df['time_index'] = (df['week'] - min_date).dt.days
    
#     future_preds = []

#     # 2. Train a separate simple model for each country
#     # (This is often more accurate than one giant model for simple trends)
#     for country in df['country'].unique():
#         country_data = df[df['country'] == country]
        
#         if len(country_data) < 2:
#             continue # Not enough data to predict
            
#         model = LinearRegression()
#         X = country_data[['time_index']]
#         y = country_data['metric']
        
#         model.fit(X, y)
        
#         # 3. Create Future Time Index
#         last_day = country_data['time_index'].max()
#         # Create input for the next N weeks (7 days * N weeks)
#         future_days = np.array([last_day + (i * 7) for i in range(1, weeks_to_predict + 1)]).reshape(-1, 1)
        
#         # 4. Predict
#         predictions = model.predict(future_days)
        
#         # 5. Construct the prediction DataFrame
#         future_dates = [country_data['week'].max() + pd.Timedelta(weeks=i) for i in range(1, weeks_to_predict + 1)]
        
#         temp_df = pd.DataFrame({
#             'week': future_dates,
#             'metric': predictions,
#             'country': country,
#             'type': 'Predicted' # Mark these as predictions
#         })
#         # Ensure no negative predictions (impossible to have negative jobs)
#         temp_df['metric'] = temp_df['metric'].clip(lower=0)
        
#         future_preds.append(temp_df)

#     # 6. Combine History (Actual) and Future (Predicted)
#     df['type'] = 'Actual'
#     if future_preds:
#         return pd.concat([df[['week', 'metric', 'country', 'type']], pd.concat(future_preds)])
#     return df

# def visualize_weekly_data(data):
#     """
#     Visualize weekly job postings by country.
#     :param data: Aggregated DataFrame with weekly job postings.
#     """
#     # 1. Ensure 'week' is actual datetime objects (Crucial for the x-axis to look good)
#     data['week'] = pd.to_datetime(data['week'])

#     # Calculate the total jobs per week across all countries
#     weekly_totals = data.groupby('week')['metric'].sum().reset_index()
    
#     # Find the first week where we have significant data (e.g., > 10 jobs total)
#     # This automatically finds where the "spike" starts
#     active_weeks = weekly_totals[weekly_totals['metric'] > 100]['week']
    
#     if not active_weeks.empty:
#         start_date = active_weeks.min()
#         # Filter the main dataframe to only show data after that start date
#         data = data[data['week'] >= start_date]

#     fig = px.area(
#         data,
#         x="week",
#         y="metric",
#         color="country",
#         title="Weekly Job Postings by Country",
#         labels={"week": "Week", "metric": "Job Postings", "country": "Country"},
#     )
#     st.plotly_chart(fig, use_container_width=True)
    
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

    st.title("📊 Phase 2: AI Job Market Forecasting")
    st.markdown("""
    This module implements **Recursive Multi-step Forecasting** using a **Random Forest Ensemble**.
    
    *Methodology Reference: Cerqueira, V. et al. "Machine Learning vs Statistical Methods for Time Series Forecasting".*
    """)
    
    # ---------------- FETCH DATA BASED ON MODE ----------------
    if mode == "📁 View Existing Database":
        # 3. Load Data
        with st.spinner('Fetching data from Cassandra...'):
            df_raw = fetch_all()
        
        if df_raw.empty:
            st.warning("⚠️ No data found in the database. Start the producer to ingest data.")
            st.stop()
        
         # ---------------- PREPROCESS DATA ----------------
        st.write("Preprocessing data...")
        #df = preprocess_data(df_raw)
        
        
        # 4. Preprocessing
        df_weekly = preprocess_weekly_data(df_raw)
        if df_weekly.empty:
            st.error("Data processing failed. Ensure data has valid dates and countries.")
            st.stop()
        st.write(df_weekly.head())
        
        # ---------------- VISUALIZE DATA ----------------
        #!!! st.write("Visualizing weekly job postings...")
        # df['week'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
        # weekly_data = df.groupby(['week', 'country'], as_index=False).agg({'metric': 'sum'})

        # 5. UI Controls
        all_countries = sorted(df_weekly['country'].unique())
        # Default to top 3 countries by volume
        top_countries = df_weekly.groupby('country')['job_count'].sum().nlargest(3).index.tolist()

        col1, col2 = st.columns([3, 1]) # Create columns for better layout
        with col1:
            selected_countries = st.multiselect(
                "Select Countries to Analyze:",
                options=all_countries,
                default=top_countries
            )
        with col2:
            st.metric("Total Jobs Analyzed", len(df_raw))

        if not selected_countries:
            st.info("Please select a country to generate predictions.")
            st.stop()
        # with col2:
        #     # The Checkbox (Placed to the right or top)
        #     select_all = st.checkbox("Select All Countries")

        # with col1:
        #     if select_all:
        #         # If checked, we disable the box and select everything
        #         selected_countries = all_countries
        #         st.info(f"✅ Displaying all {len(all_countries)} countries.")
        #     else:
        #         # Otherwise, show the picker
        #         selected_countries = st.multiselect(
        #             "Select Countries to Compare:",
        #             options=all_countries,
        #             default=top_countries
        #         )

        # 4. Filter the data based on selection
        # if not selected_countries:
        #     st.warning("⚠️ Please select at least one country to view the plot.")
        # else:
        #     filtered_data = weekly_data[weekly_data['country'].isin(selected_countries)]
            
        #     # Pass ONLY the filtered data to your plotting function
        #     visualize_weekly_data(filtered_data)

        # 6. Analysis Loop
    for country in selected_countries:
        st.markdown(f"### 🏳️ {country}")
        
        # Filter Data
        country_df = df_weekly[df_weekly['country'] == country].sort_values('week')
        

        # Need enough data for lags (4) + test (4) + val (4) + train (at least 4) = 16 weeks
        # We relax this slightly for demo purposes, but warn the user.
        MIN_WEEKS_REQUIRED = 12 
        
        if len(country_df) < MIN_WEEKS_REQUIRED:
            st.warning(f"Not enough historical data for **{country}** to train a reliable AI model (Need {MIN_WEEKS_REQUIRED}+ weeks, found {len(country_df)}). Showing raw trend only.")
            st.line_chart(country_df.set_index('week')['job_count'])
            continue

        # Feature Engineering
        df_features = create_lag_features(country_df, LAG_FEATURES)
        
        # Split
        train, val, test = split_data_time_series(df_features)
        if train.empty:
             st.error("Insufficient data for training split.")
             continue
        # Train
        model = train_model(train)
        
        # Evaluate
        rmse, mae, test_preds = evaluate_model(model, test)
        
        # # Future Forecast
        # # Get the very last window of known data (from test set)
        # last_window = df_features.iloc[-1][[f'lag_{l}' for l in LAG_FEATURES]].values
        # future_preds = recursive_forecast(model, last_window, FORECAST_HORIZON)
        
        # # Metrics Display
        # m1, m2, m3 = st.columns(3)
        # m1.metric("Model RMSE (Error)", f"{rmse:.2f}", help="Lower is better")
        # m2.metric("Next Week Prediction", f"{int(future_preds[0])}", delta=f"{int(future_preds[0] - country_df.iloc[-1]['job_count'])}")
        # m3.metric("12-Week Trend", "Growth" if future_preds[-1] > future_preds[0] else "Decline")
        
        # # Visualize
        # fig = visualize_results(country, train, val, test, test_preds, future_preds, rmse)
        # st.plotly_chart(fig, use_container_width=True)
        
        # with st.expander(f"View Detailed Data for {country}"):
        #     st.dataframe(country_df.tail(10))

        # Future Forecast
        # Get the very last window of known data (from test set or validation set)
        last_known_data = pd.concat([train, val, test]).iloc[-1]
        last_window = last_known_data[[f'lag_{l}' for l in LAG_FEATURES]].values
        
        future_preds = recursive_forecast(model, last_window, FORECAST_HORIZON)
        
        # Metrics Display
        m1, m2, m3 = st.columns(3)
        m1.metric("Model RMSE (Accuracy)", f"{rmse:.2f}", help="Root Mean Squared Error. Lower is better.")
        
        next_week_pred = int(future_preds[0])
        current_val = int(country_df.iloc[-1]['job_count'])
        delta = next_week_pred - current_val
        
        m2.metric("Next Week Prediction", f"{next_week_pred}", delta=delta)
        
        trend = "📈 Growth" if future_preds[-1] > future_preds[0] else "📉 Decline"
        m3.metric("12-Week Forecast Trend", trend)
        
        # Visualize
        fig = visualize_results(country, train, val, test, test_preds, future_preds, rmse)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"View Training Data for {country}"):
            st.dataframe(country_df.tail(10))


    else:  # Real-time Streaming Mode
        st_autorefresh(interval=3000, key="predictive_insights_refresh")
        
        df_raw = fetch_recent(LOOKBACK_MINUTES)
        
        if df_raw.empty:
            st.warning("⚠️ No live data yet. Start the producer to see real-time updates.")
            st.info("Waiting for streaming data...")
            st.stop()
        
        #df = preprocess_data(df_raw)
        
        if df.empty:
            st.warning("⚠️ No European country data in recent stream.")
            st.stop()

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
        
        st.success(f"🔴 LIVE: {len(df_raw)} jobs")
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

        visualize_weekly_data(df)
    st.divider()

    add_footer("Tibor Buti")