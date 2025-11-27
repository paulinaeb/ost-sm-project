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

# Configuration will now be dynamic based on user selection
DEFAULT_CONFIG = {
    'weekly': {
        'LAG_FEATURES': [1, 2, 3, 4],  # Use past 4 weeks
        'TEST_SIZE': 3,                 # Last 3 weeks for testing
        'VAL_SIZE': 3,                  # Previous 3 weeks for validation
        'FORECAST_HORIZON': 3,         # Periods to predict into the future
        'DATA_START_THRESHOLD': 10,     # Minimum jobs per period
        'MIN_PERIODS_REQUIRED': 12,     # Minimum periods needed
        'FREQ': 'W-MON',                # Week starting Monday
        'PERIOD_NAME': 'week',
        'TIMEDELTA_UNIT': 'weeks'
    },
    'daily': {
        'LAG_FEATURES': [1, 2, 3, 7, 14],  # Use past 1, 2, 3 days, 1 week, 2 weeks
        'TEST_SIZE': 7,                    # Last 7 days for testing
        'VAL_SIZE': 7,                     # Previous 7 days for validation
        'FORECAST_HORIZON': 7,             # Periods to predict into the future
        'DATA_START_THRESHOLD': 5,          # Minimum jobs per day
        'MIN_PERIODS_REQUIRED': 42,         # Minimum days needed (6 weeks)
        'FREQ': 'D',                        # Daily
        'PERIOD_NAME': 'date',
        'TIMEDELTA_UNIT': 'days'
    }
}

# Streaming Config
LOOKBACK_MINUTES = 60        # How far back to look for the "Live" stream

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

# --- PREPROCESSING & ML FUNCTIONS (BATCH LAYER) ---

def preprocess_temporal_data(df, config):
    """
    Aggregates raw job data into specified time periods per country.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Normalize to period based on granularity
    if config['FREQ'] == 'W-MON':
        df['period'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
    else:  # daily
        df['period'] = df['created_at'].dt.floor('D')
    
    temporal_counts = df.groupby(['country', 'period']).size().reset_index(name='job_count')
    
    if temporal_counts.empty:
        return pd.DataFrame()

    max_date = temporal_counts['period'].max()
    full_df = []

    for country in temporal_counts['country'].unique():
        country_data = temporal_counts[temporal_counts['country'] == country].sort_values('period')
        
        active_periods = country_data[country_data['job_count'] > config['DATA_START_THRESHOLD']]['period']
        
        if not active_periods.empty:
            start_date = active_periods.min()
            country_data = country_data[country_data['period'] >= start_date]
        else:
            start_date = country_data['period'].min()

        country_data = country_data.set_index('period')
        relevant_periods = pd.date_range(start=start_date, end=max_date, freq=config['FREQ'])
        
        country_data = country_data.reindex(relevant_periods, fill_value=0).reset_index().rename(columns={'index': 'period'})
        country_data['country'] = country
        full_df.append(country_data)
        
    return pd.concat(full_df, ignore_index=True)

def create_lag_features(df, config):
    """
    Creates lag features for supervised learning.
    """
    df = df.copy()
    for lag in config['LAG_FEATURES']:
        df[f'lag_{lag}'] = df['job_count'].shift(lag)
    return df.dropna()

def split_data_time_series(df, config):
    """
    Splits data into Train, Validation, and Test sets respecting temporal order.
    """
    unique_periods = sorted(df['period'].unique())
    
    if len(unique_periods) < (config['TEST_SIZE'] + config['VAL_SIZE'] + 2):
        return df, pd.DataFrame(), pd.DataFrame()
    
    test_start = unique_periods[-config['TEST_SIZE']]
    val_start = unique_periods[-(config['TEST_SIZE'] + config['VAL_SIZE'])]
    
    train = df[df['period'] < val_start]
    val = df[(df['period'] >= val_start) & (df['period'] < test_start)]
    test = df[df['period'] >= test_start]

    return train, val, test

def train_model(train_df, config):
    """
    Trains a Random Forest Regressor.
    """
    X_train = train_df[[f'lag_{l}' for l in config['LAG_FEATURES']]]
    y_train = train_df['job_count']
    
    # RandomForest is chosen for its robustness to non-linearities and lack of scaling requirement
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, test_df, config):
    """
    Calculates RMSE and MAE for the model.
    """
    if test_df.empty:
        return 0, 0, []
        
    X_test = test_df[[f'lag_{l}' for l in config['LAG_FEATURES']]]
    y_test = test_df['job_count']
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return rmse, mae, predictions

def recursive_forecast(model, last_window, config):
    """
    Generates future predictions by feeding predictions back into the model.
    """
    future_predictions = []
    current_features = list(last_window) # Should be [lag_1, lag_2, ...]
    
    for _ in range(config['FORECAST_HORIZON']):
        # Prepare features for single prediction
        # Scikit-learn expects 2D array
        features_array = np.array(current_features).reshape(1, -1)
        pred = model.predict(features_array)[0]
        pred = max(0, pred) # Relu: Jobs can't be negative
        future_predictions.append(pred)

        # Update features: shift everything to right, new pred becomes lag_1
        # current: [t-1, t-2, t-3, t-4] -> new: [pred, t-1, t-2, t-3]
        current_features = [pred] + current_features[:-1]
        
    return future_predictions

def visualize_results(country, train, val, test, test_preds, future_preds, rmse, config):
    """
    Visualizes historical data, test performance, and future forecasts.
    """
    fig = go.Figure()
    period_name = config['PERIOD_NAME'].capitalize()

    history = pd.concat([train, val])
    fig.add_trace(go.Scatter(
        x=history['period'], 
        y=history['job_count'],
        mode='lines',
        name='Historical Data (Train/Val)',
        line=dict(color='gray', width=1)
    ))

    if not test.empty:
        fig.add_trace(go.Scatter(
            x=test['period'], 
            y=test['job_count'],
            mode='lines+markers',
            name='Actual Test Data',
            line=dict(color='blue')
        ))
        
        if len(test_preds) > 0:
            fig.add_trace(go.Scatter(
                x=test['period'], 
                y=test_preds,
                mode='lines+markers',
                name=f'Model Validation (RMSE: {rmse:.2f})',
                line=dict(color='orange', dash='dot')
            ))    

    last_date = test['period'].max() if not test.empty else history['period'].max()
    
    if config['TIMEDELTA_UNIT'] == 'weeks':
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(len(future_preds))]
    else:  # days
        future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_preds))]
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_preds,
        mode='lines+markers',
        name='Future Forecast (AI)',
        line=dict(color='green', width=3)
    ))

    fig.update_layout(
        title=f"{period_name}ly Analytics & Forecast: {country}",
        xaxis_title=period_name,
        yaxis_title="Job Postings",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

# --- STREAMING FUNCTIONS (SPEED LAYER) ---

def visualize_streaming_data(df):
    """
    Simple, fast visualization for real-time data monitoring.
    """
    fig = go.Figure()
    
    # Aggregate by minute/hour to show ingestion speed
    # We use the raw 'created_at' to show density over time
    for country in df['country'].unique():
        country_df = df[df['country'] == country]
        # Group by minute for granularity
        time_series = country_df.set_index('created_at').resample('min').size()
        
        fig.add_trace(go.Scatter(
            x=time_series.index,
            y=time_series.values,
            mode='lines+markers',
            name=country,
            stackgroup='one' # Stacked area chart for volume
        ))

    fig.update_layout(
        title="Real-Time Ingestion Volume (Last Hour)",
        xaxis_title="Time (UTC)",
        yaxis_title="Jobs Ingested per Minute",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_weekly_data(data, granularity='weekly'):
    """
    Visualize temporal job postings by country.
    """
    data['period'] = pd.to_datetime(data['period'])
    config = DEFAULT_CONFIG[granularity]
    period_name = config['PERIOD_NAME'].capitalize()

    period_totals = data.groupby('period')['metric'].sum().reset_index()
    active_periods = period_totals[period_totals['metric'] > config['DATA_START_THRESHOLD'] * 2]['period']
    
    if not active_periods.empty:
        start_date = active_periods.min()
        data = data[data['period'] >= start_date]

    fig = px.area(
        data,
        x="period",
        y="metric",
        color="country",
        title=f"{period_name}ly Job Postings by Country",
        labels={"period": period_name, "metric": "Job Postings", "country": "Country"},
    )
    st.plotly_chart(fig, use_container_width=True)
    
def run():
    st.title("📈 Predictive Insights")

    TABLE = os.getenv("CASSANDRA_TABLE", "jobs")


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

    # ---------------- TIME GRANULARITY SELECTOR ----------------
    st.subheader("⏱️ Forecast Granularity")
    granularity = st.radio(
        "Select time period for forecasting:",
        ["weekly", "daily"],
        horizontal=True,
        help="Choose whether to forecast by weeks or days"
    )
    
    config = DEFAULT_CONFIG[granularity]
    period_name = config['PERIOD_NAME'].capitalize()

    # ---------------- MODE SELECTOR ----------------
    st.subheader("📊 View Mode")
    
    mode = st.radio(
        "Choose how to view data:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"],
        horizontal=True
    )

    st.title("📊 Phase 2: AI Job Market Forecasting")
    st.markdown(f"""
    This module implements **Recursive Multi-step Forecasting** using a **Random Forest Ensemble**.
    * **Scientific Approach:** Uses lag features and time-series cross-validation.
    * **Granularity:** {period_name}ly forecasts for {config['FORECAST_HORIZON']} periods ahead
    * **Data Filter:** Automatically detects the "significant start" of data (>{config['DATA_START_THRESHOLD']} jobs/{config['PERIOD_NAME']}) to ignore pre-collection noise.
    *Methodology Reference: Cerqueira, V. et al. "Machine Learning vs Statistical Methods for Time Series Forecasting".*
    """)
    
    # ---------------- FETCH DATA BASED ON MODE ----------------
    if mode == "📁 View Existing Database":
        with st.spinner('Fetching data from Cassandra...'):
            df_raw = fetch_all()
        
        if df_raw.empty:
            st.warning("⚠️ No data found in the database. Start the producer to ingest data.")
            st.stop()
        
        st.write("Preprocessing data...")
        df = preprocess_data(df_raw)
        
        df_temporal = preprocess_temporal_data(df_raw, config)
        if df_temporal.empty:
            st.error("Data processing failed. Ensure data has valid dates and countries.")
            st.stop()
        st.write(df_temporal.head())
        
        # Prepare data for visualization
        df['period'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time) if granularity == 'weekly' else df['created_at'].dt.floor('D')
        temporal_data = df.groupby(['period', 'country'], as_index=False).agg({'metric': 'sum'})

        all_countries = sorted(df_temporal['country'].unique())
        #the ones with most jobs top_countries = df_temporal.groupby('country')['job_count'].sum().nlargest(3).index.tolist()
        # Calculate RMSE for each country to find best predictions
        country_rmse = {}
    
        for country in all_countries:
            country_df = df_temporal[df_temporal['country'] == country].sort_values('period')
        
            if len(country_df) < config['MIN_PERIODS_REQUIRED']:
                continue
        
            try:
                df_features = create_lag_features(country_df, config)
                train, val, test = split_data_time_series(df_features, config)
            
                if train.empty or test.empty:
                    continue
            
                model = train_model(train, config)
                rmse, mae, test_preds = evaluate_model(model, test, config)
            
                # Only include countries with meaningful predictions (average > 0)
                if rmse is not None and test_preds is not None:
                    avg_prediction = np.mean(test_preds)
                    avg_actual = test['job_count'].mean()
                
                    # Filter: require average predictions > 0 and average actual values > threshold
                    if avg_prediction > 0 and avg_actual > config['DATA_START_THRESHOLD']:
                        country_rmse[country] = rmse
            except Exception as e:
                continue
        
        # Select top 3 countries with lowest RMSE (best predictions)
        if country_rmse:
            top_countries = sorted(country_rmse.items(), key=lambda x: x[1])[:5]
            top_countries = [country for country, rmse in top_countries]
        else:
            # Fallback to countries with most jobs if no RMSE calculated
            top_countries = df_temporal.groupby('country')['job_count'].sum().nlargest(3).index.tolist()

        col1, col2 = st.columns([3, 1])
        
        with col2:
            select_all = st.checkbox("Select All Countries")

        with col1:
            if select_all:
                selected_countries = all_countries
                st.info(f"✅ Displaying all {len(all_countries)} countries.")
            else:
                selected_countries = st.multiselect(
                    "Select Countries to Compare:",
                    options=all_countries,
                    default=top_countries
                )

        if not selected_countries:
            st.warning("⚠️ Please select at least one country to view the plot.")
        else:
            filtered_data = temporal_data[temporal_data['country'].isin(selected_countries)]
            visualize_weekly_data(filtered_data, granularity)
            # Display RMSE ranking for transparency
            if country_rmse:
                with st.expander("📊 View Prediction Accuracy by Country"):
                    rmse_df = pd.DataFrame(
                        sorted(country_rmse.items(), key=lambda x: x[1]),
                        columns=['Country', 'RMSE']
                    )
                    rmse_df['Rank'] = range(1, len(rmse_df) + 1)
                    st.dataframe(
                        rmse_df[['Rank', 'Country', 'RMSE']].style.format({'RMSE': '{:.2f}'}),
                        hide_index=True
                    )
                    st.caption("Lower RMSE = More accurate predictions")

        for country in selected_countries:
            st.markdown(f"### 🏳️ {country}") 
            country_df = df_temporal[df_temporal['country'] == country].sort_values('period')
        
            if len(country_df) < config['MIN_PERIODS_REQUIRED']:
                st.warning(f"Not enough historical data for **{country}** to train a reliable AI model (Need {config['MIN_PERIODS_REQUIRED']}+ {config['PERIOD_NAME']}s, found {len(country_df)}). Showing raw trend only.")
                st.line_chart(country_df.set_index('period')['job_count'])
                continue

            df_features = create_lag_features(country_df, config)
            train, val, test = split_data_time_series(df_features, config)
            
            if train.empty:
                st.error("Insufficient data for training split.")
                continue
                
            model = train_model(train, config)
            rmse, mae, test_preds = evaluate_model(model, test, config)

            last_known_data = pd.concat([train, val, test]).iloc[-1]
            last_window = last_known_data[[f'lag_{l}' for l in config['LAG_FEATURES']]].values
            future_preds = recursive_forecast(model, last_window, config)
        
            m1, m2, m3 = st.columns(3)
            m1.metric("Model RMSE (Accuracy)", f"{rmse:.2f}", help="Root Mean Squared Error. Lower is better.")
        
            next_period_pred = int(future_preds[0])
            current_val = int(country_df.iloc[-1]['job_count'])
            delta = next_period_pred - current_val
        
            m2.metric(f"Next {period_name} Prediction", f"{next_period_pred}", delta=delta)
        
            trend = "📈 Growth" if future_preds[-1] > future_preds[0] else "📉 Decline"
            m3.metric(f"{config['FORECAST_HORIZON']}-{period_name} Forecast Trend", trend)
        
            fig = visualize_results(country, train, val, test, test_preds, future_preds, rmse, config)
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
        
        df = preprocess_data(df_raw)
        df['period'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time) if granularity == 'weekly' else df['created_at'].dt.floor('D')
        temporal_data = df.groupby(['period', 'country'], as_index=False).agg({'metric': 'sum'})

        all_countries = sorted(temporal_data['country'].unique())
        top_countries = df.groupby('country')['metric'].sum().nlargest(5).index.tolist()

        col1, col2 = st.columns([3, 1])

        with col2:
            select_all = st.checkbox("Select All Countries")

        with col1:
            if select_all:
                selected_countries = all_countries
                st.info(f"✅ Displaying all {len(all_countries)} countries.")
            else:
                selected_countries = st.multiselect(
                    "Select Countries to Compare:",
                    options=all_countries,
                    default=top_countries
                )

        if not selected_countries:
            st.warning("⚠️ Please select at least one country to view the plot.")
        else:
            filtered_data = temporal_data[temporal_data['country'].isin(selected_countries)]
            st.success(f"🔴 LIVE: {len(df_raw)} jobs")
            st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
            visualize_weekly_data(filtered_data, granularity)
            
            df_stream = df_raw.copy()
            df_stream['created_at'] = pd.to_datetime(df_stream['created_at'])
            fig = px.histogram(df_stream, x="created_at", color="country", nbins=60, title="Ingestion Velocity (Events/Minute)")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
    st.divider()
    add_footer("Tibor Buti")