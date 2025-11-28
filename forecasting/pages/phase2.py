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
import re
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
from footer_utils import add_footer
from data_utils import fetch_recent, fetch_all, preprocess_temporal_data, build_work_mode_series, WORK_MODES
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
    raw_data['created_at'] = pd.to_datetime(raw_data['created_at'], errors='coerce', dayfirst=True, infer_datetime_format=True)
    raw_data['ingested_at'] = pd.to_datetime(raw_data['ingested_at'], errors='coerce', dayfirst=True, infer_datetime_format=True)

    # Add a 'metric' column (e.g., count of jobs per week)
    raw_data['metric'] = 1  # Each row represents one job posting

    return raw_data

# --- PREPROCESSING & ML FUNCTIONS (BATCH LAYER) ---


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
                name=f'Test Predictions (RMSE {rmse:.2f})',
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


# --- WORK MODE CLASSIFICATION & FORECASTING ---



def _add_time_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    p = pd.to_datetime(df['period'])
    df['dow'] = p.dt.weekday
    df['week'] = p.dt.isocalendar().week.astype(int)
    df['month'] = p.dt.month
    return df

def _future_dates(last_date: pd.Timestamp, horizon: int, unit: str):
    if unit == 'weeks':
        return [last_date + timedelta(weeks=i+1) for i in range(horizon)]
    return [last_date + timedelta(days=i+1) for i in range(horizon)]

def forecast_work_mode_trends_ml(series_df: pd.DataFrame, config: dict):
    forecasts = {}
    if series_df.empty:
        return forecasts
    horizon = config['FORECAST_HORIZON']
    for mode in WORK_MODES:
        s = series_df[series_df['work_mode'] == mode].sort_values('period')
        if len(s) < max(10, len(config['LAG_FEATURES']) + config['VAL_SIZE'] + config['TEST_SIZE'] + 2):
            continue
        # Prepare features
        s2 = s.rename(columns={'count': 'job_count'})[['period', 'job_count']]
        s2 = create_lag_features(s2, config)
        s2 = _add_time_features(s2, config)
        train, val, test = split_data_time_series(s2, config)
        if train.empty or test.empty:
            continue
        feature_cols = [f'lag_{l}' for l in config['LAG_FEATURES']] + ['dow', 'week', 'month']
        X_train, y_train = train[feature_cols], train['job_count']
        X_test, y_test = test[feature_cols], test['job_count']
        # Model: XGBoost if available, else RandomForest fallback
        if XGBRegressor is not None:
            model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=2
            )
        else:
            model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        # Recursive multi-step forecast with time features
        hist = pd.concat([train, val, test]).sort_values('period')
        last_row = hist.iloc[-1]
        current_lags = [last_row[f'lag_{l}'] for l in config['LAG_FEATURES']]
        last_date = pd.to_datetime(hist['period']).max()
        future_dates = _future_dates(last_date, horizon, config['TIMEDELTA_UNIT'])
        future_preds = []
        for d in future_dates:
            tf = pd.DataFrame({'period': [d]})
            tf = _add_time_features(tf, config)
            feats = current_lags + [int(tf['dow'].iloc[0]), int(tf['week'].iloc[0]), int(tf['month'].iloc[0])]
            yhat = float(model.predict(np.array(feats).reshape(1, -1))[0])
            yhat = max(0.0, yhat)
            future_preds.append(yhat)
            # update lags
            current_lags = [yhat] + current_lags[:-1]
        forecasts[mode] = (future_dates, np.array(future_preds), rmse)
    return forecasts

def _smooth_series(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    df = df.copy()
    df['count'] = df.groupby('work_mode')['count'].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return df

def _to_percentage_stack(df: pd.DataFrame) -> pd.DataFrame:
    # Convert counts to percentage-of-total per period
    df = df.copy()
    totals = df.groupby('period')['count'].transform('sum')
    df['count'] = np.where(totals > 0, df['count'] / totals * 100.0, 0.0)
    return df

def plot_work_mode_trends(series_df: pd.DataFrame, forecasts: dict, config: dict, *, smooth_window: int = 1, percent: bool = True):
    if series_df.empty:
        st.info("No classified work-mode data to display.")
        return
    # Optional smoothing and normalization
    series_proc = _smooth_series(series_df, smooth_window)
    if percent:
        series_proc = _to_percentage_stack(series_proc)
    fig = go.Figure()
    period_name = config['PERIOD_NAME'].capitalize()
    # Historical stacked area
    for mode in WORK_MODES:
        s = series_proc[series_proc['work_mode'] == mode]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s['period'],
            y=s['count'],
            mode='lines',
            name=f"{mode} (hist)",
            stackgroup='workmode'
        ))
    # Forecast overlays
    colors = {"Remote": "#1f77b4", "Hybrid": "#ff7f0e", "On-site": "#2ca02c"}
    # When displaying as percent, compute per-step totals across modes for normalization
    totals = None
    if percent and forecasts:
        # Determine horizon from any mode
        first = next(iter(forecasts.values()))
        horizon = len(first[1]) if len(first) >= 2 else 0
        if horizon > 0:
            totals = np.zeros(horizon)
            for _mode, _data in forecasts.items():
                fy = np.array(_data[1]) if len(_data) >= 2 else np.array([])
                if fy.size == horizon:
                    totals += fy
    for mode, data in forecasts.items():
        future_x, future_y = data[0], data[1]
        rmse_txt = f" (RMSE {data[2]:.2f})" if len(data) >= 3 and data[2] is not None else ""
        if percent and totals is not None and len(future_y) == len(totals):
            fy = np.array(future_y)
            future_y = np.where(totals > 0, fy / totals * 100.0, 0.0)
        fig.add_trace(go.Scatter(
            x=future_x,
            y=future_y,
            mode='lines+markers',
            name=f"{mode} forecast{rmse_txt}",
            line=dict(color=colors.get(mode, None), dash='dot', width=3)
        ))
    y_title = "Share of Postings (%)" if percent else "Job Postings"
    fig.update_layout(
        title=f"Global Work-Mode Trends ({period_name}ly)",
        xaxis_title=period_name,
        yaxis_title=y_title,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    # Add latest value annotations safely per mode
    for mode in WORK_MODES:
        dm = series_proc[series_proc['work_mode'] == mode]
        if dm.empty:
            continue
        lv = dm.sort_values('period').iloc[-1]
        suffix = '%' if percent else ''
        fig.add_annotation(x=lv['period'], y=lv['count'], text=f"{mode}: {lv['count']:.0f}{suffix}", showarrow=True, arrowhead=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_work_mode_snapshot(series_df: pd.DataFrame, forecasts: dict, config: dict):
    if series_df.empty:
        st.info("No classified work-mode data to display.")
        return
    period_name = config['PERIOD_NAME'].capitalize()
    periods = sorted(pd.to_datetime(series_df['period'].unique()))
    if not periods:
        st.info("No periods available for snapshot.")
        return
    selected = st.select_slider(f"Select {period_name}", options=periods, value=periods[-1])
    snap = series_df[series_df['period'] == selected]
    snap = snap.groupby('work_mode', as_index=False)['count'].sum()
    if snap['count'].sum() == 0:
        st.info("No postings in the selected period.")
        return
    colors = {"Remote": "#1f77b4", "Hybrid": "#ff7f0e", "On-site": "#2ca02c"}
    pie = px.pie(
        snap,
        names='work_mode',
        values='count',
        title=f"Snapshot: {period_name} {selected.strftime('%Y-%m-%d')}",
        color='work_mode',
        color_discrete_map=colors,
        hole=0.5
    )
    pie.update_traces(textposition='inside', textinfo='percent+label')
    pie.update_layout(template='plotly_dark', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(pie, use_container_width=True)

    # Trend indicators from latest period to next forecasted period
    st.caption("Model next-period trend from latest data (percent shares)")
    latest_period = series_df['period'].max()
    latest = series_df[series_df['period'] == latest_period].groupby('work_mode')['count'].sum().to_dict()
    # Sum forecasts across modes to get predicted total
    pred_next = {}
    total_pred = 0.0
    for mode in WORK_MODES:
        data = forecasts.get(mode)
        if data and len(data[1]) > 0:
            yhat1 = float(data[1][0])
            pred_next[mode] = yhat1
            total_pred += yhat1
    current_total = float(sum(latest.values())) if len(latest) > 0 else 0.0
    cols = st.columns(3)
    for i, mode in enumerate(WORK_MODES):
        with cols[i % 3]:
            curr = float(latest.get(mode, 0.0))
            nxt = float(pred_next.get(mode, np.nan))
            if np.isnan(nxt):
                st.metric(f"{mode}", "—", delta=None)
            else:
                curr_share = (curr / current_total * 100.0) if current_total > 0 else 0.0
                pred_share = (nxt / total_pred * 100.0) if total_pred > 0 else np.nan
                if np.isnan(pred_share):
                    st.metric(f"{mode}", "—", delta=None)
                else:
                    direction = "📈" if pred_share > curr_share else ("📉" if pred_share < curr_share else "➡️")
                    delta_pp = pred_share - curr_share
                    st.metric(f"{mode} {direction}", f"{curr_share:.1f}% → {pred_share:.1f}%", delta=f"{delta_pp:.1f} pp")

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

    # ---------------- MODE SELECTOR ----------------
    st.subheader("📊 View Mode")
    
    mode = st.radio(
        "Choose how to view data:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"],
        horizontal=True
    )
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
        # Fixed default selection for speed; only compute for selected countries
        preferred_defaults = ["Austria", "Bulgaria", "Poland"]
        top_countries = [c for c in preferred_defaults if c in all_countries]
        if not top_countries:
            # Fallback: first three alphabetically
            top_countries = all_countries[:3]

        col1, col2 = st.columns([3, 1])
        
        with col2:
            select_all = st.checkbox("Select All Countries")
            show_overall = st.checkbox("Show Overall Aggregate", help="Aggregate selected countries into a single series and forecast.")

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
            # RMSE ranking removed for performance; compute only for selected countries below

        # Overall aggregate (batch) forecast
        if selected_countries and 'show_overall' in locals() and show_overall:
            st.markdown("### 🌍 Overall (Aggregate of selected)")
            overall_df = df_temporal[df_temporal['country'].isin(selected_countries)] \
                .groupby('period', as_index=False)['job_count'].sum() \
                .sort_values('period')

            if len(overall_df) < config['MIN_PERIODS_REQUIRED']:
                st.info(f"Not enough historical data for Overall (need ≥ {config['MIN_PERIODS_REQUIRED']} {config['PERIOD_NAME']}s).")
            else:
                overall_features = create_lag_features(overall_df, config)
                train_o, val_o, test_o = split_data_time_series(overall_features, config)
                if train_o.empty:
                    st.info("Insufficient data to train Overall model.")
                else:
                    model_o = train_model(train_o, config)
                    rmse_o, mae_o, preds_o = evaluate_model(model_o, test_o, config)
                    last_known_o = pd.concat([train_o, val_o, test_o]).iloc[-1]
                    last_window_o = last_known_o[[f'lag_{l}' for l in config['LAG_FEATURES']]].values
                    future_o = recursive_forecast(model_o, last_window_o, config)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Overall RMSE", f"{rmse_o:.2f}")
                    next_o = int(future_o[0])
                    current_o = int(overall_df.iloc[-1]['job_count'])
                    m2.metric(f"Next {period_name} (Overall)", f"{next_o}", delta=next_o - current_o)
                    trend_o = "📈 Growth" if future_o[-1] > future_o[0] else "📉 Decline"
                    m3.metric(f"{config['FORECAST_HORIZON']}-{period_name} Trend", trend_o)

                    fig_o = visualize_results("Overall", train_o, val_o, test_o, preds_o, future_o, rmse_o, config)
                    st.plotly_chart(fig_o, use_container_width=True)

        # Work-mode trends & forecast (Global)
        with st.expander("🧪 Work-Mode Trends (Global)", expanded=True):
            st.caption("Detection via keyword rules; Forecast via gradient boosting (XGBoost, Chen & Guestrin 2016) with time-aware split. Falls back to RandomForest if XGBoost unavailable.")
            enable_wm = st.checkbox("Enable Work-Mode Forecast", key="wm_batch_toggle", value=True)
            wm_snapshot = st.checkbox("Snapshot Donut with Slider", key="wm_batch_snapshot", help="View a donut for a selected period and see model trend from latest data.")
            if enable_wm:
                wm_series = build_work_mode_series(df_raw, config)
                if wm_series.empty:
                    st.info("No work-mode information found in descriptions.")
                else:
                    forecasts = forecast_work_mode_trends_ml(wm_series, config)
                    if wm_snapshot:
                        plot_work_mode_snapshot(wm_series, forecasts, config)
                    else:
                        # Fixed smoothing window per granularity for consistency
                        default_smooth = 3 if granularity == 'weekly' else 7
                        plot_work_mode_trends(wm_series, forecasts, config, smooth_window=default_smooth, percent=True)

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
        st_autorefresh(interval=5000, key="predictive_insights_refresh")
        
        df_raw = fetch_recent(LOOKBACK_MINUTES)
        
        if df_raw.empty:
            st.warning("⚠️ No live data yet. Start the producer to see real-time updates.")
            st.info("Waiting for streaming data...")
            st.stop()
        
        df = preprocess_data(df_raw)
        df['period'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time) if granularity == 'weekly' else df['created_at'].dt.floor('D')
        temporal_data = df.groupby(['period', 'country'], as_index=False).agg({'metric': 'sum'})

        all_countries = sorted(temporal_data['country'].unique())
        # Align default selection with static mode for faster startup
        preferred_defaults = ["Austria", "Bulgaria", "Poland"]
        default_countries = [c for c in preferred_defaults if c in all_countries]
        if not default_countries:
            # Fallback: first three alphabetically
            default_countries = all_countries[:3]

        col1, col2 = st.columns([3, 1])

        with col2:
            select_all = st.checkbox("Select All Countries")
            show_overall_stream = st.checkbox("Show Overall Aggregate", help="Aggregate selected countries into a single series and forecast.")
            enable_live_forecast = st.checkbox("Enable Live Forecasting", help="Train a quick Random Forest on the recent window and predict the next period.", value=True)

        with col1:
            if select_all:
                selected_countries = all_countries
                st.info(f"✅ Displaying all {len(all_countries)} countries.")
            else:
                selected_countries = st.multiselect(
                    "Select Countries to Compare:",
                    options=all_countries,
                    default=default_countries
                )

        if not selected_countries:
            st.warning("⚠️ Please select at least one country to view the plot.")
        else:
            filtered_data = temporal_data[temporal_data['country'].isin(selected_countries)]
            st.success(f"🔴 LIVE: {len(df_raw)} jobs")
            st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
            visualize_weekly_data(filtered_data, granularity)
            
            st.divider()

            df_stream = df_raw.copy()
            df_stream['created_at'] = pd.to_datetime(df_stream['created_at'], errors='coerce', dayfirst=True, infer_datetime_format=True)
            fig = px.histogram(df_stream, x="created_at", color="country", nbins=LOOKBACK_MINUTES, title=f'Job postings', #in the last {LOOKBACK_MINUTES} minutes',
                              labels={"month of job posting"})#, "count": "Number of Job Postings", "country": "Country"})
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            if enable_live_forecast:
                st.subheader("⚡ Live Forecasts (Experimental)")
                live_config = DEFAULT_CONFIG[granularity]
                period_name = live_config['PERIOD_NAME'].capitalize()

                # Work-mode trends & forecast (Global, placed here to match static mode positioning)
                with st.expander("🧪 Work-Mode Trends (Global)", expanded=True):
                    st.caption("Detection via keyword rules; Forecast via gradient boosting (XGBoost, Chen & Guestrin 2016) with time-aware split. Falls back to RandomForest if XGBoost unavailable.")
                    enable_wm_s = st.checkbox("Enable Work-Mode Forecast (Live)", key="wm_stream_toggle", value=True)
                    wm_snapshot_s = st.checkbox("Snapshot Donut with Slider", key="wm_stream_snapshot")
                    if enable_wm_s:
                        wm_series_s = build_work_mode_series(df_raw, live_config)
                        if wm_series_s.empty:
                            st.info("No work-mode information found in live descriptions.")
                        else:
                            forecasts_s = forecast_work_mode_trends_ml(wm_series_s, live_config)
                            if wm_snapshot_s:
                                plot_work_mode_snapshot(wm_series_s, forecasts_s, live_config)
                            else:
                                default_smooth_s = 3 if granularity == 'weekly' else 7
                                plot_work_mode_trends(wm_series_s, forecasts_s, live_config, smooth_window=default_smooth_s, percent=True)

                # Overall aggregate (streaming) forecast
                if selected_countries and show_overall_stream:
                    st.markdown("#### 🌍 Overall (Aggregate of selected)")
                    overall_s = temporal_data[temporal_data['country'].isin(selected_countries)] \
                        .groupby('period', as_index=False)['metric'].sum() \
                        .rename(columns={'metric': 'job_count'}) \
                        .sort_values('period')

                    # Use lighter requirements in streaming context
                    min_needed = max(6, len(live_config['LAG_FEATURES']) + live_config['TEST_SIZE'] + 1)
                    if len(overall_s) < min_needed:
                        st.info(f"Not enough recent data for Overall (need ≥ {min_needed} {live_config['PERIOD_NAME']}s).")
                    else:
                        try:
                            features_o = create_lag_features(overall_s, live_config)
                            window_rows = max(24, min(64, len(features_o)))
                            features_o = features_o.tail(window_rows)

                            unique_p = sorted(features_o['period'].unique())
                            test_start = unique_p[-live_config['TEST_SIZE']]
                            train_o = features_o[features_o['period'] < test_start]
                            test_o = features_o[features_o['period'] >= test_start]

                            if train_o.empty or test_o.empty:
                                st.info("Skipping Overall: insufficient split for live training.")
                            else:
                                model_o = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42)
                                X_to = train_o[[f'lag_{l}' for l in live_config['LAG_FEATURES']]]
                                y_to = train_o['job_count']
                                model_o.fit(X_to, y_to)

                                rmse_o, mae_o, preds_o = evaluate_model(model_o, test_o, live_config)
                                last_known_o = features_o.iloc[-1]
                                last_window_o = last_known_o[[f'lag_{l}' for l in live_config['LAG_FEATURES']]].values
                                future_o = recursive_forecast(model_o, last_window_o, live_config)

                                next_o = int(max(0, future_o[0]))
                                current_o = int(overall_s.iloc[-1]['job_count'])
                                delta_o = next_o - current_o
                                trend_o = "📈 Growth" if future_o[-1] > future_o[0] else "📉 Decline"

                                c1, c2, c3 = st.columns(3)
                                c1.metric("Overall RMSE", f"{rmse_o:.2f}")
                                c2.metric(f"Next {period_name} (Overall)", f"{next_o}", delta=delta_o)
                                c3.metric(f"{live_config['FORECAST_HORIZON']}-{period_name} Trend", trend_o)

                                fig_o = visualize_results(
                                    "Overall",
                                    train_o,
                                    pd.DataFrame(),
                                    test_o,
                                    preds_o,
                                    future_o,
                                    rmse_o,
                                    live_config
                                )
                                st.plotly_chart(fig_o, use_container_width=True)
                        except Exception:
                            st.info("Live Overall forecast skipped due to an error.")

                # Prepare per-country live models on the recent window
                for country in selected_countries:
                    country_df = temporal_data[temporal_data['country'] == country].sort_values('period')

                    # Require minimum periods; be lighter than batch requirement
                    min_needed = max(6, len(live_config['LAG_FEATURES']) + live_config['TEST_SIZE'] + 1)
                    if len(country_df) < min_needed:
                        st.info(f"Not enough recent data for {country} (need ≥ {min_needed} {live_config['PERIOD_NAME']}s).")
                        continue

                    # Map job_count-like column name to align with feature functions
                    country_df = country_df.rename(columns={'metric': 'job_count'})

                    try:
                        features_df = create_lag_features(country_df, live_config)
                        # Use the latest N rows to keep it fast
                        window_rows = max(24, min(64, len(features_df)))
                        features_df = features_df.tail(window_rows)

                        # Quick train-test split: last TEST_SIZE periods as pseudo-test
                        unique_periods = sorted(features_df['period'].unique())
                        test_start = unique_periods[-live_config['TEST_SIZE']]
                        train_live = features_df[features_df['period'] < test_start]
                        test_live = features_df[features_df['period'] >= test_start]

                        if train_live.empty or test_live.empty:
                            st.info(f"Skipping {country}: insufficient split for live training.")
                            continue

                        model_live = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42)
                        X_train = train_live[[f'lag_{l}' for l in live_config['LAG_FEATURES']]]
                        y_train = train_live['job_count']
                        model_live.fit(X_train, y_train)

                        # Evaluate quickly on the pseudo-test
                        rmse_live, mae_live, preds_live = evaluate_model(model_live, test_live, live_config)

                        # Forecast next periods from the latest window
                        last_known = features_df.iloc[-1]
                        last_window = last_known[[f'lag_{l}' for l in live_config['LAG_FEATURES']]].values
                        future_preds_live = recursive_forecast(model_live, last_window, live_config)

                        next_pred = int(max(0, future_preds_live[0]))
                        current_val = int(country_df.iloc[-1]['job_count'])
                        delta = next_pred - current_val
                        trend = "📈 Growth" if future_preds_live[-1] > future_preds_live[0] else "📉 Decline"

                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"{country} RMSE", f"{rmse_live:.2f}")
                        c2.metric(f"Next {period_name}", f"{next_pred}", delta=delta)
                        c3.metric(f"{live_config['FORECAST_HORIZON']}-{period_name} Trend", trend)

                        # Small overlay chart for the country
                        fig_live = visualize_results(
                            country,
                            train_live,
                            pd.DataFrame(),
                            test_live,
                            preds_live,
                            future_preds_live,
                            rmse_live,
                            live_config
                        )
                        st.plotly_chart(fig_live, use_container_width=True)
                    except Exception:
                        st.info(f"Live forecast skipped for {country} due to an error.")
        
    st.divider()
    add_footer("Tibor Buti")