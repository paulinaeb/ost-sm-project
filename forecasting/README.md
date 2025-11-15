
# LinkedIn Jobs Real-Time Streaming Pipeline

Kafka → Consumer → Cassandra → Streamlit Live Dashboard.

# Directory Structure

ost-sm-project/
│
├── forecasting/
│   ├── streamlit_app.py        # Live dashboard
│   ├── cassandra_client.py     # Cassandra DB adapter
│   ├── forecasting.py
│   ├── README.md

# Streamlit Dashboard #

- Real-time job stream visualization
- Auto-updating metrics:
- total jobs
- jobs per minute
- cumulative job count
- latest arrivals
- Fully interactive with filters (country, skill, time window)

# Components #
### 1. `cassandra_client.py`
- Provides a reusable helper to create a Cassandra session
- Reads host/port/keyspace from environment variables
- Uses dict_factory to return rows as Python dictionaries
- Shared by the Streamlit dashboard and other scripts

### 2. `streamlit_app.py`
- Real-time visualization of incoming job stream
- Auto-refresh (1–3 seconds) using streamlit-autorefresh
Shows:
   - Latest job arrivals
   - Jobs per minute (live)
   - Cumulative job count
   - Top companies
   - Interactive filters (country, skill, time window)
- Connects directly to Cassandra to read live data
- Runs inside Docker for easy deployment

### 2. `forecasting.py`
- Implements a simple baseline forecasting model for job trends
- Uses only Pandas (no heavy ML libraries) for fast execution
- Main function: naive_forecast()
  - Takes a DataFrame of daily job counts → columns: ['date', 'jobs']
  - Resamples data to daily frequency, filling missing days with 0
  
- Returns a forecast DataFrame with columns:
  - date – predicted future dates
  - yhat – forecasted number of jobs per day
- Used by the Streamlit dashboard to show upcoming job trends






