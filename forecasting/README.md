
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
    ├── pages/
        ├── phase1.py
        ├── phase2.py
        ├── phase3.py   <-- Phase 3 is inside main file
        └── phase4.py


## Everyone should work on their part through phase1,2,3,4 files inside pages parellel to streamlit_app.py file ##

## Make sure not to call "st.set_page_config()" inside phase files. It must appear only once in the entire project inside streamlit_app.py. ##

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


## __init__.py marks a folder as a Python package.##



# Start the Entire Pipeline (Cassandra + Kafka + UI)

docker compose up --build

**This starts:**

Streamlit Dashboard http://localhost:8501

Kafka UI    http://localhost:8080

Cassandra Web UI    http://localhost:8081

Wait until all containers are healthy.

# Start Streaming Data

**For Windows**
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

**For MAC**
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

**To see the Updates Live**
pip install streamlit-autorefresh


# Start Kafka Producer and Consumer
python streaming/kafka_producer.py (inside venv)
python streaming/kafka_consumer.py (inside venv)

# Useful commmands

docker compose restart streamlit (for restarting the session)
docker compose restart cassandra-dev (for restarting the session)

**This one deletes the table data containing jobs to run the consumer and producer again to see the updates**
docker exec -it cassandra-dev cqlsh -e "TRUNCATE linkedin_jobs.jobs;"


**To activate the venv again**
source venv/bin/activate







