# Real-Time Analytics Dashboard

A comprehensive Streamlit-based analytics dashboard for real-time visualization and analysis of European cybersecurity job postings. This module provides five specialized analytical views for monitoring job market trends, predictions, and skill matching.

## 🏗️ Architecture

```
CSV File → Kafka Producer → Kafka Topic → Kafka Consumer → Cassandra  → Streamlit Live Dashboard.
              ↓                                              ↓
        (simulates                                  (linkedin_jobs)
         real-time)                                     (ecsf)

```

The forecasting module connects to a Cassandra database containing LinkedIn job postings and ECSF (European Cybersecurity Skills Framework) data to provide real-time analytics and insights.

## 📂 Directory Structure

```
forecasting/
├── streamlit_app.py         # Main application entry point with navigation
├── cassandra_client.py      # Cassandra database connection management
├── data_utils.py            # Shared data fetching utilities
├── footer_utils.py          # Reusable footer component
├── forecasting.py           # Naive forecasting algorithms
├── pages/
│   ├── phase1.py           # 🌍 Country Radar - Geographic job distribution
│   ├── phase2.py           # 📈 Predictive Insights 
│   ├── phase4.py           # 🔍 Matching Tracker - ECSF role matching
│   └── phase5.py           # 📡 Change Detector 
└── README.md
```

## 🚀 Features

### Main Dashboard 
Located in `streamlit_app.py` - Real-time job stream monitoring:
- **Live Metrics**: Total jobs, jobs per minute, cumulative counts
- **Latest Arrivals**: Most recent job postings with details
- **Top Companies**: Companies with most job postings
- **Auto-refresh**: Updates every N seconds
- **Filters**: Country, skill, and time window filtering
Author: Ahad

### 🌍 Country Radar 
European geographic job distribution analysis:
- **Interactive Choropleth Map**: Visualizes job distribution across European countries
- **Country Rankings**: Top countries by job count with bar charts
- **Job Distribution**: Most common job titles per country
- **Real-time Updates**: Auto-refresh with configurable lookback period
- **Country Focus**: Filters for European countries with ISO-3 code mapping
Author: Paulina

### 📈 Predictive Insights 
- Time series forecasting and trend analysis
Author: Tibor

### 🔍 Matching Tracker
ECSF role matching and similarity analysis:
- **Title Matching**: Uses fuzzy matching (RapidFuzz) to match LinkedIn job titles to ECSF roles
- **Top ECSF Roles**: Visualizes most frequently matched cybersecurity roles
- **Similarity Scores**: Token-set ratio-based similarity computation
- **Dual Database**: Connects to both `linkedin_jobs` and `ecsf` keyspaces
Author: Sameha

### 📡 Change Detector 
- Job market change detection and anomaly analysis
Author: Ahad

## 🛠️ Core Components

### 1. `cassandra_client.py`
Database connection management:
- **Environment-driven configuration**: Reads from env variables (`CASSANDRA_HOSTS`, `CASSANDRA_PORT`, `CASSANDRA_KEYSPACE`)
- **Session management**: `get_session()` returns a configured Cassandra session
- **Keyspace validation**: `validate_keyspace()` and `validate_ecsf_keyspace()` check database availability
- **Dict factory**: Returns query results as Python dictionaries for easy DataFrame conversion

### 2. `data_utils.py`
Shared data fetching utilities with caching:
- **`fetch_recent(minutes=60)`**: Retrieves jobs from the last N minutes (cached for 3s)
- **`fetch_all()`**: Fetches all jobs from database (cached for 30s)
- **`get_total_count()`**: Returns total job count (cached for 2s)
- **`fetch_all_roles_by_title()`**: Fetches ECSF role titles (cached for 20s)
- **`fetch_all_role_with_tks()`**: Fetches ECSF roles with tasks/knowledge/skills

### 3. `streamlit_app.py`
Main application with horizontal navigation:
- **Fixed Navigation Bar**: Gradient-styled buttons for phase switching
- **Dynamic Page Loading**: Uses `importlib` to load phase modules
- **Query Parameters**: Phase navigation via URL parameters
- **Page Configuration**: Single `st.set_page_config()` for entire app

### 4. `footer_utils.py`
Reusable styled footer component:
- **`add_footer(author_name)`**: Adds author attribution and project branding
- Fixed position styling with dark theme

## ⚙️ Configuration

### Database Requirements
1. **linkedin_jobs keyspace**: Contains `jobs` table with streaming LinkedIn job data
2. **ecsf keyspace**: Contains ECSF framework tables (`roles_by_title`, `role_with_tks`, etc.)

## 🚦 Getting Started

### Prerequisites
- Cassandra running (via Docker or local)
- Job data streaming pipeline active (Kafka producer/consumer)
- ECSF data loaded into Cassandra
(See the main README.md file to see detailed steps)

### Troubleshooting
If you see database connection errors:
1. Ensure Cassandra is running: `docker-compose up -d`
2. Create keyspace by running consumer: `python streaming/kafka_consumer.py`
3. Load ECSF data: `python preprocessing/ECSF/load_ecsf.py`
4. Start producer: `python streaming/kafka_producer.py`


## 🔄 Auto-Refresh
The dashboard uses `streamlit-autorefresh` for live updates:
- Configurable per page
- Interactive filters (country, skill, time window)
- Connects directly to Cassandra to read live data
- Runs inside Docker for easy deployment

## Useful commmands

docker compose restart streamlit (for restarting the session)
docker compose restart cassandra-dev (for restarting the session)

**Deletes the table data containing jobs to run the consumer and producer again to see the updates**
docker exec -it cassandra-dev cqlsh -e "TRUNCATE linkedin_jobs.jobs;"






