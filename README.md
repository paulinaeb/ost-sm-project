# 🇪🇺 Europe CyberScope — CSOMA (CyberSecurity Job Market Analyzer)

**CSOMA (CyberSecurity Job Market Analyzer)** is a data pipeline and analytics platform designed to **collect, process, and visualize cybersecurity job advertisements across Europe in (simulated) real time**.  
The system leverages open-source technologies to identify **regional trends**, **in-demand skills**, and **evolving market dynamics** in the cybersecurity sector.

---

## Stream mining Architecture

```
CSV File → Kafka Producer → Kafka Topic → Kafka Consumer → Cassandra  → Streamlit Live Dashboard.
              ↓                 ↓                            ↓
        (simulates      (KRaft mode -               (linkedin_jobs_db)
         real-time)     no ZooKeeper)               (separate database)
         
         PARALLEL EXECUTION: Producer & Consumer run simultaneously
```

---

## ⚙️ Quick Start Guide

### 0. Clone the Repository
```bash
git clone <https://github.com/paulinaeb/ost-sm-project.git>
cd ost-sm-project
```

### 1. Start Docker Services
```bash
docker-compose up -d
```
Wait for **All dependencies** to be running and healthy.

### 2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

### 3. Initialize Database Schema (for the static dataset - ECSF)
Access the Cassandra container:
```bash
docker exec -it cassandra-dev cqlsh
```
Then, inside `cqlsh`, run:
```sql
SOURCE 'preprocessing/ECSF/keyspace_tables_creation.sql';
```
*Alternatively, copy-paste the SQL script directly into the terminal.*

### 4. Load ECSF Data
```bash
python preprocessing/ECSF/load_ecsf.py
```

### 5. Start stream mining Job Ads!
```powershell
# Start consumer first (in one terminal with venv activated) - it will wait for messages & creates DB structure if needed
python streaming\kafka_consumer.py

# 2. Start producer (in another terminal with venv activated) - it will publish jobs to a kafka topic
python streaming\kafka_producer.py

# Watch as messages are consumed and stored in real-time!
# Press Ctrl+C in consumer terminal when done

# If you wish to start stream mining again, run the following command to truncate the dynamic database
docker exec -it cassandra-dev cqlsh -e "TRUNCATE linkedin_jobs.jobs;"
```

---

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit Dashboard | http://localhost:8501 | See stream mining & visualizations|
| Cassandra Web UI | http://localhost:8081 | Browse Cassandra data |
| Kafka UI | http://localhost:8080 | Monitor Kafka topics |
| Cassandra CQL | localhost:9042 | Direct CQL access |
| Kafka Broker | localhost:29092 | Producer/Consumer connection |

---

## 📊 Dashboard & Visualizations

The **Streamlit Dashboard** provides an interactive interface to explore cybersecurity job market data across Europe. 
Access it at http://localhost:8501 after starting the services.

### Navigation Tabs

The dashboard features horizontal navigation with the following sections:

| Tab | Description |
|-----|-------------|
| **📈 Dashboard** | Real-time streaming overview with live job ingestion monitoring, time-based aggregations, and quick statistics |
| **🌍 Country Radar** | European job market analysis with interactive visualizations (see below) |
| **📈 Predictive Insights** | Market trend forecasting and predictions *(coming soon)* |
| **🔍 Matching Tracker** | Job-skill matching and recommendation system *(coming soon)* |
| **📡 Change Detector** | Real-time anomaly detection and market shifts *(coming soon)* |

### Country Radar Visualizations

The **Country Radar** tab offers comprehensive European cybersecurity job market insights:

1. **🗺️ Interactive Choropleth Map**
   - Displays job distribution across European countries
   - Color-coded by job frequency (darker = more jobs)
   - Hover to see: country name, job count, and most common job title

2. **🎯 Top Jobs by Country**
   - Horizontal bar chart showing top 10 job titles
   - Country-specific metrics: total jobs, companies, job titles, and skills
   - Expandable table with recent job postings

3. **🌍 Top European Countries Ranking**
   - Bar chart of top 10 countries by job volume
   - Percentage distribution breakdown
   - Real-time statistics: total jobs across all countries

All visualizations support dual modes:
- **Database Mode**: Historical data analysis
- **Streaming Mode**: Real-time updates with 3-second refresh (auto-refresh enabled)

---

## 📁 Dataset Sources

| Type | Location | Description |
|------|-----------|--------------|
| Dynamic (simulated) | [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1Ult_m13_--7MYIEA8JGtRRzqX8hyaz3W) | Periodically updated simulated job ads |
| Static | [GitHub Dataset – ENISA ECSF](https://github.com/opliyal3/ENISA-ECSF-Dataset/tree/main) | Reference dataset for ECSF-aligned skills |

---

## 👩‍💻 Maintainers

**Europe CyberScope Team**  
Contributions and issue reports are welcome — please open a GitHub issue or submit a pull request.
