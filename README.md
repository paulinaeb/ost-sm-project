# 🇪🇺 Europe CyberScope — CSOMA (CyberSecurity Job Market Analyzer)

**CSOMA (CyberSecurity Job Market Analyzer)** is a data pipeline and analytics platform designed to **collect, process, and visualize cybersecurity job advertisements across Europe in (simulated) real time**.  
The system leverages open-source technologies to identify **regional trends**, **in-demand skills**, and **evolving market dynamics** in the cybersecurity sector.

---

## Stream mining Architecture

```
CSV File → Kafka Producer → Kafka Topic → Kafka Consumer → Cassandra  → Streamlit Live Dashboard.
              ↓                 ↓                            ↓
        (simulates      (KRaft mode -               (linkedin_jobs)
         real-time)     no ZooKeeper)                   (ecsf)
         
         PARALLEL EXECUTION: Producer & Consumer run simultaneously
```
Its technical design follows the streaming-pipeline principles described by Narkhede et al. (2017), using Apache Kafka for real-time ingestion and Apache Cassandra for scalable, reliable data storage.

---

## ⚙️ Quick Start Guide

### 0. Clone the Repository
```bash
git clone https://github.com/paulinaeb/ost-sm-project.git
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

| Tab | Description | Author |
|-----|-------------|--------|
| **📈 Dashboard** | Real-time streaming overview with live job ingestion monitoring, time-based aggregations, and quick statistics | Ahad |
| **🌍 Country Radar** | European job market analysis with interactive visualizations | Paulina |
| **📈 Predictive Insights** | Market trend forecasting and predictions | Tibor |
| **🔍 Matching Tracker** | Job-skill matching and recommendation system | Sameha |
| **📡 Change Detector** | Real-time anomaly detection and market shifts | Ahad |

The Country Radar builds on insights from Ogryzek & Jaskulski (2025), whose GIS-based choropleth mapping shows how spatial visualisation can clarify regional labour-market patterns.

All visualizations support dual modes:
- **Database Mode**: Historical data analysis
- **Streaming Mode**: Real-time updates with N-second refresh (auto-refresh enabled)

---

## 📁 Dataset Sources

| Type | Location | Description |
|------|-----------|--------------|
| Dynamic (simulated) | [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1Ult_m13_--7MYIEA8JGtRRzqX8hyaz3W) | Periodically updated simulated job ads |
| Static | [GitHub Dataset – ENISA ECSF](https://github.com/opliyal3/ENISA-ECSF-Dataset/tree/main) | Reference dataset for ECSF-aligned skills |

---

## 👥 Team Contributions

| Name | Nationality | Role & Responsibilities |
|------|-------------|------------------------|
| **Nasser Samiha** 🇸🇾 | Syrian | **Data Preprocessing & Matching Tracker**<br/>• Cleaned ECSF and LinkedIn datasets and stored ECSF in Cassandra<br/>• Built fuzzy matching pipeline for job-skill alignment<br/>• Developed ECSF/Jobs matching visualization dashboard |
| **Espejo Paulina** 🇻🇪 | Venezuelan | **Stream Mining & Country Radar**<br/>• Implemented Kafka producer/consumer for real-time job simulation and its storage to Cassandra<br/>• Designed European job market geographic visualization dashboard<br/>• Initiated containerization with Docker compose
| **Ahad Rezaul Khan** 🇧🇩 | Bangladesh | **Real-time Dashboard & Change Detection**<br/>• Created live streaming dashboard with auto-refresh<br/>• Implemented batch analytics and forecasting models<br/>• Built anomaly detection for market shifts |
| **Buti Tibor** 🇭🇺 | Hungarian | **Deployment & Predictive Insights**<br/>• Created comprehensive pipeline for deployment <br/>• Developed time-series forecasting dashboard |

---

## 👩‍💻 Maintainers

**Europe CyberScope Team**  
Contributions and issue reports are welcome — please open a GitHub issue or submit a pull request.


**Reference:**  
Narkhede, N., Shapira, G., & Palino, T. (2017). *Kafka: The definitive guide: Real-time data and stream processing at scale*. O'Reilly Media.

Ogryzek, M., & Jaskulski, M. (2025). Applying methods of exploratory data analysis and methods of modeling the unemployment rate in spatial terms in Poland. Applied Sciences, 15(8), 4136. https://doi.org/10.3390/app15084136 