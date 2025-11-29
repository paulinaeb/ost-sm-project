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
### Automated Workflow (Recommended)

Use these one-liners to get up and streaming fast:

```bash
# Linux/macOS/WSL
bash deploy.sh
```

```powershell
# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\deploy.ps1
```

One-line alternative (does the same in a fresh session):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\deploy.ps1
```
Notes:
- `Set-ExecutionPolicy -Scope Process` only applies to the current PowerShell window and is temporary.
- You only need to run it once per session before calling `.\deploy.ps1` (subsequent runs can omit it).
- If your policy already allows script execution, skip it entirely and just run `.\deploy.ps1`.

Then, to reset and restart streaming:

```bash
# Linux/macOS/WSL
bash restart_simulation.sh
```

```powershell
# Windows PowerShell
.\restart_simulation.ps1
```
If this is a brand new PowerShell session and you have not set the execution policy yet, you can chain it:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\restart_simulation.ps1
```
Otherwise just use `.\restart_simulation.ps1` directly.
### Alternatively

### 1. Start Docker Services
```bash
docker-compose up -d
```

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
### Or simply use the automated deployment scripts:
```bash
# Bash (Linux/macOS/WSL)
bash deploy.sh

# PowerShell (Windows)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\deploy.ps1
```
Wait for **All dependencies** to be running and healthy. The deploy scripts will:
- Build the app image
- Start Kafka, Cassandra, UI services, and Streamlit
- Initialize ECSF keyspace/tables and load data if needed
- Ensure linkedin_jobs keyspace/table exist and start streaming pipeline

To restart streaming cleanly using scripts:
```powershell
.\restart_simulation.ps1   # PowerShell
```
or
```bash
bash restart_simulation.sh    # Bash
```


### 6. Running in Two Modes (Docker vs Local)

The producer and consumer now auto-adapt to the environment:

- They first honor an explicit environment variable `KAFKA_BOOTSTRAP_SERVERS` if set.
- Otherwise they attempt to connect to `localhost:29092` (the Docker-mapped broker port on the host).
- If that is unreachable, they fall back to `kafka:9092` (the internal Docker Compose service name).

This means you can:

| Mode | Command | Bootstrap selected |
|------|---------|--------------------|
| Local (host) | `python streaming\kafka_consumer.py` | `localhost:29092` (if reachable) |
| Docker exec | `docker exec -it python-app python streaming/kafka_consumer.py` | `kafka:9092` |
| Forced override | `$env:KAFKA_BOOTSTRAP_SERVERS='kafka:9092'` then run scripts | explicit value |

To force a mode explicitly (Windows PowerShell):
```powershell
$env:KAFKA_BOOTSTRAP_SERVERS = 'localhost:29092'  # or 'kafka:9092'
python streaming\kafka_producer.py
```

If you see timeouts from producer/consumer:
1. Verify broker port: `Test-NetConnection -ComputerName localhost -Port 29092`
2. Check container health: `docker ps` and `docker logs kafka --tail 50`
3. Confirm topic exists in Kafka UI (http://localhost:8080) or create it.

### 7. Environment Setup Tips (Windows)

- Prefer Python 3.11 for local runs to match the Docker image.
- Always activate your venv before running producer/consumer:
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```
- If you see `ModuleNotFoundError: cassandra` or Kafka import errors, ensure you're using the venv interpreter and reinstall with:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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