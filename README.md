# 🇪🇺 Europe CyberScope — CSOMA (CyberSecurity Job Market Analyzer)

**CSOMA (CyberSecurity Job Market Analyzer)** is a data pipeline and analytics platform designed to **collect, process, and visualize cybersecurity job advertisements across Europe in (simulated) real time**.  
The system leverages open-source technologies to identify **regional trends**, **in-demand skills**, and **evolving market dynamics** in the cybersecurity sector.

---

## Stream mining Architecture

```
CSV File → Kafka Producer → Kafka Topic → Kafka Consumer → Cassandra
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
Wait for **Cassandra** to become healthy (approx. 1–2 minutes):
```bash
docker logs -f cassandra-dev
```

### 2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

### 3. Initialize Database Schema
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

### 5. Start stream mining!
```powershell
# Start consumer first (in one terminal) - it will wait for messages & creates DB structure if needed
python streaming\kafka_consumer.py

# 2. Start producer (in another terminal) - it will publish jobs to a kafka topic
python streaming\kafka_producer.py

# Watch as messages are consumed and stored in real-time!
# Press Ctrl+C in consumer terminal when done
```

---

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Cassandra Web UI | http://localhost:8081 | Browse Cassandra data |
| Kafka UI | http://localhost:8080 | Monitor Kafka topics |
| Cassandra CQL | localhost:9042 | Direct CQL access |
| Kafka Broker | localhost:29092 | Producer/Consumer connection |

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
