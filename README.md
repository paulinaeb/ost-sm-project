# 🇪🇺 Europe CyberScope — CSOMA (CyberSecurity Job Market Analyzer)

**CSOMA (CyberSecurity Job Market Analyzer)** is a data pipeline and analytics platform designed to **collect, process, and visualize cybersecurity job advertisements across Europe in (simulated) real time**.  
The system leverages open-source technologies to identify **regional trends**, **in-demand skills**, and **evolving market dynamics** in the cybersecurity sector.

---

## 📁 Dataset Sources

| Type | Location | Description |
|------|-----------|--------------|
| Dynamic (simulated) | [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1Ult_m13_--7MYIEA8JGtRRzqX8hyaz3W) | Periodically updated simulated job ads |
| Static | [GitHub Dataset – ENISA ECSF](https://github.com/opliyal3/ENISA-ECSF-Dataset/tree/main) | Reference dataset for ECSF-aligned skills |

---

## ⚙️ Quick Start Guide

### 0. Clone the Repository
```bash
git clone <https://github.com/paulinaeb/ost-sm-project.git>
# Europe CyberScope - CSOMA (CyberSecurity Job Market Analyzer)
The Cybersecurity Job Market Analyzer (CSOMA) aims to acquire, store, and analyze cybersecurity job advertisements across Europe in real time. The system uses open-source technologies to process and visualize job data, helping identify regional trends and emerging skill demands.

# Streaming Architecture

```
CSV File → Kafka Producer → Kafka Topic → Kafka Consumer → Cassandra
              ↓                 ↓                            ↓
        (simulates      (KRaft mode -               (linkedin_jobs_db)
         real-time)     no ZooKeeper)               (separate database)
         
         PARALLEL EXECUTION: Producer & Consumer run simultaneously
```

# How to run: get cassandra config and data, then start streaming!

0. Clone this repo 
git clone https://github.com/paulinaeb/ost-sm-project/
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
.\venv\Scripts\Activate.ps1
pip install requirements.txt

3. Load data with python
python preprocessing/ECSF/load_ecsf.py