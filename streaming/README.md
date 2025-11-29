# LinkedIn Jobs Real-Time Streaming Pipeline

**Author:** Paulina Espejo

This directory contains the Kafka-based streaming pipeline for real-time ingestion of LinkedIn job postings into Cassandra.

## Architecture

```
CSV File → Kafka Producer → Kafka Topic → Kafka Consumer → Cassandra
              ↓                 ↓                            ↓
        (simulates      (KRaft mode -               (linkedin_jobs_db)
         real-time)     no ZooKeeper)               (separate database)
         
         PARALLEL EXECUTION: Producer & Consumer run simultaneously
```
Its technical design follows the streaming-pipeline principles described by Narkhede et al. (2017), using Apache Kafka for real-time ingestion and Apache Cassandra for scalable, reliable data storage.

**Technology Stack:**
- **Kafka in KRaft mode**: Modern ZooKeeper-free Kafka for simplified architecture
- **Cassandra 4.1**: Distributed NoSQL database for job storage
- **Python**: kafka-python and cassandra-driver for stream processing

**Database Structure:**
- `ecsf.*` - ECSF reference data (work roles, TKS, etc.)
- `linkedin_jobs.jobs` - Streaming job postings (separate keyspace)

## Academic Context

This streaming architecture implements a distributed data pipeline using Apache Kafka for message queuing and Apache Cassandra for persistent storage, following established patterns for real-time data ingestion and processing. The architecture is based on principles described by Narkhede et al. (2017), who detail how Kafka-based streaming platforms enable scalable, fault-tolerant data pipelines that can handle high-throughput event streams while maintaining data consistency across distributed systems.

**Reference:**  
Narkhede, N., Shapira, G., & Palino, T. (2017). *Kafka: The definitive guide: Real-time data and stream processing at scale*. O'Reilly Media.

## Components

### 1. `kafka_producer.py`
- Reads `cleaned_linkedin_jobs.csv`
- Publishes each job as JSON to Kafka topic `linkedin-jobs`
- Simulates real-time arrival (10 jobs/second by default)
- Configurable speed via `JOBS_PER_SECOND` constant

### 2. `kafka_consumer.py`
- Subscribes to `linkedin-jobs` Kafka topic
- Creates `linkedin_jobs.jobs` table if it doesn't exist
- Runs **continuously** consuming messages in real-time
- Inserts each job into Cassandra immediately upon consumption
- Skips and logs errors for invalid data


**Cassandra Web UI:** http://localhost:8081
- Navigate to `linkedin_jobs` keyspace
- Query `jobs` table

**Kafka UI:** http://localhost:8080
- View `linkedin-jobs` topic
- See messages, partitions, consumer groups

**Command line verification:**
```powershell
# Count jobs in Cassandra
docker exec cassandra-dev cqlsh -e "SELECT COUNT(*) FROM linkedin_jobs_db.jobs;"

# Show sample jobs
docker exec cassandra-dev cqlsh -e "SELECT id, title, company_name FROM linkedin_jobs_db.jobs LIMIT 10;"
```

## Step-by-Step Run 

```powershell
# 1. Start consumer first (in one terminal) - it will wait for messages
python streaming\kafka_consumer.py


# 2. Start producer (in another terminal) - it will publish jobs
python streaming\kafka_producer.py

# Watch as messages are consumed and stored in real-time!
# Press Ctrl+C in consumer terminal when done
```

## Table Schema

```sql
CREATE TABLE ecsf.linkedin_jobs (
    id int PRIMARY KEY,
    title text,
    primary_description text,
    detail_url text,
    location text,
    skill text,
    company_name text,
    created_at text,
    country text,
    ingested_at timestamp
);
```

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Cassandra Web UI | http://localhost:8081 | Browse Cassandra data |
| Kafka UI | http://localhost:8080 | Monitor Kafka topics |
| Cassandra CQL | localhost:9042 | Direct CQL access |
| Kafka Broker | localhost:29092 | Producer/Consumer connection |

### Manual way

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