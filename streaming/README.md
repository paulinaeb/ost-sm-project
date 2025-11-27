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
