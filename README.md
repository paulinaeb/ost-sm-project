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

1. Start containers
docker-compose up -d

-- Wait for Cassandra to be healthy (~1-2 minutes)
docker logs -f cassandra-dev

2. Create venv and install python dependencies for loading data
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install requirements.txt

3. Load static data with python
python preprocessing/ECSF/load_ecsf.py

4. Start consumer (this creates the keyspace and a table for dynamic data in Cassandra)
python streaming/kafka_consumer.py

5. Start producer
python streaming/kafka_producer.py

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Cassandra Web UI | http://localhost:8081 | Browse Cassandra data |
| Kafka UI | http://localhost:8080 | Monitor Kafka topics |
| Cassandra CQL | localhost:9042 | Direct CQL access |
| Kafka Broker | localhost:29092 | Producer/Consumer connection |

## Dataset sources 
datasets/dynamic (simulated)
https://drive.google.com/drive/u/1/folders/1Ult_m13_--7MYIEA8JGtRRzqX8hyaz3W
datasets/static
https://github.com/opliyal3/ENISA-ECSF-Dataset/tree/main