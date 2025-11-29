"""
Kafka Producer - Simulates real-time LinkedIn job postings ingestion
Reads cleaned_linkedin_jobs.csv and publishes each job to Kafka topic
"""
import csv
import json
import time
import sys
import os
from kafka import KafkaProducer
import socket
from kafka.errors import KafkaError

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
# Adaptive bootstrap resolution
_env_bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
def _detect_bootstrap():
    if _env_bootstrap:
        return _env_bootstrap
    host_candidates = ['localhost:29092', '127.0.0.1:29092']
    for cand in host_candidates:
        host, port = cand.split(':')
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                return cand
        except Exception:
            continue
    return 'kafka:9092'

KAFKA_BOOTSTRAP_SERVERS = _detect_bootstrap()
KAFKA_TOPIC = 'linkedin-jobs'
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'dynamic_dataset', 'cleaned_linkedin_jobs.csv')

# Simulation speed
JOBS_PER_SECOND = 100  # Fast demo mode (100 jobs/second)
# JOBS_PER_SECOND = 0.5  # Realistic mode (1 job every 2 seconds)

DELAY_BETWEEN_JOBS = 1.0 / JOBS_PER_SECOND


def create_producer():
    """Create and configure Kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',  # Wait for all replicas to acknowledge
            retries=3
        )
        print(f"✓ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except KafkaError as e:
        print(f"✗ Failed to connect to Kafka: {e}")
        sys.exit(1)


def read_and_publish_jobs(producer):
    """Read CSV file and publish each job to Kafka topic"""
    if not os.path.exists(CSV_FILE_PATH):
        print(f"✗ CSV file not found: {CSV_FILE_PATH}")
        sys.exit(1)
    
    print(f"✓ Reading jobs from: {CSV_FILE_PATH}")
    print(f"✓ Publishing to topic: {KAFKA_TOPIC}")
    print(f"✓ Speed: {JOBS_PER_SECOND} jobs/second (delay: {DELAY_BETWEEN_JOBS:.3f}s)")
    print("-" * 60)
    
    job_count = 0
    start_time = time.time()
    
    with open(CSV_FILE_PATH, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Convert CSV row to JSON-friendly format
            job_message = {
                'id': int(row['ID']),
                'title': row['Title'],
                'primary_description': row['Primary Description'],
                'detail_url': row['Detail URL'],
                'location': row['Location'],
                'skill': row['Skill'],
                'company_name': row['Company Name'],
                'created_at': row['Created At'],
                'country': row['Country']
            }
            
            try:
                # Send to Kafka
                future = producer.send(KAFKA_TOPIC, value=job_message)
                # Wait for send to complete (synchronous for reliability)
                record_metadata = future.get(timeout=10)
                
                job_count += 1
                if job_count % 10 == 0:  # Log every 10 jobs
                    elapsed = time.time() - start_time
                    rate = job_count / elapsed if elapsed > 0 else 0
                    print(f"[{job_count:4d}] Published: {job_message['title'][:50]:<50} | "
                          f"Partition: {record_metadata.partition} | Rate: {rate:.1f} jobs/s")
                
                # Simulate real-time arrival
                time.sleep(DELAY_BETWEEN_JOBS)
                
            except KafkaError as e:
                print(f"✗ Error publishing job {job_message['id']}: {e}")
                continue
    
    # Final stats
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"✓ Published {job_count} jobs in {elapsed:.2f}s (avg {job_count/elapsed:.1f} jobs/s)")
    

def main():
    print("=" * 60)
    print("LinkedIn Jobs Kafka Producer - Real-time Simulation")
    print("=" * 60)
    
    producer = create_producer()
    
    try:
        read_and_publish_jobs(producer)
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    finally:
        producer.flush()
        producer.close()
        print("✓ Producer closed")


if __name__ == "__main__":
    main()
