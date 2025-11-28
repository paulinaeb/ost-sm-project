"""
Kafka Consumer - Processes LinkedIn job postings from Kafka and stores in Cassandra - (creates table if it doesn't exist)
"""
import json
import sys
from kafka import KafkaConsumer
import socket
from kafka.errors import KafkaError
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from datetime import datetime
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
# Adaptive bootstrap resolution
_env_bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
def _detect_bootstrap():
    if _env_bootstrap:
        return _env_bootstrap
    # Prefer localhost:29092 (host Docker mapped port) if reachable
    host_candidates = ['localhost:29092', '127.0.0.1:29092']
    for cand in host_candidates:
        host, port = cand.split(':')
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                return cand
        except Exception:
            continue
    # Fallback to container network name/port
    return 'kafka:9092'

KAFKA_BOOTSTRAP_SERVERS = _detect_bootstrap()
KAFKA_TOPIC = 'linkedin-jobs'
KAFKA_GROUP_ID = 'linkedin-jobs-consumer-group'
CASSANDRA_CONTACT_POINTS = os.getenv('CASSANDRA_HOSTS', '127.0.0.1').split(',')
CASSANDRA_KEYSPACE = 'linkedin_jobs'  # Separate database for streaming data
CASSANDRA_TABLE = 'jobs'


def connect_cassandra():
    """Connect to Cassandra and ensure keyspace exists"""
    try:
        cluster = Cluster(CASSANDRA_CONTACT_POINTS)
        session = cluster.connect()
        
        # Ensure keyspace exists
        session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
            WITH replication = {{'class':'SimpleStrategy','replication_factor':1}}
        """)
        session.set_keyspace(CASSANDRA_KEYSPACE)
        
        print(f"✓ Connected to Cassandra keyspace: {CASSANDRA_KEYSPACE}")
        return session
    except Exception as e:
        print(f"✗ Failed to connect to Cassandra: {e}")
        sys.exit(1)


def create_table_if_not_exists(session):
    """Create linkedin_jobs table if it doesn't exist"""
    create_table_cql = f"""
    CREATE TABLE IF NOT EXISTS {CASSANDRA_TABLE} (
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
    )
    """
    try:
        session.execute(create_table_cql)
        print(f"✓ Table {CASSANDRA_TABLE} ready")
    except Exception as e:
        print(f"✗ Error creating table: {e}")
        sys.exit(1)


def create_consumer():
    """Create and configure Kafka consumer"""
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',  # Start from beginning if no offset
            enable_auto_commit=False  # Manual commit for reliability
            # No consumer_timeout_ms - runs continuously for real-time streaming
        )
        print(f"✓ Connected to Kafka ({KAFKA_BOOTSTRAP_SERVERS}), subscribed to topic: {KAFKA_TOPIC}")
        return consumer
    except KafkaError as e:
        print(f"✗ Failed to connect to Kafka: {e}")
        sys.exit(1)


def insert_job_to_cassandra(session, job_data):
    """Insert a single job into Cassandra"""
    insert_cql = f"""
    INSERT INTO {CASSANDRA_TABLE} 
    (id, title, primary_description, detail_url, location, skill, 
     company_name, created_at, country, ingested_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        prepared = session.prepare(insert_cql)
        session.execute(prepared, (
            job_data['id'],
            job_data['title'],
            job_data['primary_description'],
            job_data['detail_url'],
            job_data['location'],
            job_data['skill'],
            job_data['company_name'],
            job_data['created_at'],
            job_data['country'],
            datetime.utcnow()
        ))
        return True
    except Exception as e:
        print(f"✗ Error inserting job {job_data.get('id', 'unknown')}: {e}")
        return False


def consume_and_store(consumer, session):
    """Consume messages from Kafka and store in Cassandra (runs continuously)"""
    print(f"✓ Consuming messages from topic: {KAFKA_TOPIC} (real-time mode)")
    print(f"✓ Storing to: {CASSANDRA_KEYSPACE}.{CASSANDRA_TABLE}")
    print("-" * 60)
    
    processed_count = 0
    error_count = 0
    
    try:
        for message in consumer:
            job_data = message.value
            
            # Insert to Cassandra immediately
            success = insert_job_to_cassandra(session, job_data)
            
            if success:
                processed_count += 1
                if processed_count % 10 == 0:  # Log every 10 jobs
                    print(f"[{processed_count:4d}] Stored: {job_data['title'][:50]:<50} | "
                          f"Offset: {message.offset}")
                
                # Commit offset after successful insert
                consumer.commit()
            else:
                error_count += 1
                # Skip and log error
                print(f"⚠ Skipped job {job_data.get('id', 'unknown')} due to error")
    
    except KeyboardInterrupt:
        print("\n\n✗ Consumer stopped by user")
        print("-" * 60)
        print(f"✓ Processed: {processed_count} jobs")
        print(f"✗ Errors: {error_count} jobs")


def main():
    print("=" * 60)
    print("LinkedIn Jobs Kafka Consumer → Cassandra")
    print("=" * 60)
    
    # Connect to Cassandra and ensure table exists
    session = connect_cassandra()
    create_table_if_not_exists(session)
    
    # Create Kafka consumer
    consumer = create_consumer()
    
    try:
        consume_and_store(consumer, session)
    finally:
        consumer.close()
        session.cluster.shutdown()
        print("✓ Consumer and Cassandra connection closed")


if __name__ == "__main__":
    main()
