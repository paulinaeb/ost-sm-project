# analytics/cassandra_client.py
import os
from cassandra.cluster import Cluster
from cassandra.query import dict_factory

# All env-driven so Docker is easy
CASSANDRA_HOSTS = os.getenv("CASSANDRA_HOSTS", "cassandra-dev").split(",")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "linkedin_jobs")
CASSANDRA_TABLE = os.getenv("CASSANDRA_TABLE", "jobs")

# New: ecsf keyspace (defaults to 'ecsf' if not provided)
ECSF_KEYSPACE = os.getenv("ECSF_KEYSPACE", "ecsf")

def get_session():
    """
    Return a Cassandra session with dict rows.
    Reuse this in any script/app that needs to read data.
    """
    cluster = Cluster(CASSANDRA_HOSTS, port=CASSANDRA_PORT)
    session = cluster.connect()
    session.set_keyspace(CASSANDRA_KEYSPACE)
    session.row_factory = dict_factory
    return session

def validate_keyspace():
    """
    Check if the configured keyspace exists in Cassandra.
    Returns tuple: (bool: exists, str: error_message or None)
    """
    try:
        cluster = Cluster(CASSANDRA_HOSTS, port=CASSANDRA_PORT)
        session = cluster.connect()
        
        # Query system schema to check if keyspace exists
        query = "SELECT keyspace_name FROM system_schema.keyspaces WHERE keyspace_name = %s"
        result = session.execute(query, (CASSANDRA_KEYSPACE,))
        
        if result.one() is None:
            return False, f"Keyspace '{CASSANDRA_KEYSPACE}' does not exist. Please create it first."
        
        session.shutdown()
        cluster.shutdown()
        return True, None
        
    except Exception as e:
        return False, f"Failed to connect to Cassandra: {str(e)}"

def validate_ecsf_keyspace():
    """
    This function check if the ECSF_KEYSPACE exists in Cassandra.
    it keeps the same structure/behavior as validate_keyspace() function
    """
    try:
        cluster = Cluster(CASSANDRA_HOSTS, port=CASSANDRA_PORT)
        session = cluster.connect()
        
        query = "SELECT keyspace_name FROM system_schema.keyspaces WHERE keyspace_name = %s"
        result = session.execute(query, (ECSF_KEYSPACE,))
        
        if result.one() is None:
            return False, f"Keyspace '{ECSF_KEYSPACE}' does not exist. Please create it first."
        
        session.shutdown()
        cluster.shutdown()
        return True, None

    except Exception as e:
        return False, f"Failed to connect to Cassandra: {str(e)}"