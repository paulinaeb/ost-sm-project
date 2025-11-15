# analytics/cassandra_client.py
import os
from cassandra.cluster import Cluster
from cassandra.query import dict_factory

# All env-driven so Docker is easy
CASSANDRA_HOSTS = os.getenv("CASSANDRA_HOSTS", "cassandra-dev").split(",")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "linkedin_jobs")
CASSANDRA_TABLE = os.getenv("CASSANDRA_TABLE", "jobs")

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
