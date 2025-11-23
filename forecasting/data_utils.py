"""
Data fetching utilities for Streamlit pages
Shared functions to fetch data from Cassandra
"""
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
from cassandra_client import get_session

# Configuration
TABLE = os.getenv("CASSANDRA_TABLE", "jobs")
ecsf_keyspace = 'ecsf'
roles_by_tile_table = "roles_by_title"
role_with_tks_table= 'role_with_tks'

@st.cache_data(ttl=3)
def fetch_recent(minutes=60):
    """
    Fetch recent jobs from Cassandra within the last N minutes
    Args:
        minutes: Number of minutes to look back
    Returns:
        DataFrame with job data
    """
    session = get_session()
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)

    query = f"""
    SELECT id, title, company_name, location, country, skill,
           created_at, ingested_at
    FROM {TABLE}
    WHERE ingested_at >= %s AND ingested_at < %s ALLOW FILTERING;
    """

    rows = session.execute(query, (start, now))
    df = pd.DataFrame(list(rows))

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ingested_at"], errors="coerce")

    for c in ["company_name", "title", "location", "country", "skill"]:
        df[c] = df[c].fillna("").astype(str)

    return df.sort_values("ts")


@st.cache_data(ttl=30)
def fetch_all():
    """
    Fetch all jobs from Cassandra database
    Returns:
        DataFrame with all job data
    """
    session = get_session()
    query = f"""
    SELECT id, title, company_name, location, country, skill,
           created_at, ingested_at
    FROM {TABLE};
    """
    rows = session.execute(query)
    df = pd.DataFrame(list(rows))

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ingested_at"], errors="coerce")

    for c in ["company_name", "title", "location", "country", "skill"]:
        df[c] = df[c].fillna("").astype(str)

    return df.sort_values("ts")


@st.cache_data(ttl=2)
def get_total_count():
    """
    Get total count of jobs in the database
    Returns:
        int: Total number of jobs
    """
    session = get_session()
    result = session.execute(f"SELECT COUNT(*) AS count FROM {TABLE};")
    row = result.one()
    return row["count"] if row else 0


@st.cache_data(ttl=20)
def fetch_all_roles_by_title():
    session = get_session()
    query = f"""
    SELECT title_key
    FROM {ecsf_keyspace}.{roles_by_tile_table};
    """
    rows = session.execute(query)
    df = pd.DataFrame(list(rows))

    if df.empty:
        return df

    return df

@st.cache_data(ttl=20)
def fetch_all_role_with_tks():
    session = get_session()
    query = f"""
    SELECT title, alt_titles, tks
    FROM {ecsf_keyspace}.{role_with_tks_table};
    """
    rows = session.execute(query)
    df = pd.DataFrame(list(rows))

    if df.empty:
        return df

    return df

