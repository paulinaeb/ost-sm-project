"""
Data fetching utilities for Streamlit pages
Shared functions to fetch data from Cassandra
"""
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
from cassandra_client import get_session
import numpy as np

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
            primary_description,
            created_at, ingested_at
    FROM {TABLE}
    WHERE ingested_at >= %s AND ingested_at < %s ALLOW FILTERING;
    """

    rows = session.execute(query, (start, now))
    df = pd.DataFrame(list(rows))

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ingested_at"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Normalize created_at too, if provided in varying formats
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", dayfirst=True, infer_datetime_format=True)

    for c in ["company_name", "title", "location", "country", "skill", "primary_description"]:
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
            primary_description,
            created_at, ingested_at
    FROM {TABLE};
    """
    rows = session.execute(query)
    df = pd.DataFrame(list(rows))

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ingested_at"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", dayfirst=True, infer_datetime_format=True)

    for c in ["company_name", "title", "location", "country", "skill", "primary_description"]:
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

# ---------------- Temporal preprocessing & Work-mode utilities ----------------

@st.cache_data(show_spinner=False)
def preprocess_temporal_data(df: pd.DataFrame, config: dict):
    """
    Aggregates raw job data into specified time periods per country.
    Cached by Streamlit to speed up re-renders.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', dayfirst=True, infer_datetime_format=True)

    # Normalize to period based on granularity
    if config['FREQ'] == 'W-MON':
        df['period'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
    else:  # daily
        df['period'] = df['created_at'].dt.floor('D')

    temporal_counts = df.groupby(['country', 'period']).size().reset_index(name='job_count')

    if temporal_counts.empty:
        return pd.DataFrame()

    max_date = temporal_counts['period'].max()
    full_df = []

    for country in temporal_counts['country'].unique():
        country_data = temporal_counts[temporal_counts['country'] == country].sort_values('period')

        active_periods = country_data[country_data['job_count'] > config['DATA_START_THRESHOLD']]['period']

        if not active_periods.empty:
            start_date = active_periods.min()
            country_data = country_data[country_data['period'] >= start_date]
        else:
            start_date = country_data['period'].min()

        country_data = country_data.set_index('period')
        relevant_periods = pd.date_range(start=start_date, end=max_date, freq=config['FREQ'])

        country_data = country_data.reindex(relevant_periods, fill_value=0).reset_index().rename(columns={'index': 'period'})
        country_data['country'] = country
        full_df.append(country_data)

    return pd.concat(full_df, ignore_index=True)

WORK_MODES = ["Remote", "Hybrid", "On-site"]

def classify_work_mode(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "Unknown"
    t = str(text)
    t = t.replace('\u2010', '-').replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
    t = t.lower()
    hybrid_kw = ["hybrid", "mix of remote", "split remote", "days in office"]
    remote_kw = ["remote", "work from home", "wfh", "telecommute", "home-based", "home based"]
    onsite_kw = ["on-site", "onsite", "on site", "office-based", "in office"]

    if any(k in t for k in hybrid_kw):
        return "Hybrid"
    if any(k in t for k in remote_kw):
        return "Remote"
    if any(k in t for k in onsite_kw):
        return "On-site"
    return "Unknown"

@st.cache_data(show_spinner=False)
def build_work_mode_series(df_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Classify postings' work-mode and aggregate counts per period."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["period", "work_mode", "count"])
    df = df_raw.copy()
    if 'created_at' not in df.columns:
        return pd.DataFrame(columns=["period", "work_mode", "count"])
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', dayfirst=True, infer_datetime_format=True)
    if config['FREQ'] == 'W-MON':
        df['period'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time)
    else:
        df['period'] = df['created_at'].dt.floor('D')
    cols = []
    if 'primary_description' in df.columns:
        cols.append('primary_description')
    for c in ['location', 'title', 'company_name']:
        if c in df.columns:
            cols.append(c)
    if not cols:
        return pd.DataFrame(columns=["period", "work_mode", "count"])
    df['__wm_text'] = df[cols].fillna("").astype(str).agg(' '.join, axis=1)
    df['work_mode'] = df['__wm_text'].apply(classify_work_mode)
    df = df[df['work_mode'].isin(WORK_MODES)]
    if df.empty:
        return pd.DataFrame(columns=["period", "work_mode", "count"])
    out = df.groupby(['period', 'work_mode']).size().reset_index(name='count').sort_values('period')
    return out

