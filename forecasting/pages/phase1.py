import os
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from cassandra_client import validate_keyspace
from data_utils import fetch_recent, fetch_all
from streamlit_autorefresh import st_autorefresh
from footer_utils import add_footer

def run():
    st.title("🌍 Country Radar - European Cybersecurity Jobs")
    
    # ---------------- SETTINGS ----------------
    TABLE = os.getenv("CASSANDRA_TABLE", "jobs")
    LOOKBACK_MINUTES = 60
    
    # European country codes (ISO-3 for Plotly)
    EUROPEAN_COUNTRIES = [
        'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
        'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
        'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE', 'GBR', 'CHE', 'NOR', 'ISL'
    ]
    
    # Country name mapping (handle variations in the data)
    COUNTRY_NAME_MAP = {
        'Austria': 'AUT', 'Belgium': 'BEL', 'Bulgaria': 'BGR', 'Croatia': 'HRV',
        'Cyprus': 'CYP', 'Czech Republic': 'CZE', 'Czechia': 'CZE', 'Denmark': 'DNK',
        'Estonia': 'EST', 'Finland': 'FIN', 'France': 'FRA', 'Germany': 'DEU',
        'Greece': 'GRC', 'Hungary': 'HUN', 'Ireland': 'IRL', 'Italy': 'ITA',
        'Latvia': 'LVA', 'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Malta': 'MLT',
        'Netherlands': 'NLD', 'Poland': 'POL', 'Portugal': 'PRT', 'Romania': 'ROU',
        'Slovakia': 'SVK', 'Slovenia': 'SVN', 'Spain': 'ESP', 'Sweden': 'SWE',
        'United Kingdom': 'GBR', 'UK': 'GBR', 'Switzerland': 'CHE', 'Norway': 'NOR',
        'Iceland': 'ISL'
    }
    
    # Reverse mapping for display
    ISO_TO_NAME = {v: k for k, v in COUNTRY_NAME_MAP.items() if k not in ['UK', 'Czechia']}
    
    # ---------------- VALIDATE KEYSPACE ----------------
    keyspace_exists, error_msg = validate_keyspace()
    if not keyspace_exists:
        st.error(f"❌ **Database Connection Error**")
        st.warning(error_msg)
        st.info("""
        **To fix this:**
        1. Make sure Cassandra is running: `docker-compose up -d`
        2. Create the database by running the consumer: `python streaming\\kafka_consumer.py`
        3. Start the producer to ingest data: `python streaming\\kafka_producer.py`
        4. Refresh this page.
        """)
        st.stop()
    
    # ---------------- HELPER FUNCTIONS ----------------
    def normalize_country(country):
        """Convert country name to ISO-3 code"""
        if pd.isna(country) or country == '':
            return None
        return COUNTRY_NAME_MAP.get(country, None)
    
    # ---------------- DATA PROCESSING ----------------
    def prepare_data(df):
        """Add ISO codes and filter European countries"""
        if df.empty:
            return df
        
        df['country_iso'] = df['country'].apply(normalize_country)
        # Filter only European countries
        df = df[df['country_iso'].isin(EUROPEAN_COUNTRIES)].copy()
        return df
    
    def aggregate_by_country(df):
        """Count jobs per country"""
        if df.empty:
            return pd.DataFrame(columns=['country_iso', 'job_count', 'country_name'])
        
        country_counts = df.groupby('country_iso').size().reset_index(name='job_count')
        country_counts['country_name'] = country_counts['country_iso'].map(
            lambda x: ISO_TO_NAME.get(x, x)
        )
        return country_counts.sort_values('job_count', ascending=False)
    
    def aggregate_by_week(df):
        """Group data by week for time slider"""
        if df.empty:
            return pd.DataFrame()
        
        df['week'] = df['ts'].dt.to_period('W').dt.start_time
        weekly = df.groupby(['week', 'country_iso']).size().reset_index(name='job_count')
        weekly['country_name'] = weekly['country_iso'].map(lambda x: ISO_TO_NAME.get(x, x))
        return weekly
    
    # ---------------- VISUALIZATIONS ----------------
    def create_choropleth_map(df, title="Job Distribution Across Europe", height=600):
        """Create European choropleth map"""
        if df.empty:
            return None
        
        country_data = aggregate_by_country(df)
        
        if country_data.empty:
            return None
        
        fig = px.choropleth(
            country_data,
            locations='country_iso',
            color='job_count',
            hover_name='country_name',
            hover_data={'country_iso': False, 'job_count': True},
            color_continuous_scale='Viridis',
            title=title,
            labels={'job_count': 'Number of Jobs'}
        )
        
        # Focus on Europe
        fig.update_geos(
            scope='europe',
            showcountries=True,
            countrycolor="lightgray",
            showcoastlines=True,
            coastlinecolor="gray",
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(
                showframe=False,
                showland=True,
                landcolor='rgb(243, 243, 243)',
            )
        )
        
        return fig
    
    def create_top_jobs_chart(df, country_iso, top_n=10):
        """Create horizontal bar chart of top jobs in a country"""
        if df.empty or country_iso is None:
            return None
        
        df_country = df[df['country_iso'] == country_iso]
        
        if df_country.empty:
            return None
        
        top_jobs = df_country['title'].value_counts().head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_jobs.values,
            y=top_jobs.index,
            orientation='h',
            marker=dict(color='rgb(55, 83, 109)')
        ))
        
        country_name = ISO_TO_NAME.get(country_iso, country_iso)
        
        fig.update_layout(
            title=f"Top {top_n} Job Types in {country_name}",
            xaxis_title="Number of Jobs",
            yaxis_title="Job Title",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def create_top_countries_chart(df, job_filter=None, top_n=10):
        """Create bar chart of top countries"""
        if df.empty:
            return None
        
        df_filtered = df.copy()
        
        if job_filter and job_filter != "All":
            df_filtered = df_filtered[df_filtered['title'] == job_filter]
        
        top_countries = df_filtered['country_iso'].value_counts().head(top_n)
        
        # Map to country names
        country_names = [ISO_TO_NAME.get(iso, iso) for iso in top_countries.index]
        
        fig = go.Figure(go.Bar(
            x=top_countries.values,
            y=country_names,
            orientation='h',
            marker=dict(
                color=top_countries.values,
                colorscale='Blues',
                showscale=False
            )
        ))
        
        title = f"Top {top_n} Countries"
        if job_filter and job_filter != "All":
            title += f" for '{job_filter}'"
        
        fig.update_layout(
            title=title,
            xaxis_title="Number of Jobs",
            yaxis_title="Country",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    # ---------------- MODE SELECTOR ----------------
    st.subheader("📊 View Mode")
    
    mode = st.radio(
        "Choose how to view data:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"],
        horizontal=True
    )
    
    # ---------------- FETCH DATA BASED ON MODE ----------------
    if mode == "📁 View Existing Database":
        df_raw = fetch_all()
        
        if df_raw.empty:
            st.warning("⚠️ No data found in the database. Start the producer to ingest data.")
            st.stop()
        
        df = prepare_data(df_raw)
        
        if df.empty:
            st.warning("⚠️ No European country data found in the database.")
            st.stop()
        
        st.success(f"✅ Loaded {len(df)} jobs from {df['country_iso'].nunique()} European countries")
        
    else:  # Real-time Streaming Mode
        st_autorefresh(interval=3000, key="country_radar_refresh")
        
        df_raw = fetch_recent(LOOKBACK_MINUTES)
        
        if df_raw.empty:
            st.warning("⚠️ No live data yet. Start the producer to see real-time updates.")
            st.info("Waiting for streaming data...")
            st.stop()
        
        df = prepare_data(df_raw)
        
        if df.empty:
            st.warning("⚠️ No European country data in recent stream.")
            st.stop()
        
        st.success(f"🔴 LIVE: {len(df)} jobs from {df['country_iso'].nunique()} countries")
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    st.divider()
    
    # ============================================================================
    # SECTION 1: CHOROPLETH MAP WITH TIME SLIDER
    # ============================================================================
    st.header("🗺️ European Job Distribution Map")
    
    # Initialize session state for selected country
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = None
    
    col_map, col_info = st.columns([3, 1])
    
    with col_map:
        # Time slider (only in database mode)
        if mode == "📁 View Existing Database" and not df.empty:
            st.subheader("📅 Time Period Selector")
            
            # Get date range
            min_date = df['ts'].min().date()
            max_date = df['ts'].max().date()
            
            # Calculate weeks
            df['week'] = df['ts'].dt.to_period('W').apply(lambda x: x.start_time.date())
            weeks = sorted(df['week'].unique())
            
            if len(weeks) > 1:
                col_slider, col_animate = st.columns([4, 1])
                
                with col_slider:
                    selected_week_idx = st.slider(
                        "Select Week",
                        min_value=0,
                        max_value=len(weeks) - 1,
                        value=len(weeks) - 1,
                        format="Week %d",
                        help="Slide to view different weeks"
                    )
                
                with col_animate:
                    animate = st.checkbox("🎬 Animate", help="Auto-play through weeks")
                
                selected_week = weeks[selected_week_idx]
                
                # Filter by selected week
                df_map = df[df['week'] == selected_week].copy()
                
                st.caption(f"📆 Showing data for week starting: **{selected_week}**")
                
                # Animation logic
                if animate:
                    if 'week_idx' not in st.session_state:
                        st.session_state.week_idx = 0
                    
                    if st.session_state.week_idx < len(weeks) - 1:
                        st.session_state.week_idx += 1
                        st.rerun()
                    else:
                        st.session_state.week_idx = 0
            else:
                df_map = df
                st.caption(f"📆 Showing all data (single week available)")
        else:
            # Streaming mode - show cumulative
            df_map = df
            st.caption("🔴 **Live Mode:** Showing cumulative data from recent stream")
        
        # Create and display map
        fig_map = create_choropleth_map(df_map, height=500)
        
        if fig_map:
            # Display map
            selected_points = st.plotly_chart(
                fig_map, 
                use_container_width=True,
                key="choropleth_map"
            )
            
            st.caption("💡 **Tip:** Darker colors indicate more job postings. Click a country to see its top jobs below.")
        else:
            st.info("No map data available for the selected period.")
    
    with col_info:
        st.subheader("📊 Quick Stats")
        
        # Metrics
        total_jobs = len(df_map)
        total_countries = df_map['country_iso'].nunique()
        total_companies = df_map['company_name'].nunique()
        
        st.metric("Total Jobs", total_jobs)
        st.metric("Countries", total_countries)
        st.metric("Companies", total_companies)
        
        st.divider()
        
        # Top country
        if not df_map.empty:
            top_country_iso = df_map['country_iso'].value_counts().idxmax()
            top_country_name = ISO_TO_NAME.get(top_country_iso, top_country_iso)
            top_country_count = df_map['country_iso'].value_counts().max()
            
            st.metric(
                "🏆 Top Country",
                top_country_name,
                delta=f"{top_country_count} jobs"
            )
            
            # Most common job
            top_job = df_map['title'].value_counts().idxmax()
            st.metric("🎯 Most Common Job", top_job, delta=f"{df_map['title'].value_counts().max()} postings")
    
    st.divider()
    
    # ============================================================================
    # SECTION 2: TOP 10 JOBS BY COUNTRY (Click-to-Filter)
    # ============================================================================
    st.header("🎯 Top Jobs by Country")
    
    col_selector, col_chart = st.columns([1, 3])
    
    with col_selector:
        st.subheader("Select Country")
        
        # Get list of countries with data
        countries_with_data = df['country_iso'].value_counts()
        country_options = {
            ISO_TO_NAME.get(iso, iso): iso 
            for iso in countries_with_data.index
        }
        
        sorted_countries = sorted(country_options.keys())
        
        selected_country_name = st.selectbox(
            "Choose a country",
            options=sorted_countries,
            help="Select a country to see its top job types"
        )
        
        selected_country_iso = country_options[selected_country_name]
        
        # Update session state
        st.session_state.selected_country = selected_country_iso
        
        st.divider()
        
        # Country-specific metrics
        df_country = df[df['country_iso'] == selected_country_iso]
        
        st.metric("Jobs in Country", len(df_country))
        st.metric("Unique Companies", df_country['company_name'].nunique())
        st.metric("Unique Job Titles", df_country['title'].nunique())
        st.metric("Unique Skills", df_country['skill'].nunique())
    
    with col_chart:
        fig_jobs = create_top_jobs_chart(df, selected_country_iso, top_n=10)
        
        if fig_jobs:
            st.plotly_chart(fig_jobs, use_container_width=True)
        else:
            st.info(f"No job data available for {selected_country_name}")
        
        # Show sample jobs
        if not df_country.empty:
            with st.expander("📋 View Recent Jobs in " + selected_country_name):
                cols_display = ['title', 'company_name', 'location', 'skill', 'ts']
                st.dataframe(
                    df_country[cols_display].sort_values('ts', ascending=False).head(20),
                    use_container_width=True
                )
    
    st.divider()
    
    # ============================================================================
    # SECTION 3: TOP 10 COUNTRIES RANKING
    # ============================================================================
    st.header("🌍 Top European Countries Ranking")
    
    col_filter, col_chart = st.columns([1, 3])
    
    with col_filter:
        st.subheader("Filter Options")
        
        # Job type filter
        job_types = ["All"] + sorted(df['title'].unique().tolist())
        
        selected_job = st.selectbox(
            "Filter by Job Type",
            options=job_types,
            help="Filter countries by specific job type"
        )
        
        st.divider()
        
        # Show percentage breakdown
        st.subheader("📊 Distribution")
        
        df_filtered = df if selected_job == "All" else df[df['title'] == selected_job]
        
        total_filtered = len(df_filtered)
        st.metric("Total Jobs", total_filtered)
        
        top_10_countries = df_filtered['country_iso'].value_counts().head(10)
        
        st.caption("**Top 10 Countries:**")
        for rank, (iso, count) in enumerate(top_10_countries.items(), 1):
            country_name = ISO_TO_NAME.get(iso, iso)
            percentage = (count / total_filtered) * 100
            st.write(f"{rank}. **{country_name}**: {count} ({percentage:.1f}%)")
    
    with col_chart:
        fig_countries = create_top_countries_chart(df, job_filter=selected_job, top_n=10)
        
        if fig_countries:
            st.plotly_chart(fig_countries, use_container_width=True)
        else:
            st.info("No data available for the selected filter.")
        
        # Geographic distribution insight
        if selected_job != "All":
            with st.expander(f"🔍 Insights for '{selected_job}'"):
                df_job = df[df['title'] == selected_job]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Jobs", len(df_job))
                col2.metric("Countries", df_job['country_iso'].nunique())
                col3.metric("Companies Hiring", df_job['company_name'].nunique())
                
                # Top companies hiring for this role
                st.caption("**Top Companies Hiring:**")
                top_companies = df_job['company_name'].value_counts().head(5)
                for company, count in top_companies.items():
                    st.write(f"• {company}: {count} jobs")
    
    # ---------------- FOOTER ----------------
    st.divider()
    add_footer("Paulina Espejo")