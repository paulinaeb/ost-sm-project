import streamlit as st
from footer_utils import add_footer
from data_utils import fetch_recent, fetch_all,fetch_all_roles_by_title,fetch_all_role_with_tks
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from rapidfuzz import fuzz
import pandas as pd
from cassandra_client import validate_keyspace, validate_ecsf_keyspace
import plotly.graph_objects as go
import plotly.express as px

def run():

    # set the lookup minutes
    LOOKBACK_MINUTES = 60

    # set the Page title
    st.title("🔍 Matching Tracker")
    #st.write("Content for Matching Tracker page coming soon...")

    # check cassandra key existence
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

    # check the existence of ecsf keyspace
    ecsf_keyspace_exists, ecsf_error_msg = validate_ecsf_keyspace()
    if not ecsf_keyspace_exists:
        st.error(f"❌ **Database Connection Error**")
        st.warning(ecsf_error_msg)
        st.info("""
        **To fix this:**
        1. Make sure Cassandra is running: `docker-compose up -d`
        2. From the command line type `docker exec -it cassandra-dev cqlsh`
        3. Create the tables schema by running , inside cqlsh,`sql SOURCE 'preprocessing/ECSF/keyspace_tables_creation.sql';`
        4. Load ECSF data `python preprocessing/ECSF/load_ecsf.py`
        5. Refresh this page.
        """)
        st.stop()

    ## set the page footer
    add_footer("Samiha Nasser")

    # ============================================================================
    # SECTION 1: ECSF titles similarity to incoming linkedin jobs
    # ============================================================================

    ## first stream mining process
    def calculate_titles_similarity(linkedin_jobs_df,ecsf_roles_by_title_df):

        # Select the columns containing titles
        linkedin_titles = linkedin_jobs_df['title'].astype(str).str.strip()
        ecsf_titles = ecsf_roles_by_title_df['title_key'].astype(str).str.strip()

        # List to store results
        results = []

        for ln_title in linkedin_titles:
            # For each LinkedIn title → compute similarity against all official titles
            similarities = [
                fuzz.token_set_ratio(ln_title, ecsf_title)
                for ecsf_title in ecsf_titles
            ]
            
            # Get the maximum similarity score and its matching ECSF title
            max_sim = max(similarities)
            best_match = ecsf_titles.iloc[similarities.index(max_sim)]
            
            results.append({
                'LinkedIn_Title': ln_title,
                'Best_ECSF_Title': best_match,
                'Similarity': max_sim
            })

        similarity_df = pd.DataFrame(results)
        print(similarity_df)
        return similarity_df
    
    def plot_top_ecsf_roles(similarity_df, top_n=10):
        """
        Visualizes the most frequently matched ECSF roles.
        - Groups by Best_ECSF_Title
        - Counts how many LinkedIn titles map to each ECSF role
        - Sorts by count
        - Plots Top N
        """

        if similarity_df is None or similarity_df.empty:
            st.info("No similarity results to show.")
            return None

        # Count LinkedIn titles associated with each ECSF role
        role_counts = (
            similarity_df
            .groupby("Best_ECSF_Title")["LinkedIn_Title"]
            .nunique()
            .sort_values(ascending=False)
        )

        # Select top_n roles
        top_roles = role_counts.head(top_n)

        if top_roles.empty:
            st.info("No roles to display.")
            return None

        # Add title
        st.subheader(f"Top {len(top_roles)} ECSF Roles by Number of Matching LinkedIn Titles")

        # Build Plotly figure
        fig = go.Figure(go.Bar(
            x=top_roles.values,
            y=top_roles.index,
            orientation='h',
            marker=dict(
                color=top_roles.values,
                colorscale="Greens",
                line=dict(color="#FFFFFF", width=0.8),
            ),
            hovertemplate="<b>%{y}</b><br>Matched Titles: <b>%{x}</b><extra></extra>",
            text=top_roles.values,
            textposition="outside",
            textfont=dict(color="black", size=12),
        ))

        fig.update_layout(
            height=max(300, 60 * len(top_roles)),
            margin=dict(l=40, r=100, t=40, b=40),
            template="plotly_white", 
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                title="<b>Number of Matching LinkedIn Titles</b>",
                titlefont=dict(size=16, color="black"),
                tickfont=dict(size=14, color="black"),
                gridcolor="lightgray",
                zeroline=False,
            ),
            yaxis=dict(
                title="<b>ECSF Role</b>",
                titlefont=dict(size=16, color="black"),
                tickfont=dict(size=14, color="black"),
                automargin=True,
            ),
            font=dict(family="Arial, sans-serif", color="black"),
        )

        # Reverse Y-axis so highest is on top
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        return fig

    # ============================================================================
    # SECTION 2: Skill gap between the official titles and the incoming linkedin jobs
    # ============================================================================

    def plot_skill_gap(linkedin_jobs_df, ecsf_role_with_tks):
        jobs = linkedin_jobs_df.copy()
        jobs['skills_clean'] = jobs['skill'].fillna("").str.lower().str.split(r'[,;•·]')
        jobs['skills_clean'] = jobs['skills_clean'].apply(lambda x: [s.strip() for s in x if s.strip()])

        # Extract official competences (T(task) + K(knowledge) + S(skill))
        def get_competences(tks):
            return [desc.lower().strip() for code, _, desc in tks]

        ecsf = ecsf_role_with_tks.copy()
        ecsf['competences'] = ecsf['tks'].apply(get_competences)

        # Simple mapping
        jobs['role'] = jobs['title'].apply(lambda t: next((r['title'] for _, r in ecsf.iterrows() if r['title'].lower() in t.lower() or any(a.lower() in t.lower() for a in r['alt_titles'])), None))
        matched = jobs.dropna(subset='role')

        results = []
        for role in ecsf['title']:
            official = set(ecsf[ecsf['title']==role]['competences'].iloc[0])
            role_jobs = matched[matched['role']==role]
            
            if role_jobs.empty:
                coverage = 0
            else:
                all_skills_set = set(s.lower() for sublist in role_jobs['skills_clean'] for s in sublist)
                covered = sum(1 for comp in official if any(kw in comp for kw in all_skills_set))
                coverage = covered / len(official)
            
            results.append({'Role': role.split(' (')[0], 'Coverage %': coverage})

        df = pd.DataFrame(results).sort_values('Coverage %')

        df = df[df['Coverage %'] > 0.001]

        st.subheader(f"Skill Gap indicator")

        fig = go.Figure(go.Bar(
            x=df['Role'],
            y=df['Coverage %'],
            text=df['Coverage %'].map("{:.0%}".format),
            textposition="outside",
            textfont=dict(color="black", size=13, family="Arial"),
            
            marker=dict(
                color=df['Coverage %'],
                colorscale="Blues",
                line=dict(color="darkblue", width=1.2),
            ),
            
            hovertemplate="<b>%{x}</b><br>Coverage: <b>%{y:.0%}</b><extra></extra>"
        ))

        fig.update_layout(
            title='',
            height=max(600, len(df) * 100),
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(color="black", size=13),
                titlefont=dict(color="black", size=16),
                title="<b>ECSF Role</b>"
            ),
            yaxis=dict(
                title="<b>Coverage of Official Competences</b>",
                tickformat=".0%",
                range=[0, 1.05],
                tickfont=dict(color="black", size=14),
                titlefont=dict(color="black", size=16),
                gridcolor="#e0e0e0"
            ),
            
            font=dict(family="Arial, sans-serif", color="black"),
            margin=dict(l=80, r=120, t=80, b=100),
            title_x=0.5
        )
      
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
    ## set the mode selection
    # ---------------- MODE SELECTOR ----------------
    ecsf_roles_by_title_df = fetch_all_roles_by_title()
    ecsf_role_with_tks_df = fetch_all_role_with_tks()
    
    st.subheader("📊 View Mode")
    
    mode = st.radio(
        "Choose how to view data:",
        ["📁 View Existing Database", "⚡ Real-time Streaming"],
        horizontal=True
    )

    
    if ecsf_roles_by_title_df.empty or ecsf_role_with_tks_df.empty:
        st.warning("⚠️ No static data found in the cassandra, Try to contact Samiha! Oh no contact info :')")
        st.stop()
    # ---------------- FETCH DATA BASED ON MODE ----------------
    if mode == "📁 View Existing Database":
        linkedin_jobs_df = fetch_all()
        if linkedin_jobs_df.empty:
            st.warning("⚠️ No data found in the database. Start the producer to ingest data.")
            st.stop()
        
        # the first visualization
        similarity_df = calculate_titles_similarity(linkedin_jobs_df,ecsf_roles_by_title_df)

        plot_top_ecsf_roles(similarity_df)

        st.divider()
        
        # the second visualization
        plot_skill_gap(linkedin_jobs_df,ecsf_role_with_tks_df)

        st.divider()
        
        #st.success(f"✅ Loaded {len(linkedin_jobs_df)} jobs")
        
    else:  # Real-time Streaming Mode
        st_autorefresh(interval=3000, key="matching_tracker_refresh")

        linkedin_jobs_df = fetch_recent(LOOKBACK_MINUTES)

        if linkedin_jobs_df.empty:
            st.warning("⚠️ No data found in the database. Start the producer to ingest data.")
            st.stop()

        st.success(f"🔴 LIVE: {len(linkedin_jobs_df)} jobs")
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

        similarity_df = calculate_titles_similarity(linkedin_jobs_df,ecsf_roles_by_title_df)

        plot_top_ecsf_roles(similarity_df)

        st.divider()

        # the second visualization
        plot_skill_gap(linkedin_jobs_df,ecsf_role_with_tks_df)
                
    st.divider()

# how to commit and push
# git checkout phase1
# git commit -m "add to matching tracker : first visualization" forecasting/pages/phase4.py
# git push origin phase1