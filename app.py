import streamlit as st
import pandas as pd
import os
import joblib
import re
from datetime import datetime, timedelta
from dateutil import parser

# Page Configuration
st.set_page_config(
    page_title="Skill-Matched Job Explorer",
    page_icon="🎯",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data and Model Loading 
@st.cache_data
def load_job_data(file_path, file_description="job data"):
   
    if not os.path.exists(file_path):
        if file_description == "today's new jobs": return pd.DataFrame() # Normal if not found
        st.error(f"Data file not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.EmptyDataError: # File exists but is empty
        if file_description == "today's new jobs": return pd.DataFrame()
        st.warning(f"{file_description.capitalize()} file '{file_path}' is empty.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_description} from '{file_path}': {e}")
        return pd.DataFrame()

@st.cache_resource # Caches resources like models
def load_model(path, model_name):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path} for {model_name}")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading {model_name} from '{path}': {e}")
        return None

# Helper Functions 
def parse_posting_date(date_str):
    # Parses string dates into datetime.date objects.
    if pd.isna(date_str) or date_str == "Not found" or not isinstance(date_str, str) or date_str.strip() == "":
        return None
    try: return parser.parse(date_str).date()
    except: return None # Catch all parsing errors silently for now

def process_user_skills(raw_skills_string):
    # Cleans and tokenizes comma-separated user skill string for TF-IDF.
    if not isinstance(raw_skills_string, str) or not raw_skills_string.strip(): return ""
    s = raw_skills_string.lower()
    tokens = [t.strip() for t in s.split(',')]
    cleaned = []
    for t in tokens:
        p = t.strip(); p = re.sub(r'\s*\([^)]*\)', '', p).strip(); p = re.sub(r'[^a-z0-9\s\.\+\#\-]', ' ', p); p = ' '.join(p.split())
        if p: cleaned.append(p)
    return ' '.join(cleaned)

# File Paths and Mappings
DATA_DIR = 'data'
MODELS_DIR = 'models'
ALL_JOBS_DATA_PATH = os.path.join(DATA_DIR, 'karkidi_jobs_processed_with_clusters.csv')
TODAYS_NEW_JOBS_PATH = os.path.join(DATA_DIR, 'todays_new_jobs.csv') # Expects this from daily scraper
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')

CLUSTER_MAPPING = {
    0: "Natural Language Processing & ML", 1: "Systems Programming (Linux/Unix) & PyTorch",
    2: "Cloud Data Services (AWS & Apache)", 3: "DevOps & Multi-Cloud Platforms (AWS/GCP)",
    4: "Data Science & Analytics (SQL & Programming)", 5: "Big Data Engineering (Java, Spark, Hadoop)"
}

# --- Load Core Assets ---
df_jobs_initial = load_job_data(ALL_JOBS_DATA_PATH, "main historical job data")
df_new_today_initial = load_job_data(TODAYS_NEW_JOBS_PATH, "today's new jobs")
tfidf_vectorizer = load_model(VECTORIZER_PATH, "TF-IDF Vectorizer")
kmeans_model = load_model(KMEANS_MODEL_PATH, "KMeans Model")

# Prepare DataFrames
df_jobs = pd.DataFrame()
if not df_jobs_initial.empty:
    df_jobs = df_jobs_initial.copy()
    df_jobs['Category_Name'] = df_jobs.get('Cluster_Label', pd.Series(dtype='float')).map(CLUSTER_MAPPING).fillna("Other")
    df_jobs['Parsed_Posting_Date'] = df_jobs.get('Posting_Date', pd.Series(dtype='str')).apply(parse_posting_date)
else: 
    df_jobs = pd.DataFrame(columns=['Title', 'Company', 'Location', 'Experience', 'Summary', 'Skills',
                                    'Posting_Date', 'Job_URL', 'Job_Type', 'Scraped_Timestamp',
                                    'Cluster_Label', 'Category_Name', 'Parsed_Posting_Date'])
df_new_today = pd.DataFrame()
if not df_new_today_initial.empty:
    df_new_today = df_new_today_initial.copy()
    if 'Predicted_Category' in df_new_today.columns:
        df_new_today['Category_Name'] = df_new_today['Predicted_Category']
    elif 'Predicted_Cluster_Label' in df_new_today.columns:
        df_new_today['Category_Name'] = df_new_today['Predicted_Cluster_Label'].map(CLUSTER_MAPPING).fillna("Other")
    else: df_new_today['Category_Name'] = "Other"

# Initialize Session State for Filters
def initialize_session_state_filters():
    today = datetime.today().date()
    # Defaults based on the main df_jobs dataset
    data_min = None
    data_max = None
    if not df_jobs.empty and 'Parsed_Posting_Date' in df_jobs.columns:
        valid_dates = df_jobs['Parsed_Posting_Date'].dropna()
        if not valid_dates.empty:
            data_min = valid_dates.min()
            data_max = min(valid_dates.max(), today)
            if data_min > data_max: data_min = data_max
    
    st.session_state.default_start_date = data_min if data_min else today - timedelta(days=30)
    st.session_state.default_end_date = data_max if data_max else today
    
    # For calendar UI bounds
    st.session_state.calendar_min_bound = datetime(st.session_state.default_start_date.year, 1, 1).date() if pd.notna(st.session_state.default_start_date) else today - timedelta(days=365*2)
    st.session_state.calendar_max_bound = today

    # Initialize filter states if they don't exist
    for key, default_val in [
        ("user_skills_input", ""), ("selected_skill_categories", []), ("selected_job_types", []),
        ("start_date_filter", st.session_state.default_start_date),
        ("end_date_filter", st.session_state.default_end_date),
        ("search_term", ""), ("sort_by_date", True), ("predicted_category_for_skills", None)
    ]:
        if key not in st.session_state: st.session_state[key] = default_val

if 'app_initialized' not in st.session_state:
    initialize_session_state_filters()
    st.session_state.app_initialized = True

# Callbacks for Filter Interactions 
def clear_all_filters_callback():
    initialize_session_state_filters() 
    st.session_state.user_skills_input = "" 
    st.session_state.predicted_category_for_skills = None
    st.session_state.selected_skill_categories = []
    st.session_state.selected_job_types = []
    st.session_state.search_term = ""

def user_skills_input_callback():
    # When user types skills, predict category and update related session state
    if tfidf_vectorizer and kmeans_model and st.session_state.user_skills_input:
        processed_skills = process_user_skills(st.session_state.user_skills_input)
        if processed_skills:
            try:
                user_skill_vector = tfidf_vectorizer.transform([processed_skills])
                predicted_cluster_label = kmeans_model.predict(user_skill_vector)[0]
                predicted_category = CLUSTER_MAPPING.get(predicted_cluster_label)
                st.session_state.predicted_category_for_skills = predicted_category
                # Auto-select the predicted category in the multiselect
                st.session_state.selected_skill_categories = [predicted_category] if predicted_category else []
            except Exception as e: # Catch broad exception from model prediction/transform
                st.session_state.predicted_category_for_skills = None; st.session_state.selected_skill_categories = []
                st.sidebar.warning(f"Could not predict from skills: {str(e)[:100]}") # Show brief error
        else: # Processed skills are empty
            st.session_state.predicted_category_for_skills = None
    elif not st.session_state.user_skills_input : # User cleared skill input
        st.session_state.predicted_category_for_skills = None


def skill_category_filter_callback():
    if st.session_state.get("predicted_category_for_skills") is not None:
        current_selection_set = set(st.session_state.selected_skill_categories)
        predicted_set = set([st.session_state.predicted_category_for_skills])
        if current_selection_set != predicted_set:
            st.session_state.user_skills_input = ""
            st.session_state.predicted_category_for_skills = None

# UI 
st.title("🎯 Skill-Matched Job Explorer")
st.markdown("Discover Karkidi.com job postings tailored to your skills. Enter your skills or use filters to explore.")

if df_jobs.empty and df_new_today.empty:
    st.error("No job data available. Please run the data collection scripts.")
elif tfidf_vectorizer is None or kmeans_model is None:
    st.warning("ML models not loaded. Skill-based category prediction will be unavailable.")

# Sidebar UI
with st.sidebar:
    st.header("🔎 Filters & Search")
    if st.button("Clear All Filters", use_container_width=True):
        clear_all_filters_callback()

    st.text_area("Enter your skills (comma-separated):", height=100, key="user_skills_input", on_change=user_skills_input_callback)
    if st.session_state.get("predicted_category_for_skills"):
        st.success(f"Skills suggest: **{st.session_state.predicted_category_for_skills}**")

    if 'Category_Name' in df_jobs.columns and not df_jobs['Category_Name'].empty:
        st.multiselect("Filter by Skill Category:", options=sorted(df_jobs['Category_Name'].unique()), key="selected_skill_categories", on_change=skill_category_filter_callback)
    
    if 'Job_Type' in df_jobs.columns and not df_jobs['Job_Type'].empty:
        valid_job_types = sorted([jt for jt in df_jobs['Job_Type'].dropna().unique() if jt != "Not found"])
        if valid_job_types: st.multiselect("Filter by Job Type:", options=valid_job_types, key="selected_job_types")
    
    st.subheader("Filter by Posting Date")
    st.date_input("Start date:", key="start_date_filter", min_value=st.session_state.calendar_min_bound, max_value=st.session_state.calendar_max_bound)
    st.date_input("End date:", key="end_date_filter", min_value=st.session_state.calendar_min_bound, max_value=st.session_state.calendar_max_bound)
    
    st.text_input("Search anything (Title, Company, etc.):", key="search_term")
    st.checkbox("Sort by Posting Date (Newest First)", key="sort_by_date")

    st.markdown("---")
    st.header("📧 Get Job Alerts!")

    st.markdown("Interested in email alerts? \nSend your Name, Email, and Skill Preferences to: test_admin_job_monitor@proton.me ")
    st.markdown("---")

# Filtering Logic (Applied to df_jobs for the main list) 
df_display_main = pd.DataFrame() 
if not df_jobs.empty: 
    df_display_main = df_jobs.copy()
    # Apply filters from session_state
    if st.session_state.selected_skill_categories:
        df_display_main = df_display_main[df_display_main['Category_Name'].isin(st.session_state.selected_skill_categories)]
    if st.session_state.selected_job_types and 'Job_Type' in df_display_main.columns:
        df_display_main = df_display_main[df_display_main['Job_Type'].isin(st.session_state.selected_job_types)]
    
    # Date Filtering
    if 'Parsed_Posting_Date' in df_display_main.columns and df_display_main['Parsed_Posting_Date'].notna().any():
        if st.session_state.start_date_filter <= st.session_state.end_date_filter:
            start_filter = st.session_state.start_date_filter
            end_filter = st.session_state.end_date_filter
            df_display_main = df_display_main[
                df_display_main['Parsed_Posting_Date'].apply(lambda x: pd.notna(x) and start_filter <= x <= end_filter)
            ]
        else: st.sidebar.error("Start date cannot be after end date.") # Show error in sidebar
    
    # Text Search
    if st.session_state.search_term:
        term = st.session_state.search_term.lower()
        search_cols = ['Title', 'Company', 'Skills', 'Summary', 'Location', 'Job_Type', 'Category_Name']
        df_display_main = df_display_main[
            df_display_main[search_cols].apply(lambda row: row.astype(str).str.lower().str.contains(term, na=False).any(), axis=1)
        ]
        
    if st.session_state.sort_by_date and 'Parsed_Posting_Date' in df_display_main.columns and df_display_main['Parsed_Posting_Date'].notna().any():
        df_display_main = df_display_main.sort_values(by='Parsed_Posting_Date', ascending=False)

# Display Sections 

# Section for Today's New Jobs
if not df_new_today.empty:
    st.markdown("---")
    st.header(f"✨ Today's New Jobs ({len(df_new_today)})")
    for index, row in df_new_today.iterrows():
        expander_title_new = f"{row.get('Title', 'N/A')} at {row.get('Company', 'N/A')} (NEW!)"
        with st.expander(expander_title_new):
            st.markdown(f"**Predicted Category:** {row.get('Category_Name', 'N/A')}")
            st.markdown(f"**Company:** {row.get('Company', 'N/A')}")
            st.markdown(f"**Location:** {row.get('Location', 'N/A')}")
            if pd.notna(row.get('Posting_Date')) and row.get('Posting_Date') != 'Not found':
                st.caption(f"Posted: {row.get('Posting_Date')}")
            job_url = row.get('Job_URL', 'Not found')
            if job_url != 'Not found' and pd.notna(job_url): st.markdown(f"[{job_url}]({job_url})")
            st.subheader("Skills:")
            skills_raw = row.get('Skills_Raw', 'Not specified') # From daily scraper
            skill_list_new = [s.strip() for s in str(skills_raw).split(',') if s.strip()]
            if skill_list_new:
                tags_html_new = "".join([f"<span style='background-color: #777; color: white; border-radius: 5px; padding: 2px 6px; margin: 2px; display: inline-block; font-size: 0.85em;'>{skill}</span>" for skill in skill_list_new])
                st.markdown(tags_html_new, unsafe_allow_html=True)
            else: st.caption("Not specified")
            st.markdown("---")
    st.markdown("---")

# Section for All Filterable Jobs
st.header(f"Browse All Jobs ({len(df_display_main)} matching filters)")
if df_display_main.empty:
    st.info("No jobs match your current filter criteria for the main dataset.")
else:
    for index, row in df_display_main.iterrows():
        expander_title = f"{row.get('Title', 'N/A')} at {row.get('Company', 'N/A')}"
        with st.expander(expander_title):
            col_main, col_meta = st.columns([3,1]) # Give more space to main info
            with col_main:
                st.subheader(f"{row.get('Title', 'N/A Title')}")
                st.markdown(f"**{row.get('Company', 'N/A Company')}** | {row.get('Location', 'N/A Location')}")
                job_url = row.get('Job_URL', 'Not found')
                if job_url != 'Not found' and pd.notna(job_url):
                    st.markdown(f"[{job_url}]({job_url})")

            with col_meta:
                st.caption(f"Category: {row.get('Category_Name', 'N/A')}")
                if pd.notna(row.get('Job_Type')) and row.get('Job_Type') != 'Not found': st.caption(f"Type: {row.get('Job_Type')}")
                if pd.notna(row.get('Posting_Date')) and row.get('Posting_Date') != 'Not found': st.caption(f"Posted: {row.get('Posting_Date')}")
                if pd.notna(row.get('Experience')) and row.get('Experience') != 'Not found': st.caption(f"Experience: {row.get('Experience')}")

            st.markdown("**Skills:**")
            skills_main = row.get('Skills', 'Not specified') # This 'Skills' comes from the main processed CSV
            skill_list_main = [s.strip() for s in str(skills_main).split(',') if s.strip()]
            if skill_list_main:
                tags_html_main = "".join([f"<span style='background-color: #e0e0e0; color: #333; border-radius: 5px; padding: 2px 6px; margin: 2px; display: inline-block; font-size: 0.85em;'>{skill}</span>" for skill in skill_list_main])
                st.markdown(tags_html_main, unsafe_allow_html=True)
            else: st.caption("Not specified")

            summary_display = row.get('Summary', 'Not found')
            if summary_display != 'Not found' and pd.notna(summary_display) and str(summary_display).strip():
                with st.container(): # Container for summary for better visual grouping
                    st.markdown("**Summary:**")
                    if len(str(summary_display)) > 250: # Threshold for using text_area
                        st.text_area("Job Summary", value=str(summary_display), height=100, disabled=True, key=f"summary_main_{index}_{row.get('Job_URL','nokey')}")
                    else:
                        st.markdown(str(summary_display))
            st.markdown("---")