import streamlit as st
import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="RoadSafe Analytics - Milestone 1", page_icon="🚦", layout="wide")

# --- 2. FUNCTIONS ---

@st.cache_data
def load_data(file_source, rows=None):
    """
    Loads data efficiently. 
    Accepts a file path string (csv/zip) or an uploaded file object.
    """
    try:
        if isinstance(file_source, str):
            # If it's a string path (local file)
            if file_source.endswith('.zip'):
                return pd.read_csv(file_source, compression='zip', nrows=rows)
            else:
                return pd.read_csv(file_source, nrows=rows)
        else:
            # If it's an uploaded file object from Streamlit
            return pd.read_csv(file_source, nrows=rows)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df, drop_threshold):
    """
    Pipeline for Milestone 1: Cleaning, Imputation, and Feature Engineering.
    """
    df_clean = df.copy()
    
    # 1. Drop columns with excessive missing values (Week 2 req)
    missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
    cols_to_drop = missing_percent[missing_percent > drop_threshold].index
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # 2. Drop duplicate rows (Week 2 req)
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # 3. Imputation (Week 2 req: "Drop or impute")
    # Rule: Drop rows where CRITICAL info (Time, Loc) is missing, Impute the rest.
    if {'Start_Lat', 'Start_Lng'}.issubset(df_clean.columns):
        df_clean = df_clean.dropna(subset=['Start_Lat', 'Start_Lng'])
        
    # Fill remaining numeric columns with Median (robust to outliers)
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Fill categorical columns with Mode
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 4. DateTime Conversion & Feature Engineering (Week 2 req)
    if 'Start_Time' in df_clean.columns:
        df_clean['Start_Time'] = pd.to_datetime(df_clean['Start_Time'], errors='coerce')
        
        # New Features
        df_clean['Hour'] = df_clean['Start_Time'].dt.hour
        df_clean['Weekday'] = df_clean['Start_Time'].dt.day_name()
        df_clean['Month'] = df_clean['Start_Time'].dt.month_name()
        df_clean['Year'] = df_clean['Start_Time'].dt.year

    # 5. Handle Outliers (Week 2 req)
    # Example: Filter unrealistic temperatures if column exists
    if 'Temperature(F)' in df_clean.columns:
        df_clean = df_clean[(df_clean['Temperature(F)'] > -50) & (df_clean['Temperature(F)'] < 130)]

    return df_clean, cols_to_drop, duplicates_removed

def encode_features(df):
    """
    Week 2 Req: Encode categorical variables if needed.
    """
    df_encoded = df.copy()
    # Example: Binary Encoding for Sunrise_Sunset
    if 'Sunrise_Sunset' in df_encoded.columns:
        # Map Night/Day to 0/1. Handle potential missing values safely.
        df_encoded['Sunrise_Sunset_Code'] = df_encoded['Sunrise_Sunset'].map({'Night': 0, 'Day': 1}).fillna(-1)
    
    return df_encoded

# --- 3. SIDEBAR ---
st.sidebar.header("⚙️ Data Settings")

# Flexible File Loading: Checks for local file OR upload
local_file_path = "archive.zip" # Or "US_Accidents_March23.csv"
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=['csv', 'zip'])

use_sample = st.sidebar.checkbox("Use Sample Data (if no file)", value=True)
row_limit = st.sidebar.slider("Rows to Load", 1000, 200000, 50000)
threshold = st.sidebar.slider("Drop Column if Missing > %", 10, 100, 40)
apply_encoding = st.sidebar.checkbox("Apply Categorical Encoding", value=False)

# --- 4. MAIN CONTENT ---

st.title("🚗 RoadSafe Analytics: Milestone 1")
st.markdown("### Week 1 & 2: Initialization, Cleaning & Preprocessing")

# Week 1 Req: Define Objectives
with st.expander("📌 Project Objectives (Week 1)"):
    st.markdown("""
    **Goal:** Analyze US road accident data to uncover trends and factors contributing to severity.
    
    **Milestone 1 Outcomes:**
    1.  **Dataset Acquisition:** Load raw Kaggle dataset.
    2.  **Exploration:** Understand schema, types, and missing data.
    3.  **Cleaning:** Handle missing values, duplicates, and outliers.
    4.  **Preprocessing:** Feature engineering (Time/Date) and Encoding.
    """)

# Load Data Logic
df_raw = None

if uploaded_file is not None:
    with st.spinner("Loading uploaded file..."):
        df_raw = load_data(uploaded_file, row_limit)
elif os.path.exists(local_file_path):
    with st.spinner(f"Loading local file: {local_file_path}..."):
        df_raw = load_data(local_file_path, row_limit)
elif use_sample:
    # Create a dummy dataframe for demonstration
    st.info("Using generated sample data for demonstration. Upload a file for real analysis.")
    dates = pd.date_range(start='1/1/2023', periods=1000, freq='H')
    data = {
        'Start_Time': dates,
        'Severity': np.random.randint(1, 5, 1000),
        'Start_Lat': np.random.uniform(25, 48, 1000),
        'Start_Lng': np.random.uniform(-120, -70, 1000),
        'Temperature(F)': np.random.uniform(-10, 100, 1000),
        'Sunrise_Sunset': np.random.choice(['Day', 'Night'], 1000),
        'Precipitation(in)': [np.nan if i % 10 == 0 else 0.0 for i in range(1000)] # 10% missing
    }
    df_raw = pd.DataFrame(data)

# Main Processing
if df_raw is not None:
    # --- STEP 1: EXPLORATION (Week 1) ---
    st.subheader("1. Data Exploration")
    
    # Requirement: Check for basic statistics and data types
    exp_tab1, exp_tab2, exp_tab3 = st.tabs(["Overview & Missing", "Basic Statistics", "Data Types"])
    
    with exp_tab1:
        col1, col2 = st.columns(2)
        col1.metric("Raw Rows", df_raw.shape[0])
        col1.metric("Raw Columns", df_raw.shape[1])
        
        # Show missing values
        missing = df_raw.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if not missing.empty:
            st.write("**Top Missing Columns:**")
            st.dataframe(missing.head(10))
        else:
            st.success("No missing values found in raw load.")
            
        st.write("First 5 Rows:", df_raw.head())

    with exp_tab2:
        st.markdown("**Basic Statistics (Numerical):**")
        st.write("This satisfies the 'Check for basic statistics' requirement.")
        st.dataframe(df_raw.describe())

    with exp_tab3:
        st.markdown("**Column Data Types (Schema):**")
        st.write("This satisfies the 'Check for data types' requirement.")
        st.dataframe(df_raw.dtypes.astype(str))

    # --- STEP 2: CLEANING & PREPROCESSING (Week 2) ---
    st.subheader("2. Cleaning & Feature Engineering")
    
    df_clean, dropped_cols, dups_removed = preprocess_data(df_raw, threshold)
    
    # Requirement: Encode categorical variables
    if apply_encoding:
        df_clean = encode_features(df_clean)
        st.success("✅ Categorical Encoding Applied (e.g., Sunrise_Sunset -> 0/1)")
    else:
        st.info("ℹ️ Enable 'Apply Categorical Encoding' in sidebar to fulfill Week 2 Requirement.")

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Cols Dropped", len(dropped_cols))
    m2.metric("Duplicates Removed", dups_removed)
    m3.metric("Final Shape", f"{df_clean.shape[0]} x {df_clean.shape[1]}")

    if len(dropped_cols) > 0:
        st.caption(f"Dropped columns (> {threshold}% missing): {list(dropped_cols)}")

    # --- STEP 3: VERIFICATION VISUALS ---
    st.subheader("3. Verification of New Features")
    st.markdown("Verifying that **Hour**, **Weekday**, and **Month** were successfully extracted.")
    
    tab1, tab2 = st.tabs(["Analysis Features", "Cleaned Data View"])
    
    with tab1:
        if 'Hour' in df_clean.columns:
            st.bar_chart(df_clean['Hour'].value_counts().sort_index())
            st.caption("Accident count by Hour (extracted from Start_Time)")
        
    with tab2:
        st.dataframe(df_clean.head(10))
        # Allow user to download the milestone result
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data (CSV)",
            data=csv,
            file_name="milestone1_cleaned_data.csv",
            mime="text/csv",
        )

else:
    st.warning("Please upload a dataset or enable 'Use Sample Data' to proceed.")