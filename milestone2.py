import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
# Sets the browser tab title and expands the layout to use the full screen width
st.set_page_config(page_title="RoadSafe Analytics", layout="wide")
st.title("🚦 RoadSafe Analytics: Milestone 2")

@st.cache_data # Decorator to memoize data loading so it doesn't reload on every interaction
def load_data():
    try:
        # Loading high-performance Parquet format instead of CSV for speed
        df = pd.read_parquet('milestone1_cleaned_data.parquet')
        
        # FEATURE ENGINEERING: Extracting time-based components
        # This allows us to analyze patterns across hours, days, and months
        if 'Start_Time' in df.columns:
            df['Start_Time'] = pd.to_datetime(df['Start_Time'])
            df['Hour'] = df['Start_Time'].dt.hour
            df['Weekday'] = df['Start_Time'].dt.day_name()
            df['Month'] = df['Start_Time'].dt.month_name()
        return df
    except Exception as e:
        st.error("Dataset not found. Please run the CSV to Parquet conversion first.")
        return None

df = load_data()

if df is not None:
    # Organizes the dashboard into two distinct chapters for the milestone
    tab1, tab2 = st.tabs(["Univariate Analysis", " Bivariate Analysis"])

    # ==========================================
    #  UNIVARIATE ANALYSIS
    # Goal: Understand the distribution and frequency of individual factors.
    # ==========================================
    with tab1:
        st.header("1. Distribution of Key Factors")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Accident Severity")
            # PIE CHART: Best for showing "part-to-whole" relationships.
            # Helps identify if most accidents are minor (1-2) or severe (3-4).
            sev_counts = df['Severity'].value_counts().reset_index()
            fig_pie = px.pie(sev_counts, values='count', names='Severity', title="Severity Proportions")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("Top Weather Conditions")
            # BAR CHART: Ideal for categorical data.
            # Quickly shows which weather condition is most frequent during accidents.
            weather_top = df['Weather_Condition'].value_counts().head(10)
            st.bar_chart(weather_top)

        st.divider()
        st.subheader("2. Temporal Trends")
        t1, t2, t3 = st.columns(3)
        
        with t1:
            # LINE CHART: Best for continuous time data.
            # Visualizes the "Rush Hour" spikes (usually morning and evening).
            st.write("**By Hour (Uncovering Rush Hour)**")
            st.line_chart(df['Hour'].value_counts().sort_index())
        
        with t2:
            # ORDERED BAR CHART: Shows which days are deadliest.
            # Reindexing ensures the days appear in calendar order rather than by count.
            st.write("**By Day of Week**")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            st.bar_chart(df['Weekday'].value_counts().reindex(day_order))
            
        with t3:
            # MONTHLY BAR CHART: Helps identify seasonal patterns (e.g., winter ice or monsoon rain).
            st.write("**By Month (Seasonal Trends)**")
            st.bar_chart(df['Month'].value_counts())

    # ==========================================
    # BIVARIATE ANALYSIS
    # Goal: Determine how variables interact and drive accident severity.
    # ==========================================
    with tab2:
        st.header("1. Correlation & Severity Drivers")
        
        # HEATMAP: Visualizes the Pearson correlation coefficient.
        # It tells us if factors like Temperature or Visibility have a linear 
        # relationship with Severity (values closer to 1 or -1 indicate strong links).
        st.subheader("Correlation: Severity vs. Weather ")
        corr_features = ['Severity', 'Temperature(F)', 'Visibility(mi)', 'Humidity(%)', 'Wind_Speed(mph)']
        available = [f for f in corr_features if f in df.columns]
        
        fig_corr, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df[available].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

        st.divider()
        st.subheader("2. Detailed Impact Analysis")
        b1, b2 = st.columns(2)

        # SAMPLING: Crucial for large datasets (4M+ rows). 
        # Taking 50k samples prevents the browser from lagging while keeping statistical accuracy.
        sample_df = df.sample(n=min(50000, len(df)))

        with b1:
            # BOXPLOT: Shows the distribution of visibility across different severity levels.
            # Helps identify if "Lower Visibility" median correlates with "Higher Severity."
            st.write("**Visibility Impact (Boxplot)**")
            fig_box = px.box(sample_df, x="Severity", y="Visibility(mi)", color="Severity")
            st.plotly_chart(fig_box, use_container_width=True)

        with b2:
            # GROUPED HISTOGRAM: Compares categorical road features against Severity.
            # It helps answer: "Are accidents at Traffic Signals more or less severe than at Junctions?"
            st.write("**Road Surface (Side-by-Side)**")
            road_feat = st.selectbox("Select Road Feature", ["Junction", "Crossing", "Stop", "Traffic_Signal"])
            fig_road = px.histogram(sample_df, x=road_feat, color="Severity", barmode="group")
            st.plotly_chart(fig_road, use_container_width=True)

        # PAIR PLOT: The ultimate multivariate tool.
        # Shows scatter plots for every pair of variables and distribution on the diagonal.
        # Wrapped in a checkbox because it is computationally expensive (slow).
        if st.checkbox("Generate Multivariate Pair Plot (Slow)"):
            with st.spinner("Analyzing relationships..."):
                pair_sample = sample_df[available].dropna().sample(1000)
                fig_pair = sns.pairplot(pair_sample, hue="Severity")
                st.pyplot(fig_pair.fig)

# Sidebar indicator to show the script has finished executing logic
st.sidebar.success("Milestone 2 Logic Ready")