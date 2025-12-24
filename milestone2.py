import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="RoadSafe Analytics Dashboard", layout="wide")

st.title("🚦 RoadSafe Analytics: Exploratory Data Analysis")
# Updated citations to cover full ranges as requested
st.markdown("""
This dashboard covers **Milestone 2** of the RoadSafe Analytics project.
* **Week 3:** Univariate Analysis (Distributions, Time, Weather, Day/Night) 
* **Week 4:** Bivariate Analysis (Correlations, Severity Impact, Congestion) 
""")

# ==========================================
# 2. DATA LOADING (Cached for performance)
# ==========================================
@st.cache_data
def load_data():
    filename = 'milestone1_cleaned_data.csv'
    try:
        df = pd.read_csv(filename)
        # Ensure Start_Time is datetime [cite: 36]
        if 'Start_Time' in df.columns:
            df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        
        # Feature Engineering (Week 2 Requirements recap) [cite: 37]
        if 'Hour' not in df.columns and 'Start_Time' in df.columns:
            df['Hour'] = df['Start_Time'].dt.hour
        if 'Weekday' not in df.columns and 'Start_Time' in df.columns:
            df['Weekday'] = df['Start_Time'].dt.day_name()
        if 'Month' not in df.columns and 'Start_Time' in df.columns:
            df['Month'] = df['Start_Time'].dt.month_name()
            
        return df
    except FileNotFoundError:
        st.error(f"File '{filename}' not found. Please ensure it is in the same directory.")
        return None

df = load_data()

if df is not None:
    # Set global plotting style
    sns.set_style("darkgrid")
    
    # Create tabs for organization
    tab1, tab2 = st.tabs(["Week 3: Univariate Analysis", "Week 4: Bivariate Analysis"])

    # ==========================================
    # WEEK 3: UNIVARIATE ANALYSIS [cite: 39]
    # ==========================================
    with tab1:
        st.header("Week 3: Univariate Analysis")
        st.markdown("Analyzing distributions of accidents by time, weather, and lighting conditions.")
        
        col1, col2 = st.columns(2)
        
        # 1. Severity Distribution [cite: 40]
        with col1:
            st.subheader("1. Accident Severity Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='Severity', data=df, palette='viridis', ax=ax)
            ax.set_title('Distribution of Accident Severity')
            ax.set_xlabel('Severity Level')
            ax.set_ylabel('Count')
            # Add count labels
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')
            st.pyplot(fig)

        # 2. Day vs Night (PIE CHART) 
        with col2:
            st.subheader("2. Day vs. Night Accidents")
            target_col = 'Civil_Twilight' if 'Civil_Twilight' in df.columns else 'Sunrise_Sunset'
            
            if target_col in df.columns:
                counts = df[target_col].value_counts()
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['gold', 'darkblue', 'gray', 'orange'])
                ax.set_title(f'Accident Proportions: {target_col}')
                st.pyplot(fig)
            else:
                st.warning("Twilight/Sunset column not found for Pie Chart.")

        st.markdown("---")
        
        # 3. Time Analysis [cite: 41]
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3. Accidents by Hour of Day")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['Hour'], bins=24, kde=False, color='skyblue', ax=ax)
            ax.set_title('Accident Frequency by Hour')
            ax.set_xticks(range(0, 24, 2))
            st.pyplot(fig)

        with col4:
            st.subheader("4. Accidents by Day of Week")
            order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='Weekday', data=df, order=order_days, palette='muted', ax=ax)
            ax.set_title('Accident Frequency by Day')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # --- Month Analysis (Required by Milestone 2) [cite: 41] ---
        st.markdown("---")
        st.subheader("5. Accidents by Month")
        order_months = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
        existing_months = [m for m in order_months if m in df['Month'].unique()]
        
        if existing_months:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(x='Month', data=df, order=existing_months, palette='coolwarm', ax=ax)
            ax.set_title('Accident Frequency by Month')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Month data not available.")

        st.markdown("---")

        # 6. Weather Conditions [cite: 42]
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("6. Top 10 Weather Conditions")
            top_weather = df['Weather_Condition'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=top_weather.values, y=top_weather.index, palette='coolwarm', ax=ax)
            ax.set_title('Top 10 Weather Conditions')
            st.pyplot(fig)

        # --- Road Infrastructure Frequency (Required by Milestone 2) [cite: 42] ---
        with col6:
            st.subheader("7. Common Road Infrastructure Involved")
            road_features = ['Bump', 'Crossing', 'Junction', 'Station', 'Stop', 'Traffic_Signal']
            actual_features = [f for f in road_features if f in df.columns]
            
            if actual_features:
                road_counts = df[actual_features].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=road_counts.values, y=road_counts.index, palette='magma', ax=ax)
                ax.set_title('Accident Frequency by Road Feature')
                ax.set_xlabel('Count of Accidents')
                st.pyplot(fig)
            else:
                st.write("No road feature columns found.")

    # ==========================================
    # WEEK 4: BIVARIATE ANALYSIS [cite: 44]
    # ==========================================
    with tab2:
        st.header("Week 4: Bivariate & Multivariate Analysis")
        st.markdown("Analyzing correlations between severity and weather, road, and traffic factors.")

        # 1. Correlation Heatmap [cite: 47]
        st.subheader("1. Correlation Heatmap (Weather & Severity)")
        numeric_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
        available_cols = [c for c in numeric_cols if c in df.columns]
        
        if available_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[available_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

        # 2. Visibility vs Severity [cite: 46]
        st.subheader("2. Impact of Visibility on Severity")
        if 'Visibility(mi)' in df.columns:
            vis_limit = st.slider("Filter Visibility Range (miles) to remove outliers", 0, 100, 10)
            subset_vis = df[df['Visibility(mi)'] <= vis_limit]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='Severity', y='Visibility(mi)', data=subset_vis, palette='Set2', ax=ax)
            ax.set_title(f'Severity vs Visibility (<= {vis_limit} miles)')
            st.pyplot(fig)

        col7, col8 = st.columns(2)

        # 3. Traffic Congestion (Distance) vs Severity [cite: 46]
        with col7:
            st.subheader("3. Traffic Congestion Impact")
            st.caption("Using 'Distance(mi)' (length of road affected) as a proxy for congestion/impact.")
            if 'Distance(mi)' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x='Distance(mi)', y='Severity', data=df, alpha=0.5, hue='Severity', palette='deep', ax=ax)
                ax.set_title('Affected Distance vs Severity')
                ax.set_xlim(0, 20)
                st.pyplot(fig)

        # 4. Road Infrastructure Features vs Severity [cite: 46]
        with col8:
            st.subheader("4. Road Features vs High Severity")
            road_features = ['Bump', 'Crossing', 'Junction', 'Station', 'Stop', 'Traffic_Signal']
            road_data = []

            for feature in road_features:
                if feature in df.columns:
                    total_cases = df[df[feature] == True].shape[0]
                    if total_cases > 0:
                        severe_cases = df[(df[feature] == True) & (df['Severity'].isin([3, 4]))].shape[0]
                        pct_severe = (severe_cases / total_cases) * 100
                        road_data.append({'Feature': feature, 'High_Severity_Pct': pct_severe})

            if road_data:
                road_df = pd.DataFrame(road_data)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Feature', y='High_Severity_Pct', data=road_df, palette='magma', ax=ax)
                ax.set_title('Percentage of Severe Accidents (Severity 3 & 4) by Feature')
                ax.set_ylabel('Percentage (%)')
                st.pyplot(fig)
        
        # 5. Pair Plot [cite: 47] - FULL DATASET VERSION
        st.subheader("5. Pair Plot (Complete Dataset)")
        st.markdown("**Note:** This plot uses the full dataset as requested. Rendering may take time depending on data size.")
        
        if len(available_cols) > 1:
            # Using the full dataframe (df) - NO SAMPLING applied
            full_data_plot = df[available_cols].dropna()
            
            # Using diag_kind='hist' is slightly faster for large datasets than 'kde'
            pair_plot_fig = sns.pairplot(full_data_plot, hue='Severity', palette='viridis', diag_kind='hist')
            
            st.pyplot(pair_plot_fig.fig)