import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats 
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ==========================================
# 1. GLOBAL CONFIG & STYLE
# ==========================================
st.set_page_config(page_title="RoadSafe | AI-Powered Analytics", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #1a5f7a !important; color: white !important; }
    .stat-card {
        padding: 20px; border-radius: 10px; background-color: #ffffff;
        border: 1px solid #e0e0e0; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (Low Latency)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('milestone1_cleaned_data.parquet')
        if 'Start_Time' in df.columns:
            df['Start_Time'] = pd.to_datetime(df['Start_Time'])
            df['Hour'] = df['Start_Time'].dt.hour
            df['Day'] = df['Start_Time'].dt.day_name()
            df['Month'] = df['Start_Time'].dt.month_name()
        return df
    except:
        return None

df = load_data()

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
st.sidebar.title("🛡️ RoadSafe Control")
if df is not None:
    # Filter by State
    states = sorted(df['State'].unique())
    sel_state = st.sidebar.selectbox("Geographic Focus:", ["National (All)"] + states)
    
    main_df = df.copy()
    if sel_state != "National (All)":
        main_df = main_df[main_df['State'] == sel_state]

    # Add City Filter (New)
    if sel_state != "National (All)":  # Only show cities if a state is selected
        available_cities = sorted(main_df['City'].dropna().unique())
        sel_cities = st.sidebar.multiselect(
            "Select Cities (Optional):",
            options=available_cities,
            default=[],
            help="Filter by specific cities within the selected state"
        )
        if sel_cities:
            main_df = main_df[main_df['City'].isin(sel_cities)]

    # Filter by Weather
    weather_list = main_df['Weather_Condition'].unique()
    sel_weather = st.sidebar.multiselect("Weather Conditions:", weather_list, default=None)
    if sel_weather:
        main_df = main_df[main_df['Weather_Condition'].isin(sel_weather)]

    st.sidebar.markdown("---")
    st.sidebar.info("Intern: Chavan Srikar\n\nProject: RoadSafe Analytics")
# ==========================================
# 4. MAIN INTERFACE WITH TABS
# ==========================================
st.title("🛣️ RoadSafe: Accident Exploratory Dashboard")

if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Univariate Analysis", 
        "🔗 Bivariate Relationships", 
        "📍 Geospatial Hotspots", 
        "🔬 Statistical Inference"
    ])

    # --- TAB 1: UNIVARIATE ---
    with tab1:
        st.subheader("Single Variable Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Accident Frequency by Severity**")
            fig_sev = px.pie(main_df, names='Severity', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_sev, use_container_width=True)
        with c2:
            st.markdown("**Top 10 Weather Conditions**")
            weather_counts = main_df['Weather_Condition'].value_counts().head(10)
            st.bar_chart(weather_counts)
        
        st.markdown("**Temporal Trend (Hourly Distribution)**")
        hour_dist = main_df['Hour'].value_counts().sort_index()
        st.line_chart(hour_dist)

    # --- TAB 2: BIVARIATE ---
    with tab2:
        st.subheader("Interaction Analysis")
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**Weather vs. Severity (Heatmap)**")
            pivot = main_df.groupby(['Weather_Condition', 'Severity']).size().unstack().fillna(0).head(15)
            fig_heat = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='YlOrRd')
            st.plotly_chart(fig_heat, use_container_width=True)
        with col_y:
            st.markdown("**Severity by Day of Week**")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig_box = px.box(main_df, x='Day', y='Severity', category_orders={'Day': day_order}, color='Day')
            st.plotly_chart(fig_box, use_container_width=True)

    # --- TAB 3: GEOSPATIAL ---
    with tab3:
        st.subheader("Geographic Risk Density")
        # Optimization: Sample data for map if too large to maintain low latency
        map_df = main_df.sample(min(len(main_df), 50000)) if len(main_df) > 50000 else main_df
        
        fig_map = px.scatter_mapbox(
            map_df, lat="Start_Lat", lon="Start_Lng", color="Severity",
            size_max=15, zoom=3.5, mapbox_style="carto-positron",
            title="Localized Accident Clusters (Sampled for Latency)",
            color_continuous_scale=px.colors.diverging.RdYlGn[::-1]
        )
        fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 4: INFERENCE ---
    with tab4:
        st.subheader("Hypothesis Testing (ANOVA)")
        target_weather = ['Clear', 'Rain', 'Snow', 'Fog']
        anova_df = main_df[main_df['Weather_Condition'].isin(target_weather)].dropna(subset=['Severity'])
        
        if len(anova_df) > 100:
            groups = [anova_df[anova_df['Weather_Condition'] == w]['Severity'] for w in target_weather]
            f_val, p_val = stats.f_oneway(*groups)
            
            # Display Metrics
            k1, k2 = st.columns(2)
            k1.metric("F-Statistic", f"{f_val:.2f}")
            k2.metric("P-Value", f"{p_val:.4e}")
            
            if p_val < 0.05:
                st.success("Significant impact detected: Weather conditions influence accident severity.")
                st.markdown("**Post-Hoc Tukey HSD Test Results**")
                tukey = pairwise_tukeyhsd(endog=anova_df['Severity'], groups=anova_df['Weather_Condition'], alpha=0.05)
                st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
            else:
                st.warning("No significant difference found for the selected filter criteria.")
        else:
            st.info("Please broaden your filters to run the statistical test.")

else:
    st.error("Missing Data: Please place 'milestone1_cleaned_data.parquet' in the root directory.")
