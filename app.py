import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IT5006 Chicago Crime Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER: FORMAT NUMBERS ---
def format_big_number(num):
    """Formats large numbers (e.g., 1,200,000 -> 1.2M, 45,000 -> 45.0K)."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:,}"

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Metric Value Styling */
    [data-testid="stMetricValue"] {
        font-size: clamp(18px, 1.8vw, 26px) !important; 
        font-weight: bold !important;
        word-wrap: break-word !important;       
        white-space: pre-wrap !important;       
        line-height: 1.2 !important;            
        height: auto !important;                
        min-height: 50px !important;            
    }
    
    [data-testid="stMetricLabel"] {
        font-size: clamp(12px, 1.2vw, 14px) !important;
        width: 100% !important;
        white-space: normal !important;
    }
    
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 35px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
        color: #333;
    }
    
    .status-box {
        padding: 10px;
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        border-radius: 4px;
        margin-bottom: 15px;
        font-weight: bold;
        color: #0f52ba;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: CUSTOM METRIC CARD ---
def custom_metric(label, value):
    st.markdown(f"""
    <div style="
        background-color: #f9f9f9; 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        padding: 10px; 
        text-align: center;
        margin-bottom: 10px;">
        <div style="font-size: 14px; color: #666; margin-bottom: 2px;">{label}</div>
        <div style="font-size: 22px; font-weight: bold; color: #333; white-space: normal;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_crime_data():
    try:
        df = pd.read_csv('Crime_Dataset_Lite.zip')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name()
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.day_name()
        for col in ['Community Area', 'Beat', 'District', 'Ward']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        return df
    except FileNotFoundError:
        st.error("❌ Crime_Dataset_Lite.zip not found.")
        return pd.DataFrame()

@st.cache_data
def load_census_data():
    census_dict = {}
    for year in range(2015, 2025):
        fname = f"CCA_{year}.geojson"
        if os.path.exists(fname):
            try:
                gdf = gpd.read_file(fname)
                df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
                
                # GEOID logic
                if 'GEOID' in df.columns:
                    df['Community Area'] = pd.to_numeric(df['GEOID'], errors='coerce').fillna(0).astype(int)
                elif 'OBJECTID' in df.columns:
                    df['Community Area'] = pd.to_numeric(df['OBJECTID'], errors='coerce').fillna(0).astype(int)
                elif len(df) == 77:
                     df['Community Area'] = range(1, 78)

                # Metrics calculations
                pop = df['TOT_POP'].replace(0, 1) if 'TOT_POP' in df.columns else 1
                hh = df['TOT_HH'].replace(0, 1) if 'TOT_HH' in df.columns else 1

                # REQUIRED FEATURES FOR REPORT ALIGNMENT
                df['Black_Pct'] = (df['BLACK'] / pop) * 100 if 'BLACK' in df.columns else 0
                
                if 'UNEMP' in df.columns and 'IN_LBFRC' in df.columns:
                    df['Labor_Force'] = df['IN_LBFRC'].replace(0, 1)
                    df['Unemployment_Rate'] = (df['UNEMP'] / df['Labor_Force']) * 100
                else: 
                    df['Unemployment_Rate'] = 0

                if 'MEDINC' in df.columns: df['Median_Income'] = df['MEDINC']
                elif 'MED_INC' in df.columns: df['Median_Income'] = df['MED_INC']
                else: df['Median_Income'] = 0

                # Additional display columns
                df['Pct_White'] = (df['WHITE'] / pop) * 100 if 'WHITE' in df.columns else 0
                df['Pct_Hispanic'] = (df['HISP'] / pop) * 100 if 'HISP' in df.columns else 0
                df['Pct_Asian'] = (df['ASIAN'] / pop) * 100 if 'ASIAN' in df.columns else 0
                
                df['Median_HomeVal'] = df.get('MED_HV', 0)
                
                pop_25 = df.get('POP_25OV', df.get('AGE_25_UP', 1)).replace(0, 1)
                df['Pop_Over25'] = pop_25
                
                df['Pct_NoHS'] = 0
                lt_hs_cols = [c for c in df.columns if c.upper() in ['LT_HS', 'EDU_LESS_HS', 'NOT_HS_GRAD']]
                if lt_hs_cols: df['Pct_NoHS'] = (df[lt_hs_cols[0]] / pop_25) * 100
                
                df['Pct_Bach'] = 0
                bach_cols = [c for c in df.columns if c.upper() in ['BACH', 'EDU_BACH', 'BACHELORS_OR_MORE']]
                if bach_cols: df['Pct_Bach'] = (df[bach_cols[0]] / pop_25) * 100

                df['Pct_ForeignBorn'] = (df['FOR_BORN'] / pop) * 100 if 'FOR_BORN' in df.columns else 0
                df['Pct_NoVeh'] = (df['NO_VEH'] / hh) * 100 if 'NO_VEH' in df.columns else 0
                df['Avg_HH_Size'] = df['POP_HH'] / hh if 'POP_HH' in df.columns else 0

                low_inc_cols = ['HCUND20K', 'HC20Kto49K']
                if all(c in df.columns for c in low_inc_cols):
                    df['Pct_LowIncome'] = (df[low_inc_cols].sum(axis=1) / hh) * 100
                else: df['Pct_LowIncome'] = 0 

                cols = ['Community Area', 'Black_Pct', 'Pct_White', 'Pct_Hispanic', 'Pct_Asian', 
                        'Pct_LowIncome', 'Median_Income', 'Median_HomeVal', 
                        'Unemployment_Rate', 'Labor_Force', 
                        'Pct_NoHS', 'Pct_Bach', 'Pop_Over25',
                        'Pct_ForeignBorn', 'Pct_NoVeh', 'Avg_HH_Size',
                        'TOT_POP', 'TOT_HH', 'WHITE', 'BLACK', 'HISP', 'ASIAN']
                
                avail = [c for c in cols if c in df.columns]
                census_dict[year] = df[avail].set_index('Community Area').fillna(0)
            except: pass
    return census_dict

@st.cache_data
def load_geography(level):
    try:
        if level == 'Community Area':
            gdf = gpd.read_file('Boundaries.geojson')
            id_col = 'area_num_1' if 'area_num_1' in gdf.columns else 'area_numbe'
            gdf['geometry_id'] = pd.to_numeric(gdf[id_col], errors='coerce').fillna(0).astype(int)
            gdf['name'] = gdf['community'].str.title()
        elif level == 'Police Beat':
            gdf = gpd.read_file('Boundaries_beat.geojson')
            gdf['geometry_id'] = pd.to_numeric(gdf['beat_num'], errors='coerce').fillna(0).astype(int)
            gdf['name'] = "Beat " + gdf['geometry_id'].astype(str)
        elif level == 'Police District':
            gdf = gpd.read_file('Boundaries_district.geojson')
            gdf['geometry_id'] = pd.to_numeric(gdf['dist_num'], errors='coerce').fillna(0).astype(int)
            gdf['name'] = "District " + gdf['geometry_id'].astype(str)
        elif level == 'Ward':
            gdf = gpd.read_file('Boundaries_ward.geojson')
            gdf['geometry_id'] = pd.to_numeric(gdf['ward'], errors='coerce').fillna(0).astype(int)
            gdf['name'] = "Ward " + gdf['geometry_id'].astype(str)
        return gdf 
    except FileNotFoundError:
        return gpd.GeoDataFrame()

# --- HELPER: CLUSTERING (AMENDED FOR SUPERIOR FEATURES) ---
def run_clustering(census_df):
    if census_df.empty: return census_df
    # Features aligned with report best model
    features = ['Median_Income', 'Unemployment_Rate', 'Black_Pct']
    X = census_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Consistent Ranking Sort
    temp_df = pd.DataFrame({'label': labels, 'income': census_df['Median_Income']})
    rank = temp_df.groupby('label')['income'].mean().sort_values(ascending=False).index
    remap = {old_label: new_label for new_label, old_label in enumerate(rank)}
    census_df = census_df.copy()
    census_df['Cluster'] = pd.Series(labels, index=X.index).map(remap)
    return census_df

# --- HELPER: BENCHMARK CALCULATOR ---
def calculate_benchmark(df, metric, denominator=None, method='mean'):
    if method == 'median':
        return df[metric].median()
    elif method == 'mean' and denominator in df.columns:
        total_num = (df[metric] / 100 * df[denominator]).sum()
        total_den = df[denominator].sum()
        return (total_num / total_den) * 100 if total_den != 0 else df[metric].mean()
    return df[metric].mean()

# --- MAIN APP ---
df_crime = load_crime_data()
census_data = load_census_data()

if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

# --- SIDEBAR ---
st.sidebar.header("Filter Controls") 
geo_level = st.sidebar.radio("Geography Level:", ('Community Area', 'Police District', 'Police Beat', 'Ward'))
years = st.sidebar.slider("Year Range:", int(df_crime['Year'].min()), int(df_crime['Year'].max()), (2020, 2024))

st.sidebar.subheader("Crime Filters")
cats = sorted(df_crime['Crime_Category'].unique().tolist())
sel_cat = st.sidebar.selectbox("Category:", ["All"] + cats)

# --- PROFILE SELECTOR ---
valid_communities = None
census_year_data = None
cluster_names = {0: "Affluent / High SES", 1: "Working Class / Mixed", 2: "Vulnerable / Low SES"}

if geo_level == 'Community Area':
    mid_year = max(2015, min(2024, int((years[0] + years[1]) / 2)))
    if mid_year in census_data:
        census_year_data = run_clustering(census_data[mid_year])
        st.sidebar.markdown("---")
        st.sidebar.subheader("Neighborhood Profile")
        sel_cluster = st.sidebar.selectbox("Filter by Archetype:", ["All Neighborhoods"] + list(cluster_names.values()))
        if sel_cluster != "All Neighborhoods":
            target = [k for k, v in cluster_names.items() if v == sel_cluster][0]
            valid_communities = census_year_data[census_year_data['Cluster'] == target].index.tolist()

# --- FILTER DATA ---
mask = (df_crime['Year'].between(years[0], years[1]))
if sel_cat != "All": mask &= (df_crime['Crime_Category'] == sel_cat)
df_filtered = df_crime[mask]

if valid_communities is not None:
    df_filtered = df_filtered[df_filtered['Community Area'].isin(valid_communities)]

# --- PREP MAP ---
merge_key = {'Community Area': 'Community Area', 'Police Beat': 'Beat', 'Police District': 'District', 'Ward': 'Ward'}[geo_level]
map_agg = df_filtered.groupby(merge_key).agg(Total=('ID', 'count'), Arrest=('Arrest', 'sum')).reset_index()
map_agg['Efficiency'] = (map_agg['Arrest'] / map_agg['Total']) * 100
map_agg['geometry_id'] = map_agg[merge_key]

gdf = load_geography(geo_level).reset_index(drop=True).merge(map_agg, on='geometry_id', how='left').fillna(0)
if geo_level == 'Community Area' and census_year_data is not None:
    gdf = gdf.merge(census_year_data, left_on='geometry_id', right_index=True, how='left').fillna(0)

# --- VISUALS ---
st.title("IT5006 Chicago Crime Dashboard")
st.markdown(f"**Analyzing:** {sel_cat} | **Years:** {years[0]}-{years[1]} | **Level:** {geo_level}")

col1, col2 = st.columns([2.5, 1])

with col1:
    metric_choice = st.radio("Metric:", ('Total Volume', 'Arrest Efficiency %'), horizontal=True)
    if st.button("Clear Map Selection"):
        st.session_state.selected_id = None
        st.rerun()

    col = 'Total' if metric_choice == 'Total Volume' else 'Efficiency'
    scale = 'Reds' if metric_choice == 'Total Volume' else 'Greens'
    
    if not gdf.empty:
        # Map Zoom
        fig = px.choropleth_map(gdf, geojson=gdf.geometry, locations=gdf.index, color=col,
                                color_continuous_scale=scale, range_color=(0, gdf[col].quantile(0.95)),
                                map_style="carto-positron", zoom=9.5, center={"lat": 41.85, "lon": -87.65},
                                opacity=0.6, hover_name='name')
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
        
        if st.session_state.selected_id:
            sel_row_map = gdf[gdf['geometry_id'] == st.session_state.selected_id]
            if not sel_row_map.empty:
                fig.add_trace(go.Choroplethmap(
                    geojson=sel_row_map.geometry.__geo_interface__,
                    locations=sel_row_map.index, z=[1],
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                    marker_line_width=4, marker_line_color='red', showscale=False))

        sel = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
        if sel and sel['selection']['points']:
            clicked_id = gdf.iloc[sel['selection']['points'][0]['point_index']]['geometry_id']
            st.session_state.selected_id = None if st.session_state.selected_id == clicked_id else clicked_id
            st.rerun()

# --- PANEL VIEW ---
sel_id = st.session_state.selected_id
sel_row = gdf[gdf['geometry_id'] == sel_id].iloc[0] if sel_id and sel_id in gdf['geometry_id'].values else None
df_view = df_filtered[df_filtered[merge_key] == sel_id] if sel_id else df_filtered

with col2:
    status = f"📍 {sel_row['name']}" if sel_row is not None else "🌎 City-Wide"
    st.markdown(f'<div class="status-box">{status}</div>', unsafe_allow_html=True)
    
    tot = len(df_view)
    pop_val = sel_row['TOT_POP'] if sel_row is not None else census_year_data['TOT_POP'].sum() if census_year_data is not None else 1
    rate_1k = (tot / pop_val * 1000) if pop_val > 0 else 0
    eff_val = (df_view['Arrest'].sum() / tot * 100) if tot > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1: custom_metric("Incidents", format_big_number(tot))
    with c2: custom_metric("Rate / 1k", f"{rate_1k:.1f}")
    with c3: custom_metric("Arrest %", f"{eff_val:.1f}%")

    if geo_level == 'Community Area' and sel_row is not None:
        st.markdown("---")
        st.markdown("#### Demographics")
        labels = ['White', 'Black', 'Hispanic', 'Asian']
        values = [sel_row['Pct_White'], sel_row['Black_Pct'], sel_row['Pct_Hispanic'], sel_row.get('Pct_Asian', 0)]
        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
        fig_donut.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_donut, use_container_width=True)

# --- PROFILE (Report Aligned) ---
if geo_level == 'Community Area' and sel_row is not None and census_year_data is not None:
    st.markdown("---")
    st.markdown('<div class="section-header">Socioeconomic Profile vs. City Average</div>', unsafe_allow_html=True)
    
    r1c1, r1c2, r1c3 = st.columns(3)
    # Comparison metrics aligned with report model
    bm_inc = calculate_benchmark(census_year_data, 'Median_Income', method='median')
    r1c1.metric("Median Income", f"${sel_row['Median_Income']:,.0f}", f"{sel_row['Median_Income']-bm_inc:+,.0f}")
    
    bm_unemp = calculate_benchmark(census_year_data, 'Unemployment_Rate', denominator='Labor_Force')
    r1c2.metric("Unemployment Rate", f"{sel_row['Unemployment_Rate']:.1f}%", f"{sel_row['Unemployment_Rate']-bm_unemp:+.1f}%", delta_color="inverse")
    
    bm_black = calculate_benchmark(census_year_data, 'Black_Pct', denominator='TOT_POP')
    r1c3.metric("Black Population", f"{sel_row['Black_Pct']:.1f}%", f"{sel_row['Black_Pct']-bm_black:+.1f}%")

st.markdown("---")
st.subheader("24-Hour Crime Profile")
if not df_view.empty:
    prof = df_view.groupby('Hour').size().reset_index(name='Count')
    st.plotly_chart(px.bar(prof, x='Hour', y='Count'), use_container_width=True)
