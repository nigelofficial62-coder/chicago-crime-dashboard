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
    page_icon="Current Activity",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER: FORMAT NUMBERS (NEW) ---
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
    /* 1. RESPONSIVE Metric Value Styling */
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
    
    div.stButton > button {
        margin-top: 28px; 
        width: 100%;
    }
    
    .benchmark-text {
        text-align: center;
        font-size: 13px;
        color: #666;
        margin-top: -10px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: CUSTOM METRIC CARD (UPDATED) ---
def custom_metric(label, value):
    # FIX: Changed white-space to 'normal' so text wraps if it's too long
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
        # Loading your Lite Zip
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
                
                if 'GEOID' in df.columns:
                    df['Community Area'] = pd.to_numeric(df['GEOID'], errors='coerce').fillna(0).astype(int)
                elif 'OBJECTID' in df.columns:
                    df['Community Area'] = pd.to_numeric(df['OBJECTID'], errors='coerce').fillna(0).astype(int)
                elif len(df) == 77:
                     df['Community Area'] = range(1, 78)

                # Metrics calculations (Kept original logic)
                if 'TOT_POP' in df.columns: pop = df['TOT_POP'].replace(0, 1)
                else: pop = 1
                
                inc_cols = ['HCUND20K', 'HC20Kto49K', 'HC50Kto75K', 'HCOV75K']
                if all(c in df.columns for c in inc_cols):
                    df['Calculated_HH'] = df[inc_cols].sum(axis=1)
                elif 'TOT_HH' in df.columns:
                    df['Calculated_HH'] = df['TOT_HH']
                else:
                    df['Calculated_HH'] = 1
                hh = df['Calculated_HH'].replace(0, 1)

                if 'WHITE' in df.columns:
                    df['Pct_White'] = (df['WHITE'] / pop) * 100
                    df['Pct_Black'] = (df['BLACK'] / pop) * 100
                    df['Pct_Hispanic'] = (df['HISP'] / pop) * 100
                    df['Pct_Asian'] = (df['ASIAN'] / pop) * 100
                
                # --- UPDATE 1: Create aliases for superior model ---
                if 'Pct_Black' in df.columns:
                    df['Black_Pct'] = df['Pct_Black']
                else:
                    df['Black_Pct'] = 0
                
                low_inc_cols = ['HCUND20K', 'HC20Kto49K']
                if all(c in df.columns for c in low_inc_cols):
                    df['Pct_LowIncome'] = (df[low_inc_cols].sum(axis=1) / hh) * 100
                else: df['Pct_LowIncome'] = 0 

                df['Pct_HighIncome'] = 0
                df['Wealth_Label'] = "Wealth (>$75k)"
                if 'INC_GT_150' in df.columns:
                    df['Pct_HighIncome'] = (df['INC_GT_150'] / hh) * 100
                    df['Wealth_Label'] = "Wealth (>$150k)"
                elif 'HCOV150K' in df.columns:
                    df['Pct_HighIncome'] = (df['HCOV150K'] / hh) * 100
                    df['Wealth_Label'] = "Wealth (>$150k)"
                elif 'HCOV75K' in df.columns:
                    df['Pct_HighIncome'] = (df['HCOV75K'] / hh) * 100
                    df['Wealth_Label'] = "Wealth (>$75k)"

                if 'MEDINC' in df.columns: df['Median_Income'] = df['MEDINC']
                elif 'MED_INC' in df.columns: df['Median_Income'] = df['MED_INC']
                else: df['Median_Income'] = 0

                if 'MED_HV' in df.columns: df['Median_HomeVal'] = df['MED_HV']
                else: df['Median_HomeVal'] = 0

                if 'UNEMP' in df.columns and 'IN_LBFRC' in df.columns:
                    df['Labor_Force'] = df['IN_LBFRC'].replace(0, 1)
                    df['Pct_Unemp'] = (df['UNEMP'] / df['Labor_Force']) * 100
                else: 
                    df['Pct_Unemp'] = 0
                    df['Labor_Force'] = 1
                
                # --- UPDATE 2: Create alias for Unemployment Rate ---
                df['Unemployment_Rate'] = df['Pct_Unemp']

                pop_25 = 1
                if 'POP_25OV' in df.columns: pop_25 = df['POP_25OV'].replace(0, 1)
                elif 'AGE_25_UP' in df.columns: pop_25 = df['AGE_25_UP'].replace(0, 1)
                df['Pop_Over25'] = pop_25
                
                df['Pct_NoHS'] = 0
                df['Pct_Bach'] = 0
                lt_hs_cols = [c for c in df.columns if c.upper() in ['LT_HS', 'EDU_LESS_HS', 'NOT_HS_GRAD']]
                if lt_hs_cols: df['Pct_NoHS'] = (df[lt_hs_cols[0]] / pop_25) * 100
                bach_cols = [c for c in df.columns if c.upper() in ['BACH', 'EDU_BACH', 'BACHELORS_OR_MORE']]
                if bach_cols: df['Pct_Bach'] = (df[bach_cols[0]] / pop_25) * 100

                if 'FOR_BORN' in df.columns: df['Pct_ForeignBorn'] = (df['FOR_BORN'] / pop) * 100
                else: df['Pct_ForeignBorn'] = 0

                if 'NO_VEH' in df.columns: df['Pct_NoVeh'] = (df['NO_VEH'] / hh) * 100
                else: df['Pct_NoVeh'] = 0

                if 'POP_HH' in df.columns: df['Avg_HH_Size'] = df['POP_HH'] / hh
                else: df['Avg_HH_Size'] = 0

                # Added new features to column list
                cols = ['Community Area', 'Pct_White', 'Pct_Black', 'Pct_Hispanic', 'Pct_Asian', 
                        'Pct_LowIncome', 'Pct_HighIncome', 'Median_Income', 'Median_HomeVal', 
                        'Pct_Unemp', 'Labor_Force', 
                        'Pct_NoHS', 'Pct_Bach', 'Pop_Over25',
                        'Pct_ForeignBorn', 'Pct_NoVeh', 'Avg_HH_Size',
                        'TOT_POP', 'Calculated_HH', 
                        'WHITE', 'BLACK', 'HISP', 'ASIAN', 'Wealth_Label',
                        'Unemployment_Rate', 'Black_Pct'] # Added here
                
                avail = [c for c in cols if c in df.columns]
                if 'Community Area' in avail:
                    census_dict[year] = df[avail].set_index('Community Area').fillna(0)
            except: pass

    if 2015 in census_dict and 2016 in census_dict:
        df15 = census_dict[2015]
        df16 = census_dict[2016]
        idx = df15.index.intersection(df16.index)
        census_dict[2014] = df15.loc[idx].copy() 

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
        st.error(f"❌ GeoJSON for {level} not found.")
        return gpd.GeoDataFrame()

# --- HELPER: CLUSTERING (UPDATED) ---
def run_clustering(census_df):
    if census_df.empty: return census_df
    
    # --- UPDATE 3: Use the Superior Feature Set ---
    features = ['Median_Income', 'Unemployment_Rate', 'Black_Pct']
    
    # Ensure all features exist
    features = [f for f in features if f in census_df.columns]
    
    X = census_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # --- UPDATE 4: Smart Sort by Income Descending (0 = Highest Income) ---
    temp_df = pd.DataFrame({'label': labels, 'score': census_df['Median_Income']})
    # Sort descending so Rank 0 is the highest income (Prosperous)
    rank = temp_df.groupby('label')['score'].mean().sort_values(ascending=False).index
    
    remap = {old_label: new_label for new_label, old_label in enumerate(rank)}
    census_df = census_df.copy()
    census_df['Cluster'] = pd.Series(labels, index=X.index).map(remap)
    return census_df

# --- HELPER: BENCHMARK CALCULATOR ---
def calculate_benchmark(df, metric, denominator=None, method='mean'):
    if method == 'median':
        return df[metric].median()
    elif method == 'mean' and denominator:
        try:
            total_numerator = (df[metric] / 100 * df[denominator]).sum()
            total_denominator = df[denominator].sum()
            if total_denominator == 0: return 0
            return (total_numerator / total_denominator) * 100
        except:
            return df[metric].mean() 
    return df[metric].mean()

# --- MAIN APP ---
df_crime = load_crime_data()
census_data = load_census_data()

# --- INIT STATE ---
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

# --- SIDEBAR ---
st.sidebar.header("Filter Controls") 
geo_level = st.sidebar.radio("Geography Level:", ('Community Area', 'Police District', 'Police Beat', 'Ward'))
years = st.sidebar.slider("Year Range:", int(df_crime['Year'].min()), int(df_crime['Year'].max()), (2020, 2024))

st.sidebar.subheader("Time Filters")
all_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
sel_months = st.sidebar.multiselect("Select Months (Optional):", all_months, default=[])

st.sidebar.subheader("Crime Filters")
cats = sorted(df_crime['Crime_Category'].unique().tolist())
sel_cat = st.sidebar.selectbox("Category:", ["All"] + cats)
sel_sub = []
if sel_cat != "All":
    subs = df_crime[df_crime['Crime_Category'] == sel_cat]['Primary Type'].unique().tolist()
    sel_sub = st.sidebar.multiselect("Subtypes:", sorted(subs))

# --- PROFILE SELECTOR ---
valid_communities = None
census_year_data = None
cluster_names = {0: "Affluent / High SES", 1: "Working Class / Mixed", 2: "Vulnerable / Low SES"}

if geo_level == 'Community Area':
    mid_year = max(2014, min(2024, int((years[0] + years[1]) / 2)))
    if mid_year in census_data:
        census_year_data = run_clustering(census_data[mid_year])
        st.sidebar.markdown("---")
        st.sidebar.subheader("Neighborhood Profile")
        sel_cluster = st.sidebar.selectbox("Filter by Archetype:", ["All Neighborhoods"] + list(cluster_names.values()))
        if sel_cluster != "All Neighborhoods":
            target_cluster = [k for k, v in cluster_names.items() if v == sel_cluster][0]
            census_filtered = census_year_data[census_year_data['Cluster'] == target_cluster]
            valid_communities = census_filtered.index.tolist()

# --- FILTER DATA ---
mask = (df_crime['Year'].between(years[0], years[1]))
if sel_months: mask &= (df_crime['Month'].isin(sel_months))
if sel_cat != "All": mask &= (df_crime['Crime_Category'] == sel_cat)
if sel_sub: mask &= (df_crime['Primary Type'].isin(sel_sub))
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
if valid_communities is not None and geo_level == 'Community Area':
    gdf = gdf[gdf['geometry_id'].isin(valid_communities)].reset_index(drop=True)

# --- VISUALS ---
st.title("IT5006 Chicago Crime Dashboard")

if years[0] == years[1]: year_text = f"{years[0]}"
else: year_text = f"{years[0]} - {years[1]}"

st.markdown(f"**Analyzing:** {sel_cat} | **Year:** {year_text} | **Level:** {geo_level}")

col1, col2 = st.columns([2.5, 1])

with col1:
    c_map_controls = st.columns([2.5, 1])
    with c_map_controls[0]:
        metric = st.radio("Metric:", ('Total Volume', 'Arrest Efficiency %'), horizontal=True)
    with c_map_controls[1]:
        if st.button("Clear Map Selection"):
            st.session_state.selected_id = None
            st.rerun()

    col = 'Total' if metric == 'Total Volume' else 'Efficiency'
    scale = 'Reds' if metric == 'Total Volume' else 'Greens'
    
    if not gdf.empty:
        h_data = {'Total':True, 'Efficiency':':.1f'}
        if 'Median_Income' in gdf.columns: h_data.update({'Median_Income': ':$,.0f'})
        
        # Base Map
        fig = px.choropleth_map(gdf, geojson=gdf.geometry, locations=gdf.index, color=col,
                                color_continuous_scale=scale, range_color=(0, gdf[col].quantile(0.95)),
                                map_style="carto-positron", zoom=9.5, center={"lat": 41.85, "lon": -87.65},
                                opacity=0.6, hover_name='name', hover_data=h_data)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
        
        # Highlight Layer
        if st.session_state.selected_id is not None:
            sel_row = gdf[gdf['geometry_id'] == st.session_state.selected_id]
            if not sel_row.empty:
                fig.add_trace(go.Choroplethmap(
                    geojson=sel_row.geometry.__geo_interface__,
                    locations=sel_row.index,
                    z=[1] * len(sel_row),
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                    marker_line_width=4,      
                    marker_line_color='red', 
                    showscale=False,
                    hoverinfo='skip',
                    name='Selected'
                ))

        # Capture Click
        sel = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
        if sel and sel['selection']['points']:
            idx = sel['selection']['points'][0]['point_index']
            clicked_id = gdf.iloc[idx]['geometry_id']
            if st.session_state.selected_id == clicked_id:
                st.session_state.selected_id = None
            else:
                st.session_state.selected_id = clicked_id
            st.rerun()
    else:
        st.warning("No data matches filters.")
        sel = None

# --- DETERMINE VIEW ---
sel_id = st.session_state.selected_id
sel_name = "City-Wide"
sel_row = None

if sel_id is not None:
    if sel_id in gdf['geometry_id'].values:
        sel_row = gdf[gdf['geometry_id'] == sel_id].iloc[0]
        sel_name = sel_row['name']
        df_view = df_filtered[df_filtered[merge_key] == sel_id]
    else:
        st.session_state.selected_id = None
        df_view = df_filtered
else:
    df_view = df_filtered

with col2:
    if sel_id:
        st.markdown(f'<div class="status-box">📍 Viewing: {sel_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box">🌎 Viewing: City-Wide</div>', unsafe_allow_html=True)
        
    tot = len(df_view)
    eff = (df_view['Arrest'].sum() / tot * 100) if tot > 0 else 0
    pop_val = sel_row['TOT_POP'] if sel_row is not None and 'TOT_POP' in sel_row else 0
    if pop_val == 0 and census_year_data is not None: pop_val = census_year_data['TOT_POP'].sum()
    rate_per_1k = (tot / pop_val * 1000) if pop_val > 0 else 0

    c1, c2, c3 = st.columns(3)
    # FIX: Applied format_big_number here
    with c1: custom_metric("Incidents", format_big_number(tot))
    with c2: custom_metric("Rate / 1k", f"{rate_per_1k:.1f}")
    with c3: custom_metric("Arrest %", f"{eff:.1f}%")

    st.markdown("#### Top Crime Types")
    if not df_view.empty:
        top_crimes = df_view['Primary Type'].value_counts().head(5)
        for i, (crime, count) in enumerate(top_crimes.items(), 1):
            st.markdown(f"<div style='font-size: 14px; margin-bottom: 4px;'><b>{i}. {crime}</b>: {count:,}</div>", unsafe_allow_html=True)
    else:
        st.write("No data.")

    if geo_level == 'Community Area' and sel_row is not None:
        st.markdown("---")
        st.markdown("#### Demographics")
        labels = ['White', 'Black', 'Hispanic', 'Asian']
        values = [sel_row['Pct_White'], sel_row['Pct_Black'], sel_row['Pct_Hispanic'], sel_row.get('Pct_Asian', 0)]
        
        color_map = {'White': '#1f77b4', 'Black': '#d62728', 'Hispanic': '#2ca02c', 'Asian': '#ff7f0e'}
        marker_colors = [color_map[l] for l in labels]

        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=marker_colors))])
        
        # Centered Legend
        fig_donut.update_layout(
            showlegend=True, 
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), 
            margin=dict(t=0, b=0, l=0, r=0), 
            height=200
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
        if census_year_data is not None:
            tot_pop_city = census_year_data['TOT_POP'].sum()
            avg_white = (census_year_data['WHITE'].sum() / tot_pop_city) * 100
            avg_black = (census_year_data['BLACK'].sum() / tot_pop_city) * 100
            avg_hisp = (census_year_data['HISP'].sum() / tot_pop_city) * 100
            avg_asian = (census_year_data['ASIAN'].sum() / tot_pop_city) * 100
            st.caption(f"**Vs. City Avg:** White: {avg_white:.1f}% | Black: {avg_black:.1f}% | Hisp: {avg_hisp:.1f}% | Asian: {avg_asian:.1f}%")

# --- SOCIOECONOMIC DASHBOARD (UI FIX: RANKINGS, SMART COLORS & >150k DEFAULT) ---
if geo_level == 'Community Area' and sel_row is not None and census_year_data is not None:
    st.markdown("---") 
    st.markdown('<div class="section-header">Socioeconomic Profile (Selected vs. City Benchmark)</div>', unsafe_allow_html=True)
    
    # 1. Helper to calculate Rank (e.g., "1st", "15th")
    def get_rank_str(df, metric, current_val, high_is_rank_1=True):
        try:
            # Rank the entire city data for this metric
            ranks = df[metric].rank(ascending=not high_is_rank_1, method='min')
            r = ranks[df[metric] == current_val].iloc[0]
            
            n = int(r)
            if 11 <= (n % 100) <= 13: suffix = 'th'
            else: suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
            return f"({n}{suffix})"
        except:
            return ""

    # 2. UI RENDERER (Fixed: Added benchmark_label back)
    def render_box(col, label, val, benchmark_val, rank_str, suffix="", prefix="", benchmark_label="Avg", is_bad_high=False):
        import textwrap 
        
        # Color Logic
        diff = val - benchmark_val
        pct_diff = (diff / benchmark_val) * 100 if benchmark_val != 0 else 0
        
        # Neutral Zone: If within 5% of city average, stay Black
        if abs(pct_diff) < 5: 
            color = "#333333" 
        elif is_bad_high:
            color = "#d62728" if diff > 0 else "#2ca02c" # Red if High (Bad)
        else:
            color = "#2ca02c" if diff > 0 else "#d62728" # Green if High (Good)

        # HTML Block
        html_code = textwrap.dedent(f"""
            <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size: 14px; color: #555; margin-bottom: 5px;">
                    {label}
                </div>
                <div style="font-size: 24px; font-weight: bold; color: {color}; margin-bottom: 8px;">
                    {prefix}{val:,.1f}{suffix} <span style="font-size: 16px; color: #888; font-weight: normal;">{rank_str}</span>
                </div>
                <div style="font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 5px;">
                    City {benchmark_label}: <b>{prefix}{benchmark_val:,.1f}{suffix}</b>
                </div>
            </div>
        """)
        
        with col:
            st.markdown(html_code, unsafe_allow_html=True)

    # --- RENDER METRICS ---
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    
    # 1. Median Income
    bm_inc = calculate_benchmark(census_year_data, 'Median_Income', method='median')
    rank_inc = get_rank_str(census_year_data, 'Median_Income', sel_row['Median_Income'], high_is_rank_1=True)
    render_box(r1c1, "Median Income", sel_row['Median_Income'], bm_inc, rank_inc, prefix="$", is_bad_high=False)
    
    # 2. Home Value
    bm_hv = calculate_benchmark(census_year_data, 'Median_HomeVal', method='median')
    rank_hv = get_rank_str(census_year_data, 'Median_HomeVal', sel_row['Median_HomeVal'], high_is_rank_1=True)
    render_box(r1c2, "Median Home Value", sel_row['Median_HomeVal'], bm_hv, rank_hv, prefix="$", is_bad_high=False)
    
    # 3. Poverty
    bm_low = calculate_benchmark(census_year_data, 'Pct_LowIncome', denominator='Calculated_HH', method='mean')
    rank_low = get_rank_str(census_year_data, 'Pct_LowIncome', sel_row['Pct_LowIncome'], high_is_rank_1=False)
    render_box(r1c3, "Poverty (<$50k)", sel_row['Pct_LowIncome'], bm_low, rank_low, suffix="%", is_bad_high=True)
    
    # 4. Wealth (Updated Default to >$150k)
    wealth_label = sel_row.get('Wealth_Label', "Wealth (>$150k)")
    bm_high = calculate_benchmark(census_year_data, 'Pct_HighIncome', denominator='Calculated_HH', method='mean')
    rank_high = get_rank_str(census_year_data, 'Pct_HighIncome', sel_row['Pct_HighIncome'], high_is_rank_1=True)
    render_box(r1c4, wealth_label, sel_row['Pct_HighIncome'], bm_high, rank_high, suffix="%", is_bad_high=False)

    r2c1, r2c2, r2c3 = st.columns(3)
    
    # 5. No High School
    bm_nohs = calculate_benchmark(census_year_data, 'Pct_NoHS', denominator='Pop_Over25', method='mean')
    rank_nohs = get_rank_str(census_year_data, 'Pct_NoHS', sel_row['Pct_NoHS'], high_is_rank_1=False)
    render_box(r2c1, "No High School Diploma", sel_row['Pct_NoHS'], bm_nohs, rank_nohs, suffix="%", is_bad_high=True)
    
    # 6. Bachelors
    bm_bach = calculate_benchmark(census_year_data, 'Pct_Bach', denominator='Pop_Over25', method='mean')
    rank_bach = get_rank_str(census_year_data, 'Pct_Bach', sel_row['Pct_Bach'], high_is_rank_1=True)
    render_box(r2c2, "Bachelor's Degree+", sel_row['Pct_Bach'], bm_bach, rank_bach, suffix="%", is_bad_high=False)
    
    # 7. Unemployment
    bm_unemp = calculate_benchmark(census_year_data, 'Pct_Unemp', denominator='Labor_Force', method='mean')
    rank_unemp = get_rank_str(census_year_data, 'Pct_Unemp', sel_row['Pct_Unemp'], high_is_rank_1=False)
    render_box(r2c3, "Unemployment Rate", sel_row['Pct_Unemp'], bm_unemp, rank_unemp, suffix="%", is_bad_high=True)

    r3c1, r3c2, r3c3 = st.columns(3)
    
    # 8. Immigrant Pop
    bm_fb = calculate_benchmark(census_year_data, 'Pct_ForeignBorn', denominator='TOT_POP', method='mean')
    rank_fb = get_rank_str(census_year_data, 'Pct_ForeignBorn', sel_row['Pct_ForeignBorn'], high_is_rank_1=True)
    render_box(r3c1, "Immigrant Population", sel_row['Pct_ForeignBorn'], bm_fb, rank_fb, suffix="%", is_bad_high=False)
    
    # 9. HH Size
    bm_hh = calculate_benchmark(census_year_data, 'Avg_HH_Size', denominator='Calculated_HH', method='mean')
    rank_hh = get_rank_str(census_year_data, 'Avg_HH_Size', sel_row['Avg_HH_Size'], high_is_rank_1=True)
    render_box(r3c2, "Average Household Size", sel_row['Avg_HH_Size'], bm_hh, rank_hh, benchmark_label="Avg")
    
    # 10. No Vehicle
    bm_noveh = calculate_benchmark(census_year_data, 'Pct_NoVeh', denominator='Calculated_HH', method='mean')
    rank_noveh = get_rank_str(census_year_data, 'Pct_NoVeh', sel_row['Pct_NoVeh'], high_is_rank_1=False)
    render_box(r3c3, "Households with No Vehicles", sel_row['Pct_NoVeh'], bm_noveh, rank_noveh, suffix="%", is_bad_high=True)

# --- ROW 3: TRENDS & HOURLY ---
st.markdown("---")
rc1, rc2 = st.columns(2)
with rc1:
    st.subheader("Monthly Trends") 
    if not df_view.empty:
        trend = df_view.groupby(pd.Grouper(key='Date', freq='ME')).size().reset_index(name='Count')
        fig = px.line(trend, x='Date', y='Count', markers=True, color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, use_container_width=True)
with rc2:
    st.subheader("24-Hour Profile") 
    if not df_view.empty:
        prof = df_view.groupby('Hour').size().reset_index(name='Count')
        fig = px.bar(prof, x='Hour', y='Count')
        fig.update_layout(xaxis=dict(type='category'))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Temporal Heatmap") 
if not df_view.empty:
    heat = df_view.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Count')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig = px.density_heatmap(heat, x='Hour', y='DayOfWeek', z='Count', color_continuous_scale='Reds',
                             category_orders={'DayOfWeek': days}, nbinsx=24, nbinsy=7)
    fig.update_traces(xgap=3, ygap=3)
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1, showgrid=False), yaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)
