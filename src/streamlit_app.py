"""
Toyota GR Cup - Racing ROI Dashboard (Ultimate Edition)
=======================================================
Interactive Pit Wall Console for telemetry analysis.
Built for the Toyota GR Cup Hackathon 2025.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis functions
from roi_engine import (
    load_and_pivot_telemetry,
    clean_telemetry,
    calculate_tire_stress,
    calculate_lap_metrics,
    calculate_roi_efficiency,
    predict_tire_failure,
    generate_coaching_advice
)

# ==========================================
# 1. CONFIGURATION & CSS (The "Pit Wall" Look)
# ==========================================
st.set_page_config(
    page_title="GR Cup | Racing ROI Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "High-Tech Racing" feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    /* Global Theme Overrides */
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117; 
    }
    ::-webkit-scrollbar-thumb {
        background: #cc0000; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #ff0000; 
    }

    /* Header Styling */
    .main-header {
        display: flex;
        align-items: center;
        background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%);
        padding: 2rem;
        border-radius: 0px 0px 20px 20px;
        border-bottom: 4px solid #cc0000;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(204, 0, 0, 0.1);
    }
    
    .header-icon {
        width: 60px;
        height: 60px;
        margin-right: 20px;
        fill: #cc0000;
    }
    
    .header-title h1 {
        color: white;
        font-weight: 700;
        margin: 0;
        font-size: 3rem;
        line-height: 1;
    }
    
    .header-title p {
        color: #888;
        margin: 5px 0 0 0;
        font-size: 1.2rem;
        font-family: 'Roboto Mono', monospace;
    }

    /* Custom Metric Card */
    .metric-card {
        background: linear-gradient(180deg, #1e2126 0%, #15171b 100%);
        border: 1px solid #333;
        border-left: 4px solid #cc0000;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(204, 0, 0, 0.1);
        border-color: #555;
    }
    
    .metric-label {
        color: #8892b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Roboto Mono', monospace;
    }

    /* Sidebar Metric Card (Compact) */
    .sidebar-card {
        background: #111;
        border: 1px solid #333;
        border-left: 3px solid #cc0000;
        padding: 12px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .sidebar-card .metric-label {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 2px;
    }
    
    .sidebar-card .metric-value {
        font-size: 1.4rem;
        color: #fff;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 5px;
        font-weight: 500;
    }
    
    .delta-pos { color: #2ecc71; }
    .delta-neg { color: #e74c3c; }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: 1px solid #333;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #888;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        transition: color 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #cc0000;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #cc0000 !important;
        border-bottom: 3px solid #cc0000 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #222;
    }
    
    /* Custom Alerts */
    .alert-box {
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.95rem;
    }
    
    .alert-success {
        background-color: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        color: #2ecc71;
    }
    
    .alert-danger {
        background-color: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        color: #ff6b6b;
    }
    
    .alert-icon {
        margin-right: 15px;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
def validate_csv_format(df):
    """
    Validate that uploaded CSV has required columns and format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Uploaded dataframe
        
    Returns:
    --------
    tuple : (bool, str)
        (is_valid, error_message)
    """
    required_cols = ['timestamp', 'vehicle_id', 'telemetry_name', 'telemetry_value']
    required_telemetry = ['speed', 'accx_can', 'accy_can', 'lap']
    
    # Check basic structure
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Check telemetry channels
    available_telemetry = df['telemetry_name'].unique()
    missing_telemetry = [t for t in required_telemetry if t not in available_telemetry]
    if missing_telemetry:
        return False, f"Missing required telemetry channels: {', '.join(missing_telemetry)}"
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['telemetry_value']):
        try:
            df['telemetry_value'] = pd.to_numeric(df['telemetry_value'], errors='coerce')
        except:
            return False, "telemetry_value column must contain numeric data"
    
    return True, "CSV format valid!"

# Update load_race_data to include validation for uploaded files
@st.cache_data(show_spinner=False)
def load_race_data(filepath, is_uploaded=False):
    """Load and process race data with caching"""
    try:
        # Load raw data first
        df_raw = pd.read_csv(filepath, low_memory=False)
        
        # Validate if uploaded
        if is_uploaded:
            is_valid, msg = validate_csv_format(df_raw)
            if not is_valid:
                st.error(f"❌ CSV Validation Failed: {msg}")
                st.stop()
            else:
                st.sidebar.success(f"✅ {msg}")
        
        # Rest of existing load_race_data code...
        df = load_and_pivot_telemetry(filepath)
        df_clean = clean_telemetry(df)
        df_stress = calculate_tire_stress(df_clean)
        lap_summary = calculate_lap_metrics(df_stress)
        lap_roi = calculate_roi_efficiency(lap_summary)
        
        # Calculate driver skill silently
        if 'total_g' not in df_clean.columns:
            df_clean['total_g'] = np.sqrt(df_clean['accx_can']**2 + df_clean['accy_can']**2)
        
        # Skill logic
        combined = len(df_clean[(df_clean['accx_can'].abs() > 0.3) & (df_clean['accy_can'].abs() > 0.3)])
        total = len(df_clean)
        trail_brake_pct = (combined / total * 100) if total > 0 else 0
        
        if trail_brake_pct > 40:
            skill_level = "PRO (Trail Braking)"
            shape = "MUSHROOM"
        elif trail_brake_pct > 25:
            skill_level = "INTERMEDIATE"
            shape = "ROUNDED DIAMOND"
        else:
            skill_level = "AMATEUR (Sequential)"
            shape = "DIAMOND"
        
        skill_analysis = {
            'trail_brake_pct': trail_brake_pct,
            'skill_level': skill_level,
            'shape': shape
        }
        
        return {
            'df_clean': df_clean,
            'lap_summary': lap_summary,
            'lap_roi': lap_roi,
            'skill_analysis': skill_analysis,
            'raw_df': df
        }
    except Exception as e:
        return None

# ==========================================
# 3. INTERACTIVE PLOTTING FUNCTIONS (Plotly)
# ==========================================
def plot_interactive_friction_circle(df, skill):
    """Creates a Plotly Friction Circle"""
    # Downsample for speed if needed
    plot_df = df.sample(n=min(10000, len(df)))
    
    fig = px.scatter(
        plot_df,
        x='accy_can',
        y='accx_can',
        color='speed',
        color_continuous_scale='Turbo',
        labels={'accy_can': 'Lateral G', 'accx_can': 'Longitudinal G', 'speed': 'Speed (kph)'},
        title=f"G-Force Limits ({skill['shape']} Shape)"
    )
    
    # Add Reference Circles
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1.5, y0=-1.5, x1=1.5, y1=1.5, line_color="lime", line_dash="dash")
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1.0, y0=-1.0, x1=1.0, y1=1.0, line_color="cyan", line_dash="dot")
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis=dict(range=[-2.5, 2.5], title="Lateral G (Turning)"),
        yaxis=dict(range=[-2.5, 2.5], title="Longitudinal G (Braking/Accel)"),
        showlegend=True
    )
    return fig

def plot_interactive_tire_stress(df_all_vehicles, selected_vehicle_id, failure_threshold=None):
    """Plotly Line Chart for Tire Stress - Shows ALL vehicles with selected one highlighted"""
    
    # Generate unique colors for each vehicle
    unique_vehicles = sorted(df_all_vehicles['vehicle_id'].unique())
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    vehicle_colors = {veh: color_palette[i % len(color_palette)] for i, veh in enumerate(unique_vehicles)}
    
    fig = go.Figure()
    
    # Plot all vehicles
    for veh in unique_vehicles:
        vehicle_df = df_all_vehicles[df_all_vehicles['vehicle_id'] == veh].sort_values('lap')
        
        # Calculate cumulative stress for this vehicle
        vehicle_df = vehicle_df.copy()
        vehicle_df['cumulative_stress'] = vehicle_df['lap_tire_stress'].cumsum()
        
        # Get vehicle number for display
        veh_num = vehicle_df['vehicle_number'].iloc[0] if len(vehicle_df) > 0 else veh
        
        is_selected = (veh == selected_vehicle_id)
        
        fig.add_trace(go.Scatter(
            x=vehicle_df['lap'],
            y=vehicle_df['cumulative_stress'],
            mode='lines+markers' if is_selected else 'lines',
            name=f"Car #{veh_num}",
            line=dict(
                color=vehicle_colors[veh],
                width=4 if is_selected else 1.5,
                dash='solid' if is_selected else 'dot'
            ),
            marker=dict(size=8 if is_selected else 4),
            opacity=1.0 if is_selected else 0.4,
            showlegend=True
        ))
    
    # Add critical failure threshold (Dynamic)
    if failure_threshold:
        max_lap = df_all_vehicles['lap'].max()
        fig.add_shape(
            type="line",
            x0=0, x1=max_lap,
            y0=failure_threshold, y1=failure_threshold,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.add_annotation(
            x=max_lap * 0.9, y=failure_threshold,
            text="PREDICTED FAILURE LIMIT",
            showarrow=False,
            yshift=10,
            font=dict(color="red", size=10)
        )
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        title="CUMULATIVE TIRE ENERGY (WORK DONE) - ALL VEHICLES",
        xaxis_title="Lap Number",
        yaxis_title="Cumulative Energy (J)",
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def metric_card(label, value, delta=None, is_sidebar=False):
    delta_html = ""
    if delta:
        color_class = "delta-pos" if float(delta) >= 0 else "delta-neg"
        arrow = "▲" if float(delta) >= 0 else "▼"
        delta_html = f'<div class="metric-delta {color_class}">{arrow} {abs(float(delta)):.2f}</div>'
    
    card_class = "sidebar-card" if is_sidebar else "metric-card"
    
    return f"""
    <div class="{card_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================
def main():
    # --- HEADER ---
    st.markdown("""
        <div class="main-header">
            <svg viewBox="0 0 100 100" class="header-icon">
                <path d="M10 50 L40 50 L50 20 L60 50 L90 50 L70 70 L80 100 L50 80 L20 100 L30 70 Z" fill="#cc0000"/>
            </svg>
            <div class="header-title">
                <h1>TOYOTA GR CUP <span style="color:#cc0000">///</span> ROI</h1>
                <p>TELEMETRY ANALYSIS & STRATEGY CONSOLE</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR CONFIGURATION ---
    from PIL import Image
    import os
    
    # Try to load local logo
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=160)
    else:
        st.sidebar.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Toyota_Gazoo_Racing_logo_2019.svg/1200px-Toyota_Gazoo_Racing_logo_2019.svg.png" width="160">
            </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### SESSION CONFIGURATION")
    
    # DATA SOURCE SELECTION
    data_source = st.sidebar.radio(
        "DATA SOURCE",
        ["Pre-loaded Race Data", "Upload Custom CSV"],
        help="Choose between pre-loaded Toyota GR Cup data or upload your own telemetry CSV"
    )
    
    filepath = None
    
    if data_source == "Pre-loaded Race Data":
        # Race Selection
        race_options = {
            "Race 1 - Sonoma (Balanced)": 'data/Sonoma/Race 1/sonoma.csv',
           
        }
        
        selected_race = st.sidebar.selectbox("SELECT RACE DATA", list(race_options.keys()))
        filepath = race_options[selected_race]
        
    else:  # Upload Custom CSV
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 UPLOAD TELEMETRY CSV")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a telemetry CSV in long format with columns: timestamp, vehicle_id, telemetry_name, telemetry_value"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                filepath = tmp_file.name
            
            st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
            st.sidebar.info(f"File size: {uploaded_file.size / 1024 / 1024:.2f} MB")
        else:
            st.sidebar.warning("⚠️ Please upload a CSV file to begin analysis")
            st.info("""
                ### 📋 Expected CSV Format
                
                Your CSV should have the following structure:
                
                **Long Format (Recommended):**
                ```
                timestamp, vehicle_id, vehicle_number, lap, telemetry_name, telemetry_value
                2025-03-30T15:02:54.012Z, GR86-002-002, 2, 1, speed, 120.5
                2025-03-30T15:02:54.012Z, GR86-002-002, 2, 1, accx_can, 0.85
                2025-03-30T15:02:54.012Z, GR86-002-002, 2, 1, accy_can, -0.65
                ```
                
                **Required Telemetry Channels:**
                - `speed` - Vehicle speed (km/h)
                - `accx_can` - Longitudinal G-force
                - `accy_can` - Lateral G-force
                - `lap` - Lap number
                - `vehicle_id` - Unique vehicle identifier
                
                **Optional:**
                - `Steering_Angle`
                - `Brake_Pressure`
                - `Throttle_Position`
                - GPS coordinates (latitude, longitude)
                
                ---
                
                ### 🎯 Quick Start Guide
                
                1. Select **"Upload Custom CSV"** in the sidebar
                2. Click **"Browse files"** and select your telemetry CSV
                3. Wait for data processing (may take 30-60s for large files)
                4. Select a vehicle from the dropdown
                5. Explore the analysis tabs!
                
                **Sample data available in pre-loaded Race 1 & 2**
            """)
            st.stop()

    # Load Data (works for both pre-loaded and uploaded)
    if filepath:
        with st.spinner(f"Processing Telemetry... This may take a moment for large files."):
            try:
                data = load_race_data(filepath)
                
                # Clean up temp file if it was uploaded
                if data_source == "Upload Custom CSV" and filepath.startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(filepath)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"⚠️ Error loading data: {str(e)}")
                st.error("""
                    **Possible issues:**
                    - CSV format doesn't match expected structure
                    - Missing required columns (speed, accx_can, accy_can, lap, vehicle_id)
                    - File corruption or encoding issues
                    
                    **Try:**
                    - Check your CSV has the correct column names
                    - Ensure data is in long format (use pre-loaded races as reference)
                    - Verify file is not corrupted
                """)
                import traceback
                with st.expander("🔍 See detailed error"):
                    st.code(traceback.format_exc())
                st.stop()

        if data is None:
            st.error("⚠️ Data processing failed! Check your file format.")
            st.stop()
    else:
        st.stop()
        
    # Vehicle Filter - FIX THE MESS
    st.sidebar.markdown("### DRIVER PROFILE")
    
    # Get unique vehicle numbers from lap_summary (cleaned data) and sort properly
    # Convert to int for proper numerical sorting, then back to string for display
    vehicle_numbers = data['lap_summary']['vehicle_number'].unique()
    
    # Handle mixed types - convert all to string, extract numeric part, sort
    import re
    def extract_number(veh_str):
        """Extract numeric part from vehicle number string"""
        match = re.search(r'\d+', str(veh_str))
        return int(match.group()) if match else 0
    
    vehicles = sorted(vehicle_numbers, key=extract_number)
    # Convert back to strings for display
    vehicles = [str(v) for v in vehicles]
    
    selected_veh_num = st.sidebar.selectbox("SELECT CAR NUMBER", vehicles)
    
    # Find the matching vehicle_id
    # Match by converting both to string
    veh_data = data['lap_summary'][data['lap_summary']['vehicle_number'].astype(str) == selected_veh_num]
    
    if veh_data.empty:
        st.sidebar.error(f"⚠️ Car {selected_veh_num} has no valid data.")
        st.stop()
        
    veh_id = veh_data['vehicle_id'].iloc[0]
    veh_stats = data['lap_roi'][data['lap_roi']['vehicle_id'] == veh_id]
    
    if veh_stats.empty:
        st.sidebar.error(f"⚠️ Car {selected_veh_num} has no valid laps.")
        st.stop()
    
    # Sidebar Stats
    st.sidebar.markdown("---")
    st.sidebar.markdown(metric_card("CURRENT LAP", int(veh_stats['lap'].max()), is_sidebar=True), unsafe_allow_html=True)
    st.sidebar.markdown(metric_card("BEST EFFICIENCY", f"{veh_stats['roi_efficiency'].max():.2f}", is_sidebar=True), unsafe_allow_html=True)
    
    # --- MAIN DASHBOARD ---
    
    # 1. TOP LEVEL METRICS (The "At a Glance" view)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_laps = len(data['lap_summary'])
        st.markdown(metric_card("TOTAL LAPS", total_laps), unsafe_allow_html=True)
        
    with col2:
        avg_eff = veh_stats['roi_efficiency'].mean()
        st.markdown(metric_card("AVG EFFICIENCY", f"{avg_eff:.2f}", f"{avg_eff-0.8:.2f}"), unsafe_allow_html=True)
        
    with col3:
        avg_stress = veh_stats['lap_tire_stress'].mean()
        st.markdown(metric_card("AVG TIRE ENERGY", f"{avg_stress:,.0f}"), unsafe_allow_html=True)
        
    with col4:
        skill = data['skill_analysis']['skill_level']
        # Shorten skill string for card
        skill_short = skill.split(" ")[0]
        st.markdown(metric_card("DRIVER SKILL", skill_short), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. TABS FOR DEEP DIVE
    tab1, tab2, tab3, tab4 = st.tabs(["ROI ANALYSIS", "FRICTION CIRCLE", "PIT STRATEGY", "RAW DATA"])

    # --- TAB 1: ROI ANALYSIS ---
    with tab1:
        st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="#cc0000" style="margin-right: 10px;">
                    <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
                </svg>
                <h3 style="margin:0">RETURN ON INVESTMENT (ROI)</h3>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # ROI Timeline
            fig_roi = px.bar(
                veh_stats, x='lap', y='roi_efficiency',
                color='roi_category',
                color_discrete_map={
                    'EXCELLENT': '#2ecc71',  # Emerald Green
                    'GOOD': '#82e0aa',       # Light Green (User wanted more green!)
                    'WASTEFUL': '#e67e22',   # Orange
                    'TERRIBLE': '#e74c3c'    # Red
                },
                title="LAP-BY-LAP EFFICIENCY SCORE"
            )
            fig_roi.update_layout(
                template="plotly_dark", 
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Rajdhani'},
                yaxis=dict(range=[-5, 5], title="ROI Score (Scaled)") # Fix scale to keep bars visible
            )
            st.plotly_chart(fig_roi, use_container_width=True)
            
        with c2:
            # Distribution Pie Chart
            roi_counts = veh_stats['roi_category'].value_counts()
            fig_pie = px.pie(
                names=roi_counts.index, values=roi_counts.values,
                color=roi_counts.index,
                color_discrete_map={
                    'EXCELLENT': '#2ecc71', 
                    'GOOD': '#82e0aa', 
                    'WASTEFUL': '#e67e22', 
                    'TERRIBLE': '#e74c3c'
                },
                hole=0.5,
                title="EFFICIENCY DISTRIBUTION"
            )
            fig_pie.update_layout(
                template="plotly_dark", 
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Rajdhani'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Coaching Advice
        st.markdown("### RACE ENGINEER RADIO")
        advice = generate_coaching_advice(veh_stats, veh_id)
        if advice:
            for item in advice[:3]:
                st.markdown(f'''
                <div class="alert-box alert-danger">
                    <span class="alert-icon">🎙️</span>
                    <div><b>ENGINEER:</b> {item}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
             st.markdown('''
                <div class="alert-box alert-success">
                    <span class="alert-icon">🎙️</span>
                    <div><b>ENGINEER:</b> Pace is good. Keep managing those tires.</div>
                </div>
             ''', unsafe_allow_html=True)

    # --- TAB 2: FRICTION CIRCLE ---
    with tab2:
        st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="#cc0000" style="margin-right: 10px;">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                    <circle cx="12" cy="12" r="3"/>
                </svg>
                <h3 style="margin:0">G-FORCE FRICTION CIRCLE</h3>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            fig_circle = plot_interactive_friction_circle(data['df_clean'], data['skill_analysis'])
            fig_circle.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Rajdhani'}
            )
            st.plotly_chart(fig_circle, use_container_width=True)
        with c2:
            st.info("""
            **HOW TO READ:**
            - **X-Axis:** Turning G-Force
            - **Y-Axis:** Braking/Accel G-Force
            
            **SHAPE ANALYSIS:**
            - 💎 **Diamond:** Amateur (Sequential Inputs)
            - 🍄 **Mushroom:** Pro (Trail Braking)
            """)
            st.markdown(metric_card("TRAIL BRAKING", f"{data['skill_analysis']['trail_brake_pct']:.1f}%"), unsafe_allow_html=True)

    # --- TAB 3: TIRE DEATH PREDICTION ---
    with tab3:
        st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="#cc0000" style="margin-right: 10px;">
                    <path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-4.42 8-8-8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/>
                </svg>
                <h3 style="margin:0">PREDICTIVE STRATEGY</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Run Prediction for selected vehicle
        prediction = predict_tire_failure(data['lap_roi'], veh_id)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(metric_card("FAILURE LAP", f"{prediction['predicted_failure_lap']:.1f}"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("LAPS LEFT", f"{prediction['laps_remaining']:.1f}"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("ENERGY RATE", f"{prediction['stress_rate']:,.0f}/lap"), unsafe_allow_html=True)
        
        # Graph showing ALL vehicles with selected one highlighted
        # Pass the dynamic threshold from the prediction
        fig_stress = plot_interactive_tire_stress(
            data['lap_roi'], 
            veh_id, 
            failure_threshold=prediction.get('failure_threshold')
        )
        
        fig_stress.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Rajdhani'}
        )
        st.plotly_chart(fig_stress, use_container_width=True)
        
        # Strategy Alert
        if prediction['laps_remaining'] < 5:
            st.markdown(f'''
                <div class="alert-box alert-danger">
                    <span class="alert-icon">⚠️</span>
                    <div><b>STRATEGY ALERT (Car #{selected_veh_num}):</b> BOX BOX BOX. Tire failure imminent within {prediction['laps_remaining']:.1f} laps.</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="alert-box alert-success">
                    <span class="alert-icon">✅</span>
                    <div><b>STRATEGY (Car #{selected_veh_num}):</b> Tires look good. {prediction['laps_remaining']:.1f} laps remaining before pit window.</div>
                </div>
            ''', unsafe_allow_html=True)

    # --- TAB 4: RAW DATA ---
    with tab4:
        st.markdown("### DETAILED TELEMETRY LOGS")
        st.dataframe(veh_stats.style.background_gradient(subset=['roi_efficiency'], cmap='RdYlGn'), use_container_width=True)
        
        csv = veh_stats.to_csv(index=False).encode('utf-8')
        st.download_button("DOWNLOAD CSV REPORT", data=csv, file_name=f"roi_report_car_{selected_veh_num}.csv", mime="text/csv")

if __name__ == "__main__":
    main()