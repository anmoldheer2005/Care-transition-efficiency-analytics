import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Care Transition & Placement Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Data loading & preprocessing
@st.cache_data
def load_data():
    file_name = "HHS_Unaccompanied_Alien_Children_Program.csv"
    try:
        df = pd.read_csv(file_name)
    except Exception:
        df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")
    
    # Clean rows and dates
    df = df.dropna(subset=['Date']).copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Clean numeric columns
    numeric_cols = [
        'Children apprehended and placed in CBP custody*',
        'Children in CBP custody',
        'Children transferred out of CBP custody',
        'Children in HHS Care',
        'Children discharged from HHS Care'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Standardize column naming
    df = df.rename(columns={
        'Children apprehended and placed in CBP custody*': 'Apprehensions',
        'Children in CBP custody': 'CBP_Custody',
        'Children transferred out of CBP custody': 'Transfers_out_CBP',
        'Children in HHS Care': 'HHS_Care',
        'Children discharged from HHS Care': 'Discharges_HHS'
    })
    
    # Feature engineering for KPIs
    df['Transfer_Efficiency_Ratio'] = np.where(df['CBP_Custody'] > 0, (df['Transfers_out_CBP'] / df['CBP_Custody']) * 100, 0)
    df['Discharge_Effectiveness_Index'] = np.where(df['HHS_Care'] > 0, (df['Discharges_HHS'] / df['HHS_Care']) * 100, 0)
    df['Pipeline_Throughput'] = np.where(df['Apprehensions'] > 0, (df['Discharges_HHS'] / df['Apprehensions']) * 100, 100)
    df['Daily_Net_Backlog'] = df['Apprehensions'] - df['Discharges_HHS']
    df['Day_Name'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6])
    
    # 7-day rolling averages
    df['Apprehensions_7d'] = df['Apprehensions'].rolling(7, min_periods=1).mean()
    df['Discharges_7d'] = df['Discharges_HHS'].rolling(7, min_periods=1).mean()
    df['HHS_Care_7d'] = df['HHS_Care'].rolling(7, min_periods=1).mean()
    
    return df

df_raw = load_data()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🎛️ Control Panel")

# Date range filter
min_date = df_raw['Date'].min().to_pydatetime()
max_date = df_raw['Date'].max().to_pydatetime()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = df_raw[(df_raw['Date'] >= pd.to_datetime(start_d)) & (df_raw['Date'] <= pd.to_datetime(end_d))].copy()
else:
    filtered_df = df_raw.copy()

st.sidebar.markdown("---")
st.sidebar.subheader("Threshold Alert Settings")
cbp_threshold = st.sidebar.slider("CBP Custody High Warning", min_value=100, max_value=500, value=300, step=25)
hhs_threshold = st.sidebar.slider("HHS Capacity Stress Warning", min_value=3000, max_value=12000, value=8500, step=500)
rolling_toggle = st.sidebar.checkbox("Use 7-Day Moving Averages", value=True)

# ----------------- MAIN DASHBOARD -----------------
st.title("🛡️ Care Transition Efficiency & Placement Outcome Analytics")
st.caption("Monitoring process velocity, intake-discharge equilibrium, and child welfare pipeline stability.")

# System Status / Visual Threshold Alerts
latest_row = filtered_df.iloc[-1] if not filtered_df.empty else df_raw.iloc[-1]
alert_col1, alert_col2 = st.columns(2)

with alert_col1:
    if latest_row['CBP_Custody'] > cbp_threshold:
        st.error(f"⚠️ **CBP Bottleneck Alert**: Active CBP load ({int(latest_row['CBP_Custody'])}) exceeds threshold ({cbp_threshold}).")
    else:
        st.success(f"✅ **CBP Transition Stable**: Current load is {int(latest_row['CBP_Custody'])} cases.")

with alert_col2:
    if latest_row['HHS_Care'] > hhs_threshold:
        st.warning(f"⚠️ **HHS Shelter Saturation Warning**: Current census ({int(latest_row['HHS_Care']):,}) exceeds threshold ({hhs_threshold:,}).")
    else:
        st.info(f"ℹ️ **HHS Shelter Status**: Active care census at {int(latest_row['HHS_Care']):,} children.")

st.markdown("---")

# Key Performance Indicators Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    avg_transfer_eff = filtered_df['Transfer_Efficiency_Ratio'].mean()
    st.metric(
        label="Transfer Efficiency Ratio (CBP → HHS)",
        value=f"{avg_transfer_eff:.1f}%",
        help="Percentage of children in CBP custody successfully transferred each day."
    )

with kpi2:
    avg_discharge_eff = filtered_df['Discharge_Effectiveness_Index'].mean()
    st.metric(
        label="Discharge Effectiveness Index",
        value=f"{avg_discharge_eff:.2f}%",
        help="Daily sponsor placement rate relative to total HHS care population."
    )

with kpi3:
    avg_throughput = filtered_df['Pipeline_Throughput'].mean()
    st.metric(
        label="Pipeline Throughput Ratio",
        value=f"{avg_throughput:.1f}%",
        help="Ratio of total exits (discharges) to total entries (apprehensions)."
    )

with kpi4:
    net_backlog_sum = filtered_df['Daily_Net_Backlog'].sum()
    st.metric(
        label="Net Intake vs Exits (Cumulative)",
        value=f"{int(net_backlog_sum):,} cases",
        delta="- Clearance Surplus" if net_backlog_sum < 0 else "+ Net Backlog",
        delta_color="normal"
    )

# ----------------- SECTION TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Care Pipeline Flow",
    "⚡ Transfer & Discharge Efficiency",
    "🚧 Bottleneck & Backlog Detection",
    "📈 Temporal & Outcome Trends"
])

# TAB 1: Care Pipeline Flow
with tab1:
    st.subheader("Multi-Stage Pipeline Movement")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        app_col = 'Apprehensions_7d' if rolling_toggle else 'Apprehensions'
        trans_col = 'Transfers_out_CBP'
        disc_col = 'Discharges_7d' if rolling_toggle else 'Discharges_HHS'
        
        fig_pipeline = go.Figure()
        fig_pipeline.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[app_col], mode='lines', name='1. Apprehensions (Entry)', line=dict(color='#E53935', width=2)))
        fig_pipeline.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[trans_col], mode='lines', name='2. CBP → HHS Transfers', line=dict(color='#FB8C00', width=2)))
        fig_pipeline.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[disc_col], mode='lines', name='3. Sponsor Discharges (Exit)', line=dict(color='#43A047', width=2)))
        fig_pipeline.update_layout(title="Daily Pipeline Flow Volume", xaxis_title="Date", yaxis_title="Children Count", hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_pipeline, use_container_width=True)
        
    with col_b:
        st.markdown("#### Cumulative Totals")
        st.write(f"• **Total Inflow (Apprehensions):** {int(filtered_df['Apprehensions'].sum()):,}")
        st.write(f"• **Total Transferred to HHS:** {int(filtered_df['Transfers_out_CBP'].sum()):,}")
        st.write(f"• **Total Placed with Sponsors:** {int(filtered_df['Discharges_HHS'].sum()):,}")
        
        avg_hhs = filtered_df['HHS_Care'].mean()
        st.write(f"• **Average Daily HHS Shelter Census:** {int(avg_hhs):,}")
        
        # Mini pie / donut chart of active distribution
        fig_donut = go.Figure(data=[go.Pie(labels=['CBP Custody', 'HHS Care'], values=[filtered_df['CBP_Custody'].mean(), filtered_df['HHS_Care'].mean()], hole=.5)])
        fig_donut.update_layout(title="Mean In-System Custody Share", showlegend=True, height=260, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

# TAB 2: Transfer & Discharge Efficiency
with tab2:
    st.subheader("Process Velocity & Efficiency Dynamics")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_trans_eff = px.line(
            filtered_df, x='Date', y='Transfer_Efficiency_Ratio',
            title="CBP Transfer Efficiency Ratio (%)",
            labels={'Transfer_Efficiency_Ratio': 'Efficiency Ratio (%)'}
        )
        fig_trans_eff.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="100% Clearance Target")
        fig_trans_eff.update_traces(line_color='#1E88E5')
        st.plotly_chart(fig_trans_eff, use_container_width=True)
        
    with col_e2:
        fig_disc_eff = px.line(
            filtered_df, x='Date', y='Discharge_Effectiveness_Index',
            title="HHS Sponsor Discharge Effectiveness Index (%)",
            labels={'Discharge_Effectiveness_Index': 'Daily Discharge % of HHS Census'}
        )
        fig_disc_eff.update_traces(line_color='#8E24AA')
        st.plotly_chart(fig_disc_eff, use_container_width=True)

# TAB 3: Bottleneck & Backlog Detection
with tab3:
    st.subheader("Care Backlog & Pressure Points")
    
    # Net daily change
    filtered_df['Backlog_Color'] = np.where(filtered_df['Daily_Net_Backlog'] > 0, 'Backlog Influx (Net +)', 'Clearance (Net -)')
    fig_backlog = px.bar(
        filtered_df, x='Date', y='Daily_Net_Backlog',
        color='Backlog_Color',
        color_discrete_map={'Backlog Influx (Net +)': '#E53935', 'Clearance (Net -)': '#00897B'},
        title="Daily Net System Inflow vs Exit (Apprehensions - Discharges)"
    )
    fig_backlog.update_layout(xaxis_title="Date", yaxis_title="Net Case Count", legend_title="Pipeline Status")
    st.plotly_chart(fig_backlog, use_container_width=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_hhs_load = px.area(filtered_df, x='Date', y='HHS_Care', title="HHS Active Care Load Over Time", color_discrete_sequence=['#3949AB'])
        fig_hhs_load.add_hline(y=hhs_threshold, line_dash="dash", line_color="red", annotation_text="Capacity Threshold")
        st.plotly_chart(fig_hhs_load, use_container_width=True)
    with col_c2:
        fig_cbp_load = px.area(filtered_df, x='Date', y='CBP_Custody', title="CBP Active Custody Load Over Time", color_discrete_sequence=['#FB8C00'])
        fig_cbp_load.add_hline(y=cbp_threshold, line_dash="dash", line_color="red", annotation_text="CBP High Warning")
        st.plotly_chart(fig_cbp_load, use_container_width=True)

# TAB 4: Temporal & Outcome Trends
with tab4:
    st.subheader("Temporal Patterns & Operational Regularity")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        # Weekday vs Weekend analysis
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_stats = filtered_df.groupby('Day_Name')[['Apprehensions', 'Discharges_HHS', 'Transfers_out_CBP']].mean().reindex(weekday_order)
        
        fig_dow = px.bar(
            weekday_stats, barmode='group',
            title="Average Activity by Day of the Week",
            labels={'value': 'Mean Daily Count', 'Day_Name': 'Day of Week'}
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        
    with col_t2:
        # Monthly aggregate trend
        filtered_df['YearMonth'] = filtered_df['Date'].dt.to_period('M').astype(str)
        monthly_df = filtered_df.groupby('YearMonth')[['Apprehensions', 'Discharges_HHS']].sum().reset_index()
        fig_monthly = px.line(monthly_df, x='YearMonth', y=['Apprehensions', 'Discharges_HHS'], title="Monthly Intake vs Placement Totals", markers=True)
        fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Total Children")
        st.plotly_chart(fig_monthly, use_container_width=True)

# ----------------- FOOTER / EXPORT -----------------
st.markdown("---")
st.subheader("📥 Export Processed Analytics")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Metrics (CSV)",
    data=csv_data,
    file_name="care_transition_efficiency_metrics.csv",
    mime="text/csv"
)