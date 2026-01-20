import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Retention Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. LOAD SYSTEM ARTIFACTS (CACHED) ---
@st.cache_resource
def load_system():
    # Define paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    system_path = os.path.join(current_dir, 'models', 'churn_system.pkl')
    db_path = os.path.join(current_dir, 'data', 'telecom_churn.db')
    
    if not os.path.exists(system_path):
        st.error("❌ Critical Error: 'churn_system.pkl' not found. Please run 'train_model.py' first.")
        st.stop()
        
    # Load the dictionary containing Model, Features, and Optimal Threshold
    system_data = joblib.load(system_path)
    return system_data['model'], system_data['features'], system_data['threshold'], db_path

model, features_trained, threshold, db_path = load_system()

# --- 2. FETCH ACTIVE DATA (REAL-TIME) ---
@st.cache_data
def get_active_customers(database_path):
    conn = sqlite3.connect(database_path)
    # We select only active customers (Churn = 0)
    query = """
    SELECT * FROM customers c
    JOIN services s ON c.customerID = s.customerID
    JOIN accounts a ON c.customerID = a.customerID
    JOIN churn_status t ON c.customerID = t.customerID
    WHERE t.Churn = 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Clean duplicate columns (Fix for the SQL JOIN issue)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

df_active = get_active_customers(db_path)

# --- 3. PREDICTION ENGINE ---
# A. One-Hot Encoding (Same physics as training)
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService',
                    'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod']

df_input = pd.get_dummies(df_active, columns=categorical_cols, drop_first=True)

# B. Alignment (Reindex to match training feature space)
df_input = df_input.reindex(columns=features_trained, fill_value=0)

# C. Calculate Probabilities
probs = model.predict_proba(df_input)[:, 1]

# D. Prepare Display Dataframe
df_display = df_active.copy()
df_display['Churn_Probability'] = probs
# Flag High Risk customers based on the Optimized Threshold
df_display['High_Risk'] = (probs >= threshold).astype(int)

# Filter: Only High Risk
high_risk_df = df_display[df_display['High_Risk'] == 1].copy()
high_risk_df = high_risk_df.sort_values(by='Churn_Probability', ascending=False)

# --- 4. DASHBOARD UI ---

# Header
st.title("📡 Telecom Retention Command Center")
st.markdown(f"**System Status:** Calibrated with Optimal Financial Threshold of **{threshold:.2%}**")
st.markdown("---")

# --- KPIs SECTION ---
col1, col2, col3, col4 = st.columns(4)

total_risk = len(high_risk_df)
pct_risk = total_risk / len(df_active) if len(df_active) > 0 else 0
revenue_at_risk = high_risk_df['MonthlyCharges'].sum()
avg_prob = high_risk_df['Churn_Probability'].mean() if not high_risk_df.empty else 0

with col1:
    st.metric("Customers in Danger Zone", f"{total_risk}", f"{pct_risk:.1%} of active base", delta_color="inverse")

with col2:
    st.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", "Recurring Revenue")

with col3:
    st.metric("Avg. Churn Probability", f"{avg_prob:.1%}", "Model Confidence")

with col4:
    # Projected Annual Loss (if no action is taken)
    st.metric("Projected Annual Loss", f"${revenue_at_risk * 12:,.0f}", "Without Retention")

st.markdown("---")

# --- VISUALIZATION SECTION ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📍 Risk Distribution by Internet Service")
    if not high_risk_df.empty:
        risk_by_internet = high_risk_df['InternetService'].value_counts().reset_index()
        risk_by_internet.columns = ['Internet Service', 'Count']
        
        fig_bar = px.bar(risk_by_internet, x='Internet Service', y='Count', 
                         text='Count', color='Count', 
                         color_continuous_scale='Reds',
                         title="Where is the bleeding coming from?")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No high risk data to display.")

with c2:
    st.subheader("💳 Payment Methods")
    if not high_risk_df.empty:
        risk_by_payment = high_risk_df['PaymentMethod'].value_counts().reset_index()
        risk_by_payment.columns = ['Method', 'Count']
        
        fig_pie = px.pie(risk_by_payment, values='Count', names='Method', 
                         title="High Risk Payment Methods", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No high risk data to display.")

# --- ACTIONABLE TABLE & FILTERS ---
st.subheader("Priority Action List")
st.markdown("Focus your retention efforts on these customers. Ordered by highest probability of leaving.")

# 1. SIDEBAR CONTROLS
with st.sidebar:
    st.header("🔧 Operational Filters")
    
    # Slider for Minimum Monthly Charge
    # We use min_value=0.0 and max_value from the data to be dynamic
    max_bill = float(df_active['MonthlyCharges'].max()) if not df_active.empty else 120.0
    min_revenue_filter = min_bill = st.number_input(
        "Factura Mensual Mínima ($)", 
        min_value=0.0, 
        max_value=150.0, 
        value=50.0,
        step=0.5)
    
    # Multiselect for Contract Type
    available_contracts = df_active['Contract'].unique()
    selected_contracts = st.multiselect(
        "Contract Type", 
        available_contracts, 
        default=available_contracts # Select all by default
    )
    
    st.info("Use these filters to assign specific lists to different agents (e.g., VIP Agents for bills > $80).")

# 2. APPLYING FILTERS LOGIC
# We ensure selected_contracts is not empty to avoid errors
if not selected_contracts:
    st.warning("Please select at least one contract type to view data.")
    filtered_df = high_risk_df[0:0] # Return empty dataframe structure
else:
    filtered_df = high_risk_df[
        (high_risk_df['MonthlyCharges'] >= min_revenue_filter) & 
        (high_risk_df['Contract'].isin(selected_contracts))
    ].copy()

# 3. DISPLAY TABLE
show_cols = ['customerID', 'Churn_Probability', 'MonthlyCharges', 'tenure', 'Contract', 'InternetService', 'PaymentMethod']

st.dataframe(
    filtered_df[show_cols].style.format({
        'Churn_Probability': '{:.1%}',
        'MonthlyCharges': '${:.2f}'
    }).background_gradient(subset=['Churn_Probability'], cmap='Reds'),
    use_container_width=True
)

# 4. DOWNLOAD BUTTON
csv_data = filtered_df[show_cols].to_csv(index=False).encode('utf-8')

st.download_button(
    label=f"📥 Download Priority List ({len(filtered_df)} leads)",
    data=csv_data,
    file_name="retention_priority_list.csv",
    mime="text/csv"
)