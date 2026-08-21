import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SAP Order Fulfillment & Stockout Risk Workbench",
    page_icon="📦",
    layout="wide"
)

# --- HEADER & BUSINESS CONTEXT ---
st.title("📦 Smart Order Fulfillment & Stockout Risk Workbench")
st.markdown("**SAP Order-to-Cash (SD) & Inventory Management (MM) Decision Support System**")
st.markdown("---")

# --- MOCK DATA GENERATION (Simulating SAP Tables VBAK / VBAP / MARD) ---
@st.cache_data
def load_sap_data():
    np.random.seed(42)
    data = {
        "Order_ID": [f"ORD-10{i}" for i in range(1, 101)],
        "Customer": [f"Client Corp {chr(65 + i%10)}" for i in range(100)],
        "Material": [f"Part-Mech-{100 + i%5}" for i in range(100)],
        "Order_Qty": np.random.randint(50, 500, 100),
        "Stock_Level": np.random.randint(100, 800, 100),
        "Lead_Time_Days": np.random.randint(2, 14, 100),
        "Credit_Limit_Exceeded": np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
    }
    df = pd.DataFrame(data)
    
    # Simple rule/ML model for Delay Risk
    df['Risk_Score'] = (
        (df['Order_Qty'] > df['Stock_Level']).astype(int) * 0.4 +
        (df['Credit_Limit_Exceeded'] * 0.3) +
        (df['Lead_Time_Days'] / 14.0 * 0.3)
    )
    df['Delayed_Risk'] = (df['Risk_Score'] > 0.5).astype(int)
    return df

df = load_sap_data()

# Train a quick ML Surrogate Model
X = df[['Order_Qty', 'Stock_Level', 'Lead_Time_Days', 'Credit_Limit_Exceeded']]
y = df['Delayed_Risk']
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

df['Predicted_Delay_Prob'] = model.predict_proba(X)[:, 1] * 100

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 SAP Process Filters")
selected_risk = st.sidebar.selectbox("Filter Risk Level", ["All Orders", "High Risk (>75% Delay Prob)", "Normal Orders"])
credit_filter = st.sidebar.checkbox("Show Only Credit Blocked Orders", value=False)

filtered_df = df.copy()
if selected_risk == "High Risk (>75% Delay Prob)":
    filtered_df = filtered_df[filtered_df['Predicted_Delay_Prob'] > 75]
elif selected_risk == "Normal Orders":
    filtered_df = filtered_df[filtered_df['Predicted_Delay_Prob'] <= 75]

if credit_filter:
    filtered_df = filtered_df[filtered_df['Credit_Limit_Exceeded'] == 1]

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Open Orders", len(df))
col2.metric("High Stockout Risk", len(df[df['Predicted_Delay_Prob'] > 75]))
col3.metric("Credit Blocked Orders", df['Credit_Limit_Exceeded'].sum())
col4.metric("Projected On-Time Delivery", f"{100 - (len(df[df['Predicted_Delay_Prob'] > 75])/len(df)*100):.1f}%")

st.markdown("---")

# --- MAIN WORKBENCH TABLE ---
st.subheader("📋 Inbound Sales Orders & AI Risk Workbench")
st.markdown("Select an order below to review fulfillment details, override credit limits, or trigger emergency stock replenishment.")

def color_risk(val):
    color = 'red' if val > 75 else ('orange' if val > 40 else 'green')
    return f'color: {color}; font-weight: bold;'

st.dataframe(
    filtered_df.style.map(color_risk, subset=['Predicted_Delay_Prob']),
    use_container_width=True
)

# --- ACTION & EXCEPTION HANDLING PANEL ---
st.markdown("---")
st.subheader("⚡ Action & Exception Handling Workbench")

selected_order = st.selectbox("Select Order ID for Action", filtered_df['Order_ID'].unique())
order_row = df[df['Order_ID'] == selected_order].iloc[0]

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"**Order Details for `{selected_order}`:**")
    st.write(f"- **Customer:** {order_row['Customer']}")
    st.write(f"- **Material:** {order_row['Material']}")
    st.write(f"- **Ordered Qty:** {order_row['Order_Qty']} units")
    st.write(f"- **Available Stock:** {order_row['Stock_Level']} units")
    st.write(f"- **Predicted Delay Risk:** `{order_row['Predicted_Delay_Prob']:.1f}%`")
    st.write(f"- **Credit Status:** {'⚠️ Blocked' if order_row['Credit_Limit_Exceeded'] == 1 else '✅ Cleared'}")

with col_b:
    st.markdown("**Manager Decision & Workflow Trigger:**")
    action = st.radio("Choose Action:", ["Approve Credit Override", "Trigger Emergency Stock Replenishment", "Hold / Escalate"])
    
    if st.button("Execute SAP Workflow Action"):
        if action == "Approve Credit Override":
            st.success(f"Successfully released credit block for {selected_order} (Logged in SAP audit trail).")
        elif action == "Trigger Emergency Stock Replenishment":
            st.warning(f"Purchase Requisition (PR) created for {order_row['Material']} to cover stock deficit.")
        else:
            st.info(f"Order {selected_order} escalated to regional sales head.")

# --- BUSINESS IMPACT / FOOTER ---
st.markdown("---")
st.markdown("*Designed for Kaar Technologies Campus Assessment — Demonstrating Process Accuracy, ML Decision Support, and Exception Controls.*")
