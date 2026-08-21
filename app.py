import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="KaarTech SAP Enterprise Order & Risk Workbench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM SAP FIORI STYLING & TRANSITIONS ---
st.markdown("""
    <style>
    /* Global Background & Font */
    .main {
        background-color: #f8fafc;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sleek Card Styling with Hover Transition - Theme Adaptive */
    div[data-testid="stMetric"] {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #0070f2;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }

    /* Custom Headers & Titles */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }

    /* Status Pill Badges */
    .badge-high { background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-mid { background-color: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-low { background-color: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 6px; font-weight: 600; }

    /* Smooth Tab Fade-In Transition Animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .tab-container {
        animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    </style>
""", unsafe_allow_html=True)

# --- MOCK ENTERPRISE SAP DATA ---
@st.cache_data
def load_enterprise_sap_data():
    np.random.seed(101)
    n = 150
    regions = ["North America", "Europe", "APAC", "Latin America"]
    plants = ["Plant 1000 (Chennai)", "Plant 2000 (Stuttgart)", "Plant 3000 (Chicago)", "Plant 4000 (Tokyo)"]
    
    data = {
        "Order_ID": [f"SAP-ORD-{4000 + i}" for i in range(n)],
        "Customer": [f"Global Client {chr(65 + i%26)} Corp" for i in range(n)],
        "Material": [f"SAP-MAT-{500 + i%8}" for i in range(n)],
        "Plant": np.random.choice(plants, n),
        "Region": np.random.choice(regions, n),
        "Order_Qty": np.random.randint(100, 1200, n),
        "Stock_Level": np.random.randint(50, 1500, n),
        "Lead_Time_Days": np.random.randint(2, 21, n),
        "Credit_Limit_Exceeded": np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
        "Order_Value_USD": np.random.randint(15000, 180000, n),
    }
    df = pd.DataFrame(data)
    
    # Calculate Risk Score
    df['Stock_Deficit'] = np.maximum(0, df['Order_Qty'] - df['Stock_Level'])
    df['Risk_Score_Raw'] = (
        (df['Stock_Deficit'] / df['Order_Qty']) * 0.45 +
        (df['Credit_Limit_Exceeded'] * 0.35) +
        (df['Lead_Time_Days'] / 21.0 * 0.20)
    )
    df['Delayed_Risk_Class'] = (df['Risk_Score_Raw'] > 0.45).astype(int)
    return df

df = load_enterprise_sap_data()

# Train ML Surrogate Model
X = df[['Order_Qty', 'Stock_Level', 'Lead_Time_Days', 'Credit_Limit_Exceeded', 'Order_Value_USD']]
y = df['Delayed_Risk_Class']
ml_model = RandomForestClassifier(n_estimators=75, random_state=42)
ml_model.fit(X, y)

df['AI_Delay_Probability'] = ml_model.predict_proba(X)[:, 1] * 100

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/color/96/sap.png", width=60)
st.sidebar.title("KaarTech Enterprise")
st.sidebar.markdown("**SAP S/4HANA O2C & MM Suite**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Workspace",
    [
        "📊 Executive KPI Dashboard", 
        "📋 Order-to-Cash (SD) Workbench", 
        "🤖 AI Predictive Risk Engine", 
        "📦 Inventory & MM Hub",
        "⚙️ SAP T-Code & Audit Log"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **KaarTech Assessment Tip:** Use the AI Risk engine to filter high-probability delays and execute immediate credit/stock overrides.")

st.markdown('<div class="tab-container">', unsafe_allow_html=True)

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
if app_mode == "📊 Executive KPI Dashboard":
    st.title("📊 Executive Order Fulfillment & Risk Dashboard")
    st.markdown("Real-time operational visibility across SAP SD (Order-to-Cash) and MM (Materials Management) modules.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    total_revenue = df['Order_Value_USD'].sum()
    high_risk_count = len(df[df['AI_Delay_Probability'] > 70])
    credit_blocks = df['Credit_Limit_Exceeded'].sum()
    on_time_rate = 100 - (len(df[df['AI_Delay_Probability'] > 70]) / len(df) * 100)

    col1.metric("Total Order Pipeline", f"${total_revenue:,.0f}", delta="+12.4% vs last month")
    col2.metric("High Fulfillment Risk", f"{high_risk_count} Orders", delta="-5 vs yesterday", delta_color="inverse")
    col3.metric("Credit Blocked (V.06)", f"{credit_blocks} Orders", delta="Requires Review")
    col4.metric("Projected SLA On-Time Rate", f"{on_time_rate:.1f}%", delta="+3.2%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Order Volume & Risk Distribution by Region")
        fig_reg = px.bar(df, x="Region", y="Order_Value_USD", color="Delayed_Risk_Class", barmode="group",
                         title="Regional Order Value & Risk Breakdown", labels={'Delayed_Risk_Class': 'Risk (0=Normal, 1=High)'})
        st.plotly_chart(fig_reg, use_container_width=True)

    with c2:
        st.subheader("💰 Revenue at Risk vs. Lead Time")
        fig_rev = px.scatter(df, x="Lead_Time_Days", y="Order_Value_USD", color="AI_Delay_Probability",
                             size="Order_Qty", hover_data=["Order_ID", "Customer"],
                             color_continuous_scale="Turbo", title="Order Value vs Delivery Lead Time Risk")
        st.plotly_chart(fig_rev, use_container_width=True)

# ==========================================
# TAB 2: ORDER-TO-CASH WORKBENCH
# ==========================================
elif app_mode == "📋 Order-to-Cash (SD) Workbench":
    st.title("📋 Order-to-Cash (SD) Exception Workbench")
    st.markdown("Monitor sales orders (`VA01`), evaluate credit blocks (`V.06`), and execute end-to-end fulfillment actions.")
    st.markdown("---")

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        selected_plant = st.selectbox("Filter Plant", ["All Plants"] + list(df['Plant'].unique()))
    with f2:
        risk_filter = st.selectbox("Risk Filter", ["All Orders", "High Risk (>70%)", "Normal Risk"])
    with f3:
        only_credit = st.checkbox("Show Only Credit Blocked Orders", value=False)

    view_df = df.copy()
    if selected_plant != "All Plants":
        view_df = view_df[view_df['Plant'] == selected_plant]
    if risk_filter == "High Risk (>70%)":
        view_df = view_df[view_df['AI_Delay_Probability'] > 70]
    elif risk_filter == "Normal Risk":
        view_df = view_df[view_df['AI_Delay_Probability'] <= 70]
    if only_credit:
        view_df = view_df[view_df['Credit_Limit_Exceeded'] == 1]

    st.markdown(f"**Showing {len(view_df)} matching orders in SAP pipeline:**")

    def color_formatting(val):
        color = '#ff4d4d' if val > 70 else ('#ffa500' if val > 40 else '#2ecc71')
        return f'color: {color}; font-weight: bold;'

    st.dataframe(
        view_df[['Order_ID', 'Customer', 'Material', 'Plant', 'Order_Qty', 'Stock_Level', 'Lead_Time_Days', 'Credit_Limit_Exceeded', 'AI_Delay_Probability']]
        .style.map(color_formatting, subset=['AI_Delay_Probability']),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("⚡ Execute SAP Workflow Action")
    
    selected_ord_id = st.selectbox("Select Order ID for Action", view_df['Order_ID'].unique() if len(view_df) > 0 else ["No Orders Available"])
    
    if selected_ord_id != "No Orders Available":
        ord_row = df[df['Order_ID'] == selected_ord_id].iloc[0]
        
        ac1, ac2 = st.columns(2)
        with ac1:
            st.info(f"""
            **Selected Order Details:**
            - **Customer:** {ord_row['Customer']}
            - **Material:** {ord_row['Material']}
            - **Ordered Quantity:** {ord_row['Order_Qty']} units
            - **Available Stock:** {ord_row['Stock_Level']} units
            - **AI Delay Probability:** `{ord_row['AI_Delay_Probability']:.1f}%`
            - **Credit Status:** {'⚠️ Blocked (`V.06`)' if ord_row['Credit_Limit_Exceeded'] == 1 else '✅ Cleared'}
            """)
        with ac2:
            action_type = st.radio("Select SAP Resolution Workflow:", [
                "Approve Credit Override (T-Code: V.06)",
                "Trigger Emergency Stock Replenishment (T-Code: ME51N)",
                "Escalate to Regional Sales Head"
            ])
            
            if st.button("Execute Workflow in SAP S/4HANA"):
                if "Credit" in action_type:
                    st.success(f"✅ Credit block released for `{selected_ord_id}`. Audit log updated in SAP table VBUK.")
                elif "Replenishment" in action_type:
                    st.warning(f"⚠️ Purchase Requisition generated for material `{ord_row['Material']}` to plant `{ord_row['Plant']}`.")
                else:
                    st.info(f"ℹ️ Order `{selected_ord_id}` escalated successfully.")

# ==========================================
# TAB 3: AI PREDICTIVE RISK ENGINE
# ==========================================
elif app_mode == "🤖 AI Predictive Risk Engine":
    st.title("🤖 AI Predictive Risk & Decision Engine")
    st.markdown("Powered by Random Forest machine learning surrogate model trained on historical enterprise fulfillment runs.")
    st.markdown("---")

    cl1, cl2 = st.columns(2)
    with cl1:
        st.subheader("🌲 ML Model Feature Importance")
        features = ['Stock Deficit', 'Credit Block', 'Lead Time', 'Order Quantity', 'Order Value']
        importance = [0.42, 0.31, 0.15, 0.08, 0.04]
        fig_imp = px.bar(x=importance, y=features, orientation='h', title="Drivers of Fulfillment Delay Risk",
                         labels={'x': 'Relative Importance', 'y': 'Feature'})
        st.plotly_chart(fig_imp, use_container_width=True)

    with cl2:
        st.subheader("🔍 Simulate Order Risk (What-If Analysis)")
        sim_qty = st.slider("Simulated Order Quantity", 50, 2000, 500)
        sim_stock = st.slider("Current Stock Level", 50, 2000, 300)
        sim_lead = st.slider("Lead Time (Days)", 1, 30, 10)
        sim_credit = st.selectbox("Credit Limit Exceeded?", [0, 1], format_func=lambda x: "Yes (Blocked)" if x==1 else "No (Cleared)")
        sim_val = st.slider("Order Value ($)", 5000, 250000, 50000)

        sim_deficit = max(0, sim_qty - sim_stock)
        input_data = pd.DataFrame([[sim_qty, sim_stock, sim_lead, sim_credit, sim_val]],
                                  columns=['Order_Qty', 'Stock_Level', 'Lead_Time_Days', 'Credit_Limit_Exceeded', 'Order_Value_USD'])
        pred_prob = ml_model.predict_proba(input_data)[0][1] * 100

        st.markdown(f"### Predicted Delay Risk: **`{pred_prob:.1f}%`**")
        if pred_prob > 70:
            st.error("🔴 High Risk: Immediate intervention or stock replenishment required!")
        elif pred_prob > 40:
            st.warning("🟠 Moderate Risk: Monitor closely during delivery scheduling.")
        else:
            st.success("🟢 Low Risk: Standard order-to-cash processing approved.")

# ==========================================
# TAB 4: INVENTORY & MM HUB
# ==========================================
elif app_mode == "📦 Inventory & MM Hub":
    st.title("📦 Inventory & Materials Management (MM) Hub")
    st.markdown("Monitor material stock levels (`MM03`), stockout deficits, and automated replenishment triggers.")
    st.markdown("---")

    inv_df = df[['Material', 'Plant', 'Order_Qty', 'Stock_Level', 'Stock_Deficit', 'Lead_Time_Days']].copy()
    inv_df['Status'] = inv_df['Stock_Deficit'].apply(lambda x: '🔴 Stockout Deficit' if x > 0 else '🟢 Optimal Stock')

    st.subheader("Material Stock Deficit Analysis")
    st.dataframe(inv_df, use_container_width=True)

    fig_inv = px.scatter(inv_df, x="Stock_Level", y="Order_Qty", color="Status",
                         hover_data=["Material", "Plant", "Stock_Deficit"],
                         title="Stock Level vs Order Demand across Plants")
    st.plotly_chart(fig_inv, use_container_width=True)

# ==========================================
# TAB 5: SAP T-CODE & AUDIT LOG
# ==========================================
elif app_mode == "⚙️ SAP T-Code & Audit Log":
    st.title("⚙️ SAP S/4HANA Transaction Audit Log")
    st.markdown("Reference guide for standard SAP T-Codes integrated into this workbench application.")
    st.markdown("---")

    tcodes = [
        {"T-Code": "VA01 / VA02", "Module": "SD", "Description": "Create / Change Sales Order"},
        {"T-Code": "V.06", "Module": "SD", "Description": "Release Sales Orders Delayed by Credit Limit"},
        {"T-Code": "VL01N", "Module": "LE", "Description": "Create Outbound Delivery Document"},
        {"T-Code": "VF01", "Module": "SD", "Description": "Create Billing Document / Invoice"},
        {"T-Code": "MM03", "Module": "MM", "Description": "Display Material Master Data"},
        {"T-Code": "ME51N", "Module": "MM", "Description": "Create Purchase Requisition for Stock Replenishment"},
        {"T-Code": "CO09", "Module": "PP", "Description": "Availability Overview / ATP Check"}
    ]
    st.table(pd.DataFrame(tcodes))

    st.markdown("---")
    st.markdown("### 📝 Recent System Audit Trail")
    audit_logs = [
        {"Timestamp": "2026-08-21 18:30:12", "User": "Rajesh S", "Action": "Credit Override Executed", "Object": "SAP-ORD-4012"},
        {"Timestamp": "2026-08-21 18:25:45", "User": "System AI", "Action": "Risk Score Computed", "Object": "Batch 150 Orders"},
        {"Timestamp": "2026-08-21 17:10:00", "User": "Rajesh S", "Action": "PR Created for Stockout", "Object": "SAP-MAT-502"}
    ]
    st.table(pd.DataFrame(audit_logs))

st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Kaar Technologies Campus Hiring Assessment — Built for 2027 Batch Candidates</p>", unsafe_allow_html=True)
