import streamlit as pd_streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import local modules
from data_generator import generate_synthetic_data
from train_model import train_fraud_detector

# Set page configuration for a premium, wide layout
st.set_page_config(
    page_title="GuardianAI | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-mode premium branding, glassmorphism, and sleek styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .stApp {
        background-color: #0e1117;
    }
    /* Brand Header styling */
    .brand-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .brand-subtitle {
        font-size: 1.1rem;
        color: #8892b0;
        margin-bottom: 2rem;
    }
    /* Metric Card Styling */
    .kpi-card {
        background: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.075);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
        margin: 0.5rem 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Alert Status Cards */
    .status-card-safe {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .status-card-fraud {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load model and preprocessing assets
@st.cache_resource
def load_model_assets(assets_path='model_assets.pkl'):
    if not os.path.exists(assets_path):
        with st.spinner("Model assets not found. Training model automatically on synthetic data..."):
            train_fraud_detector(model_output_path=assets_path)
    
    with open(assets_path, 'rb') as f:
        return pickle.load(f)

# Helper: Save custom data back to CSV
def save_dataset(df, path='credit_card_transactions.csv'):
    df.to_csv(path, index=False)

# Load assets
assets = load_model_assets()
model = assets['model']
preprocessor = assets['preprocessor']
feature_names = assets['feature_names']
metrics = assets['metrics']
feature_importances = assets['feature_importances']

# Side bar logo and details
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-top: 1rem;'><h2 style='color:#00f2fe; margin-bottom: 0;'>🛡️ GuardianAI</h2><p style='color:#8892b0; font-size:0.9rem;'>AI-Powered Fraud Prevention</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Model Status")
    st.success("Random Forest Model: Active")
    st.info(f"Training dataset size: 50,000")
    
    st.subheader("Quick Stats")
    st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
    st.metric("Precision (Fraud)", f"{metrics['precision_fraud']*100:.1f}%")
    st.metric("Recall (Fraud)", f"{metrics['recall_fraud']*100:.1f}%")
    
    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem; color:#8892b0; text-align:center;'>Developer Lab | AIML Project 2026</div>", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='brand-title'>Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>An interactive dashboard demonstrating synthetic data generation, exploratory analysis, machine learning training, real-time prediction, and streaming simulation.</p>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_prediction, tab_simulator, tab_how_it_works = st.tabs([
    "📊 System Overview", 
    "🔍 Real-Time Predictor", 
    "⚡ Live Streaming Simulator", 
    "📖 AI/ML Explainer"
])

# ----------------- TAB 1: SYSTEM OVERVIEW -----------------
with tab_overview:
    st.header("System & Dataset Overview")
    
    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Load dataset to show statistics
    df_dataset = pd.read_csv('credit_card_transactions.csv')
    total_trans = len(df_dataset)
    fraud_trans = df_dataset['is_fraud'].sum()
    fraud_rate = (fraud_trans / total_trans) * 100
    avg_amount = df_dataset['purchase_amount'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-value">{total_trans:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #ef4444;">{fraud_trans:,}</div>
            <div class="kpi-label">Fraudulent Blocked</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #f59e0b;">{fraud_rate:.3f}%</div>
            <div class="kpi-label">Class Imbalance Ratio</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #10b981;">${avg_amount:.2f}</div>
            <div class="kpi-label">Average Transaction Value</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Section
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Transaction Amounts by Category & Legitimacy")
        # Sample data for fast plotting
        sample_df = df_dataset.sample(n=min(5000, len(df_dataset)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="distance_from_home",
            y="purchase_amount",
            color=sample_df['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'}),
            color_discrete_map={'Legitimate': '#10b981', 'Fraud': '#ef4444'},
            labels={"distance_from_home": "Distance from Home (km)", "purchase_amount": "Purchase Amount ($)"},
            title="Distribution of Amount vs Distance",
            opacity=0.6,
            template="plotly_dark"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_chart2:
        st.subheader("Fraud Rate by Category")
        fraud_by_cat = df_dataset.groupby('category')['is_fraud'].mean().reset_index()
        fraud_by_cat['is_fraud'] *= 100  # Convert to percentage
        fraud_by_cat = fraud_by_cat.sort_values(by='is_fraud', ascending=True)
        
        fig_bar = px.bar(
            fraud_by_cat,
            x="is_fraud",
            y="category",
            orientation="h",
            labels={"is_fraud": "Fraud Rate (%)", "category": "Transaction Category"},
            title="Percentage of Fraudulent Transactions per Category",
            template="plotly_dark",
            color_discrete_sequence=['#4facfe']
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Sample Transactions (Scrollable Database)")
    st.dataframe(
        df_dataset.head(100).style.format({
            'distance_from_home': '{:.2f} km',
            'purchase_amount': '${:.2f}',
            'is_fraud': lambda x: '⚠️ FRAUD' if x == 1 else '✅ Safe'
        }),
        use_container_width=True
    )

# ----------------- TAB 2: REAL-TIME PREDICTOR -----------------
with tab_prediction:
    st.header("Real-Time Fraud Risk Assessor")
    st.markdown("Adjust the sliders below to simulate a single credit card transaction. The machine learning model will analyze the features in real-time and provide a risk percentage.")
    
    col_input, col_result = st.columns([3, 2])
    
    with col_input:
        st.subheader("Transaction Parameters")
        
        # Split inputs into two nested columns
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            purchase_amount = st.slider("Purchase Amount ($)", min_value=1.0, max_value=5000.0, value=75.0, step=0.5)
            distance_from_home = st.slider("Distance from Home (km)", min_value=0.1, max_value=2000.0, value=12.5, step=0.1)
            hour_of_day = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=14)
            category = st.selectbox("Transaction Category", options=['grocery', 'entertainment', 'food_dining', 'gas_transport', 'online_shopping', 'travel_lodging', 'electronics'])
            
        with subcol2:
            online_order = st.toggle("Is this an Online Order?", value=False)
            
            # Use Chip and Use PIN are physically mutually exclusive if it's online
            if online_order:
                use_chip = st.toggle("Use Chip (Disabled for Online)", value=False, disabled=True)
                use_pin = st.toggle("Use PIN (Disabled for Online)", value=False, disabled=True)
            else:
                use_chip = st.toggle("Use Card Chip?", value=True)
                use_pin = st.toggle("Enter Card PIN?", value=False)
                
        # Prepare input data dictionary
        input_data = {
            'distance_from_home': distance_from_home,
            'purchase_amount': purchase_amount,
            'category': category,
            'online_order': 1 if online_order else 0,
            'use_chip': 1 if use_chip else 0,
            'use_pin': 1 if use_pin else 0,
            'hour_of_day': hour_of_day
        }
        
    with col_result:
        st.subheader("Model Decision & Insights")
        
        # Build DataFrame from inputs
        input_df = pd.DataFrame([input_data])
        
        # Run inference
        input_processed = preprocessor.transform(input_df)
        fraud_probability = model.predict_proba(input_processed)[0, 1]
        prediction = model.predict(input_processed)[0]
        
        # Probability Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk Score (%)", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#8892b0"},
                'bar': {'color': "#ef4444" if fraud_probability > 0.5 else "#10b981"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e0e6ed", 'family': "Inter"},
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Dynamic response cards
        if prediction == 1:
            st.markdown(f"""
            <div class="status-card-fraud">
                <h4 style="color: #ef4444; margin-top:0;">⚠️ High Fraud Probability Alert</h4>
                <p style="margin: 0; font-size: 0.95rem;">
                    This transaction triggers several risk patterns. Our recommendation is to <strong>BLOCK</strong> this transaction and request verification from the customer.
                </p>
                <ul style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #fecaca;">
                    <li>Risk Probability: <strong>{fraud_probability * 100:.2f}%</strong></li>
                    <li>Decision: Decline & Flag</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-safe">
                <h4 style="color: #10b981; margin-top:0;">✅ Legitimate Transaction Approved</h4>
                <p style="margin: 0; font-size: 0.95rem;">
                    The transaction behavior matches standard customer spend profiles. It is cleared as <strong>SAFE</strong>.
                </p>
                <ul style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #d1fae5;">
                    <li>Risk Probability: <strong>{fraud_probability * 100:.2f}%</strong></li>
                    <li>Decision: Approve</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 3: LIVE STREAMING SIMULATOR -----------------
with tab_simulator:
    st.header("Live Credit Card Transaction Stream Simulator")
    st.markdown("Simulate a live stream of transactions hitting your gateway in real-time. Watch how GuardianAI screens each one and updates metrics instantly.")
    
    col_sim_ctrl, col_sim_stats = st.columns([1, 3])
    
    # Store state in session
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
    if 'sim_data' not in st.session_state:
        st.session_state.sim_data = pd.DataFrame(columns=[
            'timestamp', 'purchase_amount', 'distance_from_home', 'category', 'online_order', 'predicted_fraud', 'risk_score'
        ])
    
    with col_sim_ctrl:
        st.subheader("Simulator Settings")
        speed = st.slider("Simulation Speed (seconds/trans)", min_value=0.2, max_value=2.0, value=0.5, step=0.1)
        
        # Start and Stop Buttons
        if not st.session_state.sim_running:
            if st.button("Start Live Stream 🚀", key="btn_start_sim", type="primary", use_container_width=True):
                st.session_state.sim_running = True
                st.rerun()
        else:
            if st.button("Stop Live Stream 🛑", key="btn_stop_sim", type="secondary", use_container_width=True):
                st.session_state.sim_running = False
                st.rerun()
                
        if st.button("Clear Stream Data 🧹", key="btn_clear_sim", use_container_width=True):
            st.session_state.sim_data = pd.DataFrame(columns=[
                'timestamp', 'purchase_amount', 'distance_from_home', 'category', 'online_order', 'predicted_fraud', 'risk_score'
            ])
            st.rerun()
            
    with col_sim_stats:
        # Dynamic stats elements
        sim_stats_col1, sim_stats_col2, sim_stats_col3 = st.columns(3)
        placeholder_sc1 = sim_stats_col1.empty()
        placeholder_sc2 = sim_stats_col2.empty()
        placeholder_sc3 = sim_stats_col3.empty()
        
    st.markdown("---")
    
    # Live updating containers
    col_table, col_live_chart = st.columns([3, 2])
    
    with col_table:
        st.subheader("Live Transaction Ledger")
        table_placeholder = st.empty()
        
    with col_live_chart:
        st.subheader("Threat Ratio Over Time")
        chart_placeholder = st.empty()
        
    # Active simulation loop
    if st.session_state.sim_running:
        while st.session_state.sim_running:
            # Generate single transaction
            single_tx = generate_synthetic_data(1)
            
            # Run prediction
            tx_processed = preprocessor.transform(single_tx.drop(columns=['is_fraud']))
            risk_score = model.predict_proba(tx_processed)[0, 1]
            pred_fraud = int(model.predict(tx_processed)[0])
            
            # Construct row
            new_row = pd.DataFrame([{
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'purchase_amount': float(single_tx['purchase_amount'].iloc[0]),
                'distance_from_home': float(single_tx['distance_from_home'].iloc[0]),
                'category': single_tx['category'].iloc[0],
                'online_order': 'Yes' if single_tx['online_order'].iloc[0] == 1 else 'No',
                'predicted_fraud': pred_fraud,
                'risk_score': float(risk_score)
            }])
            
            # Add row to session data, keep only last 50 transactions to prevent memory overload
            st.session_state.sim_data = pd.concat([new_row, st.session_state.sim_data], ignore_index=True).head(50)
            
            # Update metric cards
            tot = len(st.session_state.sim_data)
            frauds = int(st.session_state.sim_data['predicted_fraud'].sum())
            fr_rate = (frauds / tot * 100) if tot > 0 else 0.0
            
            placeholder_sc1.metric("Screened Transactions", f"{tot}")
            placeholder_sc2.metric("Fraud Detected", f"{frauds}", delta=f"{frauds} flagged", delta_color="inverse")
            placeholder_sc3.metric("Live Threat Rate", f"{fr_rate:.1f}%")
            
            # Display ledger styled dataframe
            def color_fraud_row(row):
                return ['background-color: rgba(239, 68, 68, 0.25); border-left: 5px solid #ef4444;' if row['predicted_fraud'] == 1 else '' for _ in row]
            
            display_df = st.session_state.sim_data.copy()
            # Style the table
            table_placeholder.dataframe(
                display_df.style.apply(color_fraud_row, axis=1).format({
                    'purchase_amount': '${:.2f}',
                    'distance_from_home': '{:.1f} km',
                    'risk_score': '{:.1%}',
                    'predicted_fraud': lambda x: '⚠️ FRAUD' if x == 1 else '✅ Safe'
                }),
                use_container_width=True,
                height=400
            )
            
            # Display threat ratio chart
            if tot > 1:
                # Reverse dataframe to show chronological order left-to-right
                chart_df = st.session_state.sim_data.iloc[::-1].reset_index()
                chart_df['cumulative_fraud'] = chart_df['predicted_fraud'].cumsum()
                
                fig_line = px.line(
                    chart_df,
                    x='timestamp',
                    y='cumulative_fraud',
                    labels={'timestamp': 'Timestamp', 'cumulative_fraud': 'Total Blocked Frauds'},
                    title="Blocked Fraud Cumulative Trend",
                    template="plotly_dark",
                    markers=True
                )
                fig_line.update_traces(line_color='#ef4444', marker_size=6)
                fig_line.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=300
                )
                chart_placeholder.plotly_chart(fig_line, use_container_width=True)
                
            time.sleep(speed)
            
    else:
        # Static display when simulator is idle
        if len(st.session_state.sim_data) > 0:
            tot = len(st.session_state.sim_data)
            frauds = int(st.session_state.sim_data['predicted_fraud'].sum())
            fr_rate = (frauds / tot * 100) if tot > 0 else 0.0
            
            placeholder_sc1.metric("Screened Transactions", f"{tot}")
            placeholder_sc2.metric("Fraud Detected", f"{frauds}", delta=f"{frauds} flagged", delta_color="inverse")
            placeholder_sc3.metric("Live Threat Rate", f"{fr_rate:.1f}%")
            
            def color_fraud_row(row):
                return ['background-color: rgba(239, 68, 68, 0.25); border-left: 5px solid #ef4444;' if row['predicted_fraud'] == 1 else '' for _ in row]
            
            table_placeholder.dataframe(
                st.session_state.sim_data.style.apply(color_fraud_row, axis=1).format({
                    'purchase_amount': '${:.2f}',
                    'distance_from_home': '{:.1f} km',
                    'risk_score': '{:.1%}',
                    'predicted_fraud': lambda x: '⚠️ FRAUD' if x == 1 else '✅ Safe'
                }),
                use_container_width=True,
                height=400
            )
            
            chart_df = st.session_state.sim_data.iloc[::-1].reset_index()
            chart_df['cumulative_fraud'] = chart_df['predicted_fraud'].cumsum()
            fig_line = px.line(
                chart_df,
                x='timestamp',
                y='cumulative_fraud',
                labels={'timestamp': 'Timestamp', 'cumulative_fraud': 'Total Blocked Frauds'},
                title="Blocked Fraud Cumulative Trend",
                template="plotly_dark",
                markers=True
            )
            fig_line.update_traces(line_color='#ef4444')
            chart_placeholder.plotly_chart(fig_line, use_container_width=True)
        else:
            placeholder_sc1.metric("Screened Transactions", "0")
            placeholder_sc2.metric("Fraud Detected", "0")
            placeholder_sc3.metric("Live Threat Rate", "0.0%")
            table_placeholder.info("Click 'Start Live Stream' on the sidebar to begin receiving live credit card transactions.")

# ----------------- TAB 4: HOW IT WORKS (AI/ML EXPLAINER) -----------------
with tab_how_it_works:
    st.header("How the Fraud Detection System Works")
    
    st.markdown("""
    This project is a demonstration of how Financial Institutions leverage Machine Learning to intercept credit card theft and fraudulent behavior.
    Below is a breakdown of the standard AI/ML workflow used in this application.
    """)
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.subheader("1. Feature Engineering")
        st.markdown("""
        Raw transaction logs contain timestamps, account IDs, merchant names, and amounts. To train an AI, we must engineer **behavioral features**:
        *   **Distance from Home**: Calculated using customer billing address coordinates and the transaction terminal coordinates. Huge distances indicate potential card cloning or theft.
        *   **Online Order**: E-commerce transactions bypass card chip security, carrying a significantly higher base probability of fraud.
        *   **PIN vs. Signature**: Card-Present transactions where a PIN is entered are highly secure. Physical transactions with magnetic stripe swipes or signatures are high-risk.
        *   **Hour of Day**: Transactions occurring between 1 AM and 5 AM are less common and more typical of automated card draining or late-night fraud sprees.
        """)
        
        st.subheader("2. Handling Class Imbalance")
        st.markdown("""
        In real credit card transaction logs, fraud accounts for less than **0.2%** of all data. 
        *   **The Trap**: A standard machine learning model trying to maximize *Accuracy* will simply guess "Safe" for every single transaction, yielding a **99.8% Accuracy**, but catching **0% of fraud**.
        *   **The Solution**: We configure our Random Forest algorithm to use `class_weight='balanced'`. This tells the model to penalize errors on fraudulent examples significantly more, forcing it to prioritize finding the minority fraud class.
        """)
        
    with exp_col2:
        st.subheader("3. Choosing the Model (Random Forest)")
        st.markdown("""
        We use a **Random Forest Classifier**, an ensemble model composed of 100 individual **Decision Trees**. 
        *   Each tree analyzes random subsets of data and features.
        *   During inference, all 100 trees vote on whether the transaction is "Safe" or "Fraud".
        *   This voting system reduces overfitting and makes predictions highly robust.
        """)
        
        # Display Feature Importance Chart dynamically
        st.subheader("Feature Importance Analysis")
        st.markdown("Feature importances represent how much weight the model puts on each feature to arrive at its final fraud decision:")
        
        fi_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
        # Sort values
        fi_df = fi_df.sort_values(by='Importance', ascending=True)
        
        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "Gini Importance Score", "Feature": "Input Feature"},
            template="plotly_dark",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        
    st.markdown("---")
    
    st.subheader("4. Understanding Metrics: Precision vs. Recall")
    metric_explanation_col1, metric_explanation_col2 = st.columns(2)
    
    with metric_explanation_col1:
        st.markdown(f"""
        #### **Precision ({metrics['precision_fraud']*100:.1f}%)**
        *   **Definition**: Out of all transactions flagged as **Fraud**, how many were actually fraud?
        *   **Low Precision Impact**: Too many "false alarms" (False Positives). Legitimate customers get their cards blocked at supermarkets, leading to frustration.
        """)
        
    with metric_explanation_col2:
        st.markdown(f"""
        #### **Recall ({metrics['recall_fraud']*100:.1f}%)**
        *   **Definition**: Out of all actual **Fraud** transactions, how many did the system catch?
        *   **Low Recall Impact**: The system misses fraud (False Negatives), letting stolen card purchases slide through. Financial institutions lose millions of dollars.
        """)
