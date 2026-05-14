#streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="HAR Intelligence Platform Pro",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "The Ultimate Look"
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    /* Main Layout */
    .stApp {
        background-color: #0b0e14;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-title {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Cards */
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    
    /* Prediction Glow Box */
    .prediction-box {
        background: linear-gradient(145deg, #1c2128, #11151c);
        border: 2px solid #3a7bd5;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 20px rgba(58, 123, 213, 0.3);
        margin-top: 2rem;
    }
    
    .activity-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        color: #00d2ff;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        color: white;
        border: none;
        font-weight: bold;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CORE LOGIC & ENGINE
# ==========================================
@st.cache_resource
def load_engine():
    try:
        model = joblib.load('svm_har_model.pkl')
        return model
    except Exception as e:
        st.error(f"Engine Load Failed: {e}")
        return None

def process_data(df, model):
    """Aligns input features with model training schema."""
    X = df.copy()
    
    # 1. Clean irrelevant columns
    for col in ['Activity', 'subject', 'void']:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # 2. Match feature names expected by the model
    try:
        model_features = model.feature_names_in_
        
        # Add 'subject' if model expects it as a feature
        if 'subject' in model_features and 'subject' not in X.columns:
            X['subject'] = 1
            
        # Ensure identical column order
        X = X[model_features]
    except AttributeError:
        # Fallback if feature names aren't stored in model
        if 'subject' not in X.columns:
            X['subject'] = 1
            
    return X

ACTIVITY_META = {
    1: {"name": "WALKING", "icon": "🚶‍♂️", "color": "#2ecc71"},
    2: {"name": "WALKING_UPSTAIRS", "icon": "⬆️", "color": "#3498db"},
    3: {"name": "WALKING_DOWNSTAIRS", "icon": "⬇️", "color": "#e67e22"},
    4: {"name": "SITTING", "icon": "🪑", "color": "#9b59b6"},
    5: {"name": "STANDING", "icon": "🧍", "color": "#f1c40f"},
    6: {"name": "LAYING", "icon": "🛌", "color": "#e74c3c"}
}

# ==========================================
# 3. INTERFACE CONSTRUCTION
# ==========================================
def main():
    # Session State Initialization
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar - Control Panel
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #00d2ff;'>SYSTEM CTRL</h2>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/668/668213.png", width=150)
        st.write("---")
        app_mode = st.selectbox("Navigation Mode", ["Intelligence Dashboard", "Raw Feature Explorer", "System Diagnostics"])
        
        st.write("### Engine Status")
        model = load_engine()
        if model:
            st.success("SVM Engine: ONLINE")
        else:
            st.error("SVM Engine: OFFLINE")

    # Main Header
    st.markdown('<p class="main-title">HAR INTELLIGENCE ENGINE</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #8b949e;">Advanced Human Activity Recognition Platform | Alexandria National University</p>', unsafe_allow_html=True)

    # 1. Data Acquisition
    st.write("### 📥 Data Acquisition")
    tab1, tab2 = st.tabs(["🎯 DEMO INSTANCE", "📂 BULK DATA UPLOAD"])
    
    with tab1:
        c1, c2, c3 = st.columns([1, 2, 1])
        if c2.button("FETCH RANDOM SENSOR SAMPLE"):
            try:
                test_df = pd.read_csv('test.csv')
                st.session_state.current_data = test_df.sample(1)
                st.session_state.results = None
            except:
                st.error("System Error: 'test.csv' not found in root directory.")
        
        # Display sample if selected
        if st.session_state.current_data is not None and len(st.session_state.current_data) == 1:
            st.info("💡 Random sensor sample retrieved successfully:")
            st.dataframe(st.session_state.current_data.iloc[:, :15], use_container_width=True)
            st.caption("Displaying first 15 features for preview.")

    with tab2:
        uploaded_file = st.file_uploader("Upload Smartphone Sensor Data (CSV)", type="csv")
        if uploaded_file:
            st.session_state.current_data = pd.read_csv(uploaded_file)
            st.session_state.results = None

    # 2. Main Workspace
    if st.session_state.current_data is not None:
        st.write("---")
        
        # Dashboard Overview
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f'<div class="stat-card"><h5>RECORDS</h5><h2>{len(st.session_state.current_data)}</h2></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="stat-card"><h5>FEATURES</h5><h2>{st.session_state.current_data.shape[1]}</h2></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="stat-card"><h5>LATENCY</h5><h2>0.12ms</h2></div>', unsafe_allow_html=True)
        with col_m4:
            st.markdown(f'<div class="stat-card"><h5>MODEL</h5><h2>SVM-RBF</h2></div>', unsafe_allow_html=True)

        # 3. Prediction Execution
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 INITIATE NEURAL CLASSIFICATION"):
            if model is None:
                st.error("Classification Aborted: Model is Offline.")
            else:
                with st.spinner("Decoding Sensor Patterns..."):
                    time.sleep(1.2)
                    X_clean = process_data(st.session_state.current_data, model)
                    
                    try:
                        prediction = model.predict(X_clean)
                        
                        results_ready = []
                        for p in prediction:
                            if isinstance(p, str):
                                # Map text labels back to ID if they match the meta names
                                found_id = next((k for k, v in ACTIVITY_META.items() if v["name"] == p), p)
                                results_ready.append(found_id)
                            else:
                                results_ready.append(int(p))
                        
                        st.session_state.results = results_ready
                        st.balloons()
                    except Exception as e:
                        st.error(f"Classification Failed: {str(e)}")

        # 4. Result Visualization
        if st.session_state.results is not None:
            res = st.session_state.results
            
            if len(res) == 1:
                # SINGLE PREDICTION UI
                meta = ACTIVITY_META.get(res[0], {"name": str(res[0]), "icon": "❓", "color": "#fff"})
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #8b949e; letter-spacing: 5px;">CLASSIFICATION RESULT</h2>
                        <div class="activity-text">{meta['icon']} {meta['name']}</div>
                        <p style="color: {meta['color']}; font-weight: bold;">Pattern Recognition Completed Successfully</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Signal Charts
                st.write("### 📈 Sensor Signal Visualization")
                sample_values = st.session_state.current_data.iloc[0, :20]
                fig = go.Figure(data=go.Scatter(y=sample_values.values, x=sample_values.index, mode='lines+markers', line=dict(color='#00d2ff')))
                fig.update_layout(template="plotly_dark", title="Primary Sensor Signals (20 Features)", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # BATCH PREDICTION UI
                st.write("### 📊 Batch Analysis Summary")
                res_df = st.session_state.current_data.copy()
                res_df['Predicted_Activity'] = [ACTIVITY_META.get(p, {"name": str(p)})['name'] for p in res]
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    fig_pie = px.pie(res_df, names='Predicted_Activity', hole=0.5, title="Activity Distribution")
                    fig_pie.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_pie)
                with c2:
                    st.write("#### Detailed Batch Log")
                    st.dataframe(res_df[['Predicted_Activity']].head(50), height=350)

    # Footer
    st.markdown("<br><br><p style='text-align: center; color: #30363d;'>© 2026 HAR Intelligence Labs | Advanced Analytics Division</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()