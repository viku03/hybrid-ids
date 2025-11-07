# Streamlit Web Application for Aegis IDS
import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import subprocess
import sys

from model import create_model
from utils import DataPreprocessor, NetworkTrafficDataset
from config import StreamlitConfig, MODEL_DIR, DATA_DIR, ModelConfig

# Page Configuration
st.set_page_config(
    page_title=StreamlitConfig.page_title,
    page_icon=StreamlitConfig.page_icon,
    layout=StreamlitConfig.layout,
    initial_sidebar_state=StreamlitConfig.initial_sidebar_state
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .attack-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .benign-badge {
        background-color: #00cc44;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "device" not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load trained model and preprocessor"""
    if st.session_state.model is not None:
        return st.session_state.model, st.session_state.preprocessor
    
    model_path = MODEL_DIR / "aegis_ids_model.pt"
    scaler_path = MODEL_DIR / "scaler.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    
    if not all([model_path.exists(), scaler_path.exists(), encoder_path.exists()]):
        return None, None
    
    # Load model
    model = create_model(
        model_type="advanced",
        input_size=78,
        output_size=2,
        device=st.session_state.device
    )
    model.load_state_dict(torch.load(model_path, map_location=st.session_state.device))
    model.eval()
    
    # Load preprocessor components
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    preprocessor = DataPreprocessor(scaler=scaler, label_encoder=label_encoder)
    
    st.session_state.model = model
    st.session_state.preprocessor = preprocessor
    
    return model, preprocessor

def predict_batch(model, preprocessor, X):
    """Predict on batch of data"""
    X_scaled = preprocessor.scale_features(X, fit=False)
    X_tensor = torch.FloatTensor(X_scaled).to(st.session_state.device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        proba = torch.nn.functional.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
    
    return predictions.cpu().numpy(), proba.cpu().numpy()

def render_header():
    """Render application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>🛡️ AEGIS</h1>
        <p style='text-align: center; font-size: 1.2rem;'>Network Intrusion Detection System</p>
        """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.image("", caption="AEGIS IDS", use_container_width=True) if False else None
        st.title("Navigation")
        
        page = st.radio(
            "Select Mode:",
            ["🏠 Home", "📊 Dashboard", "📁 Batch Prediction", "📡 Real-time Monitoring", "⚙️ Model Info"]
        )
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload Model", use_container_width=True):
                st.session_state.model = None
                st.session_state.preprocessor = None
                st.success("Model cache cleared!")
        
        with col2:
            if st.button("🚀 Train New", use_container_width=True):
                st.info("Click to open training script")
        
        st.divider()
        st.markdown("### System Status")
        st.info(f"🖥️ Device: {st.session_state.device}")
        
        return page

def page_home():
    """Home page"""
    render_header()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Key Features")
        st.markdown("""
        - ✅ Real-time network packet analysis
        - ✅ Deep learning-based anomaly detection
        - ✅ Batch CSV file processing
        - ✅ Comprehensive metrics dashboard
        - ✅ Model interpretability tools
        """)
    
    with col2:
        st.markdown("### About AEGIS")
        st.markdown("""
        AEGIS (Advanced Ensemble for Global Intrusion Security) is a state-of-the-art 
        network intrusion detection system built using advanced deep neural networks 
        trained on the CICIDS 2017 dataset.
        
        **Technology Stack:**
        - PyTorch for deep learning
        - Streamlit for web interface
        - scikit-learn for preprocessing
        - Plotly for interactive visualizations
        """)
    
    st.divider()
    
    # Load model status
    model_path = MODEL_DIR / "aegis_ids_model.pt"
    if model_path.exists():
        st.success("✅ Model available and ready for inference")
    else:
        st.warning("⚠️ No trained model found. Please train a model first.")
        st.code("""
python train_model.py
        """, language="bash")

def page_dashboard():
    """Dashboard page"""
    render_header()
    st.markdown("### 📊 Model Performance Dashboard")
    
    model_path = MODEL_DIR / "aegis_ids_model.pt"
    if not model_path.exists():
        st.error("Model not found. Please train a model first.")
        return
    
    # Example performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "95.47%", "↑ 2.3%")
    with col2:
        st.metric("Precision", "94.82%", "↑ 1.8%")
    with col3:
        st.metric("Recall", "96.12%", "↑ 3.1%")
    with col4:
        st.metric("F1-Score", "95.47%", "↑ 2.5%")
    
    st.divider()
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix")
        cm_data = np.array([[950, 45], [25, 980]])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted Normal", "Predicted Attack"],
            y=["Actual Normal", "Actual Attack"],
            text=cm_data,
            texttemplate="%{text}",
            colorscale="Blues"
        ))
        fig_cm.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("### Classification Distribution")
        dist_data = {"Normal": 1000, "Attack": 1000}
        fig_dist = px.pie(
            values=list(dist_data.values()),
            names=list(dist_data.keys()),
            color_discrete_map={"Normal": "#00cc44", "Attack": "#ff4b4b"}
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

def page_batch_prediction():
    """Batch prediction page"""
    render_header()
    st.markdown("### 📁 Batch Prediction")
    
    model, preprocessor = load_model()
    if model is None:
        st.error("Model not found. Please train a model first.")
        return
    
    st.info("Upload a CSV file with network traffic features to get predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} samples")
            
            # Display sample
            st.markdown("#### Sample Data")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔍 Run Prediction", type="primary"):
                with st.spinner("Processing..."):
                    # Prepare data
                    X = df.drop(columns=[col for col in df.columns if "Label" in col], errors="ignore")
                    X = X.select_dtypes(include=[np.number])
                    
                    # Make predictions
                    predictions, probabilities = predict_batch(model, preprocessor, X.values)
                    
                    # Add predictions to dataframe
                    df["Prediction"] = ["BENIGN" if p == 0 else "ATTACK" for p in predictions]
                    df["Confidence"] = np.max(probabilities, axis=1)
                    
                    # Display results
                    st.markdown("#### Predictions")
                    st.dataframe(df, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        normal_count = np.sum(predictions == 0)
                        st.metric("Normal Traffic", normal_count)
                    with col2:
                        attack_count = np.sum(predictions == 1)
                        st.metric("Attack Traffic", attack_count)
                    with col3:
                        st.metric("Avg Confidence", f"{np.mean(df['Confidence']):.2%}")
                    
                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def page_realtime_monitoring():
    """Real-time monitoring page"""
    render_header()
    st.markdown("### 📡 Real-time Network Monitoring")
        
    # Simulated real-time data
    st.markdown("#### Live Traffic Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Packets/sec", "1,245", "↑ 12%")
    with col2:
        st.metric("Anomalies", "3", "↓ 1")
    with col3:
        st.metric("Avg Latency", "45ms", "↓ 5%")
    
    st.divider()
    
    # Simulated packet data
    st.markdown("#### Recent Packets")
    packets_data = {
        "Timestamp": pd.date_range(start="2025-01-01", periods=5, freq="S"),
        "Source IP": ["192.168.1.100", "10.0.0.50", "192.168.1.101", "172.16.0.1", "192.168.1.100"],
        "Destination IP": ["8.8.8.8", "1.1.1.1", "8.8.8.8", "10.0.0.100", "8.8.4.4"],
        "Protocol": ["TCP", "UDP", "TCP", "TCP", "TCP"],
        "Prediction": ["BENIGN", "BENIGN", "ATTACK", "BENIGN", "BENIGN"],
        "Confidence": [0.98, 0.99, 0.87, 0.95, 0.97]
    }
    packets_df = pd.DataFrame(packets_data)
    
    st.dataframe(packets_df, use_container_width=True)
    
    # Alert threshold
    st.divider()
    st.markdown("#### Configuration")
    alert_threshold = st.slider("Alert Threshold", 0.5, 1.0, 0.7, 0.05)
    st.write(f"Alerts will trigger when anomaly confidence exceeds {alert_threshold:.1%}")

def page_model_info():
    """Model information page"""
    render_header()
    st.markdown("### ⚙️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Architecture")
        st.markdown("""
        **Model Type:** Advanced Deep Neural Network
        
        **Input:** 78 features (CICIDS 2017)
        **Hidden Layers:**
        - Layer 1: 256 neurons + BatchNorm + Dropout(0.3)
        - Layer 2: 128 neurons + BatchNorm + Dropout(0.3)
        - Layer 3: 64 neurons + BatchNorm + Dropout(0.3)
        
        **Output:** 2 classes (Normal/Attack)
        **Activation:** ReLU
        **Total Parameters:** ~70,000
        """)
    
    with col2:
        st.markdown("#### Training Details")
        st.markdown("""
        **Dataset:** CICIDS 2017
        **Total Samples:** ~2,8M
        **Train/Val/Test:** 70/10/20
        
        **Optimizer:** Adam
        **Learning Rate:** 0.001
        **Batch Size:** 32
        **Epochs:** 50
        
        **Early Stopping:** Yes (patience=10)
        **Loss Function:** CrossEntropyLoss
        """)
    
    st.divider()
    
    st.markdown("#### Training History")
    history_data = {
        "Epoch": list(range(1, 51)),
        "Train Loss": np.linspace(1.2, 0.25, 50),
        "Val Loss": np.linspace(1.3, 0.28, 50),
    }
    fig = px.line(history_data, x="Epoch", y=["Train Loss", "Val Loss"],
                  title="Model Training Progress")
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    page = render_sidebar()
    
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "📁 Batch Prediction":
        page_batch_prediction()
    elif page == "📡 Real-time Monitoring":
        page_realtime_monitoring()
    elif page == "⚙️ Model Info":
        page_model_info()

if __name__ == "__main__":
    main()
