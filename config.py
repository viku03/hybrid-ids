import os
import torch
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Auto-detect best available device
def get_device():
    """Automatically detect the best available device"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("🔧 Using CPU (optimal for small models)")
        return "cpu"  # CPU is faster for small models like this (62K params)
    elif torch.cuda.is_available():
        print("🔧 Using CUDA (NVIDIA GPU)")
        return "cuda"
    else:
        print("🔧 Using CPU")
        return "cpu"

# Model Configuration - OPTIMIZED
class ModelConfig:
    input_size = 78  # CICIDS 2017 feature count
    hidden_sizes = [256, 128, 64]
    output_size = 2  # Binary classification (Normal/Attack)
    dropout_rate = 0.3
    learning_rate = 0.001
    weight_decay = 1e-5
    batch_size = 128  # INCREASED from 32 for speed (uses more memory but faster)
    epochs = 30  # REDUCED from 50 (early stopping will catch it anyway)
    early_stopping_patience = 5  # REDUCED from 10 (faster convergence check)
    device = "cpu"  # CPU is faster for small models with many samples
    
# Training Configuration - OPTIMIZED
class TrainingConfig:
    test_size = 0.2
    validation_size = 0.1
    random_state = 42
    num_workers = 4  # Parallel data loading (CPU-friendly)
    pin_memory = True  # Keep data in pinned memory for faster transfer
    stratified_split = True
    
# Dataset Configuration - FIXED FOR CICIDS 2017
class DatasetConfig:
    cicids_files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv"
    ]
    target_column = " Label"  # Leading space
    normal_label = "BENIGN"
    test_size = 0.2
    
# Feature Engineering
class FeatureConfig:
    exclude_features = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Bwd Packets"
    ]
    fill_nan_value = 0
    fill_inf_value = 0
    
# Streamlit Configuration
class StreamlitConfig:
    page_title = "Aegis - Network Intrusion Detection System"
    page_icon = "🛡️"
    layout = "wide"
    initial_sidebar_state = "expanded"
    max_file_size_mb = 1000
    
# API Configuration
class APIConfig:
    host = "0.0.0.0"
    port = 8000
    debug = False
    
def get_config(env: str = "development"):
    """Get configuration based on environment"""
    configs = {
        "development": {
            "model": ModelConfig,
            "training": TrainingConfig,
            "dataset": DatasetConfig,
            "feature": FeatureConfig,
            "streamlit": StreamlitConfig,
        },
        "production": {
            "model": ModelConfig,
            "training": TrainingConfig,
            "dataset": DatasetConfig,
            "feature": FeatureConfig,
            "streamlit": StreamlitConfig,
        }
    }
    return configs.get(env, configs["development"])
