# Utility Functions for Data Preprocessing and Feature Engineering
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from pathlib import Path
from config import DatasetConfig, FeatureConfig, TrainingConfig

class NetworkTrafficDataset(Dataset):
    """Custom PyTorch Dataset for network traffic data"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Feature matrix (numpy array or tensor)
            y: Labels (numpy array or tensor)
        """
        self.X = torch.FloatTensor(X) if isinstance(X, np.ndarray) else X
        self.y = torch.LongTensor(y) if isinstance(y, np.ndarray) else y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataPreprocessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, scaler=None, label_encoder=None):
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_columns = None
    
    def load_cicids_data(self, data_dir, files=None):
        """
        Load and combine CICIDS 2017 dataset files
        
        Args:
            data_dir: Directory containing CICIDS files
            files: List of filenames to load (if None, loads all available)
        Returns:
            Concatenated DataFrame
        """
        if files is None:
            files = DatasetConfig.cicids_files
        
        dfs = []
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                print(f"Loading {filename}...")
                try:
                    df = pd.read_csv(filepath)
                    dfs.append(df)
                    print(f"✓ Loaded {len(df)} samples from {filename}")
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
            else:
                print(f"⚠ File not found: {filepath}")
        
        if dfs:
            data = pd.concat(dfs, ignore_index=True)
            print(f"\nTotal samples loaded: {len(data)}")
            return data
        else:
            raise FileNotFoundError("No CICIDS data files found")
    
    def clean_data(self, df):
        """
        Clean dataset - remove duplicates, handle missing values
        
        Args:
            df: Input DataFrame
        Returns:
            Cleaned DataFrame
        """
        initial_size = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"Removed {initial_size - len(df)} duplicate rows")
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values - FIXED: Use proper pandas syntax
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # Fixed: Use proper pandas fillna without inplace
                df[col] = df[col].fillna(FeatureConfig.fill_nan_value)
        
        print(f"Final dataset size: {len(df)}")
        return df
    
    def extract_features(self, df):
        """
        Extract feature matrix and labels
        
        Args:
            df: Input DataFrame
        Returns:
            X: Feature matrix (numpy array)
            y: Labels (numpy array)
            feature_names: List of feature names
        """
        # Separate features and labels
        # CICIDS 2017 uses " Label" with leading space
        target_col = DatasetConfig.target_column
        
        if target_col not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            raise KeyError(f"Target column '{target_col}' not found in dataset")
        
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        
        # Remove unnecessary columns
        columns_to_drop = [col for col in FeatureConfig.exclude_features if col in X.columns]
        X = X.drop(columns=columns_to_drop)
        
        # Keep only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Features extracted: {X.shape[1]} features, {len(y)} samples")
        return X.values, y, numeric_cols
    
    def encode_labels(self, y, fit=True):
        """
        Encode class labels to binary (0 for BENIGN, 1 for ATTACK)
        
        Args:
            y: Raw labels
            fit: Whether to fit the encoder or use existing one
        Returns:
            Encoded labels
        """
        if fit:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(
                np.where(y == DatasetConfig.normal_label, "BENIGN", "ATTACK")
            )
            print(f"Label encoding: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        else:
            y_encoded = self.label_encoder.transform(
                np.where(y == DatasetConfig.normal_label, "BENIGN", "ATTACK")
            )
        
        return y_encoded
    
    def scale_features(self, X, fit=True):
        """
        Standardize features using StandardScaler
        
        Args:
            X: Feature matrix
            fit: Whether to fit scaler or use existing one
        Returns:
            Scaled feature matrix
        """
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            print(f"Scaler fitted on {X.shape[0]} samples")
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess(self, df, fit=True):
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers
        Returns:
            X: Preprocessed features
            y: Encoded labels
            info: Dictionary with preprocessing info
        """
        # Clean data
        df = self.clean_data(df)
        
        # Extract features
        X, y, feature_names = self.extract_features(df)
        
        # Encode labels
        y = self.encode_labels(y, fit=fit)
        
        # Scale features
        X = self.scale_features(X, fit=fit)
        
        info = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": feature_names,
            "class_distribution": {0: np.sum(y == 0), 1: np.sum(y == 1)}
        }
        
        return X, y, info


def get_class_weights(y):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        y: Label array
    Returns:
        Class weights tensor
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = torch.FloatTensor([total / (len(unique) * count) for count in counts])
    return weights


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate various classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, roc_curve
    )
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
        metrics["roc_curve"] = roc_curve(y_true, y_proba[:, 1])
    
    return metrics


def save_model(model, optimizer, scaler, label_encoder, save_path, epoch=None):
    """Save model checkpoint"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler": scaler,
        "label_encoder": label_encoder,
        "epoch": epoch
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(model, optimizer, load_path):
    """Load model checkpoint"""
    checkpoint = torch.load(load_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler = checkpoint["scaler"]
    label_encoder = checkpoint["label_encoder"]
    print(f"Model loaded from {load_path}")
    return model, optimizer, scaler, label_encoder