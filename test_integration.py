# Integration Tests
import pytest
import numpy as np
import torch
import pandas as pd
import tempfile
import os
from pathlib import Path

from model import create_model
from utils import DataPreprocessor, NetworkTrafficDataset
from train_model import Trainer
from torch.utils.data import DataLoader

@pytest.fixture
def sample_cicids_data():
    """Create sample CICIDS-like data"""
    n_samples = 200
    n_features = 78
    
    # Create feature data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    X = np.abs(X) * 100  # Make values positive like network features
    
    # Create labels
    y = np.random.choice([0, 1], n_samples)
    
    return X, y

@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

class TestDataPipeline:
    """Test complete data pipeline"""
    
    def test_preprocessor_fit_transform(self, sample_cicids_data):
        """Test preprocessor fit and transform"""
        X, y = sample_cicids_data
        
        preprocessor = DataPreprocessor()
        
        # Fit scaler
        X_scaled = preprocessor.scale_features(X, fit=True)
        
        # Check scaling
        assert X_scaled.shape == X.shape
        assert np.abs(X_scaled.mean()) < 1e-5
        assert np.abs(X_scaled.std() - 1.0) < 1e-5
    
    def test_preprocessor_transform_only(self, sample_cicids_data):
        """Test preprocessor transform without fitting"""
        X, _ = sample_cicids_data
        X_train = X[:100]
        X_test = X[100:]
        
        preprocessor = DataPreprocessor()
        
        # Fit on train
        preprocessor.scale_features(X_train, fit=True)
        
        # Transform test
        X_test_scaled = preprocessor.scale_features(X_test, fit=False)
        
        assert X_test_scaled.shape == X_test.shape

class TestModelTraining:
    """Test model training pipeline"""
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized"""
        model = create_model("advanced")
        trainer = Trainer(model)
        
        assert trainer.model is not None
        assert trainer.best_val_loss == float("inf")
    
    def test_single_batch_training(self, sample_cicids_data):
        """Test training on single batch"""
        X, y = sample_cicids_data
        
        # Create small dataset
        dataset = NetworkTrafficDataset(X[:50], y[:50])
        loader = DataLoader(dataset, batch_size=10)
        
        model = create_model("advanced")
        trainer = Trainer(model, device="cpu")
        
        # Train one batch
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        train_loss, train_acc = trainer.train_epoch(loader, criterion, optimizer)
        
        assert train_loss > 0
        assert 0 <= train_acc <= 100
    
    def test_validation_step(self, sample_cicids_data):
        """Test validation step"""
        X, y = sample_cicids_data
        
        # Create dataset
        dataset = NetworkTrafficDataset(X[:50], y[:50])
        loader = DataLoader(dataset, batch_size=10)
        
        model = create_model("advanced")
        trainer = Trainer(model, device="cpu")
        
        criterion = torch.nn.CrossEntropyLoss()
        val_loss, val_acc, preds, labels = trainer.validate(loader, criterion)
        
        assert val_loss > 0
        assert 0 <= val_acc <= 100
        assert len(preds) == 50
        assert len(labels) == 50

class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_save_load(self, temp_model_dir):
        """Test saving and loading model - FIXED VERSION"""
        # Create model in eval mode
        model = create_model("advanced")
        model.eval()  # Set to eval mode
        
        model_path = temp_model_dir / "test_model.pt"
        
        # Save model
        torch.save(model.state_dict(), model_path)
        assert model_path.exists()
        
        # Load model
        new_model = create_model("advanced")
        new_model.eval()  # Set to eval mode
        new_model.load_state_dict(torch.load(model_path, weights_only=False))
        
        # Test models produce same output with SAME seed
        torch.manual_seed(42)
        X = torch.randn(10, 78)
        
        with torch.no_grad():
            output1 = model(X)
            output2 = new_model(X)
        
        # Use approximate equality for neural networks (BatchNorm variance)
        assert torch.allclose(output1, output2, atol=1e-5)

class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_training_pipeline(self, sample_cicids_data, temp_model_dir):
        """Test complete training pipeline"""
        X, y = sample_cicids_data
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = NetworkTrafficDataset(X_train, y_train)
        val_dataset = NetworkTrafficDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Create model
        model = create_model("advanced")
        trainer = Trainer(model, device="cpu")
        
        # Quick training
        history = trainer.fit(train_loader, val_loader, epochs=2)
        
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
    
    def test_prediction_pipeline(self, sample_cicids_data):
        """Test complete prediction pipeline"""
        X, y = sample_cicids_data
        
        # Prepare data
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.scale_features(X, fit=True)
        
        # Create model
        model = create_model("advanced")
        model.eval()
        
        # Make predictions
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            predictions = model.predict(X_tensor)
            probabilities = model.predict_proba(X_tensor)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])