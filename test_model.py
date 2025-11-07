# Unit Tests for Model
import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import tempfile

from model import AdvancedNetworkIDS, create_model
from utils import DataPreprocessor, NetworkTrafficDataset, get_class_weights
from config import ModelConfig

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    n_samples = 100
    n_features = 78
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y

@pytest.fixture
def model():
    """Create test model"""
    return AdvancedNetworkIDS(
        input_size=78,
        hidden_sizes=[256, 128, 64],
        output_size=2,
        dropout_rate=0.3
    )

class TestModel:
    """Test cases for neural network model"""
    
    def test_model_initialization(self, model):
        """Test model can be initialized"""
        assert model is not None
        assert isinstance(model, AdvancedNetworkIDS)
    
    def test_model_forward_pass(self, model, sample_data):
        """Test forward pass through model"""
        X, _ = sample_data
        X_tensor = torch.FloatTensor(X)
        
        output = model(X_tensor)
        
        assert output.shape == (100, 2)
        assert not torch.isnan(output).any()
    
    def test_model_prediction(self, model, sample_data):
        """Test model prediction"""
        X, _ = sample_data
        X_tensor = torch.FloatTensor(X)
        
        predictions = model.predict(X_tensor)
        
        assert predictions.shape == (100,)
        assert torch.all((predictions == 0) | (predictions == 1))
    
    def test_model_probability(self, model, sample_data):
        """Test probability prediction"""
        X, _ = sample_data
        X_tensor = torch.FloatTensor(X)
        
        proba = model.predict_proba(X_tensor)
        
        assert proba.shape == (100, 2)
        assert torch.all(proba >= 0) and torch.all(proba <= 1)
        assert torch.allclose(proba.sum(dim=1), torch.ones(100))
    
    def test_model_parameters(self, model):
        """Test model parameters"""
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
        assert params < 1_000_000  # Should be reasonable size
    
    def test_model_gradient_flow(self, model, sample_data):
        """Test gradients flow through model"""
        X, y = sample_data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None

class TestDataUtils:
    """Test cases for data utilities"""
    
    def test_dataset_creation(self, sample_data):
        """Test dataset creation"""
        X, y = sample_data
        dataset = NetworkTrafficDataset(X, y)
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self, sample_data):
        """Test dataset indexing"""
        X, y = sample_data
        dataset = NetworkTrafficDataset(X, y)
        
        X_sample, y_sample = dataset[0]
        
        assert X_sample.shape == (78,)
        assert y_sample in [0, 1]
    
    def test_class_weights(self, sample_data):
        """Test class weight calculation"""
        _, y = sample_data
        weights = get_class_weights(y)
        
        assert weights.shape[0] == 2
        assert torch.all(weights > 0)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert preprocessor.scaler is None
        assert preprocessor.label_encoder is None

class TestModelFactory:
    """Test cases for model factory function"""
    
    def test_create_advanced_model(self):
        """Test creating advanced model"""
        model = create_model("advanced")
        assert isinstance(model, AdvancedNetworkIDS)
    
    def test_create_model_with_device(self):
        """Test creating model on specific device"""
        device = torch.device("cpu")
        model = create_model("advanced", device=device)
        
        # Check model is on correct device
        for param in model.parameters():
            assert param.device == device
    
    def test_invalid_model_type(self):
        """Test error on invalid model type"""
        with pytest.raises(ValueError):
            create_model("invalid_type")

class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self, sample_data):
        """Test full training pipeline"""
        X, y = sample_data
        
        # Create model
        model = create_model("advanced")
        
        # Create dataset
        dataset = NetworkTrafficDataset(X, y)
        
        # Create dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32)
        
        # Forward pass
        for X_batch, y_batch in loader:
            output = model(X_batch)
            assert output.shape[0] == X_batch.shape[0]
            break

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
