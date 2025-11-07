# Pytest configuration and fixtures
import pytest
import numpy as np
import torch
from pathlib import Path

@pytest.fixture(scope="session")
def device():
    """Get device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

@pytest.fixture
def sample_features():
    """Create sample feature matrix"""
    n_samples = 100
    n_features = 78
    return np.random.randn(n_samples, n_features).astype(np.float32)

@pytest.fixture
def sample_labels():
    """Create sample labels"""
    n_samples = 100
    return np.random.randint(0, 2, n_samples)

@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset seeds before each test"""
    yield
    np.random.seed(None)
    torch.manual_seed(torch.seed())
