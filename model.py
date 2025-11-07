# Neural Network Model for Network Intrusion Detection
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedNetworkIDS(nn.Module):
    """
    Advanced Deep Neural Network for Network Intrusion Detection
    Architecture:
    - Input Layer: 78 features (CICIDS 2017)
    - Hidden Layers: 256 -> 128 -> 64 neurons with dropout & batch normalization
    - Output Layer: 2 classes (Normal/Attack)
    """
    
    def __init__(self, input_size=78, hidden_sizes=[256, 128, 64], 
                 output_size=2, dropout_rate=0.3):
        super(AdvancedNetworkIDS, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layers
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier uniform weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Output logits of shape (batch_size, output_size)
        """
        # Layer 1 with batch norm and dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Layer 2 with batch norm and dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Layer 3 with batch norm and dropout
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer (no activation, will be handled by loss function)
        x = self.fc4(x)
        
        return x
    
    def predict(self, x):
        """
        Get predictions from model
        Args:
            x: Input tensor
        Returns:
            Predicted class labels
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x):
        """
        Get prediction probabilities from model
        Args:
            x: Input tensor
        Returns:
            Probability scores for each class
        """
        with torch.no_grad():
            logits = self.forward(x)
            proba = F.softmax(logits, dim=1)
        return proba

class NetworkIDS_Transformer(nn.Module):
    """
    Transformer-based model for Network Intrusion Detection
    Uses multi-head self-attention for feature interaction
    """
    
    def __init__(self, input_size=78, d_model=128, nhead=4, 
                 num_layers=3, output_size=2, dropout_rate=0.3):
        super(NetworkIDS_Transformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        """Forward pass through transformer model"""
        # Reshape for transformer (add sequence dimension)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Project to embedding dimension
        x = self.input_proj(x)  # (batch_size, 1, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        
        # Take mean pooling over sequence
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(model_type="advanced", input_size=78, output_size=2, device="cpu"):
    """
    Factory function to create model
    Args:
        model_type: "advanced" or "transformer"
        input_size: Number of input features
        output_size: Number of output classes
        device: Device to load model on
    Returns:
        Initialized model
    """
    if model_type == "advanced":
        model = AdvancedNetworkIDS(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            output_size=output_size,
            dropout_rate=0.3
        )
    elif model_type == "transformer":
        model = NetworkIDS_Transformer(
            input_size=input_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            output_size=output_size,
            dropout_rate=0.3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)
