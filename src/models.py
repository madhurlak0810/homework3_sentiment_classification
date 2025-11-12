"""
Model architectures for sentiment classification.
Implements RNN, LSTM, and Bidirectional LSTM models with configurable parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SentimentRNN(nn.Module):
    """Basic RNN model for sentiment classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3, activation: str = 'relu'):
        super(SentimentRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        if activation.lower() == 'tanh':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        else:
            # For ReLU and Sigmoid, we'll use LSTM and apply activation separately
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.activation_fn = self._get_activation_function(activation)
    
    def _get_activation_function(self, activation: str):
        """Get activation function based on string."""
        if activation.lower() == 'relu':
            return F.relu
        elif activation.lower() == 'sigmoid':
            return torch.sigmoid
        elif activation.lower() == 'tanh':
            return torch.tanh
        else:
            return F.relu  # Default
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN
        if self.activation.lower() == 'tanh':
            rnn_out, _ = self.rnn(embedded)  # (batch_size, seq_len, hidden_dim)
        else:
            # For non-tanh activations with LSTM
            rnn_out, _ = self.rnn(embedded)
            rnn_out = self.activation_fn(rnn_out)
        
        # Take the last output
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Dropout
        output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(output)  # (batch_size, 1)
        
        # Sigmoid activation for binary classification
        output = torch.sigmoid(output)
        
        return output.squeeze()  # (batch_size,)


class SentimentLSTM(nn.Module):
    """LSTM model for sentiment classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3, activation: str = 'relu'):
        super(SentimentLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.activation_fn = self._get_activation_function(activation)
    
    def _get_activation_function(self, activation: str):
        """Get activation function based on string."""
        if activation.lower() == 'relu':
            return F.relu
        elif activation.lower() == 'sigmoid':
            return torch.sigmoid
        elif activation.lower() == 'tanh':
            return torch.tanh
        else:
            return F.relu  # Default
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim)
        
        # Apply activation function to LSTM output
        lstm_out = self.activation_fn(lstm_out)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Dropout
        output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(output)  # (batch_size, 1)
        
        # Sigmoid activation for binary classification
        output = torch.sigmoid(output)
        
        return output.squeeze()  # (batch_size,)


class SentimentBiLSTM(nn.Module):
    """Bidirectional LSTM model for sentiment classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3, activation: str = 'relu'):
        super(SentimentBiLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer (input size is 2 * hidden_dim due to bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
        # Activation function
        self.activation_fn = self._get_activation_function(activation)
    
    def _get_activation_function(self, activation: str):
        """Get activation function based on string."""
        if activation.lower() == 'relu':
            return F.relu
        elif activation.lower() == 'sigmoid':
            return torch.sigmoid
        elif activation.lower() == 'tanh':
            return torch.tanh
        else:
            return F.relu  # Default
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Apply activation function to LSTM output
        lstm_out = self.activation_fn(lstm_out)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Dropout
        output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(output)  # (batch_size, 1)
        
        # Sigmoid activation for binary classification
        output = torch.sigmoid(output)
        
        return output.squeeze()  # (batch_size,)


class ModelFactory:
    """Factory class for creating models based on configuration."""
    
    @staticmethod
    def create_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
        """Create model based on type and parameters."""
        model_type = model_type.lower()
        
        if model_type == 'rnn':
            return SentimentRNN(vocab_size, **kwargs)
        elif model_type == 'lstm':
            return SentimentLSTM(vocab_size, **kwargs)
        elif model_type == 'bilstm' or model_type == 'bidirectional_lstm':
            return SentimentBiLSTM(vocab_size, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """Get model information and parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_type': model.__class__.__name__,
        }
        
        # Add model-specific info
        if hasattr(model, 'vocab_size'):
            info['vocab_size'] = model.vocab_size
        if hasattr(model, 'embedding_dim'):
            info['embedding_dim'] = model.embedding_dim
        if hasattr(model, 'hidden_dim'):
            info['hidden_dim'] = model.hidden_dim
        if hasattr(model, 'num_layers'):
            info['num_layers'] = model.num_layers
        if hasattr(model, 'activation'):
            info['activation'] = model.activation
        
        return info


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights using Xavier/Glorot initialization."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:  # For matrices
                nn.init.xavier_uniform_(param)
            else:  # For bias terms
                nn.init.zeros_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


# Configuration templates for different experiments
MODEL_CONFIGS = {
    'base_rnn': {
        'model_type': 'rnn',
        'embedding_dim': 100,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3
    },
    'base_lstm': {
        'model_type': 'lstm',
        'embedding_dim': 100,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3
    },
    'base_bilstm': {
        'model_type': 'bilstm',
        'embedding_dim': 100,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3
    }
}


if __name__ == "__main__":
    # Example usage
    vocab_size = 10000
    
    # Create different models
    models = {}
    for config_name, config in MODEL_CONFIGS.items():
        model = ModelFactory.create_model(vocab_size=vocab_size, **config)
        models[config_name] = model
        
        # Print model information
        info = ModelFactory.get_model_info(model)
        print(f"\n{config_name.upper()} Model:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Test with sample input
    print("\nTesting models with sample input:")
    sample_input = torch.randint(0, vocab_size, (2, 50))  # batch_size=2, seq_len=50
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
            print(f"{name}: Output shape = {output.shape}, Sample output = {output[:2].numpy()}")