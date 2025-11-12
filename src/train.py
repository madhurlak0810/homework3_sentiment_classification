"""
Training script for sentiment classification models.
Handles training loop, gradient clipping, model saving, and experiment tracking.
"""

import os
import time
import json
import pickle
import argparse
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ModelFactory, initialize_weights
from preprocess import preprocess_data, get_data_loaders
from utils import set_seed, EarlyStopping, MetricsTracker
from evaluate import evaluate_model

# Set random seeds for reproducibility
set_seed(42)


class SentimentTrainer:
    """Trainer class for sentiment classification models."""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any]):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer(config['optimizer'], config.get('learning_rate', 0.001))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Gradient clipping
        self.use_grad_clipping = config.get('gradient_clipping', False)
        self.clip_value = config.get('clip_value', 1.0)
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 5),
            min_delta=config.get('min_delta', 0.001)
        )
    
    def _get_optimizer(self, optimizer_name: str, learning_rate: float):
        """Create optimizer based on configuration."""
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            # Move data to device
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.use_grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy, epoch_time
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model for one epoch."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, 
              epochs: int = 10) -> Dict[str, List[float]]:
        """Train model for specified number of epochs."""
        print(f"Starting training for {epochs} epochs...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Gradient clipping: {self.use_grad_clipping}")
        print("-" * 50)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['epoch_times'].append(epoch_time)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.1f}s")
                
                # Early stopping check
                if self.early_stopping.should_stop(val_loss):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Time: {epoch_time:.1f}s")
        
        print("Training completed!")
        return history
    
    def save_model(self, filepath: str, additional_info: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_class': self.model.__class__.__name__,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
        return checkpoint


def run_experiment(config: Dict[str, Any], data_dir: str, results_dir: str) -> Dict[str, Any]:
    """Run a single experiment with given configuration."""
    print(f"\nRunning experiment: {config['experiment_name']}")
    print(f"Configuration: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or preprocess data
    sequence_length = config['sequence_length']
    
    # Always preprocess data fresh (no caching for now)
    sample_size = 1000 if config['experiment_name'] == 'quick_test' else None
    processed_data = preprocess_data(data_dir, [sequence_length], sample_size=sample_size)
    
    # Create data loaders
    train_loader = get_data_loaders(
        processed_data[sequence_length]['train_dataset'], 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    test_loader = get_data_loaders(
        processed_data[sequence_length]['test_dataset'], 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    # Create model
    model_config = {
        'embedding_dim': config['embedding_dim'],
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout'],
        'activation': config['activation']
    }
    
    model = ModelFactory.create_model(
        model_type=config['architecture'],
        vocab_size=processed_data['vocab_size'],
        **model_config
    )
    
    # Initialize weights
    initialize_weights(model)
    
    # Create trainer
    trainer = SentimentTrainer(model, device, config)
    
    # Train model
    start_time = time.time()
    history = trainer.train(train_loader, test_loader, config['epochs'])
    total_training_time = time.time() - start_time
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Calculate average epoch time
    avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
    
    # Save model
    model_filename = f"{config['experiment_name']}_model.pth"
    model_path = os.path.join(results_dir, model_filename)
    trainer.save_model(model_path, {
        'history': history,
        'test_metrics': test_metrics,
        'total_training_time': total_training_time
    })
    
    # Prepare results
    results = {
        'experiment_name': config['experiment_name'],
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'avg_epoch_time': avg_epoch_time,
        'total_training_time': total_training_time,
        'model_path': model_path
    }
    
    print(f"Experiment completed!")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Average epoch time: {avg_epoch_time:.1f}s")
    
    return results


def create_experiment_configs() -> List[Dict[str, Any]]:
    """Create all experiment configurations for the comparative study."""
    base_config = {
        'embedding_dim': 100,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.001,
        'patience': 5,
        'min_delta': 0.001,
        'clip_value': 1.0
    }
    
    experiments = []
    
    # Test different architectures
    architectures = ['rnn', 'lstm', 'bilstm']
    activations = ['relu', 'sigmoid', 'tanh']
    optimizers = ['adam', 'sgd', 'rmsprop']
    sequence_lengths = [25, 50, 100]
    gradient_clipping_options = [False, True]
    
    experiment_id = 1
    
    # Systematic experiments
    for architecture in architectures:
        for activation in activations:
            for optimizer in optimizers:
                for seq_len in sequence_lengths:
                    for grad_clip in gradient_clipping_options:
                        config = base_config.copy()
                        config.update({
                            'experiment_name': f'exp_{experiment_id:03d}_{architecture}_{activation}_{optimizer}_{seq_len}_{grad_clip}',
                            'architecture': architecture,
                            'activation': activation,
                            'optimizer': optimizer,
                            'sequence_length': seq_len,
                            'gradient_clipping': grad_clip
                        })
                        experiments.append(config)
                        experiment_id += 1
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Train sentiment classification models')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='../results', help='Results directory')
    parser.add_argument('--experiment_id', type=int, default=None, help='Specific experiment ID to run')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments')
    parser.add_argument('--config_file', type=str, default=None, help='Custom config file')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.config_file:
        # Load custom configuration
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        
        results = run_experiment(config, args.data_dir, args.results_dir)
        
        # Save results
        results_file = os.path.join(args.results_dir, f"{config['experiment_name']}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif args.run_all:
        # Run all experiments
        experiments = create_experiment_configs()
        all_results = []
        
        print(f"Running {len(experiments)} experiments...")
        
        for i, config in enumerate(experiments):
            print(f"\nProgress: {i+1}/{len(experiments)}")
            
            try:
                results = run_experiment(config, args.data_dir, args.results_dir)
                all_results.append(results)
                
                # Save individual results
                results_file = os.path.join(args.results_dir, f"{config['experiment_name']}_results.json")
                with open(results_file, 'w') as f:
                    # Make results JSON serializable
                    serializable_results = {
                        k: v for k, v in results.items() 
                        if k not in ['history']  # Skip non-serializable parts
                    }
                    json.dump(serializable_results, f, indent=2)
                
            except Exception as e:
                print(f"Error in experiment {config['experiment_name']}: {e}")
                continue
        
        # Save summary results
        summary_file = os.path.join(args.results_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            summary_results = []
            for result in all_results:
                summary = {
                    'experiment_name': result['experiment_name'],
                    'config': result['config'],
                    'test_accuracy': result['test_metrics']['accuracy'],
                    'test_f1_score': result['test_metrics']['f1_score'],
                    'avg_epoch_time': result['avg_epoch_time']
                }
                summary_results.append(summary)
            json.dump(summary_results, f, indent=2)
        
        print(f"\nAll experiments completed! Results saved to {args.results_dir}")
    
    elif args.experiment_id is not None:
        # Run specific experiment
        experiments = create_experiment_configs()
        if 0 <= args.experiment_id < len(experiments):
            config = experiments[args.experiment_id]
            results = run_experiment(config, args.data_dir, args.results_dir)
            
            # Save results
            results_file = os.path.join(args.results_dir, f"{config['experiment_name']}_results.json")
            with open(results_file, 'w') as f:
                serializable_results = {
                    k: v for k, v in results.items() 
                    if k not in ['history']
                }
                json.dump(serializable_results, f, indent=2)
        else:
            print(f"Invalid experiment ID: {args.experiment_id}")
    
    else:
        # Run a single default experiment
        config = {
            'experiment_name': 'default_lstm_experiment',
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': 'adam',
            'sequence_length': 50,
            'gradient_clipping': True,
            'embedding_dim': 100,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'patience': 5,
            'min_delta': 0.001,
            'clip_value': 1.0
        }
        
        results = run_experiment(config, args.data_dir, args.results_dir)
        
        # Save results
        results_file = os.path.join(args.results_dir, f"{config['experiment_name']}_results.json")
        with open(results_file, 'w') as f:
            serializable_results = {
                k: v for k, v in results.items() 
                if k not in ['history']
            }
            json.dump(serializable_results, f, indent=2)


if __name__ == "__main__":
    main()