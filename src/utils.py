"""
Utility functions for sentiment classification project.
Includes plotting, metrics tracking, early stopping, and reproducibility functions.
"""

import os
import random
import pickle
import json
from typing import List, Dict, Any, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some functions will not work.")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop
    
    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop."""
        return self.__call__(val_loss)


class MetricsTracker:
    """Track and store training/validation metrics."""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
    
    def update(self, epoch_metrics: Dict[str, float]):
        """Update metrics with new epoch values."""
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_acc', mode: str = 'max') -> int:
        """Get epoch with best performance for given metric."""
        if metric not in self.metrics or not self.metrics[metric]:
            return -1
        
        values = self.metrics[metric]
        if mode == 'max':
            return np.argmax(values)
        else:
            return np.argmin(values)
    
    def save(self, filepath: str):
        """Save metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)


class PlotGenerator:
    """Generate various plots for model evaluation and analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        plt.style.use('default')  # Use default style instead of seaborn
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: str = None, show: bool = True):
        """Plot training and validation loss/accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss', alpha=0.8)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', alpha=0.8)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy', alpha=0.8)
        if 'val_acc' in history and history['val_acc']:
            ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy', alpha=0.8)
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison_bar(self, results: Dict[str, Dict[str, float]], 
                          metrics: List[str] = ['accuracy', 'f1_score'], 
                          save_path: str = None, show: bool = True):
        """Plot bar chart comparing different models."""
        models = list(results.keys())
        n_metrics = len(metrics)
        n_models = len(models)
        
        fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 6))
        
        x = np.arange(n_models)
        width = 0.8 / n_metrics
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_sequence_length_analysis(self, results: Dict[int, Dict[str, float]], 
                                    metric: str = 'accuracy',
                                    save_path: str = None, show: bool = True):
        """Plot performance vs sequence length."""
        sequence_lengths = sorted(results.keys())
        values = [results[seq_len].get(metric, 0) for seq_len in sequence_lengths]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(sequence_lengths, values, 'bo-', linewidth=2, markersize=8, alpha=0.8)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Sequence Length')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on points
        for x, y in zip(sequence_lengths, values):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sequence length analysis plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None,
                            save_path: str = None, show: bool = True):
        """Plot confusion matrix heatmap."""
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_experiment_heatmap(self, experiment_results: List[Dict[str, Any]], 
                              metric: str = 'accuracy',
                              save_path: str = None, show: bool = True):
        """Plot heatmap of experiment results."""
        # Create DataFrame from experiment results
        data = []
        for result in experiment_results:
            config = result['config']
            data.append({
                'architecture': config['architecture'],
                'activation': config['activation'],
                'optimizer': config['optimizer'],
                'sequence_length': config['sequence_length'],
                'gradient_clipping': config['gradient_clipping'],
                metric: result['test_metrics'][metric]
            })
        
        df = pd.DataFrame(data)
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values=metric,
            index=['architecture', 'activation'],
            columns=['optimizer', 'gradient_clipping'],
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
        ax.set_title(f'Experiment Results: {metric.replace("_", " ").title()}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Experiment heatmap saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def create_results_summary_table(experiment_results: List[Dict[str, Any]],
                                metrics: List[str] = ['accuracy', 'f1_score']) -> pd.DataFrame:
    """Create a summary table of all experiment results."""
    summary_data = []
    
    for result in experiment_results:
        config = result['config']
        test_metrics = result['test_metrics']
        
        row = {
            'Model': config['architecture'].upper(),
            'Activation': config['activation'].title(),
            'Optimizer': config['optimizer'].title(),
            'Seq Length': config['sequence_length'],
            'Grad Clipping': 'Yes' if config['gradient_clipping'] else 'No',
            'Epoch Time (s)': result.get('avg_epoch_time', 0)
        }
        
        for metric in metrics:
            row[metric.replace('_', ' ').title()] = test_metrics.get(metric, 0)
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df


def save_results_to_csv(experiment_results: List[Dict[str, Any]], 
                       filepath: str,
                       metrics: List[str] = ['accuracy', 'f1_score']):
    """Save experiment results to CSV file."""
    df = create_results_summary_table(experiment_results, metrics)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def load_experiment_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all experiment results from a directory."""
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    print(f"Loaded {len(results)} experiment results")
    return results


def find_best_configuration(experiment_results: List[Dict[str, Any]], 
                          metric: str = 'accuracy') -> Dict[str, Any]:
    """Find the best performing configuration."""
    best_result = max(experiment_results, 
                     key=lambda x: x['test_metrics'].get(metric, 0))
    
    print(f"Best configuration based on {metric}:")
    print(f"  Model: {best_result['config']['architecture']}")
    print(f"  Activation: {best_result['config']['activation']}")
    print(f"  Optimizer: {best_result['config']['optimizer']}")
    print(f"  Sequence Length: {best_result['config']['sequence_length']}")
    print(f"  Gradient Clipping: {best_result['config']['gradient_clipping']}")
    print(f"  {metric.title()}: {best_result['test_metrics'][metric]:.4f}")
    
    return best_result


def analyze_hyperparameter_impact(experiment_results: List[Dict[str, Any]], 
                                hyperparameter: str,
                                metric: str = 'accuracy') -> Dict[str, float]:
    """Analyze the impact of a specific hyperparameter."""
    impact_analysis = {}
    
    for result in experiment_results:
        config = result['config']
        param_value = config.get(hyperparameter)
        
        if param_value not in impact_analysis:
            impact_analysis[param_value] = []
        
        impact_analysis[param_value].append(result['test_metrics'].get(metric, 0))
    
    # Calculate average performance for each parameter value
    avg_performance = {}
    for param_value, performances in impact_analysis.items():
        avg_performance[param_value] = np.mean(performances)
    
    print(f"Impact of {hyperparameter} on {metric}:")
    for param_value, avg_perf in sorted(avg_performance.items(), 
                                       key=lambda x: x[1], reverse=True):
        print(f"  {param_value}: {avg_perf:.4f}")
    
    return avg_performance


def get_system_info() -> Dict[str, str]:
    """Get system information for reproducibility reporting."""
    import platform
    import psutil
    
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'ram_total_gb': f"{psutil.virtual_memory().total / (1024**3):.1f}",
        'ram_available_gb': f"{psutil.virtual_memory().available / (1024**3):.1f}"
    }
    
    if TORCH_AVAILABLE:
        info['torch_version'] = torch.__version__
        info['cuda_available'] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
    
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    
    print("System Information:")
    print("-" * 30)
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    # Example usage
    print("Utilities module - Testing basic functionality")
    
    # Set seed
    set_seed(42)
    
    # Print system info
    print_system_info()
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    # Simulate training with decreasing validation loss
    val_losses = [0.8, 0.7, 0.6, 0.65, 0.66, 0.67]  # Should trigger early stopping
    
    print("\nTesting early stopping:")
    for i, loss in enumerate(val_losses):
        should_stop = early_stopping.should_stop(loss)
        print(f"Epoch {i+1}, Val Loss: {loss:.3f}, Should stop: {should_stop}")
        if should_stop:
            break