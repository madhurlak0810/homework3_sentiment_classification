"""
Evaluation script for sentiment classification models.
Calculates accuracy, F1-score, precision, recall, and other metrics.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)


def evaluate_model(model: nn.Module, data_loader: DataLoader, 
                  device: torch.device, detailed: bool = True) -> Dict[str, Any]:
    """
    Evaluate model on given dataset.
    
    Args:
        model: Trained model to evaluate
        data_loader: DataLoader containing evaluation data
        device: Device to run evaluation on
        detailed: Whether to return detailed metrics
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0
    inference_times = []
    
    criterion = nn.BCELoss()
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(data_loader):
            # Move data to device
            texts = texts.to(device)
            labels = labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            # Forward pass
            outputs = model(texts)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels_np)
            all_probabilities.extend(probabilities)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    
    # ROC AUC score
    try:
        roc_auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        roc_auc = 0.0  # In case of single class
    
    # Average loss
    avg_loss = total_loss / len(data_loader)
    
    # Timing metrics
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    
    # Basic metrics
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc),
        'loss': float(avg_loss),
        'avg_inference_time': float(avg_inference_time),
        'total_inference_time': float(total_inference_time),
        'num_samples': len(all_labels),
        'num_batches': len(data_loader)
    }
    
    if detailed:
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        # Per-class metrics
        metrics.update({
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'true_positives': int(cm[1, 1]),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0])
        })
        
        # Additional statistics
        metrics.update({
            'positive_samples': int(np.sum(all_labels)),
            'negative_samples': int(len(all_labels) - np.sum(all_labels)),
            'predicted_positive': int(np.sum(all_predictions)),
            'predicted_negative': int(len(all_predictions) - np.sum(all_predictions))
        })
    
    print("Evaluation completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return metrics


def evaluate_multiple_models(models: Dict[str, nn.Module], data_loader: DataLoader, 
                           device: torch.device) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple models on the same dataset.
    
    Args:
        models: Dictionary of model_name -> model
        data_loader: DataLoader containing evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of model_name -> metrics
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        print("-" * 50)
        
        model_metrics = evaluate_model(model, data_loader, device, detailed=True)
        results[model_name] = model_metrics
    
    return results


def compare_models(results: Dict[str, Dict[str, Any]], 
                  metrics: List[str] = ['accuracy', 'f1_score', 'precision', 'recall']) -> None:
    """
    Print comparison table of model performances.
    
    Args:
        results: Dictionary of model_name -> metrics
        metrics: List of metrics to compare
    """
    print("\nModel Comparison:")
    print("=" * 80)
    
    # Print header
    header = f"{'Model':<20}"
    for metric in metrics:
        header += f"{metric.title():<12}"
    print(header)
    print("-" * 80)
    
    # Print results for each model
    for model_name, model_metrics in results.items():
        row = f"{model_name:<20}"
        for metric in metrics:
            value = model_metrics.get(metric, 0.0)
            row += f"{value:<12.4f}"
        print(row)


def calculate_statistical_significance(results1: Dict[str, Any], 
                                     results2: Dict[str, Any], 
                                     metric: str = 'accuracy') -> Dict[str, float]:
    """
    Calculate statistical significance between two model results.
    Note: This is a simplified version. For proper statistical testing,
    you would need multiple runs or bootstrap sampling.
    """
    # This is a placeholder for statistical significance testing
    # In practice, you would need multiple runs of the same experiment
    # or bootstrap sampling to calculate proper confidence intervals
    
    value1 = results1.get(metric, 0.0)
    value2 = results2.get(metric, 0.0)
    
    difference = abs(value1 - value2)
    
    return {
        'metric': metric,
        'model1_value': value1,
        'model2_value': value2,
        'difference': difference,
        'note': 'Statistical significance testing requires multiple runs'
    }


def generate_evaluation_report(results: Dict[str, Dict[str, Any]], 
                             output_file: str = None) -> str:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: Dictionary of model_name -> metrics
        output_file: Optional file path to save the report
        
    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("SENTIMENT CLASSIFICATION EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Summary table
    report_lines.append("MODEL PERFORMANCE SUMMARY:")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'ROC AUC':<10}")
    report_lines.append("-" * 40)
    
    # Sort models by accuracy
    sorted_models = sorted(results.items(), 
                          key=lambda x: x[1].get('accuracy', 0), 
                          reverse=True)
    
    for model_name, metrics in sorted_models:
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('f1_score', 0)
        roc_auc = metrics.get('roc_auc', 0)
        
        report_lines.append(f"{model_name:<20} {accuracy:<10.4f} {f1_score:<10.4f} {roc_auc:<10.4f}")
    
    report_lines.append("")
    
    # Best performing model
    best_model_name, best_metrics = sorted_models[0]
    report_lines.append(f"BEST PERFORMING MODEL: {best_model_name}")
    report_lines.append("-" * 30)
    report_lines.append(f"Accuracy: {best_metrics.get('accuracy', 0):.4f}")
    report_lines.append(f"F1-Score: {best_metrics.get('f1_score', 0):.4f}")
    report_lines.append(f"Precision: {best_metrics.get('precision', 0):.4f}")
    report_lines.append(f"Recall: {best_metrics.get('recall', 0):.4f}")
    report_lines.append(f"ROC AUC: {best_metrics.get('roc_auc', 0):.4f}")
    report_lines.append("")
    
    # Detailed results for each model
    for model_name, metrics in results.items():
        report_lines.append(f"DETAILED RESULTS - {model_name.upper()}")
        report_lines.append("-" * 40)
        
        for key, value in metrics.items():
            if key not in ['confusion_matrix', 'classification_report']:
                if isinstance(value, float):
                    report_lines.append(f"{key}: {value:.4f}")
                else:
                    report_lines.append(f"{key}: {value}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            report_lines.append("\nConfusion Matrix:")
            report_lines.append("                 Predicted")
            report_lines.append("               Neg    Pos")
            report_lines.append(f"Actual Neg    {cm[0][0]:4d}   {cm[0][1]:4d}")
            report_lines.append(f"       Pos    {cm[1][0]:4d}   {cm[1][1]:4d}")
        
        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")
    
    return report_text


class ModelEvaluator:
    """Class to handle comprehensive model evaluation."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}
    
    def evaluate_single_model(self, model: nn.Module, data_loader: DataLoader, 
                            model_name: str) -> Dict[str, Any]:
        """Evaluate a single model and store results."""
        metrics = evaluate_model(model, data_loader, self.device, detailed=True)
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_batch(self, models: Dict[str, nn.Module], 
                      data_loader: DataLoader) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple models."""
        for model_name, model in models.items():
            self.evaluate_single_model(model, data_loader, model_name)
        return self.results
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, Dict[str, Any]]:
        """Get the best performing model based on specified metric."""
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluation first.")
        
        best_model = max(self.results.items(), 
                        key=lambda x: x[1].get(metric, 0))
        return best_model
    
    def generate_comparison_table(self, metrics: List[str] = None) -> str:
        """Generate a comparison table of all evaluated models."""
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        
        if not self.results:
            return "No evaluation results found."
        
        # Create table header
        table_lines = []
        header = f"{'Model':<25}"
        for metric in metrics:
            header += f"{metric.replace('_', ' ').title():<12}"
        table_lines.append(header)
        table_lines.append("-" * len(header))
        
        # Add results for each model
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].get('accuracy', 0), 
                              reverse=True)
        
        for model_name, model_metrics in sorted_results:
            row = f"{model_name:<25}"
            for metric in metrics:
                value = model_metrics.get(metric, 0.0)
                row += f"{value:<12.4f}"
            table_lines.append(row)
        
        return "\n".join(table_lines)


if __name__ == "__main__":
    # Example usage (when running as standalone script)
    print("Evaluation module - for testing purposes only")
    
    # This would typically be called from the training script
    # or a separate evaluation script with actual models and data
    pass