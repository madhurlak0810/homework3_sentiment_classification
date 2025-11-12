#!/usr/bin/env python3
"""
Script to create required plots from experimental results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_experiment_data():
    """Load experiment data from JSON file"""
    with open("results/experiment_summary.json", 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_performance_vs_sequence_length():
    """Plot Accuracy and F1 vs Sequence Length"""
    df = load_experiment_data()
    
    # Extract configuration details
    df['architecture'] = df['config'].apply(lambda x: x['architecture'])
    df['sequence_length'] = df['config'].apply(lambda x: x['sequence_length'])
    df['optimizer'] = df['config'].apply(lambda x: x['optimizer'])
    df['activation'] = df['config'].apply(lambda x: x['activation'])
    
    # Create summary by sequence length
    seq_summary = df.groupby('sequence_length').agg({
        'test_accuracy': ['mean', 'std'],
        'test_f1_score': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    seq_summary.columns = ['_'.join(col) for col in seq_summary.columns]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Accuracy vs Sequence Length
    x = seq_summary.index
    y_acc = seq_summary['test_accuracy_mean']
    yerr_acc = seq_summary['test_accuracy_std']
    
    ax1.errorbar(x, y_acc, yerr=yerr_acc, marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy vs Sequence Length\n(Mean ± Standard Deviation)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 0.85)
    
    # Add value labels
    for i, (xi, yi) in enumerate(zip(x, y_acc)):
        ax1.annotate(f'{yi:.3f}', (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot F1 vs Sequence Length
    y_f1 = seq_summary['test_f1_score_mean']
    yerr_f1 = seq_summary['test_f1_score_std']
    
    ax2.errorbar(x, y_f1, yerr=yerr_f1, marker='s', linewidth=2, markersize=8, capsize=5, color='orange')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Test F1-Score')
    ax2.set_title('Test F1-Score vs Sequence Length\n(Mean ± Standard Deviation)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 0.85)
    
    # Add value labels
    for i, (xi, yi) in enumerate(zip(x, y_f1)):
        ax2.annotate(f'{yi:.3f}', (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    plt.savefig('results/plots/performance_vs_sequence_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: results/plots/performance_vs_sequence_length.png")

def plot_architecture_comparison():
    """Plot performance comparison by architecture"""
    df = load_experiment_data()
    
    # Extract configuration details
    df['architecture'] = df['config'].apply(lambda x: x['architecture'].upper())
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy by Architecture
    df.boxplot(column='test_accuracy', by='architecture', ax=ax1)
    ax1.set_title('Test Accuracy by Architecture')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_ylim(0.4, 0.85)
    
    # F1-Score by Architecture
    df.boxplot(column='test_f1_score', by='architecture', ax=ax2)
    ax2.set_title('Test F1-Score by Architecture')
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Test F1-Score')
    ax2.set_ylim(0.4, 0.85)
    
    # Training Time by Architecture
    df.boxplot(column='avg_epoch_time', by='architecture', ax=ax3)
    ax3.set_title('Average Epoch Time by Architecture')
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel('Epoch Time (seconds)')
    
    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    
    plt.savefig('results/plots/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: results/plots/architecture_comparison.png")

def plot_optimizer_performance():
    """Plot performance by optimizer"""
    df = load_experiment_data()
    
    # Extract optimizer info
    df['optimizer'] = df['config'].apply(lambda x: x['optimizer'].upper())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy by Optimizer
    optimizer_acc = df.groupby('optimizer')['test_accuracy'].agg(['mean', 'std']).round(4)
    x_pos = np.arange(len(optimizer_acc))
    
    bars1 = ax1.bar(x_pos, optimizer_acc['mean'], yerr=optimizer_acc['std'], 
                    capsize=5, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Mean Test Accuracy')
    ax1.set_title('Test Accuracy by Optimizer\n(Mean ± Standard Deviation)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(optimizer_acc.index)
    ax1.set_ylim(0.4, 0.8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, optimizer_acc['mean'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    # F1-Score by Optimizer
    optimizer_f1 = df.groupby('optimizer')['test_f1_score'].agg(['mean', 'std']).round(4)
    
    bars2 = ax2.bar(x_pos, optimizer_f1['mean'], yerr=optimizer_f1['std'], 
                    capsize=5, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xlabel('Optimizer')
    ax2.set_ylabel('Mean Test F1-Score')
    ax2.set_title('Test F1-Score by Optimizer\n(Mean ± Standard Deviation)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(optimizer_f1.index)
    ax2.set_ylim(0.4, 0.8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, optimizer_f1['mean'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plt.savefig('results/plots/optimizer_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: results/plots/optimizer_performance.png")

def plot_training_curves_best_worst():
    """Plot training curves for best and worst performing models"""
    df = load_experiment_data()
    
    # Find best and worst experiments
    best_idx = df['test_accuracy'].idxmax()
    worst_idx = df['test_accuracy'].idxmin()
    
    best_exp = df.loc[best_idx]
    worst_exp = df.loc[worst_idx]
    
    print(f"Best model: {best_exp['experiment_name']} - Accuracy: {best_exp['test_accuracy']:.4f}")
    print(f"Worst model: {worst_exp['experiment_name']} - Accuracy: {worst_exp['test_accuracy']:.4f}")
    
    # Try to load individual result files for training curves
    best_file = f"results/{best_exp['experiment_name']}_results.json"
    worst_file = f"results/{worst_exp['experiment_name']}_results.json"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot best model training curve
    if os.path.exists(best_file):
        with open(best_file, 'r') as f:
            best_data = json.load(f)
        
        if 'training_history' in best_data:
            history = best_data['training_history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Best Model Training Curves\n{best_exp["experiment_name"]}\nFinal Accuracy: {best_exp["test_accuracy"]:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Training history not available\nfor best model', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title(f'Best Model: {best_exp["experiment_name"]}\nAccuracy: {best_exp["test_accuracy"]:.4f}')
    else:
        ax1.text(0.5, 0.5, 'Training data file not found\nfor best model', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f'Best Model: {best_exp["experiment_name"]}\nAccuracy: {best_exp["test_accuracy"]:.4f}')
    
    # Plot worst model training curve
    if os.path.exists(worst_file):
        with open(worst_file, 'r') as f:
            worst_data = json.load(f)
        
        if 'training_history' in worst_data:
            history = worst_data['training_history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title(f'Worst Model Training Curves\n{worst_exp["experiment_name"]}\nFinal Accuracy: {worst_exp["test_accuracy"]:.4f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Training history not available\nfor worst model', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f'Worst Model: {worst_exp["experiment_name"]}\nAccuracy: {worst_exp["test_accuracy"]:.4f}')
    else:
        ax2.text(0.5, 0.5, 'Training data file not found\nfor worst model', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'Worst Model: {worst_exp["experiment_name"]}\nAccuracy: {worst_exp["test_accuracy"]:.4f}')
    
    plt.tight_layout()
    
    plt.savefig('results/plots/training_curves_best_worst.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: results/plots/training_curves_best_worst.png")

def plot_activation_function_comparison():
    """Plot performance by activation function"""
    df = load_experiment_data()
    
    # Extract activation function info
    df['activation'] = df['config'].apply(lambda x: x['activation'].capitalize())
    
    # Create summary statistics
    activation_stats = df.groupby('activation').agg({
        'test_accuracy': ['mean', 'std', 'max'],
        'test_f1_score': ['mean', 'std', 'max']
    }).round(4)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Violin plot for accuracy
    df.boxplot(column='test_accuracy', by='activation', ax=ax1)
    ax1.set_title('Test Accuracy by Activation Function')
    ax1.set_xlabel('Activation Function')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_ylim(0.4, 0.85)
    
    # Violin plot for F1-score
    df.boxplot(column='test_f1_score', by='activation', ax=ax2)
    ax2.set_title('Test F1-Score by Activation Function')
    ax2.set_xlabel('Activation Function')
    ax2.set_ylabel('Test F1-Score')
    ax2.set_ylim(0.4, 0.85)
    
    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    
    plt.savefig('results/plots/activation_function_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: results/plots/activation_function_comparison.png")

def create_heatmap_analysis():
    """Create heatmap of performance across different configurations"""
    df = load_experiment_data()
    
    # Extract configuration details
    df['architecture'] = df['config'].apply(lambda x: x['architecture'].upper())
    df['optimizer'] = df['config'].apply(lambda x: x['optimizer'].upper())
    df['activation'] = df['config'].apply(lambda x: x['activation'].capitalize())
    df['sequence_length'] = df['config'].apply(lambda x: x['sequence_length'])
    
    # Create pivot table for heatmap - Architecture vs Optimizer
    pivot_acc = df.pivot_table(values='test_accuracy', 
                               index='architecture', 
                               columns='optimizer', 
                               aggfunc='mean')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap 1: Architecture vs Optimizer
    sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='viridis', ax=ax1)
    ax1.set_title('Mean Test Accuracy\nArchitecture vs Optimizer')
    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Architecture')
    
    # Heatmap 2: Activation vs Sequence Length
    pivot_seq = df.pivot_table(values='test_accuracy', 
                               index='activation', 
                               columns='sequence_length', 
                               aggfunc='mean')
    
    sns.heatmap(pivot_seq, annot=True, fmt='.4f', cmap='viridis', ax=ax2)
    ax2.set_title('Mean Test Accuracy\nActivation vs Sequence Length')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Activation Function')
    
    plt.tight_layout()
    
    plt.savefig('results/plots/performance_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: results/plots/performance_heatmaps.png")

def main():
    """Create all required plots"""
    print("Creating experimental result plots...")
    
    # Create plots directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Generate all plots
    plot_performance_vs_sequence_length()
    plot_architecture_comparison()
    plot_optimizer_performance()
    plot_training_curves_best_worst()
    plot_activation_function_comparison()
    create_heatmap_analysis()
    
    print("\nAll plots created successfully in results/plots/")
    print("\nGenerated files:")
    plot_files = [
        "performance_vs_sequence_length.png",
        "architecture_comparison.png", 
        "optimizer_performance.png",
        "training_curves_best_worst.png",
        "activation_function_comparison.png",
        "performance_heatmaps.png"
    ]
    
    for plot_file in plot_files:
        print(f"  - {plot_file}")

if __name__ == "__main__":
    main()