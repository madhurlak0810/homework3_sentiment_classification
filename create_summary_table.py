#!/usr/bin/env python3
"""
Script to create comprehensive summary table from experimental results
"""

import json
import pandas as pd
import os

def create_summary_table():
    """Create comprehensive summary table from experiment results"""
    
    # Load results
    results_file = "results/experiment_summary.json"
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total experiments: {len(data)}")
    
    if len(data) == 0:
        print("No experiment data found!")
        return
    
    # Create summary table
    summary_data = []
    
    for exp in data:
        config = exp['config']
        summary_data.append({
            'Model': config['architecture'].upper(),
            'Activation': config['activation'].capitalize(),
            'Optimizer': config['optimizer'].upper(),
            'Seq Length': config['sequence_length'],
            'Grad Clipping': 'Yes' if config['gradient_clipping'] else 'No',
            'Accuracy': exp['test_accuracy'],
            'F1': exp['test_f1_score'],
            'Epoch Time (s)': exp['avg_epoch_time']
        })
    
    # Create DataFrame and sort by accuracy
    df = pd.DataFrame(summary_data)
    df_sorted = df.sort_values('Accuracy', ascending=False)
    
    # Format numeric columns for display
    df_display = df_sorted.copy()
    df_display['Accuracy'] = df_display['Accuracy'].apply(lambda x: f"{x:.4f}")
    df_display['F1'] = df_display['F1'].apply(lambda x: f"{x:.4f}")
    df_display['Epoch Time (s)'] = df_display['Epoch Time (s)'].apply(lambda x: f"{x:.2f}")
    
    # Save to CSV
    output_file = "results/metrics.csv"
    df_display.to_csv(output_file, index=False)
    
    print(f"\nSummary table saved to: {output_file}")
    print(f"Top 10 performing models:")
    print(df_display.head(10).to_string(index=False))
    
    # Print some statistics using original numeric values
    print(f"\nPerformance Statistics:")
    print(f"Best accuracy: {df_sorted.iloc[0]['Accuracy']:.4f}")
    print(f"Best F1: {df_sorted.sort_values('F1', ascending=False).iloc[0]['F1']:.4f}")
    
    # Architecture summary
    print(f"\nArchitecture Performance Summary:")
    for arch in ['RNN', 'LSTM', 'BILSTM']:
        arch_data = df_sorted[df_sorted['Model'] == arch]
        if len(arch_data) > 0:
            print(f"{arch}: Mean Acc={arch_data['Accuracy'].mean():.4f}, "
                  f"Max Acc={arch_data['Accuracy'].max():.4f}")
    
    return df_sorted

if __name__ == "__main__":
    create_summary_table()