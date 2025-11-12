# Sentiment Classification with RNN Architectures

A comprehensive comparative analysis of RNN architectures (RNN, LSTM, Bidirectional LSTM) for sentiment classification on the IMDb movie reviews dataset.

## Experimental Results

This project systematically evaluated 162 different configurations across multiple architectures, achieving strong performance on binary sentiment classification.

### Key Performance Metrics

- **Best Accuracy**: 82.43% (BiLSTM with Sigmoid + RMSprop, sequence length 100)
- **Best F1-Score**: 82.43% (Same configuration as best accuracy)
- **Dataset**: 50,000 IMDb movie reviews (25K train, 25K test)
- **Total Experiments**: 162 configurations tested

### Top 10 Performing Models

1. **BiLSTM + Sigmoid + RMSprop**: 82.43% accuracy (seq_len=100, no gradient clipping)
2. **BiLSTM + Sigmoid + Adam**: 82.26% accuracy (seq_len=100, no gradient clipping)
3. **LSTM + ReLU + RMSprop**: 82.22% accuracy (seq_len=100, no gradient clipping)
4. **LSTM + Sigmoid + Adam**: 82.10% accuracy (seq_len=100, gradient clipping)
5. **BiLSTM + Tanh + RMSprop**: 82.10% accuracy (seq_len=100, no gradient clipping)
6. **LSTM + Sigmoid + RMSprop**: 81.83% accuracy (seq_len=100, no gradient clipping)
7. **RNN + Sigmoid + Adam**: 81.71% accuracy (seq_len=100, gradient clipping)
8. **LSTM + Tanh + Adam**: 81.62% accuracy (seq_len=100, gradient clipping)
9. **RNN + ReLU + RMSprop**: 81.61% accuracy (seq_len=100, gradient clipping)
10. **LSTM + Tanh + RMSprop**: 81.47% accuracy (seq_len=100, no gradient clipping)

### Architecture Comparison

| Architecture | Mean Accuracy | Max Accuracy | Std Dev |
|-------------|---------------|--------------|---------|
| BiLSTM      | 67.04%        | 82.43%       | 12.50%  |
| LSTM        | 67.30%        | 82.22%       | 12.73%  |
| RNN         | 66.74%        | 81.71%       | 10.87%  |

### Hyperparameter Analysis

**Sequence Length Impact:**
| Length | Mean Accuracy | Max Accuracy | Std Dev |
|--------|---------------|--------------|---------|
| 25     | 63.66%        | 71.73%       | 9.05%   |
| 50     | 67.36%        | 76.84%       | 11.55%  |
| 100    | 70.06%        | 82.43%       | 14.17%  |

**Optimizer Performance:**
| Optimizer | Mean Accuracy | Max Accuracy | Std Dev |
|-----------|---------------|--------------|---------|
| RMSprop   | 75.25%        | 82.43%       | 4.81%   |
| Adam      | 74.28%        | 82.26%       | 5.95%   |
| SGD       | 51.55%        | 72.53%       | 4.30%   |

**Activation Function Performance:**
| Activation | Mean Accuracy | Max Accuracy | Std Dev |
|------------|---------------|--------------|---------|
| Sigmoid    | 67.34%        | 82.43%       | 12.81%  |
| ReLU       | 67.17%        | 82.22%       | 12.38%  |
| Tanh       | 66.56%        | 82.10%       | 10.91%  |

### Key Findings

1. **Sequence Length Impact**: Longer sequences consistently improve performance. Moving from 25 to 100 tokens increases mean accuracy by 6.4 percentage points.

2. **Optimizer Ranking**: RMSprop significantly outperforms other optimizers with the highest mean accuracy (75.25%) and best maximum performance (82.43%). SGD performs poorly across all configurations.

3. **Architecture Insights**: While all three architectures achieve similar peak performance (81-82%), BiLSTM shows slightly better maximum accuracy but higher variance.

4. **Activation Functions**: All three activation functions perform comparably, with slight advantages for sigmoid and ReLU over tanh.

5. **Gradient Clipping**: Shows mixed results with no consistent improvement pattern across configurations.

6. **Training Efficiency**: Average training time per epoch ranges from 1.7 to 2.6 seconds, making the approach computationally efficient.

## Project Structure

```
homework3_sentiment_classification/
├── src/                           # Source code
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── models.py                 # RNN/LSTM/BiLSTM architectures
│   ├── train.py                  # Training script and experiment runner
│   ├── evaluate.py               # Evaluation metrics and testing
│   └── utils.py                  # Utility functions and plotting
├── data/                         # Dataset storage
│   └── IMDB/                     # IMDb movie review dataset
├── results/                      # Experiment results
│   ├── experiment_summary.json  # Complete experimental results (162 experiments)
│   ├── metrics.csv              # Summary table in CSV format
│   ├── plots/                   # Generated visualizations (6 PNG files)
│   │   ├── performance_vs_sequence_length.png
│   │   ├── architecture_comparison.png
│   │   ├── optimizer_performance.png
│   │   ├── training_curves_best_worst.png
│   │   ├── activation_function_comparison.png
│   │   └── performance_heatmaps.png
│   └── *.pth                    # Saved model checkpoints (162 models)
├── create_plots.py               # Script to generate result visualizations
├── create_summary_table.py       # Script to generate summary metrics table
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── report_template.md            # Project report template
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv sentiment_analysis
source sentiment_analysis/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check CUDA availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Quick Start - Single Experiment

Run a single experiment with default LSTM configuration:

```bash
cd src
python train.py
```

### Custom Configuration

Run with specific parameters using a configuration file:

```bash
cd src
python train.py --config_file custom_config.json
```

Example configuration file:
```json
{
    "experiment_name": "lstm_relu_adam_50",
    "architecture": "lstm",
    "activation": "relu",
    "optimizer": "adam",
    "sequence_length": 50,
    "gradient_clipping": true,
    "embedding_dim": 100,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001
}
```

### Full Comparative Study

Run all systematic experiments:

```bash
cd src
python train.py --run_all
```

This executes all 162 experiments (3 architectures × 3 activations × 3 optimizers × 3 sequence lengths × 2 clipping options).

### Running Specific Experiments

```bash
# Run experiment by ID (0-161)
python train.py --experiment_id 42

# Use custom directories
python train.py --data_dir /path/to/data --results_dir /path/to/results
```

## Key Components

### 1. Data Preprocessing (`preprocess.py`)
- IMDb dataset loading from CSV files
- Text cleaning and normalization
- Tokenization and vocabulary building (top 10K words)
- Sequence padding/truncation for multiple length configurations
- Dataset statistics generation

### 2. Model Architectures (`models.py`)
- **SentimentRNN**: Basic RNN with configurable activation functions
- **SentimentLSTM**: LSTM model with enhanced memory capabilities
- **SentimentBiLSTM**: Bidirectional LSTM for improved context understanding

### 3. Training Pipeline (`train.py`)
- Systematic experiment management across all configurations
- Early stopping with patience for optimal convergence
- Model checkpointing and comprehensive logging
- GPU/CPU compatibility with automatic device selection

### 4. Evaluation Metrics (`evaluate.py`)
- Accuracy, F1-Score (macro and weighted)
- Precision, Recall, and ROC AUC
- Confusion matrices and classification reports
- Training time tracking

### 5. Utilities (`utils.py`)
- Reproducibility functions with fixed random seeds
- Visualization tools for training curves and comparisons
- Results analysis and summary generation

## Performance Requirements

**Hardware:**
- CPU: Any modern multi-core processor
- RAM: 8GB minimum (16GB recommended for full experiments)
- Storage: 2GB for dataset and results
- GPU: Optional (CUDA-compatible for 3-5x speedup)

**Runtime Estimates:**
- Single experiment: 3-10 minutes (GPU), 10-30 minutes (CPU)
- Full study (162 experiments): 2-6 hours (GPU), 8-24 hours (CPU)

## Results Analysis

### Pre-computed Results Available

All 162 experiments have been completed with results available in:
- `results/experiment_summary.json`: Complete experimental data
- `results/metrics.csv`: Summary table with all configurations and performance metrics
- `results/plots/`: Six comprehensive visualization plots

### Accessing Results

Load and analyze the complete experimental results:

```python
import json
import pandas as pd

# Load complete results
with open('results/experiment_summary.json', 'r') as f:
    results = json.load(f)

# Load summary table
df = pd.read_csv('results/metrics.csv')

# View top 10 performing models
print(df.head(10))

# Find best configuration
best_exp = max(results, key=lambda x: x['test_accuracy'])
print(f"Best accuracy: {best_exp['test_accuracy']:.4f}")
```

### Generate Additional Plots

To recreate or generate additional plots:

```bash
# Generate all experimental plots
python3 create_plots.py

# Generate summary table
python3 create_summary_table.py
```

### Available Visualizations

1. **Performance vs Sequence Length**: Shows dramatic impact of sequence length
2. **Architecture Comparison**: Box plots comparing RNN, LSTM, BiLSTM
3. **Optimizer Performance**: Bar charts showing RMSprop superiority
4. **Training Curves**: Best vs worst model training progression
5. **Activation Function Comparison**: Performance across ReLU, Sigmoid, Tanh
6. **Performance Heatmaps**: Hyperparameter interaction analysis

## Reproducibility

- All experiments use fixed random seeds (42) for consistent results
- Dependencies are pinned in requirements.txt
- Complete configuration logging for every experiment
- Deterministic data preprocessing pipeline

## Troubleshooting

**Memory Issues:**
```bash
# Reduce batch size
python train.py --batch_size 16
```

**CUDA Issues:**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python train.py
```

**Dataset Download Issues:**
Ensure you have the IMDb dataset CSV files in the `data/IMDB/` directory.

## License

This project is for educational purposes as part of a machine learning course assignment.