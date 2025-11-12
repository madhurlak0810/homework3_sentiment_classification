"""
Data preprocessing module for sentiment classification.
Handles IMDb dataset loading, cleaning, tokenization, and sequence preparation.
"""

import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize

# Set random seeds for reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class SentimentDataset(Dataset):
    """Custom dataset class for sentiment classification."""
    
    def __init__(self, texts: List[List[int]], labels: List[int]):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


class TextPreprocessor:
    """Handles all text preprocessing operations."""
    
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.vocab_size = 0
        self.tokenizer = word_tokenize
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize cleaned text."""
        clean_text = self.clean_text(text)
        tokens = self.tokenizer(clean_text)
        return tokens
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from training texts."""
        print("Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize_text(text)
            word_counts.update(tokens)
        
        # Create vocabulary with most frequent words
        # Reserve indices for special tokens
        self.vocab_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        
        # Add most frequent words
        most_common_words = word_counts.most_common(self.max_vocab_size - 2)
        for word, _ in most_common_words:
            self.vocab_to_idx[word] = len(self.vocab_to_idx)
        
        # Create reverse mapping
        self.idx_to_vocab = {idx: word for word, idx in self.vocab_to_idx.items()}
        self.vocab_size = len(self.vocab_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of token indices."""
        tokens = self.tokenize_text(text)
        sequence = []
        for token in tokens:
            if token in self.vocab_to_idx:
                sequence.append(self.vocab_to_idx[token])
            else:
                sequence.append(self.vocab_to_idx['<UNK>'])  # Unknown token
        return sequence
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int) -> List[List[int]]:
        """Pad or truncate sequences to fixed length."""
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) > max_length:
                # Truncate
                padded_sequence = sequence[:max_length]
            else:
                # Pad
                padded_sequence = sequence + [self.vocab_to_idx['<PAD>']] * (max_length - len(sequence))
            padded_sequences.append(padded_sequence)
        return padded_sequences
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'vocab_to_idx': self.vocab_to_idx,
            'idx_to_vocab': self.idx_to_vocab,
            'vocab_size': self.vocab_size,
            'max_vocab_size': self.max_vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.vocab_to_idx = vocab_data['vocab_to_idx']
        self.idx_to_vocab = vocab_data['idx_to_vocab']
        self.vocab_size = vocab_data['vocab_size']
        self.max_vocab_size = vocab_data['max_vocab_size']
        print(f"Vocabulary loaded from {filepath}")


def download_imdb_dataset(data_dir: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load IMDb dataset from CSV file."""
    print("Loading IMDb dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Look for the CSV file in the data directory
    csv_path = os.path.join(data_dir, 'IMDB Dataset.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IMDb dataset not found at {csv_path}. Please ensure the dataset file exists.")
    
    print(f"Loading from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples in dataset: {len(df)}")
    
    # Shuffle the dataset to ensure random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into training and test sets (50/50 split as specified)
    split_idx = len(df) // 2
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:split_idx*2]
    
    # Extract texts and labels
    train_texts = train_df['review'].tolist()
    train_labels = [1 if label == 'positive' else 0 for label in train_df['sentiment']]
    
    test_texts = test_df['review'].tolist()
    test_labels = [1 if label == 'positive' else 0 for label in test_df['sentiment']]
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Training positive samples: {sum(train_labels)}")
    print(f"Training negative samples: {len(train_labels) - sum(train_labels)}")
    
    return train_texts, train_labels, test_texts, test_labels


def preprocess_data(data_dir: str, sequence_lengths: List[int] = [25, 50, 100], sample_size: int = None) -> Dict:
    """Complete preprocessing pipeline."""
    # Download dataset
    train_texts, train_labels, test_texts, test_labels = download_imdb_dataset(data_dir)
    
    # For quick testing, use a smaller sample
    if sample_size:
        train_texts = train_texts[:sample_size]
        train_labels = train_labels[:sample_size]
        test_texts = test_texts[:sample_size//2]
        test_labels = test_labels[:sample_size//2]
        print(f"Using sample size: {len(train_texts)} training, {len(test_texts)} test")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=5000)  # Smaller vocab for faster processing
    
    # Build vocabulary from training data
    preprocessor.build_vocabulary(train_texts)
    
    # Save vocabulary
    vocab_path = os.path.join(data_dir, 'vocabulary.pkl')
    preprocessor.save_vocabulary(vocab_path)
    
    # Convert texts to sequences
    print("Converting texts to sequences...")
    train_sequences = [preprocessor.text_to_sequence(text) for text in train_texts]
    test_sequences = [preprocessor.text_to_sequence(text) for text in test_texts]
    
    # Prepare data for different sequence lengths
    processed_data = {}
    
    for seq_len in sequence_lengths:
        print(f"Processing sequences for length {seq_len}...")
        
        # Pad sequences
        train_padded = preprocessor.pad_sequences(train_sequences, seq_len)
        test_padded = preprocessor.pad_sequences(test_sequences, seq_len)
        
        # Create datasets
        train_dataset = SentimentDataset(train_padded, train_labels)
        test_dataset = SentimentDataset(test_padded, test_labels)
        
        processed_data[seq_len] = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset)
        }
    
    # Add vocabulary info
    processed_data['vocab_size'] = preprocessor.vocab_size
    processed_data['preprocessor'] = preprocessor
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    
    # Calculate average sequence length
    original_lengths = [len(preprocessor.tokenize_text(text)) for text in train_texts[:1000]]  # Sample for efficiency
    avg_length = np.mean(original_lengths)
    print(f"Average review length (first 1000 samples): {avg_length:.1f} words")
    
    return processed_data


def get_data_loaders(dataset: SentimentDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create DataLoader for a dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    data_dir = "../data"
    
    # Preprocess data for all sequence lengths
    processed_data = preprocess_data(data_dir)
    
    # Example: Create data loaders for sequence length 50
    train_loader = get_data_loaders(processed_data[50]['train_dataset'], batch_size=32, shuffle=True)
    test_loader = get_data_loaders(processed_data[50]['test_dataset'], batch_size=32, shuffle=False)
    
    print(f"\nDataLoader created:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Show a sample batch
    for texts, labels in train_loader:
        print(f"Sample batch shape - Texts: {texts.shape}, Labels: {labels.shape}")
        break