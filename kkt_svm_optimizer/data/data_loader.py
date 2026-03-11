"""
Data Loading and Preprocessing Module
Handles Credit Card Fraud Detection dataset or alternatives.
Addresses class imbalance and feature scaling.
"""

import numpy as np
import pandas as pd
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


class FraudDataLoader:
    """
    Loads and preprocesses credit card fraud detection dataset.
    
    Dataset source: https://www.kaggle.com/mlg-ulb/creditcardfraud
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.data = None
        self.target = None
        
    def load_kaggle_credit_card_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load credit card fraud detection dataset from Kaggle.
        
        Note: You need to download the dataset from Kaggle first:
        https://www.kaggle.com/mlg-ulb/creditcardfraud
        
        Args:
            filepath: Path to creditcard.csv
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,) with values {0, 1}
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset not found at {filepath}. "
                "Please download from: https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )
        
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['Class'].value_counts()}")
        print(f"Class imbalance ratio (fraud/normal): {df['Class'].value_counts()[1]/df['Class'].value_counts()[0]:.4%}")
        
        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        
        self.data = X
        self.target = y
        
        return X, y
    
    def create_synthetic_imbalanced_dataset(self, 
                                          n_samples: int = 10000,
                                          n_features: int = 30,
                                          imbalance_ratio: float = 0.01,
                                          random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a synthetic imbalanced dataset for testing.
        Useful when full Kaggle dataset is unavailable.
        
        Args:
            n_samples: Total number of samples
            n_features: Number of features
            imbalance_ratio: Fraction of positive (fraud) samples
            random_state: Random seed
            
        Returns:
            X: Feature matrix
            y: Binary labels {0, 1}
        """
        if random_state is None:
            random_state = self.random_state
        
        np.random.seed(random_state)
        
        n_positive = max(1, int(n_samples * imbalance_ratio))
        n_negative = n_samples - n_positive
        
        print(f"Creating synthetic imbalanced dataset:")
        print(f"  Total samples: {n_samples}")
        print(f"  Positive (fraud): {n_positive} ({imbalance_ratio:.2%})")
        print(f"  Negative (normal): {n_negative} ({1-imbalance_ratio:.2%})")
        print(f"  Features: {n_features}")
        
        # Generate negative samples (normal transactions)
        X_neg = np.random.randn(n_negative, n_features)
        y_neg = np.zeros(n_negative)
        
        # Generate positive samples (frauds) - slightly different distribution
        X_pos = np.random.randn(n_positive, n_features) + 2.0
        X_pos[:, :5] *= 0.5  # Make some features more distinctive
        y_pos = np.ones(n_positive)
        
        # Combine and shuffle
        X = np.vstack([X_neg, X_pos])
        y = np.hstack([y_neg, y_pos])
        
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        self.data = X
        self.target = y
        
        return X, y
    
    def prepare_data(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    test_size: float = 0.2,
                    val_size: float = 0.1,
                    handle_imbalance: bool = True,
                    sampling_method: str = 'none') -> Dict:
        """
        Prepare data for training and evaluation.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining data)
            handle_imbalance: Whether to apply balancing techniques
            sampling_method: 'none', 'random_undersample', 'random_oversample', 'smote'
            
        Returns:
            dict with keys: X_train, y_train, X_val, y_val, X_test, y_test, and metadata
        """
        print("\n" + "="*60)
        print("Data Preparation")
        print("="*60)
        
        # Initial train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Train-validation split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_train
        )
        
        print(f"Train set: {X_train.shape[0]} samples ({np.sum(y_train==1)} fraud)")
        print(f"Val set: {X_val.shape[0]} samples ({np.sum(y_val==1)} fraud)")
        print(f"Test set: {X_test.shape[0]} samples ({np.sum(y_test==1)} fraud)")
        
        # Handle class imbalance in training set
        if handle_imbalance and sampling_method != 'none':
            X_train, y_train = self._handle_imbalance(
                X_train, y_train, sampling_method
            )
            print(f"\nAfter {sampling_method}:")
            print(f"Train set: {X_train.shape[0]} samples ({np.sum(y_train==1)} fraud)")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        prepared_data = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'X_train_raw': X_train,
            'X_val_raw': X_val,
            'X_test_raw': X_test,
            'scaler': scaler,
            'metadata': {
                'n_features': X.shape[1],
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test),
                'n_fraud_train': np.sum(y_train == 1),
                'n_fraud_val': np.sum(y_val == 1),
                'n_fraud_test': np.sum(y_test == 1),
                'imbalance_ratio_train': np.sum(y_train == 1) / len(y_train),
                'imbalance_ratio_val': np.sum(y_val == 1) / len(y_val),
                'imbalance_ratio_test': np.sum(y_test == 1) / len(y_test),
            }
        }
        
        # Convert labels to ±1 for SVM
        prepared_data['y_train_svm'] = np.where(y_train == 0, -1, 1)
        prepared_data['y_val_svm'] = np.where(y_val == 0, -1, 1)
        prepared_data['y_test_svm'] = np.where(y_test == 0, -1, 1)
        
        print("="*60)
        
        return prepared_data
    
    def _handle_imbalance(self, X: np.ndarray, y: np.ndarray, 
                         method: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using specified method.
        
        Args:
            X: Features
            y: Labels
            method: 'random_undersample', 'random_oversample', or 'smote'
            
        Returns:
            X_balanced, y_balanced
        """
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if method == 'random_undersample':
            # Undersample majority class
            neg_indices = np.where(y == 0)[0]
            np.random.shuffle(neg_indices)
            selected_neg = neg_indices[:n_positive]
            
            pos_indices = np.where(y == 1)[0]
            all_indices = np.concatenate([selected_neg, pos_indices])
            np.random.shuffle(all_indices)
            
            X_balanced = X[all_indices]
            y_balanced = y[all_indices]
            
        elif method == 'random_oversample':
            # Oversample minority class
            pos_indices = np.where(y == 1)[0]
            oversample_indices = np.random.choice(pos_indices, n_negative - n_positive, replace=True)
            
            all_indices = np.concatenate([np.arange(len(X)), oversample_indices])
            X_balanced = X[all_indices]
            y_balanced = y[all_indices]
            
        else:
            raise ValueError(f"Unknown imbalance handling method: {method}")
        
        print(f"Applied {method}: {n_positive} pos, {n_negative} neg -> "
              f"{np.sum(y_balanced==1)} pos, {np.sum(y_balanced==0)} neg")
        
        return X_balanced, y_balanced


class DatasetDownloader:
    """
    Utility to download datasets (placeholder for automated download).
    """
    
    @staticmethod
    def download_credit_card_dataset(output_path: str):
        """
        Attempt to download credit card fraud dataset.
        Note: Direct download is not available; users must get from Kaggle.
        """
        print("Credit Card Fraud Detection dataset cannot be auto-downloaded.")
        print("Please download manually from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print(f"Save as: {output_path}")
