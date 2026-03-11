"""
KKT-Guided SVM Model
Integrates KKT conditions and accelerated optimization for fraud detection.
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from kkt_svm_optimizer.core.optimizer import AcceleratedGradientOptimizer
from kkt_svm_optimizer.core.kkt_conditions import KKTConditions, KKTMonitor


class KKTGuidedSVM:
    """
    Support Vector Machine with KKT-guided active set identification.
    Dramatically reduces computational complexity for imbalanced datasets.
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 kernel: str = 'linear',
                 optimizer_method: str = 'nesterov',
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 use_kkt_filtering: bool = True,
                 kkt_check_frequency: int = 10,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize KKT-guided SVM.
        
        Args:
            C: Regularization parameter (inverse of regularization strength)
            kernel: 'linear' (currently supported)
            optimizer_method: 'sgd', 'momentum', 'nesterov', or 'fista'
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            use_kkt_filtering: Enable KKT-guided active set filtering
            kkt_check_frequency: Check KKT every N iterations
            random_state: Random seed for reproducibility
            verbose: Print progress
        """
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.use_kkt_filtering = use_kkt_filtering
        self.kkt_check_frequency = kkt_check_frequency
        self.random_state = random_state
        self.verbose = verbose
        
        assert kernel == 'linear', "Currently only linear kernel is supported"
        
        # Parameters (learned during training)
        self.w = None
        self.b = None
        self.support_vectors_ = None
        self.n_support_ = 0
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Optimizer
        self.optimizer = AcceleratedGradientOptimizer(
            method=optimizer_method,
            learning_rate=0.01,
            momentum=0.9,
            max_iter=max_iter,
            tol=tol,
            use_kkt_filtering=use_kkt_filtering,
            kkt_check_frequency=kkt_check_frequency,
            verbose=verbose
        )
        
        # KKT monitoring
        self.kkt_monitor = KKTMonitor()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KKTGuidedSVM':
        """
        Train the SVM on the given dataset.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - must be binary {-1, 1} or {0, 1}
            
        Returns:
            self: Fitted SVM
        """
        # Convert labels to ±1 if needed
        if not np.all(np.isin(y, [-1, 1])):
            y_unique = np.unique(y)
            assert len(y_unique) == 2, "SVM requires binary classification"
            y = np.where(y == y_unique[0], -1, 1)
        
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        
        if self.verbose:
            print("="*60)
            print("KKT-Guided SVM Training")
            print("="*60)
            print(f"Dataset shape: {X.shape}")
            print(f"Regularization (C): {self.C}")
            print(f"KKT filtering enabled: {self.use_kkt_filtering}")
            print(f"Optimizer method: {self.optimizer.method}")
            print("-"*60)
        
        # Optimize with accelerated gradient descent
        start_time = time.time()
        self.w, self.b, opt_history = self.optimizer.optimize(
            X_scaled, y, C=self.C
        )
        train_time = time.time() - start_time
        
        # Identify support vectors using KKT conditions
        kkt_checker = KKTConditions(tol=self.tol)
        margins = y * (X_scaled @ self.w + self.b)
        margin_violations = kkt_checker.compute_margin_violations(X_scaled, y, self.w, self.b, self.C)
        alpha = self.optimizer.kkt_checker.compute_dual_variables(
            margin_violations, np.zeros(len(y)), self.C, eta=0.01
        )
        
        sv_info = kkt_checker.identify_support_vectors(alpha, margin_violations, self.C)
        self.support_vectors_ = X_scaled[sv_info['all_sv_mask']]
        self.n_support_ = len(self.support_vectors_)
        
        # Store history
        self.opt_history = opt_history
        self.sv_info = sv_info
        self.train_time = train_time
        
        # Print training summary
        if self.verbose:
            print(f"Training time: {train_time:.2f}s")
            print(f"Support vectors identified: {self.n_support_}")
            print(f"  - Core SVs: {sv_info['n_core_sv']}")
            print(f"  - Boundary SVs: {sv_info['n_boundary_sv']}")
            print(f"  - Non-SVs: {sv_info['n_non_sv']}")
            print(f"Final loss: {opt_history['loss'][-1]:.6f}")
            print(f"Final gradient norm: {opt_history['gradient_norm'][-1]:.6f}")
            print("="*60)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            predictions: Class labels {-1, 1}
        """
        assert self.is_fitted, "Model must be fitted before prediction"
        
        X_scaled = self.scaler.transform(X)
        decision = X_scaled @ self.w + self.b
        predictions = np.sign(decision)
        predictions[predictions == 0] = 1  # Handle boundary cases
        
        return predictions.astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values (distances from hyperplane).
        
        Args:
            X: Features
            
        Returns:
            decision_values: Distance from hyperplane
        """
        assert self.is_fitted, "Model must be fitted before prediction"
        
        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.w + self.b
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates using sigmoid on decision function.
        
        Args:
            X: Features
            
        Returns:
            probabilities: Array of shape (n_samples, 2) with class probabilities
        """
        decision = self.decision_function(X)
        # Apply sigmoid: 1 / (1 + exp(-z))
        proba_positive = 1.0 / (1.0 + np.exp(-decision))
        proba_negative = 1.0 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            X: Test features
            y: Test labels (binary {0,1} or {-1,1})
            
        Returns:
            dict: Dictionary with various metrics
        """
        assert self.is_fitted, "Model must be fitted before evaluation"
        
        # Normalize labels to ±1
        if not np.all(np.isin(y, [-1, 1])):
            y_norm = np.where(y == 0, -1, 1)
        else:
            y_norm = y.copy()
        
        # Make predictions
        predictions = self.predict(X)
        decision_values = self.decision_function(X)
        
        # Convert predictions to 0/1 for metrics if needed
        pred_01 = np.where(predictions == -1, 0, 1)
        y_01 = np.where(y_norm == -1, 0, 1)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_01, pred_01),
            'precision': precision_score(y_01, pred_01, zero_division=0),
            'recall': recall_score(y_01, pred_01, zero_division=0),
            'f1_score': f1_score(y_01, pred_01, zero_division=0),
            'roc_auc': roc_auc_score(y_01, decision_values) if len(np.unique(y_01)) > 1 else 0.0,
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_01, pred_01)
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]
        
        return metrics
    
    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors."""
        assert self.is_fitted, "Model must be fitted first"
        return self.support_vectors_
    
    def get_n_support_vectors(self) -> int:
        """Get number of support vectors."""
        assert self.is_fitted, "Model must be fitted first"
        return self.n_support_
    
    def get_training_summary(self) -> Dict:
        """Get summary of training process."""
        assert self.is_fitted, "Model must be fitted first"
        
        return {
            'training_time': self.train_time,
            'n_support_vectors': self.n_support_,
            'core_svs': self.sv_info['n_core_sv'],
            'boundary_svs': self.sv_info['n_boundary_sv'],
            'final_loss': self.opt_history['loss'][-1],
            'final_gradient_norm': self.opt_history['gradient_norm'][-1],
            'total_iterations': len(self.opt_history['loss'])
        }
