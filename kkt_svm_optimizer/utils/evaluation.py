"""
Evaluation and Benchmarking Module
Compares KKT-guided SVM against standard implementations.
"""

import numpy as np
import time
from typing import Dict, Tuple
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)


class SVMBenchmark:
    """
    Benchmark KKT-guided SVM against standard implementations.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize benchmark suite.
        
        Args:
            verbose: Print detailed results
        """
        self.verbose = verbose
        self.results = {}
    
    def run_sklearn_svm(self, 
                       X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       C: float = 1.0) -> Dict:
        """
        Train and evaluate sklearn's SVM.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            C: Regularization parameter
            
        Returns:
            dict with training time, predictions, and metrics
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Training Sklearn SVM (C={})".format(C))
            print("="*60)
        
        # Convert labels to 0/1 for sklearn
        y_train_01 = np.where(y_train == -1, 0, 1)
        y_test_01 = np.where(y_test == -1, 0, 1)
        
        # Initialize SVM
        svm = SVC(kernel='linear', C=C, max_iter=1000, verbose=0)
        
        # Train
        start_time = time.time()
        svm.fit(X_train, y_train_01)
        train_time = time.time() - start_time
        
        # Predict
        predictions = svm.predict(X_test)
        decision_values = svm.decision_function(X_test)
        
        # Evaluate
        metrics = self._compute_metrics(y_test_01, predictions, decision_values)
        
        result = {
            'train_time': train_time,
            'n_support_vectors': len(svm.support_vectors_),
            'predictions': predictions,
            'decision_values': decision_values,
            'metrics': metrics,
            'model': svm
        }
        
        if self.verbose:
            print(f"Training time: {train_time:.2f}s")
            print(f"Support vectors: {result['n_support_vectors']}")
            self._print_metrics("Sklearn SVM", metrics)
        
        return result
    
    def run_kkt_svm(self, 
                   kkt_svm_model,
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_test: np.ndarray, 
                   y_test: np.ndarray) -> Dict:
        """
        Train and evaluate KKT-guided SVM.
        
        Args:
            kkt_svm_model: Instance of KKTGuidedSVM
            X_train, y_train: Training data (labels should be ±1)
            X_test, y_test: Test data
            
        Returns:
            dict with training time, predictions, and metrics
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Training KKT-Guided SVM")
            print("="*60)
        
        # Train
        start_time = time.time()
        kkt_svm_model.fit(X_train, y_train)
        train_time = kkt_svm_model.train_time
        
        # Predict
        predictions = kkt_svm_model.predict(X_test)
        decision_values = kkt_svm_model.decision_function(X_test)
        
        # Convert to 0/1 for metrics
        predictions_01 = np.where(predictions == -1, 0, 1)
        y_test_01 = np.where(y_test == -1, 0, 1)
        
        # Evaluate
        metrics = self._compute_metrics(y_test_01, predictions_01, decision_values)
        
        result = {
            'train_time': train_time,
            'n_support_vectors': kkt_svm_model.get_n_support_vectors(),
            'predictions': predictions_01,
            'decision_values': decision_values,
            'metrics': metrics,
            'model': kkt_svm_model,
            'training_summary': kkt_svm_model.get_training_summary()
        }
        
        if self.verbose:
            print(f"Training time: {train_time:.2f}s")
            print(f"Support vectors: {result['n_support_vectors']}")
            self._print_metrics("KKT-Guided SVM", metrics)
        
        return result
    
    def _compute_metrics(self, y_true: np.ndarray, 
                        predictions: np.ndarray,
                        decision_values: np.ndarray) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (0/1)
            predictions: Predicted labels (0/1)
            decision_values: Decision function values
            
        Returns:
            dict with various metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
        }
        
        # ROC-AUC (only if both classes present)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, decision_values)
        else:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _print_metrics(self, model_name: str, metrics: Dict):
        """Print metrics in formatted way."""
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    def compare_implementations(self, 
                               kkt_result: Dict, 
                               sklearn_result: Dict) -> Dict:
        """
        Compare KKT-guided SVM with sklearn SVM.
        
        Args:
            kkt_result: Result from run_kkt_svm
            sklearn_result: Result from run_sklearn_svm
            
        Returns:
            dict with comparison analysis
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Comparison: KKT-Guided SVM vs Sklearn SVM")
            print("="*60)
        
        # Time comparison
        time_reduction = (sklearn_result['train_time'] - kkt_result['train_time']) / sklearn_result['train_time']
        
        # Performance comparison
        kkt_f1 = kkt_result['metrics']['f1_score']
        sklearn_f1 = sklearn_result['metrics']['f1_score']
        f1_difference = kkt_f1 - sklearn_f1
        f1_relative_diff = f1_difference / sklearn_f1 if sklearn_f1 > 0 else 0
        
        kkt_auc = kkt_result['metrics']['roc_auc']
        sklearn_auc = sklearn_result['metrics']['roc_auc']
        auc_difference = kkt_auc - sklearn_auc
        
        comparison = {
            'time_reduction': time_reduction,
            'time_reduction_percent': time_reduction * 100,
            'speedup_factor': sklearn_result['train_time'] / kkt_result['train_time'],
            'kkt_train_time': kkt_result['train_time'],
            'sklearn_train_time': sklearn_result['train_time'],
            'kkt_f1': kkt_f1,
            'sklearn_f1': sklearn_f1,
            'f1_difference': f1_difference,
            'f1_relative_difference_percent': f1_relative_diff * 100,
            'kkt_roc_auc': kkt_auc,
            'sklearn_roc_auc': sklearn_auc,
            'auc_difference': auc_difference,
            'kkt_n_sv': kkt_result['n_support_vectors'],
            'sklearn_n_sv': sklearn_result['n_support_vectors'],
        }
        
        if self.verbose:
            print(f"\nTraining Time Comparison:")
            print(f"  Sklearn:      {sklearn_result['train_time']:.4f}s")
            print(f"  KKT-Guided:   {kkt_result['train_time']:.4f}s")
            print(f"  Time Reduction: {time_reduction*100:.1f}%")
            print(f"  Speedup Factor: {comparison['speedup_factor']:.2f}x")
            
            print(f"\nSupport Vectors:")
            print(f"  Sklearn:      {comparison['sklearn_n_sv']}")
            print(f"  KKT-Guided:   {comparison['kkt_n_sv']}")
            
            print(f"\nF1-Score Comparison:")
            print(f"  Sklearn:      {sklearn_f1:.4f}")
            print(f"  KKT-Guided:   {kkt_f1:.4f}")
            print(f"  Difference:   {f1_difference:+.4f} ({f1_relative_diff*100:+.2f}%)")
            
            print(f"\nROC-AUC Comparison:")
            print(f"  Sklearn:      {sklearn_auc:.4f}")
            print(f"  KKT-Guided:   {kkt_auc:.4f}")
            print(f"  Difference:   {auc_difference:+.4f}")
            
            # Check objectives
            print(f"\nObjective Achievement:")
            achieved_speedup = time_reduction >= 0.40
            maintained_f1 = abs(f1_relative_diff) <= 0.05  # Within 5% of sklearn
            print(f"  ✓ 40% Time Reduction: {'YES' if achieved_speedup else 'NO'} ({time_reduction*100:.1f}%)")
            print(f"  ✓ Maintain F1-Score:  {'YES' if maintained_f1 else 'NO'} ({f1_relative_diff*100:+.2f}%)")
            print("="*60)
        
        return comparison


class PerformanceAnalyzer:
    """
    Detailed performance analysis and visualization helpers.
    """
    
    @staticmethod
    def print_detailed_analysis(kkt_result: Dict, 
                               sklearn_result: Dict,
                               comparison: Dict):
        """
        Print detailed analysis of both models.
        """
        print("\n" + "="*60)
        print("Detailed Performance Analysis")
        print("="*60)
        
        # KKT Model Internal Details
        if 'training_summary' in kkt_result:
            ts = kkt_result['training_summary']
            print(f"\nKKT-Guided SVM Internal Details:")
            print(f"  Core SVs:           {ts.get('core_svs', 'N/A')}")
            print(f"  Boundary SVs:       {ts.get('boundary_svs', 'N/A')}")
            print(f"  Total SVs:          {ts.get('n_support_vectors', 'N/A')}")
            print(f"  Final Loss:         {ts.get('final_loss', 'N/A'):.6f}")
            print(f"  Iterations:         {ts.get('total_iterations', 'N/A')}")
        
        # Detailed Metrics
        print(f"\nDetailed Metrics (KKT vs Sklearn):")
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics_to_compare:
            kkt_val = kkt_result['metrics'].get(metric, 0)
            sklearn_val = sklearn_result['metrics'].get(metric, 0)
            diff = kkt_val - sklearn_val
            print(f"  {metric:12s}: {kkt_val:.4f} vs {sklearn_val:.4f} (diff: {diff:+.4f})")
        
        # Decision Boundary Similarity
        print(f"\nDecision Boundary Comparison:")
        kkt_decisions = kkt_result['decision_values']
        sklearn_decisions = sklearn_result['decision_values']
        
        # Normalize for comparison
        kkt_norm = (kkt_decisions - np.mean(kkt_decisions)) / (np.std(kkt_decisions) + 1e-8)
        sklearn_norm = (sklearn_decisions - np.mean(sklearn_decisions)) / (np.std(sklearn_decisions) + 1e-8)
        
        correlation = np.corrcoef(kkt_norm, sklearn_norm)[0, 1]
        print(f"  Decision value correlation: {correlation:.4f}")
        
        # Agreement Analysis
        kkt_preds = kkt_result['predictions']
        sklearn_preds = sklearn_result['predictions']
        agreement = np.mean(kkt_preds == sklearn_preds)
        print(f"  Prediction agreement:       {agreement:.4f}")
        
        print("="*60)
