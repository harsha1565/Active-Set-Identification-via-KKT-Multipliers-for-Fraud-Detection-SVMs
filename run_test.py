#!/usr/bin/env python
"""
KKT-Guided SVM Optimizer - Complete Pipeline Test
"""

import sys
sys.path.insert(0, r'c:\Users\Dharnish\Documents\Mini project')

import numpy as np
from kkt_svm_optimizer.data.data_loader import FraudDataLoader
from kkt_svm_optimizer.models.svm_model import KKTGuidedSVM
from kkt_svm_optimizer.utils.evaluation import SVMBenchmark

print("=" * 70)
print("KKT-GUIDED SVM OPTIMIZER - COMPLETE PIPELINE TEST")
print("=" * 70 + "\n")

# Load dataset
print("[1/7] Loading dataset...")
data_loader = FraudDataLoader(random_state=42)
X, y = data_loader.create_synthetic_imbalanced_dataset(
    n_samples=5000, n_features=20, imbalance_ratio=0.02, random_state=42)
print(f"Dataset: {X.shape}, Fraud: {np.sum(y==1)}\n")

# Prepare data
print("[2/7] Preparing data...")
prepared = data_loader.prepare_data(X, y, test_size=0.2, val_size=0.1)
X_train = prepared['X_train']
y_train = prepared['y_train_svm']
X_test = prepared['X_test']
y_test = prepared['y_test_svm']
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# Train KKT-Guided SVM
print("[3/7] Training KKT-Guided SVM...")
kkt_svm = KKTGuidedSVM(
    C=1.0, optimizer_method='nesterov', max_iter=300,
    tol=1e-4, use_kkt_filtering=True, verbose=False, random_state=42)
kkt_svm.fit(X_train, y_train)
print(f"Training time: {kkt_svm.train_time:.4f}s")
print(f"Support vectors: {kkt_svm.get_n_support_vectors()}\n")

# Evaluate
print("[4/7] Evaluating KKT-Guided SVM...")
kkt_predictions = kkt_svm.predict(X_test)
kkt_metrics = kkt_svm.evaluate(X_test, y_test)
print(f"F1-Score: {kkt_metrics['f1_score']:.4f}")
print(f"ROC-AUC: {kkt_metrics['roc_auc']:.4f}\n")

# Train baseline
print("[5/7] Training Sklearn SVM baseline...")
benchmark = SVMBenchmark(verbose=False)
y_train_01 = np.where(y_train == -1, 0, 1)
y_test_01 = np.where(y_test == -1, 0, 1)

print(f"Train classes: {np.unique(y_train_01)}, Test classes: {np.unique(y_test_01)}")

sklearn_result = benchmark.run_sklearn_svm(
    X_train, y_train_01, X_test, y_test_01, C=1.0)
print(f"Training time: {sklearn_result['train_time']:.4f}s")
print(f"Support vectors: {sklearn_result['n_support_vectors']}\n")

# Compare
print("[6/7] Comparing implementations...")
kkt_result = {
    'train_time': kkt_svm.train_time,
    'n_support_vectors': kkt_svm.get_n_support_vectors(),
    'predictions': np.where(kkt_predictions == -1, 0, 1),
    'decision_values': kkt_svm.decision_function(X_test),
    'metrics': kkt_metrics,
    'model': kkt_svm,
}

comparison = benchmark.compare_implementations(kkt_result, sklearn_result)

print(f"Time Reduction: {comparison['time_reduction']*100:.1f}%")
print(f"Speedup: {comparison['speedup_factor']:.2f}x")
print(f"F1-Score Diff: {comparison['f1_relative_difference_percent']:+.2f}%\n")

# Summary
print("[7/7] Results Summary")
print("=" * 70)
print(f"Time Reduction:        {comparison['time_reduction']*100:.1f}% (target: >=40%)")
print(f"F1-Score Preservation: {comparison['f1_relative_difference_percent']:+.2f}% (target: +-5%)")
print(f"Speedup Factor:        {comparison['speedup_factor']:.2f}x")
print(f"\nKKT-Guided SVM:  {kkt_result['train_time']:.4f}s, {comparison['kkt_f1']:.4f} F1")
print(f"Sklearn SVM:     {sklearn_result['train_time']:.4f}s, {comparison['sklearn_f1']:.4f} F1")

objectives_pass = (comparison['time_reduction'] >= 0.40 and 
                   abs(comparison['f1_relative_difference_percent']) <= 5)

print("\n" + "=" * 70)
if objectives_pass:
    print("PASS: ALL OBJECTIVES ACHIEVED!")
else:
    print("INFO: Objectives require optimization")
print("=" * 70)
