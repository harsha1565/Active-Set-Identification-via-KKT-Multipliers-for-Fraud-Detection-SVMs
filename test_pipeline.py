#!/usr/bin/env python
"""
Quick Test Script for KKT-Guided SVM
Tests all components and runs the full pipeline
"""

import sys
import os

# Add the workspace to path
sys.path.insert(0, r'c:\Users\Dharnish\Documents\Mini project')

import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("KKT-GUIDED SVM OPTIMIZER - QUICK TEST")
print("="*70 + "\n")

# Test 1: Import modules
print("[TEST 1] Importing modules...")
try:
    from kkt_svm_optimizer.data.data_loader import FraudDataLoader
    from kkt_svm_optimizer.models.svm_model import KKTGuidedSVM
    from kkt_svm_optimizer.utils.evaluation import SVMBenchmark
    print("✓ All modules imported successfully\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Load and prepare dataset
print("[TEST 2] Loading dataset...")
try:
    data_loader = FraudDataLoader(random_state=42)
    X, y = data_loader.create_synthetic_imbalanced_dataset(
        n_samples=5000, n_features=20, imbalance_ratio=0.02)
    print(f"✓ Dataset created: {X.shape}")
    print(f"  Fraud samples: {np.sum(y==1)} ({np.mean(y==1)*100:.2f}%)\n")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}\n")
    sys.exit(1)

# Test 3: Prepare data
print("[TEST 3] Preparing data...")
try:
    prepared = data_loader.prepare_data(X, y, test_size=0.2, val_size=0.1)
    X_train = prepared['X_train']
    y_train = prepared['y_train_svm']
    X_test = prepared['X_test']
    y_test = prepared['y_test_svm']
    print(f"✓ Data prepared")
    print(f"  Training: {X_train.shape}, Test: {X_test.shape}\n")
except Exception as e:
    print(f"✗ Data preparation failed: {e}\n")
    sys.exit(1)

# Test 4: Train KKT-Guided SVM
print("[TEST 4] Training KKT-Guided SVM...")
try:
    kkt_svm = KKTGuidedSVM(
        C=1.0,
        optimizer_method='nesterov',
        max_iter=300,
        tol=1e-4,
        use_kkt_filtering=True,
        kkt_check_frequency=10,
        verbose=True,
        random_state=42
    )
    kkt_svm.fit(X_train, y_train)
    print(f"✓ KKT-Guided SVM trained successfully\n")
except Exception as e:
    print(f"✗ Training failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Evaluate
print("[TEST 5] Evaluating KKT-Guided SVM...")
try:
    kkt_predictions = kkt_svm.predict(X_test)
    kkt_metrics = kkt_svm.evaluate(X_test, y_test)
    print(f"✓ Evaluation complete")
    print(f"  F1-Score: {kkt_metrics['f1_score']:.4f}")
    print(f"  Accuracy: {kkt_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {kkt_metrics['roc_auc']:.4f}\n")
except Exception as e:
    print(f"✗ Evaluation failed: {e}\n")
    sys.exit(1)

# Test 6: Compare with Sklearn
print("[TEST 6] Training baseline Sklearn SVM...")
try:
    benchmark = SVMBenchmark(verbose=False)
    y_train_01 = np.where(y_train == -1, 0, 1)
    y_test_01 = np.where(y_test == -1, 0, 1)
    sklearn_result = benchmark.run_sklearn_svm(
        X_train, y_train_01, X_test, y_test_01, C=1.0
    )
    print(f"✓ Sklearn SVM trained")
    print(f"  F1-Score: {sklearn_result['metrics']['f1_score']:.4f}\n")
except Exception as e:
    print(f"✗ Sklearn training failed: {e}\n")
    sys.exit(1)

# Test 7: Compare
print("[TEST 7] Comparing implementations...")
try:
    kkt_result = {
        'train_time': kkt_svm.train_time,
        'n_support_vectors': kkt_svm.get_n_support_vectors(),
        'predictions': np.where(kkt_predictions == -1, 0, 1),
        'decision_values': kkt_svm.decision_function(X_test),
        'metrics': kkt_metrics,
        'model': kkt_svm,
        'training_summary': kkt_svm.get_training_summary()
    }
    
    comparison = benchmark.compare_implementations(kkt_result, sklearn_result)
    
    print(f"✓ Comparison complete")
    print(f"  Time Reduction: {comparison['time_reduction']*100:.1f}%")
    print(f"  Speedup: {comparison['speedup_factor']:.2f}x")
    print(f"  F1-Score Difference: {comparison['f1_relative_difference_percent']:+.2f}%\n")
except Exception as e:
    print(f"✗ Comparison failed: {e}\n")
    sys.exit(1)

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Time Reduction:          {comparison['time_reduction']*100:.1f}% (target: ≥40%)")
print(f"F1-Score Preservation:   {comparison['f1_relative_difference_percent']:+.2f}% (target: ±5%)")
print(f"Speedup Factor:          {comparison['speedup_factor']:.2f}x")
print(f"\nKKT-Guided SVM:  {kkt_result['train_time']:.4f}s, {comparison['kkt_f1']:.4f} F1")
print(f"Sklearn SVM:     {sklearn_result['train_time']:.4f}s, {comparison['sklearn_f1']:.4f} F1")

objectives_pass = (comparison['time_reduction'] >= 0.40 and 
                   abs(comparison['f1_relative_difference_percent']) <= 5)

print("\n" + "="*70)
if objectives_pass:
    print("✓ ALL OBJECTIVES ACHIEVED - PIPELINE WORKING CORRECTLY!")
else:
    print("⚠ Some objectives require optimization")
print("="*70)
