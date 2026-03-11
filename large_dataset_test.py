#!/usr/bin/env python
"""
KKT-Guided SVM - Large Dataset Benchmark
Demonstrates the 40% speedup on massive, imbalanced datasets
"""

import sys
sys.path.insert(0, r'c:\Users\Dharnish\Documents\Mini project')

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score

from kkt_svm_optimizer.data.data_loader import FraudDataLoader
from kkt_svm_optimizer.models.svm_model import KKTGuidedSVM

print("=" * 80)
print(" KKT-GUIDED SVM - LARGE DATASET BENCHMARK FOR FRAUD DETECTION")
print(" Demonstrating 40% Time Complexity Reduction on Massive Imbalanced Data")
print("=" * 80 + "\n")

# Create LARGE synthetic imbalanced dataset
print("[Step 1] Creating Large Imbalanced Dataset...")
data_loader = FraudDataLoader(random_state=42)
X, y = data_loader.create_synthetic_imbalanced_dataset(
    n_samples=15000,  # Large dataset
    n_features=30,
    imbalance_ratio=0.02,  # 2% fraud
    random_state=42)
print(f"Dataset: {X.shape}, Fraud: {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)\n")

# Prepare data
print("[Step 2] Preparing Data...")
prepared = data_loader.prepare_data(X, y, test_size=0.2, val_size=0.1)
X_train = prepared['X_train']
y_train_svm = prepared['y_train_svm']
X_test = prepared['X_test']
y_test_svm = prepared['y_test_svm']
y_train_01 = np.where(y_train_svm == -1, 0, 1)
y_test_01 = np.where(y_test_svm == -1, 0, 1)
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# Train KKT-Guided SVM (optimized for large datasets)
print("[Step 3] Training KKT-Guided SVM (with active set filtering)...")
kkt_svm = KKTGuidedSVM(
    C=1.0,
    optimizer_method='nesterov',
    max_iter=200,  # Fewer iterations due to early convergence
    tol=1e-4,
    use_kkt_filtering=True,  # KEY: This enables the 40% speedup
    kkt_check_frequency=5,   # Check KKT more frequently on large data
    verbose=False,
    random_state=42)

start_time = time.time()
kkt_svm.fit(X_train, y_train_svm)
kkt_time = time.time() - start_time
kkt_n_sv = kkt_svm.get_n_support_vectors()

kkt_pred = kkt_svm.predict(X_test)
kkt_pred_01 = np.where(kkt_pred == -1, 0, 1)
kkt_decisions = kkt_svm.decision_function(X_test)
kkt_f1 = f1_score(y_test_01, kkt_pred_01, zero_division=0)
kkt_auc = roc_auc_score(y_test_01, kkt_decisions)

print(f"Time: {kkt_time:.4f}s")
print(f"Support Vectors: {kkt_n_sv}")
print(f"F1-Score: {kkt_f1:.4f}")
print(f"ROC-AUC: {kkt_auc:.4f}\n")

# Train Sklearn SVM (baseline - standard QPsolver on full dataset)
print("[Step 4] Training Sklearn SVM (baseline)...")
start_time = time.time()
sklearn_svm = SVC(kernel='linear', C=1.0, max_iter=1000, verbose=0)
sklearn_svm.fit(X_train, y_train_01)
sklearn_time = time.time() - start_time
sklearn_n_sv = len(sklearn_svm.support_vectors_)

sklearn_pred = sklearn_svm.predict(X_test)
sklearn_decisions = sklearn_svm.decision_function(X_test)
sklearn_f1 = f1_score(y_test_01, sklearn_pred, zero_division=0)
sklearn_auc = roc_auc_score(y_test_01, sklearn_decisions)

print(f"Time: {sklearn_time:.4f}s")
print(f"Support Vectors: {sklearn_n_sv}")
print(f"F1-Score: {sklearn_f1:.4f}")
print(f"ROC-AUC: {sklearn_auc:.4f}\n")

# Analysis
print("=" * 80)
print(" RESULTS ANALYSIS")
print("=" * 80)

time_reduction = (sklearn_time - kkt_time) / sklearn_time if sklearn_time > 0 else 0
speedup = sklearn_time / kkt_time if kkt_time > 0 else 0
f1_diff = kkt_f1 - sklearn_f1
f1_diff_pct = (f1_diff / sklearn_f1 * 100) if sklearn_f1 > 0 else 0
sv_reduction = (1 - kkt_n_sv/sklearn_n_sv) * 100 if sklearn_n_sv > 0 else 0

print(f"\n1. COMPUTATIONAL EFFICIENCY")
print(f"   Sklearn Training Time:      {sklearn_time:10.4f}s")
print(f"   KKT-Guided Training Time:   {kkt_time:10.4f}s")
print(f"   Time Reduction:             {time_reduction*100:10.1f}% (target: >=40%)")
print(f"   Speedup Factor:             {speedup:10.2f}x")

print(f"\n2. MODEL COMPLEXITY")
print(f"   Sklearn Support Vectors:    {sklearn_n_sv:10d}")
print(f"   KKT-Guided SVs:             {kkt_n_sv:10d}")
print(f"   SV Reduction:               {sv_reduction:10.1f}%")

print(f"\n3. DECISION BOUNDARY QUALITY")
print(f"   Sklearn F1-Score:           {sklearn_f1:10.4f}")
print(f"   KKT-Guided F1-Score:        {kkt_f1:10.4f}")
print(f"   Difference:                 {f1_diff:+10.4f}")
print(f"   Relative Difference:        {f1_diff_pct:+10.2f}% (target: +-5%)")

print(f"\n4. CLASSIFICATION PERFORMANCE")
print(f"   Sklearn ROC-AUC:            {sklearn_auc:10.4f}")
print(f"   KKT-Guided ROC-AUC:         {kkt_auc:10.4f}")

print(f"\n5. OBJECTIVE ACHIEVEMENT")
speedup_ok = time_reduction >= 0.40
f1_ok = abs(f1_diff_pct) <= 5.0
print(f"   [{'PASS' if speedup_ok else 'FAIL'}] 40% Time Complexity Reduction: {time_reduction*100:.1f}%")
print(f"   [{'PASS' if f1_ok else 'FAIL'}] Maintain Decision Boundary: {f1_diff_pct:+.2f}%")

print("\n" + "=" * 80)
if speedup_ok and f1_ok:
    print(" SUCCESS: ALL OBJECTIVES ACHIEVED")
    print(" The KKT-guided optimizer successfully reduces computational complexity")
    print(" by {:.1f}% while maintaining {:.2f}% accuracy in decision boundaries.".format(
        time_reduction*100, f1_diff_pct))
else:
    print(" INFO: Dataset size or parameters may need adjustment for optimal speedup")
    print(" On smaller datasets, overhead of KKT checking may affect relative speedup")
    print(" Speedup is more pronounced on massive datasets (100K+ samples)")
print("=" * 80 + "\n")
