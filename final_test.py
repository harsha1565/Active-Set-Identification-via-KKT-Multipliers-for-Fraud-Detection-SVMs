#!/usr/bin/env python
"""
KKT-Guided SVM vs Sklearn - Final Full Comparison
"""

import sys
sys.path.insert(0, r'c:\Users\Dharnish\Documents\Mini project')

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

from kkt_svm_optimizer.data.data_loader import FraudDataLoader
from kkt_svm_optimizer.models.svm_model import KKTGuidedSVM

print("=" * 75)
print(" KKT-GUIDED SVM vs SKLEARN - FRAUD DETECTION BENCHMARK")
print("=" * 75 + "\n")

# Load Dataset
print("[1] Loading Imbalanced Dataset...")
data_loader = FraudDataLoader(random_state=42)
X, y = data_loader.create_synthetic_imbalanced_dataset(
    n_samples=5000, n_features=20, imbalance_ratio=0.02, random_state=42)
print(f"    Dataset: {X.shape}, Fraud: {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)\n")

# Prepare Data
print("[2] Preparing Data...")
prepared = data_loader.prepare_data(X, y, test_size=0.2, val_size=0.1)
X_train = prepared['X_train']
y_train_svm = prepared['y_train_svm']  # +-1
X_test = prepared['X_test']
y_test_svm = prepared['y_test_svm']  # +-1
print(f"    Train: {X_train.shape}, Test: {X_test.shape}\n")

# Convert labels for sklearn
y_train_01 = np.where(y_train_svm == -1, 0, 1)
y_test_01 = np.where(y_test_svm == -1, 0, 1)

# Train KKT-Guided SVM
print("[3] Training KKT-Guided SVM...")
kkt_svm = KKTGuidedSVM(
    C=1.0, optimizer_method='nesterov', max_iter=300, tol=1e-4,
    use_kkt_filtering=True, verbose=False, random_state=42)
kkt_svm.fit(X_train, y_train_svm)
kkt_time = kkt_svm.train_time
kkt_n_sv = kkt_svm.get_n_support_vectors()

kkt_pred = kkt_svm.predict(X_test)
kkt_pred_01 = np.where(kkt_pred == -1, 0, 1)
kkt_decisions = kkt_svm.decision_function(X_test)

kkt_f1 = f1_score(y_test_01, kkt_pred_01)
kkt_acc = accuracy_score(y_test_01, kkt_pred_01)
kkt_auc = roc_auc_score(y_test_01, kkt_decisions)

print(f"    Time: {kkt_time:.4f}s")
print(f"    SVs: {kkt_n_sv}")
print(f"    F1-Score: {kkt_f1:.4f}\n")

# Train Sklearn SVM
print("[4] Training Sklearn SVM (baseline)...")
start = time.time()
sklearn_svm = SVC(kernel='linear', C=1.0, max_iter=1000, verbose=0)
sklearn_svm.fit(X_train, y_train_01)
sklearn_time = time.time() - start
sklearn_n_sv = len(sklearn_svm.support_vectors_)

sklearn_pred = sklearn_svm.predict(X_test)
sklearn_decisions = sklearn_svm.decision_function(X_test)

sklearn_f1 = f1_score(y_test_01, sklearn_pred)
sklearn_acc = accuracy_score(y_test_01, sklearn_pred)
sklearn_auc = roc_auc_score(y_test_01, sklearn_decisions)

print(f"    Time: {sklearn_time:.4f}s")
print(f"   SVs: {sklearn_n_sv}")
print(f"    F1-Score: {sklearn_f1:.4f}\n")

# Compute Improvements
time_reduction = (sklearn_time - kkt_time) / sklearn_time
speedup = sklearn_time / kkt_time
f1_diff = kkt_f1 - sklearn_f1
f1_diff_pct = (f1_diff / sklearn_f1 * 100) if sklearn_f1 > 0 else 0

print("=" * 75)
print(" COMPARISON RESULTS")
print("=" * 75)
print(f"\nTraining Time:")
print(f"  Sklearn:        {sklearn_time:.4f}s")
print(f"  KKT-Guided:     {kkt_time:.4f}s")
print(f"  Time Reduction: {time_reduction*100:.1f}% (target: >=40%)")
print(f"  Speedup:        {speedup:.2f}x")

print(f"\nModel Complexity:")
print(f"  Sklearn SVs:    {sklearn_n_sv}")
print(f"  KKT SVs:        {kkt_n_sv}")
print(f"  SV Reduction:   {(1-kkt_n_sv/sklearn_n_sv)*100:.1f}%")

print(f"\nF1-Score:")
print(f"  Sklearn:        {sklearn_f1:.4f}")
print(f"  KKT-Guided:     {kkt_f1:.4f}")
print(f"  Difference:     {f1_diff:+.4f} ({f1_diff_pct:+.2f}%)")

print(f"\nRO C-AUC:")
print(f"  Sklearn:        {sklearn_auc:.4f}")
print(f"  KKT-Guided:     {kkt_auc:.4f}")

print(f"\nObjective Achievement:")
speedup_ok = time_reduction >= 0.40
f1_ok = abs(f1_diff_pct) <= 5.0
print(f"  40% Time Reduction:  {'PASS' if speedup_ok else 'FAIL'} ({time_reduction*100:.1f}%)")
print(f"  Maintain F1-Score:   {'PASS' if f1_ok else 'FAIL'} ({f1_diff_pct:+.2f}%)")

print("\n" + "=" * 75)
if speedup_ok and f1_ok:
    print(" RESULT: ALL OBJECTIVES ACHIEVED - READY FOR PRODUCTION!")
else:
    print(" RESULT: Objectives require further optimization")
print("=" * 75 + "\n")
