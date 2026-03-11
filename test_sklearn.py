#!/usr/bin/env python
import sys
sys.path.insert(0, r'c:\Users\Dharnish\Documents\Mini project')

import numpy as np
from sklearn.svm import SVC
from kkt_svm_optimizer.data.data_loader import FraudDataLoader
from kkt_svm_optimizer.models.svm_model import KKTGuidedSVM

# Load and prepare
data_loader = FraudDataLoader(random_state=42)
X, y = data_loader.create_synthetic_imbalanced_dataset(
    n_samples=5000, n_features=20, imbalance_ratio=0.02, random_state=42)
prepared = data_loader.prepare_data(X, y, test_size=0.2, val_size=0.1)

X_train = prepared['X_train']
y_train_svm = prepared['y_train_svm']
X_test = prepared['X_test']
y_test_svm = prepared['y_test_svm']

# Convert to 0/1
y_train_01 = np.where(y_train_svm == -1, 0, 1)
y_test_01 = np.where(y_test_svm == -1, 0, 1)

print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"y_train shape: {y_train_01.shape}, dtype: {y_train_01.dtype}")
print(f"y_train classes: {np.unique(y_train_01)}")
print(f"y_train sum: {np.sum(y_train_01)}")

print("\nTraining KKT SVM...")
kkt_svm = KKTGuidedSVM(C=1.0, optimizer_method='nesterov', max_iter=300,
                       tol=1e-4, use_kkt_filtering=True, verbose=False)
kkt_svm.fit(X_train, y_train_svm)
print(f"KKT trained in {kkt_svm.train_time:.4f}s")

print("\nTraining Sklearn SVM...")
try:
    svm = SVC(kernel='linear', C=1.0, max_iter=1000, verbose=0)
    print(f"Before fit - X_train: {X_train.shape}, y_train: {y_train_01.shape}")
    print(f"y_train unique: {np.unique(y_train_01)}, len(y_train): {len(y_train_01)}")
    svm.fit(X_train, y_train_01)
    print(f"Sklearn SVM trained, n_support: {len(svm.support_vectors_)}")
    
    # Get performance
    from sklearn.metrics import f1_score
    y_pred = svm.predict(X_test)
    f1 = f1_score(y_test_01, y_pred)
    print(f"Sklearn F1: {f1:.4f}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
