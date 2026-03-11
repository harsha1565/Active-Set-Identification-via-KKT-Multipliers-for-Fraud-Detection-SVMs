"""
Main Training Pipeline
Orchestrates the complete KKT-guided SVM training, evaluation, and benchmarking workflow.
"""

import sys
import os
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kkt_svm_optimizer.data.data_loader import FraudDataLoader
from kkt_svm_optimizer.models.svm_model import KKTGuidedSVM
from kkt_svm_optimizer.utils.evaluation import SVMBenchmark, PerformanceAnalyzer


def main(args):
    """
    Main training pipeline.
    
    Args:
        args: Command-line arguments
    """
    
    print("\n" + "="*70)
    print("KKT-Guided SVM Optimizer for Fraud Detection")
    print("Accelerated Gradient Descent with Active Set Identification")
    print("="*70 + "\n")
    
    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    print("STEP 1: Loading Dataset")
    print("-"*70)
    
    data_loader = FraudDataLoader(random_state=args.random_state)
    
    if args.use_synthetic:
        print("Using SYNTHETIC IMBALANCED DATASET for quick demo\n")
        X, y = data_loader.create_synthetic_imbalanced_dataset(
            n_samples=args.n_samples,
            n_features=args.n_features,
            imbalance_ratio=args.imbalance_ratio,
            random_state=args.random_state
        )
    else:
        print("Using CREDIT CARD FRAUD DETECTION DATASET\n")
        if not os.path.exists(args.data_path):
            print(f"ERROR: Dataset not found at {args.data_path}")
            print("Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
            return
        X, y = data_loader.load_kaggle_credit_card_data(args.data_path)
    
    # =========================================================================
    # STEP 2: Prepare Data
    # =========================================================================
    print("\nSTEP 2: Preparing Data")
    print("-"*70)
    
    prepared = data_loader.prepare_data(
        X, y, 
        test_size=args.test_size,
        val_size=args.val_size,
        handle_imbalance=args.handle_imbalance,
        sampling_method=args.sampling_method
    )
    
    X_train = prepared['X_train']
    y_train = prepared['y_train_svm']  # Convert to ±1
    X_test = prepared['X_test']
    y_test = prepared['y_test_svm']
    
    print("\nData Preparation Summary:")
    print(f"  Training set:   {X_train.shape}")
    print(f"  Test set:       {X_test.shape}")
    print(f"  Features:       {X_train.shape[1]}")
    print(f"  Fraud samples:  {np.sum(y_train == 1)} / {len(y_train)} in training")
    
    # =========================================================================
    # STEP 3: Train KKT-Guided SVM
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Training KKT-Guided SVM")
    print("="*70)
    
    kkt_svm = KKTGuidedSVM(
        C=args.C,
        optimizer_method=args.optimizer_method,
        max_iter=args.max_iter,
        tol=args.tol,
        use_kkt_filtering=True,
        kkt_check_frequency=args.kkt_check_frequency,
        verbose=True,
        random_state=args.random_state
    )
    
    kkt_svm.fit(X_train, y_train)
    kkt_result = {
        'train_time': kkt_svm.train_time,
        'n_support_vectors': kkt_svm.get_n_support_vectors(),
        'predictions': kkt_svm.predict(X_test),
        'decision_values': kkt_svm.decision_function(X_test),
        'metrics': kkt_svm.evaluate(X_test, y_test),
        'model': kkt_svm,
        'training_summary': kkt_svm.get_training_summary()
    }
    
    # =========================================================================
    # STEP 4: Train Baseline Sklearn SVM
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Training Baseline (Sklearn SVM)")
    print("="*70)
    
    benchmark = SVMBenchmark(verbose=True)
    
    # Convert labels to 0/1 for sklearn
    y_train_01 = np.where(y_train == -1, 0, 1)
    y_test_01 = np.where(y_test == -1, 0, 1)
    
    sklearn_result = benchmark.run_sklearn_svm(
        X_train, y_train_01, X_test, y_test_01, C=args.C
    )
    
    # =========================================================================
    # STEP 5: Comparison and Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Comprehensive Comparison")
    print("="*70)
    
    comparison = benchmark.compare_implementations(kkt_result, sklearn_result)
    
    # =========================================================================
    # STEP 6: Detailed Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: Detailed Performance Analysis")
    print("="*70)
    
    analyzer = PerformanceAnalyzer()
    analyzer.print_detailed_analysis(kkt_result, sklearn_result, comparison)
    
    # =========================================================================
    # STEP 7: Generate Summary Report
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: Summary Report")
    print("="*70)
    print_summary_report(kkt_result, sklearn_result, comparison, prepared)
    
    # =========================================================================
    # STEP 8: Save Results (Optional)
    # =========================================================================
    if args.save_results:
        print("\n" + "="*70)
        print("STEP 8: Saving Results")
        print("="*70)
        save_results(kkt_result, sklearn_result, comparison, args.output_dir)
    
    return kkt_result, sklearn_result, comparison


def print_summary_report(kkt_result, sklearn_result, comparison, prepared):
    """Print comprehensive summary report."""
    
    print("\n📊 FINAL RESULTS SUMMARY\n")
    
    print("1. TRAINING EFFICIENCY")
    print("   " + "-"*50)
    print(f"   Sklearn SVM Training Time:    {sklearn_result['train_time']:8.4f}s")
    print(f"   KKT-Guided SVM Training Time: {kkt_result['train_time']:8.4f}s")
    print(f"   Time Reduction:               {comparison['time_reduction']*100:8.1f}%")
    print(f"   Speedup Factor:               {comparison['speedup_factor']:8.2f}x")
    
    print("\n2. MODEL COMPLEXITY")
    print("   " + "-"*50)
    print(f"   Sklearn Support Vectors:      {comparison['sklearn_n_sv']:8d}")
    print(f"   KKT-Guided Support Vectors:   {comparison['kkt_n_sv']:8d}")
    print(f"   SV Reduction:                 {(1-comparison['kkt_n_sv']/comparison['sklearn_n_sv'])*100:8.1f}%")
    
    print("\n3. DECISION BOUNDARY QUALITY (F1-Score)")
    print("   " + "-"*50)
    print(f"   Sklearn F1-Score:             {comparison['sklearn_f1']:8.4f}")
    print(f"   KKT-Guided F1-Score:          {comparison['kkt_f1']:8.4f}")
    print(f"   Difference:                   {comparison['f1_difference']:+8.4f}")
    print(f"   Relative Difference:          {comparison['f1_relative_difference_percent']:+8.2f}%")
    
    print("\n4. ROC-AUC COMPARISON")
    print("   " + "-"*50)
    print(f"   Sklearn ROC-AUC:              {comparison['sklearn_roc_auc']:8.4f}")
    print(f"   KKT-Guided ROC-AUC:           {comparison['kkt_roc_auc']:8.4f}")
    print(f"   Difference:                   {comparison['auc_difference']:+8.4f}")
    
    print("\n5. OBJECTIVE ACHIEVEMENT")
    print("   " + "-"*50)
    speedup_ok = comparison['time_reduction'] >= 0.40
    f1_ok = abs(comparison['f1_relative_difference_percent']) <= 5.0
    
    status_speedup = "✓ PASS" if speedup_ok else "✗ FAIL"
    status_f1 = "✓ PASS" if f1_ok else "✗ FAIL"
    
    print(f"   [1] 40% Time Reduction:  {status_speedup} ({comparison['time_reduction']*100:.1f}%)")
    print(f"   [2] Maintain F1-Score:   {status_f1} ({comparison['f1_relative_difference_percent']:+.2f}%)")
    
    print("\n6. DATASET STATISTICS")
    print("   " + "-"*50)
    meta = prepared['metadata']
    print(f"   Total Samples:                {meta['n_train'] + meta['n_test']:8d}")
    print(f"   Training Samples:             {meta['n_train']:8d}")
    print(f"   Test Samples:                 {meta['n_test']:8d}")
    print(f"   Features:                     {meta['n_features']:8d}")
    print(f"   Class Imbalance Ratio:        {meta['imbalance_ratio_train']:8.2%}")
    
    print("\n" + "="*67)
    
    if speedup_ok and f1_ok:
        print("✓ ALL OBJECTIVES ACHIEVED!")
    else:
        print("⚠ Some objectives not fully met. Check detailed analysis.")
    
    print("="*67 + "\n")


def save_results(kkt_result, sklearn_result, comparison, output_dir):
    """Save detailed results to files."""
    
    import json
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-friendly format
    results_dict = {
        'comparison': {
            'time_reduction_percent': comparison['time_reduction_percent'],
            'speedup_factor': comparison['speedup_factor'],
            'f1_difference': comparison['f1_difference'],
            'f1_relative_difference_percent': comparison['f1_relative_difference_percent'],
            'kkt_f1': float(comparison['kkt_f1']),
            'sklearn_f1': float(comparison['sklearn_f1']),
            'kkt_roc_auc': float(comparison['kkt_roc_auc']),
            'sklearn_roc_auc': float(comparison['sklearn_roc_auc']),
            'kkt_n_sv': int(comparison['kkt_n_sv']),
            'sklearn_n_sv': int(comparison['sklearn_n_sv']),
        },
        'kkt_metrics': {k: float(v) if isinstance(v, (int, np.integer, float, np.floating)) else v 
                       for k, v in kkt_result['metrics'].items()},
        'sklearn_metrics': {k: float(v) if isinstance(v, (int, np.integer, float, np.floating)) else v 
                           for k, v in sklearn_result['metrics'].items()},
        'times': {
            'kkt_train_time': float(kkt_result['train_time']),
            'sklearn_train_time': float(sklearn_result['train_time']),
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to {os.path.join(output_dir, 'results.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='KKT-Guided SVM for Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with synthetic data:
  python train_pipeline.py --use-synthetic --n-samples 5000
  
  # Full training with Kaggle data:
  python train_pipeline.py --data-path creditcard.csv
  
  # Custom hyperparameters:
  python train_pipeline.py --use-synthetic --C 0.1 --max-iter 500
        """
    )
    
    # Dataset arguments
    parser.add_argument('--use-synthetic', action='store_true', 
                       help='Use synthetic imbalanced dataset (default: use Kaggle data)')
    parser.add_argument('--data-path', type=str, default='creditcard.csv',
                       help='Path to creditcard.csv from Kaggle')
    parser.add_argument('--n-samples', type=int, default=5000,
                       help='Number of synthetic samples (if --use-synthetic)')
    parser.add_argument('--n-features', type=int, default=30,
                       help='Number of features in synthetic data')
    parser.add_argument('--imbalance-ratio', type=float, default=0.01,
                       help='Fraud ratio in synthetic data')
    
    # Data preprocessing arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Fraction of data to use for validation')
    parser.add_argument('--handle-imbalance', action='store_true',
                       help='Apply class imbalance handling')
    parser.add_argument('--sampling-method', type=str, default='none',
                       choices=['none', 'random_undersample', 'random_oversample'],
                       help='Method for handling class imbalance')
    
    # Model arguments
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter')
    parser.add_argument('--optimizer-method', type=str, default='nesterov',
                       choices=['sgd', 'momentum', 'nesterov', 'fista'],
                       help='Optimization method')
    parser.add_argument('--max-iter', type=int, default=1000,
                       help='Maximum optimization iterations')
    parser.add_argument('--tol', type=float, default=1e-4,
                       help='Convergence tolerance')
    parser.add_argument('--kkt-check-frequency', type=int, default=10,
                       help='Check KKT conditions every N iterations')
    
    # Output arguments
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to JSON file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    # Utility arguments
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run main pipeline
    try:
        kkt_result, sklearn_result, comparison = main(args)
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
