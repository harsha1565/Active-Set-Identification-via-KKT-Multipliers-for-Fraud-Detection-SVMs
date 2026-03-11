"""
KKT Conditions Module for SVM Optimization
Implements Karush-Kuhn-Tucker conditions for identifying active sets (support vectors)
and tracking Lagrange multipliers for constraint filtering.

Mathematical Background:
  For a convex QP problem:
    min 0.5 * w^T w + C * sum(xi_i)
    s.t. y_i(w^T x_i + b) >= 1 - xi_i, xi_i >= 0

  KKT conditions establish the relationships between:
    - Dual variables (Lagrange multipliers alpha_i)
    - Primal variables (w, b, xi_i)
    - Constraint violations
"""

import numpy as np
from typing import Tuple, Dict


class KKTConditions:
    """
    Manages KKT conditions for SVM optimization.
    Identifies support vectors and filters non-binding constraints.
    """
    
    def __init__(self, tol: float = 1e-3):
        """
        Initialize KKT conditions checker.
        
        Args:
            tol: Tolerance for KKT condition verification (default: 1e-3)
        """
        self.tol = tol
        self.kkt_violations = None
        self.support_vector_mask = None
        
    def compute_margin_violations(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray, 
                                 w: np.ndarray, 
                                 b: float, 
                                 C: float = 1.0) -> np.ndarray:
        """
        Compute margin violations for each sample.
        
        Margin violation = 1 - y_i(w^T x_i + b)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) with values ±1
            w: Weight vector (n_features,)
            b: Bias term (scalar)
            C: Regularization parameter
            
        Returns:
            margin_violations: Array of margin violations (n_samples,)
        """
        margins = y * (X @ w + b)
        margin_violations = 1.0 - margins
        return margin_violations
    
    def compute_dual_variables(self, 
                              margin_violations: np.ndarray, 
                              alpha: np.ndarray, 
                              C: float = 1.0,
                              eta: float = 0.01) -> np.ndarray:
        """
        Update Lagrange multipliers (alpha) using gradient information.
        
        Based on the dual problem gradient:
          ∇f(α) = -D(α)  where D is the Hessian of the dual problem
        
        Update rule (simplified gradient ascent on dual):
          α_i := min(C, max(0, α_i + η * margin_violation_i))
        
        Args:
            margin_violations: Current margin violations
            alpha: Current Lagrange multipliers
            C: Box constraint (C ≥ α_i ≥ 0)
            eta: Learning rate for alpha update
            
        Returns:
            alpha_updated: Updated Lagrange multipliers
        """
        alpha_new = alpha + eta * margin_violations
        alpha_new = np.clip(alpha_new, 0, C)
        return alpha_new
    
    def identify_support_vectors(self, 
                                alpha: np.ndarray, 
                                margin_violations: np.ndarray, 
                                C: float = 1.0,
                                active_tol: float = 1e-4) -> Dict:
        """
        Identify support vectors and classify constraint types based on KKT conditions.
        
        KKT Classification:
          1. Core Support Vectors (0 < α_i < C):
             - Margin exactly satisfied: |margin_i| < tol
             - Contribute actively to solution
             
          2. Boundary Support Vectors (α_i = C):
             - Possibly margin violation (margin_i < 1)
             - On upper bound of dual variable
             
          3. Non-Support Vectors (α_i = 0):
             - Margin satisfied: margin_i > -tol
             - Can be pruned from optimization
        
        Args:
            alpha: Current Lagrange multipliers (n_samples,)
            margin_violations: Margin violations (n_samples,)
            C: Box constraint
            active_tol: Tolerance for identifying active constraints
            
        Returns:
            dict with keys:
              - 'core_sv': Core support vectors (0 < α < C)
              - 'boundary_sv': Boundary support vectors (α = C)
              - 'non_sv': Non-support vectors (α = 0)
              - 'all_sv_mask': Boolean mask for all support vectors
              - 'non_sv_mask': Boolean mask for non-support vectors
              - 'sv_indices': Indices of all support vectors
              - 'non_sv_indices': Indices of non-support vectors
        """
        # Classification based on alpha values
        core_sv = (alpha > active_tol) & (alpha < C - active_tol)
        boundary_sv = alpha >= C - active_tol
        non_sv = alpha <= active_tol
        
        all_sv = core_sv | boundary_sv
        
        sv_indices = np.where(all_sv)[0]
        non_sv_indices = np.where(non_sv)[0]
        
        classification = {
            'core_sv': core_sv,
            'boundary_sv': boundary_sv,
            'non_sv': non_sv,
            'all_sv_mask': all_sv,
            'non_sv_mask': non_sv,
            'sv_indices': sv_indices,
            'non_sv_indices': non_sv_indices,
            'n_core_sv': np.sum(core_sv),
            'n_boundary_sv': np.sum(boundary_sv),
            'n_non_sv': np.sum(non_sv),
            'n_all_sv': len(sv_indices)
        }
        
        return classification
    
    def compute_kkt_violations(self, 
                              alpha: np.ndarray, 
                              margin_violations: np.ndarray, 
                              C: float = 1.0,
                              y: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Compute KKT violations for termination criteria.
        
        For sample i, KKT violation depends on its class:
          - If y_i = +1 (positive class):
              * If 0 < α_i < C: margin should be ≈ 1 (interior point)
              * If α_i = 0: margin ≥ 1 (negative slack)
              * If α_i = C: margin ≤ 1 (positive slack allowed)
              
          - If y_i = -1 (negative class): similar logic
        
        Args:
            alpha: Lagrange multipliers
            margin_violations: Margin violations (1 - margin)
            C: Box constraint
            y: Labels (optional, for detailed classification)
            
        Returns:
            kkt_violations: Per-sample violation magnitude
            max_violation: Maximum violation (scalar)
        """
        # Compute violations for each multiplier class
        violations = np.zeros_like(alpha)
        
        # Core support vectors: complementary slackness
        core_mask = (alpha > self.tol) & (alpha < C - self.tol)
        violations[core_mask] = np.abs(margin_violations[core_mask])
        
        # Boundary support vectors: only violated if margin < 1
        boundary_mask = alpha >= C - self.tol
        violations[boundary_mask] = np.maximum(0, margin_violations[boundary_mask])
        
        # Non-support vectors: only violated if margin < 1
        non_sv_mask = alpha <= self.tol
        violations[non_sv_mask] = np.maximum(0, margin_violations[non_sv_mask])
        
        max_violation = np.max(violations)
        self.kkt_violations = violations
        
        return violations, max_violation
    
    def filter_working_set(self, 
                          alpha: np.ndarray, 
                          margin_violations: np.ndarray, 
                          C: float = 1.0,
                          aggressiveness: float = 0.1) -> np.ndarray:
        """
        Filter the working set by identifying indices to include in optimization.
        
        Strategy:
          - Always include: All support vectors (0 < α < C or α = C with violations)
          - Optionally include: Some non-support vectors with large margin violations
          - Exclude: Non-support vectors far from margin (aggressively prune)
        
        This aggressive filtering is key to reducing computational complexity.
        
        Args:
            alpha: Lagrange multipliers
            margin_violations: Margin violations
            C: Box constraint
            aggressiveness: Parameter controlling pruning aggressiveness (0-1)
                           Higher = more aggressive pruning
            
        Returns:
            working_set_mask: Boolean mask for samples to include in optimization
        """
        # Always include support vectors
        support_vectors = (alpha > self.tol) & (alpha < C - self.tol)
        boundary_vectors = (alpha >= C - self.tol)
        
        # Include non-support vectors with largest violations
        non_sv = (alpha <= self.tol)
        non_sv_violations = margin_violations.copy()
        non_sv_violations[~non_sv] = -np.inf  # Exclude other types
        
        # Threshold for including non-support vectors
        n_non_sv = np.sum(non_sv)
        if n_non_sv > 0:
            threshold_percent = max(5, int(np.ceil(aggressiveness * n_non_sv)))
            threshold_idx = np.argsort(-non_sv_violations)[:threshold_percent]
            important_non_sv = np.zeros_like(non_sv)
            important_non_sv[threshold_idx] = True
        else:
            important_non_sv = np.zeros_like(non_sv, dtype=bool)
        
        working_set = support_vectors | boundary_vectors | important_non_sv
        
        return working_set
    
    def estimate_pruning_benefit(self, 
                                working_set_mask: np.ndarray) -> Tuple[int, float]:
        """
        Estimate the computational benefit of working set filtering.
        
        Returns:
            n_pruned: Number of samples pruned from optimization
            pruning_ratio: Fraction of samples pruned (0 to 1)
        """
        n_pruned = np.sum(~working_set_mask)
        pruning_ratio = n_pruned / len(working_set_mask) if len(working_set_mask) > 0 else 0
        
        return n_pruned, pruning_ratio


class KKTMonitor:
    """
    Tracks KKT conditions and statistics during optimization.
    Useful for debugging and performance analysis.
    """
    
    def __init__(self):
        """Initialize KKT monitor."""
        self.history = {
            'max_violation': [],
            'n_core_sv': [],
            'n_boundary_sv': [],
            'n_non_sv': [],
            'pruning_ratio': [],
            'active_set_size': []
        }
    
    def record(self, metrics: Dict):
        """
        Record KKT metrics at an optimization iteration.
        
        Args:
            metrics: Dictionary with keys matching self.history
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of KKT monitoring.
        
        Returns:
            dict with average and final values of tracked metrics
        """
        summary = {}
        for key, values in self.history.items():
            if len(values) > 0:
                summary[f'{key}_avg'] = np.mean(values)
                summary[f'{key}_final'] = values[-1]
        
        return summary
