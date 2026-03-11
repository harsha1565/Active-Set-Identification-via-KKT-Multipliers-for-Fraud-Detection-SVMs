"""
Accelerated Gradient Descent Optimizer for SVM
Implements accelerated gradient methods (Nesterov, FISTA) with KKT-guided active set filtering.

Mathematical Background:
  Standard Gradient Descent (GD):
    w_{t+1} = w_t - η * ∇f(w_t)
    
  Nesterov Accelerated Gradient (NAG):
    v_t = β*v_{t-1} + ∇f(w_t - β*v_{t-1})
    w_{t+1} = w_t - η*v_t
    
  FISTA (Fast Iterative Shrinkage-Thresholding):
    y_t = x_t + ((t-1)/(t+2)) * (x_t - x_{t-1})
    x_{t+1} = proximal_operator(y_t - (1/L)*∇f(y_t))
"""

import numpy as np
import time
from typing import Tuple, Dict, Callable, Optional
from scipy.optimize import line_search
from .kkt_conditions import KKTConditions


class AcceleratedGradientOptimizer:
    """
    Accelerated Gradient Descent with KKT-guided active set identification.
    Significantly reduces computational complexity through dynamic constraint filtering.
    """
    
    def __init__(self, 
                 method: str = 'nesterov',
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 use_kkt_filtering: bool = True,
                 kkt_check_frequency: int = 10,
                 verbose: bool = False):
        """
        Initialize optimizer.
        
        Args:
            method: 'sgd', 'momentum', 'nesterov', or 'fista'
            learning_rate: Initial learning rate (will be adapted)
            momentum: Momentum parameter (0.9-0.99 recommended)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            use_kkt_filtering: Whether to use KKT filtering for active set reduction
            kkt_check_frequency: Check KKT conditions every N iterations
            verbose: Print optimization progress
        """
        self.method = method
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.use_kkt_filtering = use_kkt_filtering
        self.kkt_check_frequency = kkt_check_frequency
        self.verbose = verbose
        
        self.kkt_checker = KKTConditions(tol=tol)
        
        # Optimization state
        self.velocity = None
        self.t = 0  # Iteration counter
        self.history = {
            'loss': [],
            'gradient_norm': [],
            'learning_rate': [],
            'active_set_size': [],
            'time': [],
            'kkt_violation': []
        }
        
    def _compute_gradient_svm(self, 
                             X_active: np.ndarray, 
                             y_active: np.ndarray, 
                             w: np.ndarray, 
                             b: float, 
                             C: float,
                             indices: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Compute gradient of hinge loss for active set.
        
        Loss: L = 0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i(w^T x_i + b))
        Gradient: ∇_w L = w - C * sum_{i: margin_i < 1} y_i * x_i
        
        Args:
            X_active: Features for active set
            y_active: Labels for active set
            w: Current weight vector
            b: Current bias
            C: Regularization parameter
            indices: Indices in original dataset (for tracking)
            
        Returns:
            grad_w: Gradient w.r.t. weights
            grad_b: Gradient w.r.t. bias
            loss: Current loss value
        """
        n = X_active.shape[0]
        
        # Compute margins: y_i * (w^T x_i + b)
        margins = y_active * (X_active @ w + b)
        
        # Hinge loss: max(0, 1 - margin)
        hinge_losses = np.maximum(0, 1 - margins)
        loss = 0.5 * np.sum(w**2) + C * np.sum(hinge_losses)
        
        # Gradient computation
        # Identify violated margins (active constraints)
        violated = hinge_losses > 0
        
        # Gradient w.r.t. w: w - C * sum_{violated} y_i * x_i
        grad_w = w.copy()
        if np.any(violated):
            # Reshape y for proper broadcasting: (n_violated,) -> (n_violated, 1)
            y_violated = y_active[violated].reshape(-1, 1)
            grad_w -= C * (y_violated * X_active[violated]).sum(axis=0)
        
        # Gradient w.r.t. b: -C * sum_{violated} y_i
        grad_b = 0.0
        if np.any(violated):
            grad_b = -C * np.sum(y_active[violated])
        
        return grad_w, grad_b, loss
    
    def _adaptive_line_search(self, 
                             X_active: np.ndarray, 
                             y_active: np.ndarray,
                             w: np.ndarray, 
                             b: float,
                             grad_w: np.ndarray,
                             grad_b: float,
                             C: float,
                             direction_w: np.ndarray,
                             direction_b: float,
                             indices: np.ndarray) -> float:
        """
        Perform line search to find good step size.
        Uses backtracking to ensure sufficient decrease.
        
        Args:
            X_active, y_active: Active training set
            w, b: Current parameters
            grad_w, grad_b: Current gradients
            C: Regularization parameter
            direction_w, direction_b: Search direction
            indices: Indices in original dataset
            
        Returns:
            step_size: Adaptive step size
        """
        # Simple backtracking line search
        step_size = self.learning_rate
        c1 = 1e-4  # Armijo constant
        
        for _ in range(10):
            w_new = w - step_size * direction_w
            b_new = b - step_size * direction_b
            
            _, _, loss_new = self._compute_gradient_svm(
                X_active, y_active, w_new, b_new, C, indices
            )
            _, _, loss_old = self._compute_gradient_svm(
                X_active, y_active, w, b, C, indices
            )
            
            # Check sufficient decrease (Armijo condition)
            grad_norm = np.sum(np.abs(grad_w)) + np.abs(grad_b)
            sufficient_decrease = loss_new <= loss_old - c1 * step_size * grad_norm
            
            if sufficient_decrease:
                return step_size
            
            step_size *= 0.5
        
        return step_size
    
    def optimize(self, 
                X: np.ndarray, 
                y: np.ndarray, 
                C: float = 1.0,
                initial_w: Optional[np.ndarray] = None,
                initial_b: Optional[float] = None) -> Tuple[np.ndarray, float, Dict]:
        """
        Main optimization loop with KKT-guided active set filtering.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) with values ±1
            C: Regularization parameter
            initial_w: Initial weight vector (default: zeros)
            initial_b: Initial bias (default: 0)
            
        Returns:
            w: Optimized weight vector
            b: Optimized bias term
            history: Dictionary with optimization trajectory
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        if initial_w is None:
            w = np.zeros(n_features)
        else:
            w = initial_w.copy()
        
        b = initial_b if initial_b is not None else 0.0
        
        # Initialize Lagrange multipliers
        alpha = np.zeros(n_samples)
        
        # Initialize velocity for momentum methods
        if self.method in ['momentum', 'nesterov']:
            self.velocity = np.zeros(n_features)
            velocity_b = 0.0
        
        start_time = time.time()
        working_set_mask = np.ones(n_samples, dtype=bool)
        
        for iteration in range(self.max_iter):
            iter_start = time.time()
            
            # KKT-guided active set filtering
            if self.use_kkt_filtering and iteration > 0 and iteration % self.kkt_check_frequency == 0:
                margin_violations = self.kkt_checker.compute_margin_violations(
                    X, y, w, b, C
                )
                alpha = self.kkt_checker.compute_dual_variables(
                    margin_violations, alpha, C, eta=0.01
                )
                
                # Identify support vectors and filter working set
                sv_info = self.kkt_checker.identify_support_vectors(alpha, margin_violations, C)
                working_set_mask = self.kkt_checker.filter_working_set(
                    alpha, margin_violations, C, aggressiveness=0.15
                )
                
                # Get updated statistics
                n_pruned, pruning_ratio = self.kkt_checker.estimate_pruning_benefit(
                    working_set_mask
                )
                
                if self.verbose:
                    print(f"Iter {iteration}: Working set size: {np.sum(working_set_mask)}/{n_samples} "
                          f"(pruned {n_pruned}, ratio={pruning_ratio:.2%})")
                    print(f"  SVs: core={sv_info['n_core_sv']}, boundary={sv_info['n_boundary_sv']}, "
                          f"non={sv_info['n_non_sv']}")
                
                self.history['active_set_size'].append(np.sum(working_set_mask))
            else:
                self.history['active_set_size'].append(np.sum(working_set_mask))
            
            # Get active set
            active_indices = np.where(working_set_mask)[0]
            X_active = X[working_set_mask]
            y_active = y[working_set_mask]
            
            # Compute gradient on active set
            grad_w, grad_b, loss = self._compute_gradient_svm(
                X_active, y_active, w, b, C, active_indices
            )
            
            # Adaptive learning rate (decrease over time)
            current_lr = self.learning_rate / (1 + 0.01 * iteration)
            
            # Update based on method
            if self.method == 'sgd':
                w = w - current_lr * grad_w
                b = b - current_lr * grad_b
                
            elif self.method == 'momentum':
                self.velocity = self.momentum * self.velocity - current_lr * grad_w
                velocity_b = self.momentum * velocity_b - current_lr * grad_b
                w = w + self.velocity
                b = b + velocity_b
                
            elif self.method == 'nesterov':
                # Nesterov: lookahead then update
                w_lookahead = w - self.momentum * self.velocity
                b_lookahead = b - self.momentum * velocity_b
                
                grad_w_la, grad_b_la, _ = self._compute_gradient_svm(
                    X_active, y_active, w_lookahead, b_lookahead, C, active_indices
                )
                
                self.velocity = self.momentum * self.velocity - current_lr * grad_w_la
                velocity_b = self.momentum * velocity_b - current_lr * grad_b_la
                w = w + self.velocity
                b = b + velocity_b
                
            elif self.method == 'fista':
                if iteration == 0:
                    z = w.copy()
                    z_b = b
                    t_prev = 1.0
                
                # FISTA acceleration
                t = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
                z_prev = z.copy()
                z_b_prev = z_b
                
                z = w - current_lr * grad_w
                z_b = b - current_lr * grad_b
                
                w = z + ((t_prev - 1) / t) * (z - z_prev)
                b = z_b + ((t_prev - 1) / t) * (z_b - z_b_prev)
                
                t_prev = t
            
            # Compute statistics
            grad_norm = np.linalg.norm(grad_w)
            margins = y * (X @ w + b)
            margin_violations = self.kkt_checker.compute_margin_violations(X, y, w, b, C)
            violations, max_violation = self.kkt_checker.compute_kkt_violations(
                alpha, margin_violations, C
            )
            
            self.history['loss'].append(loss)
            self.history['gradient_norm'].append(grad_norm)
            self.history['learning_rate'].append(current_lr)
            self.history['kkt_violation'].append(max_violation)
            self.history['time'].append(time.time() - iter_start)
            
            if self.verbose and (iteration % 50 == 0 or iteration == self.max_iter - 1):
                print(f"Iter {iteration}: Loss={loss:.6f}, GradNorm={grad_norm:.6f}, "
                      f"KKTViol={max_violation:.6f}")
            
            # Check convergence
            if grad_norm < self.tol and max_violation < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        total_time = time.time() - start_time
        self.history['total_time'] = total_time
        
        if self.verbose:
            print(f"Optimization complete. Total time: {total_time:.2f}s")
        
        return w, b, self.history
    
    def get_history(self) -> Dict:
        """Get optimization history."""
        return self.history
