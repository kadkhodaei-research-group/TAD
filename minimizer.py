from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)


class Minimizer:
    """Implementation of gradient descent methods for finding local minima.
    
    This is the opposite of the Dimer method - instead of finding saddle points,
    this finds local minima by following the negative gradient (force) direction.
    """
    
    def __init__(
            self,
            x: npt.NDArray[np.float64],
            force_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
            method: str = 'lbfgs',
            step_size: float = 0.1,
            max_step_size: float = 0.2,
            stopping_criteria: float = 0.01,
            line_search: bool = True,
            force_reset_threshold: float = 0.5,  # Reset optimizer if force increases by this factor
            adaptive_step: bool = True,  # Enable adaptive step sizing
            **kwargs
        ) -> None:
        """Initialize the Minimizer.
        
        Args:
            x: Initial position
            force_func: Function returning forces at given position
            method: Minimization method ('steepest', 'cg', 'lbfgs', 'fire')
            step_size: Base step size
            max_step_size: Maximum allowed step size
            stopping_criteria: Force convergence threshold (eV/Å)
            line_search: Whether to use line search
        """
        self.x = x.copy()
        self.force_func = force_func
        self.method = method.lower()
        self.step_size = step_size
        self.max_step_size = max_step_size
        self.stopping_criteria = stopping_criteria
        self.line_search = line_search
        self.force_reset_threshold = force_reset_threshold
        self.adaptive_step = adaptive_step
        
        # Dimension
        self.D = x.size
        
        # Track force history for adaptive stepping and reset detection
        self.force_history = []
        self.best_force_norm = float('inf')
        self.steps_since_improvement = 0
        self.step_size_factor = 1.0  # Adaptive step size multiplier
        
        # Initialize optimization info based on method
        if self.method == 'lbfgs':
            self.optinfo = {
                'F_old': np.zeros((1, self.D)),
                'deltaR_mem': np.ndarray(shape=(0, self.D)),
                'deltaF_mem': np.ndarray(shape=(0, self.D)),
                'num_lbfgs_mem': min(10, self.D),  # Memory size for L-BFGS
            }
        elif self.method == 'cg':
            self.optinfo = {
                'F_old': np.zeros((1, self.D)),
                'search_dir': np.zeros((1, self.D)),
                'iter': 0,
            }
        elif self.method == 'fire':
            # Fast Inertial Relaxation Engine
            self.optinfo = {
                'velocity': np.zeros((1, self.D)),
                'alpha': 0.1,
                'dt': 0.1,
                'N_steps': 0,
                'N_min': 5,
                'f_inc': 1.1,
                'f_dec': 0.5,
                'alpha_start': 0.1,
                'f_alpha': 0.99,
            }
        elif self.method == 'bfgs':
            # Full BFGS with Hessian approximation
            self.optinfo = {
                'H': np.eye(self.D),  # Initial Hessian approximation
                'F_old': np.zeros(self.D),
                'x_old': np.zeros(self.D),
                'first_step': True,
            }
        elif self.method == 'lbfgs_scipy':
            # Prepare for scipy-style L-BFGS
            self.optinfo = {
                'memory': [],  # List of (s, y) pairs
                'max_memory': min(10, self.D),
                'F_old': np.zeros(self.D),
                'x_old': np.zeros(self.D),
                'first_step': True,
            }
        else:  # steepest descent
            self.optinfo = {}
            
        # Track all observations
        self.R_all = np.ndarray(shape=(0, self.D))
        self.E_all = np.ndarray(shape=(0, 1))
        self.F_all = np.ndarray(shape=(0, self.D))
        
    def steepest_descent_step(self, F: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Simple steepest descent step."""
        # Normalize force direction
        F_norm = np.linalg.norm(F)
        if F_norm > 1e-10:
            direction = F / F_norm
            step = self.step_size * direction
            
            # Limit step size
            step_norm = np.linalg.norm(step)
            if step_norm > self.max_step_size:
                step = step * (self.max_step_size / step_norm)
        else:
            step = np.zeros_like(F)
            
        return step.ravel()
        
    def bfgs_step(self, F: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Full BFGS step with Hessian update."""
        F = F.ravel()
        
        if self.optinfo['first_step']:
            # First step - use steepest descent
            self.optinfo['first_step'] = False
            self.optinfo['F_old'] = F.copy()
            self.optinfo['x_old'] = self.x.copy()
            return self.steepest_descent_step(F)
        
        # Get stored values
        H = self.optinfo['H']
        F_old = self.optinfo['F_old']
        x_old = self.optinfo['x_old']
        
        # Compute differences
        s = self.x - x_old  # Step
        y = F - F_old  # Force difference
        
        # Check if update is valid
        sy = np.dot(s, y)
        if sy > 1e-10:  # Positive curvature condition
            # BFGS update formula
            Hs = H @ s
            sHs = np.dot(s, Hs)
            
            # Update Hessian approximation
            H = H + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs
            
            # Ensure symmetry and positive definiteness
            H = 0.5 * (H + H.T)
            
            # Add small diagonal regularization if needed
            eigvals = np.linalg.eigvalsh(H)
            if np.min(eigvals) < 1e-6:
                H += (1e-6 - np.min(eigvals)) * np.eye(self.D)
        else:
            # Reset Hessian if update would be bad
            logger.info("BFGS: Resetting Hessian due to non-positive curvature")
            H = np.eye(self.D)
        
        # Compute step: p = H * F (since H approximates inverse Hessian)
        step = H @ F
        
        # Limit step size
        step_norm = np.linalg.norm(step)
        if step_norm > self.max_step_size:
            step = step * (self.max_step_size / step_norm)
            # Reset Hessian if step was limited
            H = np.eye(self.D)
        
        # Update stored values
        self.optinfo['H'] = H
        self.optinfo['F_old'] = F.copy()
        self.optinfo['x_old'] = self.x.copy()
        
        return step
        
    def lbfgs_scipy_step(self, F: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """L-BFGS step using scipy-style two-loop recursion (more robust)."""
        F = F.ravel()
        
        if self.optinfo['first_step']:
            self.optinfo['first_step'] = False
            self.optinfo['F_old'] = F.copy()
            self.optinfo['x_old'] = self.x.copy()
            return self.steepest_descent_step(F)
        
        # Get stored values
        memory = self.optinfo['memory']
        F_old = self.optinfo['F_old']
        x_old = self.optinfo['x_old']
        
        # Compute differences
        s = self.x - x_old
        y = F - F_old
        
        # Check if update is valid
        sy = np.dot(s, y)
        if sy > 1e-10:
            # Add to memory
            memory.append((s.copy(), y.copy()))
            if len(memory) > self.optinfo['max_memory']:
                memory.pop(0)
        
        # Two-loop recursion
        q = F.copy()
        m = len(memory)
        alpha = np.zeros(m)
        
        # First loop (backward)
        for i in range(m-1, -1, -1):
            s_i, y_i = memory[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            alpha[i] = rho_i * np.dot(s_i, q)
            q = q - alpha[i] * y_i
        
        # Scaling
        if m > 0:
            s_last, y_last = memory[-1]
            gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
            gamma = np.clip(gamma, 0.01, 10.0)  # Limit scaling factor
        else:
            gamma = self.step_size
        
        r = gamma * q
        
        # Second loop (forward)
        for i in range(m):
            s_i, y_i = memory[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            beta = rho_i * np.dot(y_i, r)
            r = r + (alpha[i] - beta) * s_i
        
        step = r
        
        # Apply adaptive step sizing
        if self.adaptive_step:
            step = step * self.step_size_factor
        
        # Limit step size
        step_norm = np.linalg.norm(step)
        if step_norm > self.max_step_size:
            step = step * (self.max_step_size / step_norm)
            # Clear memory if step was limited
            self.optinfo['memory'] = []
            logger.info("L-BFGS-Scipy: Cleared memory due to step limit")
        
        # Update stored values
        self.optinfo['F_old'] = F.copy()
        self.optinfo['x_old'] = self.x.copy()
        
        return step
        
    def conjugate_gradient_step(self, F: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conjugate gradient step (Fletcher-Reeves)."""
        F = F.reshape(1, -1)
        F_old = self.optinfo['F_old']
        search_dir_old = self.optinfo['search_dir']
        iter_count = self.optinfo['iter']
        
        if iter_count == 0 or np.linalg.norm(F_old) < 1e-10:
            # First iteration or reset
            search_dir = F
        else:
            # Fletcher-Reeves formula
            beta = np.dot(F[0, :], F[0, :]) / (np.dot(F_old[0, :], F_old[0, :]) + 1e-10)
            
            # Reset if beta is negative or too large
            if beta < 0 or beta > 10:
                beta = 0
                
            search_dir = F + beta * search_dir_old
            
        # Normalize and scale
        dir_norm = np.linalg.norm(search_dir)
        if dir_norm > 1e-10:
            step = self.step_size * search_dir / dir_norm
            
            # Limit step size
            step_norm = np.linalg.norm(step)
            if step_norm > self.max_step_size:
                step = step * (self.max_step_size / step_norm)
        else:
            step = np.zeros_like(search_dir)
            
        # Update history
        self.optinfo['F_old'] = F.copy()
        self.optinfo['search_dir'] = search_dir.copy()
        self.optinfo['iter'] += 1
        
        return step.ravel()
        
    def lbfgs_step(self, F: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """L-BFGS step following the two-loop recursion."""
        F = F.reshape(1, -1)
        F_old = self.optinfo['F_old']
        deltaR_mem = self.optinfo['deltaR_mem']
        deltaF_mem = self.optinfo['deltaF_mem']
        num_mem = self.optinfo['num_lbfgs_mem']
        
        m = deltaR_mem.shape[0]
        
        # Update memory if not first iteration
        if m > 0 or np.linalg.norm(F_old) > 1e-10:
            if m > 0:
                deltaF_mem = np.vstack((deltaF_mem, F - F_old))
                
        # L-BFGS two-loop recursion
        q = F[0, :]  # We want to follow forces for minimization
        a_mem = np.zeros((m, 1))
        
        # First loop
        for k in range(m):
            s = deltaR_mem[m - 1 - k, :]
            y = deltaF_mem[m - 1 - k, :]
            rho = 1 / (np.dot(y, s) + 1e-10)
            a = rho * np.dot(s, q)
            a_mem[m - 1 - k, 0] = a
            q = q - a * y
            
        # Scaling
        if m > 0:
            s = deltaR_mem[m - 1, :]
            y = deltaF_mem[m - 1, :]
            scaling = np.dot(s, y) / (np.dot(y, y) + 1e-10)
            scaling = np.clip(scaling, 0.01, 1.0)  # Ensure positive scaling
        else:
            scaling = self.step_size
            
        r = scaling * q
        
        # Second loop
        for k in range(1, m + 1):
            s = deltaR_mem[k - 1, :]
            y = deltaF_mem[k - 1, :]
            rho = 1 / (np.dot(y, s) + 1e-10)
            b = rho * np.dot(y, r)
            r = r + s * (a_mem[k - 1, 0] - b)
            
        # The step is in the direction r
        step = r[np.newaxis]
        
        # Limit step size
        step_norm = np.linalg.norm(step)
        if step_norm > self.max_step_size:
            step = step * (self.max_step_size / step_norm)
            # Reset memory if step was limited
            self.optinfo['deltaR_mem'] = np.ndarray(shape=(0, self.D))
            self.optinfo['deltaF_mem'] = np.ndarray(shape=(0, self.D))
        else:
            # Update memory for next iteration
            if np.linalg.norm(F_old) > 1e-10:  # Not first iteration
                deltaR_mem = np.vstack((deltaR_mem, step))
                if m >= num_mem:
                    deltaR_mem = np.delete(deltaR_mem, 0, 0)
                    deltaF_mem = np.delete(deltaF_mem, 0, 0)
                self.optinfo['deltaR_mem'] = deltaR_mem
                self.optinfo['deltaF_mem'] = deltaF_mem
                
        self.optinfo['F_old'] = F.copy()
        
        return step.ravel()
        
    def fire_step(self, F: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """FIRE (Fast Inertial Relaxation Engine) step."""
        F = F.reshape(1, -1)
        v = self.optinfo['velocity']
        alpha = self.optinfo['alpha']
        dt = self.optinfo['dt']
        N_steps = self.optinfo['N_steps']
        
        # Check if moving in right direction
        P = np.dot(F[0, :], v[0, :])
        
        if P > 0:
            # Good direction - increase time step and adjust mixing
            N_steps += 1
            if N_steps > self.optinfo['N_min']:
                dt = min(dt * self.optinfo['f_inc'], self.max_step_size)
                alpha = alpha * self.optinfo['f_alpha']
        else:
            # Bad direction - reset
            v = np.zeros((1, self.D))
            alpha = self.optinfo['alpha_start']
            dt = dt * self.optinfo['f_dec']
            N_steps = 0
            
        # Update velocity (mixing with force direction)
        F_norm = np.linalg.norm(F)
        v_norm = np.linalg.norm(v)
        
        if F_norm > 1e-10:
            v = (1 - alpha) * v + alpha * (v_norm / F_norm) * F
            
        # Velocity Verlet integration
        v = v + dt * F
        step = dt * v
        
        # Limit step size
        step_norm = np.linalg.norm(step)
        if step_norm > self.max_step_size:
            step = step * (self.max_step_size / step_norm)
            v = step / dt  # Adjust velocity accordingly
            
        # Update state
        self.optinfo['velocity'] = v
        self.optinfo['alpha'] = alpha
        self.optinfo['dt'] = dt
        self.optinfo['N_steps'] = N_steps
        
        return step.ravel()
        
    def run(self) -> npt.NDArray[np.float64]:
        """Execute one minimization step.
        
        Returns:
            Step vector (displacement from current position)
        """
        # Get forces at current position
        R = self.x.reshape(1, -1)
        F = self.force_func(self.x)
        F = F.reshape(1, -1)
        
        # Store observation
        self.R_all = np.vstack((self.R_all, R)) if self.R_all.size > 0 else R
        self.E_all = np.vstack((self.E_all, [[-1.0]])) if self.E_all.size > 0 else np.array([[-1.0]])
        self.F_all = np.vstack((self.F_all, F)) if self.F_all.size > 0 else F
        
        # Check convergence
        F_max = np.max(np.abs(F))
        F_rms = np.sqrt(np.mean(F**2))
        
        # Track force history
        self.force_history.append(F_rms)
        
        # Adaptive step size adjustment
        if self.adaptive_step and len(self.force_history) > 1:
            if F_rms < self.best_force_norm:
                # Improvement - increase step size
                self.best_force_norm = F_rms
                self.steps_since_improvement = 0
                self.step_size_factor = min(self.step_size_factor * 1.1, 2.0)
            else:
                # No improvement
                self.steps_since_improvement += 1
                
                # Check if we should reset
                if F_rms > self.best_force_norm * (1 + self.force_reset_threshold):
                    logger.info(f"Force increased significantly ({F_rms:.4f} > {self.best_force_norm * (1 + self.force_reset_threshold):.4f}), resetting optimizer")
                    self._reset_optimizer()
                    self.step_size_factor = 0.5  # Reduce step size after reset
                elif self.steps_since_improvement > 5:
                    # Reduce step size if no improvement for several steps
                    self.step_size_factor = max(self.step_size_factor * 0.9, 0.1)
        else:
            self.best_force_norm = F_rms
        
        if F_rms < self.stopping_criteria:
            logger.info(f"Converged with RMS force: {F_rms:.6f}")
            return np.zeros_like(self.x)
            
        # Calculate step based on method
        if self.method == 'lbfgs':
            step = self.lbfgs_step(F)
        elif self.method == 'lbfgs_scipy':
            step = self.lbfgs_scipy_step(F)
        elif self.method == 'bfgs':
            step = self.bfgs_step(F)
        elif self.method == 'cg':
            step = self.conjugate_gradient_step(F)
        elif self.method == 'fire':
            step = self.fire_step(F)
        else:  # steepest descent
            step = self.steepest_descent_step(F)
            
        # Enhanced line search for certain methods
        if self.line_search and self.method in ['steepest', 'cg', 'bfgs', 'lbfgs_scipy']:
            step = self._armijo_line_search(step, F)
            
        # Update position
        self.x = self.x + step
        
        return step
        
    def _reset_optimizer(self):
        """Reset optimizer state when stuck."""
        if self.method == 'lbfgs':
            self.optinfo['deltaR_mem'] = np.ndarray(shape=(0, self.D))
            self.optinfo['deltaF_mem'] = np.ndarray(shape=(0, self.D))
        elif self.method == 'lbfgs_scipy':
            self.optinfo['memory'] = []
        elif self.method == 'bfgs':
            self.optinfo['H'] = np.eye(self.D)
        elif self.method == 'cg':
            self.optinfo['iter'] = 0
            self.optinfo['search_dir'] = np.zeros((1, self.D))
        elif self.method == 'fire':
            self.optinfo['velocity'] = np.zeros((1, self.D))
            self.optinfo['alpha'] = self.optinfo['alpha_start']
            self.optinfo['dt'] = 0.1
            self.optinfo['N_steps'] = 0
            
    def _armijo_line_search(self, step: npt.NDArray[np.float64], 
                           F0: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Armijo backtracking line search with force function."""
        alpha = 1.0
        c1 = 1e-4  # Armijo constant
        rho = 0.5  # Backtracking factor
        max_iter = 20
        
        # Initial force magnitude
        f0 = np.linalg.norm(F0)
        
        # Compute directional derivative (should be positive for minimization)
        # Since we're following forces, this should be positive
        grad_dot_dir = np.dot(F0.ravel(), step)
        
        if grad_dot_dir <= 0:
            # Bad direction, use simple reduction
            return 0.1 * step
        
        # Armijo condition for force-based optimization
        for i in range(max_iter):
            x_new = self.x + alpha * step
            F_new = self.force_func(x_new)
            f_new = np.linalg.norm(F_new)
            
            # We want force magnitude to decrease
            # Armijo: f(x + alpha*p) <= f(x) - c1 * alpha * |grad_f · p|
            if f_new <= f0 - c1 * alpha * grad_dot_dir:
                break
                
            alpha *= rho
            
            if alpha < 1e-10:
                # Line search failed, use small step
                alpha = 0.01
                break
                
        return alpha * step
        
    def _simple_line_search(self, step: npt.NDArray[np.float64], 
                           F0: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Simple backtracking line search."""
        alpha = 1.0
        c = 0.5  # Armijo constant
        rho = 0.5  # Backtracking factor
        
        # Initial function value (use force magnitude as proxy)
        f0 = -np.linalg.norm(F0)  # Negative because we want to minimize
        
        # Expected decrease
        expected_decrease = c * np.dot(F0.ravel(), step)
        
        # Backtracking
        for _ in range(10):  # Max 10 iterations
            x_new = self.x + alpha * step
            F_new = self.force_func(x_new)
            f_new = -np.linalg.norm(F_new)
            
            if f_new <= f0 + alpha * expected_decrease:
                break
                
            alpha *= rho
            
        return alpha * step
        
    def continue_run(self, next_x: npt.NDArray[np.float64]) -> bool:
        """Check if minimizer should continue based on force convergence."""
        force = self.force_func(next_x)
        return np.linalg.norm(force) >= self.stopping_criteria