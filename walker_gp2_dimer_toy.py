"""walker_gp2_dimer_toy.py - GP2 Dimer Walker for toy model potentials."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, List, Tuple
import os
import pickle
import time
from dimer import Dimer
from output_manager import get_output_path
from gp2_model import GP2
from gp_base import train_multitask_gp_toy_model

logger = logging.getLogger(__name__)


class WalkerGP2DimerToy:
    """GP2-accelerated dimer method for 2D toy model potentials."""
    
    def __init__(
        self,
        initial_position: npt.NDArray[np.float64],
        local_pes: Any,  # ToyModelInterface
        max_dimer_steps: int = 100,
        rotation: str = "lbfgsext",
        translation: str = "lbfgs",
        dimer_sep: float = 0.01,
        T_anglerot: float = 0.01,
        T_anglerot_init: float = 0.0873,
        T_anglerot_gp: float = 0.01,
        max_dimer_rotations: int = 10,
        num_init_rotations: int = 5,
        num_iter_rot_gp: int = 10,
        param_trans: Optional[npt.NDArray[np.float64]] = None,
        dimer_stopping_criteria: float = 0.01,
        step_size: float = 0.02,
        max_step_size: float = 0.05,
        divisor_T_dimer_gp: float = 10.0,
        max_inner_iterations: int = 50,
        disp_max: float = 0.2,
        verbose: bool = False,
        checkpoint_interval: int = 10,
        model_type: str = "MultitaskGPModel_rbf_atomic",
        use_gpu: bool = False,
        **kwargs
    ) -> None:
        """Initialize GP2 dimer walker for toy models.
        
        Args:
            initial_position: Initial position (2D for toy models)
            local_pes: ToyModelInterface for energy/force calculations
            max_dimer_steps: Maximum number of outer iterations (relaxation phases)
            rotation: Rotation method ('lbfgsext', 'lbfgs', 'cg', 'mn')
            translation: Translation method ('lbfgs', 'cg', 'newton', 'qmvv')
            dimer_sep: Dimer separation distance
            T_anglerot: Rotation convergence threshold
            T_anglerot_init: Initial rotation convergence threshold
            T_anglerot_gp: Rotation convergence threshold on GP surface
            max_dimer_rotations: Max rotations per translation
            num_init_rotations: Number of initial rotations
            num_iter_rot_gp: Max rotations per translation on GP
            param_trans: Translation parameters
            dimer_stopping_criteria: Force convergence threshold
            step_size: Base step size for translations
            max_step_size: Maximum allowed step size
            divisor_T_dimer_gp: Divisor for dynamic GP convergence
            max_inner_iterations: Max iterations per relaxation phase
            disp_max: Maximum displacement from nearest observed point
            verbose: Enable verbose output
            checkpoint_interval: Save checkpoint every N steps
            model_type: GP model type ('MultitaskGPModel_rbf_atomic' or 'MultitaskGPModel')
            use_gpu: Use GPU acceleration for GP training and inference
        """
        self.initial_position = initial_position.copy()
        self.local_pes = local_pes
        self.max_dimer_steps = max_dimer_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.step_size = step_size
        self.max_step_size = max_step_size
        self.dimer_sep = dimer_sep
        self.dimer_stopping_criteria = dimer_stopping_criteria
        self.T_anglerot_init = T_anglerot_init
        self.T_anglerot_gp = T_anglerot_gp
        self.num_init_rotations = num_init_rotations
        self.num_iter_rot_gp = num_iter_rot_gp
        self.divisor_T_dimer_gp = divisor_T_dimer_gp
        self.max_inner_iterations = max_inner_iterations
        self.disp_max = disp_max
        self.model_type = model_type
        self.use_gpu = use_gpu
        
        # System size (2D for toy models)
        self.n_dof = len(initial_position)
        if self.n_dof != 2:
            raise ValueError(f"Toy models expect 2D positions, got {self.n_dof}D")
        
        # Track evaluations
        self.eval_count = 0
        self.gp2_eval_count = 0
        self.force_evals_per_step = []
        self.gp2_evals_per_step = []
        
        # Set default translation parameters
        # For toy models, use smaller step for positive curvature to avoid overshooting
        if param_trans is None:
            # [steplength_convex, max_steplength]
            # Use very small convex step (1/10 of regular) near saddle points
            param_trans = np.array([[step_size * 0.1, max_step_size]])
        
        # GP2 will be initialized when we have training data
        self.gp2 = None
        self.gp2_initialized = False
        
        # Training data storage
        self.training_positions = []
        self.training_energies = []
        self.training_forces = []
        
        # Initialize dimer with true potential force function
        self.dimer = Dimer(
            x=initial_position,
            force_func=self._force_func_true,
            dimer_sep=dimer_sep,
            rotation_method=rotation,
            translation=translation,
            opt_type="gp2_dimer",
            max_dimer_rotations=max_dimer_rotations,
            T_anglerot=T_anglerot,
            T_anglerot_init=T_anglerot_init,
            num_iter_initrot=num_init_rotations,
            param_trans=param_trans,
            dimer_stopping_criteria=dimer_stopping_criteria
        )
        
        # Store initial orientation if provided
        if hasattr(self.dimer, 'orient') and self.dimer.orient is not None:
            self._initial_orient = self.dimer.orient.copy()
        
        # State tracking
        self.bigiter = 0  # Outer iteration counter
        self.converged = False
        self.trajectory = []  # Store (position, energy, force) tuples
        self.gp2_predictions = []  # Store GP2 predictions
        
        # Energy reference
        self.energy_reference = None
        self.reference_set = False
        
        # Table history for verbose output
        self.table_history = []
        
        # Track inner iterations and GP accuracy
        self.E_R_acc = []  # Accurate energies at dimer center
        self.maxF_R_acc = []  # Accurate max forces at dimer center
        self.E_R_gp = []  # GP energies (inner iterations)
        self.maxF_R_gp = []  # GP max forces (inner iterations)
        self.obs_at = []  # Observation points (inner iterations)
        
        if self.verbose:
            print("\n" + "="*60)
            print("GP2 DIMER WALKER FOR TOY MODELS INITIALIZED")
            print("="*60)
            print(f"Potential: {self.local_pes.potential_name}")
            print(f"Domain: {self.local_pes.pes.domain}")
            print(f"Initial position: [{initial_position[0]:.4f}, {initial_position[1]:.4f}]")
            print(f"Rotation method: {rotation}")
            print(f"Translation method: {translation}")
            print(f"Dimer separation: {dimer_sep}")
            print(f"Convergence criteria: {dimer_stopping_criteria}")
            print(f"Step size: {step_size} (max: {max_step_size})")
            print(f"\nGP2 Settings:")
            print(f"  Divisor T dimer GP: {divisor_T_dimer_gp}")
            print(f"  Max inner iterations: {max_inner_iterations}")
            print(f"  T_anglerot_gp: {T_anglerot_gp}")
            print(f"  Num iter rot GP: {num_iter_rot_gp}")
            print("="*60 + "\n")
    
    def _force_func_true(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate forces using true potential.
        
        Args:
            position: 1D position array (as called by dimer)
            
        Returns:
            Forces (1D)
        """
        # Ensure position is 1D
        position = position.ravel()
        if len(position) != 2:
            raise ValueError(f"Expected 2D position, got {len(position)}D")
        
        forces = self.local_pes.first_derivative(position, is_thermal=False)
        self.eval_count += 1
        return forces
    
    def _force_func_gp(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate forces using GP2 only.
        
        Args:
            position: 1D position array (as called by dimer)
            
        Returns:
            Forces (1D)
        """
        if not self.gp2_initialized:
            raise RuntimeError("GP2 not initialized")
        
        # Ensure position is 1D and reshape to 2D for processing
        position = position.ravel()
        if len(position) != 2:
            raise ValueError(f"Expected 2D position, got {len(position)}D")
        
        # Get GP2 prediction (convert 2D to 3D)
        pos_3d = np.array([position[0], position[1], 0.0]).reshape(1, -1)
        energy_pred, force_pred, energy_var, force_var = self.gp2.predict(pos_3d)
        
        self.gp2_eval_count += 1
        
        # Extract 2D forces from 3D prediction - handle different output formats
        if isinstance(force_pred, np.ndarray):
            if force_pred.ndim == 2:
                # Shape (1, 3) - take first row, first 2 components
                return force_pred[0, :2]
            elif force_pred.ndim == 1:
                # Shape (3,) - take first 2 components
                return force_pred[:2]
        else:
            # Scalar or other format - shouldn't happen for forces
            raise ValueError(f"Unexpected force prediction format: {type(force_pred)}")
    
    def _add_observation(self, position: npt.NDArray[np.float64], energy: float, forces: npt.NDArray[np.float64]) -> None:
        """Add observation to training data."""
        # Store as 2D for consistency
        self.training_positions.append(position.copy())
        self.training_energies.append(energy)
        self.training_forces.append(forces.copy())
    
    def _train_gp2(self) -> None:
        """Train or update GP2 model with ALL collected data."""
        n_points = len(self.training_positions)
        
        if n_points == 0:
            return
        
        # Convert 2D positions/forces to 3D for GP2 (add zero z-component)
        train_positions_2d = np.array(self.training_positions)
        train_forces_2d = np.array(self.training_forces)
        train_energies = np.array(self.training_energies)
        
        # Add zero z-component to make 3D
        train_positions = np.column_stack([train_positions_2d, np.zeros(n_points)])
        train_forces = np.column_stack([train_forces_2d, np.zeros(n_points)])
        
        if self.verbose:
            print(f"\n  [Training GP2 with ALL {len(train_energies)} points...]")
        
        try:
            # Initialize GP2 if not already done
            if self.gp2 is None:
                self.gp2 = GP2(
                    training_data=[train_positions, train_energies, train_forces],
                    path=get_output_path('data_gp2'),
                    atomic_info={
                        'n_atoms': 1, 
                        'atom_types': ['X'],
                        'n_pt': 1,
                        'pairtype': ['X-X'],
                        'atomtype_mov': ['X'],
                        'atomtype_fro': [],
                        'conf_fro': np.array([]),
                        'moving_indices': [0]
                    },
                    use_gpu=self.use_gpu
                )
                self.gp2.model_type = self.model_type
                self.gp2.energy_reference = self.energy_reference
            else:
                # Update training data
                self.gp2.training_data = [train_positions, train_energies, train_forces]
            
            # Train GP2 using appropriate method based on model type
            if self.model_type == 'MultitaskGPModel':
                # Use the new toy model training method - this is a module function
                train_multitask_gp_toy_model(
                    self.gp2,  # Pass the GP2 instance
                    training_data=self.gp2.training_data,
                    thermal_noise=None,  # No thermal noise for toy models
                    model_name="GP2"
                )
            else:
                # Use standard training method
                self.gp2.train(
                    training_data=self.gp2.training_data,
                    thermal_noise=None,  # No thermal noise for toy models
                    model_name="GP2"
                )
            
            self.gp2_initialized = True
            
            if self.verbose:
                print(f"  [GP2 training complete with {n_points} points]")
                
        except Exception as e:
            logger.error(f"GP2 training failed: {e}")
            if self.verbose:
                print(f"  [GP2 training failed: {e}]")
    
    def _evaluate_position(self, position: npt.NDArray[np.float64]) -> tuple:
        """Evaluate energy and forces at a position using true potential.
        
        Returns:
            (energy_ref, forces)
        """
        # Get energy
        energy = self.local_pes.scaler_y_value(position, is_thermal=False)
        
        # Set reference on first calculation
        if not self.reference_set:
            self.energy_reference = energy
            self.reference_set = True
            if self.gp2 is not None:
                self.gp2.energy_reference = self.energy_reference
            if self.verbose:
                print(f"Energy reference set to: {self.energy_reference:.6f}")
        
        # Apply reference
        energy_ref = energy - self.energy_reference
        
        # Get forces
        forces = self.local_pes.first_derivative(position, is_thermal=False)
        
        # Increment true evaluation counter
        self.eval_count += 1
        
        return energy_ref, forces
    
    def _gp_evaluate(self, position: npt.NDArray[np.float64]) -> tuple:
        """Evaluate energy and forces using GP2.
        
        Returns:
            (energy, forces)
        """
        if not self.gp2_initialized:
            raise RuntimeError("GP2 not initialized")
        
        # Convert to 3D for GP2
        pos_3d = np.array([position[0], position[1], 0.0]).reshape(1, -1)
        energy_pred, force_pred, energy_var, force_var = self.gp2.predict(pos_3d)
        
        # Extract energy scalar
        if isinstance(energy_pred, np.ndarray):
            energy = energy_pred.item() if energy_pred.size == 1 else energy_pred[0]
        else:
            energy = float(energy_pred)
        
        # Extract 2D forces
        if isinstance(force_pred, np.ndarray):
            if force_pred.ndim == 2:
                forces = force_pred[0, :2]
            elif force_pred.ndim == 1:
                forces = force_pred[:2]
        else:
            raise ValueError(f"Unexpected force prediction format: {type(force_pred)}")
        
        return energy, forces
    
    def _relaxation_phase(self):
        """Perform one relaxation phase on GP surface."""
        if self.verbose:
            print(f"\n" + "="*40)
            print(f"RELAXATION PHASE {self.bigiter}")
            print("="*40)
        
        # Train GP with ALL data
        self._train_gp2()
        
        # Calculate convergence threshold for GP surface
        if self.divisor_T_dimer_gp > 0 and len(self.maxF_R_acc) > 0:
            T_dimer_gp = max(
                min(self.maxF_R_acc) / self.divisor_T_dimer_gp,
                self.dimer_stopping_criteria / 10.0
            )
        else:
            T_dimer_gp = self.dimer_stopping_criteria / 10.0
        
        # Ensure GP threshold is not too small
        T_dimer_gp = max(T_dimer_gp, 0.01)
        
        if self.verbose:
            print(f"GP convergence threshold: {T_dimer_gp:.6f}")
        
        # Set current position
        R = self.dimer.x.copy()
        orient = self.dimer.orient.copy() if hasattr(self.dimer, 'orient') and self.dimer.orient is not None else None
        
        # Initialize dimer for GP surface
        self.dimer.force_func = self._force_func_gp
        self.dimer.dimer_stopping_criteria = T_dimer_gp
        
        # Track convergence
        converged = False
        inner_iter = 0
        position_history = []  # Track recent positions to detect oscillations
        
        # Inner iteration loop
        for inner_iter in range(self.max_inner_iterations):
            # Get GP predictions at current position
            E_gp, G_gp = self._gp_evaluate(R)
            maxF_gp = np.max(np.abs(G_gp))
            
            # Store GP trajectory
            self.E_R_gp.append(E_gp)
            self.maxF_R_gp.append(maxF_gp)
            
            # Check convergence on GP surface
            if maxF_gp < T_dimer_gp:
                converged = True
                if self.verbose:
                    print(f"Converged on GP surface after {inner_iter} iterations")
                break
            
            # Perform dimer step on GP surface
            self.dimer.x = R.copy()
            if orient is not None:
                self.dimer.orient = orient.copy()
            
            try:
                # Set dimer parameters for GP surface
                self.dimer.dimer_stopping_criteria = T_dimer_gp
                self.dimer.T_anglerot = self.T_anglerot_gp
                self.dimer.max_dimer_rotations = self.num_iter_rot_gp
                
                # Perform dimer step
                step_vector = self.dimer.run()
                
                # Get new position (dimer.run() already updates self.dimer.x)
                R_new = self.dimer.x.copy()
                
                # Check curvature for debugging
                if hasattr(self.dimer, 'Curv') and self.verbose:
                    curv = self.dimer.Curv
                    print(f"  Curvature: {curv:.4f} {'(negative - saddle seeking)' if curv < 0 else '(positive - climbing)'}")
                
                # Check if step is reasonable
                step_size = np.linalg.norm(R_new - R)
                if step_size > self.max_step_size:
                    # Scale back the step
                    direction = (R_new - R) / step_size
                    R_new = R + direction * self.max_step_size
                    if self.verbose:
                        print(f"  Limited step from {step_size:.4f} to {self.max_step_size}")
            except Exception as e:
                if self.verbose:
                    print(f"  Dimer step failed on GP: {e}")
                break
            
            # Check bounds and apply soft boundary handling
            domain = self.local_pes.pes.domain
            boundary_margin = 0.1  # Distance from boundary to start reducing step
            
            # Check if we're approaching boundaries
            near_boundary = False
            for i in range(2):
                if R_new[i] < domain[i][0] + boundary_margin or R_new[i] > domain[i][1] - boundary_margin:
                    near_boundary = True
                    break
            
            # If near boundary, reduce step size
            if near_boundary:
                # Calculate distance to nearest boundary
                min_dist = float('inf')
                for i in range(2):
                    dist_low = R_new[i] - domain[i][0]
                    dist_high = domain[i][1] - R_new[i]
                    min_dist = min(min_dist, dist_low, dist_high)
                
                # Scale step based on distance to boundary
                if min_dist < boundary_margin:
                    scale_factor = max(0.1, min_dist / boundary_margin)
                    step_from_R = R_new - R
                    R_new = R + step_from_R * scale_factor
                    
                    if self.verbose:
                        print(f"  Near boundary - scaled step by {scale_factor:.2f}")
            
            # Final hard limit check (should rarely trigger with soft handling)
            clipped = False
            for i in range(2):
                if R_new[i] < domain[i][0]:
                    R_new[i] = domain[i][0] + 0.01
                    clipped = True
                elif R_new[i] > domain[i][1]:
                    R_new[i] = domain[i][1] - 0.01
                    clipped = True
            
            if clipped and self.verbose:
                print(f"  Hard boundary limit applied: [{R_new[0]:.4f}, {R_new[1]:.4f}]")
            
            # Check for oscillations
            position_history.append(R_new.copy())
            if len(position_history) > 5:
                position_history.pop(0)
                # Check if we're oscillating between positions
                if len(position_history) >= 4:
                    dist1 = np.linalg.norm(position_history[-1] - position_history[-3])
                    dist2 = np.linalg.norm(position_history[-2] - position_history[-4])
                    if dist1 < 0.05 and dist2 < 0.05:
                        if self.verbose:
                            print(f"  Detected oscillation (distances: {dist1:.4f}, {dist2:.4f}), stopping relaxation")
                        break
                    
                    # Also check for getting stuck at boundary
                    if clipped and inner_iter > 5:
                        boundary_count = sum(1 for pos in position_history[-4:] 
                                           if any(abs(pos[i] - domain[i][j]) < 0.02 
                                                 for i in range(2) for j in range(2)))
                        if boundary_count >= 3:
                            if self.verbose:
                                print(f"  Stuck at boundary, stopping relaxation")
                            break
            
            # Check displacement from nearest observed point
            if len(self.training_positions) > 0:
                # Calculate distances to all observed points
                distances = [np.linalg.norm(R_new - obs_pos) for obs_pos in self.training_positions]
                min_dist_to_obs = min(distances)
                
                if min_dist_to_obs > self.disp_max:
                    if self.verbose:
                        print(f"  Displacement {min_dist_to_obs:.4f} exceeds disp_max {self.disp_max}, stopping relaxation")
                    break
            
            # Accept step
            R = R_new.copy()
            orient = self.dimer.orient.copy() if hasattr(self.dimer, 'orient') else None
            
            # Track GP evaluations
            self.gp2_eval_count += 1
        
        # Store inner iteration count
        self.obs_at.append(inner_iter)
        
        if self.verbose:
            if converged:
                print(f"Relaxation converged after {inner_iter} inner iterations")
            else:
                print(f"Relaxation stopped after {inner_iter} iterations (not converged)")
        
        # Store the final GP force magnitude for convergence checking
        if converged and len(self.maxF_R_gp) > 0:
            maxF_gp_final = self.maxF_R_gp[-1]
        else:
            # If not converged on GP or no GP evaluations, use actual force
            maxF_gp_final = None
        
        # Evaluate final position on true potential
        E_R, G_R = self._evaluate_position(R)
        maxF_R = np.max(np.abs(G_R))
        
        # Add observation for next GP training
        self._add_observation(R, E_R, G_R)
        
        # Store accurate values
        self.E_R_acc.append(E_R)
        self.maxF_R_acc.append(maxF_R)
        
        # Update dimer position
        self.dimer.x = R.copy()
        if orient is not None:
            self.dimer.orient = orient.copy()
        
        # Reset force function to true potential
        self.dimer.force_func = self._force_func_true
        self.dimer.dimer_stopping_criteria = self.dimer_stopping_criteria
        
        # Store trajectory point
        self.trajectory.append((R.copy(), E_R, G_R.copy()))
        
        # Calculate curvature by evaluating image 1
        if orient is not None:
            R1 = R + self.dimer_sep * orient.ravel()
            E1, G1 = self._evaluate_position(R1)
            
            # Ensure dimer points from low to high energy
            # If E1 < E_R, flip the orientation
            if E1 < E_R:
                orient = -orient
                self.dimer.orient = orient
                # Recalculate R1 with flipped orientation
                R1 = R + self.dimer_sep * orient.ravel()
                E1_temp = E1
                G1_temp = G1
                # Use the already calculated values by swapping
                E1 = E_R
                G1 = G_R
                # The center values become what was at image 1
                # This is approximate but avoids extra evaluation
            
            # Calculate curvature
            F_R = -G_R  # Convert gradient to force
            F1 = -G1
            curvature = 2.0 * (np.dot(F_R - F1, orient.ravel())) / self.dimer_sep
            self.dimer.Curv = curvature
        else:
            curvature = np.nan
        
        if self.verbose:
            print(f"Accurate values: E_R = {E_R:.6f}, maxF_R = {maxF_R:.6f}")
            if maxF_gp_final is not None:
                print(f"GP force at convergence: {maxF_gp_final:.6f}")
            if not np.isnan(curvature):
                print(f"Curvature: {curvature:.6f}")
        
        return R, E_R, G_R, maxF_R, curvature, maxF_gp_final
    
    def _perform_initial_rotations(self) -> None:
        """Perform initial rotations to find saddle point direction."""
        if self.verbose:
            print("\n" + "="*40)
            print("INITIAL ROTATIONS")
            print("="*40)
        
        # Initialize orientation if needed
        if not hasattr(self.dimer, 'orient') or self.dimer.orient is None:
            # Initialize along force direction
            init_forces = self.training_forces[0]  # Use stored initial forces
            if np.linalg.norm(init_forces) > 1e-10:
                orient_1d = init_forces / np.linalg.norm(init_forces)
            else:
                # Random orientation if at stationary point
                orient_1d = np.random.randn(2)
                orient_1d = orient_1d / np.linalg.norm(orient_1d)
            self.dimer.orient = orient_1d.reshape(1, -1)
        
        # Store initial orientation
        self._initial_orient = self.dimer.orient.copy()
        
        # Evaluate image 1
        R1 = self.initial_position + self.dimer_sep * self.dimer.orient.ravel()
        E1, G1 = self._evaluate_position(R1)
        self._add_observation(R1, E1, G1)
        
        # Perform initial rotations without GP
        self.dimer.x = self.initial_position.copy()
        
        # Manually perform initial rotations to collect observations
        for i in range(self.num_init_rotations):
            # Current position and orientation
            R = self.dimer.x.copy()
            orient = self.dimer.orient.copy()
            
            # Evaluate at current image 1
            R1 = R + self.dimer_sep * orient.ravel()
            E1, G1 = self._evaluate_position(R1)
            
            # Get force at center (already have it from initial evaluation)
            idx = len(self.training_positions) - 2  # Get the center position index
            G_R = self.training_forces[idx] if idx >= 0 else self.training_forces[0]
            
            # Calculate rotation force
            F_R = -G_R  # Convert gradient to force
            F1 = -G1
            F_rot = F1 - np.dot(F1, orient.ravel()) * orient.ravel()
            F_rot = F_rot - (F_R - np.dot(F_R, orient.ravel()) * orient.ravel())
            F_rot = F_rot / self.dimer_sep
            
            # Check convergence
            rotation_mag = np.linalg.norm(F_rot)
            if rotation_mag < self.T_anglerot_init:
                if self.verbose:
                    print(f"  Initial rotation {i+1} converged with |F_rot| = {rotation_mag:.6f}")
                break
            
            # Perform small rotation
            dtheta = 0.1 * rotation_mag / (1.0 + rotation_mag)  # Adaptive step
            rotation_direction = F_rot / rotation_mag
            new_orient = orient.ravel() + dtheta * rotation_direction
            new_orient = new_orient / np.linalg.norm(new_orient)
            self.dimer.orient = new_orient.reshape(1, -1)
            
            # Add observation for new image 1
            R1_new = R + self.dimer_sep * new_orient
            E1_new, G1_new = self._evaluate_position(R1_new)
            self._add_observation(R1_new, E1_new, G1_new)
            
            if self.verbose:
                print(f"  Initial rotation {i+1}: |F_rot| = {rotation_mag:.6f}, dtheta = {dtheta:.4f}")
        
        if self.verbose:
            print(f"Initial rotations complete, collected {len(self.training_positions)} observations")
    
    def run_dimer(self) -> Dict[str, Any]:
        """Run GP2-accelerated dimer optimization on toy model.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Evaluate initial position
        init_energy, init_forces = self._evaluate_position(self.initial_position)
        init_force_mag = np.linalg.norm(init_forces)
        
        # Add first observation
        self._add_observation(self.initial_position, init_energy, init_forces)
        self.trajectory.append((self.initial_position.copy(), init_energy, init_forces.copy()))
        
        if self.verbose:
            print(f"\nInitial state:")
            print(f"  Position: [{self.initial_position[0]:.4f}, {self.initial_position[1]:.4f}]")
            print(f"  Energy: {init_energy:.6f}")
            print(f"  Force magnitude: {init_force_mag:.6f}")
            print(f"  Forces: [{init_forces[0]:.6f}, {init_forces[1]:.6f}]")
        
        # Perform initial rotations to find saddle direction
        self._perform_initial_rotations()
        
        # Add initial entry to table history
        self.table_history.append({
            'Step': 0,
            'Position_X': self.initial_position[0],
            'Position_Y': self.initial_position[1],
            'Energy': init_energy,
            'Force_Mag': init_force_mag,
            'Curvature': np.nan,
            'Evaluations': len(self.training_positions),  # Count all evaluations so far
            'GP2_Evaluations': 0
        })
        
        # Print initial table
        self._print_progress_table()
        
        # Main optimization loop with relaxation phases
        while self.bigiter < self.max_dimer_steps:
            self.bigiter += 1
            step_start_evals = self.eval_count
            step_start_gp2_evals = self.gp2_eval_count
            
            # Perform relaxation phase
            R, E_R, G_R, maxF_R, curvature, maxF_gp_final = self._relaxation_phase()
            
            # Track evaluations
            step_evals = self.eval_count - step_start_evals
            step_gp2_evals = self.gp2_eval_count - step_start_gp2_evals
            self.force_evals_per_step.append(step_evals)
            self.gp2_evals_per_step.append(step_gp2_evals)
            
            # Store GP2 prediction comparison if available
            if self.gp2_initialized:
                try:
                    E_gp, G_gp = self._gp_evaluate(R)
                    self.gp2_predictions.append({
                        'position': R.copy(),
                        'energy_pred': E_gp,
                        'energy_var': 0.0,  # Not available in simplified version
                        'force_pred': G_gp,
                        'force_var': 0.0,
                        'actual_energy': E_R,
                        'actual_forces': G_R.copy()
                    })
                except:
                    pass
            
            # Store in table history
            self.table_history.append({
                'Step': self.bigiter,
                'Position_X': R[0],
                'Position_Y': R[1],
                'Energy': E_R,
                'Force_Mag': maxF_R,
                'Curvature': curvature,
                'Evaluations': step_evals,
                'GP2_Evaluations': step_gp2_evals
            })
            
            # Print full progress table
            self._print_progress_table()
            
            # Check convergence based on GP force (if available) or actual force
            if maxF_gp_final is not None:
                # Use GP force for convergence (preferred)
                convergence_force = maxF_gp_final
                force_type = "GP"
            else:
                # Fall back to actual force if GP not available
                convergence_force = maxF_R
                force_type = "actual"
            
            if convergence_force < self.dimer_stopping_criteria:
                self.converged = True
                if self.verbose:
                    print(f"\nCONVERGED! {force_type} force magnitude {convergence_force:.6f} eV/Å < {self.dimer_stopping_criteria} eV/Å")
                    print(f"(Actual force: {maxF_R:.6f} eV/Å)")
                break
            
            # Also check if we're at a saddle point (near-zero curvature with small forces)
            # Use actual force here since curvature is calculated from actual forces
            if abs(curvature) < 0.1 and maxF_R < 0.5:  # Increased curvature tolerance
                if self.verbose:
                    print(f"\nNear saddle point: |curvature| = {abs(curvature):.6f} < 0.1 and actual force = {maxF_R:.6f}")
                # Do a few more steps to refine
                if self.bigiter > 10:  # Don't stop too early
                    self.converged = True
                    break
        
        # Final evaluation
        final_pos = self.dimer.x.copy()
        final_energy, final_forces = self._evaluate_position(final_pos)
        final_force_mag = np.linalg.norm(final_forces)
        
        end_time = time.time()
        
        # Compile results
        results = {
            'converged': self.converged,
            'steps': self.bigiter,
            'final_position': final_pos,
            'final_energy': final_energy,
            'final_forces': final_forces,
            'final_force_magnitude': final_force_mag,
            'total_evaluations': self.eval_count,
            'total_gp2_evaluations': self.gp2_eval_count,
            'runtime': end_time - start_time,
            'trajectory': self.trajectory,
            'force_evals_per_step': self.force_evals_per_step,
            'gp2_evals_per_step': self.gp2_evals_per_step,
            'table_history': self.table_history,
            'gp2_predictions': self.gp2_predictions,
            'potential_info': self.local_pes.get_info(),
            'gp2_info': {
                'initialized': self.gp2_initialized,
                'training_points': len(self.training_positions)
            }
        }
        
        if hasattr(self.dimer, 'Curv'):
            results['final_curvature'] = self.dimer.Curv
        
        if self.verbose:
            print("\n" + "="*60)
            print("GP2 DIMER OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Converged: {self.converged}")
            print(f"Total outer iterations: {self.bigiter}")
            print(f"Total true evaluations: {self.eval_count}")
            print(f"Total GP2 evaluations: {self.gp2_eval_count}")
            if self.eval_count + self.gp2_eval_count > 0:
                print(f"GP2 usage rate: {self.gp2_eval_count / (self.eval_count + self.gp2_eval_count) * 100:.1f}%")
            print(f"Runtime: {end_time - start_time:.2f} seconds")
            print(f"Final position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}]")
            print(f"Final energy: {final_energy:.6f}")
            print(f"Final force magnitude: {final_force_mag:.6f}")
            if 'final_curvature' in results:
                print(f"Final curvature: {results['final_curvature']:.6f}")
            print("="*60)
        
        return results
    
    def save_checkpoint(self, filename: Optional[str] = None) -> None:
        """Save checkpoint of current state."""
        if filename is None:
            filename = get_output_path('checkpoints', 'gp2_dimer_toy_latest.pkl')
        
        checkpoint_dir = os.path.dirname(filename)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'walker': self,
            'steps': self.steps,
            'converged': self.converged,
            'eval_count': self.eval_count,
            'gp2_eval_count': self.gp2_eval_count,
            'trajectory': self.trajectory,
            'table_history': self.table_history,
            'gp2_predictions': self.gp2_predictions,
            'training_data': {
                'positions': self.training_positions,
                'energies': self.training_energies,
                'forces': self.training_forces
            },
            'dimer_state': {
                'x': self.dimer.x,
                'orient': getattr(self.dimer, 'orient', None),
                'curv': getattr(self.dimer, 'Curv', None)
            },
            'gp2_state': {
                'initialized': self.gp2_initialized
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        if self.verbose:
            print(f"  [Checkpoint saved to {filename}]")
    
    @classmethod
    def load_checkpoint(cls, filename: str, local_pes: Any) -> 'WalkerGP2DimerToy':
        """Load walker from checkpoint."""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        walker = checkpoint['walker']
        walker.local_pes = local_pes  # Reconnect PES interface
        
        # Restore dimer state
        walker.dimer.x = checkpoint['dimer_state']['x']
        if checkpoint['dimer_state']['orient'] is not None:
            walker.dimer.orient = checkpoint['dimer_state']['orient']
        if checkpoint['dimer_state']['curv'] is not None:
            walker.dimer.Curv = checkpoint['dimer_state']['curv']
        
        # Restore GP2 if it was trained
        if checkpoint['gp2_state']['initialized']:
            # Retrain with saved data
            train_positions = np.array(checkpoint['training_data']['positions'])
            train_energies = np.array(checkpoint['training_data']['energies'])
            train_forces = np.array(checkpoint['training_data']['forces'])
            
            walker.gp2.training_data = [train_positions, train_energies, train_forces]
            walker.gp2.train(
                training_data=walker.gp2.training_data,
                thermal_noise=None,
                model_name="GP2"
            )
            walker.gp2_initialized = True
        
        return walker
    
    def _print_progress_table(self) -> None:
        """Print progress table with full history."""
        print("\n" + "="*140)
        print("GP2 DIMER PROGRESS TABLE")
        print("="*140)
        print(f"{'Iter':>6} {'X (Å)':>10} {'Y (Å)':>10} {'E_GP2':>12} {'E_Actual':>12} {'E_Err%':>8} {'F_GP2':>12} {'F_Actual':>12} {'F_Err%':>8} {'Curv':>10} {'Evals':>6} {'GP2':>6}")
        print("="*140)
        
        # Print ALL entries in history
        for entry in self.table_history:
            # Get GP2 predictions if available
            e_gp2 = "---"
            f_gp2 = "---"
            e_err = "---"
            f_err = "---"
            if self.gp2_predictions and entry['Step'] > 0:
                # Find corresponding GP2 prediction
                for pred in self.gp2_predictions:
                    if np.allclose(pred['position'], [entry['Position_X'], entry['Position_Y']], atol=1e-4):
                        e_gp2 = f"{pred['energy_pred']:.6f}"
                        f_pred = pred['force_pred']
                        f_mag_gp2 = np.linalg.norm(f_pred)
                        f_gp2 = f"{f_mag_gp2:.6f}"
                        # Calculate errors
                        if abs(entry['Energy']) > 1e-10:
                            e_err_val = abs(pred['energy_pred'] - entry['Energy']) / abs(entry['Energy']) * 100
                            e_err = f"{e_err_val:.1f}"
                        if entry['Force_Mag'] > 1e-10:
                            f_err_val = abs(f_mag_gp2 - entry['Force_Mag']) / entry['Force_Mag'] * 100
                            f_err = f"{f_err_val:.1f}"
                        break
            
            # Handle NaN curvature
            curv_str = f"{entry['Curvature']:10.3f}" if not np.isnan(entry['Curvature']) else "       ---"
            
            print(f"{entry['Step']:6d} {entry['Position_X']:10.4f} {entry['Position_Y']:10.4f} "
                  f"{e_gp2:>12} {entry['Energy']:12.6f} {e_err:>8} "
                  f"{f_gp2:>12} {entry['Force_Mag']:12.6f} {f_err:>8} "
                  f"{curv_str} {entry['Evaluations']:6d} {entry['GP2_Evaluations']:6d}")
        
        print("="*140)
    
    def _print_table_row(self) -> None:
        """Print current state as table row (deprecated - use _print_progress_table)."""
        # Just call the full table print
        self._print_progress_table()