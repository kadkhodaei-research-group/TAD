"""walker_gp2_neb.py - GP2-accelerated NEB walker following atomic GP-sNEB algorithm."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, Tuple, List
import os
import pickle
import time
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from gp2_model import GP2
from atomic_structure import AtomicStructure, calculate_atomic_distance_measure
from gp_data_saver import get_gp_logger
from output_manager import get_output_path

logger = logging.getLogger(__name__)


class WalkerGP2NEB:
    """GP2-accelerated NEB method following atomic GP-sNEB algorithm."""
    
    def __init__(
        self,
        initial_path: npt.NDArray[np.float64],  # Shape: (N_images, N_atoms*3)
        local_pes: Any,  # VASP interface
        max_neb_steps: int = 100,
        # Stopping criteria
        disp_max: float = 0.5,
        ratio_at_limit: float = 2.0/3.0,
        # NEB parameters
        k_parallel: float = 1.0,
        k_perpendicular: float = 1.0,
        neb_convergence_threshold: float = 0.1,
        ci_convergence_threshold: float = 0.1,
        ci_activation_threshold: float = 0.0,  # 0 means no climbing image
        translation_method: str = "qmvv",
        step_size: float = 0.01,
        max_step_size: float = 0.2,
        # GP convergence
        divisor_T_MEP_gp: float = 10.0,
        max_inner_iterations: int = 1000,
        # Options
        num_bigiter_init: int = 1,
        num_bigiter_initparam: float = np.inf,
        num_bigiter_hess: int = 0,
        eps_hess: float = 0.001,
        # Other parameters
        verbose: bool = False,
        checkpoint_interval: int = 1,
        visualize: bool = False,
        model_type: str = "MultitaskGPModel_rbf_atomic",
        **kwargs
    ) -> None:
        """Initialize GP2-accelerated NEB walker.
        
        Args:
            initial_path: Initial path positions (N_images x 3N array)
            local_pes: VASP interface for energy/force calculations
            max_neb_steps: Maximum number of outer iterations
            disp_max: Maximum displacement from nearest observed point (relative to path length)
            ratio_at_limit: Limit for inter-atomic distance ratio
            k_parallel: Parallel spring constant
            k_perpendicular: Perpendicular spring constant
            neb_convergence_threshold: Force convergence threshold for images
            ci_convergence_threshold: Additional convergence threshold for climbing image
            ci_activation_threshold: Threshold to activate climbing image (0 = off)
            translation_method: Method for moving images ('qmvv', 'lbfgs', 'fire')
            step_size: Base step size for translations
            max_step_size: Maximum allowed step size
            divisor_T_MEP_gp: Divisor for dynamic GP convergence
            max_inner_iterations: Max iterations per relaxation phase
            num_bigiter_init: Number of iterations starting from initial path
            num_bigiter_initparam: Number of iterations with fresh hyperparameters
            num_bigiter_hess: Number of iterations using virtual Hessian
            eps_hess: Epsilon for virtual Hessian
            verbose: Enable verbose output
            checkpoint_interval: Save checkpoint every N iterations
            visualize: Visualize energy along path
            model_type: GP model type to use
        """
        self.initial_path = initial_path.copy()
        self.local_pes = local_pes
        self.max_neb_steps = max_neb_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.visualize = visualize
        self.model_type = model_type
        
        # Store translation method
        self.translation_method = translation_method
        
        # Stopping criteria
        self.disp_max = disp_max
        self.ratio_at_limit = ratio_at_limit
        self.divisor_T_MEP_gp = divisor_T_MEP_gp
        self.max_inner_iterations = max_inner_iterations
        
        # NEB parameters
        self.k_par = k_parallel
        self.k_perp = k_perpendicular
        self.T_MEP = neb_convergence_threshold
        self.T_CI = ci_convergence_threshold
        self.T_CIon_gp = ci_activation_threshold
        self.step_size = step_size
        self.max_step_size = max_step_size
        
        # Extract convergence_criterion from kwargs
        self.convergence_criterion = kwargs.get('convergence_criterion', 'max_force')
        
        # Options
        self.num_bigiter_init = num_bigiter_init
        self.num_bigiter_initparam = num_bigiter_initparam
        self.num_bigiter_hess = num_bigiter_hess
        self.eps_hess = eps_hess
        
        # System info
        self.n_images = len(initial_path)
        self.n_atoms = initial_path.shape[1] // 3
        self.n_dof = initial_path.shape[1]
        
        # Get atomic info from local_pes
        self.atomic_info = self.local_pes.get_atomic_info()
        
        # Validate atomic info
        required_keys = ['n_pt', 'pairtype', 'atomtype_mov', 'atomtype_fro', 
                        'conf_fro', 'moving_indices']
        for key in required_keys:
            if key not in self.atomic_info:
                raise ValueError(f"Missing required atomic info key: {key}")
        
        # Ensure moving indices are consistent
        moving_indices = self.atomic_info['moving_indices']
        if not isinstance(moving_indices, list):
            self.atomic_info['moving_indices'] = list(moving_indices)
        
        self.moving_indices = self.atomic_info['moving_indices']
        self.n_moving = len(self.moving_indices)
        self.n_moving_dof = self.n_moving * 3
        
        # Get atomic structure from local_pes if available
        if hasattr(self.local_pes, 'atomic_structure'):
            self.atomic_structure = self.local_pes.atomic_structure
        else:
            print("Warning: local_pes doesn't have atomic_structure attribute")
            self.atomic_structure = None
            # Try to initialize it if we have the necessary info
            if self.atomic_info.get('n_pt', 0) == 0:
                self._init_atomic_structure()
        
        # Energy reference (set on first calculation)
        self.energy_reference = None
        self.reference_set = False
        
        # Initialize GP2 model
        self.gp2 = None
        self.gp2_trained = False
        
        # Track evaluations
        self.obs_total = 0  # Total observations
        self.obs_at = []  # Observation points (inner iterations)
        
        # Calculate initial path length
        self.scale = 0.0
        for i in range(self.n_images-1):
            self.scale += np.sqrt(np.sum(np.square(initial_path[i+1,:] - initial_path[i,:])))
        
        if self.verbose:
            print(f"Initial path length: {self.scale:.4f} Å")
        
        # State tracking
        self.bigiter = 0  # Outer iteration counter
        self.converged = False
        
        # Initialize path
        self.R = initial_path.copy()  # Current path positions
        self.E_R = np.zeros((self.n_images, 1))  # Energies
        self.G_R = np.zeros((self.n_images, self.n_dof))  # Gradients (not forces!)
        
        # Latest paths
        self.R_latest_equal = np.ndarray(shape=(0, self.n_dof))  # Latest evenly spaced path
        self.R_latest_climb = None  # Latest climbing image path
        self.i_CI_latest = 0  # Latest climbing image index
        
        # Climbing image
        self.CI_on = 0  # 0 = off, 1 = on
        self.i_CI = -1  # Index of climbing image among intermediate images
        
        # Translation state
        self.V_old = np.zeros((self.n_images-2, self.n_dof))  # Velocities
        self.F_R_old = np.zeros((self.n_images-2, self.n_dof))  # Previous forces
        self.zeroV = 1  # Use zero velocity initially
        
        # Data storage
        self.R_all = np.empty((0, self.n_dof))  # All observed positions
        self.E_all = np.empty((0, 1))  # All energies
        self.G_all = np.empty((0, self.n_dof))  # All gradients
        
        # Virtual Hessian data
        self.R_h = np.ndarray(shape=(0, self.n_dof))
        self.E_h = np.ndarray(shape=(0, 1))
        self.G_h = np.ndarray(shape=(0, self.n_dof))
        
        # Accuracy tracking
        self.E_R_acc = np.ndarray(shape=(self.n_images, 0))  # Accurate energies
        self.E_R_gp = np.ndarray(shape=(self.n_images, 0))  # GP energies
        self.normF_R_acc = np.ndarray(shape=(self.n_images-2, 0))  # Accurate force norms
        self.normF_R_gp = np.ndarray(shape=(self.n_images-2, 0))  # GP force norms
        self.normFCI_acc = np.ndarray(shape=(0,))  # CI force norms (accurate)
        self.normFCI_gp = np.ndarray(shape=(0,))  # CI force norms (GP)
        
        # Hyperparameter tracking
        self.param_gp = []  # GP hyperparameters for each outer iteration
        
        # Table history for progress tracking
        self.table_history = []
        
        # Visualization
        self.figs = []
        
        if self.verbose:
            print("\n" + "="*80)
            print("GP2 NEB WALKER INITIALIZED")
            print("="*80)
            print(f"Path with {self.n_images} images")
            print(f"System size: {self.n_atoms} atoms ({self.n_dof} DOF per image)")
            print(f"Moving atoms: {self.n_moving} ({self.n_moving_dof} DOF)")
            print(f"Spring constants: k_par={k_parallel}, k_perp={k_perpendicular}")
            print(f"Convergence threshold: {neb_convergence_threshold} eV/Å")
            print(f"Climbing image: {'Enabled' if ci_activation_threshold > 0 else 'Disabled'}")
            if ci_activation_threshold > 0:
                print(f"  CI activation threshold: {ci_activation_threshold} eV/Å")
                print(f"  CI convergence threshold: {ci_convergence_threshold} eV/Å")
            print(f"Translation method: {translation_method}")
            print(f"Initial path length: {self.scale:.4f} Å")
            if num_bigiter_hess > 0:
                print(f"Virtual Hessian: Enabled for first {num_bigiter_hess} iterations")
            print("="*80 + "\n")
    
    def _init_atomic_structure(self):
        """Initialize atomic structure for managing active/frozen atoms."""
        # This method is similar to the one in walker_gp2_dimer
        # Extract positions for moving and frozen atoms
        full_pos_3d = self.initial_path[0].reshape(-1, 3)  # Use first image
        
        # Get all atom indices
        all_indices = set(range(self.n_atoms))
        moving_set = set(self.moving_indices)
        frozen_indices = list(all_indices - moving_set)
        
        moving_positions = full_pos_3d[self.moving_indices]
        frozen_positions = full_pos_3d[frozen_indices] if frozen_indices else np.empty((0, 3))
        
        # Get atom types
        moving_types = self.atomic_info['atomtype_mov']
        frozen_types = self.atomic_info.get('atomtype_fro', np.array([], dtype=np.int64))
        
        # Get activation radius from local_pes if available
        activation_radius = getattr(self.local_pes, 'activation_radius', np.inf)
        
        # Create atomic structure
        self.atomic_structure = AtomicStructure(
            moving_atoms=moving_positions,
            frozen_atoms=frozen_positions,
            moving_types=moving_types,
            frozen_types=frozen_types,
            activation_radius=activation_radius,
            moving_indices=self.moving_indices
        )
        
        # Update atomic info with structure info
        struct_info = self.atomic_structure.get_structure_info()
        self.atomic_info.update(struct_info)
        
        # Force activation of nearby frozen atoms if needed (like in demo)
        if self.atomic_info.get('n_pt', 0) == 0 and activation_radius < np.inf and len(frozen_positions) > 0:
            # Check all images on initial path for nearby frozen atoms
            for img_idx in range(self.n_images):
                img_pos_3d = self.initial_path[img_idx].reshape(-1, 3)
                moving_pos = img_pos_3d[self.moving_indices]
                self.atomic_structure.update_activated_atoms_wrapper(
                    self.initial_path[img_idx], verbose=False
                )
            
            # Update atomic info
            struct_info = self.atomic_structure.get_structure_info()
            self.atomic_info.update(struct_info)
            
            if self.verbose:
                print(f"After initial activation:")
                print(f"  Active frozen atoms: {len(self.atomic_structure.active_frozen_atoms)}")
                print(f"  Inactive frozen atoms: {len(self.atomic_structure.inactive_frozen_atoms)}")
                print(f"  Active pair types (n_pt): {self.atomic_info['n_pt']}")
    
    def _evaluate_image(self, position: npt.NDArray[np.float64], image_id: Optional[int] = None) -> Tuple[float, npt.NDArray[np.float64]]:
        """Evaluate energy and gradient at a position.
        
        Args:
            position: Atomic positions
            image_id: Optional image ID for NEB directory structure
            
        Returns:
            (energy, gradient)  # Note: gradient = -forces
        """
        # Prepare kwargs for NEB-specific parameters
        kwargs = {}
        if image_id is not None:
            kwargs['path_id'] = self.bigiter
            kwargs['image_id'] = image_id
        
        # Get energy
        energy = self.local_pes.scaler_y_value(position, is_thermal=False, **kwargs)
        
        # Set reference on first calculation
        if not self.reference_set:
            self.energy_reference = energy
            self.reference_set = True
            if self.verbose:
                print(f"Energy reference set to: {self.energy_reference:.4f} eV")
        
        # Apply reference
        energy_ref = energy - self.energy_reference
        
        # Get forces and convert to gradients
        forces = self.local_pes.first_derivative(position, is_thermal=False, **kwargs)
        gradients = -forces  # gradient = -force
        
        self.obs_total += 1
        
        return energy_ref, gradients
    
    def _evaluate_images(self, positions: npt.NDArray[np.float64], image_ids: Optional[List[int]] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluate multiple images at once.
        
        Args:
            positions: Array of positions (N_eval x n_dof)
            image_ids: Optional list of image IDs
            
        Returns:
            (energies, gradients)  # Shape: (N_eval, 1) and (N_eval, n_dof)
        """
        n_eval = len(positions)
        energies = np.zeros((n_eval, 1))
        gradients = np.zeros((n_eval, self.n_dof))
        
        for i in range(n_eval):
            image_id = image_ids[i] if image_ids is not None else None
            energies[i, 0], gradients[i, :] = self._evaluate_image(positions[i], image_id)
        
        return energies, gradients
    
    def _force_sNEB(self, R: npt.NDArray, E_R: npt.NDArray, G_R: npt.NDArray, 
                    CI_on: int = 0) -> Tuple[npt.NDArray, float, int]:
        """Calculate stabilized NEB forces on intermediate images.
        
        Returns:
            F_R: NEB forces on intermediate images (N_im-2 x D)
            normFCI: Norm of force on climbing image (0 if CI off)
            i_CI: Index of climbing image among intermediate images (-1 if CI off)
        """
        N_im = self.n_images
        D = self.n_dof
        
        # Tangent vectors
        tau = np.zeros((N_im, D))
        
        # Calculate tangents for all images
        for i in range(1, N_im-1):
            # Energy differences
            dE_plus = E_R[i+1, 0] - E_R[i, 0]
            dE_minus = E_R[i, 0] - E_R[i-1, 0]
            
            # Position differences
            dR_plus = R[i+1, :] - R[i, :]
            dR_minus = R[i, :] - R[i-1, :]
            
            # Improved tangent estimate
            if dE_plus > 0 and dE_minus > 0:
                # Local maximum
                tau[i, :] = dR_plus if dE_plus > dE_minus else dR_minus
            elif dE_plus < 0 and dE_minus < 0:
                # Local minimum
                tau[i, :] = dR_plus if dE_plus < dE_minus else dR_minus
            else:
                # Normal case
                if abs(dE_plus) > abs(dE_minus):
                    tau[i, :] = dR_plus
                elif abs(dE_plus) < abs(dE_minus):
                    tau[i, :] = dR_minus
                else:
                    tau[i, :] = dR_plus + dR_minus
            
            # Normalize
            tau_norm = np.linalg.norm(tau[i, :])
            if tau_norm > 0:
                tau[i, :] /= tau_norm
        
        # Find highest energy image for CI
        i_CI = np.argmax(E_R[1:-1, 0]) if CI_on > 0 else -1
        
        # Calculate NEB forces
        F_R = np.zeros((N_im-2, D))
        normFCI = 0.0
        
        for i in range(1, N_im-1):
            # Get negative gradient (force)
            F_i = -G_R[i, :]
            
            # Parallel component
            F_par = np.dot(F_i, tau[i, :])
            
            if CI_on > 0 and i == i_CI + 1:  # i_CI is 0-based for intermediate images
                # Climbing image: reverse parallel component
                F_R[i-1, :] = F_i - 2.0 * F_par * tau[i, :]
                normFCI = np.linalg.norm(F_R[i-1, :])
            else:
                # Regular image: remove parallel component, add spring forces
                F_perp = F_i - F_par * tau[i, :]
                
                # Spring forces
                k_plus = self.k_par
                k_minus = self.k_par
                
                # Add perpendicular spring constant contribution
                if self.k_perp > 0:
                    # Calculate perpendicular distances
                    dR_plus_perp = (R[i+1, :] - R[i, :]) - np.dot(R[i+1, :] - R[i, :], tau[i, :]) * tau[i, :]
                    dR_minus_perp = (R[i, :] - R[i-1, :]) - np.dot(R[i, :] - R[i-1, :], tau[i, :]) * tau[i, :]
                    
                    # Add perpendicular spring forces
                    F_perp += self.k_perp * (dR_plus_perp - dR_minus_perp)
                
                # Distance to neighbors
                d_plus = np.linalg.norm(R[i+1, :] - R[i, :])
                d_minus = np.linalg.norm(R[i, :] - R[i-1, :])
                
                # Parallel spring force
                F_spring_par = k_plus * d_plus - k_minus * d_minus
                
                # Total NEB force
                F_R[i-1, :] = F_perp + F_spring_par * tau[i, :]
        
        return F_R, normFCI, i_CI
    
    def _step_translation(self, R: npt.NDArray, F_R: npt.NDArray) -> npt.NDArray:
        """Move images according to NEB forces using specified method."""
        N_im = self.n_images
        R_new = R.copy()
        
        if self.translation_method == "qmvv":
            # Quick-Min Velocity Verlet
            dt = self.step_size
            
            # Update all intermediate images
            for i in range(1, N_im-1):
                idx = i - 1
                F = F_R[idx, :]
                V = self.V_old[idx, :]
                
                # Velocity update
                if self.zeroV:
                    V_new = 0.5 * dt * F
                else:
                    V_new = V + dt * F
                
                # Quick-Min projection
                Vdot = np.dot(V_new, F)
                if Vdot > 0:
                    V_new = (Vdot / np.dot(F, F)) * F
                else:
                    V_new = np.zeros_like(V_new)
                
                # Update position
                R_new[i, :] = R[i, :] + dt * V_new
                
                # Store velocity
                self.V_old[idx, :] = V_new
        
        elif self.translation_method == "lbfgs":
            # Simple steepest descent for now
            for i in range(1, N_im-1):
                idx = i - 1
                F = F_R[idx, :]
                R_new[i, :] = R[i, :] + self.step_size * F
        
        elif self.translation_method == "fire":
            # FIRE algorithm
            dt = self.step_size
            alpha_start = 0.1
            
            for i in range(1, N_im-1):
                idx = i - 1
                F = F_R[idx, :]
                V = self.V_old[idx, :]
                
                # Check if velocity is along force direction
                power = np.dot(V, F)
                
                if power > 0:
                    # Mix velocities
                    V = (1.0 - alpha_start) * V + alpha_start * np.linalg.norm(V) * F / np.linalg.norm(F)
                else:
                    # Reset velocity
                    V = np.zeros_like(V)
                
                # Velocity update
                V_new = V + dt * F
                
                # Position update
                R_new[i, :] = R[i, :] + dt * V_new
                
                # Store velocity
                self.V_old[idx, :] = V_new
        
        else:
            raise ValueError(f"Unknown translation method: {self.translation_method}")
        
        self.F_R_old = F_R.copy()
        self.zeroV = 0
        
        return R_new
    
    def _limit_step(self, R_new: npt.NDArray, R: npt.NDArray) -> npt.NDArray:
        """Limit step size if needed."""
        N_im = self.n_images
        
        # Calculate step lengths
        steplength = np.sqrt(np.sum((R_new[1:(N_im-1),:] - R[1:(N_im-1),:])**2, 1))
        
        # Calculate atom-wise step lengths
        steplength_atomwise = np.zeros((N_im-2, self.n_atoms))
        for i in range(1, N_im-1):
            for j in range(self.n_atoms):
                atom_disp = R_new[i, 3*j:3*(j+1)] - R[i, 3*j:3*(j+1)]
                steplength_atomwise[i-1, j] = np.linalg.norm(atom_disp)
        
        # Calculate limits
        if self.atomic_structure is not None:
            steplength_atomwise_limit = 0.5 * (1.0 - self.ratio_at_limit) * self._mindist_interatomic(R[1:(N_im-1),:])
        else:
            # Fallback: use max_step_size
            steplength_atomwise_limit = self.max_step_size * np.ones((N_im-2, self.n_atoms))
        
        # Check if limiting needed
        if any(steplength > 0.99 * self.disp_max * self.scale) or np.any(steplength_atomwise > 0.99 * steplength_atomwise_limit):
            step_coeff = np.min((np.ones(N_im-2), 0.99 * self.disp_max * self.scale / steplength), 0)
            step_coeff = np.min((step_coeff, 0.99 * np.min(steplength_atomwise_limit / (steplength_atomwise + 1e-10), 1)), 0)
            
            if self.verbose:
                print(f'Warning: step length limited')
            
            R_new[1:(N_im-1),:] = R[1:(N_im-1),:] + step_coeff[:,None] * (R_new[1:(N_im-1),:] - R[1:(N_im-1),:])
            self.zeroV = 1
        
        return R_new
    
    def _mindist_interatomic(self, positions: npt.NDArray) -> npt.NDArray:
        """Calculate minimum inter-atomic distances for each image."""
        n_images = len(positions)
        min_dists = np.zeros((n_images, self.n_atoms))
        
        for i in range(n_images):
            pos_3d = positions[i].reshape(-1, 3)
            for j in range(self.n_atoms):
                dists = []
                for k in range(self.n_atoms):
                    if j != k:
                        dist = np.linalg.norm(pos_3d[j] - pos_3d[k])
                        dists.append(dist)
                min_dists[i, j] = min(dists) if dists else 1.0
        
        return min_dists
    
    def _check_interatomic_distances(self, R_new: npt.NDArray[np.float64]) -> Tuple[bool, Optional[int]]:
        """Check if inter-atomic distances changed too much.
        
        Args:
            R_new: Positions of intermediate images only (shape: (n_images-2, n_dof))
        
        Returns:
            (should_stop, problematic_image_index)
        """
        if self.atomic_structure is None or len(self.R_all) == 0:
            return False, None
        
        # Check each intermediate image
        for i in range(len(R_new)):
            if self.atomic_structure.check_interatomic_distances(
                R_new[i], self.R_all, self.ratio_at_limit
            ):
                return True, i + 1  # Convert to 1-based image index
        
        return False, None
    
    def _check_displacement(self, R_new: npt.NDArray[np.float64]) -> Tuple[bool, Optional[int]]:
        """Check if displacement from nearest observed point is too large.
        
        Args:
            R_new: Positions of intermediate images only (shape: (n_images-2, n_dof))
        
        Returns:
            (should_stop, problematic_image_index)
        """
        if len(self.R_all) == 0:
            return False, None
        
        disp_nearest = np.zeros((len(R_new), 1))
        for i in range(len(R_new)):
            disp_nearest[i, 0] = np.sqrt(np.min(np.sum(np.square(R_new[i,:] - self.R_all), 1)))
        
        if np.max(disp_nearest) > self.disp_max * self.scale:
            problematic_idx = np.argmax(disp_nearest) + 1  # Convert to 1-based image index
            return True, problematic_idx
        
        return False, None
    
    def _get_hessian_points(self, R_init: npt.NDArray, eps: float) -> npt.NDArray:
        """Generate virtual Hessian points around minima.
        
        Following the demo script approach for better GP training.
        """
        min1 = R_init[0, :]
        min2 = R_init[-1, :]
        D = self.n_dof
        
        R_h = np.zeros((2*D, D))
        
        # Points around minimum 1 - perturb each coordinate
        for d in range(D):
            R_h[d, :] = min1.copy()
            R_h[d, d] += eps
        
        # Points around minimum 2 - perturb each coordinate
        for d in range(D):
            R_h[D+d, :] = min2.copy()
            R_h[D+d, d] += eps
        
        if self.verbose:
            print(f"Generated {2*D} virtual Hessian points with epsilon={eps}")
        
        return R_h
    
    def _train_gp2(self, reinit_hyperparams: bool = False):
        """Train or update GP2 model with current data."""
        if len(self.R_all) < 2:
            if self.verbose:
                print("Not enough data to train GP2")
            return
        
        # Extract moving positions and forces
        positions_moving = []
        forces_moving = []
        
        for i in range(len(self.R_all)):
            pos_3d = self.R_all[i].reshape(-1, 3)
            pos_moving = pos_3d[self.moving_indices].flatten()
            positions_moving.append(pos_moving)
            
            # Extract forces for moving atoms (gradient = -force)
            grad_3d = self.G_all[i].reshape(-1, 3)
            force_3d = -grad_3d  # Convert gradient to force
            force_moving = force_3d[self.moving_indices].flatten()
            forces_moving.append(force_moving)
        
        positions_moving = np.array(positions_moving)
        forces_moving = np.array(forces_moving)
        energies = self.E_all.flatten()
        
        # Create or update GP2
        if self.gp2 is None:
            # Initialize GP2
            training_data = [positions_moving, energies, forces_moving]
            self.gp2 = GP2(
                training_data=training_data,
                atomic_info=self.atomic_info
            )
            self.gp2.model_type = self.model_type
            self.gp2.energy_reference = self.energy_reference
        else:
            # Update existing GP2
            self.gp2.atomic_info = self.atomic_info
            self.gp2.training_data = [positions_moving, energies, forces_moving]
        
        # Train the model
        self.gp2.set_current_iteration(f"bigiter_{self.bigiter}")
        
        if self.verbose:
            print(f"\nTraining GP2 with {len(positions_moving)} data points")
            print(f"  Energy range: [{energies.min():.4f}, {energies.max():.4f}] eV")
            print(f"  n_pt: {self.atomic_info.get('n_pt')}")
        
        # Set hyperparameters if reinitializing
        if reinit_hyperparams:
            # Calculate appropriate hyperparameter ranges
            if self.bigiter < self.num_bigiter_hess and len(self.R_h) > 0:
                # Exclude Hessian points for range calculation
                start_idx = 2 * self.n_dof
                mean_y = np.mean(self.E_all[start_idx:])
                range_y = np.max(self.E_all[start_idx:]) - np.min(self.E_all[start_idx:])
            else:
                mean_y = np.mean(self.E_all)
                range_y = np.max(self.E_all) - np.min(self.E_all)
            
            # Set initial hyperparameters (following reference)
            if self.verbose:
                print(f"  Setting initial hyperparameters:")
                print(f"    Energy mean: {mean_y:.4f}")
                print(f"    Energy range: {range_y:.4f}")
        
        self.gp2.train(
            training_data=self.gp2.training_data,
            thermal_noise=None,
            model_name="GP2", 
            path=get_output_path('data_gp2'),
        )
        self.gp2_trained = True
        
        # Store hyperparameters
        if hasattr(self.gp2, 'model') and self.gp2.model is not None:
            self.param_gp.append(self._extract_hyperparameters())
    
    def _extract_hyperparameters(self) -> Dict:
        """Extract current GP hyperparameters."""
        params = {}
        if hasattr(self.gp2, 'model') and self.gp2.model is not None:
            model = self.gp2.model
            if hasattr(model, 'covar_module'):
                covar = model.covar_module
                if hasattr(covar, 'data_covar_module'):
                    base_kernel = covar.data_covar_module
                    if hasattr(base_kernel, 'magnitude'):
                        params['magnitude'] = float(base_kernel.magnitude.detach().cpu().numpy())
                    if hasattr(base_kernel, 'lengthscale'):
                        params['lengthscales'] = base_kernel.lengthscale.detach().cpu().numpy().tolist()
        return params
    
    def _gp_evaluate_path(self, R: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Evaluate entire path using GP model.
        
        Returns:
            (energies, gradients)  # Shape: (N_images, 1) and (N_images, n_dof)
        """
        if not self.gp2_trained:
            return np.zeros((self.n_images, 1)), np.zeros((self.n_images, self.n_dof))
        
        E_gp = np.zeros((self.n_images, 1))
        G_gp = np.zeros((self.n_images, self.n_dof))
        
        for i in range(self.n_images):
            # Extract moving positions
            pos_3d = R[i].reshape(-1, 3)
            pos_moving = pos_3d[self.moving_indices].flatten()
            
            # Get GP predictions
            pred_e, pred_f, _, _ = self.gp2.predict(pos_moving.reshape(1, -1))
            
            # Store energy
            E_gp[i, 0] = float(pred_e[0])
            
            # Reconstruct full gradients (gradient = -force)
            for j, idx in enumerate(self.moving_indices):
                G_gp[i, 3*idx:3*(idx+1)] = -pred_f[0, 3*j:3*(j+1)]
        
        return E_gp, G_gp
    
    def _relaxation_phase(self):
        """Perform one relaxation phase on GP surface."""
        if self.verbose:
            print(f"\n" + "="*60)
            print(f"RELAXATION PHASE {self.bigiter}")
            print("="*60)
        
        # Train GP with all data
        self._train_gp2(reinit_hyperparams=(self.bigiter <= self.num_bigiter_initparam))
        
        # Calculate convergence threshold for GP surface
        if self.divisor_T_MEP_gp > 0 and len(self.normF_R_acc.flatten()) > 0:
            T_MEP_gp = max(
                np.min(self.normF_R_acc) / self.divisor_T_MEP_gp,
                min(self.T_MEP / 10.0, self.T_CI / 10.0)
            )
        else:
            T_MEP_gp = min(self.T_MEP, self.T_CI) / 10.0
        
        if self.verbose:
            print(f"GP convergence threshold: {T_MEP_gp:.6f} eV/Å")
        
        # Set initial path for relaxation
        if self.bigiter > self.num_bigiter_init and self.R_latest_equal.shape[0] > 0:
            self.R = self.R_latest_equal.copy()
            if self.T_CIon_gp > 0:
                print('Started from latest "preliminarily converged" evenly spaced path')
            else:
                print('Started from latest converged path')
        else:
            self.R = self.initial_path.copy()
            print('Started from initial path')
        
        # Reset climbing image and velocities
        self.CI_on = 0
        self.V_old = np.zeros((self.n_images-2, self.n_dof))
        self.F_R_old = np.zeros((self.n_images-2, self.n_dof))
        self.zeroV = 1
        
        # Track convergence
        converged = False
        stopped_early = False
        stop_reason = ""
        
        # Inner iteration loop
        for inner_iter in range(self.max_inner_iterations + 1):
            # Get GP predictions
            E_R_gp, G_R_gp = self._gp_evaluate_path(self.R)
            F_R_gp, normFCI_gp, i_CI_gp = self._force_sNEB(self.R, E_R_gp, G_R_gp, self.CI_on)
            normF_R_gp = np.sqrt(np.sum(np.square(F_R_gp), 1))
            
            # Turn on climbing image if threshold reached
            if self.CI_on <= 0 and self.T_CIon_gp > 0 and np.max(normF_R_gp) < self.T_CIon_gp:
                self.R_latest_equal = self.R.copy()
                self.CI_on = 1
                i_CI_test = np.argmax(E_R_gp[1:-1, 0])
                print(f'Climbing image (image {i_CI_test+2}) turned on after {inner_iter} inner iterations')
                
                # Check if CI unchanged from previous phase
                if self.bigiter > self.num_bigiter_init and i_CI_test == self.i_CI_latest and self.R_latest_climb is not None:
                    # Test if CI position is still the same
                    E_R_test, _ = self._gp_evaluate_path(self.R_latest_climb)
                    i_CI_test2 = np.argmax(E_R_test[1:-1, 0])
                    if i_CI_test2 == self.i_CI_latest:
                        self.R = self.R_latest_climb.copy()
                        E_R_gp, G_R_gp = self._gp_evaluate_path(self.R)
                        print('CI unchanged: continued from latest converged CI-path')
                
                # Recalculate forces with CI
                F_R_gp, normFCI_gp, i_CI_gp = self._force_sNEB(self.R, E_R_gp, G_R_gp, self.CI_on)
                normF_R_gp = np.sqrt(np.sum(np.square(F_R_gp), 1))
                self.zeroV = 1
            
            # Store GP trajectory
            self.E_R_gp = np.hstack((self.E_R_gp, E_R_gp))
            self.normF_R_gp = np.hstack((self.normF_R_gp, normF_R_gp[:, np.newaxis]))
            self.normFCI_gp = np.hstack((self.normFCI_gp, normFCI_gp))
            
            # Check convergence on GP surface
            if (self.T_CIon_gp <= 0 or self.CI_on > 0) and np.max(normF_R_gp) < T_MEP_gp and inner_iter > 0:
                if self.CI_on > 0:
                    self.R_latest_climb = self.R.copy()
                    self.i_CI_latest = i_CI_gp
                    print(f'Converged on GP surface after {inner_iter} iterations (CI: image {i_CI_gp+2})')
                else:
                    self.R_latest_equal = self.R.copy()
                    print(f'Converged on GP surface after {inner_iter} iterations')
                converged = True
                break
            
            # Check if max iterations reached
            if inner_iter == self.max_inner_iterations:
                print(f'Max inner iterations ({inner_iter}) reached')
                stopped_early = True
                stop_reason = "max iterations"
                break
            
            # Move images
            R_new = self._step_translation(self.R, F_R_gp)
            
            # Update active atoms if needed
            if self.atomic_structure is not None:
                n_activated_total = 0
                for i in range(1, self.n_images-1):
                    n_activated = self.atomic_structure.update_activated_atoms_wrapper(R_new[i], self.verbose and i == 1)
                    n_activated_total += n_activated
                
                if n_activated_total > 0:
                    # Update atomic info
                    self.atomic_info.update(self.atomic_structure.get_structure_info())
                    # Retrain GP with new atomic structure
                    self._train_gp2(reinit_hyperparams=False)
                    self.zeroV = 1
            
            # Limit step if needed
            R_new = self._limit_step(R_new, self.R)
            
            # Check stopping criteria (only after first iteration)
            if inner_iter > 0:
                # Check inter-atomic distances
                stop, problem_img = self._check_interatomic_distances(R_new[1:-1])
                if stop:
                    print(f'Stopped: inter-atomic distance in image {problem_img} changes too much')
                    stopped_early = True
                    stop_reason = "inter-atomic distances"
                    break
                
                # Check displacement
                stop, problem_img = self._check_displacement(R_new[1:-1])
                if stop:
                    print(f'Stopped: image {problem_img} too far from nearest observed point')
                    stopped_early = True
                    stop_reason = "displacement"
                    break
            
            # Accept step
            self.R = R_new.copy()
            self.F_R_old = F_R_gp.copy()
        
        # Store observation point
        self.obs_at.append(self.E_R_gp.shape[1])
        
        # Evaluate intermediate images on accurate surface
        image_ids = list(range(1, self.n_images-1))
        E_inter, G_inter = self._evaluate_images(self.R[1:-1], image_ids)
        
        # Add to observations
        for i in range(self.n_images-2):
            self.R_all = np.vstack((self.R_all, self.R[i+1].reshape(1, -1)))
            self.E_all = np.vstack((self.E_all, E_inter[i].reshape(1, 1)))
            self.G_all = np.vstack((self.G_all, G_inter[i].reshape(1, -1)))
        
        # Update full path energies and gradients
        self.E_R[1:-1] = E_inter
        self.G_R[1:-1] = G_inter
        
        # Calculate accurate forces
        F_R_acc, normFCI_acc, i_CI_acc = self._force_sNEB(self.R, self.E_R, self.G_R, self.CI_on)
        normF_R_acc = np.sqrt(np.sum(np.square(F_R_acc), 1))
        
        # Store accuracy tracking
        self.E_R_acc = np.hstack((self.E_R_acc, self.E_R))
        self.normF_R_acc = np.hstack((self.normF_R_acc, normF_R_acc[:, np.newaxis]))
        self.normFCI_acc = np.hstack((self.normFCI_acc, normFCI_acc))
        
        if self.verbose:
            if self.T_CIon_gp > 0:
                print(f'\nAccurate values: meanE_R = {np.mean(self.E_R[1:-1]):.3g}, '
                      f'maxnormF_R = {np.max(normF_R_acc):.3g}, '
                      f'minnormF_R = {np.min(normF_R_acc):.3g}, '
                      f'normFCI = {normFCI_acc:.3g} (image {i_CI_acc+2})')
            else:
                print(f'\nAccurate values: meanE_R = {np.mean(self.E_R[1:-1]):.3g}, '
                      f'maxnormF_R = {np.max(normF_R_acc):.3g}, '
                      f'minnormF_R = {np.min(normF_R_acc):.3g}')
        
        # Update table history
        table_entry = {
            'step': self.bigiter,
            'E_mean': np.mean(self.E_R[1:-1]),
            'E_max': np.max(self.E_R[1:-1]),
            'maxF': np.max(normF_R_acc),
            'minF': np.min(normF_R_acc),
            'normFCI': normFCI_acc if self.CI_on else np.nan,
            'CI_img': i_CI_acc + 2 if self.CI_on else -1,
            'inner_iters': inner_iter,
            'obs_total': self.obs_total
        }
        self.table_history.append(table_entry)
    
    def run(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, int]:
        """Run GP2-accelerated NEB search.
        
        Returns:
            (final_positions, final_energies, final_gradients, climbing_image_index)
        """
        if self.verbose:
            print("\nStarting GP2-accelerated NEB search...")
        
        # Evaluate endpoints
        if self.verbose:
            print("\nEvaluating endpoint minima...")
        
        # Minimum 1
        E_min1, G_min1 = self._evaluate_image(self.R[0], image_id=0)
        self.E_R[0, 0] = E_min1
        self.G_R[0, :] = G_min1
        
        # Minimum 2
        E_min2, G_min2 = self._evaluate_image(self.R[-1], image_id=self.n_images-1)
        self.E_R[-1, 0] = E_min2
        self.G_R[-1, :] = G_min2
        
        if self.verbose:
            print(f"Minimum 1: E = {E_min1:.6f} eV")
            print(f"Minimum 2: E = {E_min2:.6f} eV")
            print(f"Energy difference: {E_min2 - E_min1:.6f} eV")
        
        # Add endpoints to observations
        self.R_all = np.vstack((self.R[0].reshape(1, -1), self.R[-1].reshape(1, -1)))
        self.E_all = np.vstack(([[E_min1]], [[E_min2]]))
        self.G_all = np.vstack((G_min1.reshape(1, -1), G_min2.reshape(1, -1)))
        
        # Activate frozen atoms near endpoints if needed
        if self.atomic_structure is not None:
            n_activated = 0
            n_activated += self.atomic_structure.update_activated_atoms_wrapper(self.R[0], False)
            n_activated += self.atomic_structure.update_activated_atoms_wrapper(self.R[-1], False)
            if n_activated > 0 and self.verbose:
                print(f"Activated {n_activated} frozen atoms near endpoints")
                self.atomic_info.update(self.atomic_structure.get_structure_info())
                print(f"Active pair types (n_pt): {self.atomic_info['n_pt']}")
        
        # Define virtual Hessian points if used
        if self.num_bigiter_hess > 0:
            self.R_h = self._get_hessian_points(self.initial_path, self.eps_hess)
            E_h, G_h = self._evaluate_images(self.R_h)
            self.E_h = E_h
            self.G_h = G_h
            
            # Check for additional frozen atom activation from Hessian points
            if self.atomic_structure is not None:
                n_activated_hess = 0
                for i in range(len(self.R_h)):
                    n_activated_hess += self.atomic_structure.update_activated_atoms_wrapper(
                        self.R_h[i], verbose=False
                    )
                if n_activated_hess > 0 and self.verbose:
                    print(f"Activated {n_activated_hess} additional frozen atoms from Hessian points")
                    self.atomic_info.update(self.atomic_structure.get_structure_info())
            
            # Add Hessian points to beginning of observations
            self.R_all = np.vstack((self.R_h, self.R_all))
            self.E_all = np.vstack((self.E_h, self.E_all))
            self.G_all = np.vstack((self.G_h, self.G_all))
        
        # Visualization setup
        if self.visualize:
            fig1 = plt.figure(figsize=(10, 6))
            plt.title('Energy along NEB path')
            plt.xlabel('Image number')
            plt.ylabel('Energy (eV)')
            self.figs.append(fig1)
            plt.ion()
        
        # Main outer iteration loop
        for self.bigiter in range(self.max_neb_steps + 1):
            # Evaluate intermediate images on accurate surface
            if self.bigiter == 0:
                # First iteration: evaluate all intermediate images
                image_ids = list(range(1, self.n_images-1))
                E_inter, G_inter = self._evaluate_images(self.R[1:-1], image_ids)
                
                # Add to observations
                for i in range(self.n_images-2):
                    self.R_all = np.vstack((self.R_all, self.R[i+1].reshape(1, -1)))
                    self.E_all = np.vstack((self.E_all, E_inter[i].reshape(1, 1)))
                    self.G_all = np.vstack((self.G_all, G_inter[i].reshape(1, -1)))
                
                # Update path energies and gradients
                self.E_R[1:-1] = E_inter
                self.G_R[1:-1] = G_inter
                
                # Calculate accurate forces
                F_R_acc, normFCI_acc, i_CI_acc = self._force_sNEB(self.R, self.E_R, self.G_R, 0)
                normF_R_acc = np.sqrt(np.sum(np.square(F_R_acc), 1))
                
                # Store accuracy tracking
                self.E_R_acc = np.hstack((self.E_R_acc, self.E_R))
                self.normF_R_acc = np.hstack((self.normF_R_acc, normF_R_acc[:, np.newaxis]))
                self.normFCI_acc = np.hstack((self.normFCI_acc, 0.0))
                self.obs_at.append(0)
                
                if self.verbose:
                    print(f'\nInitial accurate values: meanE_R = {np.mean(self.E_R[1:-1]):.3g}, '
                          f'maxnormF_R = {np.max(normF_R_acc):.3g}, '
                          f'minnormF_R = {np.min(normF_R_acc):.3g}')
                
                # Initial table entry
                table_entry = {
                    'step': 0,
                    'E_mean': np.mean(self.E_R[1:-1]),
                    'E_max': np.max(self.E_R[1:-1]),
                    'maxF': np.max(normF_R_acc),
                    'minF': np.min(normF_R_acc),
                    'normFCI': np.nan,
                    'CI_img': -1,
                    'inner_iters': 0,
                    'obs_total': self.obs_total
                }
                self.table_history.append(table_entry)
            
            # Check convergence
            if self.bigiter > 0:
                max_force = np.max(self.normF_R_acc[:, -1])
                ci_force = self.normFCI_acc[-1] if len(self.normFCI_acc) > 0 else 0.0
                
                if max_force < self.T_MEP and ci_force < self.T_CI:
                    self.converged = True
                    if self.verbose:
                        print(f"\n" + "="*80)
                        print(f"CONVERGED after {self.bigiter} relaxation phases")
                        print(f"Total evaluations: {self.obs_total}")
                        print("="*80)
                    break
            
            # Check if max iterations reached
            if self.bigiter == self.max_neb_steps:
                if self.verbose:
                    print(f"\n" + "="*80)
                    print(f"MAXIMUM ITERATIONS REACHED")
                    print(f"Total evaluations: {self.obs_total}")
                    print("="*80)
                break
            
            # Save checkpoint
            if self.bigiter % self.checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Remove virtual Hessian observations if needed
            if self.num_bigiter_hess > 0 and self.bigiter == self.num_bigiter_hess:
                n_hess = 2 * self.n_dof
                self.R_all = self.R_all[n_hess:]
                self.E_all = self.E_all[n_hess:]
                self.G_all = self.G_all[n_hess:]
            
            # Visualize if requested
            if self.visualize:
                self._visualize_path()
            
            # Print progress table
            if self.table_history:
                self._print_progress_table()
            
            # Perform relaxation phase on GP surface
            self._relaxation_phase()
        
        # Save final checkpoint
        self.save_checkpoint(final=True)
        
        # Final visualization
        if self.visualize:
            self._visualize_path(final=True)
            plt.ioff()
            plt.show()
        
        # Print summary
        if self.verbose:
            self._print_summary()
        
        # Return final results
        i_CI = self.i_CI_latest if self.CI_on else -1
        return self.R, self.E_R, self.G_R, i_CI
    
    def _visualize_path(self, final=False):
        """Visualize energy along the path."""
        plt.figure(self.figs[0].number)
        plt.clf()
        
        # Plot current path
        x = np.arange(1, self.n_images + 1)
        plt.plot(x, self.E_R[:, 0], 'bo-', markersize=8, linewidth=2, label='Current')
        
        # Mark climbing image if active
        if self.CI_on and hasattr(self, 'i_CI_latest') and self.i_CI_latest >= 0:
            plt.plot(self.i_CI_latest + 2, self.E_R[self.i_CI_latest + 1, 0], 
                    'r*', markersize=15, label='Climbing Image')
        
        # Add spline interpolation if requested
        if final or self.bigiter % 5 == 0:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(np.arange(self.n_images), self.R)
            x_fine = np.linspace(0, self.n_images - 1, 100)
            R_spline = cs(x_fine)
            
            # Would need to evaluate on spline for true visualization
            # For now just show linear interpolation
            E_interp = np.interp(x_fine, np.arange(self.n_images), self.E_R[:, 0])
            plt.plot(x_fine + 1, E_interp, 'r--', alpha=0.5, label='Interpolated')
        
        plt.xlabel('Image Number')
        plt.ylabel('Energy (eV)')
        plt.title(f'NEB Energy Profile - Iteration {self.bigiter}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if not final:
            plt.pause(0.1)
    
    def _print_progress_table(self):
        """Print progress table with history."""
        print("\n" + "="*120)
        print("NEB PROGRESS TABLE")
        print("="*120)
        print(f"{'Step':>6} {'E_mean':>12} {'E_max':>12} {'Max |F|':>12} {'Min |F|':>12} "
              f"{'|F_CI|':>12} {'CI Img':>8} {'Inner':>8} {'Total Obs':>10}")
        print(f"{' ':>6} {'(eV)':>12} {'(eV)':>12} {'(eV/Å)':>12} {'(eV/Å)':>12} "
              f"{'(eV/Å)':>12} {' ':>8} {'Iters':>8} {' ':>10}")
        print("-"*120)
        
        # Show last 20 entries (or all if fewer)
        start_idx = max(0, len(self.table_history) - 20)
        for row in self.table_history[start_idx:]:
            ci_force_str = f"{row['normFCI']:12.6f}" if not np.isnan(row['normFCI']) else f"{'---':>12}"
            ci_img_str = f"{row['CI_img']:8d}" if row['CI_img'] > 0 else f"{'---':>8}"
            
            print(f"{row['step']:6d} {row['E_mean']:12.6f} {row['E_max']:12.6f} "
                  f"{row['maxF']:12.6f} {row['minF']:12.6f} {ci_force_str} "
                  f"{ci_img_str} {row['inner_iters']:8d} {row['obs_total']:10d}")
        
        print("="*120)
    
    def _print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Converged: {self.converged}")
        print(f"Outer iterations (relaxation phases): {self.bigiter}")
        print(f"Total accurate evaluations: {self.obs_total}")
        print(f"Total GP evaluations: {len(self.E_R_gp.flatten())}")
        
        if self.obs_total > 0:
            speedup = len(self.E_R_gp.flatten()) / self.obs_total
            print(f"Speedup factor: {speedup:.1f}x")
        
        # Energy barrier
        E_max = np.max(self.E_R[1:-1, 0])
        E_min1 = self.E_R[0, 0]
        E_min2 = self.E_R[-1, 0]
        
        print(f"\nEnergy barrier:")
        print(f"  Forward: {E_max - E_min1:.6f} eV")
        print(f"  Reverse: {E_max - E_min2:.6f} eV")
        print(f"  Saddle point energy: {E_max:.6f} eV")
        
        if self.CI_on and hasattr(self, 'i_CI_latest'):
            print(f"  Climbing image: {self.i_CI_latest + 2} (1-based indexing)")
        
        # Path length
        path_length = 0.0
        for i in range(1, self.n_images):
            path_length += np.linalg.norm(self.R[i, :] - self.R[i-1, :])
        print(f"\nFinal path length: {path_length:.4f} Å")
        print(f"Initial path length: {self.scale:.4f} Å")
        print(f"Path length change: {(path_length/self.scale - 1)*100:.1f}%")
        
        # Force evolution
        if len(self.normF_R_acc) > 0:
            print(f"\nForce evolution:")
            print(f"  Initial max |F|: {np.max(self.normF_R_acc[:, 0]):.6f} eV/Å")
            print(f"  Final max |F|: {np.max(self.normF_R_acc[:, -1]):.6f} eV/Å")
            if self.CI_on and len(self.normFCI_acc) > 0:
                print(f"  Final CI |F|: {self.normFCI_acc[-1]:.6f} eV/Å")
        
        # Frozen atom statistics
        if self.atomic_structure is not None:
            print(f"\nAtomic structure:")
            print(f"  Active frozen atoms: {len(self.atomic_structure.active_frozen_atoms)}")
            print(f"  Inactive frozen atoms: {len(self.atomic_structure.inactive_frozen_atoms)}")
            print(f"  Active pair types: {self.atomic_info.get('n_pt', 0)}")
        
        print("="*80)
    
    def save_checkpoint(self, final=False):
        """Save checkpoint data."""
        checkpoint_data = {
            'walker_type': 'WalkerGP2NEB',
            'iteration': self.bigiter,
            'converged': self.converged,
            'energy_reference': self.energy_reference,
            'reference_set': self.reference_set,
            'n_images': self.n_images,
            'n_atoms': self.n_atoms,
            'n_dof': self.n_dof,
            'scale': self.scale,
            'R': self.R,
            'E_R': self.E_R,
            'G_R': self.G_R,
            'R_all': self.R_all,
            'E_all': self.E_all,
            'G_all': self.G_all,
            'R_latest_equal': self.R_latest_equal,
            'R_latest_climb': self.R_latest_climb,
            'i_CI_latest': self.i_CI_latest,
            'CI_on': self.CI_on,
            'i_CI': self.i_CI,
            'V_old': self.V_old,
            'F_R_old': self.F_R_old,
            'zeroV': self.zeroV,
            'E_R_acc': self.E_R_acc,
            'E_R_gp': self.E_R_gp,
            'normF_R_acc': self.normF_R_acc,
            'normF_R_gp': self.normF_R_gp,
            'normFCI_acc': self.normFCI_acc,
            'normFCI_gp': self.normFCI_gp,
            'obs_total': self.obs_total,
            'obs_at': self.obs_at,
            'param_gp': self.param_gp,
            'atomic_info': self.atomic_info,
            'table_history': self.table_history,
            'k_par': self.k_par,
            'k_perp': self.k_perp,
            'T_MEP': self.T_MEP,
            'T_CI': self.T_CI,
            'T_CIon_gp': self.T_CIon_gp
        }
        
        checkpoint_dir = get_output_path('checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if final:
            filename = os.path.join(checkpoint_dir, 'gp2_neb_final.pkl')
        else:
            filename = os.path.join(checkpoint_dir, f'gp2_neb_checkpoint_{self.bigiter}.pkl')
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save as latest
        latest_filename = os.path.join(checkpoint_dir, 'gp2_neb_latest.pkl')
        with open(latest_filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        if self.verbose and (self.bigiter % 10 == 0 or final):
            print(f"  [Checkpoint saved at iteration {self.bigiter}]")
    
    def load_checkpoint(self, checkpoint_file=None):
        """Load checkpoint and restore state."""
        if checkpoint_file is None:
            checkpoint_file = get_output_path('checkpoints', 'gp2_neb_latest.pkl')
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state
        self.bigiter = checkpoint['iteration']
        self.converged = checkpoint['converged']
        self.energy_reference = checkpoint['energy_reference']
        self.reference_set = checkpoint['reference_set']
        self.n_images = checkpoint['n_images']
        self.n_atoms = checkpoint['n_atoms']
        self.n_dof = checkpoint['n_dof']
        self.scale = checkpoint['scale']
        self.R = checkpoint['R']
        self.E_R = checkpoint['E_R']
        self.G_R = checkpoint['G_R']
        self.R_all = checkpoint['R_all']
        self.E_all = checkpoint['E_all']
        self.G_all = checkpoint['G_all']
        self.R_latest_equal = checkpoint['R_latest_equal']
        self.R_latest_climb = checkpoint['R_latest_climb']
        self.i_CI_latest = checkpoint['i_CI_latest']
        self.CI_on = checkpoint['CI_on']
        self.i_CI = checkpoint['i_CI']
        self.V_old = checkpoint['V_old']
        self.F_R_old = checkpoint['F_R_old']
        self.zeroV = checkpoint['zeroV']
        self.E_R_acc = checkpoint['E_R_acc']
        self.E_R_gp = checkpoint['E_R_gp']
        self.normF_R_acc = checkpoint['normF_R_acc']
        self.normF_R_gp = checkpoint['normF_R_gp']
        self.normFCI_acc = checkpoint['normFCI_acc']
        self.normFCI_gp = checkpoint['normFCI_gp']
        self.obs_total = checkpoint['obs_total']
        self.obs_at = checkpoint['obs_at']
        self.param_gp = checkpoint['param_gp']
        self.table_history = checkpoint['table_history']
        
        # Retrain GP with loaded data
        if len(self.R_all) > 1:
            self._train_gp2()
        
        if self.verbose:
            print(f"\nRestored checkpoint from iteration {self.bigiter}")
            print(f"Total observations so far: {self.obs_total}")
            print(f"Table history entries: {len(self.table_history)}")
        
        return checkpoint