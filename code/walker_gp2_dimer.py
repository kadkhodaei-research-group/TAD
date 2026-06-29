"""walker_gp2_dimer.py - GP2-accelerated dimer walker following atomic GP-dimer algorithm."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, Tuple, List
import os
import pickle
import time
from scipy.stats import norm

from dimer import Dimer
from gp2_model import GP2
from atomic_structure import AtomicStructure, calculate_atomic_distance_measure
from gp_data_saver import get_gp_logger
from output_manager import get_output_path

logger = logging.getLogger(__name__)


class WalkerGP2Dimer:
    """GP2-accelerated dimer method following atomic GP-dimer algorithm."""
    
    def __init__(
        self,
        initial_position: npt.NDArray[np.float64],
        local_pes: Any,  # VASP interface
        max_dimer_steps: int = 100,
        # Stopping criteria
        disp_max: float = 0.5,
        ratio_at_limit: float = 2.0/3.0,
        # Dimer parameters
        rotation: str = "lbfgsext",
        translation: str = "lbfgs",
        dimer_sep: float = 0.01,
        T_anglerot: float = 0.01,
        T_anglerot_init: float = 0.0873,
        T_anglerot_gp: float = 0.01,
        max_dimer_rotations: int = 10,
        num_init_rotations: int = 5,
        num_iter_rot_gp: int = 10,
        dimer_stopping_criteria: float = 0.01,
        step_size: float = 0.1,
        max_step_size: float = 0.1,
        # GP convergence
        divisor_T_dimer_gp: float = 10.0,
        max_inner_iterations: int = 1000,
        # Options
        initrot_nogp: bool = False,
        inittrans_nogp: bool = False,
        eval_image1: bool = False,
        num_bigiter_initloc: float = np.inf,
        num_bigiter_initparam: float = np.inf,
        # Other parameters
        verbose: bool = False,
        checkpoint_interval: int = 1,
        model_type: str = "MultitaskGPModel_rbf_atomic",
        use_gpu: bool = False,
        **kwargs
    ) -> None:
        """Initialize GP2-accelerated dimer walker.
        
        Args:
            initial_position: Initial atomic positions (full system, flattened)
            local_pes: VASP interface for energy/force calculations
            max_dimer_steps: Maximum number of outer iterations
            disp_max: Maximum displacement from nearest observed point
            ratio_at_limit: Limit for inter-atomic distance ratio
            rotation: Rotation method
            translation: Translation method
            dimer_sep: Dimer separation distance
            T_anglerot: Rotation convergence threshold
            T_anglerot_init: Initial rotation convergence threshold
            T_anglerot_gp: Rotation convergence on GP surface
            max_dimer_rotations: Max rotations per translation
            num_init_rotations: Number of initial rotations
            num_iter_rot_gp: Max rotations per translation on GP
            dimer_stopping_criteria: Force convergence threshold
            step_size: Base step size for translations
            max_step_size: Maximum allowed step size
            divisor_T_dimer_gp: Divisor for dynamic GP convergence
            max_inner_iterations: Max iterations per relaxation phase
            initrot_nogp: Perform initial rotations without GP
            inittrans_nogp: Perform initial translation without GP
            eval_image1: Evaluate image 1 after each phase
            num_bigiter_initloc: Number of iterations from initial location
            num_bigiter_initparam: Number of iterations with fresh hyperparameters
            verbose: Enable verbose output
            checkpoint_interval: Save checkpoint every N iterations
            model_type: GP model type to use
            use_gpu: Use GPU acceleration for GP training and inference
        """
        self.initial_position = initial_position.copy()
        self.local_pes = local_pes
        self.max_dimer_steps = max_dimer_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.model_type = model_type
        self.use_gpu = use_gpu
        
        # Store rotation and translation methods
        self.rotation = rotation
        self.translation = translation
        
        # Stopping criteria
        self.disp_max = disp_max
        self.ratio_at_limit = ratio_at_limit
        self.divisor_T_dimer_gp = divisor_T_dimer_gp
        self.max_inner_iterations = max_inner_iterations
        
        # Dimer parameters
        self.dimer_sep = dimer_sep
        self.T_anglerot = T_anglerot
        self.T_anglerot_init = T_anglerot_init
        self.T_anglerot_gp = T_anglerot_gp
        self.max_dimer_rotations = max_dimer_rotations
        self.num_init_rotations = num_init_rotations
        self.num_iter_rot_gp = num_iter_rot_gp
        self.dimer_stopping_criteria = dimer_stopping_criteria
        self.step_size = step_size
        self.max_step_size = max_step_size
        
        # Options
        self.initrot_nogp = initrot_nogp
        self.inittrans_nogp = inittrans_nogp
        self.eval_image1 = eval_image1
        self.num_bigiter_initloc = num_bigiter_initloc
        self.num_bigiter_initparam = num_bigiter_initparam
        
        # Get atomic info from local_pes (following walker.py approach)
        self.atomic_info = self.local_pes.get_atomic_info()

        # Validate atomic info (from walker.py)
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
        
        self.n_atoms = len(initial_position) // 3
        self.n_dof = len(initial_position)
        
        # Get atomic structure from local_pes if available (from walker.py)
        if hasattr(self.local_pes, 'atomic_structure'):
            self.atomic_structure = self.local_pes.atomic_structure
        else:
            # If local_pes doesn't have atomic_structure, we need to handle this
            print("Warning: local_pes doesn't have atomic_structure attribute")
            self.atomic_structure = None
        
        if self.verbose:
            print(f"\nAtomic info from local_pes:")
            print(f"  n_pt: {self.atomic_info.get('n_pt')}")
            print(f"  moving_indices: {self.atomic_info.get('moving_indices')}")
            print(f"  Number of moving atoms: {len(self.moving_indices)}")
            print(f"  Number of active frozen atoms: {len(self.atomic_info.get('atomtype_fro', []))}")
        
        # Energy reference (set on first calculation)
        self.energy_reference = None
        self.reference_set = False
        
        # Initialize GP2 model
        self.gp2 = None
        self.gp2_trained = False
        
        # Track evaluations
        self.obs_total = 0  # Total observations
        self.obs_initrot = 0  # Observations for initial rotations
        self.obs_at = []  # Observation points (inner iterations)
        
        # Tracking for adaptive threshold
        self.min_force_achieved = float('inf')
        
        # State tracking
        self.bigiter = 0  # Outer iteration counter
        self.converged = False
        
        # Data storage (following reference)
        self.R_all = np.empty((0, self.n_dof))  # All observed positions
        self.E_all = np.empty((0, 1))  # All energies
        self.G_all = np.empty((0, self.n_dof))  # All gradients (forces)
        
        # Accuracy tracking
        self.E_R_acc = []  # Accurate energies at dimer center
        self.maxF_R_acc = []  # Accurate max forces at dimer center
        self.E_R_gp = []  # GP energies (inner iterations)
        self.maxF_R_gp = []  # GP max forces (inner iterations)
        
        # Hyperparameter tracking
        self.param_gp = []  # GP hyperparameters for each outer iteration
        self.param_gp_initrot = []  # GP hyperparameters for initial rotations
        
        # Stopping criteria counters
        self.num_esmax = 0  # Stopped by max iterations
        self.num_es1 = 0  # Stopped by inter-atomic distance
        self.num_es2 = 0  # Stopped by displacement
        
        # Table history for progress tracking
        self.table_history = []
        
        # Initialize dimer
        self._init_dimer()
        
        if self.verbose:
            print("\n" + "="*60)
            print("GP2 DIMER WALKER INITIALIZED")
            print("="*60)
            print(f"System size: {self.n_atoms} atoms ({self.n_dof} DOF)")
            print(f"Moving atoms: {self.n_moving} ({self.n_moving_dof} DOF)")
            print(f"Rotation method: {rotation}")
            print(f"Translation method: {translation}")
            print(f"Dimer separation: {dimer_sep} Å")
            print(f"Convergence criteria: {dimer_stopping_criteria} eV/Å")
            print(f"Max displacement: {disp_max} Å")
            print(f"Inter-atomic ratio limit: {ratio_at_limit}")
            print("="*60 + "\n")
    
    def _init_atomic_structure(self):
        """Initialize atomic structure for managing active/frozen atoms."""
        # Extract positions for moving and frozen atoms
        full_pos_3d = self.initial_position.reshape(-1, 3)
        
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
        
        # CRITICAL: Ensure we have proper pair types for GP training
        if self.atomic_info.get('n_pt', 0) == 0:
            if self.verbose:
                print("\nWARNING: No pair types initially found!")
                
            # Force activation of nearby frozen atoms to get pair types
            if activation_radius < np.inf and len(frozen_positions) > 0:
                # Find closest frozen atoms
                distances_to_frozen = []
                for mov_pos in moving_positions:
                    for i, fro_pos in enumerate(frozen_positions):
                        dist = np.linalg.norm(mov_pos - fro_pos)
                        distances_to_frozen.append((dist, i))
                
                distances_to_frozen.sort()
                
                # Force activate at least a few closest atoms
                n_to_activate = min(10, len(frozen_positions))
                if self.verbose:
                    print(f"Force activating {n_to_activate} closest frozen atoms for GP training")
                
                for i in range(n_to_activate):
                    if i < len(distances_to_frozen):
                        _, idx = distances_to_frozen[i]
                        # Manually activate this atom
                        if idx < len(self.atomic_structure.inactive_frozen_atoms):
                            new_active = self.atomic_structure.inactive_frozen_atoms[idx:idx+1]
                            new_type = self.atomic_structure.inactive_frozen_types[idx:idx+1]
                            
                            if len(self.atomic_structure.active_frozen_atoms) == 0:
                                self.atomic_structure.active_frozen_atoms = new_active
                                self.atomic_structure.active_frozen_types = new_type
                            else:
                                self.atomic_structure.active_frozen_atoms = np.vstack((
                                    self.atomic_structure.active_frozen_atoms, new_active
                                ))
                                self.atomic_structure.active_frozen_types = np.concatenate((
                                    self.atomic_structure.active_frozen_types, new_type
                                ))
                            
                            # Update pair types
                            self.atomic_structure._update_pair_types(new_type)
                
                # Clean up inactive lists
                indices_to_remove = [distances_to_frozen[i][1] for i in range(min(n_to_activate, len(distances_to_frozen)))]
                self.atomic_structure.inactive_frozen_atoms = np.delete(
                    self.atomic_structure.inactive_frozen_atoms, indices_to_remove, axis=0
                )
                self.atomic_structure.inactive_frozen_types = np.delete(
                    self.atomic_structure.inactive_frozen_types, indices_to_remove
                )
                
                # Update atomic info
                struct_info = self.atomic_structure.get_structure_info()
                self.atomic_info.update(struct_info)
                
                if self.verbose:
                    print(f"After forced activation:")
                    print(f"  Active frozen atoms: {len(self.atomic_structure.active_frozen_atoms)}")
                    print(f"  Pair types (n_pt): {self.atomic_info['n_pt']}")
            
            # If still no pair types, we have a problem
            if self.atomic_info.get('n_pt', 0) == 0:
                print("ERROR: Still no pair types after activation!")
                print("Falling back to standard GP model")
                self.model_type = "MultitaskGPModel"
    
    def _init_dimer(self):
        """Initialize dimer with current position and orientation."""
        # Set default translation parameters
        param_trans = np.array([[self.step_size, self.max_step_size]])
        
        # Initialize dimer at current position
        self.dimer = Dimer(
            x=self.initial_position,
            force_func=self._force_func_accurate,  # Start with accurate forces
            dimer_sep=self.dimer_sep,
            rotation_method=self.rotation,
            translation=self.translation,
            opt_type="pure_dimer",
            max_dimer_rotations=self.max_dimer_rotations,
            T_anglerot=self.T_anglerot,
            T_anglerot_init=self.T_anglerot_init,
            num_iter_initrot=self.num_init_rotations,
            param_trans=param_trans,
            dimer_stopping_criteria=self.dimer_stopping_criteria
        )
        
        # Store initial state
        self.R = self.initial_position.copy()  # Current dimer center
        self.orient = None  # Will be set during initialization
        
        # For tracking converged positions
        self.R_latest_conv = None
        self.orient_latest_conv = None

    def _force_func_accurate(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate accurate forces using VASP.
        
        Args:
            position: Full system positions (can be 1D or 2D)
            
        Returns:
            Forces on all atoms (same shape as input)
        """
        # Handle both 1D and 2D input
        if position.ndim == 1:
            # Single position
            forces = self.local_pes.first_derivative(position, is_thermal=False)
            self.obs_total += 1  # Make sure this line exists
            
            if self.verbose and self.obs_total % 10 == 0:
                print(f"  [VASP evaluation #{self.obs_total}]")
            
            return forces
        else:
            # Multiple positions (2D array)
            forces_list = []
            for pos in position:
                forces = self.local_pes.first_derivative(pos, is_thermal=False)
                self.obs_total += 1  # Make sure this line exists
                
                if self.verbose and self.obs_total % 10 == 0:
                    print(f"  [VASP evaluation #{self.obs_total}]")
                
                forces_list.append(forces)
            
            return np.array(forces_list)
    
    def _force_func_gp(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate forces using GP2 model.
        
        Args:
            position: Full system positions (can be 1D or 2D)
            
        Returns:
            Forces on all atoms (same shape as input)
        """
        if self.gp2 is None or not self.gp2_trained:
            return np.zeros_like(position)
        
        # Handle both 1D and 2D input
        if position.ndim == 1:
            # Single position
            pos_3d = position.reshape(-1, 3)
            pos_moving = pos_3d[self.moving_indices].flatten()
            
            # Get GP predictions (for moving atoms only)
            _, pred_forces_moving, _, _ = self.gp2.predict(pos_moving.reshape(1, -1))
            
            # Reconstruct full forces (zeros for frozen atoms)
            full_forces = np.zeros_like(position)
            for i, idx in enumerate(self.moving_indices):
                full_forces[3*idx:3*(idx+1)] = pred_forces_moving[0, 3*i:3*(i+1)]
            
            return full_forces
        else:
            # Multiple positions (2D array)
            forces_list = []
            for pos in position:
                pos_3d = pos.reshape(-1, 3)
                pos_moving = pos_3d[self.moving_indices].flatten()
                
                # Get GP predictions (for moving atoms only)
                _, pred_forces_moving, _, _ = self.gp2.predict(pos_moving.reshape(1, -1))
                
                # Reconstruct full forces (zeros for frozen atoms)
                full_forces = np.zeros_like(pos)
                for i, idx in enumerate(self.moving_indices):
                    full_forces[3*idx:3*(idx+1)] = pred_forces_moving[0, 3*i:3*(i+1)]
                
                forces_list.append(full_forces)
            
            return np.array(forces_list)
    
    def _evaluate_position(self, position: npt.NDArray[np.float64], is_thermal: bool = False) -> Tuple[float, npt.NDArray[np.float64]]:
        """Evaluate energy and forces at a position using VASP.
        
        Returns:
            (energy, forces)
        """
        # Get energy
        energy = self.local_pes.scaler_y_value(position, is_thermal=is_thermal)
        
        # Set reference on first calculation
        if not self.reference_set:
            self.energy_reference = energy
            self.reference_set = True
            if self.verbose:
                print(f"Energy reference set to: {self.energy_reference:.4f} eV")
        
        # Apply reference
        energy_ref = energy - self.energy_reference
        
        # Get forces
        forces = self.local_pes.first_derivative(position, is_thermal=is_thermal)
        
        # INCREMENT THE COUNTER HERE
        self.obs_total += 1
        
        return energy_ref, forces
    
    def _add_observation(self, position: npt.NDArray[np.float64], energy: float, forces: npt.NDArray[np.float64]):
        """Add new observation to the dataset."""
        self.R_all = np.vstack((self.R_all, position.reshape(1, -1)))
        self.E_all = np.vstack((self.E_all, [[energy]]))
        self.G_all = np.vstack((self.G_all, forces.reshape(1, -1)))
    
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
            
            # Extract forces for moving atoms
            force_3d = self.G_all[i].reshape(-1, 3)
            force_moving = force_3d[self.moving_indices].flatten()
            forces_moving.append(force_moving)
        
        positions_moving = np.array(positions_moving)
        forces_moving = np.array(forces_moving)
        energies = self.E_all.flatten()
        
        # Create or update GP2
        if self.gp2 is None:
            # Initialize GP2 (following walker.py approach)
            training_data = [positions_moving, energies, forces_moving]
            self.gp2 = GP2(
                training_data=training_data,
                atomic_info=self.atomic_info,
                use_gpu=self.use_gpu
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
        
        self.gp2.train(
            training_data=self.gp2.training_data,
            thermal_noise=None,  # No thermal noise for GP2
            model_name="GP2",
            path=get_output_path('data_gp2'),
        )
        self.gp2_trained = True
        
        # Store hyperparameters (following reference)
        if hasattr(self.gp2, 'model') and self.gp2.model is not None:
            # Extract and store hyperparameters
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
    
    def _perform_initial_rotations(self):
        """Perform initial rotations following atomic GP-dimer algorithm."""
        if self.num_init_rotations <= 0:
            return
        
        if self.verbose:
            print("\n" + "="*40)
            print("INITIAL ROTATIONS")
            print("="*40)
        
        # Evaluate initial position if not done
        if len(self.R_all) == 0:
            E_R, G_R = self._evaluate_position(self.R)
            self._add_observation(self.R, E_R, G_R)
            self.E_R_acc.append(E_R)
            self.maxF_R_acc.append(np.max(np.abs(G_R)))

            # Max force for moving atoms only -- NEW
            G_R_3d = G_R.reshape(-1, 3)
            G_R_moving = G_R_3d[self.moving_indices]
            maxF_R_moving = np.max(np.abs(G_R_moving))
            self.maxF_R_acc.append(maxF_R_moving)
            # End of NEW
            
            if self.verbose:
                print(f"Initial values: E_R = {E_R:.6f}, maxF_R = {self.maxF_R_acc[-1]:.6f}")
        
        # Initialize orientation if not set
        if self.orient is None or self.dimer.orient is None:
            # Use force-based initialization
            G_R = self.G_all[-1]
            forces_3d = G_R.reshape(-1, 3)
            force_mags = np.linalg.norm(forces_3d, axis=1)
            max_force_atom = np.argmax(force_mags)
            
            initial_orient = np.zeros(len(G_R))
            initial_orient[3*max_force_atom:3*max_force_atom+3] = forces_3d[max_force_atom]
            initial_orient = initial_orient / np.linalg.norm(initial_orient)
            
            self.dimer.set_initial_direction(initial_orient)
            # Store as 2D array for dimer compatibility
            self.orient = initial_orient.reshape(1, -1)
            self.dimer.orient = self.orient.copy()
            
            if self.verbose:
                print(f"Set initial orientation along force on atom {max_force_atom}")
        
        # Evaluate image 1
        R1 = self.R + self.dimer_sep * self.orient.ravel()
        E1, G1 = self._evaluate_position(R1)
        self._add_observation(R1, E1, G1)
        
        if self.initrot_nogp:
            # Perform rotations without GP
            self._perform_rotations_without_gp()
        else:
            # Perform rotations with GP
            self._perform_rotations_with_gp()
        
        self.obs_initrot = self.obs_total
    
    def _perform_rotations_without_gp(self):
        """Perform initial rotations without GP (directly on accurate surface)."""
        for iter_rot in range(self.num_init_rotations):
            # Check convergence
            G_R = self.G_all[-2]  # Middle point gradient
            G1 = self.G_all[-1]   # Image 1 gradient
            
            # Calculate rotation angle
            F_rot = self._calculate_rotation_force(G_R, G1, self.orient.ravel())
            F_0 = np.linalg.norm(F_rot)
            C_0 = np.dot(-G_R + G1, self.orient.ravel()) / self.dimer_sep
            dtheta = 0.5 * np.arctan(0.5 * F_0 / np.abs(C_0)) if C_0 != 0 else 0
            
            if dtheta < self.T_anglerot_init:
                if self.verbose:
                    print(f"Initial rotations converged after {iter_rot} iterations")
                break
            
            # Rotate dimer
            self.dimer.x = self.R.copy()
            self.dimer.orient = self.orient.copy()
            step = self.dimer.run()  # This will rotate the dimer
            self.orient = self.dimer.orient.copy()
            
            # Evaluate new image 1
            R1 = self.R + self.dimer_sep * self.orient.ravel()
            E1, G1 = self._evaluate_position(R1)
            self._add_observation(R1, E1, G1)
    
    def _perform_rotations_with_gp(self):
        """Perform initial rotations with GP acceleration."""
        for bigiter_initrot in range(self.num_init_rotations + 1):
            # Check convergence on accurate surface
            if len(self.G_all) >= 2:
                G_R = self.G_all[-2]
                G1 = self.G_all[-1]
                
                F_rot = self._calculate_rotation_force(G_R, G1, self.orient.ravel())
                F_0 = np.linalg.norm(F_rot)
                C_0 = np.dot(-G_R + G1, self.orient.ravel()) / self.dimer_sep
                dtheta = 0.5 * np.arctan(0.5 * F_0 / np.abs(C_0)) if C_0 != 0 else 0
                
                if dtheta < self.T_anglerot_init:
                    if self.verbose:
                        print(f"Initial rotations converged after {bigiter_initrot} outer iterations")
                    break
            
            if bigiter_initrot == self.num_init_rotations:
                if self.verbose:
                    print(f"WARNING: Max initial rotation iterations reached")
                break
            
            # Train GP with current data
            self._train_gp2(reinit_hyperparams=True)
            
            # Store hyperparameters for initial rotations
            if self.gp2_trained:
                self.param_gp_initrot.append(self._extract_hyperparameters())
            
            # Switch dimer to use GP forces
            self.dimer.force_func = self._force_func_gp
            
            # Perform rotations on GP surface
            T_anglerot_init_gp = min(0.01, self.T_anglerot_init / 10.0)
            orient_old = self.orient.copy()
            
            for inner_iter in range(self.max_inner_iterations):
                # Get GP forces
                R01 = np.vstack((self.R, self.R + self.dimer_sep * self.orient.ravel()))
                G01_gp = np.vstack((self._force_func_gp(R01[0]), self._force_func_gp(R01[1])))
                
                # Check convergence on GP surface
                F_rot_gp = self._calculate_rotation_force(G01_gp[0], G01_gp[1], self.orient.ravel())
                F_0_gp = np.linalg.norm(F_rot_gp)
                C_0_gp = np.dot(-G01_gp[0] + G01_gp[1], self.orient.ravel()) / self.dimer_sep
                dtheta_gp = 0.5 * np.arctan(0.5 * F_0_gp / np.abs(C_0_gp)) if C_0_gp != 0 else 0
                
                if dtheta_gp < T_anglerot_init_gp:
                    break
                
                # Rotate on GP surface
                self.dimer.x = self.R.copy()
                self.dimer.orient = self.orient.copy()
                self.dimer.run()
                self.orient = self.dimer.orient.copy()
            
            # Evaluate new orientation on accurate surface
            R1 = self.R + self.dimer_sep * self.orient.ravel()
            E1, G1 = self._evaluate_position(R1)
            self._add_observation(R1, E1, G1)
            
            # Switch back to accurate forces
            self.dimer.force_func = self._force_func_accurate
    
    def _calculate_rotation_force(self, G_R: npt.NDArray, G1: npt.NDArray, orient: npt.NDArray) -> npt.NDArray:
        """Calculate rotation force for dimer."""
        # Ensure orient is 1D for calculations
        if orient.ndim == 2:
            orient = orient.ravel()
        # Force on image 1 perpendicular to dimer orientation
        F1_perp = G1 - np.dot(G1, orient) * orient
        # Force on center perpendicular to dimer orientation
        F_R_perp = G_R - np.dot(G_R, orient) * orient
        # Rotation force
        return F1_perp - F_R_perp
    
    def _check_interatomic_distances(self, R_new: npt.NDArray[np.float64]) -> bool:
        """Check if inter-atomic distances changed too much.
        
        Returns:
            True if position should be rejected
        """
        if self.atomic_structure is not None:
            return self.atomic_structure.check_interatomic_distances(
                R_new, self.R_all, self.ratio_at_limit
            )
        else:
            # Fallback if atomic_structure is not available
            return False
    
    def _check_displacement(self, R_new: npt.NDArray[np.float64]) -> bool:
        """Check if displacement from nearest observed point is too large.
        
        Returns:
            True if position should be rejected
        """
        if len(self.R_all) == 0:
            return False
        
        distances = np.sqrt(np.sum((R_new - self.R_all)**2, axis=1))
        min_dist = np.min(distances)
        
        return min_dist > self.disp_max
    
    def _relaxation_phase(self):
        """Perform one relaxation phase on GP surface."""
        if self.verbose:
            print(f"\n" + "="*40)
            print(f"RELAXATION PHASE {self.bigiter}")
            print("="*40)
            print(f"DEBUG: Starting relaxation phase, table history has {len(self.table_history)} entries")
        
        # Train GP with all data
        self._train_gp2(reinit_hyperparams=(self.bigiter <= self.num_bigiter_initparam))
        
        # Calculate convergence threshold for GP surface
        if self.divisor_T_dimer_gp > 0 and len(self.maxF_R_acc) > 0:
            T_dimer_gp = max(
                min(self.maxF_R_acc) / self.divisor_T_dimer_gp,
                self.dimer_stopping_criteria / 10.0
            )
        else:
            T_dimer_gp = self.dimer_stopping_criteria / 10.0
        
        # Set initial position for relaxation
        if self.bigiter > self.num_bigiter_initloc and self.R_latest_conv is not None:
            self.R = self.R_latest_conv.copy()
            self.orient = self.orient_latest_conv.copy()
            if self.verbose:
                print("Started from latest converged dimer")
        else:
            self.R = self.initial_position.copy()
            if self.orient is None:
                # Initialize orientation from forces
                if len(self.G_all) > 0:
                    G_R = self.G_all[0]
                    forces_3d = G_R.reshape(-1, 3)
                    force_mags = np.linalg.norm(forces_3d, axis=1)
                    max_force_atom = np.argmax(force_mags)
                    
                    orient_1d = np.zeros(len(G_R))
                    orient_1d[3*max_force_atom:3*max_force_atom+3] = forces_3d[max_force_atom]
                    orient_1d = orient_1d / np.linalg.norm(orient_1d)
                    # Store as 2D array
                    self.orient = orient_1d.reshape(1, -1)
            if self.verbose:
                print("Started from initial location")
        
        # Reset dimer with GP forces
        self.dimer.x = self.R.copy()
        self.dimer.orient = self.orient.copy()
        self.dimer.force_func = self._force_func_gp
        self.dimer.dimer_stopping_criteria = T_dimer_gp
        
        # Track convergence
        converged = False
        stopped_early = False
        stop_reason = ""
        
        # Inner iteration loop
        for inner_iter in range(self.max_inner_iterations):
            # Get GP predictions at current position
            E_gp, G_gp = self._gp_evaluate(self.R)
            maxF_gp = np.max(np.abs(G_gp))
            
            # Store GP trajectory
            self.E_R_gp.append(E_gp)
            self.maxF_R_gp.append(maxF_gp)
            
            # Check convergence on GP surface
            if maxF_gp < T_dimer_gp:
                self.R_latest_conv = self.R.copy()
                self.orient_latest_conv = self.orient.copy()
                converged = True
                if self.verbose:
                    print(f"Converged on GP surface after {inner_iter} iterations")
                break
            
            # Perform dimer step (rotation + translation)
            self.dimer.x = self.R.copy()
            self.dimer.orient = self.orient.copy()
            
            # The dimer expects its force_func to handle both single positions and pairs
            # So we need to make sure our force function can handle that
            step_vector = self.dimer.run()
            
            # Get new position
            R_new = self.dimer.x + step_vector
            
            # Update active atoms if needed
            n_activated = self.atomic_structure.update_activated_atoms_wrapper(R_new, self.verbose)
            if n_activated > 0:
                # Update atomic info
                self.atomic_info.update(self.atomic_structure.get_structure_info())
                # Need to retrain GP with new atomic structure
                self._train_gp2(reinit_hyperparams=False)
            
            # Check stopping criteria
            
            # 1. Inter-atomic distance check
            if inner_iter > 0 and self._check_interatomic_distances(R_new):
                stopped_early = True
                stop_reason = "inter-atomic distances"
                self.num_es1 += 1
                break
            
            # 2. Displacement check
            if inner_iter > 0 and self._check_displacement(R_new):
                stopped_early = True
                stop_reason = "displacement"
                self.num_es2 += 1
                break
            
            # Accept step
            self.R = R_new.copy()
            self.orient = self.dimer.orient.copy()
        
        # Check if max iterations reached
        if inner_iter == self.max_inner_iterations - 1:
            stopped_early = True
            stop_reason = "max iterations"
            self.num_esmax += 1
        
        # Store the number of inner iterations completed
        inner_iters_completed = inner_iter + 1 if 'inner_iter' in locals() else 0
        
        if self.verbose:
            if converged:
                print(f"Relaxation converged after {inner_iters_completed} inner iterations")
            elif stopped_early:
                print(f"Relaxation stopped early: {stop_reason}")
        
        # Store inner iteration count
        self.obs_at.append(len(self.E_R_gp) - 1)
        
        # Evaluate final position on accurate surface
        E_R, G_R = self._evaluate_position(self.R)
        self._add_observation(self.R, E_R, G_R)
        self.E_R_acc.append(E_R)
        self.maxF_R_acc.append(np.max(np.abs(G_R)))

        # Calculate max force for MOVING ATOMS ONLY to match GP -- NEW
        G_R_3d = G_R.reshape(-1, 3)
        G_R_moving = G_R_3d[self.moving_indices]
        maxF_R_moving = np.max(np.abs(G_R_moving))
        self.maxF_R_acc.append(maxF_R_moving)
        # End of NEW

        if self.verbose:
            print(f"Accurate values: E_R = {E_R:.6f}, maxF_R (moving atoms) = {maxF_R_moving:.6f}")
            # Also print max force for all atoms for comparison
            maxF_R_all = np.max(np.abs(G_R))
            print(f"  Max force (all atoms): {maxF_R_all:.6f}")
        
        if self.verbose:
            print(f"Accurate values: E_R = {E_R:.6f}, maxF_R = {self.maxF_R_acc[-1]:.6f}")

        # Always evaluate image 1 to get curvature (not just when eval_image1 is True)
        R1 = self.R + self.dimer_sep * self.orient.ravel()
        E1, G1 = self._evaluate_position(R1)
        if self.eval_image1:
            # Only add to observations if eval_image1 is True
            self._add_observation(R1, E1, G1)
        
        # Calculate curvatures
        curv_gp = np.nan
        curv_dimer = np.nan
        
        if self.orient is not None:
            # Get GP curvature at final position
            if self.gp2_trained:
                G_R_gp = self._force_func_gp(self.R)
                G1_gp = self._force_func_gp(R1)
                curv_gp = np.dot(-G_R_gp + G1_gp, self.orient.ravel()) / self.dimer_sep
            
            # Calculate actual curvature using the forces we just computed
            curv_dimer = np.dot(-G_R + G1, self.orient.ravel()) / self.dimer_sep
        
        # Get final GP values
        final_E_gp = self.E_R_gp[-1] if self.E_R_gp else np.nan
        final_maxF_gp = self.maxF_R_gp[-1] if self.maxF_R_gp else np.nan
        
        # Add entry to table history (without inner_iters)
        table_entry = {
            'step': self.bigiter,
            'E_GP': final_E_gp,
            'E_Actual': E_R,
            'F_GP': final_maxF_gp,
            'F_Actual': self.maxF_R_acc[-1],
            'Curvature_GP': curv_gp,
            'Curvature_dimer': curv_dimer
        }
        self.table_history.append(table_entry)
        
        # Optionally evaluate image 1
        if self.eval_image1:
            R1 = self.R + self.dimer_sep * self.orient
            E1, G1 = self._evaluate_position(R1)
            self._add_observation(R1, E1, G1)
    
    def _gp_evaluate(self, position: npt.NDArray[np.float64]) -> Tuple[float, npt.NDArray[np.float64]]:
        """Evaluate energy and forces using GP model.
        
        Returns:
            (energy, forces)
        """
        if not self.gp2_trained:
            return 0.0, np.zeros_like(position)
        
        # Extract moving positions
        pos_3d = position.reshape(-1, 3)
        pos_moving = pos_3d[self.moving_indices].flatten()
        
        # Get GP predictions
        pred_e, pred_f, _, _ = self.gp2.predict(pos_moving.reshape(1, -1))
        
        # Reconstruct full forces
        full_forces = np.zeros_like(position)
        for i, idx in enumerate(self.moving_indices):
            full_forces[3*idx:3*(idx+1)] = pred_f[0, 3*i:3*(i+1)]
        
        return float(pred_e[0]), full_forces
    
    def run(self) -> Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
        """Run GP2-accelerated dimer search."""
        if self.verbose:
            print("\nStarting GP2-accelerated dimer search...")
        
        # Evaluate initial position
        E_R, G_R = self._evaluate_position(self.initial_position)
        self._add_observation(self.initial_position, E_R, G_R)
        self.E_R_acc.append(E_R)
        self.maxF_R_acc.append(np.max(np.abs(G_R)))
        self.R = self.initial_position.copy()


        # Calculate max force for moving atoms only -- NEW
        G_R_3d = G_R.reshape(-1, 3)
        G_R_moving = G_R_3d[self.moving_indices]
        maxF_R_moving = np.max(np.abs(G_R_moving))
        self.maxF_R_acc.append(maxF_R_moving)
        # End of NEW

        
        if self.verbose:
            print(f"\nInitial values: E_R = {E_R:.6f}, maxF_R = {self.maxF_R_acc[-1]:.6f}")
        
        # Check initial convergence
        if self.maxF_R_acc[-1] < self.dimer_stopping_criteria:
            self.converged = True
            if self.verbose:
                print("Already converged at initial position!")
            return self.R, E_R, G_R
        
        # Perform initial rotations
        self._perform_initial_rotations()

        # Record initial state in table
        initial_entry = {
            'step': 0,
            'E_GP': np.nan,  # No GP yet
            'E_Actual': self.E_R_acc[-1],
            'F_GP': np.nan,
            'F_Actual': self.maxF_R_acc[-1],
            'Curvature_GP': np.nan,
            'Curvature_dimer': np.nan
        }
        self.table_history.append(initial_entry)
        
        # Initial translation if requested
        if self.inittrans_nogp:
            self._perform_initial_translation()
        
        # Main outer iteration loop
        for self.bigiter in range(1, self.max_dimer_steps + 1):
            # Save checkpoint if needed
            if self.bigiter % self.checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Perform relaxation phase on GP surface
            self._relaxation_phase()
            
            # Print the progress table after each dimer step
            if self.table_history:
                print(f"\nDEBUG: About to print table with {len(self.table_history)} entries")
                self._print_progress_table()
            else:
                print("\nDEBUG: No table history to print!")
            
            # Check convergence
            if self.maxF_R_acc[-1] < self.dimer_stopping_criteria:
                self.converged = True
                if self.verbose:
                    print(f"\n" + "="*60)
                    print(f"CONVERGED after {self.bigiter} outer iterations")
                    print(f"Total observations: {self.obs_total}")
                    print("="*60)
                break
            
            # Update minimum force for adaptive threshold
            self.min_force_achieved = min(self.min_force_achieved, self.maxF_R_acc[-1])
        
        if not self.converged and self.verbose:
            print(f"\n" + "="*60)
            print(f"MAXIMUM ITERATIONS REACHED")
            print(f"Total observations: {self.obs_total}")
            print("="*60)
        
        # Save final checkpoint
        self.save_checkpoint(final=True)
        
        # Print summary
        if self.verbose:
            self._print_summary()
        
        # Return final results
        return self.R, self.E_R_acc[-1], self.G_all[-1]
    
    def _perform_initial_translation(self):
        """Perform initial translation step without GP."""
        if self.verbose:
            print("\nPerforming initial translation step...")
        
        # Need gradient at image 1
        if len(self.R_all) < 2:
            R1 = self.R + self.dimer_sep * self.orient.ravel()
            E1, G1 = self._evaluate_position(R1)
            self._add_observation(R1, E1, G1)
        
        # Get gradients
        G_R = self.G_all[-2]
        G1 = self.G_all[-1]
        
        # Calculate curvature
        Curv = np.dot(-G_R + G1, self.orient.ravel()) / self.dimer_sep
        
        # Translation force
        F_trans = -G_R + 2 * np.dot(G_R, self.orient.ravel()) * self.orient.ravel()
        
        # Take test step
        if Curv != 0:
            self.R = self.R + 0.5 * F_trans / np.abs(Curv)
        else:
            self.R = self.R + self.step_size * F_trans / np.linalg.norm(F_trans)
        
        # Evaluate new position
        E_R, G_R = self._evaluate_position(self.R)
        self._add_observation(self.R, E_R, G_R)
        self.E_R_acc.append(E_R)
        self.maxF_R_acc.append(np.max(np.abs(G_R)))
        
        if self.verbose:
            print(f"After translation: E_R = {E_R:.6f}, maxF_R = {self.maxF_R_acc[-1]:.6f}")
    
    def _print_progress_table(self):
        """Print progress table with all history."""
        print("\n" + "="*126)
        print("DIMER PROGRESS TABLE")
        print("="*126)
        print(f"{'Step':>6} {'E_GP':>14} {'E_Actual':>14} {'F_GP':>14} {'F_Actual':>14} {'Curvature_GP':>14} {'Curvature_dimer':>16}")
        print(f"{' ':>6} {'(eV)':>14} {'(eV)':>14} {'(eV/Å)':>14} {'(eV/Å)':>14} {'(eV/Å²)':>14} {'(eV/Å²)':>16}")
        print("-"*126)
        
        # Show all entries
        for row in self.table_history:
            e_gp_str = f"{row['E_GP']:14.6f}" if not np.isnan(row['E_GP']) else f"{'---':>14}"
            e_actual_str = f"{row['E_Actual']:14.6f}" if not np.isnan(row['E_Actual']) else f"{'---':>14}"
            maxf_gp_str = f"{row['F_GP']:14.6f}" if not np.isnan(row['F_GP']) else f"{'---':>14}"
            maxf_actual_str = f"{row['F_Actual']:14.6f}" if not np.isnan(row['F_Actual']) else f"{'---':>14}"
            curv_gp_str = f"{row['Curvature_GP']:14.6f}" if not np.isnan(row['Curvature_GP']) else f"{'---':>14}"
            curv_dimer_str = f"{row['Curvature_dimer']:16.6f}" if not np.isnan(row['Curvature_dimer']) else f"{'---':>16}"
            
            print(f"{row['step']:6d} {e_gp_str} {e_actual_str} {maxf_gp_str} {maxf_actual_str} {curv_gp_str} {curv_dimer_str}")
        print("="*126)
    
    def set_initial_orientation(self, orient: npt.NDArray[np.float64]):
        """Set initial dimer orientation.
        
        Args:
            orient: Initial orientation vector (will be normalized)
        """
        orient_normalized = orient / np.linalg.norm(orient)
        # Store as 2D array for dimer compatibility
        self.orient = orient_normalized.reshape(1, -1)
        if hasattr(self, 'dimer') and self.dimer is not None:
            self.dimer.set_initial_direction(orient_normalized)
            self.dimer.orient = self.orient.copy()
        if self.verbose:
            print(f"Set initial orientation manually")
        """Print summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Converged: {self.converged}")
        print(f"Outer iterations: {self.bigiter}")
        print(f"Total observations: {self.obs_total}")
        print(f"Initial rotation observations: {self.obs_initrot}")
        print(f"Inner iterations total: {len(self.E_R_gp)}")
        
        print(f"\nStopping statistics:")
        print(f"  Max iterations reached: {self.num_esmax}")
        print(f"  Inter-atomic distance limit: {self.num_es1}")
        print(f"  Displacement limit: {self.num_es2}")
        
        print(f"\nEnergy evolution:")
        print(f"  Initial: {self.E_R_acc[0]:.6f} eV")
        print(f"  Final: {self.E_R_acc[-1]:.6f} eV")
        print(f"  Change: {self.E_R_acc[-1] - self.E_R_acc[0]:.6f} eV")
        
        print(f"\nForce evolution:")
        print(f"  Initial max |F|: {self.maxF_R_acc[0]:.6f} eV/Å")
        print(f"  Final max |F|: {self.maxF_R_acc[-1]:.6f} eV/Å")
        
        # Displacement
        displacement = np.linalg.norm(self.R - self.initial_position)
        print(f"\nTotal displacement: {displacement:.4f} Å")
        print("="*70)
    
    def save_checkpoint(self, final=False):
        """Save checkpoint data."""
        checkpoint_data = {
            'walker_type': 'WalkerGP2Dimer',
            'iteration': self.bigiter,
            'converged': self.converged,
            'energy_reference': self.energy_reference,
            'reference_set': self.reference_set,
            'R': self.R,
            'orient': self.orient,
            'R_all': self.R_all,
            'E_all': self.E_all,
            'G_all': self.G_all,
            'E_R_acc': self.E_R_acc,
            'maxF_R_acc': self.maxF_R_acc,
            'E_R_gp': self.E_R_gp,
            'maxF_R_gp': self.maxF_R_gp,
            'obs_total': self.obs_total,
            'obs_initrot': self.obs_initrot,
            'obs_at': self.obs_at,
            'param_gp': self.param_gp,
            'param_gp_initrot': self.param_gp_initrot,
            'num_esmax': self.num_esmax,
            'num_es1': self.num_es1,
            'num_es2': self.num_es2,
            'atomic_info': self.atomic_info,
            'min_force_achieved': self.min_force_achieved,
            'R_latest_conv': self.R_latest_conv,
            'orient_latest_conv': self.orient_latest_conv,
            'table_history': self.table_history  # Save table history
        }
        
        checkpoints_dir = get_output_path('checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        if final:
            filename = os.path.join(checkpoints_dir, 'gp2_dimer_final.pkl')
        else:
            filename = os.path.join(checkpoints_dir, f'gp2_dimer_checkpoint_{self.bigiter}.pkl')
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save as latest
        latest_filename = os.path.join(checkpoints_dir, 'gp2_dimer_latest.pkl')
        with open(latest_filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        if self.verbose and (self.bigiter % 10 == 0 or final):
            print(f"  [Checkpoint saved at iteration {self.bigiter}]")

    def _print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Converged: {self.converged}")
        print(f"Outer iterations: {self.bigiter}")
        print(f"Total observations: {self.obs_total}")
        print(f"Initial rotation observations: {self.obs_initrot}")
        print(f"Inner iterations total: {len(self.E_R_gp)}")
        
        print(f"\nStopping statistics:")
        print(f"  Max iterations reached: {self.num_esmax}")
        print(f"  Inter-atomic distance limit: {self.num_es1}")
        print(f"  Displacement limit: {self.num_es2}")
        
        print(f"\nEnergy evolution:")
        print(f"  Initial: {self.E_R_acc[0]:.6f} eV")
        print(f"  Final: {self.E_R_acc[-1]:.6f} eV")
        print(f"  Change: {self.E_R_acc[-1] - self.E_R_acc[0]:.6f} eV")
        
        print(f"\nForce evolution:")
        print(f"  Initial max |F|: {self.maxF_R_acc[0]:.6f} eV/Å")
        print(f"  Final max |F|: {self.maxF_R_acc[-1]:.6f} eV/Å")
        
        # Displacement
        displacement = np.linalg.norm(self.R - self.initial_position)
        print(f"\nTotal displacement: {displacement:.4f} Å")
        
        # Efficiency metrics
        if self.obs_total > self.obs_initrot:
            main_obs = self.obs_total - self.obs_initrot
            inner_iters_main = len(self.E_R_gp)
            if main_obs > 0:
                speedup = inner_iters_main / main_obs
                print(f"\nEfficiency:")
                print(f"  Main search observations: {main_obs}")
                print(f"  GP evaluations: {inner_iters_main}")
                print(f"  Speedup factor: {speedup:.1f}x")
        
        # Final curvature if available
        if self.table_history and len(self.table_history) > 0:
            last_entry = self.table_history[-1]
            if 'Curvature_dimer' in last_entry and not np.isnan(last_entry['Curvature_dimer']):
                print(f"\nFinal curvature: {last_entry['Curvature_dimer']:.6f} eV/Å²")
        
        print("="*70)
    
    def load_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            checkpoint_file = get_output_path('checkpoints', 'gp2_dimer_latest.pkl')
        """Load checkpoint and restore state."""
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state
        self.bigiter = checkpoint['iteration']
        self.converged = checkpoint['converged']
        self.energy_reference = checkpoint['energy_reference']
        self.reference_set = checkpoint['reference_set']
        self.R = checkpoint['R']
        self.orient = checkpoint['orient']
        self.R_all = checkpoint['R_all']
        self.E_all = checkpoint['E_all']
        self.G_all = checkpoint['G_all']
        self.E_R_acc = checkpoint['E_R_acc']
        self.maxF_R_acc = checkpoint['maxF_R_acc']
        self.E_R_gp = checkpoint['E_R_gp']
        self.maxF_R_gp = checkpoint['maxF_R_gp']
        self.obs_total = checkpoint['obs_total']
        self.obs_initrot = checkpoint['obs_initrot']
        self.obs_at = checkpoint['obs_at']
        self.param_gp = checkpoint['param_gp']
        self.param_gp_initrot = checkpoint['param_gp_initrot']
        self.num_esmax = checkpoint['num_esmax']
        self.num_es1 = checkpoint['num_es1']
        self.num_es2 = checkpoint['num_es2']
        self.min_force_achieved = checkpoint.get('min_force_achieved', float('inf'))
        self.R_latest_conv = checkpoint.get('R_latest_conv')
        self.orient_latest_conv = checkpoint.get('orient_latest_conv')
        
        # Restore table history if available
        self.table_history = checkpoint.get('table_history', [])
        
        # Retrain GP with loaded data
        if len(self.R_all) > 1:
            self._train_gp2()
        
        if self.verbose:
            print(f"\nRestored checkpoint from iteration {self.bigiter}")
            print(f"Total observations so far: {self.obs_total}")
            print(f"Table history entries: {len(self.table_history)}")
        
        return checkpoint