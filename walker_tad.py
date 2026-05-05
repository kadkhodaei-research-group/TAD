"""walker_dual_gp_improved.py - Enhanced dual GP walker with improvements from toy model experience."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, Tuple, List, Union
import os
import pickle
import time
from scipy.stats import norm
import warnings
import json

from dimer import Dimer
from gp2_model import GP2
from gp1_model import GP1 
from trgp1_model import ThermodynamicResidenceGP1
from atomic_structure import AtomicStructure, calculate_atomic_distance_measure
from gp_data_saver import get_gp_logger
from phonon_unfolding import read_phonopy
from pymatgen.io.vasp import Poscar
from output_manager import get_output_path


logger = logging.getLogger(__name__)


class TrustRegionManager:
    """Manages trust region for adaptive step sizes."""
    
    def __init__(self, initial_radius: float = 0.5, min_radius: float = 0.01, max_radius: float = 2.0,
                 dramatic_threshold: float = 0.1, expand_threshold: float = 0.5, 
                 dramatic_factor: float = 0.5, moderate_factor: float = 0.8, expand_factor: float = 1.2):
        self.radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.dramatic_threshold = dramatic_threshold
        self.expand_threshold = expand_threshold
        self.dramatic_factor = dramatic_factor
        self.moderate_factor = moderate_factor
        self.expand_factor = expand_factor
        self.history = []
        
        # Store previous force vector for new reduction ratio calculation
        self.previous_force_vector = None
        
    def adjust_radius(self, force_vector_before: npt.NDArray[np.float64], force_vector_after: npt.NDArray[np.float64], step_taken: float):
        """Adjust trust region based on force changes using new algorithm."""
        # Calculate force magnitudes
        force_before = np.linalg.norm(force_vector_before)
        force_after = np.linalg.norm(force_vector_after)
        
        # Calculate new reduction ratio: norm(force_vector_before - force_vector_after) / force_before
        if force_before > 0:
            force_vector_diff = force_vector_before - force_vector_after
            reduction_ratio = np.linalg.norm(force_vector_diff) / force_before
        else:
            reduction_ratio = 0
            
        # NEW_v25: REVERSED 3-prong trust region adjustment algorithm
        if reduction_ratio < self.dramatic_threshold:
            # Low reduction ratio - EXPAND (opposite of v23)
            self.radius *= self.expand_factor  # Use expand factor instead of dramatic shrink
        elif reduction_ratio < self.expand_threshold:
            # Moderate reduction ratio - NO CHANGE (instead of moderate shrink)
            pass  # Keep radius unchanged
        else:
            # High reduction ratio - SHRINK (opposite of v23)
            self.radius *= self.dramatic_factor  # Use dramatic factor for shrinking
            
        # Apply bounds
        self.radius = np.clip(self.radius, self.min_radius, self.max_radius)
        
        # Store history
        self.history.append({
            'radius': self.radius,
            'force_before': force_before,
            'force_after': force_after,
            'step_taken': step_taken,
            'reduction_ratio': reduction_ratio
        })
        
    def limit_step(self, proposed_step: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Limit step size to trust region."""
        step_norm = np.linalg.norm(proposed_step)
        if step_norm > self.radius:
            return proposed_step * (self.radius / step_norm)
        return proposed_step


class OscillationDetector:
    """Detects oscillations in position trajectory."""
    
    def __init__(self, memory: int = 5, threshold: float = 0.02):
        self.memory = memory
        self.threshold = threshold
        self.position_history = []
        
    def check_oscillation(self, position: npt.NDArray[np.float64]) -> bool:
        """Check if the walker is oscillating."""
        self.position_history.append(position.copy())
        
        if len(self.position_history) < self.memory:
            return False
            
        # Keep only recent history
        self.position_history = self.position_history[-self.memory:]
        
        # Check if positions are cycling
        for i in range(len(self.position_history) - 2):
            dist = np.linalg.norm(self.position_history[i] - self.position_history[-1])
            if dist < self.threshold:
                return True
                
        return False
        

class ForceMinimumDetector:
    """Detects when passing through force minimum (potential saddle point)."""
    
    def __init__(self, window: int = 5, increase_threshold: float = 1.5):
        self.window = window
        self.increase_threshold = increase_threshold
        self.force_history = []
        self.curvature_history = []
        
    def update(self, force: float, curvature: Optional[float] = None):
        """Update force and curvature history."""
        self.force_history.append(force)
        self.curvature_history.append(curvature)
        
        # Keep limited history
        if len(self.force_history) > self.window * 2:
            self.force_history = self.force_history[-self.window*2:]
            self.curvature_history = self.curvature_history[-self.window*2:]
            
    def detect_minimum_passed(self) -> Tuple[bool, Optional[float]]:
        """Check if we've passed through a force minimum."""
        if len(self.force_history) < self.window:
            return False, None
            
        # Find minimum in history
        min_force = min(self.force_history[:-1])  # Exclude current
        min_idx = self.force_history[:-1].index(min_force)
        
        # Check if force has increased significantly since minimum
        current_force = self.force_history[-1]
        if current_force > min_force * self.increase_threshold:
            # Check if we had negative curvature at minimum
            if min_idx < len(self.curvature_history):
                min_curvature = self.curvature_history[min_idx]
                if min_curvature is not None and min_curvature < 0:
                    return True, min_force
                    
        return False, None



class WalkerDualGPImproved:
    """Enhanced walker with improvements from toy model experience."""

    def __init__(
        self,
        initial_position: npt.NDArray[np.float64],
        local_pes: Any,
        force_constants_file: str,
        POSCAR_file: str,
        temperature: float = 300.0,
        mass: Union[float, npt.NDArray[np.float64]] = 1.0,
        num_snapshots: int = 10,
        max_dimer_steps: int = 100,
        disp_max: float = 0.5,
        ratio_at_limit: float = 2.0/3.0,
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
        divisor_T_dimer_gp: float = 10.0,
        max_inner_iterations: int = 1000,
        gp1_noise_model: str = "fixed",
        gp1_student_t_df: float = 2.0,
        gp1_use_adaptive_df: bool = False,
        gp1_adaptive_df_start_iter: int = 15,
        gp1_adaptive_df_end_iter: int = 25,
        gp1_adaptive_df_target: float = 1.0,
        gp1_remove_outliers: bool = False,
        gp1_outlier_threshold: float = 5.0,
        initrot_nogp: bool = False,
        inittrans_nogp: bool = False,
        eval_image1: bool = False,
        num_bigiter_initloc: float = np.inf,
        num_bigiter_initparam: float = np.inf,
        use_gpu: bool = False,
        parallel_eam: bool = False,
        verbose: bool = False,
        checkpoint_interval: int = 1,
        model_type: str = "MultitaskGPModel_rbf_atomic",
        # Enhanced parameters
        enable_trust_region: bool = True,
        trust_region_initial: float = 0.5,
        trust_region_min: float = 0.01,
        trust_region_max: float = 2.0,
        trust_region_dramatic_threshold: float = 0.1,
        trust_region_expand_threshold: float = 0.5,
        trust_region_dramatic_factor: float = 0.5,
        trust_region_moderate_factor: float = 0.8,
        trust_region_expand_factor: float = 1.2,
        enable_oscillation_detection: bool = True,
        oscillation_memory: int = 5,
        oscillation_threshold: float = 0.02,
        enable_force_minimum_detection: bool = True,
        force_minimum_window: int = 5,
        force_minimum_increase_threshold: float = 1.5,
        relaxed_saddle_criteria: float = 0.1,
        enable_adaptive_convergence: bool = True,
        gp1_fallback_to_averaging: bool = True,
        gp1_max_retries: int = 3,
        validate_parameters: bool = True,
        **kwargs
    ) -> None:
        # Enhanced parameters
        self.enable_trust_region = enable_trust_region
        self.trust_region_initial = trust_region_initial
        self.trust_region_min = trust_region_min
        self.trust_region_max = trust_region_max
        self.trust_region_dramatic_threshold = trust_region_dramatic_threshold
        self.trust_region_expand_threshold = trust_region_expand_threshold
        self.trust_region_dramatic_factor = trust_region_dramatic_factor
        self.trust_region_moderate_factor = trust_region_moderate_factor
        self.trust_region_expand_factor = trust_region_expand_factor

        self.enable_oscillation_detection = enable_oscillation_detection
        self.oscillation_memory = oscillation_memory
        self.oscillation_threshold = oscillation_threshold

        self.enable_force_minimum_detection = enable_force_minimum_detection
        self.force_minimum_window = force_minimum_window
        self.force_minimum_increase_threshold = force_minimum_increase_threshold

        self.relaxed_saddle_criteria = relaxed_saddle_criteria
        self.enable_adaptive_convergence = enable_adaptive_convergence

        self.gp1_fallback_to_averaging = gp1_fallback_to_averaging
        self.gp1_max_retries = gp1_max_retries

        self.validate_parameters = validate_parameters
        self.relaxation_exit_events = []
        # === Parent class initialization (from WalkerDualGP) ===
        self.initial_position = initial_position.copy()
        self.local_pes = local_pes
        
        # Configure parallel EAM if enabled
        if parallel_eam and hasattr(local_pes, 'set_parallel_eam'):
            local_pes.set_parallel_eam(True)
            if verbose:
                print("GPU-accelerated EAM enabled for thermal snapshot calculations")
        
        self.max_dimer_steps = max_dimer_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.model_type = model_type
        self.num_snapshots = num_snapshots
        self.mass = mass
        self.temperature = temperature
        self.use_gpu = use_gpu
        self.parallel_eam = parallel_eam
        
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
        


        # Read force constants for thermal sampling
        self.eigenvalues_tr, self.eigenvectors_tr = self._read_force_constants(
            POSCAR_file, force_constants_file
        )
        # ADD: Initialize thermal noise calculation
        self.trgp1 = ThermodynamicResidenceGP1(
            current_location=initial_position,
            temperature=temperature,
            mass=self.mass,
            local_pes=self.local_pes,
            path=get_output_path("data_trgp1"),
            atomic_info=self.atomic_info,
        )
        self.thermal_noise = self.trgp1.calculate_thermal_noise(
            eigenvalues_tr=self.eigenvalues_tr,
            eigenvectors_tr=self.eigenvectors_tr,
            num_snapshots=num_snapshots
        )
        
        # Divide thermal noise by number of atoms for per-atom noise
        self.thermal_noise = (self.thermal_noise[0] / self.n_atoms, 
                              self.thermal_noise[1] / self.n_atoms)
        
        if self.verbose:
            print(f"Thermal noise (per atom): |F| σ = {self.thermal_noise[0]:.3e} eV/Å, ΔE σ = {self.thermal_noise[1]:.3e} eV")
        
        # Store noise model parameters
        self.gp1_noise_model = gp1_noise_model
        self.gp1_student_t_df = gp1_student_t_df
        # Store adaptive df parameters
        self.gp1_use_adaptive_df = gp1_use_adaptive_df
        self.gp1_adaptive_df_start_iter = gp1_adaptive_df_start_iter
        self.gp1_adaptive_df_end_iter = gp1_adaptive_df_end_iter
        self.gp1_adaptive_df_target = gp1_adaptive_df_target
        self.gp1_remove_outliers = gp1_remove_outliers
        self.gp1_outlier_threshold = gp1_outlier_threshold
        # Initialize GP1 with GPU support
        self.gp1 = GP1(
            current_location=initial_position,
            temperature=temperature,
            mass=self.mass,
            local_pes=self.local_pes,
            path=get_output_path("data_gp1"),
            atomic_info=self.atomic_info,
            noise_model=self.gp1_noise_model, 
            student_t_df=self.gp1_student_t_df, 
            use_adaptive_df=self.gp1_use_adaptive_df,
            adaptive_df_start_iter=self.gp1_adaptive_df_start_iter,
            adaptive_df_end_iter=self.gp1_adaptive_df_end_iter,
            adaptive_df_target=self.gp1_adaptive_df_target,
            remove_outliers=self.gp1_remove_outliers,
            outlier_threshold=self.gp1_outlier_threshold,
            use_gpu=self.use_gpu,
        )
        self.gp1.model_type = self.model_type
        
        # Set minimum noise for custom noise models
        if self.gp1_noise_model != "fixed" and self.thermal_noise is not None:
            self.gp1.minimum_noise = self.thermal_noise
            print(f"GP1: Using minimum noise constraint from TRGP1: F={self.thermal_noise[0]:.6f} eV/Å, E={self.thermal_noise[1]:.6f} eV")
        
        # Energy reference (set on first calculation)
        self.energy_reference = None
        self.reference_set = False
        
        # Initialize GP2 model
        self.gp2 = None
        self.gp2_trained = False
        
        # Statistical tracking
        self.raw_energy_std = None
        self.raw_force_std = None
        self.raw_energy_mad = None
        self.raw_force_mad = None
        self.raw_energy_std_mad_ratio = None
        self.raw_force_std_mad_ratio = None
        self.gp1_energy_std = None
        self.gp1_force_std = None
        self.gp1_energy_mad = None
        self.gp1_force_mad = None
        self.gp1_energy_std_mad_ratio = None
        self.gp1_force_std_mad_ratio = None
        self.gp2_energy_std = None
        self.gp2_force_std = None
        self.gp1_initial_noise = None
        self.gp1_final_noise = None
        self.gp2_initial_noise = None
        self.gp2_final_noise = None
        self.avg_temperatures = []  # Track temperatures throughout run

        # Log model information
        gp_logger = get_gp_logger()

        # Log GP1 info
        gp_logger.log_model_info(
            model_name='GP1',
            model_type=self.model_type,
            atomic_info=self.atomic_info,
            additional_info={
                'temperature': temperature,
                'num_snapshots': self.num_snapshots,
                'thermal_noise': {
                    'force': self.thermal_noise[0],
                    'energy': self.thermal_noise[1]
                }
            }
        )

        # Log walker info
        gp_logger.log_model_info(
            model_name='WalkerDualGP',
            model_type='DualGP',
            atomic_info=self.atomic_info,
            additional_info={
                'max_dimer_steps': self.max_dimer_steps,
                'dimer_stopping_criteria': self.dimer_stopping_criteria,
                'disp_max': self.disp_max,
                'ratio_at_limit': self.ratio_at_limit,
                'divisor_T_dimer_gp': self.divisor_T_dimer_gp
            }
        )
        
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
        self.e_thermal_all = np.empty((0, 1))  # Raw e_thermal from GP1 at each observation
        
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
        # Initialize new components
        if self.enable_trust_region:
            self.trust_region = TrustRegionManager(
                initial_radius=self.trust_region_initial,
                min_radius=self.trust_region_min,
                max_radius=self.trust_region_max,
                dramatic_threshold=self.trust_region_dramatic_threshold,
                expand_threshold=self.trust_region_expand_threshold,
                dramatic_factor=self.trust_region_dramatic_factor,
                moderate_factor=self.trust_region_moderate_factor,
                expand_factor=self.trust_region_expand_factor
            )
            
        if self.enable_oscillation_detection:
            self.oscillation_detector = OscillationDetector(
                memory=self.oscillation_memory,
                threshold=self.oscillation_threshold
            )
            
        if self.enable_force_minimum_detection:
            self.force_minimum_detector = ForceMinimumDetector(
                window=self.force_minimum_window,
                increase_threshold=self.force_minimum_increase_threshold
            )
            
        # Additional tracking
        self.gp1_training_failures = 0
        self.trust_region_history = []
        self.oscillation_events = []
        self.force_minimum_events = []
        self.convergence_type = None
        
        # NEW_v25: Initialize relaxation step logging
        self.relaxation_log = []
        
        # Validate parameters if requested
        if self.validate_parameters:
            self._validate_parameters()
            
    def _validate_parameters(self):
        """Validate parameters based on toy model insights."""
        warnings_list = []
        
        # Temperature vs snapshots validation
        if self.temperature < 50 and self.num_snapshots > 100:
            warnings_list.append(
                f"Low temperature ({self.temperature}K) with many snapshots ({self.num_snapshots}) "
                "may be inefficient. Consider reducing snapshots."
            )
            
        if self.temperature > 5000 and self.num_snapshots < 50:
            warnings_list.append(
                f"High temperature ({self.temperature}K) with few snapshots ({self.num_snapshots}) "
                "may have high variance. Consider increasing snapshots."
            )
            
        # Convergence criteria validation
        if self.dimer_stopping_criteria < 0.05 and self.temperature > 100:
            warnings_list.append(
                f"Convergence criterion ({self.dimer_stopping_criteria} eV/Å) may be too tight "
                f"for thermal system at {self.temperature}K. Consider using relaxed criteria."
            )
            
        # Mass validation
        avg_mass = np.mean(self.mass) if isinstance(self.mass, np.ndarray) else self.mass
        if avg_mass < 10:
            warnings_list.append(
                f"Low mass ({avg_mass:.1f} amu) may cause convergence issues. "
                "Toy model found mass scaling of 1000x beneficial."
            )
            
        # Print warnings
        if warnings_list:
            print("\n" + "="*60)
            print("PARAMETER VALIDATION WARNINGS:")
            for i, warning in enumerate(warnings_list, 1):
                print(f"{i}. {warning}")
            print("="*60 + "\n")
            
    def _evaluate_position_core(self, position: npt.NDArray[np.float64], is_thermal: bool = False) -> Tuple[float, npt.NDArray[np.float64]]:
        """Core evaluation using GP1 thermal sampling (from parent WalkerDualGP)."""
        # Update GP1 location
        self.gp1.current_location = position.copy()
        self.gp1.set_current_iteration(f"eval_{self.obs_total}")

        # Get thermalized data from GP1
        x_thermal, e_thermal, f_thermal = self.gp1.get_thermalized_current_location(
            num_snapshots=self.num_snapshots,
            thermal_noise=self.thermal_noise,
            eigenvalues_tr=self.eigenvalues_tr,
            eigenvectors_tr=self.eigenvectors_tr,
        )


        # Store raw e_thermal for GP2 training shift
        self._last_e_thermal = e_thermal

        # Just use the energy from GP1 directly (already referenced)
        energy_ref = e_thermal

        # f_thermal is forces for moving atoms only
        # Reconstruct full forces with zeros for frozen atoms
        full_forces = np.zeros_like(position)
        for i, idx in enumerate(self.moving_indices):
            full_forces[3*idx:3*(idx+1)] = f_thermal[3*i:3*(i+1)]

        # LOG GP1 EVALUATION
        gp_logger = get_gp_logger()

        pos_3d = position.reshape(-1, 3)
        pos_moving = pos_3d[self.moving_indices].flatten()

        gp1_pred_e, gp1_pred_f, gp1_var_e, gp1_var_f = self.gp1.predict(pos_moving.reshape(1, -1))

        gp_logger.log_gp_prediction(
            model_name='GP1',
            iteration=self.obs_total,
            position=pos_moving,
            prediction={
                'energy': float(gp1_pred_e[0]),
                'force_norm': float(np.linalg.norm(gp1_pred_f)),
                'energy_variance': float(gp1_var_e[0]) if gp1_var_e is not None else None,
                'force_variance': float(np.mean(gp1_var_f)) if gp1_var_f is not None else None,
                'thermal_energy': float(e_thermal),
                'thermal_force_norm': float(np.linalg.norm(f_thermal)),
                'num_snapshots': self.num_snapshots,
                'reference_applied': self.reference_set
            }
        )

        self.obs_total += 1

        return energy_ref, full_forces

    def _evaluate_position(self, position: npt.NDArray[np.float64], is_thermal: bool = False) -> Tuple[float, npt.NDArray[np.float64]]:
        """Enhanced evaluation with robust GP1 handling and fallback."""
        try:
            return self._evaluate_position_core(position, is_thermal)
        except RuntimeError as e:
            if self.gp1_fallback_to_averaging and ("cholesky" in str(e).lower() or "nan" in str(e).lower()):
                self.gp1_training_failures += 1
                logger.warning(f"GP1 evaluation failed: {e}")
                logger.warning("Falling back to direct thermal averaging without GP1")
                logger.warning("Using zero forces as fallback - this is a placeholder implementation")
                return 0.0, np.zeros_like(position)
            else:
                raise
            
        
    def _relaxation_phase(self):
        """Enhanced relaxation phase with trust region and better convergence detection."""
        # Train GP2 with all data
        self._train_gp2(reinit_hyperparams=(self.bigiter <= self.num_bigiter_initparam))
        
        # Calculate adaptive convergence threshold
        if self.enable_adaptive_convergence and len(self.maxF_R_acc) > 0:
            # Use dynamic threshold based on achieved forces
            min_achieved = min(self.maxF_R_acc)
            T_dimer_gp = max(
                min_achieved / self.divisor_T_dimer_gp,
                self.dimer_stopping_criteria / 10.0,
                0.001  # Absolute minimum
            )
        else:
            T_dimer_gp = self.dimer_stopping_criteria / 10.0
            
        # Set initial position for relaxation
        if self.bigiter > self.num_bigiter_initloc and self.R_latest_conv is not None:
            self.R = self.R_latest_conv.copy()
            self.orient = self.orient_latest_conv.copy()
        else:
            self.R = self.initial_position.copy()
            if self.orient is None:
                # Initialize orientation from forces (from parent class logic)
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
                
        # Reset dimer with GP forces
        self.dimer.x = self.R.copy()
        self.dimer.orient = self.orient.copy()
        self.dimer.force_func = self._force_func_gp
        self.dimer.dimer_stopping_criteria = T_dimer_gp
        
        # Track convergence
        converged = False
        oscillating = False
        force_minimum_detected = False
        exit_reason = "max_iterations"
        iterations_completed = 0
        
        # Inner iteration loop
        for inner_iter in range(self.max_inner_iterations):
            iterations_completed = inner_iter + 1
            # Get GP predictions at current position
            E_gp, G_gp = self._gp_evaluate(self.R)
            maxF_gp = np.max(np.abs(G_gp))
            
            # Calculate curvature if available (dimer has curvature after rotation)
            curvature = self.dimer.Curv if hasattr(self.dimer, 'Curv') else None
            
            # Store GP trajectory
            self.E_R_gp.append(E_gp)
            self.maxF_R_gp.append(maxF_gp)
            
            # Store force vectors for trust region calculation
            if not hasattr(self, 'G_R_gp'):
                self.G_R_gp = []
            self.G_R_gp.append(G_gp.copy())
            
            # NEW_v25: Log relaxation step data
            relaxation_entry = {
                'outer_iteration': self.bigiter,
                'inner_iteration': inner_iter,
                'energy': E_gp,
                'max_force': maxF_gp,
                'forces': G_gp.copy(),
                'position': self.R.copy(),
                'curvature': curvature,
                'trust_region_radius': self.trust_region.radius if self.enable_trust_region else None,
                'timestamp': time.time()
            }
            self.relaxation_log.append(relaxation_entry)
            
            # Update force minimum detector
            if self.enable_force_minimum_detection:
                self.force_minimum_detector.update(maxF_gp, curvature)
                
            # Check various convergence criteria
            # 1. Standard convergence (only if not trapped in negative GP2 energy)
            if maxF_gp < T_dimer_gp and not (inner_iter > 0 and self.E_R_gp and self.E_R_gp[-1] < 0):
                self.R_latest_conv = self.R.copy()
                self.orient_latest_conv = self.orient.copy()
                converged = True
                self.convergence_type = "standard"
                if self.verbose:
                    print(f"Converged on GP surface (standard) after {inner_iter} iterations")
                exit_reason = "standard_convergence"
                break
                
            # 1b. Explicit check for GP2 forces < 0.01 eV/Å
            if maxF_gp < 0.01:
                self.R_latest_conv = self.R.copy()
                self.orient_latest_conv = self.orient.copy()
                converged = True
                self.convergence_type = "gp2_force_converged"
                if self.verbose:
                    print(f"Converged on GP surface (GP2 force < 0.01 eV/Å) after {inner_iter} iterations")
                exit_reason = "gp2_force_limit"
                break
                
            # 2. Relaxed saddle convergence
            if (curvature is not None and curvature < 0 and maxF_gp < self.relaxed_saddle_criteria
                    and not (inner_iter > 0 and self.E_R_gp and self.E_R_gp[-1] < 0)):
                self.R_latest_conv = self.R.copy()
                self.orient_latest_conv = self.orient.copy()
                converged = True
                self.convergence_type = "relaxed_saddle"
                if self.verbose:
                    print(f"Converged on GP surface (relaxed saddle) after {inner_iter} iterations")
                exit_reason = "relaxed_saddle"
                break
                
            # 3. Force minimum detection
            if self.enable_force_minimum_detection:
                passed_minimum, min_force = self.force_minimum_detector.detect_minimum_passed()
                if passed_minimum:
                    force_minimum_detected = True
                    self.force_minimum_events.append({
                        'iteration': self.bigiter,
                        'inner_iteration': inner_iter,
                        'min_force': min_force,
                        'current_force': maxF_gp
                    })
                    if self.verbose:
                        print(f"Detected force minimum passage (min: {min_force:.4f}, current: {maxF_gp:.4f})")
                    # Don't break - let it continue to find actual minimum
                    
            # Perform dimer step
            self.dimer.x = self.R.copy()
            self.dimer.orient = self.orient.copy()
            
            # Get step vector from dimer
            step_vector = self.dimer.run()
            
            # Apply trust region if enabled
            if self.enable_trust_region:
                original_step = step_vector.copy()
                step_vector = self.trust_region.limit_step(step_vector)
                if not np.allclose(original_step, step_vector):
                    logger.debug(f"Trust region limited step from {np.linalg.norm(original_step):.4f} "
                               f"to {np.linalg.norm(step_vector):.4f}")
                    
            # Get new position
            R_new = self.dimer.x + step_vector
            
            # Create dimer tracking directory for monitoring
            if hasattr(self.local_pes, 'vasp_manager'):
                dimer_dir = self.local_pes.vasp_manager.setup_dimer_run(R_new)
                if self.verbose and inner_iter % 10 == 0:
                    print(f"  [Created dimer tracking directory: {dimer_dir}]")
            
            # Check oscillation
            if self.enable_oscillation_detection:
                if self.oscillation_detector.check_oscillation(R_new):
                    oscillating = True
                    self.oscillation_events.append({
                        'iteration': self.bigiter,
                        'inner_iteration': inner_iter,
                        'position': R_new.copy()
                    })
                    # Reduce trust region and continue
                    if self.enable_trust_region:
                        self.trust_region.radius *= 0.5
                        
            # Update position
            self.R = R_new.copy()
            self.orient = self.dimer.orient.copy()
            
            # Update trust region based on force change
            if self.enable_trust_region and inner_iter > 0 and len(self.G_R_gp) > 1:
                force_vector_before = self.G_R_gp[-2]  # Previous force vector
                force_vector_after = self.G_R_gp[-1]   # Current force vector
                self.trust_region.adjust_radius(force_vector_before, force_vector_after, np.linalg.norm(step_vector))
                
            # Check stopping criteria (from parent class)
            # 1. Inter-atomic distance check
            if inner_iter > 0 and self._check_interatomic_distances(R_new):
                if self.verbose:
                    print("Stopping: inter-atomic distances")
                self.num_es1 += 1
                exit_reason = "interatomic_distance"
                break
                
            # 2. Displacement check
            if inner_iter > 0 and self._check_displacement(R_new):
                if self.verbose:
                    print("Stopping: displacement limit")
                self.num_es2 += 1
                exit_reason = "displacement_limit"
                break
                
            # Early termination for oscillation
            if oscillating and inner_iter > 20:
                exit_reason = "oscillation_limit"
                break
                
        # Record exit reason and final relaxation snapshot
        self._record_relaxation_exit(exit_reason, iterations_completed, converged)
        last_energy = self.E_R_gp[-1] if hasattr(self, 'E_R_gp') and self.E_R_gp else None
        last_force = self.maxF_R_gp[-1] if hasattr(self, 'maxF_R_gp') and self.maxF_R_gp else None
        self.relaxation_log.append({
            'outer_iteration': self.bigiter,
            'inner_iteration': iterations_completed,
            'energy': last_energy,
            'max_force': last_force,
            'curvature': None,
            'trust_region_radius': self.trust_region.radius if self.enable_trust_region else None,
            'exit_reason': exit_reason
        })

        # Evaluate at final position with accurate forces
        if self.eval_image1:
            self.E_R_acc.append(self.E_R_gp[-1])
            self.maxF_R_acc.append(self.maxF_R_gp[-1])
        else:
            # Use GP1 thermal sampling for final evaluation
            E_R, G_R = self._evaluate_position(self.R)
            self._add_observation(self.R, E_R, G_R)
            self.E_R_acc.append(E_R)
            
            # Calculate max force for moving atoms only
            G_R_3d = G_R.reshape(-1, 3)
            G_R_moving = G_R_3d[self.moving_indices]
            maxF_R_moving = np.max(np.abs(G_R_moving))
            self.maxF_R_acc.append(maxF_R_moving)
            
        # Update table history with enhanced information
        # Create table entry (from parent class logic)
        # Get GP1 predictions for comparison
        pos_3d = self.R.reshape(-1, 3)
        pos_moving = pos_3d[self.moving_indices].flatten()
        
        if self.gp1 and self.gp1.model is not None:
            gp1_pred_e, gp1_pred_f, _, _ = self.gp1.predict(pos_moving.reshape(1, -1))
            final_E_gp1 = float(gp1_pred_e[0])
            final_maxF_gp1 = np.max(np.abs(gp1_pred_f))
        else:
            final_E_gp1 = np.nan
            final_maxF_gp1 = np.nan
            
        # Get GP2 predictions
        final_E_gp2 = self.E_R_gp[-1] if self.E_R_gp else np.nan
        final_maxF_gp2 = self.maxF_R_gp[-1] if self.maxF_R_gp else np.nan
        
        # Get actual values
        E_R = self.E_R_acc[-1]
        
        # Calculate curvatures
        curv_dimer = self.dimer.Curv if hasattr(self.dimer, 'Curv') else np.nan
        curv_gp1 = np.nan  # Not easily available
        curv_gp2 = np.nan  # Not easily available
        
        # Calculate differences
        e_diff_gp2_gp1 = final_E_gp2 - final_E_gp1 if not np.isnan(final_E_gp2) and not np.isnan(final_E_gp1) else np.nan
        e_diff_gp2_actual = final_E_gp2 - E_R if not np.isnan(final_E_gp2) else np.nan
        
        table_entry = {
            'step': self.bigiter,
            # Show relative energies with respect to initial position
            'E_GP1': final_E_gp1 - self.initial_energy_for_table if not np.isnan(final_E_gp1) else np.nan,
            'E_GP2': final_E_gp2 - self.initial_energy_for_table if not np.isnan(final_E_gp2) else np.nan,
            'E_Actual': E_R - self.initial_energy_for_table,  # Relative to initial
            'F_GP1': final_maxF_gp1,
            'F_GP2': final_maxF_gp2,
            'F_Actual': self.maxF_R_acc[-1],
            'Curvature_GP1': curv_gp1,
            'Curvature_GP2': curv_gp2,
            'Curvature_dimer': curv_dimer,
            'E_diff_GP2_GP1': e_diff_gp2_gp1,
            'E_diff_GP2_Actual': e_diff_gp2_actual,
            # Enhanced information
            'convergence_type': self.convergence_type,
            'trust_region_radius': self.trust_region.radius if self.enable_trust_region else None,
            'oscillating': oscillating,
            'force_minimum_detected': force_minimum_detected,
            'inner_iterations': inner_iter
        }
        self.table_history.append(table_entry)
        
    def _check_convergence(self) -> Tuple[bool, str]:
        """Enhanced convergence checking with multiple criteria."""
        # Get latest values
        maxF_actual = self.maxF_R_acc[-1] if self.maxF_R_acc else np.inf
        maxF_gp2 = self.maxF_R_gp[-1] if self.maxF_R_gp else np.inf
        curvature = self.dimer.Curv if hasattr(self.dimer, 'Curv') else None
        latest_gp2_energy = self.E_R_gp[-1] if hasattr(self, 'E_R_gp') and self.E_R_gp else np.inf
        
        # If GP2 energy is still negative, force the outer loop to continue
        if latest_gp2_energy < 0.0:
            return False, ""
        
        # 1. Standard convergence - based on GP2 forces
        if maxF_gp2 < self.dimer_stopping_criteria:
            return True, "standard"
            
        # 2. Relaxed saddle convergence - based on GP2 forces
        if curvature is not None and curvature < 0 and maxF_gp2 < self.relaxed_saddle_criteria:
            return True, "relaxed_saddle"
            
        # 3. Force minimum passed (only if we've detected it multiple times) - based on GP2 forces
        if self.enable_force_minimum_detection and len(self.force_minimum_events) >= 2:
            # Check if GP2 force has been increasing consistently
            if len(self.maxF_R_gp) >= 3:
                if all(self.maxF_R_gp[i] < self.maxF_R_gp[i+1] for i in range(-3, -1)):
                    return True, "force_minimum_passed"
                    
        return False, ""
        
    def run(self) -> Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
        """Enhanced run method with better convergence detection."""
        if self.verbose:
            print("\nStarting Enhanced Dual GP dimer search...")
            print(f"Trust region: {'ENABLED' if self.enable_trust_region else 'DISABLED'}")
            print(f"Oscillation detection: {'ENABLED' if self.enable_oscillation_detection else 'DISABLED'}")
            print(f"Force minimum detection: {'ENABLED' if self.enable_force_minimum_detection else 'DISABLED'}")
            print(f"Relaxed saddle criteria: {self.relaxed_saddle_criteria} eV/Å")
            
        # Establish energy reference using GP1 thermal average (from parent class logic)
        if not self.reference_set:
            print("\nEstablishing energy reference using GP1 thermal average...")
            # Create initial GP1 evaluation at starting position
            self.gp1.current_location = self.initial_position.copy()
            self.gp1.set_current_iteration("reference_establishment")
            
            # IMPORTANT: Ensure GP1 has no energy reference during reference establishment
            gp1_old_reference = getattr(self.gp1, 'energy_reference', None)
            self.gp1.energy_reference = None
            self.gp1.disable_auto_referencing = True  # Disable automatic referencing
            print(f"  Temporarily cleared GP1 energy_reference (was: {gp1_old_reference})")
            print(f"  Disabled automatic energy referencing in GP1")
            
            # Get thermal average from GP1
            x_thermal_ref, e_thermal_ref, f_thermal_ref = self.gp1.get_thermalized_current_location(
                num_snapshots=self.num_snapshots,
                thermal_noise=self.thermal_noise,
                eigenvalues_tr=self.eigenvalues_tr,
                eigenvectors_tr=self.eigenvectors_tr,
                context='reference_establishment',
                dimer_step=None,
                moving_only=True
            )
            
            print(f"DEBUG: e_thermal_ref from GP1 during reference establishment: {e_thermal_ref:.4f} eV")
            
            # Set the energy reference from GP1 thermal average
            self.energy_reference = e_thermal_ref
            self.reference_set = True
            
            # Now set GP1's reference to the established value
            self.gp1.energy_reference = self.energy_reference
            self.gp1.disable_auto_referencing = False  # Re-enable automatic referencing
            
            # Store temperature information from this evaluation
            if hasattr(self.gp1, 'thermo_stats') and self.gp1.thermo_stats:
                self.initial_avg_temperature = self.gp1.thermo_stats.get('avg_temperature', self.temperature)
            else:
                self.initial_avg_temperature = self.temperature
            
            print(f"Energy reference established from GP1: {self.energy_reference:.4f} eV")
            print(f"Average temperature: {self.initial_avg_temperature:.1f} K")
        
        # Now evaluate initial position (will use the established reference)
        E_R, G_R = self._evaluate_position(self.initial_position)
        self._add_observation(self.initial_position, E_R, G_R)
        self.E_R_acc.append(E_R)
        
        # Calculate max force for moving atoms only
        G_R_3d = G_R.reshape(-1, 3)
        G_R_moving = G_R_3d[self.moving_indices]
        maxF_R_moving = np.max(np.abs(G_R_moving))
        self.maxF_R_acc.append(maxF_R_moving)
        self.R = self.initial_position.copy()
        
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
        
        # IMPORTANT: Evaluate GP1 at initial position to establish reference for ALL energies
        print("\nEvaluating GP1 at initial position to establish table reference...")
        
        # Get GP1 prediction at initial position
        initial_E_gp1 = np.nan
        initial_F_gp1 = np.nan
        
        if hasattr(self, 'gp1') and self.gp1 is not None:
            # GP1 should be trained now after initial rotations
            pos_3d = self.initial_position.reshape(-1, 3)
            pos_moving = pos_3d[self.moving_indices].flatten()
            
            try:
                pred_e_gp1, pred_f_gp1, _, _ = self.gp1.predict(pos_moving.reshape(1, -1))
                initial_E_gp1 = float(pred_e_gp1[0])
                initial_F_gp1 = np.max(np.abs(pred_f_gp1))
                
                # This is our reference for the table - GP1 energy at initial position
                self.initial_energy_for_table = initial_E_gp1
                print(f"GP1 energy at initial position: {initial_E_gp1:.6f} eV (this will be the reference)")
            except Exception as e:
                print(f"Warning: Could not get GP1 prediction at initial position: {e}")
                # Fallback to using 0K-DFT as reference
                self.initial_energy_for_table = self.E_R_acc[-1]
                print(f"Using 0K-DFT as reference: {self.initial_energy_for_table:.6f} eV")
        else:
            # Fallback if GP1 not available
            self.initial_energy_for_table = self.E_R_acc[-1]
            print(f"GP1 not available, using 0K-DFT as reference: {self.initial_energy_for_table:.6f} eV")
        
        # Record initial state in table
        initial_entry = {
            'step': 0,
            'E_GP1': 0.0000 if not np.isnan(initial_E_gp1) else np.nan,  # GP1 defines the reference
            'E_GP2': np.nan,  # No GP2 yet
            # 0K-DFT energy relative to GP1 reference
            'E_Actual': self.E_R_acc[-1] - self.initial_energy_for_table,
            'F_GP1': initial_F_gp1 if not np.isnan(initial_F_gp1) else np.nan,
            'F_GP2': np.nan,
            'F_Actual': self.maxF_R_acc[-1],
            'Curvature_GP1': np.nan,
            'Curvature_GP2': np.nan,
            'Curvature_dimer': np.nan,
            'E_diff_GP2_GP1': np.nan,
            'E_diff_GP2_Actual': np.nan
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
                
            # Perform relaxation phase
            self._relaxation_phase()
            
            # Print progress table
            if self.table_history:
                self._print_progress_table()
                
            # Check convergence with enhanced criteria
            converged, convergence_type = self._check_convergence()
            if converged:
                self.converged = True
                self.convergence_type = convergence_type
                if self.verbose:
                    print(f"\n" + "="*60)
                    print(f"CONVERGED ({convergence_type}) after {self.bigiter} outer iterations")
                    print(f"Total observations: {self.obs_total}")
                    print("="*60)
                break
                
            # Update minimum force for adaptive threshold
            self.min_force_achieved = min(self.min_force_achieved, self.maxF_R_acc[-1])
            
        if not self.converged and self.verbose:
            print(f"\n" + "="*60)
            print(f"MAXIMUM ITERATIONS REACHED")
            print(f"Total observations: {self.obs_total}")
            print(f"Best force achieved: {self.min_force_achieved:.6f} eV/Å")
            print("="*60)
            
        # Save final checkpoint with enhanced information
        self.save_checkpoint(final=True)
        
        # NEW_v25: Save relaxation log to file
        self._save_relaxation_log()

        # Calculate Vineyard attempt frequency using the last GP2 model
        if self.gp2_trained and len(self.R_all) >= 2:
            self._calculate_vineyard_frequency()

        return self.R, self.E_R_acc[-1], self.G_all[-1]
        
    def _record_relaxation_exit(self, reason: str, iterations_completed: int, converged: bool):
        """Record and persist the reason why the relaxation loop exited."""
        record = {
            'timestamp': time.time(),
            'outer_iteration': self.bigiter,
            'inner_iterations_completed': iterations_completed,
            'reason': reason,
            'converged': converged
        }
        self.relaxation_exit_events.append(record)
        log_path = get_output_path('dimer_runs', 'relaxation_exit_reasons.log')
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

    def _save_relaxation_log(self):
        """NEW_v25: Save relaxation step energy and forces to organized files."""
        if not self.relaxation_log:
            return
            
        # Save as pickle for full data
        relaxation_log_pkl = get_output_path('dimer_runs', 'relaxation_log.pkl')
        with open(relaxation_log_pkl, 'wb') as f:
            pickle.dump(self.relaxation_log, f)
            
        # Save as text file for easy reading
        relaxation_log_txt = get_output_path('dimer_runs', 'relaxation_log.txt')
        with open(relaxation_log_txt, 'w') as f:
            f.write("NEW_v25 RELAXATION STEPS LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Total relaxation steps: {len(self.relaxation_log)}\n")
            f.write(f"Converged: {self.converged}\n")
            f.write(f"Convergence type: {self.convergence_type or 'Not converged'}\n")
            f.write("="*80 + "\n\n")
            
            f.write("DETAILED RELAXATION TRAJECTORY:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Outer':>6} {'Inner':>6} {'Energy':>12} {'Max Force':>12} {'Curvature':>12} {'Trust R':>10}\n")
            f.write(f"{'Iter':>6} {'Iter':>6} {'(eV)':>12} {'(eV/Å)':>12} {'(eV/Å²)':>12} {'(Å)':>10}\n")
            f.write("-"*80 + "\n")
            
            for entry in self.relaxation_log:
                outer = entry['outer_iteration']
                if 'exit_reason' in entry:
                    trust_r = entry.get('trust_region_radius', 0.0) if entry.get('trust_region_radius') is not None else 0.0
                    energy = entry.get('energy') or 0.0
                    max_force = entry.get('max_force') or 0.0
                    f.write(f"{outer:6d} {'EXIT':>6} {energy:12.6f} {max_force:12.6f} {0.0:12.6f} {trust_r:10.4f}  [{entry['exit_reason']}]\n")
                else:
                    inner = entry['inner_iteration']
                    energy = entry['energy']
                    max_force = entry['max_force']
                    curv = entry.get('curvature', 0.0) if entry.get('curvature') is not None else 0.0
                    trust_r = entry.get('trust_region_radius', 0.0) if entry.get('trust_region_radius') is not None else 0.0
                    f.write(f"{outer:6d} {inner:6d} {energy:12.6f} {max_force:12.6f} {curv:12.6f} {trust_r:10.4f}\n")
            
            f.write("-"*80 + "\n")
            
            # Summary statistics
            f.write("\nSUMMARY STATISTICS:\n")
            f.write("-"*80 + "\n")
            
            energies = [e['energy'] for e in self.relaxation_log if e.get('energy') is not None]
            forces = [e['max_force'] for e in self.relaxation_log if e.get('max_force') is not None]
            
            if energies:
                f.write(f"Energy range: [{min(energies):.6f}, {max(energies):.6f}] eV\n")
                f.write(f"Final energy: {energies[-1]:.6f} eV\n")
            else:
                f.write("Energy range: [NA, NA] eV\n")
                f.write("Final energy: NA eV\n")
            if forces:
                f.write(f"Force range: [{min(forces):.6f}, {max(forces):.6f}] eV/Å\n")
                f.write(f"Final max force: {forces[-1]:.6f} eV/Å\n")
            else:
                f.write("Force range: [NA, NA] eV/Å\n")
                f.write("Final max force: NA eV/Å\n")
            
            if self.enable_trust_region:
                radii = [e['trust_region_radius'] for e in self.relaxation_log if e['trust_region_radius'] is not None]
                if radii:
                    f.write(f"Trust region range: [{min(radii):.4f}, {max(radii):.4f}] Å\n")
                    f.write(f"Final trust region: {radii[-1]:.4f} Å\n")
        
        if self.verbose:
            print(f"Saved relaxation log to:")
            print(f"  - {relaxation_log_pkl} (full data)")
            print(f"  - {relaxation_log_txt} (summary)")
    
    def _print_enhanced_summary(self):
        """Print enhanced summary with additional statistics."""
        print("\n" + "="*60)
        print("ENHANCED SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"Convergence type: {self.convergence_type or 'Not converged'}")
        print(f"GP1 training failures: {self.gp1_training_failures}")
        
        # Trust region statistics
        if self.enable_trust_region and self.trust_region.history:
            radii = [h['radius'] for h in self.trust_region.history]
            print(f"\nTrust Region Statistics:")
            print(f"  Initial radius: {self.trust_region_initial:.4f} Å")
            print(f"  Final radius: {radii[-1]:.4f} Å")
            print(f"  Min radius used: {min(radii):.4f} Å")
            print(f"  Max radius used: {max(radii):.4f} Å")
            
        # Oscillation statistics
        if self.oscillation_events:
            print(f"\nOscillation Events: {len(self.oscillation_events)}")
            for i, event in enumerate(self.oscillation_events[:3]):  # Show first 3
                print(f"  Event {i+1}: Outer iter {event['iteration']}, "
                      f"inner iter {event['inner_iteration']}")
                      
        # Force minimum statistics
        if self.force_minimum_events:
            print(f"\nForce Minimum Events: {len(self.force_minimum_events)}")
            for i, event in enumerate(self.force_minimum_events):
                print(f"  Event {i+1}: Min force {event['min_force']:.6f} eV/Å "
                      f"at outer iter {event['iteration']}")
                      
        print("="*60)
        
    def _save_checkpoint_base(self, final=False):
        """Base checkpoint save (from parent WalkerDualGP)."""
        self._update_statistics()

        checkpoint_data = {
            'walker_type': 'WalkerDualGP',
            'iteration': self.bigiter,
            'converged': self.converged,
            'energy_reference': self.energy_reference,
            'reference_set': self.reference_set,
            'R': self.R,
            'orient': self.orient,
            'R_all': self.R_all,
            'E_all': self.E_all,
            'G_all': self.G_all,
            'e_thermal_all': self.e_thermal_all,
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
            'table_history': self.table_history,
            'raw_energy_std': self.raw_energy_std,
            'raw_force_std': self.raw_force_std,
            'raw_energy_mad': self.raw_energy_mad,
            'raw_force_mad': self.raw_force_mad,
            'raw_energy_std_mad_ratio': self.raw_energy_std_mad_ratio,
            'raw_force_std_mad_ratio': self.raw_force_std_mad_ratio,
            'gp1_energy_std': self.gp1_energy_std,
            'gp1_force_std': self.gp1_force_std,
            'gp1_energy_mad': self.gp1_energy_mad,
            'gp1_force_mad': self.gp1_force_mad,
            'gp1_energy_std_mad_ratio': self.gp1_energy_std_mad_ratio,
            'gp1_force_std_mad_ratio': self.gp1_force_std_mad_ratio,
            'gp2_energy_std': self.gp2_energy_std,
            'gp2_force_std': self.gp2_force_std,
            'gp1_initial_noise': self.gp1_initial_noise,
            'gp1_final_noise': self.gp1_final_noise,
            'gp2_initial_noise': self.gp2_initial_noise,
            'gp2_final_noise': self.gp2_final_noise,
            'avg_temperatures': self.avg_temperatures,
            'initial_avg_temperature': getattr(self, 'initial_avg_temperature', self.temperature)
        }

        checkpoints_dir = get_output_path('checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        if final:
            filename = os.path.join(checkpoints_dir, 'gp2_dimer_final.pkl')
        else:
            filename = os.path.join(checkpoints_dir, f'gp2_dimer_checkpoint_{self.bigiter}.pkl')

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        latest_filename = os.path.join(checkpoints_dir, 'gp2_dimer_latest.pkl')
        with open(latest_filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        if self.verbose and (self.bigiter % 10 == 0 or final):
            print(f"  [Checkpoint saved at iteration {self.bigiter}]")

    def _load_checkpoint_base(self, checkpoint_file=None):
        """Base checkpoint load (from parent WalkerDualGP)."""
        if checkpoint_file is None:
            checkpoint_file = get_output_path('checkpoints', 'gp2_dimer_latest.pkl')
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)

        self.bigiter = checkpoint['iteration']
        self.converged = checkpoint['converged']
        self.energy_reference = checkpoint['energy_reference']
        self.reference_set = checkpoint['reference_set']
        self.R = checkpoint['R']
        self.orient = checkpoint['orient']
        self.R_all = checkpoint['R_all']
        self.E_all = checkpoint['E_all']
        self.G_all = checkpoint['G_all']
        self.e_thermal_all = checkpoint.get('e_thermal_all', np.zeros_like(self.E_all))
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
        self.table_history = checkpoint.get('table_history', [])

        if len(self.R_all) > 1:
            self._train_gp2()

        if self.verbose:
            print(f"\nRestored checkpoint from iteration {self.bigiter}")
            print(f"Total observations so far: {self.obs_total}")
            print(f"Table history entries: {len(self.table_history)}")

        return checkpoint

    def save_checkpoint(self, final: bool = False):
        """Enhanced checkpoint saving with complete trust region and enhanced features data."""
        # Call base checkpoint save
        self._save_checkpoint_base(final)
        
        # Prepare comprehensive trust region data
        trust_region_data = None
        if self.enable_trust_region:
            trust_region_data = {
                'enabled': True,
                'current_radius': self.trust_region.radius,
                'min_radius': self.trust_region.min_radius,
                'max_radius': self.trust_region.max_radius,
                'history': self.trust_region.history,
                'initial_radius': self.trust_region_initial,
                'trust_region_min': self.trust_region_min,
                'trust_region_max': self.trust_region_max,
            }
        
        # Prepare oscillation detector data
        oscillation_data = None
        if self.enable_oscillation_detection:
            oscillation_data = {
                'enabled': True,
                'memory': self.oscillation_detector.memory,
                'threshold': self.oscillation_detector.threshold,
                'position_history': self.oscillation_detector.position_history,
                'events': self.oscillation_events,
                'oscillation_memory': self.oscillation_memory,
                'oscillation_threshold': self.oscillation_threshold,
            }
        
        # Prepare force minimum detector data
        force_minimum_data = None
        if self.enable_force_minimum_detection:
            force_minimum_data = {
                'enabled': True,
                'window': self.force_minimum_detector.window,
                'increase_threshold': self.force_minimum_detector.increase_threshold,
                'force_history': self.force_minimum_detector.force_history,
                'curvature_history': self.force_minimum_detector.curvature_history,
                'events': self.force_minimum_events,
                'force_minimum_window': self.force_minimum_window,
                'force_minimum_increase_threshold': self.force_minimum_increase_threshold,
            }
        
        # Save comprehensive enhanced features data
        checkpoint_extra = {
            # Trust region
            'trust_region_data': trust_region_data,
            'enable_trust_region': self.enable_trust_region,
            
            # Oscillation detection
            'oscillation_data': oscillation_data,
            'enable_oscillation_detection': self.enable_oscillation_detection,
            
            # Force minimum detection
            'force_minimum_data': force_minimum_data,
            'enable_force_minimum_detection': self.enable_force_minimum_detection,
            
            # Convergence settings
            'relaxed_saddle_criteria': self.relaxed_saddle_criteria,
            'enable_adaptive_convergence': self.enable_adaptive_convergence,
            'convergence_type': self.convergence_type,
            
            # GP1 settings
            'gp1_fallback_to_averaging': self.gp1_fallback_to_averaging,
            'gp1_max_retries': self.gp1_max_retries,
            'gp1_training_failures': self.gp1_training_failures,
            
            # Parameter validation
            'validate_parameters': self.validate_parameters,
            
            # Additional state
            'min_force_achieved': getattr(self, 'min_force_achieved', float('inf')),
            'iteration': self.bigiter,
            'converged': self.converged,
        }
        
        suffix = "_final" if final else f"_step_{self.bigiter}"
        extra_file = get_output_path('checkpoints', f'checkpoint_extra{suffix}.pkl')
        
        with open(extra_file, 'wb') as f:
            pickle.dump(checkpoint_extra, f)
            
        if self.verbose:
            print(f"Saved enhanced checkpoint with trust region data to {extra_file}")
    
    def load_checkpoint(self, checkpoint_file=None):
        """Load checkpoint and restore complete state including trust region and enhanced features."""
        # Call base checkpoint load
        checkpoint = self._load_checkpoint_base(checkpoint_file)
        
        # Determine the corresponding extra checkpoint file
        if checkpoint_file is None:
            extra_file = get_output_path('checkpoints', f'checkpoint_extra_step_{self.bigiter}.pkl')
            # Try to find the latest extra checkpoint if specific one doesn't exist
            if not os.path.exists(extra_file):
                extra_file = get_output_path('checkpoints', 'checkpoint_extra_final.pkl')
        else:
            # Extract the suffix from the main checkpoint file
            import re
            match = re.search(r'(step_\d+|final)', checkpoint_file)
            if match:
                suffix = match.group(1)
                extra_file = get_output_path('checkpoints', f'checkpoint_extra_{suffix}.pkl')
            else:
                extra_file = None
        
        # Load enhanced features data if available
        if extra_file and os.path.exists(extra_file):
            with open(extra_file, 'rb') as f:
                checkpoint_extra = pickle.load(f)
            
            # Restore trust region
            if checkpoint_extra.get('trust_region_data') and self.enable_trust_region:
                tr_data = checkpoint_extra['trust_region_data']
                self.trust_region.radius = tr_data['current_radius']
                self.trust_region.min_radius = tr_data['min_radius']
                self.trust_region.max_radius = tr_data['max_radius']
                self.trust_region.history = tr_data['history']
                if self.verbose:
                    print(f"Restored trust region with radius: {self.trust_region.radius:.4f} Å")
                    print(f"  Trust region history: {len(self.trust_region.history)} entries")
            
            # Restore oscillation detector
            if checkpoint_extra.get('oscillation_data') and self.enable_oscillation_detection:
                osc_data = checkpoint_extra['oscillation_data']
                self.oscillation_detector.position_history = osc_data['position_history']
                self.oscillation_events = osc_data['events']
                if self.verbose:
                    print(f"Restored oscillation detector with {len(self.oscillation_events)} events")
            
            # Restore force minimum detector
            if checkpoint_extra.get('force_minimum_data') and self.enable_force_minimum_detection:
                fm_data = checkpoint_extra['force_minimum_data']
                self.force_minimum_detector.force_history = fm_data['force_history']
                self.force_minimum_detector.curvature_history = fm_data['curvature_history']
                self.force_minimum_events = fm_data['events']
                if self.verbose:
                    print(f"Restored force minimum detector with {len(self.force_minimum_events)} events")
            
            # Restore other settings
            self.convergence_type = checkpoint_extra.get('convergence_type')
            self.gp1_training_failures = checkpoint_extra.get('gp1_training_failures', 0)
            self.min_force_achieved = checkpoint_extra.get('min_force_achieved', float('inf'))
            
            # Restore convergence settings (in case they changed)
            self.relaxed_saddle_criteria = checkpoint_extra.get('relaxed_saddle_criteria', self.relaxed_saddle_criteria)
            self.enable_adaptive_convergence = checkpoint_extra.get('enable_adaptive_convergence', self.enable_adaptive_convergence)
            
            if self.verbose:
                print(f"Restored enhanced features from {extra_file}")
                print(f"  Convergence type: {self.convergence_type}")
                print(f"  GP1 training failures: {self.gp1_training_failures}")
                print(f"  Min force achieved: {self.min_force_achieved:.6f} eV/Å")
        else:
            if self.verbose:
                print(f"Warning: Could not find enhanced checkpoint file: {extra_file}")
                print("  Trust region and enhanced features will use default values")
        
        return checkpoint

    def _compute_gp2_hessian(self, position_moving: npt.NDArray[np.float64], epsilon: float = 1e-4) -> npt.NDArray[np.float64]:
        """Compute Hessian of GP2 energy via finite differences on moving atoms.

        Args:
            position_moving: Flattened moving atom positions (3*N_moving,)
            epsilon: Finite difference step size in Angstrom

        Returns:
            Hessian matrix of shape (3*N_moving, 3*N_moving) in eV/Å²
        """
        n = len(position_moving)
        pos = position_moving.flatten()

        e0 = float(self.gp2.predict(pos.reshape(1, -1))[0][0])

        e_plus = np.zeros(n)
        e_minus = np.zeros(n)
        for i in range(n):
            p = pos.copy()
            p[i] += epsilon
            e_plus[i] = float(self.gp2.predict(p.reshape(1, -1))[0][0])
            p = pos.copy()
            p[i] -= epsilon
            e_minus[i] = float(self.gp2.predict(p.reshape(1, -1))[0][0])

        H = np.zeros((n, n))

        # Diagonal
        for i in range(n):
            H[i, i] = (e_plus[i] - 2.0 * e0 + e_minus[i]) / (epsilon ** 2)

        # Off-diagonal
        for i in range(n):
            for j in range(i + 1, n):
                p = pos.copy()
                p[i] += epsilon
                p[j] += epsilon
                e_pp = float(self.gp2.predict(p.reshape(1, -1))[0][0])

                H[i, j] = (e_pp - e_plus[i] - e_plus[j] + e0) / (epsilon ** 2)
                H[j, i] = H[i, j]

        return H

    def _calculate_vineyard_frequency(self):
        """Calculate Vineyard attempt frequency nu* using GP2 normal modes at minimum and saddle."""
        if self.gp2 is None or not self.gp2_trained:
            return

        # Positions: initial = local minimum, current = saddle
        pos_min_full = self.R_all[0]
        pos_saddle_full = self.R

        pos_min_3d = pos_min_full.reshape(-1, 3)
        pos_saddle_3d = pos_saddle_full.reshape(-1, 3)

        pos_min_moving = pos_min_3d[self.moving_indices].flatten()
        pos_saddle_moving = pos_saddle_3d[self.moving_indices].flatten()

        n_moving_dof = len(pos_min_moving)

        # Mass vector for moving atoms
        if isinstance(self.mass, np.ndarray):
            if len(self.mass) == self.n_atoms:
                masses_moving = self.mass[self.moving_indices]
            elif len(self.mass) == self.n_moving:
                masses_moving = self.mass
            else:
                masses_moving = np.full(self.n_moving, np.mean(self.mass))
        else:
            masses_moving = np.full(self.n_moving, float(self.mass))

        mass_dof = np.repeat(masses_moving, 3)
        sqrt_mass = np.sqrt(mass_dof)
        inv_sqrt_mass = 1.0 / sqrt_mass

        # Compute GP2 Hessians
        H_min = self._compute_gp2_hessian(pos_min_moving)
        H_saddle = self._compute_gp2_hessian(pos_saddle_moving)

        # Mass-weighted Hessians (dynamical matrices): D_ij = H_ij / sqrt(m_i * m_j)
        D_min = H_min * np.outer(inv_sqrt_mass, inv_sqrt_mass)
        D_saddle = H_saddle * np.outer(inv_sqrt_mass, inv_sqrt_mass)

        # Symmetrise
        D_min = 0.5 * (D_min + D_min.T)
        D_saddle = 0.5 * (D_saddle + D_saddle.T)

        # Diagonalise
        eigenvalues_min, eigenvectors_min = np.linalg.eigh(D_min)
        eigenvalues_saddle, eigenvectors_saddle = np.linalg.eigh(D_saddle)

        # Unit conversion: eV/(amu*Å²) -> s^-2
        eV_to_J = 1.602176634e-19
        amu_to_kg = 1.66053906660e-27
        ang_to_m = 1.0e-10
        conv = eV_to_J / (amu_to_kg * ang_to_m**2)  # ~9.6485e27 s^-2

        def eigenvalues_to_freq_THz(evals):
            omega = np.sqrt(np.abs(evals) * conv)
            freq_Hz = omega / (2.0 * np.pi)
            return freq_Hz * 1e-12  # Hz -> THz

        freq_min_THz = eigenvalues_to_freq_THz(eigenvalues_min)
        freq_saddle_THz = eigenvalues_to_freq_THz(eigenvalues_saddle)

        signs_min = np.sign(eigenvalues_min)
        signs_saddle = np.sign(eigenvalues_saddle)

        n_imaginary_min = np.sum(eigenvalues_min < -1e-10)
        n_imaginary_saddle = np.sum(eigenvalues_saddle < -1e-10)

        # Vineyard formula: use real modes above threshold
        freq_threshold = 1e-3  # THz
        valid_min = (eigenvalues_min > 1e-10) & (freq_min_THz > freq_threshold)
        valid_saddle = (eigenvalues_saddle > 1e-10) & (freq_saddle_THz > freq_threshold)

        freq_min_valid = freq_min_THz[valid_min]
        freq_saddle_valid = freq_saddle_THz[valid_saddle]

        n_modes_min = len(freq_min_valid)
        n_modes_saddle = len(freq_saddle_valid)

        if n_modes_min == 0 or n_modes_saddle == 0:
            self.vineyard_frequency = None
            return

        # nu* = prod(nu_min) / prod(nu_saddle) via log-sum
        log_prod_min = np.sum(np.log(freq_min_valid))
        log_prod_saddle = np.sum(np.log(freq_saddle_valid))
        log_nu_star = log_prod_min - log_prod_saddle

        nu_star_THz = np.exp(log_nu_star)
        nu_star_Hz = nu_star_THz * 1e12

        self.vineyard_frequency = nu_star_Hz

        # Save normal modes data
        normal_modes_data = {
            'description': 'GP2-based normal modes for Vineyard attempt frequency',
            'pos_min_moving': pos_min_moving,
            'pos_min_full': pos_min_full,
            'hessian_min': H_min,
            'dynamical_matrix_min': D_min,
            'eigenvalues_min': eigenvalues_min,
            'eigenvectors_min': eigenvectors_min,
            'frequencies_min_THz': freq_min_THz,
            'signs_min': signs_min,
            'n_imaginary_min': int(n_imaginary_min),
            'pos_saddle_moving': pos_saddle_moving,
            'pos_saddle_full': pos_saddle_full,
            'hessian_saddle': H_saddle,
            'dynamical_matrix_saddle': D_saddle,
            'eigenvalues_saddle': eigenvalues_saddle,
            'eigenvectors_saddle': eigenvectors_saddle,
            'frequencies_saddle_THz': freq_saddle_THz,
            'signs_saddle': signs_saddle,
            'n_imaginary_saddle': int(n_imaginary_saddle),
            'vineyard_frequency_Hz': nu_star_Hz,
            'vineyard_frequency_THz': nu_star_THz,
            'n_modes_min_used': n_modes_min,
            'n_modes_saddle_used': n_modes_saddle,
            'masses_moving_amu': masses_moving,
            'moving_indices': self.moving_indices,
            'n_moving': self.n_moving,
            'freq_threshold_THz': freq_threshold,
            'hessian_epsilon_angstrom': 1e-4,
        }

        modes_pkl = get_output_path('dimer_runs', 'vineyard_normal_modes.pkl')
        os.makedirs(os.path.dirname(modes_pkl), exist_ok=True)
        with open(modes_pkl, 'wb') as f:
            pickle.dump(normal_modes_data, f)

        modes_txt = get_output_path('dimer_runs', 'vineyard_normal_modes.txt')
        with open(modes_txt, 'w') as f:
            f.write("VINEYARD ATTEMPT FREQUENCY - GP2 NORMAL MODES\n")
            f.write("="*70 + "\n\n")
            f.write(f"Vineyard nu* = {nu_star_THz:.6f} THz = {nu_star_Hz:.6e} Hz\n\n")
            f.write(f"Moving atoms: {self.n_moving}, DOFs: {n_moving_dof}\n")
            f.write(f"Masses (amu): {masses_moving}\n\n")

            f.write("NORMAL MODES AT LOCAL MINIMUM\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Mode':>6} {'Eigenvalue':>16} {'Freq (THz)':>14} {'Type':>10}\n")
            f.write("-"*70 + "\n")
            for i in range(n_moving_dof):
                mode_type = "imaginary" if eigenvalues_min[i] < -1e-10 else (
                    "acoustic" if abs(eigenvalues_min[i]) < 1e-10 else "real")
                f.write(f"{i+1:6d} {eigenvalues_min[i]:16.8f} {freq_min_THz[i]:14.6f} {mode_type:>10}\n")

            f.write("\nNORMAL MODES AT SADDLE POINT\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Mode':>6} {'Eigenvalue':>16} {'Freq (THz)':>14} {'Type':>10}\n")
            f.write("-"*70 + "\n")
            for i in range(n_moving_dof):
                mode_type = "imaginary" if eigenvalues_saddle[i] < -1e-10 else (
                    "acoustic" if abs(eigenvalues_saddle[i]) < 1e-10 else "real")
                f.write(f"{i+1:6d} {eigenvalues_saddle[i]:16.8f} {freq_saddle_THz[i]:14.6f} {mode_type:>10}\n")

            f.write("\n" + "="*70 + "\n")
            f.write(f"Modes used at minimum: {n_modes_min} (real, > {freq_threshold} THz)\n")
            f.write(f"Modes used at saddle:  {n_modes_saddle} (real, > {freq_threshold} THz)\n")
            f.write(f"Vineyard nu* = {nu_star_THz:.6f} THz = {nu_star_Hz:.6e} Hz\n")

    def _add_observation(self, position: npt.NDArray[np.float64], energy: float, forces: npt.NDArray[np.float64]):
        """Add new observation to the dataset."""
        self.R_all = np.vstack((self.R_all, position.reshape(1, -1)))
        self.E_all = np.vstack((self.E_all, [[energy]]))
        self.G_all = np.vstack((self.G_all, forces.reshape(1, -1)))
        e_thermal_val = getattr(self, '_last_e_thermal', 0.0)
        self.e_thermal_all = np.vstack((self.e_thermal_all, [[e_thermal_val]]))

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

    def _force_func_accurate(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate accurate forces using GP1 thermal sampling.
        
        MODIFIED: Now uses GP1 instead of direct VASP.
        """
        if position.ndim == 1:
            # Single position
            _, forces = self._evaluate_position(position)
            return forces
        else:
            # Multiple positions
            forces_list = []
            for pos in position:
                _, forces = self._evaluate_position(pos)
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
        pred_e, pred_f, var_e, var_f = self.gp2.predict(pos_moving.reshape(1, -1))

        # LOG GP2 EVALUATION
        gp_logger = get_gp_logger()
        gp_logger.log_gp_prediction(
            model_name='GP2',
            iteration=len(self.E_R_gp),  # Inner iteration count
            position=pos_moving,
            prediction={  # Changed from 'predictions' to 'prediction'
                'energy': float(pred_e[0]),
                'force_norm': float(np.linalg.norm(pred_f)),
                'max_force': float(np.max(np.abs(pred_f))),
                'energy_variance': float(var_e[0]) if var_e is not None else None,
                'force_variance': float(np.mean(var_f)) if var_f is not None else None,
                'outer_iteration': self.bigiter
            }
        )
        
        # Reconstruct full forces
        full_forces = np.zeros_like(position)
        for i, idx in enumerate(self.moving_indices):
            full_forces[3*idx:3*(idx+1)] = pred_f[0, 3*i:3*(i+1)]
        
        return float(pred_e[0]), full_forces

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

            # # Max force for moving atoms only -- NEW
            # G_R_3d = G_R.reshape(-1, 3)
            # G_R_moving = G_R_3d[self.moving_indices]
            # maxF_R_moving = np.max(np.abs(G_R_moving))
            # self.maxF_R_acc.append(maxF_R_moving)
            # # End of NEW
            
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

    def _print_progress_table(self):
        """Print progress table with all history."""
        print("\n" + "="*85)
        print("DIMER PROGRESS TABLE")
        print("="*85)

        print(f"{'':>5} {'RELATIVE ENERGY (eV)':^40} {'FORCE (eV/Å)':^40}")
        print(f"{'Step':>5} {'GP1':>13} {'GP2':>13} {'0K-DFT':>13} {'GP1':>13} {'GP2':>13} {'0K-DFT':>13}")
        print("-"*85)

        for row in self.table_history:
            step = row['step']

            def fmt(val, decimals=4):
                if np.isnan(val):
                    return "---"
                else:
                    return f"{val:.{decimals}f}"

            vals = {
                'e_gp1': fmt(row.get('E_GP1', np.nan)),
                'e_gp2': fmt(row.get('E_GP2', np.nan)),
                'e_act': fmt(row.get('E_Actual', np.nan)),
                'f_gp1': fmt(row.get('F_GP1', np.nan)),
                'f_gp2': fmt(row.get('F_GP2', np.nan)),
                'f_act': fmt(row.get('F_Actual', np.nan)),
            }

            print(f"{step:5d} {vals['e_gp1']:>13} {vals['e_gp2']:>13} {vals['e_act']:>13} "
                f"{vals['f_gp1']:>13} {vals['f_gp2']:>13} {vals['f_act']:>13}")

        print("="*85)

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
        
        # Only print energy evolution if we have data
        if len(self.E_R_acc) > 0:
            print(f"\nEnergy evolution:")
            print(f"  Initial: {self.E_R_acc[0]:.6f} eV")
            print(f"  Final: {self.E_R_acc[-1]:.6f} eV")
            print(f"  Change: {self.E_R_acc[-1] - self.E_R_acc[0]:.6f} eV")
        
        # Only print force evolution if we have data
        if len(self.maxF_R_acc) > 0:
            print(f"\nForce evolution:")
            print(f"  Initial max |F|: {self.maxF_R_acc[0]:.6f} eV/Å")
            print(f"  Final max |F|: {self.maxF_R_acc[-1]:.6f} eV/Å")
        
        # Displacement
        if hasattr(self, 'R') and hasattr(self, 'initial_position'):
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
        if hasattr(self, 'table_history') and self.table_history and len(self.table_history) > 0:
            last_entry = self.table_history[-1]
            if 'Curvature_dimer' in last_entry and not np.isnan(last_entry['Curvature_dimer']):
                print(f"\nFinal curvature: {last_entry['Curvature_dimer']:.6f} eV/Å²")
        
        print("="*70)

    def _read_force_constants(self, POSCAR_file, force_constants_file):
        """Read force constants using phonopy (copied from walker.py)."""
        phonon = read_phonopy(
            sposcar=POSCAR_file,
            force_constants=force_constants_file
        )
        phonon.symmetrize_force_constants()
        
        freqs_thz, eigvecs_A = phonon.get_frequencies_with_eigenvectors(q=[0,0,0])
        
        # Drop acoustic modes
        mask = freqs_thz > 1e-6
        freqs_thz = freqs_thz[mask]
        eigvecs_A = eigvecs_A[:, mask].real
        
        # Convert to atomic units
        atomic_time = 2.4188843265857e-17
        omega_s = 2 * np.pi * freqs_thz * 1e12 * atomic_time
        eigvecs_bohr = eigvecs_A
        
        if self.verbose:
            n = Poscar.from_file(POSCAR_file).structure.num_sites
            print(f"Read {omega_s.size} phonon modes for {n} atoms")
            print(f"Frequency range: {freqs_thz.min():.3f} - {freqs_thz.max():.3f} THz")
        
        return omega_s, eigvecs_bohr

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
        # Bake each e_thermal into the corresponding training energy for GP2
        energies = self.E_all.flatten() + self.e_thermal_all.flatten()

        # IMPORTANT: Ensure energies are referenced
        if self.energy_reference is not None and abs(np.mean(energies)) > 100:
            print(f"WARNING: GP2 training data appears unreferenced. Mean energy: {np.mean(energies):.4f}")
            print(f"This should not happen - data from GP1 should already be referenced")
    
        
        # Create or update GP2
        if self.gp2 is None:
            # Initialize GP2 with GPU support
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
            # CRITICAL: Set the same energy reference
            self.gp2.energy_reference = self.energy_reference
        
        # Train the model
        self.gp2.set_current_iteration(f"bigiter_{self.bigiter}")
        
        
        self.gp2.train(
            training_data=self.gp2.training_data,
            thermal_noise=None,  # No thermal noise for GP2
            model_name="GP2",
            path=get_output_path("data_gp2"),
        )
        self.gp2_trained = True

        # The GP2 training already logs via gp_base.py, but we can add walker-specific info
        gp_logger = get_gp_logger()
        gp_logger.log_training_progress(
            model_name='WalkerDualGP',
            iteration=self.bigiter,
            metrics={
                'gp2_training_points': len(positions_moving),
                'energy_range': float(energies.max() - energies.min()),
                'force_range': float(forces_moving.max() - forces_moving.min()),
                'n_pt': self.atomic_info.get('n_pt'),
                'observations_so_far': self.obs_total
            }
        )
        
        # Store hyperparameters if available
        if hasattr(self.gp2, 'model') and self.gp2.model is not None:
            self.param_gp.append(self._extract_hyperparameters())

    def _update_statistics(self):
        """Update statistical measures for raw data and GP predictions."""
        # Raw data statistics
        if len(self.E_all) > 1:
            self.raw_energy_std = np.std(self.E_all)
            # Calculate MAD for energy
            energy_median = np.median(self.E_all)
            self.raw_energy_mad = np.median(np.abs(self.E_all - energy_median))
            self.raw_energy_std_mad_ratio = self.raw_energy_std / (self.raw_energy_mad + 1e-10)
            
            # Force magnitudes for all evaluations
            force_mags = [np.linalg.norm(f.flatten()) for f in self.G_all]
            self.raw_force_std = np.std(force_mags)
            # Calculate MAD for forces
            force_median = np.median(force_mags)
            self.raw_force_mad = np.median(np.abs(force_mags - force_median))
            self.raw_force_std_mad_ratio = self.raw_force_std / (self.raw_force_mad + 1e-10)
        
        # GP1 statistics
        if self.gp1 and hasattr(self.gp1, 'model') and self.gp1.model is not None:
            # Get initial noise (set when GP1 was first created)
            if self.gp1_initial_noise is None:
                # For GP1 with thermal noise, extract from self.thermal_noise
                if self.thermal_noise is not None:
                    noise_force, noise_energy = self.thermal_noise
                    self.gp1_initial_noise = {
                        'energy': float(noise_energy),
                        'force': float(noise_force)
                    }
                elif hasattr(self.gp1, 'likelihood'):
                    try:
                        if hasattr(self.gp1.likelihood, 'noise'):
                            self.gp1_initial_noise = float(self.gp1.likelihood.noise.item())
                    except:
                        self.gp1_initial_noise = 0.01  # Default value
            
            # Get current noise (for GP1 with fixed noise, it should be the same)
            if hasattr(self.gp1, 'likelihood'):
                if isinstance(self.gp1_initial_noise, dict):
                    # Fixed noise case - just copy initial values
                    self.gp1_final_noise = self.gp1_initial_noise.copy()
                else:
                    # Trainable noise case
                    try:
                        if hasattr(self.gp1.likelihood, 'raw_task_noises'):
                            # Multitask likelihood with task-specific noises
                            task_noises = self.gp1.likelihood.raw_task_noises_constraint.transform(
                                self.gp1.likelihood.raw_task_noises
                            )
                            task_noises_np = task_noises.detach().cpu().numpy()
                            # Assuming task 0 is energy, task 1+ are forces
                            self.gp1_final_noise = {
                                'energy': float(task_noises_np[0]) if len(task_noises_np) > 0 else None,
                                'force': float(task_noises_np[1]) if len(task_noises_np) > 1 else None
                            }
                        elif hasattr(self.gp1.likelihood, 'noise'):
                            # Single noise value
                            self.gp1_final_noise = float(self.gp1.likelihood.noise.item())
                    except Exception as e:
                        print(f"Warning: Could not extract GP1 final noise: {e}")
                        pass
            
            # Compute GP1 prediction statistics from table history
            gp1_e_preds = [entry['E_GP1'] for entry in self.table_history if not np.isnan(entry.get('E_GP1', np.nan))]
            gp1_f_preds = [entry['F_GP1'] for entry in self.table_history if not np.isnan(entry.get('F_GP1', np.nan))]
            
            if len(gp1_e_preds) > 1:
                self.gp1_energy_std = np.std(gp1_e_preds)
                # Calculate MAD for GP1 energy predictions
                gp1_e_median = np.median(gp1_e_preds)
                self.gp1_energy_mad = np.median(np.abs(np.array(gp1_e_preds) - gp1_e_median))
                self.gp1_energy_std_mad_ratio = self.gp1_energy_std / (self.gp1_energy_mad + 1e-10)
                
            if len(gp1_f_preds) > 1:
                self.gp1_force_std = np.std(gp1_f_preds)
                # Calculate MAD for GP1 force predictions
                gp1_f_median = np.median(gp1_f_preds)
                self.gp1_force_mad = np.median(np.abs(np.array(gp1_f_preds) - gp1_f_median))
                self.gp1_force_std_mad_ratio = self.gp1_force_std / (self.gp1_force_mad + 1e-10)
        
        # GP2 statistics
        if self.gp2 and hasattr(self.gp2, 'model') and self.gp2.model is not None:
            # Get initial noise
            if self.gp2_initial_noise is None and hasattr(self.gp2, 'likelihood'):
                try:
                    if hasattr(self.gp2.likelihood, 'raw_task_noises'):
                        # Multitask likelihood with task-specific noises
                        task_noises = self.gp2.likelihood.raw_task_noises_constraint.transform(
                            self.gp2.likelihood.raw_task_noises
                        )
                        task_noises_np = task_noises.detach().cpu().numpy()
                        self.gp2_initial_noise = {
                            'energy': float(task_noises_np[0]) if len(task_noises_np) > 0 else None,
                            'force': float(task_noises_np[1]) if len(task_noises_np) > 1 else None
                        }
                    elif hasattr(self.gp2.likelihood.noise, 'noise'):
                        self.gp2_initial_noise = float(self.gp2.likelihood.noise.noise.item())
                    elif hasattr(self.gp2.likelihood, 'noise'):
                        self.gp2_initial_noise = float(self.gp2.likelihood.noise.item())
                except Exception as e:
                    print(f"Warning: Could not extract GP2 initial noise: {e}")
                    self.gp2_initial_noise = 0.1  # Default value for GP2
            
            # Get current noise
            if hasattr(self.gp2, 'likelihood'):
                try:
                    if hasattr(self.gp2.likelihood, 'raw_task_noises'):
                        # Multitask likelihood with task-specific noises
                        task_noises = self.gp2.likelihood.raw_task_noises_constraint.transform(
                            self.gp2.likelihood.raw_task_noises
                        )
                        task_noises_np = task_noises.detach().cpu().numpy()
                        self.gp2_final_noise = {
                            'energy': float(task_noises_np[0]) if len(task_noises_np) > 0 else None,
                            'force': float(task_noises_np[1]) if len(task_noises_np) > 1 else None
                        }
                    elif hasattr(self.gp2.likelihood.noise, 'noise'):
                        self.gp2_final_noise = float(self.gp2.likelihood.noise.noise.item())
                    elif hasattr(self.gp2.likelihood, 'noise'):
                        self.gp2_final_noise = float(self.gp2.likelihood.noise.item())
                except Exception as e:
                    print(f"Warning: Could not extract GP2 final noise: {e}")
                    pass
            
            # Compute GP2 prediction statistics
            gp2_e_preds = [entry['E_GP2'] for entry in self.table_history if not np.isnan(entry.get('E_GP2', np.nan))]
            gp2_f_preds = [entry['F_GP2'] for entry in self.table_history if not np.isnan(entry.get('F_GP2', np.nan))]
            
            if len(gp2_e_preds) > 1:
                self.gp2_energy_std = np.std(gp2_e_preds)
            if len(gp2_f_preds) > 1:
                self.gp2_force_std = np.std(gp2_f_preds)
        
        # Track average temperature from latest GP1 evaluation
        if hasattr(self.gp1, 'thermo_stats') and self.gp1.thermo_stats:
            latest_temp = self.gp1.thermo_stats.get('avg_temperature', self.temperature)
            self.avg_temperatures.append(latest_temp)

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
