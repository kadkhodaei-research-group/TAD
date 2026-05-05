"""walker_pure_neb.py - Pure NEB Walker with full system forces."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, List, Tuple
import os
import pickle
import time
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from output_manager import get_output_path

logger = logging.getLogger(__name__)


class WalkerPureNEB:
    """Pure NEB method using direct VASP forces for the entire system."""
    
    def __init__(
        self,
        initial_path: npt.NDArray[np.float64],  # Shape: (N_images, N_atoms*3)
        local_pes: Any,  # VASP interface
        max_neb_steps: int = 100,
        k_parallel: float = 1.0,
        k_perpendicular: float = 1.0,
        neb_convergence_threshold: float = 0.1,
        ci_convergence_threshold: float = 0.1,
        ci_activation_threshold: float = 0.0,  # 0 means no climbing image
        translation_method: str = "qmvv",
        step_size: float = 0.01,
        max_step_size: float = 0.2,
        verbose: bool = False,
        checkpoint_interval: int = 1,
        visualize: bool = False,
        keep_only_latest_path: bool = False,
        **kwargs
    ) -> None:
        """Initialize pure NEB walker.
        
        Args:
            initial_path: Initial path positions (N_images x 3N array)
            local_pes: VASP interface for energy/force calculations
            max_neb_steps: Maximum number of NEB iterations
            k_parallel: Parallel spring constant
            k_perpendicular: Perpendicular spring constant
            neb_convergence_threshold: Force convergence threshold for images
            ci_convergence_threshold: Additional convergence threshold for climbing image
            ci_activation_threshold: Threshold to activate climbing image (0 = off)
            translation_method: Method for moving images ('qmvv', 'lbfgs', 'fire')
            step_size: Base step size for translations
            max_step_size: Maximum allowed step size
            verbose: Enable verbose output
            checkpoint_interval: Save checkpoint every N steps
            visualize: Visualize energy along path
            keep_only_latest_path: Keep only the latest NEB path, removing previous ones
        """
        self.initial_path = initial_path.copy()
        self.local_pes = local_pes
        self.max_neb_steps = max_neb_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.step_size = step_size
        self.max_step_size = max_step_size
        self.visualize = visualize
        self.keep_only_latest_path = keep_only_latest_path
        
        # NEB parameters
        self.k_par = k_parallel
        self.k_perp = k_perpendicular
        self.T_MEP = neb_convergence_threshold
        self.T_CI = ci_convergence_threshold
        self.T_CIon = ci_activation_threshold
        self.translation_method = translation_method
        
        # Warn if step size might be too small for typical forces
        if self.verbose and self.step_size < 0.1:
            print(f"\nWARNING: Step size {self.step_size} might be too small for NEB.")
            print(f"         Consider using --step-size 0.1 or larger for faster convergence.")
            print(f"         Current max step size: {self.max_step_size}")
        
        # System size
        self.n_images = len(initial_path)
        self.n_atoms = initial_path.shape[1] // 3
        self.n_dof = initial_path.shape[1]
        
        # Track evaluations
        self.vasp_eval_count = 0
        self.force_evals_per_step = []
        
        # Initialize path
        self.R = initial_path.copy()  # Current path positions
        self.E_R = np.zeros((self.n_images, 1))  # Energies
        self.G_R = np.zeros((self.n_images, self.n_dof))  # Gradients (not forces!)
        
        # Energy reference (set from minimum 1)
        self.energy_reference = None
        self.reference_set = False
        
        # Climbing image
        self.CI_on = 0  # 0 = off, 1 = on
        self.i_CI = -1  # Index of climbing image
        self.R_CIon = None  # Path when CI was turned on
        
        # Translation state
        self.V_old = np.zeros((self.n_images-2, self.n_dof))  # Velocities
        self.F_R_old = np.zeros((self.n_images-2, self.n_dof))  # Previous forces
        self.zeroV = 1  # Use zero velocity initially
        
        # State tracking
        self.steps = 0
        self.converged = False
        self.trajectory = []  # Store (positions, energies, forces) tuples
        self.neb_path_counter = 0  # Track NEB path iterations
        
        # History tracking
        self.E_R_acc = np.ndarray((self.n_images, 0))  # Energy history
        self.normF_R_acc = np.ndarray((self.n_images-2, 0))  # Force norm history
        self.normFCI_acc = np.ndarray((0,))  # CI force norm history
        
        # Table history for verbose output
        self.table_history = []
        
        if self.verbose:
            print("\n" + "="*80)
            print("PURE NEB WALKER INITIALIZED")
            print("="*80)
            print(f"Path with {self.n_images} images")
            print(f"System size: {self.n_atoms} atoms ({self.n_dof} DOF per image)")
            print(f"Spring constants: k_par={k_parallel}, k_perp={k_perpendicular}")
            print(f"Convergence threshold: {neb_convergence_threshold} eV/Å")
            print(f"Climbing image: {'Enabled' if ci_activation_threshold > 0 else 'Disabled'}")
            print(f"Translation method: {translation_method}")
            print("="*80 + "\n")
        
        # Initialize checkpoint manager if available
        self.checkpoint_manager = None
        try:
            from continuation_checkpoint_system import CheckpointManager
            self.checkpoint_manager = CheckpointManager()
            if not hasattr(self.checkpoint_manager, '_save_pure_neb_state'):
                self.checkpoint_manager = None
                if self.verbose:
                    print("CheckpointManager doesn't support WalkerPureNEB - using basic checkpointing")
        except ImportError:
            if self.verbose:
                print("CheckpointManager not available - basic checkpointing will be used")
        except Exception as e:
            self.checkpoint_manager = None
            if self.verbose:
                print(f"CheckpointManager initialization failed: {e} - using basic checkpointing")
    
    def _evaluate_image(self, position: npt.NDArray[np.float64], image_id: Optional[int] = None) -> Tuple[float, npt.NDArray[np.float64]]:
        """Evaluate energy and gradient at a position.
        
        Args:
            position: Atomic positions
            image_id: Optional image ID for NEB directory structure
            
        Returns:
            (energy, gradient)  # Note: gradient = -forces
        """
        # Debug: Print position info
        if self.verbose and image_id is not None:
            pos_hash = hash(position.tobytes()) % 1000000  # Simple hash for tracking
            print(f"    Evaluating image {image_id}: position hash = {pos_hash}")
        
        # Prepare kwargs for NEB-specific parameters
        kwargs = {}
        if image_id is not None:
            kwargs['path_id'] = self.steps
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
        
        self.vasp_eval_count += 1
        
        return energy_ref, gradients
    
    def _force_sNEB(self, R: npt.NDArray, E_R: npt.NDArray, G_R: npt.NDArray, 
                    CI_on: int = 0) -> Tuple[npt.NDArray, float, int]:
        """Calculate stabilized NEB forces on intermediate images.
        
        Based on utils.force_sNEB from reference implementation.
        
        Returns:
            F_R: NEB forces on intermediate images (N_im-2 x D)
            normFCI: Norm of force on climbing image (0 if CI off)
            i_CI: Index of climbing image (-1 if CI off)
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
            
            # Improved tangent estimate (Henkelman & Jónsson 2000)
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
        i_CI = np.argmax(E_R[1:-1, 0]) + 1 if CI_on > 0 else -1
        
        # Calculate NEB forces
        F_R = np.zeros((N_im-2, D))
        normFCI = 0.0
        
        for i in range(1, N_im-1):
            # Get negative gradient (force)
            F_i = -G_R[i, :]
            
            # Parallel component
            F_par = np.dot(F_i, tau[i, :])
            
            if CI_on > 0 and i == i_CI:
                # Climbing image: reverse parallel component
                F_R[i-1, :] = F_i - 2.0 * F_par * tau[i, :]
                normFCI = np.linalg.norm(F_R[i-1, :])
            else:
                # Regular image: remove parallel component, add spring forces
                F_perp = F_i - F_par * tau[i, :]
                
                # Spring forces (parallel component only)
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
        
        return F_R, normFCI, i_CI - 1  # Convert to 0-based index for intermediate images
    
    def _step_translation(self, R: npt.NDArray, F_R: npt.NDArray) -> npt.NDArray:
        """Move images according to NEB forces using specified method.
        
        This follows the reference implementation more closely.
        """
        N_im = self.n_images
        R_new = R.copy()
        
        if self.verbose:
            print(f"\nTranslation step using method: {self.translation_method}")
            print(f"Step size: {self.step_size}, Max step size: {self.max_step_size}")
        
        if self.translation_method == "qmvv":
            # Quick-Min Velocity Verlet (following reference implementation)
            dt = self.step_size
            
            # Update all intermediate images
            for i in range(1, N_im-1):
                idx = i - 1
                F = F_R[idx, :]
                V = self.V_old[idx, :]
                
                # Velocity update (half step if zero velocity)
                if self.zeroV is not None and np.any(self.zeroV):
                    V_new = 0.5 * dt * F
                else:
                    # Full velocity verlet update
                    V_new = V + dt * F
                
                # Quick-Min projection: project velocity along force direction only if positive
                Vdot = np.dot(V_new, F)
                if Vdot > 0:
                    # Project velocity along force direction
                    V_new = (Vdot / np.dot(F, F)) * F
                else:
                    # Reset velocity if moving against force
                    V_new = np.zeros_like(V_new)
                
                # Update position
                R_new[i, :] = R[i, :] + dt * V_new
                
                # Store velocity for next step
                self.V_old[idx, :] = V_new
                
                # Debug output for first few images
                if self.verbose and i <= 3:
                    dr = R_new[i, :] - R[i, :]
                    print(f"  Image {i}: |F|={np.linalg.norm(F):.3f}, |V|={np.linalg.norm(V_new):.3f}, |dr|={np.linalg.norm(dr):.3f}")
        
        elif self.translation_method == "lbfgs":
            # L-BFGS style update (simplified - just steepest descent for now)
            for i in range(1, N_im-1):
                idx = i - 1
                F = F_R[idx, :]
                
                # Simple steepest descent step
                dr = self.step_size * F
                
                # No step limiting for standard methods
                R_new[i, :] = R[i, :] + dr
                
                if self.verbose and i <= 3:
                    print(f"  Image {i}: |F|={np.linalg.norm(F):.3f}, |dr|={np.linalg.norm(dr):.3f}")
        
        elif self.translation_method == "fire":
            # FIRE algorithm
            dt = self.step_size
            f_inc = 1.1  # factor to increase dt
            f_dec = 0.5  # factor to decrease dt  
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
                
                if self.verbose and i <= 3:
                    dr = R_new[i, :] - R[i, :]
                    print(f"  Image {i}: |F|={np.linalg.norm(F):.3f}, |V|={np.linalg.norm(V_new):.3f}, |dr|={np.linalg.norm(dr):.3f}")
        
        else:
            raise ValueError(f"Unknown translation method: {self.translation_method}")
        
        self.F_R_old = F_R.copy()
        self.zeroV = 0
        
        return R_new
    
    def run(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Run pure NEB search until convergence or max steps."""
        # Evaluate endpoints only once at the beginning
        if self.steps == 0:
            if self.verbose:
                print("Evaluating endpoint minima...")
            
            # Check if we can use batch evaluation
            use_batch = (hasattr(self.local_pes, 'prepare_batch_calculations') and 
                        hasattr(self.local_pes.vasp_manager, 'execution_mode') and 
                        self.local_pes.vasp_manager.execution_mode == 'eam' and
                        hasattr(self.local_pes.vasp_manager, 'parallel_eam') and
                        self.local_pes.vasp_manager.parallel_eam)
            
            if use_batch:
                # Evaluate endpoints in batch
                positions_list = [(0, self.R[0, :]), (self.n_images-1, self.R[-1, :])]
                self.local_pes.prepare_batch_calculations(positions_list, self.neb_path_counter)
                results = self.local_pes.wait_for_batch_results()
                
                if 0 in results:
                    E_min1_raw, F_min1 = results[0]  # Results contain (energy, forces)
                    # Set energy reference from first calculation
                    if not self.reference_set:
                        self.energy_reference = E_min1_raw
                        self.reference_set = True
                        if self.verbose:
                            print(f"Energy reference set to: {self.energy_reference:.4f} eV")
                    E_min1 = E_min1_raw - self.energy_reference
                    G_min1 = -F_min1  # Convert forces to gradients
                    self.E_R[0, 0] = E_min1
                    self.G_R[0, :] = G_min1
                else:
                    raise RuntimeError("Failed to evaluate minimum 1")
                    
                if self.n_images-1 in results:
                    E_min2_raw, F_min2 = results[self.n_images-1]  # Results contain (energy, forces)
                    E_min2 = E_min2_raw - self.energy_reference
                    G_min2 = -F_min2  # Convert forces to gradients
                    self.E_R[-1, 0] = E_min2
                    self.G_R[-1, :] = G_min2
                else:
                    raise RuntimeError("Failed to evaluate minimum 2")
            else:
                # Sequential evaluation
                # Minimum 1
                E_min1, G_min1 = self._evaluate_image(self.R[0, :], image_id=0)
                self.E_R[0, 0] = E_min1
                self.G_R[0, :] = G_min1
                
                # Minimum 2
                E_min2, G_min2 = self._evaluate_image(self.R[-1, :], image_id=self.n_images-1)
                self.E_R[-1, 0] = E_min2
                self.G_R[-1, :] = G_min2
            
            if self.verbose:
                print(f"Minimum 1: E = {E_min1:.6f} eV")
                print(f"Minimum 2: E = {E_min2:.6f} eV")
                print(f"Energy difference: {E_min2 - E_min1:.6f} eV\n")
        
        # Visualization setup
        if self.visualize:
            self.fig_energy = plt.figure(figsize=(10, 6))
            plt.ion()
        
        # Main NEB loop
        while self.steps < self.max_neb_steps and not self.converged:
            self.steps += 1
            
            # Clean up old paths BEFORE incrementing counter and creating new paths
            # This ensures old paths are removed before new ones are created
            if self.keep_only_latest_path and self.steps > 1:
                if self.verbose:
                    import time
                    print(f"\n[PRE-CLEANUP] at {time.strftime('%H:%M:%S')}: Step {self.steps}, removing path_{self.neb_path_counter:03d} before creating path_{self.neb_path_counter+1:03d}")
                self._cleanup_old_neb_paths()
            
            self.neb_path_counter += 1  # Increment path counter for NEB directory structure
            eval_start = self.vasp_eval_count
            
            # Evaluate all intermediate images
            if self.verbose:
                print(f"\n--- Step {self.steps} ---")
                print("Evaluating intermediate images...")
            
            # IMPORTANT: Re-evaluate ALL images each step (including endpoints for consistency)
            # Check if we can use batch evaluation for parallel EAM
            use_batch = (hasattr(self.local_pes, 'prepare_batch_calculations') and 
                        hasattr(self.local_pes.vasp_manager, 'execution_mode') and 
                        self.local_pes.vasp_manager.execution_mode == 'eam' and
                        hasattr(self.local_pes.vasp_manager, 'parallel_eam') and
                        self.local_pes.vasp_manager.parallel_eam)
            
            if use_batch:
                if self.verbose:
                    print("  Using batch evaluation for parallel EAM calculations...")
                
                # Prepare all images for batch calculation
                positions_list = [(i, self.R[i, :]) for i in range(self.n_images)]
                
                # No cleanup callback needed
                self.local_pes._cleanup_callback = None
                
                self.local_pes.prepare_batch_calculations(positions_list, self.neb_path_counter)
                
                # NOTE: Cleanup now happens BEFORE path creation (see above)
                # This ensures only the current path exists during calculations
                
                # Wait for all results
                results = self.local_pes.wait_for_batch_results()
                
                # Store results
                for i in range(self.n_images):
                    if i in results:
                        E_i_raw, G_i = results[i]
                        
                        # Set reference on first calculation (from first image)
                        if not self.reference_set and i == 0:
                            self.energy_reference = E_i_raw
                            self.reference_set = True
                            if self.verbose:
                                print(f"Energy reference set to: {self.energy_reference:.4f} eV")
                        
                        # Apply energy reference consistently
                        if self.reference_set and self.energy_reference is not None:
                            E_i_ref = E_i_raw - self.energy_reference
                        else:
                            E_i_ref = E_i_raw
                            
                        self.E_R[i, 0] = E_i_ref
                        self.G_R[i, :] = -G_i  # Convert forces to gradients
                    else:
                        print(f"  Warning: No result for image {i}, using previous values")
            else:
                # Sequential evaluation (original code)
                # NOTE: Cleanup already happened before incrementing counter (see above)
                for i in range(self.n_images):
                    if self.verbose and i == 0:
                        print("  Re-evaluating endpoints for consistency...")
                    E_i, G_i = self._evaluate_image(self.R[i, :], image_id=i)
                    self.E_R[i, 0] = E_i
                    self.G_R[i, :] = G_i
            
            # Calculate NEB forces
            F_R, normFCI, i_CI = self._force_sNEB(self.R, self.E_R, self.G_R, self.CI_on)
            normF_R = np.sqrt(np.sum(F_R**2, axis=1))
            
            # Debug: Print energy information
            if self.verbose:
                print(f"\nEnergies after evaluation:")
                for i in range(self.n_images):
                    print(f"  Image {i}: E = {self.E_R[i, 0]:.6f} eV")
                print(f"\nForce magnitudes on intermediate images:")
                for i in range(self.n_images-2):
                    print(f"  Image {i+1}: |F| = {normF_R[i]:.6f} eV/Å")
            
            # Turn on climbing image if threshold reached
            if self.CI_on == 0 and self.T_CIon > 0 and np.max(normF_R) < self.T_CIon:
                self.CI_on = 1
                self.R_CIon = self.R.copy()
                self.i_CI = np.argmax(self.E_R[1:-1, 0])
                F_R, normFCI, i_CI = self._force_sNEB(self.R, self.E_R, self.G_R, self.CI_on)
                normF_R = np.sqrt(np.sum(F_R**2, axis=1))
                self.zeroV = 1  # Reset velocities
                
                if self.verbose:
                    print(f"\nClimbing image activated on image {self.i_CI + 2} (1-based)")
            
            # Store history
            self.E_R_acc = np.hstack((self.E_R_acc, self.E_R))
            self.normF_R_acc = np.hstack((self.normF_R_acc, normF_R[:, np.newaxis]))
            self.normFCI_acc = np.hstack((self.normFCI_acc, normFCI))
            
            # Track evaluations
            evals_this_step = self.vasp_eval_count - eval_start
            self.force_evals_per_step.append(evals_this_step)
            
            # Store trajectory
            self.trajectory.append((self.R.copy(), self.E_R.copy(), F_R.copy()))
            
            # Verbose output
            if self.verbose:
                max_force = np.max(normF_R)
                mean_energy = np.mean(self.E_R[1:-1, 0])
                max_energy = np.max(self.E_R[1:-1, 0])
                
                row = f"{self.steps:6d} {mean_energy:12.6f} {max_energy:12.6f} {max_force:12.6f} "
                if self.CI_on:
                    row += f"{normFCI:12.6f} {i_CI+2:6d}"
                else:
                    row += f"{'---':>12} {'---':>6}"
                row += f" {self.vasp_eval_count:8d}"
                
                self.table_history.append(row)
                self._print_full_table()
            
            # Visualize if requested
            if self.visualize:
                self._visualize_path()
            
            # Check convergence
            if (self.T_CIon <= 0 or self.CI_on > 0) and np.max(normF_R) < self.T_MEP:
                if self.CI_on == 0 or normFCI < self.T_CI:
                    self.converged = True
                    if self.verbose:
                        print("\n" + "="*80)
                        print(f"{'CONVERGED':^80}")
                        print("="*80)
            
            # Add warning for high forces
            if self.verbose and np.max(normF_R) > 50.0:
                print(f"\nWARNING: Very high forces detected (max = {np.max(normF_R):.1f} eV/Å)")
                print("This suggests:")
                print("  - Atoms may be too close together")
                print("  - Initial path may pass through high-energy regions")
                print("  - Consider using a different initial path or smaller step size")
                
                # Find which image has highest force
                worst_idx = np.argmax(normF_R)
                print(f"  Worst image: {worst_idx + 1} with |F| = {normF_R[worst_idx]:.1f} eV/Å")
            
            # Move images if not converged
            if not self.converged:
                old_R = self.R.copy()
                self.R = self._step_translation(self.R, F_R)
                
                # Debug: Check how much images moved
                if self.verbose:
                    print(f"\nImage displacements after step {self.steps}:")
                    for i in range(1, self.n_images-1):
                        disp = np.linalg.norm(self.R[i, :] - old_R[i, :])
                        print(f"  Image {i}: moved {disp:.6f} Å")
            
            # Save checkpoint
            if self.steps % self.checkpoint_interval == 0:
                self.save_checkpoint()
        
        if not self.converged and self.verbose:
            print("\n" + "="*80)
            print(f"{'MAXIMUM STEPS REACHED':^80}")
            print("="*80)
        
        # Final summary
        if self.verbose:
            self._print_summary()
        
        # Save final checkpoint
        self.save_checkpoint(final=True)
        
        # Close visualization
        if self.visualize:
            plt.ioff()
            plt.show()
        
        # Return final path, energies, and gradients
        return self.R, self.E_R, self.G_R
    
    def _print_full_table(self):
        """Print the full table with header and all history."""
        print("\n" + "="*80)
        print(f"{'Step':>6} {'Mean E (eV)':>12} {'Max E (eV)':>12} {'Max |F|':>12} {'|F_CI|':>12} {'CI Img':>6} {'Evals':>8}")
        print("-" * 80)
        
        # Print all history
        for row in self.table_history:
            print(row)
        
        print("-" * 80)
    
    def _cleanup_old_neb_paths(self):
        """Remove old NEB path directories before creating new ones.
        
        Since this is called BEFORE incrementing neb_path_counter,
        we remove the path with the current counter value (the old one).
        """
        if not self.keep_only_latest_path:
            return
            
            
        if hasattr(self.local_pes, 'vasp_manager'):
            import shutil
            import glob
            import time
            
            # Get the base directory for NEB runs
            base_dir = self.local_pes.vasp_manager.base_dir
            neb_runs_dir = os.path.join(base_dir, 'neb_runs')
            
            if self.verbose:
                print(f"[CLEANUP] Looking for paths in: {neb_runs_dir}")
            
            # Check if the directory exists
            if not os.path.exists(neb_runs_dir):
                if self.verbose:
                    print(f"[CLEANUP] Directory does not exist yet: {neb_runs_dir}")
                return
                
            # The path to remove is the one with the current counter (before increment)
            old_path_name = f"path_{self.neb_path_counter:03d}"
            old_path_full = os.path.join(neb_runs_dir, old_path_name)
            
            if os.path.exists(old_path_full):
                try:
                    shutil.rmtree(old_path_full)
                    if self.verbose:
                        print(f"[PATH CLEANUP] at {time.strftime('%H:%M:%S')}: Removed old path {old_path_name} before creating path_{self.neb_path_counter+1:03d}")
                except Exception as e:
                    if self.verbose:
                        print(f"[Warning: Could not remove old path {old_path_name}: {e}]")
            else:
                if self.verbose:
                    print(f"[CLEANUP] Path {old_path_name} does not exist (may have been already cleaned)")
            
            # Also clean up any other stray paths that shouldn't exist
            # This handles cases where cleanup might have been interrupted
            for item in os.listdir(neb_runs_dir):
                if os.path.isdir(os.path.join(neb_runs_dir, item)) and item.startswith('path_'):
                    try:
                        path_num = int(item.split('_')[-1])
                        # Remove any path that's not the one we're about to create
                        if path_num != self.neb_path_counter + 1:
                            full_path = os.path.join(neb_runs_dir, item)
                            shutil.rmtree(full_path)
                            if self.verbose:
                                print(f"  [Cleaning up stray path: {item}]")
                    except:
                        continue
    
    def _print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total steps: {self.steps}")
        print(f"Total VASP evaluations: {self.vasp_eval_count}")
        print(f"Average VASP evals per step: {np.mean(self.force_evals_per_step):.1f}")
        print(f"Status: {'CONVERGED' if self.converged else 'NOT CONVERGED'}")
        
        # Energy barrier
        E_max = np.max(self.E_R[1:-1, 0])
        E_min1 = self.E_R[0, 0]
        E_min2 = self.E_R[-1, 0]
        
        print(f"\nEnergy barrier:")
        print(f"  Forward: {E_max - E_min1:.6f} eV")
        print(f"  Reverse: {E_max - E_min2:.6f} eV")
        print(f"  Saddle point energy: {E_max:.6f} eV")
        
        if self.CI_on:
            print(f"  Climbing image: {self.i_CI + 2} (1-based indexing)")
        
        # Path length
        path_length = 0.0
        for i in range(1, self.n_images):
            path_length += np.linalg.norm(self.R[i, :] - self.R[i-1, :])
        print(f"\nPath length: {path_length:.4f} Å")
        print("="*80)
    
    def _visualize_path(self):
        """Visualize energy along the path."""
        plt.figure(self.fig_energy.number)
        plt.clf()
        
        # Create normalized reaction coordinate
        distances = np.zeros(self.n_images)
        for i in range(1, self.n_images):
            distances[i] = distances[i-1] + np.linalg.norm(self.R[i, :] - self.R[i-1, :])
        distances /= distances[-1]  # Normalize to [0, 1]
        
        # Plot energy profile
        plt.plot(distances, self.E_R[:, 0], 'bo-', markersize=8, linewidth=2)
        
        # Mark climbing image
        if self.CI_on and self.i_CI >= 0:
            plt.plot(distances[self.i_CI+1], self.E_R[self.i_CI+1, 0], 'r*', markersize=15)
        
        plt.xlabel('Reaction Coordinate')
        plt.ylabel('Energy (eV)')
        plt.title(f'NEB Energy Profile - Step {self.steps}')
        plt.grid(True, alpha=0.3)
        plt.pause(0.1)
    
    def save_checkpoint(self, final=False):
        """Save checkpoint data."""
        checkpoint_data = {
            'walker_type': 'WalkerPureNEB',
            'iteration': self.steps,
            'converged': self.converged,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
            'energy_reference': self.energy_reference,
            'reference_set': self.reference_set,
            'trajectory': self.trajectory,
            'vasp_eval_count': self.vasp_eval_count,
            'force_evals_per_step': self.force_evals_per_step,
            'table_history': self.table_history,
            'neb_state': {
                'R': self.R.copy(),
                'E_R': self.E_R.copy(),
                'G_R': self.G_R.copy(),
                'CI_on': self.CI_on,
                'i_CI': self.i_CI,
                'V_old': self.V_old.copy(),
                'F_R_old': self.F_R_old.copy(),
                'zeroV': self.zeroV,
                'R_CIon': self.R_CIon.copy() if self.R_CIon is not None else None
            },
            'n_images': self.n_images,
            'n_atoms': self.n_atoms,
            'n_dof': self.n_dof,
            'k_par': self.k_par,
            'k_perp': self.k_perp,
            'T_MEP': self.T_MEP,
            'T_CI': self.T_CI,
            'T_CIon': self.T_CIon,
            'E_R_acc': self.E_R_acc,
            'normF_R_acc': self.normF_R_acc,
            'normFCI_acc': self.normFCI_acc,
            'neb_path_counter': self.neb_path_counter
        }
        
        # Use CheckpointManager if available
        if self.checkpoint_manager:
            self.checkpoint_manager.save_walker_state(self, self.steps, 
                                                     {'final_state': final} if final else None)
        else:
            # Fallback to basic checkpointing
            checkpoint_dir = get_output_path('checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if final:
                # Save as final
                filename = os.path.join(checkpoint_dir, 'pure_neb_final.pkl')
                with open(filename, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            # Keep previous checkpoint before saving new one
            latest_filename = os.path.join(checkpoint_dir, 'pure_neb_latest.pkl')
            previous_filename = os.path.join(checkpoint_dir, 'pure_neb_previous.pkl')
            
            # If latest exists, move it to previous (overwriting any existing previous)
            if os.path.exists(latest_filename):
                try:
                    if os.path.exists(previous_filename):
                        os.remove(previous_filename)
                    os.rename(latest_filename, previous_filename)
                    if self.verbose:
                        print(f"  [Moved latest checkpoint to previous]")
                except Exception as e:
                    if self.verbose:
                        print(f"  [Warning: Could not move checkpoint to previous: {e}]")
            
            # Save new latest checkpoint
            with open(latest_filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Clean up old intermediate checkpoint files
            # Remove any pure_neb_checkpoint_*.pkl files
            import glob
            old_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'pure_neb_checkpoint_*.pkl'))
            for old_file in old_checkpoints:
                try:
                    os.remove(old_file)
                    if self.verbose:
                        print(f"  [Removed old checkpoint: {os.path.basename(old_file)}]")
                except Exception as e:
                    if self.verbose:
                        print(f"  [Warning: Could not remove {os.path.basename(old_file)}: {e}]")
        
        if self.verbose and (self.steps % 10 == 0 or final):
            print(f"  [Checkpoint saved at step {self.steps}]")
    
    def load_checkpoint(self, checkpoint_file=None):
        """Load checkpoint and restore state."""
        if checkpoint_file is None:
            checkpoint_file = get_output_path('checkpoints', 'pure_neb_latest.pkl')
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state - handle both old and incremental formats
        self.steps = checkpoint.get('iteration', 0)
        self.converged = checkpoint.get('converged', False)
        
        # These fields might not exist in incremental format
        if 'energy_reference' in checkpoint:
            self.energy_reference = checkpoint['energy_reference']
            self.reference_set = checkpoint.get('reference_set', True)
        else:
            # Try to infer from neb_state
            if 'neb_state' in checkpoint and 'E_R' in checkpoint['neb_state']:
                E_R = checkpoint['neb_state']['E_R']
                if E_R is not None and hasattr(E_R, '__len__') and len(E_R) > 0:
                    # Use minimum as reference
                    import numpy as np
                    self.energy_reference = float(np.min(E_R.flatten()))
                    self.reference_set = True
                else:
                    self.energy_reference = 0.0
                    self.reference_set = False
            else:
                self.energy_reference = 0.0
                self.reference_set = False
        
        # Handle optional fields
        self.trajectory = checkpoint.get('trajectory', [])
        self.vasp_eval_count = checkpoint.get('vasp_eval_count', 0)
        self.force_evals_per_step = checkpoint.get('force_evals_per_step', [])
        self.table_history = checkpoint.get('table_history', [])
        
        # Handle history accumulators - might not exist
        if 'E_R_acc' in checkpoint:
            self.E_R_acc = checkpoint['E_R_acc']
        else:
            import numpy as np
            self.E_R_acc = np.ndarray((self.n_images, 0))
        
        if 'normF_R_acc' in checkpoint:
            self.normF_R_acc = checkpoint['normF_R_acc']
        else:
            import numpy as np
            self.normF_R_acc = np.ndarray((self.n_images-2, 0))
        
        if 'normFCI_acc' in checkpoint:
            self.normFCI_acc = checkpoint['normFCI_acc']
        else:
            import numpy as np
            self.normFCI_acc = np.ndarray((0,))
        
        # Path counter
        self.neb_path_counter = checkpoint.get('neb_path_counter', self.steps)
        
        # Restore NEB state
        neb_state = checkpoint.get('neb_state', {})
        if neb_state:
            self.R = neb_state.get('R', self.R)
            self.E_R = neb_state.get('E_R', self.E_R)
            self.G_R = neb_state.get('G_R', self.G_R)
            self.CI_on = neb_state.get('CI_on', 0)
            self.i_CI = neb_state.get('i_CI', -1)
            
            # These might not exist in incremental format
            if 'V_old' in neb_state:
                self.V_old = neb_state['V_old']
            else:
                import numpy as np
                self.V_old = np.zeros((self.n_images-2, self.n_atoms*3))
            
            if 'F_R_old' in neb_state:
                self.F_R_old = neb_state['F_R_old']
            else:
                if self.G_R is not None:
                    self.F_R_old = -self.G_R.copy()
                else:
                    import numpy as np
                    self.F_R_old = np.zeros((self.n_images, self.n_atoms*3))
            
            if 'zeroV' in neb_state:
                self.zeroV = neb_state['zeroV']
            else:
                import numpy as np
                self.zeroV = np.zeros((self.n_images-2,))
            
            if 'R_CIon' in neb_state:
                self.R_CIon = neb_state['R_CIon']
            else:
                self.R_CIon = self.R.copy() if self.R is not None else None
        
        # Restore history - these are already handled above
        # The following lines are redundant and can cause errors
        # self.E_R_acc = checkpoint['E_R_acc']
        # self.normF_R_acc = checkpoint['normF_R_acc']
        # self.normFCI_acc = checkpoint['normFCI_acc']
        
        # Restore neb_path_counter if it exists (for backward compatibility)
        self.neb_path_counter = checkpoint.get('neb_path_counter', 0)
        
        if self.verbose:
            print(f"\nRestored checkpoint from step {self.steps}")
            print(f"VASP evaluations so far: {self.vasp_eval_count}")
            
            # Print restored table
            if self.table_history:
                self._print_full_table()
        
        return checkpoint