from typing import Optional, List, Tuple, Dict, Any
import numpy.typing as npt
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
from gp_base import GPSurrogateModel
from thermal_sampling_utils import box_muller_transform
from output_manager import get_output_path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_kB_HARTREE = 3.1668114e-6  # Hartree ⋅ K⁻¹  

class GP1(GPSurrogateModel):
    """Gaussian Process model for local PES approximation with thermal sampling.
    
    This class extends GPSurrogateModel to handle thermal sampling and local
    potential energy surface (PES) approximation around specific configurations.
    It includes methods for generating thermalized snapshots and estimating
    local curvature.
    
    Attributes:
        current_location (npt.NDArray[np.float64]): Current configuration
        temperature (float): System temperature in Kelvin
        mass (float): Particle mass
        local_pes: Local potential energy surface object
        thermalized_current_location (Optional[npt.NDArray]): Thermalized config
        thermalized_x_e_f (Optional[List]): Thermalized [position, energy, force]
        path (Optional[str]): Path for saving model data
    """
    
    def __init__(
            self,
            current_location: npt.NDArray[np.float64],
            temperature: float,
            mass: float,
            local_pes: Any,
            path: Optional[str] = None,
            atomic_info: Optional[Dict] = None,
            noise_model: str = "fixed",
            student_t_df: float = 2.0,
            use_adaptive_df: bool = False,
            adaptive_df_start_iter: int = 15,
            adaptive_df_end_iter: int = 25,
            adaptive_df_target: float = 1.0,
            remove_outliers: bool = False,  # If True, detects and warns about outliers but keeps all data
            outlier_threshold: float = 5.0,  # MAD threshold for outlier detection
            use_gpu: bool = False,
        ) -> None:
        """Initialize GP1 with thermal and PES parameters.
        
        Args:
            current_location: Current atomic configuration
            temperature: System temperature in Kelvin
            mass: Particle mass (used for thermal sampling)
            local_pes: Local potential energy surface object
            path: Optional path for saving model data
            atomic_info: Atomic information for GP1
            use_gpu: Use GPU acceleration for training and inference
        """
        super().__init__(atomic_info=atomic_info, use_gpu=use_gpu)  # Initialize the parent class

        self.energy_reference = None  # Will be set by Walker when first energy is calculated

        # self.current_location = current_location
        self.current_location = current_location.reshape(-1) 
        self.temperature = temperature
        self.mass = mass
        self.local_pes = local_pes
        self.path = path

        # Initialize state variables
        self.thermalized_current_location: Optional[npt.NDArray] = None
        self.thermalized_x_e_f: Optional[List] = None
        self.zpm = True

        self.noise_model = noise_model
        self.student_t_df = student_t_df
        self.use_adaptive_df = use_adaptive_df
        self.adaptive_df_start_iter = adaptive_df_start_iter
        self.adaptive_df_end_iter = adaptive_df_end_iter
        self.adaptive_df_target = adaptive_df_target
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.minimum_noise = None  # Will be set if using custom noise models

        # -------------------------------------------------
        # Build per-atom masses in atomic units (self.masses_au)
        # -------------------------------------------------
        mass_arr = np.asarray(mass, dtype=float).ravel()  # whatever length

        # Detect if it's 3N (repeated xyz) and compress
        # n_atoms = len(current_location) // 3
        n_atoms = self.current_location.size // 3  
        # if len(mass_arr) == 3 * n_atoms:
        #     mass_per_atom = mass_arr.reshape(n_atoms, 3)[:, 0]   # take one of each triple
        # else:
        #     mass_per_atom = mass_arr                              # already N-long or scalar

        # Build per-atom masses
        mass_arr = np.asarray(mass, dtype=float).ravel()
        n_atoms = self.current_location.size // 3  # This is TOTAL atoms (215)
        
        # Handle different mass input formats
        if mass_arr.size == 1:
            # Single mass value - apply to all atoms
            self.masses_au_atom = np.full(n_atoms, mass_arr[0] * 1822.888486)
        elif mass_arr.size == n_atoms:
            # One mass per atom
            self.masses_au_atom = mass_arr * 1822.888486
        elif mass_arr.size == 3 * n_atoms:
            # Mass repeated for x,y,z - extract one per atom
            self.masses_au_atom = mass_arr.reshape(n_atoms, 3)[:, 0] * 1822.888486
        else:
            raise ValueError(f"Mass array size {mass_arr.size} doesn't match number of atoms {n_atoms}")
        # Create DOF version for all atoms
        self.masses_au_dof = np.repeat(self.masses_au_atom, 3)  # length 3N (645)
        # # → Store two versions
        # self.masses_au_atom = mass_per_atom * 1822.888486         # length N      (for CoM & K)
        # self.masses_au_dof  = np.repeat(self.masses_au_atom, 3)   # length 3N     (for √M repeats)

    def get_thermalized_current_location(self, num_snapshots: int, thermal_noise: Optional[Tuple[float, float]] = None,
                                eigenvalues_tr: Optional[npt.NDArray] = None,
                                eigenvectors_tr: Optional[npt.NDArray] = None,
                                context: str = 'unspecified', dimer_step: Optional[int] = None, moving_only=True) -> List[npt.NDArray[np.float64]]:
        """Generate and process thermalized snapshots around current location."""
        original_location = self.current_location.copy()
        
        # Start a new thermal batch
        if hasattr(self.local_pes, 'vasp_manager'):
            self.local_pes.vasp_manager.start_new_thermal_batch(reference_positions=self.current_location)
        
        # Generate thermal snapshots - FULL SYSTEM
        snapshots = self.generate_thermal_snapshot(
            num_snapshots=num_snapshots,
            eigenvalues_tr=eigenvalues_tr,
            eigenvectors_tr=eigenvectors_tr,
            context=context,
            dimer_step=dimer_step
        )

        # Ensure frozen atoms haven't moved
        if moving_only and hasattr(self, 'atomic_info'):
            original_pos_3d = self.current_location.reshape(-1, 3)
            moving_indices = self.atomic_info['moving_indices']
            for i in range(len(snapshots)):
                snap_3d = snapshots[i].reshape(-1, 3)
                # Reset frozen atom positions
                for atom_idx in range(len(snap_3d)):
                    if atom_idx not in moving_indices:
                        snap_3d[atom_idx] = original_pos_3d[atom_idx]
                snapshots[i] = snap_3d.flatten()
        
        # Get energies and forces from VASP with retry mechanism
        max_retries = 3
        retry_count = 0
        all_energies = []
        all_forces = []
        remaining_snapshots = list(enumerate(snapshots))
        
        while remaining_snapshots and retry_count < max_retries:
            if retry_count > 0:
                print(f"  Retry {retry_count}: Attempting to evaluate {len(remaining_snapshots)} failed snapshots")
            
            # Queue remaining snapshots
            for idx, snapshot in remaining_snapshots:
                self.local_pes.enqueue_snapshot(snapshot)  # Full system positions
            
            # Harvest results
            energies, forces = self.local_pes.harvest_pending_snapshots()
            
            # DEBUG: Check what energies we're getting
            if context == 'reference_establishment' and retry_count == 0:
                print(f"DEBUG GP1 REF: Harvested {len(energies)} energies from local_pes")
                if len(energies) > 0:
                    print(f"DEBUG GP1 REF: Raw energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}] eV")
                    print(f"DEBUG GP1 REF: Raw mean energy: {np.mean(energies):.4f} eV")
            
            if len(energies) == len(remaining_snapshots):
                # All snapshots evaluated successfully
                for i, (idx, _) in enumerate(remaining_snapshots):
                    all_energies.append((idx, energies[i]))
                    all_forces.append((idx, forces[i]))
                remaining_snapshots = []
            else:
                # Some snapshots failed
                n_failed = len(remaining_snapshots) - len(energies)
                print(f"  WARNING: {n_failed} snapshots failed to evaluate")
                
                # Add successful evaluations
                for i in range(len(energies)):
                    idx, _ = remaining_snapshots[i]
                    all_energies.append((idx, energies[i]))
                    all_forces.append((idx, forces[i]))
                
                # Keep only failed snapshots for retry
                remaining_snapshots = remaining_snapshots[len(energies):]
                retry_count += 1
        
        # Check if all snapshots were successfully evaluated
        if len(all_energies) != len(snapshots):
            n_success = len(all_energies)
            n_failed = len(snapshots) - n_success
            error_msg = (f"Failed to evaluate all requested snapshots after {max_retries} retries.\n"
                        f"  Requested: {len(snapshots)} snapshots\n"
                        f"  Successfully evaluated: {n_success} snapshots\n"
                        f"  Failed: {n_failed} snapshots")
            print(f"  ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        # Sort by original index to maintain order
        all_energies.sort(key=lambda x: x[0])
        all_forces.sort(key=lambda x: x[0])
        
        # Extract sorted values
        energies = np.array([e for _, e in all_energies])
        forces = [f for _, f in all_forces]
        
        # CRITICAL: Extract forces for moving atoms only
        moving_indices = self.atomic_info["moving_indices"]
        forces_moving = []
        for force_full in forces:
            force_3d = force_full.reshape(-1, 3)
            force_moving = []
            for idx in moving_indices:
                force_moving.extend(force_3d[idx])
            forces_moving.append(force_moving)
        forces = np.array(forces_moving)
        
        # Extract moving atom positions from full snapshots for GP training
        snapshots_moving = []
        for snapshot in snapshots:
            pos_3d = snapshot.reshape(-1, 3)
            moving_pos = pos_3d[self.atomic_info["moving_indices"]].flatten()
            snapshots_moving.append(moving_pos)
        snapshots_moving = np.array(snapshots_moving)




        # Handle outlier detection if enabled (but don't remove data)
        if self.remove_outliers:
            # Outlier detection based on MAD
            e_median = np.median(energies)
            e_mad = np.median(np.abs(energies - e_median))
            
            if e_mad > 0:  # Avoid division by zero
                outlier_mask = np.abs(energies - e_median) > self.outlier_threshold * e_mad
                
                if np.any(outlier_mask):
                    n_outliers = np.sum(outlier_mask)
                    outlier_indices = np.where(outlier_mask)[0]
                    print(f"  WARNING: Detected {n_outliers} potential outliers in thermal snapshots (threshold: {self.outlier_threshold} MAD)")
                    print(f"  Outlier indices: {outlier_indices.tolist()}")
                    print(f"  Outlier energies: {energies[outlier_mask].tolist()}")
                    print(f"  Energy range: [{energies.min():.6f}, {energies.max():.6f}] eV")
                    print(f"  Median: {e_median:.6f} eV, MAD: {e_mad:.6f} eV")
                    print(f"  NOTE: Keeping all snapshots as requested (no data removed)")



        
        # Handle energy referencing
        energies_for_training = energies.copy()
        if hasattr(self, 'energy_reference') and self.energy_reference is not None:
            energies_for_training = energies_for_training - self.energy_reference

        # Train GP1 on moving-atom-only data
        dataset = [snapshots_moving, energies_for_training, forces]
        
        # SPECIAL CASE: For reference establishment, return raw energy average
        if context == 'reference_establishment':
            raw_energy_average = np.mean(energies)  # Raw unreferenced energy
            print(f"REFERENCE ESTABLISHMENT: Raw energy average = {raw_energy_average:.4f} eV")
            print(f"  Energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}] eV")
            print(f"  Number of snapshots: {len(energies)}")
            print(f"  GP1 energy_reference at this point: {getattr(self, 'energy_reference', 'None')}")
            # Return raw average instead of GP prediction
            return [
                self.current_location,  # Position
                raw_energy_average,     # Raw energy (not GP prediction)
                np.mean(forces, axis=0) # Average force
            ]

        #  ADD DIAGNOSTIC ANALYSIS HERE 
        self._analyze_thermal_distribution(
            energies=energies_for_training,
            forces=forces,
            positions=snapshots_moving,
            context=context,
            dimer_step=dimer_step,
            num_snapshots=num_snapshots
        )
        
        self.train(
            training_data=dataset,
            thermal_noise=thermal_noise,
            path=self.path, 
            model_name="GP1",
            minimum_noise=self.minimum_noise
        )
        
        # Predict on moving atom positions
        thermalized_y_g = self.predict(snapshots_moving)
        
        # Get thermalized location - but this should be FULL SYSTEM
        self.thermalized_current_location = self.averaged_thermalized_snapshot(snapshots, thermalized_y_g)
        
        # Get prediction for thermalized location
        thermalized_pos_3d = self.thermalized_current_location.reshape(-1, 3)
        thermalized_moving = thermalized_pos_3d[self.atomic_info["moving_indices"]].flatten()
        
        # Get prediction for thermalized location
        energy_current_location, force_current_location, _, _ = self.predict(thermalized_moving.reshape(1, -1))

        # FIXED: Do NOT add back the reference here!
        # The walker's _evaluate_position expects referenced energies and will handle
        # any unreferencing needed for display purposes.
        # Previous code was adding back reference here, causing double application.
        #
        # if hasattr(self, 'energy_reference') and self.energy_reference is not None:
        #     energy_current_location = energy_current_location + self.energy_reference
        
        # Extract scalar energy
        if hasattr(energy_current_location, 'flatten'):
            energy_current_location = energy_current_location.flatten()[0]

        
        # The returned force is for moving atoms only
        if hasattr(force_current_location, 'flatten'):
            force_current_location = force_current_location.flatten()
        
        # Extract scalar energy
        if hasattr(energy_current_location, 'flatten'):
            energy_current_location = energy_current_location.flatten()[0]
        
        self.thermalized_x_e_f = [
            self.thermalized_current_location,  # Full system position
            energy_current_location,             # Energy
            force_current_location               # Force for moving atoms only
        ]
        
        self.current_location = original_location
        return self.thermalized_x_e_f

    def averaged_thermalized_snapshot(
        self,
        snapshots: npt.NDArray[np.float64],
        thermalized_y_g: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Calculate weighted average position from thermalized snapshots.
        
        Args:
            snapshots: Array of snapshot configurations
            thermalized_y_g: Tuple containing:
                - predicted energies
                - predicted forces  
                - energy variances
                - force variances
                
        Returns:
            npt.NDArray[np.float64]: Weighted average position
            
        Notes:
            Snapshots are weighted by inverse of their energy prediction variance,
            giving more importance to predictions with higher confidence.
        """
        predicted_energies, predicted_forces, energy_variances, force_variances = thermalized_y_g
        
        # Ensure weights are valid
        weights = energy_variances.flatten()  # Flatten to 1D
        
        # Avoid division by zero and negative weights
        weights = np.maximum(weights, 1e-8)
        weights = 1.0 / weights  # Inverse variance weighting
        
        # Check shapes match
        if len(weights) != len(snapshots):
            raise ValueError(f"Weights length {len(weights)} doesn't match snapshots length {len(snapshots)}")
        
        # Calculate weighted average along axis 0 (average across snapshots)
        try:
            result = np.average(snapshots, axis=0, weights=weights)
            return result
        except Exception as e:
            print(f"  Error in np.average: {e}")
            # Fallback to simple average
            print("  Using simple average as fallback")
            return np.mean(snapshots, axis=0)


    def generate_thermal_snapshot(
            self,
            num_snapshots: int, # take it from the arugment instead of hard coding 
            eigenvalues_tr: Optional[npt.NDArray[np.float64]] = None,
            eigenvectors_tr: Optional[npt.NDArray[np.float64]] = None,
            context: str = "unspecified",
            dimer_step: Optional[int] = None, # we need to remove this in future
            filename: str = 'thermal_snapshots.pkl'
        ) -> npt.NDArray[np.float64]:
        """Generate thermalized configurations with correct units."""
        # Physical constants
        hbar = 1.0                   # atomic units
        k_B = 3.1668114e-6           # Hartree/K
        n_atoms = self.current_location.size // 3
        n_dof   = self.current_location.size
        # precompute Bohr eigenvectors & frequencies
        eigvecs_bohr = eigenvectors_tr       # Bohr
        omega_s      = eigenvalues_tr        # a.u. time^-1
        bohr_to_A    = 0.529177210903
        tau_au_ps    = 2.4188843265857e-5    # a.u. time in ps
        # --- inside generate_thermal_snapshot ---
        thermal_snapshot = np.zeros((num_snapshots, n_dof))
        snapshot_vel     = np.zeros((num_snapshots, n_dof))

        thermal_configs = [] # List to store (u, v) tuples

        for i in range(num_snapshots):
            disp_bohr = np.zeros(n_dof)
            vel_bohr_au = np.zeros(n_dof)

            for s in range(len(omega_s)):
                mode_vec_s = eigvecs_bohr[:, s]   # Epsilon_s (non-mass-weighted), ensure it's in Bohr! (3N vector)
                xi1, xi2 = box_muller_transform()  # Standard normals
                omega_val = omega_s[s]

                if self.zpm:
                    z = hbar * omega_val / (k_B * self.temperature)
                    n_s = 0.0 if z > 50 else 1.0 / (np.exp(z) - 1.0)
                    # This A_prime_s is the mass-independent part of the amplitude scaling factor
                    A_prime_s = np.sqrt(hbar * (2 * n_s + 1) / (2 * omega_val))  # Units: Bohr
                else:
                    A_prime_s = np.sqrt(k_B * self.temperature) / omega_val # Units: Bohr * sqrt(a.u. mass / energy_unit_of_omega^2) if omega is not energy
                                                                        # If omega_s is energy (Hartree/hbar), then A_prime_s is sqrt(Hartree / (Hartree)^2) = 1/sqrt(Hartree)
                                                                        # Let's re-check TDEP's classical A_is: (1/omega_s) * sqrt(kBT/m_i)
                                                                        # So A_prime_s here should be (1/omega_s) * sqrt(kBT)
                    A_prime_s = (1.0 / omega_val) * np.sqrt(k_B * self.temperature) # Units will be Bohr * sqrt(a.u. mass)

                # Apply the mass scaling to the contribution of this mode
                # self.sqrt_inv_masses_dof should be (3N,) array: [1/sqrt(m1), 1/sqrt(m1), 1/sqrt(m1), 1/sqrt(m2), ...]
                # Make sure masses_au_atom is shape (N,)
                # In generate_thermal_snapshot
                if not hasattr(self, 'sqrt_inv_masses_dof_scaled'):
                    # This should use ALL atoms, not just moving atoms
                    self.sqrt_inv_masses_dof_scaled = np.repeat(1.0 / np.sqrt(self.masses_au_atom), 3)
                    
                # Verify the shape
                assert self.sqrt_inv_masses_dof_scaled.shape[0] == n_dof, \
                    f"Mass array shape mismatch: {self.sqrt_inv_masses_dof_scaled.shape[0]} vs {n_dof}"

                disp_contribution = self.sqrt_inv_masses_dof_scaled * (A_prime_s * xi1 * mode_vec_s)
                vel_contribution  = self.sqrt_inv_masses_dof_scaled * (-A_prime_s * xi2 * mode_vec_s * omega_val)
                
                disp_bohr   += disp_contribution
                vel_bohr_au += vel_contribution


            # COM removal & rescale (all in a.u.)
            u_cart, v_cart = self._remove_com(
                disp_bohr.reshape(n_atoms,3),
                vel_bohr_au.reshape(n_atoms,3),
                self.masses_au_atom
            )
            # u_cart, v_cart = self._exact_temperature_rescale(
            #     u_cart, v_cart,
            #     self.masses_au_atom, omega_s, eigvecs_bohr, self.temperature
            # )

            # convert to physical units for output
            disp_Å   = u_cart * bohr_to_A             # Å
            vel_Å_ps = v_cart * bohr_to_A / tau_au_ps  # Å/ps

            thermal_snapshot[i] = self.current_location + disp_Å.reshape(-1)
            snapshot_vel[i]     = vel_Å_ps.reshape(-1)
            
            # Transpose to match (3, n_atoms) shape expected by tabulate_thermo
            thermal_configs.append((
                disp_Å.T.copy(),  # Shape (3, n_atoms) displacements in Bohr
                vel_Å_ps.T.copy()  # Shape (3, n_atoms) velocities in Hartree
            ))

        # save snapshots and (optionally) velocities
        self._save_snapshot_velocity(snapshot_vel, filename+"_vel", context, dimer_step)
        thermo_stats = self._tabulate_thermo(
                configurations=thermal_configs,  # List of (u, v) tuples
                eigvecs_bohr=eigvecs_bohr,
                omega_s=omega_s,
                nconf=num_snapshots
            )
        self._save_snapshot_history(thermal_snapshot, filename, context, dimer_step, temperature_stats=thermo_stats)
        # Log the average temperature
        avg_temp = thermo_stats['avg_temperature']
        print(f"\nAverage temperature for {context} (step {dimer_step}): {avg_temp:.1f} K")
        print(f"Target temperature: {self.temperature:.1f} K")
        print(f"Deviation: {avg_temp - self.temperature:.1f} K ({(avg_temp/self.temperature - 1)*100:.1f}%)\n")
        
        # Store thermo_stats as instance variable for later access
        self.thermo_stats = thermo_stats
    
        return thermal_snapshot

    def _remove_com(self, u: np.ndarray, v: np.ndarray, masses: np.ndarray):
        """
        Remove centre-of-mass displacement and momentum.
        u, v : (N,3) arrays in Å and Å/au
        masses : (N,) array in a.u.
        """
        Mtot = masses.sum()
        com_shift = (u * masses[:, None]).sum(axis=0) / Mtot
        com_momentum = (v * masses[:, None]).sum(axis=0) / Mtot
        return u - com_shift, v - com_momentum


    def _exact_temperature_rescale(
        self,
        u: np.ndarray,
        v: np.ndarray,
        masses: np.ndarray,
        omegas: np.ndarray,          # shape (3N,) in Hartree
        eigvecs: np.ndarray,         # shape (3N,3N) – mass-weighted, the ones you already computed
        T_target: float,
    ):
        """
        Scale u and v so that (U+K)/(3 kB (N-1)) = T_target exactly.
        Implements the same λ = √(T_target/T_now) trick used in TDEP.
        """
        # --- potential energy (harmonic) ---
        # mass-weight Cartesian displacements to Q-space
        N = len(masses)
        u_mw = (u.T * np.sqrt(masses)).T       # (N,3)
        Q = eigvecs.T @ u_mw.reshape(-1)       # (3N,)
        U = 0.5 * np.sum(omegas**2 * Q**2)

        # --- kinetic energy ---
        K = 0.5 * np.sum(masses * (v**2).sum(axis=1))

        T_now = (U + K) / (3 * _kB_HARTREE * (N - 1))
        lam = np.sqrt(T_target / T_now) # λ = √(T_target/T_now) or the scaling factor

        return u * lam, v * lam
    
    def _save_snapshot_history(
            self,
            thermal_snapshot: npt.NDArray[np.float64],
            filename: str,
            context: str = "unspecified",
            dimer_step: Optional[int] = None,
            temperature_stats: Optional[Dict] = None 
        ) -> None:
        """Save thermal snapshots history with metadata and bundle grouping."""
        snapshot_data = {
            'snapshots': thermal_snapshot,
            'context': context,
            'dimer_step': dimer_step,
            'temperature': self.temperature,
            'timestamp': time.time(),
            'is_initial_point': dimer_step == 0 and context == 'dimer_step',
            'temperature_stats': temperature_stats
        }

        try:
            output_file = get_output_path(filename)
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    existing_snapshots = pickle.load(f)
            else:
                existing_snapshots = []
                
            existing_snapshots.append(snapshot_data)
                
            output_file = get_output_path(filename)
            with open(output_file, 'wb') as f:
                pickle.dump(existing_snapshots, f)
                
        except Exception as e:
            print(f"Warning: Failed to save snapshot history: {str(e)}")

    def _save_snapshot_velocity(
            self,
            velocities: npt.NDArray[np.float64],
            filename: str,
            context: str,
            dimer_step: Optional[int] = None
        ) -> None:
        """Save the velocity part of each thermal snapshot."""
        vel_data = {
            'velocities': velocities,
            'context': context,
            'dimer_step': dimer_step,
            'temperature': self.temperature,
            'timestamp': time.time(),
        }
        try:
            output_file = get_output_path(filename)
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    existing = pickle.load(f)
            else:
                existing = []
            existing.append(vel_data)
            with open(output_file, 'wb') as f:
                pickle.dump(existing, f)
        except Exception as e:
            print(f"Warning: failed to save snapshot velocities: {e}")

    def get_curvature(
            self, 
            tol: float = 1e-6
        ) -> Tuple[str, float, npt.NDArray[np.float64]]:
        """Calculate curvature type and value at current location.
        
        Calculates curvature by:
        1. Computing Hessian matrix at current location
        2. Finding eigenvalues of the Hessian
        3. Determining curvature type based on eigenvalue signs
        
        Args:
            tol: Tolerance for considering eigenvalues as zero
            
        Returns:
            Tuple containing:
                - curvature_type: "positive", "negative", or "mixed"
                - curvature_value: Sum of eigenvalues
                - eigenvalues: Array of Hessian eigenvalues
        """
        eigenvalues = np.linalg.eigvals(self.calculate_hessian(self.current_location))
        
        # Determine curvature type based on eigenvalue signs
        positive = np.all(eigenvalues > -tol)
        negative = np.all(eigenvalues < tol)
        
        # Calculate total curvature
        curvature_value = np.sum(eigenvalues)
        
        # Determine curvature type
        if positive and not negative:
            curvature_type = "positive"
        elif negative and not positive:
            curvature_type = "negative"
        else:
            curvature_type = "mixed"
            
        return curvature_type, curvature_value, eigenvalues

    def save_metadata(
            self, 
            polynomial_fit: Optional[Any] = None,
            save_path: str = 'gp1_curvature.pkl'
        ) -> None:
        """Save GP1 curvature metadata for analysis.
        
        Saves curvature information including type, value, and eigenvalues
        for later analysis and visualization.
        
        Args:
            polynomial_fit: Optional polynomial fit model for curvature calculation
            save_path: Path to save metadata
            
        Notes:
            Data is appended to existing file if it exists.
            Each entry contains curvature type, value, and eigenvalues.
        """
        # Calculate curvature using appropriate method
        if polynomial_fit:
            gp1_curvature_type, gp1_curvature_value, gp1_eigenvalues, _ = polynomial_fit.get_curvature(self.current_location)
        else:
            gp1_curvature_type, gp1_curvature_value, gp1_eigenvalues = self.get_curvature()

        # Prepare metadata entry
        new_entry = {
            "gp1_curvature_type": gp1_curvature_type,
            "gp1_curvature_value": gp1_curvature_value,
            "gp1_eigenvalues": gp1_eigenvalues,
        }

        try:
            # Load existing data if available
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    gp1_curvature = pickle.load(f)
            else:
                gp1_curvature = []

            # Append new data and save
            gp1_curvature.append(new_entry)
            with open(save_path, 'wb') as f:
                pickle.dump(gp1_curvature, f)
                
        except Exception as e:
            print(f"Warning: Failed to save GP1 metadata: {str(e)}")

    def _tabulate_thermo(self, configurations, eigvecs_bohr, omega_s, nconf):
        """
        Tabulate thermodynamic quantities using eigenvalues/eigenvectors for ep calculation.
        
        Args:
            configurations (list): List of (u, v) tuples (displacements in Bohr, velocities in Hartree units).
            eigvecs_bohr (np.ndarray): Eigenvectors (3N x 3N matrix, normalized).
            omega_s (np.ndarray): Phonon frequencies in Hartree units (square roots of dynamical matrix eigenvalues).
            lo_kb_hartree (float): Boltzmann constant in Hartree/K.
            lo_bohr_to_A (float): Bohr to Angstrom conversion factor.
            temperature (float): Target temperature in K.
            nconf (int): Number of configurations.
        """
        # --- Constants ---
        lo_kb_hartree=8.6173303E-5*1.0/27.21138602
        lo_bohr_to_A=0.529177
        # Initialize running averages
        avgtemp = 0.0
        avgmsd = 0.0  # RMSD in Å (matches Fortran's norm2 approach)
        rek = 0.0
        rep = 0.0

        _BOHR_PER_A          = 1.8897259886          # 1 Å  → 1.8897259886 Bohr
        _VEL_AAps_TO_BOHRat  = 4.569333885e-5        # 1 Å/ps → 4.5693×10⁻5 Bohr / atomic‑time

        # Store temperature data for each configuration
        temperature_data = []

        print("  no.   +/-      ek(K)          ep(K)          <ek/ep>         T(K)          <T>(K)      <msd>(A)")

        for iconf in range(1, nconf + 1):
            u, v = configurations[iconf - 1]
            na = u.shape[1]  # Number of atoms

            # --- convert snapshots: Å → Bohr   and   Å/ps → Bohr/au_time
            u = u * _BOHR_PER_A
            v = v * _VEL_AAps_TO_BOHRat

            # --- Kinetic Energy (unchanged) ---
            ek = 0.5 * np.sum(self.masses_au_atom * np.sum(v**2, axis=0)) / na
            rek += ek

            # --- Potential Energy via Normal Modes ---
            # Flatten displacement vector (3N components)
            u_mass_weighted = u.reshape(-1, order='F') * np.sqrt(self.masses_au_dof)  # Apply mass-weighting
            Q = eigvecs_bohr.T @ u_mass_weighted



            # Compute potential energy (Hartree/atom)
            ep = 0.5 * np.sum(omega_s**2 * Q**2) / na
            rep += ep

            # --- Temperature Calculations ---
            ek_K   = 2.0 * ek / (3.0 * lo_kb_hartree)      # TDEP ek(K)
            ep_K   = 2.0 * ep / (3.0 * lo_kb_hartree)      # TDEP ep(K)
            temp_K = 0.5 * (ek_K + ep_K)   # TDEP T(K)
            temp = temp_K
            ek= ek_K
            ep= ep_K

            avgtemp = (avgtemp * (iconf - 1) + temp) / iconf
            ratio = rek / rep if rep != 0 else 0

            # --- RMS Displacement ---
            rmsd_vector = np.sqrt(np.mean(np.sum(u**2, axis=0))) * lo_bohr_to_A   # √<u²>  in Å
            avgmsd = (avgmsd * (iconf - 1) + rmsd_vector) / iconf
            # --- Print results ---
            line = (f"{iconf:4d}      + "
                    f"{ek_K:13.5f}  "
                    f"{ep_K:13.5f}  "
                    f"{ratio:13.5f}  "
                    f"{temp:10.1f}  "
                    f"{avgtemp:10.1f}  "
                    f"{avgmsd:10.4f}")
            print(line)

            # Store this configuration's temperature
            temperature_data.append({
                'config': iconf,
                'ek': ek,
                'ep': ep,
                'temp': temp,
                'avgtemp_so_far': avgtemp
            })

        return {
            'avg_temperature': avgtemp,
            'avg_rmsd': avgmsd,
            'final_ek': ek,
            'final_ep': ep,
            'temperature_history': temperature_data
        }
    
    def _analyze_thermal_distribution(self, energies, forces, positions, context, dimer_step, num_snapshots):
        """Analyze the distribution of thermal snapshots to diagnose noise characteristics."""
        from scipy import stats
        import json
        import os
        from scipy import stats
        import matplotlib.pyplot as plt
        
        # 1. Energy distribution analysis
        e_mean = np.mean(energies)
        e_std = np.std(energies)
        e_median = np.median(energies)
        e_mad = np.median(np.abs(energies - e_median))

        # 2. Force distribution analysis
        force_mags = np.linalg.norm(forces, axis=1)
        f_mean = np.mean(force_mags)
        f_std = np.std(force_mags)
        f_median = np.median(force_mags)
        f_mad = np.median(np.abs(force_mags - f_median))

        # 3. Outlier detection
        e_zscore = np.abs((energies - e_mean) / e_std)
        f_zscore = np.abs((force_mags - f_mean) / f_std)
        e_outliers_z = np.sum(e_zscore > 3)
        f_outliers_z = np.sum(f_zscore > 3)

        e_q1, e_q3 = np.percentile(energies, [25, 75])
        e_iqr = e_q3 - e_q1
        e_outliers_iqr = np.sum((energies < e_q1 - 1.5*e_iqr) | (energies > e_q3 + 1.5*e_iqr))

        # 4. Kurtosis (tail heaviness)
        e_kurtosis = stats.kurtosis(energies)
        f_kurtosis = stats.kurtosis(force_mags)

        # 5. Suggest Student-t df based on analysis
        suggested_df = self._suggest_student_t_df(e_std/e_mad, e_outliers_z/num_snapshots, e_kurtosis)
        
        # 6. Save diagnostic data
        diagnostic_data = {
            'dimer_step': int(dimer_step) if dimer_step is not None else -1,  # Use -1 for None
            'context': context,
            'iteration': getattr(self, 'current_iteration', 'unknown'),  # Add iteration info
            'energy_stats': {
                'mean': float(e_mean), 'std': float(e_std), 'median': float(e_median), 'mad': float(e_mad),
                'outlier_fraction': float(e_outliers_z/num_snapshots),
                'kurtosis': float(e_kurtosis)
            },
            'force_stats': {
                'mean': float(f_mean), 'std': float(f_std), 'median': float(f_median), 'mad': float(f_mad),
                'outlier_fraction': float(f_outliers_z/num_snapshots),
                'kurtosis': float(f_kurtosis)
            },
            'suggested_df': float(suggested_df),
            'num_snapshots': int(num_snapshots),
            'timestamp': time.time()
        }
        
        # Save to JSON file
        diagnostic_file = get_output_path('thermal_diagnostics.json')
        
        # Load existing data if file exists
        if os.path.exists(diagnostic_file):
            try:
                with open(diagnostic_file, 'r') as f:
                    all_diagnostics = json.load(f)
            except:
                all_diagnostics = []
        else:
            all_diagnostics = []
        
        # Append new data
        all_diagnostics.append(diagnostic_data)
        
        # Save updated data
        with open(diagnostic_file, 'w') as f:
            json.dump(all_diagnostics, f, indent=2)
        

    def _suggest_student_t_df(self, std_mad_ratio, outlier_fraction, kurtosis):
        """Suggest appropriate Student-t degrees of freedom based on distribution characteristics."""
        # Simple heuristic based on multiple indicators
        score = 0
        
        # Std/MAD ratio (1.48 for normal)
        if std_mad_ratio > 2.0:
            score += 3
        elif std_mad_ratio > 1.7:
            score += 2
        elif std_mad_ratio > 1.5:
            score += 1
        
        # Outlier fraction
        if outlier_fraction > 0.10:  # >10% outliers
            score += 3
        elif outlier_fraction > 0.05:  # >5% outliers
            score += 2
        elif outlier_fraction > 0.02:  # >2% outliers
            score += 1
        
        # Kurtosis
        if kurtosis > 3:
            score += 3
        elif kurtosis > 1:
            score += 2
        elif kurtosis > 0:
            score += 1
        
        # Map score to df
        if score >= 7:
            return 1.0  # Very heavy tails
        elif score >= 5:
            return 1.5
        elif score >= 3:
            return 2.0
        elif score >= 1:
            return 3.0
        else:
            return 5.0  # Nearly normal