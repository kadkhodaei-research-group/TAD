#!/usr/bin/env python
"""
Walker class for GP1 path analysis.
Handles thermal sampling, GP1 fitting, and statistical analysis.
"""

import os
import logging
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation
from scipy.linalg import eigh
import copy
import numpy.typing as npt

from gp1_model import GP1
from atomic_structure import AtomicStructure
from trgp1_model import ThermodynamicResidenceGP1
from phonon_unfolding import read_phonopy
from pymatgen.io.vasp import Poscar
from output_manager import get_output_path, get_input_path


class WalkerGP1Path:
    """Walker for GP1 analysis along interpolated path."""
    
    def __init__(
        self,
        path_positions: List[np.ndarray],
        local_pes: Any,  # VASP interface
        force_constants_file: str,
        POSCAR_file: str,
        temperature: float = 300.0,
        mass: Union[float, npt.NDArray[np.float64]] = 1.0,
        num_snapshots: int = 10,
        # GP1 noise modeling options
        gp1_noise_model: str = 'fixed',
        gp1_student_t_df: float = 2.0,
        gp1_use_adaptive_df: bool = False,
        gp1_adaptive_df_start_iter: int = 15,
        gp1_adaptive_df_end_iter: int = 25,
        gp1_adaptive_df_target: float = 1.0,
        gp1_remove_outliers: bool = False,
        gp1_outlier_threshold: float = 5.0,
        # Other parameters
        model_type: str = "MultitaskGPModel_rbf_atomic",
        energy_shift: float = 0.0,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize GP1 path walker.
        
        Args:
            path_positions: List of position arrays for each image
            local_pes: VASP interface for calculations
            force_constants_file: Path to force constants file
            POSCAR_file: Path to POSCAR file
            temperature: Temperature in K
            mass: Atomic mass (scalar or array)
            num_snapshots: Number of thermal snapshots per image
            gp1_noise_model: GP1 noise model ('fixed', 'heteroscedastic', 'student_t')
            gp1_student_t_df: Degrees of freedom for Student-t likelihood
            gp1_use_adaptive_df: Enable adaptive Student-t df
            gp1_adaptive_df_start_iter: Iteration to start df adaptation
            gp1_adaptive_df_end_iter: Iteration to reach target df
            gp1_adaptive_df_target: Target df value for adaptation
            gp1_remove_outliers: Enable outlier removal in thermal snapshots
            gp1_outlier_threshold: MAD threshold for outlier detection
            model_type: GP model type to use
            energy_shift: Energy shift/reference to apply
            verbose: Enable verbose output
        """
        self.path_positions = path_positions
        self.local_pes = local_pes
        self.temperature = temperature
        self.mass = mass
        self.num_snapshots = num_snapshots
        self.force_constants_file = force_constants_file
        self.POSCAR_file = POSCAR_file
        self.gp1_noise_model = gp1_noise_model
        self.gp1_student_t_df = gp1_student_t_df
        self.gp1_use_adaptive_df = gp1_use_adaptive_df
        self.gp1_adaptive_df_start_iter = gp1_adaptive_df_start_iter
        self.gp1_adaptive_df_end_iter = gp1_adaptive_df_end_iter
        self.gp1_adaptive_df_target = gp1_adaptive_df_target
        self.gp1_remove_outliers = gp1_remove_outliers
        self.gp1_outlier_threshold = gp1_outlier_threshold
        self.model_type = model_type
        self.energy_shift = energy_shift
        self.verbose = verbose
        
        # Energy reference (will be set from initial GP1 evaluation)
        self.energy_reference = None
        self.reference_set = False
        
        # Get atomic info from local_pes (following walker_dual_gp approach)
        self.atomic_info = self.local_pes.get_atomic_info()
        
        # Validate atomic info (from walker_dual_gp)
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
        
        # Calculate dimensions
        if path_positions:
            self.n_atoms = len(path_positions[0]) // 3
            self.n_dof = len(path_positions[0])
        else:
            raise ValueError("No path positions provided")
        
        # Get atomic structure from local_pes if available
        if hasattr(self.local_pes, 'atomic_structure'):
            self.atomic_structure = self.local_pes.atomic_structure
        else:
            # If local_pes doesn't have atomic_structure, we need to handle this
            logging.warning("local_pes doesn't have atomic_structure attribute")
            self.atomic_structure = None
        
        if self.verbose:
            print(f"\nAtomic info from local_pes:")
            print(f"  n_pt: {self.atomic_info.get('n_pt')}")
            print(f"  moving_indices: {self.atomic_info.get('moving_indices')}")
            print(f"  Number of moving atoms: {len(self.moving_indices)}")
            print(f"  Number of active frozen atoms: {len(self.atomic_info.get('atomtype_fro', []))}")
        
        # Read force constants for thermal sampling
        self.eigenvalues_tr, self.eigenvectors_tr = self._read_force_constants(
            POSCAR_file, force_constants_file
        )
        
        # Initialize thermal noise calculation using ThermodynamicResidenceGP1
        self.trgp1 = ThermodynamicResidenceGP1(
            current_location=path_positions[0],  # Use first image as reference
            temperature=temperature,
            mass=self.mass,
            local_pes=self.local_pes,
            path=get_output_path("data_trgp1_path"),
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
        
        # Derived quantities
        self.n_images = len(path_positions)
        
        # Calculate thermal parameters
        self.kB = 8.617333e-5  # eV/K
        self.thermal_energy = self.kB * self.temperature
        
        # Storage for results
        self.gp1_models = []
        self.predictions = []
        self.statistics = []
        
        # Checkpoint info
        self.completed_images = 0
        
        logging.info(f"Initialized GP1 path walker with {self.n_images} images")
        logging.info(f"Moving atoms: {self.n_moving}")
    
    def _read_force_constants(self, POSCAR_file, force_constants_file):
        """Read force constants using phonopy (following walker_dual_gp pattern)."""
        # Handle input file paths
        if not os.path.isabs(POSCAR_file):
            poscar_path = get_input_path(POSCAR_file)
            if not os.path.exists(poscar_path):
                if os.path.basename(os.getcwd()) == 'scripts':
                    poscar_path = os.path.join('..', 'inputs', POSCAR_file)
                if not os.path.exists(poscar_path):
                    poscar_path = POSCAR_file  # Try current directory
        else:
            poscar_path = POSCAR_file
            
        if not os.path.isabs(force_constants_file):
            fc_path = get_input_path(force_constants_file)
            if not os.path.exists(fc_path):
                if os.path.basename(os.getcwd()) == 'scripts':
                    fc_path = os.path.join('..', 'inputs', force_constants_file)
                if not os.path.exists(fc_path):
                    fc_path = force_constants_file  # Try current directory
        else:
            fc_path = force_constants_file
        
        # Read phonopy data
        phonon = read_phonopy(
            sposcar=poscar_path,
            force_constants=fc_path
        )
        phonon.symmetrize_force_constants()
        
        # Get frequencies and eigenvectors at Gamma point
        freqs_thz, eigvecs_A = phonon.get_frequencies_with_eigenvectors(q=[0,0,0])
        
        # Drop acoustic modes (frequencies close to zero)
        mask = freqs_thz > 1e-6
        freqs_thz = freqs_thz[mask]
        eigvecs_A = eigvecs_A[:, mask].real
        
        # Convert to atomic units (following walker_dual_gp EXACTLY)
        atomic_time = 2.4188843265857e-17
        omega_s = 2 * np.pi * freqs_thz * 1e12 * atomic_time
        eigvecs_bohr = eigvecs_A
        
        if self.verbose:
            n = Poscar.from_file(poscar_path).structure.num_sites
            print(f"Read {omega_s.size} phonon modes for {n} atoms")
            print(f"Frequency range: {freqs_thz.min():.3f} - {freqs_thz.max():.3f} THz")
        
        return omega_s, eigvecs_bohr
    
    def generate_thermal_snapshots(self, positions: np.ndarray, image_index: int) -> Tuple[GP1, float, np.ndarray, np.ndarray]:
        """
        Generate thermal snapshots around given position and train GP1.
        
        Returns:
            Tuple of (trained_gp1, averaged_energy, averaged_forces, averaged_position)
        """
        logging.info(f"Generating {self.num_snapshots} thermal snapshots for image {image_index}")
        
        # Create GP1 instance with current position (following walker_dual_gp pattern)
        gp1_for_thermal = GP1(
            current_location=positions,
            temperature=self.temperature,
            mass=self.mass,
            local_pes=self.local_pes,
            path=get_output_path('data_gp1'),  # Use single data_gp1 directory
            atomic_info=self.atomic_info,
            noise_model=self.gp1_noise_model,
            student_t_df=self.gp1_student_t_df,
            use_adaptive_df=self.gp1_use_adaptive_df,
            adaptive_df_start_iter=self.gp1_adaptive_df_start_iter,
            adaptive_df_end_iter=self.gp1_adaptive_df_end_iter,
            adaptive_df_target=self.gp1_adaptive_df_target,
            remove_outliers=self.gp1_remove_outliers,
            outlier_threshold=self.gp1_outlier_threshold,
        )
        gp1_for_thermal.model_type = self.model_type
        
        # Set minimum noise for custom noise models
        if self.gp1_noise_model != "fixed" and self.thermal_noise is not None:
            gp1_for_thermal.minimum_noise = self.thermal_noise
            if self.verbose:
                print(f"GP1: Using minimum noise constraint from TRGP1: F={self.thermal_noise[0]:.6f} eV/Å, E={self.thermal_noise[1]:.6f} eV")
        
        # Set the global energy reference
        gp1_for_thermal.energy_reference = self.energy_reference
        
        # Set iteration for this image
        gp1_for_thermal.set_current_iteration(f"image_{image_index}")
        
        # Use get_thermalized_current_location like walker_dual_gp does
        # This will generate snapshots, evaluate them, train GP1, and return AVERAGED values
        x_thermal, e_thermal, f_thermal = gp1_for_thermal.get_thermalized_current_location(
            num_snapshots=self.num_snapshots,
            thermal_noise=self.thermal_noise,
            eigenvalues_tr=self.eigenvalues_tr,
            eigenvectors_tr=self.eigenvectors_tr,
            context=f'gp1_path_image_{image_index}',
            dimer_step=None,
            moving_only=True
        )
        
        # Note: get_thermalized_current_location returns:
        # - x_thermal: averaged position (moving atoms only)
        # - e_thermal: averaged energy (single value)
        # - f_thermal: averaged forces (moving atoms only)
        
        # The GP1 model is now trained, return it along with the averaged values
        return gp1_for_thermal, e_thermal, f_thermal, x_thermal
    
    def predict_and_analyze(self, gp1: GP1, image_index: int) -> Dict:
        """
        Make predictions and analyze statistics for a single GP1 model.
        
        Returns:
            Dictionary of statistics
        """
        logging.info(f"Analyzing predictions for image {image_index}")
        
        # Since we now use get_thermalized_current_location, the GP1 is already trained
        # and we need to get the training data from the GP1 model itself
        if not hasattr(gp1, 'model') or gp1.model is None:
            logging.warning(f"No trained model found in GP1 for image {image_index}")
            return {
                'image_index': image_index,
                'energy_mae': np.nan,
                'energy_rmse': np.nan,
                'energy_std': np.nan,
                'energy_mad': np.nan,
                'energy_sigma_mad_ratio': np.nan,
                'force_mae': np.nan,
                'force_rmse': np.nan,
                'force_std': np.nan,
                'force_mad': np.nan,
                'force_sigma_mad_ratio': np.nan,
                'raw_energy_std': np.nan,
                'raw_force_std': np.nan,
                'raw_energy_mad': np.nan,
                'raw_force_mad': np.nan,
                'raw_energy_sigma_mad': np.nan,
                'raw_force_sigma_mad': np.nan,
                'pred_energies': [],
                'actual_energies': [],
                'pred_forces': [],
                'actual_forces': [],
                'pred_energy_stds': [],
                'pred_force_stds': [],
                'force_mags_pred': [],
                'force_mags_actual': [],
                'mean_energy_std': np.nan,
                'mean_force_std': np.nan,
                'noise_levels': self.thermal_noise,
                'n_snapshots': 0
            }
        
        # Get training data from GP1 model
        # The model stores train_inputs (positions) and train_targets (stacked energy and forces)
        if hasattr(gp1.model, 'train_inputs') and hasattr(gp1.model, 'train_targets'):
            positions_train = gp1.model.train_inputs[0].numpy()  # Shape: (n_samples, n_features)
            targets = gp1.model.train_targets.numpy()  # Shape: (n_samples, 1 + n_features)
            energies_train = targets[:, 0]  # First column is energy
            forces_train = targets[:, 1:]  # Remaining columns are forces
        else:
            logging.warning(f"No training data accessible in GP1 model for image {image_index}")
            return {
                'image_index': image_index,
                'energy_mae': np.nan,
                'energy_rmse': np.nan,
                'energy_std': np.nan,
                'energy_mad': np.nan,
                'energy_sigma_mad_ratio': np.nan,
                'force_mae': np.nan,
                'force_rmse': np.nan,
                'force_std': np.nan,
                'force_mad': np.nan,
                'force_sigma_mad_ratio': np.nan,
                'raw_energy_std': np.nan,
                'raw_force_std': np.nan,
                'raw_energy_mad': np.nan,
                'raw_force_mad': np.nan,
                'raw_energy_sigma_mad': np.nan,
                'raw_force_sigma_mad': np.nan,
                'pred_energies': [],
                'actual_energies': [],
                'pred_forces': [],
                'actual_forces': [],
                'pred_energy_stds': [],
                'pred_force_stds': [],
                'force_mags_pred': [],
                'force_mags_actual': [],
                'mean_energy_std': np.nan,
                'mean_force_std': np.nan,
                'noise_levels': self.thermal_noise,
                'n_snapshots': 0,
                'n_training': 0,
                'error': 'No training data accessible in model'
            }
        
        energy_ref = gp1.energy_reference if hasattr(gp1, 'energy_reference') and gp1.energy_reference is not None else 0.0
        
        # Predict on training data
        pred_energies = []
        pred_forces = []
        pred_energy_stds = []
        pred_force_stds = []
        actual_energies = []
        actual_forces = []
        
        for i in range(len(positions_train)):
            pos_moving = positions_train[i]
            
            # Make prediction (GP1 predict expects moving atoms only)
            pred_e, pred_f_moving, var_e, var_f_moving = gp1.predict(pos_moving.reshape(1, -1))
            pred_e = float(pred_e[0])
            std_e = float(np.sqrt(var_e[0])) if var_e is not None else 0.0
            
            # Add back energy reference for comparison
            pred_energies.append(pred_e + energy_ref)
            pred_forces.append(pred_f_moving.flatten())
            pred_energy_stds.append(std_e)
            pred_force_stds.append(np.sqrt(var_f_moving).flatten() if var_f_moving is not None else np.zeros_like(pred_f_moving).flatten())
            actual_energies.append(energies_train[i] + energy_ref)
            actual_forces.append(forces_train[i].flatten())
        
        # Since we're using averaged values, we'll skip the original location prediction
        # The GP1 is trained on thermal snapshots around the current location
        
        # Calculate statistics
        pred_energies = np.array(pred_energies)
        actual_energies = np.array(actual_energies)
        energy_errors = pred_energies - actual_energies
        
        # Force errors (component-wise)
        force_errors = []
        force_mags_pred = []
        force_mags_actual = []
        
        for pred_f, actual_f in zip(pred_forces, actual_forces):
            error = pred_f - actual_f
            force_errors.extend(error)
            
            # Calculate magnitudes
            pred_f_3d = pred_f.reshape(-1, 3)
            actual_f_3d = actual_f.reshape(-1, 3)
            force_mags_pred.extend(np.linalg.norm(pred_f_3d, axis=1))
            force_mags_actual.extend(np.linalg.norm(actual_f_3d, axis=1))
        
        force_errors = np.array(force_errors)
        
        # Calculate MAD and ratios
        energy_mad = median_abs_deviation(energy_errors, scale='normal')
        force_mad = median_abs_deviation(force_errors, scale='normal')
        
        energy_std = np.std(energy_errors)
        force_std = np.std(force_errors)
        
        energy_sigma_mad_ratio = energy_std / energy_mad if energy_mad > 0 else 0
        force_sigma_mad_ratio = force_std / force_mad if force_mad > 0 else 0
        
        # Raw data statistics - should be on shifted energies
        # Shift actual energies by reference for meaningful statistics
        actual_energies_shifted = actual_energies - energy_ref
        raw_energy_std = np.std(actual_energies_shifted)
        raw_force_std = np.std(np.concatenate(actual_forces))
        raw_energy_mad = median_abs_deviation(actual_energies_shifted, scale='normal')
        raw_force_mad = median_abs_deviation(np.concatenate(actual_forces), scale='normal')
        
        stats = {
            'image_index': image_index,
            'energy_mae': np.mean(np.abs(energy_errors)),
            'energy_rmse': np.sqrt(np.mean(energy_errors**2)),
            'energy_std': energy_std,
            'energy_mad': energy_mad,
            'energy_sigma_mad_ratio': energy_sigma_mad_ratio,
            'force_mae': np.mean(np.abs(force_errors)),
            'force_rmse': np.sqrt(np.mean(force_errors**2)),
            'force_std': force_std,
            'force_mad': force_mad,
            'force_sigma_mad_ratio': force_sigma_mad_ratio,
            'raw_energy_std': raw_energy_std,
            'raw_force_std': raw_force_std,
            'raw_energy_mad': raw_energy_mad,
            'raw_force_mad': raw_force_mad,
            'raw_energy_sigma_mad': raw_energy_std / raw_energy_mad if raw_energy_mad > 0 else 0,
            'raw_force_sigma_mad': raw_force_std / raw_force_mad if raw_force_mad > 0 else 0,
            # Store predictions for plotting
            'pred_energies': pred_energies,
            'actual_energies': actual_energies,
            'pred_forces': pred_forces,  # Add this to store force predictions
            'actual_forces': actual_forces,  # Add this for completeness
            'pred_energy_stds': pred_energy_stds,
            'pred_force_stds': pred_force_stds,
            'force_mags_pred': force_mags_pred,
            'force_mags_actual': force_mags_actual,
            'mean_energy_std': np.mean(pred_energy_stds),
            'mean_force_std': np.mean([np.mean(std) for std in pred_force_stds]),
            'noise_levels': self.thermal_noise,
            'n_snapshots': len(positions_train),
            'n_training': len(positions_train)
        }
        
        return stats
    
    def run(self) -> Tuple[Dict, List[GP1]]:
        """
        Run the complete GP1 path analysis.
        
        Returns:
            (results_dict, gp1_models_list)
        """
        logging.info("Starting GP1 path analysis run")
        
        # CRITICAL: Establish energy reference using GP1 thermal average BEFORE processing images
        if not self.reference_set:
            print("\nEstablishing energy reference using GP1 thermal average from first image...")
            
            # Use the first image position for reference
            ref_position = self.path_positions[0]
            
            # Create initial GP1 for reference establishment
            gp1_ref = GP1(
                current_location=ref_position,
                temperature=self.temperature,
                mass=self.mass,
                local_pes=self.local_pes,
                path=get_output_path('data_gp1'),
                atomic_info=self.atomic_info,
                noise_model=self.gp1_noise_model,
                student_t_df=self.gp1_student_t_df,
                use_adaptive_df=self.gp1_use_adaptive_df,
                adaptive_df_start_iter=self.gp1_adaptive_df_start_iter,
                adaptive_df_end_iter=self.gp1_adaptive_df_end_iter,
                adaptive_df_target=self.gp1_adaptive_df_target,
                remove_outliers=self.gp1_remove_outliers,
                outlier_threshold=self.gp1_outlier_threshold,
            )
            gp1_ref.model_type = self.model_type
            
            # Set minimum noise for custom noise models
            if self.gp1_noise_model != "fixed" and self.thermal_noise is not None:
                gp1_ref.minimum_noise = self.thermal_noise
                if self.verbose:
                    print(f"GP1 (ref): Using minimum noise constraint from TRGP1: F={self.thermal_noise[0]:.6f} eV/Å, E={self.thermal_noise[1]:.6f} eV")
            
            gp1_ref.set_current_iteration("reference_establishment")
            
            # Get thermal average from GP1
            x_thermal_ref, e_thermal_ref, f_thermal_ref = gp1_ref.get_thermalized_current_location(
                num_snapshots=self.num_snapshots,
                thermal_noise=self.thermal_noise,
                eigenvalues_tr=self.eigenvalues_tr,
                eigenvectors_tr=self.eigenvectors_tr,
                context='reference_establishment',
                dimer_step=None,
                moving_only=True
            )
            
            # Set the energy reference
            self.energy_reference = e_thermal_ref
            self.reference_set = True
            
            # Store initial temperature
            if hasattr(gp1_ref, 'thermo_stats') and gp1_ref.thermo_stats:
                self.initial_avg_temperature = gp1_ref.thermo_stats.get('avg_temperature', self.temperature)
            else:
                self.initial_avg_temperature = self.temperature
            
            print(f"Energy reference established from GP1: {self.energy_reference:.4f} eV")
            print(f"Average temperature: {self.initial_avg_temperature:.1f} K")
        
        # Process each image
        for i in range(self.completed_images, self.n_images):
            logging.info(f"\nProcessing image {i+1}/{self.n_images}")
            
            # Get position for this image
            positions = self.path_positions[i]
            
            # Generate thermal snapshots and get trained GP1
            gp1, avg_energy, avg_forces, avg_pos_moving = self.generate_thermal_snapshots(positions, i)
            
            # Log averaged energy (now we only have one value, not multiple snapshots)
            logging.info(f"Image {i} averaged energy: {avg_energy:.2f} eV")
            
            # Warn if energy is very high (possible unphysical configuration)
            if abs(avg_energy) > 1000:
                logging.warning(f"Image {i} has very high energy - possible unphysical configuration!")
                logging.warning("Consider using a better interpolation method (e.g., NEB) instead of linear interpolation")
                
                # Check if this might be due to atomic overlaps
                pos_3d = positions.reshape(-1, 3)
                min_dist = float('inf')
                for j in range(len(pos_3d)):
                    for k in range(j+1, len(pos_3d)):
                        dist = np.linalg.norm(pos_3d[j] - pos_3d[k])
                        min_dist = min(min_dist, dist)
                
                logging.warning(f"Minimum interatomic distance: {min_dist:.3f} Å")
                if min_dist < 1.5:  # Typical minimum for metals
                    logging.error(f"ATOMIC OVERLAP DETECTED! Min distance {min_dist:.3f} Å is too small!")
            
            # The GP1 model is already trained in generate_thermal_snapshots
            self.gp1_models.append(gp1)
            
            # Analyze predictions
            stats = self.predict_and_analyze(gp1, i)
            self.statistics.append(stats)
            
            # Update progress
            self.completed_images = i + 1
            
            # Save checkpoint periodically
            if (i + 1) % 10 == 0:
                self.save_checkpoint()
            
            # Log progress
            if not np.isnan(stats['energy_mae']):
                logging.info(f"Image {i}: Energy MAE = {stats['energy_mae']:.6f} eV, "
                            f"Force MAE = {stats['force_mae']:.6f} eV/Å, "
                            f"σ/MAD = {stats['energy_sigma_mad_ratio']:.3f} (E), "
                            f"{stats['force_sigma_mad_ratio']:.3f} (F)")
            else:
                logging.warning(f"Image {i}: Failed to calculate statistics (no training data)")
        
        # Compile overall results
        results = self.compile_results()
        
        return results, self.gp1_models
    
    def compile_results(self) -> Dict:
        """Compile statistics across all images."""
        stats_array = self.statistics
        
        # Identify outlier images (those with extremely high errors)
        # Filter out NaN values first
        valid_stats = [s for s in stats_array if not np.isnan(s['energy_mae'])]
        if not valid_stats:
            logging.warning("No valid statistics found - all images had errors")
            valid_stats = stats_array  # Use all stats even with NaN
        
        energy_maes = [s['energy_mae'] for s in valid_stats if not np.isnan(s['energy_mae'])]
        if energy_maes:
            median_energy_mae = np.median(energy_maes)
            mad_energy_mae = median_abs_deviation(energy_maes, scale='normal')
            
            # Mark images as outliers if their MAE is more than 5 MADs from median
            outlier_threshold = median_energy_mae + 5 * mad_energy_mae
            non_outlier_stats = [s for s in valid_stats if not np.isnan(s['energy_mae']) and s['energy_mae'] < outlier_threshold]
            outlier_stats = [s for s in valid_stats if not np.isnan(s['energy_mae']) and s['energy_mae'] >= outlier_threshold]
        else:
            non_outlier_stats = []
            outlier_stats = []
        
        logging.info(f"Identified {len(outlier_stats)} outlier images out of {len(stats_array)}")
        if outlier_stats:
            outlier_indices = [s['image_index'] for s in outlier_stats]
            logging.info(f"Outlier image indices: {outlier_indices}")
        
        # Calculate statistics excluding outliers for more meaningful results
        if non_outlier_stats:
            avg_energy_mae = np.nanmean([s['energy_mae'] for s in non_outlier_stats])
            avg_force_mae = np.nanmean([s['force_mae'] for s in non_outlier_stats])
            avg_energy_sigma_mad = np.nanmean([s['energy_sigma_mad_ratio'] for s in non_outlier_stats])
            avg_force_sigma_mad = np.nanmean([s['force_sigma_mad_ratio'] for s in non_outlier_stats])
            
            # Raw data statistics (excluding outliers)
            avg_raw_energy_std = np.nanmean([s['raw_energy_std'] for s in non_outlier_stats])
            avg_raw_force_std = np.nanmean([s['raw_force_std'] for s in non_outlier_stats])
            avg_raw_energy_sigma_mad = np.nanmean([s['raw_energy_sigma_mad'] for s in non_outlier_stats])
            avg_raw_force_sigma_mad = np.nanmean([s['raw_force_sigma_mad'] for s in non_outlier_stats])
        else:
            # All images are outliers or invalid - use all data with nanmean
            avg_energy_mae = np.nanmean([s['energy_mae'] for s in stats_array])
            avg_force_mae = np.nanmean([s['force_mae'] for s in stats_array])
            avg_energy_sigma_mad = np.nanmean([s['energy_sigma_mad_ratio'] for s in stats_array])
            avg_force_sigma_mad = np.nanmean([s['force_sigma_mad_ratio'] for s in stats_array])
            
            avg_raw_energy_std = np.nanmean([s['raw_energy_std'] for s in stats_array])
            avg_raw_force_std = np.nanmean([s['raw_force_std'] for s in stats_array])
            avg_raw_energy_sigma_mad = np.nanmean([s['raw_energy_sigma_mad'] for s in stats_array])
            avg_raw_force_sigma_mad = np.nanmean([s['raw_force_sigma_mad'] for s in stats_array])
        
        # Thermal noise (single value from trgp1)
        avg_force_noise = self.thermal_noise[0]
        avg_energy_noise = self.thermal_noise[1]
        
        results = {
            'n_images': self.n_images,
            'n_snapshots': self.num_snapshots,
            'total_evaluations': self.n_images * (self.num_snapshots + 1),  # +1 for original
            'avg_energy_error': avg_energy_mae,
            'std_energy_error': np.nanstd([s['energy_mae'] for s in non_outlier_stats]) if non_outlier_stats else np.nanstd([s['energy_mae'] for s in stats_array]),
            'avg_force_error': avg_force_mae,
            'std_force_error': np.nanstd([s['force_mae'] for s in non_outlier_stats]) if non_outlier_stats else np.nanstd([s['force_mae'] for s in stats_array]),
            'avg_energy_sigma_mad_ratio': avg_energy_sigma_mad,
            'avg_force_sigma_mad_ratio': avg_force_sigma_mad,
            'all_images': stats_array,  # Store all image statistics
            'raw_energy_std': avg_raw_energy_std,
            'raw_force_std': avg_raw_force_std,
            'raw_energy_sigma_mad': avg_raw_energy_sigma_mad,
            'raw_force_sigma_mad': avg_raw_force_sigma_mad,
            'avg_force_noise': avg_force_noise,
            'avg_energy_noise': avg_energy_noise,
            'per_image_stats': stats_array,
            'thermal_noise': self.thermal_noise,  # Single thermal noise from trgp1
            'temperature': self.temperature,
            'noise_model': self.gp1_noise_model,
            'n_outlier_images': len(outlier_stats),
            'outlier_indices': [s['image_index'] for s in outlier_stats] if outlier_stats else []
        }
        
        return results
    
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save checkpoint."""
        if filename is None:
            checkpoint_dir = get_output_path('checkpoints')
            filename = os.path.join(checkpoint_dir, 'gp1_path_latest.pkl')
        
        # Create a copy of self without unpickleable GP models
        walker_copy = copy.copy(self)
        walker_copy.gp1_models = []  # Don't save GP models
        walker_copy.trgp1 = None  # Don't save trgp1
        
        # Update per-image statistics with noise and temperature info
        per_image_stats = []
        for i, stats in enumerate(self.statistics):
            enhanced_stats = stats.copy()
            
            # Get GP1 model for this image if available
            if i < len(self.gp1_models) and self.gp1_models[i]:
                gp1 = self.gp1_models[i]
                
                # Get noise information
                if hasattr(gp1, 'likelihood'):
                    try:
                        # Initial noise from thermal noise if available
                        if self.thermal_noise is not None:
                            noise_force, noise_energy = self.thermal_noise
                            enhanced_stats['gp1_initial_noise'] = {
                                'energy': float(noise_energy),
                                'force': float(noise_force)
                            }
                        else:
                            enhanced_stats['gp1_initial_noise'] = 0.01  # Default initial
                        
                        # Final noise from likelihood
                        if hasattr(gp1.likelihood, 'raw_task_noises'):
                            # Multitask likelihood with task-specific noises
                            task_noises = gp1.likelihood.raw_task_noises_constraint.transform(
                                gp1.likelihood.raw_task_noises
                            )
                            task_noises_np = task_noises.detach().cpu().numpy()
                            enhanced_stats['gp1_final_noise'] = {
                                'energy': float(task_noises_np[0]) if len(task_noises_np) > 0 else None,
                                'force': float(task_noises_np[1]) if len(task_noises_np) > 1 else None
                            }
                        elif hasattr(gp1.likelihood, 'noise'):
                            enhanced_stats['gp1_final_noise'] = float(gp1.likelihood.noise.item())
                        else:
                            enhanced_stats['gp1_final_noise'] = 0.01
                    except Exception as e:
                        print(f"Warning: Could not extract GP1 noise for image {i}: {e}")
                        enhanced_stats['gp1_initial_noise'] = 0.01
                        enhanced_stats['gp1_final_noise'] = 0.01
                
                # Get temperature info
                if hasattr(gp1, 'thermo_stats') and gp1.thermo_stats:
                    enhanced_stats['avg_temperature'] = gp1.thermo_stats.get('avg_temperature', self.temperature)
                else:
                    enhanced_stats['avg_temperature'] = self.temperature
            
            per_image_stats.append(enhanced_stats)
        
        checkpoint_data = {
            'walker': walker_copy,
            'completed_images': self.completed_images,
            # Note: training_data is now stored inside each GP1 model
            'statistics': self.statistics,
            'thermal_noise': self.thermal_noise,  # Single thermal noise from trgp1
            # New fields
            'energy_reference': self.energy_reference,
            'reference_set': self.reference_set,
            'per_image_stats': per_image_stats,
            'initial_avg_temperature': getattr(self, 'initial_avg_temperature', self.temperature)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logging.info(f"Checkpoint saved to {filename}")
    
    def create_analysis_plots(self, results: Dict, gp1_models: List[GP1]):
        """Create comprehensive analysis plots."""
        # Check if there's any valid data
        valid_stats = [s for s in results['per_image_stats'] 
                      if not np.isnan(s.get('energy_mae', np.nan))]
        
        # Even if no valid statistics, we'll create diagnostic plots
        if not valid_stats:
            logging.warning("No valid statistics available. Creating diagnostic plots only.")
            self.create_diagnostic_plots(results, gp1_models)
            return
        
        plots_dir = get_output_path('plots')
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots - updated to 5x3 grid
        fig = plt.figure(figsize=(24, 25))
        
        # Row 1: Error Analysis
        # 1. Energy prediction errors along path
        ax1 = plt.subplot(5, 3, 1)
        self.plot_energy_errors_along_path(ax1, results)
        
        # 2. Force prediction errors along path
        ax2 = plt.subplot(5, 3, 2)
        self.plot_force_errors_along_path(ax2, results)
        
        # 3. σ/MAD ratios along path
        ax3 = plt.subplot(5, 3, 3)
        self.plot_sigma_mad_ratios(ax3, results)
        
        # Row 2: Parity Plots
        # 4. Energy prediction scatter plot
        ax4 = plt.subplot(5, 3, 4)
        self.plot_energy_parity(ax4, results)
        
        # 5. Force prediction scatter plot
        ax5 = plt.subplot(5, 3, 5)
        self.plot_force_parity(ax5, results)
        
        # 6. Uncertainty calibration
        ax6 = plt.subplot(5, 3, 6)
        self.plot_uncertainty_calibration(ax6, results)
        
        # Row 3: Energy Profiles
        # 7. GP1 vs Actual Energy Profiles
        ax7 = plt.subplot(5, 3, 7)
        self.plot_gp1_vs_actual_energy_profile(ax7, results)
        
        # 8. Energy landscape (actual with thermal spread)
        ax8 = plt.subplot(5, 3, 8)
        self.plot_energy_landscape(ax8, results)
        
        # 9. GP1 Energy Profile with Uncertainty
        ax9 = plt.subplot(5, 3, 9)
        self.plot_gp1_energy_profile_with_uncertainty(ax9, results)
        
        # Row 4: Statistics and Analysis
        # 10. Raw data statistics along path
        ax10 = plt.subplot(5, 3, 10)
        self.plot_raw_data_statistics(ax10, results)
        
        # 11. GP1 prediction uncertainties along path
        ax11 = plt.subplot(5, 3, 11)
        self.plot_gp1_prediction_uncertainties(ax11, results)
        
        # 12. Per-Image Error Distribution
        ax12 = plt.subplot(5, 3, 12)
        self.plot_error_distribution_per_image(ax12, results)
        
        # Row 5: Additional Analysis
        # 13. Original location predictions
        ax13 = plt.subplot(5, 3, 13)
        self.plot_original_predictions(ax13, results)
        
        # 14. Thermal noise analysis
        ax14 = plt.subplot(5, 3, 14)
        self.plot_thermal_noise(ax14, results)
        
        # 15. Summary statistics
        ax15 = plt.subplot(5, 3, 15)
        self.plot_summary_stats(ax15, results)
        
        plt.suptitle('GP1 Path Analysis Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plot_file = os.path.join(plots_dir, 'gp1_path_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logging.info(f"Analysis plots saved to {plot_file}")
        plt.close()
    
    def create_diagnostic_plots(self, results: Dict, gp1_models: List[GP1]):
        """Create diagnostic plots when no valid statistics are available."""
        plots_dir = get_output_path('plots')
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with diagnostic plots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Energy along path (if available)
        ax1 = plt.subplot(2, 2, 1)
        if 'all_images' in results:
            image_indices = []
            energies = []
            for img_data in results['all_images']:
                if 'avg_energy' in img_data and img_data['avg_energy'] is not None:
                    image_indices.append(img_data['image_index'])
                    energies.append(img_data['avg_energy'])
            
            if energies:
                ax1.plot(image_indices, energies, 'o-', linewidth=2, markersize=8)
                ax1.set_xlabel('Image Index')
                ax1.set_ylabel('Energy (eV)')
                ax1.set_title('Average Energy along Path')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No energy data available', 
                        transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title('Energy along Path')
        
        # 2. Thermal noise information
        ax2 = plt.subplot(2, 2, 2)
        thermal_info_text = f"Thermal Noise Analysis\n\n"
        thermal_info_text += f"Temperature: {self.temperature:.1f} K\n"
        thermal_info_text += f"Force noise (σ_F): {self.thermal_noise[0]:.3f} eV/Å\n"
        thermal_info_text += f"Energy noise (σ_E): {self.thermal_noise[1]:.3f} eV\n"
        thermal_info_text += f"Snapshots per image: {self.num_snapshots}\n"
        
        ax2.text(0.1, 0.9, thermal_info_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=12, family='monospace')
        ax2.axis('off')
        ax2.set_title('Thermal Sampling Parameters')
        
        # 3. Per-image training status
        ax3 = plt.subplot(2, 2, 3)
        image_indices = []
        training_counts = []
        statuses = []
        
        for stats in results['per_image_stats']:
            image_indices.append(stats['image_index'])
            if 'n_training' in stats and stats['n_training'] is not None:
                training_counts.append(stats['n_training'])
                statuses.append('Valid')
            else:
                training_counts.append(0)
                statuses.append('Failed')
        
        colors = ['green' if s == 'Valid' else 'red' for s in statuses]
        ax3.bar(image_indices, training_counts, color=colors, alpha=0.7)
        ax3.set_xlabel('Image Index')
        ax3.set_ylabel('Training Data Points')
        ax3.set_title('GP1 Training Data per Image')
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Valid'),
                          Patch(facecolor='red', alpha=0.7, label='Failed')]
        ax3.legend(handles=legend_elements)
        
        # 4. Summary information
        ax4 = plt.subplot(2, 2, 4)
        summary_text = f"Analysis Summary\n\n"
        summary_text += f"Total images: {results['n_images']}\n"
        summary_text += f"Failed images: {sum(1 for s in statuses if s == 'Failed')}\n"
        summary_text += f"Total evaluations: {results['total_evaluations']}\n"
        summary_text += f"\nFailure Reasons:\n"
        
        # Collect failure reasons
        for stats in results['per_image_stats']:
            if stats.get('error'):
                summary_text += f"  Image {stats['image_index']}: {stats['error'][:50]}...\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, family='monospace')
        ax4.axis('off')
        ax4.set_title('Diagnostic Summary')
        
        plt.suptitle('GP1 Path Analysis Diagnostics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plot_file = os.path.join(plots_dir, 'gp1_path_diagnostics.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logging.info(f"Diagnostic plots saved to {plot_file}")
        plt.close()
    
    def plot_energy_errors_along_path(self, ax, results):
        """Plot energy prediction errors along the path."""
        stats = results['per_image_stats']
        valid_stats = [s for s in stats if not np.isnan(s.get('energy_mae', np.nan))]
        
        if not valid_stats:
            ax.text(0.5, 0.5, 'No valid data available', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Energy Error (eV)')
            ax.set_title('Energy Prediction Errors Along Path')
            return
        
        indices = [s['image_index'] for s in valid_stats]
        mae = [s['energy_mae'] for s in valid_stats]
        rmse = [s['energy_rmse'] for s in valid_stats]
        
        ax.plot(indices, mae, 'bo-', label='MAE', markersize=6)
        ax.plot(indices, rmse, 'rs--', label='RMSE', markersize=6)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Energy Error (eV)')
        ax.set_title('Energy Prediction Errors Along Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_force_errors_along_path(self, ax, results):
        """Plot force prediction errors along the path."""
        stats = results['per_image_stats']
        valid_stats = [s for s in stats if not np.isnan(s.get('force_mae', np.nan))]
        
        if not valid_stats:
            ax.text(0.5, 0.5, 'No valid data available', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Force Error (eV/Å)')
            ax.set_title('Force Prediction Errors Along Path')
            return
        
        indices = [s['image_index'] for s in valid_stats]
        mae = [s['force_mae'] for s in valid_stats]
        rmse = [s['force_rmse'] for s in valid_stats]
        
        ax.plot(indices, mae, 'go-', label='MAE', markersize=6)
        ax.plot(indices, rmse, 'ms--', label='RMSE', markersize=6)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Force Error (eV/Å)')
        ax.set_title('Force Prediction Errors Along Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_sigma_mad_ratios(self, ax, results):
        """Plot σ/MAD ratios along the path."""
        stats = results['per_image_stats']
        valid_stats = [s for s in stats if not np.isnan(s.get('energy_sigma_mad_ratio', np.nan))]
        
        if not valid_stats:
            ax.text(0.5, 0.5, 'No valid data available', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('σ/MAD Ratio')
            ax.set_title('Uncertainty Quantification Along Path')
            return
        
        indices = [s['image_index'] for s in valid_stats]
        energy_ratios = [s['energy_sigma_mad_ratio'] for s in valid_stats]
        force_ratios = [s['force_sigma_mad_ratio'] for s in valid_stats]
        
        ax.plot(indices, energy_ratios, 'bo-', label='Energy', markersize=6)
        ax.plot(indices, force_ratios, 'ro-', label='Force', markersize=6)
        ax.axhline(y=1.46, color='k', linestyle='--', alpha=0.5, label='Target (1.46)')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('σ/MAD Ratio')
        ax.set_title('Uncertainty Quantification Along Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 3)
    
    def plot_energy_parity(self, ax, results):
        """Plot predicted vs actual energies."""
        all_pred = []
        all_actual = []
        
        for stats in results['per_image_stats']:
            if not np.isnan(stats.get('energy_mae', np.nan)):
                all_pred.extend(stats.get('pred_energies', []))
                all_actual.extend(stats.get('actual_energies', []))
        
        if not all_pred or not all_actual:
            ax.text(0.5, 0.5, 'No valid data available', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_xlabel('Actual Energy (eV)')
            ax.set_ylabel('Predicted Energy (eV)')
            ax.set_title('Energy Parity Plot')
            return
        
        all_pred = np.array(all_pred)
        all_actual = np.array(all_actual)
        
        # Remove mean for better visualization
        mean_e = np.mean(all_actual)
        all_pred -= mean_e
        all_actual -= mean_e
        
        ax.scatter(all_actual, all_pred, alpha=0.5, s=20)
        
        # Perfect correlation line
        min_val = min(all_actual.min(), all_pred.min())
        max_val = max(all_actual.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Calculate R²
        if len(all_actual) > 1:
            r2 = np.corrcoef(all_actual, all_pred)[0, 1]**2
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Actual Energy - Mean (eV)')
        ax.set_ylabel('Predicted Energy - Mean (eV)')
        ax.set_title('Energy Parity Plot')
        ax.grid(True, alpha=0.3)
    
    def plot_force_parity(self, ax, results):
        """Plot predicted vs actual force magnitudes."""
        all_pred = []
        all_actual = []
        
        for stats in results['per_image_stats']:
            if not np.isnan(stats.get('force_mae', np.nan)):
                all_pred.extend(stats.get('force_mags_pred', []))
                all_actual.extend(stats.get('force_mags_actual', []))
        
        if not all_pred or not all_actual:
            ax.text(0.5, 0.5, 'No valid data available', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_xlabel('Actual |Force| (eV/Å)')
            ax.set_ylabel('Predicted |Force| (eV/Å)')
            ax.set_title('Force Parity Plot')
            return
        
        all_pred = np.array(all_pred)
        all_actual = np.array(all_actual)
        
        # Subsample for visualization
        if len(all_pred) > 5000:
            idx = np.random.choice(len(all_pred), 5000, replace=False)
            all_pred = all_pred[idx]
            all_actual = all_actual[idx]
        
        ax.scatter(all_actual, all_pred, alpha=0.3, s=10)
        
        # Perfect correlation line
        max_val = max(all_actual.max(), all_pred.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        # Calculate R²
        if len(all_actual) > 1:
            r2 = np.corrcoef(all_actual, all_pred)[0, 1]**2
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Actual |Force| (eV/Å)')
        ax.set_ylabel('Predicted |Force| (eV/Å)')
        ax.set_title('Force Magnitude Parity Plot')
        ax.grid(True, alpha=0.3)
    
    def plot_uncertainty_calibration(self, ax, results):
        """Plot uncertainty calibration."""
        # Collect prediction errors and uncertainties
        energy_errors = []
        energy_stds = []
        
        for stats in results['per_image_stats']:
            pred = stats['pred_energies']
            actual = stats['actual_energies']
            stds = stats['pred_energy_stds']
            
            errors = np.abs(pred - actual)
            energy_errors.extend(errors)
            energy_stds.extend(stds)
        
        energy_errors = np.array(energy_errors)
        energy_stds = np.array(energy_stds)
        
        # Bin by predicted uncertainty
        n_bins = 10
        bins = np.percentile(energy_stds, np.linspace(0, 100, n_bins + 1))
        
        bin_centers = []
        actual_errors = []
        
        for i in range(n_bins):
            mask = (energy_stds >= bins[i]) & (energy_stds < bins[i + 1])
            if np.sum(mask) > 0:
                bin_centers.append(np.mean(energy_stds[mask]))
                actual_errors.append(np.mean(energy_errors[mask]))
        
        if bin_centers:
            ax.plot(bin_centers, actual_errors, 'bo-', markersize=8, label='Actual')
            ax.plot([0, max(bin_centers)], [0, max(bin_centers)], 'k--', alpha=0.5, label='Perfect')
            ax.set_xlabel('Predicted Uncertainty (eV)')
            ax.set_ylabel('Actual Error (eV)')
            ax.set_title('Uncertainty Calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_raw_data_statistics(self, ax, results):
        """Plot raw data statistics along path."""
        stats = results['per_image_stats']
        indices = [s['image_index'] for s in stats]
        energy_stds = [s['raw_energy_std'] for s in stats]
        force_stds = [s['raw_force_std'] for s in stats]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(indices, energy_stds, 'bo-', label='Energy Std', markersize=6)
        line2 = ax2.plot(indices, force_stds, 'ro-', label='Force Std', markersize=6)
        
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Energy Std Dev (eV)', color='b')
        ax2.set_ylabel('Force Std Dev (eV/Å)', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        
        ax.set_title('Raw Data Variability Along Path')
        ax.grid(True, alpha=0.3)
    
    def plot_energy_landscape(self, ax, results):
        """Plot energy landscape along path."""
        # Get mean energies for each image
        mean_energies = []
        for i, stats in enumerate(results['per_image_stats']):
            mean_e = np.mean(stats['actual_energies'])
            mean_energies.append(mean_e)
        
        mean_energies = np.array(mean_energies)
        
        # Normalize to start at zero
        mean_energies -= mean_energies[0]
        
        # Create reaction coordinate
        reaction_coord = np.linspace(0, 1, len(mean_energies))
        
        ax.plot(reaction_coord, mean_energies, 'b-', linewidth=2)
        
        # Add thermal spread visualization
        thermal_spreads = [s['raw_energy_std'] for s in results['per_image_stats']]
        upper = mean_energies + np.array(thermal_spreads)
        lower = mean_energies - np.array(thermal_spreads)
        ax.fill_between(reaction_coord, lower, upper, alpha=0.3, label='Thermal spread (±σ)')
        
        # Mark endpoints and maximum
        ax.plot(0, mean_energies[0], 'go', markersize=10, label='Local Min')
        ax.plot(1, mean_energies[-1], 'ro', markersize=10, label='Saddle')
        
        max_idx = np.argmax(mean_energies)
        ax.plot(reaction_coord[max_idx], mean_energies[max_idx], 'r*', 
                markersize=15, label=f'Max ({mean_energies[max_idx]:.3f} eV)')
        
        ax.set_xlabel('Reaction Coordinate')
        ax.set_ylabel('Relative Energy (eV)')
        ax.set_title('Energy Profile Along Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_sigma_mad_distribution(self, ax, results):
        """Plot distribution of σ/MAD ratios."""
        stats = results['per_image_stats']
        energy_ratios = [s['energy_sigma_mad_ratio'] for s in stats]
        force_ratios = [s['force_sigma_mad_ratio'] for s in stats]
        
        bins = np.linspace(0, 3, 20)
        
        ax.hist(energy_ratios, bins=bins, alpha=0.5, label='Energy', color='blue')
        ax.hist(force_ratios, bins=bins, alpha=0.5, label='Force', color='red')
        ax.axvline(x=1.46, color='k', linestyle='--', alpha=0.5, label='Target')
        ax.axvline(x=np.mean(energy_ratios), color='blue', linestyle=':', 
                   label=f'Energy mean: {np.mean(energy_ratios):.2f}')
        ax.axvline(x=np.mean(force_ratios), color='red', linestyle=':', 
                   label=f'Force mean: {np.mean(force_ratios):.2f}')
        
        ax.set_xlabel('σ/MAD Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of σ/MAD Ratios')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_original_predictions(self, ax, results):
        """Plot predictions at original (non-snapshot) locations."""
        stats = results['per_image_stats']
        
        # Skip if we don't have original location data (using averaged snapshots)
        if not stats or 'orig_energy_error' not in stats[0]:
            ax.text(0.5, 0.5, 'Using averaged thermal snapshots\n(no separate original location predictions)', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title('Original Location Predictions')
            ax.axis('off')
            return
        
        indices = [s['image_index'] for s in stats]
        energy_errors = [s['orig_energy_error'] for s in stats]
        force_mae = [s['orig_force_mae'] for s in stats]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(indices, energy_errors, 'bo-', label='Energy Error', markersize=6)
        line2 = ax2.plot(indices, force_mae, 'ro-', label='Force MAE', markersize=6)
        
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Energy Error (eV)', color='b')
        ax2.set_ylabel('Force MAE (eV/Å)', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        
        ax.set_title('Predictions at Original Locations')
        ax.grid(True, alpha=0.3)
        
        # Add zero line for energy
        ax.axhline(y=0, color='b', linestyle=':', alpha=0.5)
    
    def plot_thermal_noise(self, ax, results):
        """Plot thermal noise estimates."""
        # Since we now use a single thermal noise estimate from trgp1
        if 'thermal_noises' not in results:
            # Display the single thermal noise value we're using
            thermal_info = f"""Thermal Noise (Single Estimate)

Temperature: {self.temperature:.0f} K
Force noise (σ_F): {self.thermal_noise[0]:.3f} eV/Å
Energy noise (σ_E): {self.thermal_noise[1]:.3f} eV

Note: Using unified thermal noise
estimate from TRGP1 calculation"""
            ax.text(0.1, 0.5, thermal_info, transform=ax.transAxes, 
                    verticalalignment='center', fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            ax.set_title('Thermal Noise Estimate')
            ax.axis('off')
            return
        
        # Extract thermal noise values
        force_noises = [n[0] for n in results['thermal_noises']]
        energy_noises = [n[1] for n in results['thermal_noises']]
        
        indices = list(range(len(force_noises)))
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(indices, energy_noises, 'bo-', label='Energy Noise', markersize=6)
        line2 = ax2.plot(indices, force_noises, 'ro-', label='Force Noise', markersize=6)
        
        # Add theoretical values if available
        if hasattr(self, 'theoretical_thermal_noise'):
            ax.axhline(y=self.theoretical_thermal_noise[1], color='b', linestyle='--', 
                      alpha=0.5, label='Theory (E)')
            ax2.axhline(y=self.theoretical_thermal_noise[0], color='r', linestyle='--', 
                       alpha=0.5, label='Theory (F)')
        
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Energy Noise (eV)', color='b')
        ax2.set_ylabel('Force Noise (eV/Å)', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        
        ax.set_title('Thermal Noise Estimates')
        ax.grid(True, alpha=0.3)
    
    def plot_summary_stats(self, ax, results):
        """Plot summary statistics table."""
        ax.axis('off')
        
        # Create summary text
        summary_text = f"""GP1 Path Analysis Summary
{'='*40}

Images analyzed: {results['n_images']}
Snapshots per image: {results['n_snapshots']}
Total evaluations: {results['total_evaluations']}

Average Prediction Errors:
  Energy: {results['avg_energy_error']:.6f} ± {results['std_energy_error']:.6f} eV
  Force: {results['avg_force_error']:.6f} ± {results['std_force_error']:.6f} eV/Å

Uncertainty Quantification (σ/MAD):
  Energy: {results['avg_energy_sigma_mad_ratio']:.3f} (target: 1.46)
  Force: {results['avg_force_sigma_mad_ratio']:.3f} (target: 1.46)

Raw Data Statistics:
  Energy std: {results['raw_energy_std']:.6f} eV
  Force std: {results['raw_force_std']:.6f} eV/Å
  Energy σ/MAD: {results['raw_energy_sigma_mad']:.3f}
  Force σ/MAD: {results['raw_force_sigma_mad']:.3f}

Thermal Noise:
  Force: {results['avg_force_noise']:.6f} eV/Å
  Energy: {results['avg_energy_noise']:.6f} eV

Temperature: {self.temperature} K
Noise model: {self.gp1_noise_model}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.9))
    
    def plot_gp1_vs_actual_energy_profile(self, ax, results):
        """Plot GP1 predictions vs actual energies along the path."""
        # Get mean energies for each image from actual data
        actual_mean_energies = []
        gp1_mean_energies = []
        
        for i, stats in enumerate(results['per_image_stats']):
            actual_mean = np.mean(stats['actual_energies'])
            actual_mean_energies.append(actual_mean)
            
            # Get GP1 predictions
            pred_mean = np.mean(stats['pred_energies'])
            gp1_mean_energies.append(pred_mean)
        
        actual_mean_energies = np.array(actual_mean_energies)
        gp1_mean_energies = np.array(gp1_mean_energies)
        
        # Normalize to start at zero
        actual_mean_energies -= actual_mean_energies[0]
        gp1_mean_energies -= gp1_mean_energies[0]
        
        # Create reaction coordinate
        reaction_coord = np.linspace(0, 1, len(actual_mean_energies))
        
        # Plot both profiles
        ax.plot(reaction_coord, actual_mean_energies, 'b-', linewidth=2.5, 
                label='Actual (EAM)', marker='o', markersize=6)
        ax.plot(reaction_coord, gp1_mean_energies, 'r--', linewidth=2.5, 
                label='GP1 Prediction', marker='s', markersize=6)
        
        # Mark key points
        max_idx_actual = np.argmax(actual_mean_energies)
        max_idx_gp1 = np.argmax(gp1_mean_energies)
        
        ax.annotate(f'Actual Max: {actual_mean_energies[max_idx_actual]:.3f} eV',
                    xy=(reaction_coord[max_idx_actual], actual_mean_energies[max_idx_actual]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.7),
                    fontsize=8, color='white')
        
        ax.annotate(f'GP1 Max: {gp1_mean_energies[max_idx_gp1]:.3f} eV',
                    xy=(reaction_coord[max_idx_gp1], gp1_mean_energies[max_idx_gp1]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.7),
                    fontsize=8, color='white')
        
        ax.set_xlabel('Reaction Coordinate')
        ax.set_ylabel('Relative Energy (eV)')
        ax.set_title('GP1 vs Actual Energy Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_gp1_energy_profile_with_uncertainty(self, ax, results):
        """Plot GP1 energy profile with uncertainty bands."""
        # Get GP1 predictions and uncertainties for each image
        gp1_mean_energies = []
        gp1_std_energies = []
        
        for i, stats in enumerate(results['per_image_stats']):
            pred_mean = np.mean(stats['pred_energies'])
            gp1_mean_energies.append(pred_mean)
            
            # Average uncertainty across snapshots
            pred_std = np.mean(stats['pred_energy_stds'])
            gp1_std_energies.append(pred_std)
        
        gp1_mean_energies = np.array(gp1_mean_energies)
        gp1_std_energies = np.array(gp1_std_energies)
        
        # Normalize to start at zero
        gp1_mean_energies -= gp1_mean_energies[0]
        
        # Create reaction coordinate
        reaction_coord = np.linspace(0, 1, len(gp1_mean_energies))
        
        # Plot mean with confidence intervals
        ax.plot(reaction_coord, gp1_mean_energies, 'g-', linewidth=2.5, 
                label='GP1 Mean Prediction')
        
        # Add uncertainty bands
        ax.fill_between(reaction_coord, 
                        gp1_mean_energies - 2*gp1_std_energies,
                        gp1_mean_energies + 2*gp1_std_energies,
                        alpha=0.3, color='green', label='±2σ Uncertainty')
        
        ax.fill_between(reaction_coord, 
                        gp1_mean_energies - gp1_std_energies,
                        gp1_mean_energies + gp1_std_energies,
                        alpha=0.5, color='green', label='±1σ Uncertainty')
        
        # Mark barrier
        max_idx = np.argmax(gp1_mean_energies)
        ax.plot(reaction_coord[max_idx], gp1_mean_energies[max_idx], 'r*', 
                markersize=15, label=f'Barrier: {gp1_mean_energies[max_idx]:.3f} eV')
        
        # Add text showing average uncertainty
        avg_uncertainty = np.mean(gp1_std_energies)
        ax.text(0.05, 0.95, f'Avg. σ = {avg_uncertainty:.4f} eV', 
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Reaction Coordinate')
        ax.set_ylabel('Relative Energy (eV)')
        ax.set_title('GP1 Energy Profile with Uncertainty Bands')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_error_distribution_per_image(self, ax, results):
        """Plot distribution of errors for selected images."""
        # Select a few representative images
        n_images = len(results['per_image_stats'])
        selected_indices = [0, n_images//4, n_images//2, 3*n_images//4, n_images-1]
        
        energy_errors_by_image = []
        labels = []
        
        for idx in selected_indices:
            if idx < len(results['per_image_stats']):
                stats = results['per_image_stats'][idx]
                pred = np.array(stats['pred_energies'])
                actual = np.array(stats['actual_energies'])
                errors = pred - actual
                energy_errors_by_image.append(errors)
                labels.append(f'Image {idx}')
        
        # Create violin plot
        parts = ax.violinplot(energy_errors_by_image, positions=range(len(energy_errors_by_image)),
                              showmeans=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Energy Prediction Error (eV)')
        ax.set_title('Error Distribution for Selected Images')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_gp1_prediction_uncertainties(self, ax, results):
        """Plot standard deviation of GP1 predictions along the path."""
        stats = results['per_image_stats']
        indices = [s['image_index'] for s in stats]
        
        # Calculate std dev of GP1 predictions for each image
        gp1_energy_stds = []
        gp1_force_stds = []
        
        for stat in stats:
            # Std dev of GP1 energy predictions across snapshots
            if 'pred_energies' in stat and len(stat['pred_energies']) > 1:
                energy_std = np.std(stat['pred_energies'])
                gp1_energy_stds.append(energy_std)
            else:
                gp1_energy_stds.append(0)
            
            # Std dev of GP1 force predictions across snapshots
            if 'pred_forces' in stat and stat['pred_forces']:
                # Flatten all force components from all snapshots
                all_force_components = []
                for force_pred in stat['pred_forces']:
                    all_force_components.extend(force_pred.flatten())
                force_std = np.std(all_force_components) if all_force_components else 0
                gp1_force_stds.append(force_std)
            else:
                gp1_force_stds.append(0)
        
        # Create twin axis for force
        ax2 = ax.twinx()
        
        # Plot standard deviations
        line1 = ax.plot(indices, gp1_energy_stds, 'g-', linewidth=2.5, 
                        marker='o', markersize=6, label='Energy Std (GP1 Predictions)')
        line2 = ax2.plot(indices, gp1_force_stds, 'orange', linewidth=2.5, 
                         marker='s', markersize=6, label='Force Std (GP1 Predictions)')
        
        # Labels and formatting
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Energy Std Dev (eV)', color='g')
        ax2.set_ylabel('Force Std Dev (eV/Å)', color='orange')
        ax.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        
        ax.set_title('GP1 Prediction Variability Along Path')
        ax.grid(True, alpha=0.3)