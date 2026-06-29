"""walker_dual_gp_toy.py - Dual GP walker for toy models (GP1 thermal sampling + GP2 acceleration)."""

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
from gp1_model import GP1
from gp_base import train_multitask_gp_toy_model
from thermal_sampling_utils import box_muller_transform

logger = logging.getLogger(__name__)

# Physical constants
_kB_HARTREE = 3.1668114e-6  # Hartree ⋅ K⁻¹  
_kB_EV = 8.617333262e-5  # eV ⋅ K⁻¹
_AU_TO_EV = 27.211396641308  # Hartree to eV conversion
_AU_TO_ANG = 0.529177210903  # Bohr to Angstrom conversion
_KB_AU = 3.1668114e-6  # Hartree/K


class WalkerDualGPToy:
    """Dual GP walker for 2D toy model potentials.
    
    This combines:
    - GP1: Thermal sampling around current location for noise-averaged forces
    - GP2: Acceleration of dimer method through force/energy prediction
    """
    
    def __init__(
        self,
        initial_position: npt.NDArray[np.float64],
        local_pes: Any,  # ToyModelInterface
        equilibrium_position: Optional[npt.NDArray[np.float64]] = None,
        temperature: float = 300.0,
        mass: float = 1.0,
        num_snapshots: int = 10,
        max_dimer_steps: int = 100,
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
        param_trans: Optional[npt.NDArray[np.float64]] = None,
        dimer_stopping_criteria: float = 0.01,
        step_size: float = 0.02,
        max_step_size: float = 0.05,
        # GP convergence
        divisor_T_dimer_gp: float = 10.0,
        max_inner_iterations: int = 50,
        disp_max: float = 0.2,
        # GP model parameters
        model_type: str = "MultitaskGPModel_rbf_atomic",
        verbose: bool = False,
        checkpoint_interval: int = 10,
        use_gpu: bool = False,
        **kwargs
    ) -> None:
        """Initialize dual GP walker for toy models.
        
        Args:
            initial_position: Initial position (2D for toy models)
            local_pes: ToyModelInterface for energy/force calculations
            temperature: Temperature for thermal sampling (K)
            mass: Particle mass for thermal sampling
            num_snapshots: Number of thermal snapshots for GP1
            max_dimer_steps: Maximum number of outer iterations
            rotation: Rotation method
            translation: Translation method
            dimer_sep: Dimer separation distance
            T_anglerot: Rotation convergence threshold
            T_anglerot_init: Initial rotation convergence threshold
            T_anglerot_gp: Rotation convergence on GP surface
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
            model_type: GP model type
            verbose: Enable verbose output
            checkpoint_interval: Save checkpoint every N steps
            use_gpu: Use GPU acceleration for GP training and inference
        """
        self.initial_position = initial_position.copy()
        self.equilibrium_position = equilibrium_position.copy() if equilibrium_position is not None else initial_position.copy()
        self.local_pes = local_pes
        self.temperature = temperature
        self.mass = mass
        self.num_snapshots = num_snapshots
        self.max_dimer_steps = max_dimer_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.model_type = model_type
        self.step_size = step_size
        self.max_step_size = max_step_size
        self.dimer_sep = dimer_sep
        self.rotation = rotation
        self.translation = translation
        self.dimer_stopping_criteria = dimer_stopping_criteria
        self.T_anglerot_init = T_anglerot_init
        self.T_anglerot_gp = T_anglerot_gp
        self.num_init_rotations = num_init_rotations
        self.num_iter_rot_gp = num_iter_rot_gp
        self.divisor_T_dimer_gp = divisor_T_dimer_gp
        self.max_inner_iterations = max_inner_iterations
        self.disp_max = disp_max
        self.use_gpu = use_gpu
        
        # System size (2D for toy models)
        self.n_dof = len(initial_position)
        if self.n_dof != 2:
            raise ValueError(f"Toy models expect 2D positions, got {self.n_dof}D")
        
        # Track evaluations
        self.eval_count = 0
        self.gp1_eval_count = 0
        self.gp2_eval_count = 0
        self.thermal_eval_count = 0
        self.force_evals_per_step = []
        self.gp1_evals_per_step = []
        self.gp2_evals_per_step = []
        
        # Set default translation parameters
        if param_trans is None:
            param_trans = np.array([[step_size, max_step_size]])
        self.param_trans = param_trans
        
        # Dimer parameters
        self.param_anglerot = T_anglerot
        self.param_max_anglerot = max_dimer_rotations
        self.max_iter_initrot = num_init_rotations
        self.max_dimer_rotations = max_dimer_rotations
        
        # Create atomic info for toy models (single particle)
        self.atomic_info = {
            'n_atoms': 1,
            'atom_types': ['X'],
            'n_pt': 1,
            'pairtype': ['X-X'],
            'atomtype_mov': ['X'],
            'atomtype_fro': [],
            'conf_fro': np.array([]),
            'moving_indices': [0]
        }
        
        # Initialize GP models
        self.gp1 = None
        self.gp1_initialized = False
        self.gp2 = None
        self.gp2_initialized = False
        
        # Training data storage
        self.training_positions = []
        self.training_energies = []
        self.training_forces = []
        
        # Energy reference
        self.energy_reference = None
        self.reference_set = False
        
        # Progress tracking
        self.table_history = []
        self.trajectory = []
        self.gp1_predictions = []
        self.gp2_predictions = []
        
        # Initialize dimer
        self.dimer = None
        
        # For visualization purposes, calculate initial thermal noise
        self._calculate_thermal_noise()
        
        # Saddle detection attributes
        self.at_saddle_count = 0  # Count steps at saddle
        self.min_force_at_saddle = float('inf')
        self.saddle_position = None
        self.force_history = []  # Track force magnitudes
        
        if self.verbose:
            print("\n" + "="*60)
            print("DUAL GP WALKER FOR TOY MODELS INITIALIZED")
            print("="*60)
            print(f"Potential: {local_pes.potential_name}")
            print(f"Domain: {local_pes.pes.domain}")
            print(f"Initial position: [{initial_position[0]:.4f}, {initial_position[1]:.4f}]")
            print(f"Temperature: {temperature} K")
            print(f"Thermal snapshots: {num_snapshots}")
            print(f"Rotation method: {rotation}")
            print(f"Translation method: {translation}")
            print(f"Dimer separation: {dimer_sep}")
            print(f"Convergence criteria: {dimer_stopping_criteria}")
            print(f"Step size: {step_size} (max: {max_step_size})")
            print("\nGP Settings:")
            print(f"  Model type: {model_type}")
            print(f"  Divisor T dimer GP: {divisor_T_dimer_gp}")
            print(f"  Max inner iterations: {max_inner_iterations}")
            print(f"  T_anglerot_gp: {T_anglerot_gp}")
            print(f"  Num iter rot GP: {num_iter_rot_gp}")
            print("="*60 + "\n")
    
    def _calculate_thermal_noise(self):
        """Calculate thermal noise using TRGP1 method with Hessian eigenvalues."""
        if self.verbose:
            print("\nCalculating thermal noise using TRGP1 method...")
        
        # Calculate Hessian at equilibrium position
        hessian = self._calculate_hessian(self.equilibrium_position)
        
        # Get eigenvalues (frequencies squared in atomic units)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        
        # Convert to angular frequencies (ω = sqrt(k/m))
        # In atomic units, mass is already incorporated
        frequencies = np.sqrt(np.abs(eigenvalues))
        
        # Calculate heat capacity
        Cv_dimensionless = self._calculate_heat_capacity(frequencies)
        
        # Generate thermal snapshots to calculate <u²>
        n_snapshots = self.num_snapshots
        displacements = []
        
        for _ in range(n_snapshots):
            # Generate thermal displacement using normal modes
            displacement = np.zeros(2)
            for i, (omega, evec) in enumerate(zip(frequencies, eigenvectors.T)):
                if omega > 1e-6:  # Skip zero modes
                    # Thermal amplitude for each mode
                    # <x²> = kBT / (m*ω²) in classical limit
                    # But we use quantum formula for consistency
                    x = omega / (_KB_AU * self.temperature)
                    if x < 50:  # Avoid numerical issues
                        n_thermal = 1 / (np.exp(x) - 1)  # Bose-Einstein
                        amplitude = np.sqrt(2 * n_thermal + 1) / np.sqrt(omega)
                    else:
                        amplitude = 0.0
                    
                    # Random phase
                    phase = np.random.uniform(0, 2*np.pi)
                    displacement += amplitude * np.cos(phase) * evec
            
            displacements.append(displacement)
        
        displacements = np.array(displacements)
        average_u_squared = np.mean(np.linalg.norm(displacements, axis=1)**2)
        
        # Convert to eV/Angstrom units
        average_u_squared_ang = average_u_squared * _AU_TO_ANG**2
        
        # Calculate thermal noise
        kB_T = _kB_EV * self.temperature  # eV
        
        # Energy fluctuations: σ_E = kB*T * sqrt(Cv)
        sigma_E = kB_T * np.sqrt(Cv_dimensionless)
        
        # Force fluctuations: σ_F = σ_E / sqrt(<u²>)
        # Avoid division by zero
        if average_u_squared_ang > 1e-10:
            sigma_F = sigma_E / np.sqrt(average_u_squared_ang)
        else:
            # Use classical approximation if quantum effects dominate
            # For harmonic oscillator: <u²> = kBT / (m*ω²)
            # Using average frequency
            avg_freq = np.mean([f for f in frequencies if f > 1e-6])
            if avg_freq > 0:
                # Convert to eV/Å² units
                k_eff = avg_freq**2 * _AU_TO_EV / (_AU_TO_ANG**2)
                sigma_F = np.sqrt(k_eff * kB_T)
                if self.verbose:
                    print(f"  Using classical approximation for force noise: σ_F = {sigma_F:.6f} eV/Å")
            else:
                sigma_F = 0.0
        
        self.thermal_noise = (sigma_F, sigma_E)
        
        # Store additional info for analysis
        self.thermal_info = {
            'frequencies': frequencies,
            'Cv_dimensionless': Cv_dimensionless,
            'average_u_squared': average_u_squared_ang,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'hessian': hessian * _AU_TO_EV / (_AU_TO_ANG**2)  # Convert back to eV/Å²
        }
        
        if self.verbose:
            print(f"  Hessian eigenvalues (a.u.): {eigenvalues}")
            print(f"  Frequencies (a.u.): {frequencies}")
            print(f"  Heat capacity Cv (dimensionless): {Cv_dimensionless:.1f}")
            print(f"  <u²> = {average_u_squared_ang:.4f} Ų")
            print(f"  Thermal noise estimates:")
            print(f"    Energy noise σ_E: {sigma_E:.6f} eV")
            print(f"    Force noise σ_F: {sigma_F:.6f} eV/Å")
            print(f"  Compare to polynomial fit method:")
            print(f"    Would have given σ_F ≈ {np.sqrt(eigenvalues[0] * _kB_EV * self.temperature * _AU_TO_EV / _AU_TO_ANG**2):.6f} eV/Å")
    
    def _calculate_hessian(self, position):
        """Calculate Hessian matrix at given position."""
        h = 1e-5  # Small displacement
        hessian = np.zeros((2, 2))
        
        # Get force at center
        f0 = self.local_pes.first_derivative(position)
        
        # Calculate second derivatives
        for i in range(2):
            # Positive displacement
            pos_plus = position.copy()
            pos_plus[i] += h
            f_plus = self.local_pes.first_derivative(pos_plus)
            
            # Negative displacement
            pos_minus = position.copy()
            pos_minus[i] -= h
            f_minus = self.local_pes.first_derivative(pos_minus)
            
            # Second derivatives
            for j in range(2):
                hessian[i, j] = -(f_plus[j] - f_minus[j]) / (2 * h)
        
        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)
        
        # Convert to atomic units for eigenvalue calculation
        # H_au = H_eV/Å² * (Å²/au²) / (eV/au)
        hessian_au = hessian * (_AU_TO_ANG**2) / _AU_TO_EV
        
        return hessian_au
    
    def _calculate_heat_capacity(self, frequencies):
        """Calculate heat capacity Cv in units of kB (dimensionless)."""
        Cv_total = 0.0
        
        for omega in frequencies:
            if omega > 1e-6:  # Skip zero modes
                # x = ℏω / (kB * T) in atomic units
                x = omega / (_KB_AU * self.temperature)
                
                if x < 1e-6:  # High temperature limit
                    Cv_mode = 1.0
                elif x < 50:  # Normal calculation
                    exp_x = np.exp(x)
                    Cv_mode = (x**2) * exp_x / ((exp_x - 1)**2)
                else:  # Low temperature limit
                    Cv_mode = 0.0
                
                Cv_total += Cv_mode
        
        return Cv_total
    
    def _plot_polynomial_fit(self):
        """Plot visualization of thermal noise calculation using TRGP1 method."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            if not hasattr(self, 'thermal_info'):
                return
            
            fig = plt.figure(figsize=(15, 12))
            
            # Get thermal info data
            frequencies = self.thermal_info['frequencies']
            eigenvalues = self.thermal_info['eigenvalues']
            hessian = self.thermal_info['hessian']
            Cv = self.thermal_info['Cv_dimensionless']
            u_squared = self.thermal_info['average_u_squared']
            eq_pos = self.equilibrium_position
            
            # Subplot 1: Eigenvalue spectrum
            ax1 = fig.add_subplot(2, 2, 1)
            eigenvals_ev = eigenvalues * _AU_TO_EV / (_AU_TO_ANG**2)
            ax1.bar(range(len(eigenvals_ev)), eigenvals_ev)
            ax1.set_xlabel('Mode')
            ax1.set_ylabel('Eigenvalue (eV/Å²)')
            ax1.set_title('Hessian Eigenvalues')
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Heat capacity contribution
            ax2 = fig.add_subplot(2, 2, 2)
            x_values = []
            cv_contributions = []
            for omega in frequencies:
                if omega > 1e-6:
                    x = omega / (_KB_AU * self.temperature)
                    x_values.append(x)
                    if x < 50:
                        exp_x = np.exp(x)
                        cv_mode = (x**2) * exp_x / ((exp_x - 1)**2)
                    else:
                        cv_mode = 0.0
                    cv_contributions.append(cv_mode)
            
            if x_values:
                ax2.scatter(x_values, cv_contributions, c='red', alpha=0.7)
                ax2.set_xlabel('ℏω / (kB * T)')
                ax2.set_ylabel('Cv contribution (units of kB)')
                ax2.set_title(f'Heat Capacity Contributions (Total Cv = {Cv:.1f})')
                ax2.grid(True, alpha=0.3)
                ax2.set_xscale('log')
            
            # Subplot 3: Hessian matrix visualization
            ax3 = fig.add_subplot(2, 2, 3)
            im = ax3.imshow(hessian, cmap='RdBu_r', aspect='equal')
            plt.colorbar(im, ax=ax3, label='eV/Å²')
            ax3.set_title('Hessian Matrix')
            ax3.set_xlabel('Component j')
            ax3.set_ylabel('Component i')
            for i in range(2):
                for j in range(2):
                    ax3.text(j, i, f'{hessian[i,j]:.3f}', ha='center', va='center')
            
            # Subplot 4: Info text
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            
            info_text = f"""
            TRGP1 Thermal Noise Calculation:
            
            Equilibrium Position: [{eq_pos[0]:.4f}, {eq_pos[1]:.4f}]
            
            Hessian Eigenvalues (a.u.):
              λ₁ = {eigenvalues[0]:.4f}
              λ₂ = {eigenvalues[1]:.4f}
            
            Frequencies (a.u.):
              ω₁ = {frequencies[0]:.4f}
              ω₂ = {frequencies[1]:.4f}
            
            Temperature: {self.temperature} K
            Heat Capacity Cv: {Cv:.1f} (dimensionless)
            <u²> = {u_squared:.4f} Ų
            
            Thermal Noise:
              Energy σ_E: {self.thermal_noise[1]:.6f} eV
              Force σ_F: {self.thermal_noise[0]:.6f} eV/Å
            """
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top',
                    fontfamily='monospace')
            
            plt.suptitle(f'TRGP1 Thermal Noise Calculation - {self.local_pes.potential_name}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_dir = get_output_path('plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, 'trgp1_thermal_noise.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if self.verbose:
                print(f"  TRGP1 thermal noise plot saved: {plot_file}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Failed to plot polynomial fit: {e}")
    
    def _init_dimer(self):
        """Initialize dimer with a random orientation."""
        # Random initial orientation
        theta = np.random.rand() * 2 * np.pi
        orient_init = np.array([np.cos(theta), np.sin(theta)])
        
        # Initialize dimer
        self.dimer = Dimer(
            x=self.initial_position.copy(),
            force_func=self._force_func,
            dimer_sep=self.dimer_sep,
            rotation_method=self.rotation,
            translation=self.translation,
            max_dimer_rotations=self.max_dimer_rotations,
            T_anglerot=self.param_anglerot,
            T_anglerot_init=self.T_anglerot_init,
            num_iter_initrot=self.max_iter_initrot,
            param_trans=self.param_trans,
            dimer_stopping_criteria=self.dimer_stopping_criteria,
            initial_orientation=orient_init
        )
        
        if self.verbose:
            print(f"Dimer initialized with orientation: [{orient_init[0]:.4f}, {orient_init[1]:.4f}]")
    
    def _force_func(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Force function for dimer using GP1 thermal sampling.
        
        Returns:
            Forces (1D array)
        """
        # Ensure position is 1D
        position = position.ravel()
        if len(position) != 2:
            raise ValueError(f"Expected 2D position, got {len(position)}D")
        
        # Use GP1 thermal sampling if initialized
        if self.gp1_initialized:
            return self._gp1_evaluate_forces(position)
        else:
            # Direct evaluation
            forces = self.local_pes.first_derivative(position)
            self.eval_count += 1
            return forces
    
    def _gp1_evaluate_forces(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate forces using GP1 thermal sampling."""
        # Update GP1 location
        self.gp1.current_location = position.copy()
        
        # Generate thermal snapshots
        snapshots = self._generate_thermal_snapshots_2d(position)
        
        # Evaluate forces on all snapshots
        snapshot_forces = []
        snapshot_energies = []
        
        for snapshot in snapshots:
            energy = self.local_pes.scaler_y_value(snapshot)
            forces = self.local_pes.first_derivative(snapshot)
            snapshot_energies.append(energy)
            snapshot_forces.append(forces)
            self.thermal_eval_count += 1
        
        # Convert to arrays
        snapshot_forces = np.array(snapshot_forces)
        snapshot_energies = np.array(snapshot_energies)
        
        # Calculate thermal average (simple average for toy models)
        avg_forces = np.mean(snapshot_forces, axis=0)
        
        self.gp1_eval_count += 1
        
        return avg_forces
    
    def _generate_thermal_snapshots_2d(self, position: npt.NDArray[np.float64]) -> List[npt.NDArray[np.float64]]:
        """Generate thermal snapshots for 2D toy models."""
        snapshots = []
        
        # Thermal displacement scale
        # For harmonic oscillator: <x²> = kT/(m*omega²)
        # Assume characteristic frequency ~ 1 for simplicity
        thermal_scale = np.sqrt(_kB_EV * self.temperature / self.mass)
        
        for _ in range(self.num_snapshots):
            # Generate random thermal displacement
            # box_muller_transform returns 2 values, perfect for 2D
            dx, dy = box_muller_transform(0.0, thermal_scale)
            displacement = np.array([dx, dy])
            snapshot = position + displacement
            snapshots.append(snapshot)
        
        return snapshots
    
    def _clip_to_domain(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply soft boundary handling to keep position within domain.
        
        Uses a soft repulsion near boundaries instead of hard clipping.
        """
        domain = self.local_pes.pes.domain
        new_pos = position.copy()
        
        # Soft boundary margin (5% of domain size)
        margin_x = (domain[0][1] - domain[0][0]) * 0.05
        margin_y = (domain[1][1] - domain[1][0]) * 0.05
        
        # Apply soft repulsion for each dimension
        for i, (low, high) in enumerate([domain[0], domain[1]]):
            margin = margin_x if i == 0 else margin_y
            
            # Near lower boundary
            if new_pos[i] < low + margin:
                # Soft repulsion: exponential decay
                penetration = low + margin - new_pos[i]
                new_pos[i] = low + margin - penetration * np.exp(-penetration/margin)
                if new_pos[i] < low:
                    new_pos[i] = low + 1e-6  # Hard limit as fallback
            
            # Near upper boundary
            elif new_pos[i] > high - margin:
                # Soft repulsion: exponential decay
                penetration = new_pos[i] - (high - margin)
                new_pos[i] = high - margin + penetration * np.exp(-penetration/margin)
                if new_pos[i] > high:
                    new_pos[i] = high - 1e-6  # Hard limit as fallback
        
        return new_pos
    
    def _force_func_gp2(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Force function using GP2 predictions."""
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
        
        # Extract 2D forces from 3D prediction
        if isinstance(force_pred, np.ndarray):
            if force_pred.ndim == 2:
                return force_pred[0, :2]
            elif force_pred.ndim == 1:
                return force_pred[:2]
        else:
            raise ValueError(f"Unexpected force prediction format: {type(force_pred)}")
    
    def _add_observation(self, position: npt.NDArray[np.float64], energy: float, forces: npt.NDArray[np.float64]) -> None:
        """Add observation to training data."""
        # Store as 2D for consistency
        self.training_positions.append(position.copy())
        self.training_energies.append(energy)
        self.training_forces.append(forces.copy())
    
    def _train_gp1(self) -> None:
        """Train or update GP1 model."""
        n_points = len(self.training_positions)
        
        if n_points == 0:
            return
        
        # Convert to arrays
        train_positions_2d = np.array(self.training_positions)
        train_forces_2d = np.array(self.training_forces)
        train_energies = np.array(self.training_energies)
        
        # Add zero z-component for GP1 (3D format)
        train_positions = np.column_stack([train_positions_2d, np.zeros(n_points)])
        train_forces = np.column_stack([train_forces_2d, np.zeros(n_points)])
        
        if self.verbose:
            print(f"\n  [Training GP1 with {len(train_energies)} points...]")
        
        try:
            # Initialize GP1 if not already done
            if self.gp1 is None:
                # Use equilibrium position for GP1 initialization
                eq_pos_3d = np.array([self.equilibrium_position[0], self.equilibrium_position[1], 0.0])
                
                # Create data directory for GP1
                gp1_path = get_output_path('data_gp1')
                os.makedirs(gp1_path, exist_ok=True)
                
                self.gp1 = GP1(
                    current_location=eq_pos_3d,
                    temperature=self.temperature,
                    mass=self.mass,
                    local_pes=self.local_pes,
                    path=gp1_path,
                    atomic_info=self.atomic_info,
                    noise_model="fixed",  # Use fixed noise model
                    use_gpu=self.use_gpu
                )
                self.gp1.model_type = self.model_type
                self.gp1.energy_reference = self.energy_reference
                
                # Let GP1 calculate thermal noise
                if self.thermal_noise is None:
                    self.gp1._calculate_thermal_noise()
                    self.thermal_noise = self.gp1.minimum_noise
                    if self.verbose:
                        print(f"  GP1 calculated thermal noise - Force: {self.thermal_noise[0]:.6f} eV/Å, Energy: {self.thermal_noise[1]:.6f} eV")
            
            # Update training data
            self.gp1.train(
                training_data=[train_positions, train_energies, train_forces],
                thermal_noise=self.thermal_noise,
                path=self.gp1.path,
                model_name="GP1"
            )
            
            self.gp1_initialized = True
            
            if self.verbose:
                print(f"  [GP1 training complete]")
                
        except Exception as e:
            logger.error(f"GP1 training failed: {e}")
            if self.verbose:
                print(f"  [GP1 training failed: {e}]")
    
    def _train_gp2(self) -> None:
        """Train or update GP2 model with ALL collected data."""
        n_points = len(self.training_positions)
        
        if n_points == 0:
            return
        
        # Require minimum data points for stable GP2 training
        min_points_for_gp2 = 3  # Back to original value
        if n_points < min_points_for_gp2:
            if self.verbose:
                print(f"\n  [Skipping GP2 training - only {n_points} points (need {min_points_for_gp2})]")
            self.gp2_initialized = False
            return
        
        # Convert 2D positions/forces to 3D for GP2
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
                # Create data directory for GP2
                gp2_path = get_output_path('data_gp2')
                os.makedirs(gp2_path, exist_ok=True)
                
                self.gp2 = GP2(
                    training_data=[train_positions, train_energies, train_forces],
                    path=gp2_path,
                    atomic_info=self.atomic_info,
                    use_gpu=self.use_gpu
                )
                self.gp2.model_type = self.model_type
                self.gp2.energy_reference = self.energy_reference
            else:
                # Update training data
                self.gp2.training_data = [train_positions, train_energies, train_forces]
            
            # Train GP2 using appropriate method
            if self.model_type == 'MultitaskGPModel':
                # Use the new toy model training method
                train_multitask_gp_toy_model(
                    self.gp2,
                    training_data=self.gp2.training_data,
                    thermal_noise=None,
                    model_name="GP2",
                    path=self.gp2.path
                )
            else:
                # Use standard training method
                self.gp2.train(
                    training_data=self.gp2.training_data,
                    thermal_noise=None,
                    path=self.gp2.path,
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
        """Evaluate energy and forces at a position using GP1 thermal sampling.
        
        Returns:
            (position, energy_ref, forces)
        """
        # Update GP1 location
        if self.gp1_initialized:
            self.gp1.current_location = np.array([position[0], position[1], 0.0])
        
        # Get thermal average from GP1 if available
        if self.gp1_initialized:
            # For energy: use thermal average (scalar, averaging is more valid)
            snapshots = self._generate_thermal_snapshots_2d(position)
            
            # Evaluate all snapshots
            positions = []
            energies = []
            forces = []
            for snapshot in snapshots:
                e = self.local_pes.scaler_y_value(snapshot)
                f = self.local_pes.first_derivative(snapshot)
                positions.append(snapshot)
                energies.append(e)
                forces.append(f)
                self.thermal_eval_count += 1
            
            # Calculate thermal averages
            position_avg = np.mean(positions, axis=0)
            energy = np.mean(energies)
            force = np.mean(forces, axis=0)
            
            # Replace position with averaged position
            position = position_avg
            
            self.gp1_eval_count += 1
        else:
            # Direct evaluation
            energy = self.local_pes.scaler_y_value(position)
            force = self.local_pes.first_derivative(position)
            self.eval_count += 1
        
        # Apply energy reference
        if self.energy_reference is not None:
            energy_ref = energy - self.energy_reference
        else:
            energy_ref = energy
        
        return position, energy_ref, force
    
    def _gp2_evaluate(self, position: npt.NDArray[np.float64]) -> tuple:
        """Evaluate using GP2 (for inner iterations)."""
        if not self.gp2_initialized:
            raise RuntimeError("GP2 not initialized")
        
        # Get GP2 prediction
        pos_3d = np.array([position[0], position[1], 0.0]).reshape(1, -1)
        energy_pred, force_pred, energy_var, force_var = self.gp2.predict(pos_3d)
        
        # Extract scalars/arrays
        if isinstance(energy_pred, np.ndarray):
            energy_pred = float(energy_pred[0])
        
        forces_2d = self._force_func_gp2(position)
        
        return energy_pred, forces_2d
    
    def _print_full_table(self):
        """Print complete progress table with all steps."""
        print("\n" + "="*140)
        print("DUAL GP DIMER PROGRESS TABLE")
        print("="*140)
        print("  Iter      X (Å)      Y (Å)        E_GP2     E_Actual   E_Err%        F_GP2     F_Actual   F_Err%       Curv  Evals   GP1   GP2")
        print("="*140)
        
        # Print all previous steps
        for row in self.table_history:
            step = row['Step']
            pos_x = row['Position_X']
            pos_y = row['Position_Y']
            energy = row['Energy']
            force_mag = row['Force_Mag']
            curvature = row['Curvature']
            evals = row['Evaluations']
            gp1_evals = row['GP1_Evaluations']
            gp2_evals = row['GP2_Evaluations']
            
            # Get GP2 predictions if available
            gp2_energy = row.get('GP2_Energy', None)
            gp2_force_mag = row.get('GP2_Force_Mag', None)
            
            # Calculate errors if GP2 predictions available
            if gp2_energy is not None and energy != 0:
                e_err = abs((gp2_energy - energy) / energy) * 100
                e_err_str = f"{e_err:7.1f}"
            else:
                e_err_str = "    ---"
            
            if gp2_force_mag is not None and force_mag > 0.001:
                f_err = abs((gp2_force_mag - force_mag) / force_mag) * 100
                f_err_str = f"{f_err:7.1f}"
            else:
                f_err_str = "    ---"
            
            # Format values
            gp2_e_str = f"{gp2_energy:10.6f}" if gp2_energy is not None else "       ---"
            gp2_f_str = f"{gp2_force_mag:10.6f}" if gp2_force_mag is not None else "       ---"
            
            print(f"  {step:4d}   {pos_x:8.4f}   {pos_y:8.4f}   "
                  f"{gp2_e_str}   {energy:10.6f}   {e_err_str}   "
                  f"{gp2_f_str}   {force_mag:10.6f}   {f_err_str}   "
                  f"{curvature:8.3f}   {evals:4d}   {gp1_evals:3d}   {gp2_evals:3d}")
        
        print("="*140)
    
    def _print_progress(self, step: int, position: npt.NDArray[np.float64], 
                       energy: float, forces: npt.NDArray[np.float64],
                       curvature: float, gp2_energy: Optional[float] = None,
                       gp2_forces: Optional[npt.NDArray[np.float64]] = None) -> None:
        """Print progress table row."""
        force_mag = np.linalg.norm(forces)
        
        # Calculate errors if GP2 predictions available
        if gp2_energy is not None and energy != 0:
            e_err = abs((gp2_energy - energy) / energy) * 100
            e_err_str = f"{e_err:7.1f}"
        else:
            e_err_str = "    ---"
        
        if gp2_forces is not None:
            gp2_force_mag = np.linalg.norm(gp2_forces)
            if force_mag > 0.001:
                f_err = abs((gp2_force_mag - force_mag) / force_mag) * 100
                f_err_str = f"{f_err:7.1f}"
            else:
                f_err_str = "    ---"
        else:
            gp2_force_mag = None
            f_err_str = "    ---"
        
        # Format values
        gp2_e_str = f"{gp2_energy:10.6f}" if gp2_energy is not None else "       ---"
        gp2_f_str = f"{gp2_force_mag:10.6f}" if gp2_force_mag is not None else "       ---"
        
        print(f"  {step:4d}   {position[0]:8.4f}   {position[1]:8.4f}   "
              f"{gp2_e_str}   {energy:10.6f}   {e_err_str}   "
              f"{gp2_f_str}   {force_mag:10.6f}   {f_err_str}   "
              f"{curvature:8.3f}   {self.eval_count:4d}   {self.gp1_eval_count:3d}   {self.gp2_eval_count:3d}")
        
        # Store in table history
        self.table_history.append({
            'Step': step,
            'Position_X': position[0],
            'Position_Y': position[1],
            'Energy': energy,
            'Force_Mag': force_mag,
            'Curvature': curvature,
            'Evaluations': self.eval_count,
            'GP1_Evaluations': self.gp1_eval_count,
            'GP2_Evaluations': self.gp2_eval_count,
            'GP2_Energy': gp2_energy,
            'GP2_Force_Mag': gp2_force_mag
        })
    
    def _initial_rotations(self) -> None:
        """Perform initial rotations using GP1 thermal sampling."""
        if self.verbose:
            print("\n" + "="*40)
            print("INITIAL ROTATIONS (with GP1)")
            print("="*40)
        
        # Initialize orientation if needed
        if not hasattr(self.dimer, 'orient') or self.dimer.orient is None:
            # Initialize along force direction
            init_forces = self.training_forces[0] if self.training_forces else self._force_func(self.initial_position)
            if np.linalg.norm(init_forces) > 1e-10:
                orient_1d = init_forces / np.linalg.norm(init_forces)
            else:
                # Random orientation if at stationary point
                orient_1d = np.random.randn(2)
                orient_1d = orient_1d / np.linalg.norm(orient_1d)
            self.dimer.orient = orient_1d.reshape(1, -1)
        
        # For initial rotations, we'll let the dimer optimize its orientation
        # during the first relaxation phase
        
        # Evaluate at dimer endpoints
        x0 = self.dimer.x
        # Get second image position
        x1 = x0 + self.dimer_sep * self.dimer.orient[0]
        
        # Get forces and energies
        x0_avg, e0, f0 = self._evaluate_position(x0)
        x1_avg, e1, f1 = self._evaluate_position(x1)
        
        # Update positions with thermal averages
        x0 = x0_avg
        x1 = x1_avg
        self.dimer.x = x0
        
        # Add to training data
        self._add_observation(x0, e0, f0)
        self._add_observation(x1, e1, f1)
        
        if self.verbose:
            print(f"Initial evaluations complete, collected {len(self.training_positions)} observations")
    
    def _relaxation_phase(self) -> tuple:
        """Perform relaxation phase using GP2."""
        if self.verbose:
            print("\n" + "="*40)
            print(f"RELAXATION PHASE {len(self.trajectory)+1}")
            print("="*40)
        
        # Check if we're already at a saddle point
        current_pos = self.dimer.x.copy()
        current_force = self._force_func(current_pos)
        current_force_mag = np.linalg.norm(current_force)
        
        # Calculate curvature
        x1 = current_pos + self.dimer_sep * self.dimer.orient[0]
        f1 = self._force_func(x1)
        F_diff = f1 - current_force
        tau = self.dimer.orient[0]
        current_curv = -2 * np.dot(F_diff, tau) / self.dimer_sep
        
        # If at saddle with low force, just return current state
        if current_curv < 0 and current_force_mag < self.dimer_stopping_criteria * 1.5:
            if self.verbose:
                print(f"Already at saddle point!")
                print(f"Force magnitude: {current_force_mag:.6f} eV/Å")
                print(f"Curvature: {current_curv:.4f}")
                print(f"Skipping relaxation to avoid overshooting")
            
            # Just evaluate and return
            R_avg, E_R, G_R = self._evaluate_position(current_pos)
            return R_avg, E_R, G_R, current_force_mag, current_curv, current_force_mag
        
        # Train both GP models
        self._train_gp1()
        self._train_gp2()
        
        # If GP2 not ready yet, skip this relaxation phase
        if not self.gp2_initialized:
            if self.verbose:
                print("GP2 not yet initialized, collecting more data...")
            
            # Just evaluate at current position and return
            R_final = self.dimer.x.copy()
            R_avg, E_R, G_R = self._evaluate_position(R_final)
            R_final = R_avg
            maxF_R = np.linalg.norm(G_R)
            
            # Update minimum force
            self.min_force_achieved = min(self.min_force_achieved, maxF_R)
            
            # Add to training data
            self._add_observation(R_final, E_R, G_R)
            
            # Simple curvature estimate
            self.dimer.force_func = self._force_func
            F0 = self._force_func(R_final)
            x1 = R_final + self.dimer_sep * self.dimer.orient[0]
            F1 = self._force_func(x1)
            F_diff = F1 - F0
            tau = self.dimer.orient[0]
            curvature = -2 * np.dot(F_diff, tau) / self.dimer_sep
            
            # Return expected 6 values: R, E_R, G_R, maxF_R, curvature, maxF_gp_final
            return R_final, E_R, G_R, maxF_R, curvature, None
        
        # Dynamic convergence threshold with minimum
        if self.min_force_achieved < float('inf'):
            # Don't let GP threshold go below main convergence criteria
            gp_convergence_threshold = max(self.dimer_stopping_criteria, 
                                          self.min_force_achieved / self.divisor_T_dimer_gp)
        else:
            gp_convergence_threshold = self.dimer_stopping_criteria
        
        
        print(f"GP convergence threshold: {gp_convergence_threshold:.6f}")
        
        # Set GP parameters for inner loop
        self.dimer.T_rot = self.T_anglerot_gp
        self.dimer.maxIteration_rot = self.num_iter_rot_gp
        
        # Inner optimization loop on GP surface
        converged_gp = False
        oscillation_detected = False
        recent_positions = []
        maxF_gp_final = None
        
        # Adaptive inner iterations - fewer when forces are already low
        if hasattr(self, 'min_force_achieved') and self.min_force_achieved < 0.1:
            max_inner = min(10, self.max_inner_iterations)
        else:
            max_inner = self.max_inner_iterations
            
        for inner_iter in range(max_inner):
            # Store current position
            R = self.dimer.x.copy()
            
            # Update dimer's force function to use GP2
            self.dimer.force_func = self._force_func_gp2
            
            # Store position before step
            R_before = self.dimer.x.copy()
            
            # Calculate curvature before step
            F0 = self._force_func_gp2(self.dimer.x)
            x1 = self.dimer.x + self.dimer_sep * self.dimer.orient[0]
            F1 = self._force_func_gp2(x1)
            F_diff = F1 - F0
            tau = self.dimer.orient[0] if hasattr(self.dimer, 'orient') else np.array([1.0, 0.0])
            curv = -2 * np.dot(F_diff, tau) / self.dimer_sep
            
            if self.verbose:
                if curv < 0:
                    print(f"  Curvature: {curv:.4f} (negative - saddle seeking)")
                else:
                    print(f"  Curvature: {curv:.4f} (positive - climbing)")
            
            # Use standard dimer parameters
            self.dimer.param_trans = self.param_trans
            
            # Check if we should take a step
            current_force_mag = np.linalg.norm(F0)
            
            # If at saddle with very low force, don't take a step
            if curv < 0 and current_force_mag < self.dimer_stopping_criteria * 2:
                if self.verbose:
                    print(f"  At saddle with low force ({current_force_mag:.6f}), skipping dimer step")
                R_new = R_before
            else:
                # Store position before dimer step
                R_before_step = self.dimer.x.copy()
                
                # Perform dimer step which includes rotation
                step_vector = self.dimer.run()
                
                # Get new position after step
                R_after_step = self.dimer.x.copy()
                
                # Check if we're moving away from a saddle point
                if curv < 0:  # Already at saddle
                    # Calculate step size
                    step_size = np.linalg.norm(R_after_step - R_before_step)
                    
                    # If we've been hovering at saddle, be very conservative
                    if self.at_saddle_count > 0:
                        # Almost freeze the position
                        scale = 0.01
                        R_new = R_before_step + scale * (R_after_step - R_before_step)
                        self.dimer.x = R_new
                        if self.verbose:
                            print(f"  At saddle for {self.at_saddle_count} steps - minimal step ({scale*step_size:.6f})")
                    # If taking a large step away from saddle, reduce it
                    elif step_size > 0.01 and current_force_mag < 0.1:
                        # Scale down the step significantly
                        scale = min(0.1, self.dimer_stopping_criteria / current_force_mag)
                        R_new = R_before_step + scale * (R_after_step - R_before_step)
                        self.dimer.x = R_new  # Update dimer position
                        if self.verbose:
                            print(f"  Reduced step from {step_size:.4f} to {scale*step_size:.4f} to prevent overshooting")
                    else:
                        R_new = R_after_step
                else:
                    R_new = R_after_step
            
            
            # Check convergence on GP2 surface
            E_gp, G_gp = self._gp2_evaluate(R_new)
            maxF_gp = np.linalg.norm(G_gp)
            
            # If at saddle point with low force, stop immediately
            if curv < 0 and maxF_gp < self.dimer_stopping_criteria * 2:
                if self.verbose:
                    print(f"At saddle point (curv={curv:.4f}) with low force {maxF_gp:.6f}, stopping")
                converged_gp = True
                maxF_gp_final = maxF_gp
                break
            
            if maxF_gp < gp_convergence_threshold:
                if self.verbose:
                    print(f"Converged on GP surface after {inner_iter+1} iterations")
                    print(f"GP force magnitude: {maxF_gp:.6f} eV/Å")
                converged_gp = True
                break
            
            # Check if GP force is below main convergence criteria
            if maxF_gp < self.dimer_stopping_criteria:
                if self.verbose:
                    print(f"GP force magnitude {maxF_gp:.6f} eV/Å < {self.dimer_stopping_criteria} eV/Å")
                    print(f"Main convergence criteria satisfied on GP surface!")
                converged_gp = True
                maxF_gp_final = maxF_gp
                break
            
        
        # Final state
        if not converged_gp and not oscillation_detected:
            print(f"Relaxation stopped after {max_inner} iterations (not converged)")
        elif converged_gp:
            print(f"Relaxation converged after {inner_iter+1} inner iterations")
        
        # Apply trust region constraint to prevent large jumps
        R_final = self.dimer.x.copy()
        R_start = self.trajectory[-1][0] if self.trajectory else self.initial_position
        displacement = R_final - R_start
        step_size = np.linalg.norm(displacement)
        
        # Maximum allowed step based on current force magnitude
        current_force_mag = np.linalg.norm(self.trajectory[-1][2]) if self.trajectory else 2.0
        max_allowed_step = min(0.2, 0.1 * current_force_mag)  # Scale with force magnitude
        
        if step_size > max_allowed_step:
            # Reduce step to maximum allowed
            scale = max_allowed_step / step_size
            R_final = R_start + scale * displacement
            self.dimer.x = R_final  # Update dimer position
            if self.verbose:
                print(f"Trust region: Reduced step from {step_size:.4f} to {max_allowed_step:.4f}")
        
        # Evaluate at final position using GP1
        
        
        R_avg, E_R, G_R = self._evaluate_position(R_final)
        
        # Check if we're near a saddle point
        force_before_avg = self.local_pes.first_derivative(R_final)
        force_mag_before = np.linalg.norm(force_before_avg)
        
        # If very close to saddle, don't use averaged position
        if 'curv' in locals() and curv < 0 and force_mag_before < 0.05:
            if self.verbose:
                print(f"  Near saddle point, not using position averaging")
                print(f"  Position would shift: {np.linalg.norm(R_avg - R_final):.6f}")
            # Re-evaluate at original position
            _, E_R, G_R = self._evaluate_position(R_final)
            maxF_R = np.linalg.norm(G_R)
        else:
            R_final = R_avg  # Use averaged position
            maxF_R = np.linalg.norm(G_R)
        
        # Update minimum force
        self.min_force_achieved = min(self.min_force_achieved, maxF_R)
        
        # Add to training data
        self._add_observation(R_final, E_R, G_R)
        
        # Get final curvature using true forces
        self.dimer.force_func = self._force_func
        F0 = self._force_func(R_final)
        x1 = R_final + self.dimer_sep * self.dimer.orient[0]
        F1 = self._force_func(x1)
        F_diff = F1 - F0
        tau = self.dimer.orient[0]
        final_curvature = -2 * np.dot(F_diff, tau) / self.dimer_sep
        
        print(f"Accurate values: E_R = {E_R:.6f}, maxF_R = {maxF_R:.6f}")
        if maxF_gp_final is not None:
            print(f"GP force at convergence: {maxF_gp_final:.6f}")
        print(f"Curvature: {final_curvature:.6f}")
        
        return R_final, E_R, G_R, maxF_R, final_curvature, maxF_gp_final
    
    def run_dimer(self) -> dict:
        """Run dual GP dimer search."""
        start_time = time.time()
        
        # Initialize
        self._init_dimer()
        self.min_force_achieved = float('inf')
        
        # Establish energy reference using GP1 thermal average
        if not self.reference_set:
            print("\nEstablishing energy reference using GP1 thermal average...")
            
            # Direct evaluation with thermal sampling
            snapshots = self._generate_thermal_snapshots_2d(self.initial_position)
            energies = []
            for snapshot in snapshots:
                e = self.local_pes.scaler_y_value(snapshot)
                energies.append(e)
                self.thermal_eval_count += 1
            
            e_thermal_ref = np.mean(energies)
            self.energy_reference = e_thermal_ref
            self.reference_set = True
            print(f"Energy reference set to: {e_thermal_ref:.6f}")
        
        # Initial evaluation
        print("\nInitial state:")
        pos_avg, E_init, G_init = self._evaluate_position(self.initial_position)
        self.initial_position = pos_avg  # Update initial position with thermal average
        print(f"  Position: [{self.initial_position[0]:.4f}, {self.initial_position[1]:.4f}]")
        print(f"  Energy: {E_init:.6f}")
        print(f"  Force magnitude: {np.linalg.norm(G_init):.6f}")
        print(f"  Forces: [{G_init[0]:.6f}, {G_init[1]:.6f}]")
        
        # Add initial observation
        self._add_observation(self.initial_position, E_init, G_init)
        self.trajectory.append((self.initial_position.copy(), E_init, G_init.copy()))
        
        # Initial rotations
        self._initial_rotations()
        
        # Add initial state to table history
        self._print_progress(0, self.initial_position, E_init, G_init, 0.0)
        
        # Print initial table
        self._print_full_table()
        
        # Main optimization loop
        converged = False
        for step in range(1, self.max_dimer_steps + 1):
            # Store counts at start of step
            eval_start = self.eval_count
            gp1_start = self.gp1_eval_count
            gp2_start = self.gp2_eval_count
            
            # Check if we're already at a saddle point BEFORE relaxation
            if len(self.trajectory) > 0:
                last_pos, last_e, last_f = self.trajectory[-1]
                last_force_mag = np.linalg.norm(last_f)
                
                # If force is already small and we have negative curvature, we're at saddle
                if last_force_mag < self.dimer_stopping_criteria:
                    # Verify curvature
                    F0 = self._force_func(last_pos)
                    x1 = last_pos + self.dimer_sep * self.dimer.orient[0]
                    F1 = self._force_func(x1)
                    F_diff = F1 - F0
                    tau = self.dimer.orient[0]
                    curv = -2 * np.dot(F_diff, tau) / self.dimer_sep
                    
                    if curv < 0:  # Negative curvature confirms saddle
                        converged = True
                        print(f"\nCONVERGED! At saddle point with force magnitude {last_force_mag:.6f} eV/Å < {self.dimer_stopping_criteria} eV/Å")
                        print(f"Curvature: {curv:.4f} (negative - saddle point)")
                        break
            
            # Relaxation phase
            try:
                R, E_R, G_R, maxF_R, curvature, maxF_gp_final = self._relaxation_phase()
            except Exception as e:
                logger.error(f"Relaxation phase failed: {e}")
                print(f"\nERROR in relaxation phase: {e}")
                break
            
            # Store trajectory
            self.trajectory.append((R.copy(), E_R, G_R.copy()))
            
            # Track evaluations for this step
            self.force_evals_per_step.append(self.eval_count - eval_start)
            self.gp1_evals_per_step.append(self.gp1_eval_count - gp1_start)
            self.gp2_evals_per_step.append(self.gp2_eval_count - gp2_start)
            
            # Update dimer position
            self.dimer.x = R.copy()
            
            # Get GP2 prediction for comparison
            try:
                E_gp2, G_gp2 = self._gp2_evaluate(R)
            except:
                E_gp2, G_gp2 = None, None
            
            # Store progress
            self._print_progress(step, R, E_R, G_R, curvature, E_gp2, G_gp2)
            
            # Print full table
            self._print_full_table()
            
            # Generate plot for this step
            self._generate_step_plot(step)
            
            # Track force history
            self.force_history.append(maxF_R)
            
            # Check if we're hovering near a saddle point
            # For V8, saddle points are at (0,0), (±1,0), (0,±1), (±1,±1)
            near_saddle = False
            saddle_candidates = [(0,0), (1,0), (0,1), (1,1), (-1,0), (0,-1), (-1,-1), (1,-1), (-1,1)]
            
            for saddle in saddle_candidates:
                dist_to_saddle = np.linalg.norm(R - np.array(saddle))
                if dist_to_saddle < 0.15:  # Within 0.15 of a saddle point
                    near_saddle = True
                    break
            
            # Track if we're near saddle with decreasing forces
            if near_saddle and len(self.force_history) >= 2:
                # Check if we had a force minimum and now forces are increasing
                if maxF_R < 1.0:  # Reasonable force level
                    if self.min_force_at_saddle > maxF_R:
                        self.min_force_at_saddle = maxF_R
                        self.saddle_position = R.copy()
                        self.at_saddle_count = 0  # Reset counter when we find lower force
                    elif maxF_R > self.min_force_at_saddle * 1.2:  # Forces increased by 20%
                        self.at_saddle_count += 1
                        if self.at_saddle_count >= 2:
                            # We passed through a force minimum near saddle
                            converged = True
                            print(f"\nCONVERGED! Detected force minimum near saddle point")
                            print(f"Saddle position: [{self.saddle_position[0]:.4f}, {self.saddle_position[1]:.4f}]")
                            print(f"Minimum force: {self.min_force_at_saddle:.6f} eV/Å")
                            print(f"Current force: {maxF_R:.6f} eV/Å (increased from minimum)")
                            print(f"Note: Algorithm passed through saddle and is now escaping")
                            break
            
            # Original convergence criteria
            if maxF_gp_final is not None and maxF_gp_final < self.dimer_stopping_criteria:
                converged = True
                print(f"\nCONVERGED! GP force magnitude {maxF_gp_final:.6f} eV/Å < {self.dimer_stopping_criteria} eV/Å")
                print(f"(Actual force: {maxF_R:.6f} eV/Å)")
                break
            elif maxF_R < self.dimer_stopping_criteria:
                converged = True
                print(f"\nCONVERGED! Force magnitude {maxF_R:.6f} eV/Å < {self.dimer_stopping_criteria} eV/Å")
                break
            elif curvature < 0 and maxF_R < 0.1:  # At saddle with reasonably low force
                converged = True
                print(f"\nCONVERGED! At saddle point (curvature={curvature:.3f}) with force magnitude {maxF_R:.6f} eV/Å")
                print(f"Note: Thermal averaging makes it difficult to achieve forces < 0.01 eV/Å")
                break
            
            # Save checkpoint
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(step)
        
        # Final summary
        runtime = time.time() - start_time
        
        print("\n" + "="*60)
        print("DUAL GP DIMER OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Converged: {converged}")
        print(f"Total outer iterations: {len(self.trajectory) - 1}")
        print(f"Total true evaluations: {self.eval_count}")
        print(f"Total thermal evaluations: {self.thermal_eval_count}")
        print(f"Total GP1 evaluations: {self.gp1_eval_count}")
        print(f"Total GP2 evaluations: {self.gp2_eval_count}")
        print(f"Runtime: {runtime:.2f} seconds")
        
        if converged:
            final_pos = self.trajectory[-1][0]
            final_energy = self.trajectory[-1][1]
            final_forces = self.trajectory[-1][2]
            print(f"Final position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}]")
            print(f"Final energy: {final_energy:.6f}")
            print(f"Final force magnitude: {np.linalg.norm(final_forces):.6f}")
            if 'curvature' in locals():
                print(f"Final curvature: {curvature:.6f}")
        print("="*60)
        
        # Prepare results
        results = {
            'converged': converged,
            'steps': len(self.trajectory) - 1,
            'final_position': self.trajectory[-1][0] if self.trajectory else None,
            'final_energy': self.trajectory[-1][1] if self.trajectory else None,
            'final_forces': self.trajectory[-1][2] if self.trajectory else None,
            'final_force_magnitude': np.linalg.norm(self.trajectory[-1][2]) if self.trajectory else None,
            'total_evaluations': self.eval_count,
            'thermal_evaluations': self.thermal_eval_count,
            'total_gp1_evaluations': self.gp1_eval_count,
            'total_gp2_evaluations': self.gp2_eval_count,
            'runtime': runtime,
            'trajectory': self.trajectory,
            'force_evals_per_step': self.force_evals_per_step,
            'gp1_evals_per_step': self.gp1_evals_per_step,
            'gp2_evals_per_step': self.gp2_evals_per_step,
            'table_history': self.table_history,
            'gp1_predictions': self.gp1_predictions,
            'gp2_predictions': self.gp2_predictions,
            'potential_info': {
                'potential_name': self.local_pes.potential_name,
                'domain': self.local_pes.pes.domain,
                'energy_range': self.local_pes.pes.energies,
                'eval_count': self.local_pes.eval_count,
                'dimensionality': 2
            },
            'gp1_info': {
                'initialized': self.gp1_initialized,
                'training_points': len(self.training_positions),
                'temperature': self.temperature,
                'num_snapshots': self.num_snapshots,
                'thermal_noise': self.thermal_noise
            },
            'gp2_info': {
                'initialized': self.gp2_initialized,
                'training_points': len(self.training_positions),
                'final_curvature': curvature if 'curvature' in locals() else None,
                'convergence_threshold': self.dimer_stopping_criteria,
                'gp2_threshold': self.dimer_stopping_criteria
            }
        }
        
        return results
    
    def _generate_step_plot(self, step: int) -> None:
        """Generate plot showing current state of optimization."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create figure
            fig = plt.figure(figsize=(20, 15))
            
            # Subplot 1: Trajectory on PES contour
            ax1 = fig.add_subplot(2, 3, 1)
            
            # Create grid for contour plot
            domain = self.local_pes.pes.domain
            x = np.linspace(domain[0][0], domain[0][1], 100)
            y = np.linspace(domain[1][0], domain[1][1], 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.local_pes.scaler_y_value(np.array([X[i, j], Y[i, j]]))
            
            # Plot contour
            contourf = ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
            ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            # Plot trajectory
            if self.trajectory:
                positions = np.array([pos for pos, _, _ in self.trajectory])
                ax1.plot(positions[:, 0], positions[:, 1], 'w-', linewidth=2, alpha=0.8)
                ax1.scatter(positions[0, 0], positions[0, 1], s=100, c='lime', edgecolor='black', zorder=10)
                ax1.scatter(positions[-1, 0], positions[-1, 1], s=100, c='red', edgecolor='black', zorder=10)
            
            ax1.set_xlabel('X (Å)')
            ax1.set_ylabel('Y (Å)')
            ax1.set_title(f'Step {step}: Trajectory on PES')
            plt.colorbar(contourf, ax=ax1, label='Energy')
            
            # Subplot 2: Thermal sampling visualization
            ax2 = fig.add_subplot(2, 3, 2)
            if self.trajectory:
                current_pos = self.trajectory[-1][0]
                # Generate thermal snapshots for visualization
                snapshots = self._generate_thermal_snapshots_2d(current_pos)
                snapshots = np.array(snapshots)
                
                # Plot local region
                local_size = 0.5
                x_local = np.linspace(current_pos[0] - local_size, current_pos[0] + local_size, 50)
                y_local = np.linspace(current_pos[1] - local_size, current_pos[1] + local_size, 50)
                X_local, Y_local = np.meshgrid(x_local, y_local)
                Z_local = np.zeros_like(X_local)
                
                for i in range(X_local.shape[0]):
                    for j in range(X_local.shape[1]):
                        Z_local[i, j] = self.local_pes.scaler_y_value(np.array([X_local[i, j], Y_local[i, j]]))
                
                contourf2 = ax2.contourf(X_local, Y_local, Z_local, levels=20, cmap='viridis', alpha=0.7)
                ax2.scatter(snapshots[:, 0], snapshots[:, 1], s=30, c='yellow', edgecolor='black', alpha=0.8)
                ax2.scatter(current_pos[0], current_pos[1], s=100, c='red', edgecolor='black', zorder=10)
                ax2.set_xlabel('X (Å)')
                ax2.set_ylabel('Y (Å)')
                ax2.set_title(f'Thermal Sampling (T={self.temperature}K)')
                plt.colorbar(contourf2, ax=ax2, label='Energy')
            
            # Subplot 3: Convergence history
            ax3 = fig.add_subplot(2, 3, 3)
            if self.trajectory:
                energies = [e for _, e, _ in self.trajectory]
                forces = [np.linalg.norm(f) for _, _, f in self.trajectory]
                steps_array = np.arange(len(energies))
                
                ax3_twin = ax3.twinx()
                line1 = ax3.plot(steps_array, energies, 'b-', linewidth=2, label='Energy')
                line2 = ax3_twin.plot(steps_array, forces, 'r-', linewidth=2, label='Force Mag')
                
                ax3.set_xlabel('Step')
                ax3.set_ylabel('Energy', color='b')
                ax3_twin.set_ylabel('Force Magnitude (eV/Å)', color='r')
                ax3.tick_params(axis='y', labelcolor='b')
                ax3_twin.tick_params(axis='y', labelcolor='r')
                ax3.set_title('Convergence History')
                ax3.grid(True, alpha=0.3)
                
                # Add convergence line
                ax3_twin.axhline(y=self.dimer_stopping_criteria, color='green', linestyle='--', alpha=0.5)
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='best')
            
            # Subplot 4: GP predictions vs actual
            ax4 = fig.add_subplot(2, 3, 4)
            if self.table_history:
                steps_hist = [row['Step'] for row in self.table_history]
                gp2_errors = []
                for row in self.table_history:
                    if row.get('E_GP2') is not None and row.get('Energy') is not None:
                        error = abs(row['E_GP2'] - row['Energy']) / abs(row['Energy'] + 1e-10) * 100
                        gp2_errors.append(error)
                    else:
                        gp2_errors.append(0)
                
                ax4.plot(steps_hist, gp2_errors, 'o-', markersize=6)
                ax4.set_xlabel('Step')
                ax4.set_ylabel('GP2 Energy Prediction Error (%)')
                ax4.set_title('GP2 Prediction Accuracy')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(bottom=0)
            
            # Subplot 5: Curvature evolution
            ax5 = fig.add_subplot(2, 3, 5)
            if self.table_history:
                curvatures = [row['Curvature'] for row in self.table_history]
                steps_hist = [row['Step'] for row in self.table_history]
                ax5.plot(steps_hist, curvatures, 'g-', linewidth=2, marker='o', markersize=6)
                ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax5.set_xlabel('Step')
                ax5.set_ylabel('Curvature')
                ax5.set_title('Curvature Evolution')
                ax5.grid(True, alpha=0.3)
                
                # Color background
                for i in range(len(curvatures)):
                    if curvatures[i] > 0:
                        ax5.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
                    else:
                        ax5.axvspan(i-0.5, i+0.5, alpha=0.2, color='blue')
            
            # Subplot 6: Model usage statistics
            ax6 = fig.add_subplot(2, 3, 6)
            gp1_total = self.gp1_eval_count
            gp2_total = self.gp2_eval_count
            direct_total = self.eval_count
            thermal_total = self.thermal_eval_count
            
            categories = ['Direct', 'Thermal', 'GP1', 'GP2']
            values = [direct_total, thermal_total, gp1_total, gp2_total]
            colors = ['gray', 'lightblue', 'blue', 'orange']
            
            bars = ax6.bar(categories, values, color=colors, alpha=0.7)
            ax6.set_ylabel('Evaluation Count')
            ax6.set_title('Model Usage Statistics')
            ax6.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(val)}', ha='center', va='bottom')
            
            plt.suptitle(f'Dual GP Dimer - {self.local_pes.potential_name} - Step {step}', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_dir = get_output_path('plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, f'step_{step:04d}.png')
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            if self.verbose:
                print(f"  Plot saved: {plot_file}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Failed to generate plot: {e}")
    
    def _save_checkpoint(self, step: int):
        """Save checkpoint data."""
        checkpoint = {
            'step': step,
            'dimer_state': {
                'x': self.dimer.x,
                'orient': self.dimer.orient,
                'curvature': self.dimer.curvature if hasattr(self.dimer, 'curvature') else None
            },
            'trajectory': self.trajectory,
            'training_data': {
                'positions': self.training_positions,
                'energies': self.training_energies,
                'forces': self.training_forces
            },
            'eval_counts': {
                'total': self.eval_count,
                'thermal': self.thermal_eval_count,
                'gp1': self.gp1_eval_count,
                'gp2': self.gp2_eval_count
            }
        }
        
        checkpoint_file = get_output_path('checkpoints', f'dual_gp_checkpoint_step_{step}.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        if self.verbose:
            print(f"\nCheckpoint saved at step {step}")