"""walker_pure_dimer_toy.py - Pure Dimer Walker for toy model potentials."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any
import os
import pickle
import time
from dimer import Dimer
from output_manager import get_output_path

logger = logging.getLogger(__name__)


class WalkerPureDimerToy:
    """Pure dimer method for 2D toy model potentials."""
    
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
        max_dimer_rotations: int = 10,
        num_init_rotations: int = 5,
        param_trans: Optional[npt.NDArray[np.float64]] = None,
        dimer_stopping_criteria: float = 0.01,
        step_size: float = 0.05,  # Smaller step size for toy models
        max_step_size: float = 0.1,  # Smaller max step for toy models
        verbose: bool = False,
        checkpoint_interval: int = 1,
        **kwargs
    ) -> None:
        """Initialize pure dimer walker for toy models.
        
        Args:
            initial_position: Initial position (2D for toy models)
            local_pes: ToyModelInterface for energy/force calculations
            max_dimer_steps: Maximum number of dimer steps
            rotation: Rotation method ('lbfgsext', 'lbfgs', 'cg', 'mn')
            translation: Translation method ('lbfgs', 'cg', 'newton', 'qmvv')
            dimer_sep: Dimer separation distance
            T_anglerot: Rotation convergence threshold
            T_anglerot_init: Initial rotation convergence threshold
            max_dimer_rotations: Max rotations per translation
            param_trans: Translation parameters
            dimer_stopping_criteria: Force convergence threshold
            step_size: Base step size for translations
            max_step_size: Maximum allowed step size
            verbose: Enable verbose output
            checkpoint_interval: Save checkpoint every N steps
        """
        self.initial_position = initial_position.copy()
        self.local_pes = local_pes
        self.max_dimer_steps = max_dimer_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.step_size = step_size
        self.max_step_size = max_step_size
        self.dimer_stopping_criteria = dimer_stopping_criteria
        
        # System size (2D for toy models)
        self.n_dof = len(initial_position)
        if self.n_dof != 2:
            raise ValueError(f"Toy models expect 2D positions, got {self.n_dof}D")
        
        # Track evaluations
        self.eval_count = 0
        self.force_evals_per_step = []
        
        # Set default translation parameters
        if param_trans is None:
            param_trans = np.array([[step_size, max_step_size]])
        
        # Initialize dimer with toy model positions
        self.dimer = Dimer(
            x=initial_position,
            force_func=self._force_func,
            dimer_sep=dimer_sep,
            rotation_method=rotation,
            translation=translation,
            opt_type="pure_dimer",
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
        self.steps = 0
        self.converged = False
        self.trajectory = []  # Store (position, energy, force) tuples
        
        # Energy reference (for consistency with other walkers)
        self.energy_reference = None
        self.reference_set = False
        
        # Table history for verbose output
        self.table_history = []
        
        if self.verbose:
            print("\n" + "="*60)
            print("PURE DIMER WALKER FOR TOY MODELS INITIALIZED")
            print("="*60)
            print(f"Potential: {self.local_pes.potential_name}")
            print(f"Domain: {self.local_pes.pes.domain}")
            print(f"Initial position: [{initial_position[0]:.4f}, {initial_position[1]:.4f}]")
            print(f"Rotation method: {rotation}")
            print(f"Translation method: {translation}")
            print(f"Dimer separation: {dimer_sep}")
            print(f"Convergence criteria: {dimer_stopping_criteria}")
            print(f"Step size: {step_size} (max: {max_step_size})")
            print("="*60 + "\n")
    
    def _force_func(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate forces for toy model.
        
        Args:
            position: 2D position
            
        Returns:
            Forces (2D)
        """
        forces = self.local_pes.first_derivative(position, is_thermal=False)
        self.eval_count += 1
        
        if self.verbose and self.eval_count % 20 == 0:
            print(f"  [Evaluation #{self.eval_count}]")
        
        return forces
    
    def _evaluate_position(self, position: npt.NDArray[np.float64]) -> tuple:
        """Evaluate energy and forces at a position.
        
        Returns:
            (energy, forces, force_magnitude)
        """
        # Get energy
        energy = self.local_pes.scaler_y_value(position, is_thermal=False)
        
        # Set reference on first calculation
        if not self.reference_set:
            self.energy_reference = energy
            self.reference_set = True
            if self.verbose:
                print(f"Energy reference set to: {self.energy_reference:.6f}")
        
        # Apply reference
        energy_ref = energy - self.energy_reference
        
        # Get forces
        forces = self.local_pes.first_derivative(position, is_thermal=False)
        
        # Calculate force magnitude
        force_mag = np.linalg.norm(forces)
        
        return energy_ref, forces, force_mag
    
    def run_dimer(self) -> Dict[str, Any]:
        """Run dimer optimization on toy model.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Evaluate initial position
        init_energy, init_forces, init_force_mag = self._evaluate_position(self.initial_position)
        self.trajectory.append((self.initial_position.copy(), init_energy, init_forces.copy()))
        
        if self.verbose:
            print(f"\nInitial state:")
            print(f"  Position: [{self.initial_position[0]:.4f}, {self.initial_position[1]:.4f}]")
            print(f"  Energy: {init_energy:.6f}")
            print(f"  Force magnitude: {init_force_mag:.6f}")
            print(f"  Forces: [{init_forces[0]:.6f}, {init_forces[1]:.6f}]")
            print("\n" + "="*92)
            print(f"{'Step':>4} {'X (Å)':>8} {'Y (Å)':>8} {'Energy (eV)':>12} {'|F| (eV/Å)':>12} {'Fx (eV/Å)':>10} {'Fy (eV/Å)':>10} {'Curv (eV/Å²)':>12} {'Evals':>6}")
            print("="*92)
        
        # Main dimer loop
        while self.steps < self.max_dimer_steps:
            step_start_evals = self.eval_count
            
            # Check for NaN orientation
            if hasattr(self.dimer, 'orient') and self.dimer.orient is not None:
                if np.any(np.isnan(self.dimer.orient)):
                    if self.verbose:
                        print(f"\nWARNING: NaN detected in dimer orientation at step {self.steps}")
                    # Reinitialize with last good position
                    if len(self.trajectory) > 1:
                        last_good_pos = self.trajectory[-2][0]
                        self.dimer.x = last_good_pos.copy()
                        if hasattr(self, '_initial_orient'):
                            self.dimer.orient = self._initial_orient.copy()
                    else:
                        break
            
            # Perform dimer step
            try:
                self.dimer.run()
            except Exception as e:
                logger.error(f"Dimer step failed: {e}")
                if self.verbose:
                    print(f"\nERROR: Dimer step failed at step {self.steps}: {e}")
                break
            
            self.steps += 1
            
            # Get current state
            current_pos = self.dimer.x.copy()
            current_energy, current_forces, current_force_mag = self._evaluate_position(current_pos)
            
            # Store trajectory
            self.trajectory.append((current_pos.copy(), current_energy, current_forces.copy()))
            
            # Track evaluations
            step_evals = self.eval_count - step_start_evals
            self.force_evals_per_step.append(step_evals)
            
            # Get curvature (note: Dimer class uses capital C)
            curvature = getattr(self.dimer, 'Curv', np.nan)
            
            # Store in table history
            self.table_history.append({
                'Step': self.steps,
                'Position_X': current_pos[0],
                'Position_Y': current_pos[1],
                'Energy': current_energy,
                'Force_Mag': current_force_mag,
                'Curvature': curvature,
                'Evaluations': step_evals
            })
            
            # Verbose output - formatted table row
            if self.verbose:
                print(f"{self.steps:4d} {current_pos[0]:8.4f} {current_pos[1]:8.4f} {current_energy:12.6f} "
                      f"{current_force_mag:12.6f} {current_forces[0]:10.4f} {current_forces[1]:10.4f} "
                      f"{curvature:12.3f} {step_evals:6d}")
            
            # Check convergence
            if current_force_mag < self.dimer_stopping_criteria:
                self.converged = True
                if self.verbose:
                    print("="*92)
                    print(f"CONVERGED! Force magnitude {current_force_mag:.6f} eV/Å < {self.dimer_stopping_criteria} eV/Å")
                break
            
            # Save checkpoint (disabled for toy models due to pickle issues)
            # if self.steps % self.checkpoint_interval == 0:
            #     self.save_checkpoint()
        
        # Final evaluation
        final_pos = self.dimer.x.copy()
        final_energy, final_forces, final_force_mag = self._evaluate_position(final_pos)
        
        end_time = time.time()
        
        # Compile results
        results = {
            'converged': self.converged,
            'steps': self.steps,
            'final_position': final_pos,
            'final_energy': final_energy,
            'final_forces': final_forces,
            'final_force_magnitude': final_force_mag,
            'total_evaluations': self.eval_count,
            'runtime': end_time - start_time,
            'trajectory': self.trajectory,
            'force_evals_per_step': self.force_evals_per_step,
            'table_history': self.table_history,
            'potential_info': self.local_pes.get_info()
        }
        
        if hasattr(self.dimer, 'Curv'):
            results['final_curvature'] = self.dimer.Curv
        
        if self.verbose:
            print("\n" + "="*60)
            print("DIMER OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Converged: {self.converged}")
            print(f"Total steps: {self.steps}")
            print(f"Total evaluations: {self.eval_count}")
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
            filename = get_output_path('checkpoints', 'pure_dimer_toy_latest.pkl')
        
        checkpoint_dir = os.path.dirname(filename)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'walker': self,
            'steps': self.steps,
            'converged': self.converged,
            'eval_count': self.eval_count,
            'trajectory': self.trajectory,
            'table_history': self.table_history,
            'dimer_state': {
                'x': self.dimer.x,
                'orient': getattr(self.dimer, 'orient', None),
                'curv': getattr(self.dimer, 'curv', None)
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        if self.verbose:
            print(f"  Checkpoint saved to {filename}")
    
    @classmethod
    def load_checkpoint(cls, filename: str, local_pes: Any) -> 'WalkerPureDimerToy':
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
            walker.dimer.curv = checkpoint['dimer_state']['curv']
        
        return walker