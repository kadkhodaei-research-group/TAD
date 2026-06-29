"""walker_pure_dimer_clean.py - Pure Dimer Walker with full system forces."""

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


class WalkerPureDimer:
    """Pure dimer method using direct VASP forces for the entire system."""
    
    def __init__(
        self,
        initial_position: npt.NDArray[np.float64],
        local_pes: Any,  # VASP interface
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
        step_size: float = 0.1,
        max_step_size: float = 0.2,
        verbose: bool = False,
        checkpoint_interval: int = 1,
        **kwargs
    ) -> None:
        """Initialize pure dimer walker.
        
        Args:
            initial_position: Initial atomic positions (full system, flattened)
            local_pes: VASP interface for energy/force calculations
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
        
        # System size
        self.n_atoms = len(initial_position) // 3
        self.n_dof = len(initial_position)
        
        # Track evaluations
        self.vasp_eval_count = 0
        self.force_evals_per_step = []
        
        # Set default translation parameters
        if param_trans is None:
            param_trans = np.array([[step_size, max_step_size]])
        
        # Initialize dimer with full system positions
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
        
        # Energy reference (set on first calculation)
        self.energy_reference = None
        self.reference_set = False
        
        # Table history for verbose output
        self.table_history = []
        
        if self.verbose:
            print("\n" + "="*60)
            print("PURE DIMER WALKER INITIALIZED")
            print("="*60)
            print(f"System size: {self.n_atoms} atoms ({self.n_dof} DOF)")
            print(f"Rotation method: {rotation}")
            print(f"Translation method: {translation}")
            print(f"Dimer separation: {dimer_sep} Å")
            print(f"Convergence criteria: {dimer_stopping_criteria} eV/Å")
            print(f"Step size: {step_size} Å (max: {max_step_size} Å)")
            print("="*60 + "\n")
        
        # Initialize checkpoint manager if available
        # self.checkpoint_manager = None
        # try:
        #     from continuation_checkpoint_system import CheckpointManager
        #     self.checkpoint_manager = CheckpointManager()
        # except ImportError:
        #     if self.verbose:
        #         print("CheckpointManager not available - basic checkpointing will be used")
                # Initialize checkpoint manager if available
        self.checkpoint_manager = None
        try:
            from continuation_checkpoint_system import CheckpointManager
            self.checkpoint_manager = CheckpointManager()
            # Test if it can handle WalkerPureDimer
            if not hasattr(self.checkpoint_manager, '_save_pure_dimer_state'):
                # CheckpointManager doesn't have the method we need
                self.checkpoint_manager = None
                if self.verbose:
                    print("CheckpointManager doesn't support WalkerPureDimer - using basic checkpointing")
        except ImportError:
            if self.verbose:
                print("CheckpointManager not available - basic checkpointing will be used")
        except Exception as e:
            # Any other error - fall back to basic checkpointing
            self.checkpoint_manager = None
            if self.verbose:
                print(f"CheckpointManager initialization failed: {e} - using basic checkpointing")
        
        # For backward compatibility with checkpoint system
        self.moving_indices = None
    
    def _force_func(self, position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate forces for the entire system using VASP.
        
        Args:
            position: Full system positions (flattened)
            
        Returns:
            Forces on all atoms (flattened)
        """
        # Get forces from VASP/EAM
        # IMPORTANT: first_derivative now returns FORCES directly, not gradients!
        forces = self.local_pes.first_derivative(position, is_thermal=False)
        self.vasp_eval_count += 1
        
        if self.verbose and self.vasp_eval_count % 10 == 0:
            print(f"  [VASP evaluation #{self.vasp_eval_count}]")
        
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
                print(f"Energy reference set to: {self.energy_reference:.4f} eV")
        
        # Apply reference
        energy_ref = energy - self.energy_reference
        
        # Get forces (now returns actual forces, not gradients)
        forces = self.local_pes.first_derivative(position, is_thermal=False)
        
        # Calculate force magnitude (RMS)
        force_mag = np.sqrt(np.mean(forces**2))
        
        return energy_ref, forces, force_mag
    
    def run(self) -> tuple:
        """Run pure dimer search until convergence or max steps."""
        # Evaluate initial position only if starting fresh
        if self.steps == 0:
            energy, forces, force_mag = self._evaluate_position(self.initial_position)
            self.trajectory.append((self.initial_position.copy(), energy, forces.copy()))
            
            if self.verbose:
                max_force = np.max(np.abs(forces))
                row = f"{self.steps:6d} {energy:12.6f} {force_mag:12.6f} {max_force:12.6f} {'---':>10} {self.vasp_eval_count:8d}"
                self.table_history.append(row)
                # Print initial table
                self._print_full_table()
            
            # If no orientation set, use force-based initialization
            if self.dimer.orient is None:
                forces_3d = forces.reshape(-1, 3)
                force_mags = np.linalg.norm(forces_3d, axis=1)
                max_force_atom = np.argmax(force_mags)
                
                initial_orient = np.zeros(len(forces))
                initial_orient[3*max_force_atom:3*max_force_atom+3] = forces_3d[max_force_atom]
                initial_orient = initial_orient / np.linalg.norm(initial_orient)
                
                self.dimer.set_initial_direction(initial_orient)
                if self.verbose:
                    print(f"Auto-set dimer orientation along force on atom {max_force_atom} (|F|={force_mags[max_force_atom]:.4f} eV/Å)")
        
        orients = []
        
        # Main dimer loop
        while self.steps < self.max_dimer_steps and not self.converged:
            self.steps += 1
            eval_start = self.vasp_eval_count
            
            # Get current position from trajectory
            current_pos = self.trajectory[-1][0]
            self.dimer.x = current_pos.copy()

            # CRITICAL: Don't reset the dimer orientation!
            # The dimer should maintain its orientation from the previous step
            # Only reset if this is the very first step after initialization
            if self.steps == 1 and hasattr(self, '_initial_orient'):
                self.dimer.orient = self._initial_orient
                delattr(self, '_initial_orient')
            
            # Perform one dimer step (rotation + translation)
            step_vector = self.dimer.run()

            # add self.dimer.orient to list of orientations
            orients.append(self.dimer.orient.copy())
            # see if the orientation has changed

            # Update position
            new_pos = self.dimer.x + step_vector
            
            # Evaluate new position
            energy, forces, force_mag = self._evaluate_position(new_pos)
            
            # Get curvature estimate from dimer
            if hasattr(self.dimer, 'Curv'):
                curv = self.dimer.Curv
            else:
                # Estimate from force projections
                if self.dimer.orient is not None:
                    orient = self.dimer.orient.ravel()
                    f_par = np.dot(forces, orient)
                    curv = -f_par / self.dimer.dimer_sep
                else:
                    curv = np.nan
            
            # Store trajectory
            self.trajectory.append((new_pos.copy(), energy, forces.copy()))
            
            # Track evaluations
            evals_this_step = self.vasp_eval_count - eval_start
            self.force_evals_per_step.append(evals_this_step)
            
            # Verbose output
            if self.verbose:
                max_force = np.max(np.abs(forces))
                curv_str = f"{curv:10.4f}" if not np.isnan(curv) else "---"
                row = f"{self.steps:6d} {energy:12.6f} {force_mag:12.6f} {max_force:12.6f} {curv_str} {self.vasp_eval_count:8d}"
                self.table_history.append(row)
                
                # Print full table
                self._print_full_table()
            
            # Check convergence (RMS force)
            if force_mag < self.dimer.dimer_stopping_criteria:
                self.converged = True
                if self.verbose:
                    print("\n" + "="*70)
                    print(f"{'CONVERGED':^70}")
                    print("="*70)
            
            # Save checkpoint
            if self.steps % self.checkpoint_interval == 0:
                self.save_checkpoint()
        
        if not self.converged and self.verbose:
            print("\n" + "="*70)
            print(f"{'MAXIMUM STEPS REACHED':^70}")
            print("="*70)
        
        # Final summary
        if self.verbose:
            self._print_summary()
        
        # Save final checkpoint
        self.save_checkpoint(final=True)
        
        # Return final position, energy, forces
        final_pos = self.trajectory[-1][0]
        final_energy = self.trajectory[-1][1]
        final_forces = self.trajectory[-1][2]
        
        return final_pos, final_energy, final_forces
    
    def _print_full_table(self):
        """Print the full table with header and all history."""
        print("\n" + "="*70)
        print(f"{'Step':>6} {'Energy (eV)':>12} {'RMS Force':>12} {'Max |F|':>12} {'Curv':>10} {'Evals':>8}")
        print("-" * 70)
        
        # Print all history
        for row in self.table_history:
            print(row)
        
        print("-" * 70)
    
    def _print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total steps: {self.steps}")
        print(f"Total VASP evaluations: {self.vasp_eval_count}")
        print(f"Average VASP evals per step: {np.mean(self.force_evals_per_step):.1f}")
        print(f"Status: {'CONVERGED' if self.converged else 'NOT CONVERGED'}")
        
        # Energy/force evolution
        energies = [t[1] for t in self.trajectory]
        forces_rms = [np.sqrt(np.mean(t[2]**2)) for t in self.trajectory]
        forces_max = [np.max(np.abs(t[2])) for t in self.trajectory]
        
        print(f"\nEnergy change: {energies[-1] - energies[0]:.6f} eV")
        print(f"Initial RMS force: {forces_rms[0]:.6f} eV/Å")
        print(f"Final RMS force: {forces_rms[-1]:.6f} eV/Å")
        print(f"Initial max |F|: {forces_max[0]:.6f} eV/Å")
        print(f"Final max |F|: {forces_max[-1]:.6f} eV/Å")
        
        # Displacement
        initial_pos = self.trajectory[0][0]
        final_pos = self.trajectory[-1][0]
        displacement = np.linalg.norm(final_pos - initial_pos)
        max_atom_disp = np.max(np.linalg.norm((final_pos - initial_pos).reshape(-1, 3), axis=1))
        
        print(f"\nTotal displacement (RMS): {displacement/np.sqrt(self.n_atoms):.4f} Å")
        print(f"Maximum atomic displacement: {max_atom_disp:.4f} Å")
        
        # Path length
        path_length = 0.0
        for i in range(1, len(self.trajectory)):
            path_length += np.linalg.norm(self.trajectory[i][0] - self.trajectory[i-1][0])
        print(f"Path length: {path_length:.4f} Å")
        print("="*70)
    
    def save_checkpoint(self, final=False):
        """Save checkpoint data."""
        checkpoint_data = {
            'walker_type': 'WalkerPureDimer',
            'iteration': self.steps,
            'converged': self.converged,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
            'energy_reference': self.energy_reference,
            'reference_set': self.reference_set,
            'trajectory': self.trajectory,
            'vasp_eval_count': self.vasp_eval_count,
            'force_evals_per_step': self.force_evals_per_step,
            'table_history': self.table_history,
            'dimer_state': {
                'x': self.dimer.x.copy(),
                'orient': self.dimer.orient.copy() if self.dimer.orient is not None else None,
                'dimer_stopping_criteria': self.dimer.dimer_stopping_criteria,
                'rotation_method': self.dimer.rotation_method,
                'translation_method': self.dimer.translation_method
            },
            'n_atoms': self.n_atoms,
            'n_dof': self.n_dof,
            'step_size': self.step_size,
            'max_step_size': self.max_step_size
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
                filename = os.path.join(checkpoint_dir, 'pure_dimer_final.pkl')
            else:
                filename = os.path.join(checkpoint_dir, f'pure_dimer_checkpoint_{self.steps}.pkl')
            
            with open(filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Also save as latest
            latest_filename = os.path.join(checkpoint_dir, 'pure_dimer_latest.pkl')
            with open(latest_filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        
        if self.verbose and (self.steps % 10 == 0 or final):
            print(f"  [Checkpoint saved at step {self.steps}]")
    
    def load_checkpoint(self, checkpoint_file=None):
        """Load checkpoint and restore state."""
        if checkpoint_file is None:
            checkpoint_file = get_output_path('checkpoints', 'pure_dimer_latest.pkl')
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state
        self.steps = checkpoint['iteration']
        self.converged = checkpoint['converged']
        self.energy_reference = checkpoint['energy_reference']
        self.reference_set = checkpoint['reference_set']
        self.trajectory = checkpoint['trajectory']
        self.vasp_eval_count = checkpoint['vasp_eval_count']
        self.force_evals_per_step = checkpoint['force_evals_per_step']
        self.table_history = checkpoint.get('table_history', [])
        self.n_atoms = checkpoint.get('n_atoms', self.n_atoms)
        self.n_dof = checkpoint.get('n_dof', self.n_dof)
        
        # Restore dimer state
        dimer_state = checkpoint['dimer_state']
        self.dimer.x = dimer_state['x']
        if dimer_state['orient'] is not None:
            self.dimer.orient = dimer_state['orient']
        
        if self.verbose:
            print(f"\nRestored checkpoint from step {self.steps}")
            print(f"VASP evaluations so far: {self.vasp_eval_count}")
            
            # Print restored table
            if self.table_history:
                self._print_full_table()
        
        return checkpoint