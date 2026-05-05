"""walker_minimizer.py - Local Minimum Walker using gradient descent methods."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any
import os
import pickle
import time
from minimizer import Minimizer
from output_manager import get_output_path

logger = logging.getLogger(__name__)


class WalkerMinimizer:
    """Walker for finding local minima using gradient descent methods."""
    
    def __init__(
        self,
        initial_position: npt.NDArray[np.float64],
        local_pes: Any,  # VASP interface
        max_steps: int = 100,
        method: str = "lbfgs",
        step_size: float = 0.1,
        max_step_size: float = 0.2,
        stopping_criteria: float = 0.01,
        line_search: bool = True,
        force_reset_threshold: float = 0.5,
        adaptive_step: bool = True,
        verbose: bool = False,
        checkpoint_interval: int = 1,
        **kwargs
    ) -> None:
        """Initialize minimization walker.
        
        Args:
            initial_position: Initial atomic positions (full system, flattened)
            local_pes: VASP interface for energy/force calculations
            max_steps: Maximum number of minimization steps
            method: Minimization method ('steepest', 'cg', 'lbfgs', 'fire')
            step_size: Base step size
            max_step_size: Maximum allowed step size
            stopping_criteria: Force convergence threshold (eV/Å)
            line_search: Whether to use line search
            verbose: Enable verbose output
            checkpoint_interval: Save checkpoint every N steps
        """
        self.initial_position = initial_position.copy()
        self.local_pes = local_pes
        self.max_steps = max_steps
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.method = method
        
        # System size
        self.n_atoms = len(initial_position) // 3
        self.n_dof = len(initial_position)
        
        # Track evaluations
        self.vasp_eval_count = 0
        self.force_evals_per_step = []
        
        # Initialize minimizer
        self.minimizer = Minimizer(
            x=initial_position,
            force_func=self._force_func,
            method=method,
            step_size=step_size,
            max_step_size=max_step_size,
            stopping_criteria=stopping_criteria,
            line_search=line_search,
            force_reset_threshold=force_reset_threshold,
            adaptive_step=adaptive_step
        )
        
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
            print("LOCAL MINIMIZATION WALKER INITIALIZED")
            print("="*60)
            print(f"System size: {self.n_atoms} atoms ({self.n_dof} DOF)")
            print(f"Minimization method: {method}")
            print(f"Step size: {step_size} Å (max: {max_step_size} Å)")
            print(f"Convergence criteria: {stopping_criteria} eV/Å")
            print(f"Line search: {'enabled' if line_search else 'disabled'}")
            print("="*60 + "\n")
        
        # Initialize checkpoint manager if available
        self.checkpoint_manager = None
        try:
            # Import the extension to patch CheckpointManager
            import minimizer_checkpoint_extension
            from continuation_checkpoint_system import CheckpointManager
            self.checkpoint_manager = CheckpointManager()
        except ImportError:
            if self.verbose:
                print("CheckpointManager not available - basic checkpointing will be used")
        
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
        
        # Get forces
        forces = self.local_pes.first_derivative(position, is_thermal=False)
        
        # Calculate force magnitude (RMS)
        force_mag = np.sqrt(np.mean(forces**2))
        
        return energy_ref, forces, force_mag
    
    def run(self) -> tuple:
        """Run minimization until convergence or max steps."""
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
        
        # Main minimization loop
        while self.steps < self.max_steps and not self.converged:
            self.steps += 1
            eval_start = self.vasp_eval_count
            
            # Get current position from trajectory
            current_pos = self.trajectory[-1][0]
            self.minimizer.x = current_pos.copy()
            
            # Perform one minimization step
            step_vector = self.minimizer.run()
            
            # Update position
            new_pos = self.minimizer.x.copy()
            
            # Evaluate new position
            energy, forces, force_mag = self._evaluate_position(new_pos)
            
            # Calculate step size for display
            step_size = np.linalg.norm(step_vector)
            
            # Store trajectory
            self.trajectory.append((new_pos.copy(), energy, forces.copy()))
            
            # Track evaluations
            evals_this_step = self.vasp_eval_count - eval_start
            self.force_evals_per_step.append(evals_this_step)
            
            # Verbose output
            if self.verbose:
                max_force = np.max(np.abs(forces))
                step_str = f"{step_size:10.4f}"
                row = f"{self.steps:6d} {energy:12.6f} {force_mag:12.6f} {max_force:12.6f} {step_str} {self.vasp_eval_count:8d}"
                self.table_history.append(row)
                
                # Print full table
                self._print_full_table()
            
            # Check convergence (RMS force)
            if force_mag < self.minimizer.stopping_criteria:
                self.converged = True
                if self.verbose:
                    print("\n" + "="*70)
                    print(f"{'CONVERGED TO LOCAL MINIMUM':^70}")
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
        print(f"{'Step':>6} {'Energy (eV)':>12} {'RMS Force':>12} {'Max |F|':>12} {'Step Size':>10} {'Evals':>8}")
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
            'walker_type': 'WalkerMinimizer',
            'iteration': self.steps,
            'converged': self.converged,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
            'energy_reference': self.energy_reference,
            'reference_set': self.reference_set,
            'trajectory': self.trajectory,
            'vasp_eval_count': self.vasp_eval_count,
            'force_evals_per_step': self.force_evals_per_step,
            'table_history': self.table_history,
            'minimizer_state': {
                'x': self.minimizer.x.copy(),
                'method': self.minimizer.method,
                'stopping_criteria': self.minimizer.stopping_criteria,
                'optinfo': self.minimizer.optinfo.copy()
            },
            'n_atoms': self.n_atoms,
            'n_dof': self.n_dof,
            'method': self.method
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
                filename = os.path.join(checkpoint_dir, 'minimizer_final.pkl')
            else:
                filename = os.path.join(checkpoint_dir, f'minimizer_checkpoint_{self.steps}.pkl')
            
            with open(filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Also save as latest
            latest_filename = os.path.join(checkpoint_dir, 'minimizer_latest.pkl')
            with open(latest_filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        
        if self.verbose and (self.steps % 10 == 0 or final):
            print(f"  [Checkpoint saved at step {self.steps}]")
    
    def load_checkpoint(self, checkpoint_file=None):
        """Load checkpoint and restore state."""
        if checkpoint_file is None:
            checkpoint_file = get_output_path('checkpoints', 'minimizer_latest.pkl')
        # Try using the extended checkpoint system first
        try:
            import minimizer_checkpoint_extension
            from continuation_checkpoint_system import restore_walker_from_checkpoint
            
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Use the extended restore function
            restore_walker_from_checkpoint(checkpoint, self)
            
            if self.verbose:
                print(f"\nRestored checkpoint from step {self.steps}")
                print(f"VASP evaluations so far: {self.vasp_eval_count}")
                
                # Print restored table
                if self.table_history:
                    self._print_full_table()
            
            return checkpoint
            
        except ImportError:
            # Fallback to basic checkpoint loading
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Restore state manually
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
            
            # Restore minimizer state
            min_state = checkpoint['minimizer_state']
            self.minimizer.x = min_state['x']
            self.minimizer.optinfo = min_state['optinfo']
            
            if self.verbose:
                print(f"\nRestored checkpoint from step {self.steps}")
                print(f"VASP evaluations so far: {self.vasp_eval_count}")
                
                # Print restored table
                if self.table_history:
                    self._print_full_table()
            
            return checkpoint