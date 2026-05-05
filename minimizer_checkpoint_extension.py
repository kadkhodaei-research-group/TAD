"""Extension to continuation_checkpoint_system.py for WalkerMinimizer support."""

import pickle
import os
import time
import logging
import shutil
from typing import Dict, Optional

# Monkey patch the CheckpointManager to add minimizer support
def patch_checkpoint_manager():
    """Add WalkerMinimizer support to CheckpointManager."""
    from continuation_checkpoint_system import CheckpointManager
    
    # Save the original save_walker_state method
    original_save_walker_state = CheckpointManager.save_walker_state
    
    def new_save_walker_state(self, walker, iteration: int, additional_data: Optional[Dict] = None):
        """Extended save_walker_state that handles WalkerMinimizer."""
        walker_type = walker.__class__.__name__
        
        if walker_type == 'WalkerMinimizer':
            self._save_minimizer_state(walker, iteration, additional_data)
        else:
            # Call original method for other walker types
            original_save_walker_state(self, walker, iteration, additional_data)
    
    def _save_minimizer_state(self, walker, iteration: int, additional_data: Optional[Dict] = None):
        """Save WalkerMinimizer state."""
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        
        checkpoint_data = {
            # Basic info
            'walker_type': 'WalkerMinimizer',
            'iteration': iteration,
            'timestamp': timestamp,
            'converged': walker.converged,
            
            # Walker state
            'steps': walker.steps,
            'energy_reference': walker.energy_reference,
            'reference_set': walker.reference_set,
            
            # System info
            'n_atoms': walker.n_atoms,
            'n_dof': walker.n_dof,
            'method': walker.method,
            
            # Trajectory
            'trajectory': walker.trajectory,
            
            # VASP evaluation tracking
            'vasp_eval_count': walker.vasp_eval_count,
            'force_evals_per_step': walker.force_evals_per_step,
            
            # Verbose output history
            'table_history': walker.table_history,
            
            # Minimizer state
            'minimizer_state': {
                'x': walker.minimizer.x.copy(),
                'method': walker.minimizer.method,
                'step_size': walker.minimizer.step_size,
                'max_step_size': walker.minimizer.max_step_size,
                'stopping_criteria': walker.minimizer.stopping_criteria,
                'line_search': walker.minimizer.line_search,
                'optinfo': walker.minimizer.optinfo.copy(),
                'D': walker.minimizer.D,
            },
        }
        
        # Save observation arrays from minimizer
        if hasattr(walker.minimizer, 'R_all'):
            checkpoint_data['minimizer_state']['R_all'] = walker.minimizer.R_all.copy()
        if hasattr(walker.minimizer, 'E_all'):
            checkpoint_data['minimizer_state']['E_all'] = walker.minimizer.E_all.copy()
        if hasattr(walker.minimizer, 'F_all'):
            checkpoint_data['minimizer_state']['F_all'] = walker.minimizer.F_all.copy()
        
        # Add any additional data
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Save with timestamp
        filename = f'{self.checkpoint_dir}/minimizer_checkpoint_{iteration}_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Update latest symlink
        latest_link = f'{self.checkpoint_dir}/minimizer_latest.pkl'
        if os.path.exists(latest_link):
            os.remove(latest_link)
        shutil.copy2(filename, latest_link)
        
        self.logger.info(f"Saved minimizer checkpoint at iteration {iteration}")
    
    # Attach new methods to CheckpointManager
    CheckpointManager.save_walker_state = new_save_walker_state
    CheckpointManager._save_minimizer_state = _save_minimizer_state


def restore_minimizer_from_checkpoint(checkpoint, walker):
    """Restore WalkerMinimizer state from checkpoint."""
    logging.info(f"Restoring WalkerMinimizer from checkpoint at iteration {checkpoint['iteration']}")
    
    # Restore basic state
    walker.steps = checkpoint['steps']
    walker.converged = checkpoint['converged']
    walker.energy_reference = checkpoint['energy_reference']
    walker.reference_set = checkpoint['reference_set']
    
    # Restore system info
    walker.n_atoms = checkpoint['n_atoms']
    walker.n_dof = checkpoint['n_dof']
    walker.method = checkpoint.get('method', 'lbfgs')
    
    # Restore trajectory
    walker.trajectory = checkpoint['trajectory']
    
    # Restore VASP tracking
    walker.vasp_eval_count = checkpoint['vasp_eval_count']
    walker.force_evals_per_step = checkpoint['force_evals_per_step']
    
    # Restore table history
    walker.table_history = checkpoint.get('table_history', [])
    
    # Restore minimizer state
    min_state = checkpoint['minimizer_state']
    walker.minimizer.x = min_state['x'].copy()
    walker.minimizer.method = min_state['method']
    walker.minimizer.step_size = min_state['step_size']
    walker.minimizer.max_step_size = min_state['max_step_size']
    walker.minimizer.stopping_criteria = min_state['stopping_criteria']
    walker.minimizer.line_search = min_state['line_search']
    walker.minimizer.optinfo = min_state['optinfo'].copy()
    walker.minimizer.D = min_state['D']
    
    # Restore observation arrays
    if 'R_all' in min_state:
        walker.minimizer.R_all = min_state['R_all'].copy()
    if 'E_all' in min_state:
        walker.minimizer.E_all = min_state['E_all'].copy()
    if 'F_all' in min_state:
        walker.minimizer.F_all = min_state['F_all'].copy()
    
    logging.info(f"WalkerMinimizer restored to iteration {walker.steps}")
    logging.info(f"VASP evaluations so far: {walker.vasp_eval_count}")
    
    return walker


# Extend restore_walker_from_checkpoint to handle WalkerMinimizer
def patch_restore_walker_from_checkpoint():
    """Patch the restore function to handle WalkerMinimizer."""
    try:
        import continuation_checkpoint_system
        
        # Get the original function
        original_restore = continuation_checkpoint_system.restore_walker_from_checkpoint
        
        def new_restore_walker_from_checkpoint(checkpoint, walker):
            """Extended restore function that handles WalkerMinimizer."""
            walker_type = checkpoint.get('walker_type', walker.__class__.__name__)
            
            if walker_type == 'WalkerMinimizer':
                return restore_minimizer_from_checkpoint(checkpoint, walker)
            else:
                # Call original restore for other walker types
                return original_restore(checkpoint, walker)
        
        # Replace the function in the module
        continuation_checkpoint_system.restore_walker_from_checkpoint = new_restore_walker_from_checkpoint
        
    except ImportError:
        # If continuation_checkpoint_system doesn't exist, that's ok
        pass


# Apply patches when this module is imported
patch_checkpoint_manager()
patch_restore_walker_from_checkpoint()