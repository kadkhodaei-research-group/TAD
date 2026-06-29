import pickle
import os
import time
import numpy as np
import logging
from typing import Dict, Any, Optional
import shutil

class CheckpointManager:
    """Manages saving and loading of checkpoints for continuation runs."""
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        if checkpoint_dir is None:
            # Use the output manager if available
            try:
                from output_manager import get_output_path
                self.checkpoint_dir = get_output_path('checkpoints')
            except ImportError:
                # Fallback to relative path if output_manager not available
                self.checkpoint_dir = 'checkpoints'
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_walker_state(self, walker, iteration: int, additional_data: Optional[Dict] = None):
        """Save complete walker state for continuation."""
        
        # Detect walker type
        walker_type = walker.__class__.__name__
        
        if walker_type == 'WalkerGP2':
            self._save_gp2_walker_state(walker, iteration, additional_data)
        elif walker_type == 'WalkerPureDimer':
            self._save_pure_dimer_state(walker, iteration, additional_data)
        else:
            self._save_full_walker_state(walker, iteration, additional_data)

    def _save_pure_dimer_state(self, walker, iteration: int, additional_data: Optional[Dict] = None):
        """Save WalkerPureDimer state - CLEAN VERSION without moving atom references."""
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        
        checkpoint_data = {
            # Basic info
            'walker_type': 'WalkerPureDimer',
            'iteration': iteration,
            'timestamp': timestamp,
            'converged': walker.converged,
            
            # Walker state
            'steps': walker.steps,
            'energy_reference': walker.energy_reference,
            'reference_set': walker.reference_set,
            
            # Full system info
            'n_atoms': walker.n_atoms,
            'n_dof': walker.n_dof,
            
            # Trajectory - full system positions, energies, and forces
            'trajectory': walker.trajectory,
            
            # VASP evaluation tracking
            'vasp_eval_count': walker.vasp_eval_count,
            'force_evals_per_step': walker.force_evals_per_step,
            
            # Verbose output history
            'table_history': walker.table_history,
            
            # Dimer state - full system
            'dimer_state': {
                'x': walker.dimer.x.copy(),  # Full system position
                'orient': walker.dimer.orient.copy() if walker.dimer.orient is not None else None,
                'dimer_stopping_criteria': walker.dimer.dimer_stopping_criteria,
                'rotation_method': walker.dimer.rotation_method,
                'translation_method': walker.dimer.translation_method,
                'dimer_sep': walker.dimer.dimer_sep,
                'T_anglerot': walker.dimer.T_anglerot,
                'T_anglerot_init': walker.dimer.T_anglerot_init,
                'max_dimer_rotations': walker.dimer.max_dimer_rotations,
            },
            
            # Step size parameters
            'step_size': walker.step_size,
            'max_step_size': walker.max_step_size,
        }
        
        # Save picklable parts of dimer internal state
        if hasattr(walker.dimer, 'rotinfo'):
            rotinfo_copy = {}
            for key, value in walker.dimer.rotinfo.items():
                try:
                    pickle.dumps(value)
                    rotinfo_copy[key] = value
                except:
                    pass
            checkpoint_data['dimer_state']['rotinfo'] = rotinfo_copy
        
        if hasattr(walker.dimer, 'transinfo'):
            transinfo_copy = {}
            for key, value in walker.dimer.transinfo.items():
                if key == 'potential':  # Skip function
                    continue
                try:
                    pickle.dumps(value)
                    transinfo_copy[key] = value
                except:
                    pass
            checkpoint_data['dimer_state']['transinfo'] = transinfo_copy
        
        # Add any additional data
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Save with timestamp
        filename = f'{self.checkpoint_dir}/pure_dimer_checkpoint_{iteration}_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Update latest symlink
        latest_link = f'{self.checkpoint_dir}/pure_dimer_latest.pkl'
        if os.path.exists(latest_link):
            os.remove(latest_link)
        shutil.copy2(filename, latest_link)
        
        self.logger.info(f"Saved pure dimer checkpoint at iteration {iteration}")
    
    def _save_gp2_walker_state(self, walker, iteration: int, additional_data: Optional[Dict] = None):
        """Save WalkerGP2 state (GP2 only, no thermal sampling)."""
        checkpoint = {
            # Basic info
            'iteration': iteration,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
            'walker_type': 'WalkerGP2',
            'converged': walker.converged,
            
            # Walker state
            'steps': walker.steps,
            'energy_reference': walker.energy_reference,
            'reference_set': walker.reference_set,
            'min_force_achieved': getattr(walker, 'min_force_achieved', float('inf')),
            
            # Full position trajectory
            'full_position_trajectory': walker.full_position_trajectory.copy() if hasattr(walker, 'full_position_trajectory') else [],

            # Verbose history for GP2
            'gp2_verbose_history': walker.verbose_logger.gp2_history if hasattr(walker.verbose_logger, 'gp2_history') else [],
            
            # GP2 state only
            'gp2_training_data': [
                walker.gp2.training_data[0].copy(),
                walker.gp2.training_data[1].copy(),
                walker.gp2.training_data[2].copy(),
            ],
            'gp2_model_state': walker.gp2.model.state_dict() if walker.gp2.model else None,
            'gp2_likelihood_state': walker.gp2.likelihood.state_dict() if walker.gp2.likelihood else None,
            'gp2_energy_reference': getattr(walker.gp2, 'energy_reference', None),
            
            # Dimer state
            'dimer_x': walker.dimer.x.copy(),
            'dimer_orient': walker.dimer.orient.copy() if hasattr(walker.dimer, 'orient') and walker.dimer.orient is not None else None,
            'dimer_tau': walker.dimer.tau.copy() if hasattr(walker.dimer, 'tau') else None,
            'dimer_angle': walker.dimer.angle if hasattr(walker.dimer, 'angle') else None,
            'dimer_translation_history': getattr(walker.dimer, 'translation_history', None),
            'dimer_rotation_history': getattr(walker.dimer, 'rotation_history', None),
            
            # Atomic info
            'atomic_info': walker.atomic_info.copy(),
            
            # Stopping criteria parameters
            'ratio_at_limit': walker.ratio_at_limit,
            'disp_max': walker.disp_max,
            'divisor_T_dimer_gp': walker.divisor_T_dimer_gp,
            'dimer_stopping_criteria': walker.dimer_stopping_criteria,
            
            # Model type
            'model_type': walker.model_type,
        }
        
        # Add any additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        # Save checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iteration}.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save as latest
        latest_file = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pkl')
        with open(latest_file, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        logging.info(f"Saved WalkerGP2 checkpoint at iteration {iteration}")

    def _save_full_walker_state(self, walker, iteration: int, additional_data: Optional[Dict] = None):
        """Save complete walker state for continuation."""
        checkpoint = {
            # Basic info
            'iteration': iteration,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
            'converged': walker.converged,
            
            # Walker state
            'steps': walker.steps,
            'energy_reference': walker.energy_reference,
            'reference_set': walker.reference_set,
            'min_force_achieved': getattr(walker, 'min_force_achieved', float('inf')),

            # Save full position trajectory
            'full_position_trajectory': walker.full_position_trajectory.copy() if hasattr(walker, 'full_position_trajectory') else [],
            
            # GP1 state
            'gp1_current_location': walker.gp1.current_location.copy(),
            'gp1_model_state': walker.gp1.model.state_dict() if walker.gp1.model else None,
            'gp1_likelihood_state': walker.gp1.likelihood.state_dict() if walker.gp1.likelihood else None,
            'gp1_energy_reference': getattr(walker.gp1, 'energy_reference', None),
            
            # GP2 state
            'gp2_training_data': [
                walker.gp2.training_data[0].copy(),  # positions
                walker.gp2.training_data[1].copy(),  # energies
                walker.gp2.training_data[2].copy(),  # forces
            ],
            'gp2_model_state': walker.gp2.model.state_dict() if walker.gp2.model else None,
            'gp2_likelihood_state': walker.gp2.likelihood.state_dict() if walker.gp2.likelihood else None,
            'gp2_energy_reference': getattr(walker.gp2, 'energy_reference', None),
            
            # Dimer state
            'dimer_x': walker.dimer.x.copy(),
            'dimer_orient': walker.dimer.orient.copy() if hasattr(walker.dimer, 'orient') and walker.dimer.orient is not None else None,
            'dimer_tau': walker.dimer.tau.copy() if hasattr(walker.dimer, 'tau') else None,
            'dimer_angle': walker.dimer.angle if hasattr(walker.dimer, 'angle') else None,
            'dimer_translation_history': getattr(walker.dimer, 'translation_history', None),
            'dimer_rotation_history': getattr(walker.dimer, 'rotation_history', None),
            
            # Thermal noise
            'thermal_noise': walker.thermal_noise,
            
            # Verbose history for table recreation - CRITICAL!
            'verbose_history': walker.verbose_logger.verbose_history if hasattr(walker, 'verbose_logger') else [],
            
            # VASP manager state (if needed)
            'vasp_run_counts': self._get_vasp_counts(walker),
            
            # Atomic info (in case it changed due to activation)
            'atomic_info': walker.atomic_info.copy(),
            
            # Stopping criteria parameters
            'ratio_at_limit': walker.ratio_at_limit,
            'disp_max': walker.disp_max,
            'divisor_T_dimer_gp': walker.divisor_T_dimer_gp,
            'dimer_stopping_criteria': walker.dimer_stopping_criteria,
        }
        
        # Add any additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        # Save checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iteration}.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save as latest
        latest_file = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pkl')
        with open(latest_file, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        logging.info(f"Saved checkpoint at iteration {iteration}")

    def load_latest_checkpoint(self):
        """Load the most recent checkpoint."""
        latest_file = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pkl')
        if os.path.exists(latest_file):
            with open(latest_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logging.info(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
            return checkpoint
        return None
        
    def _get_vasp_counts(self, walker):
        """Extract VASP run counts if available."""
        if hasattr(walker.local_pes, 'vasp_manager'):
            vasp_mgr = walker.local_pes.vasp_manager
            return {
                'thermal_count': vasp_mgr.thermal_count if hasattr(vasp_mgr, 'thermal_count') else 0,
                'dimer_count': vasp_mgr.dimer_count if hasattr(vasp_mgr, 'dimer_count') else 0,
            }
        return {}


def restore_pure_dimer_from_checkpoint(checkpoint, walker):
    """Restore WalkerPureDimer state from checkpoint - CLEAN VERSION."""
    logging.info(f"Restoring WalkerPureDimer from checkpoint at iteration {checkpoint['iteration']}")
    
    # Restore basic state
    walker.steps = checkpoint['steps']
    walker.converged = checkpoint['converged']
    walker.energy_reference = checkpoint['energy_reference']
    walker.reference_set = checkpoint['reference_set']
    
    # Restore system info
    walker.n_atoms = checkpoint['n_atoms']
    walker.n_dof = checkpoint['n_dof']
    
    # Restore trajectory
    walker.trajectory = checkpoint['trajectory']
    
    # Restore VASP tracking
    walker.vasp_eval_count = checkpoint['vasp_eval_count']
    walker.force_evals_per_step = checkpoint['force_evals_per_step']
    
    # Restore table history
    walker.table_history = checkpoint.get('table_history', [])
    
    # Restore step sizes
    walker.step_size = checkpoint.get('step_size', 0.1)
    walker.max_step_size = checkpoint.get('max_step_size', 0.2)
    
    # Restore dimer state
    dimer_state = checkpoint['dimer_state']
    walker.dimer.x = dimer_state['x'].copy()
    if dimer_state.get('orient') is not None:
        walker.dimer.orient = dimer_state['orient'].copy()
    
    # Restore dimer parameters
    walker.dimer.dimer_stopping_criteria = dimer_state.get('dimer_stopping_criteria', 0.01)
    walker.dimer.rotation_method = dimer_state.get('rotation_method', 'lbfgsext')
    walker.dimer.translation_method = dimer_state.get('translation_method', 'lbfgs')
    walker.dimer.dimer_sep = dimer_state.get('dimer_sep', 0.01)
    walker.dimer.T_anglerot = dimer_state.get('T_anglerot', 0.01)
    walker.dimer.T_anglerot_init = dimer_state.get('T_anglerot_init', 0.0873)
    walker.dimer.max_dimer_rotations = dimer_state.get('max_dimer_rotations', 10)
    
    # Restore dimer internal state if available
    if 'rotinfo' in dimer_state and hasattr(walker.dimer, 'rotinfo'):
        for key, value in dimer_state['rotinfo'].items():
            walker.dimer.rotinfo[key] = value
    
    if 'transinfo' in dimer_state and hasattr(walker.dimer, 'transinfo'):
        for key, value in dimer_state['transinfo'].items():
            if key != 'potential':  # Don't overwrite the function
                walker.dimer.transinfo[key] = value
    
    logging.info(f"WalkerPureDimer restored to iteration {walker.steps}")
    logging.info(f"VASP evaluations so far: {walker.vasp_eval_count}")
    
    return walker


def restore_walker_from_checkpoint(checkpoint, walker):
    """Restore walker state from checkpoint."""
    walker_type = checkpoint.get('walker_type', walker.__class__.__name__)
    
    if walker_type == 'WalkerPureDimer':
        return restore_pure_dimer_from_checkpoint(checkpoint, walker)
    elif walker_type == 'WalkerGP2':
        return restore_gp2_walker_from_checkpoint(checkpoint, walker)
    else:
        # Original restore function for full walker
        return restore_full_walker_from_checkpoint(checkpoint, walker)


def restore_full_walker_from_checkpoint(checkpoint, walker):
    """Original restore function for full walker with GP1 and GP2."""
    logging.info(f"Restoring walker from checkpoint at iteration {checkpoint['iteration']}")
    
    # Restore basic state
    walker.steps = checkpoint['steps']
    walker.converged = checkpoint['converged']
    walker.energy_reference = checkpoint['energy_reference']
    walker.reference_set = checkpoint['reference_set']
    walker.min_force_achieved = checkpoint.get('min_force_achieved', float('inf'))
    
    # IMPORTANT: Set a flag to indicate we're restored from checkpoint
    walker._restored_from_checkpoint = True
    
    # Restore stopping criteria parameters
    walker.ratio_at_limit = checkpoint.get('ratio_at_limit', 2.0/3.0)
    walker.disp_max = checkpoint.get('disp_max', 0.5)
    walker.divisor_T_dimer_gp = checkpoint.get('divisor_T_dimer_gp', 10.0)
    walker.dimer_stopping_criteria = checkpoint.get('dimer_stopping_criteria', 0.01)
    
    # Restore GP1 state
    walker.gp1.current_location = checkpoint['gp1_current_location'].copy()
    if checkpoint['gp1_model_state'] and walker.gp1.model:
        walker.gp1.model.load_state_dict(checkpoint['gp1_model_state'])
        walker.gp1.likelihood.load_state_dict(checkpoint['gp1_likelihood_state'])
    
    # CRITICAL: Restore energy reference for GP1
    if 'gp1_energy_reference' in checkpoint:
        walker.gp1.energy_reference = checkpoint['gp1_energy_reference']
    elif walker.energy_reference is not None:
        walker.gp1.energy_reference = walker.energy_reference
    
    # Restore GP2 training data
    walker.gp2.training_data = [
        checkpoint['gp2_training_data'][0].copy(),
        checkpoint['gp2_training_data'][1].copy(),
        checkpoint['gp2_training_data'][2].copy(),
    ]
    
    # CRITICAL: Restore energy reference for GP2
    if 'gp2_energy_reference' in checkpoint:
        walker.gp2.energy_reference = checkpoint['gp2_energy_reference']
    elif walker.energy_reference is not None:
        walker.gp2.energy_reference = walker.energy_reference
    
    # Retrain GP2 with the restored data
    logging.info("Retraining GP2 with restored training data...")
    walker.gp2.train(
        training_data=walker.gp2.training_data,
        thermal_noise=checkpoint['thermal_noise'],
        path=walker.gp2.path,
        model_name="GP2"
    )
    
    # Restore dimer state
    walker.dimer.x = checkpoint['dimer_x'].copy()
    if checkpoint['dimer_tau'] is not None:
        walker.dimer.tau = checkpoint['dimer_tau'].copy()
    if checkpoint['dimer_angle'] is not None:
        walker.dimer.angle = checkpoint['dimer_angle']
    if 'dimer_orient' in checkpoint and checkpoint['dimer_orient'] is not None:
        walker.dimer.orient = checkpoint['dimer_orient'].copy()
    
    # Restore histories
    if checkpoint.get('dimer_translation_history') is not None:
        walker.dimer.translation_history = checkpoint['dimer_translation_history']
    if checkpoint.get('dimer_rotation_history') is not None:
        walker.dimer.rotation_history = checkpoint['dimer_rotation_history']
    
    # CRITICAL: Restore verbose history for table continuity
    if 'verbose_history' in checkpoint:
        if hasattr(walker, 'verbose_logger'):
            walker.verbose_logger.verbose_history = checkpoint['verbose_history']
            print(f"Restored {len(checkpoint['verbose_history'])} verbose history entries to verbose_logger")
        else:
            # If verbose_logger doesn't exist, store it temporarily
            walker._temp_verbose_history = checkpoint['verbose_history']
            print(f"Stored {len(checkpoint['verbose_history'])} verbose history entries temporarily")
    
    # Restore atomic info (in case atoms were activated)
    if 'atomic_info' in checkpoint:
        walker.atomic_info = checkpoint['atomic_info']
        walker.gp1.atomic_info = checkpoint['atomic_info']
        walker.gp2.atomic_info = checkpoint['atomic_info']
    
    # Restore full position trajectory
    if 'full_position_trajectory' in checkpoint:
        walker.full_position_trajectory = checkpoint['full_position_trajectory']
        logging.info(f"Restored {len(walker.full_position_trajectory)} full position trajectory points")
    else:
        # If not in checkpoint, initialize with current dimer position
        walker.full_position_trajectory = [walker.dimer.x.copy()]
        logging.info("Initialized full position trajectory with current dimer position")
    
    # Update VASP manager counts if available
    if hasattr(walker.local_pes, 'vasp_manager') and 'vasp_run_counts' in checkpoint:
        vasp_counts = checkpoint['vasp_run_counts']
        if 'thermal_count' in vasp_counts:
            walker.local_pes.vasp_manager.thermal_count = vasp_counts['thermal_count']
        if 'dimer_count' in vasp_counts:
            walker.local_pes.vasp_manager.dimer_count = vasp_counts['dimer_count']
    
    logging.info(f"Walker restored to iteration {walker.steps}")
    logging.info(f"Energy reference: {walker.energy_reference}")
    logging.info(f"GP1 energy reference: {walker.gp1.energy_reference}")
    logging.info(f"GP2 energy reference: {walker.gp2.energy_reference}")
    
    return walker


def restore_gp2_walker_from_checkpoint(checkpoint, walker):
    """Restore WalkerGP2 state from checkpoint."""
    logging.info(f"Restoring WalkerGP2 from checkpoint at iteration {checkpoint['iteration']}")
    
    # Restore basic state
    walker.steps = checkpoint['steps']
    walker.converged = checkpoint['converged']
    walker.energy_reference = checkpoint['energy_reference']
    walker.reference_set = checkpoint['reference_set']
    walker.min_force_achieved = checkpoint.get('min_force_achieved', float('inf'))
    
    # Set restoration flag
    walker._restored_from_checkpoint = True
    
    # Restore stopping criteria
    walker.ratio_at_limit = checkpoint.get('ratio_at_limit', 2.0/3.0)
    walker.disp_max = checkpoint.get('disp_max', 0.5)
    walker.divisor_T_dimer_gp = checkpoint.get('divisor_T_dimer_gp', 10.0)
    walker.dimer_stopping_criteria = checkpoint.get('dimer_stopping_criteria', 0.01)
    
    # Restore GP2 training data
    walker.gp2.training_data = [
        checkpoint['gp2_training_data'][0].copy(),
        checkpoint['gp2_training_data'][1].copy(),
        checkpoint['gp2_training_data'][2].copy(),
    ]

    # Restore GP2 verbose history
    if 'gp2_verbose_history' in checkpoint and hasattr(walker, 'verbose_logger'):
        walker.verbose_logger.gp2_history = checkpoint['gp2_verbose_history']
        logging.info(f"Restored {len(checkpoint['gp2_verbose_history'])} GP2 verbose history entries")
    
    # Restore energy reference for GP2
    if 'gp2_energy_reference' in checkpoint:
        walker.gp2.energy_reference = checkpoint['gp2_energy_reference']
    elif walker.energy_reference is not None:
        walker.gp2.energy_reference = walker.energy_reference
    
    # Retrain GP2
    logging.info("Retraining GP2 with restored training data...")
    walker.gp2.train(
        training_data=walker.gp2.training_data,
        thermal_noise=None,
        path=walker.gp2.path,
        model_name="GP2"
    )
    
    # Restore dimer state
    walker.dimer.x = checkpoint['dimer_x'].copy()
    if checkpoint.get('dimer_tau') is not None:
        walker.dimer.tau = checkpoint['dimer_tau'].copy()
    if checkpoint.get('dimer_angle') is not None:
        walker.dimer.angle = checkpoint['dimer_angle']
    if checkpoint.get('dimer_orient') is not None:
        walker.dimer.orient = checkpoint['dimer_orient'].copy()
    
    # Restore atomic info
    if 'atomic_info' in checkpoint:
        walker.atomic_info = checkpoint['atomic_info']
        walker.gp2.atomic_info = checkpoint['atomic_info']
    
    # Restore trajectory
    if 'full_position_trajectory' in checkpoint:
        walker.full_position_trajectory = checkpoint['full_position_trajectory']
        logging.info(f"Restored {len(walker.full_position_trajectory)} trajectory points")
    
    logging.info(f"WalkerGP2 restored to iteration {walker.steps}")
    return walker