#!/usr/bin/env python
"""
Incremental checkpoint system for NEB calculations.
Instead of rewriting 3GB files, this appends new data incrementally.
"""

import pickle
import json
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import gzip
import shutil


class IncrementalCheckpoint:
    """Efficient incremental checkpoint system for NEB."""
    
    def __init__(self, checkpoint_dir: str, prefix: str = "pure_neb"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        
        # File paths
        self.state_file = self.checkpoint_dir / f"{prefix}_state.pkl"
        self.trajectory_file = self.checkpoint_dir / f"{prefix}_trajectory.pkl"
        self.incremental_dir = self.checkpoint_dir / f"{prefix}_incremental"
        self.incremental_dir.mkdir(exist_ok=True)
        
        # Metadata file (small, JSON for easy reading)
        self.metadata_file = self.checkpoint_dir / f"{prefix}_metadata.json"
        
        # Initialize or load existing data
        self.metadata = self._load_metadata()
        self.current_chunk = self.metadata.get('current_chunk', 0)
        
    def _load_metadata(self) -> Dict:
        """Load or initialize metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_chunk': 0,
            'total_steps': 0,
            'chunks': []
        }
    
    def save_state(self, state_data: Dict[str, Any]):
        """Save the current state (small data that changes each step)."""
        # Remove large trajectory data if present
        state_only = state_data.copy()
        state_only.pop('trajectory', None)
        state_only.pop('reference_set', None)  # This can be large too
        
        # Save to temporary file first, then rename (atomic operation)
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            pickle.dump(state_only, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Atomic rename
        temp_file.replace(self.state_file)
        
        # Update metadata
        self.metadata['total_steps'] = state_data.get('iteration', 0)
        self.metadata['last_save'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self._save_metadata()
    
    def append_trajectory(self, step: int, positions: np.ndarray, energies: np.ndarray, 
                         forces: Optional[np.ndarray] = None, **kwargs):
        """Append trajectory data to incremental files."""
        chunk_size = 100  # Save every 100 steps to a new chunk
        chunk_id = step // chunk_size
        
        # Create chunk file path
        chunk_file = self.incremental_dir / f"chunk_{chunk_id:06d}.pkl.gz"
        
        # Load existing chunk or create new
        if chunk_file.exists():
            with gzip.open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
        else:
            chunk_data = {
                'steps': [],
                'positions': [],
                'energies': [],
                'forces': [],
                'extras': []
            }
        
        # Append new data
        chunk_data['steps'].append(step)
        chunk_data['positions'].append(positions.copy())
        chunk_data['energies'].append(energies.copy())
        if forces is not None:
            chunk_data['forces'].append(forces.copy())
        chunk_data['extras'].append(kwargs)
        
        # Save chunk (compressed)
        with gzip.open(chunk_file, 'wb') as f:
            pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update metadata
        if chunk_id not in self.metadata['chunks']:
            self.metadata['chunks'].append(chunk_id)
            self.current_chunk = chunk_id
    
    def append_table_row(self, row: str):
        """Append a row to the progress table (for live monitoring)."""
        table_file = self.checkpoint_dir / f"{self.prefix}_progress.txt"
        with open(table_file, 'a') as f:
            f.write(row + '\n')
    
    def load_state(self) -> Dict[str, Any]:
        """Load the current state."""
        if not self.state_file.exists():
            return {}
        
        with open(self.state_file, 'rb') as f:
            return pickle.load(f)
    
    def load_trajectory(self, last_n_steps: Optional[int] = None) -> Dict[str, List]:
        """Load trajectory data efficiently."""
        trajectory = {
            'steps': [],
            'positions': [],
            'energies': [],
            'forces': [],
            'extras': []
        }
        
        # Load chunks
        chunk_files = sorted(self.incremental_dir.glob("chunk_*.pkl.gz"))
        
        # If only want last N steps, figure out which chunks to load
        if last_n_steps is not None:
            total_steps = self.metadata.get('total_steps', 0)
            start_step = max(0, total_steps - last_n_steps)
            start_chunk = start_step // 100
            chunk_files = [f for f in chunk_files 
                          if int(f.stem.split('_')[1]) >= start_chunk]
        
        # Load relevant chunks
        for chunk_file in chunk_files:
            with gzip.open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            
            # Append to trajectory
            for key in trajectory:
                if key in chunk_data:
                    trajectory[key].extend(chunk_data[key])
        
        # Filter if needed
        if last_n_steps is not None and len(trajectory['steps']) > last_n_steps:
            # Keep only last N
            for key in trajectory:
                if trajectory[key]:
                    trajectory[key] = trajectory[key][-last_n_steps:]
        
        return trajectory
    
    def load_full_checkpoint(self) -> Dict[str, Any]:
        """Load full checkpoint (state + trajectory)."""
        checkpoint = self.load_state()
        checkpoint['trajectory'] = self.load_trajectory()
        return checkpoint
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary without loading all data."""
        summary = {
            'metadata': self.metadata,
            'state_size': self.state_file.stat().st_size if self.state_file.exists() else 0,
            'num_chunks': len(list(self.incremental_dir.glob("chunk_*.pkl.gz"))),
            'total_size': sum(f.stat().st_size for f in self.incremental_dir.glob("*.pkl.gz"))
        }
        return summary
    
    def cleanup_old_chunks(self, keep_last_n: int = 1000):
        """Remove old trajectory chunks to save space."""
        chunk_files = sorted(self.incremental_dir.glob("chunk_*.pkl.gz"))
        
        if len(chunk_files) <= keep_last_n // 100:
            return
        
        # Delete old chunks
        n_to_delete = len(chunk_files) - (keep_last_n // 100)
        for chunk_file in chunk_files[:n_to_delete]:
            chunk_file.unlink()
            chunk_id = int(chunk_file.stem.split('_')[1])
            if chunk_id in self.metadata['chunks']:
                self.metadata['chunks'].remove(chunk_id)
        
        self._save_metadata()


class NEBCheckpointAdapter:
    """Adapter to integrate incremental checkpoints with existing NEB walker."""
    
    def __init__(self, walker, checkpoint_dir: str):
        self.walker = walker
        self.checkpoint = IncrementalCheckpoint(checkpoint_dir)
        
    def save(self, final: bool = False):
        """Save checkpoint using incremental system."""
        # Save state (small data)
        state_data = {
            'walker_type': 'WalkerPureNEB',
            'iteration': self.walker.steps,
            'converged': self.walker.converged,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
            'vasp_eval_count': self.walker.vasp_eval_count,
            'neb_state': {
                'R': self.walker.R.copy(),
                'E_R': self.walker.E_R.copy(),
                'G_R': self.walker.G_R.copy(),
                'CI_on': self.walker.CI_on,
                'i_CI': self.walker.i_CI,
                'n_images': self.walker.n_images,
                'n_atoms': self.walker.n_atoms,
            },
            'energy_reference': self.walker.energy_reference if hasattr(self.walker, 'energy_reference') else 0.0,
            'reference_set': self.walker.reference_set if hasattr(self.walker, 'reference_set') else True,
            'trajectory': [],  # Don't save full trajectory in state
            'force_evals_per_step': self.walker.force_evals_per_step if hasattr(self.walker, 'force_evals_per_step') else [],
            'table_history': self.walker.table_history[-100:] if hasattr(self.walker, 'table_history') else [],  # Last 100 entries
            'neb_path_counter': self.walker.neb_path_counter if hasattr(self.walker, 'neb_path_counter') else self.walker.steps,
            'parameters': {
                'k_par': self.walker.k_par,
                'k_perp': self.walker.k_perp,
                'T_MEP': self.walker.T_MEP,
                'T_CI': self.walker.T_CI,
                'T_CIon': self.walker.T_CIon,
                'translation_method': self.walker.translation_method
            }
        }
        
        self.checkpoint.save_state(state_data)
        
        # Save trajectory incrementally
        if hasattr(self.walker, 'R') and self.walker.R is not None:
            self.checkpoint.append_trajectory(
                step=self.walker.steps,
                positions=self.walker.R,
                energies=self.walker.E_R,
                forces=self.walker.G_R,
                CI_on=self.walker.CI_on,
                i_CI=self.walker.i_CI
            )
        
        # Save table row for live monitoring
        if hasattr(self.walker, 'table_history') and self.walker.table_history:
            if len(self.walker.table_history) > 0:
                self.checkpoint.append_table_row(self.walker.table_history[-1])
        
        if final:
            print(f"\n✓ Final checkpoint saved (incremental)")
            summary = self.checkpoint.get_summary()
            print(f"  State file: {summary['state_size']/1024/1024:.1f} MB")
            print(f"  Trajectory chunks: {summary['num_chunks']}")
            print(f"  Total trajectory size: {summary['total_size']/1024/1024:.1f} MB")
    
    def load(self) -> bool:
        """Load checkpoint if exists."""
        state = self.checkpoint.load_state()
        if not state:
            return False
        
        # Restore walker state
        neb_state = state.get('neb_state', {})
        self.walker.R = neb_state.get('R')
        self.walker.E_R = neb_state.get('E_R')
        self.walker.G_R = neb_state.get('G_R')
        self.walker.CI_on = neb_state.get('CI_on', 0)
        self.walker.i_CI = neb_state.get('i_CI', -1)
        self.walker.steps = state.get('iteration', 0)
        self.walker.converged = state.get('converged', False)
        self.walker.vasp_eval_count = state.get('vasp_eval_count', 0)
        
        # Don't load full trajectory to walker (too big)
        # It can be loaded on-demand for visualization
        
        print(f"✓ Restored checkpoint from step {self.walker.steps}")
        print(f"  VASP evaluations so far: {self.walker.vasp_eval_count}")
        
        return True


# Backwards compatibility wrapper
def create_checkpoint_manager(walker, checkpoint_dir: str):
    """Create a checkpoint manager for the walker."""
    return NEBCheckpointAdapter(walker, checkpoint_dir)