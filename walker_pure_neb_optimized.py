#!/usr/bin/env python
"""
Optimized WalkerPureNEB with incremental checkpointing.
Key improvements:
1. Incremental checkpoint saves (no 3GB rewrites)
2. Trajectory data saved separately in chunks
3. Live monitoring support
4. Backwards compatible
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Import the original walker
from walker_pure_neb import WalkerPureNEB

# Import incremental checkpoint system
from incremental_checkpoint import IncrementalCheckpoint, NEBCheckpointAdapter


class WalkerPureNEBOptimized(WalkerPureNEB):
    """Optimized NEB walker with incremental checkpointing."""
    
    def __init__(self, *args, checkpoint_dir: Optional[str] = None, 
                 use_incremental: bool = True, **kwargs):
        """Initialize optimized walker.
        
        Args:
            use_incremental: If True, use incremental checkpoint system.
                           If False, fall back to original monolithic saves.
        """
        super().__init__(*args, **kwargs)
        
        self.use_incremental = use_incremental
        
        if use_incremental and checkpoint_dir:
            # Use incremental checkpoint system
            self.checkpoint_adapter = NEBCheckpointAdapter(self, checkpoint_dir)
            print("✓ Using incremental checkpoint system")
        else:
            self.checkpoint_adapter = None
            if use_incremental:
                print("⚠ Incremental checkpointing disabled (no checkpoint_dir)")
    
    def save_checkpoint(self, final: bool = False):
        """Save checkpoint using incremental system if available."""
        if self.checkpoint_adapter:
            # Use incremental system
            self.checkpoint_adapter.save(final=final)
            
            # Don't save trajectory to walker (memory optimization)
            # Clear trajectory after saving to prevent memory growth
            if hasattr(self, 'trajectory') and len(self.trajectory) > 100:
                # Keep only last 100 for in-memory operations
                self.trajectory = self.trajectory[-100:]
        else:
            # Fall back to original implementation
            super().save_checkpoint(final=final)
    
    def load_checkpoint(self, checkpoint_file: Optional[str] = None) -> bool:
        """Load checkpoint using incremental system if available."""
        if self.checkpoint_adapter:
            # Use incremental system
            return self.checkpoint_adapter.load()
        else:
            # Fall back to original implementation
            return super().load_checkpoint(checkpoint_file)
    
    def _evaluate_image(self, position: np.ndarray, image_id: int = -1, **kwargs) -> Tuple[float, np.ndarray]:
        """Evaluate single image and optionally save intermediate data."""
        # Call parent implementation
        energy, gradient = super()._evaluate_image(position, image_id, **kwargs)
        
        # Save intermediate results for live monitoring (lightweight)
        if self.checkpoint_adapter and self.steps % 10 == 0:
            # Save current state without full trajectory
            self._save_light_state()
        
        return energy, gradient
    
    def _save_light_state(self):
        """Save lightweight state for live monitoring."""
        if not self.checkpoint_adapter:
            return
        
        # Only save essential current state
        state_data = {
            'walker_type': 'WalkerPureNEB',
            'iteration': self.steps,
            'converged': self.converged,
            'timestamp': time.strftime('%Y%m%d-%H%M%S'),
            'vasp_eval_count': self.vasp_eval_count,
            'neb_state': {
                'R': self.R.copy() if self.R is not None else None,
                'E_R': self.E_R.copy() if self.E_R is not None else None,
                'G_R': self.G_R.copy() if self.G_R is not None else None,
                'CI_on': self.CI_on,
                'i_CI': self.i_CI,
                'n_images': self.n_images,
                'n_atoms': self.n_atoms,
            },
            'parameters': {
                'k_par': self.k_par,
                'k_perp': self.k_perp,
                'T_MEP': self.T_MEP,
                'T_CI': self.T_CI,
                'T_CIon': self.T_CIon,
                'translation_method': self.translation_method
            }
        }
        
        # Save state only (fast)
        self.checkpoint_adapter.checkpoint.save_state(state_data)
    
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run NEB with optimized checkpointing."""
        
        # Try to restore from checkpoint
        if self.checkpoint_adapter:
            restored = self.checkpoint_adapter.load()
            if restored:
                print(f"Restored from incremental checkpoint at step {self.steps}")
        
        # Run the optimization
        result = super().run()
        
        # Final save
        if self.checkpoint_adapter:
            self.save_checkpoint(final=True)
        
        return result


def create_optimized_walker(positions: np.ndarray, 
                           local_pes,
                           output_dir: str,
                           **kwargs) -> WalkerPureNEBOptimized:
    """Factory function to create optimized walker with proper checkpoint setup."""
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create walker with incremental checkpointing
    walker = WalkerPureNEBOptimized(
        positions=positions,
        local_pes=local_pes,
        checkpoint_dir=str(checkpoint_dir),
        use_incremental=True,
        **kwargs
    )
    
    return walker


# Monkey-patch option for existing code
def patch_existing_walker():
    """Patch the existing WalkerPureNEB to use incremental checkpointing.
    
    This allows existing code to benefit from optimizations without changes.
    """
    import walker_pure_neb
    from output_manager import get_output_path
    
    original_save = walker_pure_neb.WalkerPureNEB.save_checkpoint
    original_init = walker_pure_neb.WalkerPureNEB.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        # Set up incremental checkpointing
        try:
            from incremental_checkpoint import NEBCheckpointAdapter
            import os
            
            # Try to get checkpoint dir from output manager, fall back to local
            try:
                checkpoint_dir = get_output_path('checkpoints')
            except:
                # OutputManager not initialized, use local directory
                checkpoint_dir = 'checkpoints'
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._checkpoint_adapter = NEBCheckpointAdapter(self, checkpoint_dir)
            self._checkpoint_dir = checkpoint_dir
            print("✓ Incremental checkpoint system activated")
        except Exception as e:
            self._checkpoint_adapter = None
            self._checkpoint_dir = None
    
    def new_save(self, final=False):
        if hasattr(self, '_checkpoint_adapter') and self._checkpoint_adapter:
            # Use incremental save
            self._checkpoint_adapter.save(final=final)
            
            # Memory optimization: clear large trajectory
            if hasattr(self, 'trajectory') and len(self.trajectory) > 100:
                self.trajectory = self.trajectory[-100:]
                
            # Don't also call original save (avoid double-saving)
            return
            
        # Only use original if incremental not available
        original_save(self, final=final)
    
    original_load = walker_pure_neb.WalkerPureNEB.load_checkpoint
    
    def new_load(self, checkpoint_file=None):
        if hasattr(self, '_checkpoint_adapter') and self._checkpoint_adapter:
            # Use incremental loader
            success = self._checkpoint_adapter.load()
            if success:
                return True
        # Fall back to original
        return original_load(self, checkpoint_file)
    
    # Apply patches
    walker_pure_neb.WalkerPureNEB.__init__ = new_init
    walker_pure_neb.WalkerPureNEB.save_checkpoint = new_save
    walker_pure_neb.WalkerPureNEB.load_checkpoint = new_load


if __name__ == "__main__":
    # Test the incremental checkpoint system
    print("Testing incremental checkpoint system...")
    
    checkpoint_dir = "test_checkpoint"
    ckpt = IncrementalCheckpoint(checkpoint_dir)
    
    # Simulate saving data
    for step in range(5):
        state = {
            'iteration': step,
            'converged': False,
            'energy': np.random.random()
        }
        ckpt.save_state(state)
        
        positions = np.random.random((10, 50, 3))
        energies = np.random.random(10)
        ckpt.append_trajectory(step, positions, energies)
        
        print(f"Saved step {step}")
    
    # Load summary
    summary = ckpt.get_summary()
    print(f"\nCheckpoint summary:")
    print(f"  Total steps: {summary['metadata']['total_steps']}")
    print(f"  Chunks: {summary['num_chunks']}")
    print(f"  Total size: {summary['total_size'] / 1024:.1f} KB")
    
    # Load last 2 steps
    trajectory = ckpt.load_trajectory(last_n_steps=2)
    print(f"\nLoaded last 2 steps: {trajectory['steps']}")
    
    print("\n✓ Test complete!")