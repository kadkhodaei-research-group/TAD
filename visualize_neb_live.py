#!/usr/bin/env python
"""
Live NEB visualization that works with incremental checkpoints.
Can be run while NEB is still running to monitor progress.

Key features:
- Reads incremental checkpoint files efficiently
- Shows energy profile evolution
- Displays convergence metrics
- Can be run repeatedly without loading huge files
"""

import numpy as np
import os
# Suppress Qt warnings in WSL
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
import gzip
import argparse
from typing import Optional, Dict, List
import time


def find_latest_run():
    """Find the latest run directory."""
    outputs_dir = Path("../outputs")
    if not outputs_dir.exists():
        outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        return None
        
    run_dirs = sorted([d for d in outputs_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("run_")])
    
    if run_dirs:
        return run_dirs[-1]
    return None


def load_incremental_checkpoint(checkpoint_dir: Path, last_n_steps: Optional[int] = None):
    """Load data from incremental checkpoint efficiently."""
    
    # Check for incremental checkpoint files
    metadata_file = checkpoint_dir / "pure_neb_metadata.json"
    state_file = checkpoint_dir / "pure_neb_state.pkl"
    incremental_dir = checkpoint_dir / "pure_neb_incremental"
    
    # If incremental files don't exist, fall back to legacy
    if not metadata_file.exists() and not state_file.exists():
        print("Using legacy checkpoint format (incremental not active yet)")
        return load_legacy_checkpoint(checkpoint_dir, last_n_steps)
    
    # Load metadata if exists
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'total_steps': 0}
    
    # Load current state
    state_file = checkpoint_dir / "pure_neb_state.pkl"
    if state_file.exists():
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
    else:
        state = {}
    
    # Load trajectory from chunks
    incremental_dir = checkpoint_dir / "pure_neb_incremental"
    trajectory = {
        'steps': [],
        'energies': [],
        'positions': [],
        'forces': [],
        'CI_on': [],
        'i_CI': []
    }
    
    if incremental_dir.exists():
        chunk_files = sorted(incremental_dir.glob("chunk_*.pkl.gz"))
        
        # Determine which chunks to load
        if last_n_steps is not None:
            total_steps = metadata.get('total_steps', 0)
            start_step = max(0, total_steps - last_n_steps)
            start_chunk = start_step // 100
            chunk_files = [f for f in chunk_files 
                          if int(f.stem.split('_')[1]) >= start_chunk]
        
        # Load chunks
        for chunk_file in chunk_files:
            try:
                with gzip.open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
            except (EOFError, gzip.BadGzipFile) as e:
                print(f"Warning: Corrupted chunk file {chunk_file.name}, skipping...")
                continue
            
            trajectory['steps'].extend(chunk_data.get('steps', []))
            trajectory['energies'].extend(chunk_data.get('energies', []))
            trajectory['positions'].extend(chunk_data.get('positions', []))
            trajectory['forces'].extend(chunk_data.get('forces', []))
            
            # Handle extras
            for extra in chunk_data.get('extras', []):
                trajectory['CI_on'].append(extra.get('CI_on', 0))
                trajectory['i_CI'].append(extra.get('i_CI', -1))
    
    # Load progress table
    table_history = []
    progress_file = checkpoint_dir / "pure_neb_progress.txt"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            table_history = f.readlines()
    
    return {
        'metadata': metadata,
        'state': state,
        'trajectory': trajectory,
        'table_history': table_history
    }


def load_legacy_checkpoint(checkpoint_dir: Path, last_n_steps: Optional[int] = None):
    """Load old-style monolithic checkpoint."""
    checkpoint_file = checkpoint_dir / "pure_neb_latest.pkl"
    
    if not checkpoint_file.exists():
        # Try to find any pkl file
        pkl_files = list(checkpoint_dir.glob("*.pkl"))
        if pkl_files:
            checkpoint_file = pkl_files[0]
        else:
            return None
    
    print(f"Loading legacy checkpoint: {checkpoint_file}")
    print(f"File size: {checkpoint_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract trajectory if needed
    if last_n_steps is not None and 'trajectory' in data:
        traj = data['trajectory']
        if len(traj) > last_n_steps:
            data['trajectory'] = traj[-last_n_steps:]
    
    return {
        'metadata': {'total_steps': data.get('iteration', 0)},
        'state': data.get('neb_state', {}),
        'trajectory': {'energies': data.get('trajectory', [])},
        'table_history': data.get('table_history', [])
    }


def plot_neb_evolution(data: Dict, output_file: Optional[str] = None):
    """Create visualization of NEB evolution."""
    
    metadata = data.get('metadata', {})
    state = data.get('state', {})
    trajectory = data.get('trajectory', {})
    
    # Get current state
    current_step = metadata.get('total_steps', 0)
    neb_state = state.get('neb_state', {})
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Current energy profile
    ax1 = plt.subplot(2, 3, 1)
    if 'E_R' in neb_state and neb_state['E_R'] is not None:
        try:
            energies = np.array(neb_state['E_R'])
            # Handle case where energies might be nested arrays
            if energies.ndim > 1:
                energies = energies.flatten()
            
            n_images = len(energies)
            x = np.arange(n_images)
            
            # Normalize energies
            e_min = np.min(energies)
            e_norm = energies - e_min
            
            ax1.plot(x, e_norm, 'bo-', linewidth=2, markersize=8)
            
            # Mark climbing image if active
            i_CI = neb_state.get('i_CI', -1)
            if i_CI >= 0 and i_CI < len(e_norm):
                ax1.plot(i_CI, e_norm[i_CI], 'r*', markersize=15, label=f'CI (img {i_CI})')
            
            # Mark highest energy
            i_max = np.argmax(e_norm)
            max_energy = float(e_norm.flat[i_max])  # Use flat to safely get scalar
            ax1.plot(i_max, max_energy, 'gs', markersize=10, 
                    label=f'Max: {max_energy:.3f} eV')
            
            ax1.set_xlabel('Image Index')
            ax1.set_ylabel('Energy (eV)')
            ax1.set_title(f'Energy Profile at Step {current_step}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error plotting energies:\n{str(e)[:50]}', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('Energy Profile - Error')
    
    # 2. Energy evolution over time
    ax2 = plt.subplot(2, 3, 2)
    if trajectory.get('energies'):
        try:
            # Handle mixed data types in energy history
            energy_history = []
            for e in trajectory['energies']:
                if isinstance(e, (list, np.ndarray)):
                    # If it's an array of energies (multiple images)
                    energy_history.append(np.array(e).flatten())
                else:
                    # Single energy value
                    energy_history.append([float(e) if e is not None else 0.0])
            
            steps = trajectory.get('steps', list(range(len(energy_history))))
            
            if len(energy_history) > 0:
                # Plot max energy evolution
                max_energies = []
                for energies in energy_history:
                    if len(energies) > 1:
                        # Multiple images - calculate barrier
                        e_arr = np.array(energies)
                        max_energies.append(np.max(e_arr) - np.min(e_arr))
                    else:
                        # Single value
                        max_energies.append(float(energies[0]) if len(energies) > 0 else 0.0)
                
                if max_energies:
                    ax2.plot(steps[:len(max_energies)], max_energies, 'b-', label='Barrier Height')
            
                # Mark when CI was activated
                ci_history = trajectory.get('CI_on', [])
                if ci_history:
                    ci_on_steps = [s for s, ci in zip(steps, ci_history) if ci > 0]
                    if ci_on_steps:
                        ax2.axvline(ci_on_steps[0], color='r', linestyle='--', 
                                   alpha=0.5, label='CI Activated')
                
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Barrier Height (eV)')
                ax2.set_title('Barrier Evolution')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error plotting evolution:\n{str(e)[:50]}', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Energy Evolution - Error')
    
    # 3. Force convergence
    ax3 = plt.subplot(2, 3, 3)
    if 'G_R' in neb_state:
        forces = neb_state['G_R']
        force_norms = np.linalg.norm(forces.reshape(len(forces), -1), axis=1)
        
        x = np.arange(len(force_norms))
        ax3.bar(x, force_norms, color='blue', alpha=0.7)
        
        # Mark convergence threshold
        T_MEP = state.get('parameters', {}).get('T_MEP', 0.1)
        ax3.axhline(T_MEP, color='r', linestyle='--', label=f'Threshold: {T_MEP} eV/Å')
        
        # Mark max force
        i_max = np.argmax(force_norms)
        max_force_val = float(force_norms[i_max])
        ax3.bar(i_max, max_force_val, color='red', alpha=0.8)
        ax3.text(i_max, max_force_val, f'{max_force_val:.3f}', 
                ha='center', va='bottom')
        
        ax3.set_xlabel('Image Index')
        ax3.set_ylabel('|Force| (eV/Å)')
        ax3.set_title(f'Force Magnitudes (Max: {float(np.max(force_norms)):.3f})')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Convergence history
    ax4 = plt.subplot(2, 3, 4)
    if trajectory.get('forces'):
        try:
            force_history = trajectory['forces']
            steps = trajectory.get('steps', list(range(len(force_history))))
            
            max_forces = []
            mean_forces = []
            for forces in force_history:
                if forces is not None and len(forces) > 0:
                    try:
                        # Handle different force formats
                        f_array = np.array(forces)
                        if f_array.size > 0:
                            # Reshape only if we have valid data
                            if f_array.ndim == 1:
                                # 1D array - already force magnitudes
                                f_norms = np.abs(f_array)
                            else:
                                # Multi-dimensional - compute norms
                                f_norms = np.linalg.norm(f_array.reshape(-1, 3), axis=1)
                            max_forces.append(float(np.max(f_norms)))
                            mean_forces.append(float(np.mean(f_norms)))
                    except:
                        # Skip problematic entries
                        pass
        
            if max_forces:
                ax4.plot(steps[:len(max_forces)], max_forces, 'r-', label='Max |F|', linewidth=2)
                ax4.plot(steps[:len(mean_forces)], mean_forces, 'b-', label='Mean |F|', linewidth=1)
                
                T_MEP = state.get('parameters', {}).get('T_MEP', 0.1)
                ax4.axhline(T_MEP, color='g', linestyle='--', alpha=0.5, 
                           label=f'Threshold: {T_MEP}')
                
                ax4.set_xlabel('Step')
                ax4.set_ylabel('Force (eV/Å)')
                ax4.set_title('Force Convergence')
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error plotting forces:\n{str(e)[:50]}', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Force Convergence - Error')
    
    # 5. Energy path in 2D (if we have positions)
    ax5 = plt.subplot(2, 3, 5)
    if 'R' in neb_state:
        positions = neb_state['R']
        n_images = len(positions)
        
        # Simple reaction coordinate (distance from first image)
        reaction_coords = []
        for i in range(n_images):
            if i == 0:
                reaction_coords.append(0)
            else:
                dist = np.linalg.norm(positions[i] - positions[0])
                reaction_coords.append(dist)
        
        energies = neb_state.get('E_R', np.zeros(n_images))
        e_norm = energies - np.min(energies)
        
        ax5.plot(reaction_coords, e_norm, 'bo-', linewidth=2, markersize=8)
        ax5.set_xlabel('Reaction Coordinate (Å)')
        ax5.set_ylabel('Energy (eV)')
        ax5.set_title('Energy vs Reaction Coordinate')
        ax5.grid(True, alpha=0.3)
    
    # 6. Step timing and efficiency
    ax6 = plt.subplot(2, 3, 6)
    table_history = data.get('table_history', [])
    if table_history:
        # Parse the last few entries
        recent = table_history[-20:] if len(table_history) > 20 else table_history
        
        info_text = f"Total Steps: {current_step}\n"
        info_text += f"Total Images: {neb_state.get('n_images', 'N/A')}\n"
        info_text += f"Atoms per Image: {neb_state.get('n_atoms', 'N/A')}\n"
        info_text += f"\nRecent Progress:\n"
        info_text += "─" * 40 + "\n"
        
        for line in recent[-10:]:
            info_text += line.strip() + "\n"
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top')
        ax6.axis('off')
        ax6.set_title('Run Information')
    
    # Overall title
    fig.suptitle(f'NEB Evolution - Step {current_step}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_file}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Live NEB visualization')
    parser.add_argument('--run-dir', type=str, help='Run directory path')
    parser.add_argument('--no-watch', action='store_true',
                       help='Disable watch mode (single snapshot only)')
    parser.add_argument('--last-n', type=int, 
                       help='Only load last N steps')
    parser.add_argument('--output', type=str, 
                       help='Output file for plot')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Smart defaults:
    # 1. Always use latest run unless specified
    # 2. Always watch unless --no-watch is given
    # 3. Auto-detect if run is complete
    
    # Find run directory (default to latest)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Default to latest run
        run_dir = find_latest_run()
        if not run_dir:
            # Try current directory
            if (Path.cwd() / "checkpoints").exists():
                run_dir = Path.cwd()
            else:
                print("No run directories found!")
                print("Run from scripts/ directory or specify --run-dir")
                return
        else:
            print(f"Auto-selected latest run: {run_dir.name}")
    
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"No checkpoints directory in {run_dir}")
        return
    
    print(f"Monitoring: {run_dir}")
    
    # Smart detection: Check if run is still active
    data = load_incremental_checkpoint(checkpoint_dir, args.last_n)
    if not data:
        print("No checkpoint data found")
        return
    
    state = data.get('state', {})
    is_converged = state.get('converged', False)
    
    # Decide whether to watch or not
    watch_mode = not args.no_watch and not is_converged
    
    if is_converged:
        print("✓ NEB already converged! Generating final visualization...")
        output_file = args.output or str(run_dir / "neb_final.png")
        plot_neb_evolution(data, output_file)
        print(f"Saved to: {output_file}")
        
    elif watch_mode:
        # Watch mode - update periodically
        print(f"📊 Watch mode activated (updates every {args.interval}s)")
        print("Press Ctrl+C to stop monitoring")
        print("-" * 50)
        
        try:
            while True:
                data = load_incremental_checkpoint(checkpoint_dir, args.last_n)
                if data:
                    output_file = args.output or str(run_dir / "neb_evolution.png")
                    plot_neb_evolution(data, output_file)
                    
                    metadata = data.get('metadata', {})
                    state = data.get('state', {})
                    neb_state = state.get('neb_state', {})
                    
                    # Print status
                    step = metadata.get('total_steps', 0)
                    if 'G_R' in neb_state and neb_state['G_R'] is not None:
                        forces = neb_state['G_R']
                        max_force = np.max(np.linalg.norm(forces.reshape(len(forces), -1), axis=1))
                        print(f"[{time.strftime('%H:%M:%S')}] Step {step:5d} | Max Force: {max_force:.4f} eV/Å", end='\r')
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Step {step:5d}", end='\r')
                    
                    # Check if converged
                    if state.get('converged', False):
                        print(f"\n✓ NEB Converged at step {step}!")
                        break
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\nStopped monitoring")
            print(f"Final plot saved to: {output_file}")
    else:
        # Single snapshot mode
        print("Generating single snapshot...")
        output_file = args.output or str(run_dir / "neb_snapshot.png")
        plot_neb_evolution(data, output_file)
        print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()