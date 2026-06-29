#!/usr/bin/env python
"""
Enhanced NEB evolution visualization showing the complete history.
Features:
- Smart sampling to avoid overcrowding
- Energy profile evolution with heatmap
- Convergence trends
- Interactive elements
"""

import numpy as np
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
import pickle
import gzip
import argparse
import time


def load_full_trajectory(checkpoint_dir: Path):
    """Load complete trajectory from incremental checkpoint."""
    
    print("Loading full trajectory data...")
    
    # Check for incremental checkpoint
    incremental_dir = checkpoint_dir / "pure_neb_incremental"
    state_file = checkpoint_dir / "pure_neb_state.pkl"
    metadata_file = checkpoint_dir / "pure_neb_metadata.json"
    
    # Load metadata
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        total_steps = metadata.get('total_steps', 0)
        print(f"Total steps in checkpoint: {total_steps}")
    else:
        metadata = {}
        total_steps = 0
    
    # Load current state
    if state_file.exists():
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
    else:
        state = {}
    
    # Load all chunks
    trajectory = {
        'steps': [],
        'energies': [],
        'forces': [],
        'barriers': [],
        'max_forces': []
    }
    
    if incremental_dir.exists():
        chunk_files = sorted(incremental_dir.glob("chunk_*.pkl.gz"))
        print(f"Loading {len(chunk_files)} chunks...")
        
        for i, chunk_file in enumerate(chunk_files):
            if i % 10 == 0:
                print(f"  Loading chunk {i}/{len(chunk_files)}...")
            
            try:
                with gzip.open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                # Extract energy barriers for each step
                for j, energies in enumerate(chunk_data.get('energies', [])):
                    step = chunk_data['steps'][j] if 'steps' in chunk_data else len(trajectory['steps'])
                    trajectory['steps'].append(step)
                    
                    # Handle different energy formats
                    if isinstance(energies, (list, np.ndarray)) and len(energies) > 1:
                        e_arr = np.array(energies)
                        barrier = float(np.max(e_arr) - np.min(e_arr))
                        trajectory['barriers'].append(barrier)
                        trajectory['energies'].append(e_arr)
                    else:
                        # Single value or empty
                        trajectory['barriers'].append(0.0)
                        trajectory['energies'].append(energies)
                    
                    # Extract max forces if available
                    if 'forces' in chunk_data and j < len(chunk_data['forces']):
                        forces = chunk_data['forces'][j]
                        if forces is not None and len(forces) > 0:
                            try:
                                f_arr = np.array(forces)
                                if f_arr.size > 0:
                                    if f_arr.ndim > 1:
                                        f_norms = np.linalg.norm(f_arr.reshape(-1, 3), axis=1)
                                    else:
                                        f_norms = np.abs(f_arr)
                                    trajectory['max_forces'].append(float(np.max(f_norms)))
                                else:
                                    trajectory['max_forces'].append(None)
                            except:
                                trajectory['max_forces'].append(None)
                        else:
                            trajectory['max_forces'].append(None)
                    else:
                        trajectory['max_forces'].append(None)
                        
            except Exception as e:
                print(f"  Warning: Error loading {chunk_file.name}: {e}")
                continue
    
    print(f"Loaded {len(trajectory['steps'])} trajectory points")
    
    # Also load from legacy checkpoint if needed
    legacy_file = checkpoint_dir / "pure_neb_latest.pkl"
    if legacy_file.exists() and len(trajectory['steps']) == 0:
        print("Loading from legacy checkpoint...")
        with open(legacy_file, 'rb') as f:
            legacy_data = pickle.load(f)
        
        if 'trajectory' in legacy_data:
            # Process legacy trajectory
            for i, entry in enumerate(legacy_data['trajectory']):
                trajectory['steps'].append(i)
                # Process energy/barrier data
                if isinstance(entry, dict):
                    trajectory['energies'].append(entry.get('energy', 0))
                elif isinstance(entry, tuple) and len(entry) > 0:
                    trajectory['energies'].append(entry[0])
                else:
                    trajectory['energies'].append(0)
                trajectory['barriers'].append(0)  # Will calculate later
    
    return state, metadata, trajectory


def smart_sample(data, max_points=500):
    """Intelligently sample data to avoid overcrowding."""
    n = len(data)
    if n <= max_points:
        return np.arange(n), data
    
    # Use logarithmic sampling - more points near the end
    indices = []
    
    # Always include first and last
    indices.append(0)
    indices.append(n-1)
    
    # Add evenly spaced points
    step = n // (max_points - 2)
    indices.extend(range(step, n-1, step))
    
    # Add more recent points
    recent_start = max(0, n - max_points // 4)
    indices.extend(range(recent_start, n, max(1, (n - recent_start) // (max_points // 4))))
    
    # Sort and remove duplicates
    indices = sorted(set(indices))
    
    return np.array(indices), [data[i] for i in indices if i < len(data)]


def create_evolution_plot(state, metadata, trajectory, output_file=None):
    """Create comprehensive evolution visualization."""
    
    print("\nCreating enhanced evolution visualization...")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    
    # Get current NEB state
    neb_state = state.get('neb_state', {})
    current_step = metadata.get('total_steps', len(trajectory['steps']))
    
    # ========== 1. Energy Profile Evolution (Multiple Profiles) ==========
    ax1 = plt.subplot(3, 4, (1, 2))
    
    if trajectory['energies']:
        # Sample 40 profiles evenly spaced throughout the run
        n_profiles_to_show = 40
        total_steps = len(trajectory['energies'])
        
        if total_steps > n_profiles_to_show:
            # Sample indices evenly across the entire run
            sample_indices = np.linspace(0, total_steps-1, n_profiles_to_show, dtype=int)
        else:
            sample_indices = range(total_steps)
        
        # Create colormap for progression
        cmap = plt.cm.viridis
        
        for idx_num, idx in enumerate(sample_indices):
            e = trajectory['energies'][idx]
            if isinstance(e, (list, np.ndarray)) and len(e) > 1:
                e_arr = np.array(e)
                # Ensure 1D array
                if e_arr.ndim > 1:
                    e_arr = e_arr.flatten()
                
                # Normalize energies
                e_norm = e_arr - np.min(e_arr)
                
                # Calculate color and alpha based on progression
                progress = idx / total_steps
                color = cmap(progress)
                
                # Make early profiles more transparent, later ones more opaque
                alpha = 0.2 + 0.6 * progress
                
                # Line width - thicker for more recent
                linewidth = 0.5 + 1.5 * progress
                
                # Plot
                x = np.arange(len(e_norm))
                step_num = trajectory['steps'][idx] if idx < len(trajectory['steps']) else idx
                
                # Only label first, middle, and last for clarity
                if idx_num == 0:
                    ax1.plot(x, e_norm, color=color, alpha=alpha, linewidth=linewidth,
                            label=f'Step {step_num}')
                elif idx_num == len(sample_indices) // 2:
                    ax1.plot(x, e_norm, color=color, alpha=alpha, linewidth=linewidth,
                            label=f'Step {step_num}')
                elif idx_num == len(sample_indices) - 1:
                    ax1.plot(x, e_norm, color=color, alpha=alpha, linewidth=linewidth,
                            label=f'Step {step_num}')
                else:
                    ax1.plot(x, e_norm, color=color, alpha=alpha, linewidth=linewidth)
        
        # Add colorbar to show progression
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=total_steps))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, label='Step')
        
        ax1.set_xlabel('Image Index')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'Energy Profile Evolution ({n_profiles_to_show} profiles from {total_steps} steps)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
    
    # ========== 2. Current Energy Profile ==========
    ax2 = plt.subplot(3, 4, (3, 4))
    
    if 'E_R' in neb_state and neb_state['E_R'] is not None:
        try:
            energies = np.array(neb_state['E_R'])
            if energies.ndim > 1:
                energies = energies.flatten()
            
            n_images = len(energies)
            x = np.arange(n_images)
            e_norm = energies - np.min(energies)
            
            # Main profile
            ax2.plot(x, e_norm, 'bo-', linewidth=2, markersize=8, label='Current')
            
            # Show evolution with faded lines
            if trajectory['energies']:
                # Sample some earlier profiles
                sample_indices = np.linspace(0, len(trajectory['energies'])-1, 
                                           min(10, len(trajectory['energies'])), dtype=int)
                for idx in sample_indices[:-1]:  # Exclude the last (current)
                    e = trajectory['energies'][idx]
                    if isinstance(e, (list, np.ndarray)) and len(e) > 1:
                        e_arr = np.array(e)
                        e_old = e_arr - np.min(e_arr)
                        alpha = 0.1 + 0.4 * (idx / len(trajectory['energies']))
                        ax2.plot(range(len(e_old)), e_old, 'gray', alpha=alpha, linewidth=0.5)
            
            # Mark special points
            i_max = np.argmax(e_norm)
            ax2.plot(i_max, e_norm[i_max], 'r*', markersize=15, 
                    label=f'Barrier: {e_norm[i_max]:.3f} eV')
            
            # CI if active
            i_CI = neb_state.get('i_CI', -1)
            if i_CI >= 0 and i_CI < len(e_norm):
                ax2.plot(i_CI, e_norm[i_CI], 'gs', markersize=12, label=f'CI (img {i_CI})')
            
            ax2.set_xlabel('Image Index')
            ax2.set_ylabel('Energy (eV)')
            ax2.set_title(f'Energy Profile at Step {current_step}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        except Exception as e:
            ax2.text(0.5, 0.5, 'Error displaying profile', 
                    transform=ax2.transAxes, ha='center')
    
    # ========== 3. Barrier Height Evolution ==========
    ax3 = plt.subplot(3, 4, (5, 6))
    
    if trajectory['barriers']:
        # Smart sample for plotting
        indices, barriers = smart_sample(trajectory['barriers'], max_points=1000)
        steps = [trajectory['steps'][i] for i in indices if i < len(trajectory['steps'])]
        
        # Remove None values
        valid_data = [(s, b) for s, b in zip(steps, barriers) if b is not None and b > 0]
        if valid_data:
            steps, barriers = zip(*valid_data)
            
            ax3.plot(steps, barriers, 'b-', linewidth=1.5, alpha=0.8)
            ax3.fill_between(steps, 0, barriers, alpha=0.3)
            
            # Mark important points
            if len(barriers) > 0:
                max_barrier_idx = np.argmax(barriers)
                ax3.plot(steps[max_barrier_idx], barriers[max_barrier_idx], 'r*', 
                        markersize=12, label=f'Max: {barriers[max_barrier_idx]:.3f} eV')
                
                # Current barrier
                if len(barriers) > 0:
                    ax3.plot(steps[-1], barriers[-1], 'go', markersize=8, 
                            label=f'Current: {barriers[-1]:.3f} eV')
            
            # Add trend line
            if len(steps) > 10:
                z = np.polyfit(steps[-min(100, len(steps)):], 
                              barriers[-min(100, len(barriers)):], 1)
                p = np.poly1d(z)
                ax3.plot(steps[-min(100, len(steps)):], 
                        p(steps[-min(100, len(steps)):]), 
                        'r--', alpha=0.5, label='Recent trend')
            
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Barrier Height (eV)')
            ax3.set_title('Barrier Evolution')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
    
    # ========== 4. Force Convergence ==========
    ax4 = plt.subplot(3, 4, (7, 8))
    
    if trajectory['max_forces']:
        # Filter out None values and sample
        valid_forces = [(i, f) for i, f in enumerate(trajectory['max_forces']) 
                       if f is not None and f > 0]
        
        if valid_forces:
            indices, forces = zip(*valid_forces)
            steps = [trajectory['steps'][i] for i in indices if i < len(trajectory['steps'])]
            
            # Smart sample if too many points
            if len(forces) > 1000:
                sample_idx = np.linspace(0, len(forces)-1, 1000, dtype=int)
                forces = [forces[i] for i in sample_idx]
                steps = [steps[i] for i in sample_idx]
            
            ax4.semilogy(steps, forces, 'r-', linewidth=1, alpha=0.7, label='Max |F|')
            
            # Add rolling average
            if len(forces) > 20:
                window = min(50, len(forces) // 10)
                rolling_avg = np.convolve(forces, np.ones(window)/window, mode='valid')
                rolling_steps = steps[window//2:len(rolling_avg)+window//2]
                ax4.semilogy(rolling_steps, rolling_avg, 'b-', linewidth=2, 
                           alpha=0.8, label='Moving avg')
            
            # Convergence threshold
            T_MEP = state.get('parameters', {}).get('T_MEP', 0.1)
            ax4.axhline(T_MEP, color='g', linestyle='--', linewidth=2,
                       label=f'Threshold: {T_MEP} eV/Å')
            
            # Mark converged region
            converged_steps = [s for s, f in zip(steps, forces) if f < T_MEP]
            if converged_steps:
                ax4.axvspan(converged_steps[0], steps[-1], alpha=0.2, color='green')
            
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Max Force (eV/Å)')
            ax4.set_title('Force Convergence')
            ax4.grid(True, alpha=0.3, which='both')
            ax4.legend()
    
    # ========== 5. Convergence Metrics ==========
    ax5 = plt.subplot(3, 4, 9)
    
    # Calculate convergence metrics
    if trajectory['max_forces'] and trajectory['barriers']:
        recent_forces = [f for f in trajectory['max_forces'][-100:] if f is not None]
        recent_barriers = [b for b in trajectory['barriers'][-100:] if b is not None and b > 0]
        
        metrics_text = f"Step: {current_step}\n"
        metrics_text += f"Converged: {state.get('converged', False)}\n\n"
        
        if recent_forces:
            metrics_text += f"Current Max Force: {recent_forces[-1]:.4f} eV/Å\n"
            metrics_text += f"Avg (last 100): {np.mean(recent_forces):.4f} eV/Å\n"
            valid_forces = [f for f in trajectory['max_forces'] if f is not None]
            if valid_forces:
                metrics_text += f"Min achieved: {np.min(valid_forces):.4f} eV/Å\n\n"
        
        if recent_barriers:
            metrics_text += f"Current Barrier: {recent_barriers[-1]:.3f} eV\n"
            metrics_text += f"Avg (last 100): {np.mean(recent_barriers):.3f} eV\n"
            
            # Calculate stability
            if len(recent_barriers) > 10:
                std = np.std(recent_barriers[-10:])
                metrics_text += f"Stability (σ): {std:.4f} eV\n"
        
        ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax5.axis('off')
    ax5.set_title('Convergence Metrics')
    
    # ========== 6. Energy Distribution ==========
    ax6 = plt.subplot(3, 4, 10)
    
    if trajectory['barriers']:
        valid_barriers = [b for b in trajectory['barriers'] if b is not None and b > 0]
        if valid_barriers:
            ax6.hist(valid_barriers, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax6.axvline(np.mean(valid_barriers), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(valid_barriers):.3f}')
            ax6.axvline(np.median(valid_barriers), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(valid_barriers):.3f}')
            
            ax6.set_xlabel('Barrier Height (eV)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Barrier Distribution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    # ========== 7. Step Timing Analysis ==========
    ax7 = plt.subplot(3, 4, 11)
    
    if len(trajectory['steps']) > 1:
        # Calculate time per step (assuming uniform sampling)
        step_diffs = np.diff(trajectory['steps'][:1000])  # First 1000 for performance
        
        if len(step_diffs) > 0:
            ax7.hist(step_diffs, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax7.set_xlabel('Steps Between Saves')
            ax7.set_ylabel('Frequency')
            ax7.set_title('Checkpoint Frequency')
            ax7.grid(True, alpha=0.3)
    
    # ========== 8. Path Length Evolution ==========
    ax8 = plt.subplot(3, 4, 12)
    
    if 'R' in neb_state and neb_state['R'] is not None:
        positions = neb_state['R']
        path_lengths = []
        
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            path_lengths.append(dist)
        
        if path_lengths:
            x = range(len(path_lengths))
            ax8.bar(x, path_lengths, color='purple', alpha=0.7)
            ax8.set_xlabel('Image Pair')
            ax8.set_ylabel('Distance (Å)')
            ax8.set_title(f'Path Segment Lengths (Total: {sum(path_lengths):.2f} Å)')
            ax8.grid(True, alpha=0.3)
    
    # Overall title and layout
    fig.suptitle(f'NEB Evolution Analysis - {current_step} Steps - {len(trajectory["steps"])} Data Points', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Enhanced visualization saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Enhanced NEB evolution visualization')
    parser.add_argument('--run-dir', type=str, help='Run directory path')
    parser.add_argument('--output', type=str, help='Output file name')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output image')
    
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Find latest run
        for base in ["../outputs", "outputs"]:
            if Path(base).exists():
                runs = sorted([d for d in Path(base).iterdir() 
                              if d.is_dir() and d.name.startswith("run_")])
                if runs:
                    run_dir = runs[-1]
                    print(f"Using latest run: {run_dir}")
                    break
        else:
            print("No run directory found!")
            return
    
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"No checkpoints in {run_dir}")
        return
    
    # Load data
    state, metadata, trajectory = load_full_trajectory(checkpoint_dir)
    
    # Create visualization
    output_file = args.output or str(run_dir / "neb_evolution_enhanced.png")
    create_evolution_plot(state, metadata, trajectory, output_file)
    
    print(f"\n✅ Visualization complete!")
    print(f"   Output: {output_file}")
    print(f"   Total steps analyzed: {len(trajectory['steps'])}")
    
    if trajectory['barriers']:
        valid_barriers = [b for b in trajectory['barriers'] if b is not None and b > 0]
        if valid_barriers:
            print(f"   Final barrier: {valid_barriers[-1]:.3f} eV")
            print(f"   Best barrier: {min(valid_barriers):.3f} eV")


if __name__ == "__main__":
    main()