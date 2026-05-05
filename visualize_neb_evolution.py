#!/usr/bin/env python
"""Visualize NEB evolution showing all images side by side.

Supports organized run directory structure:
- Use --run-dir to specify a specific run directory
- Use --latest to automatically find the most recent run
- Use --output-subdir to specify subdirectory for outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib.patches import Circle  # Not needed for energy-only visualization
import pickle
import argparse
from pathlib import Path
from typing import Optional
# from pymatgen.io.vasp import Poscar  # Not needed for energy-only visualization

from output_manager import get_output_path, get_latest_run_dir


def load_neb_results(results_file='results/pure_neb_results.pkl'):
    """Load NEB results from pickle file."""
    # Try to resolve path through output manager
    try:
        resolved_path = get_output_path(results_file)
        if Path(resolved_path).exists():
            results_file = resolved_path
    except:
        pass  # Fall back to original path
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def load_neb_checkpoint(checkpoint_file='checkpoints/pure_neb_latest.pkl'):
    """Load NEB checkpoint to get trajectory."""
    # Try to resolve path through output manager
    try:
        resolved_path = get_output_path(checkpoint_file)
        if Path(resolved_path).exists():
            checkpoint_file = resolved_path
    except:
        pass  # Fall back to original path
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


# Functions for atom visualization - removed since we're only showing energy profiles


def create_neb_animation(checkpoint_file='checkpoints/pure_neb_latest.pkl',
                        initial_poscar='POSCAR_Mo_initial',
                        final_poscar='POSCAR_Mo_final',
                        output_file='neb_evolution.gif',
                        atom_indices=None,
                        view_axis=2,  # 0=x, 1=y, 2=z
                        image_spacing=3.0,
                        fps=10,
                        align_111=False,
                        show_neighbors=False,
                        neighbor_cutoff=3.5,
                        bcc_110_view=False):
    """Create animation of NEB evolution showing only energy profile.
    
    Args:
        checkpoint_file: Path to NEB checkpoint
        initial_poscar: Initial structure file (not used in energy-only mode)
        final_poscar: Final structure file (not used in energy-only mode)
        output_file: Output animation file
        atom_indices: Not used in energy-only mode
        view_axis: Not used in energy-only mode
        image_spacing: Not used in energy-only mode
        fps: Frames per second for animation
        align_111: Not used in energy-only mode
        show_neighbors: Not used in energy-only mode
        neighbor_cutoff: Not used in energy-only mode
        bcc_110_view: Not used in energy-only mode
    """
    # Load checkpoint
    checkpoint = load_neb_checkpoint(checkpoint_file)
    trajectory = checkpoint['trajectory']
    n_images = checkpoint['n_images']
    
    # Skip atom detection and visualization - we're only showing energy profile
    
    # Setup figure for energy profile only
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize plot elements
    line, = ax.plot([], [], 'bo-', markersize=8, linewidth=2)
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12)
    
    def init():
        """Initialize animation."""
        line.set_data([], [])
        return line, step_text
    
    def animate(frame):
        """Animate frame."""
        if frame >= len(trajectory):
            return line, step_text
        
        # Get current step data
        _, energies, _ = trajectory[frame]
        
        # Create reaction coordinate
        x = np.arange(n_images)
        y = energies.flatten()
        
        # Update plot
        line.set_data(x, y)
        
        # Update text with more detailed info
        mean_e = np.mean(energies[1:-1])
        max_e = np.max(energies[1:-1])
        barrier_forward = max_e - energies[0, 0]
        barrier_reverse = max_e - energies[-1, 0]
        
        # Find saddle point index
        saddle_idx = np.argmax(energies[1:-1]) + 1
        
        step_text.set_text(f'Step {frame+1}/{len(trajectory)}\n'
                          f'Mean E: {mean_e:.3f} eV\n'
                          f'Saddle Point: Image {saddle_idx} at {max_e:.3f} eV\n'
                          f'Forward Barrier: {barrier_forward:.3f} eV\n'
                          f'Reverse Barrier: {barrier_reverse:.3f} eV')
        
        return line, step_text
    
    # Set axis properties
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('NEB Energy Profile Evolution')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits based on trajectory data
    if trajectory:
        all_energies = []
        for _, energies, _ in trajectory:
            all_energies.extend(energies.flatten())
        
        ax.set_xlim(-0.5, n_images - 0.5)
        y_min = min(all_energies) - 0.1
        y_max = max(all_energies) + 0.1
        ax.set_ylim(y_min, y_max)
        
        # Add horizontal line at zero if energies are referenced
        if abs(y_min) < 1.0:  # Likely referenced energies
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Create animation
    print(f"Creating animation with {len(trajectory)} frames...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(trajectory), interval=1000/fps,
                                 blit=True, repeat=True)
    
    # Resolve output path through output manager
    try:
        resolved_output = get_output_path(output_file)
        # Ensure directory exists
        Path(resolved_output).parent.mkdir(parents=True, exist_ok=True)
        output_file = resolved_output
    except:
        pass  # Fall back to original path
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    if output_file.endswith('.gif'):
        anim.save(output_file, writer='pillow', fps=fps)
    elif output_file.endswith('.mp4'):
        anim.save(output_file, writer='ffmpeg', fps=fps)
    else:
        print(f"Unknown file format. Saving as GIF.")
        anim.save(output_file + '.gif', writer='pillow', fps=fps)
    
    print(f"Animation saved to {output_file}")
    
    # Also show the plot
    plt.show()
    
    return fig, anim


def create_energy_profile_animation(checkpoint_file='checkpoints/pure_neb_latest.pkl',
                                  output_file='neb_energy_profile.gif',
                                  fps=10):
    """Create animation of energy profile evolution."""
    # Load checkpoint
    checkpoint = load_neb_checkpoint(checkpoint_file)
    trajectory = checkpoint['trajectory']
    n_images = checkpoint['n_images']
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize plot
    line, = ax.plot([], [], 'bo-', markersize=8, linewidth=2)
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12)
    
    def init():
        line.set_data([], [])
        return line, step_text
    
    def animate(frame):
        if frame >= len(trajectory):
            return line, step_text
        
        # Get energies
        _, energies, _ = trajectory[frame]
        
        # Create reaction coordinate
        x = np.arange(n_images)
        y = energies.flatten()
        
        # Update plot
        line.set_data(x, y)
        
        # Update text
        barrier = np.max(energies[1:-1]) - energies[0, 0]
        step_text.set_text(f'Step {frame+1}/{len(trajectory)}\n'
                         f'Barrier: {float(barrier):.3f} eV')
        
        return line, step_text
    
    # Set axis properties
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('NEB Energy Profile Evolution')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    if trajectory:
        all_energies = []
        for _, energies, _ in trajectory:
            all_energies.extend(energies.flatten())
        
        ax.set_xlim(-0.5, n_images - 0.5)
        ax.set_ylim(min(all_energies) - 0.1, max(all_energies) + 0.1)
    
    # Create animation
    print(f"Creating energy profile animation...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(trajectory), interval=1000/fps,
                                 blit=True, repeat=True)
    
    # Resolve output path through output manager
    try:
        resolved_output = get_output_path(output_file)
        # Ensure directory exists
        Path(resolved_output).parent.mkdir(parents=True, exist_ok=True)
        output_file = resolved_output
    except:
        pass  # Fall back to original path
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    if output_file.endswith('.gif'):
        anim.save(output_file, writer='pillow', fps=fps)
    elif output_file.endswith('.mp4'):
        anim.save(output_file, writer='ffmpeg', fps=fps)
    
    print(f"Energy profile animation saved to {output_file}")
    
    return fig, anim


def resolve_file_paths(checkpoint: str, initial_poscar: str, final_poscar: str, 
                       run_dir: Optional[str] = None, latest: bool = False) -> tuple:
    """Resolve file paths from run directory or use provided paths."""
    if latest:
        try:
            run_dir = get_latest_run_dir()
            print(f"Using latest run directory: {run_dir}")
        except Exception as e:
            print(f"Could not find latest run directory: {e}")
            print("Using default file paths")
            return checkpoint, initial_poscar, final_poscar
    
    if run_dir:
        # Look for files in the specified run directory
        run_path = Path(run_dir)
        
        # Try to find checkpoint in checkpoints subdirectory
        checkpoint_candidates = [
            run_path / "checkpoints" / Path(checkpoint).name,
            run_path / checkpoint,
            Path(checkpoint)
        ]
        
        checkpoint_resolved = checkpoint
        for candidate in checkpoint_candidates:
            if candidate.exists():
                checkpoint_resolved = str(candidate)
                break
        
        # Try to find POSCAR files in inputs or run directory
        # Also check parent directory for inputs folder
        parent_inputs = run_path.parent.parent / "inputs" if run_path.parent.name == "outputs" else None
        
        initial_candidates = [
            run_path / "inputs" / Path(initial_poscar).name,
            run_path / initial_poscar,
            Path("../inputs") / Path(initial_poscar).name,  # Check parent inputs
            Path("inputs") / Path(initial_poscar).name,
            Path(initial_poscar)
        ]
        
        if parent_inputs and parent_inputs.exists():
            initial_candidates.insert(2, parent_inputs / Path(initial_poscar).name)
        
        final_candidates = [
            run_path / "inputs" / Path(final_poscar).name,
            run_path / final_poscar,
            Path("../inputs") / Path(final_poscar).name,  # Check parent inputs
            Path("inputs") / Path(final_poscar).name,
            Path(final_poscar)
        ]
        
        if parent_inputs and parent_inputs.exists():
            final_candidates.insert(2, parent_inputs / Path(final_poscar).name)
        
        initial_resolved = initial_poscar
        for candidate in initial_candidates:
            if candidate.exists():
                initial_resolved = str(candidate)
                break
        
        final_resolved = final_poscar
        for candidate in final_candidates:
            if candidate.exists():
                final_resolved = str(candidate)
                break
        
        print(f"Looking for files in run directory: {run_dir}")
        print(f"Checkpoint: {checkpoint_resolved} {'(found)' if Path(checkpoint_resolved).exists() else '(not found)'}")
        print(f"Initial POSCAR: {initial_resolved} {'(found)' if Path(initial_resolved).exists() else '(not found)'}")
        print(f"Final POSCAR: {final_resolved} {'(found)' if Path(final_resolved).exists() else '(not found)'}")
        
        return checkpoint_resolved, initial_resolved, final_resolved
    
    return checkpoint, initial_poscar, final_poscar

def main():
    parser = argparse.ArgumentParser(description='Visualize NEB energy profile evolution')
    
    parser.add_argument('--checkpoint', default='checkpoints/pure_neb_latest.pkl',
                        help='Path to NEB checkpoint file')
    parser.add_argument('--initial-poscar', default='POSCAR_Mo_initial',
                        help='Initial POSCAR file (not used for energy-only visualization)')
    parser.add_argument('--final-poscar', default='POSCAR_Mo_final',
                        help='Final POSCAR file (not used for energy-only visualization)')
    parser.add_argument('--output', default='neb_energy_evolution.gif',
                        help='Output animation file')
    parser.add_argument('--run-dir', help='Specific run directory to analyze')
    parser.add_argument('--latest', action='store_true', help='Use the most recent run directory')
    parser.add_argument('--output-subdir', default='neb_animations', 
                        help='Subdirectory within run for outputs (default: neb_animations)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second')
    # Keep these arguments for backward compatibility but they won't be used
    parser.add_argument('--atoms', type=int, nargs='+', default=None,
                        help='(Not used in energy-only mode)')
    parser.add_argument('--view-axis', type=int, default=2, choices=[0, 1, 2],
                        help='(Not used in energy-only mode)')
    parser.add_argument('--image-spacing', type=float, default=3.0,
                        help='(Not used in energy-only mode)')
    parser.add_argument('--energy-profile', action='store_true',
                        help='(Deprecated - energy profile is now the default)')
    parser.add_argument('--align-111', action='store_true',
                        help='(Not used in energy-only mode)')
    parser.add_argument('--show-neighbors', action='store_true',
                        help='(Not used in energy-only mode)')
    parser.add_argument('--neighbor-cutoff', type=float, default=3.5,
                        help='(Not used in energy-only mode)')
    parser.add_argument('--bcc-110-view', action='store_true',
                        help='(Not used in energy-only mode)')
    
    args = parser.parse_args()
    
    # If no specific run directory is provided and not using --latest, 
    # automatically try to find the latest run
    if not args.run_dir and not args.latest:
        # Check if the default checkpoint exists
        if not Path(args.checkpoint).exists():
            print("Default checkpoint not found. Automatically searching for latest run...")
            args.latest = True
    
    # Resolve file paths based on run directory options
    checkpoint, initial_poscar, final_poscar = resolve_file_paths(
        args.checkpoint, args.initial_poscar, args.final_poscar, 
        args.run_dir, args.latest)
    
    # Determine output file path
    if args.run_dir or args.latest:
        # Get the run directory path
        if args.latest:
            try:
                run_dir = get_latest_run_dir()
            except:
                run_dir = None
        else:
            run_dir = args.run_dir
            
        if run_dir:
            output_dir = Path(run_dir) / args.output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_dir / args.output)
        else:
            output_file = args.output
    else:
        output_file = args.output
    
    # Create energy profile animation (this is now the main and only animation)
    create_neb_animation(
        checkpoint_file=checkpoint,
        initial_poscar=initial_poscar,
        final_poscar=final_poscar,
        output_file=output_file,
        atom_indices=args.atoms,
        view_axis=args.view_axis,
        image_spacing=args.image_spacing,
        fps=args.fps,
        align_111=args.align_111,
        show_neighbors=args.show_neighbors,
        neighbor_cutoff=args.neighbor_cutoff,
        bcc_110_view=args.bcc_110_view
    )


if __name__ == "__main__":
    main()