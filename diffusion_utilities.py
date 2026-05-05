#!/usr/bin/env python
"""
Utility functions for diffusion analysis.
Includes trajectory analysis, visualization, and post-processing tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
import matplotlib.animation as animation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import sys


def visualize_diffusion_paths(trajectory_file: str,
                            species_to_track: List[str],
                            output_file: str = 'diffusion_paths.png',
                            projection: str = 'xy',
                            n_frames: Optional[int] = None):
    """
    Visualize diffusion paths of tracked species.
    
    Args:
        trajectory_file: Path to trajectory file
        species_to_track: List of species to visualize
        output_file: Output image file
        projection: 'xy', 'xz', 'yz', or '3d'
        n_frames: Number of frames to use (None for all)
    """
    traj = Trajectory(trajectory_file)
    
    if n_frames:
        # Sample frames evenly
        indices = np.linspace(0, len(traj)-1, n_frames, dtype=int)
        frames = [traj[i] for i in indices]
    else:
        frames = traj
        
    # Get tracked atom indices
    symbols = frames[0].get_chemical_symbols()
    tracked_indices = [i for i, sym in enumerate(symbols) if sym in species_to_track]
    
    # Extract positions
    positions = []
    for atoms in frames:
        pos = atoms.get_positions()[tracked_indices]
        positions.append(pos)
    positions = np.array(positions)
    
    # Unwrap trajectories for periodic boundaries
    cell = frames[0].get_cell()
    for i in range(1, len(positions)):
        delta = positions[i] - positions[i-1]
        # Check for jumps across boundaries
        for j in range(3):
            if cell[j, j] > 0:  # Only for periodic dimensions
                jumps = np.round(delta[:, j] / cell[j, j])
                positions[i, :, j] -= jumps * cell[j, j]
            
    # Create figure
    if projection == '3d':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Color map for different atoms
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tracked_indices)))
    
    # Plot trajectories
    for i, atom_idx in enumerate(tracked_indices):
        trajectory = positions[:, i, :]
        
        if projection == '3d':
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   color=colors[i], alpha=0.7, linewidth=1)
            # Mark start and end
            ax.scatter(*trajectory[0], color=colors[i], s=100, marker='o', 
                      edgecolor='black', linewidth=2)
            ax.scatter(*trajectory[-1], color=colors[i], s=100, marker='s',
                      edgecolor='black', linewidth=2)
        else:
            # Project onto 2D
            if projection == 'xy':
                x, y = trajectory[:, 0], trajectory[:, 1]
            elif projection == 'xz':
                x, y = trajectory[:, 0], trajectory[:, 2]
            elif projection == 'yz':
                x, y = trajectory[:, 1], trajectory[:, 2]
                
            ax.plot(x, y, color=colors[i], alpha=0.7, linewidth=1)
            ax.scatter(x[0], y[0], color=colors[i], s=100, marker='o',
                      edgecolor='black', linewidth=2)
            ax.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s',
                      edgecolor='black', linewidth=2)
            
    # Add cell boundaries
    if projection != '3d':
        if projection == 'xy':
            ax.add_patch(plt.Rectangle((0, 0), cell[0, 0], cell[1, 1],
                                     fill=False, edgecolor='black', linewidth=2))
        elif projection == 'xz':
            ax.add_patch(plt.Rectangle((0, 0), cell[0, 0], cell[2, 2],
                                     fill=False, edgecolor='black', linewidth=2))
        elif projection == 'yz':
            ax.add_patch(plt.Rectangle((0, 0), cell[1, 1], cell[2, 2],
                                     fill=False, edgecolor='black', linewidth=2))
                                     
    # Labels and formatting
    if projection == '3d':
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
    else:
        ax.set_xlabel(f'{projection[0].upper()} (Å)')
        ax.set_ylabel(f'{projection[1].upper()} (Å)')
        ax.set_aspect('equal')
        
    ax.set_title(f'Diffusion Paths ({", ".join(species_to_track)})')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='gray', markeredgecolor='black',
                            markersize=10, label='Start'),
                      Line2D([0], [0], marker='s', color='w',
                            markerfacecolor='gray', markeredgecolor='black', 
                            markersize=10, label='End')]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diffusion paths saved to {output_file}")


def create_diffusion_movie(trajectory_file: str,
                         species_to_track: List[str],
                         output_file: str = 'diffusion_movie.mp4',
                         fps: int = 10,
                         trail_length: int = 50):
    """
    Create an animation of diffusion.
    
    Args:
        trajectory_file: Path to trajectory file
        species_to_track: Species to animate
        output_file: Output movie file
        fps: Frames per second
        trail_length: Number of frames to show as trail
    """
    traj = Trajectory(trajectory_file)
    
    # Get tracked atom indices
    symbols = traj[0].get_chemical_symbols()
    tracked_indices = [i for i, sym in enumerate(symbols) if sym in species_to_track]
    other_indices = [i for i in range(len(symbols)) if i not in tracked_indices]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize plot elements
    cell = traj[0].get_cell()
    ax.set_xlim(0, cell[0, 0])
    ax.set_ylim(0, cell[1, 1])
    ax.set_aspect('equal')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    
    # Plot static atoms
    static_pos = traj[0].get_positions()[other_indices]
    ax.scatter(static_pos[:, 0], static_pos[:, 1], c='lightgray', s=50, alpha=0.5)
    
    # Initialize moving atoms
    tracked_pos = traj[0].get_positions()[tracked_indices]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tracked_indices)))
    
    scatters = []
    trails = []
    
    for i in range(len(tracked_indices)):
        scatter = ax.scatter(tracked_pos[i, 0], tracked_pos[i, 1], 
                           c=[colors[i]], s=100, edgecolor='black', linewidth=2)
        scatters.append(scatter)
        
        trail, = ax.plot([], [], color=colors[i], alpha=0.5, linewidth=1)
        trails.append(trail)
        
    # Animation function
    trail_data = [[] for _ in range(len(tracked_indices))]
    
    def animate(frame):
        atoms = traj[frame]
        
        # Update tracked atoms
        for i, idx in enumerate(tracked_indices):
            pos = atoms.get_positions()[idx]
            
            # Handle periodic boundaries
            pos = pos % np.diag(cell)
            
            scatters[i].set_offsets([pos[0], pos[1]])
            
            # Update trail
            trail_data[i].append(pos[:2])
            if len(trail_data[i]) > trail_length:
                trail_data[i].pop(0)
                
            if len(trail_data[i]) > 1:
                trail_xy = np.array(trail_data[i])
                trails[i].set_data(trail_xy[:, 0], trail_xy[:, 1])
                
        ax.set_title(f'MD Diffusion - Frame {frame}/{len(traj)}')
        return scatters + trails
        
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(traj),
                                 interval=1000/fps, blit=True)
                                 
    # Save animation
    anim.save(output_file, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()
    
    print(f"Diffusion movie saved to {output_file}")


def analyze_diffusion_anisotropy(trajectory_file: str,
                               species_to_track: List[str],
                               output_file: str = 'anisotropy_analysis.png'):
    """
    Analyze directional diffusion (anisotropy).
    
    Args:
        trajectory_file: Path to trajectory
        species_to_track: Species to analyze
        output_file: Output figure
    """
    from md_diffusion_analyzer import MDDiffusionAnalyzer
    
    traj = Trajectory(trajectory_file)
    analyzer = MDDiffusionAnalyzer(traj[0], species_to_track=species_to_track)
    
    # Calculate MSD in each direction
    time_data, msd_total = analyzer.calculate_msd(trajectory_file)
    
    # Calculate directional MSDs
    n_frames = len(traj)
    n_atoms = len(analyzer.tracked_indices)
    
    # Extract positions
    positions = np.zeros((n_frames, n_atoms, 3))
    for i, atoms in enumerate(traj):
        positions[i] = atoms.get_positions()[analyzer.tracked_indices]
        
    # Calculate MSD for each direction
    msd_x = np.zeros(len(time_data))
    msd_y = np.zeros(len(time_data))
    msd_z = np.zeros(len(time_data))
    
    for i, dt in enumerate(range(1, len(time_data) + 1)):
        dr = positions[dt:] - positions[:-dt]
        msd_x[i] = np.mean(dr[:, :, 0]**2)
        msd_y[i] = np.mean(dr[:, :, 1]**2)
        msd_z[i] = np.mean(dr[:, :, 2]**2)
        
    # Calculate directional diffusion coefficients
    from scipy import stats
    
    # Fit linear region (20-80%)
    start = int(0.2 * len(time_data))
    end = int(0.8 * len(time_data))
    
    Dx = stats.linregress(time_data[start:end], msd_x[start:end]).slope / 2
    Dy = stats.linregress(time_data[start:end], msd_y[start:end]).slope / 2
    Dz = stats.linregress(time_data[start:end], msd_z[start:end]).slope / 2
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot directional MSDs
    ax1.plot(time_data, msd_x, 'r-', label=f'X: D={Dx:.2e} Ų/ps')
    ax1.plot(time_data, msd_y, 'g-', label=f'Y: D={Dy:.2e} Ų/ps')
    ax1.plot(time_data, msd_z, 'b-', label=f'Z: D={Dz:.2e} Ų/ps')
    ax1.plot(time_data, msd_total, 'k--', label='Total', linewidth=2)
    
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('MSD (Ų)')
    ax1.set_title('Directional Mean Square Displacement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of diffusion coefficients
    directions = ['X', 'Y', 'Z']
    D_values = [Dx, Dy, Dz]
    colors = ['red', 'green', 'blue']
    
    ax2.bar(directions, D_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Diffusion Coefficient (Ų/ps)')
    ax2.set_title('Directional Diffusion Coefficients')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add anisotropy ratio
    D_max = max(D_values)
    D_min = min(D_values)
    anisotropy_ratio = D_max / D_min if D_min > 0 else np.inf
    
    ax2.text(0.5, 0.95, f'Anisotropy ratio: {anisotropy_ratio:.2f}',
            transform=ax2.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Anisotropy analysis saved to {output_file}")
    
    return {'Dx': Dx, 'Dy': Dy, 'Dz': Dz, 'anisotropy_ratio': anisotropy_ratio}


def extract_jump_statistics(trajectory_file: str,
                          species: str,
                          jump_threshold: float = 3.0,
                          output_file: str = 'jump_analysis.png'):
    """
    Extract and visualize jump statistics.
    
    Args:
        trajectory_file: Path to trajectory
        species: Species to analyze
        jump_threshold: Minimum jump distance (Å)
        output_file: Output figure
    """
    traj = Trajectory(trajectory_file)
    
    # Get atom indices for species
    symbols = traj[0].get_chemical_symbols()
    species_indices = [i for i, sym in enumerate(symbols) if sym == species]
    
    # Collect all jumps
    all_jumps = []
    jump_vectors = []
    
    for atom_idx in species_indices:
        positions = []
        for atoms in traj:
            positions.append(atoms.get_positions()[atom_idx])
        positions = np.array(positions)
        
        # Find jumps
        for i in range(1, len(positions)):
            delta = positions[i] - positions[i-1]
            dist = np.linalg.norm(delta)
            
            if dist > jump_threshold:
                all_jumps.append({
                    'atom': atom_idx,
                    'frame': i,
                    'distance': dist,
                    'vector': delta
                })
                jump_vectors.append(delta)
                
    if not all_jumps:
        print("No jumps found!")
        return
        
    # Analyze jump statistics
    jump_distances = [j['distance'] for j in all_jumps]
    jump_vectors = np.array(jump_vectors)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Jump distance histogram
    ax = axes[0, 0]
    ax.hist(jump_distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(jump_distances), color='red', linestyle='--',
              label=f'Mean: {np.mean(jump_distances):.2f} Å')
    ax.set_xlabel('Jump Distance (Å)')
    ax.set_ylabel('Count')
    ax.set_title('Jump Distance Distribution')
    ax.legend()
    
    # 2. Jump direction analysis (2D projection)
    ax = axes[0, 1]
    ax.scatter(jump_vectors[:, 0], jump_vectors[:, 1], alpha=0.5)
    ax.set_xlabel('ΔX (Å)')
    ax.set_ylabel('ΔY (Å)')
    ax.set_title('Jump Vectors (XY projection)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Jump frequency over time
    ax = axes[1, 0]
    jump_frames = [j['frame'] for j in all_jumps]
    hist, bins = np.histogram(jump_frames, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Convert to jump rate
    time_window = len(traj) / len(bins)
    jump_rate = hist / (time_window * len(species_indices))
    
    ax.plot(bin_centers, jump_rate, 'g-', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Jump Rate (jumps/frame/atom)')
    ax.set_title('Jump Frequency Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""Jump Statistics Summary
    
Total jumps: {len(all_jumps)}
Atoms analyzed: {len(species_indices)}
Average jump distance: {np.mean(jump_distances):.2f} ± {np.std(jump_distances):.2f} Å
Jump frequency: {len(all_jumps)/(len(traj)*len(species_indices)):.4f} jumps/frame/atom

Most common jump distances:
"""
    
    # Add histogram of common distances
    hist, bins = np.histogram(jump_distances, bins=10)
    for i in range(3):  # Top 3
        idx = np.argmax(hist)
        summary_text += f"  {bins[idx]:.1f}-{bins[idx+1]:.1f} Å: {hist[idx]} jumps\n"
        hist[idx] = 0
        
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace')
           
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Jump analysis saved to {output_file}")
    
    return all_jumps


def find_latest_trajectory():
    """Find the latest trajectory file in outputs directory."""
    outputs_dir = Path("../outputs")
    if not outputs_dir.exists():
        outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        return None
        
    # Find latest run directory
    run_dirs = sorted([d for d in outputs_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("run_")])
    
    if not run_dirs:
        return None
        
    # Search for trajectory files in latest run
    latest_run = run_dirs[-1]
    
    # Common locations for MD trajectories
    search_paths = [
        latest_run / "md_diffusion_analysis" / "T_*" / "md_trajectory.traj",
        latest_run / "md_trajectory.traj",
        latest_run / "*.traj"
    ]
    
    for pattern in search_paths:
        if "*" in str(pattern):
            files = list(latest_run.glob(pattern.name))
            if files:
                # Get the most recent
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return files[0]
        elif pattern.exists():
            return pattern
            
    return None


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Diffusion analysis utilities')
    parser.add_argument('trajectory', nargs='?', help='Path to trajectory file (auto-detect if not specified)')
    parser.add_argument('--species', nargs='+', required=True,
                       help='Species to analyze')
    parser.add_argument('--visualize-paths', action='store_true',
                       help='Visualize diffusion paths')
    parser.add_argument('--make-movie', action='store_true',
                       help='Create diffusion movie')
    parser.add_argument('--analyze-anisotropy', action='store_true',
                       help='Analyze directional diffusion')
    parser.add_argument('--analyze-jumps', action='store_true',
                       help='Analyze atomic jumps')
    parser.add_argument('--jump-threshold', type=float, default=3.0,
                       help='Jump distance threshold (Å)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: same as trajectory)')
                       
    args = parser.parse_args()
    
    # Auto-detect trajectory if not specified
    if not args.trajectory:
        print("No trajectory specified. Searching for latest...")
        trajectory_path = find_latest_trajectory()
        if trajectory_path:
            print(f"Found: {trajectory_path}")
            args.trajectory = str(trajectory_path)
        else:
            print("ERROR: Could not find trajectory file!")
            print("Please specify trajectory path or run from correct directory")
            sys.exit(1)
    
    # Set output directory
    trajectory_path = Path(args.trajectory)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use same directory as trajectory
        output_dir = trajectory_path.parent
    
    output_dir.mkdir(exist_ok=True)
    
    if args.visualize_paths:
        output_file = output_dir / 'diffusion_paths.png'
        visualize_diffusion_paths(args.trajectory, args.species, 
                                output_file=str(output_file))
        
    if args.make_movie:
        output_file = output_dir / 'diffusion_movie.mp4'
        create_diffusion_movie(args.trajectory, args.species,
                             output_file=str(output_file))
        
    if args.analyze_anisotropy:
        output_file = output_dir / 'anisotropy_analysis.png'
        analyze_diffusion_anisotropy(args.trajectory, args.species,
                                   output_file=str(output_file))
        
    if args.analyze_jumps:
        for species in args.species:
            output_file = output_dir / f'jump_analysis_{species}.png'
            extract_jump_statistics(args.trajectory, species, args.jump_threshold,
                                  output_file=str(output_file))
    
    print(f"\nAll outputs saved to: {output_dir}")