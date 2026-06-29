#!/usr/bin/env python3
"""
Script to animate POSCARs from dimer_runs folder showing the entire cell
with the moving atom highlighted in a different color.
Can automatically detect which atoms are moving.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element
import argparse
from pathlib import Path
from output_manager import get_output_path


def get_element_colors(structure):
    """Get colors for each element type."""
    # Define a color palette for different elements
    color_palette = {
        'H': '#FFFFFF',
        'Li': '#CC80FF',
        'Be': '#C2FF00',
        'B': '#FFB5B5',
        'C': '#909090',
        'N': '#3050F8',
        'O': '#FF0D0D',
        'F': '#90E050',
        'Na': '#AB5CF2',
        'Mg': '#8AFF00',
        'Al': '#BFA6A6',
        'Si': '#F0C8A0',
        'P': '#FF8000',
        'S': '#FFFF30',
        'Cl': '#1FF01F',
        'K': '#8F40D4',
        'Ca': '#3DFF00',
        'Ti': '#BFC2C7',
        'V': '#A6A6AB',
        'Cr': '#8A99C7',
        'Mn': '#9C7AC7',
        'Fe': '#E06633',
        'Co': '#F090A0',
        'Ni': '#50D050',
        'Cu': '#C88033',
        'Zn': '#7D80B0',
        'Ga': '#C28F8F',
        'Ge': '#668F8F',
        'As': '#BD80E3',
        'Se': '#FFA100',
        'Br': '#A62929',
        'Rb': '#702EB0',
        'Sr': '#00FF00',
        'Y': '#94FFFF',
        'Zr': '#94E0E0',
        'Nb': '#73C2C9',
        'Mo': '#54B5B5',
        'Tc': '#3B9E9E',
        'Ru': '#248F8F',
        'Rh': '#0A7D8C',
        'Pd': '#006985',
        'Ag': '#C0C0C0',
        'Cd': '#FFD98F',
        'In': '#A67573',
        'Sn': '#668080',
        'Sb': '#9E63B5',
        'Te': '#D47A00',
        'I': '#940094',
        'W': '#2194D6',
    }
    
    colors = []
    for site in structure:
        element = site.specie.symbol
        if element in color_palette:
            colors.append(color_palette[element])
        else:
            # Default color for elements not in palette
            colors.append('#808080')
    
    return colors


def detect_moving_atoms(positions_list, threshold=0.1):
    """Automatically detect which atoms are moving significantly.
    
    Args:
        positions_list: List of position arrays from POSCARs
        threshold: Minimum displacement (in Angstroms) to be considered "moving"
    
    Returns:
        List of atom indices that are moving
    """
    if len(positions_list) < 2:
        return []
    
    # Calculate total displacement for each atom
    initial_pos = positions_list[0]
    final_pos = positions_list[-1]
    
    displacements = np.linalg.norm(final_pos - initial_pos, axis=1)
    
    # Find atoms with displacement above threshold
    moving_indices = np.where(displacements > threshold)[0]
    
    # Sort by displacement magnitude (largest first)
    moving_indices = moving_indices[np.argsort(-displacements[moving_indices])]
    
    print(f"\nDetected {len(moving_indices)} moving atoms (threshold: {threshold} Å):")
    for idx in moving_indices[:10]:  # Show top 10
        print(f"  Atom {idx}: displacement = {displacements[idx]:.3f} Å")
    if len(moving_indices) > 10:
        print(f"  ... and {len(moving_indices) - 10} more")
    
    return moving_indices.tolist()


def analyze_trajectory(positions_list):
    """Analyze the trajectory to provide statistics."""
    if len(positions_list) < 2:
        return {}
    
    initial_pos = positions_list[0]
    final_pos = positions_list[-1]
    
    # Calculate displacements
    displacements = np.linalg.norm(final_pos - initial_pos, axis=1)
    
    # Calculate path lengths for each atom
    path_lengths = np.zeros(len(initial_pos))
    for i in range(1, len(positions_list)):
        step_disp = np.linalg.norm(positions_list[i] - positions_list[i-1], axis=1)
        path_lengths += step_disp
    
    stats = {
        'max_displacement': np.max(displacements),
        'max_disp_atom': np.argmax(displacements),
        'mean_displacement': np.mean(displacements),
        'n_atoms_moved': np.sum(displacements > 0.01),  # Atoms that moved > 0.01 Å
        'max_path_length': np.max(path_lengths),
        'max_path_atom': np.argmax(path_lengths),
    }
    
    return stats


def read_poscar_sequence(dimer_runs_dir, auto_detect=True, manual_indices=None):
    """Read all POSCARs from numbered directories in dimer_runs."""
    poscars = []
    positions_list = []
    
    # Find all numbered directories
    numbered_dirs = []
    for item in os.listdir(dimer_runs_dir):
        item_path = os.path.join(dimer_runs_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            numbered_dirs.append(int(item))
    
    # Sort directories numerically
    numbered_dirs.sort()
    
    print(f"Found {len(numbered_dirs)} dimer directories")
    
    for dir_num in numbered_dirs:
        dir_path = os.path.join(dimer_runs_dir, str(dir_num))
        poscar_path = os.path.join(dir_path, 'POSCAR')
        
        if os.path.exists(poscar_path):
            try:
                poscar = Poscar.from_file(poscar_path)
                poscars.append(poscar)
                positions_list.append(poscar.structure.cart_coords)
                print(f"Read POSCAR from directory {dir_num}")
            except Exception as e:
                print(f"Error reading POSCAR from directory {dir_num}: {e}")
    
    if not poscars:
        raise ValueError(f"No valid POSCARs found in {dimer_runs_dir}")
    
    # Analyze trajectory
    stats = analyze_trajectory(positions_list)
    print(f"\nTrajectory Statistics:")
    print(f"  Maximum displacement: {stats.get('max_displacement', 0):.3f} Å (atom {stats.get('max_disp_atom', 'N/A')})")
    print(f"  Mean displacement: {stats.get('mean_displacement', 0):.3f} Å")
    print(f"  Atoms that moved > 0.01 Å: {stats.get('n_atoms_moved', 0)}")
    
    # Determine moving indices
    if auto_detect and manual_indices is None:
        moving_indices = detect_moving_atoms(positions_list)
        if not moving_indices:
            print("\nNo significantly moving atoms detected. Using atom with largest displacement.")
            moving_indices = [stats.get('max_disp_atom', 0)]
    else:
        moving_indices = manual_indices or [0]
        print(f"\nUsing manually specified moving atoms: {moving_indices}")
    
    return poscars, positions_list, numbered_dirs, moving_indices


def create_animation(poscars, positions_list, moving_indices, output_file='dimer_animation.gif', 
                    zoom_factor=8.0, view_angles=(35.26, 45)):
    """Create animation of the dimer run with highlighted moving atoms."""
    
    # Get the first structure for setup
    structure = poscars[0].structure
    n_atoms = len(structure)
    
    # Get colors for all atoms
    atom_colors = get_element_colors(structure)
    atom_sizes = [300] * n_atoms  # Default size
    
    # Highlight moving atoms
    moving_color = '#FF0000'  # Red for moving atoms
    for idx in moving_indices:
        atom_colors[idx] = moving_color
        atom_sizes[idx] = 500  # Larger size for moving atoms
    
    # Create figure with better background
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    # Calculate center of mass for better zooming
    all_positions = np.vstack(positions_list)
    
    # Focus on moving atoms if specified, otherwise use center of mass
    if moving_indices:
        moving_atom_positions = np.vstack([pos[moving_indices] for pos in positions_list])
        center = moving_atom_positions.mean(axis=0)
        max_displacement = np.max(np.abs(moving_atom_positions - center))
    else:
        center = all_positions.mean(axis=0)
        max_displacement = np.std(all_positions)
    
    # Set tighter limits for zoom
    margin = max_displacement + zoom_factor
    
    ax.set_xlim([center[0] - margin, center[0] + margin])
    ax.set_ylim([center[1] - margin, center[1] + margin])
    ax.set_zlim([center[2] - margin, center[2] + margin])
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Set viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Initial scatter plot
    positions = positions_list[0]
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=atom_colors, s=atom_sizes, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    # Add title with better positioning
    title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, fontsize=16, 
                      ha='center', weight='bold')
    
    # Add legend with better positioning
    from matplotlib.patches import Patch
    legend_elements = []
    
    # Get unique elements
    unique_elements = list(set([site.specie.symbol for site in structure]))
    for element in unique_elements:
        # Find color for this element
        for i, site in enumerate(structure):
            if site.specie.symbol == element and i not in moving_indices:
                color = get_element_colors(structure)[i]
                legend_elements.append(Patch(facecolor=color, edgecolor='black', label=element))
                break
    
    # Add moving atom to legend
    if moving_indices:
        elements_of_moving = set([structure[idx].specie.symbol for idx in moving_indices])
        label = f'Moving atoms ({", ".join(elements_of_moving)})'
        legend_elements.append(Patch(facecolor=moving_color, edgecolor='black', label=label))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.8)
    
    # Draw unit cell with thinner, more transparent lines
    lattice = structure.lattice.matrix
    cell_lines = []
    
    # Define unit cell edges
    edges = [
        ([0, 0, 0], [1, 0, 0]),
        ([0, 0, 0], [0, 1, 0]),
        ([0, 0, 0], [0, 0, 1]),
        ([1, 0, 0], [1, 1, 0]),
        ([1, 0, 0], [1, 0, 1]),
        ([0, 1, 0], [1, 1, 0]),
        ([0, 1, 0], [0, 1, 1]),
        ([0, 0, 1], [1, 0, 1]),
        ([0, 0, 1], [0, 1, 1]),
        ([1, 1, 0], [1, 1, 1]),
        ([1, 0, 1], [1, 1, 1]),
        ([0, 1, 1], [1, 1, 1])
    ]
    
    for edge in edges:
        start = np.dot(edge[0], lattice)
        end = np.dot(edge[1], lattice)
        line, = ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                       'gray', alpha=0.2, linewidth=0.5, linestyle='--')
        cell_lines.append(line)
    
    # Add trajectory traces for moving atoms
    if moving_indices:
        for idx in moving_indices[:5]:  # Limit to 5 atoms for clarity
            trajectory_line, = ax.plot([], [], [], 'r-', alpha=0.3, linewidth=1)
    
    # Animation update function
    def update(frame):
        positions = positions_list[frame]
        
        # Update scatter plot positions
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        # Update title
        if frame > 0 and moving_indices:
            displacements = [np.linalg.norm(positions[idx] - positions_list[0][idx]) 
                           for idx in moving_indices]
            max_disp = max(displacements)
            title.set_text(f'Step {frame + 1}/{len(positions_list)} | Max displacement: {max_disp:.3f} Å')
        else:
            title.set_text(f'Step {frame + 1}/{len(positions_list)}')
        
        return scatter, title
    
    # Create animation with better settings
    anim = animation.FuncAnimation(fig, update, frames=len(positions_list),
                                  interval=400, blit=False, repeat=True)
    
    # Make the plot area larger by reducing margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    # Save animation
    print(f"\nSaving animation to {output_file}...")
    if output_file.endswith('.gif'):
        anim.save(output_file, writer='pillow', fps=2)
    elif output_file.endswith('.mp4'):
        anim.save(output_file, writer='ffmpeg', fps=2)
    else:
        print("Unknown file format. Saving as GIF.")
        anim.save(output_file + '.gif', writer='pillow', fps=2)
    
    print(f"Animation saved to {output_file}")
    
    # Also show the plot
    plt.show()


def create_trajectory_plot(positions_list, moving_indices, output_file='trajectory_plot.png'):
    """Create a static plot showing the trajectory of the moving atoms."""
    
    if not moving_indices:
        print("No moving atoms to plot trajectory for.")
        return
    
    # Create figure with subplots for each moving atom (up to 4)
    n_atoms_to_plot = min(len(moving_indices), 4)
    fig, axes = plt.subplots(n_atoms_to_plot, 1, figsize=(10, 3*n_atoms_to_plot))
    if n_atoms_to_plot == 1:
        axes = [axes]
    
    steps = np.arange(len(positions_list))
    
    for i, atom_idx in enumerate(moving_indices[:n_atoms_to_plot]):
        ax = axes[i]
        
        # Extract positions for this atom
        atom_positions = np.array([pos[atom_idx] for pos in positions_list])
        
        # Plot x, y, z coordinates
        ax.plot(steps, atom_positions[:, 0], 'b-', label='X', linewidth=2)
        ax.plot(steps, atom_positions[:, 1], 'g-', label='Y', linewidth=2)
        ax.plot(steps, atom_positions[:, 2], 'r-', label='Z', linewidth=2)
        
        ax.set_ylabel('Coordinate (Å)')
        ax.set_title(f'Atom {atom_idx} Trajectory')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if i == n_atoms_to_plot - 1:
            ax.set_xlabel('Dimer Step')
    
    plt.tight_layout()
    # Use plots directory relative to the output file if available
    output_dir = Path(output_file).parent if 'output_file' in locals() else Path('plots')
    output_dir.mkdir(exist_ok=True)
    coord_output = output_dir / 'dimer_coordinates.png'
    fig.savefig(coord_output, dpi=150)
    
    # Create displacement plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for i, atom_idx in enumerate(moving_indices[:5]):  # Limit to 5 atoms
        atom_positions = np.array([pos[atom_idx] for pos in positions_list])
        displacements = np.linalg.norm(atom_positions - atom_positions[0], axis=1)
        ax2.plot(steps, displacements, 'o-', label=f'Atom {atom_idx}', 
                linewidth=2, markersize=4, alpha=0.8)
    
    ax2.set_xlabel('Dimer Step')
    ax2.set_ylabel('Total Displacement (Å)')
    ax2.set_title('Atomic Displacements from Initial Position')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    disp_output = output_dir / 'dimer_displacement.png'
    fig2.savefig(disp_output, dpi=150)
    
    print(f"Trajectory plots saved")


def main():
    parser = argparse.ArgumentParser(description='Animate POSCARs from dimer runs')
    parser.add_argument('--dimer-runs-dir', type=str, default='vasp_runs/dimer_runs',
                       help='Path to dimer_runs directory')
    parser.add_argument('--moving-indices', type=int, nargs='+', default=None,
                       help='Indices of moving atoms (0-based). If not specified, will auto-detect.')
    parser.add_argument('--auto-detect', action='store_true', default=True,
                       help='Automatically detect moving atoms (default: True)')
    parser.add_argument('--no-auto-detect', dest='auto_detect', action='store_false',
                       help='Disable automatic detection of moving atoms')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Displacement threshold for auto-detection (Å)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output animation file (gif or mp4, default: saves to current run directory)')
    parser.add_argument('--trajectory-plot', action='store_true',
                       help='Also create static trajectory plots')
    parser.add_argument('--zoom', type=float, default=8.0,
                       help='Zoom factor (smaller = more zoomed in)')
    parser.add_argument('--elevation', type=float, default=35.26,
                       help='Viewing elevation angle (default: 35.26 for [111] view)')
    parser.add_argument('--azimuth', type=float, default=45,
                       help='Viewing azimuth angle (default: 45 for [111] view)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.dimer_runs_dir):
        print(f"Error: Directory {args.dimer_runs_dir} does not exist!")
        return
    
    # Read all POSCARs
    try:
        # Set threshold for auto-detection
        if args.auto_detect and args.moving_indices is None:
            print(f"Auto-detection enabled with threshold: {args.threshold} Å")
        
        poscars, positions_list, dir_numbers, moving_indices = read_poscar_sequence(
            args.dimer_runs_dir, 
            auto_detect=args.auto_detect,
            manual_indices=args.moving_indices
        )
        
        print(f"\nSuccessfully read {len(poscars)} POSCARs")
        print(f"Animating {len(moving_indices)} moving atoms: {moving_indices}")
        
        # Set default output path if not specified
        if args.output is None:
            # Try to infer run directory from dimer_runs path
            dimer_path = Path(args.dimer_runs_dir)
            if 'outputs' in dimer_path.parts and 'vasp_runs' in dimer_path.parts:
                # Path like outputs/run_*/vasp_runs/dimer_runs
                run_dir = dimer_path.parent.parent  # Go up to run directory
                plots_dir = run_dir / 'plots'
                plots_dir.mkdir(exist_ok=True)
                args.output = str(plots_dir / 'dimer_animation.gif')
            else:
                # Initialize OutputManager only when needed
                from output_manager import OutputManager
                try:
                    get_output_path('test')
                except RuntimeError:
                    OutputManager.setup()
                args.output = get_output_path('plots', 'dimer_animation.gif')
        elif not os.path.isabs(args.output):
            # If relative path, try to put it in appropriate plots directory
            dimer_path = Path(args.dimer_runs_dir)
            if 'outputs' in dimer_path.parts and 'vasp_runs' in dimer_path.parts:
                run_dir = dimer_path.parent.parent
                plots_dir = run_dir / 'plots'
                plots_dir.mkdir(exist_ok=True)
                args.output = str(plots_dir / args.output)
            else:
                # Use current directory
                args.output = args.output
        
        # Create animation
        create_animation(poscars, positions_list, moving_indices, args.output,
                        zoom_factor=args.zoom, view_angles=(args.elevation, args.azimuth))
        
        # Create trajectory plots if requested
        if args.trajectory_plot:
            create_trajectory_plot(positions_list, moving_indices, args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()