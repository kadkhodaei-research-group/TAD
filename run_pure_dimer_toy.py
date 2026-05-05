#!/usr/bin/env python
"""Run pure dimer method on toy model potentials."""

import argparse
import os
import sys
import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
from walker_pure_dimer_toy import WalkerPureDimerToy
from toy_model_interface import ToyModelInterface
from output_manager import OutputManager, get_output_path


def setup_logging():
    """Set up logging."""
    logging.getLogger().handlers.clear()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    log_file = get_output_path('logs', 'pure_dimer_toy_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")


def plot_trajectory(results: dict, potential_name: str):
    """Plot the dimer trajectory on the potential energy surface."""
    # Import 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with better layout - much larger
    fig = plt.figure(figsize=(32, 20))
    
    # Create more sophisticated subplot grid with more spacing
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1.3, 0.8], width_ratios=[1.4, 1.1, 1.1, 1.1], 
                          hspace=0.35, wspace=0.35)
    
    # Main plots
    ax1 = fig.add_subplot(gs[0:2, 0])  # Main contour plot (larger, left side)
    ax3d = fig.add_subplot(gs[0, 1], projection='3d')  # 3D surface plot
    ax_vec = fig.add_subplot(gs[0, 2])  # Vector field plot
    ax_energy_profile = fig.add_subplot(gs[0, 3])  # Energy along dimer
    
    # History plots
    ax2 = fig.add_subplot(gs[1, 1])  # Energy/Force history
    ax3 = fig.add_subplot(gs[1, 2:])  # Curvature history
    
    # Bottom info
    ax_info = fig.add_subplot(gs[2, :])  # Information panel
    
    # Extract trajectory
    positions = np.array([pos for pos, _, _ in results['trajectory']])
    energies = np.array([e for _, e, _ in results['trajectory']])
    forces = np.array([f for _, _, f in results['trajectory']])
    force_mags = np.array([np.linalg.norm(f) for f in forces])
    
    # Get potential info
    pot_info = results['potential_info']
    domain = pot_info['domain']
    
    # Create finer grid for contour plot
    x = np.linspace(domain[0][0], domain[0][1], 200)
    y = np.linspace(domain[1][0], domain[1][1], 200)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate potential on grid
    from toy_model_interface import ToyModelInterface
    pes_interface = ToyModelInterface(potential_name, np.array([0, 0]))
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = pes_interface.energy_func(np.array([X[i, j], Y[i, j]]))
    
    # Plot 1: Enhanced contour with trajectory
    # Use filled contour for background
    contourf = ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    # Add contour lines without labels
    contour = ax1.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot full trajectory as a continuous line
    ax1.plot(positions[:, 0], positions[:, 1], 'w-', linewidth=3, alpha=0.8, 
             label='Trajectory', zorder=5)
    ax1.plot(positions[:, 0], positions[:, 1], 'k--', linewidth=1.5, alpha=0.5, zorder=5)
    
    # Add markers along the trajectory
    n_markers = min(10, len(positions))
    marker_indices = np.linspace(0, len(positions)-1, n_markers, dtype=int)
    for idx in marker_indices[1:-1]:  # Skip start and end
        ax1.plot(positions[idx, 0], positions[idx, 1], 'o', 
                color='yellow', markersize=6, markeredgecolor='black', 
                markeredgewidth=1, zorder=6)
    
    # Mark start and end points
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, 
             label=f'Start ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}) Å', 
             markeredgecolor='black', markeredgewidth=2, zorder=7)
    ax1.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=15, 
             label=f'End ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}) Å',
             markeredgecolor='black', markeredgewidth=2, zorder=7)
    
    # Add force vectors at selected points
    stride = max(1, len(positions) // 10)  # Show up to 10 force vectors
    for i in range(0, len(positions), stride):
        if i < len(forces):
            # Scale force vectors for visibility
            scale = 0.05 / (np.max(force_mags) + 1e-10)
            ax1.arrow(positions[i, 0], positions[i, 1], 
                     -forces[i, 0] * scale, -forces[i, 1] * scale,
                     head_width=0.02, head_length=0.01, fc='red', ec='red', alpha=0.5)
    
    ax1.set_xlabel('X (Å)', fontsize=16)
    ax1.set_ylabel('Y (Å)', fontsize=16)
    
    # Convert potential name to LaTeX notation
    latex_names = {
        'V8': r'$V = -\sin(\pi x)\sin(\pi y)$',
        'V4': r'$V = (y + x)^2 + 10(y - x^2)^2$',
        'V5': r'$V = -\sin(\pi x)\sin(\pi y)$',
        'Muller-Brown': 'Müller-Brown Potential',
        'Karplus': 'Karplus Potential',
        'three_hole_pot': 'Three-Hole Potential'
    }
    title_name = latex_names.get(potential_name, potential_name)
    ax1.set_title(f'{title_name} - Dimer Trajectory', fontsize=18)
    ax1.legend(loc='best', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Add colorbar with units
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('Energy (eV)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Plot 2: 3D plot - zoom in around trajectory
    # Determine zoom region based on trajectory bounds
    x_margin = 0.2
    y_margin = 0.2
    x_min, x_max = positions[:, 0].min() - x_margin, positions[:, 0].max() + x_margin
    y_min, y_max = positions[:, 1].min() - y_margin, positions[:, 1].max() + y_margin
    
    # Create finer grid for 3D visualization in zoomed region
    x_3d = np.linspace(x_min, x_max, 40)
    y_3d = np.linspace(y_min, y_max, 40)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    Z_3d = np.zeros_like(X_3d)
    for i in range(X_3d.shape[0]):
        for j in range(X_3d.shape[1]):
            Z_3d[i, j] = pes_interface.energy_func(np.array([X_3d[i, j], Y_3d[i, j]]))
    
    # Plot 3D surface with better coloring
    surf = ax3d.plot_surface(X_3d, Y_3d, Z_3d, cmap='coolwarm', alpha=0.7, 
                            antialiased=True, linewidth=0, edgecolor='none',
                            vmin=Z_3d.min(), vmax=Z_3d.max())
    
    # Add contour lines at the bottom
    contours = ax3d.contour(X_3d, Y_3d, Z_3d, zdir='z', offset=Z_3d.min()-0.1, 
                           cmap='gray', alpha=0.4, linewidths=1)
    
    # Plot trajectory on 3D surface with elevation
    traj_z = np.array([pes_interface.energy_func(pos) for pos in positions])
    # Elevate trajectory slightly above surface for visibility
    traj_z_elevated = traj_z + 0.02
    ax3d.plot(positions[:, 0], positions[:, 1], traj_z_elevated, 'black', 
              linewidth=4, alpha=0.8, zorder=10)
    ax3d.plot(positions[:, 0], positions[:, 1], traj_z_elevated, 'yellow', 
              linewidth=2, zorder=11)
    
    # Mark start and end points
    ax3d.scatter(positions[0, 0], positions[0, 1], traj_z_elevated[0], 
                color='green', s=200, edgecolor='black', linewidth=2, zorder=12)
    ax3d.scatter(positions[-1, 0], positions[-1, 1], traj_z_elevated[-1], 
                color='red', s=200, marker='*', edgecolor='black', linewidth=2, zorder=12)
    
    # Set 3D view angle for better visibility
    ax3d.view_init(elev=30, azim=-60)
    ax3d.set_xlabel('X (Å)', fontsize=14)
    ax3d.set_ylabel('Y (Å)', fontsize=14)
    ax3d.set_zlabel('Energy (eV)', fontsize=14)
    ax3d.set_title('3D View - Zoomed Region', fontsize=16)
    ax3d.tick_params(axis='both', which='major', labelsize=12)
    ax3d.set_box_aspect([1,1,0.5])  # Make z-axis shorter for better view
    
    # Plot 3: Vector field showing force directions
    # Create grid for vector field
    x_vec = np.linspace(x_min, x_max, 15)
    y_vec = np.linspace(y_min, y_max, 15)
    X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
    
    # Calculate forces at grid points
    U_vec = np.zeros_like(X_vec)
    V_vec = np.zeros_like(Y_vec)
    for i in range(X_vec.shape[0]):
        for j in range(X_vec.shape[1]):
            force = pes_interface.first_derivative(np.array([X_vec[i,j], Y_vec[i,j]]))
            U_vec[i,j] = force[0]
            V_vec[i,j] = force[1]
    
    # Normalize vectors for display
    magnitude = np.sqrt(U_vec**2 + V_vec**2)
    U_norm = U_vec / (magnitude + 1e-10)
    V_norm = V_vec / (magnitude + 1e-10)
    
    # Create zoomed contour background for vector field
    X_vec_bg, Y_vec_bg = np.meshgrid(x_vec, y_vec)
    Z_vec_bg = np.zeros_like(X_vec_bg)
    for i in range(X_vec_bg.shape[0]):
        for j in range(X_vec_bg.shape[1]):
            Z_vec_bg[i, j] = pes_interface.energy_func(np.array([X_vec_bg[i, j], Y_vec_bg[i, j]]))
    
    # Plot vector field
    ax_vec.contourf(X_vec_bg, Y_vec_bg, Z_vec_bg, levels=20, cmap='RdBu_r', alpha=0.3)
    ax_vec.quiver(X_vec, Y_vec, U_norm, V_norm, magnitude, 
                  cmap='plasma', scale=20, width=0.003, 
                  headwidth=3, headlength=4)
    
    # Overlay trajectory
    ax_vec.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=2, alpha=0.7)
    ax_vec.plot(positions[0, 0], positions[0, 1], 'go', markersize=10)
    ax_vec.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=12)
    
    ax_vec.set_xlabel('X (Å)', fontsize=14)
    ax_vec.set_ylabel('Y (Å)', fontsize=14)
    ax_vec.set_title('Force Vector Field', fontsize=16)
    ax_vec.set_aspect('equal')
    ax_vec.grid(True, alpha=0.3)
    ax_vec.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 4: Energy profile along dimer direction
    if 'table_history' in results and len(results['table_history']) > 0:
        # For each step, calculate energy along dimer direction
        n_points = 50
        dimer_range = np.linspace(-0.2, 0.2, n_points)
        
        # Take a few representative steps
        n_profiles = min(5, len(positions))
        step_indices = np.linspace(0, len(positions)-1, n_profiles, dtype=int)
        
        for idx in step_indices:
            pos = positions[idx]
            # Get dimer direction (approximate from trajectory)
            if idx < len(positions) - 1:
                direction = positions[idx+1] - positions[idx]
            else:
                direction = positions[idx] - positions[idx-1]
            
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1.0, 0.0])
            
            # Calculate energy along dimer
            energies_along = []
            for s in dimer_range:
                test_pos = pos + s * direction
                energies_along.append(pes_interface.energy_func(test_pos))
            
            # Plot with gradient color
            color = plt.cm.viridis(idx / (len(positions)-1))
            ax_energy_profile.plot(dimer_range, energies_along, 
                                 color=color, alpha=0.7, 
                                 label=f'Step {idx+1}' if idx in [0, len(positions)-1] else '')
        
        ax_energy_profile.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax_energy_profile.set_xlabel('Distance along dimer (Å)', fontsize=14)
        ax_energy_profile.set_ylabel('Energy (eV)', fontsize=14)
        ax_energy_profile.set_title('Energy Profile Along Dimer', fontsize=16)
        ax_energy_profile.grid(True, alpha=0.3)
        ax_energy_profile.legend(fontsize=12)
        ax_energy_profile.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 5: Energy and force magnitude vs steps
    steps = np.arange(len(energies))
    
    ax2_energy = ax2
    ax2_force = ax2.twinx()
    
    # Energy plot with markers
    line1 = ax2_energy.plot(steps, energies, 'b-o', markersize=4, label='Energy')
    ax2_energy.set_xlabel('Step', fontsize=16)
    ax2_energy.set_ylabel('Energy (eV)', color='b', fontsize=16)
    ax2_energy.tick_params(axis='y', labelcolor='b', labelsize=14)
    ax2_energy.grid(True, alpha=0.3)
    
    # Force magnitude plot on LINEAR scale
    line2 = ax2_force.plot(steps, force_mags, 'r-s', markersize=4, label='|Force|')
    ax2_force.set_ylabel('Force Magnitude (eV/Å)', color='r', fontsize=16)
    ax2_force.tick_params(axis='y', labelcolor='r', labelsize=14)
    
    # Add convergence threshold line
    ax2_force.axhline(y=results.get('convergence_threshold', 0.01), 
                      color='g', linestyle='--', linewidth=2, alpha=0.7, label='Convergence')
    
    ax2.set_title('Convergence History', fontsize=18)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best', fontsize=14)
    
    # Plot 6: Curvature history
    if 'table_history' in results and results['table_history']:
        curvatures = [entry.get('Curvature', np.nan) for entry in results['table_history']]
        valid_curv = [c for c in curvatures if not np.isnan(c)]
        
        if valid_curv:
            ax3.plot(steps[:len(curvatures)], curvatures, 'g-o', markersize=6, 
                    linewidth=2, label='Curvature', markeredgecolor='darkgreen')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
            ax3.set_xlabel('Step', fontsize=16)
            ax3.set_ylabel('Curvature (eV/Å²)', fontsize=16)
            ax3.set_title('Curvature Evolution', fontsize=18)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', which='major', labelsize=14)
            
            # Highlight negative curvature regions (saddle points)
            neg_curv = np.array(curvatures) < 0
            if np.any(neg_curv):
                ax3.fill_between(steps[:len(curvatures)], 
                               np.min(curvatures) * 1.1, 0, 
                               where=neg_curv, alpha=0.2, color='red', 
                               label='Saddle Region')
                ax3.legend(loc='best', fontsize=14)
            
            # Add final curvature annotation
            if len(curvatures) > 0 and not np.isnan(curvatures[-1]):
                ax3.annotate(f'Final: {curvatures[-1]:.2f} eV/Å²', 
                           xy=(len(curvatures)-1, curvatures[-1]),
                           xytext=(len(curvatures)-1, curvatures[-1] + 0.5),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                           fontsize=14, ha='center')
    else:
        ax3.text(0.5, 0.5, 'No curvature data available', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=16)
        ax3.set_title('Curvature Evolution', fontsize=18)
    
    # Bottom information panel with detailed summary
    ax_info.axis('off')
    
    # Create formatted summary table
    info_data = []
    info_data.append(['Parameter', 'Initial', 'Final', 'Change'])
    info_data.append(['Position X (Å)', f'{positions[0, 0]:.6f}', f'{positions[-1, 0]:.6f}', 
                     f'{positions[-1, 0] - positions[0, 0]:.6f}'])
    info_data.append(['Position Y (Å)', f'{positions[0, 1]:.6f}', f'{positions[-1, 1]:.6f}', 
                     f'{positions[-1, 1] - positions[0, 1]:.6f}'])
    info_data.append(['Energy (eV)', f'{energies[0]:.6f}', f'{energies[-1]:.6f}', 
                     f'{energies[-1] - energies[0]:.6f}'])
    info_data.append(['|Force| (eV/Å)', f'{force_mags[0]:.6f}', f'{force_mags[-1]:.6e}', 
                     f'{force_mags[-1] - force_mags[0]:.6f}'])
    
    if 'final_curvature' in results and not np.isnan(results['final_curvature']):
        info_data.append(['Curvature (eV/Å²)', 'N/A', f'{results["final_curvature"]:.3f}', 
                         'Saddle' if results['final_curvature'] < 0 else 'Not Saddle'])
    
    # Create table
    table = ax_info.table(cellText=info_data[1:], colLabels=info_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.5)
    
    # Style the table
    for i in range(len(info_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#2C3E50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 3 and 'Saddle' in str(info_data[i][j]):
                    cell.set_facecolor('#E8F8F5')
                    cell.set_text_props(color='green', weight='bold')
                else:
                    cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    
    # Add additional info text
    add_info = f"Total Steps: {results['steps']} | Total Evaluations: {results['total_evaluations']} | "
    add_info += f"Converged: {'Yes' if results['converged'] else 'No'} | "
    add_info += f"Runtime: {results.get('runtime', 0):.2f}s"
    ax_info.text(0.5, 0.1, add_info, transform=ax_info.transAxes, 
                ha='center', fontsize=16, weight='bold')
    
    plt.suptitle(f'Dimer Method Results - {title_name}', fontsize=24, y=0.98)
    try:
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    except:
        # Fallback if tight_layout fails with mixed subplot types
        plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.02)
    
    # Save plot
    plot_file = get_output_path('plots', f'dimer_trajectory_{potential_name}.png')
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    print(f"\nTrajectory plot saved to: {plot_file}")
    plt.close()


def pure_dimer_toy_search(
    potential_name: str,
    initial_position: np.ndarray,
    system_params: dict
) -> tuple:
    """Run pure dimer search on toy model potential."""
    
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info(f"Starting pure dimer search on {potential_name} potential")
    
    try:
        # Create toy model interface
        local_pes = ToyModelInterface(potential_name, initial_position)
        
        # Extract dimer parameters
        dimer_params = {
            'max_dimer_steps': system_params.get('max_dimer_steps', 100),
            'rotation': system_params.get('rotation_method', 'lbfgsext'),
            'translation': system_params.get('translation_method', 'lbfgs'),
            'dimer_sep': system_params.get('dimer_sep', 0.01),
            'T_anglerot': system_params.get('T_anglerot', 0.01),
            'T_anglerot_init': system_params.get('T_anglerot_init', 0.0873),
            'max_dimer_rotations': system_params.get('max_dimer_rotations', 10),
            'num_init_rotations': system_params.get('num_init_rotations', 5),
            'dimer_stopping_criteria': system_params.get('dimer_stopping_criteria', 0.01),
            'step_size': system_params.get('step_size', 0.05),
            'max_step_size': system_params.get('max_step_size', 0.1),
            'verbose': system_params.get('verbose', True),
            'checkpoint_interval': system_params.get('checkpoint_interval', 10)
        }
        
        # Create walker
        walker = WalkerPureDimerToy(
            initial_position=initial_position,
            local_pes=local_pes,
            **dimer_params
        )
        
        # Run dimer optimization
        results = walker.run_dimer()
        
        # Add convergence threshold to results for plotting
        results['convergence_threshold'] = dimer_params['dimer_stopping_criteria']
        
        # Save results
        results_file = get_output_path('results', f'dimer_toy_{potential_name}_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {results_file}")
        
        # Generate plots
        plot_trajectory(results, potential_name)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Potential: {potential_name}")
        print(f"Initial position: [{initial_position[0]:.4f}, {initial_position[1]:.4f}]")
        print(f"Converged: {results['converged']}")
        print(f"Steps taken: {results['steps']}")
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"Final position: [{results['final_position'][0]:.4f}, {results['final_position'][1]:.4f}]")
        print(f"Final energy: {results['final_energy']:.6f}")
        print(f"Final force magnitude: {results['final_force_magnitude']:.6e}")
        if 'final_curvature' in results:
            print(f"Final curvature: {results['final_curvature']:.6f}")
            if results['final_curvature'] < 0:
                print("SUCCESS: Found saddle point (negative curvature)!")
            else:
                print("WARNING: Not at saddle point (positive curvature)")
        print("="*60)
        
        return walker, results
        
    except Exception as e:
        logging.error(f"Pure dimer search failed: {str(e)}")
        raise
    finally:
        # Restore stdout
        sys.stdout = tee.terminal


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run pure dimer method on toy model potentials'
    )
    
    # Potential selection
    parser.add_argument(
        '--potential',
        type=str,
        default='Muller-Brown',
        choices=['Muller-Brown', 'Leps', 'Leps-HO', 'Karplus', 'Wolfe-Quapp', 
                 'three_hole_pot', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'],
        help='Toy potential to use'
    )
    
    # Initial position
    parser.add_argument(
        '--initial-x',
        type=float,
        default=-0.5,
        help='Initial X coordinate'
    )
    parser.add_argument(
        '--initial-y',
        type=float,
        default=0.5,
        help='Initial Y coordinate'
    )
    
    # Dimer parameters
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum number of dimer steps'
    )
    parser.add_argument(
        '--rotation-method',
        type=str,
        default='lbfgsext',
        choices=['lbfgsext', 'lbfgs', 'cg', 'mn'],
        help='Dimer rotation method'
    )
    parser.add_argument(
        '--translation-method',
        type=str,
        default='lbfgs',
        choices=['lbfgs', 'cg', 'newton', 'qmvv'],
        help='Dimer translation method'
    )
    parser.add_argument(
        '--dimer-sep',
        type=float,
        default=0.01,
        help='Dimer separation distance'
    )
    parser.add_argument(
        '--convergence',
        type=float,
        default=0.01,
        help='Force convergence threshold'
    )
    parser.add_argument(
        '--step-size',
        type=float,
        default=0.05,
        help='Base step size'
    )
    parser.add_argument(
        '--max-step-size',
        type=float,
        default=0.1,
        help='Maximum step size'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Base directory for outputs'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this run'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Initialize output directory structure
    OutputManager.setup(
        base_dir=args.output_dir,
        run_name=args.run_name or f"toy_{args.potential}"
    )
    
    setup_logging()
    
    # Save run metadata
    OutputManager.save_run_metadata({
        'script': 'run_pure_dimer_toy.py',
        'potential': args.potential,
        'initial_position': [args.initial_x, args.initial_y],
        'parameters': vars(args)
    })
    
    # Set up system parameters
    system_params = {
        'max_dimer_steps': args.max_steps,
        'rotation_method': args.rotation_method,
        'translation_method': args.translation_method,
        'dimer_sep': args.dimer_sep,
        'dimer_stopping_criteria': args.convergence,
        'step_size': args.step_size,
        'max_step_size': args.max_step_size,
        'verbose': args.verbose,
        'checkpoint_interval': 10
    }
    
    # Initial position
    initial_position = np.array([args.initial_x, args.initial_y])
    
    # Run dimer search
    walker, results = pure_dimer_toy_search(
        potential_name=args.potential,
        initial_position=initial_position,
        system_params=system_params
    )
    
    return 0 if results['converged'] else 1


if __name__ == "__main__":
    sys.exit(main())