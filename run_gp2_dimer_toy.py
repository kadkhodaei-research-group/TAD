#!/usr/bin/env python
"""Run GP2-accelerated dimer method on toy model potentials."""

import argparse
import os
import sys
import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
from walker_gp2_dimer_toy import WalkerGP2DimerToy
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
    
    log_file = get_output_path('logs', 'gp2_dimer_toy_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")


def plot_gp2_trajectory(results: dict, potential_name: str):
    """Plot the GP2 dimer trajectory with additional GP2-specific visualizations."""
    # Import 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with comprehensive layout
    fig = plt.figure(figsize=(32, 24))
    
    # Create subplot grid
    gs = fig.add_gridspec(4, 4, height_ratios=[1.3, 1.3, 1.0, 0.8], 
                          width_ratios=[1.4, 1.1, 1.1, 1.1], 
                          hspace=0.35, wspace=0.35)
    
    # Main plots
    ax1 = fig.add_subplot(gs[0:2, 0])  # Main contour plot
    ax3d = fig.add_subplot(gs[0, 1], projection='3d')  # 3D surface plot
    ax_vec = fig.add_subplot(gs[0, 2])  # Vector field plot
    ax_gp2_pred = fig.add_subplot(gs[0, 3])  # GP2 predictions
    
    # History plots
    ax2 = fig.add_subplot(gs[1, 1])  # Energy/Force history
    ax3 = fig.add_subplot(gs[1, 2])  # Curvature history
    ax_gp2_usage = fig.add_subplot(gs[1, 3])  # GP2 usage
    
    # GP2 analysis plots
    ax_gp2_error = fig.add_subplot(gs[2, 0:2])  # GP2 prediction errors
    ax_gp2_uncertainty = fig.add_subplot(gs[2, 2:])  # GP2 uncertainty
    
    # Bottom info
    ax_info = fig.add_subplot(gs[3, :])  # Information panel
    
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
    contourf = ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    contour = ax1.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot full trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'w-', linewidth=3, alpha=0.8, 
             label='Trajectory', zorder=5)
    ax1.plot(positions[:, 0], positions[:, 1], 'k--', linewidth=1.5, alpha=0.5, zorder=5)
    
    # Mark GP2 evaluation points
    if 'gp2_predictions' in results and results['gp2_predictions']:
        gp2_positions = np.array([pred['position'] for pred in results['gp2_predictions']])
        ax1.scatter(gp2_positions[:, 0], gp2_positions[:, 1], 
                   c='cyan', s=30, marker='o', edgecolor='black', 
                   linewidth=0.5, alpha=0.7, label='GP2 Used', zorder=6)
    
    # Mark training points
    if 'table_history' in results:
        true_eval_mask = np.array([entry['Evaluations'] > 0 for entry in results['table_history']])
        if np.any(true_eval_mask):
            # table_history includes initial position
            if len(positions) == len(true_eval_mask):
                true_eval_positions = positions[true_eval_mask]
            else:
                # Handle mismatch in lengths
                min_len = min(len(positions), len(true_eval_mask))
                true_eval_positions = positions[:min_len][true_eval_mask[:min_len]]
            ax1.scatter(true_eval_positions[:, 0], true_eval_positions[:, 1], 
                       c='red', s=40, marker='s', edgecolor='black', 
                       linewidth=0.5, alpha=0.7, label='True Eval', zorder=6)
    
    # Mark start and end points
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, 
             label=f'Start ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}) Å', 
             markeredgecolor='black', markeredgewidth=2, zorder=7)
    ax1.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=15, 
             label=f'End ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}) Å',
             markeredgecolor='black', markeredgewidth=2, zorder=7)
    
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
    ax1.set_title(f'{title_name} - GP2 Dimer Trajectory', fontsize=18)
    ax1.legend(loc='best', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('Energy (eV)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Plot 2: 3D plot - zoom in around trajectory
    x_margin = 0.2
    y_margin = 0.2
    x_min, x_max = positions[:, 0].min() - x_margin, positions[:, 0].max() + x_margin
    y_min, y_max = positions[:, 1].min() - y_margin, positions[:, 1].max() + y_margin
    
    x_3d = np.linspace(x_min, x_max, 40)
    y_3d = np.linspace(y_min, y_max, 40)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    Z_3d = np.zeros_like(X_3d)
    for i in range(X_3d.shape[0]):
        for j in range(X_3d.shape[1]):
            Z_3d[i, j] = pes_interface.energy_func(np.array([X_3d[i, j], Y_3d[i, j]]))
    
    surf = ax3d.plot_surface(X_3d, Y_3d, Z_3d, cmap='coolwarm', alpha=0.7, 
                            antialiased=True, linewidth=0, edgecolor='none')
    
    # Plot trajectory on 3D surface
    traj_z = np.array([pes_interface.energy_func(pos) for pos in positions])
    traj_z_elevated = traj_z + 0.02
    ax3d.plot(positions[:, 0], positions[:, 1], traj_z_elevated, 'black', 
              linewidth=4, alpha=0.8, zorder=10)
    
    ax3d.view_init(elev=30, azim=-60)
    ax3d.set_xlabel('X (Å)', fontsize=14)
    ax3d.set_ylabel('Y (Å)', fontsize=14)
    ax3d.set_zlabel('Energy (eV)', fontsize=14)
    ax3d.set_title('3D View - Zoomed Region', fontsize=16)
    ax3d.tick_params(axis='both', which='major', labelsize=12)
    ax3d.set_box_aspect([1,1,0.5])
    
    # Plot 3: Vector field
    x_vec = np.linspace(x_min, x_max, 15)
    y_vec = np.linspace(y_min, y_max, 15)
    X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
    
    U_vec = np.zeros_like(X_vec)
    V_vec = np.zeros_like(Y_vec)
    for i in range(X_vec.shape[0]):
        for j in range(X_vec.shape[1]):
            force = pes_interface.first_derivative(np.array([X_vec[i,j], Y_vec[i,j]]))
            U_vec[i,j] = force[0]
            V_vec[i,j] = force[1]
    
    magnitude = np.sqrt(U_vec**2 + V_vec**2)
    U_norm = U_vec / (magnitude + 1e-10)
    V_norm = V_vec / (magnitude + 1e-10)
    
    ax_vec.quiver(X_vec, Y_vec, U_norm, V_norm, magnitude, 
                  cmap='plasma', scale=20, width=0.003)
    ax_vec.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=2, alpha=0.7)
    ax_vec.set_xlabel('X (Å)', fontsize=14)
    ax_vec.set_ylabel('Y (Å)', fontsize=14)
    ax_vec.set_title('Force Vector Field', fontsize=16)
    ax_vec.set_aspect('equal')
    ax_vec.grid(True, alpha=0.3)
    ax_vec.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 4: GP2 predictions vs actual
    if 'gp2_predictions' in results and results['gp2_predictions']:
        gp2_preds = results['gp2_predictions']
        pred_steps = np.arange(len(gp2_preds))
        actual_e = np.array([p['actual_energy'] for p in gp2_preds])
        pred_e = np.array([p['energy_pred'] for p in gp2_preds])
        pred_std = np.sqrt(np.array([p['energy_var'] for p in gp2_preds]))
        
        ax_gp2_pred.errorbar(pred_steps, pred_e, yerr=2*pred_std, 
                            fmt='b-', alpha=0.7, label='GP2 Prediction ±2σ')
        ax_gp2_pred.plot(pred_steps, actual_e, 'r--', label='Actual Energy')
        ax_gp2_pred.set_xlabel('GP2 Evaluation', fontsize=14)
        ax_gp2_pred.set_ylabel('Energy (eV)', fontsize=14)
        ax_gp2_pred.set_title('GP2 Energy Predictions', fontsize=16)
        ax_gp2_pred.legend(fontsize=12)
        ax_gp2_pred.grid(True, alpha=0.3)
        ax_gp2_pred.tick_params(axis='both', which='major', labelsize=12)
    else:
        ax_gp2_pred.text(0.5, 0.5, 'No GP2 predictions yet', 
                        transform=ax_gp2_pred.transAxes, ha='center', va='center', fontsize=14)
        ax_gp2_pred.set_title('GP2 Energy Predictions', fontsize=16)
    
    # Plot 5: Energy and force magnitude vs steps
    steps = np.arange(len(energies))
    
    ax2_energy = ax2
    ax2_force = ax2.twinx()
    
    line1 = ax2_energy.plot(steps, energies, 'b-o', markersize=4, label='Energy')
    ax2_energy.set_xlabel('Step', fontsize=16)
    ax2_energy.set_ylabel('Energy (eV)', color='b', fontsize=16)
    ax2_energy.tick_params(axis='y', labelcolor='b', labelsize=14)
    ax2_energy.grid(True, alpha=0.3)
    
    line2 = ax2_force.plot(steps, force_mags, 'r-s', markersize=4, label='|Force|')
    ax2_force.set_ylabel('Force Magnitude (eV/Å)', color='r', fontsize=16)
    ax2_force.tick_params(axis='y', labelcolor='r', labelsize=14)
    ax2_force.axhline(y=results.get('convergence_threshold', 0.01), 
                      color='g', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_title('Convergence History', fontsize=18)
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
            
            neg_curv = np.array(curvatures) < 0
            if np.any(neg_curv):
                ax3.fill_between(steps[:len(curvatures)], 
                               np.min(curvatures) * 1.1, 0, 
                               where=neg_curv, alpha=0.2, color='red', 
                               label='Saddle Region')
                ax3.legend(loc='best', fontsize=14)
    
    # Plot 7: GP2 usage rate
    if 'gp2_evals_per_step' in results and 'force_evals_per_step' in results:
        gp2_evals = np.array(results['gp2_evals_per_step'])
        true_evals = np.array(results['force_evals_per_step'])
        total_per_step = gp2_evals + true_evals
        gp2_rate = np.zeros_like(gp2_evals, dtype=float)
        mask = total_per_step > 0
        gp2_rate[mask] = gp2_evals[mask] / total_per_step[mask] * 100
        
        ax_gp2_usage.bar(np.arange(len(gp2_rate)), gp2_rate, color='cyan', alpha=0.7)
        ax_gp2_usage.set_xlabel('Step', fontsize=16)
        ax_gp2_usage.set_ylabel('GP2 Usage (%)', fontsize=16)
        ax_gp2_usage.set_title('GP2 Usage Rate', fontsize=18)
        ax_gp2_usage.set_ylim(0, 105)
        ax_gp2_usage.grid(True, alpha=0.3, axis='y')
        ax_gp2_usage.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 8: GP2 prediction errors
    if 'gp2_predictions' in results and results['gp2_predictions']:
        gp2_preds = results['gp2_predictions']
        pred_steps = np.arange(len(gp2_preds))
        
        # Energy errors
        actual_e = np.array([p['actual_energy'] for p in gp2_preds])
        pred_e = np.array([p['energy_pred'] for p in gp2_preds])
        energy_errors = np.abs(pred_e - actual_e)
        
        ax_gp2_error.plot(pred_steps, energy_errors, 'b-o', label='Energy Error')
        ax_gp2_error.set_xlabel('GP2 Evaluation', fontsize=16)
        ax_gp2_error.set_ylabel('Absolute Error (eV)', fontsize=16)
        ax_gp2_error.set_title('GP2 Prediction Errors', fontsize=18)
        ax_gp2_error.legend(fontsize=14)
        ax_gp2_error.grid(True, alpha=0.3)
        ax_gp2_error.tick_params(axis='both', which='major', labelsize=14)
        ax_gp2_error.set_yscale('log')
    else:
        ax_gp2_error.text(0.5, 0.5, 'No GP2 predictions available', 
                         transform=ax_gp2_error.transAxes, ha='center', va='center', fontsize=16)
        ax_gp2_error.set_title('GP2 Prediction Errors', fontsize=18)
    
    # Plot 9: GP2 uncertainty evolution
    if 'gp2_predictions' in results and results['gp2_predictions']:
        gp2_preds = results['gp2_predictions']
        pred_steps = np.arange(len(gp2_preds))
        pred_std = np.sqrt(np.array([p['energy_var'] for p in gp2_preds]))
        
        ax_gp2_uncertainty.plot(pred_steps, pred_std, 'g-s', markersize=6)
        ax_gp2_uncertainty.axhline(y=results.get('gp2_threshold', 0.01), 
                                  color='red', linestyle='--', label='Threshold')
        ax_gp2_uncertainty.set_xlabel('GP2 Evaluation', fontsize=16)
        ax_gp2_uncertainty.set_ylabel('Uncertainty (eV)', fontsize=16)
        ax_gp2_uncertainty.set_title('GP2 Uncertainty Evolution', fontsize=18)
        ax_gp2_uncertainty.legend(fontsize=14)
        ax_gp2_uncertainty.grid(True, alpha=0.3)
        ax_gp2_uncertainty.tick_params(axis='both', which='major', labelsize=14)
    else:
        ax_gp2_uncertainty.text(0.5, 0.5, 'No GP2 predictions available', 
                               transform=ax_gp2_uncertainty.transAxes, ha='center', va='center', fontsize=16)
        ax_gp2_uncertainty.set_title('GP2 Uncertainty Evolution', fontsize=18)
    
    # Bottom information panel
    ax_info.axis('off')
    
    # Create summary table
    info_data = []
    info_data.append(['Parameter', 'Initial', 'Final', 'GP2 Info'])
    info_data.append(['Position X (Å)', f'{positions[0, 0]:.6f}', f'{positions[-1, 0]:.6f}', 
                     f'Training Points: {results["gp2_info"]["training_points"]}'])
    info_data.append(['Position Y (Å)', f'{positions[0, 1]:.6f}', f'{positions[-1, 1]:.6f}', 
                     f'GP2 Evals: {results["total_gp2_evaluations"]}'])
    info_data.append(['Energy (eV)', f'{energies[0]:.6f}', f'{energies[-1]:.6f}', 
                     f'True Evals: {results["total_evaluations"]}'])
    info_data.append(['|Force| (eV/Å)', f'{force_mags[0]:.6f}', f'{force_mags[-1]:.6e}', 
                     f'GP2 Rate: {results["total_gp2_evaluations"]/(results["total_evaluations"]+results["total_gp2_evaluations"])*100:.1f}%'])
    
    if 'final_curvature' in results and not np.isnan(results['final_curvature']):
        info_data.append(['Curvature (eV/Å²)', 'N/A', f'{results["final_curvature"]:.3f}', 
                         'Saddle' if results['final_curvature'] < 0 else 'Not Saddle'])
    
    # Create table
    table = ax_info.table(cellText=info_data[1:], colLabels=info_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.12, 0.12, 0.2])
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
                cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    
    # Add additional info
    add_info = f"Total Steps: {results['steps']} | Converged: {'Yes' if results['converged'] else 'No'} | "
    add_info += f"Runtime: {results.get('runtime', 0):.2f}s | "
    add_info += f"Efficiency Gain: {(results['total_evaluations']+results['total_gp2_evaluations'])/results['total_evaluations']:.1f}x"
    ax_info.text(0.5, 0.1, add_info, transform=ax_info.transAxes, 
                ha='center', fontsize=16, weight='bold')
    
    plt.suptitle(f'GP2 Dimer Method Results - {title_name}', fontsize=24, y=0.98)
    try:
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    except:
        plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.02)
    
    # Save plot
    plot_file = get_output_path('plots', f'gp2_dimer_trajectory_{potential_name}.png')
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    print(f"\nTrajectory plot saved to: {plot_file}")
    plt.close()


def gp2_dimer_toy_search(
    potential_name: str,
    initial_position: np.ndarray,
    system_params: dict
) -> tuple:
    """Run GP2 dimer search on toy model potential."""
    
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info(f"Starting GP2 dimer search on {potential_name} potential")
    
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
            'step_size': system_params.get('step_size', 0.02),
            'max_step_size': system_params.get('max_step_size', 0.05),
            'verbose': system_params.get('verbose', True),
            'checkpoint_interval': system_params.get('checkpoint_interval', 10),
            # GP2-specific parameters
            'gp2_update_interval': system_params.get('gp2_update_interval', 5),
            'gp2_threshold': system_params.get('gp2_threshold', 0.01),
            'gp2_min_train_points': system_params.get('gp2_min_train_points', 10),
            'gp2_max_train_points': system_params.get('gp2_max_train_points', 100),
            'gp2_noise': system_params.get('gp2_noise', 1e-4),
            'gp2_variance': system_params.get('gp2_variance', 1.0),
            'disp_max': system_params.get('disp_max', 0.2),
            'gp2_length_scales': system_params.get('gp2_length_scales', None),
            'use_gp2_forces': system_params.get('use_gp2_forces', True),
            'model_type': system_params.get('model_type', 'MultitaskGPModel_rbf_atomic')
        }
        
        # Create walker
        walker = WalkerGP2DimerToy(
            initial_position=initial_position,
            local_pes=local_pes,
            **dimer_params
        )
        
        # Run dimer optimization
        results = walker.run_dimer()
        
        # Add convergence threshold and GP2 threshold to results for plotting
        results['convergence_threshold'] = dimer_params['dimer_stopping_criteria']
        results['gp2_threshold'] = dimer_params['gp2_threshold']
        
        # Save results
        results_file = get_output_path('results', f'gp2_dimer_toy_{potential_name}_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {results_file}")
        
        # Save GP2 model if trained (skip for now due to save_model signature issues)
        # if walker.gp2_initialized:
        #     gp2_file = get_output_path('models', f'gp2_model_{potential_name}.pkl')
        #     walker.gp2.save_model(gp2_file)
        #     print(f"GP2 model saved to: {gp2_file}")
        
        # Generate plots
        plot_gp2_trajectory(results, potential_name)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Potential: {potential_name}")
        print(f"Initial position: [{initial_position[0]:.4f}, {initial_position[1]:.4f}]")
        print(f"Converged: {results['converged']}")
        print(f"Steps taken: {results['steps']}")
        print(f"Total true evaluations: {results['total_evaluations']}")
        print(f"Total GP2 evaluations: {results['total_gp2_evaluations']}")
        print(f"GP2 usage rate: {results['total_gp2_evaluations']/(results['total_evaluations']+results['total_gp2_evaluations'])*100:.1f}%")
        print(f"Efficiency gain: {(results['total_evaluations']+results['total_gp2_evaluations'])/results['total_evaluations']:.1f}x")
        print(f"Final position: [{results['final_position'][0]:.4f}, {results['final_position'][1]:.4f}]")
        print(f"Final energy: {results['final_energy']:.6f}")
        print(f"Final force magnitude: {results['final_force_magnitude']:.6e}")
        if 'final_curvature' in results:
            print(f"Final curvature: {results['final_curvature']:.6f}")
            curv = results['final_curvature']
            force_mag = results['final_force_magnitude']
            
            # More nuanced saddle point detection
            if curv < -0.1:
                print("SUCCESS: Found saddle point (negative curvature)!")
            elif -0.1 <= curv <= 0.1 and force_mag < 0.1:
                print("SUCCESS: Very close to saddle point (near-zero curvature, small forces)")
            elif curv > 10.0 and force_mag < 0.1:
                print("WARNING: Likely at a local minimum (large positive curvature)")
            elif curv < -10.0 and force_mag < 0.1:
                print("WARNING: Likely at a local maximum (large negative curvature)")
            else:
                print("INFO: Final point has positive curvature - may need more iterations")
        print("="*60)
        
        return walker, results
        
    except Exception as e:
        logging.error(f"GP2 dimer search failed: {str(e)}")
        raise
    finally:
        # Restore stdout
        sys.stdout = tee.terminal


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run GP2-accelerated dimer method on toy model potentials'
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
    
    # GP2 parameters
    parser.add_argument(
        '--gp2-update-interval',
        type=int,
        default=5,
        help='Steps between GP2 model updates'
    )
    parser.add_argument(
        '--gp2-threshold',
        type=float,
        default=0.01,
        help='GP2 uncertainty threshold for using predictions'
    )
    parser.add_argument(
        '--gp2-min-points',
        type=int,
        default=10,
        help='Minimum training points for GP2'
    )
    parser.add_argument(
        '--gp2-max-points',
        type=int,
        default=100,
        help='Maximum training points for GP2'
    )
    parser.add_argument(
        '--gp2-noise',
        type=float,
        default=1e-4,
        help='GP2 noise parameter'
    )
    parser.add_argument(
        '--no-gp2-forces',
        action='store_true',
        help='Disable GP2 force predictions'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='MultitaskGPModel_rbf_atomic',
        choices=['MultitaskGPModel_rbf_atomic', 'MultitaskGPModel'],
        help='GP model type (MultitaskGPModel uses standard RBF kernel with better length scale control)'
    )
    
    # GPU options
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration for GP training and inference'
    )
    parser.add_argument(
        '--no-gpu-fallback',
        action='store_true',
        help='Fail if GPU is requested but unavailable'
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
        run_name=args.run_name or f"gp2_toy_{args.potential}"
    )
    
    setup_logging()
    
    # Save run metadata
    OutputManager.save_run_metadata({
        'script': 'run_gp2_dimer_toy.py',
        'potential': args.potential,
        'initial_position': [args.initial_x, args.initial_y],
        'parameters': vars(args)
    })
    
    # Check GPU availability if requested
    if args.gpu:
        import torch
        
        # Print diagnostic information
        if args.verbose:
            print(f"\nGPU Diagnostics:")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if hasattr(torch.version, 'cuda'):
                print(f"  CUDA version (PyTorch built with): {torch.version.cuda}")
            if torch.cuda.is_available():
                print(f"  CUDA device count: {torch.cuda.device_count()}")
                print(f"  Current CUDA device: {torch.cuda.current_device()}")
                print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
            else:
                print("  Note: PyTorch may need to be reinstalled with CUDA support")
                print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        if not torch.cuda.is_available():
            if args.no_gpu_fallback:
                error_msg = ("GPU requested but CUDA is not available.\n"
                           "This could be because:\n"
                           "1. PyTorch is installed without CUDA support\n"
                           "2. NVIDIA drivers are not properly installed\n"
                           "3. CUDA toolkit version mismatch\n"
                           "To reinstall PyTorch with CUDA support, run:\n"
                           "  pip install torch --index-url https://download.pytorch.org/whl/cu118")
                raise RuntimeError(error_msg)
            else:
                logging.warning("GPU requested but CUDA not available. Falling back to CPU.")
                if args.verbose:
                    print("  Falling back to CPU computation")
    
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
        'checkpoint_interval': 10,
        # GP2 parameters
        'gp2_update_interval': args.gp2_update_interval,
        'gp2_threshold': args.gp2_threshold,
        'gp2_min_train_points': args.gp2_min_points,
        'gp2_max_train_points': args.gp2_max_points,
        'gp2_noise': args.gp2_noise,
        'use_gp2_forces': not args.no_gp2_forces,
        'model_type': args.model_type,
        # GPU options
        'use_gpu': args.gpu
    }
    
    # Initial position
    initial_position = np.array([args.initial_x, args.initial_y])
    
    # Run GP2 dimer search
    walker, results = gp2_dimer_toy_search(
        potential_name=args.potential,
        initial_position=initial_position,
        system_params=system_params
    )
    
    return 0 if results['converged'] else 1


if __name__ == "__main__":
    sys.exit(main())