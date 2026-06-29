#!/usr/bin/env python
"""Run dual GP method on toy model potentials (GP1 thermal sampling + GP2 acceleration)."""

import argparse
import os
import sys
import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
from walker_dual_gp_toy import WalkerDualGPToy
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
    
    log_file = get_output_path('logs', 'dual_gp_toy_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")


def plot_dual_gp_trajectory(results: dict, potential_name: str):
    """Plot the dual GP dimer trajectory with comprehensive visualizations."""
    # Import 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with comprehensive layout
    fig = plt.figure(figsize=(24, 18))
    
    # Create subplot grid
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 0.8], 
                          width_ratios=[1.4, 1.1, 1.1, 1.1], 
                          hspace=0.35, wspace=0.35)
    
    # Main plots
    ax1 = fig.add_subplot(gs[0:2, 0])  # Main contour plot
    ax3d = fig.add_subplot(gs[0, 1], projection='3d')  # 3D surface plot
    ax_thermal = fig.add_subplot(gs[0, 2])  # Thermal sampling visualization
    ax_dual_pred = fig.add_subplot(gs[0, 3])  # GP1 vs GP2 predictions
    
    # History plots
    ax2 = fig.add_subplot(gs[1, 1])  # Energy/Force history
    ax3 = fig.add_subplot(gs[1, 2])  # Curvature history
    ax_usage = fig.add_subplot(gs[1, 3])  # GP usage
    
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
    
    # Create grid for contour plot
    x = np.linspace(domain[0][0], domain[0][1], 200)
    y = np.linspace(domain[1][0], domain[1][1], 200)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate potential on grid
    from toy_model_interface import ToyModelInterface
    pes_interface = ToyModelInterface(potential_name, np.array([0, 0]))
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = pes_interface.scaler_y_value(np.array([X[i, j], Y[i, j]]))
    
    # Plot 1: Enhanced contour with trajectory
    contourf = ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    contour = ax1.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot full trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'w-', linewidth=3, alpha=0.8, 
             label='Trajectory', zorder=5)
    ax1.plot(positions[:, 0], positions[:, 1], 'k--', linewidth=1.5, alpha=0.5, zorder=5)
    
    # Mark key points
    ax1.scatter(positions[0, 0], positions[0, 1], s=200, c='lime', 
                edgecolor='black', linewidth=2, zorder=10, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], s=200, c='red', 
                edgecolor='black', linewidth=2, zorder=10, label='End')
    
    # Mark all evaluation points
    for i, (pos, e, f) in enumerate(results['trajectory'][1:], 1):
        ax1.scatter(pos[0], pos[1], s=50, c='white', edgecolor='black', 
                    alpha=0.7, zorder=6)
        ax1.annotate(f'{i}', (pos[0], pos[1]), xytext=(3, 3), 
                     textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_title(f'Dual GP Dimer Trajectory - {potential_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax1, label='Energy')
    
    # Plot 2: 3D surface
    ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, 
                      linewidth=0, antialiased=True)
    
    # Add trajectory on 3D plot
    ax3d.plot(positions[:, 0], positions[:, 1], energies, 'r-', 
              linewidth=3, label='Trajectory')
    ax3d.scatter(positions[0, 0], positions[0, 1], energies[0], 
                 s=100, c='lime', edgecolor='black')
    ax3d.scatter(positions[-1, 0], positions[-1, 1], energies[-1], 
                 s=100, c='red', edgecolor='black')
    
    ax3d.set_xlabel('X (Å)')
    ax3d.set_ylabel('Y (Å)')
    ax3d.set_zlabel('Energy')
    ax3d.set_title('3D View - Energy Surface')
    ax3d.view_init(elev=30, azim=225)
    
    # Plot 3: Thermal sampling visualization
    ax_thermal.set_title('Thermal Sampling Effect')
    ax_thermal.set_xlabel('Step')
    ax_thermal.set_ylabel('Average Force Magnitude (eV/Å)')
    
    # Plot force magnitudes
    steps = np.arange(len(force_mags))
    ax_thermal.plot(steps, force_mags, 'b-', linewidth=2, label='GP1 Thermal Average')
    
    # Add temperature info
    temp_info = results.get('gp1_info', {})
    temp = temp_info.get('temperature', 300)
    n_snap = temp_info.get('num_snapshots', 10)
    ax_thermal.text(0.02, 0.98, f'T = {temp} K\n{n_snap} snapshots', 
                    transform=ax_thermal.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax_thermal.grid(True, alpha=0.3)
    ax_thermal.legend()
    
    # Plot 4: GP1 vs GP2 predictions
    ax_dual_pred.set_title('GP Model Usage')
    ax_dual_pred.set_xlabel('Step')
    ax_dual_pred.set_ylabel('Evaluations per Step')
    
    # Extract evaluation counts
    gp1_evals = results.get('gp1_evals_per_step', [])
    gp2_evals = results.get('gp2_evals_per_step', [])
    
    if gp1_evals and gp2_evals:
        steps = np.arange(1, len(gp1_evals) + 1)
        width = 0.35
        
        ax_dual_pred.bar(steps - width/2, gp1_evals, width, label='GP1', color='blue', alpha=0.7)
        ax_dual_pred.bar(steps + width/2, gp2_evals, width, label='GP2', color='orange', alpha=0.7)
        
        ax_dual_pred.legend()
        ax_dual_pred.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Energy/Force history
    ax2_twin = ax2.twinx()
    
    # Create steps array for plotting
    plot_steps = np.arange(len(energies))
    
    line1 = ax2.plot(plot_steps, energies, 'b-', linewidth=2, label='Energy')
    line2 = ax2_twin.plot(plot_steps, force_mags, 'r-', linewidth=2, label='Force')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Energy', color='b')
    ax2_twin.set_ylabel('Force Magnitude (eV/Å)', color='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Convergence History')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')
    
    # Add convergence line
    ax2_twin.axhline(y=results['gp2_info']['convergence_threshold'], 
                     color='green', linestyle='--', alpha=0.5, 
                     label='Convergence Threshold')
    
    # Plot 6: Curvature evolution
    curvatures = [row['Curvature'] for row in results['table_history']]
    curv_steps = np.arange(len(curvatures))
    ax3.plot(curv_steps, curvatures, 'g-', linewidth=2, marker='o', markersize=6)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Curvature')
    ax3.set_title('Curvature Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Color background based on curvature sign
    for i in range(len(curvatures)):
        if curvatures[i] > 0:
            ax3.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
        else:
            ax3.axvspan(i-0.5, i+0.5, alpha=0.2, color='blue')
    
    # Plot 7: GP usage statistics
    total_thermal = results.get('thermal_evaluations', 0)
    total_gp1 = results.get('total_gp1_evaluations', 0)
    total_gp2 = results.get('total_gp2_evaluations', 0)
    total_direct = results.get('total_evaluations', 0)
    
    categories = ['Direct\nEvals', 'Thermal\nSamples', 'GP1\nCalls', 'GP2\nCalls']
    values = [total_direct, total_thermal, total_gp1, total_gp2]
    colors = ['gray', 'lightblue', 'blue', 'orange']
    
    bars = ax_usage.bar(categories, values, color=colors, alpha=0.7)
    ax_usage.set_ylabel('Count')
    ax_usage.set_title('Total Evaluation Counts')
    ax_usage.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax_usage.text(bar.get_x() + bar.get_width()/2., height,
                      f'{int(val)}', ha='center', va='bottom')
    
    # Information panel
    ax_info.axis('off')
    info_text = f"""
    Dual GP Dimer Results - {potential_name}
    
    Initial position: ({results['trajectory'][0][0][0]:.4f}, {results['trajectory'][0][0][1]:.4f})
    Final position: ({results['final_position'][0]:.4f}, {results['final_position'][1]:.4f}) | 
    Final energy: {results['final_energy']:.6f} | Final force: {results['final_force_magnitude']:.6f} eV/Å
    
    GP1 Info: Temperature: {temp} K | Snapshots: {n_snap} | Thermal noise: F={results['gp1_info']['thermal_noise'][0]:.4f} eV/Å, E={results['gp1_info']['thermal_noise'][1]:.4f} eV
    GP2 Info: Training points: {results['gp2_info']['training_points']} | Final curvature: {results['gp2_info']['final_curvature']:.4f}
    
    Performance: Steps: {results['steps']} | Runtime: {results['runtime']:.1f}s | 
    Efficiency gain: {(total_thermal + total_gp1 + total_gp2) / max(total_direct, 1):.1f}x
    """
    
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=11, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Add title with convergence status
    convergence_status = "CONVERGED" if results['converged'] else "NOT CONVERGED"
    fig.suptitle(f'Dual GP Dimer Method Results - {potential_name} - {convergence_status}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    return fig


def dual_gp_toy_search(
    potential_name: str,
    initial_position: np.ndarray,
    system_params: dict,
    equilibrium_position: np.ndarray = None
) -> tuple:
    """Run dual GP search on toy model potential."""
    
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info(f"Starting dual GP search on {potential_name} potential")
    
    try:
        # Create toy model interface
        local_pes = ToyModelInterface(potential_name, initial_position)
        
        # Extract parameters
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
            # Thermal sampling parameters
            'temperature': system_params.get('temperature', 300.0),
            'mass': system_params.get('mass', 1.0),
            'num_snapshots': system_params.get('num_snapshots', 10),
            # GP parameters
            'divisor_T_dimer_gp': system_params.get('divisor_T_dimer_gp', 10.0),
            'max_inner_iterations': system_params.get('max_inner_iterations', 50),
            'disp_max': system_params.get('disp_max', 0.2),
            'model_type': system_params.get('model_type', 'MultitaskGPModel_rbf_atomic')
        }
        
        # Create walker
        walker = WalkerDualGPToy(
            initial_position=initial_position,
            local_pes=local_pes,
            equilibrium_position=equilibrium_position,
            **dimer_params
        )
        
        # Run dimer search
        results = walker.run_dimer()
        
        # Plot results
        if system_params.get('plot_results', True):
            try:
                fig = plot_dual_gp_trajectory(results, potential_name)
                plot_file = get_output_path('plots', f'dual_gp_trajectory_{potential_name}.png')
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"\nTrajectory plot saved to: {plot_file}")
            except Exception as e:
                logging.error(f"Failed to create plot: {e}")
        
        # Save results
        results_file = get_output_path('results', f'dual_gp_toy_{potential_name}_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Potential: {potential_name}")
        print(f"Temperature: {dimer_params['temperature']} K")
        print(f"Thermal snapshots: {dimer_params['num_snapshots']}")
        print(f"Initial position: [{initial_position[0]:.4f}, {initial_position[1]:.4f}]")
        print(f"Converged: {results['converged']}")
        print(f"Steps taken: {results['steps']}")
        print(f"Total direct evaluations: {results['total_evaluations']}")
        print(f"Total thermal samples: {results['thermal_evaluations']}")
        print(f"Total GP1 evaluations: {results['total_gp1_evaluations']}")
        print(f"Total GP2 evaluations: {results['total_gp2_evaluations']}")
        
        if results['converged']:
            efficiency = (results['thermal_evaluations'] + results['total_gp1_evaluations'] + 
                         results['total_gp2_evaluations']) / max(results['total_evaluations'], 1)
            print(f"Efficiency gain: {efficiency:.1f}x")
            print(f"Final position: [{results['final_position'][0]:.4f}, {results['final_position'][1]:.4f}]")
            print(f"Final energy: {results['final_energy']:.6f}")
            print(f"Final force magnitude: {results['final_force_magnitude']:.6e}")
            print(f"Final curvature: {results['gp2_info']['final_curvature']:.6f}")
            
            # Check if at saddle point
            curv = results['gp2_info']['final_curvature']
            force_mag = results['final_force_magnitude']
            
            if curv < -0.1:
                print("SUCCESS: Found saddle point (negative curvature)!")
            elif -0.1 <= curv <= 0.1 and force_mag < 0.1:
                print("SUCCESS: Very close to saddle point (near-zero curvature, small forces)")
            elif curv > 0.1:
                if force_mag < 0.1:
                    print("WARNING: Converged to local extremum (positive curvature)")
                else:
                    print("WARNING: Not at saddle point (positive curvature)")
        else:
            print("Search did not converge within maximum steps")
        
        print("="*60)
        
        return walker, results
        
    except Exception as e:
        logging.error(f"Dual GP search failed: {e}")
        raise
    finally:
        # Restore stdout
        sys.stdout = sys.__stdout__


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run dual GP dimer method on toy model potentials'
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
    
    # Equilibrium position for thermal noise calculation
    parser.add_argument(
        '--equilibrium-x',
        type=float,
        default=None,
        help='Equilibrium X coordinate for thermal noise calculation (default: use initial position)'
    )
    parser.add_argument(
        '--equilibrium-y',
        type=float,
        default=None,
        help='Equilibrium Y coordinate for thermal noise calculation (default: use initial position)'
    )
    
    # Thermal sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=300.0,
        help='Temperature for thermal sampling (K)'
    )
    parser.add_argument(
        '--mass',
        type=float,
        default=1.0,
        help='Particle mass'
    )
    parser.add_argument(
        '--num-snapshots',
        type=int,
        default=10,
        help='Number of thermal snapshots for GP1'
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
        help='Force convergence threshold (eV/Å)'
    )
    parser.add_argument(
        '--step-size',
        type=float,
        default=0.02,
        help='Base step size'
    )
    parser.add_argument(
        '--max-step-size',
        type=float,
        default=0.05,
        help='Maximum step size'
    )
    
    # GP parameters
    parser.add_argument(
        '--max-inner-iterations',
        type=int,
        default=50,
        help='Maximum inner iterations on GP surface'
    )
    parser.add_argument(
        '--divisor-T-dimer-gp',
        type=float,
        default=10.0,
        help='Divisor for dynamic GP convergence threshold'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='MultitaskGPModel_rbf_atomic',
        choices=['MultitaskGPModel_rbf_atomic', 'MultitaskGPModel'],
        help='GP model type'
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
        run_name=args.run_name or f"dual_gp_toy_{args.potential}"
    )
    
    setup_logging()
    
    # Save run metadata
    OutputManager.save_run_metadata({
        'script': 'run_dual_gp_toy.py',
        'potential': args.potential,
        'initial_position': [args.initial_x, args.initial_y],
        'temperature': args.temperature,
        'num_snapshots': args.num_snapshots,
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
        'temperature': args.temperature,
        'mass': args.mass,
        'num_snapshots': args.num_snapshots,
        'max_inner_iterations': args.max_inner_iterations,
        'divisor_T_dimer_gp': args.divisor_T_dimer_gp,
        'model_type': args.model_type,
        'use_gpu': args.gpu,
        'plot_results': True
    }
    
    # Initial position
    initial_position = np.array([args.initial_x, args.initial_y])
    
    # Equilibrium position (use initial if not specified)
    if args.equilibrium_x is not None and args.equilibrium_y is not None:
        equilibrium_position = np.array([args.equilibrium_x, args.equilibrium_y])
    else:
        equilibrium_position = None
    
    # Run dual GP search
    walker, results = dual_gp_toy_search(
        potential_name=args.potential,
        initial_position=initial_position,
        system_params=system_params,
        equilibrium_position=equilibrium_position
    )
    
    return 0 if results['converged'] else 1


if __name__ == "__main__":
    sys.exit(main())