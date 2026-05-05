#!/usr/bin/env python
"""Plot checkpoint data from walker runs - supports both old and new formats."""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
from output_manager import get_output_path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint from file."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def detect_walker_type(checkpoint_data: Dict) -> str:
    """Detect the walker type from checkpoint data."""
    if 'walker_type' in checkpoint_data:
        return checkpoint_data['walker_type']
    
    # Check for GP1 path analysis specific data
    if 'walker' in checkpoint_data and hasattr(checkpoint_data.get('walker'), 'path_positions'):
        return 'WalkerGP1Path'
    if 'path_positions' in checkpoint_data and 'per_image_stats' in checkpoint_data:
        return 'WalkerGP1Path'
    if 'training_data' in checkpoint_data and 'thermal_noises' in checkpoint_data and 'statistics' in checkpoint_data:
        # Additional check for GP1 path structure
        if 'completed_images' in checkpoint_data:
            return 'WalkerGP1Path'
    
    # Check for thermal noise - key indicator of dual GP
    if 'thermal_noise' in checkpoint_data and checkpoint_data['thermal_noise'] is not None:
        if 'bigiter' in checkpoint_data or 'outer_iteration' in checkpoint_data:
            return 'WalkerDualGP'
    
    # Try to infer from data structure
    if 'bigiter' in checkpoint_data or 'outer_iteration' in checkpoint_data:
        return 'WalkerGP2Dimer'
    elif 'gp1' in checkpoint_data:
        return 'Walker'
    elif 'steps' in checkpoint_data:
        return 'WalkerPureDimer'
    return 'Unknown'


def get_iteration_count(checkpoint_data: Dict) -> int:
    """Get iteration count handling both old and new formats."""
    if 'iteration' in checkpoint_data:
        return checkpoint_data['iteration']
    elif 'bigiter' in checkpoint_data:
        return checkpoint_data['bigiter']
    elif 'outer_iteration' in checkpoint_data:
        return checkpoint_data['outer_iteration']
    elif 'steps' in checkpoint_data:
        return checkpoint_data['steps']
    return 0


def print_summary(checkpoint_data: Dict):
    """Print summary of checkpoint data."""
    print("\n" + "="*60)
    print(" "*20 + "CHECKPOINT SUMMARY" + " "*20)
    print("="*60)
    
    walker_type = detect_walker_type(checkpoint_data)
    print(f"Walker Type: {walker_type}")
    
    iteration = get_iteration_count(checkpoint_data)
    
    if walker_type in ['WalkerGP2Dimer', 'WalkerDualGP']:
        # GP-based dimer walkers
        print(f"Outer Iterations: {checkpoint_data.get('bigiter', checkpoint_data.get('outer_iteration', 'N/A'))}")
        print(f"Inner Iterations: {len(checkpoint_data.get('E_R_gp', []))}")
        print(f"Total Observations: {checkpoint_data.get('obs_total', 'N/A')}")
        print(f"Initial Rotation Observations: {checkpoint_data.get('obs_initrot', 'N/A')}")
        print(f"Converged: {checkpoint_data.get('converged', False)}")
        
        # Additional info for dual GP
        if walker_type == 'WalkerDualGP':
            if 'temperature' in checkpoint_data:
                print(f"Temperature: {checkpoint_data['temperature']} K")
            if 'thermal_noise' in checkpoint_data and checkpoint_data['thermal_noise']:
                print(f"Thermal noise: σ_F = {checkpoint_data['thermal_noise'][0]:.3e}, σ_E = {checkpoint_data['thermal_noise'][1]:.3e}")
        
        if 'maxF_R_acc' in checkpoint_data and len(checkpoint_data['maxF_R_acc']) > 0:
            print(f"Best Max Force: {np.min(checkpoint_data['maxF_R_acc']):.6f} eV/Å")
            print(f"Latest Max Force: {checkpoint_data['maxF_R_acc'][-1]:.6f} eV/Å")
        
        print(f"\nStopping Statistics:")
        print(f"  Max iterations reached: {checkpoint_data.get('num_esmax', 0)}")
        print(f"  Inter-atomic distance limit: {checkpoint_data.get('num_es1', 0)}")
        print(f"  Displacement limit: {checkpoint_data.get('num_es2', 0)}")
        
        # Print table history if available
        if 'table_history' in checkpoint_data and len(checkpoint_data['table_history']) > 0:
            print(f"\nProgress Table Entries: {len(checkpoint_data['table_history'])}")
            
    elif walker_type == 'Walker':
        # Original TD-SPF walker
        print(f"Iteration: {iteration}")
        print(f"Converged: {checkpoint_data.get('converged', False)}")
        print(f"Temperature: {checkpoint_data.get('temperature', 'N/A')} K")
        
    elif walker_type == 'WalkerPureDimer':
        # Pure dimer walker
        print(f"Steps: {iteration}")
        print(f"Converged: {checkpoint_data.get('converged', False)}")
        print(f"VASP Evaluations: {checkpoint_data.get('vasp_eval_count', 0)}")
        
    elif walker_type == 'WalkerGP1Path':
        # GP1 path analysis walker
        # Extract walker object if available
        walker = checkpoint_data.get('walker', None)
        if walker:
            n_images = walker.n_images if hasattr(walker, 'n_images') else checkpoint_data.get('n_images', 'N/A')
            temperature = walker.temperature if hasattr(walker, 'temperature') else checkpoint_data.get('temperature', 'N/A')
        else:
            n_images = checkpoint_data.get('n_images', 'N/A')
            temperature = checkpoint_data.get('temperature', 'N/A')
            
        print(f"Images analyzed: {checkpoint_data.get('completed_images', 0)}")
        print(f"Total images: {n_images}")
        print(f"Temperature: {temperature} K")
        
        if 'thermal_noises' in checkpoint_data and checkpoint_data['thermal_noises']:
            avg_force_noise = np.mean([n[0] for n in checkpoint_data['thermal_noises']])
            avg_energy_noise = np.mean([n[1] for n in checkpoint_data['thermal_noises']])
            print(f"Avg thermal noise: σ_F = {avg_force_noise:.3e} eV/Å, σ_E = {avg_energy_noise:.3e} eV")
        
        if 'statistics' in checkpoint_data and checkpoint_data['statistics']:
            print(f"\nLatest statistics:")
            latest = checkpoint_data['statistics'][-1]
            energy_mae = latest.get('energy_mae', 'N/A')
            force_mae = latest.get('force_mae', 'N/A')
            if isinstance(energy_mae, (int, float)):
                print(f"  Energy MAE: {energy_mae:.6f} eV")
            else:
                print(f"  Energy MAE: {energy_mae}")
            if isinstance(force_mae, (int, float)):
                print(f"  Force MAE: {force_mae:.6f} eV/Å")
            else:
                print(f"  Force MAE: {force_mae}")
        
    print("="*60)


def plot_gp2_dimer_analysis(checkpoint_data: Dict, output_dir: str = '.'):
    """Create comprehensive analysis plots for GP2 Dimer walker."""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Outer iteration convergence with table data
    ax1 = fig.add_subplot(gs[0, :])
    plot_outer_convergence(ax1, checkpoint_data)
    
    # 2. Inner iteration history (GP values)
    ax2 = fig.add_subplot(gs[1, :])
    plot_inner_iterations(ax2, checkpoint_data)
    
    # 3. GP vs Actual comparison
    ax3 = fig.add_subplot(gs[2, 0])
    plot_gp_vs_actual(ax3, checkpoint_data)
    
    # 4. Efficiency analysis
    ax4 = fig.add_subplot(gs[2, 1])
    plot_efficiency(ax4, checkpoint_data)
    
    # 5. Observation distribution
    ax5 = fig.add_subplot(gs[2, 2])
    plot_observation_distribution(ax5, checkpoint_data)
    
    # 6. Hyperparameter evolution
    ax6 = fig.add_subplot(gs[3, 0])
    plot_hyperparameter_evolution(ax6, checkpoint_data)
    
    # 7. Curvature analysis
    ax7 = fig.add_subplot(gs[3, 1])
    plot_curvature_analysis(ax7, checkpoint_data)
    
    # 8. System info
    ax8 = fig.add_subplot(gs[3, 2])
    plot_system_info(ax8, checkpoint_data)
    
    plt.suptitle('GP2 Dimer Walker Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, f'gp2_dimer_analysis_iter{get_iteration_count(checkpoint_data)}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nGP2 Dimer analysis saved to: {output_file}")
    plt.close()


def plot_outer_convergence(ax, checkpoint_data):
    """Plot outer iteration convergence with both GP and actual values."""
    table_history = checkpoint_data.get('table_history', [])
    
    if table_history:
        # Extract data
        steps = []
        e_actual = []
        e_gp = []
        f_actual = []
        f_gp = []
        
        for row in table_history:
            step = row.get('step', len(steps))
            
            # Get values
            ea = row.get('E_Actual', np.nan)
            eg = row.get('E_GP', np.nan)
            fa = row.get('F_Actual', np.nan)
            fg = row.get('F_GP', np.nan)
            
            # Add all data points (we'll handle NaN in plotting)
            steps.append(step)
            e_actual.append(ea)
            e_gp.append(eg)
            f_actual.append(fa)
            f_gp.append(fg)
        
        if steps:
            # Plot energies
            color = 'tab:blue'
            ax.set_xlabel('Outer Iteration')
            ax.set_ylabel('Energy (eV)', color=color)
            
            # Plot actual energy
            mask = ~np.isnan(e_actual)
            if np.any(mask):
                ax.plot(np.array(steps)[mask], np.array(e_actual)[mask], 
                       'o-', color=color, linewidth=2, markersize=8, label='E Actual')
            
            # Plot GP energy
            mask = ~np.isnan(e_gp)
            if np.any(mask):
                ax.plot(np.array(steps)[mask], np.array(e_gp)[mask], 
                       's--', color='lightblue', linewidth=1.5, markersize=6, 
                       alpha=0.7, label='E GP')
            
            ax.tick_params(axis='y', labelcolor=color)
            ax.legend(loc='upper left')
            
            # Plot forces on twin axis
            ax_twin = ax.twinx()
            color = 'tab:red'
            ax_twin.set_ylabel('Max |Force| (eV/Å)', color=color)
            
            # Plot actual force
            mask = ~np.isnan(f_actual)
            if np.any(mask):
                ax_twin.plot(np.array(steps)[mask], np.array(f_actual)[mask], 
                           '^-', color=color, linewidth=2, markersize=8, label='F Actual')
            
            # Plot GP force
            mask = ~np.isnan(f_gp)
            if np.any(mask):
                ax_twin.plot(np.array(steps)[mask], np.array(f_gp)[mask], 
                           'v--', color='lightcoral', linewidth=1.5, markersize=6, 
                           alpha=0.7, label='F GP')
            
            ax_twin.tick_params(axis='y', labelcolor=color)
            
            # Add convergence line
            if 'dimer_stopping_criteria' in checkpoint_data:
                ax_twin.axhline(y=checkpoint_data['dimer_stopping_criteria'], 
                               color='green', linestyle='--', alpha=0.5, 
                               label=f"Target: {checkpoint_data['dimer_stopping_criteria']} eV/Å")
            
            ax_twin.legend(loc='upper right')
            
            # Mark initial rotation phase
            if 'obs_initrot' in checkpoint_data and checkpoint_data['obs_initrot'] > 0:
                ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
                ax.text(0, ax.get_ylim()[1]*0.95, 'Init Rot', ha='center', va='top', fontsize=8)
    else:
        # Fall back to arrays
        plot_from_arrays(ax, checkpoint_data)
    
    ax.set_title('GP Predictions and Accurate Values at Each Outer Iteration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_from_arrays(ax, checkpoint_data):
    """Plot from array data when table history is not available."""
    E_R_acc = checkpoint_data.get('E_R_acc', np.array([]))
    maxF_R_acc = checkpoint_data.get('maxF_R_acc', np.array([]))
    
    if len(E_R_acc) > 0:
        color = 'tab:blue'
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Energy (eV)', color=color)
        ax.plot(E_R_acc, 'o-', color=color, linewidth=2, markersize=8)
        ax.tick_params(axis='y', labelcolor=color)
        
        ax_twin = ax.twinx()
        color = 'tab:red'
        ax_twin.set_ylabel('Max |Force| (eV/Å)', color=color)
        ax_twin.plot(maxF_R_acc, 's-', color=color, linewidth=2, markersize=8)
        ax_twin.tick_params(axis='y', labelcolor=color)
        
        # Add convergence line
        if 'dimer_stopping_criteria' in checkpoint_data:
            ax_twin.axhline(y=checkpoint_data['dimer_stopping_criteria'], 
                           color='green', linestyle='--', alpha=0.5, 
                           label=f"Target: {checkpoint_data['dimer_stopping_criteria']}")


def plot_inner_iterations(ax, checkpoint_data):
    """Plot inner iteration history with phase markers."""
    E_R_gp = checkpoint_data.get('E_R_gp', np.array([]))
    maxF_R_gp = checkpoint_data.get('maxF_R_gp', np.array([]))
    obs_at = checkpoint_data.get('obs_at', np.array([]))
    
    if len(E_R_gp) > 0:
        # Plot energy
        color = 'tab:blue'
        ax.set_xlabel('Inner Iteration')
        ax.set_ylabel('GP Energy (eV)', color=color)
        ax.plot(E_R_gp, '-', color=color, linewidth=1, alpha=0.7)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Mark observation points
        if len(obs_at) > 0:
            for i, obs in enumerate(obs_at[1:], 1):  # Skip first which is 0
                ax.axvline(x=obs, color='gray', linestyle=':', alpha=0.5)
                ax.text(obs, ax.get_ylim()[1]*0.95, f'Obs {i}', 
                       ha='center', va='top', fontsize=8, rotation=45)
        
        # Plot force on twin axis
        ax_twin = ax.twinx()
        color = 'tab:red'
        ax_twin.set_ylabel('GP Max |Force| (eV/Å)', color=color)
        ax_twin.plot(maxF_R_gp, '-', color=color, linewidth=1, alpha=0.7)
        ax_twin.tick_params(axis='y', labelcolor=color)
        
        # Add dynamic threshold if available
        if 'divisor_T_dimer_gp' in checkpoint_data:
            divisor = checkpoint_data['divisor_T_dimer_gp']
            if 'maxF_R_acc' in checkpoint_data and len(checkpoint_data['maxF_R_acc']) > 0:
                min_force = min(checkpoint_data['maxF_R_acc'])
                threshold = max(min_force / divisor, 
                              checkpoint_data.get('dimer_stopping_criteria', 0.01) / 10.0)
                ax_twin.axhline(y=threshold, color='orange', linestyle='--', alpha=0.5,
                              label=f'GP Target: {threshold:.4f}')
                ax_twin.legend(loc='upper right')
        
        ax.set_title('GP Predictions During Relaxation Phases', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)


def plot_gp_vs_actual(ax, checkpoint_data):
    """Plot GP predictions vs actual values."""
    table_history = checkpoint_data.get('table_history', [])
    
    if table_history and len(table_history) > 1:
        # Extract GP and actual values - handle different key names
        gp_energies = []
        actual_energies = []
        gp_forces = []
        actual_forces = []
        
        for row in table_history[1:]:  # Skip first row
            # Energy values
            e_gp = row.get('E_GP', row.get('energy_gp', np.nan))
            e_actual = row.get('E_Actual', row.get('energy_actual', np.nan))
            if not np.isnan(e_gp) and not np.isnan(e_actual):
                gp_energies.append(e_gp)
                actual_energies.append(e_actual)
            
            # Force values
            f_gp = row.get('maxF_GP', row.get('max_force_gp', np.nan))
            f_actual = row.get('maxF_Actual', row.get('max_force_actual', np.nan))
            if not np.isnan(f_gp) and not np.isnan(f_actual):
                gp_forces.append(f_gp)
                actual_forces.append(f_actual)
        
        if gp_energies and actual_energies:
            # Energy correlation
            ax.scatter(actual_energies, gp_energies, alpha=0.6, s=50, label='Energy')
            
            # Perfect correlation line
            min_e = min(min(actual_energies), min(gp_energies))
            max_e = max(max(actual_energies), max(gp_energies))
            ax.plot([min_e, max_e], [min_e, max_e], 'k--', alpha=0.5, label='Perfect')
            
            # Calculate R²
            if len(actual_energies) > 2:
                r2_e = np.corrcoef(actual_energies, gp_energies)[0, 1]**2
                ax.text(0.05, 0.95, f'Energy R² = {r2_e:.3f}', 
                       transform=ax.transAxes, fontsize=10, va='top')
            
            ax.set_xlabel('Actual Energy (eV)')
            ax.set_ylabel('GP Energy (eV)')
            ax.set_title('GP vs Actual Predictions', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data\nfor correlation plot', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP vs Actual Predictions', fontsize=12, fontweight='bold')


def plot_curvature_analysis(ax, checkpoint_data):
    """Plot curvature analysis from table history."""
    table_history = checkpoint_data.get('table_history', [])
    
    if table_history and len(table_history) > 1:
        steps = []
        curv_gp = []
        curv_dimer = []
        
        for row in table_history:
            # Try different possible curvature keys
            c_dimer = row.get('Curvature_dimer', row.get('curvature_dimer', row.get('curvature', np.nan)))
            if not np.isnan(c_dimer):
                steps.append(row.get('step', len(steps)))
                curv_dimer.append(c_dimer)
                c_gp = row.get('Curvature_GP', row.get('curvature_gp', np.nan))
                curv_gp.append(c_gp)
        
        if steps:
            # Plot both curvatures
            ax.plot(steps, curv_dimer, 'bo-', label='Actual Curvature', markersize=8, linewidth=2)
            if not all(np.isnan(curv_gp)):
                ax.plot(steps, curv_gp, 'rs--', label='GP Curvature', markersize=6, linewidth=1.5, alpha=0.7)
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Outer Iteration')
            ax.set_ylabel('Curvature (eV/Å²)')
            ax.set_title('Dimer Curvature Evolution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text for final curvature
            if curv_dimer:
                final_curv = curv_dimer[-1]
                ax.text(0.95, 0.05, f'Final: {final_curv:.4f}', 
                       transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No curvature data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dimer Curvature Evolution', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No curvature data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Dimer Curvature Evolution', fontsize=12, fontweight='bold')


def plot_efficiency(ax, checkpoint_data):
    """Plot efficiency analysis."""
    obs_total = checkpoint_data.get('obs_total', 0)
    obs_initrot = checkpoint_data.get('obs_initrot', 0)
    inner_iters = len(checkpoint_data.get('E_R_gp', []))
    
    if obs_total > 0:
        # Breakdown of observations
        obs_main = obs_total - obs_initrot
        
        labels = []
        sizes = []
        colors = []
        
        if obs_initrot > 0:
            labels.extend(['Initial Rotations', 'Main Search'])
            sizes.extend([obs_initrot, obs_main])
            colors.extend(['#ff9999', '#66b3ff'])
        else:
            labels.append('Observations')
            sizes.append(obs_total)
            colors.append('#ff9999')
        
        if inner_iters > 0:
            labels.append('GP Evaluations')
            sizes.append(inner_iters)
            colors.append('#90ee90')
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct=lambda pct: f'{int(pct*sum(sizes)/100)}',
                                         startangle=90)
        
        # Calculate speedup
        if inner_iters > 0 and obs_main > 0:
            speedup = inner_iters / obs_main
            ax.set_title(f'Efficiency: {speedup:.1f}x speedup\n(Total: {obs_total} obs + {inner_iters} GP)', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Observation Distribution\n(Total: {obs_total})', 
                        fontsize=12, fontweight='bold')


def plot_observation_distribution(ax, checkpoint_data):
    """Plot distribution of relaxation phase lengths."""
    obs_at = checkpoint_data.get('obs_at', np.array([]))
    
    if len(obs_at) > 1:
        gaps = np.diff(obs_at)
        
        # Create histogram
        ax.hist(gaps, bins=min(20, len(gaps)), color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Inner Iterations Between Observations')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Relaxation Phase Lengths', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        textstr = f'Mean: {np.mean(gaps):.1f}\nStd: {np.std(gaps):.1f}\nMax: {np.max(gaps)}\nMin: {np.min(gaps)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    else:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution of Relaxation Phase Lengths', fontsize=12, fontweight='bold')


def plot_hyperparameter_evolution(ax, checkpoint_data):
    """Plot GP hyperparameter evolution."""
    param_gp = checkpoint_data.get('param_gp', [])
    
    if param_gp:
        iterations = list(range(len(param_gp)))
        
        # Extract magnitudes and mean lengthscales
        magnitudes = []
        mean_lengthscales = []
        
        for params in param_gp:
            if isinstance(params, dict):
                magnitudes.append(params.get('magnitude', np.nan))
                ls = params.get('lengthscales', [])
                if isinstance(ls, list) and ls:
                    mean_lengthscales.append(np.mean(ls))
                else:
                    mean_lengthscales.append(np.nan)
        
        # Plot if we have data
        if magnitudes and not all(np.isnan(magnitudes)):
            ax2 = ax.twinx()
            
            # Magnitude
            line1 = ax.plot(iterations, magnitudes, 'b-o', label='Magnitude', markersize=6)
            ax.set_xlabel('Outer Iteration')
            ax.set_ylabel('Magnitude', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Lengthscale
            if mean_lengthscales and not all(np.isnan(mean_lengthscales)):
                line2 = ax2.plot(iterations, mean_lengthscales, 'r-s', label='Mean Lengthscale', markersize=6)
                ax2.set_ylabel('Mean Lengthscale', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='best')
            
            ax.set_title('GP Hyperparameter Evolution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hyperparameter data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('GP Hyperparameter Evolution', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No hyperparameter data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP Hyperparameter Evolution', fontsize=12, fontweight='bold')


def plot_curvature_analysis(ax, checkpoint_data):
    """Plot curvature analysis from table history."""
    table_history = checkpoint_data.get('table_history', [])
    
    if table_history and len(table_history) > 1:
        steps = []
        curv_gp = []
        curv_dimer = []
        
        for row in table_history:
            if not np.isnan(row.get('Curvature_dimer', np.nan)):
                steps.append(row['step'])
                curv_gp.append(row.get('Curvature_GP', np.nan))
                curv_dimer.append(row['Curvature_dimer'])
        
        if steps:
            # Plot both curvatures
            ax.plot(steps, curv_dimer, 'bo-', label='Actual Curvature', markersize=8, linewidth=2)
            if not all(np.isnan(curv_gp)):
                ax.plot(steps, curv_gp, 'rs--', label='GP Curvature', markersize=6, linewidth=1.5, alpha=0.7)
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Outer Iteration')
            ax.set_ylabel('Curvature (eV/Å²)')
            ax.set_title('Dimer Curvature Evolution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text for final curvature
            if curv_dimer:
                final_curv = curv_dimer[-1]
                ax.text(0.95, 0.05, f'Final: {final_curv:.4f}', 
                       transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No curvature data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dimer Curvature Evolution', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No curvature data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Dimer Curvature Evolution', fontsize=12, fontweight='bold')


def plot_system_info(ax, checkpoint_data):
    """Plot system information."""
    ax.axis('off')
    
    # Get iteration count
    bigiter = checkpoint_data.get('bigiter', checkpoint_data.get('outer_iteration', 0))
    
    # Get atomic info
    atomic_info = checkpoint_data.get('atomic_info', {})
    n_moving = len(atomic_info.get('moving_indices', []))
    n_pt = atomic_info.get('n_pt', 0)
    
    info_text = f"""GP2 Dimer System Info
{'─' * 35}
Outer iterations: {bigiter}
Total observations: {checkpoint_data.get('obs_total', 0)}
Initial rotations: {checkpoint_data.get('obs_initrot', 0)}
Inner iterations: {len(checkpoint_data.get('E_R_gp', []))}
Converged: {checkpoint_data.get('converged', False)}

Stopping counts:
  Max iterations: {checkpoint_data.get('num_esmax', 0)}
  Inter-atomic dist: {checkpoint_data.get('num_es1', 0)}
  Displacement: {checkpoint_data.get('num_es2', 0)}

System:
  Moving atoms: {n_moving}
  Pair types: {n_pt}
  Total atoms: {checkpoint_data.get('n_atoms', 'N/A')}

Parameters:
  disp_max: {checkpoint_data.get('disp_max', 'N/A')} Å
  ratio_at_limit: {checkpoint_data.get('ratio_at_limit', 'N/A')}
  divisor_T_dimer_gp: {checkpoint_data.get('divisor_T_dimer_gp', 'N/A')}
"""
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.9))


def plot_dual_gp_analysis(checkpoint_data: Dict, output_dir: str = '.'):
    """Create comprehensive analysis plots for Dual GP walker."""
    # Create figure with multiple subplots - increased to 7 rows
    fig = plt.figure(figsize=(26, 30), constrained_layout=True)
    gs = GridSpec(7, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Outer iteration convergence with table data
    ax1 = fig.add_subplot(gs[0, :])
    plot_outer_convergence(ax1, checkpoint_data)
    
    # 2. Inner iteration history (GP values)
    ax2 = fig.add_subplot(gs[1, :])
    plot_inner_iterations(ax2, checkpoint_data)
    
    # 3. GP vs Actual comparison
    ax3 = fig.add_subplot(gs[2, 0])
    plot_gp_vs_actual(ax3, checkpoint_data)
    
    # 4. Efficiency analysis
    ax4 = fig.add_subplot(gs[2, 1])
    plot_efficiency(ax4, checkpoint_data)
    
    # 5. Observation distribution
    ax5 = fig.add_subplot(gs[2, 2])
    plot_observation_distribution(ax5, checkpoint_data)
    
    # 6. Hyperparameter evolution
    ax6 = fig.add_subplot(gs[3, 0])
    plot_hyperparameter_evolution(ax6, checkpoint_data)
    
    # 7. Curvature analysis
    ax7 = fig.add_subplot(gs[3, 1])
    plot_curvature_analysis(ax7, checkpoint_data)
    
    # 8. Thermal sampling info (unique to dual GP)
    ax8 = fig.add_subplot(gs[3, 2])
    plot_thermal_info(ax8, checkpoint_data)
    
    # 9. Raw data variability (new)
    ax9 = fig.add_subplot(gs[4, 0])
    plot_dual_gp_raw_variability(ax9, checkpoint_data)
    
    # 10. GP1 prediction variability (new)
    ax10 = fig.add_subplot(gs[4, 1])
    plot_dual_gp_prediction_variability(ax10, checkpoint_data)
    
    # 11. Thermal snapshot analysis (new)
    ax11 = fig.add_subplot(gs[4, 2])
    plot_dual_gp_thermal_analysis(ax11, checkpoint_data)
    
    # 12. GP1 training data std dev
    ax12 = fig.add_subplot(gs[5, 0])
    plot_dual_gp1_training_std(ax12, checkpoint_data)
    
    # 13. Raw thermal evaluation std dev
    ax13 = fig.add_subplot(gs[5, 1])
    plot_dual_raw_thermal_std(ax13, checkpoint_data)
    
    # 14. System info (spanning bottom row)
    ax14 = fig.add_subplot(gs[6, :])
    plot_dual_gp_system_info(ax14, checkpoint_data)
    
    plt.suptitle('Dual GP Walker Analysis (GP1 Thermal + GP2 Acceleration)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Save
    output_file = os.path.join(output_dir, f'dual_gp_analysis_iter{get_iteration_count(checkpoint_data)}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nDual GP analysis saved to: {output_file}")
    plt.close()


def plot_thermal_info(ax, checkpoint_data):
    """Plot thermal sampling information unique to dual GP."""
    ax.axis('off')
    
    # Extract thermal info
    temperature = checkpoint_data.get('temperature', 'N/A')
    thermal_noise = checkpoint_data.get('thermal_noise', None)
    num_snapshots = checkpoint_data.get('num_snapshots', 'N/A')
    
    info_text = f"""Thermal Sampling Info
{'─' * 30}
Temperature: {temperature} K
Snapshots per evaluation: {num_snapshots}

"""
    
    if thermal_noise:
        info_text += f"""Thermal Noise:
  Force σ: {thermal_noise[0]:.3e} eV/Å
  Energy σ: {thermal_noise[1]:.3e} eV

"""
    
    # Add GP1 info if available
    if 'gp1_trained' in checkpoint_data:
        info_text += f"""GP1 Status:
  Trained: {checkpoint_data.get('gp1_trained', False)}
  Current location set: {checkpoint_data.get('gp1_location_set', False)}
"""
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.9))
    
    ax.set_title('Thermal Sampling Parameters', fontsize=12, fontweight='bold')


def plot_dual_gp_raw_variability(ax, checkpoint_data):
    """Plot evolution of energy and force magnitudes from accurate evaluations."""
    E_R_acc = checkpoint_data.get('E_R_acc', [])
    maxF_R_acc = checkpoint_data.get('maxF_R_acc', [])
    
    if not E_R_acc or not maxF_R_acc:
        ax.text(0.5, 0.5, 'No accurate evaluation data available', ha='center', va='center')
        ax.set_title('Accurate Evaluation Evolution')
        return
    
    # Handle different array lengths by using the minimum
    min_length = min(len(E_R_acc), len(maxF_R_acc))
    E_R_acc = E_R_acc[:min_length]
    maxF_R_acc = maxF_R_acc[:min_length]
    iterations = list(range(min_length))
    
    # Create twin axis for forces
    ax2 = ax.twinx()
    
    # Plot energies
    line1 = ax.plot(iterations, E_R_acc, 'b-o', markersize=6, label='Energy', linewidth=2)
    ax.set_xlabel('Evaluation Index')
    ax.set_ylabel('Energy (eV)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # Plot forces
    line2 = ax2.plot(iterations, maxF_R_acc, 'r-s', markersize=6, label='Max |F|', linewidth=2)
    ax2.set_ylabel('Max Force (eV/Å)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Calculate and display variability
    if len(E_R_acc) > 1:
        energy_std = np.std(E_R_acc)
        force_std = np.std(maxF_R_acc)
        
        info_text = f"""Variability (std dev):
Energy: {energy_std:.3e} eV
Force: {force_std:.3e} eV/Å
Total evaluations: {len(E_R_acc)}"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))
    
    ax.set_title('Accurate Evaluation Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')


def plot_dual_gp_prediction_variability(ax, checkpoint_data):
    """Plot GP model prediction errors from table history."""
    table_history = checkpoint_data.get('table_history', [])
    
    if not table_history:
        ax.text(0.5, 0.5, 'No table history available', ha='center', va='center')
        ax.set_title('GP Model Prediction Errors')
        return
    
    # Extract data from table history
    steps = []
    gp1_energy_errors = []
    gp2_energy_errors = []
    gp1_force_errors = []
    gp2_force_errors = []
    
    for entry in table_history:
        if 'E_Actual' in entry and not np.isnan(entry['E_Actual']):
            steps.append(entry['step'])
            
            # Energy errors
            if 'E_GP1' in entry and not np.isnan(entry['E_GP1']):
                gp1_energy_errors.append(abs(entry['E_GP1'] - entry['E_Actual']))
            else:
                gp1_energy_errors.append(np.nan)
                
            if 'E_GP2' in entry and not np.isnan(entry['E_GP2']):
                gp2_energy_errors.append(abs(entry['E_GP2'] - entry['E_Actual']))
            else:
                gp2_energy_errors.append(np.nan)
                
            # Force errors
            if 'F_GP1' in entry and not np.isnan(entry['F_GP1']) and 'F_Actual' in entry:
                gp1_force_errors.append(abs(entry['F_GP1'] - entry['F_Actual']))
            else:
                gp1_force_errors.append(np.nan)
                
            if 'F_GP2' in entry and not np.isnan(entry['F_GP2']) and 'F_Actual' in entry:
                gp2_force_errors.append(abs(entry['F_GP2'] - entry['F_Actual']))
            else:
                gp2_force_errors.append(np.nan)
    
    if not steps:
        ax.text(0.5, 0.5, 'No valid data in table history', ha='center', va='center')
        ax.set_title('GP Model Prediction Errors')
        return
    
    # Create twin axis
    ax2 = ax.twinx()
    
    # Plot energy errors
    ax.plot(steps, gp1_energy_errors, 'g-o', label='GP1 Energy', markersize=6, linewidth=2)
    ax.plot(steps, gp2_energy_errors, 'b-s', label='GP2 Energy', markersize=6, linewidth=2)
    ax.set_ylabel('Energy Error (eV)', color='darkgreen')
    ax.tick_params(axis='y', labelcolor='darkgreen')
    
    # Plot force errors
    ax2.plot(steps, gp1_force_errors, 'orange', linestyle='--', marker='^', 
             label='GP1 Force', markersize=6, linewidth=2)
    ax2.plot(steps, gp2_force_errors, 'red', linestyle='--', marker='v', 
             label='GP2 Force', markersize=6, linewidth=2)
    ax2.set_ylabel('Force Error (eV/Å)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    ax.set_xlabel('Iteration')
    ax.set_title('GP Model Prediction Errors vs Actual', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    
    # Add statistics
    valid_gp1_e = [e for e in gp1_energy_errors if not np.isnan(e)]
    valid_gp2_e = [e for e in gp2_energy_errors if not np.isnan(e)]
    valid_gp1_f = [f for f in gp1_force_errors if not np.isnan(f)]
    valid_gp2_f = [f for f in gp2_force_errors if not np.isnan(f)]
    
    stats_text = "Mean Errors:\n"
    if valid_gp1_e:
        stats_text += f"GP1 Energy: {np.mean(valid_gp1_e):.3e} eV\n"
    if valid_gp2_e:
        stats_text += f"GP2 Energy: {np.mean(valid_gp2_e):.3e} eV\n"
    if valid_gp1_f:
        stats_text += f"GP1 Force: {np.mean(valid_gp1_f):.3e} eV/Å\n"
    if valid_gp2_f:
        stats_text += f"GP2 Force: {np.mean(valid_gp2_f):.3e} eV/Å"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))


def plot_dual_gp_thermal_analysis(ax, checkpoint_data):
    """Plot thermal noise analysis and GP model comparisons."""
    thermal_noise = checkpoint_data.get('thermal_noise', None)
    temperature = checkpoint_data.get('temperature', None)
    num_snapshots = checkpoint_data.get('num_snapshots', None)
    table_history = checkpoint_data.get('table_history', [])
    
    # Create a comprehensive analysis view
    ax.axis('off')
    
    info_text = "Thermal Analysis Summary\n" + "="*40 + "\n\n"
    
    # Thermal parameters
    if thermal_noise:
        info_text += f"Thermal Noise Parameters:\n"
        info_text += f"  Force σ: {thermal_noise[0]:.3e} eV/Å\n"
        info_text += f"  Energy σ: {thermal_noise[1]:.3e} eV\n"
        info_text += f"  Temperature: {temperature} K\n"
        info_text += f"  Snapshots per eval: {num_snapshots}\n\n"
    
    # GP1 vs GP2 comparison from table history
    if table_history:
        # Extract GP differences
        gp1_gp2_energy_diffs = []
        gp1_gp2_force_diffs = []
        
        for entry in table_history:
            if ('E_GP1' in entry and not np.isnan(entry['E_GP1']) and 
                'E_GP2' in entry and not np.isnan(entry['E_GP2'])):
                gp1_gp2_energy_diffs.append(abs(entry['E_GP1'] - entry['E_GP2']))
            
            if ('F_GP1' in entry and not np.isnan(entry['F_GP1']) and 
                'F_GP2' in entry and not np.isnan(entry['F_GP2'])):
                gp1_gp2_force_diffs.append(abs(entry['F_GP1'] - entry['F_GP2']))
        
        if gp1_gp2_energy_diffs:
            info_text += "GP1 vs GP2 Differences:\n"
            info_text += f"  Mean Energy Diff: {np.mean(gp1_gp2_energy_diffs):.3e} eV\n"
            info_text += f"  Max Energy Diff: {np.max(gp1_gp2_energy_diffs):.3e} eV\n"
            
        if gp1_gp2_force_diffs:
            info_text += f"  Mean Force Diff: {np.mean(gp1_gp2_force_diffs):.3e} eV/Å\n"
            info_text += f"  Max Force Diff: {np.max(gp1_gp2_force_diffs):.3e} eV/Å\n\n"
    
    # Analysis of convergence behavior
    E_R_acc = checkpoint_data.get('E_R_acc', [])
    maxF_R_acc = checkpoint_data.get('maxF_R_acc', [])
    
    if E_R_acc and maxF_R_acc and len(E_R_acc) > 1:
        info_text += "Convergence Analysis:\n"
        
        # Energy changes between evaluations
        energy_changes = np.diff(E_R_acc)
        info_text += f"  Energy change std dev: {np.std(energy_changes):.3e} eV\n"
        info_text += f"  Max energy change: {np.max(np.abs(energy_changes)):.3e} eV\n"
        
        # Force convergence
        force_changes = np.diff(maxF_R_acc)
        info_text += f"  Force change std dev: {np.std(force_changes):.3e} eV/Å\n"
        info_text += f"  Final force reduction: {(maxF_R_acc[0] - maxF_R_acc[-1]):.3e} eV/Å\n"
    
    # Display the text
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.9))
    
    ax.set_title('Thermal and Convergence Analysis', fontsize=12, fontweight='bold')


def plot_dual_gp1_training_std(ax, checkpoint_data):
    """Plot standard deviation of GP1 predictions on training data positions."""
    # This plot analyzes GP1's prediction variability when evaluated at training positions
    
    # From table history, we can analyze GP1 predictions
    table_history = checkpoint_data.get('table_history', [])
    
    if table_history and len(table_history) > 1:
        # Collect GP1 predictions with iteration numbers
        iterations = []
        gp1_energies = []
        gp1_forces = []
        
        for i, entry in enumerate(table_history):
            if 'E_GP1' in entry and not np.isnan(entry['E_GP1']):
                iterations.append(i)
                gp1_energies.append(entry['E_GP1'])
                gp1_forces.append(entry['F_GP1'] if 'F_GP1' in entry and not np.isnan(entry['F_GP1']) else np.nan)
        
        if gp1_energies and len(gp1_energies) > 1:
            # Create two subplots - one for energy, one for forces
            ax.clear()
            
            # Energy subplot
            ax2 = ax.twinx()
            
            # Plot energy values
            color_energy = 'tab:blue'
            ax.plot(iterations, gp1_energies, 'o-', color=color_energy, alpha=0.6, label='GP1 Energy')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('GP1 Energy (eV)', color=color_energy)
            ax.tick_params(axis='y', labelcolor=color_energy)
            
            # Plot forces on secondary axis
            color_force = 'tab:red'
            valid_forces = [(i, f) for i, f in zip(iterations, gp1_forces) if not np.isnan(f)]
            if valid_forces:
                force_iters, force_vals = zip(*valid_forces)
                ax2.plot(force_iters, force_vals, 's-', color=color_force, alpha=0.6, label='GP1 Max Force')
                ax2.set_ylabel('GP1 Max Force (eV/Å)', color=color_force)
                ax2.tick_params(axis='y', labelcolor=color_force)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            
            # Add statistics text
            energy_std = np.std(gp1_energies)
            energy_mean = np.mean(gp1_energies)
            stats_text = f"Energy: μ={energy_mean:.3f}, σ={energy_std:.2e} eV"
            if valid_forces:
                force_std = np.std(force_vals)
                force_mean = np.mean(force_vals)
                stats_text += f"\nForce: μ={force_mean:.3f}, σ={force_std:.2e} eV/Å"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
        else:
            ax.text(0.5, 0.5, 'Insufficient data for analysis', 
                   transform=ax.transAxes, ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'No GP1 prediction data available', 
               transform=ax.transAxes, ha='center', va='center')
    
    ax.set_title('GP1 Prediction Variability', fontsize=12, fontweight='bold')


def plot_dual_raw_thermal_std(ax, checkpoint_data):
    """Plot standard deviation of raw thermal evaluations."""
    # This analyzes the actual evaluation data variability
    E_R_acc = checkpoint_data.get('E_R_acc', [])
    maxF_R_acc = checkpoint_data.get('maxF_R_acc', [])
    num_snapshots = checkpoint_data.get('num_snapshots', 10)
    temperature = checkpoint_data.get('temperature', 'N/A')
    thermal_noise = checkpoint_data.get('thermal_noise', None)
    
    if E_R_acc or maxF_R_acc:
        # Clear axis and prepare for plotting
        ax.clear()
        
        # Create subplots for energy and forces
        if E_R_acc and maxF_R_acc:
            # Use twin axes for both energy and forces
            ax2 = ax.twinx()
            
            # Plot energy data
            if len(E_R_acc) > 1:
                iterations_e = list(range(len(E_R_acc)))
                color_energy = 'tab:blue'
                
                # Plot raw energy values
                ax.plot(iterations_e, E_R_acc, 'o-', color=color_energy, alpha=0.6, label='Raw Energy')
                ax.set_xlabel('Evaluation')
                ax.set_ylabel('Energy (eV)', color=color_energy)
                ax.tick_params(axis='y', labelcolor=color_energy)
                
                # Plot expected thermal noise band for energy
                if thermal_noise:
                    expected_energy_std = thermal_noise[1]
                    energy_mean = np.mean(E_R_acc)
                    ax.axhline(energy_mean, color=color_energy, linestyle='--', alpha=0.5)
                    ax.fill_between(iterations_e, 
                                  energy_mean - expected_energy_std, 
                                  energy_mean + expected_energy_std,
                                  color=color_energy, alpha=0.2, label='Expected ±σ_E')
            
            # Plot force data
            if len(maxF_R_acc) > 1:
                iterations_f = list(range(len(maxF_R_acc)))
                color_force = 'tab:red'
                
                # Plot raw force values
                ax2.plot(iterations_f, maxF_R_acc, 's-', color=color_force, alpha=0.6, label='Max |Force|')
                ax2.set_ylabel('Max |Force| (eV/Å)', color=color_force)
                ax2.tick_params(axis='y', labelcolor=color_force)
                
                # Plot expected thermal noise band for forces
                if thermal_noise:
                    expected_force_std = thermal_noise[0]
                    force_mean = np.mean(maxF_R_acc)
                    ax2.axhline(force_mean, color=color_force, linestyle='--', alpha=0.5)
                    ax2.fill_between(iterations_f,
                                   force_mean - expected_force_std,
                                   force_mean + expected_force_std,
                                   color=color_force, alpha=0.2, label='Expected ±σ_F')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            
        elif E_R_acc:
            # Only energy data available
            if len(E_R_acc) > 1:
                iterations = list(range(len(E_R_acc)))
                ax.plot(iterations, E_R_acc, 'o-', color='tab:blue', alpha=0.6, label='Raw Energy')
                
                if thermal_noise:
                    expected_energy_std = thermal_noise[1]
                    energy_mean = np.mean(E_R_acc)
                    ax.axhline(energy_mean, color='tab:blue', linestyle='--', alpha=0.5)
                    ax.fill_between(iterations,
                                  energy_mean - expected_energy_std,
                                  energy_mean + expected_energy_std,
                                  color='tab:blue', alpha=0.2, label='Expected ±σ_E')
                
                ax.set_xlabel('Evaluation')
                ax.set_ylabel('Energy (eV)')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        elif maxF_R_acc:
            # Only force data available
            if len(maxF_R_acc) > 1:
                iterations = list(range(len(maxF_R_acc)))
                ax.plot(iterations, maxF_R_acc, 's-', color='tab:red', alpha=0.6, label='Max |Force|')
                
                if thermal_noise:
                    expected_force_std = thermal_noise[0]
                    force_mean = np.mean(maxF_R_acc)
                    ax.axhline(force_mean, color='tab:red', linestyle='--', alpha=0.5)
                    ax.fill_between(iterations,
                                  force_mean - expected_force_std,
                                  force_mean + expected_force_std,
                                  color='tab:red', alpha=0.2, label='Expected ±σ_F')
                
                ax.set_xlabel('Evaluation')
                ax.set_ylabel('Max |Force| (eV/Å)')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # Add statistics information
        stats_text = f"T={temperature}K, {num_snapshots} snapshots/eval"
        if E_R_acc and len(E_R_acc) > 1:
            energy_std = np.std(E_R_acc)
            energy_mean = np.mean(E_R_acc)
            stats_text += f"\nE: μ={energy_mean:.3f}, σ={energy_std:.2e} eV"
            if thermal_noise:
                ratio_e = energy_std / thermal_noise[1]
                stats_text += f" (ratio={ratio_e:.2f})"
        
        if maxF_R_acc and len(maxF_R_acc) > 1:
            force_std = np.std(maxF_R_acc)
            force_mean = np.mean(maxF_R_acc)
            stats_text += f"\nF: μ={force_mean:.3f}, σ={force_std:.2e} eV/Å"
            if thermal_noise:
                ratio_f = force_std / thermal_noise[0]
                stats_text += f" (ratio={ratio_f:.2f})"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    else:
        # No data available
        ax.axis('off')
        ax.text(0.5, 0.5, 'No evaluation data available',
               transform=ax.transAxes, ha='center', va='center')
    
    ax.set_title('Raw Data Thermal Variability', fontsize=12, fontweight='bold')


def plot_dual_gp_system_info(ax, checkpoint_data):
    """Plot comprehensive system information for dual GP."""
    ax.axis('off')
    
    # Get iteration count
    bigiter = checkpoint_data.get('bigiter', checkpoint_data.get('outer_iteration', 0))
    
    # Get atomic info
    atomic_info = checkpoint_data.get('atomic_info', {})
    n_moving = len(atomic_info.get('moving_indices', []))
    n_pt = atomic_info.get('n_pt', 0)
    
    # Calculate efficiency metrics
    obs_total = checkpoint_data.get('obs_total', 0)
    obs_initrot = checkpoint_data.get('obs_initrot', 0)
    inner_iters = len(checkpoint_data.get('E_R_gp', []))
    
    efficiency_text = ""
    if obs_total > obs_initrot and inner_iters > 0:
        obs_main = obs_total - obs_initrot
        speedup = inner_iters / obs_main if obs_main > 0 else 0
        efficiency_text = f"""
Efficiency:
  Main observations: {obs_main}
  GP evaluations: {inner_iters}
  Speedup: {speedup:.1f}x
"""
    
    # Create three columns of info
    col1_text = f"""System Configuration
{'─' * 25}
Moving atoms: {n_moving}
Pair types: {n_pt}
Total atoms: {checkpoint_data.get('n_atoms', 'N/A')}
Temperature: {checkpoint_data.get('temperature', 'N/A')} K

Stopping Parameters:
  disp_max: {checkpoint_data.get('disp_max', 'N/A')} Å
  ratio_at_limit: {checkpoint_data.get('ratio_at_limit', 'N/A')}
  divisor_T_dimer_gp: {checkpoint_data.get('divisor_T_dimer_gp', 'N/A')}
"""
    
    col2_text = f"""Progress Summary
{'─' * 25}
Outer iterations: {bigiter}
Total observations: {obs_total}
  Initial rotations: {obs_initrot}
  Main search: {obs_total - obs_initrot}
Inner iterations: {inner_iters}
Converged: {checkpoint_data.get('converged', False)}

Stopping Statistics:
  Max iterations: {checkpoint_data.get('num_esmax', 0)}
  Inter-atomic dist: {checkpoint_data.get('num_es1', 0)}
  Displacement: {checkpoint_data.get('num_es2', 0)}
"""
    
    col3_text = f"""GP Models Status
{'─' * 25}
GP1 (Thermal Sampling):
  Energy reference: {checkpoint_data.get('energy_reference', 'N/A'):.4f} eV
  Thermal noise: {'Yes' if checkpoint_data.get('thermal_noise') else 'No'}
  
GP2 (Acceleration):
  Trained: {checkpoint_data.get('gp2_trained', False)}
  Training points: {len(checkpoint_data.get('R_all', []))}
{efficiency_text}
"""
    
    # Plot three columns
    ax.text(0.02, 0.95, col1_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))
    
    ax.text(0.35, 0.95, col2_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    ax.text(0.68, 0.95, col3_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))


def plot_pure_dimer_analysis(checkpoint_data: Dict, output_dir: str = '.'):
    """Create analysis plots for Pure Dimer walker."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pure Dimer Walker Analysis', fontsize=16, fontweight='bold')
    
    # Extract data from trajectory: (positions, energy, forces)
    trajectory = checkpoint_data.get('trajectory', [])
    energies = []
    max_forces = []
    rms_forces = []
    displacements = []
    
    if trajectory:
        initial_pos = None
        for i, entry in enumerate(trajectory):
            if len(entry) >= 3:
                positions, energy, forces = entry[:3]
                energies.append(energy)
                
                # Calculate force statistics
                forces_array = np.array(forces).reshape(-1, 3)
                force_mags = np.linalg.norm(forces_array, axis=1)
                max_forces.append(np.max(force_mags))
                rms_forces.append(np.sqrt(np.mean(force_mags**2)))
                
                # Calculate displacement from initial position
                if initial_pos is None:
                    initial_pos = np.array(positions)
                    displacements.append(0.0)
                else:
                    pos_diff = np.array(positions) - initial_pos
                    pos_diff_3d = pos_diff.reshape(-1, 3)
                    displacement = np.sqrt(np.mean(np.sum(pos_diff_3d**2, axis=1)))
                    displacements.append(displacement)
    
    # 1. Energy and force convergence
    ax = axes[0, 0]
    
    if energies:
        ax2 = ax.twinx()
        
        # Energy
        line1 = ax.plot(energies, 'b-o', label='Energy', markersize=6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy (eV)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Force
        if max_forces:
            line2 = ax2.plot(max_forces, 'r-s', label='Max |Force|', markersize=6)
            ax2.set_ylabel('Max |Force| (eV/Å)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add convergence line
            conv_criteria = checkpoint_data.get('force_tolerance', 0.01)
            ax2.axhline(y=conv_criteria, color='green', linestyle='--', alpha=0.5)
        
        ax.set_title('Convergence History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 2. Step sizes
    ax = axes[0, 1]
    step_sizes = checkpoint_data.get('step_sizes', [])
    
    if step_sizes:
        ax.plot(step_sizes, 'g-o', markersize=6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Step Size (Å)')
        ax.set_title('Step Size History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 3. Rotation angles
    ax = axes[1, 0]
    rotation_angles = checkpoint_data.get('rotation_angles', [])
    
    if rotation_angles:
        ax.plot(rotation_angles, 'm-o', markersize=6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Rotation Angle (radians)')
        ax.set_title('Dimer Rotation History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 4. Summary info
    ax = axes[1, 1]
    ax.axis('off')
    
    # Format final values safely
    final_energy = f"{energies[-1]:.6f}" if energies else "N/A"
    final_max_force = f"{max_forces[-1]:.6f}" if max_forces else "N/A"
    
    info_text = f"""Pure Dimer Summary
{'─' * 25}
Steps: {checkpoint_data.get('steps', 0)}
VASP evaluations: {checkpoint_data.get('vasp_eval_count', 0)}
Converged: {checkpoint_data.get('converged', False)}
Final energy: {final_energy} eV
Final max force: {final_max_force} eV/Å
"""
    
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'pure_dimer_analysis_step{checkpoint_data.get("steps", 0)}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPure Dimer analysis saved to: {output_file}")
    plt.close()


def plot_minimizer_analysis(checkpoint_data: Dict, output_dir: str = '.'):
    """Create analysis plots for WalkerMinimizer."""
    # Extract trajectory data
    trajectory = checkpoint_data.get('trajectory', [])
    if len(trajectory) < 2:
        print("Insufficient trajectory data for plotting")
        return
    
    # Parse trajectory data: (positions, energy, forces)
    energies = []
    max_forces = []
    rms_forces = []
    steps = []
    
    for i, entry in enumerate(trajectory):
        if len(entry) >= 3:
            positions, energy, forces = entry[:3]
            energies.append(energy)
            
            # Calculate force statistics
            forces_array = np.array(forces).reshape(-1, 3)
            force_mags = np.linalg.norm(forces_array, axis=1)
            max_forces.append(np.max(force_mags))
            rms_forces.append(np.sqrt(np.mean(force_mags**2)))
            steps.append(i)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Energy convergence
    ax1 = fig.add_subplot(gs[0, 0])
    if len(energies) > 1:
        # Plot relative to initial energy
        energy_ref = energies[0]
        rel_energies = np.array(energies) - energy_ref
        ax1.plot(steps, rel_energies, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Relative Energy (eV)')
        ax1.set_title('Energy Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Add final energy change
        final_change = rel_energies[-1]
        ax1.text(0.05, 0.95, f'Final ΔE: {final_change:.6f} eV', 
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Force convergence
    ax2 = fig.add_subplot(gs[0, 1])
    if len(max_forces) > 1:
        ax2.semilogy(steps, max_forces, 'r-', linewidth=2, marker='s', markersize=4, label='Max |F|')
        ax2.semilogy(steps, rms_forces, 'g-', linewidth=2, marker='^', markersize=4, label='RMS |F|')
        ax2.axhline(y=0.01, color='k', linestyle='--', alpha=0.7, label='Target (0.01 eV/Å)')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Force (eV/Å)')
        ax2.set_title('Force Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add convergence info
        final_max_f = max_forces[-1]
        converged_text = 'Converged' if final_max_f < 0.01 else 'Not Converged'
        ax2.text(0.05, 0.95, f'Final Max F: {final_max_f:.6f} eV/Å\n{converged_text}', 
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen' if final_max_f < 0.01 else 'lightcoral', alpha=0.8))
    
    # 3. Energy vs Max Force scatter
    ax3 = fig.add_subplot(gs[1, 0])
    if len(energies) > 1 and len(max_forces) > 1:
        rel_energies = np.array(energies) - energies[0]
        scatter = ax3.scatter(max_forces, rel_energies, c=steps, cmap='viridis', 
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Max Force (eV/Å)')
        ax3.set_ylabel('Relative Energy (eV)')
        ax3.set_title('Energy vs Force Correlation')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Step')
        
        # Add start and end markers
        ax3.scatter(max_forces[0], rel_energies[0], s=100, c='red', marker='o', 
                   edgecolors='black', linewidth=2, label='Start', zorder=5)
        ax3.scatter(max_forces[-1], rel_energies[-1], s=100, c='blue', marker='*', 
                   edgecolors='black', linewidth=2, label='End', zorder=5)
        ax3.legend()
    
    # 4. Force evaluations per step
    ax4 = fig.add_subplot(gs[1, 1])
    force_evals = checkpoint_data.get('force_evals_per_step', [])
    if force_evals:
        eval_steps = list(range(len(force_evals)))
        ax4.bar(eval_steps, force_evals, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Force Evaluations')
        ax4.set_title('Computational Efficiency')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        total_evals = sum(force_evals)
        avg_evals = np.mean(force_evals)
        ax4.text(0.05, 0.95, f'Total: {total_evals}\nAvg: {avg_evals:.1f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 5. Minimizer progress table (if available)
    ax5 = fig.add_subplot(gs[2, :])
    table_history = checkpoint_data.get('table_history', [])
    if table_history and len(table_history) > 1:
        # Parse table entries to get numerical data
        step_data = []
        for entry in table_history[1:]:  # Skip header
            try:
                parts = entry.strip().split()
                if len(parts) >= 4:
                    step = int(parts[0])
                    energy_change = float(parts[1])
                    max_force = float(parts[2])
                    rms_force = float(parts[3])
                    step_data.append([step, energy_change, max_force, rms_force])
            except (ValueError, IndexError):
                continue
        
        if step_data:
            step_data = np.array(step_data)
            
            # Create twin axes for energy and force
            ax5_twin = ax5.twinx()
            
            # Plot energy change
            line1 = ax5.plot(step_data[:, 0], step_data[:, 1], 'b-', linewidth=2, 
                           marker='o', markersize=3, label='Energy Change', alpha=0.8)
            
            # Plot forces
            line2 = ax5_twin.plot(step_data[:, 0], step_data[:, 2], 'r-', linewidth=2, 
                                marker='s', markersize=3, label='Max Force', alpha=0.8)
            line3 = ax5_twin.plot(step_data[:, 0], step_data[:, 3], 'g-', linewidth=2, 
                                marker='^', markersize=3, label='RMS Force', alpha=0.8)
            
            # Formatting
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Energy Change (eV)', color='blue')
            ax5_twin.set_ylabel('Force (eV/Å)', color='red')
            ax5.set_title('Minimization Progress (Table Data)')
            ax5.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax5.legend(lines, labels, loc='upper right')
            
            # Color the y-axis labels
            ax5.tick_params(axis='y', labelcolor='blue')
            ax5_twin.tick_params(axis='y', labelcolor='red')
    else:
        ax5.text(0.5, 0.5, 'No table history available', 
                transform=ax5.transAxes, ha='center', va='center', fontsize=12)
        ax5.set_title('Minimization Progress Table')
    
    plt.suptitle('Minimizer Walker Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])  # Leave space for info box at bottom
    
    # Add overall info text box after tight_layout
    info_text = f"""Method: {checkpoint_data.get('method', 'Unknown')} | Steps: {checkpoint_data.get('steps', 0)} | Converged: {checkpoint_data.get('converged', False)} | Atoms: {checkpoint_data.get('n_atoms', 'Unknown')} | Force Evals: {checkpoint_data.get('vasp_eval_count', 'Unknown')}"""
    
    fig.text(0.5, 0.01, info_text, fontsize=9, horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))
    
    # Save plot
    output_file = os.path.join(output_dir, f'minimizer_analysis_step{checkpoint_data.get("steps", 0)}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nMinimizer analysis saved to: {output_file}")
    plt.close()


def plot_pure_neb_analysis(checkpoint_data: Dict, output_dir: str = '.'):
    """Create analysis plots for WalkerPureNEB."""
    # Extract NEB state data
    neb_state = checkpoint_data.get('neb_state', {})
    n_images = checkpoint_data.get('n_images', 0)
    
    if not neb_state or n_images == 0:
        print("Insufficient NEB data for plotting")
        return
    
    # Extract energies and forces for all images
    E_R = neb_state.get('E_R', [])
    G_R = neb_state.get('G_R', [])
    
    # Extract trajectory data for convergence plots
    trajectory = checkpoint_data.get('trajectory', [])
    
    # Extract climbing image info
    CI_on = neb_state.get('CI_on', False)
    i_CI = neb_state.get('i_CI', -1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Energy profile along path
    ax1 = fig.add_subplot(gs[0, :])
    if len(E_R) == n_images:
        # Convert E_R to a simple array if needed
        E_R_array = np.array(E_R).flatten() if isinstance(E_R, np.ndarray) else E_R
        
        # Calculate reaction coordinate (normalized distance along path)
        reaction_coord = np.linspace(0, 1, n_images)
        
        # Plot energy profile
        ax1.plot(reaction_coord, E_R_array, 'bo-', linewidth=2, markersize=8, label='NEB Path')
        
        # Mark initial and final states
        ax1.plot(0, E_R_array[0], 'gs', markersize=12, label='Initial')
        ax1.plot(1, E_R_array[-1], 'rs', markersize=12, label='Final')
        
        # Mark climbing image if active
        if CI_on and 0 < i_CI < n_images - 1:
            ax1.plot(reaction_coord[i_CI], E_R_array[i_CI], 'r*', markersize=15, label=f'CI (Image {i_CI})')
        
        # Find and mark transition state (highest energy)
        ts_idx = np.argmax(E_R_array[1:-1]) + 1  # Exclude endpoints
        ax1.plot(reaction_coord[ts_idx], E_R_array[ts_idx], 'mo', markersize=10, label=f'TS (Image {ts_idx})')
        
        ax1.set_xlabel('Reaction Coordinate')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('Energy Profile Along NEB Path', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add barrier information
        forward_barrier = float(E_R_array[ts_idx] - E_R_array[0])
        reverse_barrier = float(E_R_array[ts_idx] - E_R_array[-1])
        ax1.text(0.02, 0.98, f'Forward barrier: {forward_barrier:.3f} eV\nReverse barrier: {reverse_barrier:.3f} eV',
                transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Force magnitudes for each image
    ax2 = fig.add_subplot(gs[1, 0])
    if len(G_R) > 0:
        # Calculate force magnitudes for each image
        force_mags = []
        for g in G_R:
            if len(g) > 0:
                g_array = np.array(g).reshape(-1, 3)
                mag = np.max(np.linalg.norm(g_array, axis=1))
                force_mags.append(mag)
            else:
                force_mags.append(0.0)
        
        image_indices = np.arange(n_images)
        ax2.bar(image_indices, force_mags, color='orange', alpha=0.7)
        ax2.axhline(y=checkpoint_data.get('T_MEP', 0.1), color='r', linestyle='--', 
                   label=f'Convergence: {checkpoint_data.get("T_MEP", 0.1)}')
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Max |Force| (eV/Å)')
        ax2.set_title('Force Magnitudes by Image')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Convergence history
    ax3 = fig.add_subplot(gs[1, 1])
    if len(trajectory) > 0:
        steps = []
        max_forces = []
        
        for i, traj_entry in enumerate(trajectory):
            if len(traj_entry) >= 3:
                _, _, forces = traj_entry[:3]
                if isinstance(forces, list) and len(forces) > 0:
                    # Calculate max force across all images
                    max_f = 0
                    for img_forces in forces:
                        if len(img_forces) > 0:
                            f_array = np.array(img_forces).reshape(-1, 3)
                            max_f = max(max_f, np.max(np.linalg.norm(f_array, axis=1)))
                    max_forces.append(max_f)
                    steps.append(i)
        
        if steps:
            ax3.semilogy(steps, max_forces, 'b-', linewidth=2)
            ax3.axhline(y=checkpoint_data.get('T_MEP', 0.1), color='r', linestyle='--', 
                       label=f'Target: {checkpoint_data.get("T_MEP", 0.1)}')
            ax3.set_xlabel('NEB Step')
            ax3.set_ylabel('Max |Force| (eV/Å)')
            ax3.set_title('NEB Convergence History')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # 4. Image spacing analysis
    ax4 = fig.add_subplot(gs[1, 2])
    R = neb_state.get('R', [])
    if len(R) == n_images:
        # Calculate distances between adjacent images
        distances = []
        for i in range(1, n_images):
            R_i = np.array(R[i])
            R_prev = np.array(R[i-1])
            dist = np.linalg.norm(R_i - R_prev)
            distances.append(dist)
        
        ax4.bar(np.arange(1, n_images), distances, color='green', alpha=0.7)
        ax4.set_xlabel('Image Pair (i, i-1)')
        ax4.set_ylabel('Distance (Å)')
        ax4.set_title('Image Spacing Along Path')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add average spacing
        avg_dist = np.mean(distances)
        ax4.axhline(y=avg_dist, color='r', linestyle='--', label=f'Average: {avg_dist:.3f}')
        ax4.legend()
    
    # 5. Energy vs iteration for all images
    ax5 = fig.add_subplot(gs[2, :2])
    if len(trajectory) > 0:
        # Extract energies for each image over iterations
        image_energies = {i: [] for i in range(n_images)}
        
        for traj_entry in trajectory:
            if len(traj_entry) >= 2:
                _, energies = traj_entry[:2]
                if isinstance(energies, list) and len(energies) == n_images:
                    for i, e in enumerate(energies):
                        image_energies[i].append(e)
        
        # Find TS index if we have E_R data
        ts_idx = None
        if len(E_R) == n_images:
            E_R_array = np.array(E_R).flatten() if isinstance(E_R, np.ndarray) else E_R
            ts_idx = np.argmax(E_R_array[1:-1]) + 1
        
        # Plot energy evolution for each image
        for i in range(n_images):
            if image_energies[i]:
                label = None
                if i == 0:
                    label = 'Initial'
                elif i == n_images - 1:
                    label = 'Final'
                elif CI_on and i == i_CI:
                    label = f'CI (Image {i})'
                elif ts_idx is not None and i == ts_idx:
                    label = f'TS (Image {i})'
                
                ax5.plot(image_energies[i], linewidth=1.5, label=label, 
                        alpha=0.8 if label else 0.5)
        
        ax5.set_xlabel('NEB Step')
        ax5.set_ylabel('Energy (eV)')
        ax5.set_title('Energy Evolution for All Images')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)
    
    # 6. Table history summary
    ax6 = fig.add_subplot(gs[2, 2])
    table_history = checkpoint_data.get('table_history', [])
    
    # Create summary text
    summary_text = f"NEB Summary\n{'='*20}\n\n"
    summary_text += f"Total steps: {checkpoint_data.get('iteration', 0)}\n"
    summary_text += f"Converged: {'Yes' if checkpoint_data.get('converged', False) else 'No'}\n"
    summary_text += f"Images: {n_images}\n"
    summary_text += f"Climbing image: {'On' if CI_on else 'Off'}\n"
    
    if CI_on and 0 < i_CI < n_images - 1:
        summary_text += f"CI index: {i_CI}\n"
    
    summary_text += f"\nForce evaluations: {checkpoint_data.get('vasp_eval_count', 0)}\n"
    
    # Get final max force
    if len(G_R) > 0:
        final_max_force = 0
        for g in G_R:
            if len(g) > 0:
                g_array = np.array(g).reshape(-1, 3)
                final_max_force = max(final_max_force, np.max(np.linalg.norm(g_array, axis=1)))
        summary_text += f"Final max force: {final_max_force:.4f} eV/Å\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=10, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax6.axis('off')
    
    # Add overall title
    plt.suptitle(f'NEB Analysis - {checkpoint_data.get("timestamp", "Unknown time")}', 
                fontsize=16, fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neb_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created NEB analysis plot: {os.path.join(output_dir, 'neb_analysis.png')}")


def plot_gp2_neb_analysis(checkpoint_data: Dict, output_dir: str = '.'):
    """Create comprehensive analysis plots for GP2 NEB walker."""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. NEB path energy profile and convergence
    ax1 = fig.add_subplot(gs[0, :])
    plot_neb_energy_profile(ax1, checkpoint_data)
    
    # 2. Outer iteration convergence (GP and actual values)
    ax2 = fig.add_subplot(gs[1, :])
    plot_neb_outer_convergence(ax2, checkpoint_data)
    
    # 3. Inner iteration history (GP predictions)
    ax3 = fig.add_subplot(gs[2, 0])
    plot_neb_inner_iterations(ax3, checkpoint_data)
    
    # 4. GP vs Actual comparison
    ax4 = fig.add_subplot(gs[2, 1])
    plot_neb_gp_vs_actual(ax4, checkpoint_data)
    
    # 5. Efficiency analysis
    ax5 = fig.add_subplot(gs[2, 2])
    plot_neb_efficiency(ax5, checkpoint_data)
    
    # 6. Force distribution across images
    ax6 = fig.add_subplot(gs[3, 0])
    plot_neb_force_distribution(ax6, checkpoint_data)
    
    # 7. Path evolution
    ax7 = fig.add_subplot(gs[3, 1])
    plot_neb_path_evolution(ax7, checkpoint_data)
    
    # 8. System info
    ax8 = fig.add_subplot(gs[3, 2])
    plot_neb_system_info(ax8, checkpoint_data)
    
    plt.suptitle('GP2 NEB Walker Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, f'gp2_neb_analysis_iter{get_iteration_count(checkpoint_data)}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nGP2 NEB analysis saved to: {output_file}")
    plt.close()


def plot_neb_energy_profile(ax, checkpoint_data):
    """Plot NEB energy profile along the path."""
    neb_state = checkpoint_data.get('neb_state', {})
    n_images = checkpoint_data.get('n_images', 0)
    E_R = neb_state.get('E_R', [])
    
    if len(E_R) == n_images and n_images > 0:
        # Convert to array and flatten if needed
        E_R_array = np.array(E_R).flatten() if isinstance(E_R, np.ndarray) else E_R
        
        # Create reaction coordinate
        reaction_coord = np.linspace(0, 1, n_images)
        
        # Plot energy profile
        ax.plot(reaction_coord, E_R_array, 'bo-', linewidth=2, markersize=8, label='NEB Path')
        
        # Mark endpoints
        ax.plot(0, E_R_array[0], 'gs', markersize=12, label='Initial')
        ax.plot(1, E_R_array[-1], 'rs', markersize=12, label='Final')
        
        # Find and mark transition state
        ts_idx = np.argmax(E_R_array[1:-1]) + 1
        ax.plot(reaction_coord[ts_idx], E_R_array[ts_idx], 'r*', markersize=15, label=f'TS (Image {ts_idx})')
        
        # Check for climbing image
        CI_on = neb_state.get('CI_on', False)
        i_CI = neb_state.get('i_CI', -1)
        if CI_on and 0 < i_CI < n_images - 1:
            ax.plot(reaction_coord[i_CI], E_R_array[i_CI], 'mo', markersize=10, label=f'CI (Image {i_CI})')
        
        ax.set_xlabel('Reaction Coordinate')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('NEB Energy Profile', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add barrier information
        forward_barrier = float(E_R_array[ts_idx] - E_R_array[0])
        reverse_barrier = float(E_R_array[ts_idx] - E_R_array[-1])
        ax.text(0.02, 0.98, f'Forward barrier: {forward_barrier:.3f} eV\nReverse barrier: {reverse_barrier:.3f} eV',
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No NEB energy data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('NEB Energy Profile', fontsize=14, fontweight='bold')


def plot_neb_outer_convergence(ax, checkpoint_data):
    """Plot outer iteration convergence for NEB."""
    table_history = checkpoint_data.get('table_history', [])
    
    if table_history:
        # Extract data from table history
        steps = []
        max_forces_actual = []
        max_forces_gp = []
        
        for row in table_history:
            step = row.get('step', len(steps))
            f_actual = row.get('maxF_Actual', row.get('max_force_actual', np.nan))
            f_gp = row.get('maxF_GP', row.get('max_force_gp', np.nan))
            
            steps.append(step)
            max_forces_actual.append(f_actual)
            max_forces_gp.append(f_gp)
        
        if steps:
            # Plot actual forces
            mask = ~np.isnan(max_forces_actual)
            if np.any(mask):
                ax.semilogy(np.array(steps)[mask], np.array(max_forces_actual)[mask], 
                           'ro-', linewidth=2, markersize=8, label='Actual Max Force')
            
            # Plot GP forces
            mask = ~np.isnan(max_forces_gp)
            if np.any(mask):
                ax.semilogy(np.array(steps)[mask], np.array(max_forces_gp)[mask], 
                           'bs--', linewidth=1.5, markersize=6, alpha=0.7, label='GP Max Force')
            
            # Add convergence threshold
            if 'T_MEP' in checkpoint_data:
                ax.axhline(y=checkpoint_data['T_MEP'], color='green', linestyle='--', 
                          alpha=0.5, label=f"Target: {checkpoint_data['T_MEP']} eV/Å")
            
            ax.set_xlabel('Outer Iteration')
            ax.set_ylabel('Max |Force| (eV/Å)')
            ax.set_title('NEB Convergence History', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        # Fallback to array data
        maxF_R_acc = checkpoint_data.get('maxF_R_acc', np.array([]))
        if len(maxF_R_acc) > 0:
            ax.semilogy(maxF_R_acc, 'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Outer Iteration')
            ax.set_ylabel('Max |Force| (eV/Å)')
            ax.set_title('NEB Convergence History', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)


def plot_neb_inner_iterations(ax, checkpoint_data):
    """Plot inner GP iterations for NEB."""
    maxF_R_gp = checkpoint_data.get('maxF_R_gp', np.array([]))
    obs_at = checkpoint_data.get('obs_at', np.array([]))
    
    if len(maxF_R_gp) > 0:
        ax.semilogy(maxF_R_gp, '-', color='tab:red', linewidth=1, alpha=0.7)
        ax.set_xlabel('Inner Iteration')
        ax.set_ylabel('GP Max |Force| (eV/Å)')
        ax.set_title('GP Inner Iterations', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark observation points
        if len(obs_at) > 0:
            for i, obs in enumerate(obs_at[1:], 1):
                ax.axvline(x=obs, color='gray', linestyle=':', alpha=0.5)
        
        # Add dynamic threshold
        if 'divisor_T_MEP_gp' in checkpoint_data and 'maxF_R_acc' in checkpoint_data:
            if len(checkpoint_data['maxF_R_acc']) > 0:
                min_force = min(checkpoint_data['maxF_R_acc'])
                threshold = max(min_force / checkpoint_data['divisor_T_MEP_gp'], 
                              checkpoint_data.get('T_MEP', 0.1) / 10.0)
                ax.axhline(y=threshold, color='orange', linestyle='--', alpha=0.5,
                          label=f'GP Target: {threshold:.4f}')
                ax.legend()
    else:
        ax.text(0.5, 0.5, 'No inner iteration data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP Inner Iterations', fontsize=12, fontweight='bold')


def plot_neb_gp_vs_actual(ax, checkpoint_data):
    """Plot GP vs actual predictions for NEB."""
    table_history = checkpoint_data.get('table_history', [])
    
    if table_history and len(table_history) > 1:
        gp_forces = []
        actual_forces = []
        
        for row in table_history[1:]:  # Skip first row
            f_gp = row.get('maxF_GP', row.get('max_force_gp', np.nan))
            f_actual = row.get('maxF_Actual', row.get('max_force_actual', np.nan))
            if not np.isnan(f_gp) and not np.isnan(f_actual):
                gp_forces.append(f_gp)
                actual_forces.append(f_actual)
        
        if gp_forces and actual_forces:
            ax.scatter(actual_forces, gp_forces, alpha=0.6, s=50, label='Max Force')
            
            # Perfect correlation line
            min_f = min(min(actual_forces), min(gp_forces))
            max_f = max(max(actual_forces), max(gp_forces))
            ax.plot([min_f, max_f], [min_f, max_f], 'k--', alpha=0.5, label='Perfect')
            
            # Calculate R²
            if len(actual_forces) > 2:
                r2 = np.corrcoef(actual_forces, gp_forces)[0, 1]**2
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                       transform=ax.transAxes, fontsize=10, va='top')
            
            ax.set_xlabel('Actual Max Force (eV/Å)')
            ax.set_ylabel('GP Max Force (eV/Å)')
            ax.set_title('GP vs Actual Forces', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data\nfor correlation plot', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP vs Actual Forces', fontsize=12, fontweight='bold')


def plot_neb_efficiency(ax, checkpoint_data):
    """Plot efficiency analysis for GP2 NEB."""
    obs_total = checkpoint_data.get('obs_total', 0)
    obs_initrot = checkpoint_data.get('obs_initrot', 0)
    inner_iters = len(checkpoint_data.get('maxF_R_gp', []))
    
    if obs_total > 0:
        # Breakdown of evaluations
        obs_main = obs_total - obs_initrot
        
        labels = []
        sizes = []
        colors = []
        
        if obs_initrot > 0:
            labels.extend(['Initial Path', 'Main Search'])
            sizes.extend([obs_initrot, obs_main])
            colors.extend(['#ff9999', '#66b3ff'])
        else:
            labels.append('VASP Evaluations')
            sizes.append(obs_total)
            colors.append('#ff9999')
        
        if inner_iters > 0:
            labels.append('GP Evaluations')
            sizes.append(inner_iters)
            colors.append('#90ee90')
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct=lambda pct: f'{int(pct*sum(sizes)/100)}',
                                         startangle=90)
        
        # Calculate speedup
        if inner_iters > 0 and obs_main > 0:
            speedup = inner_iters / obs_main
            ax.set_title(f'Efficiency: {speedup:.1f}x speedup\n(Total: {obs_total} VASP + {inner_iters} GP)', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Evaluation Distribution\n(Total: {obs_total})', 
                        fontsize=12, fontweight='bold')


def plot_neb_force_distribution(ax, checkpoint_data):
    """Plot force distribution across NEB images."""
    neb_state = checkpoint_data.get('neb_state', {})
    G_R = neb_state.get('G_R', [])
    n_images = checkpoint_data.get('n_images', 0)
    
    if len(G_R) == n_images and n_images > 0:
        force_mags = []
        for g in G_R:
            if len(g) > 0:
                g_array = np.array(g).reshape(-1, 3)
                mag = np.max(np.linalg.norm(g_array, axis=1))
                force_mags.append(mag)
            else:
                force_mags.append(0.0)
        
        image_indices = np.arange(n_images)
        colors = ['green' if i == 0 or i == n_images-1 else 'orange' for i in range(n_images)]
        
        ax.bar(image_indices, force_mags, color=colors, alpha=0.7)
        ax.axhline(y=checkpoint_data.get('T_MEP', 0.1), color='r', linestyle='--', 
                   label=f'Target: {checkpoint_data.get("T_MEP", 0.1)} eV/Å')
        
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Max |Force| (eV/Å)')
        ax.set_title('Force Distribution by Image', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')


def plot_neb_path_evolution(ax, checkpoint_data):
    """Plot how the NEB path evolves."""
    trajectory = checkpoint_data.get('trajectory', [])
    n_images = checkpoint_data.get('n_images', 0)
    
    if len(trajectory) > 1 and n_images > 0:
        # Extract energies over iterations for key images
        initial_energies = []
        final_energies = []
        ts_energies = []
        
        # Find TS index from current state
        neb_state = checkpoint_data.get('neb_state', {})
        E_R = neb_state.get('E_R', [])
        ts_idx = None
        if len(E_R) == n_images:
            E_R_array = np.array(E_R).flatten() if isinstance(E_R, np.ndarray) else E_R
            ts_idx = np.argmax(E_R_array[1:-1]) + 1
        
        for traj_entry in trajectory:
            if len(traj_entry) >= 2:
                _, energies = traj_entry[:2]
                if isinstance(energies, list) and len(energies) == n_images:
                    initial_energies.append(energies[0])
                    final_energies.append(energies[-1])
                    if ts_idx is not None:
                        ts_energies.append(energies[ts_idx])
        
        if initial_energies:
            iterations = list(range(len(initial_energies)))
            ax.plot(iterations, initial_energies, 'g-', linewidth=2, label='Initial')
            ax.plot(iterations, final_energies, 'r-', linewidth=2, label='Final')
            if ts_energies and ts_idx is not None:
                ax.plot(iterations, ts_energies, 'b-', linewidth=2, label=f'TS (Image {ts_idx})')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Energy (eV)')
            ax.set_title('Path Energy Evolution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough trajectory data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Path Energy Evolution', fontsize=12, fontweight='bold')


def plot_neb_system_info(ax, checkpoint_data):
    """Plot system information for GP2 NEB."""
    ax.axis('off')
    
    # Get iteration count
    bigiter = checkpoint_data.get('bigiter', checkpoint_data.get('outer_iteration', 0))
    
    # Get NEB state info
    neb_state = checkpoint_data.get('neb_state', {})
    n_images = checkpoint_data.get('n_images', 0)
    CI_on = neb_state.get('CI_on', False)
    i_CI = neb_state.get('i_CI', -1)
    
    # Get atomic info
    atomic_info = checkpoint_data.get('atomic_info', {})
    n_moving = len(atomic_info.get('moving_indices', []))
    
    info_text = f"""GP2 NEB System Info
{'─' * 35}
Outer iterations: {bigiter}
Total observations: {checkpoint_data.get('obs_total', 0)}
Initial path setup: {checkpoint_data.get('obs_initrot', 0)}
Inner iterations: {len(checkpoint_data.get('maxF_R_gp', []))}
Converged: {checkpoint_data.get('converged', False)}

NEB Configuration:
  Images: {n_images}
  Climbing image: {'On' if CI_on else 'Off'}
  CI index: {i_CI if CI_on and i_CI > 0 else 'N/A'}
  Spring constants:
    k_parallel: {checkpoint_data.get('k_par', 'N/A')}
    k_perpendicular: {checkpoint_data.get('k_perp', 'N/A')}

System:
  Moving atoms: {n_moving}
  Total atoms: {checkpoint_data.get('n_atoms', 'N/A')}

Convergence:
  Target: {checkpoint_data.get('T_MEP', 'N/A')} eV/Å
  GP divisor: {checkpoint_data.get('divisor_T_MEP_gp', 'N/A')}
"""
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.9))


def plot_gp1_path_analysis(checkpoint_data: Dict, output_dir: str = '.'):
    """Create comprehensive analysis plots for GP1 Path walker."""
    # Extract walker object if available
    walker = checkpoint_data.get('walker', None)
    
    # Try to get results from different possible locations
    results = None
    
    # First try: direct results key
    if 'results' in checkpoint_data:
        results = checkpoint_data['results']
    # Second try: from walker object
    elif walker and hasattr(walker, 'results'):
        results = walker.results
    
    # If no results found, try to reconstruct from checkpoint data
    if results is None:
        print("WARNING: No results found in checkpoint. Creating basic plots from available data.")
        
        # Extract data from walker if available
        if walker:
            n_images = walker.n_images if hasattr(walker, 'n_images') else len(checkpoint_data.get('statistics', []))
            temperature = walker.temperature if hasattr(walker, 'temperature') else 300
            path_positions = walker.path_positions if hasattr(walker, 'path_positions') else None
        else:
            n_images = len(checkpoint_data.get('statistics', []))
            temperature = 300
            path_positions = None
            
        results = {
            'n_images': n_images,
            'completed_images': checkpoint_data.get('completed_images', 0),
            'temperature': temperature,
            'per_image_stats': checkpoint_data.get('statistics', []),
            'thermal_noises': checkpoint_data.get('thermal_noises', []),
        }
        
        # Add path positions to checkpoint data if available
        if path_positions is not None:
            checkpoint_data['path_positions'] = path_positions
    
    # Create figure with multiple subplots - 5x3 grid matching walker_gp1_path
    fig = plt.figure(figsize=(24, 25))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Error Analysis
    # 1. Energy prediction errors along path
    ax1 = fig.add_subplot(gs[0, 0])
    plot_gp1_energy_errors(ax1, results)
    
    # 2. Force prediction errors along path
    ax2 = fig.add_subplot(gs[0, 1])
    plot_gp1_force_errors(ax2, results)
    
    # 3. σ/MAD ratios along path
    ax3 = fig.add_subplot(gs[0, 2])
    plot_gp1_sigma_mad_ratios(ax3, results)
    
    # Row 2: Parity Plots
    # 4. Energy parity plot
    ax4 = fig.add_subplot(gs[1, 0])
    plot_gp1_energy_parity(ax4, results)
    
    # 5. Force parity plot
    ax5 = fig.add_subplot(gs[1, 1])
    plot_gp1_force_parity(ax5, results)
    
    # 6. Progress tracking
    ax6 = fig.add_subplot(gs[1, 2])
    plot_gp1_progress(ax6, checkpoint_data)
    
    # Row 3: Energy Profiles
    # 7. GP1 vs Actual Energy Profiles
    ax7 = fig.add_subplot(gs[2, 0])
    plot_gp1_vs_actual_profile(ax7, results)
    
    # 8. Energy landscape
    ax8 = fig.add_subplot(gs[2, 1])
    plot_gp1_energy_landscape(ax8, results)
    
    # 9. GP1 Energy with Uncertainty
    ax9 = fig.add_subplot(gs[2, 2])
    plot_gp1_energy_uncertainty(ax9, results)
    
    # Row 4: Statistics
    # 10. Raw data statistics
    ax10 = fig.add_subplot(gs[3, 0])
    plot_gp1_raw_statistics(ax10, results)
    
    # 11. GP1 prediction uncertainties
    ax11 = fig.add_subplot(gs[3, 1])
    plot_gp1_prediction_uncertainties(ax11, results)
    
    # 12. Summary info
    ax12 = fig.add_subplot(gs[3, 2])
    plot_gp1_system_info(ax12, checkpoint_data, results)
    
    # Row 5: Additional Analysis
    # 13. Error distribution
    ax13 = fig.add_subplot(gs[4, 0])
    plot_gp1_error_distribution(ax13, results)
    
    # 14. Uncertainty calibration
    ax14 = fig.add_subplot(gs[4, 1])
    plot_gp1_uncertainty_calibration(ax14, results)
    
    # 15. Path visualization (if available)
    ax15 = fig.add_subplot(gs[4, 2])
    plot_gp1_path_visualization(ax15, checkpoint_data)
    
    plt.suptitle('GP1 Path Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'gp1_path_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"GP1 path analysis plot saved to: {output_file}")


# GP1 Path plotting helper functions
def plot_gp1_energy_errors(ax, results):
    """Plot energy prediction errors along the path."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Energy Prediction Errors')
        return
        
    stats = results['per_image_stats']
    indices = [s.get('image_index', i) for i, s in enumerate(stats)]
    mae = [s.get('energy_mae', 0) for s in stats]
    rmse = [s.get('energy_rmse', 0) for s in stats]
    
    ax.plot(indices, mae, 'bo-', label='MAE', markersize=6)
    ax.plot(indices, rmse, 'rs--', label='RMSE', markersize=6)
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Energy Error (eV)')
    ax.set_title('Energy Prediction Errors Along Path')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_gp1_force_errors(ax, results):
    """Plot force prediction errors along the path."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Force Prediction Errors')
        return
        
    stats = results['per_image_stats']
    indices = [s.get('image_index', i) for i, s in enumerate(stats)]
    mae = [s.get('force_mae', 0) for s in stats]
    rmse = [s.get('force_rmse', 0) for s in stats]
    
    ax.plot(indices, mae, 'go-', label='MAE', markersize=6)
    ax.plot(indices, rmse, 'ms--', label='RMSE', markersize=6)
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Force Error (eV/Å)')
    ax.set_title('Force Prediction Errors Along Path')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_gp1_sigma_mad_ratios(ax, results):
    """Plot σ/MAD ratios along the path."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Uncertainty Quantification')
        return
        
    stats = results['per_image_stats']
    indices = [s.get('image_index', i) for i, s in enumerate(stats)]
    energy_ratios = [s.get('energy_sigma_mad_ratio', 0) for s in stats]
    force_ratios = [s.get('force_sigma_mad_ratio', 0) for s in stats]
    
    ax.plot(indices, energy_ratios, 'bo-', label='Energy', markersize=6)
    # Limit force ratios for visibility
    force_ratios_clipped = [min(r, 10) for r in force_ratios]
    ax.plot(indices, force_ratios_clipped, 'ro-', label='Force (clipped at 10)', markersize=6)
    ax.axhline(y=1.46, color='k', linestyle='--', alpha=0.5, label='Target (1.46)')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('σ/MAD Ratio')
    ax.set_title('Uncertainty Quantification Along Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)


def plot_gp1_energy_parity(ax, results):
    """Plot predicted vs actual energies."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Energy Parity Plot')
        return
        
    all_pred = []
    all_actual = []
    
    for stats in results['per_image_stats']:
        if 'pred_energies' in stats and 'actual_energies' in stats:
            all_pred.extend(stats['pred_energies'])
            all_actual.extend(stats['actual_energies'])
    
    if not all_pred:
        ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Energy Parity Plot')
        return
        
    all_pred = np.array(all_pred)
    all_actual = np.array(all_actual)
    
    # Remove mean for better visualization
    mean_e = np.mean(all_actual)
    all_pred -= mean_e
    all_actual -= mean_e
    
    ax.scatter(all_actual, all_pred, alpha=0.5, s=20)
    
    # Perfect correlation line
    min_val = min(all_actual.min(), all_pred.min())
    max_val = max(all_actual.max(), all_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Calculate R²
    if len(all_actual) > 1:
        r2 = np.corrcoef(all_actual, all_pred)[0, 1]**2
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Actual Energy - Mean (eV)')
    ax.set_ylabel('Predicted Energy - Mean (eV)')
    ax.set_title('Energy Parity Plot')
    ax.grid(True, alpha=0.3)


def plot_gp1_force_parity(ax, results):
    """Plot predicted vs actual force magnitudes."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Force Parity Plot')
        return
        
    all_pred = []
    all_actual = []
    
    for stats in results['per_image_stats']:
        if 'force_mags_pred' in stats and 'force_mags_actual' in stats:
            all_pred.extend(stats['force_mags_pred'])
            all_actual.extend(stats['force_mags_actual'])
    
    if not all_pred:
        ax.text(0.5, 0.5, 'No force data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Force Parity Plot')
        return
        
    all_pred = np.array(all_pred)
    all_actual = np.array(all_actual)
    
    # Subsample for visualization
    if len(all_pred) > 5000:
        idx = np.random.choice(len(all_pred), 5000, replace=False)
        all_pred = all_pred[idx]
        all_actual = all_actual[idx]
    
    ax.scatter(all_actual, all_pred, alpha=0.3, s=10)
    
    # Perfect correlation line
    max_val = max(all_actual.max(), all_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Calculate R²
    if len(all_actual) > 1:
        r2 = np.corrcoef(all_actual, all_pred)[0, 1]**2
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Actual |Force| (eV/Å)')
    ax.set_ylabel('Predicted |Force| (eV/Å)')
    ax.set_title('Force Magnitude Parity Plot')
    ax.grid(True, alpha=0.3)


def plot_gp1_progress(ax, checkpoint_data):
    """Plot progress of GP1 path analysis."""
    completed = checkpoint_data.get('completed_images', 0)
    total = checkpoint_data.get('n_images', 20)
    
    # Create progress bar
    ax.barh(0, completed/total, height=0.5, color='green', alpha=0.7, label='Completed')
    ax.barh(0, (total-completed)/total, left=completed/total, height=0.5, 
            color='lightgray', alpha=0.7, label='Remaining')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Progress')
    ax.set_yticks([])
    ax.set_title(f'Analysis Progress: {completed}/{total} images')
    
    # Add text
    ax.text(0.5, 0, f'{completed}/{total} ({100*completed/total:.1f}%)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.legend(loc='upper right')


def plot_gp1_vs_actual_profile(ax, results):
    """Plot GP1 predictions vs actual energies along the path."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP1 vs Actual Energy Profiles')
        return
        
    # Get mean energies for each image
    actual_mean_energies = []
    gp1_mean_energies = []
    
    for stats in results['per_image_stats']:
        if 'actual_energies' in stats:
            actual_mean = np.mean(stats['actual_energies'])
            actual_mean_energies.append(actual_mean)
        
        if 'pred_energies' in stats:
            pred_mean = np.mean(stats['pred_energies'])
            gp1_mean_energies.append(pred_mean)
    
    if not actual_mean_energies:
        ax.text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP1 vs Actual Energy Profiles')
        return
        
    actual_mean_energies = np.array(actual_mean_energies)
    gp1_mean_energies = np.array(gp1_mean_energies)
    
    # Normalize to start at zero
    actual_mean_energies -= actual_mean_energies[0]
    gp1_mean_energies -= gp1_mean_energies[0]
    
    # Create reaction coordinate
    reaction_coord = np.linspace(0, 1, len(actual_mean_energies))
    
    # Plot both profiles
    ax.plot(reaction_coord, actual_mean_energies, 'b-', linewidth=2.5, 
            label='Actual (EAM)', marker='o', markersize=6)
    ax.plot(reaction_coord, gp1_mean_energies, 'r--', linewidth=2.5, 
            label='GP1 Prediction', marker='s', markersize=6)
    
    ax.set_xlabel('Reaction Coordinate')
    ax.set_ylabel('Relative Energy (eV)')
    ax.set_title('GP1 vs Actual Energy Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_gp1_energy_landscape(ax, results):
    """Plot energy landscape with thermal spread."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Energy Landscape')
        return
        
    # Get mean energies for each image
    mean_energies = []
    thermal_spreads = []
    
    for stats in results['per_image_stats']:
        if 'actual_energies' in stats:
            mean_e = np.mean(stats['actual_energies'])
            mean_energies.append(mean_e)
        
        if 'raw_energy_std' in stats:
            thermal_spreads.append(stats['raw_energy_std'])
    
    if not mean_energies:
        ax.text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Energy Landscape')
        return
        
    mean_energies = np.array(mean_energies)
    mean_energies -= mean_energies[0]
    
    reaction_coord = np.linspace(0, 1, len(mean_energies))
    
    ax.plot(reaction_coord, mean_energies, 'b-', linewidth=2)
    
    # Add thermal spread if available
    if thermal_spreads and len(thermal_spreads) == len(mean_energies):
        upper = mean_energies + np.array(thermal_spreads)
        lower = mean_energies - np.array(thermal_spreads)
        ax.fill_between(reaction_coord, lower, upper, alpha=0.3, label='Thermal spread (±σ)')
    
    ax.set_xlabel('Reaction Coordinate')
    ax.set_ylabel('Relative Energy (eV)')
    ax.set_title('Energy Profile Along Path')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_gp1_energy_uncertainty(ax, results):
    """Plot GP1 energy profile with uncertainty bands."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP1 Energy Uncertainty')
        return
        
    gp1_mean_energies = []
    gp1_std_energies = []
    
    for stats in results['per_image_stats']:
        if 'pred_energies' in stats:
            pred_mean = np.mean(stats['pred_energies'])
            gp1_mean_energies.append(pred_mean)
        
        if 'pred_energy_stds' in stats:
            pred_std = np.mean(stats['pred_energy_stds'])
            gp1_std_energies.append(pred_std)
    
    if not gp1_mean_energies:
        ax.text(0.5, 0.5, 'No GP1 predictions', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP1 Energy Uncertainty')
        return
        
    gp1_mean_energies = np.array(gp1_mean_energies)
    gp1_mean_energies -= gp1_mean_energies[0]
    
    reaction_coord = np.linspace(0, 1, len(gp1_mean_energies))
    
    ax.plot(reaction_coord, gp1_mean_energies, 'g-', linewidth=2.5, label='GP1 Mean')
    
    if gp1_std_energies and len(gp1_std_energies) == len(gp1_mean_energies):
        gp1_std_energies = np.array(gp1_std_energies)
        ax.fill_between(reaction_coord, 
                        gp1_mean_energies - 2*gp1_std_energies,
                        gp1_mean_energies + 2*gp1_std_energies,
                        alpha=0.3, color='green', label='±2σ')
    
    ax.set_xlabel('Reaction Coordinate')
    ax.set_ylabel('Relative Energy (eV)')
    ax.set_title('GP1 Energy Profile with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_gp1_raw_statistics(ax, results):
    """Plot raw data statistics along path."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Raw Data Statistics')
        return
        
    indices = []
    energy_stds = []
    force_stds = []
    
    for i, stats in enumerate(results['per_image_stats']):
        indices.append(i)
        energy_stds.append(stats.get('raw_energy_std', 0))
        force_stds.append(stats.get('raw_force_std', 0))
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(indices, energy_stds, 'b-', label='Energy Std', linewidth=2)
    line2 = ax2.plot(indices, force_stds, 'r-', label='Force Std', linewidth=2)
    
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Energy Std Dev (eV)', color='b')
    ax2.set_ylabel('Force Std Dev (eV/Å)', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    
    ax.set_title('Raw Data Variability Along Path')
    ax.grid(True, alpha=0.3)


def plot_gp1_thermal_noise(ax, results):
    """Plot thermal noise estimates."""
    if 'thermal_noises' not in results or not results['thermal_noises']:
        ax.text(0.5, 0.5, 'No thermal noise data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Thermal Noise Estimates')
        return
        
    force_noises = [n[0] for n in results['thermal_noises']]
    energy_noises = [n[1] for n in results['thermal_noises']]
    
    indices = list(range(len(force_noises)))
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(indices, energy_noises, 'bo-', label='Energy Noise', markersize=6)
    line2 = ax2.plot(indices, force_noises, 'ro-', label='Force Noise', markersize=6)
    
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Energy Noise (eV)', color='b')
    ax2.set_ylabel('Force Noise (eV/Å)', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    
    ax.set_title('Thermal Noise Estimates')
    ax.grid(True, alpha=0.3)


def plot_gp1_system_info(ax, checkpoint_data, results):
    """Plot system information."""
    ax.axis('off')
    
    # Format numerical values with fallback to N/A
    def format_value(value, fmt=".6f"):
        if isinstance(value, (int, float)):
            return f"{value:{fmt}}"
        return str(value)
    
    avg_energy_error = format_value(results.get('avg_energy_error', 'N/A'), '.6f')
    avg_force_error = format_value(results.get('avg_force_error', 'N/A'), '.6f')
    avg_energy_sigma_mad = format_value(results.get('avg_energy_sigma_mad_ratio', 'N/A'), '.3f')
    avg_force_sigma_mad = format_value(results.get('avg_force_sigma_mad_ratio', 'N/A'), '.3f')
    avg_force_noise = format_value(results.get('avg_force_noise', 'N/A'), '.6f')
    avg_energy_noise = format_value(results.get('avg_energy_noise', 'N/A'), '.6f')
    
    info_text = f"""GP1 Path Analysis Summary
{'='*40}

Images: {checkpoint_data.get('completed_images', 0)}/{checkpoint_data.get('n_images', 'N/A')}
Temperature: {checkpoint_data.get('temperature', 'N/A')} K
Snapshots per image: {results.get('n_snapshots', 'N/A')}

Average Errors:
  Energy: {avg_energy_error} eV
  Force: {avg_force_error} eV/Å

Uncertainty (σ/MAD):
  Energy: {avg_energy_sigma_mad}
  Force: {avg_force_sigma_mad}

Thermal Noise:
  Force: {avg_force_noise} eV/Å
  Energy: {avg_energy_noise} eV
"""
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.9))


def plot_gp1_error_distribution(ax, results):
    """Plot error distribution for selected images."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error Distribution')
        return
        
    n_images = len(results['per_image_stats'])
    selected_indices = [0, n_images//4, n_images//2, 3*n_images//4, n_images-1]
    
    energy_errors_by_image = []
    labels = []
    
    for idx in selected_indices:
        if idx < len(results['per_image_stats']):
            stats = results['per_image_stats'][idx]
            if 'pred_energies' in stats and 'actual_energies' in stats:
                pred = np.array(stats['pred_energies'])
                actual = np.array(stats['actual_energies'])
                if len(pred) > 0 and len(actual) > 0:
                    errors = pred - actual
                    energy_errors_by_image.append(errors)
                    labels.append(f'Img {idx}')
    
    # Check if we have any non-empty error data
    if not energy_errors_by_image or all(len(e) == 0 for e in energy_errors_by_image):
        ax.text(0.5, 0.5, 'No error data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error Distribution')
        return
        
    parts = ax.violinplot(energy_errors_by_image, positions=range(len(energy_errors_by_image)),
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Image')
    ax.set_ylabel('Energy Error (eV)')
    ax.set_title('Error Distribution for Selected Images')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')


def plot_gp1_uncertainty_calibration(ax, results):
    """Plot uncertainty calibration."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Uncertainty Calibration')
        return
        
    energy_errors = []
    energy_stds = []
    
    for stats in results['per_image_stats']:
        if 'pred_energies' in stats and 'actual_energies' in stats and 'pred_energy_stds' in stats:
            pred = stats['pred_energies']
            actual = stats['actual_energies']
            stds = stats['pred_energy_stds']
            
            errors = np.abs(np.array(pred) - np.array(actual))
            energy_errors.extend(errors)
            energy_stds.extend(stds)
    
    if not energy_errors:
        ax.text(0.5, 0.5, 'No uncertainty data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Uncertainty Calibration')
        return
        
    energy_errors = np.array(energy_errors)
    energy_stds = np.array(energy_stds)
    
    # Bin by predicted uncertainty
    n_bins = 10
    bin_edges = np.linspace(0, np.max(energy_stds), n_bins + 1)
    actual_coverage = []
    predicted_coverage = []
    
    for i in range(n_bins):
        mask = (energy_stds >= bin_edges[i]) & (energy_stds < bin_edges[i+1])
        if np.sum(mask) > 0:
            # Actual fraction within 1σ
            actual_frac = np.mean(energy_errors[mask] <= energy_stds[mask])
            actual_coverage.append(actual_frac)
            predicted_coverage.append(0.683)  # 1σ coverage
    
    if actual_coverage:
        ax.plot(predicted_coverage, actual_coverage, 'bo-', markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.set_xlabel('Predicted Coverage')
        ax.set_ylabel('Actual Coverage')
        ax.set_title('Uncertainty Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Uncertainty Calibration')


def plot_gp1_path_visualization(ax, checkpoint_data):
    """Visualize the path if position data is available."""
    if 'path_positions' not in checkpoint_data:
        ax.text(0.5, 0.5, 'No path data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Path Visualization')
        return
        
    path_positions = checkpoint_data['path_positions']
    n_images = len(path_positions)
    
    # Simple visualization: show displacement from first image
    displacements = []
    for i in range(n_images):
        disp = np.linalg.norm(path_positions[i] - path_positions[0])
        displacements.append(disp)
    
    ax.plot(range(n_images), displacements, 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Displacement from Initial (Å)')
    ax.set_title('Path Displacement Profile')
    ax.grid(True, alpha=0.3)
    
    # Mark endpoints
    ax.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Initial')
    ax.axvline(x=n_images-1, color='r', linestyle='--', alpha=0.5, label='Final')
    ax.legend()


def plot_gp1_prediction_uncertainties(ax, results):
    """Plot standard deviation of GP1 predictions along the path."""
    if 'per_image_stats' not in results or not results['per_image_stats']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP1 Prediction Variability')
        return
        
    stats = results['per_image_stats']
    indices = [s.get('image_index', i) for i, s in enumerate(stats)]
    
    # Calculate std dev of GP1 predictions for each image
    gp1_energy_stds = []
    gp1_force_stds = []
    
    for stat in stats:
        # Std dev of GP1 energy predictions across snapshots
        if 'pred_energies' in stat and len(stat.get('pred_energies', [])) > 1:
            energy_std = np.std(stat['pred_energies'])
            gp1_energy_stds.append(energy_std)
        else:
            gp1_energy_stds.append(0)
        
        # Std dev of GP1 force predictions across snapshots
        if 'pred_forces' in stat and stat['pred_forces']:
            # Flatten all force components from all snapshots
            all_force_components = []
            for force_pred in stat['pred_forces']:
                all_force_components.extend(force_pred.flatten())
            force_std = np.std(all_force_components) if all_force_components else 0
            gp1_force_stds.append(force_std)
        else:
            gp1_force_stds.append(0)
    
    if not gp1_energy_stds or all(x == 0 for x in gp1_energy_stds):
        ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GP1 Prediction Variability')
        return
        
    # Create twin axis for force
    ax2 = ax.twinx()
    
    # Plot standard deviations
    line1 = ax.plot(indices, gp1_energy_stds, 'g-', linewidth=2.5, 
                    marker='o', markersize=6, label='Energy Std (GP1 Predictions)')
    line2 = ax2.plot(indices, gp1_force_stds, 'orange', linewidth=2.5, 
                     marker='s', markersize=6, label='Force Std (GP1 Predictions)')
    
    # Labels and formatting
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Energy Std Dev (eV)', color='g')
    ax2.set_ylabel('Force Std Dev (eV/Å)', color='orange')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    
    ax.set_title('GP1 Prediction Variability Along Path')
    ax.grid(True, alpha=0.3)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Plot checkpoint data from walker runs')
    parser.add_argument('checkpoint_file', nargs='?', help='Path to checkpoint file')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots (default: current run plots directory)')
    parser.add_argument('--run-dir', default=None,
                        help='Specific run directory to analyze (e.g., outputs/run_20240115_143022)')
    parser.add_argument('--latest', action='store_true',
                        help='Analyze the latest run (uses outputs/latest symlink)')
    parser.add_argument('--output-subdir', default='analysis',
                        help='Subdirectory within run directory to save plots')
    args = parser.parse_args()
    
    # Initialize output directory as None - will be determined later
    output_dir = None
    
    # Determine output directory based on arguments (but defer creation for auto-detection)
    if args.latest:
        # Use latest run
        latest_link = Path("outputs/latest")
        if latest_link.exists():
            run_dir = latest_link.resolve()
            output_dir = str(run_dir / args.output_subdir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            raise FileNotFoundError("No latest run found. Run a calculation first.")
    elif args.run_dir:
        # Use specific run directory
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
        output_dir = str(run_dir / args.output_subdir)
        os.makedirs(output_dir, exist_ok=True)
    elif args.checkpoint_file:
        # Specific checkpoint file provided - infer output directory
        if args.output_dir is None:
            checkpoint_path = Path(args.checkpoint_file)
            # Look for pattern like outputs/run_*/checkpoints/file.pkl
            if 'outputs' in checkpoint_path.parts and 'checkpoints' in checkpoint_path.parts:
                run_dir = checkpoint_path.parent.parent  # Go up from checkpoints to run dir
                output_dir = str(run_dir / 'plots')
                os.makedirs(output_dir, exist_ok=True)
            else:
                # Fallback to current directory
                output_dir = 'plots'
                os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
    elif args.output_dir:
        # Output directory specified but no checkpoint
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    # If none of the above, output_dir remains None and will be set during auto-detection

    if output_dir:
        print(f"Output directory: {output_dir}")
    
    # Find and load checkpoint
    checkpoint_files = []
    if args.checkpoint_file:
        checkpoint_files = [args.checkpoint_file]
    else:
        # Auto-detect checkpoint files
        print("No checkpoint file specified. Searching for checkpoints...")
        
        # Auto-detect: look in latest run first, then current directory
        search_dirs = []
        
        # Try to find latest run
        try:
            # Look for outputs directory - check current directory and parent directory
            possible_outputs = [Path("outputs"), Path("../outputs")]
            outputs_dir = None
            
            for out_path in possible_outputs:
                if out_path.exists():
                    outputs_dir = out_path
                    break
            
            if outputs_dir:
                print(f"Found outputs directory: {outputs_dir}")
                
                # Look for outputs/latest symlink first
                latest_link = outputs_dir / "latest"
                if latest_link.exists() and latest_link.is_symlink():
                    latest_run = latest_link.resolve()
                    search_dirs.append(latest_run)
                    print(f"Found latest run: {latest_run}")
                else:
                    # Fall back to finding newest run directory
                    run_dirs = [d for d in outputs_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('run_')]
                    if run_dirs:
                        # Sort by modification time, newest first
                        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                        search_dirs.append(latest_run)
                        print(f"Found newest run: {latest_run}")
            else:
                print("No outputs directory found in current or parent directory")
        except Exception as e:
            print(f"Could not find latest run: {e}")
        
        # Also search current directory as fallback
        search_dirs.append(Path("."))
        
        # Define checkpoint patterns (in order of preference)
        checkpoint_patterns = [
            # Latest checkpoints (preferred)
            'dual_gp_latest.pkl',
            'gp2_dimer_latest.pkl', 
            'pure_dimer_latest.pkl',
            'pure_neb_latest.pkl',
            'gp2_neb_latest.pkl',
            'minimizer_latest.pkl',
            'walker_checkpoint_latest.pkl',
            'checkpoint_latest.pkl',
            # Final checkpoints
            'dual_gp_final.pkl',
            'gp2_dimer_final.pkl',
            'pure_dimer_final.pkl', 
            'pure_neb_final.pkl',
            'gp2_neb_final.pkl',
            'minimizer_final.pkl',
            'walker_checkpoint_final.pkl',
            'checkpoint_final.pkl',
        ]
        
        # Search for checkpoints
        found_checkpoints = []
        for search_dir in search_dirs:
            checkpoints_dir = search_dir / 'checkpoints'
            if checkpoints_dir.exists():
                print(f"Searching in: {checkpoints_dir}")
                
                # Look for pattern-based checkpoints first
                for pattern in checkpoint_patterns:
                    checkpoint_path = checkpoints_dir / pattern
                    if checkpoint_path.exists():
                        found_checkpoints.append(str(checkpoint_path))
                        print(f"  Found: {pattern}")
                
                # If no pattern matches, look for any .pkl files in checkpoints/
                if not found_checkpoints:
                    pkl_files = list(checkpoints_dir.glob("*.pkl"))
                    if pkl_files:
                        # Sort by modification time, newest first
                        pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        for pkl_file in pkl_files[:3]:  # Show top 3
                            found_checkpoints.append(str(pkl_file))
                            print(f"  Found: {pkl_file.name}")
                
                # If we found checkpoints in this directory, stop searching
                if found_checkpoints:
                    break
        
        checkpoint_files = found_checkpoints
    
    if not checkpoint_files:
        print("No checkpoint file found!")
        print("Please either:")
        print("  1. Specify a checkpoint file: python plot_checkpoint.py /path/to/checkpoint.pkl")
        print("  2. Run a calculation first to create checkpoints")
        print("  3. Use --latest or --run-dir to specify where to look")
        return
    
    checkpoint_file = checkpoint_files[0]
    print(f"Loading checkpoint from: {checkpoint_file}")
    
    # Set output directory if not already set
    if output_dir is None:
        checkpoint_path = Path(checkpoint_file)
        if 'outputs' in checkpoint_path.parts and 'checkpoints' in checkpoint_path.parts:
            # Set output directory to be in the same run as the checkpoint
            run_dir = checkpoint_path.parent.parent  # Go up from checkpoints to run dir
            output_dir = str(run_dir / 'plots')
            os.makedirs(output_dir, exist_ok=True)
            print(f"Auto-setting output directory to: {output_dir}")
        else:
            # Fallback to current directory
            output_dir = 'plots'
            os.makedirs(output_dir, exist_ok=True)
            print(f"Using fallback output directory: {output_dir}")
    
    try:
        checkpoint_data = load_checkpoint(checkpoint_file)
        walker_type = detect_walker_type(checkpoint_data)
        
        print_summary(checkpoint_data)
        
        # Create appropriate plots based on walker type
        if walker_type == 'WalkerDualGP':
            plot_dual_gp_analysis(checkpoint_data, output_dir)
        elif walker_type == 'WalkerGP2Dimer':
            plot_gp2_dimer_analysis(checkpoint_data, output_dir)
        elif walker_type == 'WalkerPureDimer':
            plot_pure_dimer_analysis(checkpoint_data, output_dir)
        elif walker_type == 'WalkerMinimizer':
            plot_minimizer_analysis(checkpoint_data, output_dir)
        elif walker_type == 'WalkerPureNEB':
            plot_pure_neb_analysis(checkpoint_data, output_dir)
        elif walker_type == 'WalkerGP2NEB':
            plot_gp2_neb_analysis(checkpoint_data, output_dir)
        elif walker_type == 'WalkerGP1Path':
            plot_gp1_path_analysis(checkpoint_data, output_dir)
        else:
            print(f"\nPlotting for {walker_type} not fully implemented yet.")
            print("Basic summary has been printed above.")
        
        print(f"Analysis plots saved to: {output_dir}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# Keep all the existing plotting functions (plot_outer_convergence, plot_inner_iterations, etc.)
# They work for both WalkerGP2Dimer and WalkerDualGP

if __name__ == "__main__":
    main()