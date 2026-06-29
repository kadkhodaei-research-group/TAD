#!/usr/bin/env python
"""
Visualize NEB results from Walker checkpoint format.
Handles the E_R_acc format found in walker checkpoints.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import gc

def find_latest_checkpoint():
    """Find the latest checkpoint file."""
    outputs_dir = Path("../outputs")
    if not outputs_dir.exists():
        outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("No outputs directory found!")
        return None
        
    run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        print("No run directories found!")
        return None
        
    latest_run = run_dirs[-1]
    print(f"Using latest run directory: {latest_run}")
    
    checkpoint_path = latest_run / "checkpoints" / "pure_neb_latest.pkl"
    
    if checkpoint_path.exists():
        return checkpoint_path
    else:
        # Try other names
        checkpoint_dir = latest_run / "checkpoints"
        pkl_files = list(checkpoint_dir.glob("*.pkl"))
        if pkl_files:
            pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return pkl_files[0]
            
    return None

def load_walker_checkpoint(checkpoint_path, last_n_steps=None):
    """Load walker checkpoint efficiently."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
        
    # Extract key information
    n_images = data.get('n_images', 50)
    n_iterations = data.get('iteration', 0)
    
    print(f"\nCheckpoint info:")
    print(f"  Walker type: {data.get('walker_type', 'Unknown')}")
    print(f"  Number of images: {n_images}")
    print(f"  Total iterations: {n_iterations}")
    print(f"  Converged: {data.get('converged', False)}")
    
    # Extract energy data
    E_R_acc = data.get('E_R_acc', None)
    
    if E_R_acc is None:
        print("ERROR: No E_R_acc data found!")
        return None
        
    print(f"  Energy data shape: {E_R_acc.shape}")
    
    # Convert to energy profiles over iterations
    # E_R_acc is (n_images, n_iterations)
    if last_n_steps:
        E_R_acc = E_R_acc[:, -last_n_steps:]
        start_iter = n_iterations - last_n_steps
    else:
        start_iter = 0
        
    # Transpose to get (n_iterations, n_images)
    energy_profiles = E_R_acc.T
    
    # Get reference energy
    energy_reference = data.get('energy_reference', 0.0)
    
    # Create result dictionary
    result = {
        'energy_profiles': energy_profiles,
        'n_images': n_images,
        'n_iterations': n_iterations,
        'start_iter': start_iter,
        'energy_reference': energy_reference,
        'converged': data.get('converged', False),
        'force_norms': data.get('normF_R_acc', None),
        'table_history': data.get('table_history', None)
    }
    
    # Clean up
    del data
    gc.collect()
    
    return result

def plot_neb_evolution(result, output_dir):
    """Plot NEB evolution from walker checkpoint."""
    energy_profiles = result['energy_profiles']
    n_images = result['n_images']
    start_iter = result['start_iter']
    energy_reference = result['energy_reference']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nPlotting {len(energy_profiles)} iterations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Energy profile evolution
    ax = axes[0, 0]
    n_profiles = len(energy_profiles)
    
    # Select profiles to show
    if n_profiles <= 20:
        indices = range(n_profiles)
    else:
        # Show ~10 profiles evenly spaced
        indices = np.linspace(0, n_profiles-1, 10, dtype=int)
        
    for i in indices:
        alpha = 0.3 + 0.7 * (i / (n_profiles-1))
        color = plt.cm.viridis(i / (n_profiles-1))
        iter_num = start_iter + i
        ax.plot(range(n_images), energy_profiles[i] - energy_reference, 
                'o-', alpha=alpha, color=color, markersize=3,
                label=f'Iter {iter_num}' if i in [indices[0], indices[-1]] else '')
                
    # Highlight final profile
    final_profile = energy_profiles[-1] - energy_reference
    ax.plot(range(n_images), final_profile, 'ko-', linewidth=2.5, 
            markersize=7, label='Final', zorder=10)
            
    ax.set_xlabel('Image Index', fontsize=12)
    ax.set_ylabel('Energy - E_ref (eV)', fontsize=12)
    ax.set_title('NEB Energy Profile Evolution', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # 2. Final energy profile with barriers
    ax = axes[0, 1]
    ax.plot(range(n_images), final_profile, 'ko-', linewidth=2, markersize=8)
    
    # Calculate barriers
    E_initial = final_profile[0]
    E_final = final_profile[-1]
    E_max = np.max(final_profile)
    E_max_idx = np.argmax(final_profile)
    
    forward_barrier = E_max - E_initial
    reverse_barrier = E_max - E_final
    reaction_energy = E_final - E_initial
    
    # Add annotations
    ax.axhline(y=E_initial, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=E_final, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=E_max, color='green', linestyle='--', alpha=0.5)
    
    # Add text box
    textstr = f'Forward: {forward_barrier:.3f} eV\nReverse: {reverse_barrier:.3f} eV\nΔE: {reaction_energy:.3f} eV'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    ax.set_xlabel('Image Index', fontsize=12)
    ax.set_ylabel('Energy - E_ref (eV)', fontsize=12)
    ax.set_title(f'Final NEB Profile (TS at image {E_max_idx})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 3. Barrier convergence
    ax = axes[1, 0]
    
    # Calculate barrier for each iteration
    forward_barriers = []
    reverse_barriers = []
    
    for profile in energy_profiles[::max(1, len(energy_profiles)//100)]:  # Sample for speed
        profile_rel = profile - energy_reference
        E_init = profile_rel[0]
        E_fin = profile_rel[-1]
        E_mx = np.max(profile_rel)
        forward_barriers.append(E_mx - E_init)
        reverse_barriers.append(E_mx - E_fin)
        
    iterations = np.linspace(start_iter, start_iter + n_profiles - 1, len(forward_barriers))
    
    ax.plot(iterations, forward_barriers, 'b-', label='Forward barrier')
    ax.plot(iterations, reverse_barriers, 'r-', label='Reverse barrier')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Barrier Height (eV)', fontsize=12)
    ax.set_title('Barrier Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Force convergence (if available)
    ax = axes[1, 1]
    
    if result.get('force_norms') is not None:
        force_norms = result['force_norms']
        # Plot max force norm over iterations
        max_forces = np.max(force_norms, axis=0)
        iterations = range(len(max_forces))
        
        ax.semilogy(iterations[::max(1, len(iterations)//1000)], 
                    max_forces[::max(1, len(iterations)//1000)], 'g-')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Max Force Norm (eV/Å)', fontsize=12)
        ax.set_title('Force Convergence', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
    else:
        ax.text(0.5, 0.5, 'Force data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Force Convergence', fontsize=14)
        
    plt.tight_layout()
    
    # Save plots
    output_file = output_dir / 'neb_walker_analysis.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    
    output_file_png = output_dir / 'neb_walker_analysis.png'
    plt.savefig(output_file_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file_png}")
    
    plt.close()
    
    # Save summary
    summary_file = output_dir / 'neb_walker_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("NEB WALKER CALCULATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total iterations: {result['n_iterations']}\n")
        f.write(f"Number of images: {n_images}\n")
        f.write(f"Converged: {result['converged']}\n")
        f.write(f"Reference energy: {energy_reference:.6f} eV\n\n")
        
        f.write("Final Energy Profile:\n")
        f.write(f"  Initial state: {E_initial:.6f} eV (relative to reference)\n")
        f.write(f"  Final state: {E_final:.6f} eV (relative to reference)\n")
        f.write(f"  Transition state: {E_max:.6f} eV at image {E_max_idx}\n\n")
        
        f.write("Barriers:\n")
        f.write(f"  Forward: {forward_barrier:.3f} eV\n")
        f.write(f"  Reverse: {reverse_barrier:.3f} eV\n")
        f.write(f"  Reaction energy: {reaction_energy:.3f} eV\n")
        
        # Save final profile data
        f.write("\nFinal Profile Data (Image, Energy-Eref):\n")
        for i, E in enumerate(final_profile):
            f.write(f"{i:4d}  {E:12.6f}\n")
            
    print(f"Saved summary to: {summary_file}")
    
    # Also save just the final profile
    np.savetxt(output_dir / 'final_profile.dat', 
               np.column_stack((range(n_images), final_profile)),
               header='Image_Index Energy-Eref(eV)', fmt='%4d %12.6f')
    print(f"Saved final profile data to: {output_dir}/final_profile.dat")
    
    return forward_barrier, reverse_barrier, reaction_energy

def main():
    parser = argparse.ArgumentParser(description='Visualize NEB walker checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--last-n-steps', type=int, 
                       help='Only plot last N iterations (for large files)')
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint()
        
    if not checkpoint_path or not checkpoint_path.exists():
        print("ERROR: Could not find checkpoint file!")
        return
        
    # Load data
    result = load_walker_checkpoint(checkpoint_path, args.last_n_steps)
    
    if result is None:
        print("ERROR: Could not load checkpoint data!")
        return
        
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / 'neb_visualization'
        
    # Create plots
    forward, reverse, reaction = plot_neb_evolution(result, output_dir)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL NEB RESULTS")
    print("="*60)
    print(f"Forward barrier:  {forward:.3f} eV")
    print(f"Reverse barrier:  {reverse:.3f} eV") 
    print(f"Reaction energy:  {reaction:.3f} eV")
    print("="*60)

if __name__ == "__main__":
    main()