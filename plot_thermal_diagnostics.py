#!/usr/bin/env python
"""Plot thermal noise diagnostics from GP1 analysis."""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.gridspec import GridSpec
from pathlib import Path
from output_manager import get_output_path

def load_diagnostics(filename='thermal_diagnostics.json'):
    """Load diagnostic data from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_thermal_evolution(data, output_file='thermal_diagnostics.png'):
    """Create comprehensive diagnostic plots."""
    # For plotting, we'll use the index as x-axis if dimer_step is not available
    valid_data = []
    for i, d in enumerate(data):
        dimer_step = d.get('dimer_step')
        # Handle None, null, or missing dimer_step
        if dimer_step is not None and dimer_step >= 0:
            x_value = dimer_step
        else:
            x_value = i  # Use index if no valid dimer step
        valid_data.append((x_value, d))
    
    if not valid_data:
        print("No data to plot!")
        return
    
    # Sort by x_value
    valid_data.sort(key=lambda x: x[0])
    x_values = [x[0] for x in valid_data]
    data_points = [x[1] for x in valid_data]
    
    # Energy statistics
    e_means = [d['energy_stats']['mean'] for d in data_points]
    e_stds = [d['energy_stats']['std'] for d in data_points]
    e_mads = [d['energy_stats']['mad'] for d in data_points]
    e_outliers = [d['energy_stats']['outlier_fraction'] * 100 for d in data_points]
    e_kurtosis = [d['energy_stats']['kurtosis'] for d in data_points]
    
    # Force statistics
    f_stds = [d['force_stats']['std'] for d in data_points]
    f_outliers = [d['force_stats']['outlier_fraction'] * 100 for d in data_points]
    
    # Suggested df
    suggested_dfs = [d['suggested_df'] for d in data_points]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Determine x-axis label - check if any valid dimer steps exist
    has_valid_dimer_steps = any(d.get('dimer_step') is not None and d.get('dimer_step') >= 0 for d in data)
    xlabel = 'Dimer Step' if has_valid_dimer_steps else 'Evaluation Index'
    
    # 1. Energy standard deviation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_values, e_stds, 'b-o', label='Std Dev', markersize=6)
    ax1.plot(x_values, e_mads, 'g--s', label='MAD', markersize=6)
    ax1.set_ylabel('Energy Spread (eV)')
    ax1.set_title('Energy Distribution Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Force standard deviation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_values, f_stds, 'r-o', markersize=6)
    ax2.set_ylabel('Force Std Dev (eV/Å)')
    ax2.set_title('Force Distribution Spread')
    ax2.grid(True, alpha=0.3)
    
    # 3. Outlier fraction
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x_values, e_outliers, 'b-o', label='Energy', markersize=6)
    ax3.plot(x_values, f_outliers, 'r--s', label='Force', markersize=6)
    ax3.set_ylabel('Outlier Fraction (%)')
    ax3.set_title('Outlier Detection (z-score > 3)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Kurtosis (tail heaviness)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x_values, e_kurtosis, 'm-o', markersize=6)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Normal')
    ax4.set_ylabel('Excess Kurtosis')
    ax4.set_title('Tail Heaviness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Std/MAD ratio
    ax5 = fig.add_subplot(gs[2, 0])
    std_mad_ratios = [s/m if m > 0 else 0 for s, m in zip(e_stds, e_mads)]
    ax5.plot(x_values, std_mad_ratios, 'c-o', markersize=6)
    ax5.axhline(y=1.48, color='k', linestyle='--', alpha=0.5, label='Normal')
    ax5.set_ylabel('Std/MAD Ratio')
    ax5.set_title('Robustness Indicator')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Suggested Student-t df
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(x_values, suggested_dfs, 'g-o', markersize=6)
    ax6.set_ylabel('Suggested df')
    ax6.set_title('Recommended Student-t Parameter')
    ax6.set_ylim(0, 6)
    ax6.grid(True, alpha=0.3)
    
    # 7. Energy mean evolution
    ax7 = fig.add_subplot(gs[3, :])
    ax7.plot(x_values, e_means, 'k-o', markersize=6)
    ax7.set_xlabel(xlabel)
    ax7.set_ylabel('Mean Energy (eV)')
    ax7.set_title('Energy Evolution')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Thermal Snapshot Distribution Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Also create a simple comparison plot
    comparison_file = os.path.join(os.path.dirname(output_file), 'thermal_comparison.png')
    create_comparison_plot(x_values, e_stds, xlabel, comparison_file)

def create_comparison_plot(x_values, e_stds, xlabel, output_file='thermal_comparison.png'):
    """Create a simple comparison plot showing key metrics."""
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, e_stds, 'o-', linewidth=2, markersize=8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Energy Std Dev (eV)', fontsize=12)
    plt.title('Energy Variation in Thermal Snapshots', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved to {output_file}")

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("THERMAL DIAGNOSTICS SUMMARY")
    print("="*60)
    
    if not data:
        print("No data found.")
        return
    
    print(f"\nTotal evaluations analyzed: {len(data)}")
    
    # Find entry with highest std dev
    e_stds = [d['energy_stats']['std'] for d in data]
    max_std_idx = np.argmax(e_stds)
    max_std_entry = data[max_std_idx]
    
    print(f"Maximum energy std dev: {e_stds[max_std_idx]:.4f} eV")
    dimer_step = max_std_entry.get('dimer_step')
    if dimer_step is not None and dimer_step >= 0:
        print(f"  at dimer step: {dimer_step}")
    print(f"  at evaluation index: {max_std_idx}")
    
    # Average suggested df
    suggested_dfs = [d['suggested_df'] for d in data]
    print(f"\nStudent-t df statistics:")
    print(f"  Average suggested df: {np.mean(suggested_dfs):.2f}")
    print(f"  Range: [{min(suggested_dfs):.1f}, {max(suggested_dfs):.1f}]")
    
    # Outlier statistics
    e_outlier_fracs = [d['energy_stats']['outlier_fraction'] for d in data]
    print(f"\nOutlier statistics:")
    print(f"  Average outlier fraction: {np.mean(e_outlier_fracs)*100:.1f}%")
    print(f"  Maximum outlier fraction: {max(e_outlier_fracs)*100:.1f}%")
    
    # Show trend
    if len(data) > 3:
        first_std = np.mean(e_stds[:3])
        last_std = np.mean(e_stds[-3:])
        print(f"\nTrend analysis:")
        print(f"  Average std dev (first 3): {first_std:.4f} eV")
        print(f"  Average std dev (last 3): {last_std:.4f} eV")
        print(f"  Change: {(last_std/first_std - 1)*100:+.1f}%")
    
    # Energy reference info
    e_means = [d['energy_stats']['mean'] for d in data]
    print(f"\nEnergy reference check:")
    print(f"  First mean energy: {e_means[0]:.2f} eV")
    print(f"  Last mean energy: {e_means[-1]:.2f} eV")
    if abs(e_means[0]) > 1000:
        print("  Note: First energy appears to be unreferenced (raw DFT energy)")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Plot thermal noise diagnostics')
    parser.add_argument('--input', default='thermal_diagnostics.json', 
                        help='Input JSON file with diagnostic data')
    parser.add_argument('--output', default='thermal_diagnostics.png',
                        help='Output plot filename')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics')
    parser.add_argument('--run-dir', default=None,
                        help='Specific run directory to analyze (e.g., outputs/run_20240115_143022)')
    parser.add_argument('--latest', action='store_true',
                        help='Analyze the latest run (uses outputs/latest symlink)')
    parser.add_argument('--output-subdir', default='analysis',
                        help='Subdirectory within run directory to save plots')
    
    args = parser.parse_args()
    
    # Determine output directory based on arguments
    if args.latest:
        # Use latest run
        outputs_base = os.path.join('..', 'outputs') if os.path.basename(os.getcwd()) == 'scripts' else 'outputs'
        latest_link = Path(outputs_base) / "latest"
        if latest_link.exists():
            run_dir = latest_link.resolve()
            output_dir = str(run_dir / args.output_subdir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Try to find latest run manually
            try:
                from output_manager import get_latest_run_dir
                run_dir = Path(get_latest_run_dir())
                output_dir = str(run_dir / args.output_subdir)
                os.makedirs(output_dir, exist_ok=True)
            except:
                raise FileNotFoundError("No latest run found. Run a calculation first.")
    elif args.run_dir:
        # Use specific run directory
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
        output_dir = str(run_dir / args.output_subdir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Default behavior - look for thermal diagnostics in the latest run
        outputs_base = os.path.join('..', 'outputs') if os.path.basename(os.getcwd()) == 'scripts' else 'outputs'
        
        # Try to find the latest run
        latest_link = Path(outputs_base) / "latest"
        if latest_link.exists() and latest_link.is_symlink():
            run_dir = latest_link.resolve()
            output_dir = str(run_dir / "thermal_diagnostics_plots")
        else:
            # Try to use get_latest_run_dir function
            try:
                from output_manager import get_latest_run_dir
                run_dir = Path(get_latest_run_dir())
                output_dir = str(run_dir / "thermal_diagnostics_plots")
            except:
                # Fallback to old behavior
                output_dir = os.path.join(outputs_base, 'thermal_diagnostics_plots')
        
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    
    # Determine input and output file paths
    if args.latest or args.run_dir:
        # Look for thermal diagnostics in the run directory
        input_file = str(Path(output_dir).parent / 'thermal_diagnostics.json')
        if not os.path.exists(input_file):
            input_file = args.input  # Fall back to specified input
    else:
        # For default behavior, also look in the run directory
        if 'run_' in str(output_dir):
            # We found a run directory, look for thermal diagnostics there
            input_file = str(Path(output_dir).parent / 'thermal_diagnostics.json')
            if not os.path.exists(input_file):
                input_file = args.input  # Fall back to specified input
        else:
            input_file = args.input
    
    output_file = os.path.join(output_dir, 'thermal_diagnostics_plot.png')
    
    # Load data
    try:
        data = load_diagnostics(input_file)
        print(f"Loaded {len(data)} diagnostic entries from {input_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return
    
    # Create plots
    plot_thermal_evolution(data, output_file)
    
    # Print summary
    print_summary(data)
    
    print(f"Analysis plots saved to: {output_dir}")

if __name__ == "__main__":
    main()