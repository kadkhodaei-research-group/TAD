#!/usr/bin/env python
"""
Simple script to run existing plotting scripts on all experimental runs.
Uses plot_checkpoint.py and plot_thermal_diagnostics.py for each run.
"""

import os
import subprocess
import shutil
from pathlib import Path
from output_manager import get_output_path

# Results directory - use output manager for proper organization
try:
    from output_manager import OutputManager
    try:
        get_output_path('test')
    except RuntimeError:
        OutputManager.setup()
    RESULTS_BASE_DIR = get_output_path('dual_gp_results')
except:
    # Fallback for backward compatibility
    RESULTS_BASE_DIR = 'dual_gp_results'

def find_experiment_folders():
    """Find all experiment folders in results directory."""
    if not os.path.exists(RESULTS_BASE_DIR):
        print(f"Error: Results directory '{RESULTS_BASE_DIR}' not found!")
        return []
    
    # Get all timestamped folders
    exp_folders = []
    for folder in os.listdir(RESULTS_BASE_DIR):
        folder_path = os.path.join(RESULTS_BASE_DIR, folder)
        if os.path.isdir(folder_path) and folder != 'plots':
            # Check if it has the expected files
            checkpoint_file = os.path.join(folder_path, 'gp2_dimer_latest.pkl')
            thermal_file = os.path.join(folder_path, 'thermal_diagnostics.json')
            
            if os.path.exists(checkpoint_file):
                exp_folders.append(folder_path)
    
    return sorted(exp_folders)

def plot_checkpoint_for_run(exp_folder):
    """Run plot_checkpoint.py for a specific experimental run."""
    checkpoint_file = os.path.join(exp_folder, 'gp2_dimer_latest.pkl')
    
    if not os.path.exists(checkpoint_file):
        print(f"  No checkpoint file found in {exp_folder}")
        return
    
    # Create output directory for this run's plots
    plots_dir = os.path.join(exp_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to experiment folder so plots are saved there
        os.chdir(exp_folder)
        
        # Run plot_checkpoint.py
        cmd = ['python', os.path.join(original_dir, 'plot_checkpoint.py'), 
               'gp2_dimer_latest.pkl']
        
        print(f"  Running plot_checkpoint.py...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  Error running plot_checkpoint.py: {result.stderr}")
        else:
            # Move generated plots to plots subdirectory
            for file in os.listdir('.'):
                if file.endswith('.png') and file.startswith(('dual_gp_analysis', 'gp2_dimer_analysis')):
                    shutil.move(file, os.path.join('plots', file))
                    print(f"    Generated: {file}")
    
    finally:
        # Return to original directory
        os.chdir(original_dir)

def plot_thermal_diagnostics_for_run(exp_folder):
    """Run plot_thermal_diagnostics.py for a specific experimental run."""
    thermal_file = os.path.join(exp_folder, 'thermal_diagnostics.json')
    
    if not os.path.exists(thermal_file):
        print(f"  No thermal diagnostics file found in {exp_folder}")
        return
    
    # Create output directory for this run's plots
    plots_dir = os.path.join(exp_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to experiment folder
        os.chdir(exp_folder)
        
        # Run plot_thermal_diagnostics.py
        cmd = ['python', os.path.join(original_dir, 'plot_thermal_diagnostics.py'), 
               '--input', 'thermal_diagnostics.json',
               '--output', os.path.join('plots', 'thermal_diagnostics.png')]
        
        print(f"  Running plot_thermal_diagnostics.py...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  Error running plot_thermal_diagnostics.py: {result.stderr}")
        else:
            print(f"    Generated: thermal_diagnostics.png")
            
            # Also generate comparison plot
            if os.path.exists(os.path.join('plots', 'thermal_comparison.png')):
                print(f"    Generated: thermal_comparison.png")
    
    finally:
        # Return to original directory
        os.chdir(original_dir)

def create_summary_page():
    """Create a simple HTML page linking to all plots."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Dual GP Experiment Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .experiment { margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; }
        .experiment h2 { margin-top: 0; }
        .plots { display: flex; flex-wrap: wrap; gap: 10px; }
        .plot-link { display: block; margin: 5px 0; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>Dual GP Experiment Results - All Plots</h1>
"""
    
    exp_folders = find_experiment_folders()
    
    for exp_folder in exp_folders:
        folder_name = os.path.basename(exp_folder)
        plots_dir = os.path.join(exp_folder, 'plots')
        
        if os.path.exists(plots_dir):
            html_content += f'<div class="experiment">\n'
            html_content += f'<h2>{folder_name}</h2>\n'
            
            # List all PNG files in plots directory
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            
            if plot_files:
                html_content += '<div class="plots">\n'
                for plot_file in sorted(plot_files):
                    rel_path = os.path.join(folder_name, 'plots', plot_file)
                    html_content += f'<a class="plot-link" href="{rel_path}">{plot_file}</a>\n'
                html_content += '</div>\n'
            else:
                html_content += '<p>No plots found</p>\n'
            
            html_content += '</div>\n'
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    index_file = os.path.join(RESULTS_BASE_DIR, 'index.html')
    with open(index_file, 'w') as f:
        f.write(html_content)
    
    print(f"\nCreated summary page: {index_file}")

def main():
    """Main function to plot all experimental runs."""
    print("="*80)
    print("PLOTTING ALL EXPERIMENTAL RUNS")
    print("="*80)
    
    # Check if required plotting scripts exist
    required_scripts = ['plot_checkpoint.py', 'plot_thermal_diagnostics.py']
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"\nError: Required script '{script}' not found!")
            print("Please ensure the plotting scripts are in the current directory.")
            return
    
    # Find all experiment folders
    exp_folders = find_experiment_folders()
    
    if not exp_folders:
        print("\nNo experiment folders found!")
        return
    
    print(f"\nFound {len(exp_folders)} experiment folders")
    
    # Process each experiment
    for i, exp_folder in enumerate(exp_folders, 1):
        folder_name = os.path.basename(exp_folder)
        print(f"\n[{i}/{len(exp_folders)}] Processing: {folder_name}")
        
        # Run checkpoint plotting
        plot_checkpoint_for_run(exp_folder)
        
        # Run thermal diagnostics plotting
        plot_thermal_diagnostics_for_run(exp_folder)
    
    # Create summary HTML page
    create_summary_page()
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print(f"\nAll plots have been generated in each experiment's 'plots' subdirectory")
    print(f"Open {os.path.join(RESULTS_BASE_DIR, 'index.html')} to browse all plots")

if __name__ == "__main__":
    main()
