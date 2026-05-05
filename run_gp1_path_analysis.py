#!/usr/bin/env python
"""
Main script for GP1 path analysis between local minimum and saddle point.
Creates interpolated images, fits separate GP1 models with thermal snapshots,
and analyzes prediction statistics.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from walker_gp1_path import WalkerGP1Path
from vasp_manager import VASPManager
from vasp_interface import VASPInterface
from atomic_structure import AtomicStructure
from output_manager import OutputManager, get_output_path, get_input_path
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element


def get_mass_vector(structure):
    """Return mass array for all atoms."""
    atom_masses = np.array([
        Element(site.specie.symbol).atomic_mass
        for site in structure
    ], dtype=float)
    return np.repeat(atom_masses, 3)  # Repeat for x,y,z


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    log_file = get_output_path('logs', 'gp1_path_analysis.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> Tuple[Dict, str, str]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GP1 Path Analysis')
    
    # Required positional arguments
    parser.add_argument('local_min_poscar', 
                        help='POSCAR file for local minimum (looked for in inputs/ directory)')
    parser.add_argument('saddle_poscar', 
                        help='POSCAR file for saddle point (looked for in inputs/ directory)')
    
    # Path parameters
    parser.add_argument('--n-images', type=int, default=100,
                        help='Number of interpolated images (default: 100)')
    parser.add_argument('--interpolation-method', choices=['linear', 'idpp'], 
                        default='linear',
                        help='Method for generating initial path (default: linear)')
    
    # System parameters
    parser.add_argument('--neighbor-cutoff', type=float, default=5.0,
                        help='Cutoff distance for nearest neighbors in Angstroms (default: 5.0)')
    parser.add_argument('--max-neighbors', type=int, default=50,
                        help='Maximum number of nearest neighbors to consider (default: 50)')
    parser.add_argument('--activation-radius', type=float, default=10.0,
                        help='Activation radius for identifying atoms near saddle (default: 10.0)')
    parser.add_argument('--energy-shift', type=float, default=0.0,
                        help='Energy shift/reference to apply (default: 0.0)')
    parser.add_argument('--moving-indices', type=int, nargs='+', default=None,
                        help='Indices of atoms allowed to move (0-indexed). If not specified, all atoms can move')
    parser.add_argument('--orient-atom-direction', type=str, default=None,
                        help='Atom index and direction for orientation (format: "atom_idx:x,y,z")')
    
    # Force constants and thermal parameters
    parser.add_argument('--force-constants-file', default='FORCE_CONSTANTS',
                        help='Force constants file for thermal noise calculation (default: FORCE_CONSTANTS)')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature in K (default: 300)')
    parser.add_argument('--num-snapshots', type=int, default=10,
                        help='Number of thermal snapshots per image (default: 10)')
    parser.add_argument('--md-timestep', type=float, default=1.0,
                        help='MD timestep in fs (default: 1.0)')
    parser.add_argument('--md-steps', type=int, default=10,
                        help='MD steps between snapshots (default: 10)')
    
    # GP1 parameters
    parser.add_argument('--gp1-noise-model', 
                        choices=['fixed', 'heteroscedastic', 'student_t'],
                        default='fixed',
                        help='GP1 noise model (default: fixed)')
    parser.add_argument('--gp1-student-t-df', type=float, default=2.0,
                        help='Degrees of freedom for Student-t likelihood (lower = heavier tails)')
    # Adaptive df arguments
    parser.add_argument('--gp1-use-adaptive-df', action='store_true',
                        help='Enable adaptive Student-t df')
    parser.add_argument('--gp1-adaptive-df-start-iter', type=int, default=15,
                        help='Iteration to start df adaptation')
    parser.add_argument('--gp1-adaptive-df-end-iter', type=int, default=25,
                        help='Iteration to reach target df')
    parser.add_argument('--gp1-adaptive-df-target', type=float, default=1.0,
                        help='Target df value for adaptation')
    # Other GP1 arguments
    parser.add_argument('--gp1-remove-outliers', action='store_true',
                        help='Enable outlier removal in thermal snapshots')
    parser.add_argument('--gp1-outlier-threshold', type=float, default=5.0,
                        help='MAD threshold for outlier detection')
    
    # Execution parameters
    parser.add_argument('--execution-mode', choices=['direct', 'queue', 'mock', 'eam'], 
                        default='direct',
                        help='Execution mode for calculations')
    parser.add_argument('--eam-potential-file', default=None,
                        help='EAM potential file for EAM mode')
    parser.add_argument('--parallel-eam', action='store_true',
                        help='Enable parallel execution of EAM calculations')
    parser.add_argument('--eam-n-workers', type=int, default=None,
                        help='Number of parallel workers for EAM (default: number of CPU cores)')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to checkpoint file to resume from')
    
    # Output parameters
    parser.add_argument('--output-dir', default=None,
                        help='Base directory for all outputs (default: ./outputs)')
    parser.add_argument('--run-name', default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--input-dir', default=None,
                        help='Directory containing input files (default: ./inputs)')
    
    # Other parameters
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plotting (useful for HPC runs)')
    
    args = parser.parse_args()
    
    # Create system parameters dictionary
    system_params = {
        'n_images': args.n_images,
        'interpolation_method': args.interpolation_method,
        'neighbor_cutoff': args.neighbor_cutoff,
        'max_neighbors': args.max_neighbors,
        'activation_radius': args.activation_radius,
        'energy_shift': args.energy_shift,
        'force_constants_file': args.force_constants_file,
        'temperature': args.temperature,
        'num_snapshots': args.num_snapshots,
        'md_timestep': args.md_timestep,
        'md_steps': args.md_steps,
        'gp1_noise_model': args.gp1_noise_model,
        'gp1_student_t_df': args.gp1_student_t_df,
        'gp1_use_adaptive_df': args.gp1_use_adaptive_df,
        'gp1_adaptive_df_start_iter': args.gp1_adaptive_df_start_iter,
        'gp1_adaptive_df_end_iter': args.gp1_adaptive_df_end_iter,
        'gp1_adaptive_df_target': args.gp1_adaptive_df_target,
        'gp1_remove_outliers': args.gp1_remove_outliers,
        'gp1_outlier_threshold': args.gp1_outlier_threshold,
        'execution_mode': args.execution_mode,
        'eam_potential_file': args.eam_potential_file,
        'parallel_eam': args.parallel_eam,
        'eam_n_workers': args.eam_n_workers,
        'no_plots': args.no_plots,
        'output_dir': args.output_dir,
        'run_name': args.run_name,
        'input_dir': args.input_dir,
        'moving_indices': args.moving_indices,
        'orient_atom_direction': args.orient_atom_direction,
    }
    
    return system_params, args.local_min_poscar, args.saddle_poscar, args.checkpoint


def load_poscar_structure(poscar_file: str, moving_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict, str]:
    """Load structure from POSCAR file. Returns (positions, structure_info, poscar_path)."""
    # Handle input file path
    if not os.path.isabs(poscar_file):
        poscar_path = get_input_path(poscar_file)
        if not os.path.exists(poscar_path):
            # Backward compatibility - check current directory
            if os.path.exists(poscar_file):
                poscar_path = os.path.abspath(poscar_file)
                logging.warning(f"Found {poscar_file} in current directory")
                logging.warning("Consider moving it to inputs/ directory")
            else:
                # Try parent directory's inputs if running from scripts/
                if os.path.basename(os.getcwd()) == 'scripts':
                    parent_input_path = os.path.join('..', 'inputs', poscar_file)
                    if os.path.exists(parent_input_path):
                        poscar_path = parent_input_path
                    else:
                        raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
                else:
                    raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
    else:
        poscar_path = poscar_file
    
    # Read POSCAR file
    with open(poscar_path, 'r') as f:
        lines = f.readlines()
    
    # Parse lattice vectors
    scale = float(lines[1].strip())
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)])
    lattice *= scale
    
    # Parse atom types and counts
    atom_types = lines[5].split()
    atom_counts = [int(x) for x in lines[6].split()]
    
    # Check for selective dynamics
    selective_dynamics = lines[7].strip()[0].upper() in ['S', 'T']
    coord_start = 8 if selective_dynamics else 7
    
    # Parse positions
    positions = []
    moving_flags = []
    
    for i in range(coord_start + 1, coord_start + 1 + sum(atom_counts)):
        pos_data = lines[i].split()
        positions.append([float(x) for x in pos_data[:3]])
        
        if selective_dynamics:
            # T T T means moving, F F F means frozen
            flags = pos_data[3:6]
            is_moving = all(f.upper() == 'T' for f in flags)
            moving_flags.append(is_moving)
        else:
            # All atoms are moving if no selective dynamics
            moving_flags.append(True)
    
    positions = np.array(positions)
    
    # Convert to Cartesian if needed
    if lines[coord_start].strip()[0].upper() in ['D', 'R']:
        # Direct/fractional coordinates
        positions = positions @ lattice
    
    # Determine moving indices
    if moving_indices is not None:
        # Use specified moving indices
        frozen_indices = [i for i in range(len(positions)) if i not in moving_indices]
    else:
        # Use selective dynamics or all atoms
        moving_indices = [i for i, flag in enumerate(moving_flags) if flag]
        frozen_indices = [i for i, flag in enumerate(moving_flags) if not flag]
    
    structure_info = {
        'lattice': lattice,
        'atom_types': atom_types,
        'atom_counts': atom_counts,
        'n_atoms': sum(atom_counts),
        'moving_indices': moving_indices,
        'frozen_indices': frozen_indices,
        'n_moving': len(moving_indices)
    }
    
    return positions.flatten(), structure_info, poscar_path


def create_interpolated_path(initial_poscar_path: str, final_poscar_path: str, 
                           n_images: int, interpolation_method: str = 'linear') -> List[np.ndarray]:
    """Create interpolated path between two structures using linear or IDPP method.
    
    Args:
        initial_poscar_path: Path to initial POSCAR file
        final_poscar_path: Path to final POSCAR file  
        n_images: Number of images to create (including endpoints)
        interpolation_method: 'linear' or 'idpp'
        
    Returns:
        List of position arrays for each image
    """
    # Read structures using pymatgen
    initial_struct = Poscar.from_file(initial_poscar_path, check_for_potcar=False).structure
    final_struct = Poscar.from_file(final_poscar_path, check_for_potcar=False).structure
    
    # Ensure structures are compatible
    if len(initial_struct) != len(final_struct):
        raise ValueError("Initial and final structures must have the same number of atoms")
    
    # Get fractional coordinates
    initial_frac = initial_struct.frac_coords
    final_frac = final_struct.frac_coords
    n_atoms = len(initial_struct)
    
    # Debug: Check which atoms move significantly
    logging.info(f"\nAnalyzing atomic movements:")
    for i in range(n_atoms):
        # Calculate displacement considering periodic boundary conditions
        frac_diff = final_frac[i] - initial_frac[i]
        # Apply minimum image convention
        frac_diff = frac_diff - np.round(frac_diff)
        cart_diff = initial_struct.lattice.get_cartesian_coords(frac_diff)
        dist = np.linalg.norm(cart_diff)
        if dist > 0.01:  # Only log atoms that move significantly
            logging.info(f"  Atom {i}: moves {dist:.4f} Å")
    
    # Generate path
    path = []
    
    if interpolation_method == 'linear':
        logging.info(f"\nGenerating linear interpolation path with {n_images} images...")
        
        for img in range(n_images):
            t = img / (n_images - 1)
            
            # Interpolate each atom considering periodic boundaries
            frac_coords = np.zeros((n_atoms, 3))
            for atom in range(n_atoms):
                # Calculate displacement in fractional coordinates
                disp = final_frac[atom] - initial_frac[atom]
                
                # Apply minimum image convention - find shortest path
                disp = disp - np.round(disp)
                
                # Interpolate
                frac_coords[atom] = initial_frac[atom] + t * disp
                
                # Wrap back into cell
                frac_coords[atom] = frac_coords[atom] - np.floor(frac_coords[atom])
            
            # Convert to Cartesian and flatten
            cart_coords = initial_struct.lattice.get_cartesian_coords(frac_coords)
            path.append(cart_coords.flatten())
    
    elif interpolation_method == 'idpp':
        # Start with linear interpolation
        logging.info(f"\nGenerating IDPP interpolation (starting with linear)...")
        
        for img in range(n_images):
            t = img / (n_images - 1)
            
            # Interpolate each atom considering periodic boundaries
            frac_coords = np.zeros((n_atoms, 3))
            for atom in range(n_atoms):
                # Calculate displacement in fractional coordinates
                disp = final_frac[atom] - initial_frac[atom]
                
                # Apply minimum image convention
                disp = disp - np.round(disp)
                
                # Interpolate
                frac_coords[atom] = initial_frac[atom] + t * disp
                
                # Wrap back into cell
                frac_coords[atom] = frac_coords[atom] - np.floor(frac_coords[atom])
            
            # Convert to Cartesian and flatten
            cart_coords = initial_struct.lattice.get_cartesian_coords(frac_coords)
            
            # Add small random perturbation to intermediate images for IDPP
            if 0 < img < n_images - 1:
                perturbation = 0.01 * np.random.randn(n_atoms, 3)
                cart_coords += perturbation
            
            path.append(cart_coords.flatten())
    
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    
    # Final check: ensure no atoms are too close together
    logging.info(f"\nChecking for atomic overlaps in interpolated path...")
    min_allowed_dist = 1.5  # Angstroms
    
    for img in range(n_images):
        coords = path[img].reshape(-1, 3)
        min_dist = float('inf')
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Consider periodic images
                diff = coords[j] - coords[i]
                # Apply minimum image convention in Cartesian space
                for k in range(3):
                    cell_length = initial_struct.lattice.abc[k]
                    if abs(diff[k]) > cell_length / 2:
                        diff[k] = diff[k] - np.sign(diff[k]) * cell_length
                
                dist = np.linalg.norm(diff)
                min_dist = min(min_dist, dist)
        
        if min_dist < min_allowed_dist:
            logging.warning(f"WARNING: Image {img} has minimum distance {min_dist:.3f} Å < {min_allowed_dist} Å")
            logging.warning("Consider using fewer images or a different interpolation method")
    
    return path


def gp1_path_analysis(
    system_params: Dict,
    local_min_poscar: str,
    saddle_poscar: str,
    checkpoint_file: Optional[str] = None
) -> Tuple[Dict, List[Dict]]:
    """
    Main function for GP1 path analysis.
    
    Returns:
        results: Dictionary containing analysis results
        gp1_models: List of fitted GP1 models
    """
    # Set up stdout redirection
    import sys
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info("Starting GP1 path analysis")
    
    # Load structures
    logging.info("Loading structures...")
    local_min_pos, structure_info, local_min_poscar_path = load_poscar_structure(
        local_min_poscar, moving_indices=system_params.get('moving_indices')
    )
    saddle_pos, _, saddle_poscar_path = load_poscar_structure(
        saddle_poscar, moving_indices=system_params.get('moving_indices')
    )
    
    # Get mass vector from structure
    local_min_structure = Poscar.from_file(local_min_poscar_path, check_for_potcar=False).structure
    mass_vector = get_mass_vector(local_min_structure)
    
    # Create interpolated path
    logging.info(f"Creating {system_params['n_images']} interpolated images using {system_params['interpolation_method']} method...")
    path_positions = create_interpolated_path(
        local_min_poscar_path, saddle_poscar_path,  # Use the actual file paths
        system_params['n_images'],
        system_params['interpolation_method']
    )
    
    # Initialize appropriate calculator based on execution mode
    from vasp_manager import VASPManager
    from vasp_interface import VASPInterface
    
    # Use the full path to POSCAR file from load_poscar_structure
    poscar_path = local_min_poscar_path
    
    # Create VASP manager (handles all execution modes including EAM)
    vasp_mgr = VASPManager(
        execution_mode=system_params['execution_mode'],
        base_dir=get_output_path("vasp_runs"),
        user_poscar_path=poscar_path,
        eam_potential_file=system_params.get('eam_potential_file'),
        parallel_eam=system_params.get('parallel_eam', False),
        eam_n_workers=system_params.get('eam_n_workers')
    )
    
    # Create VASP interface (works with all execution modes)
    local_pes = VASPInterface(
        vasp_manager=vasp_mgr,
        poscar_file=poscar_path,
        activation_radius=system_params['activation_radius'],
        moving_indices=structure_info['moving_indices']
    )
    
    # Create walker
    if checkpoint_file:
        logging.info(f"Loading from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        walker = checkpoint_data['walker']
        walker.local_pes = local_pes  # Reconnect VASP interface
    else:
        walker = WalkerGP1Path(
            path_positions=path_positions,
            local_pes=local_pes,
            force_constants_file=system_params['force_constants_file'],
            POSCAR_file=local_min_poscar_path,  # Use the full path from load_poscar_structure
            temperature=system_params['temperature'],
            mass=mass_vector,  # Use proper mass vector
            num_snapshots=system_params['num_snapshots'],
            gp1_noise_model=system_params['gp1_noise_model'],
            gp1_student_t_df=system_params['gp1_student_t_df'],
            gp1_use_adaptive_df=system_params['gp1_use_adaptive_df'],
            gp1_adaptive_df_start_iter=system_params['gp1_adaptive_df_start_iter'],
            gp1_adaptive_df_end_iter=system_params['gp1_adaptive_df_end_iter'],
            gp1_adaptive_df_target=system_params['gp1_adaptive_df_target'],
            gp1_remove_outliers=system_params['gp1_remove_outliers'],
            gp1_outlier_threshold=system_params['gp1_outlier_threshold'],
            energy_shift=system_params['energy_shift'],
            verbose=True,
        )
    
    # Run the analysis
    results, gp1_models = walker.run()
    
    # Save results
    results_dir = get_output_path('results')
    
    # Save raw results (without GP1 models which can't be pickled)
    results_file = os.path.join(results_dir, 'gp1_path_analysis_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'system_params': system_params,
            'structure_info': structure_info,
            'path_positions': path_positions
        }, f)
    logging.info(f"Results saved to: {results_file}")
    
    # Save final checkpoint
    checkpoint_dir = get_output_path('checkpoints')
    final_checkpoint = os.path.join(checkpoint_dir, 'gp1_path_final.pkl')
    walker.save_checkpoint(final_checkpoint)
    
    # Create plots if requested
    if not system_params['no_plots']:
        logging.info("Creating analysis plots...")
        walker.create_analysis_plots(results, gp1_models)
    
    # Print summary
    print_analysis_summary(results)
    
    return results, gp1_models


def print_analysis_summary(results: Dict):
    """Print summary of analysis results."""
    print("\n" + "="*60)
    print(" "*20 + "GP1 PATH ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal images analyzed: {results['n_images']}")
    print(f"Snapshots per image: {results['n_snapshots']}")
    print(f"Total VASP evaluations: {results['total_evaluations']}")
    
    if results.get('n_outlier_images', 0) > 0:
        print(f"\nWARNING: {results['n_outlier_images']} outlier images detected!")
        print(f"Outlier indices: {results['outlier_indices']}")
        print("Statistics below exclude outliers for more meaningful results.")
    
    print("\nPrediction Statistics (averaged over all models):")
    print(f"  Energy prediction error: {results['avg_energy_error']:.6f} ± {results['std_energy_error']:.6f} eV")
    print(f"  Force prediction error: {results['avg_force_error']:.6f} ± {results['std_force_error']:.6f} eV/Å")
    
    print("\nUncertainty Quantification:")
    print(f"  Energy σ/MAD ratio: {results['avg_energy_sigma_mad_ratio']:.3f} (target: 1.46)")
    print(f"  Force σ/MAD ratio: {results['avg_force_sigma_mad_ratio']:.3f} (target: 1.46)")
    
    print("\nRaw Data Statistics:")
    print(f"  Energy std dev: {results['raw_energy_std']:.6f} eV")
    print(f"  Force std dev: {results['raw_force_std']:.6f} eV/Å")
    print(f"  Energy σ/MAD: {results['raw_energy_sigma_mad']:.3f}")
    print(f"  Force σ/MAD: {results['raw_force_sigma_mad']:.3f}")
    
    print("\nThermal Noise:")
    print(f"  Average force noise: {results['avg_force_noise']:.6f} eV/Å")
    print(f"  Average energy noise: {results['avg_energy_noise']:.6f} eV")
    
    if results.get('n_outlier_images', 0) > 0:
        print("\n" + "!"*60)
        print("! IMPORTANT: Linear interpolation created unphysical configurations!")
        print("! Consider using NEB or other path optimization methods instead.")
        print("!"*60)
    
    print("="*60)


def main():
    """Main entry point."""
    # Parse arguments
    system_params, local_min_poscar, saddle_poscar, checkpoint_file = parse_arguments()
    
    # Clean up any stray eam_file_calculator.py in the scripts directory
    # This is a workaround for a bug in eam_file_executor.py
    import os
    eam_calc_path = os.path.join(os.path.dirname(__file__), 'eam_file_calculator.py')
    if os.path.exists(eam_calc_path):
        os.remove(eam_calc_path)
        logging.info("Removed stray eam_file_calculator.py from scripts directory")
    
    # Initialize output directory structure
    OutputManager.setup(
        base_dir=system_params.get('output_dir', None),
        run_name=system_params.get('run_name', None),
        input_dir=system_params.get('input_dir', None)
    )
    
    # Save run metadata
    OutputManager.save_run_metadata({
        'script': 'run_gp1_path_analysis.py',
        'parameters': system_params,
        'local_min_poscar': local_min_poscar,
        'saddle_poscar': saddle_poscar
    })
    
    # Set up logging
    setup_logging(system_params.get('log_level', 'INFO'))
    
    print("Starting GP1 Path Analysis with parameters:")
    print(f"  Images: {system_params['n_images']}")
    print(f"  Temperature: {system_params['temperature']} K")
    print(f"  Snapshots per image: {system_params['num_snapshots']}")
    print(f"  GP1 noise model: {system_params['gp1_noise_model']}")
    print(f"  Activation radius: {system_params['activation_radius']} Å")
    
    try:
        # Run analysis
        results, gp1_models = gp1_path_analysis(
            system_params,
            local_min_poscar,
            saddle_poscar,
            checkpoint_file
        )
        
        logging.info("GP1 path analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error during GP1 path analysis: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()