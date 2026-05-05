#!/usr/bin/env python
"""Run dimer method with GP2 surrogate model for acceleration."""

import argparse
import os
import sys
import logging
import numpy as np
from walker_gp2_dimer import WalkerGP2Dimer
from pymatgen.core.periodic_table import Element
from vasp_manager import VASPManager, cleanup
from pymatgen.io.vasp import Poscar
from vasp_interface import VASPInterface
from pymatgen.core import Structure
from output_manager import OutputManager, get_output_path, get_input_path


def setup_logging():
    """Set up logging."""
    logging.getLogger().handlers.clear()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(get_output_path('logs', 'gp2_dimer_search.log'), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")

def find_nearest_neighbors(structure, center_indices, cutoff_1nn=None, cutoff_2nn=None, auto_detect=True):
    """Find nearest neighbors of specified atoms.
    
    Args:
        structure: Pymatgen Structure object
        center_indices: List of atom indices to find neighbors for
        cutoff_1nn: Cutoff for 1st nearest neighbors (Å)
        cutoff_2nn: Cutoff for 2nd nearest neighbors (Å)  
        auto_detect: If True, automatically detect cutoffs from structure
        
    Returns:
        dict with 'first_neighbors' and 'second_neighbors' lists
    """
    import numpy as np
    from collections import defaultdict
    
    positions = structure.cart_coords
    n_atoms = len(positions)
    
    # Calculate all distances from center atoms
    all_neighbors = defaultdict(list)
    
    for center_idx in center_indices:
        center_pos = positions[center_idx]
        distances = []
        
        # Calculate distances to all other atoms
        for i in range(n_atoms):
            if i != center_idx:
                dist = np.linalg.norm(positions[i] - center_pos)
                distances.append((dist, i))
        
        # Sort by distance
        distances.sort()
        
        if auto_detect and (cutoff_1nn is None or cutoff_2nn is None):
            # Auto-detect cutoffs by finding gaps in distance distribution
            dist_values = [d[0] for d in distances]
            
            # Find significant gaps (> 0.5 Å)
            gaps = []
            for i in range(len(dist_values)-1):
                gap = dist_values[i+1] - dist_values[i]
                if gap > 0.5:
                    gaps.append((i, gap, dist_values[i], dist_values[i+1]))
            
            if cutoff_1nn is None and len(gaps) >= 1:
                # First gap defines 1st nearest neighbors
                cutoff_1nn = (gaps[0][2] + gaps[0][3]) / 2
                print(f"Auto-detected 1st NN cutoff: {cutoff_1nn:.2f} Å")
            
            if cutoff_2nn is None and len(gaps) >= 2:
                # Second gap defines 2nd nearest neighbors
                cutoff_2nn = (gaps[1][2] + gaps[1][3]) / 2
                print(f"Auto-detected 2nd NN cutoff: {cutoff_2nn:.2f} Å")
        
        # Set defaults if not found
        if cutoff_1nn is None:
            cutoff_1nn = 3.5  # Default for many metals
        if cutoff_2nn is None:
            cutoff_2nn = 5.0
        
        # Classify neighbors
        first_neighbors = []
        second_neighbors = []
        
        for dist, idx in distances:
            if dist <= cutoff_1nn:
                first_neighbors.append(idx)
            elif dist <= cutoff_2nn:
                second_neighbors.append(idx)
        
        all_neighbors[center_idx] = {
            '1nn': first_neighbors,
            '2nn': second_neighbors,
            'cutoffs': (cutoff_1nn, cutoff_2nn)
        }
    
    # Combine all unique neighbors
    all_1nn = set()
    all_2nn = set()
    
    for center_idx in center_indices:
        all_1nn.update(all_neighbors[center_idx]['1nn'])
        all_2nn.update(all_neighbors[center_idx]['2nn'])
    
    # Remove any overlap and original atoms
    all_2nn = all_2nn - all_1nn - set(center_indices)
    all_1nn = all_1nn - set(center_indices)
    
    return {
        'first_neighbors': sorted(list(all_1nn)),
        'second_neighbors': sorted(list(all_2nn)),
        'details': all_neighbors
    }

def gp2_dimer_search(poscar_file: str, system_params: dict) -> tuple:
    """Run dimer search with GP2 surrogate model."""
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info("Starting GP2-accelerated dimer search")
    
    # Check if this is a continuation
    is_continuation = system_params.get('continuation', False)
    
    if is_continuation:
        logging.info("Continuation run - preserving existing data")
    else:
        # Clean up previous runs
        cleanup()
    
    setup_logging()
    
    try:
        # Setup structure
        structure = Poscar.from_file(poscar_file, check_for_potcar=False).structure
        
        # Create VASP manager
        work_dir = get_output_path('vasp_runs')
        
        execution_mode = system_params.get('execution_mode', 'mpi')
        mpi_command = system_params.get('mpi_command', None)
        vasp_command = system_params.get('vasp_command', 'vasp_gam')
        eam_potential_file = system_params.get('eam_potential_file', None)
        kim_model = system_params.get('kim_model', 'EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000')
        
        print(f"\nInitializing calculations with execution mode: {execution_mode}")
        
        vasp_mgr = VASPManager(
            base_dir=os.path.join(work_dir, "vasp_runs"),
            user_poscar_path=poscar_file,
            execution_mode=execution_mode,
            mpi_command=mpi_command,
            vasp_command=vasp_command,
            eam_potential_file=eam_potential_file,
            skip_thermal=True,  # No thermal sampling needed
            kim_model_name=kim_model,
        )
        
        # Get activation radius and moving indices
        activation_radius = system_params.get('activation_radius', 10.0)
        moving_indices = system_params.get('moving_indices', [0])
        
        local_pes = VASPInterface(
            vasp_manager=vasp_mgr,
            poscar_file=poscar_file,
            activation_radius=activation_radius,
            moving_indices=moving_indices
        )
        
        # Get initial position (full system)
        initial_position = structure.cart_coords.flatten()
        
        # Create walker
        if not is_continuation:
            walker = WalkerGP2Dimer(
                initial_position=initial_position,
                local_pes=local_pes,
                max_dimer_steps=system_params["max_dimer_steps"],
                # Stopping criteria
                disp_max=system_params.get("disp_max", 0.5),
                ratio_at_limit=system_params.get("ratio_at_limit", 2.0/3.0),
                # Dimer parameters
                rotation=system_params.get("rotation", "lbfgsext"),
                translation=system_params.get("translation", "lbfgs"),
                dimer_sep=system_params.get("dimer_sep", 0.01),
                T_anglerot=system_params.get("T_anglerot", 0.01),
                T_anglerot_init=system_params.get("T_anglerot_init", 0.0873),
                T_anglerot_gp=system_params.get("T_anglerot_gp", 0.01),
                max_dimer_rotations=system_params.get("max_dimer_rotations", 10),
                num_init_rotations=system_params.get("num_init_rotations", 5),
                num_iter_rot_gp=system_params.get("num_iter_rot_gp", 10),
                dimer_stopping_criteria=system_params.get("dimer_stopping_criteria", 0.01),
                step_size=system_params.get("step_size", 0.1),
                max_step_size=system_params.get("max_step_size", 0.1),
                # GP convergence
                divisor_T_dimer_gp=system_params.get("divisor_T_dimer_gp", 10.0),
                max_inner_iterations=system_params.get("max_inner_iterations", 1000),
                # Options
                initrot_nogp=system_params.get("initrot_nogp", False),
                inittrans_nogp=system_params.get("inittrans_nogp", False),
                eval_image1=system_params.get("eval_image1", False),
                num_bigiter_initloc=system_params.get("num_bigiter_initloc", np.inf),
                num_bigiter_initparam=system_params.get("num_bigiter_initparam", np.inf),
                # Other parameters
                verbose=system_params["verbose"],
                checkpoint_interval=system_params.get('checkpoint_interval', 1),
                model_type=system_params.get("model_type", "MultitaskGPModel_rbf_atomic"),
            )
            
            # Set initial orientation if specified
            initial_orient_method = system_params.get('initial_orient_method', 'auto')
            if initial_orient_method == 'manual':
                manual_orient = system_params.get('manual_orient', None)
                if manual_orient is not None:
                    walker.dimer.set_initial_direction(np.array(manual_orient))
        else:
            # Create walker and restore from checkpoint
            walker = WalkerGP2Dimer(
                initial_position=initial_position,
                local_pes=local_pes,
                max_dimer_steps=1,  # Will be updated
                verbose=system_params["verbose"]
            )
            
            # Load checkpoint
            checkpoint = walker.load_checkpoint()
            
            # Update max steps for continuation
            steps_completed = walker.steps
            additional_steps = system_params["max_dimer_steps"]
            walker.max_dimer_steps = steps_completed + additional_steps
            
            logging.info(f"Continuing from step {steps_completed}")
            logging.info(f"Will run {additional_steps} additional steps")
        
        # Run search
        logging.info("Starting walker run")
        final_pos, final_energy, final_forces = walker.run()
        
        logging.info("Walker run completed")
        
        # Save results
        import pickle
        results = {
            'final_position': final_pos,
            'final_energy': final_energy,
            'final_forces': final_forces,
            'converged': walker.converged,
            'outer_iterations': walker.bigiter,
            'inner_iterations': len(walker.E_R_gp),
            'obs_total': walker.obs_total,
            'obs_initrot': walker.obs_initrot,
            'R_all': walker.R_all,
            'E_all': walker.E_all,
            'G_all': walker.G_all,
            'E_R_acc': walker.E_R_acc,
            'maxF_R_acc': walker.maxF_R_acc,
            'E_R_gp': walker.E_R_gp,
            'maxF_R_gp': walker.maxF_R_gp,
            'obs_at': walker.obs_at,
            'num_esmax': walker.num_esmax,
            'num_es1': walker.num_es1,
            'num_es2': walker.num_es2,
            'n_atoms': walker.n_atoms
        }
        
        results_dir = get_output_path('results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'gp2_dimer_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Also save final structure
        final_coords = final_pos.reshape(-1, 3)
        final_structure = Structure(
            lattice=structure.lattice,
            species=structure.species,
            coords=final_coords,
            coords_are_cartesian=True
        )
        final_poscar = Poscar(final_structure)
        final_poscar.write_file(os.path.join(results_dir, 'POSCAR_final'))
        
        print(f"\nFinal results:")
        print(f"  Energy: {final_energy:.6f} eV")
        print(f"  RMS Force: {np.sqrt(np.mean(final_forces**2)):.6f} eV/Å")
        print(f"  Max |Force|: {np.max(np.abs(final_forces)):.6f} eV/Å")
        print(f"  Converged: {walker.converged}")
        print(f"  Outer iterations: {walker.bigiter}")
        print(f"  Total observations: {walker.obs_total}")
        print(f"  Inner iterations: {len(walker.E_R_gp)}")
        print(f"\nStopping statistics:")
        print(f"  Max iterations reached: {walker.num_esmax}")
        print(f"  Inter-atomic distance limit: {walker.num_es1}")
        print(f"  Displacement limit: {walker.num_es2}")
        print(f"\nFinal structure saved to: {os.path.join(results_dir, 'POSCAR_final')}")
        
        return final_pos, final_energy, final_forces
        
    except Exception as e:
        logging.error(f"Error during GP2 dimer search: {e}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Dimer Method with GP2 Surrogate Model (Atomic GP-Dimer Style)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required files
    parser.add_argument('--poscar-file', default='POSCAR',
                        help='Initial POSCAR structure (looked for in inputs/ directory)')
    
    # Physical parameters
    parser.add_argument('--activation-radius', type=float, default=10.0,
                        help='Activation radius (Å) for frozen atoms')
    parser.add_argument('--moving-indices', type=int, nargs='+', default=[0],
                        help='Atom indices that can move')
    
    # Stopping criteria (following atomic GP-dimer)
    parser.add_argument('--disp-max', type=float, default=0.5,
                        help='Maximum displacement from nearest observed point (Å)')
    parser.add_argument('--ratio-at-limit', type=float, default=2.0/3.0,
                        help='Limit for inter-atomic distance ratio')
    
    # Dimer parameters
    parser.add_argument('--max-dimer-steps', type=int, default=100,
                        help='Maximum number of outer iterations')
    parser.add_argument('--rotation', choices=['mn', 'cg', 'lbfgs', 'lbfgsext'],
                        default='lbfgsext',
                        help='Dimer rotation method')
    parser.add_argument('--translation', choices=['newton', 'cg', 'lbfgs', 'qmvv'],
                        default='lbfgs',
                        help='Dimer translation method')
    parser.add_argument('--dimer-sep', type=float, default=0.01,
                        help='Dimer separation distance (Å)')
    parser.add_argument('--T-anglerot', type=float, default=0.01,
                        help='Rotation convergence threshold (radians)')
    parser.add_argument('--T-anglerot-init', type=float, default=0.0873,
                        help='Initial rotation convergence threshold (radians)')
    parser.add_argument('--T-anglerot-gp', type=float, default=0.01,
                        help='Rotation convergence threshold on GP surface')
    parser.add_argument('--max-dimer-rotations', type=int, default=10,
                        help='Maximum rotations per translation')
    parser.add_argument('--num-init-rotations', type=int, default=5,
                        help='Maximum number of initial rotations')
    parser.add_argument('--num-iter-rot-gp', type=int, default=10,
                        help='Maximum rotations per translation on GP surface')
    parser.add_argument('--dimer-stopping-criteria', type=float, default=0.01,
                        help='Max force convergence threshold (eV/Å)')
    
    # GP convergence
    parser.add_argument('--divisor-T-dimer-gp', type=float, default=10.0,
                        help='Divisor for dynamic GP convergence threshold')
    
    # Relaxation phase
    parser.add_argument('--max-inner-iterations', type=int, default=1000,
                        help='Maximum iterations per relaxation phase')
    
    # Initial phase options
    parser.add_argument('--initrot-nogp', action='store_true',
                        help='Perform initial rotations without GP')
    parser.add_argument('--inittrans-nogp', action='store_true',
                        help='Perform initial translation without GP')
    parser.add_argument('--eval-image1', action='store_true',
                        help='Evaluate image 1 after each phase')
    
    # Restart options
    parser.add_argument('--num-bigiter-initloc', type=float, default=np.inf,
                        help='Number of iterations starting from initial location')
    parser.add_argument('--num-bigiter-initparam', type=float, default=np.inf,
                        help='Number of iterations with fresh hyperparameters')
    
    # Step size control
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Base step size for translations (Å)')
    parser.add_argument('--max-step-size', type=float, default=0.1,
                        help='Maximum allowed step size (Å)')
    
    # Model type
    parser.add_argument('--model-type',
                        choices=['MultitaskGPModel_rbf_atomic',
                                 'BatchIndependentMultitaskGPModel_rbf',
                                 'GPModelWithDerivatives_rbf_atomic'],
                        default='MultitaskGPModel_rbf_atomic',
                        help='GP model type')
    
    # Execution mode
    parser.add_argument('--execution-mode',
                        choices=['mock', 'mpi', 'slurm', 'stampede3', 'anvil', 'eam'],
                        default='mpi',
                        help='Execution mode')
    parser.add_argument('--mpi-command', default=None,
                        help='Custom MPI command')
    parser.add_argument('--vasp-command', default='vasp_gam',
                        help='VASP executable name')
    parser.add_argument('--eam-potential-file', default=None,
                        help='Path to EAM potential file')
    parser.add_argument('--kim-model',
                        default='EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000',
                        help='KIM model name for EAM calculations')
    
    # In run_gp2_dimer.py, add arguments:
    parser.add_argument('--initial-orient-method', 
                        choices=['auto', 'random', 'manual'],
                        default='auto',
                        help='Method for setting initial dimer orientation')
    parser.add_argument('--manual-orient', type=float, nargs='+',
                        help='Manual initial orientation vector (must be same dimension as system)')
    parser.add_argument('--orient-atom-direction', type=str,
                    help='Atom index and direction, e.g., "52:-1,-1,-1"')
    

    parser.add_argument('--include-neighbors', 
                        choices=['none', '1nn', '2nn', '1nn+2nn'],
                        default='none',
                        help='Include nearest neighbors in moving atoms')
    parser.add_argument('--nn-cutoff-1', type=float, default=None,
                        help='Cutoff distance for 1st nearest neighbors (Å)')
    parser.add_argument('--nn-cutoff-2', type=float, default=None,
                        help='Cutoff distance for 2nd nearest neighbors (Å)')
    
    # GPU options
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration for GP training and inference')
    parser.add_argument('--no-gpu-fallback', action='store_true',
                        help='Fail if GPU is requested but unavailable')
    
    # Misc
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--continuation', action='store_true',
                        help='Continue from checkpoint')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save checkpoint every N iterations')
    
    # Output management
    parser.add_argument('--output-dir', default=None,
                        help='Base directory for all outputs (default: ./outputs)')
    parser.add_argument('--run-name', default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--input-dir', default=None,
                        help='Directory containing input files (default: ./inputs)')
    
    args = parser.parse_args()

    
    # Build system parameters
    system_params = {
        'activation_radius': args.activation_radius,
        'moving_indices': args.moving_indices,
        'max_dimer_steps': args.max_dimer_steps,
        # Stopping criteria
        'disp_max': args.disp_max,
        'ratio_at_limit': args.ratio_at_limit,
        # Dimer parameters
        'rotation': args.rotation,
        'translation': args.translation,
        'dimer_sep': args.dimer_sep,
        'T_anglerot': args.T_anglerot,
        'T_anglerot_init': args.T_anglerot_init,
        'T_anglerot_gp': args.T_anglerot_gp,
        'max_dimer_rotations': args.max_dimer_rotations,
        'num_init_rotations': args.num_init_rotations,
        'num_iter_rot_gp': args.num_iter_rot_gp,
        'dimer_stopping_criteria': args.dimer_stopping_criteria,
        'divisor_T_dimer_gp': args.divisor_T_dimer_gp,
        'max_inner_iterations': args.max_inner_iterations,
        'step_size': args.step_size,
        'max_step_size': args.max_step_size,
        # Options
        'initrot_nogp': args.initrot_nogp,
        'inittrans_nogp': args.inittrans_nogp,
        'eval_image1': args.eval_image1,
        'num_bigiter_initloc': args.num_bigiter_initloc,
        'num_bigiter_initparam': args.num_bigiter_initparam,
        # Other
        'model_type': args.model_type,
        'verbose': args.verbose,
        'execution_mode': args.execution_mode,
        'mpi_command': args.mpi_command,
        'vasp_command': args.vasp_command,
        'eam_potential_file': args.eam_potential_file,
        'continuation': args.continuation,
        'checkpoint_interval': args.checkpoint_interval,
        'kim_model': args.kim_model,
        'use_gpu': args.gpu,
        'no_gpu_fallback': args.no_gpu_fallback,
    }

    # Store raw arguments for structure processing later
    system_params.update({
        'raw_moving_indices': args.moving_indices,
        'include_neighbors': args.include_neighbors,
        'nn_cutoff_1': args.nn_cutoff_1,
        'nn_cutoff_2': args.nn_cutoff_2,
        'orient_atom_direction': args.orient_atom_direction,
        'output_dir': args.output_dir,
        'run_name': args.run_name,
        'input_dir': args.input_dir,
    })
    
    return args.poscar_file, system_params


def process_structure_parameters(poscar_path: str, system_params: dict) -> dict:
    """Process structure-dependent parameters after OutputManager is set up."""
    structure = Poscar.from_file(poscar_path, check_for_potcar=False).structure
    
    # Process moving indices with neighbors
    moving_indices = system_params['raw_moving_indices']
    if system_params['include_neighbors'] != 'none':
        print(f"\nFinding nearest neighbors for atoms: {moving_indices}")
        
        neighbors = find_nearest_neighbors(
            structure, 
            moving_indices,
            cutoff_1nn=system_params['nn_cutoff_1'],
            cutoff_2nn=system_params['nn_cutoff_2'],
            auto_detect=True
        )
        
        # Add neighbors based on user choice
        if system_params['include_neighbors'] in ['1nn', '1nn+2nn']:
            moving_indices.extend(neighbors['first_neighbors'])
            print(f"Added {len(neighbors['first_neighbors'])} first nearest neighbors: {neighbors['first_neighbors']}")
        
        if system_params['include_neighbors'] in ['2nn', '1nn+2nn']:
            moving_indices.extend(neighbors['second_neighbors'])
            print(f"Added {len(neighbors['second_neighbors'])} second nearest neighbors: {neighbors['second_neighbors']}")
        
        # Remove duplicates and sort
        moving_indices = sorted(list(set(moving_indices)))
        
        print(f"\nFinal moving atoms ({len(moving_indices)} total): {moving_indices}")
        
        # Print a visual representation
        print("\nAtom types:")
        for idx in range(structure.num_sites):
            status = "MOVE" if idx in moving_indices else "FROZEN"
            species = structure[idx].species_string
            print(f"  Atom {idx}: {species} - {status}")

    system_params['moving_indices'] = moving_indices

    # Handle manual orientation
    if system_params['orient_atom_direction']:
        parts = system_params['orient_atom_direction'].split(':')
        atom_idx = int(parts[0])
        direction = list(map(float, parts[1].split(',')))

        n_atoms = structure.num_sites
        
        # Create full orientation
        manual_orient = np.zeros(n_atoms * 3)

        manual_orient[atom_idx * 3] = direction[0]
        manual_orient[atom_idx * 3 + 1] = direction[1]
        manual_orient[atom_idx * 3 + 2] = direction[2]
        
        system_params['manual_orient'] = manual_orient.tolist()
        system_params['initial_orient_method'] = 'manual'
    
    return system_params


if __name__ == "__main__":
    try:
        poscar_file, system_params = parse_arguments()
        
        # Initialize output directory structure
        OutputManager.setup(
            base_dir=system_params.get('output_dir', None),
            run_name=system_params.get('run_name', None),
            input_dir=system_params.get('input_dir', None)
        )
        
        # Save run metadata
        OutputManager.save_run_metadata({
            'script': 'run_gp2_dimer.py',
            'parameters': system_params,
            'poscar_file': poscar_file
        })
        
        # Handle input file path now that OutputManager is set up
        if not os.path.isabs(poscar_file):
            poscar_path = get_input_path(poscar_file)
            if not os.path.exists(poscar_path):
                # Backward compatibility
                if os.path.exists(poscar_file):
                    poscar_path = os.path.abspath(poscar_file)
                    print(f"Warning: Found {poscar_file} in current directory")
                    print(f"Consider moving it to inputs/ directory")
                else:
                    raise FileNotFoundError(f"File not found in inputs/: {poscar_file}")
        else:
            poscar_path = poscar_file
        
        # Process structure-dependent parameters
        system_params = process_structure_parameters(poscar_path, system_params)
        
        # Check GPU availability if requested
        if system_params.get('use_gpu', False):
            import torch
            
            # Print diagnostic information
            if system_params.get('verbose', False):
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
                if system_params.get('no_gpu_fallback', False):
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
                    if system_params.get('verbose', False):
                        print("  Falling back to CPU computation")
        
        print("Starting GP2-accelerated dimer search with parameters:")
        print("\nStopping Criteria:")
        print(f"  Max displacement: {system_params['disp_max']} Å")
        print(f"  Inter-atomic ratio limit: {system_params['ratio_at_limit']}")
        
        print("\nDimer Parameters:")
        print(f"  Max outer iterations: {system_params['max_dimer_steps']}")
        print(f"  Max inner iterations: {system_params['max_inner_iterations']}")
        print(f"  Convergence: {system_params['dimer_stopping_criteria']} eV/Å")
        print(f"  Step size: {system_params['step_size']} Å (max: {system_params['max_step_size']} Å)")
        
        print("\nGP Parameters:")
        print(f"  GP convergence divisor: {system_params['divisor_T_dimer_gp']}")
        print(f"  Initial rotations without GP: {system_params['initrot_nogp']}")
        print(f"  Initial translation without GP: {system_params['inittrans_nogp']}")
        
        final_pos, final_energy, final_forces = gp2_dimer_search(
            poscar_file=poscar_path,
            system_params=system_params
        )
        
        print("\nGP2 dimer search completed successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()