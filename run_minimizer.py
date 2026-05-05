#!/usr/bin/env python
"""Run local minimization to find local minima."""

import argparse
import os
import sys
import logging
import numpy as np
from walker_minimizer import WalkerMinimizer
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
    
    log_file = get_output_path('logs', 'minimization_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")

def minimization_search(poscar_file: str, system_params: dict) -> tuple:
    """Run minimization search to find local minimum."""
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info("Starting minimization search")
    
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
        work_dir = os.path.dirname(os.path.abspath(poscar_file))
        poscar_path = os.path.abspath(poscar_file)
        
        execution_mode = system_params.get('execution_mode', 'mpi')
        mpi_command = system_params.get('mpi_command', None)
        vasp_command = system_params.get('vasp_command', 'vasp_gam')
        eam_potential_file = system_params.get('eam_potential_file', None)
        kim_model = system_params.get('kim_model', 'EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000')
        
        print(f"\nInitializing calculations with execution mode: {execution_mode}")
        
        vasp_mgr = VASPManager(
            base_dir=get_output_path('vasp_runs'),
            user_poscar_path=poscar_path,
            execution_mode=execution_mode,
            mpi_command=mpi_command,
            vasp_command=vasp_command,
            eam_potential_file=eam_potential_file,
            skip_thermal=True,  # No thermal sampling needed
            kim_model_name=kim_model,
        )
        
        # Get activation radius and moving indices for VASPInterface
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
            walker = WalkerMinimizer(
                initial_position=initial_position,
                local_pes=local_pes,
                max_steps=system_params["max_steps"],
                method=system_params.get("method", "lbfgs"),
                step_size=system_params.get("step_size", 0.1),
                max_step_size=system_params.get("max_step_size", 0.2),
                stopping_criteria=system_params.get("stopping_criteria", 0.01),
                line_search=system_params.get("line_search", True),
                force_reset_threshold=system_params.get("force_reset_threshold", 0.5),
                adaptive_step=system_params.get("adaptive_step", True),
                verbose=system_params["verbose"],
                checkpoint_interval=system_params.get('checkpoint_interval', 1)
            )
        else:
            # Create walker and restore from checkpoint
            walker = WalkerMinimizer(
                initial_position=initial_position,
                local_pes=local_pes,
                max_steps=1,  # Will be updated
                method=system_params.get("method", "lbfgs"),
                step_size=system_params.get("step_size", 0.1),
                max_step_size=system_params.get("max_step_size", 0.2),
                stopping_criteria=system_params.get("stopping_criteria", 0.01),
                line_search=system_params.get("line_search", True),
                verbose=system_params["verbose"],
                checkpoint_interval=system_params.get('checkpoint_interval', 1)
            )
            
            # Load checkpoint
            checkpoint = walker.load_checkpoint()
            
            # Update max steps for continuation
            steps_completed = walker.steps
            additional_steps = system_params["max_steps"]
            walker.max_steps = steps_completed + additional_steps
            
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
            'steps': walker.steps,
            'trajectory': walker.trajectory,
            'vasp_eval_count': walker.vasp_eval_count,
            'n_atoms': walker.n_atoms
        }
        
        results_dir = get_output_path('results')
        with open(os.path.join(results_dir, 'minimizer_results.pkl'), 'wb') as f:
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
        final_poscar.write_file(os.path.join(results_dir, 'POSCAR_minimum'))
        
        print(f"\nFinal results:")
        print(f"  Energy: {final_energy:.6f} eV")
        print(f"  RMS Force: {np.sqrt(np.mean(final_forces**2)):.6f} eV/Å")
        print(f"  Max |Force|: {np.max(np.abs(final_forces)):.6f} eV/Å")
        print(f"  Converged: {walker.converged}")
        print(f"  Total VASP evaluations: {walker.vasp_eval_count}")
        print(f"\nFinal structure saved to: {os.path.join(results_dir, 'POSCAR_minimum')}")
        
        return final_pos, final_energy, final_forces
        
    except Exception as e:
        logging.error(f"Error during minimization search: {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Local Minimization Method',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required files
    parser.add_argument('--poscar-file', default='POSCAR',
                        help='Initial POSCAR structure (searched in inputs/ directory)')
    
    # Physical parameters
    parser.add_argument('--activation-radius', type=float, default=10.0,
                        help='Activation radius (Å) for VASPInterface')
    parser.add_argument('--moving-indices', type=int, nargs='+', default=[0],
                        help='Atom indices for VASPInterface tracking')
    
    # Minimization parameters
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum number of minimization steps')
    parser.add_argument('--method', 
                        choices=['steepest', 'cg', 'lbfgs', 'lbfgs_scipy', 'bfgs', 'fire'],
                        default='lbfgs_scipy',
                        help='Minimization method (lbfgs_scipy is most robust)')
    parser.add_argument('--step-size', type=float, default=0.05,
                        help='Base step size (Å)')
    parser.add_argument('--max-step-size', type=float, default=0.1,
                        help='Maximum allowed step size (Å)')
    parser.add_argument('--stopping-criteria', type=float, default=0.01,
                        help='RMS force convergence threshold (eV/Å)')
    parser.add_argument('--no-line-search', dest='line_search', 
                        action='store_false', default=True,
                        help='Disable line search')
    parser.add_argument('--force-reset-threshold', type=float, default=0.5,
                        help='Reset optimizer if force increases by this factor')
    parser.add_argument('--no-adaptive-step', dest='adaptive_step',
                        action='store_false', default=True,
                        help='Disable adaptive step sizing')
    
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
    
    # Output management
    parser.add_argument('--output-dir', default=None,
                        help='Base directory for all outputs (default: ./outputs)')
    parser.add_argument('--run-name', default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--input-dir', default=None,
                        help='Directory containing input files (default: ./inputs)')
    
    # Misc
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--continuation', action='store_true',
                        help='Continue from checkpoint')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--kim-model',
                    default='EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000',
                    help='KIM model name for EAM calculations')
    
    args = parser.parse_args()
    
    # Initialize output directory structure
    OutputManager.setup(
        base_dir=args.output_dir,
        run_name=args.run_name,
        input_dir=args.input_dir
    )
    
    # Handle input file paths
    poscar_file = args.poscar_file
    if not os.path.isabs(poscar_file):
        poscar_path = get_input_path(poscar_file)
        if not os.path.exists(poscar_path):
            # Backward compatibility
            if os.path.exists(poscar_file):
                poscar_path = os.path.abspath(poscar_file)
                print(f"Warning: Found {poscar_file} in current directory")
                print(f"Consider moving it to inputs/ directory")
            else:
                parser.error(f"File not found in inputs/: {poscar_file}")
        else:
            poscar_file = poscar_path
    
    # Build system parameters
    system_params = {
        'output_dir': args.output_dir,
        'run_name': args.run_name,
        'input_dir': args.input_dir,
        'activation_radius': args.activation_radius,
        'moving_indices': args.moving_indices,
        'max_steps': args.max_steps,
        'method': args.method,
        'step_size': args.step_size,
        'max_step_size': args.max_step_size,
        'stopping_criteria': args.stopping_criteria,
        'line_search': args.line_search,
        'force_reset_threshold': args.force_reset_threshold,
        'adaptive_step': args.adaptive_step,
        'verbose': args.verbose,
        'execution_mode': args.execution_mode,
        'mpi_command': args.mpi_command,
        'vasp_command': args.vasp_command,
        'eam_potential_file': args.eam_potential_file,
        'continuation': args.continuation,
        'checkpoint_interval': args.checkpoint_interval,
        'kim_model': args.kim_model,
    }
    
    return poscar_file, system_params

if __name__ == "__main__":
    try:
        poscar_file, system_params = parse_arguments()
        
        # Save run metadata
        OutputManager.save_run_metadata({
            'script': 'run_minimizer.py',
            'parameters': system_params,
            'poscar_file': poscar_file
        })
        
        print("Starting minimization search with parameters:")
        for key, value in system_params.items():
            print(f"  {key}: {value}")
        
        final_pos, final_energy, final_forces = minimization_search(
            poscar_file=poscar_file,
            system_params=system_params
        )
        
        print("\nMinimization search completed successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()