#!/usr/bin/env python
"""Run pure dimer method without GP models - Clean version."""

import argparse
import os
import sys
import logging
import numpy as np
from walker_pure_dimer import WalkerPureDimer
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
    
    log_file = get_output_path('logs', 'pure_dimer_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")

def pure_dimer_search(poscar_file: str, system_params: dict) -> tuple:
    """Run pure dimer search without GP models."""
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info("Starting pure dimer search")
    
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
        # Note: These are still used by VASPInterface for its internal tracking
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
        
        # Create walker - no longer needs moving_indices
        if not is_continuation:
            walker = WalkerPureDimer(
                initial_position=initial_position,
                local_pes=local_pes,
                max_dimer_steps=system_params["max_dimer_steps"],
                rotation=system_params.get("rotation", "lbfgsext"),
                translation=system_params.get("translation", "lbfgs"),
                dimer_sep=system_params.get("dimer_sep", 0.01),
                T_anglerot=system_params.get("T_anglerot", 0.01),
                T_anglerot_init=system_params.get("T_anglerot_init", 0.0873),
                max_dimer_rotations=system_params.get("max_dimer_rotations", 10),
                num_init_rotations=system_params.get("num_init_rotations", 5),
                dimer_stopping_criteria=system_params.get("dimer_stopping_criteria", 0.01),
                step_size=system_params.get("step_size", 0.02),
                max_step_size=system_params.get("max_step_size", 0.05),
                verbose=system_params["verbose"],
                checkpoint_interval=system_params.get('checkpoint_interval', 1)
            )
            
            # Set initial orientation if specified
            initial_orient_method = system_params.get('initial_orient_method', 'auto')
            if initial_orient_method == 'auto':
                # Will be auto-set in walker.run() based on forces
                pass
            elif initial_orient_method == 'random':
                # Force random initialization
                walker.dimer.initialize_direction()
            elif initial_orient_method == 'manual':
                # Use user-specified orientation
                manual_orient = system_params.get('manual_orient', None)
                if manual_orient is not None:
                    walker.set_initial_orientation(np.array(manual_orient))
        else:
            # Create walker and restore from checkpoint
            walker = WalkerPureDimer(
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
            'steps': walker.steps,
            'trajectory': walker.trajectory,
            'vasp_eval_count': walker.vasp_eval_count,
            'n_atoms': walker.n_atoms
        }
        
        results_dir = get_output_path('results')
        with open(os.path.join(results_dir, 'pure_dimer_results.pkl'), 'wb') as f:
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
        print(f"  Total VASP evaluations: {walker.vasp_eval_count}")
        print(f"\nFinal structure saved to: {os.path.join(results_dir, 'POSCAR_final')}")
        
        return final_pos, final_energy, final_forces
        
    except Exception as e:
        logging.error(f"Error during pure dimer search: {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Pure Dimer Method (No GP Models)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required files
    parser.add_argument('--poscar-file', default='POSCAR',
                        help='Initial POSCAR structure (searched in inputs/ directory)')
    
    # Physical parameters
    parser.add_argument('--activation-radius', type=float, default=10.0,
                        help='Activation radius (Å) for VASPInterface')
    parser.add_argument('--moving-indices', type=int, nargs='+', default=[0],
                        help='Atom indices for VASPInterface tracking (not used by dimer)')
    
    # Dimer parameters
    parser.add_argument('--max-dimer-steps', type=int, default=100,
                        help='Maximum number of dimer steps')
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
    parser.add_argument('--max-dimer-rotations', type=int, default=10,
                        help='Maximum rotations per translation')
    parser.add_argument('--dimer-stopping-criteria', type=float, default=0.01,
                        help='RMS force convergence threshold (eV/Å)')
    parser.add_argument('--num-init-rotations', type=int, default=5,
                        help='Maximum number of initial rotations')
    
    # Step size control
    parser.add_argument('--step-size', type=float, default=0.02,
                        help='Base step size for translations (Å)')
    parser.add_argument('--max-step-size', type=float, default=0.05,
                        help='Maximum allowed step size (Å)')
    
    # Initial orientation control
    parser.add_argument('--initial-orient-method', 
                        choices=['auto', 'random', 'manual'],
                        default='auto',
                        help='Method for setting initial dimer orientation')
    parser.add_argument('--manual-orient', type=float, nargs='+',
                        help='Manual initial orientation vector (for manual method)')
    
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
        'max_dimer_steps': args.max_dimer_steps,
        'rotation': args.rotation,
        'translation': args.translation,
        'dimer_sep': args.dimer_sep,
        'T_anglerot': args.T_anglerot,
        'T_anglerot_init': args.T_anglerot_init,
        'max_dimer_rotations': args.max_dimer_rotations,
        'dimer_stopping_criteria': args.dimer_stopping_criteria,
        'step_size': args.step_size,
        'max_step_size': args.max_step_size,
        'initial_orient_method': args.initial_orient_method,
        'manual_orient': args.manual_orient,
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
            'script': 'run_pure_dimer.py',
            'parameters': system_params,
            'poscar_file': poscar_file
        })
        
        print("Starting pure dimer search with parameters:")
        for key, value in system_params.items():
            print(f"  {key}: {value}")
        
        final_pos, final_energy, final_forces = pure_dimer_search(
            poscar_file=poscar_file,
            system_params=system_params
        )
        
        print("\nPure dimer search completed successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()