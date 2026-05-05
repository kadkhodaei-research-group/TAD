#!/usr/bin/env python
"""Run dimer method with enhanced dual GP approach based on toy model insights."""

import argparse
import os
import sys
import logging
import platform
import numpy as np
from itertools import product
from walker_tad import WalkerDualGPImproved
from pymatgen.core.periodic_table import Element
from vasp_manager import VASPManager, cleanup
from pymatgen.io.vasp import Poscar
from vasp_interface import VASPInterface
from pymatgen.core import Structure
from output_manager import OutputManager, get_output_path, get_input_path


_BASIS_LIBRARY = {
    "sc": np.array([
        [0.0, 0.0, 0.0],
    ], dtype=float),
    "bcc": np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ], dtype=float),
    "fcc": np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ], dtype=float),
    "diamond": np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ], dtype=float),
}


def _pbc_diff(frac_to, frac_from):
    d = np.asarray(frac_to) - np.asarray(frac_from)
    d -= np.round(d)
    return d


def _possible_supercells(natoms, nbasis, vacancies=1, max_rep=10):
    ideal_natoms = natoms + vacancies
    if ideal_natoms % nbasis != 0:
        return []
    ncells = ideal_natoms // nbasis
    candidates = []
    for nx in range(1, max_rep + 1):
        for ny in range(1, max_rep + 1):
            for nz in range(1, max_rep + 1):
                if nx * ny * nz == ncells:
                    candidates.append((nx, ny, nz))
    candidates.sort(key=lambda r: (
        max(r) - min(r),
        abs(r[0] - r[1]) + abs(r[1] - r[2]) + abs(r[0] - r[2]),
    ))
    return candidates


def _generate_ideal_sites(basis, reps):
    nx, ny, nz = reps
    reps_array = np.array([nx, ny, nz], dtype=float)
    sites = []
    for i, j, k in product(range(nx), range(ny), range(nz)):
        cell_origin = np.array([i, j, k], dtype=float)
        for b in basis:
            site = (cell_origin + b) / reps_array
            sites.append(site % 1.0)
    return np.array(sites)


def _matching_score(frac_coords, lattice, ideal_sites, vacancies=1):
    atom_to_site_distances = []
    for atom in frac_coords:
        diffs_frac = np.array([_pbc_diff(site, atom) for site in ideal_sites])
        diffs_cart = diffs_frac @ lattice
        dists = np.linalg.norm(diffs_cart, axis=1)
        atom_to_site_distances.append(np.min(dists))
    atom_to_site_distances = np.array(atom_to_site_distances)

    site_to_atom_distances = []
    for site in ideal_sites:
        diffs_frac = np.array([_pbc_diff(atom, site) for atom in frac_coords])
        diffs_cart = diffs_frac @ lattice
        dists = np.linalg.norm(diffs_cart, axis=1)
        site_to_atom_distances.append(np.min(dists))
    site_to_atom_distances = np.array(site_to_atom_distances)

    sorted_site_distances = np.sort(site_to_atom_distances)
    nonvac_site_distances = sorted_site_distances[:-vacancies]

    rms_atom_to_site = float(np.sqrt(np.mean(atom_to_site_distances ** 2)))
    rms_nonvac_site_to_atom = float(np.sqrt(np.mean(nonvac_site_distances ** 2)))
    score = rms_atom_to_site + rms_nonvac_site_to_atom

    return {
        "score": float(score),
        "site_to_atom_distances": site_to_atom_distances,
    }


def auto_detect_moving_atom_and_direction(
    poscar_file,
    vacancies=1,
    allowed_types=("sc", "bcc", "fcc", "diamond"),
    max_rep=10,
    tie_tol=1e-6,
):
    """Detect the moving atom (nearest to a vacancy) and its cartesian direction
    toward that vacancy by fitting the POSCAR against ideal lattice templates.

    Returns (moving_index_0_based, unit_direction_cart, crystal_type, reps).
    The atom index is already 0-based (matches this codebase's convention)."""
    structure = Poscar.from_file(poscar_file, check_for_potcar=False).structure
    lattice = np.array(structure.lattice.matrix, dtype=float)
    frac_coords = np.array(structure.frac_coords, dtype=float) % 1.0
    natoms = len(frac_coords)

    candidates = []
    for crystal_type in allowed_types:
        basis = _BASIS_LIBRARY[crystal_type]
        nbasis = len(basis)
        reps_list = _possible_supercells(natoms, nbasis, vacancies=vacancies, max_rep=max_rep)
        for reps in reps_list:
            ideal_sites = _generate_ideal_sites(basis, reps)
            score_info = _matching_score(frac_coords, lattice, ideal_sites, vacancies=vacancies)
            candidates.append({
                "crystal_type": crystal_type,
                "reps": reps,
                "ideal_sites": ideal_sites,
                **score_info,
            })

    if not candidates:
        raise ValueError(
            "Auto-detection failed: no compatible crystal structure found. "
            "Specify --moving-indices and --orient-atom-direction manually."
        )

    candidates.sort(key=lambda x: x["score"])
    best = candidates[0]
    ideal_sites = best["ideal_sites"]
    site_to_atom_distances = best["site_to_atom_distances"]

    vacancy_site_index = int(np.argmax(site_to_atom_distances))
    vacancy_frac = ideal_sites[vacancy_site_index]

    vectors_frac = np.array([_pbc_diff(vacancy_frac, atom) for atom in frac_coords])
    vectors_cart = vectors_frac @ lattice
    distances = np.linalg.norm(vectors_cart, axis=1)
    min_distance = float(np.min(distances))
    nearest_indices = np.where(np.abs(distances - min_distance) <= tie_tol)[0]

    nearest_idx = int(nearest_indices[0])
    direction_cart = vectors_cart[nearest_idx]
    unit_direction = direction_cart / np.linalg.norm(direction_cart)

    return nearest_idx, unit_direction, best["crystal_type"], best["reps"]


def setup_logging():
    """Set up logging."""
    logging.getLogger().handlers.clear()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    log_file = get_output_path('logs', 'dual_gp_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logging.info("Logging system initialized")


def get_mass_vector(structure):
    """Return mass array for all atoms."""
    atom_masses = np.array([
        Element(site.specie.symbol).atomic_mass
        for site in structure
    ], dtype=float)
    return np.repeat(atom_masses, 3)  # Repeat for x,y,z


def find_nearest_neighbors(structure, center_indices, cutoff_1nn=None, cutoff_2nn=None, auto_detect=True):
    """Find nearest neighbors of specified atoms."""
    from collections import defaultdict

    positions = structure.cart_coords
    n_atoms = len(positions)

    all_neighbors = defaultdict(list)

    for center_idx in center_indices:
        center_pos = positions[center_idx]
        distances = []

        for i in range(n_atoms):
            if i != center_idx:
                dist = np.linalg.norm(positions[i] - center_pos)
                distances.append((dist, i))

        distances.sort()

        if auto_detect and (cutoff_1nn is None or cutoff_2nn is None):
            dist_values = [d[0] for d in distances]

            gaps = []
            for i in range(len(dist_values)-1):
                gap = dist_values[i+1] - dist_values[i]
                if gap > 0.5:
                    gaps.append((i, gap, dist_values[i], dist_values[i+1]))

            if cutoff_1nn is None and len(gaps) >= 1:
                cutoff_1nn = (gaps[0][2] + gaps[0][3]) / 2
                print(f"Auto-detected 1st NN cutoff: {cutoff_1nn:.2f} Å")

            if cutoff_2nn is None and len(gaps) >= 2:
                cutoff_2nn = (gaps[1][2] + gaps[1][3]) / 2
                print(f"Auto-detected 2nd NN cutoff: {cutoff_2nn:.2f} Å")

        if cutoff_1nn is None:
            cutoff_1nn = 3.5
        if cutoff_2nn is None:
            cutoff_2nn = 5.0

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

    all_1nn = set()
    all_2nn = set()

    for center_idx in center_indices:
        all_1nn.update(all_neighbors[center_idx]['1nn'])
        all_2nn.update(all_neighbors[center_idx]['2nn'])

    all_2nn = all_2nn - all_1nn - set(center_indices)
    all_1nn = all_1nn - set(center_indices)

    return {
        'first_neighbors': sorted(list(all_1nn)),
        'second_neighbors': sorted(list(all_2nn)),
        'details': all_neighbors
    }

# Detect if we're on macOS and handle JAX accordingly
IS_MACOS = platform.system() == 'Darwin'
if IS_MACOS:
    # Mock JAX to avoid AVX instruction errors on Apple Silicon with x86 Python
    try:
        import jax
        # JAX is available, check if it works
        jax.devices()
    except (ImportError, RuntimeError) as e:
        if "AVX instructions" in str(e) or "jax" not in sys.modules:
            # Need to mock JAX with a more complete mock
            from unittest.mock import MagicMock
            
            # Create a more complete JAX mock
            jax_mock = MagicMock()
            jax_mock.__version__ = "0.0.0 (mocked)"
            jax_mock.devices = MagicMock(return_value=[])
            jax_mock.numpy = MagicMock()
            jax_mock.config = MagicMock()
            
            sys.modules['jax'] = jax_mock
            sys.modules['jax.numpy'] = jax_mock.numpy
            sys.modules['jax.config'] = jax_mock.config
            sys.modules['jaxlib'] = MagicMock()
            print("Note: Running on macOS, JAX mocked for compatibility")


def dual_gp_search_improved(poscar_file: str, system_params: dict) -> tuple:
    """Run dimer search with enhanced dual GP approach."""
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info("Starting enhanced dual GP dimer search")
    
    # Check if this is a continuation
    is_continuation = system_params.get('continuation', False)
    
    if is_continuation:
        logging.info("Continuation run - preserving existing data")
    else:
        # Clean up previous runs
        cleanup()

    # Create results directory in output structure
    results_dir = get_output_path('results')
    setup_logging()
    
    try:
        # Setup structure
        structure = Poscar.from_file(poscar_file, check_for_potcar=False).structure
        
        # Get mass vector
        mass = get_mass_vector(structure)
        
        # Create VASP manager
        work_dir = os.path.dirname(os.path.abspath(poscar_file))
        poscar_path = os.path.abspath(poscar_file)
        
        execution_mode = system_params.get('execution_mode', 'mpi')
        mpi_command = system_params.get('mpi_command', None)
        vasp_command = system_params.get('vasp_command', 'vasp_gam')
        eam_potential_file = system_params.get('eam_potential_file', None)
        kim_model = system_params.get('kim_model', 'EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000')
        # If an EAM potential file is provided for a multi-component system (e.g. .eam.alloy),
        # don't use the default KIM model since it may not support all species
        if eam_potential_file and '.alloy' in eam_potential_file and kim_model == 'EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000':
            kim_model = None
        
        print(f"\nInitializing calculations with execution mode: {execution_mode}")
        
        vasp_mgr = VASPManager(
            base_dir=get_output_path('vasp_runs'),
            user_poscar_path=poscar_path,
            execution_mode=execution_mode,
            mpi_command=mpi_command,
            vasp_command=vasp_command,
            eam_potential_file=eam_potential_file,
            skip_thermal=False,  # We DO need thermal sampling for GP1
            kim_model_name=kim_model,
            parallel_eam=system_params.get('parallel_eam', False),
            eam_n_workers=system_params.get('eam_n_workers'),
            use_gpu=system_params.get('use_gpu', False),
            gpu_fallback=not system_params.get('no_gpu_fallback', False),
            vasp_input_dir=system_params.get('vasp_input_dir'),
            temperature=system_params.get('temperature'),
        )
        
        # Get moving indices
        # Set activation_radius to 10 for NEW_v24
        activation_radius = 10.0
        moving_indices = system_params.get('moving_indices', [0])
        
        local_pes = VASPInterface(
            vasp_manager=vasp_mgr,
            poscar_file=poscar_file,
            activation_radius=activation_radius,
            moving_indices=moving_indices
        )
        
        # Get initial position (full system)
        initial_position = structure.cart_coords.flatten()
        
        # Create enhanced walker
        if not is_continuation:
            walker = WalkerDualGPImproved(
                initial_position=initial_position,
                local_pes=local_pes,
                force_constants_file=system_params["force_constants_file"],
                POSCAR_file=poscar_file,
                temperature=system_params["temperature"],
                mass=mass,
                num_snapshots=system_params["num_snapshots"],
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
                # GP1 noise modeling options
                gp1_noise_model=system_params.get("gp1_noise_model", "fixed"),
                gp1_student_t_df=system_params.get("gp1_student_t_df", 2.0),
                gp1_use_adaptive_df=system_params.get("gp1_use_adaptive_df", False),
                gp1_adaptive_df_start_iter=system_params.get("gp1_adaptive_df_start_iter", 15),
                gp1_adaptive_df_end_iter=system_params.get("gp1_adaptive_df_end_iter", 25),
                gp1_adaptive_df_target=system_params.get("gp1_adaptive_df_target", 1.0),
                gp1_remove_outliers=system_params.get("gp1_remove_outliers", False),
                gp1_outlier_threshold=system_params.get("gp1_outlier_threshold", 5.0),
                # Options
                initrot_nogp=system_params.get("initrot_nogp", False),
                inittrans_nogp=system_params.get("inittrans_nogp", False),
                eval_image1=system_params.get("eval_image1", False),
                num_bigiter_initloc=system_params.get("num_bigiter_initloc", np.inf),
                num_bigiter_initparam=system_params.get("num_bigiter_initparam", np.inf),
                # GPU options
                use_gpu=system_params.get("use_gpu", False),
                parallel_eam=system_params.get("parallel_eam", False),
                # Other parameters
                verbose=system_params["verbose"],
                checkpoint_interval=system_params.get('checkpoint_interval', 1),
                model_type=system_params.get("model_type", "MultitaskGPModel_rbf_atomic"),
                # NEW ENHANCED PARAMETERS (NEW_v24: No trust region)
                enable_oscillation_detection=system_params.get("enable_oscillation_detection", True),
                oscillation_memory=system_params.get("oscillation_memory", 5),
                oscillation_threshold=system_params.get("oscillation_threshold", 0.02),
                enable_force_minimum_detection=system_params.get("enable_force_minimum_detection", True),
                force_minimum_window=system_params.get("force_minimum_window", 5),
                force_minimum_increase_threshold=system_params.get("force_minimum_increase_threshold", 1.5),
                relaxed_saddle_criteria=system_params.get("relaxed_saddle_criteria", 0.1),
                enable_adaptive_convergence=system_params.get("enable_adaptive_convergence", True),
                gp1_fallback_to_averaging=system_params.get("gp1_fallback_to_averaging", True),
                gp1_max_retries=system_params.get("gp1_max_retries", 3),
                validate_parameters=system_params.get("validate_parameters", True),
            )
            
            # Set initial orientation if specified
            initial_orient_method = system_params.get('initial_orient_method', 'auto')
            if initial_orient_method == 'manual':
                manual_orient = system_params.get('manual_orient', None)
                if manual_orient is not None:
                    if 'manual_orient' in system_params:
                        walker.set_initial_orientation(np.array(system_params['manual_orient']))
        else:
            # Continuation mode - create walker and restore from checkpoint
            walker = WalkerDualGPImproved(
                initial_position=initial_position,
                local_pes=local_pes,
                force_constants_file=system_params["force_constants_file"],
                POSCAR_file=poscar_file,
                temperature=system_params["temperature"],
                mass=mass,
                num_snapshots=system_params["num_snapshots"],
                max_dimer_steps=1,  # Will be updated after loading checkpoint
                # All parameters from system_params
                **{k: v for k, v in system_params.items() 
                   if k not in ['force_constants_file', 'temperature', 'num_snapshots', 'max_dimer_steps']}
            )
            
            # Load checkpoint
            checkpoint = walker.load_checkpoint()
            
            # Update max steps for continuation
            steps_completed = walker.bigiter
            additional_steps = system_params["max_dimer_steps"]
            walker.max_dimer_steps = steps_completed + additional_steps
            
            logging.info(f"Continuing from step {steps_completed}")
            logging.info(f"Will run {additional_steps} additional steps")
            
            # NEW_v24: No trust region logging
        
        # Run search
        logging.info("Starting enhanced walker run")
        final_pos, final_energy, final_forces = walker.run()
        
        # Save results with enhanced information
        import pickle
        results = {
            'final_position': final_pos,
            'final_energy': final_energy,
            'final_forces': final_forces,
            'converged': walker.converged,
            'convergence_type': walker.convergence_type,
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
            'n_atoms': walker.n_atoms,
            'thermal_noise': walker.thermal_noise,
            'temperature': walker.temperature,
            # Enhanced information
            'oscillation_events': walker.oscillation_events,
            'force_minimum_events': walker.force_minimum_events,
            'gp1_training_failures': walker.gp1_training_failures,
            'min_force_achieved': walker.min_force_achieved,
        }
        
        results_dir = get_output_path('results')
        with open(os.path.join(results_dir, 'dual_gp_results_enhanced.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Also save to dimer_runs directory
        dimer_runs_dir = get_output_path('dimer_runs')
        
        # Save dimer trajectory
        trajectory_data = {
            'positions': walker.R_all,
            'energies': walker.E_all,
            'forces': walker.G_all,
            'E_R_acc': walker.E_R_acc,
            'maxF_R_acc': walker.maxF_R_acc,
            'E_R_gp': walker.E_R_gp,
            'maxF_R_gp': walker.maxF_R_gp,
            'table_history': walker.table_history,
            'convergence_type': walker.convergence_type,
            'converged': walker.converged,
            'total_iterations': walker.bigiter,
            'total_evaluations': walker.obs_total
        }
        
        with open(os.path.join(dimer_runs_dir, 'trajectory.pkl'), 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        # Save trajectory in text format for easy viewing
        with open(os.path.join(dimer_runs_dir, 'trajectory.dat'), 'w') as f:
            f.write("# Dimer Trajectory\n")
            f.write("# Step  Energy(eV)  MaxForce(eV/A)  GP2_Energy(eV)  GP2_MaxForce(eV/A)\n")
            for i in range(len(walker.E_R_acc)):
                energy = walker.E_R_acc[i]
                max_force = walker.maxF_R_acc[i]
                gp2_energy = walker.E_R_gp[i] if i < len(walker.E_R_gp) else np.nan
                gp2_force = walker.maxF_R_gp[i] if i < len(walker.maxF_R_gp) else np.nan
                f.write(f"{i+1:5d}  {energy:12.6f}  {max_force:14.6f}  {gp2_energy:14.6f}  {gp2_force:17.6f}\n")
        
        # Save final results summary
        with open(os.path.join(dimer_runs_dir, 'results_summary.txt'), 'w') as f:
            f.write("Enhanced Dual GP Dimer Search Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Converged: {walker.converged}\n")
            f.write(f"Convergence type: {walker.convergence_type}\n")
            f.write(f"Total iterations: {walker.bigiter}\n")
            f.write(f"Total evaluations: {walker.obs_total}\n")
            f.write(f"Final energy: {final_energy:.6f} eV\n")
            f.write(f"Final max force: {np.max(np.abs(final_forces)):.6f} eV/Å\n")
            f.write(f"Final RMS force: {np.sqrt(np.mean(final_forces**2)):.6f} eV/Å\n")
            f.write(f"\nGP1 training failures: {walker.gp1_training_failures}\n")
            f.write(f"Oscillation events: {len(walker.oscillation_events)}\n")
            f.write(f"Force minimum events: {len(walker.force_minimum_events)}\n")
            f.write(f"Min force achieved: {walker.min_force_achieved:.6f} eV/Å\n")
            
            # NEW_v24: No trust region output
        
        # Save detailed table history
        if walker.table_history:
            with open(os.path.join(dimer_runs_dir, 'progress_table.txt'), 'w') as f:
                # Write header
                f.write("="*150 + "\n")
                f.write("| Step | E_GP1  | E_GP2  | E_Actual | F_GP1  | F_GP2  | F_Actual | Curv_dimer | E_diff_GP2_GP1 | E_diff_GP2_Actual | Inner Iters |\n")
                f.write("|      | (eV)   | (eV)   |   (eV)   | (eV/Å) | (eV/Å) |  (eV/Å)  |  (eV/Å²)   |     (eV)       |       (eV)        |             |\n")
                f.write("="*150 + "\n")
                
                # Write each entry
                for entry in walker.table_history:
                    step = entry.get('step', 0)
                    e_gp1 = entry.get('E_GP1', np.nan)
                    e_gp2 = entry.get('E_GP2', np.nan)
                    e_actual = entry.get('E_Actual', np.nan)
                    f_gp1 = entry.get('F_GP1', np.nan)
                    f_gp2 = entry.get('F_GP2', np.nan)
                    f_actual = entry.get('F_Actual', np.nan)
                    curv = entry.get('Curvature_dimer', np.nan)
                    e_diff_gp2_gp1 = entry.get('E_diff_GP2_GP1', np.nan)
                    e_diff_gp2_actual = entry.get('E_diff_GP2_Actual', np.nan)
                    inner_iters = entry.get('inner_iterations', 0)
                    
                    f.write(f"| {step:4d} | ")
                    f.write(f"{e_gp1:6.3f} | " if not np.isnan(e_gp1) else "   --- | ")
                    f.write(f"{e_gp2:6.3f} | " if not np.isnan(e_gp2) else "   --- | ")
                    f.write(f"{e_actual:8.3f} | ")
                    f.write(f"{f_gp1:6.4f} | " if not np.isnan(f_gp1) else "  ---  | ")
                    f.write(f"{f_gp2:6.4f} | " if not np.isnan(f_gp2) else "  ---  | ")
                    f.write(f"{f_actual:8.4f} | ")
                    f.write(f"{curv:10.5f} | " if not np.isnan(curv) else "     ---    | ")
                    f.write(f"{e_diff_gp2_gp1:14.6f} | " if not np.isnan(e_diff_gp2_gp1) else "      ---      | ")
                    f.write(f"{e_diff_gp2_actual:17.6f} | " if not np.isnan(e_diff_gp2_actual) else "        ---       | ")
                    f.write(f"{inner_iters:11d} |\n")
                
                f.write("="*150 + "\n")
        
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
        final_poscar.write_file(os.path.join(dimer_runs_dir, 'POSCAR_final'))
        
        
        return final_pos, final_energy, final_forces
        
    except Exception as e:
        logging.error(f"Error during enhanced dual GP search: {e}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Enhanced Dual GP Dimer Method with improvements from toy model experience',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required files
    parser.add_argument('--poscar-file', default='POSCAR',
                        help='Initial POSCAR structure (searched in inputs/ directory)')
    parser.add_argument('--force-constants-file', default='FORCE_CONSTANTS',
                        help='FORCE_CONSTANTS file for phonon modes (searched in inputs/ directory)')
    
    # Temperature and thermal sampling
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature (K) for thermal sampling')
    parser.add_argument('--num-snapshots', type=int, default=10,
                        help='Number of thermal snapshots per GP1 evaluation')
    
    # Physical parameters
    # REMOVED activation-radius for NEW_v24 - no confinement region
    parser.add_argument('--moving-indices', type=int, nargs='+', default=None,
                        help='Atom indices that can move (auto-detected from POSCAR if not provided)')
    
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
    
    # ENHANCED CONVERGENCE PARAMETERS
    parser.add_argument('--relaxed-saddle-criteria', type=float, default=0.1,
                        help='Relaxed force criteria for saddle points (eV/Å)')
    parser.add_argument('--enable-adaptive-convergence', action='store_true', default=True,
                        help='Use adaptive convergence thresholds')
    
    # GP convergence
    parser.add_argument('--divisor-T-dimer-gp', type=float, default=10.0,
                        help='Divisor for dynamic GP convergence threshold')
    
    # Relaxation phase
    parser.add_argument('--max-inner-iterations', type=int, default=1000,
                        help='Maximum iterations per relaxation phase')
    
    # TRUST REGION PARAMETERS
    parser.add_argument('--enable-trust-region', action='store_true', default=True,
                        help='Enable trust region logic during relaxation')
    parser.add_argument('--trust-region-initial', type=float, default=0.5,
                        help='Initial trust region radius')
    parser.add_argument('--trust-region-min', type=float, default=0.01,
                        help='Minimum allowed trust region radius')
    parser.add_argument('--trust-region-max', type=float, default=2.0,
                        help='Maximum allowed trust region radius')
    parser.add_argument('--trust-region-dramatic-threshold', type=float, default=0.1,
                        help='Reduction ratio threshold for dramatic shrink')
    parser.add_argument('--trust-region-expand-threshold', type=float, default=0.5,
                        help='Reduction ratio threshold for expansion')
    parser.add_argument('--trust-region-dramatic-factor', type=float, default=0.5,
                        help='Factor to shrink the trust region dramatically')
    parser.add_argument('--trust-region-moderate-factor', type=float, default=0.8,
                        help='Factor to shrink the trust region moderately')
    parser.add_argument('--trust-region-expand-factor', type=float, default=1.2,
                        help='Factor to expand the trust region')
    
    # OSCILLATION DETECTION
    parser.add_argument('--enable-oscillation-detection', action='store_true', default=True,
                        help='Enable oscillation detection')
    parser.add_argument('--oscillation-memory', type=int, default=5,
                        help='Number of positions to check for oscillation')
    parser.add_argument('--oscillation-threshold', type=float, default=0.02,
                        help='Distance threshold for oscillation detection (Å)')
    
    # FORCE MINIMUM DETECTION
    parser.add_argument('--enable-force-minimum-detection', action='store_true', default=False,
                        help='Enable force minimum detection')
    parser.add_argument('--force-minimum-window', type=int, default=5,
                        help='Window size for force minimum detection')
    parser.add_argument('--force-minimum-increase-threshold', type=float, default=1.5,
                        help='Force increase threshold for minimum detection')
    
    # GP1 ROBUSTNESS
    parser.add_argument('--gp1-fallback-to-averaging', action='store_true', default=True,
                        help='Fallback to thermal averaging if GP1 training fails')
    parser.add_argument('--gp1-max-retries', type=int, default=3,
                        help='Maximum retries for GP1 training')
    
    # PARAMETER VALIDATION
    parser.add_argument('--validate-parameters', action='store_true', default=True,
                        help='Validate parameters based on toy model insights')
    parser.add_argument('--no-validate-parameters', dest='validate_parameters', 
                        action='store_false',
                        help='Disable parameter validation warnings')
    
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
    
    # Noise model
    parser.add_argument('--gp1-noise-model', 
                    choices=['fixed', 'heteroscedastic', 'student_t'],
                    default='fixed',
                    help='Noise model for GP1')
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

    # Outlier removal
    parser.add_argument('--gp1-remove-outliers', action='store_true',
                        help='Enable outlier removal in thermal snapshots')
    parser.add_argument('--gp1-outlier-threshold', type=float, default=5.0,
                        help='MAD threshold for outlier detection')
    
    # Execution mode
    parser.add_argument('--execution-mode',
                        choices=['mock', 'mpi', 'vasp', 'slurm', 'stampede3', 'anvil', 'eam', 'gaop'],
                        default='mpi',
                        help='Execution mode')
    parser.add_argument('--mpi-command', default=None,
                        help='Custom MPI command')
    parser.add_argument('--vasp-command', default='vasp_gam',
                        help='VASP executable name')
    parser.add_argument('--vasp-input-dir', default=None,
                        help='Directory with VASP input files (INCAR, KPOINTS, POTCAR)')
    parser.add_argument('--eam-potential-file', default=None,
                        help='Path to EAM potential file')
    parser.add_argument('--parallel-eam', action='store_true',
                        help='Enable parallel execution of EAM calculations')
    parser.add_argument('--eam-n-workers', type=int, default=None,
                        help='Number of parallel workers for EAM (default: number of CPU cores)')
    parser.add_argument('--kim-model',
                        default='EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000',
                        help='KIM model name for EAM calculations')
    
    # GPU options
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration for GP training and inference')
    parser.add_argument('--no-gpu-fallback', action='store_true',
                        help='Fail if GPU is requested but unavailable')
    
    # Initial orientation
    parser.add_argument('--initial-orient-method', 
                        choices=['auto', 'random', 'manual'],
                        default='auto',
                        help='Method for setting initial dimer orientation')
    parser.add_argument('--manual-orient', type=float, nargs='+',
                        help='Manual initial orientation vector (must be same dimension as system)')
    parser.add_argument('--orient-atom-direction', type=str,
                        help='Atom index and direction, e.g., "52:-1,-1,-1" '
                             '(auto-detected from POSCAR if not provided)')
    
    # Nearest neighbors
    parser.add_argument('--include-neighbors', 
                        choices=['none', '1nn', '2nn', '1nn+2nn'],
                        default='none',
                        help='Include nearest neighbors in moving atoms')
    parser.add_argument('--nn-cutoff-1', type=float, default=None,
                        help='Cutoff distance for 1st nearest neighbors (Å)')
    parser.add_argument('--nn-cutoff-2', type=float, default=None,
                        help='Cutoff distance for 2nd nearest neighbors (Å)')
    
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
    
    force_constants_file = args.force_constants_file
    if not os.path.isabs(force_constants_file):
        fc_path = get_input_path(force_constants_file)
        if not os.path.exists(fc_path):
            # Backward compatibility
            if os.path.exists(force_constants_file):
                fc_path = os.path.abspath(force_constants_file)
                print(f"Warning: Found {force_constants_file} in current directory")
                print(f"Consider moving it to inputs/ directory")
            else:
                parser.error(f"File not found in inputs/: {force_constants_file}")
        else:
            force_constants_file = fc_path
    
    # Build system parameters
    system_params = {
        'force_constants_file': force_constants_file,
        'output_dir': args.output_dir,
        'run_name': args.run_name,
        'input_dir': args.input_dir,
        'temperature': args.temperature,
        'num_snapshots': args.num_snapshots,
        # 'activation_radius' removed for NEW_v24 - no confinement region,
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
        # Enhanced parameters
        'relaxed_saddle_criteria': args.relaxed_saddle_criteria,
        'enable_adaptive_convergence': args.enable_adaptive_convergence,
        'enable_trust_region': args.enable_trust_region,
        'trust_region_initial': args.trust_region_initial,
        'trust_region_min': args.trust_region_min,
        'trust_region_max': args.trust_region_max,
        'trust_region_dramatic_threshold': args.trust_region_dramatic_threshold,
        'trust_region_expand_threshold': args.trust_region_expand_threshold,
        'trust_region_dramatic_factor': args.trust_region_dramatic_factor,
        'trust_region_moderate_factor': args.trust_region_moderate_factor,
        'trust_region_expand_factor': args.trust_region_expand_factor,
        'enable_oscillation_detection': args.enable_oscillation_detection,
        'oscillation_memory': args.oscillation_memory,
        'oscillation_threshold': args.oscillation_threshold,
        'enable_force_minimum_detection': args.enable_force_minimum_detection,
        'force_minimum_window': args.force_minimum_window,
        'force_minimum_increase_threshold': args.force_minimum_increase_threshold,
        'gp1_fallback_to_averaging': args.gp1_fallback_to_averaging,
        'gp1_max_retries': args.gp1_max_retries,
        'validate_parameters': args.validate_parameters,
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
        'vasp_input_dir': args.vasp_input_dir,
        'eam_potential_file': args.eam_potential_file,
        'continuation': args.continuation,
        'checkpoint_interval': args.checkpoint_interval,
        'kim_model': args.kim_model,
        'gp1_noise_model': args.gp1_noise_model,
        'gp1_student_t_df': args.gp1_student_t_df,
        'gp1_use_adaptive_df': args.gp1_use_adaptive_df,
        'gp1_adaptive_df_start_iter': args.gp1_adaptive_df_start_iter,
        'gp1_adaptive_df_end_iter': args.gp1_adaptive_df_end_iter,
        'gp1_adaptive_df_target': args.gp1_adaptive_df_target,
        'gp1_remove_outliers': args.gp1_remove_outliers,
        'gp1_outlier_threshold': args.gp1_outlier_threshold,
        'parallel_eam': args.parallel_eam,
        'eam_n_workers': args.eam_n_workers,
        'use_gpu': args.gpu,
        'no_gpu_fallback': args.no_gpu_fallback,
    }
    
    structure = Poscar.from_file(poscar_file, check_for_potcar=False).structure

    # Auto-detect moving atom and/or initial direction from POSCAR if not given.
    # The detector locates a vacancy in a crystalline template and returns the
    # nearest atom (0-based index) and its unit-vector direction toward the vacancy.
    need_auto_moving = args.moving_indices is None
    need_auto_orient = (
        args.orient_atom_direction is None
        and args.manual_orient is None
        and args.initial_orient_method == 'auto'
    )
    if need_auto_moving or need_auto_orient:
        auto_idx, unit_dir, crystal_type, reps = auto_detect_moving_atom_and_direction(poscar_file)
        print(f"\nAuto-detected from POSCAR: {crystal_type} {reps}, "
              f"moving atom index = {auto_idx}, "
              f"unit direction (cart) = ({unit_dir[0]:.6f}, {unit_dir[1]:.6f}, {unit_dir[2]:.6f})")
        if need_auto_moving:
            args.moving_indices = [auto_idx]
        if need_auto_orient:
            args.orient_atom_direction = (
                f"{auto_idx}:{unit_dir[0]:.10f},{unit_dir[1]:.10f},{unit_dir[2]:.10f}"
            )

    # Process moving indices with neighbors (same as run_gp2_dimer.py)
    moving_indices = args.moving_indices
    if args.include_neighbors != 'none':
        print(f"\nFinding nearest neighbors for atoms: {moving_indices}")
        
        neighbors = find_nearest_neighbors(
            structure, 
            moving_indices,
            cutoff_1nn=args.nn_cutoff_1,
            cutoff_2nn=args.nn_cutoff_2,
            auto_detect=True
        )
        
        # Add neighbors based on user choice
        if args.include_neighbors in ['1nn', '1nn+2nn']:
            moving_indices.extend(neighbors['first_neighbors'])
            print(f"Added {len(neighbors['first_neighbors'])} first nearest neighbors: {neighbors['first_neighbors']}")
        
        if args.include_neighbors in ['2nn', '1nn+2nn']:
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
    
    # Process manual orientation
    if args.orient_atom_direction:
        parts = args.orient_atom_direction.split(':')
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
    
    return poscar_file, system_params


if __name__ == "__main__":
    try:
        poscar_file, system_params = parse_arguments()
        
        # Save run metadata
        OutputManager.save_run_metadata({
            'script': 'run_tad.py',
            'parameters': system_params,
            'poscar_file': poscar_file
        })
        
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
        
        # Check JAX GPU support for parallel EAM
        if system_params.get('parallel_eam', False):
            try:
                import jax
                print(f"\nJAX GPU Support:")
                # Safely get version (might be mocked)
                if hasattr(jax, '__version__'):
                    print(f"  JAX version: {jax.__version__}")
                elif IS_MACOS:
                    print(f"  JAX mocked for macOS compatibility")
                
                # Check devices if not mocked
                if hasattr(jax, 'devices') and callable(jax.devices):
                    try:
                        devices = jax.devices()
                        print(f"  Available devices: {devices}")
                        
                        # Check for GPU devices
                        gpu_devices = [d for d in devices if hasattr(d, 'platform') and d.platform == 'gpu']
                        if gpu_devices:
                            print(f"  GPU devices available: {len(gpu_devices)}")
                            print(f"  Using GPU-accelerated EAM calculations")
                        else:
                            print("  No GPU devices found for JAX, using CPU for EAM")
                    except:
                        if IS_MACOS:
                            print("  Using CPU-parallel EAM (macOS)")
                        else:
                            print("  JAX device detection failed, using CPU")
                elif IS_MACOS:
                    print("  Using CPU-parallel EAM executor (macOS)")
            except ImportError:
                print("  JAX not available, using standard EAM executor")
        
        print("\nStarting Enhanced Dual GP dimer search with parameters:")
        print("\nThermal Sampling:")
        print(f"  Temperature: {system_params['temperature']} K")
        print(f"  Snapshots per evaluation: {system_params['num_snapshots']}")
        print(f"  Force constants file: {system_params['force_constants_file']}")
        
        print("\nStopping Criteria:")
        print(f"  Max displacement: {system_params['disp_max']} Å")
        print(f"  Inter-atomic ratio limit: {system_params['ratio_at_limit']}")
        print(f"  Standard convergence: {system_params['dimer_stopping_criteria']} eV/Å")
        print(f"  Relaxed saddle convergence: {system_params['relaxed_saddle_criteria']} eV/Å")
        
        print("\nDimer Parameters:")
        print(f"  Max outer iterations: {system_params['max_dimer_steps']}")
        print(f"  Max inner iterations: {system_params['max_inner_iterations']}")
        print(f"  Step size: {system_params['step_size']} Å (max: {system_params['max_step_size']} Å)")
        
        print("\nEnhanced Features:")
        # NEW_v24: No trust region output
        print(f"  Oscillation detection: {'ENABLED' if system_params['enable_oscillation_detection'] else 'DISABLED'}")
        print(f"  Force minimum detection: {'ENABLED' if system_params['enable_force_minimum_detection'] else 'DISABLED'}")
        print(f"  Adaptive convergence: {'ENABLED' if system_params['enable_adaptive_convergence'] else 'DISABLED'}")
        print(f"  GP1 fallback: {'ENABLED' if system_params['gp1_fallback_to_averaging'] else 'DISABLED'}")
        print(f"  Parameter validation: {'ENABLED' if system_params['validate_parameters'] else 'DISABLED'}")
        
        print("\nGP Parameters:")
        print(f"  Model type: {system_params['model_type']}")
        print(f"  GP convergence divisor: {system_params['divisor_T_dimer_gp']}")
        print(f"  Initial rotations without GP: {system_params['initrot_nogp']}")
        print(f"  Initial translation without GP: {system_params['inittrans_nogp']}")
        
        final_pos, final_energy, final_forces = dual_gp_search_improved(
            poscar_file=poscar_file,
            system_params=system_params
        )
        
        print("\nEnhanced Dual GP dimer search completed successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
