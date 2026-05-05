#!/usr/bin/env python
"""Run NEB method with GP2 surrogate model for acceleration."""

import argparse
import os
import sys
import logging
import numpy as np
import pickle
from walker_gp2_neb import WalkerGP2NEB
from pymatgen.core.periodic_table import Element
from vasp_manager import VASPManager, cleanup
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
from scipy.interpolate import interp1d
from output_manager import OutputManager, get_output_path, get_input_path


def setup_logging():
    """Set up logging."""
    logging.getLogger().handlers.clear()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    log_file = get_output_path('logs', 'gp2_neb_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")


def find_nearest_neighbors(structure,
                           center_indices,
                           cutoff_1nn=None,
                           cutoff_2nn=None,
                           auto_detect=True,
                           rcut=8.0):
    """
    Return unique 1st / 2nd nearest neighbours for the chosen sites,
    honouring periodic boundaries.

    Returns
    -------
    {
        'first_neighbors':  [indices],
        'second_neighbors': [indices],
        'details': {centre: {'1nn': [...],
                             '2nn': [...],
                             'cutoffs': (c1, c2)}}
    }
    """
    from collections import defaultdict
    import numpy as np

    # Helper: extract (distance, index) irrespective of neighbour layout
    def _dist_idx(nbr):
        """
        Works with:
        • PeriodicNeighbor  (has .nn_distance, .index)
        • tuple forms       (site, dist, image[, idx])
        """
        # PeriodicNeighbor
        if hasattr(nbr, "nn_distance"):            # modern object
            return nbr.nn_distance, nbr.index

        # tuple layouts
        if isinstance(nbr, tuple):
            if len(nbr) == 4:                      # (site, dist, image, idx)
                return nbr[1], nbr[3]
            if len(nbr) == 3:
                # third entry = idx  OR  image-vector
                if isinstance(nbr[2], int):
                    return nbr[1], nbr[2]          # (site, dist, idx)
                return nbr[1], structure.index(nbr[0])  # (site, dist, image)
        raise ValueError(f"Cannot interpret neighbour: {nbr!r}")

    # Neighbour search radius
    search_r = (rcut if auto_detect and
                (cutoff_1nn is None or cutoff_2nn is None)
                else max(cutoff_1nn or 0, cutoff_2nn or 0, 1.0))

    all_ngh = structure.get_all_neighbors(search_r, include_index=True)

    details = defaultdict(dict)
    union_1nn, union_2nn = set(), set()

    for idx in center_indices:
        neigh = sorted(all_ngh[idx], key=lambda n: _dist_idx(n)[0])
        dists = np.array([_dist_idx(n)[0] for n in neigh])

        # detect cut-offs per centre atom (if requested)
        c1, c2 = cutoff_1nn, cutoff_2nn
        if auto_detect and (c1 is None or c2 is None):
            gaps = np.diff(dists[:20])
            big  = np.where(gaps / dists[:19] > 0.30)[0]    # >30 % jump
            if c1 is None and len(big) >= 1:
                c1 = 0.5 * (dists[big[0]] + dists[big[0] + 1])
            if c2 is None and len(big) >= 2:
                c2 = 0.5 * (dists[big[1]] + dists[big[1] + 1])

        c1 = c1 or 3.5       # generic defaults (metals)
        c2 = c2 or 5.0

        # classify neighbours 
        n1, n2 = [], []
        for n in neigh:
            dist, j = _dist_idx(n)
            if dist <= c1:
                n1.append(j)
            elif dist <= c2:
                n2.append(j)
            else:
                break        # sorted list → beyond 2-nn shell

        details[idx]['1nn']     = n1
        details[idx]['2nn']     = n2
        details[idx]['cutoffs'] = (c1, c2)

        union_1nn.update(n1)
        union_2nn.update(n2)

    # remove overlaps and centres themselves
    union_1nn.difference_update(center_indices)
    union_2nn.difference_update(center_indices, union_1nn)

    return {
        'first_neighbors':  sorted(union_1nn),
        'second_neighbors': sorted(union_2nn),
        'details':          details
    }


def generate_initial_path(initial_poscar: str, final_poscar: str, n_images: int, 
                         interpolation_method: str = 'linear') -> np.ndarray:
    """Generate initial path between two structures.
    
    Args:
        initial_poscar: Path to initial POSCAR
        final_poscar: Path to final POSCAR
        n_images: Total number of images (including endpoints)
        interpolation_method: 'linear' or 'idpp'
        
    Returns:
        Initial path as (n_images, n_atoms*3) array in CARTESIAN coordinates
    """
    # Read structures
    initial_struct = Poscar.from_file(initial_poscar, check_for_potcar=False).structure
    final_struct = Poscar.from_file(final_poscar, check_for_potcar=False).structure
    
    # Ensure structures are compatible
    if len(initial_struct) != len(final_struct):
        raise ValueError("Initial and final structures must have the same number of atoms")
    
    # Get fractional coordinates
    initial_frac = initial_struct.frac_coords
    final_frac = final_struct.frac_coords
    
    # Analyze atomic movements
    print(f"\nAnalyzing atomic movements:")
    n_atoms = len(initial_struct)
    moving_atoms = []
    
    for i in range(n_atoms):
        # Calculate displacement considering periodic boundary conditions
        frac_diff = final_frac[i] - initial_frac[i]
        # Apply minimum image convention
        frac_diff = frac_diff - np.round(frac_diff)
        cart_diff = initial_struct.lattice.get_cartesian_coords(frac_diff)
        dist = np.linalg.norm(cart_diff)
        
        if dist > 0.01:  # Atoms that move significantly
            moving_atoms.append(i)
            print(f"  Atom {i}: moves {dist:.4f} Å")
    
    print(f"\nTotal moving atoms: {len(moving_atoms)}")
    
    # Generate path
    path = np.zeros((n_images, n_atoms * 3))
    
    if interpolation_method == 'linear':
        print(f"\nGenerating linear interpolation path with {n_images} images...")
        
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
            path[img, :] = cart_coords.flatten()
    
    elif interpolation_method == 'idpp':
        # Start with linear interpolation then add perturbation
        print(f"\nGenerating IDPP interpolation...")
        
        # First do linear interpolation
        for img in range(n_images):
            t = img / (n_images - 1)
            
            frac_coords = np.zeros((n_atoms, 3))
            for atom in range(n_atoms):
                disp = final_frac[atom] - initial_frac[atom]
                disp = disp - np.round(disp)
                frac_coords[atom] = initial_frac[atom] + t * disp
                frac_coords[atom] = frac_coords[atom] - np.floor(frac_coords[atom])
            
            cart_coords = initial_struct.lattice.get_cartesian_coords(frac_coords)
            path[img, :] = cart_coords.flatten()
            
            # Add small random perturbation to intermediate images
            if 0 < img < n_images - 1:
                perturbation = 0.01 * np.random.randn(len(path[img, :]))
                path[img, :] += perturbation
    
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    
    # Verify path
    print(f"\nPath verification:")
    for img in range(n_images):
        if img > 0:
            dist = np.linalg.norm(path[img] - path[img-1])
            print(f"  Image {img-1} -> {img}: step = {dist:.4f} Å")
    
    # Check for atomic overlaps
    print(f"\nChecking for atomic overlaps...")
    min_allowed_dist = 1.5  # Angstroms
    
    for img in range(n_images):
        coords = path[img, :].reshape(-1, 3)
        min_dist = float('inf')
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[j] - coords[i])
                min_dist = min(min_dist, dist)
        
        if min_dist < min_allowed_dist:
            print(f"  WARNING: Image {img} has atoms closer than {min_allowed_dist} Å (min dist = {min_dist:.3f} Å)")
        else:
            print(f"  Image {img}: OK (min dist = {min_dist:.3f} Å)")
    
    return path


class VASPManagerNEB(VASPManager):
    """Extended VASP Manager with proper NEB directory support."""
    
    def __init__(self, **kwargs):
        # Extract base_dir to ensure we use the right path
        base_dir = kwargs.get('base_dir', 'vasp_runs')
        # Override base_dir to ensure NEB-specific directory
        kwargs['base_dir'] = base_dir
        
        super().__init__(**kwargs)
        
        # Override the run directories to ensure NEB uses correct path
        self.run_dirs["neb"] = os.path.join(self.base_dir, "neb_runs")
        # Directories are already created by VASPManager initialization
        
        # NEB-specific counters
        self.neb_path_counter = 0
        self.neb_image_counters = {}
        
        print(f"VASPManagerNEB initialized with base_dir: {self.base_dir}")
        print(f"NEB run directory: {self.run_dirs['neb']}")
    
    def setup_neb_run(self, image_positions: np.ndarray, path_id: int, image_id: int) -> str:
        """Set up a VASP run for a single NEB image.
        
        Args:
            image_positions: Positions for this image
            path_id: NEB path iteration number
            image_id: Image number in the path
            
        Returns:
            Run directory path
        """
        # Create directory structure: neb_runs/path_XXX/image_YY/
        path_dir = os.path.join(self.run_dirs["neb"], f"path_{path_id:03d}")
        image_dir = os.path.join(path_dir, f"image_{image_id:02d}")
        os.makedirs(image_dir, exist_ok=True)
        
        # For EAM, just write POSCAR and submit
        if self.execution_mode == "eam":
            from vasp_manager import read_structure, get_structure_from_positions, write_structure
            if self.user_poscar_path and os.path.exists(self.user_poscar_path):
                template_structure = read_structure(self.user_poscar_path)[0]
            else:
                n_atoms = image_positions.size // 3
                from pymatgen.core import Structure, Lattice
                lattice = Lattice.cubic(10.0 * (n_atoms ** (1/3)))
                species = ["Mo"] * n_atoms
                coords = image_positions.reshape(-1, 3)
                template_structure = Structure(lattice, species, coords, coords_are_cartesian=True)
            
            new_structure = get_structure_from_positions(image_positions, template_structure)
            write_structure(os.path.join(image_dir, 'POSCAR'), new_structure)
            
            # Create minimal placeholder files
            for fname in ['INCAR', 'KPOINTS', 'POTCAR']:
                fpath = os.path.join(image_dir, fname)
                if not os.path.exists(fpath):
                    with open(fpath, 'w') as f:
                        f.write(f"# Placeholder {fname} for EAM calculation\n")
            
            return image_dir
        
        # For non-EAM calculations
        from vasp_manager import read_structure, get_structure_from_positions, write_structure
        
        if self.user_poscar_path and os.path.exists(self.user_poscar_path):
            template_structure = read_structure(self.user_poscar_path)[0]
            input_dir = os.path.dirname(self.user_poscar_path)
        else:
            raise ValueError("No valid user POSCAR provided for NEB run")
        
        new_structure = get_structure_from_positions(image_positions, template_structure)
        write_structure(os.path.join(image_dir, 'POSCAR'), new_structure)
        
        # Copy input files
        for fname in ['INCAR', 'POTCAR', 'KPOINTS', 'job.sh']:
            src = os.path.join(input_dir, fname)
            dst = os.path.join(image_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
        
        return image_dir
    
    def setup_and_run_vasp(self, positions: np.ndarray, run_type: str = "main", **kwargs) -> str:
        """Override to handle NEB runs with proper directory structure."""
        if run_type == "neb":
            # Extract path and image info from kwargs
            path_id = kwargs.get('path_id', self.neb_path_counter)
            image_id = kwargs.get('image_id', 0)
            
            # Set up the run directory
            run_dir = self.setup_neb_run(positions, path_id, image_id)
            
            # Submit the job
            job_info = self.executor.submit_job(run_dir, run_type)
            
            return run_dir
        else:
            # Use parent method for non-NEB runs
            return super().setup_and_run_vasp(positions, run_type, **kwargs)


class PureNEBInterface:
    """Interface for NEB calculations that properly routes to neb_runs directory."""
    
    def __init__(self, vasp_manager, poscar_file: str, n_atoms: int):
        """Initialize pure NEB interface.
        
        Args:
            vasp_manager: VASPManager instance
            poscar_file: Path to reference POSCAR
            n_atoms: Number of atoms in the system
        """
        self.vasp_manager = vasp_manager
        self.user_poscar_path = os.path.abspath(poscar_file)
        self.n_atoms = n_atoms
        self.current_run_dir = None
        self.current_path_id = 0
        
    def scaler_y_value(self, position: np.ndarray, is_thermal: bool = False, **kwargs) -> float:
        """Calculate energy at given position.
        
        For NEB, kwargs may contain path_id and image_id.
        """
        # Extract NEB-specific parameters
        path_id = kwargs.get('path_id', self.current_path_id)
        image_id = kwargs.get('image_id', None)
        
        if image_id is not None:
            # NEB image calculation - ensure it goes to neb_runs
            run_dir = self.vasp_manager.setup_and_run_vasp(
                positions=position,
                run_type='neb',  # This ensures it goes to neb_runs
                path_id=path_id,
                image_id=image_id
            )
            print(f"  NEB calculation: path_{path_id:03d}/image_{image_id:02d} -> {run_dir}")
        else:
            # Regular calculation
            run_dir = self.vasp_manager.setup_and_run_vasp(
                positions=position,
                run_type='main'
            )
            print(f"  Regular calculation -> {run_dir}")
        
        self.current_run_dir = run_dir
        self.current_path_id = path_id
        
        # Wait for completion
        vasp_run = self.vasp_manager.wait_for_completion(run_dir)
        
        # Return energy
        if not hasattr(vasp_run, 'energy'):
            return -123.456789  # Default mock energy
        return vasp_run.energy
    
    def first_derivative(self, position: np.ndarray, is_thermal: bool = False, **kwargs) -> np.ndarray:
        """Get forces at given position.
        
        IMPORTANT: This returns FORCES (F = -∇E), not gradients!
        """
        # Extract NEB-specific parameters
        path_id = kwargs.get('path_id', self.current_path_id)
        image_id = kwargs.get('image_id', None)
        
        if image_id is not None:
            # NEB image calculation - ensure it goes to neb_runs
            run_dir = self.vasp_manager.setup_and_run_vasp(
                positions=position,
                run_type='neb',  # This ensures it goes to neb_runs
                path_id=path_id,
                image_id=image_id
            )
        else:
            # Regular calculation
            run_dir = self.vasp_manager.setup_and_run_vasp(
                positions=position,
                run_type='main'
            )
        
        self.current_run_dir = run_dir
        self.current_path_id = path_id
        
        # Wait for completion and get forces
        run = self.vasp_manager.wait_for_completion(run_dir)
        
        # Return forces directly - DO NOT NEGATE!
        return run.flattened_forces


def gp2_neb_search(initial_poscar: str, final_poscar: str, system_params: dict) -> tuple:
    """Run NEB search with GP2 surrogate model."""
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info("Starting GP2-accelerated NEB search")
    
    # Check if this is a continuation
    is_continuation = system_params.get('continuation', False)
    
    if is_continuation:
        logging.info("Continuation run - preserving existing data")
    else:
        # Clean up previous runs
        cleanup()
    
    setup_logging()
    
    try:
        # Generate initial path
        n_images = system_params['n_images']
        
        if not is_continuation:
            neb_initial_path = generate_initial_path(
                initial_poscar, 
                final_poscar, 
                n_images,
                system_params.get('interpolation_method', 'linear')
            )
            logging.info(f"Generated initial path with {n_images} images")
        else:
            # For continuation, we'll load the path from checkpoint
            neb_initial_path = None
        
        # Handle input file paths
        if not os.path.isabs(initial_poscar):
            initial_path = get_input_path(initial_poscar)
            if not os.path.exists(initial_path):
                # Backward compatibility - check current directory
                if os.path.exists(initial_poscar):
                    initial_path = os.path.abspath(initial_poscar)
                    logging.warning(f"Found {initial_poscar} in current directory")
                    logging.warning("Consider moving it to inputs/ directory")
                else:
                    raise FileNotFoundError(f"Initial POSCAR file not found: {initial_poscar}")
        else:
            initial_path = initial_poscar
            
        if not os.path.isabs(final_poscar):
            final_path = get_input_path(final_poscar)
            if not os.path.exists(final_path):
                # Backward compatibility - check current directory
                if os.path.exists(final_poscar):
                    final_path = os.path.abspath(final_poscar)
                    logging.warning(f"Found {final_poscar} in current directory")
                    logging.warning("Consider moving it to inputs/ directory")
                else:
                    raise FileNotFoundError(f"Final POSCAR file not found: {final_poscar}")
        else:
            final_path = final_poscar
            
        # Update paths
        initial_poscar = initial_path
        final_poscar = final_path
        
        # Setup structure from initial POSCAR
        structure = Poscar.from_file(initial_poscar, check_for_potcar=False).structure
        
        # Create VASP manager with NEB support
        poscar_path = initial_poscar
        
        execution_mode = system_params.get('execution_mode', 'mpi')
        mpi_command = system_params.get('mpi_command', None)
        vasp_command = system_params.get('vasp_command', 'vasp_gam')
        eam_potential_file = system_params.get('eam_potential_file', None)
        kim_model = system_params.get('kim_model', 'EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000')
        
        print(f"\nInitializing calculations with execution mode: {execution_mode}")
        
        # Create VASPManagerNEB with proper base directory
        print(f"\nCreating VASPManagerNEB for NEB calculations...")
        vasp_mgr = VASPManagerNEB(
            base_dir=get_output_path('vasp_runs'),
            user_poscar_path=poscar_path,
            execution_mode=execution_mode,
            mpi_command=mpi_command,
            vasp_command=vasp_command,
            eam_potential_file=eam_potential_file,
            skip_thermal=True,
            kim_model_name=kim_model,
        )
        
        print(f"VASPManagerNEB created with directories:")
        for name, path in vasp_mgr.run_dirs.items():
            print(f"  {name}: {path}")
        
        # Get activation radius and moving indices
        activation_radius = system_params.get('activation_radius', 10.0)
        moving_indices = system_params.get('moving_indices', list(range(len(structure))))
        
        # CRITICAL: Ensure moving_indices is not empty
        if len(moving_indices) == 0:
            print("\nWARNING: No moving indices specified. Using all atoms as moving.")
            moving_indices = list(range(len(structure)))
        
        # Create VASP interface for atomic structure tracking
        from vasp_interface import VASPInterface
        vasp_interface = VASPInterface(
            vasp_manager=vasp_mgr,
            poscar_file=initial_poscar,
            activation_radius=activation_radius,
            moving_indices=moving_indices
        )
        
        # Create PureNEBInterface for actual NEB calculations
        neb_interface = PureNEBInterface(
            vasp_manager=vasp_mgr,
            poscar_file=initial_poscar,
            n_atoms=len(structure)
        )
        
        # Add atomic info methods to neb_interface
        neb_interface.atomic_structure = vasp_interface.atomic_structure
        neb_interface.get_atomic_info = vasp_interface.get_atomic_info
        
        # Use neb_interface as local_pes
        local_pes = neb_interface
        
        # Create walker
        if not is_continuation:
            walker = WalkerGP2NEB(
                initial_path=neb_initial_path,
                local_pes=local_pes,
                max_neb_steps=system_params["max_neb_steps"],
                # NEB parameters
                k_parallel=system_params.get("k_parallel", 1.0),
                k_perpendicular=system_params.get("k_perpendicular", 1.0),
                neb_convergence_threshold=system_params.get("neb_convergence_threshold", 0.1),
                ci_convergence_threshold=system_params.get("ci_convergence_threshold", 0.1),
                ci_activation_threshold=system_params.get("ci_activation_threshold_gp", 0.0),
                convergence_criterion=system_params.get("convergence_criterion", "max_force"),
                # GP convergence
                divisor_T_MEP_gp=system_params.get("divisor_T_MEP_gp", 10.0),
                max_inner_iterations=system_params.get("max_inner_iterations", 10000),
                # Stopping criteria
                disp_max=system_params.get("disp_max", 0.5),
                ratio_at_limit=system_params.get("ratio_at_limit", 2.0/3.0),
                # Translation parameters
                translation_method=system_params.get("translation_method", "qmvv"),
                step_size=system_params.get("step_size", 0.01),
                max_step_size=system_params.get("max_step_size", 0.2),
                # Other options
                num_bigiter_init=system_params.get("num_bigiter_init", 1),
                num_bigiter_hess=system_params.get("num_bigiter_hess", 0),
                eps_hess=system_params.get("eps_hess", 0.001),
                verbose=system_params["verbose"],
                checkpoint_interval=system_params.get('checkpoint_interval', 1),
                visualize=system_params.get('visualize', False),
                model_type=system_params.get("model_type", "MultitaskGPModel_rbf_atomic"),
            )
        else:
            # Create walker and restore from checkpoint
            walker = WalkerGP2NEB(
                initial_path=np.zeros((n_images, len(structure)*3)),  # Placeholder
                local_pes=local_pes,
                max_neb_steps=1,  # Will be updated
                verbose=system_params["verbose"]
            )
            
            # Load checkpoint
            checkpoint = walker.load_checkpoint()
            
            # Update max steps for continuation
            steps_completed = walker.bigiter
            additional_steps = system_params["max_neb_steps"]
            walker.max_neb_steps = steps_completed + additional_steps
            
            logging.info(f"Continuing from step {steps_completed}")
            logging.info(f"Will run {additional_steps} additional steps")
        
        # Run search
        logging.info("Starting walker run")
        final_path, final_energies, final_gradients, ci_index = walker.run()
        
        logging.info("Walker run completed")
        
        # Save results
        results = {
            'final_path': final_path,
            'final_energies': final_energies,
            'final_gradients': final_gradients,
            'converged': walker.converged,
            'convergence_criterion': walker.convergence_criterion,
            'outer_iterations': walker.bigiter,
            'inner_iterations': walker.E_R_gp.shape[1] if walker.E_R_gp.size > 0 else 0,
            'vasp_eval_count': walker.obs_total,
            'n_images': walker.n_images,
            'n_atoms': walker.n_atoms,
            'ci_index': ci_index,
            'R_all': walker.R_all,
            'E_all': walker.E_all,
            'G_all': walker.G_all,
            'E_R_acc': walker.E_R_acc,
            'normF_R_acc': walker.normF_R_acc,
            'table_history': walker.table_history
        }
        
        results_dir = get_output_path('results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'gp2_neb_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save all image structures
        for i in range(walker.n_images):
            image_coords = final_path[i, :].reshape(-1, 3)
            image_structure = Structure(
                lattice=structure.lattice,
                species=structure.species,
                coords=image_coords,
                coords_are_cartesian=True
            )
            image_poscar = Poscar(image_structure)
            image_poscar.write_file(os.path.join(results_dir, f'POSCAR_image_{i:02d}'))
        
        # Find saddle point
        saddle_idx = np.argmax(final_energies[1:-1]) + 1
        saddle_energy = final_energies[saddle_idx, 0]
        
        print(f"\nFinal results:")
        print(f"  Saddle point energy: {saddle_energy:.6f} eV")
        print(f"  Forward barrier: {saddle_energy - final_energies[0, 0]:.6f} eV")
        print(f"  Reverse barrier: {saddle_energy - final_energies[-1, 0]:.6f} eV")
        print(f"  Saddle point: Image {saddle_idx} (0-based)")
        print(f"  Converged: {walker.converged}")
        print(f"  Convergence criterion: {walker.convergence_criterion}")
        print(f"  Outer iterations: {walker.bigiter}")
        print(f"  Total VASP evaluations: {walker.obs_total}")
        print(f"  Total inner iterations: {walker.E_R_gp.shape[1] if walker.E_R_gp.size > 0 else 0}")
        if walker.obs_total > 0:
            speedup = walker.E_R_gp.shape[1] / walker.obs_total
            print(f"  Speedup factor: {speedup:.1f}x")
        print(f"\nStructures saved to: {os.path.join(results_dir, 'POSCAR_image_*')}")
        
        return final_path, final_energies, final_gradients
        
    except Exception as e:
        logging.error(f"Error during GP2 NEB search: {e}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='GP2-Accelerated NEB Method',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required files
    parser.add_argument('--initial-poscar', default='POSCAR_initial',
                        help='Initial structure POSCAR (searched in inputs/ directory)')
    parser.add_argument('--final-poscar', default='POSCAR_final',
                        help='Final structure POSCAR (searched in inputs/ directory)')
    
    # Physical parameters
    parser.add_argument('--activation-radius', type=float, default=10.0,
                        help='Activation radius (Å) for frozen atoms')
    parser.add_argument('--moving-indices', type=int, nargs='+', default=None,
                        help='Atom indices that can move (default: all)')
    # Neighbor inclusion
    parser.add_argument('--include-neighbors', 
                        choices=['none', '1nn', '2nn', '1nn+2nn'],
                        default='none',
                        help='Include nearest neighbors in moving atoms')
    parser.add_argument('--nn-cutoff-1', type=float, default=None,
                        help='Cutoff distance for 1st nearest neighbors (Å)')
    parser.add_argument('--nn-cutoff-2', type=float, default=None,
                        help='Cutoff distance for 2nd nearest neighbors (Å)')
    
    # NEB parameters
    parser.add_argument('--n-images', type=int, default=7,
                        help='Total number of images (including endpoints)')
    parser.add_argument('--interpolation-method', choices=['linear', 'idpp'], 
                        default='linear',
                        help='Method for generating initial path')
    parser.add_argument('--max-neb-steps', type=int, default=100,
                        help='Maximum number of outer iterations')
    parser.add_argument('--k-parallel', type=float, default=1.0,
                        help='Parallel spring constant')
    parser.add_argument('--k-perpendicular', type=float, default=1.0,
                        help='Perpendicular spring constant')
    parser.add_argument('--neb-convergence-threshold', type=float, default=0.1,
                        help='Force convergence threshold (eV/Å)')
    parser.add_argument('--ci-convergence-threshold', type=float, default=0.1,
                        help='Climbing image convergence threshold (eV/Å)')
    parser.add_argument('--ci-activation-threshold-gp', type=float, default=0.0,
                        help='Force threshold to activate climbing image on GP (0 = disabled)')
    parser.add_argument('--convergence-criterion', 
                        choices=['max_force', 'moving_atoms_only'],
                        default='max_force',
                        help='Convergence criterion for saddle point finding')
    
    # GP parameters
    parser.add_argument('--divisor-T-MEP-gp', type=float, default=10.0,
                        help='Divisor for dynamic GP convergence threshold')
    parser.add_argument('--max-inner-iterations', type=int, default=10000,
                        help='Maximum iterations per relaxation phase')
    
    # Stopping criteria
    parser.add_argument('--disp-max', type=float, default=0.5,
                        help='Maximum displacement from nearest observed point (relative to path length)')
    parser.add_argument('--ratio-at-limit', type=float, default=2.0/3.0,
                        help='Limit for inter-atomic distance ratio')
    
    # Translation method
    parser.add_argument('--translation-method', 
                        choices=['qmvv', 'lbfgs', 'fire'],
                        default='qmvv',
                        help='Method for moving images')
    parser.add_argument('--step-size', type=float, default=0.01,
                        help='Base step size for translations')
    parser.add_argument('--max-step-size', type=float, default=0.2,
                        help='Maximum allowed step size')
    
    # Algorithm options
    parser.add_argument('--num-bigiter-init', type=int, default=1,
                        help='Number of iterations starting from initial path')
    parser.add_argument('--num-bigiter-hess', type=int, default=0,
                        help='Number of iterations with virtual Hessian')
    parser.add_argument('--eps-hess', type=float, default=0.001,
                        help='Epsilon for virtual Hessian')
    
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
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize energy profile during optimization')
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
    initial_poscar = args.initial_poscar
    if not os.path.isabs(initial_poscar):
        initial_path = get_input_path(initial_poscar)
        if not os.path.exists(initial_path):
            # Backward compatibility
            if os.path.exists(initial_poscar):
                initial_path = os.path.abspath(initial_poscar)
                print(f"Warning: Found {initial_poscar} in current directory")
                print(f"Consider moving it to inputs/ directory")
            else:
                parser.error(f"File not found in inputs/: {initial_poscar}")
        else:
            initial_poscar = initial_path
    
    final_poscar = args.final_poscar
    if not os.path.isabs(final_poscar):
        final_path = get_input_path(final_poscar)
        if not os.path.exists(final_path):
            # Backward compatibility
            if os.path.exists(final_poscar):
                final_path = os.path.abspath(final_poscar)
                print(f"Warning: Found {final_poscar} in current directory")
                print(f"Consider moving it to inputs/ directory")
            else:
                parser.error(f"File not found in inputs/: {final_poscar}")
        else:
            final_poscar = final_path
    
    # Get structure to determine default moving indices
    structure = Poscar.from_file(initial_poscar, check_for_potcar=False).structure
    if args.moving_indices is None:
        moving_indices = list(range(len(structure)))
    else:
        moving_indices = args.moving_indices
    
    # Load initial structure to process moving indices with neighbors
    initial_structure = Poscar.from_file(initial_poscar, check_for_potcar=False).structure
    
    if args.include_neighbors != 'none':
        print(f"\nFinding nearest neighbors for atoms: {moving_indices}")
        
        neighbors = find_nearest_neighbors(
            initial_structure, 
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
        for idx in range(initial_structure.num_sites):
            status = "MOVE" if idx in moving_indices else "FROZEN"
            species = initial_structure[idx].species_string
            print(f"  Atom {idx}: {species} - {status}")

    
    # Build system parameters
    system_params = {
        'output_dir': args.output_dir,
        'run_name': args.run_name,
        'input_dir': args.input_dir,
        'activation_radius': args.activation_radius,
        'moving_indices': moving_indices,
        'n_images': args.n_images,
        'interpolation_method': args.interpolation_method,
        'max_neb_steps': args.max_neb_steps,
        'k_parallel': args.k_parallel,
        'k_perpendicular': args.k_perpendicular,
        'neb_convergence_threshold': args.neb_convergence_threshold,
        'ci_convergence_threshold': args.ci_convergence_threshold,
        'ci_activation_threshold_gp': args.ci_activation_threshold_gp,
        'convergence_criterion': args.convergence_criterion,
        'divisor_T_MEP_gp': args.divisor_T_MEP_gp,
        'max_inner_iterations': args.max_inner_iterations,
        'disp_max': args.disp_max,
        'ratio_at_limit': args.ratio_at_limit,
        'translation_method': args.translation_method,
        'step_size': args.step_size,
        'max_step_size': args.max_step_size,
        'num_bigiter_init': args.num_bigiter_init,
        'num_bigiter_hess': args.num_bigiter_hess,
        'eps_hess': args.eps_hess,
        'model_type': args.model_type,
        'verbose': args.verbose,
        'visualize': args.visualize,
        'execution_mode': args.execution_mode,
        'mpi_command': args.mpi_command,
        'vasp_command': args.vasp_command,
        'eam_potential_file': args.eam_potential_file,
        'continuation': args.continuation,
        'checkpoint_interval': args.checkpoint_interval,
        'kim_model': args.kim_model,
    }
    
    return initial_poscar, final_poscar, system_params


if __name__ == "__main__":
    try:
        initial_poscar, final_poscar, system_params = parse_arguments()
        
        # Initialize output directory structure
        OutputManager.setup(
            base_dir=system_params.get('output_dir', None),
            run_name=system_params.get('run_name', None),
            input_dir=system_params.get('input_dir', None)
        )
        
        # Save run metadata
        OutputManager.save_run_metadata({
            'script': 'run_gp2_neb.py',
            'parameters': system_params,
            'initial_poscar': initial_poscar,
            'final_poscar': final_poscar
        })
        
        print("Starting GP2-accelerated NEB search with parameters:")
        print("\nNEB Parameters:")
        print(f"  Images: {system_params['n_images']}")
        print(f"  Spring constants: k_par={system_params['k_parallel']}, k_perp={system_params['k_perpendicular']}")
        print(f"  Convergence: {system_params['neb_convergence_threshold']} eV/Å")
        print(f"  Convergence criterion: {system_params['convergence_criterion']}")
        print(f"  Climbing image: {'Enabled' if system_params['ci_activation_threshold_gp'] > 0 else 'Disabled'}")
        
        print("\nGP Parameters:")
        print(f"  Max outer iterations: {system_params['max_neb_steps']}")
        print(f"  Max inner iterations: {system_params['max_inner_iterations']}")
        print(f"  GP convergence divisor: {system_params['divisor_T_MEP_gp']}")
        print(f"  Model type: {system_params['model_type']}")
        
        print("\nStopping Criteria:")
        print(f"  Max displacement: {system_params['disp_max']} (relative to path length)")
        print(f"  Inter-atomic ratio limit: {system_params['ratio_at_limit']}")
        
        final_path, final_energies, final_gradients = gp2_neb_search(
            initial_poscar=initial_poscar,
            final_poscar=final_poscar,
            system_params=system_params
        )
        
        print("\nGP2 NEB search completed successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()