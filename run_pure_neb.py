#!/usr/bin/env python
"""Run pure NEB method without GP models."""

import argparse
import os
import sys
import logging
import time
import numpy as np
from walker_pure_neb import WalkerPureNEB

# Automatically activate incremental checkpoint system if available
try:
    from walker_pure_neb_optimized import patch_existing_walker
    patch_existing_walker()
    # Will print success message when patch is applied
except ImportError:
    pass  # Incremental system not available, use standard
except Exception:
    pass  # Could not patch, use standard
from pymatgen.core.periodic_table import Element
from vasp_manager import VASPManager, cleanup
from pymatgen.io.vasp import Poscar
from vasp_interface import VASPInterface
from pymatgen.core import Structure
from scipy.interpolate import interp1d
from output_manager import OutputManager, get_output_path, get_input_path


class PureNEBInterface:
    """Simplified interface for pure NEB without GP-related atom tracking."""
    
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
        self.neb_path_counter = 0
        self._pending_neb_batch = []  # For batch calculations
        self._current_batch_info = None  # Store batch info for wait_for_batch
        
    def scaler_y_value(self, position: np.ndarray, is_thermal: bool = False, **kwargs) -> float:
        """Calculate energy at given position.
        
        For NEB, kwargs may contain path_id and image_id.
        """
        # Extract NEB-specific parameters
        path_id = kwargs.get('path_id', self.neb_path_counter)
        image_id = kwargs.get('image_id', None)
        
        if image_id is not None:
            # NEB image calculation
            if hasattr(self.vasp_manager, 'neb_path_counter'):
                # Update the path counter
                self.vasp_manager.neb_path_counter = path_id
            
            run_dir = self.vasp_manager.setup_and_run_vasp(
                positions=position,
                run_type='neb',
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
        path_id = kwargs.get('path_id', self.neb_path_counter)
        image_id = kwargs.get('image_id', None)
        
        if image_id is not None:
            # NEB image calculation
            if hasattr(self.vasp_manager, 'neb_path_counter'):
                # Update the path counter
                self.vasp_manager.neb_path_counter = path_id
                
            run_dir = self.vasp_manager.setup_and_run_vasp(
                positions=position,
                run_type='neb',
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
        
        # Wait for completion and get forces
        run = self.vasp_manager.wait_for_completion(run_dir)
        
        # Return forces directly - DO NOT NEGATE!
        return run.flattened_forces
    
    def prepare_batch_calculations(self, positions_list: list, path_id: int) -> list:
        """Prepare batch of NEB calculations for parallel execution.
        
        Args:
            positions_list: List of (image_id, positions) tuples
            path_id: NEB path iteration number
            
        Returns:
            List of run directories
        """
        run_dirs = []
        
        if hasattr(self.vasp_manager, 'execution_mode') and self.vasp_manager.execution_mode == 'eam':
            print(f"\nPreparing batch of {len(positions_list)} NEB calculations for parallel EAM execution...")
            
            import time
            print(f"[PATH CREATION] at {time.strftime('%H:%M:%S')}: Creating path_{path_id:03d} with {len(positions_list)} images")
            
            # Set up all runs without waiting
            for image_id, positions in positions_list:
                if hasattr(self.vasp_manager, 'neb_path_counter'):
                    self.vasp_manager.neb_path_counter = path_id
                
                # Just setup, don't wait
                run_dir = self.vasp_manager.setup_neb_run(positions, path_id, image_id)
                run_dirs.append((image_id, run_dir))
                self._pending_neb_batch.append((image_id, run_dir))
            
            # Note: Cleanup is now done BEFORE path creation in walker_pure_neb.py
            # so we don't need the callback mechanism here anymore
            
            # Submit batch for parallel execution
            if hasattr(self.vasp_manager.executor, 'submit_batch'):
                all_dirs = [rd[1] for rd in run_dirs]
                print(f"Submitting batch of {len(all_dirs)} calculations...")
                batch_info = self.vasp_manager.executor.submit_batch(all_dirs)
                # Store batch_info for later use in wait_for_batch
                self._current_batch_info = batch_info
        else:
            # For non-EAM, just prepare the list
            for image_id, positions in positions_list:
                run_dirs.append((image_id, None))  # Will be calculated sequentially
        
        return run_dirs
    
    def wait_for_batch_results(self) -> dict:
        """Wait for all pending batch calculations to complete.
        
        Returns:
            Dictionary mapping image_id to (energy, forces)
        """
        results = {}
        
        if self._pending_neb_batch:
            print(f"Waiting for {len(self._pending_neb_batch)} calculations to complete...")
            
            # Wait for all to complete
            if hasattr(self.vasp_manager.executor, 'wait_for_batch'):
                all_dirs = [rd[1] for rd in self._pending_neb_batch]
                batch_info = getattr(self, '_current_batch_info', None)
                runs = self.vasp_manager.executor.wait_for_batch(batch_info, all_dirs)
                
                # Map results back to image IDs
                for (image_id, run_dir), run in zip(self._pending_neb_batch, runs):
                    if hasattr(run, 'has_error') and run.has_error:
                        print(f"  Warning: Image {image_id} failed, using fallback values")
                        # Use reasonable fallback values
                        results[image_id] = (-123.456, np.zeros(self.n_atoms * 3))
                    else:
                        results[image_id] = (run.energy, run.flattened_forces)
            else:
                # Non-batch mode - wait individually
                for image_id, run_dir in self._pending_neb_batch:
                    run = self.vasp_manager.wait_for_completion(run_dir)
                    results[image_id] = (run.energy, run.flattened_forces)
            
            # Clear the batch
            self._pending_neb_batch = []
        
        return results


def setup_logging():
    """Set up logging."""
    logging.getLogger().handlers.clear()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    log_file = get_output_path('logs', 'pure_neb_search.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")


def generate_initial_path(initial_poscar: str, final_poscar: str, n_images: int, 
                         interpolation_method: str = 'linear') -> np.ndarray:
    """Generate initial path between two structures using linear or image dependent interpolation.
    
    Args:
        initial_poscar: Path to initial POSCAR
        final_poscar: Path to final POSCAR
        n_images: Total number of images (including endpoints)
        interpolation_method: 'linear' or 'idpp' (Image Dependent Pair Potential)
        
    Returns:
        Initial path as (n_images, n_atoms*3) array in CARTESIAN coordinates
    """
    # Read structures
    initial_struct = Poscar.from_file(initial_poscar, check_for_potcar=False).structure
    final_struct = Poscar.from_file(final_poscar, check_for_potcar=False).structure
    
    # Ensure structures are compatible
    if len(initial_struct) != len(final_struct):
        raise ValueError("Initial and final structures must have the same number of atoms")
    
    # Get fractional coordinates first
    initial_frac = initial_struct.frac_coords
    final_frac = final_struct.frac_coords
    
    # Debug: Check which atoms move significantly
    print(f"\nAnalyzing atomic movements:")
    n_atoms = len(initial_struct)
    for i in range(n_atoms):
        # Calculate displacement considering periodic boundary conditions
        frac_diff = final_frac[i] - initial_frac[i]
        # Apply minimum image convention
        frac_diff = frac_diff - np.round(frac_diff)
        cart_diff = initial_struct.lattice.get_cartesian_coords(frac_diff)
        dist = np.linalg.norm(cart_diff)
        if dist > 0.01:  # Only print atoms that move significantly
            print(f"  Atom {i}: moves {dist:.4f} Å")
            print(f"    Initial (frac): {initial_frac[i]}")
            print(f"    Final (frac):   {final_frac[i]}")
    
    # Generate path with proper periodic boundary handling
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
                
                # Apply minimum image convention - find shortest path
                disp = disp - np.round(disp)
                
                # Interpolate
                frac_coords[atom] = initial_frac[atom] + t * disp
                
                # Wrap back into cell
                frac_coords[atom] = frac_coords[atom] - np.floor(frac_coords[atom])
            
            # Convert to Cartesian and flatten
            cart_coords = initial_struct.lattice.get_cartesian_coords(frac_coords)
            path[img, :] = cart_coords.flatten()
        
        # Debug: Verify path
        print(f"\nPath verification:")
        for img in range(n_images):
            if img > 0:
                dist = np.linalg.norm(path[img] - path[img-1])
                print(f"  Image {img-1} -> {img}: step = {dist:.4f} Å")
    
    elif interpolation_method == 'idpp':
        # Start with linear interpolation
        print(f"\nGenerating IDPP interpolation (starting with linear)...")
        
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
            
            # Add small random perturbation to intermediate images
            if 0 < img < n_images - 1:
                perturbation = 0.01 * np.random.randn(len(path[img, :]))
                path[img, :] += perturbation
    
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    
    # Final check: ensure no atoms are too close together
    print(f"\nChecking for atomic overlaps in interpolated path...")
    min_allowed_dist = 1.5  # Angstroms
    
    for img in range(n_images):
        coords = path[img, :].reshape(-1, 3)
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
            print(f"  WARNING: Image {img} has atoms closer than {min_allowed_dist} Å (min dist = {min_dist:.3f} Å)")
        else:
            print(f"  Image {img}: OK (min dist = {min_dist:.3f} Å)")
    
    return path


def pure_neb_search(initial_poscar: str, final_poscar: str, system_params: dict) -> tuple:
    """Run pure NEB search without GP models."""
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
    sys.stdout = tee
    
    logging.info("Starting pure NEB search")
    
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
            initial_path = generate_initial_path(
                initial_poscar, 
                final_poscar, 
                n_images,
                system_params.get('interpolation_method', 'linear')
            )
            logging.info(f"Generated initial path with {n_images} images")
        else:
            # For continuation, we'll load the path from checkpoint
            initial_path = None
        
        # Setup structure from initial POSCAR
        structure = Poscar.from_file(initial_poscar, check_for_potcar=False).structure
        
        # Create VASP manager with NEB support
        work_dir = os.path.dirname(os.path.abspath(initial_poscar))
        poscar_path = os.path.abspath(initial_poscar)
        
        execution_mode = system_params.get('execution_mode', 'mpi')
        mpi_command = system_params.get('mpi_command', None)
        vasp_command = system_params.get('vasp_command', 'vasp_gam')
        eam_potential_file = system_params.get('eam_potential_file', None)
        kim_model = system_params.get('kim_model', 'EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000')
        
        print(f"\nInitializing calculations with execution mode: {execution_mode}")
        
        # Modified VASPManager to support NEB runs
        use_gpu = system_params.get('use_gpu', False)
        if use_gpu:
            print(f"\n🚀 Creating GPU-accelerated VASPManagerNEB for NEB calculations...")
        else:
            print(f"\nCreating VASPManagerNEB for NEB calculations...")
        
        vasp_mgr = VASPManagerNEB(
            base_dir=get_output_path('vasp_runs'),
            user_poscar_path=poscar_path,
            execution_mode=execution_mode,
            mpi_command=mpi_command,
            vasp_command=vasp_command,
            eam_potential_file=eam_potential_file,
            skip_thermal=True,  # No thermal sampling needed
            kim_model_name=kim_model,
            parallel_eam=system_params.get('parallel_eam', False),
            eam_n_workers=system_params.get('eam_n_workers'),
            use_gpu=use_gpu,
            gpu_fallback=system_params.get('gpu_fallback', True),
        )
        
        print(f"VASPManagerNEB created with directories:")
        for name, path in vasp_mgr.run_dirs.items():
            print(f"  {name}: {path}")
        
        # For pure NEB, we don't need moving/frozen atom distinctions
        # Create a simplified interface
        local_pes = PureNEBInterface(
            vasp_manager=vasp_mgr,
            poscar_file=initial_poscar,
            n_atoms=len(structure)
        )
        
        # Create walker
        if not is_continuation:
            walker = WalkerPureNEB(
                initial_path=initial_path,
                local_pes=local_pes,
                max_neb_steps=system_params["max_neb_steps"],
                k_parallel=system_params.get("k_parallel", 1.0),
                k_perpendicular=system_params.get("k_perpendicular", 1.0),
                neb_convergence_threshold=system_params.get("neb_convergence_threshold", 0.1),
                ci_convergence_threshold=system_params.get("ci_convergence_threshold", 0.1),
                ci_activation_threshold=system_params.get("ci_activation_threshold", 0.0),
                translation_method=system_params.get("translation_method", "qmvv"),
                step_size=system_params.get("step_size", 0.01),
                max_step_size=system_params.get("max_step_size", 0.2),
                verbose=system_params["verbose"],
                checkpoint_interval=system_params.get('checkpoint_interval', 1),
                visualize=system_params.get('visualize', False),
                keep_only_latest_path=system_params.get('keep_only_latest_path', False)
            )
        else:
            # For continuation, need to find checkpoint file
            checkpoint_file = system_params.get('checkpoint_file')
            
            if checkpoint_file is None:
                # Try to find the latest checkpoint
                import glob
                
                # Look in the parent outputs directory for the most recent run
                parent_output_dir = os.path.dirname(get_output_path())
                checkpoint_patterns = [
                    os.path.join(parent_output_dir, '*/checkpoints/pure_neb_state.pkl'),  # New incremental format
                    os.path.join(parent_output_dir, '*/checkpoints/pure_neb_latest.pkl'),  # Old format
                    os.path.join(parent_output_dir, 'latest/checkpoints/pure_neb_state.pkl'),
                    os.path.join(parent_output_dir, 'latest/checkpoints/pure_neb_latest.pkl'),
                    'checkpoints/pure_neb_state.pkl',  # Current directory
                    'checkpoints/pure_neb_latest.pkl',  # Current directory old format
                    'outputs/latest/checkpoints/pure_neb_state.pkl',  # Legacy location
                    'outputs/latest/checkpoints/pure_neb_latest.pkl',  # Legacy location old format
                ]
                
                checkpoint_file = None
                for pattern in checkpoint_patterns:
                    files = glob.glob(pattern)
                    if files:
                        # Get the most recent one
                        checkpoint_file = max(files, key=os.path.getmtime)
                        break
                
                if checkpoint_file is None:
                    raise FileNotFoundError(
                        "No checkpoint file found. Please specify --checkpoint-file or ensure a previous run exists."
                    )
                
                logging.info(f"Found checkpoint file: {checkpoint_file}")
            else:
                # Verify the specified file exists
                if not os.path.exists(checkpoint_file):
                    raise FileNotFoundError(f"Specified checkpoint file not found: {checkpoint_file}")
            
            # Create walker and restore from checkpoint
            walker = WalkerPureNEB(
                initial_path=np.zeros((n_images, len(structure)*3)),  # Placeholder
                local_pes=local_pes,
                max_neb_steps=1,  # Will be updated
                verbose=system_params["verbose"],
                keep_only_latest_path=system_params.get("keep_only_latest_path", False)
            )
            
            # Load checkpoint
            checkpoint = walker.load_checkpoint(checkpoint_file)
            
            # Update max steps for continuation
            steps_completed = walker.steps
            additional_steps = system_params["max_neb_steps"]
            walker.max_neb_steps = steps_completed + additional_steps
            
            logging.info(f"Continuing from step {steps_completed}")
            logging.info(f"Will run {additional_steps} additional steps")
        
        # Run search
        logging.info("Starting walker run")
        final_path, final_energies, final_gradients = walker.run()
        
        logging.info("Walker run completed")
        
        # Save results
        import pickle
        results = {
            'final_path': final_path,
            'final_energies': final_energies,
            'final_gradients': final_gradients,
            'converged': walker.converged,
            'steps': walker.steps,
            'trajectory': walker.trajectory,
            'vasp_eval_count': walker.vasp_eval_count,
            'n_images': walker.n_images,
            'n_atoms': walker.n_atoms,
            'ci_index': walker.i_CI if walker.CI_on else -1
        }
        
        results_dir = get_output_path('results')
        with open(os.path.join(results_dir, 'pure_neb_results.pkl'), 'wb') as f:
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
        print(f"  Total VASP evaluations: {walker.vasp_eval_count}")
        print(f"\nStructures saved to: {os.path.join(results_dir, 'POSCAR_image_*')}")
        
        return final_path, final_energies, final_gradients
        
    except Exception as e:
        logging.error(f"Error during pure NEB search: {e}")
        raise


class VASPManagerNEB(VASPManager):
    """Extended VASP Manager with NEB support."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add NEB run directory
        self.run_dirs["neb"] = os.path.join(self.base_dir, "neb_runs")
        os.makedirs(self.run_dirs["neb"], exist_ok=True)
        
        # NEB-specific counters
        self.neb_path_counter = 0
        self.neb_image_counters = {}
    
    def setup_neb_run(self, image_positions: np.ndarray, path_id: int, image_id: int) -> str:
        """Set up a VASP run for a single NEB image.
        
        Args:
            image_positions: Positions for this image
            path_id: NEB path iteration number
            image_id: Image number in the path
            
        Returns:
            Run directory path
        """
        # Create directory structure: neb_runs/path_XX/image_YY/
        path_dir = os.path.join(self.run_dirs["neb"], f"path_{path_id:03d}")
        image_dir = os.path.join(path_dir, f"image_{image_id:02d}")
        os.makedirs(image_dir, exist_ok=True)
        
        # Log path creation (only for first image to avoid spam)
        if image_id == 0:
            print(f"\n[PATH CREATED] {os.path.basename(path_dir)} at {time.strftime('%H:%M:%S')}")
        
        # For EAM, just write POSCAR and submit
        if self.execution_mode == "eam":
            # Use the user POSCAR as template for EAM
            from vasp_manager import read_structure, get_structure_from_positions, write_structure
            if self.user_poscar_path and os.path.exists(self.user_poscar_path):
                template_structure = read_structure(self.user_poscar_path)[0]
            else:
                # Create a simple structure based on positions
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
        """Override to handle NEB runs."""
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Pure NEB Method (No GP Models)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required files
    parser.add_argument('--initial-poscar', default='POSCAR_initial',
                        help='Initial structure POSCAR (searched in inputs/ directory)')
    parser.add_argument('--final-poscar', default='POSCAR_final',
                        help='Final structure POSCAR (searched in inputs/ directory)')
    
    # NEB parameters
    parser.add_argument('--n-images', type=int, default=7,
                        help='Total number of images (including endpoints)')
    parser.add_argument('--interpolation-method', choices=['linear', 'idpp'], 
                        default='linear',
                        help='Method for generating initial path')
    parser.add_argument('--max-neb-steps', type=int, default=100,
                        help='Maximum number of NEB iterations')
    parser.add_argument('--k-parallel', type=float, default=1.0,
                        help='Parallel spring constant')
    parser.add_argument('--k-perpendicular', type=float, default=1.0,
                        help='Perpendicular spring constant')
    parser.add_argument('--neb-convergence-threshold', type=float, default=0.1,
                        help='Force convergence threshold (eV/Å)')
    parser.add_argument('--ci-convergence-threshold', type=float, default=0.1,
                        help='Climbing image convergence threshold (eV/Å)')
    parser.add_argument('--ci-activation-threshold', type=float, default=0.0,
                        help='Force threshold to activate climbing image (0 = disabled)')
    
    # Translation method
    parser.add_argument('--translation-method', 
                        choices=['qmvv', 'lbfgs', 'fire'],
                        default='qmvv',
                        help='Method for moving images')
    parser.add_argument('--step-size', type=float, default=0.01,
                        help='Base step size for translations')
    parser.add_argument('--max-step-size', type=float, default=0.2,
                        help='Maximum allowed step size')
    

    
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
    parser.add_argument('--parallel-eam', action='store_true',
                        help='Enable parallel execution of EAM calculations')
    parser.add_argument('--eam-n-workers', type=int, default=None,
                        help='Number of parallel workers for EAM (default: number of CPU cores)')
    
    # GPU acceleration options
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU acceleration for EAM calculations (requires JAX)')
    parser.add_argument('--no-gpu-fallback', action='store_true',
                        help='Fail if GPU requested but not available (default: fallback to CPU)')
    
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
    parser.add_argument('--checkpoint-file', default=None,
                        help='Path to checkpoint file for continuation (if not specified, looks in latest run)')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--keep-only-latest-path', action='store_true',
                        help='Keep only the latest NEB path in neb_runs folder, removing all previous paths')
    
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
    
    # Build system parameters
    system_params = {
        'output_dir': args.output_dir,
        'run_name': args.run_name,
        'input_dir': args.input_dir,
        'n_images': args.n_images,
        'interpolation_method': args.interpolation_method,
        'max_neb_steps': args.max_neb_steps,
        'k_parallel': args.k_parallel,
        'k_perpendicular': args.k_perpendicular,
        'neb_convergence_threshold': args.neb_convergence_threshold,
        'ci_convergence_threshold': args.ci_convergence_threshold,
        'ci_activation_threshold': args.ci_activation_threshold,
        'translation_method': args.translation_method,
        'step_size': args.step_size,
        'max_step_size': args.max_step_size,
        'verbose': args.verbose,
        'visualize': args.visualize,
        'execution_mode': args.execution_mode,
        'mpi_command': args.mpi_command,
        'vasp_command': args.vasp_command,
        'eam_potential_file': args.eam_potential_file,
        'continuation': args.continuation,
        'checkpoint_file': args.checkpoint_file,
        'checkpoint_interval': args.checkpoint_interval,
        'kim_model': args.kim_model,
        'parallel_eam': args.parallel_eam,
        'eam_n_workers': args.eam_n_workers,
        'use_gpu': args.gpu,
        'gpu_fallback': not args.no_gpu_fallback,
        'keep_only_latest_path': args.keep_only_latest_path,
    }
    
    return initial_poscar, final_poscar, system_params


if __name__ == "__main__":
    try:
        initial_poscar, final_poscar, system_params = parse_arguments()
        
        # Save run metadata
        OutputManager.save_run_metadata({
            'script': 'run_pure_neb.py',
            'parameters': system_params,
            'initial_poscar': initial_poscar,
            'final_poscar': final_poscar
        })
        
        print("Starting pure NEB search with parameters:")
        for key, value in system_params.items():
            print(f"  {key}: {value}")
        
        final_path, final_energies, final_gradients = pure_neb_search(
            initial_poscar=initial_poscar,
            final_poscar=final_poscar,
            system_params=system_params
        )
        
        print("\nPure NEB search completed successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()