from typing import Tuple, List, Optional, Any
import logging
import os
import shutil
import numpy as np
import json
import numpy.typing as npt
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from vasp_executors import get_hpc_executor, detect_hpc_system
from output_manager import OutputManager, get_output_path


def cleanup():        
    if os.path.exists(get_output_path('data')):
        shutil.rmtree(get_output_path('data'))
    if os.path.exists(get_output_path('data_gp1')):
        shutil.rmtree(get_output_path('data_gp1'))
    if os.path.exists(get_output_path('data_gp2')):
        shutil.rmtree(get_output_path('data_gp2'))
    if os.path.exists(get_output_path('data_trgp1')):
        shutil.rmtree(get_output_path('data_trgp1'))
    if os.path.exists(get_output_path('thermal_snapshots.pkl')):
        os.remove(get_output_path('thermal_snapshots.pkl'))
    if os.path.exists(get_output_path('thermal_snapshots.pkl_vel')):
        os.remove(get_output_path('thermal_snapshots.pkl_vel'))
    if os.path.exists(get_output_path('fit_data_snapshots.pkl')):
        os.remove(get_output_path('fit_data_snapshots.pkl'))
    if os.path.exists(get_output_path('gp2_curvature.pkl')):
        os.remove(get_output_path('gp2_curvature.pkl'))
    if os.path.exists(get_output_path('gp1_curvature.pkl')):
        os.remove(get_output_path('gp1_curvature.pkl'))
    if os.path.exists(get_output_path('vasp_runs')):
        shutil.rmtree(get_output_path('vasp_runs'))
    if os.path.exists(get_output_path('saddle_point_search.log')):
        os.remove(get_output_path('saddle_point_search.log'))
    if os.path.exists(get_output_path('thermal_diagnostics.json')):
        os.remove(get_output_path('thermal_diagnostics.json'))
    # remove log files (log_file.txt and log_file_bfgs.txt)
    if os.path.exists(get_output_path('log_file.txt')):
        os.remove(get_output_path('log_file.txt'))
    if os.path.exists(get_output_path('log_file_bfgs.txt')):
        os.remove(get_output_path('log_file_bfgs.txt'))

def read_structure(filename: str) -> Tuple[Structure, List[str], List[int]]:
    """Read atomic structure from POSCAR/CONTCAR using pymatgen.
    
    Args:
        filename: Path to POSCAR/CONTCAR file
        
    Returns:
        Tuple containing:
            - structure: pymatgen Structure object
            - atom_types: List of atomic species
            - atom_counts: List of number of atoms per species
            
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is corrupted or empty
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Structure file {filename} not found")
        
    if os.path.getsize(filename) == 0:
        raise ValueError(f"Structure file {filename} is empty")
        
    poscar = Poscar.from_file(filename, check_for_potcar=False)
    structure = poscar.structure
    atom_types = poscar.site_symbols
    atom_counts = poscar.natoms
    
    return structure, atom_types, atom_counts

def read_latest_structure(directory: str) -> Structure:
    """Read structure preferring CONTCAR over POSCAR.
    
    Args:
        directory: VASP calculation directory
        
    Returns:
        Structure object from CONTCAR if valid, else from POSCAR
        
    Raises:
        FileNotFoundError: If neither file exists
    """
    contcar = os.path.join(directory, 'CONTCAR')
    poscar = os.path.join(directory, 'POSCAR')
    
    if os.path.exists(contcar) and os.path.getsize(contcar) > 0:
        try:
            return Poscar.from_file(contcar).structure
        except Exception:
            print("Warning: CONTCAR exists but is corrupted. Falling back to POSCAR.")
    
    if not os.path.exists(poscar):
        raise FileNotFoundError("Neither CONTCAR nor POSCAR found")
        
    return Poscar.from_file(poscar).structure

def write_structure(
        filename: str,
        structure: Structure,
        direct_coords: bool = True
    ) -> None:
    """Write structure to POSCAR format using pymatgen.
    
    Args:
        filename: Output file path
        structure: pymatgen Structure object
        direct_coords: Whether to write in direct coordinates
    """
    # Create Poscar object with correct initialization
    poscar = Poscar(
        structure=structure,
        selective_dynamics=None,
        true_names=True  # Use actual element names
    )
    
    # Force direct/cartesian coordinates if needed
    if not direct_coords:
        poscar.structure.add_oxidation_state_by_site([0] * len(structure))
        poscar.structure.convert_to_cartesian()
    
    # Write the POSCAR file
    poscar.write_file(filename)

def read_forces(filename: str) -> npt.NDArray[np.float64]:
    """Read forces directly from OUTCAR file.
    
    Args:
        filename: Path to OUTCAR file
        
    Returns:
        Nx3 array of forces
    """
    forces = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        read_forces = False
        for i, line in enumerate(lines):
            if 'TOTAL-FORCE' in line:
                for j in range(i+2, len(lines)):
                    if '----------' in lines[j]:
                        break
                    parts = lines[j].split()
                    if len(parts) >= 6:
                        try:
                            force = [float(parts[3]), float(parts[4]), float(parts[5])]
                            forces.append(force)
                        except (ValueError, IndexError):
                            # Stop if we hit non-numeric content or wrong format
                            break
                    elif len(parts) == 0:
                        # Stop at empty line
                        break
                break
                
    if not forces:
        raise ValueError("No forces found in OUTCAR")
        
    return np.array(forces)

def read_energy(filename: str) -> float:
    """Read energy directly from OUTCAR file.
    
    Args:
        filename: Path to OUTCAR file
        
    Returns:
        Final energy value
    """
    with open(filename, 'r') as f:
        for line in f:
            if 'free  energy   TOTEN' in line:
                return float(line.split()[4])
    raise ValueError("No energy found in OUTCAR")


def get_structure_from_positions(
        positions: npt.NDArray[np.float64],
        template_structure: Structure
    ) -> Structure:
    """Create new Structure with updated positions.
    
    Args:
        positions: Nx3 array of new positions (or 3N flattened array)
        template_structure: Template structure with correct lattice/types
        
    Returns:
        New Structure object with updated positions
        
    Raises:
        ValueError: If positions shape doesn't match structure
    """
    if positions.size != len(template_structure) * 3:
        raise ValueError(f"Expected {len(template_structure) * 3} coordinates, got {positions.size}")
        
    positions = positions.reshape(-1, 3)
    if len(positions) != len(template_structure):
        raise ValueError("Mismatch between number of positions and structure sites")
    
    # Create new structure with same lattice and species
    new_structure = Structure(
        lattice=template_structure.lattice,
        species=[site.specie for site in template_structure],
        coords=positions,
        coords_are_cartesian=True  # Specify that these are cartesian coordinates
    )
    
    return new_structure

def verify_vasp_inputs(directory: str) -> bool:
    """Verify all required VASP input files exist and are valid."""
    required_files = {
        'INCAR': Incar,
        'KPOINTS': Kpoints,
        'POTCAR': None
    }
    
    # Check each required file
    for fname, cls in required_files.items():
        fpath = os.path.join(directory, fname)
        if not os.path.exists(fpath):
            print(f"Missing {fname} in {directory}")
            return False
            
        if cls is not None:
            try:
                cls.from_file(fpath)
            except Exception as e:
                print(f"Error reading {fname} in {directory}: {str(e)}")
                return False
                
    # Check for either POSCAR or CONTCAR
    if not (os.path.exists(os.path.join(directory, 'POSCAR')) or 
            os.path.exists(os.path.join(directory, 'CONTCAR'))):
        print(f"Neither POSCAR nor CONTCAR found in {directory}")
        return False
        
    return True

def cleanup_vasp_run(directory: str, keep_essential: bool = True) -> None:
    """Clean up VASP directory, keeping only essential files.
    
    Args:
        directory: VASP calculation directory
        keep_essential: Whether to keep CONTCAR, OUTCAR, OSZICAR
    """
    essential_files = {'CONTCAR', 'OUTCAR', 'OSZICAR'} if keep_essential else set()
    for fname in os.listdir(directory):
        if fname not in essential_files and fname.startswith(('CHG', 'WAVECAR', 'EIGENVAL')):
            try:
                os.remove(os.path.join(directory, fname))
            except OSError:
                pass

def setup_next_vasp_run(
        prev_run_dir: str,
        next_run_dir: str,
        positions: npt.NDArray[np.float64],
        copy_wavecar: bool = True
    ) -> None:
    """Set up next VASP calculation directory."""
    logging.info(f"Setting up VASP run: prev={prev_run_dir}, next={next_run_dir}")
    os.makedirs(next_run_dir, exist_ok=True)
    
    # Copy input files
    input_files = {
        'INCAR': Incar,
        'POTCAR': None,
        'KPOINTS': Kpoints,
        'job.sh': None
    }
    
    for fname, cls in input_files.items():
        src = os.path.join(prev_run_dir, fname)
        dst = os.path.join(next_run_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            if cls is None:
                shutil.copy2(src, dst)
            else:
                obj = cls.from_file(src)
                obj.write_file(dst)
                
    # Create new structure and write POSCAR
    # Try to read template structure from previous run directory
    try:
        template_structure = read_latest_structure(prev_run_dir)
        print(f"DEBUG: Read template structure from {prev_run_dir}")
        print(f"DEBUG: Template structure species: {[str(s) for s in template_structure.species]}")
    except:
        # If that fails, try the user POSCAR
        if hasattr(VASPManager, '_user_poscar_path') and VASPManager._user_poscar_path:
            template_structure = read_structure(VASPManager._user_poscar_path)[0]
            print(f"DEBUG: Read template structure from user POSCAR: {VASPManager._user_poscar_path}")
            print(f"DEBUG: Template structure species: {[str(s) for s in template_structure.species]}")
        else:
            raise ValueError(f"Cannot find template structure for {positions.size} coordinates")

    # Check if dimensions match
    if len(template_structure) * 3 != positions.size:
        # Try user POSCAR as fallback
        if hasattr(VASPManager, '_user_poscar_path') and VASPManager._user_poscar_path:
            try:
                user_struct = read_structure(VASPManager._user_poscar_path)[0]
                if len(user_struct) * 3 == positions.size:
                    template_structure = user_struct
                    print(f"DEBUG: Using user POSCAR due to size mismatch")
                    print(f"DEBUG: User POSCAR species: {[str(s) for s in template_structure.species]}")
                else:
                    raise ValueError(f"Position size mismatch: expected {len(template_structure)*3}, got {positions.size}")
            except:
                raise ValueError(f"Cannot find template structure matching {positions.size} coordinates")
        else:
            raise ValueError(f"Position size mismatch: expected {len(template_structure)*3}, got {positions.size}")

    new_structure = get_structure_from_positions(positions, template_structure)
    write_structure(os.path.join(next_run_dir, 'POSCAR'), new_structure)

    # Verify all files exist
    required_files = ['INCAR', 'POTCAR', 'KPOINTS', 'job.sh', 'POSCAR']
    for fname in required_files:
        fpath = os.path.join(next_run_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file {fname} not created in {next_run_dir}")

class VASPRun:
    """Class to manage VASP calculation data."""
    
    def __init__(self, directory: str) -> None:
        """Initialize from VASP directory."""
        self.directory = directory
        self.has_error = False
        self.error_message = None
        
        # Check if this is an EAM calculation
        eam_results_file = os.path.join(directory, 'eam_results.json')
        if os.path.exists(eam_results_file):
            # Handle EAM results
            print(f"Reading EAM results from {eam_results_file}")
            try:
                with open(eam_results_file, 'r') as f:
                    results = json.load(f)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON from {eam_results_file}")
                print(f"  JSONDecodeError: {e}")
                # Check file size
                file_size = os.path.getsize(eam_results_file)
                print(f"  File size: {file_size} bytes")
                if file_size == 0:
                    raise RuntimeError(f"EAM results file is empty: {eam_results_file}")
                else:
                    raise RuntimeError(f"EAM results file is corrupted: {eam_results_file}")
            
            if 'error' in results:
                print(f"WARNING: EAM calculation had error in directory {directory}")
                print(f"  Error message: {results['error']}")
                print(f"  Using fallback values")
                # Use fallback values instead of raising error
                self.energy = results.get('energy', -1750.0)
                self.forces = np.array(results.get('forces', [[0.0, 0.0, 0.0]] * results.get('n_atoms', 215)))
                self.has_error = True
                self.error_message = results['error']
                return
            
            # Read structure from POSCAR
            poscar_path = os.path.join(directory, 'POSCAR')
            if os.path.exists(poscar_path):
                self.structure, self.atom_types, self.atom_counts = read_structure(poscar_path)
            else:
                raise FileNotFoundError(f"POSCAR not found in {directory}")
            
            # Get results
            self.energy = results['energy']
            self.forces = np.array(results['forces'])
            
            # Ensure forces shape is correct
            if self.forces.shape != (len(self.structure), 3):
                if self.forces.ndim == 1:
                    self.forces = self.forces.reshape(-1, 3)
                    
        else:
            # Check if this might be a failed EAM calculation
            poscar_path = os.path.join(directory, 'POSCAR')
            if os.path.exists(poscar_path) and not os.path.exists(os.path.join(directory, 'OUTCAR')):
                # This looks like an EAM calculation that didn't produce results
                print(f"ERROR: No eam_results.json found in {directory}")
                print(f"  Directory contents: {os.listdir(directory)}")
                raise RuntimeError(f"EAM calculation failed: No results file found in {directory}")
            
            # Original VASP handling
            # Read structure
            self.structure, self.atom_types, self.atom_counts = read_structure(
                os.path.join(directory, 'CONTCAR')
                if os.path.exists(os.path.join(directory, 'CONTCAR'))
                else os.path.join(directory, 'POSCAR')
            )
            
            # Read energy and forces
            outcar = os.path.join(directory, 'OUTCAR')
            if not os.path.exists(outcar):
                raise FileNotFoundError(f"OUTCAR not found in {directory}")
                
            self.forces = read_forces(outcar)
            self.energy = read_energy(outcar)
        
    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Get atomic positions as Nx3 array."""
        return self.structure.cart_coords
    
    @property
    def flattened_positions(self) -> npt.NDArray[np.float64]:
        """Get flattened atomic positions as 3N array."""
        return self.positions.flatten()
    
    @property
    def flattened_forces(self) -> npt.NDArray[np.float64]:
        """Get flattened forces as 3N array."""
        return self.forces.flatten()


class VASPManager:
    """Manages VASP calculations with support for different execution modes."""
    
    def __init__(self,
                base_dir: str,
                *,
                main_dir: str = "dimer_runs",
                thermal_dir: str = "thermal_snapshots",
                user_poscar_path: Optional[str] = None,
                snapshots_per_batch: int = 1000,
                execution_mode: str = "mpi",  # New parameter
                mpi_command: Optional[str] = None,  # For custom MPI commands
                vasp_command: str = "vasp_gam",
                eam_potential_file: Optional[str] = None,
                skip_thermal: bool = False,
                kim_model_name: Optional[str] = None,
                parallel_eam: bool = False,  # Enable parallel EAM execution
                eam_n_workers: Optional[int] = None,  # Number of EAM workers
                use_gpu: bool = False,  # Enable GPU acceleration
                gpu_fallback: bool = True,  # Fallback to CPU if GPU unavailable
                vasp_input_dir: Optional[str] = None,  # Directory with VASP input files (INCAR, KPOINTS, POTCAR)
                temperature: Optional[int] = None,  # Temperature for temperature-specific VASP inputs
) -> None:
        """
        Initialize VASP Manager with specified execution mode.
        
        Parameters
        ----------
        base_dir : str
            Root folder for VASP calculations
        main_dir : str
            Sub-folder for dimer optimization runs
        thermal_dir : str
            Sub-folder for thermal-snapshot runs
        user_poscar_path : str, optional
            Path to user's POSCAR file
        snapshots_per_batch : int
            Number of snapshots per thermal batch
        execution_mode : str
            One of: "mock", "mpi", "slurm", "stampede3", "eam", "vasp"
        mpi_command : str, optional
            Custom MPI command (e.g., "mpirun -np 64" or "srun -n 48")
        vasp_command : str
            VASP executable name
        """
        self.base_dir = get_output_path(base_dir if base_dir else "vasp_runs")
        # Handle relative path for user POSCAR
        if user_poscar_path and not os.path.isabs(user_poscar_path):
            self.user_poscar_path = OutputManager.get_input_path(user_poscar_path)
        else:
            self.user_poscar_path = user_poscar_path
        # Store as class variable for access in static methods
        VASPManager._user_poscar_path = user_poscar_path
        print(f"DEBUG: VASPManager initialized with user_poscar_path: {user_poscar_path}")
        self.snapshots_per_batch = snapshots_per_batch
        self.execution_mode = execution_mode.lower()
        self.eam_potential_file = eam_potential_file
        self.kim_model_name = kim_model_name
        self.parallel_eam = parallel_eam
        self.eam_n_workers = eam_n_workers
        self.use_gpu = use_gpu
        self.gpu_fallback = gpu_fallback
        self.vasp_input_dir = vasp_input_dir
        self.temperature = temperature

        # Directory map - conditionally include thermal
        if skip_thermal:
            self.run_dirs = {
                "main": os.path.join(base_dir, main_dir),
            }
        else:
            self.run_dirs = {
                "main": os.path.join(base_dir, main_dir),
                "thermal": os.path.join(base_dir, thermal_dir),
            }
        
        # Create directories
        for name, path in self.run_dirs.items():
            os.makedirs(path, exist_ok=True)
        
        # Initialize the appropriate executor
        self._init_executor(execution_mode, mpi_command, vasp_command)
        
        # Initialize run counters
        self.run_counters = {k: self._get_next_run_number(v) for k, v in self.run_dirs.items()}
        
        # Thermal batch management
        # Only initialize thermal batch if not skipping
        if not skip_thermal:
            first_batch_dir = os.path.join(self.run_dirs["thermal"], "1")
            os.makedirs(first_batch_dir, exist_ok=True)

        self.current_thermal_batch = 0
        self.thermal_run_counter = 1
        self.current_thermal_batch_dirs: List[str] = []
        self.current_thermal_batch_info: Any = None  # Stores job IDs/processes
        
        # For tracking pending thermal runs
        self._pending_thermal_dirs: List[str] = []
        
        print(f"\nInitializing VASPManager:")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Execution mode: {self.execution_mode}")
        for name, path in self.run_dirs.items():
            print(f"  {name}: {path}")
    
    def _init_executor(self, execution_mode: str, mpi_command: Optional[str], vasp_command: str):
        """Initialize the appropriate executor based on execution mode."""        
        # Detect system for logging
        detected_system = detect_hpc_system()
        print(f"  Detected HPC system: {detected_system}")
        
        if execution_mode == "mock":
            from vasp_executors import MockVASPExecutor
            self.executor = MockVASPExecutor(self.user_poscar_path)
            print("  Using MOCK executor for testing")
            
        elif execution_mode == "mpi":
            if mpi_command is None:
                mpi_command = "mpirun -np 64"  # Default
            from vasp_executors import MPIVASPExecutor
            self.executor = MPIVASPExecutor(mpi_command, vasp_command)
            print(f"  Using MPI executor: {mpi_command} {vasp_command}")
            
        elif execution_mode == "slurm":
            from vasp_executors import SLURMVASPExecutor
            self.executor = SLURMVASPExecutor()
            print("  Using SLURM executor")
            
        elif execution_mode == "stampede3":
            # Check if we're actually on Stampede3
            if detected_system == "stampede3":
                # On Stampede3 compute node - use direct execution
                from vasp_executors import Stampede3VASPExecutor
                self.executor = Stampede3VASPExecutor(self.base_dir)
                print("  Using Stampede3 executor (direct execution on compute nodes)")
            elif detected_system == "anvil":
                # On Anvil - use SLURM submission
                from vasp_executors import AnvilVASPExecutor
                self.executor = AnvilVASPExecutor()
                print("  Detected Anvil system - using Anvil executor (SLURM submission)")
            else:
                # Unknown system - try Stampede3 executor anyway
                self.executor = Stampede3VASPExecutor(self.base_dir)
                print(f"  Using Stampede3 executor on {detected_system} system")
                
        elif execution_mode == "anvil":
            from vasp_executors import AnvilVASPExecutor  # Add this import
            self.executor = AnvilVASPExecutor()
            print("  Using Anvil executor (SLURM submission)")

        elif execution_mode == "eam":
            # Check if GPU acceleration is requested
            if self.use_gpu:
                # Try to use GPU-accelerated executor with careful import
                gpu_executor_loaded = False
                try:
                    # Check if we're on macOS first
                    import platform
                    if platform.system() == 'Darwin':
                        # Use CPU-parallel executor on macOS
                        from eam_executor_cpu_parallel import EAMExecutorCPUParallel as EAMExecutorGPU
                        print("  ℹ️  macOS detected, using CPU-parallel executor instead of GPU")
                    else:
                        # Try regular GPU executor on Linux
                        from eam_executor_gpu import EAMExecutorGPU
                    
                    kim_model = getattr(self, 'kim_model_name', None)
                    if isinstance(kim_model, tuple):
                        kim_model = kim_model[0] if kim_model else None
                    
                    self.executor = EAMExecutorGPU(
                        potential_file=self.eam_potential_file,
                        kim_model=kim_model,
                        use_gpu=True,
                        fallback_to_cpu=self.gpu_fallback,
                        parallel_eam=self.parallel_eam,
                        n_workers=self.eam_n_workers
                    )
                    gpu_executor_loaded = True
                    if platform.system() == 'Darwin':
                        print(f"  💻 Using CPU-parallel EAM executor with {self.eam_n_workers or 'auto'} workers")
                    else:
                        print(f"  🚀 Using GPU-accelerated EAM executor")
                    if self.eam_potential_file:
                        print(f"     Potential: {self.eam_potential_file}")
                    elif kim_model:
                        print(f"     KIM model: {kim_model}")
                except (ImportError, RuntimeError) as e:
                    # JAX import failed (e.g., AVX instructions issue on Mac)
                    if not self.gpu_fallback:
                        raise RuntimeError(f"GPU executor not available: {e}")
                    
                    # Fall back to standard executor
                    gpu_executor_loaded = False
                    if "AVX instructions" in str(e):
                        print("  ⚠️  JAX incompatible with system (x86 Python on ARM Mac)")
                    else:
                        print(f"  ⚠️  GPU executor not available: {str(e)[:100]}")
                    print("     Falling back to standard EAM executor")
                
                # If GPU executor failed to load and we're falling back
                if not gpu_executor_loaded:
                    if self.eam_potential_file:
                        from eam_file_executor import EAMFileExecutor
                        self.executor = EAMFileExecutor(
                            potential_file=self.eam_potential_file,
                            user_poscar_path=self.user_poscar_path,
                            parallel_eam=self.parallel_eam,
                            n_workers=self.eam_n_workers
                        )
                    else:
                        kim_model = getattr(self, 'kim_model_name', "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000")
                        if isinstance(kim_model, tuple):
                            kim_model = kim_model[0] if kim_model else "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000"
                        from eam_executor_isolated import IsolatedEAMExecutor
                        self.executor = IsolatedEAMExecutor(
                            kim_model_name=kim_model,
                            user_poscar_path=self.user_poscar_path
                        )
            
            # Standard CPU execution
            elif self.parallel_eam:
                # Use CPU-parallel executor for parallel EAM without GPU
                import platform
                from eam_executor_cpu_parallel import EAMExecutorCPUParallel
                
                kim_model = getattr(self, 'kim_model_name', None)
                if isinstance(kim_model, tuple):
                    kim_model = kim_model[0] if kim_model else None
                
                self.executor = EAMExecutorCPUParallel(
                    potential_file=self.eam_potential_file,
                    kim_model=kim_model,
                    parallel_eam=True,
                    n_workers=self.eam_n_workers
                )
                print(f"  💻 Using CPU-parallel EAM executor with {self.eam_n_workers or 'auto'} workers")
                if self.eam_potential_file:
                    print(f"     Potential: {self.eam_potential_file}")
                elif kim_model:
                    print(f"     KIM model: {kim_model}")
            elif self.eam_potential_file:
                # Use potential file
                from eam_file_executor import EAMFileExecutor
                self.executor = EAMFileExecutor(
                    potential_file=self.eam_potential_file,
                    user_poscar_path=self.user_poscar_path,
                    parallel_eam=False,  # Already handled above
                    n_workers=self.eam_n_workers
                )
                print(f"  Using EAM file executor with potential: {self.eam_potential_file}")
            else:
                # Use KIM model
                kim_model = getattr(self, 'kim_model_name', "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000")
                
                # Fix: Ensure kim_model is a string, not a tuple
                if isinstance(kim_model, tuple):
                    kim_model = kim_model[0] if kim_model else "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000"
                
                from eam_executor_isolated import IsolatedEAMExecutor
                self.executor = IsolatedEAMExecutor(
                    kim_model_name=kim_model,
                    user_poscar_path=self.user_poscar_path
                )
                print(f"  Using KIM EAM executor with model: {kim_model}")
            
        elif execution_mode == "vasp":
            from vasp_executors import VASPLocalExecutor
            # Parse MPI ranks from mpi_command if provided (e.g., "-np 4")
            import re
            num_ranks = 4  # default
            if mpi_command:
                match = re.search(r'-np\s+(\d+)', mpi_command)
                if match:
                    num_ranks = int(match.group(1))

            self.executor = VASPLocalExecutor(
                vasp_command=vasp_command,
                mpi_command=mpi_command,
                num_mpi_ranks=num_ranks,
                user_poscar_path=self.user_poscar_path
            )
            print(f"  Using local VASP executor ({num_ranks} MPI ranks)")

        elif execution_mode == "vasp-gpu":
            # GPU-accelerated VASP execution
            from vasp_gpu_executor import GPUVASPExecutor
            
            # Parse GPU-specific options from mpi_command if provided
            num_gpus = 1
            gpu_devices = None
            
            # Check for GPU configuration in mpi_command (format: "gpu:N" or "gpu:0,1")
            if mpi_command and "gpu:" in mpi_command:
                gpu_config = mpi_command.split("gpu:")[1].split()[0]
                if "," in gpu_config:
                    # Specific GPU devices
                    gpu_devices = gpu_config
                    num_gpus = len(gpu_config.split(","))
                else:
                    # Number of GPUs
                    num_gpus = int(gpu_config)
                # Clean up mpi_command
                mpi_command = mpi_command.split("gpu:")[0].strip()
            
            if not mpi_command:
                mpi_command = "mpirun --allow-run-as-root --mca btl_vader_single_copy_mechanism none"
            
            self.executor = GPUVASPExecutor(
                vasp_command=vasp_command if vasp_command else "vasp_std",
                mpi_command=mpi_command,
                num_gpus=num_gpus,
                gpu_devices=gpu_devices
            )
            print(f"  Using GPU VASP executor with {num_gpus} GPU(s)")
            
        elif execution_mode == "auto":
            # Automatic selection based on detected system
            self.executor = get_hpc_executor(self.base_dir, "auto")
            print(f"  Auto-selected executor for {detected_system}")
            
        elif execution_mode == "gaop":
            from gaop_executor import GAOPExecutor
            self.executor = GAOPExecutor(
                potential_file=self.eam_potential_file,
            )
            print(f"  Using GAOP/LAMMPS executor")
            print(f"     Potential: {self.eam_potential_file}")

        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

    def _get_next_run_number(self, directory: str) -> int:
        """Get next available run number for given directory."""
        if not os.path.exists(directory):
            return 1
        existing_dirs = [d for d in os.listdir(directory) 
                        if os.path.isdir(os.path.join(directory, d)) 
                        and d.isdigit()]
        return max(map(int, existing_dirs), default=0) + 1

    def get_reference_run_dir(self, run_type: str) -> str:
        """Get the reference directory for the next VASP run."""
        if run_type == "thermal":
            if self.thermal_run_counter == 1:
                if self.user_poscar_path and os.path.exists(self.user_poscar_path):
                    return os.path.dirname(self.user_poscar_path)
                else:
                    return self.base_dir
            # For subsequent runs, use the previous run in the current batch
            return os.path.join(self.run_dirs["thermal"],
                            str(self.current_thermal_batch),
                            str(self.thermal_run_counter - 1))
        else:  # "main" / dimer
            # For the very first main run, use the user's POSCAR directory
            if self.run_counters["main"] == 1:
                if self.user_poscar_path and os.path.exists(self.user_poscar_path):
                    return os.path.dirname(self.user_poscar_path)
                else:
                    return self.base_dir
            # For subsequent runs, use the previous run
            prev = max(1, self.run_counters["main"] - 1)
            return os.path.join(self.run_dirs["main"], str(prev))
    
    def setup_and_run_vasp(self,
                        positions: np.ndarray,
                        run_type: str = "main",
                        *,
                        snapshots_per_batch: int | None = None,
                        start_new_batch: bool = False) -> str:
        """Set up and run VASP calculation."""
        if run_type not in self.run_dirs:
            raise ValueError(f"Invalid run type: {run_type!r}")
        
        max_snaps = snapshots_per_batch or self.snapshots_per_batch
        
        # Determine the run directory
        if run_type == "thermal":
            if start_new_batch or self.thermal_run_counter > max_snaps:
                self.start_new_thermal_batch(reference_positions=positions)
            batch_dir = os.path.join(self.run_dirs["thermal"], str(self.current_thermal_batch))
            current_dir = os.path.join(batch_dir, str(self.thermal_run_counter))
        else:
            next_idx = self._get_next_run_number(self.run_dirs["main"])
            self.run_counters["main"] = next_idx
            current_dir = os.path.join(self.run_dirs["main"], str(next_idx))
        
        os.makedirs(current_dir, exist_ok=True)
        
        # For EAM/GAOP, just write POSCAR and submit
        if self.execution_mode in ("eam", "gaop"):
            # ALWAYS use the user POSCAR as template for EAM
            if self.user_poscar_path and os.path.exists(self.user_poscar_path):
                template_structure = read_structure(self.user_poscar_path)[0]
            else:
                # Create a simple structure based on positions
                n_atoms = positions.size // 3
                # Assume Mo atoms and create a simple cubic cell
                from pymatgen.core import Structure, Lattice
                lattice = Lattice.cubic(10.0 * (n_atoms ** (1/3)))  # Simple estimate
                species = ["Mo"] * n_atoms
                coords = positions.reshape(-1, 3)
                template_structure = Structure(lattice, species, coords, coords_are_cartesian=True)
            
            new_structure = get_structure_from_positions(positions, template_structure)
            write_structure(os.path.join(current_dir, 'POSCAR'), new_structure)
            
            # Create minimal placeholder files to satisfy checks
            for fname in ['INCAR', 'KPOINTS', 'POTCAR']:
                fpath = os.path.join(current_dir, fname)
                if not os.path.exists(fpath):
                    with open(fpath, 'w') as f:
                        f.write(f"# Placeholder {fname} for EAM calculation\n")
            
            # Submit the job
            job_info = self.executor.submit_job(current_dir, run_type)
            
            # Track thermal runs
            if run_type == "thermal":
                self.current_thermal_batch_dirs.append(current_dir)
                self.thermal_run_counter += 1
                if self.execution_mode in ["stampede3", "anvil", "eam", "gaop"]:
                    self._pending_thermal_dirs.append(current_dir)

            return current_dir

        # For VASP mode: write POSCAR and copy/generate VASP input files
        if self.execution_mode == "vasp":
            # Use user POSCAR as template
            if self.user_poscar_path and os.path.exists(self.user_poscar_path):
                template_structure = read_structure(self.user_poscar_path)[0]
            else:
                raise ValueError("user_poscar_path must be set for VASP execution mode")

            new_structure = get_structure_from_positions(positions, template_structure)
            write_structure(os.path.join(current_dir, 'POSCAR'), new_structure)

            # Find and copy VASP input files using smart search
            ref_dir = os.path.dirname(os.path.abspath(self.user_poscar_path))
            for fname in ['INCAR', 'KPOINTS', 'POTCAR']:
                dst = os.path.join(current_dir, fname)
                if not os.path.exists(dst):
                    src = self._find_vasp_input_file(fname, ref_dir, template_structure)
                    if src:
                        shutil.copy2(src, dst)
                    else:
                        # Last resort: generate sensible defaults
                        self._generate_default_vasp_input(fname, current_dir, template_structure)

            # Submit the job
            job_info = self.executor.submit_job(current_dir, run_type)

            # Track thermal runs
            if run_type == "thermal":
                self.current_thermal_batch_dirs.append(current_dir)
                self.thermal_run_counter += 1
                self._pending_thermal_dirs.append(current_dir)

            return current_dir

        # Original code for non-EAM calculations continues here...
        prev_dir = self.get_reference_run_dir(run_type)
        
        try:
            setup_next_vasp_run(prev_run_dir=prev_dir, 
                            next_run_dir=current_dir, 
                            positions=positions)
            
            # Submit the job using the executor
            job_info = self.executor.submit_job(current_dir, run_type)
            
            # Track thermal runs
            if run_type == "thermal":
                self.current_thermal_batch_dirs.append(current_dir)
                self.thermal_run_counter += 1
                
                if self.execution_mode in ["stampede3", "anvil", "eam"]:
                    self._pending_thermal_dirs.append(current_dir)
            
            return current_dir
            
        except Exception as err:
            print(f"Error setting up VASP run: {err}")
            shutil.rmtree(current_dir, ignore_errors=True)
            raise
    
    def setup_dimer_run(self, positions: np.ndarray) -> str:
        """Set up a new dimer run directory with the given positions."""
        next_idx = self._get_next_run_number(self.run_dirs["main"])
        self.run_counters["main"] = next_idx
        run_dir = os.path.join(self.run_dirs["main"], str(next_idx))
        os.makedirs(run_dir, exist_ok=True)
        
        if self.user_poscar_path and os.path.exists(self.user_poscar_path):
            template_structure = read_structure(self.user_poscar_path)[0]
            input_dir = os.path.dirname(self.user_poscar_path)
        else:
            raise ValueError("No valid user POSCAR provided for dimer run")
        
        new_structure = get_structure_from_positions(positions, template_structure)
        write_structure(os.path.join(run_dir, 'POSCAR'), new_structure)
        
        # Copy input files
        for fname in ['INCAR', 'POTCAR', 'KPOINTS', 'job.sh']:
            src = os.path.join(input_dir, fname)
            dst = os.path.join(run_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        return run_dir
    
    def start_new_thermal_batch(self, reference_positions: np.ndarray = None) -> str:
        """Begin the next batch of thermal-snapshot VASP runs."""
        self.current_thermal_batch += 1
        self.thermal_run_counter = 1
        batch_dir = os.path.join(self.run_dirs["thermal"], str(self.current_thermal_batch))
        os.makedirs(batch_dir, exist_ok=True)
        print(f"\nStarted thermal batch {self.current_thermal_batch} in: {batch_dir}")
        self.current_thermal_batch_dirs = []
        self.current_thermal_batch_info = None
        return batch_dir
    
    def _find_vasp_input_file(self, fname: str, poscar_ref_dir: str,
                               template_structure: Structure) -> Optional[str]:
        """Find a VASP input file using multiple naming conventions and locations.

        Search order:
        1. vasp_input_dir with temperature-specific names (e.g., INCAR_2600K)
        2. vasp_input_dir with generic names (e.g., INCAR)
        3. vasp_input_dir with species-suffixed names (e.g., POTCAR_ZrO2)
        4. POSCAR reference directory with standard names

        Args:
            fname: Target file name (INCAR, KPOINTS, or POTCAR)
            poscar_ref_dir: Directory containing the user's POSCAR file
            template_structure: Reference structure for species info

        Returns:
            Path to the found file, or None if not found
        """
        candidates = []

        # Build species string for POTCAR naming (e.g., "ZrO2" from Zr+O species)
        seen = []
        for site in template_structure:
            sym = str(site.specie)
            if sym not in seen:
                seen.append(sym)
        species_str = "".join(seen)  # e.g., "ZrO"

        if self.vasp_input_dir and os.path.isdir(self.vasp_input_dir):
            vasp_dir = self.vasp_input_dir

            if fname == 'INCAR':
                # Temperature-specific names first
                if self.temperature is not None:
                    temp_int = int(self.temperature)
                    candidates.append(os.path.join(vasp_dir, f'INCAR_{temp_int}K'))
                    candidates.append(os.path.join(vasp_dir, f'INCAR_{temp_int}'))
                    candidates.append(os.path.join(vasp_dir, f'INCAR_{self.temperature}K'))
                    candidates.append(os.path.join(vasp_dir, f'INCAR_{self.temperature}'))
                candidates.append(os.path.join(vasp_dir, 'INCAR'))

            elif fname == 'KPOINTS':
                candidates.append(os.path.join(vasp_dir, 'KPOINTS'))

            elif fname == 'POTCAR':
                # Try species-suffixed names, then common suffixes
                candidates.append(os.path.join(vasp_dir, f'POTCAR_{species_str}'))
                # Also try with number suffix (e.g., POTCAR_ZrO2)
                for suffix in [species_str + "2", species_str]:
                    candidates.append(os.path.join(vasp_dir, f'POTCAR_{suffix}'))
                candidates.append(os.path.join(vasp_dir, 'POTCAR'))

        # Fallback: POSCAR reference directory with standard names
        candidates.append(os.path.join(poscar_ref_dir, fname))

        # Deduplicate while preserving order
        seen_paths = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen_paths:
                seen_paths.add(c)
                unique_candidates.append(c)

        for candidate in unique_candidates:
            if os.path.exists(candidate):
                print(f"  Found {fname}: {candidate}")
                return candidate

        return None

    def _generate_default_vasp_input(self, fname: str, run_dir: str,
                                       template_structure: Structure) -> None:
        """Generate default VASP input file if not provided by user.

        Args:
            fname: File name (INCAR, KPOINTS, or POTCAR)
            run_dir: Target run directory
            template_structure: Reference structure for species info
        """
        fpath = os.path.join(run_dir, fname)

        if fname == 'INCAR':
            incar = Incar({
                'SYSTEM': 'TD_SPF_calculation',
                'PREC': 'Accurate',
                'ENCUT': 520,
                'EDIFF': 1E-6,
                'IBRION': -1,
                'NSW': 0,
                'ISMEAR': 0,
                'SIGMA': 0.05,
                'LREAL': 'Auto',
                'ALGO': 'Normal',
                'NELM': 200,
                'LWAVE': False,
                'LCHARG': False,
            })
            incar.write_file(fpath)
            print(f"  Generated default INCAR in {run_dir}")

        elif fname == 'KPOINTS':
            kpoints = Kpoints.gamma_automatic(kpts=(1, 1, 1))
            kpoints.write_file(fpath)
            print(f"  Generated default KPOINTS (Gamma 1x1x1) in {run_dir}")

        elif fname == 'POTCAR':
            # POTCAR cannot be auto-generated without pseudopotential library
            # Try using pymatgen if VASP_PSP_DIR is set
            try:
                from pymatgen.io.vasp import Potcar
                species = []
                seen = set()
                for site in template_structure:
                    sym = str(site.specie)
                    if sym not in seen:
                        species.append(sym)
                        seen.add(sym)
                potcar = Potcar(species)
                potcar.write_file(fpath)
                print(f"  Auto-generated POTCAR for species: {species}")
            except Exception as e:
                print(f"  WARNING: POTCAR not found and could not be auto-generated: {e}")
                print(f"  Please place POTCAR in: {os.path.dirname(os.path.abspath(self.user_poscar_path))}")
                print(f"  VASP calculations will fail without POTCAR")

    def submit_pending_thermal_batch(self) -> Any:
        """Submit all pending thermal calculations as a batch."""
        print(f"\nVASPManager.submit_pending_thermal_batch called")
        print(f"  execution_mode: {self.execution_mode}")
        print(f"  _pending_thermal_dirs: {len(self._pending_thermal_dirs)}")
        
        if self.execution_mode in ["stampede3", "anvil", "eam", "vasp"]:
            # These modes support batch submission
            if not self._pending_thermal_dirs:
                print("  WARNING: _pending_thermal_dirs is empty!")
                return None
            print(f"  Calling executor.submit_batch with {len(self._pending_thermal_dirs)} dirs")
            batch_info = self.executor.submit_batch(self._pending_thermal_dirs)
            self.current_thermal_batch_info = batch_info
            self.current_thermal_batch_dirs = self._pending_thermal_dirs.copy()
            self._pending_thermal_dirs = []
            return batch_info
        else:
            print(f"  Mode {self.execution_mode} doesn't use batch submission")
            # Other modes don't need explicit batch submission
            return None
    
    def wait_for_current_thermal_batch(self, check_interval: int = 30) -> List[VASPRun]:
        """Wait for all jobs in the current thermal batch to complete."""
        if not self.current_thermal_batch_dirs:
            return []
        
        print(f"Waiting for {len(self.current_thermal_batch_dirs)} calculations to complete...")
        print(f"  current_thermal_batch_info: {self.current_thermal_batch_info}")
        print(f"  execution_mode: {self.execution_mode}")
        
        # For EAM mode without batch info, we need to submit the jobs first
        if self.execution_mode == "eam" and self.current_thermal_batch_info is None:
            print("  EAM mode detected with no batch info - jobs may need to be submitted")
            # The jobs should have been submitted by submit_pending_thermal_batch
            # If not, we have a problem
            if self._pending_thermal_dirs:
                print(f"  WARNING: {len(self._pending_thermal_dirs)} jobs still pending!")
        
        # Use executor's batch wait method
        results = self.executor.wait_for_batch(
            self.current_thermal_batch_info,
            self.current_thermal_batch_dirs
        )
        
        # Clean up
        self.current_thermal_batch_dirs.clear()
        self.current_thermal_batch_info = None
        
        return results
    
    def wait_for_completion(self, run_dir: str, 
                          max_wait_time: int = 86400, 
                          check_interval: int = 30) -> VASPRun:
        """Wait for a specific VASP job to complete."""
        # For EAM, results are instant
        if self.execution_mode == "eam":
            from eam_executor import EAMRun
            return EAMRun(run_dir)
        # For thermal runs that are part of a batch, use batch waiting
        if run_dir in self.current_thermal_batch_dirs and self.execution_mode == "stampede3":
            # For Stampede3, we need to wait for the whole batch
            results = self.wait_for_current_thermal_batch(check_interval)
            # Find the specific result
            for vr in results:
                if getattr(vr, 'run_dir', None) == run_dir:
                    return vr
            raise ValueError(f"Run directory {run_dir} not found in batch results")
        
        # For individual runs, use executor's wait method
        return self.executor.wait_for_completion(run_dir, 
                                               max_wait_time=max_wait_time,
                                               check_interval=check_interval)
    
    def get_latest_thermal_run(self) -> str:
        """Get the path to the latest thermal run."""
        if self.thermal_run_counter == 1:
            raise ValueError("No thermal runs completed yet")
        
        batch_dir = os.path.join(self.run_dirs['thermal'], str(self.current_thermal_batch))
        return os.path.join(batch_dir, str(self.thermal_run_counter - 1))
