#!/usr/bin/env python
"""
s-TDEP calculation script that integrates with existing VASP/EAM infrastructure.
Performs iterative TDEP calculations with proper file and folder management.

Version 1.7.0 - Added --continue flag to resume interrupted runs
             - Detects last completed step and continues from there
             - Loads original parameters from run_metadata.json
             - Restores snapshot count from last step
Version 1.6.3 - Added --skip-existing flag to skip runs if output folder exists
Version 1.6.2 - Improved logging for soft mode eigenvalue calculation
             - Now logs when calculation starts and if it fails
             - Better error messages for missing dependencies (phonopy/vibes)
Version 1.6.1 - Clarified snapshot increase logic:
             - FC norm not converged: proceed to next step (no snapshot increase)
             - Soft mode eigenvalues not converged: increase snapshots
             - Free energy change too large: increase snapshots
Version 1.6.0 - Improved eigenvalue convergence check using soft mode at q=2/3[111]
             - Previous versions checked Gamma point which was always ~0 THz
             - Now checks the physically relevant soft mode for BCC stability
"""

__version__ = "1.7.0"

import os
import sys

# Set OMP_NUM_THREADS to 1 to avoid symmetry detection issues in extract_forceconstants
os.environ['OMP_NUM_THREADS'] = '1'

import shutil
import subprocess
import time
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your existing infrastructure
from vasp_manager import VASPManager
from vasp_executors import get_hpc_executor, detect_hpc_system
from output_manager import OutputManager, get_output_path, get_input_path

# Import for soft mode eigenvalue calculation
try:
    from convert_tdep_fc_to_phonopy import parse_tdep_forceconstant
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from ase.io import read as ase_read
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False


class STDEPCalculator:
    """Main class for s-TDEP calculations with VASP/EAM support."""
    
    def __init__(
        self,
        n_snapshots: int = 100,
        temperature: float = 1400.0,
        max_frequency: Optional[float] = None,
        debye_temperature: Optional[float] = None,
        n_steps: int = 10,
        rc2: float = 12.0,
        rc3: float = 4.0,
        qpoint_grid: List[int] = None,
        execution_mode: str = "mpi",
        vasp_command: str = "vasp_gam",
        mpi_command: Optional[str] = None,
        eam_potential_file: Optional[str] = None,
        kim_model: Optional[str] = None,
        gulp_library_file: Optional[str] = None,
        lammps_potential_file: Optional[str] = None,
        base_dir: str = "tdep_calculations",
        verbose: bool = True,
        show_tdep_output: bool = True,
        convergence_threshold: float = 0.001,
        convergence_factor: float = 2.0,
        snapshot_increment: int = 50,
        max_snapshots: int = 500,
        norotational: bool = False,
        nohuang: bool = False,
        nohermitian: bool = False,
        # New parameters for v1.5.0
        supercell_file: Optional[str] = None,
        unitcell_file: Optional[str] = None,
        min_steps: int = 1,
        fc_convergence: bool = False,
        fc_convergence_threshold: float = 0.005,
        eigenvalue_convergence: bool = False,
        eigenvalue_threshold: float = 0.03,
        # New parameters for v1.7.0 (continuation support)
        continue_from: Optional[str] = None,
        start_step: int = 1,
        initial_snapshots: Optional[int] = None,
        # VASP GPU batch parameters
        vasp_batch_size: int = 10,
        num_gpus: int = 6,
        gpu_ids: Optional[List[int]] = None,
        parallel: bool = False,
        # Cleanup options
        cleanup_calc: bool = False
    ):
        """
        Initialize s-TDEP calculator.
        
        Args:
            n_snapshots: Number of snapshots per step
            temperature: Temperature in Kelvin
            max_frequency: Maximum frequency for first step (THz). Mutually exclusive with debye_temperature
            debye_temperature: Debye temperature for first step (K). Mutually exclusive with max_frequency
            n_steps: Number of TDEP iterations
            rc2: Cutoff for second order force constants
            rc3: Cutoff for third order force constants
            qpoint_grid: Q-point grid for anharmonic free energy [nx, ny, nz]
            execution_mode: "mpi", "slurm", "stampede3", "anvil", "eam", "gulp", "lammps", "mock"
            vasp_command: VASP executable name
            mpi_command: Custom MPI command
            eam_potential_file: Path to EAM potential file
            kim_model: KIM model name for EAM
            gulp_library_file: Path to GULP library file
            lammps_potential_file: Path to LAMMPS potential file
            base_dir: Base directory for calculations
            verbose: Enable verbose output
            show_tdep_output: Show real-time output from TDEP commands
            convergence_threshold: Threshold for free energy convergence (eV/atom)
            convergence_factor: Factor for convergence threshold to trigger snapshot increase
            snapshot_increment: Number of snapshots to add if not converged
            max_snapshots: Maximum number of snapshots to try
            norotational: Disable rotational invariance in force constant extraction
            nohuang: Disable Huang invariances in force constant extraction
            nohermitian: Disable hermitian constraint in force constant extraction
            supercell_file: Custom supercell POSCAR file path (will be copied to infile.ssposcar)
            unitcell_file: Custom unit cell POSCAR file path (will be copied to infile.ucposcar)
            min_steps: Minimum number of TDEP steps before checking convergence
            fc_convergence: Enable force constant norm convergence check
            fc_convergence_threshold: Threshold for FC norm convergence (relative change, default 0.005 - Mo achieves ~0.001-0.004 when converged, Zr stays at ~0.03)
            eigenvalue_convergence: Enable soft mode eigenvalue stability check at q=2/3[111]
            eigenvalue_threshold: Threshold for soft mode eigenvalue change in THz (default 0.03 - based on Zr convergence analysis)
            continue_from: Path to existing run folder to continue from (None for new run)
            start_step: Step number to start from (used internally for continuation)
            initial_snapshots: Initial snapshot count for continuation (overrides n_snapshots for first continued step)
            cleanup_calc: Delete calc_* directories after processing to save disk space
        """
        self.n_snapshots = n_snapshots
        self.temperature = temperature
        self.max_frequency = max_frequency
        self.debye_temperature = debye_temperature
        self.n_steps = n_steps
        self.rc2 = rc2
        self.rc3 = rc3
        self.qpoint_grid = qpoint_grid if qpoint_grid else [5, 5, 5]
        self.execution_mode = execution_mode
        self.vasp_command = vasp_command
        self.mpi_command = mpi_command
        self.eam_potential_file = eam_potential_file
        self.kim_model = kim_model
        self.gulp_library_file = gulp_library_file
        self.lammps_potential_file = lammps_potential_file
        self.base_dir = Path(base_dir).absolute()
        self.verbose = verbose
        self.show_tdep_output = show_tdep_output
        self.convergence_threshold = convergence_threshold
        self.convergence_factor = convergence_factor
        self.snapshot_increment = snapshot_increment
        self.max_snapshots = max_snapshots
        self.norotational = norotational
        self.nohuang = nohuang
        self.nohermitian = nohermitian

        # New parameters for v1.5.0
        self.supercell_file = supercell_file
        self.unitcell_file = unitcell_file
        self.min_steps = min_steps
        self.fc_convergence = fc_convergence
        self.fc_convergence_threshold = fc_convergence_threshold
        self.eigenvalue_convergence = eigenvalue_convergence
        self.eigenvalue_threshold = eigenvalue_threshold

        # Continuation support (v1.7.0)
        self.continue_from = continue_from
        self.start_step = start_step
        self.initial_snapshots = initial_snapshots

        # VASP GPU batch parameters
        self.vasp_batch_size = vasp_batch_size
        self.num_gpus = num_gpus
        self.gpu_ids = gpu_ids
        self.parallel = parallel

        # Cleanup options
        self.cleanup_calc = cleanup_calc

        # Track free energies for convergence
        self.free_energies = []
        self.snapshot_counts = []

        # Track FC norms and soft mode eigenvalues for strict convergence checks
        self.fc_norms = []
        self.softmode_eigenvalues = []  # Eigenvalues at q=2/3[111]

        # Load previous convergence data if continuing
        if self.continue_from:
            self._load_previous_convergence_data()

        # Files to copy between steps
        self.files_to_copy = [
            "process.py", "infile.forceconstant", "INCAR",
            "infile.forces", "infile.meta", "infile.positions", "infile.ssposcar",
            "infile.stat", "infile.ucposcar", "job.sh", "KPOINTS", "POTCAR",
        ]
        
        # Create base directory first (before logging)
        self.base_dir.mkdir(exist_ok=True)
        
        # Setup logging (after directory exists)
        self._setup_logging()

        # Log version information
        self.logger.info(f"s-TDEP Calculator version {__version__}")

        # Handle custom POSCAR files (copy and rename to infile.ssposcar/infile.ucposcar)
        self._setup_poscar_files()

        # Initialize executor directly (not VASPManager)
        self._init_executor()

    def _load_previous_convergence_data(self):
        """Load previous convergence data when continuing a run."""
        convergence_file = self.base_dir / "convergence_data.txt"
        if not convergence_file.exists():
            return

        print(f"Loading previous convergence data from {convergence_file}")

        with open(convergence_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        step = int(parts[0])
                        snapshots = int(parts[1])
                        free_energy = float(parts[2])
                        self.free_energies.append(free_energy)
                        self.snapshot_counts.append(snapshots)

                        # Load FC norm if available
                        if len(parts) >= 4 and parts[3] != '-':
                            try:
                                fc_norm = float(parts[3])
                                self.fc_norms.append(fc_norm)
                            except ValueError:
                                pass

                    except (ValueError, IndexError):
                        continue

        if self.free_energies:
            print(f"  Loaded {len(self.free_energies)} previous steps")
            print(f"  Free energy history: {[f'{fe:.6f}' for fe in self.free_energies]}")
            print(f"  Snapshot counts: {self.snapshot_counts}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.base_dir / "stdep_calculation.log"
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_poscar_files(self):
        """
        Handle custom POSCAR file inputs.

        If user provides custom supercell_file or unitcell_file, copy them to the
        base directory with the required TDEP names (infile.ssposcar, infile.ucposcar).
        """
        # Handle supercell file
        if self.supercell_file:
            src_path = Path(self.supercell_file)
            if not src_path.exists():
                # Try to find in inputs directory
                try:
                    src_path = Path(get_input_path(self.supercell_file))
                except:
                    pass

            if not src_path.exists():
                raise FileNotFoundError(f"Supercell file not found: {self.supercell_file}")

            # Copy to base_dir as infile.ssposcar (where _init_executor looks for it)
            dest_path = self.base_dir / "infile.ssposcar"
            shutil.copy2(src_path, dest_path)
            self.logger.info(f"Copied supercell file '{self.supercell_file}' to '{dest_path}'")

        # Handle unitcell file
        if self.unitcell_file:
            src_path = Path(self.unitcell_file)
            if not src_path.exists():
                # Try to find in inputs directory
                try:
                    src_path = Path(get_input_path(self.unitcell_file))
                except:
                    pass

            if not src_path.exists():
                raise FileNotFoundError(f"Unit cell file not found: {self.unitcell_file}")

            # Copy to base_dir as infile.ucposcar (consistent with supercell handling)
            dest_path = self.base_dir / "infile.ucposcar"
            shutil.copy2(src_path, dest_path)
            self.logger.info(f"Copied unit cell file '{self.unitcell_file}' to '{dest_path}'")

    def _init_executor(self):
        """Initialize the appropriate executor directly."""
        # Look for infile.ssposcar in base_dir first, then inputs directory
        user_poscar = self.base_dir / "infile.ssposcar"
        
        if not user_poscar.exists():
            # Try to find in inputs directory
            try:
                input_path = get_input_path("infile.ssposcar")
                if Path(input_path).exists():
                    shutil.copy2(input_path, user_poscar)
                    self.logger.info(f"Copied infile.ssposcar from inputs/ directory to {self.base_dir}")
                else:
                    # Also check current directory as fallback
                    if Path("infile.ssposcar").exists():
                        shutil.copy2("infile.ssposcar", user_poscar)
                        self.logger.info(f"Copied infile.ssposcar from current directory to {self.base_dir}")
                    else:
                        raise FileNotFoundError("infile.ssposcar not found in current directory or inputs/")
            except Exception as e:
                raise FileNotFoundError(f"infile.ssposcar not found: {e}")
        
        self.user_poscar_path = str(user_poscar)
        
        # For EAM mode with TDEP, we need special handling
        if self.execution_mode == "eam":
            # Import and use the enhanced EAM executor
            try:
                from eam_executor_md import create_tdep_compatible_eam_executor
                
                # Create TDEP-compatible EAM executor
                self.executor = create_tdep_compatible_eam_executor(
                    kim_model_name=self.kim_model,
                    potential_file=self.eam_potential_file,
                    user_poscar_path=self.user_poscar_path,
                    temperature=self.temperature,
                    timestep=1.0,  # fs, matching VASP default
                    perform_md_step=True  # Enable MD for TDEP
                )
                
                self.logger.info("Using TDEP-compatible EAM executor with MD support")
                
            except ImportError:
                self.logger.error("Enhanced EAM executor not found!")
                raise
        elif self.execution_mode == "gulp":
            # Import and use the GULP executor
            try:
                from gulp_executor_md import create_tdep_compatible_gulp_executor

                if not self.gulp_library_file:
                    raise ValueError("GULP execution mode requires gulp_library_file parameter")

                # Create TDEP-compatible GULP executor
                self.executor = create_tdep_compatible_gulp_executor(
                    library_file=self.gulp_library_file,
                    user_poscar_path=self.user_poscar_path,
                    temperature=self.temperature,
                    timestep=1.0,  # fs, matching VASP default
                    perform_md_step=True  # Enable MD for TDEP
                )

                self.logger.info("Using TDEP-compatible GULP executor with MD support")
                self.logger.info(f"GULP library file: {self.gulp_library_file}")

            except ImportError:
                self.logger.error("GULP executor not found!")
                raise
        elif self.execution_mode == "lammps":
            # Import and use the LAMMPS executor
            try:
                from lammps_executor_md import create_tdep_compatible_lammps_executor

                if not self.lammps_potential_file:
                    raise ValueError("LAMMPS execution mode requires lammps_potential_file parameter")

                # Create TDEP-compatible LAMMPS executor
                self.executor = create_tdep_compatible_lammps_executor(
                    lammps_potential_file=self.lammps_potential_file,
                    user_poscar_path=self.user_poscar_path,
                    temperature=self.temperature,
                    timestep=1.0,  # fs, matching VASP default
                    perform_md_step=True  # Enable MD for TDEP
                )

                self.logger.info("Using TDEP-compatible LAMMPS executor with MD support")
                self.logger.info(f"LAMMPS potential file: {self.lammps_potential_file}")

            except ImportError:
                self.logger.error("LAMMPS executor not found!")
                raise
        else:
            # For VASP modes, get the appropriate executor
            self.executor = get_hpc_executor(
                str(self.base_dir),
                self.execution_mode,
                mpi_command=self.mpi_command,
                vasp_command=self.vasp_command,
                user_poscar_path=self.user_poscar_path,
                vasp_batch_size=self.vasp_batch_size,
                num_gpus=self.num_gpus,
                gpu_ids=self.gpu_ids,
                parallel=self.parallel
            )
    
    def _run_tdep_command(self, cmd: List[str], command_name: str, timeout: Optional[int] = None):
        """Run a TDEP command with real-time output display."""
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        if self.show_tdep_output:
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Collect output while displaying it
            output_lines = []
            try:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    line = line.rstrip()
                    output_lines.append(line)
                    # Print with indentation for clarity
                    print(f"  [{command_name}] {line}")
                    
                    # Also log important lines
                    if any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'negative']):
                        self.logger.warning(f"{command_name} output: {line}")
                        
                process.wait()
                
            except KeyboardInterrupt:
                process.terminate()
                raise
                
            output = '\n'.join(output_lines)
            returncode = process.returncode
            
        else:
            # Run silently (original behavior)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            output = result.stdout
            returncode = result.returncode
            
        return returncode, output

    @staticmethod
    def detect_continuation_state(run_folder: str) -> Tuple[int, int, Dict[str, Any], bool]:
        """
        Detect the state of an existing run for continuation.

        Args:
            run_folder: Path to the run folder (e.g., outputs/Zr_defected_small_1300K)

        Returns:
            Tuple of (next_step, snapshot_count, original_params, already_converged)
            - next_step: The step number to continue from
            - snapshot_count: The number of snapshots to use for next step (may be increased)
            - original_params: Dictionary of original run parameters
            - already_converged: True if the previous run already met convergence criteria

        Raises:
            FileNotFoundError: If run folder or required files don't exist
            ValueError: If no completed steps found
        """
        run_path = Path(run_folder)
        if not run_path.exists():
            raise FileNotFoundError(f"Run folder not found: {run_folder}")

        # Load original parameters from run_metadata.json
        metadata_file = run_path / "run_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"run_metadata.json not found in {run_folder}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        original_params = metadata.get('parameters', {})

        # Find the tdep_calculations directory
        tdep_dir = run_path / "tdep_calculations"
        if not tdep_dir.exists():
            raise FileNotFoundError(f"tdep_calculations directory not found in {run_folder}")

        # Find all step directories and determine the last completed one
        step_dirs = sorted(tdep_dir.glob("step_*"), key=lambda p: int(p.name.split('_')[1]))

        if not step_dirs:
            raise ValueError(f"No step directories found in {tdep_dir}")

        # Find the last step with outfile.forceconstant (indicates completion)
        last_completed_step = 0
        last_snapshot_count = original_params.get('snapshots', 100)

        for step_dir in step_dirs:
            step_num = int(step_dir.name.split('_')[1])
            fc_file = step_dir / "outfile.forceconstant"

            if fc_file.exists():
                last_completed_step = step_num
                # Count calc directories to get snapshot count
                calc_dirs = list(step_dir.glob("calc_*"))
                if calc_dirs:
                    last_snapshot_count = len(calc_dirs)
                else:
                    # Fallback: count contcar_conf* files (calc dirs may have been cleaned up)
                    contcar_files = list(step_dir.glob("contcar_conf*"))
                    if contcar_files:
                        last_snapshot_count = len(contcar_files)

        if last_completed_step == 0:
            # Check if step_1 exists but has no forceconstant (incomplete first step)
            if (tdep_dir / "step_1").exists():
                # Check if there's an infile.forceconstant from base dir (for step 1)
                # In this case, start from step 1
                return 1, original_params.get('snapshots', 100), original_params, False
            raise ValueError(f"No completed steps found in {tdep_dir}. "
                           "Cannot continue - try running from scratch.")

        # Next step is last_completed + 1
        next_step = last_completed_step + 1

        # Check if we should increase snapshots for the next step based on convergence data
        next_snapshot_count = last_snapshot_count
        convergence_file = tdep_dir / "convergence_data.txt"
        free_energies = []
        snapshot_counts = []

        if convergence_file.exists():
            # Parse convergence data to check if snapshot increase is needed

            with open(convergence_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            step = int(parts[0])
                            snapshots = int(parts[1])
                            free_energy = float(parts[2])
                            free_energies.append(free_energy)
                            snapshot_counts.append(snapshots)
                        except (ValueError, IndexError):
                            continue

            # Check if the last step would have triggered a snapshot increase
            if len(free_energies) >= 2:
                convergence_threshold = original_params.get('convergence_threshold', 0.001)
                convergence_factor = original_params.get('convergence_factor', 10.0)
                snapshot_increment = original_params.get('snapshot_increment', 100)
                max_snapshots = original_params.get('max_snapshots', 1000)

                recent_change = abs(free_energies[-1] - free_energies[-2])
                threshold_for_increase = convergence_threshold * convergence_factor

                if recent_change > threshold_for_increase and last_snapshot_count < max_snapshots:
                    next_snapshot_count = min(last_snapshot_count + snapshot_increment, max_snapshots)
                    print(f"  Snapshot increase triggered: free energy change {recent_change:.6f} > {threshold_for_increase:.6f}")
            elif len(free_energies) == 1:
                # Only one step completed, check if free energy was None (negative eigenvalues)
                # In that case, increase snapshots
                pass  # Can't determine from convergence_data.txt alone

        # Check if the previous run already converged
        already_converged = False
        if convergence_file.exists() and len(free_energies) >= 2:
            conv_threshold = original_params.get('convergence_threshold', 0.001)
            min_steps = original_params.get('min_steps', 1)
            fc_conv_enabled = original_params.get('fc_convergence', False)
            fc_conv_threshold = original_params.get('fc_convergence_threshold', 0.005)

            if len(free_energies) >= min_steps:
                fe_change = abs(free_energies[-1] - free_energies[-2])
                fe_ok = fe_change < conv_threshold

                fc_ok = True  # Assume OK if FC convergence not enabled
                if fc_conv_enabled and len(free_energies) >= 2:
                    # Re-parse FC norms from convergence data
                    fc_norms = []
                    with open(convergence_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('#') or not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 4:
                                try:
                                    fc_norms.append(float(parts[3]))
                                except (ValueError, IndexError):
                                    continue
                    if len(fc_norms) >= 2:
                        fc_prev = fc_norms[-2]
                        fc_rel_change = abs(fc_norms[-1] - fc_prev) / fc_prev if fc_prev > 0 else float('inf')
                        fc_ok = fc_rel_change < fc_conv_threshold

                if fe_ok and fc_ok:
                    already_converged = True

        print(f"Continuation state detected:")
        print(f"  Last completed step: {last_completed_step}")
        print(f"  Snapshots in last step: {last_snapshot_count}")
        if next_snapshot_count != last_snapshot_count:
            print(f"  Snapshots for next step: {next_snapshot_count} (increased)")
        else:
            print(f"  Snapshots for next step: {next_snapshot_count}")
        if already_converged:
            print(f"  Previous run ALREADY CONVERGED - no further steps needed")
        else:
            print(f"  Will continue from step: {next_step}")

        return next_step, next_snapshot_count, original_params, already_converged

    def run(self):
        """Run the complete s-TDEP calculation workflow."""
        # Determine starting step and initial snapshots
        start_step = self.start_step
        current_snapshots = self.initial_snapshots if self.initial_snapshots else self.n_snapshots

        if start_step > 1:
            self.logger.info(f"CONTINUING s-TDEP calculation from step {start_step}")
            self.logger.info(f"Using {current_snapshots} snapshots (from previous run)")
        else:
            self.logger.info(f"Starting s-TDEP calculation with {self.n_steps} steps")

        self.logger.info(f"Temperature: {self.temperature} K, Initial snapshots per step: {current_snapshots}")
        self.logger.info(f"Q-point grid: {' '.join(map(str, self.qpoint_grid))}")
        self.logger.info(f"Execution mode: {self.execution_mode}")
        self.logger.info(f"Convergence threshold: {self.convergence_threshold} eV/atom")

        # Save original directory
        original_dir = Path.cwd()

        try:
            # Change to base directory
            os.chdir(self.base_dir)

            # Check required files
            self._check_required_files()

            # Run iterative TDEP steps
            for step in range(start_step, self.n_steps + 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Starting TDEP Step {step}/{self.n_steps}")
                self.logger.info(f"Using {current_snapshots} snapshots")
                self.logger.info(f"{'='*60}")
            
                step_dir = self.base_dir / f"step_{step}"
                step_dir.mkdir(exist_ok=True)
                
                # Change to step directory
                os.chdir(step_dir)
            
                try:
                    # Copy necessary files
                    self._copy_files_to_step(step)
                
                    # Generate snapshots with current count
                    self._generate_snapshots(step, current_snapshots)
                    
                    # Prepare VASP/EAM directories
                    snapshot_dirs = self._prepare_calculation_directories()
                    
                    # Run calculations directly in TDEP directories
                    self._run_calculations_in_tdep_dirs(snapshot_dirs, step)
                    
                    # Collect results
                    self._collect_results()

                    # Cleanup calc_* directories if requested
                    self._cleanup_calc_directories()

                    # Extract force constants
                    self._extract_force_constants()
                    
                    # Calculate anharmonic free energy and get the value
                    free_energy = self._calculate_anharmonic_free_energy()
                    
                    # Track the values
                    if free_energy is not None:
                        self.free_energies.append(free_energy)
                        self.snapshot_counts.append(current_snapshots)
                        self.logger.info(f"Step {step} free energy: {free_energy:.6f} eV/atom")
                        
                        # Update convergence plot after each step
                        self._save_convergence_data()
                        self._create_convergence_plot()
                        
                        # Log convergence history
                        if len(self.free_energies) > 1:
                            self.logger.info("Free energy history:")
                            for i, (fe, ns) in enumerate(zip(self.free_energies, self.snapshot_counts)):
                                if i == 0:
                                    self.logger.info(f"  Step {i+1}: {fe:.6f} eV/atom ({ns} snapshots)")
                                else:
                                    change = fe - self.free_energies[i-1]
                                    self.logger.info(f"  Step {i+1}: {fe:.6f} eV/atom ({ns} snapshots) [change: {change:+.6f}]")
                    else:
                        self.logger.warning(f"Step {step}: No free energy obtained (negative eigenvalues)")
                        # Still update plot to show the gap
                        self._save_convergence_data()
                        self._create_convergence_plot()
                    
                    # Check convergence
                    converged = self._check_convergence(step)
                    if converged:
                        self.logger.info("All convergence criteria met! Stopping iterations.")
                        break
                    elif not converged and current_snapshots < self.max_snapshots:
                        # Decide whether to increase snapshots
                        should_increase = False
                        increase_reason = ""

                        if free_energy is None:
                            # Always increase for negative eigenvalues
                            should_increase = True
                            increase_reason = "due to negative eigenvalues"
                        elif len(self.free_energies) >= 2:
                            # Check if the change is too large (poor convergence)
                            recent_change = abs(self.free_energies[-1] - self.free_energies[-2])
                            threshold_for_increase = self.convergence_threshold * self.convergence_factor

                            if recent_change > threshold_for_increase:
                                should_increase = True
                                increase_reason = f"for better free energy convergence (change {recent_change:.6f} > {self.convergence_factor}×threshold {threshold_for_increase:.6f})"

                        # Check if FC norm convergence failed - do NOT increase snapshots
                        # FC norm is about the quality of force constant fitting, not sampling
                        # More snapshots won't help FC norm - need to wait for better convergence
                        # So we just proceed to next step without increasing snapshots

                        # Check if soft mode eigenvalue convergence failed - DO increase snapshots
                        # Soft mode instability indicates insufficient sampling
                        if self.eigenvalue_convergence and len(self.softmode_eigenvalues) >= 2:
                            ev_current = self.softmode_eigenvalues[-1]
                            ev_previous = self.softmode_eigenvalues[-2]
                            if ev_current is not None and ev_previous is not None:
                                max_ev_change = np.max(np.abs(ev_current - ev_previous))
                                if max_ev_change >= self.eigenvalue_threshold:
                                    should_increase = True
                                    increase_reason = f"for soft mode convergence (change {max_ev_change:.4f} THz > threshold {self.eigenvalue_threshold} THz)"

                        if should_increase:
                            current_snapshots += self.snapshot_increment
                            self.logger.info(f"Increasing snapshots to {current_snapshots} {increase_reason}")
                        
                except Exception as e:
                    self.logger.error(f"Error in step {step}: {e}")
                    raise
                finally:
                    # Return to base directory
                    os.chdir(self.base_dir)
        
            self.logger.info("\ns-TDEP calculation completed successfully!")
            self._print_summary()
            self._create_convergence_plot()
        finally:
            # Always restore original directory
            os.chdir(original_dir)
        
    def _check_required_files(self):
        """Check that all required input files exist."""
        # Try to find files in inputs directory first
        required_files = [
            "infile.ssposcar",
            "infile.ucposcar",
            "INCAR",
            "KPOINTS",
            "POTCAR",
            "process.py"
        ]
        
        # For EAM, GULP, and LAMMPS, POTCAR is not strictly required
        if self.execution_mode in ["eam", "gulp", "lammps"]:
            required_files.remove("POTCAR")
        
        # Check files and copy them from inputs/ if needed
        for fname in required_files[:]:
            dest_path = self.base_dir / fname
            if not dest_path.exists():
                # Try to find in inputs directory
                try:
                    input_path = get_input_path(fname)
                    if Path(input_path).exists():
                        shutil.copy2(input_path, dest_path)
                        self.logger.info(f"Copied {fname} from inputs/ directory to {self.base_dir}")
                    else:
                        # Also check current directory as fallback
                        if Path(fname).exists():
                            shutil.copy2(fname, dest_path)
                            self.logger.info(f"Copied {fname} from current directory to {self.base_dir}")
                        else:
                            # Special case for process.py - check scripts directory
                            if fname == "process.py":
                                scripts_path = Path(__file__).parent / "process.py"
                                if scripts_path.exists():
                                    shutil.copy2(scripts_path, dest_path)
                                    self.logger.info(f"Copied {fname} from scripts directory to {self.base_dir}")
                                else:
                                    self.logger.warning(f"File {fname} not found in inputs/, current, or scripts directory")
                            else:
                                self.logger.warning(f"File {fname} not found in inputs/ or current directory")
                except:
                    # Also check current directory as fallback
                    if Path(fname).exists():
                        shutil.copy2(fname, dest_path)
                        self.logger.info(f"Copied {fname} from current directory to {self.base_dir}")
                    elif fname == "process.py":
                        # Special case for process.py - check scripts directory
                        scripts_path = Path(__file__).parent / "process.py"
                        if scripts_path.exists():
                            shutil.copy2(scripts_path, dest_path)
                            self.logger.info(f"Copied {fname} from scripts directory to {self.base_dir}")
        
        missing_files = []
        for fname in required_files:
            if not (self.base_dir / fname).exists():
                missing_files.append(fname)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
        
        # Check if job.sh exists, create default if not
        if not (self.base_dir / "job.sh").exists() and self.execution_mode in ["slurm", "stampede3", "anvil"]:
            self._create_default_job_script()
            
    def _create_default_job_script(self):
        """Create a default job script for SLURM systems."""
        content = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=02:00:00
#SBATCH -J tdep_calc
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -p shared

# Load modules (adjust as needed)
module load vasp

# Run VASP
srun vasp_gam
"""
        job_path = self.base_dir / "job.sh"
        with open(job_path, "w") as f:
            f.write(content)
        os.chmod(job_path, 0o755)
        self.logger.info("Created default job.sh script")
        
    def _copy_files_to_step(self, step: int):
        """Copy necessary files to the step directory."""
        if step == 1:
            # First step: copy from base directory
            source_dir = self.base_dir
        else:
            # Subsequent steps: copy from previous step
            source_dir = self.base_dir / f"step_{step-1}"
        
        for fname in self.files_to_copy:
            src = source_dir / fname
            if src.exists():
                shutil.copy2(src, fname)
                self.logger.debug(f"Copied {fname} to step_{step}")
            elif fname not in ["infile.forceconstant"] or step > 1:
                # forceconstant won't exist in step 1
                self.logger.warning(f"File {fname} not found in {source_dir}")
                
    def _generate_snapshots(self, step: int, num_snapshots: int):
        """Generate thermal snapshots using canonical_configuration."""
        self.logger.info(f"Generating {num_snapshots} snapshots at {self.temperature} K")

        # Clean up old snapshot files before generating new ones
        old_snapshots = list(Path(".").glob("contcar_conf[0-9][0-9][0-9][0-9]"))
        if old_snapshots:
            self.logger.info(f"Cleaning up {len(old_snapshots)} old snapshot files")
            for f in old_snapshots:
                f.unlink()

        # Build command
        cmd = [
            "canonical_configuration",
            "-n", str(num_snapshots),
            "-t", str(self.temperature)
        ]
        
        # Add frequency constraint only for first step
        if step == 1:
            if self.debye_temperature is not None:
                cmd.extend(["--debye_temperature", str(self.debye_temperature)])
                self.logger.info(f"Using Debye temperature: {self.debye_temperature} K")
            elif self.max_frequency is not None:
                cmd.extend(["--maximum_frequency", str(self.max_frequency)])
                self.logger.info(f"Using maximum frequency: {self.max_frequency} THz")
            # If neither is specified, canonical_configuration will use its defaults
        
        # Run canonical_configuration with real-time output
        returncode, output = self._run_tdep_command(cmd, "canonical_configuration")
        
        if returncode != 0:
            self.logger.error(f"canonical_configuration failed:")
            self.logger.error(f"Output: {output}")
            raise RuntimeError("Failed to generate snapshots")
        
        # Check that snapshots were created
        snapshot_files = list(Path(".").glob("contcar_conf[0-9][0-9][0-9][0-9]"))
        if len(snapshot_files) != num_snapshots:
            raise RuntimeError(f"Expected {num_snapshots} snapshots, found {len(snapshot_files)}")
        
        self.logger.info(f"Successfully generated {len(snapshot_files)} snapshots")
        
    def _prepare_calculation_directories(self) -> List[Path]:
        """Prepare directories for VASP/EAM calculations."""
        snapshot_files = sorted(Path(".").glob("contcar_conf[0-9][0-9][0-9][0-9]"))
        snapshot_dirs = []
        
        for i, snapshot_file in enumerate(snapshot_files, 1):
            # Create directory
            calc_dir = Path(f"calc_{i:04d}")
            calc_dir.mkdir(exist_ok=True)
            
            # Copy snapshot as POSCAR - THIS PRESERVES VELOCITIES!
            shutil.copy2(snapshot_file, calc_dir / "POSCAR")
            
            # Copy input files
            for fname in ["INCAR", "KPOINTS", "POTCAR", "job.sh"]:
                if Path(fname).exists():
                    shutil.copy2(fname, calc_dir / fname)
            
            snapshot_dirs.append(calc_dir)
        
        self.logger.info(f"Prepared {len(snapshot_dirs)} calculation directories")
        return snapshot_dirs
        
    def _run_calculations_in_tdep_dirs(self, snapshot_dirs: List[Path], step: int):
        """Run VASP/EAM calculations directly in TDEP directories."""
        self.logger.info(f"Running {len(snapshot_dirs)} calculations using {self.execution_mode}")

        # For vasp_gpu mode, use batch submission for parallel execution
        if self.execution_mode == "vasp_gpu":
            run_dirs = [str(calc_dir.absolute()) for calc_dir in snapshot_dirs]
            self.logger.info(f"Using batch submission with {self.vasp_batch_size} parallel jobs ({self.vasp_batch_size // self.num_gpus} per GPU)")

            # submit_batch runs all calculations in parallel and returns when done
            batch_results = self.executor.submit_batch(run_dirs)

            # Check results
            completed = sum(1 for r in batch_results if r.get('status') == 'COMPLETED')
            failed = sum(1 for r in batch_results if r.get('status') == 'FAILED')

            if failed > 0:
                self.logger.warning(f"VASP batch: {completed} completed, {failed} failed")
            else:
                self.logger.info(f"VASP batch: all {completed} calculations completed successfully")

            self.logger.info("All calculations completed")
            return

        # For other modes, submit jobs individually
        job_infos = []
        failed_jobs = []
        for calc_dir in snapshot_dirs:
            abs_calc_dir = calc_dir.absolute()

            try:
                # Submit job directly
                job_info = self.executor.submit_job(str(abs_calc_dir), 'thermal')
                job_infos.append((abs_calc_dir, job_info))
            except Exception as e:
                self.logger.error(f"Failed to submit job in {calc_dir}: {e}")
                failed_jobs.append(calc_dir)
                # For EAM mode, this is likely a fatal error
                if self.execution_mode == "eam":
                    self.logger.error("EAM calculation failed. Common causes:")
                    self.logger.error("  1. Missing Python packages (numpy, ase, kim-api)")
                    self.logger.error("  2. Incorrect potential file path")
                    self.logger.error("  3. Script permission issues")
                    raise RuntimeError(f"Failed to submit EAM job: {e}")

        if failed_jobs:
            self.logger.warning(f"Failed to submit {len(failed_jobs)} jobs")

        if not job_infos:
            raise RuntimeError("No jobs were successfully submitted")

        # Wait for all calculations based on execution mode
        if hasattr(self.executor, 'wait_for_batch'):
            # Batch execution (Stampede3, Anvil)
            self.logger.info("Waiting for batch calculations to complete...")
            self.executor.wait_for_batch(
                [ji[1] for ji in job_infos],  # job info
                [str(ji[0]) for ji in job_infos]  # directories
            )
        else:
            # Individual job waiting
            self.logger.info("Waiting for individual calculations to complete...")
            for i, (calc_dir, job_info) in enumerate(job_infos, 1):
                if i % 10 == 0:
                    self.logger.info(f"  Progress: {i}/{len(job_infos)}")
                try:
                    self.executor.wait_for_completion(str(calc_dir), job_info)
                except Exception as e:
                    self.logger.error(f"Error waiting for job in {calc_dir}: {e}")
                    if self.execution_mode == "eam":
                        raise

        self.logger.info("All calculations completed")
        
    def _collect_results(self):
        """Collect results using process.py."""
        self.logger.info("Collecting results from OUTCAR files")
        
        # Find all OUTCAR files
        outcar_files = sorted(list(Path(".").glob("calc_*/OUTCAR")))
        
        # Run process.py with individual files instead of glob pattern
        cmd = [sys.executable, "process.py"] + [str(f) for f in outcar_files]
        
        self.logger.info(f"Running process.py with {len(outcar_files)} files")
        
        # Run with real-time output if verbose
        if self.show_tdep_output:
            # Pass the full command, not abbreviated
            returncode, output = self._run_tdep_command(cmd, "process.py")
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            returncode = result.returncode
            output = result.stdout if result.returncode == 0 else result.stderr
        
        if returncode != 0:
            self.logger.error(f"process.py failed:")
            self.logger.error(f"Output: {output}")
            
            # Additional debug info
            self.logger.error(f"Working directory: {os.getcwd()}")
            self.logger.error(f"OUTCAR files found: {[str(f) for f in outcar_files]}")
            
            raise RuntimeError("Failed to collect results")
        
        # Check output files
        expected_outputs = ["infile.meta", "infile.positions", "infile.forces", "infile.stat"]
        for fname in expected_outputs:
            if not Path(fname).exists():
                raise RuntimeError(f"Expected output file {fname} not created")
        
        self.logger.info("Successfully collected results")

    def _cleanup_calc_directories(self):
        """Remove calc_* directories after results have been processed."""
        if not self.cleanup_calc:
            return

        calc_dirs = sorted(list(Path(".").glob("calc_*")))
        if calc_dirs:
            self.logger.info(f"Cleaning up {len(calc_dirs)} calc_* directories...")
            for calc_dir in calc_dirs:
                try:
                    shutil.rmtree(calc_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove {calc_dir}: {e}")
            self.logger.info("Cleanup complete")

    def _extract_force_constants(self):
        """Extract force constants from the collected data."""
        self.logger.info("Extracting force constants")
        
        # Build command
        cmd = [
            "extract_forceconstants",
            "-rc2", str(self.rc2),
            "-rc3", str(self.rc3),
            "-U0",
            "--printforcemap"
        ]
        
        # Add symmetry constraint flags if specified
        if self.norotational:
            cmd.append("--norotational")
            self.logger.info("Using --norotational flag (disabling rotational invariance)")
        if self.nohuang:
            cmd.append("--nohuang")
            self.logger.info("Using --nohuang flag (disabling Huang invariances)")
        if self.nohermitian:
            cmd.append("--nohermitian")
            self.logger.info("Using --nohermitian flag (disabling hermitian constraint)")
        
        # Run with the configured parameters
        returncode, output = self._run_tdep_command(cmd, "extract_forceconstants")
        
        # If it fails with symmetry error, try different strategies (only if flags not already set)
        if returncode != 0 and ("symmetry error" in output.lower() or "bad operation" in output.lower()):
            self.logger.warning("Symmetry error detected, trying alternative approaches")
            
            # Strategy 1: Try with --norotational and --nohuang flags (if not already set)
            if not (self.norotational and self.nohuang):
                self.logger.warning("Trying with --norotational and --nohuang flags to relax symmetry constraints")
                cmd_relaxed = [
                    "extract_forceconstants",
                    "-rc2", str(self.rc2),
                    "-rc3", str(self.rc3),
                    "-U0",
                    "--printforcemap",
                    "--norotational",  # Turn off rotational invariance
                    "--nohuang"        # Turn off Huang invariances
                ]
                returncode, output = self._run_tdep_command(cmd_relaxed, "extract_forceconstants")
            
            # Strategy 2: Also add --nohermitian if still failing (if not already set)
            if returncode != 0 and not self.nohermitian:
                self.logger.warning("Trying with all symmetry relaxations: --norotational --nohuang --nohermitian")
                cmd_all_relaxed = [
                    "extract_forceconstants",
                    "-rc2", str(self.rc2),
                    "-rc3", str(self.rc3),
                    "-U0",
                    "--printforcemap",
                    "--norotational",
                    "--nohuang",
                    "--nohermitian"
                ]
                returncode, output = self._run_tdep_command(cmd_all_relaxed, "extract_forceconstants")
            
            # Strategy 3: Try without third-order force constants
            if returncode != 0 and self.rc3 > 0:
                self.logger.warning("Trying without third-order force constants (rc3=-1) and relaxed symmetry")
                cmd_no_rc3 = [
                    "extract_forceconstants",
                    "-rc2", str(self.rc2),
                    "-rc3", "-1",  # Disable third-order
                    "-U0",
                    "--printforcemap",
                    "--norotational",
                    "--nohuang"
                ]
                returncode, output = self._run_tdep_command(cmd_no_rc3, "extract_forceconstants")
            
            # Strategy 4: Try with slightly different rc2 cutoff
            if returncode != 0:
                self.logger.warning("Trying with slightly adjusted rc2 cutoff and relaxed symmetry")
                for adjustment in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                    new_rc2 = self.rc2 + adjustment
                    if new_rc2 > 0:  # Make sure it's positive
                        cmd_adjusted = [
                            "extract_forceconstants",
                            "-rc2", str(new_rc2),
                            "-rc3", str(self.rc3),
                            "-U0",
                            "--printforcemap",
                            "--norotational",
                            "--nohuang"
                        ]
                        self.logger.info(f"Trying rc2={new_rc2:.2f} with relaxed symmetry")
                        returncode, output = self._run_tdep_command(cmd_adjusted, "extract_forceconstants")
                        if returncode == 0:
                            self.logger.info(f"Success with rc2={new_rc2:.2f} and relaxed symmetry")
                            break
            
            # Strategy 5: Try without -U0 flag
            if returncode != 0:
                self.logger.warning("Trying without -U0 flag and with relaxed symmetry")
                cmd_no_u0 = [
                    "extract_forceconstants",
                    "-rc2", str(self.rc2),
                    "-rc3", str(self.rc3),
                    "--printforcemap",
                    "--norotational",
                    "--nohuang"
                ]
                returncode, output = self._run_tdep_command(cmd_no_u0, "extract_forceconstants")
            
            # Strategy 6: Try with only second order and minimal flags
            if returncode != 0:
                self.logger.warning("Trying minimal command with only second order and relaxed symmetry")
                cmd_minimal = [
                    "extract_forceconstants",
                    "-rc2", str(self.rc2),
                    "-rc3", "-1",
                    "--norotational",
                    "--nohuang"
                ]
                returncode, output = self._run_tdep_command(cmd_minimal, "extract_forceconstants")
            
            # Strategy 7: If still failing, check if it's a fundamental structure issue
            if returncode != 0 and "bad operation" in output.lower():
                self.logger.error("This appears to be a fundamental symmetry issue with the structure.")
                self.logger.error("The developers recommend checking symmetry precision.")
                self.logger.error("Possible solutions:")
                self.logger.error("1. Ensure infile.ucposcar and infile.ssposcar are perfectly consistent")
                self.logger.error("2. Use TDEP's symmetry refinement tools before running")
                self.logger.error("3. Consider using a structure without broken symmetry")
                self.logger.error("4. Check that positions use at least 12 decimal places")
                self.logger.error("5. See: https://github.com/tdep-developers/tdep-tutorials/tree/main/00_preparation/refine_symmetry")
        
        if returncode != 0:
            self.logger.error(f"extract_forceconstants failed after all attempts:")
            self.logger.error(f"Last output: {output}")
            
            # Provide more helpful error message
            if "bad operation z singlets" in output.lower():
                self.logger.error("\nThis error typically indicates:")
                self.logger.error("- The structure has broken symmetry that TDEP cannot handle")
                self.logger.error("- Atomic positions might not have sufficient precision")
                self.logger.error("- The unit cell and supercell are inconsistent")
                self.logger.error("\nTDEP requires perfect symmetry - even small deviations cause failures")
                
            raise RuntimeError("Failed to extract force constants")
        
        # Create symlink
        if Path("outfile.forceconstant").exists():
            if Path("infile.forceconstant").exists():
                Path("infile.forceconstant").unlink()
            Path("infile.forceconstant").symlink_to("outfile.forceconstant")
            self.logger.info("Force constants extracted successfully")

            # Calculate and store FC norm if convergence check is enabled
            if self.fc_convergence:
                fc_norm = self._calculate_fc_norm("outfile.forceconstant")
                if fc_norm is not None:
                    self.fc_norms.append(fc_norm)
                    self.logger.info(f"Force constant norm: {fc_norm:.6f}")

            # Calculate soft mode eigenvalues at q=2/3[111] if convergence check is enabled
            if self.eigenvalue_convergence:
                self.logger.info("Calculating soft mode eigenvalues at q=2/3[111]...")
                eigenvalues = self._calculate_softmode_eigenvalues()
                if eigenvalues is not None:
                    self.softmode_eigenvalues.append(eigenvalues)
                    self.logger.info(f"Soft mode eigenvalues at q=2/3[111]: {eigenvalues} THz")
                    self.logger.info(f"Lowest soft mode frequency: {eigenvalues[0]:.4f} THz")
                else:
                    self.logger.warning("Failed to calculate soft mode eigenvalues - check if phonopy and vibes are installed")
        else:
            raise RuntimeError("outfile.forceconstant not created")

    def _calculate_fc_norm(self, fc_file: str) -> Optional[float]:
        """
        Calculate the Frobenius norm of the force constant matrix.

        This is used to check convergence of the force constants between steps.
        """
        try:
            # Parse the TDEP force constant file
            fc_path = Path(fc_file)
            if not fc_path.exists():
                return None

            with open(fc_path, 'r') as f:
                lines = f.readlines()

            # Read number of atoms and cutoff
            first_line = lines[0].split()
            n_atoms = int(first_line[0])

            # Parse force constants and calculate norm
            fc_sum_sq = 0.0
            line_idx = 2  # Skip header lines

            for i in range(n_atoms):
                if line_idx >= len(lines):
                    break
                n_neighbors = int(lines[line_idx].split()[0])
                line_idx += 1

                for _ in range(n_neighbors):
                    line_idx += 1  # neighbor index
                    line_idx += 1  # lattice vector

                    # Read 3x3 force constant matrix
                    for _ in range(3):
                        if line_idx < len(lines):
                            values = [float(x) for x in lines[line_idx].split()]
                            fc_sum_sq += sum(v**2 for v in values)
                            line_idx += 1

            return np.sqrt(fc_sum_sq)

        except Exception as e:
            self.logger.warning(f"Error calculating FC norm: {e}")
            return None

    def _ase_to_phonopy_atoms(self, ase_atoms):
        """Convert ASE Atoms to PhonopyAtoms."""
        return PhonopyAtoms(
            symbols=ase_atoms.get_chemical_symbols(),
            cell=ase_atoms.get_cell(),
            scaled_positions=ase_atoms.get_scaled_positions()
        )

    def _calculate_softmode_eigenvalues(self) -> Optional[np.ndarray]:
        """
        Calculate phonon eigenvalues at q=2/3[111] (soft mode for BCC metals).

        This is the physically relevant wavevector for BCC phase stability,
        where the LA phonon softens before the BCC->HCP/omega transformation.

        Returns eigenvalues in THz, sorted from lowest to highest.
        """
        if not PHONOPY_AVAILABLE:
            self.logger.warning("Phonopy/vibes not available for soft mode eigenvalue check. "
                              "Install with: pip install phonopy vibes")
            return None

        try:
            # Read structures
            ucposcar = Path("infile.ucposcar")
            ssposcar = Path("infile.ssposcar")
            fc_file = Path("outfile.forceconstant")

            missing = [str(f) for f in [ucposcar, ssposcar, fc_file] if not f.exists()]
            if missing:
                self.logger.warning(f"Missing files for soft mode calculation: {missing}")
                return None

            self.logger.debug(f"Reading structures from {ucposcar} and {ssposcar}")
            uc_ase = ase_read(str(ucposcar), format='vasp')
            sc_ase = ase_read(str(ssposcar), format='vasp')
            self.logger.debug(f"Unit cell: {len(uc_ase)} atoms, Supercell: {len(sc_ase)} atoms")

            # Convert TDEP FC to phonopy format
            self.logger.debug("Converting TDEP force constants to phonopy format...")
            fc_phonopy = parse_tdep_forceconstant(
                fc_file=str(fc_file),
                primitive=uc_ase,
                supercell=sc_ase,
                two_dim=False,
                reduce_fc=False
            )
            self.logger.debug(f"Force constants shape: {fc_phonopy.shape}")

            # Create Phonopy object
            unitcell = self._ase_to_phonopy_atoms(uc_ase)
            uc_cell = uc_ase.get_cell()
            sc_cell = sc_ase.get_cell()
            supercell_matrix = np.round(np.linalg.solve(uc_cell.T, sc_cell.T).T).astype(int)
            self.logger.debug(f"Supercell matrix:\n{supercell_matrix}")

            phonon = Phonopy(unitcell, supercell_matrix)
            phonon.force_constants = fc_phonopy

            # Calculate frequencies at q=2/3[111] - the soft mode wavevector for BCC
            q_softmode = [2.0/3.0, 2.0/3.0, 2.0/3.0]
            phonon.run_qpoints([q_softmode])
            frequencies = phonon.get_qpoints_dict()['frequencies'][0]

            return np.sort(frequencies)

        except ImportError as e:
            self.logger.warning(f"Import error in soft mode calculation: {e}. "
                              "Make sure phonopy and vibes are installed.")
            return None
        except Exception as e:
            self.logger.warning(f"Error calculating soft mode eigenvalues: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _calculate_anharmonic_free_energy(self):
        """Calculate anharmonic free energy using anharmonic_free_energy command."""
        self.logger.info("Calculating anharmonic free energy")
        self.logger.info(f"Using q-point grid: {' '.join(map(str, self.qpoint_grid))}")
        
        # Build command with configurable q-point grid
        cmd = [
            "anharmonic_free_energy",
            "--qpoint_grid"
        ] + [str(q) for q in self.qpoint_grid]
        
        try:
            # Run with real-time output and longer timeout
            returncode, output = self._run_tdep_command(cmd, "anharmonic_free_energy", timeout=600)
            
            if returncode != 0:
                self.logger.warning(f"anharmonic_free_energy failed with return code {returncode}")
                
                # Check for specific errors
                if "negative eigenvalues" in output.lower():
                    self.logger.warning("Negative eigenvalues detected - system may be unstable at this temperature")
                    self.logger.warning("Consider: 1) Using a lower temperature, 2) Checking structure stability")
                    self.logger.warning("          3) Increasing number of snapshots, 4) Using tighter convergence")
                    # Still try to read the output file even with negative eigenvalues
                    
            self.logger.info("Anharmonic free energy calculation completed")
            
            # Parse and log key results
            for line in output.split('\n'):
                if any(keyword in line.lower() for keyword in ['free energy', 'temperature', 'entropy', 'heat capacity']):
                    self.logger.info(f"  Result: {line.strip()}")
            
            # Always attempt to read the output file regardless of exit code
            return self._read_anharmonic_energy()
                        
        except subprocess.TimeoutExpired:
            self.logger.warning("anharmonic_free_energy timed out after 10 minutes")
            # Still try to read any partial output
            return self._read_anharmonic_energy()
        except Exception as e:
            self.logger.warning(f"Error running anharmonic_free_energy: {e}")
            return None
    
    def _read_anharmonic_energy(self):
        """Read the anharmonic free energy from output file."""
        # Try both possible filenames
        output_files = ["outfile.anharmonic_energy", "outfile.anharmonic_free_energy"]
        outfile = None
        
        for filename in output_files:
            test_path = Path(filename)
            if test_path.exists():
                outfile = test_path
                self.logger.info(f"Found anharmonic energy output file: {filename}")
                break
        
        if not outfile:
            self.logger.warning(f"No anharmonic energy output file found. Tried: {', '.join(output_files)}")
            return None
        
        try:
            with open(outfile, 'r') as f:
                lines = f.readlines()
            
            # Parse the file more robustly
            for line in lines:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try to parse as data line
                try:
                    values = line.split()
                    if len(values) >= 2:
                        # Try to convert second value to float
                        free_energy = float(values[1])
                        self.logger.info(f"Read anharmonic free energy (2nd order cumulant): {free_energy:.6f} eV/atom")
                        return free_energy
                except (ValueError, IndexError):
                    # Not a data line, continue
                    continue
            
            self.logger.warning(f"Could not parse free energy from {outfile}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error reading {outfile}: {e}")
            return None
            
    def _check_convergence(self, step: int) -> bool:
        """Check if free energy has converged.

        Convergence criteria (all must be satisfied):
        1. Minimum number of steps completed (--min-steps)
        2. Free energy change below threshold
        3. If --fc-convergence: FC norm change below threshold
        4. If --eigenvalue-convergence: Gamma eigenvalues stable
        """
        # Check minimum steps first
        if step < self.min_steps:
            self.logger.info(f"Step {step}/{self.min_steps} - waiting for minimum steps before convergence check")
            return False

        if len(self.free_energies) < 2:
            return False

        converged = True

        # Check convergence of free energy
        current = self.free_energies[-1]
        previous = self.free_energies[-2]
        diff = abs(current - previous)

        self.logger.info(f"Free energy change: {diff:.6f} eV/atom (threshold: {self.convergence_threshold})")

        if diff >= self.convergence_threshold:
            self.logger.info("Free energy not yet converged")
            converged = False
        else:
            self.logger.info("Free energy converged!")

        # Check FC norm convergence if enabled
        if self.fc_convergence and len(self.fc_norms) >= 2:
            fc_current = self.fc_norms[-1]
            fc_previous = self.fc_norms[-2]
            fc_rel_change = abs(fc_current - fc_previous) / fc_previous if fc_previous > 0 else float('inf')

            self.logger.info(f"FC norm relative change: {fc_rel_change:.6f} (threshold: {self.fc_convergence_threshold})")

            if fc_rel_change >= self.fc_convergence_threshold:
                self.logger.info("FC norm not yet converged")
                converged = False
            else:
                self.logger.info("FC norm converged!")

        # Check soft mode eigenvalue stability at q=2/3[111] if enabled
        if self.eigenvalue_convergence and len(self.softmode_eigenvalues) >= 2:
            ev_current = self.softmode_eigenvalues[-1]
            ev_previous = self.softmode_eigenvalues[-2]

            if ev_current is not None and ev_previous is not None:
                # For soft mode at q=2/3[111], check the maximum change in any eigenvalue
                # This is more stringent than just checking the lowest frequency
                max_ev_change = np.max(np.abs(ev_current - ev_previous))
                lowest_current = ev_current[0]
                lowest_previous = ev_previous[0]
                lowest_change = abs(lowest_current - lowest_previous)

                self.logger.info(f"Soft mode at q=2/3[111]:")
                self.logger.info(f"  Lowest frequency: {lowest_current:.4f} THz (prev: {lowest_previous:.4f})")
                self.logger.info(f"  Lowest freq change: {lowest_change:.4f} THz")
                self.logger.info(f"  Max eigenvalue change: {max_ev_change:.4f} THz (threshold: {self.eigenvalue_threshold} THz)")

                if max_ev_change >= self.eigenvalue_threshold:
                    self.logger.info("Soft mode eigenvalues not yet stable - need more sampling")
                    converged = False
                else:
                    self.logger.info("Soft mode eigenvalues stable!")

        # Also compare force constants file size (informational only)
        if step >= 2:
            current_fc = Path("outfile.forceconstant")
            previous_fc = Path(f"../step_{step-1}/outfile.forceconstant")

            if current_fc.exists() and previous_fc.exists():
                current_size = current_fc.stat().st_size
                previous_size = previous_fc.stat().st_size
                size_diff = abs(current_size - previous_size)

                if size_diff < 100:  # bytes
                    self.logger.info(f"Force constant file size stabilized (diff: {size_diff} bytes)")

        return converged
        
    def _print_summary(self):
        """Print summary of the calculation."""
        self.logger.info("\n" + "="*60)
        self.logger.info("s-TDEP CALCULATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Script version: {__version__}")
        self.logger.info(f"Total steps completed: {min(self.n_steps, len(list(self.base_dir.glob('step_*'))))}")
        self.logger.info(f"Temperature: {self.temperature} K")
        self.logger.info(f"Initial snapshots per step: {self.n_snapshots}")
        self.logger.info(f"Q-point grid: {' '.join(map(str, self.qpoint_grid))}")
        self.logger.info(f"Execution mode: {self.execution_mode}")
        
        # Print free energy convergence
        if self.free_energies:
            self.logger.info("\nFree Energy Convergence:")
            self.logger.info("Step | Snapshots | Free Energy (eV/atom) | Change")
            self.logger.info("-" * 55)
            for i, (fe, ns) in enumerate(zip(self.free_energies, self.snapshot_counts)):
                if i == 0:
                    self.logger.info(f"{i+1:4d} | {ns:9d} | {fe:20.6f} | -")
                else:
                    change = fe - self.free_energies[i-1]
                    self.logger.info(f"{i+1:4d} | {ns:9d} | {fe:20.6f} | {change:+.6f}")
        
        # Find final force constants
        final_fc = None
        for step in range(self.n_steps, 0, -1):
            fc_path = self.base_dir / f"step_{step}" / "outfile.forceconstant"
            if fc_path.exists():
                final_fc = fc_path
                break
        
        if final_fc:
            self.logger.info(f"\nFinal force constants: {final_fc}")
            
            # Copy to base directory for easy access
            final_dest = self.base_dir / "final_forceconstant"
            shutil.copy2(final_fc, final_dest)
            self.logger.info(f"Copied to: {final_dest}")
        
        # Save convergence data
        self._save_convergence_data()
        
        self.logger.info("="*60)
    
    def _save_convergence_data(self):
        """Save convergence data to a file."""
        if not self.free_energies:
            return

        convergence_file = self.base_dir / "convergence_data.txt"
        with open(convergence_file, 'w') as f:
            f.write("# s-TDEP Convergence Data\n")
            f.write(f"# Script version: {__version__}\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write(f"# Q-point grid: {' '.join(map(str, self.qpoint_grid))}\n")
            f.write(f"# Convergence threshold (free energy): {self.convergence_threshold} eV/atom\n")
            f.write(f"# Min steps before convergence check: {self.min_steps}\n")

            # Write FC convergence settings if enabled
            if self.fc_convergence:
                f.write(f"# FC convergence check: enabled\n")
                f.write(f"# FC convergence threshold: {self.fc_convergence_threshold}\n")
            else:
                f.write(f"# FC convergence check: disabled\n")

            # Write eigenvalue convergence settings if enabled
            if self.eigenvalue_convergence:
                f.write(f"# Eigenvalue convergence check: enabled (q=2/3[111] soft mode)\n")
                f.write(f"# Eigenvalue threshold: {self.eigenvalue_threshold} THz\n")
            else:
                f.write(f"# Eigenvalue convergence check: disabled\n")

            f.write("#\n")

            # Determine which columns to write
            has_fc = len(self.fc_norms) > 0
            has_ev = len(self.softmode_eigenvalues) > 0

            # Write header
            header = "# Step Snapshots Free_Energy_eV_per_atom"
            if has_fc:
                header += " FC_Norm FC_Norm_Change"
            if has_ev:
                header += " Softmode_Freq_THz Max_Freq_Change_THz"
            f.write(header + "\n")

            # Write data rows
            for i, (fe, ns) in enumerate(zip(self.free_energies, self.snapshot_counts)):
                line = f"{i+1} {ns} {fe:.6f}"

                # Add FC norm data if available
                if has_fc and i < len(self.fc_norms):
                    fc_norm = self.fc_norms[i]
                    line += f" {fc_norm:.4f}"

                    # FC norm change (None for first step)
                    if i > 0 and i < len(self.fc_norms):
                        fc_change = abs(self.fc_norms[i] - self.fc_norms[i-1]) / self.fc_norms[i-1]
                        line += f" {fc_change:.6f}"
                    else:
                        line += " -"
                elif has_fc:
                    line += " - -"

                # Add soft mode eigenvalue data if available
                if has_ev and i < len(self.softmode_eigenvalues):
                    ev = self.softmode_eigenvalues[i]
                    if ev is not None:
                        # Lowest frequency at q=2/3[111]
                        lowest = ev[0]
                        line += f" {lowest:.4f}"

                        # Max eigenvalue change at q=2/3[111]
                        if i > 0 and i < len(self.softmode_eigenvalues) and self.softmode_eigenvalues[i-1] is not None:
                            ev_prev = self.softmode_eigenvalues[i-1]
                            max_change = np.max(np.abs(ev - ev_prev))
                            line += f" {max_change:.4f}"
                        else:
                            line += " -"
                    else:
                        line += " - -"
                elif has_ev:
                    line += " - -"

                f.write(line + "\n")

        self.logger.info(f"Convergence data saved to: {convergence_file}")
    
    def _create_convergence_plot(self):
        """Create a gnuplot script for visualizing convergence."""
        if not self.free_energies:
            return
        
        plot_file = self.base_dir / "plot_convergence.gnuplot"
        with open(plot_file, 'w') as f:
            f.write("#!/usr/bin/env gnuplot\n")
            f.write("set terminal png size 1200,800 font 'Arial,14'\n")
            f.write("set output 'convergence_plot.png'\n")
            f.write("set xlabel 'Step'\n")
            f.write("set ylabel 'Free Energy (eV/atom)'\n")
            f.write(f"set title 's-TDEP Free Energy Convergence at {self.temperature} K'\n")
            f.write("set grid\n")
            f.write("set key top right\n")
            
            # Calculate y-range with some padding
            min_fe = min(self.free_energies)
            max_fe = max(self.free_energies)
            range_fe = max_fe - min_fe
            if range_fe < 0.01:  # If very small range, add fixed padding
                y_min = min_fe - 0.005
                y_max = max_fe + 0.005
            else:
                y_min = min_fe - 0.1 * range_fe
                y_max = max_fe + 0.1 * range_fe
            f.write(f"set yrange [{y_min}:{y_max}]\n")
            
            # Main plot with lines and points
            f.write("plot 'convergence_data.txt' using 1:3 with linespoints lw 2 pt 7 ps 1.5 title 'Free Energy',\\\n")
            
            # Add horizontal line for final value
            if len(self.free_energies) > 1:
                f.write(f"     {self.free_energies[-1]} with lines lt 2 lw 2 dashtype 2 title 'Current Value = {self.free_energies[-1]:.6f} eV/atom',\\\n")
                
                # Add convergence threshold bands if we have enough data
                if len(self.free_energies) > 2:
                    target = self.free_energies[-1]
                    upper = target + self.convergence_threshold
                    lower = target - self.convergence_threshold
                    f.write(f"     {upper} with lines lt 3 lw 1 dashtype 3 notitle,\\\n")
                    f.write(f"     {lower} with lines lt 3 lw 1 dashtype 3 notitle\n")
                else:
                    f.write("\n")
            else:
                f.write(f"     {self.free_energies[-1]} with lines lt 2 lw 2 title 'Value = {self.free_energies[-1]:.6f} eV/atom'\n")
            
            f.write("\n# To generate the plot, run: gnuplot plot_convergence.gnuplot\n")
        
        # Try to run gnuplot if available
        try:
            subprocess.run(['gnuplot', str(plot_file)], cwd=self.base_dir, check=True)
            self.logger.info(f"Convergence plot updated: {self.base_dir}/convergence_plot.png")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Don't log every time, it's noisy


def main():
    """Main entry point for the s-TDEP script."""
    parser = argparse.ArgumentParser(
        description='Run s-TDEP calculations with VASP/EAM support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add version argument
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    # TDEP parameters
    parser.add_argument('-n', '--snapshots', type=int, default=100,
                        help='Number of snapshots per step')
    parser.add_argument('-t', '--temperature', type=float, default=1400.0,
                        help='Temperature in Kelvin')
    parser.add_argument('--max-frequency', type=float, default=None,
                        help='Maximum frequency for first step (THz). Cannot be used with --debye-temperature')
    parser.add_argument('--debye-temperature', type=float, default=None,
                        help='Debye temperature (K) for first step. Cannot be used with --max-frequency')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of TDEP iterations')
    parser.add_argument('--rc2', type=float, default=12.0,
                        help='Cutoff for second order force constants')
    parser.add_argument('--rc3', type=float, default=4.0,
                        help='Cutoff for third order force constants (use -1 to disable)')
    parser.add_argument('--qpoint-grid', nargs=3, type=int, default=[5, 5, 5],
                        help='Q-point grid for anharmonic free energy calculation')
    
    # Execution parameters
    parser.add_argument('--execution-mode',
                        choices=['mock', 'mpi', 'slurm', 'stampede3', 'anvil', 'eam', 'gulp', 'lammps', 'vasp_gpu'],
                        default='mpi',
                        help='Execution mode for calculations')
    parser.add_argument('--mpi-command', default=None,
                        help='Custom MPI command')
    parser.add_argument('--vasp-command', default='vasp_gam',
                        help='VASP executable name')
    parser.add_argument('--vasp-batch-size', type=int, default=5,
                        help='Number of parallel VASP jobs (default: 5). Jobs are distributed across GPUs.')
    parser.add_argument('--num-gpus', type=int, default=6,
                        help='Number of GPUs available for VASP calculations (default: 6)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='GPU device IDs to use, comma or space-separated (e.g., "1,2,3,4,5" or "1 2 3 4 5")')
    parser.add_argument('--parallel', action='store_true',
                        help='Each job uses ALL GPUs (spread across GPUs via MPS). Without this flag, '
                             'each job uses 1 GPU and jobs are distributed across GPUs.')
    parser.add_argument('--eam-potential-file', default=None,
                        help='Path to EAM potential file')
    parser.add_argument('--kim-model', default=None,
                        help='KIM model name for EAM')
    parser.add_argument('--gulp-library-file', default=None,
                        help='Path to GULP library file (e.g., gaop2.lib)')
    parser.add_argument('--lammps-potential-file', default=None,
                        help='Path to LAMMPS potential file (e.g., gaop2.lammps)')

    # Output management
    parser.add_argument('--output-dir', default=None,
                        help='Base directory for all outputs (default: ./outputs)')
    parser.add_argument('--run-name', default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--input-dir', default=None,
                        help='Directory containing input files (default: ./inputs)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip run if output folder already exists')
    parser.add_argument('--continue', dest='continue_run', default=None, metavar='FOLDER',
                        help='Continue an interrupted run from the given folder path (e.g., outputs/Zr_defected_small_1300K)')

    # Other options
    parser.add_argument('--base-dir', default='tdep_calculations',
                        help='Subdirectory name for TDEP calculations within output structure')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-tdep-output', action='store_true',
                        help='Suppress real-time TDEP command output')
    parser.add_argument('--convergence-threshold', type=float, default=0.001,
                        help='Free energy convergence threshold (eV/atom)')
    parser.add_argument('--convergence-factor', type=float, default=2.0,
                        help='Factor for convergence threshold to trigger snapshot increase')
    parser.add_argument('--snapshot-increment', type=int, default=50,
                        help='Number of snapshots to add if not converged')
    parser.add_argument('--max-snapshots', type=int, default=500,
                        help='Maximum number of snapshots to try')
    parser.add_argument('--omp-threads', type=int, default=1,
                        help='Number of OpenMP threads (default: 1 for stability)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without executing')
    
    # Symmetry constraint flags for force constant extraction
    parser.add_argument('--norotational', action='store_true',
                        help='Disable rotational invariance in force constant extraction')
    parser.add_argument('--nohuang', action='store_true',
                        help='Disable Huang invariances in force constant extraction')
    parser.add_argument('--nohermitian', action='store_true',
                        help='Disable hermitian constraint in force constant extraction')

    # Custom POSCAR file options (v1.5.0)
    parser.add_argument('--supercell', default=None,
                        help='Custom supercell POSCAR file (will be copied to inputs/infile.ssposcar)')
    parser.add_argument('--unitcell', default=None,
                        help='Custom unit cell POSCAR file (will be copied to inputs/infile.ucposcar)')

    # Statistical convergence options (v1.5.0)
    parser.add_argument('--min-steps', type=int, default=1,
                        help='Minimum number of TDEP steps before checking convergence')
    parser.add_argument('--fc-convergence', action='store_true',
                        help='Enable force constant norm convergence check')
    parser.add_argument('--fc-convergence-threshold', type=float, default=0.005,
                        help='Threshold for FC norm convergence (relative change). Mo achieves ~0.001-0.004 when converged.')
    parser.add_argument('--eigenvalue-convergence', action='store_true',
                        help='Enable soft mode eigenvalue stability check at q=2/3[111] (BCC soft mode)')
    parser.add_argument('--eigenvalue-threshold', type=float, default=0.03,
                        help='Threshold for soft mode eigenvalue change in THz. Based on Zr analysis: typical changes 0.01-0.03 THz when converging.')

    # Cleanup options
    parser.add_argument('--cleanup-calc', action='store_true',
                        help='Delete calc_* directories after processing to save disk space')

    args = parser.parse_args()

    # Handle continuation mode
    continue_start_step = 1
    continue_snapshots = None
    if args.continue_run:
        print(f"Continuation mode: loading state from {args.continue_run}")
        try:
            continue_start_step, continue_snapshots, original_params, already_converged = \
                STDEPCalculator.detect_continuation_state(args.continue_run)

            if already_converged:
                print(f"Run already converged at step {continue_start_step - 1}. Skipping.")
                print(f"To force continuation, delete the last entry from convergence_data.txt")
                return

            # Override args with original parameters (user can still override some via command line)
            # Only override if user didn't explicitly set them
            param_defaults = {
                'snapshots': 100, 'temperature': 1400.0, 'max_frequency': None,
                'debye_temperature': None, 'steps': 10, 'rc2': 12.0, 'rc3': 4.0,
                'qpoint_grid': [5, 5, 5], 'execution_mode': 'mpi',
                'convergence_threshold': 0.001, 'convergence_factor': 2.0,
                'snapshot_increment': 50, 'max_snapshots': 500,
                'min_steps': 1, 'fc_convergence': False,
                'fc_convergence_threshold': 0.005, 'eigenvalue_convergence': False,
                'eigenvalue_threshold': 0.03
            }

            # Use original params for values the user didn't explicitly provide
            for param, default_val in param_defaults.items():
                arg_val = getattr(args, param.replace('-', '_'), None)
                if arg_val == default_val and param in original_params:
                    setattr(args, param.replace('-', '_'), original_params[param])

            # Critical: use the original potential files and execution mode
            if 'eam_potential_file' in original_params and original_params['eam_potential_file']:
                args.eam_potential_file = original_params['eam_potential_file']
            if 'kim_model' in original_params and original_params['kim_model']:
                args.kim_model = original_params['kim_model']
            if 'execution_mode' in original_params:
                args.execution_mode = original_params['execution_mode']
            if 'supercell' in original_params:
                args.supercell = original_params['supercell']
            if 'unitcell' in original_params:
                args.unitcell = original_params['unitcell']

            # Set base_dir to the existing tdep_calculations folder
            run_path = Path(args.continue_run)
            args.base_dir = str(run_path / "tdep_calculations")
            args.output_dir = str(run_path.parent)
            args.run_name = run_path.name

            print(f"Loaded original parameters from {args.continue_run}")
            print(f"Will continue from step {continue_start_step} with {continue_snapshots} snapshots")

        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Check mutual exclusivity of max_frequency and debye_temperature
    if args.max_frequency is not None and args.debye_temperature is not None:
        parser.error("--max-frequency and --debye-temperature are mutually exclusive. Please specify only one.")

    # If neither is specified, use default max_frequency
    if args.max_frequency is None and args.debye_temperature is None:
        args.max_frequency = 6.0
        if not args.continue_run:
            print("Using default --max-frequency=6.0 THz")

    # Check if output folder already exists and skip if requested
    if args.skip_existing and args.run_name and not args.continue_run:
        # Determine the output directory path
        if args.output_dir:
            output_base = Path(args.output_dir).absolute()
        else:
            # Default: look for outputs in parent of scripts directory or current directory
            current_dir = Path.cwd()
            if current_dir.name == 'scripts':
                output_base = current_dir.parent / 'outputs'
            else:
                output_base = current_dir / 'outputs'

        run_dir = output_base / args.run_name
        if run_dir.exists():
            print(f"Output folder already exists: {run_dir}")
            print("Skipping this run (--skip-existing is enabled)")
            return

    # Initialize output directory structure (skip for continuation mode)
    if not args.dry_run and not args.continue_run:
        OutputManager.setup(
            base_dir=args.output_dir,
            run_name=args.run_name,
            input_dir=args.input_dir
        )

        # Save run metadata
        OutputManager.save_run_metadata({
            'script': 'run_stdep.py',
            'parameters': vars(args)
        })

        # Update base_dir to be within output structure
        args.base_dir = get_output_path(args.base_dir)

    # For continuation mode, set up the output path correctly
    if args.continue_run and not args.dry_run:
        # OutputManager needs to point to the existing run folder
        run_path = Path(args.continue_run)
        OutputManager._run_dir = run_path
        OutputManager._output_base = run_path.parent

    # Set up stdout redirection
    # TeeOutput class defined inline to avoid external dependency
    class TeeOutput:
        """Duplicate stdout to both terminal and a file."""
        def __init__(self, file_path, append=False):
            self.terminal = sys.stdout
            mode = 'a' if append else 'w'
            self.log = open(file_path, mode, buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def close(self):
            self.log.close()

    if args.continue_run:
        stdout_file = str(Path(args.continue_run) / 'std.out')
    else:
        stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file, append=args.continue_run is not None)
    sys.stdout = tee
    
    # Set OMP threads
    os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)
    
    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(f"Script version: {__version__}")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        return
    
    # Create and run calculator
    try:
        calculator = STDEPCalculator(
            n_snapshots=args.snapshots,
            temperature=args.temperature,
            max_frequency=args.max_frequency,
            debye_temperature=args.debye_temperature,
            n_steps=args.steps,
            rc2=args.rc2,
            rc3=args.rc3,
            qpoint_grid=args.qpoint_grid,
            execution_mode=args.execution_mode,
            vasp_command=args.vasp_command,
            mpi_command=args.mpi_command,
            eam_potential_file=args.eam_potential_file,
            kim_model=args.kim_model,
            gulp_library_file=args.gulp_library_file,
            lammps_potential_file=args.lammps_potential_file,
            base_dir=args.base_dir,
            verbose=args.verbose,
            show_tdep_output=not args.no_tdep_output,
            convergence_threshold=args.convergence_threshold,
            convergence_factor=args.convergence_factor,
            snapshot_increment=args.snapshot_increment,
            max_snapshots=args.max_snapshots,
            norotational=args.norotational,
            nohuang=args.nohuang,
            nohermitian=args.nohermitian,
            # New v1.5.0 parameters
            supercell_file=args.supercell,
            unitcell_file=args.unitcell,
            min_steps=args.min_steps,
            fc_convergence=args.fc_convergence,
            fc_convergence_threshold=args.fc_convergence_threshold,
            eigenvalue_convergence=args.eigenvalue_convergence,
            eigenvalue_threshold=args.eigenvalue_threshold,
            # New v1.7.0 parameters (continuation support)
            continue_from=args.continue_run,
            start_step=continue_start_step,
            initial_snapshots=continue_snapshots,
            # VASP GPU batch parameters
            vasp_batch_size=args.vasp_batch_size,
            num_gpus=args.num_gpus,
            gpu_ids=[int(x) for x in args.gpu_ids.replace(',', ' ').split()] if args.gpu_ids else None,
            parallel=args.parallel,
            # Cleanup options
            cleanup_calc=args.cleanup_calc
        )

        calculator.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
