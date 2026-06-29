#!/usr/bin/env python
"""
s-TDEP calculation script that integrates with existing VASP/EAM infrastructure.
Performs iterative TDEP calculations with proper file and folder management.

Version 1.4.0 - Use --norotational --nohuang --nohermitian for broken symmetry
"""

__version__ = "1.4.0"

import os
import sys

# Set OMP_NUM_THREADS to 1 to avoid symmetry detection issues in extract_forceconstants
os.environ['OMP_NUM_THREADS'] = '1'

import shutil
import subprocess
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your existing infrastructure
from vasp_manager import VASPManager
from vasp_executors import get_hpc_executor, detect_hpc_system
from output_manager import OutputManager, get_output_path, get_input_path


class STDEPCalculator:
    """Main class for s-TDEP calculations with VASP/EAM support."""
    
    def __init__(
        self,
        n_snapshots: int = 100,
        temperature: float = 1400.0,
        max_frequency: Optional[float] = 6.0,
        n_steps: int = 10,
        rc2: float = 12.0,
        rc3: float = 4.0,
        qpoint_grid: List[int] = None,
        execution_mode: str = "mpi",
        vasp_command: str = "vasp_gam",
        mpi_command: Optional[str] = None,
        eam_potential_file: Optional[str] = None,
        kim_model: Optional[str] = None,
        base_dir: str = "tdep_calculations",
        verbose: bool = True,
        show_tdep_output: bool = True,
        convergence_threshold: float = 0.001,
        snapshot_increment: int = 50,
        max_snapshots: int = 500
    ):
        """
        Initialize s-TDEP calculator.
        
        Args:
            n_snapshots: Number of snapshots per step
            temperature: Temperature in Kelvin
            max_frequency: Maximum frequency for first step (THz)
            n_steps: Number of TDEP iterations
            rc2: Cutoff for second order force constants
            rc3: Cutoff for third order force constants
            qpoint_grid: Q-point grid for anharmonic free energy [nx, ny, nz]
            execution_mode: "mpi", "slurm", "stampede3", "anvil", "eam", "mock"
            vasp_command: VASP executable name
            mpi_command: Custom MPI command
            eam_potential_file: Path to EAM potential file
            kim_model: KIM model name for EAM
            base_dir: Base directory for calculations
            verbose: Enable verbose output
            show_tdep_output: Show real-time output from TDEP commands
            convergence_threshold: Threshold for free energy convergence (eV/atom)
            snapshot_increment: Number of snapshots to add if not converged
            max_snapshots: Maximum number of snapshots to try
        """
        self.n_snapshots = n_snapshots
        self.temperature = temperature
        self.max_frequency = max_frequency
        self.n_steps = n_steps
        self.rc2 = rc2
        self.rc3 = rc3
        self.qpoint_grid = qpoint_grid if qpoint_grid else [5, 5, 5]
        self.execution_mode = execution_mode
        self.vasp_command = vasp_command
        self.mpi_command = mpi_command
        self.eam_potential_file = eam_potential_file
        self.kim_model = kim_model
        self.base_dir = Path(base_dir).absolute()
        self.verbose = verbose
        self.show_tdep_output = show_tdep_output
        self.convergence_threshold = convergence_threshold
        self.snapshot_increment = snapshot_increment
        self.max_snapshots = max_snapshots
        
        # Track free energies for convergence
        self.free_energies = []
        self.snapshot_counts = []
        
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
        
        # Initialize executor directly (not VASPManager)
        self._init_executor()
        
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
        else:
            # For VASP modes, get the appropriate executor
            self.executor = get_hpc_executor(
                str(self.base_dir),
                self.execution_mode,
                mpi_command=self.mpi_command,
                vasp_command=self.vasp_command,
                user_poscar_path=self.user_poscar_path
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
        
    def run(self):
        """Run the complete s-TDEP calculation workflow."""
        self.logger.info(f"Starting s-TDEP calculation with {self.n_steps} steps")
        self.logger.info(f"Temperature: {self.temperature} K, Initial snapshots per step: {self.n_snapshots}")
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
            
            # Current number of snapshots (can increase if not converged)
            current_snapshots = self.n_snapshots
            
            # Run iterative TDEP steps
            for step in range(1, self.n_steps + 1):
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
                        self.logger.info("Free energy converged! Stopping iterations.")
                        break
                    elif not converged and free_energy is None and current_snapshots < self.max_snapshots:
                        # If we got negative eigenvalues, increase snapshots
                        current_snapshots += self.snapshot_increment
                        self.logger.info(f"Increasing snapshots to {current_snapshots} for better statistics")
                        
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
        
        # For EAM, POTCAR is not strictly required
        if self.execution_mode == "eam":
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
        
        # Build command
        cmd = [
            "canonical_configuration",
            "-n", str(num_snapshots),
            "-t", str(self.temperature)
        ]
        
        # Add maximum frequency only for first step
        if step == 1 and self.max_frequency is not None:
            cmd.extend(["--maximum_frequency", str(self.max_frequency)])
            self.logger.info(f"Using maximum frequency: {self.max_frequency} THz")
        
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
        
        # Submit all jobs directly in the TDEP directories
        job_infos = []
        for calc_dir in snapshot_dirs:
            abs_calc_dir = calc_dir.absolute()
            
            # Submit job directly
            job_info = self.executor.submit_job(str(abs_calc_dir), 'thermal')
            job_infos.append((abs_calc_dir, job_info))
        
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
                self.executor.wait_for_completion(str(calc_dir), job_info)
        
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
        
        # Try with standard parameters first
        returncode, output = self._run_tdep_command(cmd, "extract_forceconstants")
        
        # If it fails with symmetry error, try different strategies
        if returncode != 0 and ("symmetry error" in output.lower() or "bad operation" in output.lower()):
            self.logger.warning("Symmetry error detected, trying alternative approaches")
            
            # Strategy 1: Try with --norotational and --nohuang flags (for broken symmetry)
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
            
            # Strategy 2: Also add --nohermitian if still failing
            if returncode != 0:
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
        else:
            raise RuntimeError("outfile.forceconstant not created")
    
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
        """Check if free energy has converged."""
        if len(self.free_energies) < 2:
            return False
        
        # Check convergence of free energy
        if len(self.free_energies) >= 2:
            current = self.free_energies[-1]
            previous = self.free_energies[-2]
            diff = abs(current - previous)
            
            self.logger.info(f"Free energy change: {diff:.6f} eV/atom (threshold: {self.convergence_threshold})")
            
            if diff < self.convergence_threshold:
                self.logger.info("Free energy converged!")
                return True
        
        # Also compare force constants if desired
        if step >= 2:
            current_fc = Path("outfile.forceconstant")
            previous_fc = Path(f"../step_{step-1}/outfile.forceconstant")
            
            if current_fc.exists() and previous_fc.exists():
                current_size = current_fc.stat().st_size
                previous_size = previous_fc.stat().st_size
                size_diff = abs(current_size - previous_size)
                
                if size_diff < 100:  # bytes
                    self.logger.info(f"Force constant file size stabilized (diff: {size_diff} bytes)")
            
        return False
        
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
            f.write("# s-TDEP Free Energy Convergence\n")
            f.write(f"# Script version: {__version__}\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write(f"# Q-point grid: {' '.join(map(str, self.qpoint_grid))}\n")
            f.write("# Step Snapshots Free_Energy_eV_per_atom\n")
            for i, (fe, ns) in enumerate(zip(self.free_energies, self.snapshot_counts)):
                f.write(f"{i+1} {ns} {fe:.6f}\n")
        
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
    parser.add_argument('--max-frequency', type=float, default=6.0,
                        help='Maximum frequency for first step (THz)')
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
                        choices=['mock', 'mpi', 'slurm', 'stampede3', 'anvil', 'eam', 'vasp'],
                        default='mpi',
                        help='Execution mode for calculations')
    parser.add_argument('--mpi-command', default=None,
                        help='Custom MPI command')
    parser.add_argument('--vasp-command', default='vasp_gam',
                        help='VASP executable name')
    parser.add_argument('--eam-potential-file', default=None,
                        help='Path to EAM potential file')
    parser.add_argument('--kim-model', default=None,
                        help='KIM model name for EAM')
    
    # Output management
    parser.add_argument('--output-dir', default=None,
                        help='Base directory for all outputs (default: ./outputs)')
    parser.add_argument('--run-name', default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--input-dir', default=None,
                        help='Directory containing input files (default: ./inputs)')
    
    # Other options
    parser.add_argument('--base-dir', default='tdep_calculations',
                        help='Subdirectory name for TDEP calculations within output structure')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-tdep-output', action='store_true',
                        help='Suppress real-time TDEP command output')
    parser.add_argument('--convergence-threshold', type=float, default=0.001,
                        help='Free energy convergence threshold (eV/atom)')
    parser.add_argument('--snapshot-increment', type=int, default=50,
                        help='Number of snapshots to add if not converged')
    parser.add_argument('--max-snapshots', type=int, default=500,
                        help='Maximum number of snapshots to try')
    parser.add_argument('--omp-threads', type=int, default=1,
                        help='Number of OpenMP threads (default: 1 for stability)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Initialize output directory structure
    if not args.dry_run:
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
    
    # Set up stdout redirection
    from tee_output import TeeOutput
    stdout_file = get_output_path('std.out')
    tee = TeeOutput(stdout_file)
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
            n_steps=args.steps,
            rc2=args.rc2,
            rc3=args.rc3,
            qpoint_grid=args.qpoint_grid,
            execution_mode=args.execution_mode,
            vasp_command=args.vasp_command,
            mpi_command=args.mpi_command,
            eam_potential_file=args.eam_potential_file,
            kim_model=args.kim_model,
            base_dir=args.base_dir,
            verbose=args.verbose,
            show_tdep_output=not args.no_tdep_output,
            convergence_threshold=args.convergence_threshold,
            snapshot_increment=args.snapshot_increment,
            max_snapshots=args.max_snapshots
        )
        
        calculator.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()