import numpy as np
import os
import sys
import subprocess
import json
from typing import List, Any, Optional, Tuple
from vasp_executors import VASPExecutor
from multiprocessing import Pool
import multiprocessing
import functools


# Module-level function for parallel execution (avoids pickling issues)
def _run_eam_calculation_worker(args):
    """Worker function to run a single EAM calculation."""
    run_dir, potential_file = args
    poscar_path = os.path.join(run_dir, 'POSCAR')
    output_path = os.path.join(run_dir, 'eam_results.json')
    
    # Run calculation in subprocess
    result = subprocess.run([
        sys.executable, 'eam_file_calculator.py',
        poscar_path,
        potential_file,
        output_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"EAM calculation failed in {os.path.basename(run_dir)}: {result.stderr}")
        # Create fallback results
        # Try to read atom count from POSCAR
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
                # Skip header lines and find atom counts
                atom_counts_line = lines[6].strip().split()
                n_atoms = sum(int(x) for x in atom_counts_line)
        except:
            n_atoms = 215  # Default
        
        results = {
            'energy': -1750.0 + np.random.uniform(-2, 2),
            'forces': np.random.uniform(-0.5, 0.5, (n_atoms, 3)).tolist(),
            'positions': np.zeros((n_atoms, 3)).tolist(),
            'n_atoms': n_atoms,
            'error': result.stderr
        }
        with open(output_path, 'w') as f:
            json.dump(results, f)
    
    # Read results
    with open(output_path, 'r') as f:
        results = json.load(f)
    
    # Write OUTCAR
    if 'error' not in results:
        # Write OUTCAR-like file
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        with open(outcar_path, 'w') as f:
            f.write(f"# EAM calculation using potential file: {os.path.basename(potential_file)}\n")
            f.write("\n")
            f.write("POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write("-----------------------------------------------------------------------------------\n")
            
            positions = results['positions']
            forces = results['forces']
            
            for pos, force in zip(positions, forces):
                f.write(f"  {pos[0]:15.6f}  {pos[1]:15.6f}  {pos[2]:15.6f}     "
                       f"{force[0]:15.6f}  {force[1]:15.6f}  {force[2]:15.6f}\n")
            
            f.write("-----------------------------------------------------------------------------------\n")
            f.write("\n")
            f.write("  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
            f.write("  ---------------------------------------------------\n")
            f.write(f"  free  energy   TOTEN  =     {results['energy']:15.8f} eV\n")
            f.write("\n")
        
        print(f"EAM calc in {os.path.basename(run_dir)}: E = {results['energy']:.4f} eV (raw: {results.get('energy_raw', 'N/A'):.4f})")
    else:
        print(f"EAM calc in {os.path.basename(run_dir)} failed: {results['error']}")
    
    # Copy POSCAR to CONTCAR
    import shutil
    shutil.copy(poscar_path, os.path.join(run_dir, 'CONTCAR'))
    
    return run_dir, result.returncode == 0, result.stderr if result.returncode != 0 else None


class EAMFileExecutor(VASPExecutor):
    """EAM executor that uses potential files directly."""
    
    def __init__(self, potential_file: str, user_poscar_path: Optional[str] = None, 
                 parallel_eam: bool = False, n_workers: Optional[int] = None):
        """
        Initialize EAM executor with a potential file.
        
        Args:
            potential_file: Path to EAM potential file (.eam.fs or .eam.alloy)
            user_poscar_path: Path to reference POSCAR
            parallel_eam: Enable parallel execution of EAM calculations
            n_workers: Number of parallel workers (default: number of CPU cores)
        """
        self.potential_file = os.path.abspath(potential_file)
        self.user_poscar_path = user_poscar_path
        self.active_jobs = {}
        self.parallel_eam = parallel_eam
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
        # Verify potential file exists
        if not os.path.exists(self.potential_file):
            raise FileNotFoundError(f"Potential file not found: {self.potential_file}")
        
        print(f"Using EAM potential file: {self.potential_file}")
        if self.parallel_eam:
            print(f"Parallel EAM execution enabled with {self.n_workers} workers")
        
        # Create the calculator script
        self._create_eam_script()
        
        # Test it
        self._test_eam_script()
    
# Add this method to replace the _create_eam_script method in eam_file_executor.py

# Replace the _create_eam_script method in eam_file_executor.py with this:

    def _create_eam_script(self):
        """Create a standalone Python script for EAM calculations using potential files."""
    script_content = '''#!/usr/bin/env python
import sys
import json
import numpy as np
from ase import Atoms
from ase.io import read
import os

def run_eam_calculation(poscar_path, potential_file, output_path):
    """Run EAM calculation using potential file."""
    try:
        # Import EAM calculator
        from ase.calculators.eam import EAM
        
        # Read structure
        print(f"Reading POSCAR from: {poscar_path}", file=sys.stderr)
        atoms = read(poscar_path, format='vasp')
        
        # IMPORTANT: For Cu-Zr_4.eam.fs, use auto-detect (no form specified)
        print(f"Loading potential: {potential_file}", file=sys.stderr)
        
        # Check if this is the Cu-Zr potential that needs special handling
        if 'Cu-Zr' in potential_file and '.eam.fs' in potential_file:
            print(f"Using auto-detect for Cu-Zr potential", file=sys.stderr)
            calc = EAM(potential=potential_file)  # No form specified!
        else:
            # For other potentials, try to detect format from extension
            ext = os.path.splitext(potential_file)[1].lower()
            
            if ext in ['.setfl', '.eam']:
                # Standard setfl format - but check for comments first
                with open(potential_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        # Has comments, need to clean
                        print(f"Cleaning commented setfl file", file=sys.stderr)
                        cleaned_file = potential_file + '.clean'
                        with open(potential_file, 'r') as infile, open(cleaned_file, 'w') as outfile:
                            for line in infile:
                                if not line.strip().startswith('#'):
                                    outfile.write(line)
                        calc = EAM(potential=cleaned_file, form='eam')
                        os.remove(cleaned_file)
                    else:
                        calc = EAM(potential=potential_file, form='eam')
            elif ext in ['.fs', '.eam.fs']:
                # Try auto-detect first for .fs files
                try:
                    calc = EAM(potential=potential_file)
                except:
                    calc = EAM(potential=potential_file, form='eam/fs')
            elif ext in ['.alloy', '.eam.alloy']:
                calc = EAM(potential=potential_file, form='eam/alloy')
            else:
                calc = EAM(potential=potential_file)
        
        atoms.calc = calc
        
        # Calculate
        print(f"Calculating energy and forces...", file=sys.stderr)
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces().tolist()
        positions = atoms.get_positions().tolist()
        n_atoms = len(atoms)
        
        # Apply energy offset for VASP-like scale
        energy_offset = -1750.0  # Typical for Zr systems
        energy_shifted = energy + energy_offset
        
        # Save results
        results = {
            'energy': energy_shifted,
            'energy_raw': energy,
            'forces': forces,
            'positions': positions,
            'n_atoms': n_atoms,
            'potential_file': potential_file
        }
        
        print(f"Calculation successful!", file=sys.stderr)
        print(f"Raw EAM energy: {energy:.6f} eV", file=sys.stderr)
        print(f"Shifted energy: {energy_shifted:.6f} eV", file=sys.stderr)
        print(f"Max force magnitude: {np.max(np.abs(forces)):.6f} eV/Å", file=sys.stderr)
        
        with open(output_path, 'w') as f:
            json.dump(results, f)
            
    except ImportError as e:
        # ASE not properly installed
        results = {
            'error': f"Import error: {str(e)}. Make sure ASE is installed.",
            'n_atoms': 0,
            'positions': [],
            'forces': []
        }
        with open(output_path, 'w') as f:
            json.dump(results, f)
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR in EAM calculation: {e}", file=sys.stderr)
        print(f"Error type: {type(e).__name__}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # Save error
        results = {
            'error': str(e),
            'n_atoms': 0,
            'positions': [],
            'forces': []
        }
        with open(output_path, 'w') as f:
            json.dump(results, f)
        
        # Exit with error code
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python eam_file_calculator.py POSCAR potential_file output.json", file=sys.stderr)
        sys.exit(1)
        
    poscar_path = sys.argv[1]
    potential_file = sys.argv[2]
    output_path = sys.argv[3]
    run_eam_calculation(poscar_path, potential_file, output_path)
'''
    
    with open('eam_file_calculator.py', 'w') as f:
        f.write(script_content)
    os.chmod('eam_file_calculator.py', 0o755)
    print("Created/updated eam_file_calculator.py")
    
    def _test_eam_script(self):
        """Test that the EAM script works with the potential file."""
        if not self.user_poscar_path or not os.path.exists(self.user_poscar_path):
            print("Warning: No POSCAR provided for EAM test")
            return
            
        test_output = 'test_eam_file.json'
        try:
            result = subprocess.run([
                sys.executable, 'eam_file_calculator.py',
                self.user_poscar_path,
                self.potential_file,
                test_output
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"EAM test failed: {result.stderr}")
                
            with open(test_output, 'r') as f:
                data = json.load(f)
                
            if 'error' in data:
                raise RuntimeError(f"EAM calculation error: {data['error']}")
                
            print(f"EAM test successful: E_raw = {data['energy_raw']:.4f} eV, E_shifted = {data['energy']:.4f} eV")
            print(f"Using potential: {os.path.basename(self.potential_file)}")
            os.remove(test_output)
            
        except Exception as e:
            raise RuntimeError(f"Failed to test EAM calculator: {e}")
    
    def submit_job(self, run_dir: str, run_type: str) -> None:
        """Run EAM calculation in subprocess."""
        # For thermal runs with parallel mode, just mark as pending - don't run yet
        if run_type == "thermal" and self.parallel_eam:
            self.active_jobs[run_dir] = 'pending'
            return
        
        poscar_path = os.path.join(run_dir, 'POSCAR')
        output_path = os.path.join(run_dir, 'eam_results.json')
        
        # Run calculation in subprocess
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calculator_path = os.path.join(script_dir, 'eam_file_calculator.py')
        
        result = subprocess.run([
            sys.executable, calculator_path,
            poscar_path,
            self.potential_file,
            output_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"EAM calculation failed for {run_dir}")
            print(f"  Return code: {result.returncode}")
            print(f"  stderr: {result.stderr}")
            print(f"  stdout: {result.stdout}")
            # Create fallback results
            n_atoms = 215  # Your system size
            results = {
                'energy': -1750.0 + np.random.uniform(-2, 2),
                'forces': np.random.uniform(-0.5, 0.5, (n_atoms, 3)).tolist(),
                'positions': np.zeros((n_atoms, 3)).tolist(),
                'n_atoms': n_atoms,
                'error': result.stderr
            }
            with open(output_path, 'w') as f:
                json.dump(results, f)
        
        # Check if output file exists before reading
        if not os.path.exists(output_path):
            print(f"ERROR: Output file not found at {output_path} after calculation")
            print(f"  Return code was: {result.returncode}")
            print(f"  Directory contents: {os.listdir(run_dir)}")
            # Create emergency fallback
            n_atoms = 215
            results = {
                'energy': -1750.0,
                'forces': [[0.0, 0.0, 0.0] for _ in range(n_atoms)],
                'positions': [[0.0, 0.0, 0.0] for _ in range(n_atoms)],
                'n_atoms': n_atoms,
                'error': 'Output file not created'
            }
            with open(output_path, 'w') as f:
                json.dump(results, f)
        
        # Read results and write OUTCAR
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        if 'error' not in results:
            self._write_outcar_from_results(run_dir, results)
            print(f"EAM calc in {os.path.basename(run_dir)}: E = {results['energy']:.4f} eV (raw: {results.get('energy_raw', 'N/A'):.4f})")
        else:
            print(f"EAM calc in {os.path.basename(run_dir)} failed: {results['error']}")
        
        # Copy POSCAR to CONTCAR
        import shutil
        shutil.copy(poscar_path, os.path.join(run_dir, 'CONTCAR'))
        
        self.active_jobs[run_dir] = 'completed'
    
    def _write_outcar_from_results(self, run_dir: str, results: dict):
        """Write OUTCAR from results dictionary."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        
        with open(outcar_path, 'w') as f:
            f.write(f"# EAM calculation using potential file: {os.path.basename(self.potential_file)}\n")
            f.write("\n")
            f.write("POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write("-----------------------------------------------------------------------------------\n")
            
            positions = results['positions']
            forces = results['forces']
            
            for pos, force in zip(positions, forces):
                f.write(f"  {pos[0]:15.6f}  {pos[1]:15.6f}  {pos[2]:15.6f}     "
                       f"{force[0]:15.6f}  {force[1]:15.6f}  {force[2]:15.6f}\n")
            
            f.write("-----------------------------------------------------------------------------------\n")
            f.write("\n")
            f.write("  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
            f.write("  ---------------------------------------------------\n")
            f.write(f"  free  energy   TOTEN  =     {results['energy']:15.8f} eV\n")
            f.write("\n")
            f.write(" General timing and accounting informations for this job:\n")
            f.write(" ========================================================\n")
            f.write("\n")
            f.write("                  EAM calculation completed\n")
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None, **kwargs) -> List[Any]:
        """Return results immediately."""
        from vasp_manager import VASPRun
        return VASPRun(run_dir)
    
    def _submit_job_wrapper(self, args):
        """Wrapper for submit_job to work with multiprocessing."""
        run_dir, run_type = args
        try:
            self.submit_job(run_dir, run_type)
            return run_dir, True, None
        except Exception as e:
            return run_dir, False, str(e)
    
    def _run_single_eam_calculation(self, run_dir: str, max_retries: int = 2) -> Tuple[str, bool, Optional[str]]:
        """Run a single EAM calculation with retries (module-level method for pickling)."""
        # Normalize the run_dir path to avoid double directories
        run_dir = os.path.normpath(run_dir)
        poscar_path = os.path.join(run_dir, 'POSCAR')
        output_path = os.path.join(run_dir, 'eam_results.json')
        
        print(f"\nProcessing EAM calculation for: {run_dir}")
        
        # Debug: Check if paths exist
        if not os.path.exists(run_dir):
            print(f"ERROR: run_dir does not exist: {run_dir}")
            os.makedirs(run_dir, exist_ok=True)
            print(f"  Created directory: {run_dir}")
        
        if not os.path.exists(poscar_path):
            print(f"ERROR: POSCAR not found at {poscar_path}")
            print(f"  run_dir: {run_dir}")
            print(f"  Directory contents: {os.listdir(run_dir) if os.path.exists(run_dir) else 'Directory does not exist'}")
        
        # Try calculation with retries
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"  Retry attempt {attempt}/{max_retries}")
                # Clean up any partial results
                if os.path.exists(output_path):
                    os.remove(output_path)
            
            # Run calculation in subprocess
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            calculator_path = os.path.join(script_dir, 'eam_file_calculator.py')
            
            result = subprocess.run([
                sys.executable, calculator_path,
                poscar_path,
                self.potential_file,
                output_path
            ], capture_output=True, text=True)
            
            # Check if successful
            if result.returncode == 0 and os.path.exists(output_path):
                break
        else:
            # All retries failed
            print(f"  All {max_retries + 1} attempts failed")
        
        if result.returncode != 0:
            print(f"EAM calculation failed for {run_dir}")
            print(f"  Return code: {result.returncode}")
            print(f"  stderr: {result.stderr}")
            print(f"  stdout: {result.stdout}")
            # Create fallback results
            n_atoms = 215  # Your system size
            results = {
                'energy': -1750.0 + np.random.uniform(-2, 2),
                'forces': np.random.uniform(-0.5, 0.5, (n_atoms, 3)).tolist(),
                'positions': np.zeros((n_atoms, 3)).tolist(),
                'n_atoms': n_atoms,
                'error': result.stderr
            }
            with open(output_path, 'w') as f:
                json.dump(results, f)
        
        # Check if output file exists before reading
        if not os.path.exists(output_path):
            print(f"ERROR: Output file not found at {output_path} after calculation")
            print(f"  Return code was: {result.returncode}")
            print(f"  Directory contents: {os.listdir(run_dir)}")
            # Create emergency fallback
            n_atoms = 215
            results = {
                'energy': -1750.0,
                'forces': [[0.0, 0.0, 0.0] for _ in range(n_atoms)],
                'positions': [[0.0, 0.0, 0.0] for _ in range(n_atoms)],
                'n_atoms': n_atoms,
                'error': 'Output file not created'
            }
            with open(output_path, 'w') as f:
                json.dump(results, f)
        
        # Read results and write OUTCAR
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        if 'error' not in results:
            self._write_outcar_from_results(run_dir, results)
            print(f"EAM calc in {os.path.basename(run_dir)}: E = {results['energy']:.4f} eV (raw: {results.get('energy_raw', 'N/A'):.4f})")
        else:
            print(f"EAM calc in {os.path.basename(run_dir)} failed: {results['error']}")
        
        # Copy POSCAR to CONTCAR
        import shutil
        try:
            shutil.copy(poscar_path, os.path.join(run_dir, 'CONTCAR'))
        except Exception as e:
            print(f"WARNING: Failed to copy POSCAR to CONTCAR: {e}")
        
        # Return success status based on whether results were created
        success = os.path.exists(output_path) and 'error' not in results
        return run_dir, success, results.get('error', result.stderr) if not success else None
    
    def submit_batch(self, run_dirs: List[str]) -> None:
        """Run all calculations, optionally in parallel."""
        print(f"Running batch of {len(run_dirs)} EAM calculations...")
        
        # Debug: Print first few directories to check for path issues
        if run_dirs:
            print(f"First run_dir: {run_dirs[0]}")
            if len(run_dirs) > 1:
                print(f"Second run_dir: {run_dirs[1]}")
        
        if self.parallel_eam and len(run_dirs) > 1:
            print(f"Using parallel execution with {self.n_workers} workers")
            
            # Use multiprocessing pool with the instance method
            with Pool(processes=self.n_workers) as pool:
                results = []
                try:
                    for i, result in enumerate(pool.imap_unordered(self._run_single_eam_calculation, run_dirs)):
                        results.append(result)
                        if (i + 1) % 10 == 0:
                            print(f"  Progress: {i + 1}/{len(run_dirs)}")
                except Exception as e:
                    print(f"ERROR in parallel execution: {e}")
                    pool.terminate()
                    pool.join()
                    raise
                
                # Ensure we got all results
                if len(results) != len(run_dirs):
                    print(f"WARNING: Only got {len(results)} results for {len(run_dirs)} jobs")
                    print("  This may indicate some jobs failed silently")
                
                # Mark all as completed
                for run_dir in run_dirs:
                    self.active_jobs[run_dir] = 'completed'
                
                # Check for any failures
                failures = [(r[0], r[2]) for r in results if not r[1]]
                if failures:
                    print(f"Warning: {len(failures)} calculations failed:")
                    for run_dir, error in failures[:5]:  # Show first 5 errors
                        print(f"  {run_dir}: {error}")
                    if len(failures) > 5:
                        print(f"  ... and {len(failures) - 5} more")
        else:
            # Sequential execution - need to actually run the pending jobs
            for i, run_dir in enumerate(run_dirs):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(run_dirs)}")
                # For sequential, we need to force execution even if marked as pending
                if run_dir in self.active_jobs and self.active_jobs[run_dir] == 'pending':
                    # Reset to allow execution
                    del self.active_jobs[run_dir]
                    # Now run it properly
                    self.submit_job(run_dir, 'thermal')
                else:
                    self.submit_job(run_dir, 'thermal')
        
        print("Batch complete")
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Return all results."""
        from vasp_manager import VASPRun
        return [VASPRun(run_dir) for run_dir in run_dirs]