import numpy as np
import os
import subprocess
import json
from typing import List, Any, Optional
from vasp_executors import VASPExecutor
import sys
from output_manager import get_output_path

class IsolatedEAMExecutor(VASPExecutor):
    """EAM executor that runs calculations in a subprocess - CLEAN VERSION."""
    
    def __init__(self, kim_model_name: str = "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000",
                 user_poscar_path: Optional[str] = None):
        print(f"\n=== IsolatedEAMExecutor.__init__ ===")
        print(f"kim_model_name: {kim_model_name}")
        print(f"user_poscar_path: {user_poscar_path}")
        
        # Fix: Handle tuple case
        if isinstance(kim_model_name, tuple):
            kim_model_name = kim_model_name[0] if kim_model_name else "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000"
        
        self.kim_model_name = kim_model_name
        self.user_poscar_path = user_poscar_path
        self.active_jobs = {}
        
        # Create the EAM calculation script
        self._create_eam_script()
        
        # Test that it works
        if self.user_poscar_path and os.path.exists(self.user_poscar_path):
            print(f"Testing with POSCAR: {self.user_poscar_path}")
            self._test_eam_script()
        else:
            print(f"Warning: No POSCAR provided for testing")
    
    def _create_eam_script(self):
        """Create the clean EAM calculation script."""
        # Use the clean script content from above
        script_content = '''#!/usr/bin/env python
import sys
import json
import numpy as np
from ase import Atoms
from ase.io import read
from ase.calculators.kim import KIM

def run_eam_calculation(poscar_path, kim_model, output_path):
    """Run EAM calculation and save REAL results - no modifications."""
    import os
    import tempfile
    
    try:
        # Change to output directory to contain any log files
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.chdir(output_dir)
        
        # Read structure
        print(f"Reading POSCAR from: {poscar_path}", file=sys.stderr)
        atoms = read(poscar_path, format='vasp')
        
        # Set up calculator
        calc = KIM(kim_model)
        atoms.calc = calc
        
        # Calculate REAL energy and forces
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces().tolist()
        positions = atoms.get_positions().tolist()
        n_atoms = len(atoms)
        
        # Save REAL results - NO MODIFICATIONS
        results = {
            'energy': energy,  # Raw energy from EAM
            'forces': forces,  # Raw forces from EAM
            'positions': positions,
            'n_atoms': n_atoms,
            'kim_model': kim_model
        }
        
        print(f"Raw EAM energy: {energy:.6f} eV", file=sys.stderr)
        print(f"Max force magnitude: {np.max(np.abs(forces)):.6f} eV/Å", file=sys.stderr)
        
        with open(output_path, 'w') as f:
            json.dump(results, f)
            
    except Exception as e:
        print(f"ERROR in EAM calculation: {e}", file=sys.stderr)
        # DO NOT CREATE FAKE DATA - FAIL PROPERLY
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
    poscar_path = sys.argv[1]
    kim_model = sys.argv[2]
    output_path = sys.argv[3]
    run_eam_calculation(poscar_path, kim_model, output_path)
'''
        
        # Create script in proper location
        script_path = get_output_path('eam_calculator.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        self.script_path = script_path
        print(f"Created clean eam_calculator.py at: {script_path}")
    
    def _test_eam_script(self):
        """Test that the EAM script works."""
        print("\n=== Testing EAM script ===")
        test_output = get_output_path('test_eam.json')
        
        try:
            result = subprocess.run([
                sys.executable, self.script_path,
                self.user_poscar_path,
                self.kim_model_name,
                test_output
            ], capture_output=True, text=True)
            
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"stderr: {result.stderr}")
            
            if result.returncode != 0:
                raise RuntimeError(f"EAM test failed with return code {result.returncode}")
                
            with open(test_output, 'r') as f:
                data = json.load(f)
            
            if 'error' in data:
                raise RuntimeError(f"EAM calculation error: {data['error']}")
                
            print(f"EAM test successful:")
            print(f"  Raw energy: {data['energy']:.4f} eV")
            print(f"  Number of atoms: {data['n_atoms']}")
            os.remove(test_output)
            
        except Exception as e:
            print(f"Failed to test EAM calculator: {e}")
            if os.path.exists(test_output):
                os.remove(test_output)
            raise
    
    def submit_job(self, run_dir: str, run_type: str) -> Any:
        """Run EAM calculation in subprocess."""
        poscar_path = os.path.join(run_dir, 'POSCAR')
        output_path = os.path.join(run_dir, 'eam_results.json')
        
        print(f"\n=== Running EAM calculation ===")
        print(f"Run directory: {run_dir}")
        
        # Run calculation in subprocess
        result = subprocess.run([
            sys.executable, self.script_path,
            poscar_path,
            self.kim_model_name,
            output_path
        ], capture_output=True, text=True)
        
        if result.stderr:
            print(f"EAM output:\n{result.stderr}")
        
        # Read results and check for errors
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read EAM results: {e}")
        
        # Check for errors - FAIL PROPERLY, no fallback
        if 'error' in results:
            error_msg = results['error']
            raise RuntimeError(f"EAM calculation failed: {error_msg}")
        
        # Write OUTCAR with REAL results
        self._write_outcar_from_results(run_dir, results)
        
        # Copy POSCAR to CONTCAR
        import shutil
        shutil.copy(poscar_path, os.path.join(run_dir, 'CONTCAR'))
        
        self.active_jobs[run_dir] = 'completed'
        
        print(f"EAM calc in {os.path.basename(run_dir)}: E = {results['energy']:.4f} eV (raw, unmodified)")
        
        return None
    
    def _write_outcar_from_results(self, run_dir: str, results: dict):
        """Write OUTCAR from REAL results."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        
        with open(outcar_path, 'w') as f:
            f.write(f"# EAM calculation using KIM model: {self.kim_model_name}\n")
            f.write("# RAW, UNMODIFIED RESULTS\n")
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
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None, **kwargs) -> Any:
        """Return results immediately."""
        from vasp_manager import VASPRun
        return VASPRun(run_dir)
    
    def submit_batch(self, run_dirs: List[str]) -> Any:
        """Run all calculations."""
        for run_dir in run_dirs:
            self.submit_job(run_dir, 'thermal')
        return None
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Return all results."""
        from vasp_manager import VASPRun
        return [VASPRun(run_dir) for run_dir in run_dirs]