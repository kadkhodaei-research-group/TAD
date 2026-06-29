#!/usr/bin/env python
"""
eam_executor_md.py - Enhanced EAM executor that can handle single-step MD simulations
to match VASP's behavior with NSW=1 and velocities from TDEP snapshots.

This file should be placed in the same directory as run_stdep.py
"""

import numpy as np
import os
import subprocess
import json
import sys
from typing import List, Any, Optional, Tuple
from vasp_executors import VASPExecutor
from vasp_manager import VASPRun
from pathlib import Path


class EAMExecutorWithMD(VASPExecutor):
    """EAM executor that supports single-step MD for TDEP compatibility."""
    
    def __init__(self, 
                 kim_model_name: str = "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000",
                 user_poscar_path: Optional[str] = None,
                 temperature: Optional[float] = None,
                 timestep: float = 1.0,  # fs
                 perform_md_step: bool = True):
        """
        Initialize enhanced EAM executor.
        
        Args:
            kim_model_name: KIM model to use
            user_poscar_path: Reference POSCAR path
            temperature: Temperature for MD (K) - if None, use from velocities
            timestep: MD timestep in femtoseconds
            perform_md_step: Whether to perform MD step or just calculate forces
        """
        # Handle tuple case
        if isinstance(kim_model_name, tuple):
            kim_model_name = kim_model_name[0] if kim_model_name else "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000"
            
        self.kim_model_name = kim_model_name
        self.user_poscar_path = user_poscar_path
        self.temperature = temperature
        self.timestep = timestep
        self.perform_md_step = perform_md_step
        self.active_jobs = {}
        
        # Create the enhanced calculator script
        self._create_eam_md_script()
        
        # Test it
        if self.user_poscar_path and os.path.exists(self.user_poscar_path):
            self._test_eam_script()
    
    def _create_eam_md_script(self):
        """Create EAM script that can handle MD steps with velocities."""
        script_content = '''#!/usr/bin/env python
import sys
import json
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.calculators.kim import KIM
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import warnings
warnings.filterwarnings('ignore')

def parse_poscar_with_velocities(filename):
    """Parse POSCAR file including velocities if present - TDEP compatible version."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header to get number of atoms
    # Line 0: comment
    # Line 1: scaling factor
    # Lines 2-4: lattice vectors
    # Line 5: element names (might be present)
    # Line 6 or 5: atom counts
    # Line 7 or 6: "Direct" or "Cartesian"
    
    # Find the line with atom counts
    n_atoms = 0
    for i in range(5, 8):  # Check lines 5, 6, 7
        if i >= len(lines):
            break
        line = lines[i].strip()
        # Check if it's a line of numbers
        parts = line.split()
        try:
            counts = [int(x) for x in parts]
            if all(c > 0 for c in counts):  # Valid atom counts
                n_atoms = sum(counts)
                break
        except ValueError:
            continue
    
    if n_atoms == 0:
        raise ValueError("Could not determine number of atoms from POSCAR")
    
    print(f"Detected {n_atoms} atoms in POSCAR", file=sys.stderr)
    
    # Read structure using ASE - this handles the positions correctly
    from ase.io import read
    atoms = read(filename, format='vasp')
    
    # Now find velocities
    # Find the position coordinate type line
    pos_line = None
    for i, line in enumerate(lines):
        if i > 10:  # Don't look too far
            break
        if any(marker in line.strip().lower() for marker in ['cartesian', 'direct', 'selective']):
            pos_line = i
            print(f"Found position marker at line {i}: {line.strip()}", file=sys.stderr)
            break
    
    if pos_line is None:
        print("WARNING: No position marker found, cannot locate velocities", file=sys.stderr)
        return atoms, None
    
    # Velocities start after positions
    # Account for possible blank lines
    vel_start = pos_line + 1 + n_atoms
    
    # Skip blank lines
    while vel_start < len(lines) and lines[vel_start].strip() == '':
        vel_start += 1
    
    if vel_start >= len(lines):
        print("No velocities found (reached end of file)", file=sys.stderr)
        return atoms, None
    
    # Check if we have enough lines for velocities
    if vel_start + n_atoms > len(lines):
        print(f"Not enough lines for velocities: need {n_atoms} lines starting at {vel_start}, but file has {len(lines)} lines", file=sys.stderr)
        return atoms, None
    
    # Try to read velocities
    velocities = []
    for i in range(vel_start, vel_start + n_atoms):
        line = lines[i].strip()
        if not line:
            print(f"Empty line at {i} when expecting velocity", file=sys.stderr)
            return atoms, None
        
        parts = line.split()
        if len(parts) < 3:
            print(f"Line {i} has only {len(parts)} values, need 3", file=sys.stderr)
            return atoms, None
        
        try:
            vel = [float(parts[0]), float(parts[1]), float(parts[2])]
            velocities.append(vel)
        except ValueError as e:
            print(f"Error parsing velocity at line {i}: {e}", file=sys.stderr)
            return atoms, None
    
    # Convert from VASP units (Angstrom/fs) to ASE units
    velocities = np.array(velocities) * units.Ang / units.fs
    print(f"Successfully parsed {len(velocities)} velocities", file=sys.stderr)
    
    # Quick sanity check on temperature
    masses = atoms.get_masses()
    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    n_dof = 3 * len(atoms)
    temperature = 2.0 * kinetic_energy / (n_dof * units.kB)
    print(f"Calculated temperature from velocities: {temperature:.1f} K", file=sys.stderr)
    
    return atoms, velocities

def calculate_temperature_from_velocities(atoms, velocities):
    """Calculate temperature from velocities."""
    if velocities is None:
        return None
    
    masses = atoms.get_masses()
    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    n_dof = 3 * len(atoms)  # Assuming no constraints
    
    # Temperature from kinetic energy: E_kin = (n_dof/2) * k_B * T
    temperature = 2.0 * kinetic_energy / (n_dof * units.kB)
    
    return temperature

def run_eam_calculation(poscar_path, kim_model, output_path, 
                       temperature=None, timestep=1.0, perform_md=True):
    """Run EAM calculation with optional MD step."""
    try:
        # Read structure and velocities
        print(f"Reading POSCAR from: {poscar_path}", file=sys.stderr)
        atoms, velocities = parse_poscar_with_velocities(poscar_path)
        
        # Set up calculator
        calc = KIM(kim_model)
        atoms.calc = calc
        
        # Handle velocities
        if velocities is not None:
            atoms.set_velocities(velocities)
            actual_temp = calculate_temperature_from_velocities(atoms, velocities)
            print(f"Found velocities in POSCAR, T = {actual_temp:.1f} K", file=sys.stderr)
        else:
            actual_temp = None
            print(f"No velocities found in POSCAR", file=sys.stderr)
        
        # Store initial positions
        initial_positions = atoms.get_positions().copy()
        
        # Perform MD step if requested and velocities exist
        if perform_md and velocities is not None:
            print(f"Performing single MD step with dt = {timestep} fs", file=sys.stderr)
            
            # Set up MD
            dyn = VelocityVerlet(atoms, timestep * units.fs)
            
            # Run one step
            dyn.run(1)
            
            # Get final state
            final_positions = atoms.get_positions()
            final_velocities = atoms.get_velocities()
            
            # Calculate how much atoms moved
            max_displacement = np.max(np.linalg.norm(final_positions - initial_positions, axis=1))
            print(f"Max displacement: {max_displacement:.6f} Ang", file=sys.stderr)
        else:
            # No MD, just use current positions
            final_positions = initial_positions
            final_velocities = velocities
        
        # Calculate energy and forces at final positions
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        
        # If we moved atoms, recalculate forces at original positions too
        # This helps with consistency checks
        if perform_md and velocities is not None:
            atoms.set_positions(initial_positions)
            initial_forces = atoms.get_forces()
        else:
            initial_forces = forces
        
        # Prepare results
        results = {
            'energy': energy,
            'forces': forces.tolist(),
            'initial_forces': initial_forces.tolist(),
            'positions': final_positions.tolist(),
            'initial_positions': initial_positions.tolist(),
            'n_atoms': len(atoms),
            'kim_model': kim_model,
            'temperature': actual_temp,
            'md_performed': perform_md and velocities is not None,
            'timestep': timestep if perform_md else 0.0
        }
        
        # Add velocities if present
        if final_velocities is not None:
            results['velocities'] = (final_velocities / units.Ang * units.fs).tolist()
        
        print(f"Energy: {energy:.6f} eV", file=sys.stderr)
        print(f"Max |F|: {np.max(np.abs(forces)):.6f} eV/Ang", file=sys.stderr)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"ERROR in EAM calculation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        results = {
            'error': str(e),
            'n_atoms': 0,
            'positions': [],
            'forces': []
        }
        with open(output_path, 'w') as f:
            json.dump(results, f)
        
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python eam_md_calculator.py POSCAR kim_model output.json [temperature] [timestep] [perform_md]", 
              file=sys.stderr)
        sys.exit(1)
    
    poscar_path = sys.argv[1]
    kim_model = sys.argv[2]
    output_path = sys.argv[3]
    
    # Optional arguments
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else None
    timestep = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    perform_md = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else True
    
    run_eam_calculation(poscar_path, kim_model, output_path, 
                       temperature, timestep, perform_md)
'''
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eam_md_calculator.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        self._calculator_script_path = script_path
        print(f"Created eam_md_calculator.py at {script_path}")
    
    def _test_eam_script(self):
        """Test the EAM MD script."""
        print("\n=== Testing EAM MD script ===")
        test_output = 'test_eam_md.json'
        
        try:
            # Test without MD first
            result = subprocess.run([
                sys.executable, 'eam_md_calculator.py',
                self.user_poscar_path,
                self.kim_model_name,
                test_output,
                str(self.temperature) if self.temperature else "300",
                str(self.timestep),
                "false"  # No MD for test
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"stderr: {result.stderr}")
                raise RuntimeError(f"EAM test failed with return code {result.returncode}")
            
            with open(test_output, 'r') as f:
                data = json.load(f)
            
            if 'error' in data:
                raise RuntimeError(f"EAM calculation error: {data['error']}")
            
            print(f"EAM MD test successful:")
            print(f"  Energy: {data['energy']:.4f} eV")
            print(f"  Number of atoms: {data['n_atoms']}")
            print(f"  MD performed: {data.get('md_performed', False)}")
            
            os.remove(test_output)
            
        except Exception as e:
            print(f"Failed to test EAM calculator: {e}")
            if os.path.exists(test_output):
                os.remove(test_output)
            raise
    
    def submit_job(self, run_dir: str, run_type: str) -> Any:
        """Run EAM calculation with MD step if velocities present."""
        poscar_path = os.path.join(run_dir, 'POSCAR')
        output_path = os.path.join(run_dir, 'eam_results.json')
        
        # Check if this is a TDEP calculation by looking for velocities
        # or checking parent directory name
        is_tdep = 'tdep_calculations' in str(run_dir) or 'step_' in str(run_dir)
        
        print(f"\n=== Running EAM calculation ===")
        print(f"Run directory: {run_dir}")
        print(f"TDEP mode: {is_tdep}")
        print(f"Perform MD: {self.perform_md_step and is_tdep}")
        
        # Use the stored absolute path from when the script was created
        calculator_script = getattr(self, '_calculator_script_path', None)
        if calculator_script is None or not os.path.exists(calculator_script):
            # Fallback: look next to this source file
            calculator_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eam_md_calculator.py')

        if not os.path.exists(calculator_script):
            raise RuntimeError(f"Cannot find eam_md_calculator.py script. Expected at: {calculator_script}")
        
        print(f"Using calculator script: {calculator_script}")
        
        # Run calculation
        cmd = [
            sys.executable, calculator_script,  # Use absolute path
            poscar_path,
            self.kim_model_name,
            output_path,
            str(self.temperature) if self.temperature else "0",
            str(self.timestep),
            str(self.perform_md_step and is_tdep).lower()
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stderr:
            print(f"EAM output:\n{result.stderr}")
        
        # Read results
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read EAM results: {e}")
        
        if 'error' in results:
            raise RuntimeError(f"EAM calculation failed: {results['error']}")
        
        # Write OUTCAR
        self._write_outcar_from_results(run_dir, results)
        
        # Copy POSCAR to CONTCAR (with updated positions if MD was performed)
        if results.get('md_performed', False):
            self._write_contcar_with_positions(run_dir, results)
        else:
            import shutil
            shutil.copy(poscar_path, os.path.join(run_dir, 'CONTCAR'))
        
        self.active_jobs[run_dir] = 'completed'
        
        temp_str = f"T={results.get('temperature', 0):.1f}K" if results.get('temperature') else "no vel"
        md_str = "MD" if results.get('md_performed') else "static"
        print(f"EAM calc completed: E = {results['energy']:.4f} eV, {temp_str}, {md_str}")
        
        return None
    
    def _write_contcar_with_positions(self, run_dir: str, results: dict):
        """Write CONTCAR with updated positions from MD."""
        from ase.io import read, write
        
        # Read original structure
        poscar_path = os.path.join(run_dir, 'POSCAR')
        atoms = read(poscar_path, format='vasp')
        
        # Update positions
        new_positions = np.array(results['positions'])
        atoms.set_positions(new_positions)
        
        # If we have velocities, we need to write them too
        # ASE's VASP writer doesn't support velocities directly,
        # so we'll do it manually
        contcar_path = os.path.join(run_dir, 'CONTCAR')
        
        if 'velocities' in results:
            # Write manually to include velocities
            self._write_poscar_with_velocities(atoms, results['velocities'], contcar_path)
        else:
            write(contcar_path, atoms, format='vasp', direct=False)
    
    def _write_poscar_with_velocities(self, atoms, velocities, filename):
        """Write POSCAR/CONTCAR with velocities."""
        # This is a simplified version - you might want to use pymatgen
        # for more robust POSCAR writing with velocities
        from ase.io.vasp import write_vasp
        
        # First write normal POSCAR
        write_vasp(filename, atoms, direct=False, sort=False)
        
        # Then append velocities
        with open(filename, 'a') as f:
            f.write('\n')
            for vel in velocities:
                f.write(f"{vel[0]:20.16f} {vel[1]:20.16f} {vel[2]:20.16f}\n")
    
    def _write_outcar_from_results(self, run_dir: str, results: dict):
        """Write OUTCAR from results including MD information in VASP format."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        
        n_atoms = results['n_atoms']
        energy = results['energy']
        # Get temperature from results, then from self, then default to 300K
        temperature = results.get('temperature')
        if temperature is None:
            temperature = self.temperature
        if temperature is None:
            temperature = 300.0
        timestep = results.get('timestep', 1.0)
        
        with open(outcar_path, 'w') as f:
            # Write header info that process.py expects
            f.write("VASP compatible output from EAM calculation\n")
            f.write(f"# Using KIM model: {self.kim_model_name}\n")
            f.write("\n")
            
            # Write POTIM (required by process.py)
            f.write(f"   POTIM  =    {timestep:.4f}    time-step for ionic-motion\n")
            f.write("\n")
            
            # Write NIONS (required by process.py) - needs specific format
            f.write(f"   IONS per primitive cell:   NIONS =    {n_atoms}\n")
            f.write("\n")
            
            # Write TEBEG (required by process.py)
            f.write(f"   TEBEG  =    {temperature:.1f}; TEEND  =   {temperature:.1f}     temperature during run\n")
            f.write("\n")
            
            # Write ISMEAR and SIGMA (expected by process.py)
            f.write("   ISMEAR =     -1; SIGMA  =   0.20000   broadening in eV -4-tet -1-fermi 0-gaus\n")
            f.write("\n")
            
            # Write IBRION
            f.write("   IBRION =      0    ionic relax: 0-MD 1-quasi-New 2-CG\n")
            f.write("\n")
            
            # Write PSTRESS (process.py looks for this)
            f.write("   PSTRESS=    0.0 pullay stress\n")
            f.write("\n")
            
            # Write lattice vectors section (required by process.py)
            poscar_path = os.path.join(run_dir, 'POSCAR')
            if os.path.exists(poscar_path):
                from ase.io import read
                atoms = read(poscar_path, format='vasp')
                cell = atoms.get_cell()
                
                # Format 1: Lattice vectors (for reLatt)
                f.write("   Lattice vectors:\n")
                f.write("\n")
                for i in range(3):
                    axis = ['A', 'B', 'C'][i]
                    f.write(f"   {axis} = (  {cell[i,0]:11.7f},  {cell[i,1]:11.7f},  {cell[i,2]:11.7f} )\n")
                f.write("\n")
                
                # Format 2: direct lattice vectors (for reLatt2)
                f.write("      direct lattice vectors                 reciprocal lattice vectors\n")
                for i in range(3):
                    f.write(f"    {cell[i,0]:11.7f}  {cell[i,1]:11.7f}  {cell[i,2]:11.7f}"
                           f"     {0.0:11.7f}  {0.0:11.7f}  {0.0:11.7f}\n")
                f.write("\n")
            
            # Write stress section (dummy values) with format process.py expects
            f.write("  FORCE on cell =-STRESS in cart. coord.  units (eV):\n")
            f.write("  Direction    XX          YY          ZZ          XY          YZ          ZX\n")
            f.write("  --------------------------------------------------------------------------------------\n")
            f.write("  Alpha Z     0.00000     0.00000     0.00000\n")
            f.write("  Ewald       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Hartree     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  E(xc)       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Local       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  n-local     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  augment     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Kinetic     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Fock        0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  -------------------------------------------------------------------------------------\n")
            f.write("  Total       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  in kB       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  external pressure =        0.00 kB  Pullay stress =        0.00 kB\n")
            f.write("\n")
            
            # Write energy section that process.py expects (ion-electron TOTEN)
            ekin = 0.001  # Small non-zero value to avoid parsing issues
            f.write(f"%  ion-electron   TOTEN  =     {energy:15.8f} eV\n")
            f.write(f"   energy EKIN   =        {ekin:.6f}\n")  # Must be next line!
            if temperature and temperature > 0:
                f.write(f" kin. lattice  EKIN_LAT=        {ekin:.6f}  (temperature {temperature:8.2f} K)\n")
            else:
                f.write(f" kin. lattice  EKIN_LAT=        {ekin:.6f}  (temperature     0.00 K)\n")
            f.write(" nose E_kin+E_pot  =        0.000000 (temperature     0.00 K)\n")  # dummy line
            f.write(" nose E_kin+E_pot  =        0.000000 (temperature     0.00 K)\n")  # dummy line
            f.write(" ---------------------------------------------------\n")
            f.write(f" total energy   ETOTAL =      {energy:15.8f} eV\n")
            f.write("\n")
            
            # Write positions and forces section
            f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write(" -----------------------------------------------------------------------------------\n")
            
            positions = results['positions']
            forces = results['forces']
            
            for pos, force in zip(positions, forces):
                f.write(f"     {pos[0]:10.6f}  {pos[1]:10.6f}  {pos[2]:10.6f}"
                       f"     {force[0]:12.6f}  {force[1]:12.6f}  {force[2]:12.6f}\n")
            
            f.write(" -----------------------------------------------------------------------------------\n")
            f.write(f"    total drift:                           0.000000      0.000000      0.000000\n")
            f.write("\n")
            
            # Add final marker that process.py looks for
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
        print(f"Running batch of {len(run_dirs)} EAM calculations...")
        for i, run_dir in enumerate(run_dirs):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(run_dirs)}")
            self.submit_job(run_dir, 'thermal')
        print("Batch complete")
        return None
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Return all results."""
        from vasp_manager import VASPRun
        return [VASPRun(run_dir) for run_dir in run_dirs]


class EAMFileExecutorWithMD(VASPExecutor):
    """EAM executor for potential files with MD support for TDEP."""
    
    def __init__(self, 
                 potential_file: str,
                 user_poscar_path: Optional[str] = None,
                 temperature: Optional[float] = None,
                 timestep: float = 1.0,
                 perform_md_step: bool = True):
        """
        Initialize EAM file executor with MD support.
        
        Args:
            potential_file: Path to EAM potential file
            user_poscar_path: Reference POSCAR path
            temperature: Temperature for MD (K) - if None, use from velocities
            timestep: MD timestep in femtoseconds
            perform_md_step: Whether to perform MD step or just calculate forces
        """
        self.potential_file = os.path.abspath(potential_file)
        self.user_poscar_path = user_poscar_path
        self.temperature = temperature
        self.timestep = timestep
        self.perform_md_step = perform_md_step
        self.active_jobs = {}
        
        # Verify potential file exists
        if not os.path.exists(self.potential_file):
            raise FileNotFoundError(f"Potential file not found: {self.potential_file}")
        
        print(f"Using EAM potential file: {self.potential_file}")
        
        # Create the enhanced calculator script for file-based potentials
        self._create_eam_file_md_script()
        
        # Test it
        if self.user_poscar_path and os.path.exists(self.user_poscar_path):
            self._test_eam_script()
    
    def _create_eam_file_md_script(self):
        """Create EAM script for file-based potentials with MD support."""
        # Most of the script is the same, but we need to handle file-based potentials
        script_content = '''#!/usr/bin/env python
import sys
import json
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.calculators.eam import EAM
from ase.md.verlet import VelocityVerlet
import os
import warnings
warnings.filterwarnings('ignore')

def parse_poscar_with_velocities(filename):
    """Parse POSCAR file including velocities if present - TDEP compatible version."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header to get number of atoms
    # Line 0: comment
    # Line 1: scaling factor
    # Lines 2-4: lattice vectors
    # Line 5: element names (might be present)
    # Line 6 or 5: atom counts
    # Line 7 or 6: "Direct" or "Cartesian"
    
    # Find the line with atom counts
    n_atoms = 0
    for i in range(5, 8):  # Check lines 5, 6, 7
        if i >= len(lines):
            break
        line = lines[i].strip()
        # Check if it's a line of numbers
        parts = line.split()
        try:
            counts = [int(x) for x in parts]
            if all(c > 0 for c in counts):  # Valid atom counts
                n_atoms = sum(counts)
                break
        except ValueError:
            continue
    
    if n_atoms == 0:
        raise ValueError("Could not determine number of atoms from POSCAR")
    
    print(f"Detected {n_atoms} atoms in POSCAR", file=sys.stderr)
    
    # Read structure using ASE - this handles the positions correctly
    from ase.io import read
    atoms = read(filename, format='vasp')
    
    # Now find velocities
    # Find the position coordinate type line
    pos_line = None
    for i, line in enumerate(lines):
        if i > 10:  # Don't look too far
            break
        if any(marker in line.strip().lower() for marker in ['cartesian', 'direct', 'selective']):
            pos_line = i
            print(f"Found position marker at line {i}: {line.strip()}", file=sys.stderr)
            break
    
    if pos_line is None:
        print("WARNING: No position marker found, cannot locate velocities", file=sys.stderr)
        return atoms, None
    
    # Velocities start after positions
    # Account for possible blank lines
    vel_start = pos_line + 1 + n_atoms
    
    # Skip blank lines
    while vel_start < len(lines) and lines[vel_start].strip() == '':
        vel_start += 1
    
    if vel_start >= len(lines):
        print("No velocities found (reached end of file)", file=sys.stderr)
        return atoms, None
    
    # Check if we have enough lines for velocities
    if vel_start + n_atoms > len(lines):
        print(f"Not enough lines for velocities: need {n_atoms} lines starting at {vel_start}, but file has {len(lines)} lines", file=sys.stderr)
        return atoms, None
    
    # Try to read velocities
    velocities = []
    for i in range(vel_start, vel_start + n_atoms):
        line = lines[i].strip()
        if not line:
            print(f"Empty line at {i} when expecting velocity", file=sys.stderr)
            return atoms, None
        
        parts = line.split()
        if len(parts) < 3:
            print(f"Line {i} has only {len(parts)} values, need 3", file=sys.stderr)
            return atoms, None
        
        try:
            vel = [float(parts[0]), float(parts[1]), float(parts[2])]
            velocities.append(vel)
        except ValueError as e:
            print(f"Error parsing velocity at line {i}: {e}", file=sys.stderr)
            return atoms, None
    
    # Convert from VASP units (Angstrom/fs) to ASE units
    velocities = np.array(velocities) * units.Ang / units.fs
    print(f"Successfully parsed {len(velocities)} velocities", file=sys.stderr)
    
    # Quick sanity check on temperature
    masses = atoms.get_masses()
    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    n_dof = 3 * len(atoms)
    temperature = 2.0 * kinetic_energy / (n_dof * units.kB)
    print(f"Calculated temperature from velocities: {temperature:.1f} K", file=sys.stderr)
    
    return atoms, velocities

def calculate_temperature_from_velocities(atoms, velocities):
    """Calculate temperature from velocities."""
    if velocities is None:
        return None
    
    masses = atoms.get_masses()
    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    n_dof = 3 * len(atoms)
    
    temperature = 2.0 * kinetic_energy / (n_dof * units.kB)
    return temperature

def setup_eam_calculator(potential_file):
    """Set up EAM calculator with proper format detection."""
    # Check if this is the Cu-Zr potential that needs special handling
    if 'Cu-Zr' in potential_file and '.eam.fs' in potential_file:
        print(f"Using auto-detect for Cu-Zr potential", file=sys.stderr)
        calc = EAM(potential=potential_file)  # No form specified!
    else:
        # For other potentials, try to detect format from extension
        ext = os.path.splitext(potential_file)[1].lower()
        
        if ext in ['.setfl', '.eam']:
            # Standard setfl format - check for comments first
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
    
    return calc

def run_eam_calculation(poscar_path, potential_file, output_path, 
                       temperature=None, timestep=1.0, perform_md=True):
    """Run EAM calculation with optional MD step."""
    try:
        # Read structure and velocities
        print(f"Reading POSCAR from: {poscar_path}", file=sys.stderr)
        atoms, velocities = parse_poscar_with_velocities(poscar_path)
        
        # Set up calculator
        print(f"Loading potential: {potential_file}", file=sys.stderr)
        calc = setup_eam_calculator(potential_file)
        atoms.calc = calc
        
        # Handle velocities
        if velocities is not None:
            atoms.set_velocities(velocities)
            actual_temp = calculate_temperature_from_velocities(atoms, velocities)
            print(f"Found velocities in POSCAR, T = {actual_temp:.1f} K", file=sys.stderr)
        else:
            actual_temp = None
            print(f"No velocities found in POSCAR", file=sys.stderr)
        
        # Store initial positions
        initial_positions = atoms.get_positions().copy()
        
        # Perform MD step if requested and velocities exist
        if perform_md and velocities is not None:
            print(f"Performing single MD step with dt = {timestep} fs", file=sys.stderr)
            
            # Set up MD
            dyn = VelocityVerlet(atoms, timestep * units.fs)
            
            # Run one step
            dyn.run(1)
            
            # Get final state
            final_positions = atoms.get_positions()
            final_velocities = atoms.get_velocities()
            
            # Calculate displacement
            max_displacement = np.max(np.linalg.norm(final_positions - initial_positions, axis=1))
            print(f"Max displacement: {max_displacement:.6f} Ang", file=sys.stderr)
        else:
            # No MD, just use current positions
            final_positions = initial_positions
            final_velocities = velocities
        
        # Calculate energy and forces at final positions
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        
        # Apply energy offset for VASP-like scale
        energy_offset = -1750.0  # Typical for Zr systems
        energy_shifted = energy + energy_offset
        
        # Prepare results
        results = {
            'energy': energy_shifted,
            'energy_raw': energy,
            'forces': forces.tolist(),
            'positions': final_positions.tolist(),
            'initial_positions': initial_positions.tolist(),
            'n_atoms': len(atoms),
            'potential_file': potential_file,
            'temperature': actual_temp,
            'md_performed': perform_md and velocities is not None,
            'timestep': timestep if perform_md else 0.0
        }
        
        # Add velocities if present
        if final_velocities is not None:
            results['velocities'] = (final_velocities / units.Ang * units.fs).tolist()
        
        print(f"Raw EAM energy: {energy:.6f} eV", file=sys.stderr)
        print(f"Shifted energy: {energy_shifted:.6f} eV", file=sys.stderr)
        print(f"Max |F|: {np.max(np.abs(forces)):.6f} eV/Ang", file=sys.stderr)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"ERROR in EAM calculation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        results = {
            'error': str(e),
            'n_atoms': 0,
            'positions': [],
            'forces': []
        }
        with open(output_path, 'w') as f:
            json.dump(results, f)
        
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python eam_file_md_calculator.py POSCAR potential_file output.json [temperature] [timestep] [perform_md]", 
              file=sys.stderr)
        sys.exit(1)
    
    poscar_path = sys.argv[1]
    potential_file = sys.argv[2]
    output_path = sys.argv[3]
    
    # Optional arguments
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else None
    timestep = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    perform_md = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else True
    
    run_eam_calculation(poscar_path, potential_file, output_path, 
                       temperature, timestep, perform_md)
'''
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eam_file_md_calculator.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        self._calculator_script_path = script_path
        print(f"Created eam_file_md_calculator.py at {script_path}")
    
    def _test_eam_script(self):
        """Test the EAM file MD script."""
        print("\n=== Testing EAM file MD script ===")
        test_output = 'test_eam_file_md.json'
        
        try:
            # Test without MD first
            result = subprocess.run([
                sys.executable, 'eam_file_md_calculator.py',
                self.user_poscar_path,
                self.potential_file,
                test_output,
                str(self.temperature) if self.temperature else "300",
                str(self.timestep),
                "false"  # No MD for test
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"stderr: {result.stderr}")
                raise RuntimeError(f"EAM test failed with return code {result.returncode}")
            
            with open(test_output, 'r') as f:
                data = json.load(f)
            
            if 'error' in data:
                raise RuntimeError(f"EAM calculation error: {data['error']}")
            
            print(f"EAM file MD test successful:")
            print(f"  Energy (shifted): {data['energy']:.4f} eV")
            print(f"  Raw energy: {data['energy_raw']:.4f} eV")
            print(f"  Number of atoms: {data['n_atoms']}")
            
            os.remove(test_output)
            
        except Exception as e:
            print(f"Failed to test EAM calculator: {e}")
            if os.path.exists(test_output):
                os.remove(test_output)
            raise
    
    def submit_job(self, run_dir: str, run_type: str) -> Any:
        """Run EAM calculation with MD step if velocities present."""
        poscar_path = os.path.join(run_dir, 'POSCAR')
        output_path = os.path.join(run_dir, 'eam_results.json')
        
        # Check if this is a TDEP calculation
        is_tdep = 'tdep_calculations' in str(run_dir) or 'step_' in str(run_dir)
        
        print(f"\n=== Running EAM file calculation ===")
        print(f"Run directory: {run_dir}")
        print(f"Potential file: {os.path.basename(self.potential_file)}")
        print(f"TDEP mode: {is_tdep}")
        print(f"Perform MD: {self.perform_md_step and is_tdep}")
        
        # Use the stored absolute path from when the script was created
        calculator_script = getattr(self, '_calculator_script_path', None)
        if calculator_script is None or not os.path.exists(calculator_script):
            # Fallback: look next to this source file
            calculator_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eam_file_md_calculator.py')

        if not os.path.exists(calculator_script):
            raise RuntimeError("Cannot find eam_file_md_calculator.py script")
        
        print(f"Using calculator script: {calculator_script}")
        
        # Run calculation
        cmd = [
            sys.executable, calculator_script,
            poscar_path,
            self.potential_file,
            output_path,
            str(self.temperature) if self.temperature else "0",
            str(self.timestep),
            str(self.perform_md_step and is_tdep).lower()
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stderr:
            print(f"EAM output:\n{result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"EAM calculation failed with return code {result.returncode}")
        
        # Read results
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read EAM results: {e}")
        
        if 'error' in results:
            raise RuntimeError(f"EAM calculation failed: {results['error']}")
        
        # Write OUTCAR
        self._write_outcar_from_results(run_dir, results)
        
        # Copy POSCAR to CONTCAR
        import shutil
        shutil.copy(poscar_path, os.path.join(run_dir, 'CONTCAR'))
        
        self.active_jobs[run_dir] = 'completed'
        
        temp_str = f"T={results.get('temperature', 0):.1f}K" if results.get('temperature') else "no vel"
        md_str = "MD" if results.get('md_performed') else "static"
        print(f"EAM calc completed: E = {results['energy']:.4f} eV (raw: {results['energy_raw']:.4f}), {temp_str}, {md_str}")
        
        return None
    
    def _write_outcar_from_results(self, run_dir: str, results: dict):
        """Write OUTCAR from results including MD information in VASP format."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        
        n_atoms = results['n_atoms']
        energy = results['energy']
        # Get temperature from results, then from self, then default to 300K
        temperature = results.get('temperature')
        if temperature is None:
            temperature = self.temperature
        if temperature is None:
            temperature = 300.0
        timestep = results.get('timestep', 1.0)
        
        with open(outcar_path, 'w') as f:
            # Write header info that process.py expects
            f.write("VASP compatible output from EAM calculation\n")
            f.write(f"# Using potential file: {os.path.basename(self.potential_file)}\n")
            f.write("\n")
            
            # Write POTIM (required by process.py)
            f.write(f"   POTIM  =    {timestep:.4f}    time-step for ionic-motion\n")
            f.write("\n")
            
            # Write NIONS (required by process.py) - needs specific format
            f.write(f"   IONS per primitive cell:   NIONS =    {n_atoms}\n")
            f.write("\n")
            
            # Write TEBEG (required by process.py)
            f.write(f"   TEBEG  =    {temperature:.1f}; TEEND  =   {temperature:.1f}     temperature during run\n")
            f.write("\n")
            
            # Write ISMEAR and SIGMA (expected by process.py)
            f.write("   ISMEAR =     -1; SIGMA  =   0.20000   broadening in eV -4-tet -1-fermi 0-gaus\n")
            f.write("\n")
            
            # Write IBRION
            f.write("   IBRION =      0    ionic relax: 0-MD 1-quasi-New 2-CG\n")
            f.write("\n")
            
            # Write PSTRESS (process.py looks for this)
            f.write("   PSTRESS=    0.0 pullay stress\n")
            f.write("\n")
            
            # Write lattice vectors section (if available from POSCAR)
            poscar_path = os.path.join(run_dir, 'POSCAR')
            if os.path.exists(poscar_path):
                from ase.io import read
                atoms = read(poscar_path, format='vasp')
                cell = atoms.get_cell()
                
                f.write("      direct lattice vectors                 reciprocal lattice vectors\n")
                for i in range(3):
                    f.write(f"    {cell[i,0]:11.7f}  {cell[i,1]:11.7f}  {cell[i,2]:11.7f}"
                           f"     {0.0:11.7f}  {0.0:11.7f}  {0.0:11.7f}\n")
                f.write("\n")
            
            # Write stress section (dummy values)
            f.write("  FORCE on cell =-STRESS in cart. coord.  units (eV):\n")
            f.write("  Direction    XX          YY          ZZ          XY          YZ          ZX\n")
            f.write("  --------------------------------------------------------------------------------------\n")
            f.write("  Alpha Z     0.00000     0.00000     0.00000\n")
            f.write("  Ewald       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Hartree     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  E(xc)       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Local       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  n-local     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  augment     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Kinetic     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  Fock        0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  -------------------------------------------------------------------------------------\n")
            f.write("  Total       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  in kB       0.00000     0.00000     0.00000     0.00000     0.00000     0.00000\n")
            f.write("  external pressure =        0.00 kB  Pullay stress =        0.00 kB\n")
            f.write("\n")
            
            # Write energy section in VASP format (process.py looks for specific patterns)
            # Write energy section that process.py expects (ion-electron TOTEN)
            ekin = 0.001  # Small non-zero value to avoid parsing issues
            f.write(f"%  ion-electron   TOTEN  =     {energy:15.8f} eV\n")
            f.write(f"   energy EKIN   =        {ekin:.6f}\n")  # Must be next line!
            if temperature and temperature > 0:
                f.write(f" kin. lattice  EKIN_LAT=        {ekin:.6f}  (temperature {temperature:8.2f} K)\n")
            else:
                f.write(f" kin. lattice  EKIN_LAT=        {ekin:.6f}  (temperature     0.00 K)\n")
            f.write(" nose E_kin+E_pot  =        0.000000 (temperature     0.00 K)\n")  # dummy line
            f.write(" nose E_kin+E_pot  =        0.000000 (temperature     0.00 K)\n")  # dummy line
            f.write(" ---------------------------------------------------\n")
            f.write(f" total energy   ETOTAL =      {energy:15.8f} eV\n")
            f.write("\n")
            
            # Write positions and forces section
            f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write(" -----------------------------------------------------------------------------------\n")
            
            positions = results['positions']
            forces = results['forces']
            
            for pos, force in zip(positions, forces):
                f.write(f"     {pos[0]:10.6f}  {pos[1]:10.6f}  {pos[2]:10.6f}"
                       f"     {force[0]:12.6f}  {force[1]:12.6f}  {force[2]:12.6f}\n")
            
            f.write(" -----------------------------------------------------------------------------------\n")
            f.write(f"    total drift:                           0.000000      0.000000      0.000000\n")
            f.write("\n")
            
            # Add final marker that process.py looks for
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
        print(f"Running batch of {len(run_dirs)} EAM file calculations...")
        for i, run_dir in enumerate(run_dirs):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(run_dirs)}")
            self.submit_job(run_dir, 'thermal')
        print("Batch complete")
        return None
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Return all results."""
        from vasp_manager import VASPRun
        return [VASPRun(run_dir) for run_dir in run_dirs]


# Update the factory function to support file-based potentials
def create_tdep_compatible_eam_executor(kim_model_name=None, 
                                       potential_file=None,
                                       user_poscar_path=None,
                                       temperature=None,
                                       timestep=1.0,
                                       perform_md_step=True):
    """Factory function to create TDEP-compatible EAM executor."""
    
    if potential_file:
        # Use file-based EAM executor with MD support
        return EAMFileExecutorWithMD(
            potential_file=potential_file,
            user_poscar_path=user_poscar_path,
            temperature=temperature,
            timestep=timestep,
            perform_md_step=perform_md_step
        )
    else:
        # Use KIM model
        kim_model = kim_model_name or "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000"
        return EAMExecutorWithMD(
            kim_model_name=kim_model,
            user_poscar_path=user_poscar_path,
            temperature=temperature,
            timestep=timestep,
            perform_md_step=perform_md_step
        )