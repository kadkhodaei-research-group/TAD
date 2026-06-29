#!/usr/bin/env python
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
