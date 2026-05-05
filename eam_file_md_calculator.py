#!/usr/bin/env python
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
