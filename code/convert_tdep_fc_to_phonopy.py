#!/usr/bin/env python3
"""
TDEP to Phonopy Force Constants Converter
=========================================
This script converts TDEP force constants to Phonopy format.

Input files:
  - infile.forceconstant: TDEP force constants
  - infile.ucposcar: Primitive cell (looked for in inputs/ directory)
  - infile.ssposcar: Supercell (looked for in inputs/ directory)

Output files:
  - phonopy_force_constants.npy: Binary NumPy format
  - phonopy_force_constants.txt: Human-readable format
  - FORCE_CONSTANTS: Phonopy standard format
  - force_constants_summary.txt: Summary and statistics
"""

# --- Imports ---
import numpy as np
import os
from ase.io import read
from vibes.phonopy.utils import remap_force_constants
from pathlib import Path
from ase import Atoms
from phonopy.file_IO import write_FORCE_CONSTANTS
from output_manager import get_input_path, get_output_path

# --- Parsing function ---
def parse_tdep_forceconstant(
    fc_file="infile.forceconstant",
    primitive=None,
    supercell=None,
    fortran=True,
    two_dim=True,
    symmetrize=False,
    reduce_fc=True,
    eps=1e-13,
    tol=1e-5,
    format="vasp",
):
    """
    Parse TDEP force constants into phonopy format.
    Returns an array of shape (n_uc, n_sc, 3, 3) if reduce_fc=False, two_dim=False.
    """
    # Handle primitive cell
    if primitive is None:
        primitive = "infile.ucposcar"
    
    if isinstance(primitive, Atoms):
        uc = primitive
    else:
        # Try to find the file in inputs/ directory first
        if not os.path.isabs(primitive):
            primitive_path = get_input_path(primitive)
            if not os.path.exists(primitive_path):
                # Fallback to current directory for backward compatibility
                if os.path.exists(primitive):
                    primitive_path = primitive
                    print(f"Warning: Found {primitive} in current directory")
                    print(f"Consider moving it to inputs/ directory")
                else:
                    raise RuntimeError(f"Primitive cell file not found: {primitive}")
        else:
            primitive_path = primitive
        
        if Path(primitive_path).exists():
            uc = read(primitive_path, format=format)
        else:
            raise RuntimeError("primitive cell missing")
    
    # Handle supercell
    if supercell is None:
        supercell = "infile.ssposcar"
        
    if isinstance(supercell, Atoms):
        sc = supercell
    else:
        # Try to find the file in inputs/ directory first
        if not os.path.isabs(supercell):
            supercell_path = get_input_path(supercell)
            if not os.path.exists(supercell_path):
                # Fallback to current directory for backward compatibility
                if os.path.exists(supercell):
                    supercell_path = supercell
                    print(f"Warning: Found {supercell} in current directory")
                    print(f"Consider moving it to inputs/ directory")
                else:
                    raise RuntimeError(f"Supercell file not found: {supercell}")
        else:
            supercell_path = supercell
        
        if Path(supercell_path).exists():
            sc = read(supercell_path, format=format)
        else:
            raise RuntimeError("supercell missing")
    uc.wrap(eps=tol)
    sc.wrap(eps=tol)
    n_uc = len(uc)
    n_sc = len(sc)
    force_constants = np.zeros((n_uc, n_sc, 3, 3))
    with open(fc_file) as fo:
        n_atoms = int(next(fo).split()[0])
        cutoff = float(next(fo).split()[0])
        assert n_atoms == n_uc, f"Mismatch: n_atoms {n_atoms} vs n_uc {n_uc}"
        print(f".. # atoms = {n_atoms}, Real-space cutoff= {cutoff:.3f} Ang")
        for i1 in range(n_atoms):
            n_neighbors = int(next(fo).split()[0])
            for _ in range(n_neighbors):
                i2 = int(next(fo).split()[0]) - 1
                lp = np.array(next(fo).split(), dtype=float)
                phi = np.array([next(fo).split() for _ in range(3)], dtype=float)
                r_target = uc.positions[i2] + np.dot(lp, uc.cell[:])
                for ii, r1 in enumerate(sc.positions):
                    r_diff = np.abs(r_target - r1)
                    sc_temp = sc.get_cell(complete=True)
                    r_diff = np.linalg.solve(sc_temp.T, r_diff.T).T % 1.0
                    r_diff -= np.floor(r_diff + eps)
                    if np.sum(r_diff) < tol:
                        force_constants[i1, ii, :, :] += phi
    if not reduce_fc or two_dim:
        force_constants = remap_force_constants(force_constants, uc, sc, symmetrize=symmetrize)
    if two_dim:
        fc2d = force_constants.swapaxes(2, 1).reshape(2 * (3 * n_sc,))
        return fc2d
    else:
        return force_constants

# --- Helper functions ---
def save_force_constants(fc_array, filename, description=""):
    """
    Save force constants to both .npy format and a human-readable text file.
    """
    # Save as numpy array for easy loading
    np.save(filename + ".npy", fc_array)
    print(f"Saved force constants to {filename}.npy")
    
    # Save as human-readable text file
    with open(filename + ".txt", 'w') as f:
        f.write(f"# Force Constants\n")
        if description:
            f.write(f"# {description}\n")
        f.write(f"# Shape: {fc_array.shape}\n")
        f.write(f"# Format: fc[i_atom_prim, j_atom_supercell, alpha, beta]\n")
        f.write("#\n")
        
        n_prim, n_super, _, _ = fc_array.shape
        
        # Only write first few atom pairs to keep file size reasonable
        max_pairs = min(5, n_prim)
        f.write(f"# Showing first {max_pairs} atom pairs (out of {n_prim}x{n_super} total)\n\n")
        
        for i in range(max_pairs):
            for j in range(min(3, n_super)):
                f.write(f"\n# Atom pair: ({i}, {j})\n")
                for alpha in range(3):
                    for beta in range(3):
                        f.write(f"fc[{i},{j},{alpha},{beta}] = {fc_array[i,j,alpha,beta]:16.10f}\n")
    
    print(f"Saved human-readable force constants to {filename}.txt")

def print_force_constants_summary(fc_array, label=""):
    """
    Print a summary of the force constants including statistics.
    """
    print(f"\n{'='*60}")
    print(f"Force Constants Summary{' - ' + label if label else ''}")
    print(f"{'='*60}")
    print(f"Shape: {fc_array.shape}")
    print(f"Min value: {np.min(fc_array):.6e}")
    print(f"Max value: {np.max(fc_array):.6e}")
    print(f"Mean value: {np.mean(fc_array):.6e}")
    print(f"Std deviation: {np.std(fc_array):.6e}")
    print(f"Number of zero elements: {np.sum(np.abs(fc_array) < 1e-10)}")
    print(f"Total number of elements: {fc_array.size}")
    
    # Print a sample of the force constants
    print(f"\nSample force constant values (first atom pair):")
    print(f"fc[0,0,:,:] =")
    print(fc_array[0,0,:,:])
    print(f"{'='*60}\n")

# --- Main function ---
def main():
    print("TDEP to Phonopy Force Constants Converter")
    print("=========================================\n")
    
    # Check if required files exist
    required_files = ["infile.forceconstant", "infile.ucposcar", "infile.ssposcar"]
    for file in required_files:
        if not Path(file).exists():
            print(f"Error: Required file '{file}' not found!")
            return
    
    # Parse TDEP force constants
    print("Parsing TDEP force constants...")
    fc_phonopy = parse_tdep_forceconstant(
        fc_file="infile.forceconstant",
        primitive="infile.ucposcar",
        supercell="infile.ssposcar",
        two_dim=False,
        reduce_fc=False
    )
    
    print(f"\nForce constants shape: {fc_phonopy.shape}")
    print("(n_atoms_primitive, n_atoms_supercell, 3, 3)")
    
    # Save and display force constants information
    save_force_constants(fc_phonopy, "phonopy_force_constants", 
                        description="Force constants parsed from TDEP format")
    print_force_constants_summary(fc_phonopy, "TDEP-parsed")
    
    # Save force constants in Phonopy FORCE_CONSTANTS format
    write_FORCE_CONSTANTS(fc_phonopy, filename="FORCE_CONSTANTS")
    print("\nSaved force constants in Phonopy FORCE_CONSTANTS format")
    
    # Read primitive cell for summary
    prim_ase = read("infile.ucposcar", format="vasp")
    
    # Create a summary file
    with open("force_constants_summary.txt", 'w') as f:
        f.write("Force Constants Conversion Summary\n")
        f.write("==================================\n\n")
        f.write(f"Primitive cell atoms: {len(prim_ase)}\n")
        f.write(f"Primitive cell composition: {prim_ase.get_chemical_formula()}\n")
        f.write(f"Force constants shape: {fc_phonopy.shape}\n")
        f.write(f"Supercell atoms: {fc_phonopy.shape[1]}\n\n")
        f.write("Files created:\n")
        f.write("- phonopy_force_constants.npy: Binary format for Python/NumPy\n")
        f.write("- phonopy_force_constants.txt: Human-readable format (sample)\n")
        f.write("- FORCE_CONSTANTS: Phonopy standard format\n")
        f.write("- force_constants_summary.txt: This summary file\n\n")
        f.write("To load force constants in Python:\n")
        f.write(">>> import numpy as np\n")
        f.write(">>> fc = np.load('phonopy_force_constants.npy')\n\n")
        f.write("To use with Phonopy:\n")
        f.write(">>> from phonopy import Phonopy\n")
        f.write(">>> from phonopy.file_IO import parse_FORCE_CONSTANTS\n")
        f.write(">>> force_constants = parse_FORCE_CONSTANTS('FORCE_CONSTANTS')\n")
    
    print("\nCreated force_constants_summary.txt with usage instructions")
    print("\nConversion completed successfully!")

if __name__ == "__main__":
    main()