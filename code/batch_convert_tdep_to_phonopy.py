#!/usr/bin/env python3
"""
Batch convert TDEP force constants to Phonopy FORCE_CONSTANTS format
for all ZrO2 EAM temperature directories.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add scripts dir to path so we can import the converter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phonopy.file_IO import write_FORCE_CONSTANTS
from ase.io import read


def parse_tdep_forceconstant_no_vibes(fc_file, primitive, supercell,
                                      eps=1e-13, tol=1e-5, fmt="vasp"):
    """
    Parse TDEP force constants into a (n_uc, n_sc, 3, 3) numpy array.
    Standalone version that does NOT require the vibes package.
    Works when primitive cell == supercell (no remapping needed).
    """
    uc = read(primitive, format=fmt)
    sc = read(supercell, format=fmt)
    uc.wrap(eps=tol)
    sc.wrap(eps=tol)
    n_uc = len(uc)
    n_sc = len(sc)
    force_constants = np.zeros((n_uc, n_sc, 3, 3))

    with open(fc_file) as fo:
        n_atoms = int(next(fo).split()[0])
        cutoff = float(next(fo).split()[0])
        assert n_atoms == n_uc, f"Mismatch: n_atoms {n_atoms} vs n_uc {n_uc}"
        print(f"  # atoms = {n_atoms}, cutoff = {cutoff:.3f} Ang")

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

    return force_constants

EAM_INPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "inputs", "zro2_inputs_eam"
)

TEMPS = [2600, 2800, 2900, 3000]

def convert_one(temp_dir):
    """Convert a single temperature directory."""
    fc_file = os.path.join(temp_dir, "infile.forceconstant")

    # Find the POSCAR files (named POSCAR_uc_XXXX and POSCAR_ss_XXXX)
    uc_files = [f for f in os.listdir(temp_dir) if f.startswith("POSCAR_uc_")]
    ss_files = [f for f in os.listdir(temp_dir) if f.startswith("POSCAR_ss_")]

    if not uc_files or not ss_files:
        print(f"  ERROR: Missing POSCAR files in {temp_dir}")
        return False

    uc_path = os.path.join(temp_dir, uc_files[0])
    ss_path = os.path.join(temp_dir, ss_files[0])

    print(f"  FC file  : {fc_file}")
    print(f"  UC POSCAR: {uc_path}")
    print(f"  SS POSCAR: {ss_path}")

    # Parse TDEP -> phonopy (4D array: n_uc x n_sc x 3 x 3)
    fc = parse_tdep_forceconstant_no_vibes(
        fc_file=fc_file,
        primitive=uc_path,
        supercell=ss_path,
    )
    print(f"  FC shape : {fc.shape}")

    # Write Phonopy FORCE_CONSTANTS file
    out_path = os.path.join(temp_dir, "FORCE_CONSTANTS")
    write_FORCE_CONSTANTS(fc, filename=out_path)
    print(f"  Written  : {out_path}")
    return True


def main():
    base = os.path.realpath(EAM_INPUT_DIR)
    print(f"Base directory: {base}\n")

    for temp in TEMPS:
        temp_dir = os.path.join(base, f"{temp}_defected")
        print(f"--- {temp} K ---")
        if not os.path.isdir(temp_dir):
            print(f"  SKIP: directory not found: {temp_dir}")
            continue
        ok = convert_one(temp_dir)
        status = "OK" if ok else "FAILED"
        print(f"  Status: {status}\n")

    print("All conversions done.")


if __name__ == "__main__":
    main()
