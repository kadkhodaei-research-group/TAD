"""GAOP2 potential executor using LAMMPS for ZrO2 systems.

Uses the GAOP2 interatomic potential (charged ionic with Ewald summation)
via an external LAMMPS binary. Handles coordinate transformations between
ASE/Cartesian and LAMMPS triclinic frames.
"""

import os
import numpy as np
import subprocess
import tempfile
import shutil
from ase import Atoms
from ase.io import read as ase_read
from ase.calculators.lammpsrun import Prism
from pymatgen.io.vasp import Poscar


from vasp_manager import VASPRun


class GAOPExecutor:
    """Executor for GAOP2 potential calculations via LAMMPS.

    This executor writes LAMMPS input files, runs a serial LAMMPS binary,
    and parses energies/forces with proper coordinate back-transformation.
    """

    # Charge map for GAOP2 potential
    CHARGE_MAP = {'Zr': 2.044, 'Y': 1.533, 'O': -1.022}
    # LAMMPS atom type map
    TYPE_MAP = {'Zr': 1, 'Y': 2, 'O': 3}
    # Masses
    MASS_MAP = {1: 91.224, 2: 88.906, 3: 15.999}

    def __init__(self, potential_file, lmp_executable=None, timeout=120, **kwargs):
        """
        Args:
            potential_file: Path to gaop2.lammps potential definition file
            lmp_executable: Path to LAMMPS binary. Auto-detected if None.
            timeout: LAMMPS execution timeout in seconds
        """
        self.potential_file = os.path.abspath(potential_file)
        self.timeout = timeout

        if lmp_executable is None:
            # Try to find LAMMPS
            for name in ['lmp', 'lmp_serial', 'lmp_mpi']:
                path = shutil.which(name)
                if path:
                    lmp_executable = path
                    break
            if lmp_executable is None:
                raise RuntimeError("LAMMPS executable not found. Provide lmp_executable path.")

        self.lmp_executable = lmp_executable

        if not os.path.exists(self.potential_file):
            raise FileNotFoundError(f"GAOP potential file not found: {self.potential_file}")

        print(f"GAOP executor initialized:")
        print(f"  Potential: {self.potential_file}")
        print(f"  LAMMPS: {self.lmp_executable}")

    def submit_job(self, run_dir, run_type='main'):
        """Run a GAOP2/LAMMPS calculation for a single structure.

        Args:
            run_dir: Directory containing POSCAR file
            run_type: 'main' or 'thermal' (unused, for interface compatibility)

        Returns:
            VASPRun object with results directory
        """
        poscar_path = os.path.join(run_dir, 'POSCAR')

        # Read structure via pymatgen -> ASE
        poscar = Poscar.from_file(poscar_path)
        structure = poscar.structure
        atoms = Atoms(
            symbols=[site.specie.symbol for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )

        # Compute energy and forces
        energy, forces = self._compute(atoms, run_dir)

        # Write OUTCAR-like output for compatibility with existing parsers
        self._write_mock_outcar(run_dir, energy, forces, atoms)

        return VASPRun(run_dir)

    def _compute(self, atoms, work_dir):
        """Run LAMMPS GAOP2 calculation and return energy, forces."""
        atoms = atoms.copy()
        atoms.wrap(eps=1e-11)

        # Assign charges
        symbols = atoms.get_chemical_symbols()
        charges = [self.CHARGE_MAP.get(sym, 0.0) for sym in symbols]

        # Enforce charge neutrality by uniform redistribution
        # (handles O vacancy defects where the missing O leaves excess positive charge)
        total_charge = sum(charges)
        if abs(total_charge) > 1e-6:
            correction = -total_charge / len(atoms)
            charges = [c + correction for c in charges]

        # Atom types
        atom_types = [self.TYPE_MAP.get(sym, 1) for sym in symbols]

        # Prism for coordinate transformation
        prism = Prism(atoms.get_cell())
        xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()

        positions = atoms.get_positions()
        n_atoms = len(atoms)

        # Write lammps.data
        data_file = os.path.join(work_dir, 'lammps.data')
        with open(data_file, 'w') as f:
            f.write('GAOP2 calculation\n\n')
            f.write(f'{n_atoms} atoms\n')
            f.write('3 atom types\n\n')
            f.write(f'{0.0:.10f} {xhi:.10f} xlo xhi\n')
            f.write(f'{0.0:.10f} {yhi:.10f} ylo yhi\n')
            f.write(f'{0.0:.10f} {zhi:.10f} zlo zhi\n')
            if abs(xy) > 1e-10 or abs(xz) > 1e-10 or abs(yz) > 1e-10:
                f.write(f'{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n')
            f.write('\nMasses\n\n')
            f.write('1 91.224   # Zr\n')
            f.write('2 88.906   # Y\n')
            f.write('3 15.999   # O\n')
            f.write('\nAtoms  # charge\n\n')
            for i, (pos, at, ch) in enumerate(zip(positions, atom_types, charges), 1):
                lp = prism.vector_to_lammps(pos)
                f.write(f'{i} {at} {ch:.6f} {lp[0]:.10f} {lp[1]:.10f} {lp[2]:.10f}\n')

        # Write lammps.in
        in_file = os.path.join(work_dir, 'lammps.in')
        with open(in_file, 'w') as f:
            f.write('units metal\n')
            f.write('atom_style charge\n')
            f.write('boundary p p p\n\n')
            f.write('read_data lammps.data\n\n')
            # Insert potential (replace ewald with pppm for speed)
            with open(self.potential_file, 'r') as pf:
                content = pf.read().replace('kspace_style  ewald', 'kspace_style pppm')
                content = content.replace('kspace_style ewald', 'kspace_style pppm')
                f.write(content)
            f.write('\nneigh_modify every 1 delay 0 check yes\n')
            f.write('thermo 0\n')
            f.write('compute pe all pe\n\n')
            f.write('run 0\n\n')
            f.write('dump fd all custom 1 forces.dump id type q fx fy fz\n')
            f.write('dump_modify fd format line "%d %d %.6f %.15g %.15g %.15g"\n')
            f.write('run 0\n\n')
            f.write('print "Energy: $(c_pe)" file energy.txt\n')

        # Run LAMMPS
        result = subprocess.run(
            [self.lmp_executable, '-in', 'lammps.in', '-log', 'lammps.log'],
            capture_output=True, text=True, timeout=self.timeout,
            cwd=work_dir, env=os.environ
        )
        if result.returncode != 0:
            raise RuntimeError(f"LAMMPS failed in {work_dir}:\n{result.stderr[-500:]}")

        # Parse energy
        energy_file = os.path.join(work_dir, 'energy.txt')
        with open(energy_file) as f:
            energy = float(f.read().split()[-1])

        # Parse forces with coordinate back-transformation
        forces = np.zeros((n_atoms, 3))
        dump_file = os.path.join(work_dir, 'forces.dump')
        with open(dump_file) as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('ITEM: ATOMS'):
                data_start = i + 1
                break

        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) >= 6:
                aid = int(parts[0])
                f_lmp = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                forces[aid - 1] = prism.vector_to_ase(f_lmp)  # Back-transform to Cartesian

        return energy, forces

    def _write_mock_outcar(self, run_dir, energy, forces, atoms):
        """Write a simplified OUTCAR-like file for compatibility with existing parsers."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        with open(outcar_path, 'w') as f:
            f.write("GAOP2/LAMMPS calculation results\n")
            f.write(f"  free  energy   TOTEN  =    {energy:.8f} eV\n")
            f.write(f"  energy  without entropy=    {energy:.8f}  "
                    f"energy(sigma->0) =    {energy:.8f}\n\n")
            f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write(" -----------------------------------------------------------------------------------\n")
            positions = atoms.get_positions()
            for i in range(len(atoms)):
                f.write(f"  {positions[i][0]:12.6f}  {positions[i][1]:12.6f}  {positions[i][2]:12.6f}"
                        f"  {forces[i][0]:12.6f}  {forces[i][1]:12.6f}  {forces[i][2]:12.6f}\n")
            f.write(" -----------------------------------------------------------------------------------\n")

    def parse_vasp_output(self, run_dir):
        """Parse results from a completed calculation. For interface compatibility."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        energy = None
        forces = []

        with open(outcar_path) as f:
            for line in f:
                if 'free  energy   TOTEN' in line:
                    energy = float(line.split()[-2])
                if 'TOTAL-FORCE' in line:
                    next(f)  # skip dashes
                    forces = []
                    while True:
                        fline = next(f)
                        if '---' in fline:
                            break
                        parts = fline.split()
                        forces.append([float(parts[3]), float(parts[4]), float(parts[5])])

        return energy, np.array(forces) if forces else None

    def run_batch(self, run_dirs, run_type='thermal'):
        """Run multiple calculations sequentially.

        Args:
            run_dirs: List of directories, each containing a POSCAR
            run_type: 'main' or 'thermal'

        Returns:
            List of VASPRun objects
        """
        results = []
        for run_dir in run_dirs:
            try:
                result = self.submit_job(run_dir, run_type)
                results.append(result)
            except Exception as e:
                print(f"GAOP calculation failed in {run_dir}: {e}")
                results.append(None)
        return results

    def wait_for_batch(self, batch_info=None, run_dirs=None):
        """Return results for all directories (calculations already completed in submit_job)."""
        if run_dirs:
            return [VASPRun(run_dir) for run_dir in run_dirs]
        return []

    def wait_for_completion(self, run_dir, job_info=None, **kwargs):
        """Return results immediately (calculation already completed)."""
        return VASPRun(run_dir)

    def check_batch_status(self, batch_info=None):
        """GAOP calculations complete synchronously."""
        return 'completed'
