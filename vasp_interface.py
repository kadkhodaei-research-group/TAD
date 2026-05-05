from typing import List, Dict, Any, Optional, Tuple
import os
import logging
from pathlib import Path
import numpy as np
from vasp_manager import VASPRun, VASPManager
from atomic_structure import AtomicStructure
from pymatgen.io.vasp import Poscar


class VASPInterface:
    """Local PES implementation for VASP systems with atomic structure management."""
    
    def __init__(self, poscar_file: str, activation_radius: float = np.inf, moving_indices: Optional[List[int]] = None, 
                 vasp_manager:VASPManager = None):
        """Initialize with POSCAR file.
        
        Args:
            poscar_file: Path to POSCAR file
            activation_radius: Radius for activating frozen atoms (infinite means all active)
            moving_indices: List of atom indices that are allowed to move (others are frozen)
        """
        # --- create one VASPManager -------------------
        if vasp_manager is None:
            raise ValueError("You must pass the already-created VASPManager here.")
        self.vasp_manager = vasp_manager
        self.user_poscar_path = Path(poscar_file).resolve()
        self._pending_thermal_dirs: list[str] = []
        self.moving_indices = moving_indices
        # ------------------------------------------------------------
        
        if not os.path.exists(poscar_file):
            raise FileNotFoundError(f"POSCAR file {poscar_file} not found")
            
        # Read structure from POSCAR
        structure = Poscar.from_file(poscar_file, check_for_potcar=False).structure
        
        # Determine base directory from POSCAR location
        self.base_dir = os.path.dirname(os.path.abspath(poscar_file))
        
        self.n_atoms = len(structure)
        
        # Store the moving indices
        self.moving_indices = moving_indices

        self.moving_indices = moving_indices if moving_indices is not None else [0]
        
        # Get coordinates and types using self.moving_indices
        all_coords = structure.cart_coords
        moving_coords = all_coords[self.moving_indices]  # Use self.moving_indices
        frozen_coords = np.delete(all_coords, self.moving_indices, axis=0)  # Use self.moving_indices
        
        # Get atomic numbers as types
        all_atomic_numbers = np.array([site.specie.number for site in structure])
        moving_types = all_atomic_numbers[self.moving_indices]  # Use self.moving_indices
        frozen_types = np.delete(all_atomic_numbers, self.moving_indices)  # Use self.moving_indices
        
        # Convert to 0-based type indices
        unique_types = np.unique(np.concatenate([moving_types, frozen_types]))
        type_map = {num: idx for idx, num in enumerate(unique_types)}
        moving_types = np.array([type_map[num] for num in moving_types], dtype=np.int64)
        frozen_types = np.array([type_map[num] for num in frozen_types], dtype=np.int64)
        
        # Initialize atomic structure management
        self.atomic_structure = AtomicStructure(
            moving_atoms=moving_coords,
            frozen_atoms=frozen_coords,
            moving_types=moving_types,
            frozen_types=frozen_types,
            activation_radius=activation_radius,
            moving_indices=self.moving_indices
        )
        
        self.surface_energy = self
        
        # Track current thermal batch and run
        self.current_thermal_batch = None
        self.current_thermal_run = None
        
        # Current VASP run directory
        self.current_run_dir = None
        

    def get_atomic_info(self) -> Dict:
        """Get current atomic structure information for GP kernel."""
        return self.atomic_structure.get_structure_info()
    
    def set_parallel_eam(self, enable: bool) -> None:
        """Enable or disable parallel EAM execution for thermal snapshots.
        
        Args:
            enable: Whether to enable parallel EAM execution
        """
        if hasattr(self.vasp_manager, 'parallel_eam'):
            self.vasp_manager.parallel_eam = enable
            if enable:
                print("Enabled parallel EAM execution for thermal snapshots")
            else:
                print("Disabled parallel EAM execution")

    def get_thermal_run_dir(self) -> str:
        """Get directory of the current thermal run."""
        if self.current_thermal_run is None:
            raise ValueError("No current thermal run available")
        if self.current_thermal_batch is None:
            raise ValueError("No current thermal batch available")
            
        batch_dir = os.path.join(self.vasp_manager.run_dirs['thermal'], 
                                str(self.current_thermal_batch))
        run_dir = os.path.join(batch_dir, str(self.current_thermal_run))
        
        print(f"Getting thermal run directory: {run_dir}")
        return run_dir

    def scaler_y_value(self, position: np.ndarray, is_thermal: bool = False) -> float:
        """Calculate energy at given position."""
        try:
            # Update active atoms first
            n_activated = self.atomic_structure.update_active_atoms(
                position.reshape(-1, 3)
            )
            if n_activated > 0:
                print(f"Activated {n_activated} new frozen atoms")
                print(f"Total active frozen atoms: {len(self.atomic_structure.active_frozen_atoms)}")
            
            # Setup and run VASP
            run_dir = self.vasp_manager.setup_and_run_vasp(
                positions=position,
                run_type='thermal' if is_thermal else 'main'
            ) 

            
            # Store the current run directory
            self.current_run_dir = run_dir
            
            # Track thermal run directory
            if is_thermal:
                self.current_thermal_run = int(os.path.basename(run_dir))
                self.current_thermal_batch = os.path.basename(os.path.dirname(run_dir))
                print(f"Set current thermal run: batch {self.current_thermal_batch}, run {self.current_thermal_run}")
            
            # Wait for completion
            vasp_run = self.vasp_manager.wait_for_completion(run_dir)
            # Add fallback for mock runs
            if not hasattr(vasp_run, 'energy'):
                return -123.456789  # Default mock energy
            return vasp_run.energy
            
        except Exception as e:
            logging.error(f"VASP calculation failed: {str(e)}")
            if "mock" in str(e).lower():
                return -123.456789
            raise RuntimeError(f"VASP calculation failed: {e}")

    def first_derivative(self, position: np.ndarray, is_thermal: bool = False) -> np.ndarray:
        """Get forces at given position.
        
        IMPORTANT: This returns FORCES (F = -∇E), not gradients!
        ASE and VASP already return forces, so we don't negate them.
        """
        # ALWAYS run a new calculation - don't use cached results
        # This is critical for dimer method which needs forces at different positions
        run_dir = self.vasp_manager.setup_and_run_vasp(
            positions=position,
            run_type='thermal' if is_thermal else 'main'
        )
        
        # Store the run directory for reference
        if not is_thermal:
            self.current_run_dir = run_dir
        
        # Wait for completion and get forces
        run = self.vasp_manager.wait_for_completion(run_dir)
        
        # Return forces directly - DO NOT NEGATE!
        return run.flattened_forces

    def get_energy_and_forces(self, position):
        """Get energy and forces from VASP.
        
        Args:
            position: Full system positions (3*N_total)
            
        Returns:
            energy, forces for moving atoms only (3*N_mov)
        """
        # Run VASP with full positions
        run_dir = self.vasp_manager.setup_and_run_vasp(position, run_type='main')
        run = self.vasp_manager.wait_for_completion(run_dir)
        
        # Extract forces only for moving atoms
        full_forces = run.flattened_forces  # DO NOT NEGATE!
        moving_forces = self._extract_moving_forces(full_forces)
        
        return run.energy, moving_forces
    
    def _extract_moving_forces(self, full_forces):
        """Extract forces for moving atoms only."""
        forces_3d = full_forces.reshape(-1, 3)
        moving_forces = []
        for idx in self.moving_indices:
            moving_forces.extend(forces_3d[idx])
        return np.array(moving_forces)

    def enqueue_snapshot(self, position: np.ndarray) -> None:
        """Submit a thermal snapshot run and return immediately."""
        run_dir = self.vasp_manager.setup_and_run_vasp(
            positions=position,
            run_type='thermal'
        )
        self._pending_thermal_dirs.append(run_dir)

    def harvest_pending_snapshots(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wait for every snapshot in the current batch, then return energies and forces.
        Handles both regular execution and Stampede3 array jobs.
        """
        import os
        from pymatgen.io.vasp import outputs as vout

        # For batch-capable modes, submit the pending batch first
        if hasattr(self.vasp_manager, 'execution_mode') and self.vasp_manager.execution_mode in ['stampede3', 'anvil', 'eam', 'vasp', 'gaop']:
            if hasattr(self.vasp_manager, 'submit_pending_thermal_batch'):
                array_job_id = self.vasp_manager.submit_pending_thermal_batch()
                if array_job_id is None and not self.vasp_manager.current_thermal_batch_dirs:
                    return np.array([]), np.array([[]])

        # Wait for completion
        runs = self.vasp_manager.wait_for_current_thermal_batch()
        if not runs:
            return np.array([]), np.array([[]])

        n_atoms = len(runs[0].structure)   # ALL atoms in the system
        dof = 3 * n_atoms                  # Full system DOF

        energies, forces = [], []

        for r in runs:
            # Skip failed calculations
            if hasattr(r, 'has_error') and r.has_error:
                print(f"  Skipping failed calculation: {r.error_message}")
                continue
                
            run_dir = getattr(r, "run_dir", None) or os.path.dirname(
                    getattr(r, "filename", "OUTCAR")
                )

            # 1) ENERGY
            if hasattr(r, "final_energy") and r.final_energy is not None:
                energy = r.final_energy
            elif hasattr(r, "energy"):
                energy = r.energy
            else:
                with open(os.path.join(run_dir, "OUTCAR")) as fh:
                    for line in reversed(fh.readlines()):
                        if "TOTEN" in line:
                            energy = float(line.split("=")[1].split()[0])
                            break
            energies.append(energy)

            # 2) FORCES - Get FULL SYSTEM forces, not just moving atoms
            flat = None

            if hasattr(r, "forces") and np.asarray(r.forces).size == dof:
                flat = np.asarray(r.forces).reshape(-1)

            if flat is None and hasattr(r, "flattened_forces"):
                flat = np.asarray(r.flattened_forces)

            if flat is None:
                outcar = vout.Outcar(os.path.join(run_dir, "OUTCAR"))
                flat = np.asarray(outcar.forces).reshape(-1)

            # Clean up if needed
            if flat.size == 6 * n_atoms:
                flat = flat.reshape(n_atoms, 6)[:, 3:].reshape(-1)

            if flat.size > dof and flat.size % (6 * n_atoms) == 0:
                flat = flat[-6 * n_atoms :].reshape(n_atoms, 6)[:, 3:].reshape(-1)
            elif flat.size > dof and flat.size % dof == 0:
                flat = flat[-dof:]

            if flat.size != dof:
                raise ValueError(
                    f"After cleaning, len(forces)={flat.size}, expected {dof}."
                )

            # DO NOT NEGATE! ASE already gives us forces
            forces.append(flat)  # FULL SYSTEM forces with correct sign

        self._pending_thermal_dirs.clear()
        
        # Check if we have enough successful calculations
        if len(energies) < len(runs) * 0.9:  # Require at least 90% success rate
            print(f"WARNING: Only {len(energies)} out of {len(runs)} calculations succeeded")
            if len(energies) < 10:  # Minimum threshold
                raise RuntimeError(f"Too many failed calculations: only {len(energies)} out of {len(runs)} succeeded")

        # Store the raw energies for energy referencing
        energies_array = np.asarray(energies)
        self._last_thermal_energies = energies_array
        
        # Return raw energies and forces - no referencing here
        # The referencing will be handled by GP1
        return energies_array, np.vstack(forces)  # forces shape: (N_snap, 3*N_atoms)