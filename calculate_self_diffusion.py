#!/usr/bin/env python
"""Calculate self-diffusion using dimer method and local minimization.

This script:
1. Takes an initial configuration (possibly with a defect)
2. Finds the local minimum
3. Uses dimer method to find saddle point
4. Finds the final state minimum
5. Calculates activation energy and diffusion parameters
"""

import argparse
import os
import sys
import logging
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import matplotlib.pyplot as plt
from scipy import constants

# Import our methods
from walker_pure_dimer import WalkerPureDimer
from walker_minimizer import WalkerMinimizer
from vasp_manager import VASPManager, cleanup
from vasp_interface import VASPInterface
from output_manager import OutputManager, get_output_path, get_input_path


def setup_logging():
    """Set up logging."""
    logging.getLogger().handlers.clear()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    log_file = get_output_path('logs', 'self_diffusion_calculation.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")


class SelfDiffusionCalculator:
    """Calculate self-diffusion barriers and rates."""
    
    def __init__(self, system_params: dict):
        self.system_params = system_params
        self.results = {
            'initial_minimum': None,
            'saddle_point': None,
            'final_minimum': None,
            'activation_energy': None,
            'reverse_activation_energy': None,
            'attempt_frequency': None,
            'diffusion_coefficient': None,
            'trajectories': {}
        }
        
    def find_initial_minimum(self, poscar_file: str, local_pes) -> Tuple[np.ndarray, float, np.ndarray]:
        """Find the initial local minimum."""
        logging.info("Finding initial local minimum...")
        
        structure = Poscar.from_file(poscar_file, check_for_potcar=False).structure
        initial_position = structure.cart_coords.flatten()
        
        # Temporarily disable CheckpointManager for minimizer if it conflicts
        import minimizer_checkpoint_extension  # Ensure extension is loaded
        
        # Create minimizer walker
        walker = WalkerMinimizer(
            initial_position=initial_position,
            local_pes=local_pes,
            max_steps=self.system_params.get('min_max_steps', 100),
            method=self.system_params.get('min_method', 'fire'),
            step_size=self.system_params.get('min_step_size', 0.05),
            max_step_size=self.system_params.get('min_max_step_size', 0.1),
            stopping_criteria=self.system_params.get('min_stopping_criteria', 0.005),
            verbose=self.system_params.get('verbose', False),
            checkpoint_interval=10  # Less frequent checkpointing
        )
        
        # Run minimization
        final_pos, final_energy, final_forces = walker.run()
        
        if walker.converged:
            logging.info(f"Initial minimum found: E = {final_energy:.6f} eV")
        else:
            logging.warning("Initial minimization did not fully converge")
            
        self.results['initial_minimum'] = {
            'position': final_pos,
            'energy': final_energy,
            'forces': final_forces,
            'converged': walker.converged,
            'trajectory': walker.trajectory,
            'n_atoms': walker.n_atoms,
            'energy_reference': walker.energy_reference
        }
        
        # Save initial minimum structure
        self._save_structure(final_pos, structure, 'initial_minimum.vasp')
        
        return final_pos, final_energy, final_forces
    
    def find_saddle_point(self, initial_pos: np.ndarray, local_pes) -> Tuple[np.ndarray, float, np.ndarray]:
        """Find saddle point starting from initial minimum."""
        logging.info("Finding saddle point using dimer method...")
        
        # Create dimer walker - it will use its own checkpoint system
        walker = WalkerPureDimer(
            initial_position=initial_pos,
            local_pes=local_pes,
            max_dimer_steps=self.system_params.get('dimer_max_steps', 100),
            rotation=self.system_params.get('dimer_rotation', 'lbfgsext'),
            translation=self.system_params.get('dimer_translation', 'lbfgs'),
            dimer_sep=self.system_params.get('dimer_sep', 0.01),
            dimer_stopping_criteria=self.system_params.get('dimer_stopping_criteria', 0.01),
            step_size=self.system_params.get('dimer_step_size', 0.02),
            max_step_size=self.system_params.get('dimer_max_step_size', 0.05),
            verbose=self.system_params.get('verbose', False),
            checkpoint_interval=10  # Less frequent checkpointing
        )
        
        # Set initial dimer orientation if specified
        if 'dimer_initial_direction' in self.system_params:
            direction = np.array(self.system_params['dimer_initial_direction'])
            walker.dimer.set_initial_direction(direction)
        
        # Store walker reference for later use
        self._last_dimer_walker = walker
        
        # Run dimer search
        final_pos, final_energy, final_forces = walker.run()
        
        if walker.converged:
            logging.info(f"Saddle point found: E = {final_energy:.6f} eV")
            # Check if it's actually a saddle point by looking at curvature
            if hasattr(walker.dimer, 'Curv') and walker.dimer.Curv < 0:
                logging.info(f"Negative curvature confirmed: {walker.dimer.Curv:.4f}")
            else:
                logging.warning("Curvature check suggests this might not be a saddle point")
        else:
            logging.warning("Dimer search did not fully converge")
            
        self.results['saddle_point'] = {
            'position': final_pos,
            'energy': final_energy,
            'forces': final_forces,
            'converged': walker.converged,
            'trajectory': walker.trajectory,
            'curvature': getattr(walker.dimer, 'Curv', None)
        }
        
        # Save saddle point structure
        structure = Poscar.from_file(self.system_params['poscar_file'], check_for_potcar=False).structure
        self._save_structure(final_pos, structure, 'saddle_point.vasp')
        
        return final_pos, final_energy, final_forces
    
    def find_final_minimum(self, saddle_pos: np.ndarray, local_pes) -> Tuple[np.ndarray, float, np.ndarray]:
        """Find final minimum by following mode from saddle point."""
        logging.info("Finding final minimum from saddle point...")
        
        # Get the dimer walker that found the saddle point
        # We need to extract the unstable mode direction from it
        displacement = self.system_params.get('mode_following_displacement', 0.1)
        
        # Try to get the dimer orientation from the previous walker
        if hasattr(self, '_last_dimer_walker'):
            dimer_walker = self._last_dimer_walker
            if hasattr(dimer_walker, 'dimer') and hasattr(dimer_walker.dimer, 'orient'):
                unstable_mode = dimer_walker.dimer.orient.copy()
                logging.info(f"Using dimer orientation as unstable mode")
                
                # Make sure we go "forward" along the mode
                # Check which direction leads away from initial minimum
                initial_pos = self.results['initial_minimum']['position']
                test_forward = saddle_pos + 0.1 * unstable_mode
                test_backward = saddle_pos - 0.1 * unstable_mode
                
                dist_forward = np.linalg.norm(test_forward - initial_pos)
                dist_backward = np.linalg.norm(test_backward - initial_pos)
                
                if dist_backward > dist_forward:
                    unstable_mode = -unstable_mode
                    logging.info("Reversed unstable mode direction")
                
                initial_pos = saddle_pos + displacement * unstable_mode
                logging.info(f"Displaced along unstable mode by {displacement} Å")
            else:
                logging.warning("Could not extract dimer orientation")
                initial_pos = saddle_pos + displacement * np.random.randn(len(saddle_pos))
                initial_pos = saddle_pos + displacement * (initial_pos - saddle_pos) / np.linalg.norm(initial_pos - saddle_pos)
        else:
            # Fallback: use gradient at saddle point
            logging.info("Using gradient-based displacement")
            forces = local_pes.first_derivative(saddle_pos, is_thermal=False)
            if np.linalg.norm(forces) > 1e-6:
                # Follow the force direction (downhill)
                direction = forces / np.linalg.norm(forces)
                initial_pos = saddle_pos + displacement * direction
            else:
                # Last resort: random displacement
                logging.warning("Forces at saddle are too small, using random displacement")
                initial_pos = saddle_pos + displacement * np.random.randn(len(saddle_pos))
                initial_pos = saddle_pos + displacement * (initial_pos - saddle_pos) / np.linalg.norm(initial_pos - saddle_pos)
        
        # Import extension for minimizer
        import minimizer_checkpoint_extension
        
        # Now minimize from the displaced position
        walker = WalkerMinimizer(
            initial_position=initial_pos,
            local_pes=local_pes,
            max_steps=self.system_params.get('min_max_steps', 100),
            method=self.system_params.get('min_method', 'fire'),
            step_size=self.system_params.get('min_step_size', 0.05),
            max_step_size=self.system_params.get('min_max_step_size', 0.1),
            stopping_criteria=self.system_params.get('min_stopping_criteria', 0.005),
            verbose=self.system_params.get('verbose', False),
            checkpoint_interval=10  # Less frequent checkpointing
        )
        
        # Run minimization
        final_pos, final_energy, final_forces = walker.run()
        
        if walker.converged:
            logging.info(f"Final minimum found: E = {final_energy:.6f} eV")
        else:
            logging.warning("Final minimization did not fully converge")
            
        self.results['final_minimum'] = {
            'position': final_pos,
            'energy': final_energy,
            'forces': final_forces,
            'converged': walker.converged,
            'trajectory': walker.trajectory
        }
        
        # Save final minimum structure
        structure = Poscar.from_file(self.system_params['poscar_file'], check_for_potcar=False).structure
        self._save_structure(final_pos, structure, 'final_minimum.vasp')
        
        return final_pos, final_energy, final_forces
    
    def calculate_activation_energies(self):
        """Calculate forward activation energy."""
        if not all(key in self.results for key in ['initial_minimum', 'saddle_point']):
            raise ValueError("Missing required energy calculations")
        
        E_initial = self.results['initial_minimum']['energy']
        E_saddle = self.results['saddle_point']['energy']
        
        # Forward barrier (initial -> saddle)
        activation_energy_raw = E_saddle - E_initial
        
        # Apply migration correction if provided
        migration_correction = self.system_params.get('migration_correction', 0.0)
        
        self.results['activation_energy_raw'] = activation_energy_raw
        self.results['activation_energy'] = activation_energy_raw + migration_correction
        self.results['migration_correction'] = migration_correction
        
        logging.info(f"Forward activation energy (raw): {activation_energy_raw:.4f} eV")
        
        if migration_correction != 0.0:
            logging.info(f"Migration energy correction: {migration_correction:.4f} eV")
            logging.info(f"Forward activation energy (corrected): {self.results['activation_energy']:.4f} eV")
        else:
            logging.info(f"Forward activation energy: {self.results['activation_energy']:.4f} eV")
        
    def estimate_attempt_frequency(self):
        """Estimate attempt frequency using harmonic approximation or TST if possible."""
        # Try to calculate TST effective frequency if we have the necessary structures
        if (self.system_params.get('calculate_tst_frequency', False) and 
            'initial_minimum' in self.results and 'saddle_point' in self.results):
            
            try:
                # Need to save current structures for frequency calculation
                initial_struct = Poscar.from_file(self.system_params['poscar_file'], check_for_potcar=False).structure
                
                # Save current minimum state
                initial_coords = self.results['initial_minimum']['position'].reshape(-1, 3)
                initial_struct_min = Structure(
                    lattice=initial_struct.lattice,
                    species=initial_struct.species,
                    coords=initial_coords,
                    coords_are_cartesian=True
                )
                initial_poscar = Poscar(initial_struct_min)
                initial_file = get_output_path('results', 'initial_for_freq.vasp')
                initial_poscar.write_file(initial_file)
                
                # Save saddle state
                saddle_coords = self.results['saddle_point']['position'].reshape(-1, 3)
                saddle_struct = Structure(
                    lattice=initial_struct.lattice,
                    species=initial_struct.species,
                    coords=saddle_coords,
                    coords_are_cartesian=True
                )
                saddle_poscar = Poscar(saddle_struct)
                saddle_file = get_output_path('results', 'saddle_for_freq.vasp')
                saddle_poscar.write_file(saddle_file)
                
                # Calculate TST frequency
                nu_tst = self.calculate_effective_frequency_tst(
                    initial_file, saddle_file, self.system_params
                )
                
                self.results['effective_frequency_tst'] = nu_tst
                self.results['attempt_frequency'] = nu_tst
                logging.info(f"TST effective frequency: {nu_tst:.2e} Hz ({nu_tst/1e12:.3f} THz)")
                
            except Exception as e:
                logging.warning(f"Could not calculate TST frequency: {e}")
                # Fall back to default
                self.results['attempt_frequency'] = 1e13  # Hz (typical for metals)
                logging.info(f"Using default attempt frequency: {self.results['attempt_frequency']:.2e} Hz")
        else:
            # Use simple estimate
            self.results['attempt_frequency'] = 1e13  # Hz (typical for metals)
            logging.info(f"Estimated attempt frequency: {self.results['attempt_frequency']:.2e} Hz")
    def calculate_effective_frequency_tst(
            self,
            initial_state_file: str,
            saddle_state_file: str,
            system_params_orig: dict
    ) -> float:
        """Calculate effective frequency ν* using TST formula.
        
        ν* = Π(ν_i) / Π(ν_j') where:
        - ν_i are the 3N-3 normal frequencies at initial state (vacancy minimum)
        - ν_j' are the 3N-4 non-imaginary frequencies at transition state (saddle)
        
        Args:
            initial_state_file: POSCAR file for initial state (relaxed vacancy)
            saddle_state_file: POSCAR file for saddle point
            system_params_orig: System parameters
            
        Returns:
            Effective frequency in Hz
        """
        logging.info("Calculating effective frequency using TST...")
        
        # Import phonopy functions
        try:
            import phonopy_eam_clean as phonopy_calc
            from phonopy import Phonopy
            from phonopy.interface.vasp import read_vasp
            
            if not phonopy_calc.PHONOPY_AVAILABLE:
                logging.warning("Phonopy not available, using default frequency")
                return 1e13  # Default 10 THz
        except ImportError:
            logging.warning("phonopy_eam_clean not found, using default frequency")
            return 1e13
        
        # Setup directories in output directory
        work_dir = get_output_path('phonopy_calculations')
        
        # Phonopy parameters - use smaller cell for frequency calculation
        dim = [1, 1, 1]  # Unit cell is sufficient for local frequencies
        mp_grid = [1, 1, 1]  # Gamma point only
        
        try:
            # Calculate frequencies at initial state (vacancy minimum)
            logging.info("  Calculating frequencies at initial state (vacancy minimum)...")
            initial_dir = os.path.join(work_dir, "phonopy_initial_freq")
            os.makedirs(initial_dir, exist_ok=True)
            os.makedirs(os.path.join(initial_dir, 'displacements'), exist_ok=True)
            
            # Copy initial state
            import shutil
            shutil.copy(initial_state_file, os.path.join(initial_dir, 'POSCAR'))
            
            # Get structure info
            initial_structure = Poscar.from_file(initial_state_file, check_for_potcar=False).structure
            n_atoms = len(initial_structure)
            
            # Create Phonopy object directly
            from pathlib import Path
            unitcell = read_vasp(os.path.join(initial_dir, 'POSCAR'))
            phonon_initial = Phonopy(unitcell, dim)
            phonon_initial.generate_displacements(distance=0.01)
            
            # Calculate forces for displacements
            supercells = phonon_initial.get_supercells_with_displacements()
            forces_list = []
            
            if system_params_orig['execution_mode'] == 'eam':
                if system_params_orig.get('kim_model'):
                    from ase.calculators.kim import KIM
                    calc = KIM(system_params_orig['kim_model'])
                else:
                    from ase.calculators.eam import EAM
                    calc = EAM(potential=system_params_orig.get('eam_potential_file'))
                    
                # Calculate forces
                from ase import Atoms
                for i, scell in enumerate(supercells):
                    if scell is not None:
                        # Convert to ASE atoms
                        ase_atoms = Atoms(
                            symbols=scell.get_chemical_symbols(),
                            scaled_positions=scell.get_scaled_positions(),
                            cell=scell.get_cell(),
                            pbc=True
                        )
                        ase_atoms.calc = calc
                        forces = ase_atoms.get_forces()
                        forces_list.append(forces)
            
            # Set forces and calculate frequencies
            phonon_initial.forces = forces_list
            phonon_initial.produce_force_constants()
            
            # Get frequencies at Gamma point (in THz)
            phonon_initial.set_mesh([1, 1, 1])
            frequencies_initial = phonon_initial.get_frequencies([0, 0, 0])  # Gamma point
            
            # Remove acoustic modes (3 lowest frequencies)
            frequencies_initial = np.sort(frequencies_initial)[3:]  # Skip 3 acoustic modes
            frequencies_initial = frequencies_initial[frequencies_initial > 0.1]  # Remove near-zero
            
            logging.info(f"    Found {len(frequencies_initial)} optical modes at initial state")
            logging.info(f"    Frequency range: {frequencies_initial.min():.2f} - {frequencies_initial.max():.2f} THz")
            
            # Calculate frequencies at saddle point
            logging.info("  Calculating frequencies at saddle point...")
            saddle_dir = os.path.join(work_dir, "phonopy_saddle_freq")
            os.makedirs(saddle_dir, exist_ok=True)
            os.makedirs(os.path.join(saddle_dir, 'displacements'), exist_ok=True)
            
            # Copy saddle state
            shutil.copy(saddle_state_file, os.path.join(saddle_dir, 'POSCAR'))
            
            # Similar calculation for saddle
            unitcell_saddle = read_vasp(os.path.join(saddle_dir, 'POSCAR'))
            phonon_saddle = Phonopy(unitcell_saddle, dim)
            phonon_saddle.generate_displacements(distance=0.01)
            
            # Calculate forces for saddle
            supercells_saddle = phonon_saddle.get_supercells_with_displacements()
            forces_list_saddle = []
            
            for i, scell in enumerate(supercells_saddle):
                if scell is not None:
                    ase_atoms = Atoms(
                        symbols=scell.get_chemical_symbols(),
                        scaled_positions=scell.get_scaled_positions(),
                        cell=scell.get_cell(),
                        pbc=True
                    )
                    ase_atoms.calc = calc
                    forces = ase_atoms.get_forces()
                    forces_list_saddle.append(forces)
            
            phonon_saddle.forces = forces_list_saddle
            phonon_saddle.produce_force_constants()
            
            # Get frequencies at saddle
            phonon_saddle.set_mesh([1, 1, 1])
            frequencies_saddle = phonon_saddle.get_frequencies([0, 0, 0])
            
            # At saddle point, we should have one imaginary frequency
            # Remove it and the acoustic modes
            frequencies_saddle_real = []
            n_imaginary = 0
            
            for freq in frequencies_saddle:
                if freq < -0.1:  # Imaginary frequency (negative)
                    n_imaginary += 1
                    logging.info(f"    Found imaginary frequency: {freq:.2f} THz")
                elif freq > 0.1:  # Real frequency (skip near-zero acoustic)
                    frequencies_saddle_real.append(freq)
            
            frequencies_saddle_real = np.array(frequencies_saddle_real)
            
            logging.info(f"    Found {len(frequencies_saddle_real)} real modes at saddle")
            logging.info(f"    Found {n_imaginary} imaginary mode(s) at saddle")
            
            if n_imaginary != 1:
                logging.warning(f"    Expected 1 imaginary mode at saddle, found {n_imaginary}")
            
            # Calculate effective frequency using TST formula
            # ν* = Π(ν_i) / Π(ν_j')
            # To avoid numerical issues, use: log(ν*) = Σlog(ν_i) - Σlog(ν_j')
            
            log_nu_initial = np.sum(np.log(frequencies_initial))
            log_nu_saddle = np.sum(np.log(frequencies_saddle_real))
            
            log_nu_star = log_nu_initial - log_nu_saddle
            nu_star_thz = np.exp(log_nu_star)
            
            # Convert from THz to Hz
            nu_star_hz = nu_star_thz * 1e12
            
            logging.info(f"\n  TST effective frequency calculation:")
            logging.info(f"    Initial state: {len(frequencies_initial)} modes")
            logging.info(f"    Saddle point: {len(frequencies_saddle_real)} real modes")
            logging.info(f"    ν* = {nu_star_thz:.3f} THz = {nu_star_hz:.2e} Hz")
            
            return nu_star_hz
            
        except Exception as e:
            logging.error(f"Error calculating effective frequency: {e}")
            logging.info("Using default frequency 1e13 Hz")
            return 1e13
    
    def calculate_jump_distance_crystallographic(self, structure, jump_type='nn'):
        """Calculate crystallographic jump distance.
        
        Args:
            structure: pymatgen Structure object
            jump_type: 'nn' for nearest-neighbor jump
            
        Returns:
            Jump distance in Angstroms
        """
        lattice = structure.lattice
        
        # Determine structure type from lattice parameters
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        
        # Check if cubic (all angles 90° and a=b=c)
        is_cubic = (abs(alpha - 90) < 1e-3 and abs(beta - 90) < 1e-3 and 
                   abs(gamma - 90) < 1e-3 and abs(a - b) < 1e-3 and abs(b - c) < 1e-3)
        
        if is_cubic:
            # For BCC, nearest neighbor distance is (√3/2) * a
            # For FCC, nearest neighbor distance is (√2/2) * a
            
            # Detect BCC vs FCC based on structure
            # This is a simplified check - you might need to improve this
            n_atoms = len(structure)
            volume = structure.volume
            atoms_per_unit_volume = n_atoms / volume
            
            # BCC has 2 atoms per unit cell, FCC has 4
            # But we might have a supercell, so check packing fraction
            a0 = a  # Lattice parameter
            
            # Try to determine from system params if available
            structure_type = self.system_params.get('structure_type', 'bcc')
            
            if structure_type == 'bcc':
                d = np.sqrt(3) / 2 * a0
                logging.info(f"  BCC structure detected: a₀ = {a0:.3f} Å")
                logging.info(f"  Nearest-neighbor jump distance: d = (√3/2) × a₀ = {d:.3f} Å")
            elif structure_type == 'fcc':
                d = np.sqrt(2) / 2 * a0
                logging.info(f"  FCC structure detected: a₀ = {a0:.3f} Å")
                logging.info(f"  Nearest-neighbor jump distance: d = (√2/2) × a₀ = {d:.3f} Å")
            else:
                # Default to BCC for Mo
                d = np.sqrt(3) / 2 * a0
                logging.info(f"  Assuming BCC structure: a₀ = {a0:.3f} Å")
                logging.info(f"  Nearest-neighbor jump distance: d = (√3/2) × a₀ = {d:.3f} Å")
                
            return d
        else:
            logging.warning("Non-cubic structure detected, using actual jump distance")
            return None
        
    def calculate_formation_energy(self, perfect_structure_file: str, system_params_orig: dict) -> float:
        """Calculate vacancy formation energy.
        
        Args:
            perfect_structure_file: POSCAR file for perfect structure
            system_params_orig: Original system parameters
            
        Returns:
            Formation energy in eV
        """
        logging.info("Calculating vacancy formation energy...")
        
        # Load perfect structure
        perfect_structure = Poscar.from_file(perfect_structure_file, check_for_potcar=False).structure
        n_atoms_perfect = len(perfect_structure)
        
        # Create a new VASP manager specifically for the perfect structure
        logging.info("Creating new VASP manager for perfect structure...")
        perfect_poscar_path = os.path.abspath(perfect_structure_file)
        
        # Use output directory for perfect structure runs
        perfect_work_dir = get_output_path("vasp_runs_perfect")
        
        perfect_vasp_mgr = VASPManager(
            base_dir=perfect_work_dir,  # Use output directory
            user_poscar_path=perfect_poscar_path,
            execution_mode=system_params_orig['execution_mode'],
            mpi_command=system_params_orig.get('mpi_command'),
            vasp_command=system_params_orig.get('vasp_command', 'vasp_gam'),
            eam_potential_file=system_params_orig.get('eam_potential_file'),
            skip_thermal=True,
            kim_model_name=system_params_orig.get('kim_model'),
        )
        
        # Create a new VASPInterface for the perfect structure
        perfect_pes = VASPInterface(
            vasp_manager=perfect_vasp_mgr,
            poscar_file=perfect_structure_file,
            activation_radius=system_params_orig.get('activation_radius', 10.0),
            moving_indices=[0]  # Just use first atom for interface
        )
        
        # Minimize perfect structure to get its energy
        logging.info("Minimizing perfect structure...")
        perfect_pos = perfect_structure.cart_coords.flatten()
        
        # Create minimizer for perfect structure
        perfect_walker = WalkerMinimizer(
            initial_position=perfect_pos,
            local_pes=perfect_pes,
            max_steps=self.system_params.get('min_max_steps', 100),
            method=self.system_params.get('min_method', 'fire'),
            step_size=self.system_params.get('min_step_size', 0.05),
            max_step_size=self.system_params.get('min_max_step_size', 0.1),
            stopping_criteria=self.system_params.get('min_stopping_criteria', 0.005),
            verbose=False,  # Less verbose for perfect structure
            checkpoint_interval=50
        )
        
        # Run minimization
        perfect_final_pos, perfect_energy_raw, perfect_forces = perfect_walker.run()
        
        # Note: perfect_energy_raw is relative to perfect_walker's reference
        # We need the absolute energy
        perfect_energy_absolute = perfect_energy_raw + perfect_walker.energy_reference
        
        # Get energy of structure with vacancy (from our initial minimum)
        if self.results['initial_minimum'] is None:
            raise ValueError("Initial minimum not calculated yet")
        
        # Get absolute energy of vacancy structure
        vacancy_energy_relative = self.results['initial_minimum']['energy']
        vacancy_energy_reference = self.results['initial_minimum'].get('energy_reference', 
                                                                       self.results.get('energy_reference', 0.0))
        vacancy_energy_absolute = vacancy_energy_relative + vacancy_energy_reference
        
        n_atoms_vacancy = self.results['initial_minimum']['n_atoms']
        
        # Formation energy: E_f = E_vacancy - (N-1)/N * E_perfect
        # This accounts for the missing atom
        E_formation = vacancy_energy_absolute - (n_atoms_vacancy / n_atoms_perfect) * perfect_energy_absolute
        
        # Apply formation energy correction if provided
        formation_correction = system_params_orig.get('formation_correction', 0.0)
        E_formation_corrected = E_formation + formation_correction
        
        self.results['formation_energy_raw'] = E_formation
        self.results['formation_energy'] = E_formation_corrected
        self.results['formation_correction'] = formation_correction
        self.results['perfect_energy'] = perfect_energy_absolute
        self.results['vacancy_energy_absolute'] = vacancy_energy_absolute
        
        logging.info(f"Perfect structure energy: {perfect_energy_absolute:.6f} eV ({n_atoms_perfect} atoms)")
        logging.info(f"Vacancy structure energy: {vacancy_energy_absolute:.6f} eV ({n_atoms_vacancy} atoms)")
        logging.info(f"Vacancy formation energy (raw): {E_formation:.4f} eV")
        if formation_correction != 0.0:
            logging.info(f"Formation energy correction: {formation_correction:.4f} eV")
            logging.info(f"Vacancy formation energy (corrected): {E_formation_corrected:.4f} eV")
        else:
            logging.info(f"Vacancy formation energy: {E_formation:.4f} eV")
        
        # Save minimized perfect structure
        perfect_coords = perfect_final_pos.reshape(-1, 3)
        perfect_structure_min = Structure(
            lattice=perfect_structure.lattice,
            species=perfect_structure.species,
            coords=perfect_coords,
            coords_are_cartesian=True
        )
        perfect_poscar = Poscar(perfect_structure_min)
        perfect_poscar.write_file(get_output_path('results', 'perfect_minimized.vasp'))
        
        return E_formation
    
    def calculate_vibrational_entropy_total(self, structure_file: str, calc_type: str, 
                                          system_params_orig: dict, temperature: float = 300.0) -> float:
        """Calculate TOTAL vibrational entropy (not per atom) using phonopy.
        
        Args:
            structure_file: POSCAR file for the structure
            calc_type: Type of calculation ('perfect' or 'vacancy')
            system_params_orig: Original system parameters
            temperature: Temperature in K for entropy calculation
            
        Returns:
            Total vibrational entropy at temperature in k_B units
        """
        logging.info(f"Calculating total vibrational entropy for {calc_type} structure at {temperature}K...")
        
        # Import phonopy script functions
        try:
            import phonopy_eam_clean as phonopy_calc
            if not phonopy_calc.PHONOPY_AVAILABLE:
                logging.warning("Phonopy not available, returning zero entropy")
                return 0.0
        except ImportError:
            logging.warning("phonopy_eam_clean not found, returning zero entropy")
            return 0.0
        
        # Setup directory for phonon calculation in output directory
        work_dir = get_output_path('phonopy_calculations')
        calc_dir = os.path.join(work_dir, f"phonopy_{calc_type}")
        
        # Phonopy parameters
        dim = system_params_orig.get('phonopy_dim', [2, 2, 2])
        mp_grid = system_params_orig.get('phonopy_mp', [10, 10, 10])
        
        # Create directory structure with all subdirectories
        os.makedirs(calc_dir, exist_ok=True)
        os.makedirs(os.path.join(calc_dir, 'displacements'), exist_ok=True)
        os.makedirs(os.path.join(calc_dir, 'forces'), exist_ok=True)
        os.makedirs(os.path.join(calc_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(calc_dir, 'plots'), exist_ok=True)
        
        try:
            # Copy structure file
            import shutil
            shutil.copy(structure_file, os.path.join(calc_dir, 'POSCAR'))
            
            # Generate displacements
            logging.info(f"  Generating displacements with dim={dim}")
            # Convert calc_dir to Path object for phonopy_calc
            from pathlib import Path
            calc_dir_path = Path(calc_dir)
            
            # Important: ensure displacements directory exists before calling phonopy
            disp_dir = calc_dir_path / 'displacements'
            disp_dir.mkdir(parents=True, exist_ok=True)
            
            disp_files = phonopy_calc.run_phonopy_calculation(
                structure_file, calc_dir_path, dim
            )
            
            if len(disp_files) == 0:
                logging.error("No displacement files generated")
                return 0.0
            
            # Setup calculator
            if system_params_orig['execution_mode'] == 'eam':
                if system_params_orig.get('kim_model'):
                    from ase.calculators.kim import KIM
                    calc = KIM(system_params_orig['kim_model'])
                else:
                    from ase.calculators.eam import EAM
                    calc = EAM(potential=system_params_orig.get('eam_potential_file'))
            else:
                logging.warning("Only EAM mode supported for entropy calculation")
                return 0.0
            
            # Calculate forces
            logging.info(f"  Calculating forces for {len(disp_files)} displacements...")
            natoms, forces_data = phonopy_calc.calculate_forces(
                disp_files, calc, calc_dir_path
            )
            
            # Write FORCE_SETS
            force_sets_file = phonopy_calc.write_force_sets_fixed(
                natoms, forces_data, calc_dir_path
            )
            
            # Need to ensure we calculate up to at least the desired temperature
            tmax = max(int(temperature) + 100, 500)
            
            # Run phonopy to get thermal properties
            logging.info(f"  Running phonopy analysis (tmax={tmax}K)...")
            success = phonopy_calc.run_phonopy_analysis_direct(
                calc_dir_path, dim, mp_grid, tmax=tmax, band_path=None, forces_data=forces_data
            )
            
            if not success:
                # Try simpler method
                success = phonopy_calc.run_simple_phonopy_analysis(
                    calc_dir_path, dim, mp_grid, tmax=tmax, band_path=None, forces_data=forces_data
                )
            
            # Try multiple possible locations for thermal properties
            thermal_files = [
                os.path.join(calc_dir, 'results', 'thermal_properties.dat'),
                os.path.join(calc_dir, 'thermal_properties.dat'),
                os.path.join(calc_dir, 'results', 'thermal_properties.yaml'),
                os.path.join(calc_dir, 'thermal_properties.yaml')
            ]
            
            thermal_data_found = False
            S_total_kB = 0.0
            
            for thermal_file in thermal_files:
                if os.path.exists(thermal_file):
                    logging.info(f"  Found thermal properties file: {thermal_file}")
                    
                    if thermal_file.endswith('.dat'):
                        # Read data file
                        try:
                            data = np.loadtxt(thermal_file, skiprows=1)
                            if data.size == 0:
                                logging.warning(f"  Empty data file: {thermal_file}")
                                continue
                                
                            temps = data[:, 0]
                            entropy = data[:, 2]  # J/K/mol
                            
                            # Debug: print some values
                            logging.info(f"  Temperature range: {temps[0]:.1f} - {temps[-1]:.1f} K")
                            logging.info(f"  Entropy range: {entropy[0]:.2f} - {entropy[-1]:.2f} J/K/mol")
                            
                            # Find entropy at desired temperature
                            idx = np.argmin(np.abs(temps - temperature))
                            actual_T = temps[idx]
                            S_total_J = entropy[idx]  # J/K/mol for the entire supercell
                            
                            if abs(actual_T - temperature) > 50:
                                logging.warning(f"  Large temperature difference: requested {temperature}K, found {actual_T}K")
                            
                            thermal_data_found = True
                            break
                        except Exception as e:
                            logging.error(f"  Error reading {thermal_file}: {e}")
                            continue
                    
                    elif thermal_file.endswith('.yaml'):
                        # Try to parse YAML file
                        try:
                            import yaml
                            with open(thermal_file, 'r') as f:
                                thermal_yaml = yaml.safe_load(f)
                            
                            # Look for thermal properties in different possible locations
                            thermal_props = None
                            if isinstance(thermal_yaml, dict):
                                if 'thermal_properties' in thermal_yaml:
                                    thermal_props = thermal_yaml['thermal_properties']
                                elif 'phonon' in thermal_yaml and 'thermal_properties' in thermal_yaml['phonon']:
                                    thermal_props = thermal_yaml['phonon']['thermal_properties']
                            
                            if thermal_props and isinstance(thermal_props, list):
                                # Find closest temperature
                                best_idx = -1
                                best_diff = float('inf')
                                
                                for i, tp in enumerate(thermal_props):
                                    if 'temperature' in tp:
                                        diff = abs(tp['temperature'] - temperature)
                                        if diff < best_diff:
                                            best_diff = diff
                                            best_idx = i
                                
                                if best_idx >= 0:
                                    tp = thermal_props[best_idx]
                                    actual_T = tp['temperature']
                                    S_total_J = tp.get('entropy', 0.0)
                                    
                                    logging.info(f"  Found entropy at {actual_T}K: {S_total_J} J/K/mol")
                                    thermal_data_found = True
                                    break
                        except Exception as e:
                            logging.error(f"  Error parsing {thermal_file}: {e}")
                            continue
            
            if not thermal_data_found:
                logging.error("  No thermal properties data found!")
                logging.info("  Attempting direct phonopy calculation...")
                
                # Try direct calculation using phonopy Python API
                try:
                    from phonopy import Phonopy
                    from phonopy.interface.vasp import read_vasp
                    
                    unitcell = read_vasp(os.path.join(calc_dir, 'POSCAR'))
                    phonon = Phonopy(unitcell, dim)
                    
                    # Set forces
                    if hasattr(phonon, 'dataset'):
                        # Create dataset from forces
                        displacements = []
                        for i, forces in enumerate(forces_data):
                            displacements.append({'forces': forces})
                        phonon.dataset = {'first_atoms': displacements}
                    
                    # Calculate force constants
                    phonon.produce_force_constants()
                    
                    # Calculate thermal properties
                    phonon.run_mesh(mp_grid)
                    phonon.run_thermal_properties(t_min=0, t_max=tmax, t_step=10)
                    
                    # Get thermal properties dict
                    tp_dict = phonon.get_thermal_properties_dict()
                    temps = tp_dict['temperatures']
                    entropy = tp_dict['entropy']
                    
                    # Find closest temperature
                    idx = np.argmin(np.abs(temps - temperature))
                    S_total_J = entropy[idx]
                    
                    logging.info(f"  Direct calculation: S = {S_total_J} J/K/mol at {temps[idx]}K")
                    thermal_data_found = True
                    
                except Exception as e:
                    logging.error(f"  Direct phonopy calculation failed: {e}")
                    return 0.0
            
            if thermal_data_found and S_total_J > 0:
                # Convert from J/K/mol to k_B units
                # k_B = 1.380649e-23 J/K
                # N_A = 6.02214076e23 /mol
                k_B_SI = 1.380649e-23
                N_A = 6.02214076e23
                
                # S in k_B units = S(J/K/mol) / (N_A * k_B)
                S_total_kB = S_total_J / (N_A * k_B_SI)
                
                # Get actual number of atoms in the supercell
                supercell_natoms = natoms  # This is the supercell size
                unitcell_natoms = len(Poscar.from_file(structure_file, check_for_potcar=False).structure)
                supercell_factor = np.prod(dim)
                
                logging.info(f"\n  Entropy calculation details:")
                logging.info(f"    Unit cell atoms: {unitcell_natoms}")
                logging.info(f"    Supercell atoms: {supercell_natoms}")
                logging.info(f"    Supercell factor: {supercell_factor}")
                logging.info(f"    Total vibrational entropy at {temperature}K: {S_total_J:.3f} J/K/mol")
                logging.info(f"    Total vibrational entropy: {S_total_kB:.3f} k_B (for entire supercell)")
                logging.info(f"    Per unit cell: {S_total_kB/supercell_factor:.3f} k_B")
                
                # Return entropy per unit cell, not per supercell
                return S_total_kB / supercell_factor
            else:
                logging.warning(f"Invalid entropy value: {S_total_J} J/K/mol")
                return 0.0
                
        except Exception as e:
            logging.error(f"Error in entropy calculation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def calculate_formation_entropy(self, perfect_struct_file: str, 
                                  vacancy_struct_file: str,
                                  system_params_orig: dict,
                                  temperature: float = 300.0) -> float:
        """Calculate formation entropy from phonon calculations.
        
        Formation entropy: ΔS_v = S_vib(N-1) - (N-1)/N * S_vib(N)
        
        Args:
            perfect_struct_file: Perfect structure POSCAR (should be relaxed)
            vacancy_struct_file: Vacancy structure POSCAR (should be relaxed)
            system_params_orig: System parameters
            temperature: Temperature for entropy calculation
            
        Returns:
            S_f in k_B units
        """
        # Use the RELAXED structures for phonopy calculations
        # These should have been created earlier in the calculation
        perfect_relaxed_file = get_output_path('results', 'perfect_minimized.vasp')
        vacancy_relaxed_file = vacancy_struct_file  # This should already be the relaxed one
        
        # Check if relaxed files exist
        if not os.path.exists(perfect_relaxed_file):
            logging.warning(f"Relaxed perfect structure not found at {perfect_relaxed_file}, using unrelaxed")
            perfect_relaxed_file = perfect_struct_file
        else:
            logging.info(f"Using relaxed perfect structure: {perfect_relaxed_file}")
            
        if not os.path.exists(vacancy_relaxed_file):
            logging.warning(f"Relaxed vacancy structure not found at {vacancy_relaxed_file}")
            
        # Get number of atoms from the relaxed structures
        perfect_structure = Poscar.from_file(perfect_relaxed_file, check_for_potcar=False).structure
        vacancy_structure = Poscar.from_file(vacancy_relaxed_file, check_for_potcar=False).structure
        n_perfect = len(perfect_structure)
        n_vacancy = len(vacancy_structure)
        
        logging.info(f"\nCalculating formation entropy:")
        logging.info(f"  Perfect structure (relaxed): {n_perfect} atoms")
        logging.info(f"  Vacancy structure (relaxed): {n_vacancy} atoms")
        
        # Calculate TOTAL vibrational entropy for RELAXED perfect structure
        S_perfect_total = self.calculate_vibrational_entropy_total(
            perfect_relaxed_file, 'perfect', system_params_orig, temperature
        )
        
        # Calculate TOTAL vibrational entropy for RELAXED vacancy structure
        S_vacancy_total = self.calculate_vibrational_entropy_total(
            vacancy_relaxed_file, 'vacancy', system_params_orig, temperature
        )
        
        # Formation entropy calculation
        # ΔS_v = S_vib(N-1) - (N-1)/N * S_vib(N)
        S_f = S_vacancy_total - (n_vacancy / n_perfect) * S_perfect_total
        
        logging.info(f"\nFormation entropy calculation:")
        logging.info(f"  S_vib(perfect relaxed, total): {S_perfect_total:.3f} k_B")
        logging.info(f"  S_vib(vacancy relaxed, total): {S_vacancy_total:.3f} k_B")
        logging.info(f"  (N-1)/N factor: {n_vacancy/n_perfect:.6f}")
        logging.info(f"  Formation entropy S_f: {S_f:.3f} k_B")
        
        return S_f
    
    def calculate_diffusion_coefficient_complete(self, temperature: float = 300.0,
                                               S_f: float = None, S_m: float = None) -> float:
        """Calculate diffusion coefficient with full vacancy thermodynamics.
        
        Uses the formula: D = C_v × d² × Γ
        where:
        - C_v = exp(S_f/k_B) × exp(-H_f/k_B T) is vacancy concentration
        - d is the jump distance (crystallographic for BCC/FCC)
        - Γ = ν* × exp(-H_m/k_B T) is the jump rate
        
        Args:
            temperature: Temperature in Kelvin
            S_f: Formation entropy in k_B units (if None, calculate or use zero)
            S_m: Migration entropy in k_B units (should be ~0 for vacancy mechanism)
            
        Returns:
            Diffusion coefficient in m²/s
        """
        if self.results['activation_energy'] is None:
            raise ValueError("Migration energy not calculated")
            
        # Constants
        k_B = constants.k / constants.e  # Boltzmann constant in eV/K
        
        # Enthalpies
        H_m = self.results['activation_energy']  # Migration enthalpy
        
        # Check if we have formation energy
        if 'formation_energy' not in self.results:
            raise ValueError("Formation energy not calculated. Please provide perfect structure with --perfect-structure")
        
        H_f = self.results['formation_energy']
        
        # Entropies (in units of k_B)
        if S_f is None:
            S_f = self.results.get('formation_entropy', 0.0)
            if S_f is None:
                S_f = 0.0
                logging.info("Using S_f = 0 (no formation entropy contribution)")
        
        # Migration entropy should be ~0 for vacancy diffusion
        if S_m is None:
            S_m = 0.0
            logging.info("Using S_m = 0 (typical for vacancy migration)")
            
        self.results['formation_entropy'] = S_f
        self.results['migration_entropy'] = S_m
        
        # Calculate crystallographic jump distance
        structure = Poscar.from_file(self.system_params['poscar_file'], check_for_potcar=False).structure
        d_crystal = self.calculate_jump_distance_crystallographic(structure)
        
        # Also calculate actual jump distance for comparison
        if 'jump_distance' not in self.results:
            initial_pos = self.results['initial_minimum']['position'].reshape(-1, 3)
            final_pos = self.results['final_minimum']['position'].reshape(-1, 3)
            displacements = final_pos - initial_pos
            distances = np.linalg.norm(displacements, axis=1)
            self.results['jump_distance_actual'] = np.max(distances)
            logging.info(f"  Actual atomic displacement: {self.results['jump_distance_actual']:.3f} Å")
        
        # Use crystallographic distance if available, otherwise use actual
        if d_crystal is not None:
            d = d_crystal
            self.results['jump_distance'] = d_crystal
            logging.info(f"  Using crystallographic jump distance: {d:.3f} Å")
        else:
            d = self.results.get('jump_distance_actual', self.results.get('jump_distance'))
            logging.info(f"  Using actual jump distance: {d:.3f} Å")
        
        d_m = d * 1e-10  # Convert to meters
        
        # Attempt frequency - try to use TST value if available
        nu = self.results.get('effective_frequency_tst', self.results['attempt_frequency'])
        
        # Calculate vacancy concentration: C_v = exp(S_f/k_B) × exp(-H_f/k_B T)
        C_v = np.exp(S_f) * np.exp(-H_f / (k_B * temperature))
        
        # Calculate jump rate: Γ = ν* × exp(-H_m/k_B T)
        Gamma = nu * np.exp(-H_m / (k_B * temperature))
        
        # Diffusion coefficient: D = C_v × d² × Γ
        D = C_v * d_m**2 * Gamma
        
        # Also calculate with correlation factor for comparison
        structure_type = self.system_params.get('structure_type', 'bcc')
        correlation_factors = {'fcc': 0.7815, 'bcc': 0.7272, 'sc': 0.6531, 'hcp': 0.7815}
        f = correlation_factors.get(structure_type, 0.7272)
        
        # Alternative formula: D = f × d² × ν × exp(S_f + S_m) × exp(-(H_f + H_m)/(k_B T))
        D_alt = f * d_m**2 * nu * np.exp(S_f + S_m) * np.exp(-(H_f + H_m) / (k_B * temperature))
        
        # Store detailed results
        self.results['diffusion_details'] = {
            'vacancy_concentration': C_v,
            'jump_rate': Gamma,
            'jump_distance': d,
            'jump_distance_actual': self.results.get('jump_distance_actual', d),
            'effective_frequency': nu,
            'correlation_factor': f,
            'pre_exponential': C_v * d_m**2 * nu,
            'total_activation_energy': H_f + H_m,
            'formation_enthalpy': H_f,
            'migration_enthalpy': H_m,
            'formation_entropy': S_f,
            'migration_entropy': S_m,
            'temperature': temperature,
            'diffusion_alt': D_alt
        }
        
        logging.info(f"\nComplete diffusion calculation at {temperature} K:")
        logging.info(f"  Formation enthalpy (H_f): {H_f:.3f} eV")
        logging.info(f"  Migration enthalpy (H_m): {H_m:.3f} eV") 
        logging.info(f"  Total activation energy (Q): {H_f + H_m:.3f} eV")
        logging.info(f"  Formation entropy (S_f): {S_f:.3f} k_B")
        logging.info(f"  Migration entropy (S_m): {S_m:.3f} k_B")
        logging.info(f"  Jump distance (crystallographic): {d:.3f} Å")
        logging.info(f"  Effective frequency (ν*): {nu:.2e} Hz ({nu/1e12:.3f} THz)")
        logging.info(f"  Vacancy concentration (C_v): {C_v:.2e}")
        logging.info(f"  Jump rate (Γ): {Gamma:.2e} Hz")
        logging.info(f"  Diffusion coefficient (D = C_v × d² × Γ): {D:.2e} m²/s")
        logging.info(f"  Diffusion coefficient (alternative with f): {D_alt:.2e} m²/s")
        
        return D
    
    def calculate_diffusion_coefficient(self, temperature: float = 300.0) -> float:
        """Calculate diffusion coefficient using simple migration-only model.
        
        This is used when formation energy is not available.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Diffusion coefficient in m²/s
        """
        if self.results['activation_energy'] is None:
            raise ValueError("Activation energy not calculated")
            
        # Constants
        k_B = constants.k / constants.e  # Boltzmann constant in eV/K
        
        # Get structural info
        structure = Poscar.from_file(self.system_params['poscar_file'], check_for_potcar=False).structure
        
        # Estimate jump distance (distance between initial and final minima)
        initial_pos = self.results['initial_minimum']['position'].reshape(-1, 3)
        final_pos = self.results['final_minimum']['position'].reshape(-1, 3)
        
        # Find which atom moved the most
        displacements = final_pos - initial_pos
        distances = np.linalg.norm(displacements, axis=1)
        max_displacement_idx = np.argmax(distances)
        jump_distance = distances[max_displacement_idx]
        
        # Only log once
        if 'jump_distance' not in self.results:
            logging.info(f"Jump distance: {jump_distance:.3f} Å")
            logging.info(f"Jumping atom index: {max_displacement_idx}")
            self.results['jump_distance'] = jump_distance
        
        # Calculate jump rate using TST
        nu = self.results['attempt_frequency']
        E_a = self.results['activation_energy']
        
        jump_rate = nu * np.exp(-E_a / (k_B * temperature))
        
        # For self-diffusion in cubic systems: D = (1/6) * a² * Γ
        # where a is jump distance and Γ is jump rate
        # The factor depends on crystal structure and jump mechanism
        
        # Simplified: D = (jump_distance²) * jump_rate / 6
        D = (jump_distance * 1e-10)**2 * jump_rate / 6  # m²/s
        
        self.results['jump_rate'] = jump_rate
        self.results['temperature'] = temperature
        
        logging.info(f"Jump rate at {temperature} K: {jump_rate:.2e} Hz")
        logging.info(f"Diffusion coefficient at {temperature} K: {D:.2e} m²/s")
        
        return D
    
    def calculate_diffusion_over_temperature_range(self, T_min: float = 300.0, 
                                                  T_max: float = 2000.0, 
                                                  n_points: int = 50,
                                                  use_complete_model: bool = None):
        """Calculate diffusion coefficients over a temperature range.
        
        Args:
            T_min: Minimum temperature (K)
            T_max: Maximum temperature (K)
            n_points: Number of temperature points
            use_complete_model: If True, use complete vacancy model. If None, auto-detect.
            
        Returns:
            Dictionary with temperature array and diffusion coefficient array
        """
        if self.results['activation_energy'] is None:
            raise ValueError("Activation energy not calculated")
            
        # Determine which model to use
        if use_complete_model is None:
            # Use complete model if we have formation energy
            use_complete_model = 'formation_energy' in self.results
        
        # Create temperature array
        temperatures = np.linspace(T_min, T_max, n_points)
        diffusion_coeffs = []
        
        logging.info(f"Calculating diffusion coefficients from {T_min} K to {T_max} K...")
        logging.info(f"Using {'complete vacancy' if use_complete_model else 'simple migration'} model")
        
        # Get entropy values if available
        S_f = self.results.get('formation_entropy', 0.0)
        S_m = self.results.get('migration_entropy', 0.0)
        
        for T in temperatures:
            if use_complete_model:
                D = self.calculate_diffusion_coefficient_complete(T, S_f, S_m)
            else:
                D = self.calculate_diffusion_coefficient(T)
            diffusion_coeffs.append(D)
            
        self.results['temperature_range'] = {
            'temperatures': temperatures,
            'diffusion_coefficients': np.array(diffusion_coeffs),
            'T_min': T_min,
            'T_max': T_max,
            'model_used': 'complete' if use_complete_model else 'simple'
        }
        
        return self.results['temperature_range']
    
    def plot_arrhenius(self):
        """Create Arrhenius plot (log D vs 1/T)."""
        if 'temperature_range' not in self.results:
            logging.warning("No temperature range data available for Arrhenius plot")
            return
            
        T_data = self.results['temperature_range']['temperatures']
        D_data = self.results['temperature_range']['diffusion_coefficients']
        
        # Calculate 1/T in units of 1000/K for better x-axis values
        inv_T = 1000.0 / T_data
        
        # Calculate log10(D) 
        log_D = np.log10(D_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Arrhenius plot
        ax1.plot(inv_T, log_D, 'o-', markersize=6, linewidth=2, color='darkblue')
        
        # Fit a line to get activation energy (for verification)
        # log10(D) = log10(D0) - Ea/(2.303*kB*T)
        # So slope = -Ea/(2.303*kB)
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, log_D)
        
        # Plot fit line
        fit_line = slope * inv_T + intercept
        ax1.plot(inv_T, fit_line, '--', color='red', linewidth=2, 
                label=f'Linear fit (R² = {r_value**2:.4f})')
        
        # Calculate activation energy from slope
        k_B_eV = constants.k / constants.e  # in eV/K
        E_a_from_fit = -slope * 2.303 * k_B_eV * 1000  # Convert back from 1000/K
        
        ax1.set_xlabel('1000/T (K⁻¹)', fontsize=12)
        ax1.set_ylabel('log₁₀(D) (m²/s)', fontsize=12)
        ax1.set_title('Arrhenius Plot for Self-Diffusion', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add text box with activation energy
        model_type = self.results['temperature_range'].get('model_used', 'simple')
        if model_type == 'complete' and 'diffusion_details' in self.results:
            Q_total = self.results['diffusion_details']['total_activation_energy']
            H_f = self.results['diffusion_details']['formation_enthalpy']
            H_m = self.results['diffusion_details']['migration_enthalpy']
            S_f = self.results['diffusion_details'].get('formation_entropy', 0.0)
            textstr = (f'Total Q (from fit) = {E_a_from_fit:.3f} eV\n'
                      f'Total Q (calc) = {Q_total:.3f} eV\n'
                      f'H_f = {H_f:.3f} eV, H_m = {H_m:.3f} eV\n'
                      f'S_f = {S_f:.3f} k_B')
        else:
            textstr = f'E_a (from fit) = {E_a_from_fit:.3f} eV\nE_a (migration) = {self.results["activation_energy"]:.3f} eV'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Add temperature labels on top x-axis
        ax1_top = ax1.twiny()
        # Select a few temperature points to label
        temp_labels = [300, 500, 700, 1000, 1500, 2000]
        inv_temp_labels = [1000.0/T for T in temp_labels if T_data.min() <= T <= T_data.max()]
        temp_label_strings = [str(T) for T in temp_labels if T_data.min() <= T <= T_data.max()]
        ax1_top.set_xlim(ax1.get_xlim())
        ax1_top.set_xticks(inv_temp_labels)
        ax1_top.set_xticklabels(temp_label_strings)
        ax1_top.set_xlabel('Temperature (K)', fontsize=12)
        
        # Subplot 2: Linear scale D vs T
        ax2.plot(T_data, D_data * 1e4, 'o-', markersize=6, linewidth=2, color='darkgreen')
        ax2.set_xlabel('Temperature (K)', fontsize=12)
        ax2.set_ylabel('D (10⁻⁴ m²/s)', fontsize=12)
        ax2.set_title('Diffusion Coefficient vs Temperature', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Use scientific notation for y-axis if needed
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plots_dir = get_output_path('plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'diffusion_arrhenius_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Arrhenius plot saved to {os.path.join(plots_dir, 'diffusion_arrhenius_plot.png')}")
        logging.info(f"Activation energy from Arrhenius fit: {E_a_from_fit:.3f} eV")
        
        # Also create a simple single Arrhenius plot
        plt.figure(figsize=(8, 6))
        plt.plot(inv_T, log_D, 'o', markersize=8, color='darkblue', label='Data')
        plt.plot(inv_T, fit_line, '--', color='red', linewidth=2, label='Fit')
        
        plt.xlabel('1000/T (K⁻¹)', fontsize=14)
        plt.ylabel('log₁₀(D) (m²/s)', fontsize=14)
        plt.title('Arrhenius Plot for Self-Diffusion', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add activation energy text
        plt.text(0.02, 0.02, f'E_a = {self.results["activation_energy"]:.3f} eV', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'diffusion_arrhenius_simple.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Simple Arrhenius plot saved to {os.path.join(plots_dir, 'diffusion_arrhenius_simple.png')}")
    
    def plot_energy_profile(self):
        """Plot the energy profile along the diffusion path."""
        if not all(key in self.results for key in ['initial_minimum', 'saddle_point', 'final_minimum']):
            logging.warning("Cannot plot: missing energy calculations")
            return
            
        # Create simple reaction coordinate
        energies = [
            self.results['initial_minimum']['energy'],
            self.results['saddle_point']['energy'],
            self.results['final_minimum']['energy']
        ]
        
        labels = ['Initial', 'Saddle', 'Final']
        x = [0, 0.5, 1]
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, energies, 'o-', markersize=10, linewidth=2)
        
        # Add forward activation energy arrow
        E_a_forward = self.results['activation_energy']
        
        plt.annotate('', xy=(0.5, energies[1]), xytext=(0, energies[0]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        plt.text(0.25, (energies[0] + energies[1])/2, f'{E_a_forward:.3f} eV',
                ha='center', va='bottom', color='red', fontsize=12, fontweight='bold')
        
        plt.xlabel('Reaction Coordinate', fontsize=12)
        plt.ylabel('Energy (eV)', fontsize=12)
        plt.title('Self-Diffusion Energy Profile', fontsize=14)
        plt.xticks(x, labels)
        plt.grid(True, alpha=0.3)
        
        # Add a text box with energy values
        textstr = f'E_initial = {energies[0]:.3f} eV\nE_saddle = {energies[1]:.3f} eV\nE_final = {energies[2]:.3f} eV\nE_a = {E_a_forward:.3f} eV'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        plots_dir = get_output_path('plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'diffusion_energy_profile.png'), dpi=300)
        plt.close()
        
        logging.info(f"Energy profile saved to {os.path.join(plots_dir, 'diffusion_energy_profile.png')}")
    
    def _save_structure(self, positions: np.ndarray, template_structure: Structure, filename: str):
        """Save structure to file."""
        coords = positions.reshape(-1, 3)
        new_structure = Structure(
            lattice=template_structure.lattice,
            species=template_structure.species,
            coords=coords,
            coords_are_cartesian=True
        )
        poscar = Poscar(new_structure)
        poscar.write_file(get_output_path('results', filename))
        
    def save_results(self):
        """Save all results to file."""
        results_dir = get_output_path('results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save pickle file with all data
        with open(os.path.join(results_dir, 'self_diffusion_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
            
        # Save human-readable summary
        with open(os.path.join(results_dir, 'self_diffusion_summary.txt'), 'w') as f:
            f.write("SELF-DIFFUSION CALCULATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            if self.results['initial_minimum']:
                f.write(f"Initial Minimum Energy: {self.results['initial_minimum']['energy']:.6f} eV\n")
                f.write(f"Initial Converged: {self.results['initial_minimum']['converged']}\n\n")
                
            if self.results['saddle_point']:
                f.write(f"Saddle Point Energy: {self.results['saddle_point']['energy']:.6f} eV\n")
                f.write(f"Saddle Converged: {self.results['saddle_point']['converged']}\n")
                if self.results['saddle_point']['curvature'] is not None:
                    f.write(f"Curvature at Saddle: {self.results['saddle_point']['curvature']:.4f}\n\n")
                    
            if self.results['final_minimum']:
                f.write(f"Final Minimum Energy: {self.results['final_minimum']['energy']:.6f} eV\n")
                f.write(f"Final Converged: {self.results['final_minimum']['converged']}\n\n")
                
            if self.results['activation_energy'] is not None:
                f.write(f"Forward Activation Energy (raw): {self.results.get('activation_energy_raw', self.results['activation_energy']):.4f} eV\n")
                if self.results.get('migration_correction', 0.0) != 0.0:
                    f.write(f"Migration Correction: {self.results['migration_correction']:.4f} eV\n")
                    f.write(f"Forward Activation Energy (corrected): {self.results['activation_energy']:.4f} eV\n\n")
                else:
                    f.write("\n")
                
            if 'formation_energy' in self.results:
                f.write(f"Formation Energy (raw): {self.results.get('formation_energy_raw', self.results['formation_energy']):.4f} eV\n")
                if self.results.get('formation_correction', 0.0) != 0.0:
                    f.write(f"Formation Correction: {self.results['formation_correction']:.4f} eV\n")
                    f.write(f"Formation Energy (corrected): {self.results['formation_energy']:.4f} eV\n")
                else:
                    f.write(f"Formation Energy: {self.results['formation_energy']:.4f} eV\n")
                f.write(f"Formation Entropy: {self.results.get('formation_entropy', 0.0):.3f} k_B\n")
                f.write(f"Migration Entropy: {self.results.get('migration_entropy', 0.0):.3f} k_B\n\n")
                
            if 'jump_distance' in self.results:
                f.write(f"Jump Distance (crystallographic): {self.results['jump_distance']:.3f} Å\n")
                if 'jump_distance_actual' in self.results:
                    f.write(f"Jump Distance (actual displacement): {self.results['jump_distance_actual']:.3f} Å\n")
                f.write(f"Attempt Frequency: {self.results['attempt_frequency']:.2e} Hz")
                if 'effective_frequency_tst' in self.results:
                    f.write(f" ({self.results['effective_frequency_tst']/1e12:.3f} THz)")
                f.write("\n\n")
                
            if 'temperature_range' in self.results:
                f.write("TEMPERATURE-DEPENDENT DIFFUSION\n")
                f.write("-"*30 + "\n")
                T_data = self.results['temperature_range']['temperatures']
                D_data = self.results['temperature_range']['diffusion_coefficients']
                
                # Write selected temperature points
                indices = np.linspace(0, len(T_data)-1, min(20, len(T_data)), dtype=int)
                f.write("T (K)    D (m²/s)\n")
                for i in indices:
                    f.write(f"{T_data[i]:6.0f}   {D_data[i]:.2e}\n")
                    
        # Also save temperature-dependent data as CSV
        if 'temperature_range' in self.results:
            import csv
            with open(os.path.join(results_dir, 'diffusion_vs_temperature.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Temperature (K)', 'Diffusion Coefficient (m²/s)', 
                               '1000/T (K⁻¹)', 'log10(D)'])
                
                T_data = self.results['temperature_range']['temperatures']
                D_data = self.results['temperature_range']['diffusion_coefficients']
                
                for T, D in zip(T_data, D_data):
                    writer.writerow([T, D, 1000.0/T, np.log10(D)])
                    
            logging.info(f"Temperature-dependent data saved to {os.path.join(results_dir, 'diffusion_vs_temperature.csv')}")
                
        logging.info(f"Results saved to {os.path.join(results_dir, 'self_diffusion_results.pkl')} and {os.path.join(results_dir, 'self_diffusion_summary.txt')}")


def run_self_diffusion_calculation(poscar_file: str, system_params: dict):
    """Run complete self-diffusion calculation."""
    
    # Setup
    setup_logging()
    logging.info("Starting self-diffusion calculation")
    
    # Handle input file path
    if not os.path.isabs(poscar_file):
        poscar_path = get_input_path(poscar_file)
        if not os.path.exists(poscar_path):
            # Backward compatibility - check current directory
            if os.path.exists(poscar_file):
                poscar_path = os.path.abspath(poscar_file)
                logging.warning(f"Found {poscar_file} in current directory")
                logging.warning("Consider moving it to inputs/ directory")
            else:
                raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
    else:
        poscar_path = poscar_file
    
    # Update poscar_file to use the resolved path
    poscar_file = poscar_path
    system_params['poscar_file'] = poscar_path  # Update system_params with resolved path
    
    # Clean up previous runs unless continuation
    if not system_params.get('continuation', False):
        cleanup()
    
    # Create VASP/EAM interface
    work_dir = get_output_path("vasp_runs")
    
    vasp_mgr = VASPManager(
        base_dir=work_dir,  # work_dir already includes "vasp_runs"
        user_poscar_path=poscar_path,
        execution_mode=system_params['execution_mode'],
        mpi_command=system_params.get('mpi_command'),
        vasp_command=system_params.get('vasp_command', 'vasp_gam'),
        eam_potential_file=system_params.get('eam_potential_file'),
        skip_thermal=True,
        kim_model_name=system_params.get('kim_model'),
    )
    
    local_pes = VASPInterface(
        vasp_manager=vasp_mgr,
        poscar_file=poscar_file,
        activation_radius=system_params.get('activation_radius', 10.0),
        moving_indices=system_params.get('moving_indices', [0])
    )
    
    # Create calculator
    calculator = SelfDiffusionCalculator(system_params)
    
    try:
        # Step 1: Find initial minimum
        initial_pos, initial_E, initial_F = calculator.find_initial_minimum(poscar_file, local_pes)
        
        # Save minimized vacancy structure for entropy calculation
        vacancy_coords = initial_pos.reshape(-1, 3)
        vacancy_structure = Poscar.from_file(poscar_file, check_for_potcar=False).structure
        vacancy_structure_min = Structure(
            lattice=vacancy_structure.lattice,
            species=vacancy_structure.species,
            coords=vacancy_coords,
            coords_are_cartesian=True
        )
        vacancy_poscar = Poscar(vacancy_structure_min)
        vacancy_poscar.write_file(get_output_path('results', 'vacancy_minimized.vasp'))
        
        # Step 2: Find saddle point
        saddle_pos, saddle_E, saddle_F = calculator.find_saddle_point(initial_pos, local_pes)
        
        # Step 3: Find final minimum
        final_pos, final_E, final_F = calculator.find_final_minimum(saddle_pos, local_pes)
        
        # Step 4: Calculate activation energies
        calculator.calculate_activation_energies()
        
        # Step 4.5: Calculate formation energy if perfect structure provided
        if system_params.get('perfect_structure'):
            perfect_structure_path = system_params['perfect_structure']
            # Handle perfect structure path similar to poscar file
            if not os.path.isabs(perfect_structure_path):
                input_path = get_input_path(perfect_structure_path)
                if os.path.exists(input_path):
                    perfect_structure_path = input_path
                elif os.path.exists(perfect_structure_path):
                    # Backward compatibility
                    perfect_structure_path = os.path.abspath(perfect_structure_path)
                    logging.warning(f"Found {system_params['perfect_structure']} in current directory")
                    logging.warning("Consider moving it to inputs/ directory")
                else:
                    raise FileNotFoundError(f"Perfect structure file not found: {perfect_structure_path}")
            
            calculator.calculate_formation_energy(perfect_structure_path, system_params)
            
            # Step 4.6: Calculate formation entropy if requested
            if system_params.get('calculate_entropy', False):
                logging.info("\nCalculating formation entropy from phonon calculations...")
                
                # Use the temperature specified or default to 300K
                entropy_temp = system_params.get('entropy_temperature', 300.0)
                
                # IMPORTANT: Use the RELAXED structures for entropy calculation
                # The perfect structure was already relaxed and saved as perfect_minimized.vasp
                # The vacancy structure was already relaxed and saved as vacancy_minimized.vasp
                S_f = calculator.calculate_formation_entropy(
                    get_output_path('results', 'perfect_minimized.vasp'),  # Use relaxed perfect structure
                    get_output_path('results', 'vacancy_minimized.vasp'),   # Use relaxed vacancy structure
                    system_params,
                    temperature=entropy_temp
                )
                
                # Store calculated entropy
                calculator.results['formation_entropy'] = S_f
                calculator.results['calculated_formation_entropy'] = S_f
                
                # Migration entropy is ~0 for vacancy mechanism
                S_m = 0.0
                calculator.results['migration_entropy'] = S_m
                
                logging.info(f"\nEntropy results:")
                logging.info(f"  Formation entropy S_f: {S_f:.3f} k_B")
                logging.info(f"  Migration entropy S_m: {S_m:.3f} k_B (set to zero)")
        
        # Step 5: Estimate attempt frequency
        calculator.estimate_attempt_frequency()
        
        # Step 6: Calculate diffusion coefficient over temperature range
        T_min = system_params.get('temperature_min', 300.0)
        T_max = system_params.get('temperature_max', 2000.0)
        n_points = system_params.get('temperature_points', 50)
        
        calculator.calculate_diffusion_over_temperature_range(T_min, T_max, n_points)
        
        # Step 7: Plot energy profile
        calculator.plot_energy_profile()
        
        # Step 8: Plot Arrhenius plot
        calculator.plot_arrhenius()
        
        # Step 9: Save results
        calculator.save_results()
        
        logging.info("Self-diffusion calculation completed successfully")
        
        return calculator.results
        
    except Exception as e:
        logging.error(f"Error during self-diffusion calculation: {e}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Calculate self-diffusion using dimer method',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument('--poscar-file', default='POSCAR',
                        help='Initial POSCAR structure (with defect)')
    
    # Physical parameters
    parser.add_argument('--activation-radius', type=float, default=10.0,
                        help='Activation radius (Å) for VASPInterface')
    parser.add_argument('--moving-indices', type=int, nargs='+', default=[0],
                        help='Atom indices for VASPInterface tracking')
    parser.add_argument('--temperature-min', type=float, default=300.0,
                        help='Minimum temperature for Arrhenius plot (K)')
    parser.add_argument('--temperature-max', type=float, default=2000.0,
                        help='Maximum temperature for Arrhenius plot (K)')
    parser.add_argument('--temperature-points', type=int, default=50,
                        help='Number of temperature points for Arrhenius plot')
    
    # Minimization parameters
    parser.add_argument('--min-method', default='fire',
                        choices=['steepest', 'cg', 'lbfgs', 'lbfgs_scipy', 'bfgs', 'fire'],
                        help='Minimization method')
    parser.add_argument('--min-max-steps', type=int, default=100,
                        help='Max steps for minimization')
    parser.add_argument('--min-step-size', type=float, default=0.05,
                        help='Step size for minimization')
    parser.add_argument('--min-max-step-size', type=float, default=0.1,
                        help='Max step size for minimization')
    parser.add_argument('--min-stopping-criteria', type=float, default=0.005,
                        help='Force convergence for minimization (eV/Å)')
    
    # Dimer parameters
    parser.add_argument('--dimer-max-steps', type=int, default=100,
                        help='Max steps for dimer search')
    parser.add_argument('--dimer-rotation', default='lbfgsext',
                        help='Dimer rotation method')
    parser.add_argument('--dimer-translation', default='lbfgs',
                        help='Dimer translation method')
    parser.add_argument('--dimer-sep', type=float, default=0.01,
                        help='Dimer separation')
    parser.add_argument('--dimer-step-size', type=float, default=0.02,
                        help='Dimer step size')
    parser.add_argument('--dimer-max-step-size', type=float, default=0.05,
                        help='Dimer max step size')
    parser.add_argument('--dimer-stopping-criteria', type=float, default=0.01,
                        help='Force convergence for dimer (eV/Å)')
    
    # Thermodynamic parameters
    parser.add_argument('--perfect-structure', default=None,
                        help='POSCAR file for perfect structure (to calculate formation energy)')
    parser.add_argument('--structure-type', choices=['fcc', 'bcc', 'sc', 'hcp'],
                        default='bcc', help='Crystal structure type for correlation factor')
    parser.add_argument('--calculate-entropy', action='store_true',
                        help='Calculate formation entropy from phonon frequencies')
    parser.add_argument('--entropy-temperature', type=float, default=300.0,
                        help='Temperature for entropy calculation (K)')
    parser.add_argument('--calculate-tst-frequency', action='store_true',
                        help='Calculate effective frequency using TST formula')
    
    # Energy corrections
    parser.add_argument('--formation-correction', type=float, default=0.0,
                        help='Correction to add to formation energy (eV)')
    parser.add_argument('--migration-correction', type=float, default=0.0,
                        help='Correction to add to migration/barrier energy (eV)')
    
    # Phonopy parameters for entropy calculation
    parser.add_argument('--phonopy-dim', type=int, nargs=3, default=[2, 2, 2],
                        help='Supercell dimensions for phonopy')
    parser.add_argument('--phonopy-mp', type=int, nargs=3, default=[10, 10, 10],
                        help='MP grid for phonopy DOS')
    
    # Mode following
    parser.add_argument('--mode-following-displacement', type=float, default=0.1,
                        help='Displacement along unstable mode (Å)')
    
    # Execution mode
    parser.add_argument('--execution-mode',
                        choices=['mock', 'mpi', 'slurm', 'eam'],
                        default='mpi',
                        help='Execution mode')
    parser.add_argument('--mpi-command', default=None,
                        help='Custom MPI command')
    parser.add_argument('--vasp-command', default='vasp_gam',
                        help='VASP executable')
    parser.add_argument('--eam-potential-file', default=None,
                        help='EAM potential file')
    parser.add_argument('--kim-model', default=None,
                        help='KIM model name')
    
    # Misc
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--continuation', action='store_true',
                        help='Continue from previous run')
    
    # Output management
    parser.add_argument('--output-dir', default=None,
                        help='Base directory for all outputs (default: ./outputs)')
    parser.add_argument('--run-name', default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--input-dir', default=None,
                        help='Directory containing input files (default: ./inputs)')
    
    args = parser.parse_args()
    
    # Build system parameters
    system_params = {
        'poscar_file': args.poscar_file,
        'activation_radius': args.activation_radius,
        'moving_indices': args.moving_indices,
        'temperature_min': args.temperature_min,
        'temperature_max': args.temperature_max,
        'temperature_points': args.temperature_points,
        
        # Minimization
        'min_method': args.min_method,
        'min_max_steps': args.min_max_steps,
        'min_step_size': args.min_step_size,
        'min_max_step_size': args.min_max_step_size,
        'min_stopping_criteria': args.min_stopping_criteria,
        
        # Dimer
        'dimer_max_steps': args.dimer_max_steps,
        'dimer_rotation': args.dimer_rotation,
        'dimer_translation': args.dimer_translation,
        'dimer_sep': args.dimer_sep,
        'dimer_step_size': args.dimer_step_size,
        'dimer_max_step_size': args.dimer_max_step_size,
        'dimer_stopping_criteria': args.dimer_stopping_criteria,
        
        # Mode following
        'mode_following_displacement': args.mode_following_displacement,
        
        # Thermodynamic parameters
        'perfect_structure': args.perfect_structure,
        'structure_type': args.structure_type,
        'calculate_entropy': args.calculate_entropy,
        'entropy_temperature': args.entropy_temperature,
        'calculate_tst_frequency': args.calculate_tst_frequency,
        'formation_correction': args.formation_correction,
        'migration_correction': args.migration_correction,
        'phonopy_dim': args.phonopy_dim,
        'phonopy_mp': args.phonopy_mp,
        
        # Execution
        'execution_mode': args.execution_mode,
        'mpi_command': args.mpi_command,
        'vasp_command': args.vasp_command,
        'eam_potential_file': args.eam_potential_file,
        'kim_model': args.kim_model,
        
        # Misc
        'verbose': args.verbose,
        'continuation': args.continuation,
        
        # Output management
        'output_dir': args.output_dir,
        'run_name': args.run_name,
        'input_dir': args.input_dir
    }
    
    return system_params


if __name__ == "__main__":
    try:
        system_params = parse_arguments()
        
        # Initialize output directory structure
        OutputManager.setup(
            base_dir=system_params.get('output_dir', None),
            run_name=system_params.get('run_name', None),
            input_dir=system_params.get('input_dir', None)
        )
        
        # Save run metadata
        OutputManager.save_run_metadata({
            'script': 'calculate_self_diffusion.py',
            'parameters': system_params,
            'poscar_file': system_params['poscar_file']
        })
        
        print("\nSelf-Diffusion Calculation Parameters:")
        print("="*50)
        for key, value in system_params.items():
            if 'method' in key or 'criteria' in key or 'step' in key:
                print(f"  {key}: {value}")
        print("="*50 + "\n")
        
        results = run_self_diffusion_calculation(
            system_params['poscar_file'],
            system_params
        )
        
        print("\n" + "="*60)
        print("SELF-DIFFUSION CALCULATION COMPLETE")
        print("="*60)
        
        if results['activation_energy'] is not None:
            if 'activation_energy_raw' in results:
                print(f"Forward activation energy (raw): {results['activation_energy_raw']:.4f} eV")
                if results.get('migration_correction', 0.0) != 0.0:
                    print(f"Migration correction: +{results['migration_correction']:.4f} eV")
                    print(f"Forward activation energy (corrected): {results['activation_energy']:.4f} eV")
            else:
                print(f"Forward activation energy: {results['activation_energy']:.4f} eV")
            
        if 'formation_energy' in results:
            if 'formation_energy_raw' in results:
                print(f"Formation energy (raw): {results['formation_energy_raw']:.4f} eV")
                if results.get('formation_correction', 0.0) != 0.0:
                    print(f"Formation correction: +{results['formation_correction']:.4f} eV")
                    print(f"Formation energy (corrected): {results['formation_energy']:.4f} eV")
            else:
                print(f"Formation energy: {results['formation_energy']:.4f} eV")
            print(f"Formation entropy: {results.get('formation_entropy', 0.0):.3f} k_B")
            print(f"Migration entropy: {results.get('migration_entropy', 0.0):.3f} k_B")
            
        if 'temperature_range' in results:
            T_data = results['temperature_range']['temperatures']
            D_data = results['temperature_range']['diffusion_coefficients']
            print(f"\nDiffusion coefficients calculated from {T_data[0]:.0f} K to {T_data[-1]:.0f} K")
            print(f"D at {T_data[0]:.0f} K: {D_data[0]:.2e} m²/s")
            print(f"D at {T_data[-1]:.0f} K: {D_data[-1]:.2e} m²/s")
            
        print(f"\nResults saved in {get_output_path('results')} directory:")
        print("  - self_diffusion_summary.txt: Full summary")
        print("  - diffusion_vs_temperature.csv: Temperature-dependent data")
        print("  - diffusion_arrhenius_plot.png: Arrhenius plot")
        print("  - diffusion_energy_profile.png: Energy barrier profile")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()