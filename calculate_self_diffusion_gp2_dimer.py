#!/usr/bin/env python
"""Calculate self-diffusion using GP2-accelerated dimer method and local minimization.

This script:
1. Takes an initial configuration (possibly with a defect)
2. Finds the local minimum
3. Uses GP2-accelerated dimer method to find saddle point
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
from walker_gp2_dimer import WalkerGP2Dimer
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
    
    log_file = get_output_path('logs', 'self_diffusion_gp2_calculation.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info("Logging system initialized")


class SelfDiffusionGP2Calculator:
    """Calculate self-diffusion barriers and rates using GP2-accelerated dimer."""
    
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
            'trajectories': {},
            'gp2_statistics': {}  # Store GP2-specific statistics
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
        """Find saddle point starting from initial minimum using GP2-accelerated dimer."""
        logging.info("Finding saddle point using GP2-accelerated dimer method...")
        
        # Create GP2 dimer walker
        walker = WalkerGP2Dimer(
            initial_position=initial_pos,
            local_pes=local_pes,
            max_dimer_steps=self.system_params.get('dimer_max_steps', 100),
            # Stopping criteria
            disp_max=self.system_params.get('gp2_disp_max', 0.5),
            ratio_at_limit=self.system_params.get('gp2_ratio_at_limit', 2.0/3.0),
            # Dimer parameters
            rotation=self.system_params.get('dimer_rotation', 'lbfgsext'),
            translation=self.system_params.get('dimer_translation', 'lbfgs'),
            dimer_sep=self.system_params.get('dimer_sep', 0.01),
            T_anglerot=self.system_params.get('dimer_T_anglerot', 0.01),
            T_anglerot_init=self.system_params.get('dimer_T_anglerot_init', 0.0873),
            T_anglerot_gp=self.system_params.get('gp2_T_anglerot_gp', 0.01),
            max_dimer_rotations=self.system_params.get('dimer_max_rotations', 10),
            num_init_rotations=self.system_params.get('dimer_num_init_rotations', 5),
            num_iter_rot_gp=self.system_params.get('gp2_num_iter_rot_gp', 10),
            dimer_stopping_criteria=self.system_params.get('dimer_stopping_criteria', 0.01),
            step_size=self.system_params.get('dimer_step_size', 0.1),
            max_step_size=self.system_params.get('dimer_max_step_size', 0.1),
            # GP convergence
            divisor_T_dimer_gp=self.system_params.get('gp2_divisor_T_dimer_gp', 10.0),
            max_inner_iterations=self.system_params.get('gp2_max_inner_iterations', 1000),
            # Options
            initrot_nogp=self.system_params.get('gp2_initrot_nogp', False),
            inittrans_nogp=self.system_params.get('gp2_inittrans_nogp', False),
            eval_image1=self.system_params.get('gp2_eval_image1', False),
            num_bigiter_initloc=self.system_params.get('gp2_num_bigiter_initloc', np.inf),
            num_bigiter_initparam=self.system_params.get('gp2_num_bigiter_initparam', np.inf),
            # Other parameters
            verbose=self.system_params.get('verbose', False),
            checkpoint_interval=10,  # Less frequent checkpointing
            model_type=self.system_params.get('gp2_model_type', 'MultitaskGPModel_rbf_atomic')
        )
        
        # Set initial dimer orientation if specified
        initial_orient_method = self.system_params.get('initial_orient_method', 'auto')
        if initial_orient_method == 'manual' and 'dimer_initial_direction' in self.system_params:
            direction = np.array(self.system_params['dimer_initial_direction'])
            walker.set_initial_orientation(direction)
        elif initial_orient_method == 'random':
            # Let the walker use random initialization
            logging.info("Using random initial dimer orientation")
            # The walker will handle this internally
        # 'auto' will be handled by the walker based on forces
        
        # Store walker reference for later use
        self._last_dimer_walker = walker
        
        # Run GP2-accelerated dimer search
        final_pos, final_energy, final_forces = walker.run()
        
        if walker.converged:
            logging.info(f"Saddle point found: E = {final_energy:.6f} eV")
            # Check if it's actually a saddle point by looking at curvature
            if hasattr(walker, 'table_history') and len(walker.table_history) > 0:
                last_entry = walker.table_history[-1]
                if 'Curvature_dimer' in last_entry and last_entry['Curvature_dimer'] < 0:
                    logging.info(f"Negative curvature confirmed: {last_entry['Curvature_dimer']:.4f}")
                else:
                    logging.warning("Curvature check suggests this might not be a saddle point")
        else:
            logging.warning("GP2 dimer search did not fully converge")
            
        # Extract GP2 statistics
        self.results['gp2_statistics'] = {
            'outer_iterations': walker.bigiter,
            'total_observations': walker.obs_total,
            'initial_rotation_observations': walker.obs_initrot,
            'inner_iterations_total': len(walker.E_R_gp),
            'stopping_statistics': {
                'max_iterations': walker.num_esmax,
                'interatomic_distance': walker.num_es1,
                'displacement': walker.num_es2
            }
        }
        
        # Calculate speedup factor
        if walker.obs_total > walker.obs_initrot:
            main_obs = walker.obs_total - walker.obs_initrot
            inner_iters = len(walker.E_R_gp)
            if main_obs > 0:
                speedup = inner_iters / main_obs
                self.results['gp2_statistics']['speedup_factor'] = speedup
                logging.info(f"GP2 speedup factor: {speedup:.1f}x")
            
        # Optionally use GP2 energy instead of actual energy
        if self.system_params.get('use_gp2_energy_for_saddle', False) and walker.gp2_trained:
            # Get GP2 prediction at final position
            final_energy_gp2, _ = walker._gp_evaluate(final_pos)
            logging.info(f"Using GP2 predicted energy for saddle: {final_energy_gp2:.6f} eV (actual: {final_energy:.6f} eV)")
            energy_for_calculation = final_energy_gp2
        else:
            energy_for_calculation = final_energy
            logging.info(f"Using actual energy for saddle: {final_energy:.6f} eV")
        
        self.results['saddle_point'] = {
            'position': final_pos,
            'energy': energy_for_calculation,  # Use selected energy type
            'energy_actual': final_energy,  # Always store actual energy
            'energy_gp2': final_energy_gp2 if 'final_energy_gp2' in locals() else None,
            'forces': final_forces,
            'converged': walker.converged,
            'trajectory': [(walker.R_all[i], walker.E_all[i,0], walker.G_all[i]) for i in range(len(walker.R_all))],
            'curvature': last_entry.get('Curvature_dimer', None) if 'last_entry' in locals() else None,
            'orientation': walker.orient.copy() if walker.orient is not None else None  # Store just the orientation
        }
        
        # Save saddle point structure
        structure = Poscar.from_file(self.system_params['poscar_file'], check_for_potcar=False).structure
        self._save_structure(final_pos, structure, 'saddle_point.vasp')
        
        return final_pos, final_energy, final_forces
    
    def find_final_minimum(self, saddle_pos: np.ndarray, local_pes) -> Tuple[np.ndarray, float, np.ndarray]:
        """Find final minimum by following mode from saddle point."""
        logging.info("Finding final minimum from saddle point...")
        
        # Get the dimer walker that found the saddle point
        displacement = self.system_params.get('mode_following_displacement', 0.1)
        
        # Try to get the dimer orientation from the stored saddle point data
        if 'saddle_point' in self.results and self.results['saddle_point'].get('orientation') is not None:
            # GP2 walker stores orient as 2D array, so flatten it
            unstable_mode = self.results['saddle_point']['orientation'].ravel()
            logging.info(f"Using stored GP2 dimer orientation as unstable mode")
            
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
                
                # Import and use the method from the original calculator
                from calculate_self_diffusion import SelfDiffusionCalculator
                temp_calc = SelfDiffusionCalculator(self.system_params)
                nu_tst = temp_calc.calculate_effective_frequency_tst(
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
    
    def calculate_jump_distance_crystallographic(self, structure, jump_type='nn'):
        """Calculate crystallographic jump distance."""
        # Import from original
        from calculate_self_diffusion import SelfDiffusionCalculator
        temp_calc = SelfDiffusionCalculator(self.system_params)
        return temp_calc.calculate_jump_distance_crystallographic(structure, jump_type)
    
    def calculate_formation_energy(self, perfect_structure_file: str, system_params_orig: dict) -> float:
        """Calculate vacancy formation energy."""
        # Import from original and use its method
        from calculate_self_diffusion import SelfDiffusionCalculator
        temp_calc = SelfDiffusionCalculator(self.system_params)
        temp_calc.results = self.results  # Share results
        E_formation = temp_calc.calculate_formation_energy(perfect_structure_file, system_params_orig)
        
        # Copy back results
        self.results.update(temp_calc.results)
        return E_formation
    
    def calculate_vibrational_entropy_total(self, structure_file: str, calc_type: str, 
                                          system_params_orig: dict, temperature: float = 300.0) -> float:
        """Calculate TOTAL vibrational entropy using phonopy."""
        # Import from original
        from calculate_self_diffusion import SelfDiffusionCalculator
        temp_calc = SelfDiffusionCalculator(self.system_params)
        return temp_calc.calculate_vibrational_entropy_total(structure_file, calc_type, 
                                                           system_params_orig, temperature)
    
    def calculate_formation_entropy(self, perfect_struct_file: str, 
                                  vacancy_struct_file: str,
                                  system_params_orig: dict,
                                  temperature: float = 300.0) -> float:
        """Calculate formation entropy from phonon calculations."""
        # Import from original
        from calculate_self_diffusion import SelfDiffusionCalculator
        temp_calc = SelfDiffusionCalculator(self.system_params)
        return temp_calc.calculate_formation_entropy(perfect_struct_file, vacancy_struct_file,
                                                   system_params_orig, temperature)
    
    def calculate_diffusion_coefficient_complete(self, temperature: float = 300.0,
                                               S_f: float = None, S_m: float = None) -> float:
        """Calculate diffusion coefficient with full vacancy thermodynamics."""
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
        
        # For vacancy diffusion, we use the crystallographic jump distance
        # since we don't track the final position
        if d_crystal is not None:
            d = d_crystal
            self.results['jump_distance'] = d_crystal
            logging.info(f"  Using crystallographic jump distance: {d:.3f} Å")
        else:
            # Default to typical BCC nearest neighbor distance
            d = 2.8  # Typical for many metals
            self.results['jump_distance'] = d
            logging.warning(f"  Using default jump distance: {d:.3f} Å")
        
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
        """Calculate diffusion coefficient using simple migration-only model."""
        if self.results['activation_energy'] is None:
            raise ValueError("Activation energy not calculated")
            
        # Constants
        k_B = constants.k / constants.e  # Boltzmann constant in eV/K
        
        # Get structural info
        structure = Poscar.from_file(self.system_params['poscar_file'], check_for_potcar=False).structure
        
        # Use crystallographic jump distance for vacancy diffusion
        jump_distance = self.calculate_jump_distance_crystallographic(structure)
        if jump_distance is None:
            # Default to typical value
            jump_distance = 2.8  # Å, typical for many metals
            logging.warning(f"Using default jump distance: {jump_distance:.3f} Å")
        
        # Only log once
        if 'jump_distance' not in self.results:
            logging.info(f"Jump distance (crystallographic): {jump_distance:.3f} Å")
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
        """Calculate diffusion coefficients over a temperature range."""
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
        # Import from original
        from calculate_self_diffusion import SelfDiffusionCalculator
        temp_calc = SelfDiffusionCalculator(self.system_params)
        temp_calc.results = self.results  # Share results
        temp_calc.plot_arrhenius()
    
    def plot_energy_profile(self):
        """Plot the energy profile along the diffusion path."""
        if not all(key in self.results for key in ['initial_minimum', 'saddle_point']):
            logging.warning("Cannot plot: missing energy calculations")
            return
            
        # Create simple reaction coordinate for initial -> saddle
        energies = [
            self.results['initial_minimum']['energy'],
            self.results['saddle_point']['energy']
        ]
        
        labels = ['Initial', 'Saddle']
        x = [0, 1]
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, energies, 'o-', markersize=10, linewidth=2)
        
        # Add forward activation energy arrow
        E_a_forward = self.results['activation_energy']
        
        plt.annotate('', xy=(1, energies[1]), xytext=(0, energies[0]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        plt.text(0.5, (energies[0] + energies[1])/2, f'{E_a_forward:.3f} eV',
                ha='center', va='bottom', color='red', fontsize=12, fontweight='bold')
        
        plt.xlabel('Reaction Coordinate', fontsize=12)
        plt.ylabel('Energy (eV)', fontsize=12)
        plt.title('Self-Diffusion Energy Profile', fontsize=14)
        plt.xticks(x, labels)
        plt.grid(True, alpha=0.3)
        
        # Add a text box with energy values
        textstr = f'E_initial = {energies[0]:.3f} eV\nE_saddle = {energies[1]:.3f} eV\nE_a = {E_a_forward:.3f} eV'
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
        with open(os.path.join(results_dir, 'self_diffusion_gp2_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
            
        # Save human-readable summary
        with open(os.path.join(results_dir, 'self_diffusion_gp2_summary.txt'), 'w') as f:
            f.write("SELF-DIFFUSION CALCULATION RESULTS (GP2-ACCELERATED DIMER)\n")
            f.write("="*60 + "\n\n")
            
            if self.results['initial_minimum']:
                f.write(f"Initial Minimum Energy: {self.results['initial_minimum']['energy']:.6f} eV\n")
                f.write(f"Initial Converged: {self.results['initial_minimum']['converged']}\n\n")
                
            if self.results['saddle_point']:
                f.write(f"Saddle Point Energy: {self.results['saddle_point']['energy']:.6f} eV\n")
                f.write(f"Saddle Converged: {self.results['saddle_point']['converged']}\n")
                if self.results['saddle_point']['curvature'] is not None:
                    f.write(f"Curvature at Saddle: {self.results['saddle_point']['curvature']:.4f}\n\n")
                else:
                    f.write("\n")
                
            if self.results['activation_energy'] is not None:
                f.write(f"Forward Activation Energy (raw): {self.results.get('activation_energy_raw', self.results['activation_energy']):.4f} eV\n")
                if self.results.get('migration_correction', 0.0) != 0.0:
                    f.write(f"Migration Correction: {self.results['migration_correction']:.4f} eV\n")
                    f.write(f"Forward Activation Energy (corrected): {self.results['activation_energy']:.4f} eV\n\n")
                else:
                    f.write("\n")
                
            # GP2 statistics
            if 'gp2_statistics' in self.results:
                f.write("GP2 DIMER STATISTICS\n")
                f.write("-"*30 + "\n")
                stats = self.results['gp2_statistics']
                f.write(f"Outer iterations: {stats.get('outer_iterations', 'N/A')}\n")
                f.write(f"Total observations: {stats.get('total_observations', 'N/A')}\n")
                f.write(f"Initial rotation observations: {stats.get('initial_rotation_observations', 'N/A')}\n")
                f.write(f"Inner iterations total: {stats.get('inner_iterations_total', 'N/A')}\n")
                if 'speedup_factor' in stats:
                    f.write(f"GP2 speedup factor: {stats['speedup_factor']:.1f}x\n")
                
                stop_stats = stats.get('stopping_statistics', {})
                f.write(f"\nStopping statistics:\n")
                f.write(f"  Max iterations reached: {stop_stats.get('max_iterations', 0)}\n")
                f.write(f"  Inter-atomic distance limit: {stop_stats.get('interatomic_distance', 0)}\n")
                f.write(f"  Displacement limit: {stop_stats.get('displacement', 0)}\n\n")
                
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
            with open(os.path.join(results_dir, 'diffusion_vs_temperature_gp2.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Temperature (K)', 'Diffusion Coefficient (m²/s)', 
                               '1000/T (K⁻¹)', 'log10(D)'])
                
                T_data = self.results['temperature_range']['temperatures']
                D_data = self.results['temperature_range']['diffusion_coefficients']
                
                for T, D in zip(T_data, D_data):
                    writer.writerow([T, D, 1000.0/T, np.log10(D)])
                    
            logging.info(f"Temperature-dependent data saved to {os.path.join(results_dir, 'diffusion_vs_temperature_gp2.csv')}")
                
        logging.info(f"Results saved to {os.path.join(results_dir, 'self_diffusion_gp2_results.pkl')} and {os.path.join(results_dir, 'self_diffusion_gp2_summary.txt')}")


def run_self_diffusion_gp2_calculation(poscar_file: str, system_params: dict):
    """Run complete self-diffusion calculation with GP2-accelerated dimer."""
    
    # Setup
    setup_logging()
    logging.info("Starting self-diffusion calculation with GP2-accelerated dimer")
    
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
    calculator = SelfDiffusionGP2Calculator(system_params)
    
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
        
        # Step 2: Find saddle point using GP2-accelerated dimer
        saddle_pos, saddle_E, saddle_F = calculator.find_saddle_point(initial_pos, local_pes)
        
        # Step 3: Calculate activation energies (skip final minimum)
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
                
                # Use the RELAXED structures for entropy calculation
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
        
        logging.info("Self-diffusion calculation with GP2-accelerated dimer completed successfully")
        
        return calculator.results
        
    except Exception as e:
        logging.error(f"Error during self-diffusion calculation: {e}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Calculate self-diffusion using GP2-accelerated dimer method',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument('--poscar-file', default='POSCAR',
                        help='Initial POSCAR structure (with defect, looked for in inputs/ directory)')
    
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
    
    # GP2 Dimer parameters
    parser.add_argument('--dimer-max-steps', type=int, default=100,
                        help='Max outer iterations for GP2 dimer search')
    parser.add_argument('--gp2-disp-max', type=float, default=0.5,
                        help='Maximum displacement from nearest observed point (Å)')
    parser.add_argument('--gp2-ratio-at-limit', type=float, default=2.0/3.0,
                        help='Limit for inter-atomic distance ratio')
    parser.add_argument('--dimer-rotation', default='lbfgsext',
                        help='Dimer rotation method')
    parser.add_argument('--dimer-translation', default='lbfgs',
                        help='Dimer translation method')
    parser.add_argument('--dimer-sep', type=float, default=0.01,
                        help='Dimer separation')
    parser.add_argument('--dimer-T-anglerot', type=float, default=0.01,
                        help='Rotation convergence threshold')
    parser.add_argument('--dimer-T-anglerot-init', type=float, default=0.0873,
                        help='Initial rotation convergence threshold')
    parser.add_argument('--gp2-T-anglerot-gp', type=float, default=0.01,
                        help='Rotation convergence on GP surface')
    parser.add_argument('--dimer-max-rotations', type=int, default=10,
                        help='Maximum rotations per translation')
    parser.add_argument('--dimer-num-init-rotations', type=int, default=5,
                        help='Number of initial rotations')
    parser.add_argument('--gp2-num-iter-rot-gp', type=int, default=10,
                        help='Max rotations per translation on GP')
    parser.add_argument('--dimer-step-size', type=float, default=0.1,
                        help='Dimer step size')
    parser.add_argument('--dimer-max-step-size', type=float, default=0.1,
                        help='Dimer max step size')
    parser.add_argument('--dimer-stopping-criteria', type=float, default=0.01,
                        help='Force convergence for dimer (eV/Å)')
    
    # GP2 specific parameters
    parser.add_argument('--gp2-divisor-T-dimer-gp', type=float, default=10.0,
                        help='Divisor for dynamic GP convergence threshold')
    parser.add_argument('--gp2-max-inner-iterations', type=int, default=1000,
                        help='Maximum iterations per relaxation phase')
    parser.add_argument('--gp2-initrot-nogp', action='store_true',
                        help='Perform initial rotations without GP')
    parser.add_argument('--gp2-inittrans-nogp', action='store_true',
                        help='Perform initial translation without GP')
    parser.add_argument('--gp2-eval-image1', action='store_true',
                        help='Evaluate image 1 after each phase')
    parser.add_argument('--gp2-num-bigiter-initloc', type=float, default=np.inf,
                        help='Number of iterations from initial location')
    parser.add_argument('--gp2-num-bigiter-initparam', type=float, default=np.inf,
                        help='Number of iterations with fresh hyperparameters')
    parser.add_argument('--gp2-model-type', 
                        choices=['MultitaskGPModel_rbf_atomic',
                                 'BatchIndependentMultitaskGPModel_rbf',
                                 'GPModelWithDerivatives_rbf_atomic'],
                        default='MultitaskGPModel_rbf_atomic',
                        help='GP model type')
    
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
    
    # Initial orientation control
    parser.add_argument('--initial-orient-method', 
                        choices=['auto', 'random', 'manual'],
                        default='auto',
                        help='Method for setting initial dimer orientation')
    parser.add_argument('--manual-orient', type=float, nargs='+',
                        help='Manual initial orientation vector (must be same dimension as system)')
    parser.add_argument('--orient-atom-direction', type=str,
                        help='Atom index and direction, e.g., "52:-1,-1,-1"')
    
    # Nearest neighbor options
    parser.add_argument('--include-neighbors', 
                        choices=['none', '1nn', '2nn', '1nn+2nn'],
                        default='none',
                        help='Include nearest neighbors in moving atoms')
    parser.add_argument('--nn-cutoff-1', type=float, default=None,
                        help='Cutoff distance for 1st nearest neighbors (Å)')
    parser.add_argument('--nn-cutoff-2', type=float, default=None,
                        help='Cutoff distance for 2nd nearest neighbors (Å)')
    
    # Energy source selection
    parser.add_argument('--use-gp2-energy-for-saddle', action='store_true',
                        help='Use GP2 predicted energy instead of actual energy for saddle point (default: use actual)')
    
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
    
    # Handle input file path
    poscar_file_for_parsing = args.poscar_file
    if not os.path.isabs(poscar_file_for_parsing):
        # Check if file exists in inputs directory
        # First try relative to current directory
        input_path = os.path.join('inputs', poscar_file_for_parsing) if args.input_dir is None else os.path.join(args.input_dir, poscar_file_for_parsing)
        
        # If not found and we're in scripts/, try parent directory
        if not os.path.exists(input_path) and os.path.basename(os.getcwd()) == 'scripts':
            parent_input_path = os.path.join('..', 'inputs', poscar_file_for_parsing)
            if os.path.exists(parent_input_path):
                input_path = parent_input_path
        
        if os.path.exists(input_path):
            poscar_file_for_parsing = input_path
        elif not os.path.exists(poscar_file_for_parsing):
            # If not found in inputs/ and not in current dir, raise error
            print(f"Warning: POSCAR file '{poscar_file_for_parsing}' not found in inputs/ directory or current directory")
    
    # Load structure for processing
    structure = Poscar.from_file(poscar_file_for_parsing, check_for_potcar=False).structure
    
    # Process moving indices with neighbors if requested
    moving_indices = args.moving_indices
    if args.include_neighbors != 'none':
        # Import the find_nearest_neighbors function from run_gp2_dimer
        from run_gp2_dimer import find_nearest_neighbors
        
        print(f"\nFinding nearest neighbors for atoms: {moving_indices}")
        
        neighbors = find_nearest_neighbors(
            structure, 
            moving_indices,
            cutoff_1nn=args.nn_cutoff_1,
            cutoff_2nn=args.nn_cutoff_2,
            auto_detect=True
        )
        
        # Add neighbors based on user choice
        if args.include_neighbors in ['1nn', '1nn+2nn']:
            moving_indices.extend(neighbors['first_neighbors'])
            print(f"Added {len(neighbors['first_neighbors'])} first nearest neighbors: {neighbors['first_neighbors']}")
        
        if args.include_neighbors in ['2nn', '1nn+2nn']:
            moving_indices.extend(neighbors['second_neighbors'])
            print(f"Added {len(neighbors['second_neighbors'])} second nearest neighbors: {neighbors['second_neighbors']}")
        
        # Remove duplicates and sort
        moving_indices = sorted(list(set(moving_indices)))
        
        print(f"\nFinal moving atoms ({len(moving_indices)} total): {moving_indices}")
    
    # Process initial orientation
    manual_orient = None
    initial_orient_method = args.initial_orient_method
    
    # Handle orient-atom-direction (overrides other methods)
    if args.orient_atom_direction:
        parts = args.orient_atom_direction.split(':')
        atom_idx = int(parts[0])
        direction = list(map(float, parts[1].split(',')))
        
        n_atoms = structure.num_sites
        
        # Create full orientation vector
        manual_orient = np.zeros(n_atoms * 3)
        manual_orient[atom_idx * 3] = direction[0]
        manual_orient[atom_idx * 3 + 1] = direction[1]
        manual_orient[atom_idx * 3 + 2] = direction[2]
        
        # Normalize
        norm = np.linalg.norm(manual_orient)
        if norm > 0:
            manual_orient = manual_orient / norm
        
        initial_orient_method = 'manual'
        print(f"\nSet initial dimer orientation:")
        print(f"  Atom {atom_idx} displaced in direction {direction}")
    
    # Handle manual-orient
    elif args.manual_orient is not None:
        manual_orient = np.array(args.manual_orient)
        norm = np.linalg.norm(manual_orient)
        if norm > 0:
            manual_orient = manual_orient / norm
        initial_orient_method = 'manual'
        print(f"\nUsing manual initial orientation vector")
    
    # Build system parameters
    system_params = {
        'poscar_file': args.poscar_file,
        'activation_radius': args.activation_radius,
        'moving_indices': moving_indices,  # Use the processed moving_indices
        'temperature_min': args.temperature_min,
        'temperature_max': args.temperature_max,
        'temperature_points': args.temperature_points,
        
        # Minimization
        'min_method': args.min_method,
        'min_max_steps': args.min_max_steps,
        'min_step_size': args.min_step_size,
        'min_max_step_size': args.min_max_step_size,
        'min_stopping_criteria': args.min_stopping_criteria,
        
        # GP2 Dimer
        'dimer_max_steps': args.dimer_max_steps,
        'gp2_disp_max': args.gp2_disp_max,
        'gp2_ratio_at_limit': args.gp2_ratio_at_limit,
        'dimer_rotation': args.dimer_rotation,
        'dimer_translation': args.dimer_translation,
        'dimer_sep': args.dimer_sep,
        'dimer_T_anglerot': args.dimer_T_anglerot,
        'dimer_T_anglerot_init': args.dimer_T_anglerot_init,
        'gp2_T_anglerot_gp': args.gp2_T_anglerot_gp,
        'dimer_max_rotations': args.dimer_max_rotations,
        'dimer_num_init_rotations': args.dimer_num_init_rotations,
        'gp2_num_iter_rot_gp': args.gp2_num_iter_rot_gp,
        'dimer_step_size': args.dimer_step_size,
        'dimer_max_step_size': args.dimer_max_step_size,
        'dimer_stopping_criteria': args.dimer_stopping_criteria,
        'gp2_divisor_T_dimer_gp': args.gp2_divisor_T_dimer_gp,
        'gp2_max_inner_iterations': args.gp2_max_inner_iterations,
        'gp2_initrot_nogp': args.gp2_initrot_nogp,
        'gp2_inittrans_nogp': args.gp2_inittrans_nogp,
        'gp2_eval_image1': args.gp2_eval_image1,
        'gp2_num_bigiter_initloc': args.gp2_num_bigiter_initloc,
        'gp2_num_bigiter_initparam': args.gp2_num_bigiter_initparam,
        'gp2_model_type': args.gp2_model_type,
        
        # Mode following
        'mode_following_displacement': args.mode_following_displacement,
        'dimer_initial_direction': manual_orient.tolist() if manual_orient is not None else None,
        'initial_orient_method': initial_orient_method,
        
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
        'use_gp2_energy_for_saddle': args.use_gp2_energy_for_saddle,
        
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
            'script': 'calculate_self_diffusion_gp2_dimer.py',
            'parameters': system_params,
            'poscar_file': system_params['poscar_file']
        })
        
        print("\nSelf-Diffusion Calculation Parameters (GP2-Accelerated Dimer):")
        print("="*60)
        print("GP2 Dimer Settings:")
        print(f"  Max outer iterations: {system_params['dimer_max_steps']}")
        print(f"  Max inner iterations: {system_params['gp2_max_inner_iterations']}")
        print(f"  Displacement limit: {system_params['gp2_disp_max']} Å")
        print(f"  Inter-atomic ratio limit: {system_params['gp2_ratio_at_limit']}")
        print(f"  GP convergence divisor: {system_params['gp2_divisor_T_dimer_gp']}")
        print(f"  Model type: {system_params['gp2_model_type']}")
        print("="*60 + "\n")
        
        results = run_self_diffusion_gp2_calculation(
            system_params['poscar_file'],
            system_params
        )
        
        print("\n" + "="*60)
        print("SELF-DIFFUSION CALCULATION COMPLETE (GP2-ACCELERATED)")
        print("="*60)
        
        if results['activation_energy'] is not None:
            if 'activation_energy_raw' in results:
                print(f"Forward activation energy (raw): {results['activation_energy_raw']:.4f} eV")
                if results.get('migration_correction', 0.0) != 0.0:
                    print(f"Migration correction: +{results['migration_correction']:.4f} eV")
                    print(f"Forward activation energy (corrected): {results['activation_energy']:.4f} eV")
            else:
                print(f"Forward activation energy: {results['activation_energy']:.4f} eV")
        
        # Print GP2 statistics
        if 'gp2_statistics' in results:
            stats = results['gp2_statistics']
            print(f"\nGP2 Dimer Performance:")
            print(f"  Outer iterations: {stats.get('outer_iterations', 'N/A')}")
            print(f"  Total observations: {stats.get('total_observations', 'N/A')}")
            print(f"  Inner iterations: {stats.get('inner_iterations_total', 'N/A')}")
            if 'speedup_factor' in stats:
                print(f"  Speedup factor: {stats['speedup_factor']:.1f}x")
            
        if 'formation_energy' in results:
            if 'formation_energy_raw' in results:
                print(f"\nFormation energy (raw): {results['formation_energy_raw']:.4f} eV")
                if results.get('formation_correction', 0.0) != 0.0:
                    print(f"Formation correction: +{results['formation_correction']:.4f} eV")
                    print(f"Formation energy (corrected): {results['formation_energy']:.4f} eV")
            else:
                print(f"\nFormation energy: {results['formation_energy']:.4f} eV")
            print(f"Formation entropy: {results.get('formation_entropy', 0.0):.3f} k_B")
            print(f"Migration entropy: {results.get('migration_entropy', 0.0):.3f} k_B")
            
        if 'temperature_range' in results:
            T_data = results['temperature_range']['temperatures']
            D_data = results['temperature_range']['diffusion_coefficients']
            print(f"\nDiffusion coefficients calculated from {T_data[0]:.0f} K to {T_data[-1]:.0f} K")
            print(f"D at {T_data[0]:.0f} K: {D_data[0]:.2e} m²/s")
            print(f"D at {T_data[-1]:.0f} K: {D_data[-1]:.2e} m²/s")
            
        print(f"\nResults saved in {get_output_path('results')} directory:")
        print("  - self_diffusion_gp2_summary.txt: Full summary")
        print("  - diffusion_vs_temperature_gp2.csv: Temperature-dependent data")
        print("  - diffusion_arrhenius_plot.png: Arrhenius plot")
        print("  - diffusion_energy_profile.png: Energy barrier profile")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()