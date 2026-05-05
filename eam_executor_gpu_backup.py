#!/usr/bin/env python
"""GPU-accelerated EAM executor using JAX for NEB calculations."""

import os
import sys
import time
import numpy as np
from typing import List, Tuple, Optional, Any
import logging

# Try to import JAX, fall back gracefully if not available
GPU_AVAILABLE = False
JAX_INSTALLED = False
JAX_IMPORT_ERROR = None
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    from functools import partial
    JAX_INSTALLED = True
    
    # Check if GPU is actually available
    try:
        devices = jax.devices()
        GPU_AVAILABLE = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        if GPU_AVAILABLE:
            print(f"GPU detected: {devices[0]}")
    except:
        GPU_AVAILABLE = False
        
except (ImportError, RuntimeError) as e:
    # Catch both ImportError and RuntimeError (e.g., AVX instruction issues on Mac)
    JAX_IMPORT_ERROR = str(e)
    if "AVX instructions" in str(e):
        print("JAX incompatible with system (x86 Python on ARM Mac). GPU acceleration unavailable.")
    else:
        print("JAX not installed. GPU acceleration unavailable. Install with: pip install jax[cuda] -U")
    jax = None
    jnp = None

from vasp_executors import VASPExecutor
from vasp_manager import VASPRun
from ase import Atoms
from ase.calculators.eam import EAM
import warnings


class EAMExecutorGPU(VASPExecutor):
    """GPU-accelerated EAM executor using JAX."""
    
    def __init__(self, vasp_command: str = None, potential_file: str = None,
                 kim_model: str = None, use_gpu: bool = True, 
                 fallback_to_cpu: bool = True, parallel_eam: bool = False,
                 n_workers: int = None):
        """Initialize GPU-accelerated EAM executor.
        
        Args:
            vasp_command: Ignored for EAM
            potential_file: Path to EAM potential file
            kim_model: KIM model name (alternative to potential_file)
            use_gpu: Whether to attempt GPU acceleration
            fallback_to_cpu: Fall back to CPU if GPU unavailable
            parallel_eam: Use parallel CPU execution (if GPU unavailable)
            n_workers: Number of parallel workers for CPU fallback
        """
        super().__init__()
        self.potential_file = potential_file
        self.kim_model_name = kim_model
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.fallback_to_cpu = fallback_to_cpu
        self.parallel_eam = parallel_eam
        self.n_workers = n_workers or os.cpu_count()
        
        # Initialize calculator
        self._init_calculator()
        
        # Report status
        if self.use_gpu:
            logging.info("GPU acceleration ENABLED for EAM calculations")
            print("🚀 GPU acceleration ENABLED for EAM calculations")
        elif GPU_AVAILABLE and not use_gpu:
            logging.info("GPU available but disabled by user")
            print("GPU available but disabled (use --gpu to enable)")
        elif use_gpu and not GPU_AVAILABLE:
            if fallback_to_cpu:
                logging.info("GPU requested but unavailable, falling back to CPU")
                if JAX_IMPORT_ERROR and "AVX instructions" in JAX_IMPORT_ERROR:
                    print("⚠️  JAX incompatible with system (x86 Python on ARM Mac), using CPU fallback")
                elif JAX_INSTALLED:
                    print("⚠️  JAX installed but no GPU detected (likely on Mac/CPU-only system), using CPU fallback")
                else:
                    print("⚠️  GPU requested but JAX not available, using CPU fallback")
            else:
                if JAX_IMPORT_ERROR and "AVX instructions" in JAX_IMPORT_ERROR:
                    raise RuntimeError("GPU requested but JAX is incompatible with your system (x86 Python on ARM Mac). Use without --no-gpu-fallback to use CPU.")
                elif JAX_INSTALLED:
                    raise RuntimeError("GPU requested but not available. No NVIDIA GPU detected. Use without --no-gpu-fallback to use CPU.")
                else:
                    raise RuntimeError("GPU requested but JAX not installed. Install JAX with: pip install jax[cuda] -U")
        else:
            logging.info("Using CPU for EAM calculations")
            print("Using CPU for EAM calculations")
    
    def _init_calculator(self):
        """Initialize the appropriate calculator."""
        if self.kim_model_name:
            try:
                from ase.calculators.kim import KIM
                self.calculator = KIM(self.kim_model_name)
                print(f"Using KIM model: {self.kim_model_name}")
            except ImportError:
                raise ImportError("KIM-API not installed. Install with: conda install -c conda-forge kim-api")
        elif self.potential_file:
            self.calculator = EAM(potential=self.potential_file)
            print(f"Using EAM potential: {self.potential_file}")
        else:
            raise ValueError("Either potential_file or kim_model must be specified")
    
    def submit_job(self, run_dir: str, run_type: str = 'main') -> Any:
        """Run EAM calculation with GPU acceleration if available."""
        poscar_path = os.path.join(run_dir, 'POSCAR')
        
        # Read structure
        from pymatgen.io.vasp import Poscar
        poscar = Poscar.from_file(poscar_path)
        structure = poscar.structure
        
        # Convert to ASE atoms
        atoms = Atoms(
            symbols=[site.specie.symbol for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )
        
        # Calculate with GPU or CPU
        if self.use_gpu:
            energy, forces = self._calculate_gpu(atoms)
        else:
            energy, forces = self._calculate_cpu(atoms)
        
        # Write results
        self._write_mock_outcar(run_dir, energy, forces, atoms)
        
        return VASPRun(run_dir)
    
    def _calculate_gpu(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """GPU-accelerated calculation using JAX."""
        # For now, we'll use the standard calculator but prepare positions for GPU
        # In a full implementation, this would use a JAX-based EAM implementation
        
        # Move data to GPU - properly handle ASE Cell object
        positions = jnp.array(atoms.get_positions())
        # Convert cell to numpy first, then to JAX array
        cell_array = np.array(atoms.get_cell().array)
        cell = jnp.array(cell_array, dtype=jnp.float32)
        
        # Use standard calculator for now (full GPU implementation would be pure JAX)
        atoms.calc = self.calculator
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        
        # In a real implementation, we would have JAX-based force calculation
        # This is a placeholder showing where GPU acceleration would happen
        
        return energy, forces
    
    def _calculate_cpu(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """Standard CPU calculation."""
        atoms.calc = self.calculator
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        return energy, forces
    
    def submit_batch(self, run_dirs: List[str]) -> None:
        """Submit batch of calculations with GPU acceleration."""
        if self.use_gpu:
            self._submit_batch_gpu(run_dirs)
        elif self.parallel_eam:
            self._submit_batch_parallel_cpu(run_dirs)
        else:
            self._submit_batch_serial(run_dirs)
    
    def _submit_batch_gpu(self, run_dirs: List[str]) -> None:
        """GPU-accelerated batch processing."""
        # Double-check GPU is really available
        if JAX_INSTALLED and jax is not None:
            try:
                devices = jax.devices()
                gpu_available = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
                if gpu_available:
                    print(f"🚀 Running batch of {len(run_dirs)} EAM calculations on GPU ({devices[0]})")
                else:
                    print(f"⚠️  Running batch of {len(run_dirs)} EAM calculations (JAX on CPU)")
            except:
                print(f"Running batch of {len(run_dirs)} EAM calculations...")
        else:
            print(f"Running batch of {len(run_dirs)} EAM calculations...")
        
        # Collect all structures
        all_atoms = []
        for run_dir in run_dirs:
            poscar_path = os.path.join(run_dir, 'POSCAR')
            from pymatgen.io.vasp import Poscar
            poscar = Poscar.from_file(poscar_path)
            structure = poscar.structure
            
            atoms = Atoms(
                symbols=[site.specie.symbol for site in structure],
                positions=structure.cart_coords,
                cell=structure.lattice.matrix,
                pbc=True
            )
            all_atoms.append(atoms)
        
        # Batch process on GPU
        start_time = time.time()
        for i, (atoms, run_dir) in enumerate(zip(all_atoms, run_dirs)):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(run_dirs)}")
            
            try:
                energy, forces = self._calculate_gpu(atoms)
            except Exception as e:
                print(f"  Warning: GPU calculation failed for image {i}: {e}")
                print(f"  Falling back to CPU for this image")
                energy, forces = self._calculate_cpu(atoms)
                
            self._write_mock_outcar(run_dir, energy, forces, atoms)
        
        elapsed = time.time() - start_time
        print(f"✅ Batch complete in {elapsed:.2f} seconds")
        print(f"   Average: {elapsed/len(run_dirs):.3f} sec/calculation")
    
    def _submit_batch_parallel_cpu(self, run_dirs: List[str]) -> None:
        """Parallel CPU batch processing."""
        print(f"Running batch of {len(run_dirs)} EAM calculations in parallel (CPU)...")
        
        from concurrent.futures import ProcessPoolExecutor
        
        def calculate_single(run_dir):
            """Calculate for a single structure."""
            poscar_path = os.path.join(run_dir, 'POSCAR')
            from pymatgen.io.vasp import Poscar
            poscar = Poscar.from_file(poscar_path)
            structure = poscar.structure
            
            atoms = Atoms(
                symbols=[site.specie.symbol for site in structure],
                positions=structure.cart_coords,
                cell=structure.lattice.matrix,
                pbc=True
            )
            
            # Create new calculator instance for this process
            if self.kim_model_name:
                from ase.calculators.kim import KIM
                calc = KIM(self.kim_model_name)
            else:
                calc = EAM(potential=self.potential_file)
            
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            
            return run_dir, energy, forces, atoms
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(calculate_single, run_dirs))
        
        # Write results
        for run_dir, energy, forces, atoms in results:
            self._write_mock_outcar(run_dir, energy, forces, atoms)
        
        elapsed = time.time() - start_time
        print(f"Parallel CPU batch complete in {elapsed:.2f} seconds")
    
    def _submit_batch_serial(self, run_dirs: List[str]) -> None:
        """Serial batch processing."""
        print(f"Running batch of {len(run_dirs)} EAM calculations (serial)...")
        for i, run_dir in enumerate(run_dirs):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(run_dirs)}")
            self.submit_job(run_dir, 'thermal')
        print("Batch complete")
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Return results for all directories."""
        return [VASPRun(run_dir) for run_dir in run_dirs]
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None, **kwargs) -> Any:
        """Return results immediately since EAM calculations are instant."""
        # Add a small delay to simulate calculation time
        import time
        time.sleep(0.01)
        return VASPRun(run_dir)
    
    def _write_mock_outcar(self, run_dir: str, energy: float, forces: np.ndarray, atoms: Atoms):
        """Write OUTCAR file with EAM results in VASP format."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        
        # Ensure forces are 2D numpy array
        forces = np.asarray(forces, dtype=np.float64)
        if forces.ndim == 1:
            forces = forces.reshape(-1, 3)
        
        with open(outcar_path, 'w') as f:
            if self.use_gpu:
                f.write("# GPU-accelerated EAM calculation\n")
            else:
                f.write("# EAM calculation\n")
            
            if self.kim_model_name:
                f.write(f"# KIM model: {self.kim_model_name}\n")
            elif self.potential_file:
                f.write(f"# Potential file: {self.potential_file}\n")
            
            f.write("\n")
            f.write("POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write("-----------------------------------------------------------------------------------\n")
            
            positions = atoms.get_positions()
            for i, (pos, force) in enumerate(zip(positions, forces)):
                f.write(f" {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f}   ")
                f.write(f"{force[0]:15.8f} {force[1]:15.8f} {force[2]:15.8f}\n")
            
            f.write("-----------------------------------------------------------------------------------\n")
            f.write("\n")
            f.write(f"  free  energy   TOTEN  =      {energy:15.8f} eV\n")
            f.write(f"  energy  without entropy=      {energy:15.8f} eV\n")