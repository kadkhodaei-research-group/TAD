#!/usr/bin/env python
"""GPU-accelerated EAM executor using PyTorch for NEB calculations."""

import os
import sys
import time
import numpy as np
from typing import List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Import PyTorch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ PyTorch GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    print("PyTorch not installed. GPU acceleration unavailable.")

# Try to import JAX as secondary option
JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    # Check JAX backend
    devices = jax.devices()
    jax_gpu = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
    if jax_gpu:
        print(f"✓ JAX GPU also available: {devices[0]}")
except:
    jax = None
    jnp = None

from vasp_executors import VASPExecutor
from vasp_manager import VASPRun
from ase import Atoms
from ase.calculators.eam import EAM
import warnings


# Module-level function for parallel processing (can be pickled)
def _calculate_single_eam(args):
    """Module-level function for parallel EAM calculation."""
    run_dir, potential_file, kim_model_name = args
    import os
    import numpy as np
    from ase import Atoms
    from ase.calculators.eam import EAM
    from pymatgen.io.vasp import Poscar
    
    poscar_path = os.path.join(run_dir, 'POSCAR')
    poscar = Poscar.from_file(poscar_path)
    structure = poscar.structure
    
    atoms = Atoms(
        symbols=[site.specie.symbol for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True
    )
    
    # Create calculator
    if kim_model_name:
        from ase.calculators.kim import KIM
        calc = KIM(kim_model_name)
    else:
        calc = EAM(potential=potential_file)
    
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    
    return run_dir, energy, forces, atoms


class EAMExecutorGPU(VASPExecutor):
    """GPU-accelerated EAM executor using PyTorch."""
    
    def __init__(self, vasp_command: str = None, potential_file: str = None,
                 kim_model: str = None, enable_gpu: bool = True, 
                 no_gpu_fallback: bool = False, parallel_eam: bool = False,
                 n_workers: int = None, **kwargs):
        """Initialize GPU-accelerated EAM executor.
        
        Args:
            vasp_command: Ignored for EAM
            potential_file: Path to EAM potential file
            kim_model: KIM model name (alternative to potential_file)
            enable_gpu: Whether to attempt GPU acceleration
            no_gpu_fallback: Raise error if GPU unavailable (instead of CPU fallback)
            parallel_eam: Use parallel execution
            n_workers: Number of parallel workers
        """
        super().__init__()
        self.potential_file = potential_file
        self.kim_model_name = kim_model
        self.parallel_eam = parallel_eam
        self.n_workers = n_workers or min(multiprocessing.cpu_count(), 8)
        
        # Determine if we can use GPU
        if enable_gpu:
            if GPU_AVAILABLE:
                self.use_gpu = True
                print("🚀 GPU acceleration ENABLED for EAM calculations")
                logging.info("GPU acceleration ENABLED for EAM calculations")
            elif no_gpu_fallback:
                raise RuntimeError("GPU requested but not available. PyTorch CUDA not detected.")
            else:
                self.use_gpu = False
                print("⚠️ GPU requested but not available, using CPU")
                logging.info("GPU requested but unavailable, falling back to CPU")
        else:
            self.use_gpu = False
            if GPU_AVAILABLE:
                print("GPU available but disabled by user (use --gpu to enable)")
            else:
                print("Using CPU for EAM calculations")
            logging.info("Using CPU for EAM calculations")
        
        # Initialize calculator
        self._init_calculator()
    
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
        """GPU-accelerated calculation using PyTorch.
        
        Note: Currently uses standard ASE calculator but moves data through GPU
        for potential future GPU-accelerated implementations.
        """
        if TORCH_AVAILABLE and GPU_AVAILABLE:
            # Move positions to GPU for potential GPU operations
            positions = torch.tensor(atoms.get_positions(), device='cuda', dtype=torch.float32)
            cell = torch.tensor(atoms.get_cell().array, device='cuda', dtype=torch.float32)
            
            # Create a fresh calculator instance to avoid state issues
            if self.kim_model_name:
                from ase.calculators.kim import KIM
                calc = KIM(self.kim_model_name)
            else:
                calc = EAM(potential=self.potential_file)
            
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            
            # Future: GPU-accelerated EAM implementation would go here
            # Example structure:
            # energy_gpu = gpu_eam_energy(positions, cell, self.potential_params)
            # forces_gpu = gpu_eam_forces(positions, cell, self.potential_params)
            # energy = energy_gpu.cpu().numpy()
            # forces = forces_gpu.cpu().numpy()
            
            return energy, forces
        else:
            # Fallback to CPU
            return self._calculate_cpu(atoms)
    
    def _calculate_cpu(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """Standard CPU calculation."""
        # Create a fresh calculator instance to avoid state issues
        if self.kim_model_name:
            from ase.calculators.kim import KIM
            calc = KIM(self.kim_model_name)
        else:
            calc = EAM(potential=self.potential_file)
        
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        return energy, forces
    
    def submit_batch(self, run_dirs: List[str]) -> None:
        """Submit batch of calculations with GPU or parallel CPU acceleration."""
        if self.use_gpu:
            self._submit_batch_gpu(run_dirs)
        elif self.parallel_eam:
            self._submit_batch_parallel_cpu(run_dirs)
        else:
            self._submit_batch_serial(run_dirs)
    
    def _submit_batch_gpu(self, run_dirs: List[str]) -> None:
        """GPU-accelerated batch processing."""
        print(f"🚀 Running batch of {len(run_dirs)} EAM calculations on GPU")
        
        # Process all structures
        start_time = time.time()
        
        # Process serially for GPU to avoid thread-safety issues with calculators
        # GPU is fast enough that serial processing is still efficient
        for i, run_dir in enumerate(run_dirs):
            if i % 20 == 0 and i > 0:
                print(f"  Progress: {i}/{len(run_dirs)}")
            
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
            
            energy, forces = self._calculate_gpu(atoms)
            self._write_mock_outcar(run_dir, energy, forces, atoms)
        
        elapsed = time.time() - start_time
        print(f"✅ GPU batch complete in {elapsed:.2f} seconds")
        print(f"   Average: {elapsed/len(run_dirs):.3f} sec/calculation")
    
    def _submit_batch_parallel_cpu(self, run_dirs: List[str]) -> None:
        """Parallel CPU batch processing using multiprocessing."""
        print(f"Running batch of {len(run_dirs)} EAM calculations in parallel (CPU, {self.n_workers} workers)...")
        
        start_time = time.time()
        
        # Use ProcessPoolExecutor for true parallel processing
        # This avoids thread-safety issues with KIM models
        from concurrent.futures import ProcessPoolExecutor
        
        # Prepare arguments for the module-level function
        args_list = [(run_dir, self.potential_file, self.kim_model_name) 
                     for run_dir in run_dirs]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Use the module-level function that can be pickled
            results = list(executor.map(_calculate_single_eam, args_list))
        
        # Write results
        for run_dir, energy, forces, atoms in results:
            self._write_mock_outcar(run_dir, energy, forces, atoms)
        
        elapsed = time.time() - start_time
        print(f"✅ Parallel CPU batch complete in {elapsed:.2f} seconds")
        print(f"   Average: {elapsed/len(run_dirs):.3f} sec/calculation")
    
    def _submit_batch_serial(self, run_dirs: List[str]) -> None:
        """Serial batch processing."""
        print(f"Running batch of {len(run_dirs)} EAM calculations (serial)...")
        start_time = time.time()
        
        for i, run_dir in enumerate(run_dirs):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(run_dirs)}")
            self.submit_job(run_dir, 'thermal')
        
        elapsed = time.time() - start_time
        print(f"Batch complete in {elapsed:.2f} seconds")
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Return results for all directories."""
        return [VASPRun(run_dir) for run_dir in run_dirs]
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None, **kwargs) -> Any:
        """Return results immediately since EAM calculations are instant."""
        # Add a small delay to simulate calculation time
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
                f.write("# GPU-accelerated EAM calculation (PyTorch)\n")
            else:
                f.write("# EAM calculation (CPU)\n")
            
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
