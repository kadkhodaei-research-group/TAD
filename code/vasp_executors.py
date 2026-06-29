from typing import Optional, List, Dict, Any, Tuple
import subprocess
import os
import time
import shutil
import numpy as np
from abc import ABC, abstractmethod
from pymatgen.io.vasp import Poscar

class VASPExecutor(ABC):
    """Abstract base class for VASP execution strategies."""
    
    @abstractmethod
    def submit_job(self, run_dir: str, run_type: str) -> Any:
        """Submit a VASP job."""
        pass
    
    @abstractmethod
    def wait_for_completion(self, run_dir: str, job_info: Any = None, **kwargs) -> Any:
        """Wait for job completion and return results."""
        pass
    
    @abstractmethod
    def submit_batch(self, run_dirs: List[str]) -> Any:
        """Submit multiple jobs as a batch."""
        pass
    
    @abstractmethod
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Wait for batch completion and return results."""
        pass


def get_hpc_executor(base_dir: str, execution_mode: str = "auto", **kwargs) -> VASPExecutor:
    """
    Automatically select the appropriate VASP executor based on the HPC system.
    
    Args:
        base_dir: Base directory for VASP runs
        execution_mode: Execution mode - "auto", "mock", "mpi", "slurm", "stampede3", "anvil", "vasp"
    
    Returns:
        Appropriate VASPExecutor instance
    """
    
    # Handle explicit execution modes
    if execution_mode == "mock":
        from vasp_executors import MockVASPExecutor
        return MockVASPExecutor(base_dir)

    if execution_mode == "eam":
        from eam_executor import EAMExecutor
        kim_model = kwargs.get('kim_model_name', "EAM_Dynamo_MendelevAckland_2007_Zr__MO_537826574817_000")
        user_poscar = kwargs.get('user_poscar_path')
        return EAMExecutor(kim_model_name=kim_model, user_poscar_path=user_poscar)
    
    if execution_mode == "mpi":
        from vasp_executors import MPIVASPExecutor
        return MPIVASPExecutor()
    
    if execution_mode == "slurm":
        from vasp_executors import SLURMVASPExecutor
        return SLURMVASPExecutor()

    if execution_mode == "vasp":
        from vasp_executors import VASPLocalExecutor
        vasp_command = kwargs.get('vasp_command', 'vasp_gam')
        mpi_command = kwargs.get('mpi_command')
        user_poscar = kwargs.get('user_poscar_path')
        return VASPLocalExecutor(
            vasp_command=vasp_command,
            mpi_command=mpi_command,
            user_poscar_path=user_poscar
        )

    # Auto-detect or handle specific systems
    if execution_mode in ["auto", "stampede3", "anvil"]:
        # Detect which system we're on
        system_name = detect_hpc_system()
        
        if execution_mode != "auto":
            # User specified a system, but let's verify we're on the right one
            if execution_mode == "stampede3" and system_name != "stampede3":
                print(f"Warning: Requested Stampede3 executor but detected system is {system_name}")
            elif execution_mode == "anvil" and system_name != "anvil":
                print(f"Warning: Requested Anvil executor but detected system is {system_name}")
        
        # Select executor based on detected or requested system
        if system_name == "stampede3" or execution_mode == "stampede3":
            from vasp_executors import Stampede3VASPExecutor
            return Stampede3VASPExecutor(base_dir)
        elif system_name == "anvil" or execution_mode == "anvil":
            from vasp_executors import AnvilVASPExecutor
            return AnvilVASPExecutor()
        elif system_name == "bridges2":
            from vasp_executors import MPIVASPExecutor
            return MPIVASPExecutor("mpirun -np 64", "vasp_gam")
        else:
            # Default fallback
            print(f"Unknown system {system_name}, defaulting to generic SLURM executor")
            from vasp_executors import SLURMVASPExecutor
            return SLURMVASPExecutor()
    
    raise ValueError(f"Unknown execution mode: {execution_mode}")


def detect_hpc_system() -> str:
    """
    Detect which HPC system we're running on.
    
    Returns:
        System name: "stampede3", "anvil", "bridges2", or "unknown"
    """
    import os
    import subprocess
    
    # Check TACC_SYSTEM for Stampede3
    if os.environ.get('TACC_SYSTEM', '').lower() == 'stampede3':
        return 'stampede3'
    
    # Check hostname for Anvil
    hostname = os.environ.get('HOSTNAME', '').lower()
    if 'anvil' in hostname:
        return 'anvil'
    
    # Check for Anvil-specific environment
    if any(env.startswith('ANVIL') for env in os.environ):
        return 'anvil'
    
    # Check SLURM cluster name
    cluster_name = os.environ.get('SLURM_CLUSTER_NAME', '').lower()
    if 'anvil' in cluster_name:
        return 'anvil'
    elif 'stampede' in cluster_name:
        return 'stampede3'
    elif 'bridges' in cluster_name:
        return 'bridges2'
    
    # Check for PSC (Pittsburgh Supercomputing Center) for Bridges-2
    if 'PSC' in os.environ or 'BRIDGES' in os.environ:
        return 'bridges2'
    
    # Try to detect from module system
    try:
        result = subprocess.run(['module', 'list'], capture_output=True, text=True)
        module_output = result.stdout + result.stderr
        if 'TACC' in module_output:
            return 'stampede3'
        elif 'anvil' in module_output.lower():
            return 'anvil'
    except:
        pass
    
    return 'unknown'

class MockVASPExecutor(VASPExecutor):
    """Mock VASP executor for testing - generates fake results instantly."""
    
    def __init__(self, user_poscar_path: str):
        self.user_poscar_path = user_poscar_path
        
    def submit_job(self, run_dir: str, run_type: str) -> None:
        """Create mock output files immediately."""
        self._create_mock_vasp_output(run_dir)
        return None
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None, **kwargs) -> Any:
        """Instant return for mock runs."""
        # Import here to avoid circular import
        from vasp_manager import VASPRun
        return VASPRun(run_dir)
    
    def submit_batch(self, run_dirs: List[str]) -> None:
        """Create mock outputs for all directories."""
        for run_dir in run_dirs:
            self._create_mock_vasp_output(run_dir)
        return None
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Return mock results for all directories."""
        # Import here to avoid circular import
        from vasp_manager import VASPRun
        return [VASPRun(run_dir) for run_dir in run_dirs]

    def _create_mock_vasp_output(self, run_dir: str) -> None:
        """Generate fake VASP output files with random energy + forces."""
        template = Poscar.from_file(self.user_poscar_path).structure
        n_atoms = len(template)
        
        # Mock forces (±0.5 eV/Å) - more reasonable range
        mock_forces = np.random.uniform(-0.5, 0.5, (n_atoms, 3))
        
        # Mock total energy centered on -123 eV, ±2 eV window - smaller variation
        total_energy = -123.0 + np.random.uniform(-2.0, 2.0)
        
        # Write OUTCAR
        outcar_path = os.path.join(run_dir, "OUTCAR")
        with open(outcar_path, "w") as f:
            f.write("POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write("----------------------------------------------------\n")
            for fx, fy, fz in mock_forces:
                f.write(f"  0.000000  0.000000  0.000000   {fx:.6f}  {fy:.6f}  {fz:.6f}\n")
            f.write("----------------------------------------------------\n")
            f.write("FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
            f.write(f"free  energy   TOTEN  =       {total_energy: .8f} eV\n")
            f.write("General timing and accounting informations\n")
        
        # Copy POSCAR → CONTCAR
        shutil.copy(self.user_poscar_path, os.path.join(run_dir, "CONTCAR"))


class MPIVASPExecutor(VASPExecutor):
    """Regular MPI VASP executor for systems like Bridges2."""
    
    def __init__(self, mpi_command: str = "mpirun -np 64", vasp_command: str = "vasp_gam"):
        self.mpi_command = mpi_command
        self.vasp_command = vasp_command
        self.active_jobs = {}  # run_dir -> subprocess.Popen
        
    def submit_job(self, run_dir: str, run_type: str) -> subprocess.Popen:
        """Submit VASP job using mpirun."""
        cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            cmd = f"{self.mpi_command} {self.vasp_command}"
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.active_jobs[run_dir] = process
            print(f"Submitted {run_type} job in {run_dir} (PID: {process.pid})")
            return process
        finally:
            os.chdir(cwd)
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None, 
                          max_wait_time: int = 86400, check_interval: int = 30) -> Any:
        """Wait for VASP job to complete."""
        process = job_info or self.active_jobs.get(run_dir)
        
        if process:
            process.wait()
            del self.active_jobs[run_dir]
        
        # Verify calculation completed
        start_time = time.time()
        while not self._is_calculation_complete(run_dir):
            if time.time() - start_time > max_wait_time:
                raise TimeoutError(f"VASP job in {run_dir} did not complete within {max_wait_time/3600:.1f} hours")
            time.sleep(check_interval)
        
        # Import here to avoid circular import
        from vasp_manager import VASPRun
        return VASPRun(run_dir)
    
    def submit_batch(self, run_dirs: List[str]) -> List[subprocess.Popen]:
        """Submit multiple jobs in parallel."""
        processes = []
        for run_dir in run_dirs:
            process = self.submit_job(run_dir, 'thermal')
            processes.append(process)
        return processes
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Wait for all processes in batch to complete."""
        processes = batch_info or []
        
        # Wait for all processes
        for process in processes:
            process.wait()
        
        # Clear active jobs
        for run_dir in run_dirs:
            self.active_jobs.pop(run_dir, None)
        
        # Verify all calculations completed and parse results
        results = []
        for run_dir in run_dirs:
            while not self._is_calculation_complete(run_dir):
                time.sleep(5)
            from vasp_manager import VASPRun
            results.append(VASPRun(run_dir))
        
        return results
    
    def _is_calculation_complete(self, run_dir: str) -> bool:
        """Check if VASP calculation has completed successfully."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        if not os.path.exists(outcar_path):
            return False
        
        try:
            with open(outcar_path, 'r') as f:
                outcar_content = f.read()
                return 'General timing and accounting informations' in outcar_content
        except:
            return False


class SLURMVASPExecutor(VASPExecutor):
    """SLURM-based VASP executor for systems with regular SLURM access."""
    
    def __init__(self):
        self.active_jobs = {}  # run_dir -> job_id
        
    def submit_job(self, run_dir: str, run_type: str) -> str:
        """Submit VASP job to SLURM queue."""
        job_script = os.path.join(run_dir, 'job.sh')
        if not os.path.exists(job_script):
            raise FileNotFoundError(f"Job script not found: {job_script}")
        
        cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            result = subprocess.run(['sbatch', 'job.sh'], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"SLURM submission failed: {result.stderr}")
            
            # Extract job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            self.active_jobs[run_dir] = job_id
            
            print(f"Submitted {run_type} SLURM job {job_id} in {run_dir}")
            return job_id
            
        finally:
            os.chdir(cwd)
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None,
                          max_wait_time: int = 86400, check_interval: int = 30) -> Any:
        """Wait for SLURM job to complete."""
        job_id = job_info or self.active_jobs.get(run_dir)
        
        if job_id:
            # Wait for SLURM job
            while not self._is_slurm_job_complete(job_id):
                time.sleep(check_interval)
            
            del self.active_jobs[run_dir]
        
        # Verify calculation completed
        start_time = time.time()
        while not self._is_calculation_complete(run_dir):
            if time.time() - start_time > max_wait_time:
                raise TimeoutError(f"VASP job in {run_dir} did not complete")
            time.sleep(check_interval)
        from vasp_manager import VASPRun
        return VASPRun(run_dir)
    
    def submit_batch(self, run_dirs: List[str]) -> List[str]:
        """Submit multiple jobs individually."""
        job_ids = []
        for run_dir in run_dirs:
            job_id = self.submit_job(run_dir, 'thermal')
            job_ids.append(job_id)
        return job_ids
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Wait for all jobs in batch to complete."""
        job_ids = batch_info or []
        
        # Wait for all jobs
        for job_id in job_ids:
            while not self._is_slurm_job_complete(job_id):
                time.sleep(30)
        
        # Clear active jobs
        for run_dir in run_dirs:
            self.active_jobs.pop(run_dir, None)
        
        # Parse results
        results = []
        for run_dir in run_dirs:
            while not self._is_calculation_complete(run_dir):
                time.sleep(5)
            from vasp_manager import VASPRun
            results.append(VASPRun(run_dir))
        
        return results
    
    def _is_slurm_job_complete(self, job_id: str) -> bool:
        """Check if a SLURM job has completed."""
        try:
            result = subprocess.run(['squeue', '-j', job_id, '-h'], 
                                  capture_output=True, text=True)
            return len(result.stdout.strip()) == 0
        except:
            return True
    
    def _is_calculation_complete(self, run_dir: str) -> bool:
        """Check if VASP calculation has completed successfully."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        if not os.path.exists(outcar_path):
            return False
        
        try:
            with open(outcar_path, 'r') as f:
                outcar_content = f.read()
                return 'General timing and accounting informations' in outcar_content
        except:
            return False


class Stampede3VASPExecutor(VASPExecutor):
    """Executor for Stampede3 that runs VASP directly on allocated compute nodes."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.pending_thermal_dirs = []
        self.current_thermal_batch_dirs = []
        
        # Detect number of allocated nodes from SLURM environment
        self.num_nodes = self._get_allocated_nodes()
        self.processes_per_node = 48  # Stampede3 SKX nodes have 48 cores
        
        print(f"Stampede3 Executor initialized with {self.num_nodes} nodes")
        print(f"Will run {self.num_nodes} VASP jobs in parallel")
    
    def _get_allocated_nodes(self) -> int:
        """Detect number of nodes allocated to current job."""
        # Check SLURM environment variables
        if 'SLURM_JOB_NUM_NODES' in os.environ:
            return int(os.environ['SLURM_JOB_NUM_NODES'])
        elif 'SLURM_NNODES' in os.environ:
            return int(os.environ['SLURM_NNODES'])
        else:
            print("Warning: Could not detect number of allocated nodes, defaulting to 1")
            return 1
    
    def submit_job(self, run_dir: str, run_type: str) -> None:
        """Queue job for execution (thermal runs) or execute immediately (dimer runs)."""
        if run_type == 'thermal':
            # Queue for batch execution
            self.pending_thermal_dirs.append(run_dir)
            return None
        else:
            # Execute dimer run immediately using all available nodes
            print(f"Running dimer calculation in {run_dir}")
            self._run_vasp_on_nodes(run_dir, self.num_nodes)
            return None
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None, **kwargs) -> Any:
        """For dimer runs, VASP has already completed. Just parse results."""
        # Verify calculation completed
        max_wait = kwargs.get('max_wait_time', 300)
        start_time = time.time()
        
        while not self._is_calculation_complete(run_dir):
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"VASP calculation in {run_dir} did not complete within {max_wait}s")
            time.sleep(5)

        from vasp_manager import VASPRun
        return VASPRun(run_dir)
    
    def submit_batch(self, run_dirs: List[str]) -> None:
        """Execute pending thermal calculations in parallel batches."""
        if not run_dirs:
            run_dirs = self.pending_thermal_dirs
        
        if not run_dirs:
            return None
        
        self.current_thermal_batch_dirs = run_dirs.copy()
        self.pending_thermal_dirs = []
        
        print(f"\nExecuting {len(run_dirs)} thermal calculations in batches of {self.num_nodes}")
        
        # Process in batches based on number of available nodes
        for i in range(0, len(run_dirs), self.num_nodes):
            batch = run_dirs[i:i + self.num_nodes]
            print(f"\nBatch {i//self.num_nodes + 1}: Running {len(batch)} calculations")
            
            # Run VASP jobs in parallel
            processes = []
            for j, run_dir in enumerate(batch):
                print(f"  Starting VASP in {os.path.basename(run_dir)} on node subset {j}")
                proc = self._start_vasp_on_node_subset(run_dir, j, 1)
                processes.append((run_dir, proc))
            
            # Wait for all processes in this batch to complete
            for run_dir, proc in processes:
                print(f"  Waiting for {os.path.basename(run_dir)}...", end='', flush=True)
                proc.wait()
                if proc.returncode == 0:
                    print(" Done")
                else:
                    print(f" Failed with code {proc.returncode}")
                    # Check error output
                    if os.path.exists(os.path.join(run_dir, 'vasp.err')):
                        with open(os.path.join(run_dir, 'vasp.err'), 'r') as f:
                            print(f"    Error: {f.read()}")
        
        return None
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Parse results from completed thermal calculations."""
        results = []
        failed_runs = []
        
        for i, run_dir in enumerate(run_dirs):
            print(f"Parsing results {i+1}/{len(run_dirs)}: {os.path.basename(run_dir)}")
            
            if self._is_calculation_complete(run_dir):
                try:
                    from vasp_manager import VASPRun
                    vr = VASPRun(run_dir)
                    vr.run_dir = run_dir
                    results.append(vr)
                    print(f"  ✓ Success")
                except Exception as e:
                    print(f"  ✗ Failed to parse: {e}")
                    failed_runs.append(run_dir)
            else:
                print(f"  ✗ Calculation incomplete")
                failed_runs.append(run_dir)
        
        if failed_runs:
            print(f"\nWarning: {len(failed_runs)} calculations failed:")
            for run_dir in failed_runs[:5]:  # Show first 5
                print(f"  - {run_dir}")
            if len(failed_runs) > 5:
                print(f"  ... and {len(failed_runs) - 5} more")
        
        self.current_thermal_batch_dirs = []
        return results
    
    def _run_vasp_on_nodes(self, run_dir: str, num_nodes: int) -> None:
        """Run VASP using multiple nodes and wait for completion."""
        proc = self._start_vasp_on_node_subset(run_dir, 0, num_nodes)
        proc.wait()
        
        if proc.returncode != 0:
            raise RuntimeError(f"VASP failed with return code {proc.returncode}")
    
    def _start_vasp_on_node_subset(self, run_dir: str, node_offset: int, num_nodes: int) -> subprocess.Popen:
        """Start VASP on a subset of allocated nodes."""
        # Calculate MPI ranks
        start_rank = node_offset * self.processes_per_node
        num_processes = num_nodes * self.processes_per_node
        
        # Build ibrun command with rank offset
        cmd = [
            'ibrun',
            '-n', str(num_processes),
            '-o', str(start_rank),
            'vasp_gam'
        ]
        
        # Set up output redirection
        output_file = os.path.join(run_dir, 'vasp.out')
        error_file = os.path.join(run_dir, 'vasp.err')
        
        with open(output_file, 'w') as out, open(error_file, 'w') as err:
            proc = subprocess.Popen(
                cmd,
                cwd=run_dir,
                stdout=out,
                stderr=err,
                env={**os.environ, 'OMP_NUM_THREADS': '1'}
            )
        
        return proc
    
    def _is_calculation_complete(self, run_dir: str) -> bool:
        """Check if VASP calculation has completed successfully."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        if not os.path.exists(outcar_path):
            return False
        
        try:
            with open(outcar_path, 'r') as f:
                outcar_content = f.read()
                # Check for successful completion markers
                return ('General timing and accounting informations' in outcar_content or
                        'Total CPU time used' in outcar_content)
        except:
            return False
    
    def get_status_summary(self) -> str:
        """Get a summary of executor status."""
        return (f"Stampede3 Executor: {self.num_nodes} nodes allocated, "
                f"{len(self.pending_thermal_dirs)} jobs pending")


class Stampede3MultiNodeVASPExecutor(Stampede3VASPExecutor):
    """Enhanced executor that can run multiple single-node jobs per allocated node."""
    
    def __init__(self, base_dir: str):
        super().__init__(base_dir)
        # For thermal snapshots, we can run multiple single-node jobs per physical node
        self.jobs_per_node = 1  # Can be increased if jobs use less than 48 cores
    
    def submit_batch(self, run_dirs: List[str]) -> None:
        """Execute thermal calculations with better node utilization."""
        if not run_dirs:
            run_dirs = self.pending_thermal_dirs
        
        if not run_dirs:
            return None
        
        self.current_thermal_batch_dirs = run_dirs.copy()
        self.pending_thermal_dirs = []
        
        total_parallel_jobs = self.num_nodes * self.jobs_per_node
        print(f"\nExecuting {len(run_dirs)} thermal calculations")
        print(f"Running {total_parallel_jobs} jobs in parallel ({self.jobs_per_node} per node)")
        
        # Process in batches
        for i in range(0, len(run_dirs), total_parallel_jobs):
            batch = run_dirs[i:i + total_parallel_jobs]
            batch_num = i // total_parallel_jobs + 1
            print(f"\nBatch {batch_num}: Running {len(batch)} calculations")
            
            # Start all jobs in parallel
            processes = []
            for j, run_dir in enumerate(batch):
                node_id = j // self.jobs_per_node
                job_on_node = j % self.jobs_per_node
                
                # Calculate resources per job
                cores_per_job = self.processes_per_node // self.jobs_per_node
                start_rank = node_id * self.processes_per_node + job_on_node * cores_per_job
                
                print(f"  Job {j+1}: {os.path.basename(run_dir)} on node {node_id} ranks {start_rank}-{start_rank+cores_per_job-1}")
                
                cmd = [
                    'ibrun',
                    '-n', str(cores_per_job),
                    '-o', str(start_rank),
                    'vasp_gam'
                ]
                
                output_file = os.path.join(run_dir, 'vasp.out')
                error_file = os.path.join(run_dir, 'vasp.err')
                
                with open(output_file, 'w') as out, open(error_file, 'w') as err:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=run_dir,
                        stdout=out,
                        stderr=err,
                        env={**os.environ, 'OMP_NUM_THREADS': '1'}
                    )
                
                processes.append((run_dir, proc))
            
            # Wait for all processes in batch
            print("\n  Waiting for batch to complete...")
            for run_dir, proc in processes:
                dirname = os.path.basename(run_dir)
                proc.wait()
                if proc.returncode == 0:
                    print(f"    ✓ {dirname}")
                else:
                    print(f"    ✗ {dirname} (exit code: {proc.returncode})")
            
            print(f"  Batch {batch_num} complete")
        
        print(f"\nAll {len(run_dirs)} calculations completed")
        return None
    
class AnvilVASPExecutor(VASPExecutor):
    """Anvil executor that submits VASP jobs through SLURM - Fixed version."""
    
    def __init__(self):
        self.active_jobs = {}  # run_dir -> job_id
        self.pending_thermal_dirs = []
        self.current_thermal_batch_info = None
        self.current_thermal_batch_dirs = []
        self._pending_thermal_dirs = []
        print("AnvilVASPExecutor initialized (fixed version)")
        
    def submit_job(self, run_dir: str, run_type: str) -> str:
        """Submit VASP job to SLURM queue on Anvil."""
        print(f"\nAnvilVASPExecutor.submit_job called:")
        print(f"  run_dir: {run_dir}")
        print(f"  run_type: {run_type}")
        
        # Create job script if it doesn't exist
        job_script = os.path.join(run_dir, 'job.sh')
        if not os.path.exists(job_script):
            self._create_job_script(run_dir, run_type)
        
        # For ALL job types, submit immediately (like Bridges2)
        cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            result = subprocess.run(['sbatch', 'job.sh'], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"SLURM submission failed: {result.stderr}")
            
            # Extract job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            self.active_jobs[run_dir] = job_id
            
            print(f"  ✓ Submitted {run_type} SLURM job {job_id}")
            
            # For thermal runs, track them for batch collection later
            if run_type == 'thermal':
                self.pending_thermal_dirs.append(run_dir)
                self._pending_thermal_dirs.append(run_dir)
                self.current_thermal_batch_dirs.append(run_dir)
                # Store job ID for later batch waiting
                if self.current_thermal_batch_info is None:
                    self.current_thermal_batch_info = {}
                self.current_thermal_batch_info[run_dir] = job_id
            
            return job_id
            
        finally:
            os.chdir(cwd)
    
    def wait_for_completion(self, run_dir: str, job_info: Any = None,
                          max_wait_time: int = 86400, check_interval: int = 30) -> Any:
        """Wait for SLURM job to complete."""
        job_id = job_info or self.active_jobs.get(run_dir)
        
        if job_id:
            # Wait for SLURM job
            start_time = time.time()
            while not self._is_slurm_job_complete(job_id):
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"Job {job_id} did not complete within {max_wait_time/3600:.1f} hours")
                time.sleep(check_interval)
            
            if run_dir in self.active_jobs:
                del self.active_jobs[run_dir]
        
        # Verify calculation completed
        start_time = time.time()
        while not self._is_calculation_complete(run_dir):
            if time.time() - start_time > 300:  # 5 minute timeout for file completion
                raise TimeoutError(f"VASP output in {run_dir} not complete after job finished")
            time.sleep(5)
        from vasp_manager import VASPRun
        return VASPRun(run_dir)
    
    def submit_batch(self, run_dirs: List[str]) -> Dict[str, str]:
        """For Anvil, jobs are already submitted individually, just return the job IDs."""
        print(f"\nAnvilVASPExecutor.submit_batch called")
        print(f"  run_dirs passed in: {len(run_dirs) if run_dirs else 0}")
        
        # Since jobs were already submitted in submit_job, just return the collected info
        if self.current_thermal_batch_info:
            print(f"  Returning {len(self.current_thermal_batch_info)} already-submitted jobs")
            # Clear the pending lists since we're "submitting" the batch
            self.pending_thermal_dirs = []
            self._pending_thermal_dirs = []
            return self.current_thermal_batch_info
        
        # If for some reason we don't have batch info, return empty
        print("  WARNING: No batch info found")
        return {}
    
    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Wait for all jobs in batch to complete."""
        job_ids = batch_info or self.current_thermal_batch_info
        
        print(f"\nAnvilVASPExecutor.wait_for_batch called")
        print(f"  Number of job_ids: {len(job_ids) if job_ids else 0}")
        print(f"  Number of run_dirs: {len(run_dirs)}")
        
        if not job_ids:
            print("ERROR: No job IDs to wait for!")
            return []
        
        print(f"\nWaiting for {len(job_ids)} thermal calculations to complete...")
        
        # Wait for all jobs
        completed = set()
        start_time = time.time()
        max_wait = 3600  # 1 hour max for thermal calculations
        
        while len(completed) < len(job_ids):
            for run_dir, job_id in job_ids.items():
                if run_dir in completed:
                    continue
                
                if self._is_slurm_job_complete(job_id):
                    completed.add(run_dir)
                    print(f"  Job {job_id} completed ({len(completed)}/{len(job_ids)})")
            
            if time.time() - start_time > max_wait:
                print(f"Warning: Timeout waiting for jobs. {len(completed)}/{len(job_ids)} completed.")
                break
            
            if len(completed) < len(job_ids):
                time.sleep(30)  # Check every 30 seconds
        
        # Clear active jobs
        for run_dir in job_ids:
            self.active_jobs.pop(run_dir, None)
        
        # Wait a bit for file system sync
        print("Waiting for file system sync...")
        time.sleep(5)
        
        # Parse results
        results = []
        failed_runs = []
        
        for run_dir in run_dirs:
            if run_dir not in completed:
                print(f"  Warning: {run_dir} did not complete")
                failed_runs.append(run_dir)
                continue
            
            # Wait for OUTCAR to be complete
            print(f"  Checking {os.path.basename(run_dir)}...", end='', flush=True)
            wait_count = 0
            while not self._is_calculation_complete(run_dir) and wait_count < 60:
                time.sleep(2)
                wait_count += 1
            
            if not self._is_calculation_complete(run_dir):
                print(" incomplete")
                failed_runs.append(run_dir)
                continue
            
            try:
                from vasp_manager import VASPRun
                vr = VASPRun(run_dir)
                vr.run_dir = run_dir
                results.append(vr)
                print(" ✓")
            except Exception as e:
                print(f" ✗ Parse error: {e}")
                failed_runs.append(run_dir)
        
        if failed_runs:
            print(f"\nWarning: {len(failed_runs)} calculations failed or incomplete")
        
        # Clear batch info
        self.current_thermal_batch_info = None
        self.current_thermal_batch_dirs = []
        
        print(f"\nSuccessfully parsed {len(results)}/{len(run_dirs)} results")
        
        return results
    
    def _create_job_script(self, run_dir: str, run_type: str):
        """Create SLURM job script for Anvil."""
        if run_type == 'thermal':
            # Thermal snapshots - quick calculation
            walltime = "00:30:00"
            job_name = f"snap_{os.path.basename(run_dir)}"
        else:
            # Dimer runs - longer calculation
            walltime = "04:00:00"
            job_name = f"dimer_{os.path.basename(run_dir)}"
        
        script_content = f"""#!/bin/bash
    #SBATCH -A MAT200013
    #SBATCH --nodes=1
    #SBATCH --ntasks=128
    #SBATCH --time={walltime}
    #SBATCH -J {job_name}
    #SBATCH -o myjob.o%j
    #SBATCH -e myjob.e%j
    #SBATCH -p shared

    # Debug info
    echo "Job started on $(date)"
    echo "Running on node: $(hostname)"
    echo "Working directory: $(pwd)"

    # List input files
    echo "Input files:"
    ls -la

    # Load modules - FIXED FOR ANVIL
    module purge
    module load gcc/11.2.0 openmpi/4.0.6 hdf5/1.10.7
    module load vasp/5.4.4.pl2
    module list

    # Run VASP
    echo "Starting VASP..."
    srun -n $SLURM_NTASKS --kill-on-bad-exit vasp_gam

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "VASP completed successfully"
    else
        echo "VASP failed with exit code $?"
    fi

    echo "Job completed on $(date)"
    """
        
        job_script = os.path.join(run_dir, 'job.sh')
        with open(job_script, 'w') as f:
            f.write(script_content)
        os.chmod(job_script, 0o755)
    
    def _is_slurm_job_complete(self, job_id: str) -> bool:
        """Check if a SLURM job has completed."""
        try:
            result = subprocess.run(['squeue', '-j', job_id, '-h'],
                                  capture_output=True, text=True)
            # If job is not in queue, it's complete
            return len(result.stdout.strip()) == 0
        except:
            return True

    def _is_calculation_complete(self, run_dir: str) -> bool:
        """Check if VASP calculation has completed successfully."""
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        if not os.path.exists(outcar_path):
            return False

        try:
            with open(outcar_path, 'r') as f:
                outcar_content = f.read()
                return 'General timing and accounting informations' in outcar_content
        except:
            return False


class VASPLocalExecutor(VASPExecutor):
    """GPU-aware executor for running VASP locally using mpirun.

    Distributes VASP calculations across available GPUs:
    - Thermal snapshots: 1 job per GPU in parallel (1 MPI rank each).
    - Dimer/main runs: all GPUs together (1 MPI rank per GPU).
    No MPS daemon is used – each VASP process owns its GPU exclusively.
    """

    # --- Paths ---------------------------------------------------------------
    MPIRUN_PATH = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun"
    OPAL_PREFIX = "/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/comm_libs/12.3/openmpi4/openmpi-4.1.5"
    VASP_DIR = "/opt/vasp/vasp.6.3.0/bin"

    # --- GPU + static-calc INCAR overrides -----------------------------------
    GPU_INCAR_SETTINGS = {
        'ALGO': 'Normal',       # Blocked Davidson – most stable on GPU
        'LREAL': 'Auto',        # Real-space projection
        'NCORE': '1',           # 1 core per orbital band for GPU
        'NSIM': '8',            # Bands optimised simultaneously
    }

    # --- Files cleaned before a retry ----------------------------------------
    RETRY_CLEANUP_FILES = [
        'WAVECAR', 'CHGCAR', 'CHG', 'vasprun.xml', 'OUTCAR',
        'OSZICAR', 'CONTCAR', 'EIGENVAL', 'DOSCAR', 'PCDAT',
        'XDATCAR', 'IBZKPT', 'REPORT', 'vasp.out', 'vasp.err',
        'job_info.json',
    ]

    MAX_RETRIES = 3

    def __init__(self,
                 vasp_command: str = "vasp_gam",
                 mpi_command: Optional[str] = None,
                 num_mpi_ranks: int = 1,
                 user_poscar_path: Optional[str] = None,
                 gpu_ids: Optional[List[int]] = None,
                 batch_size: Optional[int] = None):
        self.vasp_command = vasp_command
        self.user_poscar_path = user_poscar_path
        self.active_jobs: Dict[str, str] = {}

        # ---- Locate binaries ------------------------------------------------
        self.vasp_path = self._find_binary(
            vasp_command,
            [f"{self.VASP_DIR}/{vasp_command}",
             f"/opt/vasp/bin/{vasp_command}",
             f"/usr/local/bin/{vasp_command}"])
        self.mpi_path = mpi_command or self._find_binary(
            "mpirun",
            [self.MPIRUN_PATH, "/usr/local/bin/mpirun"])

        # ---- Detect GPUs ----------------------------------------------------
        if gpu_ids is not None:
            self.gpu_ids = list(gpu_ids)
        else:
            self.gpu_ids = self._detect_gpus()
        self.num_gpus = len(self.gpu_ids)

        # ---- MPI config: 1 rank per GPU (standard for GPU VASP) -------------
        if self.num_gpus > 0:
            self.batch_size = batch_size if batch_size else 50
            self.ranks_per_job = 1          # 1 MPI rank per GPU
        else:
            self.batch_size = 1
            self.ranks_per_job = num_mpi_ranks

        # ---- Round-robin counter for single-GPU job assignment ---------------
        self._rr_counter = 0

        # ---- Kill any stale MPS daemon that could block CUDA ----------------
        self._kill_stale_mps()

        # ---- Summary --------------------------------------------------------
        print(f"VASPLocalExecutor initialised:")
        print(f"  VASP binary : {self.vasp_path}")
        print(f"  MPI command : {self.mpi_path}")
        print(f"  GPUs        : {self.num_gpus}  {self.gpu_ids}")
        print(f"  Batch size  : {self.batch_size} (parallel thermal jobs)")
        print(f"  MPI ranks/job: {self.ranks_per_job}")

    # =========================================================================
    # Binary / GPU detection
    # =========================================================================
    @staticmethod
    def _find_binary(name: str, search_paths: List[str]) -> str:
        for p in search_paths:
            if os.path.exists(p):
                return p
        try:
            r = subprocess.run(['which', name], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
        print(f"  WARNING: Could not locate {name}, will rely on PATH")
        return name

    @staticmethod
    def _detect_gpus() -> List[int]:
        """Detect available NVIDIA GPUs via nvidia-smi."""
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                ids = [int(x.strip()) for x in r.stdout.strip().split('\n')]
                print(f"  Detected {len(ids)} GPU(s): {ids}")
                return ids
        except Exception:
            pass
        print("  No GPUs detected – falling back to CPU-only execution")
        return []

    @staticmethod
    def _kill_stale_mps() -> None:
        """Kill any leftover MPS daemon that can block CUDA init."""
        try:
            subprocess.run(['pkill', '-9', '-f', 'nvidia-cuda-mps'],
                           capture_output=True, timeout=5)
        except Exception:
            pass

    # =========================================================================
    # INCAR preparation for GPU
    # =========================================================================
    def _prepare_incar_for_gpu(self, run_dir: str) -> None:
        """Add GPU-optimal settings to the INCAR copy in *run_dir*.

        Only touches ALGO, LREAL, NCORE, NSIM — all other user-provided
        INCAR tags are left exactly as-is.
        """
        incar_path = os.path.join(run_dir, 'INCAR')
        if not os.path.exists(incar_path):
            return

        with open(incar_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        updated_keys: set = set()

        for line in lines:
            stripped = line.upper().strip()
            if not stripped or stripped.startswith('!') or stripped.startswith('#'):
                new_lines.append(line)
                continue
            replaced = False
            for key, value in self.GPU_INCAR_SETTINGS.items():
                parts = stripped.replace('=', ' ').split()
                if parts and parts[0] == key:
                    new_lines.append(f'{key} = {value}\n')
                    updated_keys.add(key)
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)

        # Append any GPU settings not already present
        missing = set(self.GPU_INCAR_SETTINGS) - updated_keys
        if missing:
            new_lines.append('\n! GPU settings (added automatically)\n')
            for key in sorted(missing):
                new_lines.append(f'{key} = {self.GPU_INCAR_SETTINGS[key]}\n')

        with open(incar_path, 'w') as f:
            f.writelines(new_lines)

    # =========================================================================
    # Build per-job environment (no MPS)
    # =========================================================================
    def _make_env(self, gpu_id: Optional[int] = None,
                  all_gpus: bool = False) -> Dict[str, str]:
        """Return a copy of os.environ with GPU / MPI vars set."""
        env = os.environ.copy()
        # Remove any MPS variables to avoid stale-daemon issues
        env.pop('CUDA_MPS_PIPE_DIRECTORY', None)
        env.pop('CUDA_MPS_LOG_DIRECTORY', None)
        if all_gpus:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in self.gpu_ids)
        elif gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['OMP_NUM_THREADS'] = '1'
        env['OPAL_PREFIX'] = self.OPAL_PREFIX
        env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
        env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
        return env

    # =========================================================================
    # Async process helpers
    # =========================================================================
    def _start_vasp_async(self, run_dir: str, gpu_id: int) -> subprocess.Popen:
        """Launch VASP asynchronously on a single GPU, return Popen handle."""
        self._prepare_incar_for_gpu(run_dir)
        env = self._make_env(gpu_id=gpu_id)
        cmd = f"{self.mpi_path} -np {self.ranks_per_job} {self.vasp_path}"

        out_f = open(os.path.join(run_dir, 'vasp.out'), 'w')
        err_f = open(os.path.join(run_dir, 'vasp.err'), 'w')

        proc = subprocess.Popen(cmd, shell=True, cwd=run_dir,
                                stdout=out_f, stderr=err_f, env=env)
        proc._out_file = out_f
        proc._err_file = err_f
        proc._run_dir = run_dir
        proc._gpu_id = gpu_id
        proc._start_time = time.time()
        return proc

    def _finish_vasp_async(self, proc: subprocess.Popen) -> str:
        """Wait for an async VASP process and return 'COMPLETED' or 'FAILED'."""
        proc.wait()
        proc._out_file.close()
        proc._err_file.close()
        elapsed = time.time() - proc._start_time
        status = "COMPLETED" if proc.returncode == 0 else "FAILED"
        print(f"    GPU {proc._gpu_id} | {os.path.basename(proc._run_dir)} "
              f"| {status} ({elapsed:.0f}s)")
        self.active_jobs[proc._run_dir] = status.lower()

        import json
        info = {"status": status, "run_dir": proc._run_dir,
                "gpu_id": proc._gpu_id, "mpi_ranks": self.ranks_per_job,
                "elapsed_seconds": elapsed}
        with open(os.path.join(proc._run_dir, 'job_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        return status

    # =========================================================================
    # Run VASP on a single GPU (blocking, round-robin assignment)
    # =========================================================================
    def _run_vasp_single_gpu(self, run_dir: str) -> str:
        """Run a single VASP job on one GPU (blocking), assigned round-robin."""
        self._prepare_incar_for_gpu(run_dir)
        gpu_id = self.gpu_ids[self._rr_counter % self.num_gpus] if self.num_gpus else 0
        self._rr_counter += 1
        env = self._make_env(gpu_id=gpu_id)
        cmd = f"{self.mpi_path} -np {self.ranks_per_job} {self.vasp_path}"

        print(f"  Running VASP on GPU {gpu_id} ({self.ranks_per_job} rank): "
              f"{os.path.basename(run_dir)}")
        out_f = os.path.join(run_dir, 'vasp.out')
        err_f = os.path.join(run_dir, 'vasp.err')
        with open(out_f, 'w') as out, open(err_f, 'w') as err:
            p = subprocess.Popen(cmd, shell=True, cwd=run_dir,
                                 stdout=out, stderr=err, env=env)
            p.wait()

        status = "COMPLETED" if p.returncode == 0 else "FAILED"
        if status == "FAILED":
            msg = ""
            try:
                with open(err_f) as f:
                    msg = f.read()[:500]
            except Exception:
                pass
            print(f"  WARNING: VASP {status} (rc={p.returncode}) in {run_dir}")
            if msg:
                print(f"  stderr: {msg}")
        else:
            print(f"  VASP completed on GPU {gpu_id}: {os.path.basename(run_dir)}")
        self.active_jobs[run_dir] = status.lower()
        return status

    # =========================================================================
    # Parallel GPU batch (for thermal snapshots)
    # =========================================================================
    def _run_parallel_on_gpus(self, run_dirs: List[str],
                              retry_attempt: int = 0) -> Dict[str, dict]:
        """Run *run_dirs* in parallel batches across GPUs."""
        results: Dict[str, dict] = {}

        for batch_start in range(0, len(run_dirs), self.batch_size):
            batch = run_dirs[batch_start:batch_start + self.batch_size]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(run_dirs) + self.batch_size - 1) // self.batch_size
            print(f"\n  Batch {batch_num}/{total_batches}: "
                  f"{len(batch)} jobs on {self.num_gpus} GPU(s)")

            # Assign GPUs round-robin and launch
            active: List[subprocess.Popen] = []
            for i, rd in enumerate(batch):
                gpu_id = self.gpu_ids[i % self.num_gpus] if self.num_gpus else 0
                proc = self._start_vasp_async(rd, gpu_id)
                active.append(proc)

            # Poll until all are done
            while active:
                still_running = []
                for proc in active:
                    if proc.poll() is None:
                        still_running.append(proc)
                    else:
                        st = self._finish_vasp_async(proc)
                        results[proc._run_dir] = {'status': st,
                                                   'run_dir': proc._run_dir}
                active = still_running
                if active:
                    time.sleep(5)

        # ---- Retry failed jobs -----------------------------------------------
        failed = [rd for rd, info in results.items() if info['status'] == 'FAILED']
        missing = [rd for rd in run_dirs if rd not in results]
        to_retry = failed + missing

        if to_retry and retry_attempt < self.MAX_RETRIES:
            print(f"\n  Retrying {len(to_retry)} failed jobs "
                  f"(attempt {retry_attempt + 1}/{self.MAX_RETRIES})")
            for rd in to_retry:
                self._cleanup_failed_run(rd)
            retry_results = self._run_parallel_on_gpus(to_retry,
                                                       retry_attempt + 1)
            results.update(retry_results)
        elif to_retry:
            print(f"  WARNING: {len(to_retry)} jobs still failed after "
                  f"{self.MAX_RETRIES} retries")

        return results

    def _cleanup_failed_run(self, run_dir: str) -> None:
        """Remove VASP output files before retrying."""
        for fname in self.RETRY_CLEANUP_FILES:
            fpath = os.path.join(run_dir, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    # =========================================================================
    # VASPExecutor interface
    # =========================================================================
    def submit_job(self, run_dir: str, run_type: str) -> None:
        """Submit a VASP job.

        Thermal runs are queued for parallel batch execution.
        Main/dimer runs are executed immediately using all GPUs.
        """
        if run_type == 'thermal':
            self.active_jobs[run_dir] = 'queued'
            return None
        print(f"\nRunning dimer VASP calculation in {run_dir}")
        self._run_vasp_single_gpu(run_dir)
        return None

    def wait_for_completion(self, run_dir: str, job_info: Any = None,
                            **kwargs) -> Any:
        """Return parsed results (job already finished)."""
        if not self._is_calculation_complete(run_dir):
            raise RuntimeError(
                f"VASP calculation not complete in {run_dir}. "
                f"Check vasp.err for details.")
        from vasp_manager import VASPRun
        return VASPRun(run_dir)

    def submit_batch(self, run_dirs: List[str]) -> Dict[str, dict]:
        """Execute thermal snapshots in GPU-parallel batches."""
        if not run_dirs:
            return {}

        print(f"\nRunning {len(run_dirs)} VASP thermal calculations "
              f"across {self.num_gpus} GPU(s) (batch_size={self.batch_size})")
        results = self._run_parallel_on_gpus(run_dirs)

        ok = sum(1 for v in results.values() if v['status'] == 'COMPLETED')
        print(f"\nThermal batch done: {ok}/{len(run_dirs)} succeeded")
        return results

    def wait_for_batch(self, batch_info: Any, run_dirs: List[str]) -> List[Any]:
        """Parse VASPRun objects from completed calculations."""
        results = []
        failed_runs = []

        for run_dir in run_dirs:
            if self._is_calculation_complete(run_dir):
                try:
                    from vasp_manager import VASPRun
                    vr = VASPRun(run_dir)
                    vr.run_dir = run_dir
                    results.append(vr)
                except Exception as e:
                    print(f"  Failed to parse {run_dir}: {e}")
                    failed_runs.append(run_dir)
            else:
                print(f"  Incomplete: {run_dir}")
                failed_runs.append(run_dir)

        if failed_runs:
            print(f"\nWARNING: {len(failed_runs)}/{len(run_dirs)} "
                  f"calculations failed or incomplete")

        print(f"Successfully parsed {len(results)}/{len(run_dirs)} VASP results")
        return results

    # =========================================================================
    # Completion check
    # =========================================================================
    def _is_calculation_complete(self, run_dir: str) -> bool:
        outcar = os.path.join(run_dir, 'OUTCAR')
        if not os.path.exists(outcar):
            return False
        try:
            with open(outcar, 'r') as f:
                txt = f.read()
                return ('General timing and accounting informations' in txt
                        or 'Total CPU time used' in txt)
        except Exception:
            return False