#!/bin/bash
#SBATCH -A MAT200013      # Allocation name 
#SBATCH --nodes=1         # Total # of nodes
#SBATCH --ntasks=32      # Total # of MPI tasks
#SBATCH --time=00:40:00   # Total run time limit (hh:mm:ss)
#SBATCH -J vasp_run       # Job name
#SBATCH -o myjob.o%j      # Name of stdout output file
#SBATCH -e myjob.e%j      # Name of stderr error file
#SBATCH -p shared         # Queue (partition) name

# Manage processing environment, load compilers and applications.
module --force purge
module load modtree/cpu
module load gcc/11.2.0
module load openmpi/4.1.6
module load vasp/6.3.0

# Launch MPI code
srun -n $SLURM_NTASKS vasp_gam