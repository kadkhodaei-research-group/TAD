#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=vasptest
#SBATCH -t 00:30:00
#SBATCH -o vasp_job.out
#SBATCH -e vasp_job.err

ulimit -s unlimited
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/packages/hdf5/hdf5-1.14.5/GNU/lib
echo "SLURM_NTASKS: " $SLURM_NTASKS

module purge
module load openmpi/5.0.3-gcc13.2.1
module load intel-mkl/2023.2.0 cuda/11.7.1

mpirun -np $SLURM_NTASKS /opt/packages/VASP/VASP6.5/6.5.1-gnu/vasp_gam
################