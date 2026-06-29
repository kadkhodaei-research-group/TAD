#!/bin/bash
#SBATCH -N 1
##SBATCH -p RM
##SBATCH --ntasks-per-node=128

#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=64

#SBATCH -t 23:00:00
#SBATCH -o run1.out
#SBATCH -e error.err


ulimit -s unlimited
#module load intel/20.4 intelmpi/20.4-intel20.4 VASP/6.3+vtst-intel
#module load intelmpi gcc/10.2.0 VASP/6.3+vtst-intel
export OMP_NUM_THREADS=1

module purge
module load intel/2021.10.0
module load intelmpi/2021.10.0
module load VASP/6.3+vtst-intel

# enable conda in non interactive shell
source /jet/home/sfatta2/anaconda3/etc/profile.d/conda.sh   # adjust path to your conda
conda activate /ocean/projects/mat200005p/sfatta2/td_spf_py311_env

# run the job 
python run_tdspf.py \
    --poscar-file POSCAR_interpolated \
    --force-constants-file FORCE_CONSTANTS \
    --execution-mode slurm \
    --temperature 1400 \
    --moving-indices 180 \
    --max-dimer-steps 50 \
    --number-of-thermal-snapshots 50 \
    --delta-of-initial-gp2-training-points 0.05 \
    --number-of-initial-gp2-training-points 5 \
    --activation-radius 7.5 \
    --verbose