#!/bin/bash
#SBATCH -N 1
#SBATCH -p skx-dev
#SBATCH -t 00:30:00
#SBATCH -n 48
#SBATCH -A TG-MAT200013
#SBATCH -J snap




module load vasp


# enable conda in non interactive shell
source /home1/07550/tg868495/anaconda3/etc/profile.d/conda.sh   # adjust path to your conda
conda activate /scratch/07550/tg868495/td_spf_py311_env

# run the job 
python run_tdspf.py \
    --poscar-file POSCAR \
    --force-constants-file FORCE_CONSTANTS \
    --execution-mode anvil \
    --temperature 1400 \
    --moving-indices 173 \
    --max-dimer-steps 50 \
    --number-of-thermal-snapshots 50 \
    --delta-of-initial-gp2-training-points 0.01 \
    --number-of-initial-gp2-training-points 5 \
    --verbose
