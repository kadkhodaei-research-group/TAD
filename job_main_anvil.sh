#!/bin/bash
# FILENAME: job_anvil.sh
#SBATCH -A MAT200013         # Allocation name
#SBATCH --nodes=1            # Controller node
#SBATCH --ntasks=32           # Controller doesn't need many cores
#SBATCH --time=23:00:00      # Total run time limit
#SBATCH -J TD_SPF_Controller # Job name
#SBATCH -o td_spf_%j.out     # stdout output file
#SBATCH -e td_spf_%j.err     # stderr error file
#SBATCH -p shared            # Queue (partition) name
  
# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
# Load required modules
module --force purge  # Force purge to remove sticky modules
module load modtree/cpu
module load gcc/11.2.0
module load openmpi/4.1.6
module load vasp/6.3.0
# Show loaded modules for verification
echo "Loaded modules:"
module list
# Set up Python environment
source /anvil/scratch/x-sfatta2/anaconda3/etc/profile.d/conda.sh
conda activate td_spf_py311
# Verify Python and key packages
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
# Change to working directory
cd /anvil/scratch/x-sfatta2/TD_SPF/real_system


# Run the main script
echo "Starting TD-SPF calculation..."
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
