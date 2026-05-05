#!/bin/bash
# FILENAME: job_anvil_continuation.sh
#SBATCH -A MAT200013         # Allocation name
#SBATCH --nodes=1            # Controller node
#SBATCH --ntasks=32          # Controller doesn't need many cores
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
module --force purge
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

# Check if this is a continuation run
CHECKPOINT_FILE="checkpoints/checkpoint_latest.pkl"
CONTINUATION_FLAG=""

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Found checkpoint file. This will be a continuation run."
    CONTINUATION_FLAG="--continuation"
    
    # Extract iteration number from checkpoint
    ITERATION=$(python -c "
import pickle
with open('$CHECKPOINT_FILE', 'rb') as f:
    checkpoint = pickle.load(f)
    print(checkpoint['iteration'])
" 2>/dev/null)
    
    echo "Continuing from iteration: $ITERATION"
else
    echo "No checkpoint found. Starting fresh run."
fi

# Run the main script
echo "Starting TD-SPF calculation..."
python run_tdspf.py \
    --poscar-file POSCAR \
    --force-constants-file FORCE_CONSTANTS \
    --execution-mode anvil \
    --temperature 1400 \
    --moving-indices 214 \
    --max-dimer-steps 50 \
    --number-of-thermal-snapshots 50 \
    --delta-of-initial-gp2-training-points 0.05 \
    --number-of-initial-gp2-training-points 5 \
    --activation-radius 7.5 \
    $CONTINUATION_FLAG \
    --verbose

# Check exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
    echo "Job completed successfully on $(date)"
else
    echo "Job failed with exit status $EXIT_STATUS on $(date)"
    
    # Check if we can continue in a new job
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "Checkpoint exists. You can resubmit this job to continue."
    fi
fi
