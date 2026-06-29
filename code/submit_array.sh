#!/bin/bash
# This script submits a SLURM job array for VASP calculations
# Usage: ./submit_array.sh <num_jobs> <base_dir> <job_type>

NUM_JOBS=$1
BASE_DIR=$2
JOB_TYPE=$3  # "thermal" or "dimer"

if [ "$JOB_TYPE" == "thermal" ]; then
    TIME="00:30:00"
    JOB_NAME="snap"
else
    TIME="02:00:00"
    JOB_NAME="dimer"
fi

# Submit the array job and capture the job ID
sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_array
#SBATCH --array=1-${NUM_JOBS}
#SBATCH -N 1
#SBATCH -p skx
#SBATCH -t ${TIME}
#SBATCH -n 48
#SBATCH -A TG-MAT200013
#SBATCH --output=${BASE_DIR}/run_%a.out
#SBATCH --error=${BASE_DIR}/error_%a.err

# Load VASP module
module load vasp

# Navigate to the specific directory for this array task
cd ${BASE_DIR}/\${SLURM_ARRAY_TASK_ID}

# Run VASP
ibrun vasp_gam > vasp.out
EOF
