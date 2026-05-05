#!/bin/bash

# Sequential temperature study launcher for ZrO2 (Zirconia) system - EAM mode
# Runs two studies:
#   1) outputs_50snapshots_200repititions_eam: 50 snapshots, 200 repetitions
#   2) outputs_100snapshots_100repititions_eam: 100 snapshots, 100 repetitions
# Uses EAM potential for energy/force calculations (not VASP)

# Resolve paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUTS_DIR="${BASE_DIR}/inputs"
ZRO2_INPUT_DIR="${INPUTS_DIR}/zro2_inputs_eam"

echo "========================================================================"
echo "SEQUENTIAL ZrO2 EAM TEMPERATURE STUDY LAUNCHER"
echo "========================================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory:   ${BASE_DIR}"
echo "Inputs directory: ${INPUTS_DIR}"
echo "ZrO2 inputs:      ${ZRO2_INPUT_DIR}"
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using GPU device(s): ${CUDA_VISIBLE_DEVICES}"
fi
echo ""

# Configuration
RUNS_PER_GPU=10  # Number of simultaneous runs per GPU

# Detect available GPUs
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
PARALLEL_RUNS=$((RUNS_PER_GPU * NUM_GPUS))
echo "Available GPUs: ${NUM_GPUS}"
echo "Runs per GPU: ${RUNS_PER_GPU}"
echo "Total parallel runs: ${PARALLEL_RUNS}"

# Temperature array for ZrO2
ZRO2_TEMPS=(2600 2800 2900 3000)

# ============================================================================
# FUNCTION: Run ZrO2 temperature study (EAM mode)
# ============================================================================
run_zro2_study() {
    local OUTPUT_DIR=$1
    local NUM_SNAPSHOTS=$2
    local REPETITIONS=$3

    local TIMESTAMP=$(date +%s)
    local STUDY_OUTPUT="${BASE_DIR}/${OUTPUT_DIR}/zro2/temperature_study_${TIMESTAMP}"

    echo ""
    echo "========================================================================"
    echo "STARTING ZrO2 EAM STUDY: ${OUTPUT_DIR}"
    echo "  Snapshots: ${NUM_SNAPSHOTS}"
    echo "  Repetitions: ${REPETITIONS}"
    echo "  Temperatures: ${ZRO2_TEMPS[*]}"
    echo "  Output: ${STUDY_OUTPUT}"
    echo "========================================================================"

    # Create output directory
    mkdir -p "${STUDY_OUTPUT}"

    # Build and export hyperparameters with the specified num-snapshots
    export HYPERPARAMS="--num-snapshots ${NUM_SNAPSHOTS} --max-inner-iterations 50 --max-dimer-steps 100 --relaxed-saddle-criteria 0.01 --dimer-stopping-criteria 0.01 --enable-trust-region --trust-region-initial 0.5 --trust-region-dramatic-threshold 0.1 --trust-region-expand-threshold 0.5 --trust-region-dramatic-factor 0.5 --trust-region-expand-factor 1.2 --step-size 0.1 --max-step-size 0.1 --enable-oscillation-detection --validate-parameters --parallel-eam --gpu --no-gpu-fallback --verbose"
    export STUDY_OUTPUT
    export SCRIPT_DIR ZRO2_INPUT_DIR

    # Function to run a single ZrO2 EAM simulation
    run_zro2() {
        local job_id=$1
        local temp=$2
        local rep=$3
        local gpu_id=$((job_id % NUM_GPUS))
        local output_dir="${STUDY_OUTPUT}/T_${temp}K/run_$(printf "%03d" $rep)"

        # Find the actual POSCAR_ss file (naming may not match temp exactly)
        local poscar_file=$(ls "${ZRO2_INPUT_DIR}/${temp}_defected"/POSCAR_ss_* 2>/dev/null | head -1)

        cd "${SCRIPT_DIR}"
        CUDA_VISIBLE_DEVICES=${gpu_id} python run_tad.py \
            --poscar-file "${poscar_file}" \
            --force-constants-file "${ZRO2_INPUT_DIR}/${temp}_defected/FORCE_CONSTANTS" \
            --execution-mode eam \
            --eam-potential-file potentials/UZrO.eam.alloy \
            --moving-indices 75 \
            --orient-atom-direction 75:0,0,1 \
            --temperature ${temp} \
            ${HYPERPARAMS} \
            --output-dir "${output_dir}" \
            > "${output_dir}.log" 2>&1
    }

    # Export function and GPU count
    export -f run_zro2
    export NUM_GPUS

    # Generate commands with job IDs for GPU round-robin
    local commands_file="/tmp/zro2_eam_commands_${TIMESTAMP}.txt"
    > "$commands_file"

    local job_id=0
    for temp in "${ZRO2_TEMPS[@]}"; do
        mkdir -p "${STUDY_OUTPUT}/T_${temp}K"
        for ((rep=1; rep<=REPETITIONS; rep++)); do
            echo "run_zro2 ${job_id} ${temp} ${rep}" >> "$commands_file"
            job_id=$((job_id + 1))
        done
    done

    # Count total runs
    local TOTAL_RUNS=$((${#ZRO2_TEMPS[@]} * REPETITIONS))

    echo "------------------------------------------------------------------------"
    echo "ZrO2 EAM runs: ${TOTAL_RUNS} (${#ZRO2_TEMPS[@]} temperatures x ${REPETITIONS} repetitions)"
    echo "Parallel workers: ${PARALLEL_RUNS}"
    echo "------------------------------------------------------------------------"
    echo ""
    echo "Starting ZrO2 EAM execution..."
    echo ""

    # Run all commands in parallel (EAM is lightweight)
    cat "$commands_file" | xargs -I {} -P ${PARALLEL_RUNS} bash -c "{}"

    # Clean up
    rm -f "$commands_file"

    echo ""
    echo "========================================================================"
    echo "ZrO2 EAM STUDY COMPLETE: ${OUTPUT_DIR}"
    echo "Results: ${STUDY_OUTPUT}"
    echo "========================================================================"
}

# Export the temperatures array for use in functions
export ZRO2_TEMPS

# ============================================================================
# RUN THE STUDIES
# ============================================================================

echo ""
echo "Will run 2 studies:"
echo "  1) outputs_50snapshots_200repititions_eam"
echo "     - ZrO2 EAM: 50 snapshots, 200 reps, temps: ${ZRO2_TEMPS[*]}"
echo "  2) outputs_100snapshots_100repititions_eam"
echo "     - ZrO2 EAM: 100 snapshots, 100 reps, temps: ${ZRO2_TEMPS[*]}"
echo ""

# ============================================================================
# Study 1: 50 snapshots, 200 repetitions
# ============================================================================
run_zro2_study "outputs_50snapshots_200repititions_eam" 50 200

# ============================================================================
# Study 2: 100 snapshots, 100 repetitions
# ============================================================================
run_zro2_study "outputs_100snapshots_100repititions_eam" 100 100

echo ""
echo "========================================================================"
echo "ALL ZrO2 EAM STUDIES COMPLETE"
echo "========================================================================"
echo "Results locations:"
echo ""
echo "Study 1 (50 snapshots, 200 reps):"
echo "  - ${BASE_DIR}/outputs_50snapshots_200repititions_eam/zro2/"
echo "Study 2 (100 snapshots, 100 reps):"
echo "  - ${BASE_DIR}/outputs_100snapshots_100repititions_eam/zro2/"
echo "========================================================================"
