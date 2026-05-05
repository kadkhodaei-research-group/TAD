#!/bin/bash

# Run Zr temperature study only
# This script launches Zr runs from the NEW_v54 scripts directory

echo "========================================================================"
echo "LAUNCHING Zr TEMPERATURE STUDY"
echo "========================================================================"

# Configuration
PARALLEL_RUNS=10  # Number of simultaneous runs
TIMESTAMP=$(date +%s)

# Base directories
ZR_BASE="/workspace/TD_SPF/real_system/NEW_v54"
FC_DIR="/workspace/TD_SPF/real_system/FCs"

# Output directories
ZR_OUTPUT="${ZR_BASE}/outputs/zr/temperature_study_${TIMESTAMP}"

# Create output directories
mkdir -p "${ZR_OUTPUT}"

# Study parameters
ZR_TEMPS=(1300 1400 1500 1600 1700 1800)
REPETITIONS=100

# Hyperparameters
HYPERPARAMS="--num-snapshots 100 --max-inner-iterations 50 --max-dimer-steps 100 \
--relaxed-saddle-criteria 0.01 --dimer-stopping-criteria 0.01 \
--enable-trust-region --trust-region-initial 0.5 \
--trust-region-dramatic-threshold 0.1 --trust-region-expand-threshold 0.5 \
--trust-region-dramatic-factor 0.5 --trust-region-expand-factor 1.2 \
--step-size 0.1 --max-step-size 0.1 \
--enable-oscillation-detection --validate-parameters \
--parallel-eam --gpu --no-gpu-fallback --verbose"

echo "Zr study output: ${ZR_OUTPUT}"
echo "Parallel runs: ${PARALLEL_RUNS}"
echo "------------------------------------------------------------------------"

# Function to run a single Zr simulation
run_zr() {
    local temp=$1
    local rep=$2
    local output_dir="${ZR_OUTPUT}/T_${temp}K/run_$(printf "%03d" $rep)"
    
    cd "${ZR_BASE}/scripts"
    python run_dual_gp_improved.py \
        --poscar-file ../inputs/POSCAR_Zr \
        --force-constants-file "${FC_DIR}/FORCE_CONSTANTS_EAM_Zr_defected_${temp}" \
        --execution-mode eam \
        --eam-potential-file potentials/Cu-Zr_4.eam.fs \
        --moving-indices 214 \
        --temperature ${temp} \
        ${HYPERPARAMS} \
        --output-dir "${output_dir}" \
        > "${output_dir}.log" 2>&1
}


# Export functions for parallel execution
export -f run_zr
export ZR_BASE FC_DIR ZR_OUTPUT HYPERPARAMS

# Export CUDA_VISIBLE_DEVICES if set
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES
    echo "Using GPU device: $CUDA_VISIBLE_DEVICES"
fi

# Generate Zr commands
echo "Generating Zr runs..."
> /tmp/zr_commands_${TIMESTAMP}.txt
for temp in "${ZR_TEMPS[@]}"; do
    mkdir -p "${ZR_OUTPUT}/T_${temp}K"
    for ((rep=1; rep<=REPETITIONS; rep++)); do
        echo "run_zr ${temp} ${rep}" >> /tmp/zr_commands_${TIMESTAMP}.txt
    done
done

# Count total runs
ZR_TOTAL=$((${#ZR_TEMPS[@]} * REPETITIONS))

echo "------------------------------------------------------------------------"
echo "Zr runs: ${ZR_TOTAL} (${#ZR_TEMPS[@]} temperatures × ${REPETITIONS} repetitions)"
echo "Total runs: ${ZR_TOTAL}"
echo "========================================================================"
echo ""
echo "Starting parallel execution..."
echo ""

# Run all commands in parallel with proper environment
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    cat /tmp/zr_commands_${TIMESTAMP}.txt | \
        xargs -I {} -P ${PARALLEL_RUNS} bash -c "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; {}"
else
    cat /tmp/zr_commands_${TIMESTAMP}.txt | \
        xargs -I {} -P ${PARALLEL_RUNS} bash -c "{}"
fi

# Clean up
rm -f /tmp/zr_commands_${TIMESTAMP}.txt

echo ""
echo "========================================================================"
echo "ALL RUNS COMPLETE"
echo "========================================================================"
echo "Zr results: ${ZR_OUTPUT}"
echo "========================================================================"