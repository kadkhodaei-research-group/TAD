#!/bin/bash

# Sequential temperature study launcher for ZrO2 (Zirconia) system
# Runs one study:
#   outputs_50snapshots_10repititions: 50 snapshots, 10 repetitions
# Uses VASP for energy/force calculations (not EAM/MEAM)

# Resolve paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUTS_DIR="${BASE_DIR}/inputs"
ZRO2_INPUT_DIR="${INPUTS_DIR}/zro2_inputs_td_spf"
VASP_INPUT_DIR="${ZRO2_INPUT_DIR}/ZrO2_vasp_input_files"

echo "========================================================================"
echo "SEQUENTIAL ZrO2 TEMPERATURE STUDY LAUNCHER"
echo "========================================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory:   ${BASE_DIR}"
echo "Inputs directory: ${INPUTS_DIR}"
echo "ZrO2 inputs:      ${ZRO2_INPUT_DIR}"
echo "VASP inputs:      ${VASP_INPUT_DIR}"
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using GPU device(s): ${CUDA_VISIBLE_DEVICES}"
fi
echo ""

# Configuration
PARALLEL_RUNS=1  # One run at a time (VASP is resource-intensive)

# Temperature array for ZrO2
ZRO2_TEMPS=(2600 2800 2900 3000)

# ============================================================================
# FUNCTION: Run ZrO2 temperature study
# ============================================================================
run_zro2_study() {
    local OUTPUT_DIR=$1
    local NUM_SNAPSHOTS=$2
    local REPETITIONS=$3

    local TIMESTAMP=$(date +%s)
    local STUDY_OUTPUT="${BASE_DIR}/${OUTPUT_DIR}/zro2/temperature_study_${TIMESTAMP}"

    echo ""
    echo "========================================================================"
    echo "STARTING ZrO2 STUDY: ${OUTPUT_DIR}"
    echo "  Snapshots: ${NUM_SNAPSHOTS}"
    echo "  Repetitions: ${REPETITIONS}"
    echo "  Temperatures: ${ZRO2_TEMPS[*]}"
    echo "  Output: ${STUDY_OUTPUT}"
    echo "========================================================================"

    # Create output directory
    mkdir -p "${STUDY_OUTPUT}"

    # Build and export hyperparameters with the specified num-snapshots
    # NOTE: No --parallel-eam, no --kim-model (using VASP, not EAM/MEAM)
    export HYPERPARAMS="--num-snapshots ${NUM_SNAPSHOTS} --max-inner-iterations 50 --max-dimer-steps 100 --relaxed-saddle-criteria 0.05 --dimer-stopping-criteria 0.05 --enable-trust-region --trust-region-initial 0.5 --trust-region-dramatic-threshold 0.1 --trust-region-expand-threshold 0.5 --trust-region-dramatic-factor 0.5 --trust-region-expand-factor 1.2 --step-size 0.1 --max-step-size 0.1 --enable-oscillation-detection --validate-parameters --gpu --no-gpu-fallback --verbose"
    export STUDY_OUTPUT
    export SCRIPT_DIR ZRO2_INPUT_DIR VASP_INPUT_DIR

    # Function to run a single ZrO2 simulation
    run_zro2() {
        local temp=$1
        local rep=$2
        local output_dir="${STUDY_OUTPUT}/T_${temp}K/run_$(printf "%03d" $rep)"

        cd "${SCRIPT_DIR}"
        python run_tad.py \
            --poscar-file "${ZRO2_INPUT_DIR}/${temp}_defected/POSCAR_ss_${temp}" \
            --force-constants-file "${ZRO2_INPUT_DIR}/${temp}_defected/FORCE_CONSTANTS" \
            --execution-mode vasp \
            --vasp-command vasp_std \
            --vasp-input-dir "${VASP_INPUT_DIR}" \
            --moving-indices 75 \
            --orient-atom-direction 75:0,0,1 \
            --temperature ${temp} \
            ${HYPERPARAMS} \
            --output-dir "${output_dir}" \
            > "${output_dir}.log" 2>&1
    }

    # Export function
    export -f run_zro2

    # Export CUDA_VISIBLE_DEVICES if set
    if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
        export CUDA_VISIBLE_DEVICES
    fi

    # Generate commands
    local commands_file="/tmp/zro2_commands_${TIMESTAMP}.txt"
    > "$commands_file"

    for temp in "${ZRO2_TEMPS[@]}"; do
        mkdir -p "${STUDY_OUTPUT}/T_${temp}K"
        for ((rep=1; rep<=REPETITIONS; rep++)); do
            echo "run_zro2 ${temp} ${rep}" >> "$commands_file"
        done
    done

    # Count total runs
    local TOTAL_RUNS=$((${#ZRO2_TEMPS[@]} * REPETITIONS))

    echo "------------------------------------------------------------------------"
    echo "ZrO2 runs: ${TOTAL_RUNS} (${#ZRO2_TEMPS[@]} temperatures x ${REPETITIONS} repetitions)"
    echo "Parallel workers: ${PARALLEL_RUNS}"
    echo "------------------------------------------------------------------------"
    echo ""
    echo "Starting ZrO2 execution..."
    echo ""

    # Run all commands with PARALLEL_RUNS=1 (sequential for VASP)
    cat "$commands_file" | xargs -I {} -P ${PARALLEL_RUNS} bash -c "{}"

    # Clean up
    rm -f "$commands_file"

    echo ""
    echo "========================================================================"
    echo "ZrO2 STUDY COMPLETE: ${OUTPUT_DIR}"
    echo "Results: ${STUDY_OUTPUT}"
    echo "========================================================================"
}

# Export the temperatures array for use in functions
export ZRO2_TEMPS

# ============================================================================
# RUN THE STUDY (ZrO2 only, 50 snapshots, 10 repetitions)
# ============================================================================

echo ""
echo "Will run 1 study:"
echo "  1) outputs_50snapshots_10repititions"
echo "     - ZrO2: 50 snapshots, 10 reps, temps: ${ZRO2_TEMPS[*]}"
echo ""

# ============================================================================
# Study: 50 snapshots, 10 repetitions
# ============================================================================
run_zro2_study "outputs_50snapshots_10repititions" 50 10

echo ""
echo "========================================================================"
echo "ZrO2 STUDY COMPLETE"
echo "========================================================================"
echo "Results location:"
echo ""
echo "Study (50 snapshots, 10 reps):"
echo "  - ${BASE_DIR}/outputs_50snapshots_10repititions/zro2/"
echo "========================================================================"
