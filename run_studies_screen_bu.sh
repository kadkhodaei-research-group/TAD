#!/bin/bash

# Sequential temperature study launcher for both Zr and Mo
# Runs two studies:
# 1) outputs_100snapshots_100repititions: 100 snapshots, 100 repetitions
# 2) outputs_50snapshots_10repititions: 50 snapshots, 10 repetitions

# Resolve paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUTS_DIR="${BASE_DIR}/inputs"
FC_DIR="${INPUTS_DIR}/FCs"

echo "========================================================================"
echo "SEQUENTIAL Zr + Mo TEMPERATURE STUDIES LAUNCHER"
echo "========================================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory:   ${BASE_DIR}"
echo "Inputs directory: ${INPUTS_DIR}"
echo "FC directory:     ${FC_DIR}"
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using GPU device(s): ${CUDA_VISIBLE_DEVICES}"
fi
echo ""

# Configuration
PARALLEL_RUNS=10  # Number of simultaneous runs within each study

# Temperature arrays
ZR_TEMPS=(1300 1400 1500 1600 1700 1800)
MO_TEMPS=(200 300 1400)

# ============================================================================
# FUNCTION: Run Zr temperature study
# ============================================================================
run_zr_study() {
    local OUTPUT_DIR=$1
    local NUM_SNAPSHOTS=$2
    local REPETITIONS=$3

    local TIMESTAMP=$(date +%s)
    local STUDY_OUTPUT="${BASE_DIR}/${OUTPUT_DIR}/zr/temperature_study_${TIMESTAMP}"

    echo ""
    echo "========================================================================"
    echo "STARTING Zr STUDY: ${OUTPUT_DIR}"
    echo "  Snapshots: ${NUM_SNAPSHOTS}"
    echo "  Repetitions: ${REPETITIONS}"
    echo "  Temperatures: ${ZR_TEMPS[*]}"
    echo "  Output: ${STUDY_OUTPUT}"
    echo "========================================================================"

    # Create output directory
    mkdir -p "${STUDY_OUTPUT}"

    # Build and export hyperparameters with the specified num-snapshots
    export HYPERPARAMS="--num-snapshots ${NUM_SNAPSHOTS} --max-inner-iterations 50 --max-dimer-steps 100 --relaxed-saddle-criteria 0.01 --dimer-stopping-criteria 0.01 --enable-trust-region --trust-region-initial 0.5 --trust-region-dramatic-threshold 0.1 --trust-region-expand-threshold 0.5 --trust-region-dramatic-factor 0.5 --trust-region-expand-factor 1.2 --step-size 0.1 --max-step-size 0.1 --enable-oscillation-detection --validate-parameters --parallel-eam --gpu --no-gpu-fallback --verbose"
    export STUDY_OUTPUT
    export SCRIPT_DIR INPUTS_DIR FC_DIR

    # Function to run a single Zr simulation
    run_zr() {
        local temp=$1
        local rep=$2
        local output_dir="${STUDY_OUTPUT}/T_${temp}K/run_$(printf "%03d" $rep)"

        cd "${SCRIPT_DIR}"
        python run_tad.py \
            --poscar-file "${INPUTS_DIR}/POSCAR_Zr" \
            --force-constants-file "${FC_DIR}/FORCE_CONSTANTS_EAM_Zr_defected_${temp}" \
            --execution-mode eam \
            --eam-potential-file potentials/Zr_1.eam.fs \
            --moving-indices 214 \
            --temperature ${temp} \
            ${HYPERPARAMS} \
            --output-dir "${output_dir}" \
            > "${output_dir}.log" 2>&1
    }

    # Export function
    export -f run_zr

    # Export CUDA_VISIBLE_DEVICES if set
    if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
        export CUDA_VISIBLE_DEVICES
    fi

    # Generate commands
    local commands_file="/tmp/zr_commands_${TIMESTAMP}.txt"
    > "$commands_file"

    for temp in "${ZR_TEMPS[@]}"; do
        mkdir -p "${STUDY_OUTPUT}/T_${temp}K"
        for ((rep=1; rep<=REPETITIONS; rep++)); do
            echo "run_zr ${temp} ${rep}" >> "$commands_file"
        done
    done

    # Count total runs
    local TOTAL_RUNS=$((${#ZR_TEMPS[@]} * REPETITIONS))

    echo "------------------------------------------------------------------------"
    echo "Zr runs: ${TOTAL_RUNS} (${#ZR_TEMPS[@]} temperatures x ${REPETITIONS} repetitions)"
    echo "Parallel workers: ${PARALLEL_RUNS}"
    echo "------------------------------------------------------------------------"
    echo ""
    echo "Starting Zr parallel execution..."
    echo ""

    # Run all commands in parallel with proper environment
    cat "$commands_file" | xargs -I {} -P ${PARALLEL_RUNS} bash -c "{}"

    # Clean up
    rm -f "$commands_file"

    echo ""
    echo "========================================================================"
    echo "Zr STUDY COMPLETE: ${OUTPUT_DIR}"
    echo "Results: ${STUDY_OUTPUT}"
    echo "========================================================================"
}

# ============================================================================
# FUNCTION: Run Mo temperature study
# ============================================================================
run_mo_study() {
    local OUTPUT_DIR=$1
    local NUM_SNAPSHOTS=$2
    local REPETITIONS=$3

    local TIMESTAMP=$(date +%s)
    local STUDY_OUTPUT="${BASE_DIR}/${OUTPUT_DIR}/mo/temperature_study_${TIMESTAMP}"

    echo ""
    echo "========================================================================"
    echo "STARTING Mo STUDY: ${OUTPUT_DIR}"
    echo "  Snapshots: ${NUM_SNAPSHOTS}"
    echo "  Repetitions: ${REPETITIONS}"
    echo "  Temperatures: ${MO_TEMPS[*]}"
    echo "  Output: ${STUDY_OUTPUT}"
    echo "========================================================================"

    # Create output directory
    mkdir -p "${STUDY_OUTPUT}"

    # Build and export hyperparameters with the specified num-snapshots
    export HYPERPARAMS="--num-snapshots ${NUM_SNAPSHOTS} --max-inner-iterations 50 --max-dimer-steps 100 --relaxed-saddle-criteria 0.01 --dimer-stopping-criteria 0.01 --enable-trust-region --trust-region-initial 0.5 --trust-region-dramatic-threshold 0.1 --trust-region-expand-threshold 0.5 --trust-region-dramatic-factor 0.5 --trust-region-expand-factor 1.2 --step-size 0.1 --max-step-size 0.1 --enable-oscillation-detection --validate-parameters --parallel-eam --gpu --no-gpu-fallback --verbose"
    export STUDY_OUTPUT
    export SCRIPT_DIR INPUTS_DIR FC_DIR

    # Function to run a single Mo simulation
    run_mo() {
        local temp=$1
        local rep=$2
        local output_dir="${STUDY_OUTPUT}/T_${temp}K/run_$(printf "%03d" $rep)"

        cd "${SCRIPT_DIR}"
        python run_tad.py \
            --poscar-file "${INPUTS_DIR}/POSCAR_Mo" \
            --force-constants-file "${FC_DIR}/FORCE_CONSTANTS_EAM_Mo_defected_${temp}" \
            --execution-mode eam \
            --kim-model MEAM_LAMMPS_ParkFellingerLenosky_2012_Mo__MO_269937397263_002 \
            --orient-atom-direction 52:1,1,1 \
            --moving-indices 52 \
            --temperature ${temp} \
            ${HYPERPARAMS} \
            --output-dir "${output_dir}" \
            > "${output_dir}.log" 2>&1
    }

    # Export function
    export -f run_mo

    # Export CUDA_VISIBLE_DEVICES if set
    if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
        export CUDA_VISIBLE_DEVICES
    fi

    # Generate commands
    local commands_file="/tmp/mo_commands_${TIMESTAMP}.txt"
    > "$commands_file"

    for temp in "${MO_TEMPS[@]}"; do
        mkdir -p "${STUDY_OUTPUT}/T_${temp}K"
        for ((rep=1; rep<=REPETITIONS; rep++)); do
            echo "run_mo ${temp} ${rep}" >> "$commands_file"
        done
    done

    # Count total runs
    local TOTAL_RUNS=$((${#MO_TEMPS[@]} * REPETITIONS))

    echo "------------------------------------------------------------------------"
    echo "Mo runs: ${TOTAL_RUNS} (${#MO_TEMPS[@]} temperatures x ${REPETITIONS} repetitions)"
    echo "Parallel workers: ${PARALLEL_RUNS}"
    echo "------------------------------------------------------------------------"
    echo ""
    echo "Starting Mo parallel execution..."
    echo ""

    # Run all commands in parallel with proper environment
    cat "$commands_file" | xargs -I {} -P ${PARALLEL_RUNS} bash -c "{}"

    # Clean up
    rm -f "$commands_file"

    echo ""
    echo "========================================================================"
    echo "Mo STUDY COMPLETE: ${OUTPUT_DIR}"
    echo "Results: ${STUDY_OUTPUT}"
    echo "========================================================================"
}

# Export the temperatures arrays for use in functions
export ZR_TEMPS
export MO_TEMPS

# ============================================================================
# RUN THE TWO STUDIES SEQUENTIALLY (Zr + Mo for each)
# ============================================================================

echo ""
echo "Will run 2 studies sequentially (each includes Zr and Mo):"
echo "  1) outputs_100snapshots_100repititions"
echo "     - Zr: 100 snapshots, 100 reps, temps: ${ZR_TEMPS[*]}"
echo "     - Mo: 100 snapshots, 100 reps, temps: ${MO_TEMPS[*]}"
echo "  2) outputs_50snapshots_10repititions"
echo "     - Zr: 50 snapshots, 10 reps, temps: ${ZR_TEMPS[*]}"
echo "     - Mo: 50 snapshots, 10 reps, temps: ${MO_TEMPS[*]}"
echo ""

# ============================================================================
# Study 1: 100 snapshots, 100 repetitions
# ============================================================================
run_zr_study "outputs_100snapshots_100repititions" 100 100
run_mo_study "outputs_100snapshots_100repititions" 100 100

# ============================================================================
# Study 2: 50 snapshots, 10 repetitions
# ============================================================================
run_zr_study "outputs_50snapshots_10repititions" 50 10
run_mo_study "outputs_50snapshots_10repititions" 50 10

echo ""
echo "========================================================================"
echo "ALL FOUR STUDIES COMPLETE (2 x Zr + 2 x Mo)"
echo "========================================================================"
echo "Results locations:"
echo ""
echo "Study 1 (100 snapshots, 100 reps):"
echo "  - ${BASE_DIR}/outputs_100snapshots_100repititions/zr/"
echo "  - ${BASE_DIR}/outputs_100snapshots_100repititions/mo/"
echo ""
echo "Study 2 (50 snapshots, 10 reps):"
echo "  - ${BASE_DIR}/outputs_50snapshots_10repititions/zr/"
echo "  - ${BASE_DIR}/outputs_50snapshots_10repititions/mo/"
echo "========================================================================"
