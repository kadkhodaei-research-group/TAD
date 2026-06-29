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

# ============================================================================
# Argument parsing
# Usage:
#   ./run_studies_screen.sh                     # Run all (default)
#   ./run_studies_screen.sh --system zr         # Only Zr, both studies
#   ./run_studies_screen.sh --system mo         # Only Mo, both studies
#   ./run_studies_screen.sh --study 50/10       # Both systems, only 50snap/10rep
#   ./run_studies_screen.sh --study 100/100     # Both systems, only 100snap/100rep
#   ./run_studies_screen.sh --system zr --study 50/10  # Only Zr 50snap/10rep
# ============================================================================
RUN_SYSTEM="all"   # all, zr, mo
RUN_STUDY="all"    # all, 100/100, 50/10

while [[ $# -gt 0 ]]; do
    case $1 in
        --system)
            RUN_SYSTEM="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            if [[ "$RUN_SYSTEM" != "zr" && "$RUN_SYSTEM" != "mo" && "$RUN_SYSTEM" != "all" ]]; then
                echo "ERROR: --system must be 'zr', 'mo', or 'all'. Got: $RUN_SYSTEM"
                exit 1
            fi
            shift 2
            ;;
        --study)
            RUN_STUDY="$2"
            if [[ "$RUN_STUDY" != "100/100" && "$RUN_STUDY" != "50/10" && "$RUN_STUDY" != "all" ]]; then
                echo "ERROR: --study must be '100/100', '50/10', or 'all'. Got: $RUN_STUDY"
                exit 1
            fi
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--system zr|mo|all] [--study 100/100|50/10|all]"
            echo ""
            echo "Options:"
            echo "  --system   System to run: zr, mo, or all (default: all)"
            echo "  --study    Study config: 100/100, 50/10, or all (default: all)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run everything"
            echo "  $0 --system zr                  # Only Zr, both studies"
            echo "  $0 --system zr --study 50/10    # Only Zr, 50 snapshots / 10 reps"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "SEQUENTIAL Zr + Mo TEMPERATURE STUDIES LAUNCHER"
echo "========================================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory:   ${BASE_DIR}"
echo "Inputs directory: ${INPUTS_DIR}"
echo "FC directory:     ${FC_DIR}"
echo "System filter:    ${RUN_SYSTEM}"
echo "Study filter:     ${RUN_STUDY}"
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
        local job_id=$1
        local temp=$2
        local rep=$3
        local gpu_id=$((job_id % NUM_GPUS))
        local output_dir="${STUDY_OUTPUT}/T_${temp}K/run_$(printf "%03d" $rep)"

        cd "${SCRIPT_DIR}"
        CUDA_VISIBLE_DEVICES=${gpu_id} python run_tad.py \
            --poscar-file "${INPUTS_DIR}/POSCAR_Zr" \
            --force-constants-file "${FC_DIR}/FORCE_CONSTANTS_EAM_Zr_defected_${temp}" \
            --execution-mode eam \
            --eam-potential-file potentials/Cu-Zr_4.eam.fs \
            --temperature ${temp} \
            ${HYPERPARAMS} \
            --output-dir "${output_dir}" \
            > "${output_dir}.log" 2>&1
    }

    # Export function and GPU count
    export -f run_zr
    export NUM_GPUS

    # Generate commands with job IDs for GPU round-robin
    local commands_file="/tmp/zr_commands_${TIMESTAMP}.txt"
    > "$commands_file"

    local job_id=0
    for temp in "${ZR_TEMPS[@]}"; do
        mkdir -p "${STUDY_OUTPUT}/T_${temp}K"
        for ((rep=1; rep<=REPETITIONS; rep++)); do
            echo "run_zr ${job_id} ${temp} ${rep}" >> "$commands_file"
            job_id=$((job_id + 1))
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
        local job_id=$1
        local temp=$2
        local rep=$3
        local gpu_id=$((job_id % NUM_GPUS))
        local output_dir="${STUDY_OUTPUT}/T_${temp}K/run_$(printf "%03d" $rep)"

        cd "${SCRIPT_DIR}"
        CUDA_VISIBLE_DEVICES=${gpu_id} python run_tad.py \
            --poscar-file "${INPUTS_DIR}/POSCAR_Mo" \
            --force-constants-file "${FC_DIR}/FORCE_CONSTANTS_EAM_Mo_defected_${temp}" \
            --execution-mode eam \
            --kim-model MEAM_LAMMPS_ParkFellingerLenosky_2012_Mo__MO_269937397263_002 \
            --temperature ${temp} \
            ${HYPERPARAMS} \
            --output-dir "${output_dir}" \
            > "${output_dir}.log" 2>&1
    }

    # Export function and GPU count
    export -f run_mo
    export NUM_GPUS

    # Generate commands with job IDs for GPU round-robin
    local commands_file="/tmp/mo_commands_${TIMESTAMP}.txt"
    > "$commands_file"

    local job_id=0
    for temp in "${MO_TEMPS[@]}"; do
        mkdir -p "${STUDY_OUTPUT}/T_${temp}K"
        for ((rep=1; rep<=REPETITIONS; rep++)); do
            echo "run_mo ${job_id} ${temp} ${rep}" >> "$commands_file"
            job_id=$((job_id + 1))
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
# Build run plan based on filters
# ============================================================================
STUDIES_PLANNED=0

echo ""
echo "Planned runs:"

if [[ "$RUN_STUDY" == "all" || "$RUN_STUDY" == "100/100" ]]; then
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "zr" ]]; then
        echo "  - Zr: 100 snapshots, 100 reps, temps: ${ZR_TEMPS[*]}"
        STUDIES_PLANNED=$((STUDIES_PLANNED + 1))
    fi
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "mo" ]]; then
        echo "  - Mo: 100 snapshots, 100 reps, temps: ${MO_TEMPS[*]}"
        STUDIES_PLANNED=$((STUDIES_PLANNED + 1))
    fi
fi

if [[ "$RUN_STUDY" == "all" || "$RUN_STUDY" == "50/10" ]]; then
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "zr" ]]; then
        echo "  - Zr: 50 snapshots, 10 reps, temps: ${ZR_TEMPS[*]}"
        STUDIES_PLANNED=$((STUDIES_PLANNED + 1))
    fi
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "mo" ]]; then
        echo "  - Mo: 50 snapshots, 10 reps, temps: ${MO_TEMPS[*]}"
        STUDIES_PLANNED=$((STUDIES_PLANNED + 1))
    fi
fi

echo ""
echo "Total study runs: ${STUDIES_PLANNED}"
echo ""

# ============================================================================
# Study 1: 100 snapshots, 100 repetitions
# ============================================================================
if [[ "$RUN_STUDY" == "all" || "$RUN_STUDY" == "100/100" ]]; then
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "zr" ]]; then
        run_zr_study "outputs_100snapshots_100repititions" 100 100
    fi
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "mo" ]]; then
        run_mo_study "outputs_100snapshots_100repititions" 100 100
    fi
fi

# ============================================================================
# Study 2: 50 snapshots, 10 repetitions
# ============================================================================
if [[ "$RUN_STUDY" == "all" || "$RUN_STUDY" == "50/10" ]]; then
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "zr" ]]; then
        run_zr_study "outputs_50snapshots_10repititions" 50 10
    fi
    if [[ "$RUN_SYSTEM" == "all" || "$RUN_SYSTEM" == "mo" ]]; then
        run_mo_study "outputs_50snapshots_10repititions" 50 10
    fi
fi

echo ""
echo "========================================================================"
echo "ALL REQUESTED STUDIES COMPLETE (${STUDIES_PLANNED} runs)"
echo "========================================================================"
