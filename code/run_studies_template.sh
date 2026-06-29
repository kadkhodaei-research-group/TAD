#!/bin/bash
# ============================================================================
# Generic temperature-study launcher.
#
# This script reads its setup from a config file (default: ./studies.conf next
# to this script) and launches run_tad.py for every (system, study, temperature,
# repetition) combination defined there.
#
# To use:
#   1. Copy studies.conf.template to studies.conf (or pass --config <path>)
#   2. Fill in the variables for your systems and studies.
#   3. Run:
#        ./run_studies_template.sh
#        ./run_studies_template.sh --system <name>     # only that system
#        ./run_studies_template.sh --study  <name>     # only that study
#        ./run_studies_template.sh --config <path>     # alternate config file
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUTS_DIR="${BASE_DIR}/inputs"

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
CONFIG_FILE=""
RUN_SYSTEM="all"
RUN_STUDY="all"

print_usage() {
    cat <<EOF
Usage: $0 [--config FILE] [--system NAME|all] [--study NAME|all]

Options:
  --config FILE     Path to the studies config file
                    (default: ./studies.conf next to this script)
  --system NAME     Run only the named system (must appear in SYSTEMS)
                    or 'all' (default).
  --study  NAME     Run only the named study (must appear in STUDIES,
                    or in the system's per-system SYS_<NAME>_STUDIES)
                    or 'all' (default).
  -h, --help        Show this message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --system) RUN_SYSTEM="$2";  shift 2 ;;
        --study)  RUN_STUDY="$2";   shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "${CONFIG_FILE}" ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/studies.conf"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    echo "       Start from studies.conf.template (next to this script)." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Load the user config
# ----------------------------------------------------------------------------
# The config file is bash-sourced. It must define SYSTEMS and either a global
# STUDIES array (used for every system) or a per-system SYS_<NAME>_STUDIES.
# It must also define HYPERPARAMS unless every system supplies its own
# SYS_<NAME>_HYPERPARAMS.
#
# Required globals:
#   SYSTEMS                array of system short-names, e.g. (zr mo)
#
# Optional globals (with defaults):
#   RUNS_PER_GPU=10        used for default PARALLEL_RUNS = RUNS_PER_GPU*NUM_GPUS
#   STUDIES                array of "<name>:<num_snapshots>:<repetitions>",
#                          used for any system that doesn't override
#   HYPERPARAMS            string of common run_tad.py flags, used for any
#                          system that doesn't override
#
# Required per-system (for each NAME in SYSTEMS):
#   SYS_<NAME>_POSCAR              path under inputs/ (or absolute);
#                                   may contain {TEMP} and shell globs (* ? […])
#   SYS_<NAME>_FC_PATTERN          path under inputs/ (or absolute);
#                                   may contain {TEMP} and shell globs
#   SYS_<NAME>_EXECUTION_MODE      e.g. "eam", "vasp", "mpi", ...
#   SYS_<NAME>_TEMPERATURES        bash array of temperatures (K)
#
# Optional per-system:
#   SYS_<NAME>_EAM_POTENTIAL_FILE   adds --eam-potential-file
#   SYS_<NAME>_KIM_MODEL            adds --kim-model
#   SYS_<NAME>_VASP_COMMAND         adds --vasp-command
#   SYS_<NAME>_VASP_INPUT_DIR       adds --vasp-input-dir (under inputs/ or abs)
#   SYS_<NAME>_MOVING_INDICES       adds --moving-indices (else auto-detect)
#   SYS_<NAME>_ORIENT_ATOM_DIRECTION adds --orient-atom-direction (else auto)
#   SYS_<NAME>_HYPERPARAMS          override of global HYPERPARAMS
#   SYS_<NAME>_PARALLEL_RUNS        override of global PARALLEL_RUNS
#   SYS_<NAME>_STUDIES              override of global STUDIES
#   SYS_<NAME>_OUTPUT_LABEL         override of <name> in output path
RUNS_PER_GPU=10
STUDIES=()
HYPERPARAMS=""
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

# Validation
if ! declare -p SYSTEMS >/dev/null 2>&1 || [[ ${#SYSTEMS[@]} -eq 0 ]]; then
    echo "ERROR: SYSTEMS array is not set or empty in ${CONFIG_FILE}" >&2
    exit 1
fi

for sys in "${SYSTEMS[@]}"; do
    for field in POSCAR FC_PATTERN EXECUTION_MODE TEMPERATURES; do
        var="SYS_${sys}_${field}"
        if ! declare -p "${var}" >/dev/null 2>&1; then
            echo "ERROR: ${var} is not set in ${CONFIG_FILE}" >&2
            exit 1
        fi
    done

    # Each system needs studies (global or per-system) and hyperparams
    if ! declare -p "SYS_${sys}_STUDIES" >/dev/null 2>&1 && [[ ${#STUDIES[@]} -eq 0 ]]; then
        echo "ERROR: System '${sys}' has no studies. Define STUDIES (global) or SYS_${sys}_STUDIES." >&2
        exit 1
    fi
    sys_hyper_var="SYS_${sys}_HYPERPARAMS"
    if [[ -z "${!sys_hyper_var:-}" && -z "${HYPERPARAMS}" ]]; then
        echo "ERROR: System '${sys}' has no hyperparameters. Define HYPERPARAMS (global) or SYS_${sys}_HYPERPARAMS." >&2
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------
echo "========================================================================"
echo "GENERIC TEMPERATURE STUDY LAUNCHER"
echo "========================================================================"
echo "Config file:      ${CONFIG_FILE}"
echo "Script directory: ${SCRIPT_DIR}"
echo "Base directory:   ${BASE_DIR}"
echo "Inputs directory: ${INPUTS_DIR}"
echo "System filter:    ${RUN_SYSTEM}"
echo "Study filter:     ${RUN_STUDY}"
echo "Systems defined:  ${SYSTEMS[*]}"
[[ ${#STUDIES[@]} -gt 0 ]] && echo "Global studies:   ${STUDIES[*]}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
DEFAULT_PARALLEL_RUNS=$((RUNS_PER_GPU * NUM_GPUS))
echo "Available GPUs:           ${NUM_GPUS}"
echo "Default RUNS_PER_GPU:     ${RUNS_PER_GPU}"
echo "Default parallel workers: ${DEFAULT_PARALLEL_RUNS}"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# Resolve a path: substitute {TEMP}, prepend INPUTS_DIR if not absolute,
# and (if the path contains shell-glob characters) expand the glob and use
# the first match. Errors out if a glob produces no match.
resolve_input_path() {
    local pattern="$1"
    local temp="$2"
    local kind="$3"  # for error messages

    # Substitute {TEMP}
    local path="${pattern//\{TEMP\}/${temp}}"

    # Prepend inputs dir if relative
    if [[ "${path}" != /* ]]; then
        path="${INPUTS_DIR}/${path}"
    fi

    # Glob expansion if pattern contains *, ?, or [
    if [[ "${path}" == *[\*\?\[]* ]]; then
        local matches
        shopt -s nullglob
        # shellcheck disable=SC2206
        matches=( ${path} )
        shopt -u nullglob
        if [[ ${#matches[@]} -eq 0 ]]; then
            echo "ERROR: ${kind} pattern matched no files: ${path}" >&2
            return 1
        fi
        path="${matches[0]}"
    fi

    printf '%s' "${path}"
}

# ----------------------------------------------------------------------------
# Main loop: systems x studies x temperatures x repetitions
# ----------------------------------------------------------------------------
for sys in "${SYSTEMS[@]}"; do
    if [[ "${RUN_SYSTEM}" != "all" && "${RUN_SYSTEM}" != "${sys}" ]]; then
        continue
    fi

    # Per-system variables (with global fallback where applicable)
    var_poscar="SYS_${sys}_POSCAR"
    var_fc="SYS_${sys}_FC_PATTERN"
    var_exec="SYS_${sys}_EXECUTION_MODE"
    var_eam="SYS_${sys}_EAM_POTENTIAL_FILE"
    var_kim="SYS_${sys}_KIM_MODEL"
    var_vasp_cmd="SYS_${sys}_VASP_COMMAND"
    var_vasp_dir="SYS_${sys}_VASP_INPUT_DIR"
    var_moving="SYS_${sys}_MOVING_INDICES"
    var_orient="SYS_${sys}_ORIENT_ATOM_DIRECTION"
    var_hyper="SYS_${sys}_HYPERPARAMS"
    var_par="SYS_${sys}_PARALLEL_RUNS"
    var_label="SYS_${sys}_OUTPUT_LABEL"

    poscar_pattern="${!var_poscar}"
    fc_pattern="${!var_fc}"
    exec_mode="${!var_exec}"
    eam_file="${!var_eam:-}"
    kim_model="${!var_kim:-}"
    vasp_command="${!var_vasp_cmd:-}"
    vasp_input_dir="${!var_vasp_dir:-}"
    moving_indices="${!var_moving:-}"
    orient_atom_direction="${!var_orient:-}"
    sys_hyperparams="${!var_hyper:-${HYPERPARAMS}}"
    sys_parallel_runs="${!var_par:-${DEFAULT_PARALLEL_RUNS}}"
    output_label="${!var_label:-${sys}}"

    # Resolve VASP input dir if given (relative -> inputs/, leave absolute alone)
    if [[ -n "${vasp_input_dir}" && "${vasp_input_dir}" != /* ]]; then
        vasp_input_dir="${INPUTS_DIR}/${vasp_input_dir}"
    fi

    declare -n temps_ref="SYS_${sys}_TEMPERATURES"

    # Determine the studies for this system
    if declare -p "SYS_${sys}_STUDIES" >/dev/null 2>&1; then
        declare -n local_studies_ref="SYS_${sys}_STUDIES"
        local_studies=( "${local_studies_ref[@]}" )
    else
        local_studies=( "${STUDIES[@]}" )
    fi

    for study_def in "${local_studies[@]}"; do
        IFS=':' read -r study_name num_snapshots reps <<< "${study_def}"

        if [[ -z "${study_name}" || -z "${num_snapshots}" || -z "${reps}" ]]; then
            echo "ERROR: bad study entry '${study_def}' for system '${sys}' (expected NAME:SNAPSHOTS:REPS)" >&2
            exit 1
        fi
        if [[ "${RUN_STUDY}" != "all" && "${RUN_STUDY}" != "${study_name}" ]]; then
            continue
        fi

        TIMESTAMP=$(date +%s)
        STUDY_OUTPUT="${BASE_DIR}/outputs_${study_name}/${output_label}/temperature_study_${TIMESTAMP}"
        mkdir -p "${STUDY_OUTPUT}"

        echo ""
        echo "------------------------------------------------------------------------"
        echo "STARTING study='${study_name}' system='${sys}'"
        echo "  POSCAR pattern:   ${poscar_pattern}"
        echo "  FC pattern:       ${fc_pattern}"
        echo "  Mode:             ${exec_mode}"
        [[ -n "${eam_file}" ]]              && echo "  EAM potential:    ${eam_file}"
        [[ -n "${kim_model}" ]]             && echo "  KIM model:        ${kim_model}"
        [[ -n "${vasp_command}" ]]          && echo "  VASP command:     ${vasp_command}"
        [[ -n "${vasp_input_dir}" ]]        && echo "  VASP input dir:   ${vasp_input_dir}"
        [[ -n "${moving_indices}" ]]        && echo "  Moving indices:   ${moving_indices} (manual override)"
        [[ -n "${orient_atom_direction}" ]] && echo "  Orient atom/dir:  ${orient_atom_direction} (manual override)"
        echo "  Snapshots:        ${num_snapshots}"
        echo "  Repetitions:      ${reps}"
        echo "  Temperatures:     ${temps_ref[*]}"
        echo "  Output:           ${STUDY_OUTPUT}"
        echo "  Parallel workers: ${sys_parallel_runs}"
        echo "------------------------------------------------------------------------"

        commands_file="$(mktemp -t "studies_${sys}_${study_name}_XXXXXX.txt")"
        : > "${commands_file}"

        job_id=0
        for temp in "${temps_ref[@]}"; do
            mkdir -p "${STUDY_OUTPUT}/T_${temp}K"

            poscar_path=$(resolve_input_path "${poscar_pattern}" "${temp}" "POSCAR") || exit 1
            fc_path=$(resolve_input_path "${fc_pattern}" "${temp}" "FORCE_CONSTANTS") || exit 1

            extra_flags=""
            [[ -n "${eam_file}" ]]              && extra_flags+=" --eam-potential-file ${eam_file}"
            [[ -n "${kim_model}" ]]             && extra_flags+=" --kim-model ${kim_model}"
            [[ -n "${vasp_command}" ]]          && extra_flags+=" --vasp-command ${vasp_command}"
            [[ -n "${vasp_input_dir}" ]]        && extra_flags+=" --vasp-input-dir ${vasp_input_dir}"
            [[ -n "${moving_indices}" ]]        && extra_flags+=" --moving-indices ${moving_indices}"
            [[ -n "${orient_atom_direction}" ]] && extra_flags+=" --orient-atom-direction ${orient_atom_direction}"

            for ((rep=1; rep<=reps; rep++)); do
                gpu_id=$((job_id % NUM_GPUS))
                output_dir="${STUDY_OUTPUT}/T_${temp}K/run_$(printf '%03d' "${rep}")"

                cmd="cd '${SCRIPT_DIR}' && CUDA_VISIBLE_DEVICES=${gpu_id} python run_tad.py"
                cmd+=" --poscar-file '${poscar_path}'"
                cmd+=" --force-constants-file '${fc_path}'"
                cmd+=" --execution-mode ${exec_mode}"
                cmd+="${extra_flags}"
                cmd+=" --temperature ${temp}"
                cmd+=" --num-snapshots ${num_snapshots}"
                cmd+=" ${sys_hyperparams}"
                cmd+=" --output-dir '${output_dir}'"
                cmd+=" > '${output_dir}.log' 2>&1"

                printf '%s\n' "${cmd}" >> "${commands_file}"
                job_id=$((job_id + 1))
            done
        done

        echo "Total runs queued: ${job_id}"
        echo "Launching..."
        # shellcheck disable=SC2002
        cat "${commands_file}" | xargs -d '\n' -I {} -P "${sys_parallel_runs}" bash -c "{}"
        rm -f "${commands_file}"

        echo "DONE study='${study_name}' system='${sys}'"
    done
done

echo ""
echo "========================================================================"
echo "ALL REQUESTED STUDIES COMPLETE"
echo "========================================================================"
