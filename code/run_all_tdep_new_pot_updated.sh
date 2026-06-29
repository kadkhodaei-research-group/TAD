#!/bin/bash
#
# TDEP calculations for Mo, Zr, and ZrO2 systems
# Updated with custom output directory and cleanup options
#
# Cell sizes:
#   Mo:         54/53 atoms (3x3x3 supercell)
#   Zr large:   216/215 atoms (6x6x6 supercell)
#   Zr small:   54/53 atoms (3x3x3 supercell)
#   ZrO2:       4x4x4 supercells at various temperatures
#
# Usage:
#   ./run_all_tdep_new_pot.sh [options] <system>
#
# Options:
#   --output-dir <path>    Custom output directory (default: ../outputs)
#   --cleanup-steps        Keep only last 2 TDEP step folders
#   --cleanup-calc         Delete calc_* directories after processing (saves disk space)
#   --zro2-potential <pot> ZrO2 potential: 'gaop' (default) or 'uzro'
#
# Systems:
#   mo_perfect, mo_defected, mo
#   mo_equilibrium_volume          (perfect only, multiple volumes)
#   zr_perfect, zr_defected, zr
#   zr_perfect_small, zr_defected_small, zr_small
#   zr_small_equilibrium_volume    (perfect only, multiple volumes)
#   zro2_2600, zro2_2800, zro2_2900, zro2_3000, zro2, zro2_all
#   all, all_small
#
# Examples:
#   ./run_all_tdep_new_pot.sh zr_small
#   ./run_all_tdep_new_pot.sh --output-dir ../outputs_custom zro2_2600
#   ./run_all_tdep_new_pot.sh --cleanup-steps --output-dir ../outputs_zro2 zro2
#   ./run_all_tdep_new_pot.sh --cleanup-steps --cleanup-calc zro2_2600  # Both cleanup options
#   ./run_all_tdep_new_pot.sh --zro2-potential uzro zro2_small  # Use UZrO potential
#

# Change to scripts directory
cd "$(dirname "$0")"

# ============================================================================
# PARSE COMMAND LINE OPTIONS
# ============================================================================
OUTPUT_DIR="../outputs"  # Default output directory
CLEANUP_STEPS=NO        # Default: don't cleanup old steps
CLEANUP_CALC=NO         # Default: don't cleanup calc_* directories
ZRO2_POTENTIAL_CHOICE="gaop"  # Default: use GAOP2 potential (options: gaop, uzro)
SYSTEM=""               # System to run

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cleanup-steps)
            CLEANUP_STEPS=YES
            shift
            ;;
        --cleanup-calc)
            CLEANUP_CALC=YES
            shift
            ;;
        --zro2-potential)
            ZRO2_POTENTIAL_CHOICE="$2"
            if [[ "$ZRO2_POTENTIAL_CHOICE" != "gaop" && "$ZRO2_POTENTIAL_CHOICE" != "uzro" ]]; then
                echo "ERROR: --zro2-potential must be 'gaop' or 'uzro'"
                exit 1
            fi
            shift 2
            ;;
        *)
            SYSTEM="$1"
            shift
            ;;
    esac
done

# Create absolute path for output directory
# Handle relative paths correctly - realpath might not work if dir doesn't exist
if [[ "$OUTPUT_DIR" != /* ]]; then
    # If it starts with ../, we want to go up from scripts directory
    if [[ "$OUTPUT_DIR" == ../* ]]; then
        # Go up one level from scripts and append the rest
        OUTPUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/${OUTPUT_DIR#../}"
    else
        # For other relative paths, use current directory
        OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
    fi
fi

# Create the directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Cleanup old steps: $CLEANUP_STEPS"
echo "  Cleanup calc_* dirs: $CLEANUP_CALC"
echo "  ZrO2 potential: $ZRO2_POTENTIAL_CHOICE"
echo "  System to run: ${SYSTEM:-all}"
echo "============================================================"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build cleanup-calc flag for run_stdep.py
CLEANUP_CALC_FLAG=""
if [ "$CLEANUP_CALC" = "YES" ]; then
    CLEANUP_CALC_FLAG="--cleanup-calc"
fi

# ============================================================================
# CONTINUATION MODE
# ============================================================================
# Set to YES to continue interrupted runs instead of starting fresh
# When YES: uses --continue flag to resume from last completed step
# When NO:  uses --skip-existing to skip already completed runs
CONTINUE_MODE=YES

# Common parameters
SNAPSHOTS=100
STEPS=1000
RC2=6
RC3=-1
CONVERGENCE_THRESHOLD=0.005
SNAPSHOT_INCREMENT=50
MAX_SNAPSHOTS=20000
MIN_STEPS=10

# Convergence thresholds
FC_THRESHOLD=0.005           # FC norm relative change threshold
EIGENVALUE_THRESHOLD=0.03    # Soft mode eigenvalue change at q=2/3[111] in THz

# Equilibrium volume scaling factors (V/V0)
VOLUME_FACTORS=(0.97 0.99 1.00 1.01 1.03 1.05)

# ============================================================================
# Mo Parameters
# ============================================================================
MO_KIM_MODEL="MEAM_LAMMPS_ParkFellingerLenosky_2012_Mo__MO_269937397263_002"
MO_TEMPERATURES=(200 300 1400)
MO_QGRID="26 26 26"

# ============================================================================
# Zr Parameters
# ============================================================================
ZR_EAM_POTENTIAL="potentials/Zr_1.eam.fs"
ZR_TEMPERATURES=(1300 1400 1500 1600 1700 1800)
ZR_QGRID_SMALL_CELL="26 26 26"
ZR_QGRID_LARGE_PERFECT="23 23 23"   # For perfect large cell (216 atoms)
ZR_QGRID_LARGE_DEFECTED="11 11 11"     # For defected large cell (215 atoms)
# ============================================================================
# ZrO2 Parameters (4x4x4 supercell - 192 atoms perfect, 191 defected)
# ============================================================================
# Potential options for ZrO2:
#   gaop - GAOP2 potential (LAMMPS format) - better for high-T ZrO2
#   uzro - UZrO.eam.alloy potential (EAM format) - original U-Zr-O potential
ZRO2_GAOP_POTENTIAL="potentials/ScienceDirect_files_13Oct2025_18-11-03/GAOP/potentials/lammps/gaop2.lammps"
ZRO2_UZRO_POTENTIAL="potentials/UZrO.eam.alloy"
ZRO2_TEMPERATURES=(2600 2800 2900 3000)
ZRO2_QGRID="6 6 6"  # Reduced from 9 9 9 for ZrO2
ZRO2_INPUT_BASE="../inputs/ZrO2_scaled_structures_444"
ZRO2_MAX_FREQ=25  # Max frequency for canonical_configuration (first step)

# Set active potential based on user choice
if [ "$ZRO2_POTENTIAL_CHOICE" = "uzro" ]; then
    ZRO2_ACTIVE_POTENTIAL="$ZRO2_UZRO_POTENTIAL"
    ZRO2_EXECUTION_MODE="eam"
    ZRO2_POTENTIAL_FLAG="--eam-potential-file"
    echo "Using UZrO.eam.alloy potential for ZrO2 (EAM mode)"
else
    ZRO2_ACTIVE_POTENTIAL="$ZRO2_GAOP_POTENTIAL"
    ZRO2_EXECUTION_MODE="lammps"
    ZRO2_POTENTIAL_FLAG="--lammps-potential-file"
    echo "Using GAOP2 potential for ZrO2 (LAMMPS mode)"
fi

# ============================================================================
# ZrO2 Small Parameters (3x3x3 supercell - 81 atoms perfect, 80 defected)
# ============================================================================
ZRO2_SMALL_INPUT_BASE="../inputs/ZrO2_scaled_structures_333"
ZRO2_SMALL_QGRID="23 23 23"  # Can use larger grid for smaller cell

# ============================================================================
# Special run_tdep for ZrO2 (no eigenvalue convergence, max_freq=25)
# ============================================================================
run_tdep_zro2() {
    local name=$1
    local temp=$2
    local supercell=$3
    local unitcell=$4
    local potential_args=$5
    local qgrid=$6
    local custom_output_dir=${7:-$OUTPUT_DIR}  # Use custom dir if provided

    local run_name="${name}_${temp}K"
    local run_folder="${custom_output_dir}/${run_name}"

    echo "============================================================"
    echo "Running ZrO2: $name at ${temp}K"
    echo "Output: $run_folder"
    echo "Potential: $ZRO2_ACTIVE_POTENTIAL ($ZRO2_EXECUTION_MODE mode)"
    echo "Special settings: Max freq=$ZRO2_MAX_FREQ, No eigenvalue convergence"
    echo "============================================================"

    # Check if we should continue an existing run
    if [ "$CONTINUE_MODE" = "YES" ] && [ -d "$run_folder" ]; then
        echo "CONTINUATION MODE: Resuming from $run_folder"
        echo "  Overriding q-grid to: $qgrid"

        # Cleanup old steps before continuing
        cleanup_old_steps "$run_folder"

        # Start background cleanup monitor if cleanup is enabled
        if [ "$CLEANUP_STEPS" = "YES" ]; then
            (
                while true; do
                    sleep 30  # Check every 30 seconds
                    if [ -d "${run_folder}/tdep_calculations" ]; then
                        # Count step directories
                        step_dirs=($(ls -d ${run_folder}/tdep_calculations/step_* 2>/dev/null | sort -V))
                        n_steps=${#step_dirs[@]}

                        # Keep only the last 2 steps
                        if [ $n_steps -gt 2 ]; then
                            for ((i=0; i<$((n_steps-2)); i++)); do
                                if [ -d "${step_dirs[$i]}" ]; then
                                    echo "  [Cleanup] Removing old step: $(basename ${step_dirs[$i]})"
                                    rm -rf "${step_dirs[$i]}"
                                fi
                            done
                        fi
                    fi

                    # Check if the main process is still running
                    if ! ps -p $$ > /dev/null 2>&1; then
                        break
                    fi
                done
            ) &
            CLEANUP_PID=$!
        fi

        python run_stdep.py \
            --continue "$run_folder" \
            --qpoint-grid $qgrid \
            --max-snapshots $MAX_SNAPSHOTS \
            --convergence-threshold $CONVERGENCE_THRESHOLD \
            --snapshot-increment $SNAPSHOT_INCREMENT \
            $ZRO2_POTENTIAL_FLAG $ZRO2_ACTIVE_POTENTIAL \
            $CLEANUP_CALC_FLAG \
            --verbose

        # Kill the cleanup monitor if it's running
        if [ ! -z "$CLEANUP_PID" ]; then
            kill $CLEANUP_PID 2>/dev/null
            wait $CLEANUP_PID 2>/dev/null
        fi
    else
        # Normal run for ZrO2 (with special parameters)

        # Start background cleanup monitor if cleanup is enabled
        if [ "$CLEANUP_STEPS" = "YES" ]; then
            (
                while true; do
                    sleep 30  # Check every 30 seconds
                    if [ -d "${run_folder}/tdep_calculations" ]; then
                        # Count step directories
                        step_dirs=($(ls -d ${run_folder}/tdep_calculations/step_* 2>/dev/null | sort -V))
                        n_steps=${#step_dirs[@]}

                        # Keep only the last 2 steps
                        if [ $n_steps -gt 2 ]; then
                            for ((i=0; i<$((n_steps-2)); i++)); do
                                if [ -d "${step_dirs[$i]}" ]; then
                                    echo "  [Cleanup] Removing old step: $(basename ${step_dirs[$i]})"
                                    rm -rf "${step_dirs[$i]}"
                                fi
                            done
                        fi
                    fi

                    # Check if the main process is still running
                    if ! ps -p $$ > /dev/null 2>&1; then
                        break
                    fi
                done
            ) &
            CLEANUP_PID=$!
        fi

        # Run WITHOUT eigenvalue convergence and WITH max-frequency=25 for canonical_configuration
        python run_stdep.py \
            --execution-mode $ZRO2_EXECUTION_MODE \
            $ZRO2_POTENTIAL_FLAG $ZRO2_ACTIVE_POTENTIAL \
            --temperature $temp \
            --steps $STEPS \
            --snapshots $SNAPSHOTS \
            --rc2 $RC2 \
            --rc3 $RC3 \
            --qpoint-grid $qgrid \
            --convergence-threshold $CONVERGENCE_THRESHOLD \
            --snapshot-increment $SNAPSHOT_INCREMENT \
            --max-snapshots $MAX_SNAPSHOTS \
            --min-steps $MIN_STEPS \
            --fc-convergence \
            --fc-convergence-threshold $FC_THRESHOLD \
            --max-frequency $ZRO2_MAX_FREQ \
            --supercell "$supercell" \
            --unitcell "$unitcell" \
            --run-name "${run_name}" \
            --output-dir "${custom_output_dir}" \
            --skip-existing \
            $CLEANUP_CALC_FLAG \
            --verbose

        # Kill the cleanup monitor if it's running
        if [ ! -z "$CLEANUP_PID" ]; then
            kill $CLEANUP_PID 2>/dev/null
            wait $CLEANUP_PID 2>/dev/null
        fi
    fi

    # Final cleanup after completion
    cleanup_old_steps "$run_folder"

    echo ""
}

# ============================================================================
# Cleanup function for TDEP steps
# ============================================================================
cleanup_old_steps() {
    local run_folder=$1

    if [ "$CLEANUP_STEPS" = "YES" ]; then
        local tdep_dir="${run_folder}/tdep_calculations"
        if [ -d "$tdep_dir" ]; then
            # Count step directories
            local step_dirs=($(ls -d ${tdep_dir}/step_* 2>/dev/null | sort -V))
            local n_steps=${#step_dirs[@]}

            # Keep only the last 2 steps
            if [ $n_steps -gt 2 ]; then
                echo "  Cleaning up old TDEP steps (keeping last 2)..."
                for ((i=0; i<$((n_steps-2)); i++)); do
                    echo "    Removing ${step_dirs[$i]}"
                    rm -rf "${step_dirs[$i]}"
                done
            fi
        fi
    fi
}

# ============================================================================
# Helper function to run TDEP
# ============================================================================
run_tdep() {
    local name=$1
    local temp=$2
    local supercell=$3
    local unitcell=$4
    local potential_args=$5
    local qgrid=$6
    local custom_output_dir=${7:-$OUTPUT_DIR}  # Use custom dir if provided

    local run_name="${name}_${temp}K"
    local run_folder="${custom_output_dir}/${run_name}"

    echo "============================================================"
    echo "Running: $name at ${temp}K"
    echo "Output: $run_folder"
    echo "============================================================"

    # Check if we should continue an existing run
    if [ "$CONTINUE_MODE" = "YES" ] && [ -d "$run_folder" ]; then
        echo "CONTINUATION MODE: Resuming from $run_folder"
        echo "  Overriding q-grid to: $qgrid"

        # Cleanup old steps before continuing
        cleanup_old_steps "$run_folder"

        # Start background cleanup monitor if cleanup is enabled
        if [ "$CLEANUP_STEPS" = "YES" ]; then
            (
                while true; do
                    sleep 30  # Check every 30 seconds
                    if [ -d "${run_folder}/tdep_calculations" ]; then
                        # Count step directories
                        step_dirs=($(ls -d ${run_folder}/tdep_calculations/step_* 2>/dev/null | sort -V))
                        n_steps=${#step_dirs[@]}

                        # Keep only the last 2 steps
                        if [ $n_steps -gt 2 ]; then
                            for ((i=0; i<$((n_steps-2)); i++)); do
                                if [ -d "${step_dirs[$i]}" ]; then
                                    echo "  [Cleanup] Removing old step: $(basename ${step_dirs[$i]})"
                                    rm -rf "${step_dirs[$i]}"
                                fi
                            done
                        fi
                    fi

                    # Check if the main process is still running
                    if ! ps -p $$ > /dev/null 2>&1; then
                        break
                    fi
                done
            ) &
            CLEANUP_PID=$!
        fi

        python run_stdep.py \
            --continue "$run_folder" \
            --qpoint-grid $qgrid \
            --max-snapshots $MAX_SNAPSHOTS \
            --convergence-threshold $CONVERGENCE_THRESHOLD \
            --snapshot-increment $SNAPSHOT_INCREMENT \
            $CLEANUP_CALC_FLAG \
            --verbose

        # Kill the cleanup monitor if it's running
        if [ ! -z "$CLEANUP_PID" ]; then
            kill $CLEANUP_PID 2>/dev/null
            wait $CLEANUP_PID 2>/dev/null
        fi
    else
        # Normal run (with --skip-existing)

        # Start background cleanup monitor if cleanup is enabled
        if [ "$CLEANUP_STEPS" = "YES" ]; then
            (
                while true; do
                    sleep 30  # Check every 30 seconds
                    if [ -d "${run_folder}/tdep_calculations" ]; then
                        # Count step directories
                        step_dirs=($(ls -d ${run_folder}/tdep_calculations/step_* 2>/dev/null | sort -V))
                        n_steps=${#step_dirs[@]}

                        # Keep only the last 2 steps
                        if [ $n_steps -gt 2 ]; then
                            for ((i=0; i<$((n_steps-2)); i++)); do
                                if [ -d "${step_dirs[$i]}" ]; then
                                    echo "  [Cleanup] Removing old step: $(basename ${step_dirs[$i]})"
                                    rm -rf "${step_dirs[$i]}"
                                fi
                            done
                        fi
                    fi

                    # Check if the main process is still running
                    if ! ps -p $$ > /dev/null 2>&1; then
                        break
                    fi
                done
            ) &
            CLEANUP_PID=$!
        fi

        python run_stdep.py \
            --execution-mode eam \
            $potential_args \
            --temperature $temp \
            --steps $STEPS \
            --snapshots $SNAPSHOTS \
            --rc2 $RC2 \
            --rc3 $RC3 \
            --qpoint-grid $qgrid \
            --convergence-threshold $CONVERGENCE_THRESHOLD \
            --snapshot-increment $SNAPSHOT_INCREMENT \
            --max-snapshots $MAX_SNAPSHOTS \
            --min-steps $MIN_STEPS \
            --fc-convergence \
            --fc-convergence-threshold $FC_THRESHOLD \
            --eigenvalue-convergence \
            --eigenvalue-threshold $EIGENVALUE_THRESHOLD \
            --supercell "$supercell" \
            --unitcell "$unitcell" \
            --run-name "${run_name}" \
            --output-dir "${custom_output_dir}" \
            --skip-existing \
            $CLEANUP_CALC_FLAG \
            --verbose

        # Kill the cleanup monitor if it's running
        if [ ! -z "$CLEANUP_PID" ]; then
            kill $CLEANUP_PID 2>/dev/null
            wait $CLEANUP_PID 2>/dev/null
        fi
    fi

    # Final cleanup after completion
    cleanup_old_steps "$run_folder"

    echo ""
}

# ============================================================================
# Mo Perfect
# ============================================================================
run_mo_perfect() {
    echo "##############################################################"
    echo "# Mo Perfect System"
    echo "##############################################################"

    for temp in "${MO_TEMPERATURES[@]}"; do
        run_tdep "Mo_perfect" $temp \
            "POSCAR_Mo_perfect_supercell" \
            "POSCAR_Mo_perfect_unitcell" \
            "--kim-model $MO_KIM_MODEL" \
            "$MO_QGRID" \
            "$OUTPUT_DIR"
    done
}

# ============================================================================
# Mo Defected
# ============================================================================
run_mo_defected() {
    echo "##############################################################"
    echo "# Mo Defected System"
    echo "##############################################################"

    for temp in "${MO_TEMPERATURES[@]}"; do
        run_tdep "Mo_defected" $temp \
            "POSCAR_Mo_defected_supercell" \
            "POSCAR_Mo_defected_unitcell" \
            "--kim-model $MO_KIM_MODEL" \
            "$MO_QGRID" \
            "$OUTPUT_DIR"
    done
}

# ============================================================================
# Zr Perfect
# ============================================================================
run_zr_perfect() {
    echo "##############################################################"
    echo "# Zr Perfect System (Q-grid: $ZR_QGRID_LARGE_PERFECT)"
    echo "##############################################################"

    for temp in "${ZR_TEMPERATURES[@]}"; do
        run_tdep "Zr_perfect" $temp \
            "POSCAR_Zr_perfect_supercell" \
            "POSCAR_Zr_perfect_unitcell" \
            "--eam-potential-file $ZR_EAM_POTENTIAL" \
            "$ZR_QGRID_LARGE_PERFECT" \
            "$OUTPUT_DIR"
    done
}

# ============================================================================
# Zr Defected (large - 215 atoms)
# ============================================================================
run_zr_defected() {
    echo "##############################################################"
    echo "# Zr Defected System (large - 215 atoms, Q-grid: $ZR_QGRID_LARGE_DEFECTED)"
    echo "##############################################################"

    for temp in "${ZR_TEMPERATURES[@]}"; do
        run_tdep "Zr_defected" $temp \
            "POSCAR_Zr_defected_supercell" \
            "POSCAR_Zr_defected_unitcell" \
            "--eam-potential-file $ZR_EAM_POTENTIAL" \
            "$ZR_QGRID_LARGE_DEFECTED" \
            "$OUTPUT_DIR"
    done
}

# ============================================================================
# Zr Perfect Small (54 atoms - 3x3x3 supercell)
# ============================================================================
run_zr_perfect_small() {
    echo "##############################################################"
    echo "# Zr Perfect System (small - 54 atoms)"
    echo "##############################################################"

    for temp in "${ZR_TEMPERATURES[@]}"; do
        run_tdep "Zr_perfect_small" $temp \
            "POSCAR_Zr_perfect_supercell_54atoms" \
            "POSCAR_Zr_perfect_unitcell_2atoms" \
            "--eam-potential-file $ZR_EAM_POTENTIAL" \
            "$ZR_QGRID_SMALL_CELL" \
            "$OUTPUT_DIR"
    done
}

# ============================================================================
# Zr Defected Small (53 atoms - 3x3x3 supercell with vacancy)
# ============================================================================
run_zr_defected_small() {
    echo "##############################################################"
    echo "# Zr Defected System (small - 53 atoms)"
    echo "##############################################################"

    for temp in "${ZR_TEMPERATURES[@]}"; do
        run_tdep "Zr_defected_small" $temp \
            "POSCAR_Zr_defected_supercell_53atoms" \
            "POSCAR_Zr_defected_unitcell_53atoms" \
            "--eam-potential-file $ZR_EAM_POTENTIAL" \
            "$ZR_QGRID_SMALL_CELL" \
            "$OUTPUT_DIR"
    done
}

# ============================================================================
# Equilibrium Volume Helper
# ============================================================================
# Generates volume-scaled POSCAR files and returns the filenames.
# Uses generate_scaled_poscar.py to scale lattice vectors by (V/V0)^(1/3).
# Scaled files are written to ../inputs/ with a _volX.XX suffix.
generate_scaled_poscars() {
    local supercell_base=$1    # e.g. POSCAR_Zr_perfect_supercell_54atoms
    local unitcell_base=$2     # e.g. POSCAR_Zr_perfect_unitcell_2atoms
    local vol_factor=$3        # e.g. 0.97

    local inputs_dir
    inputs_dir="$(cd "$(dirname "$0")/.." && pwd)/inputs"
    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"

    local vol_tag
    vol_tag=$(printf "%.2f" "$vol_factor")

    local sc_in="${inputs_dir}/${supercell_base}"
    local uc_in="${inputs_dir}/${unitcell_base}"
    local sc_out="${inputs_dir}/${supercell_base}_vol${vol_tag}"
    local uc_out="${inputs_dir}/${unitcell_base}_vol${vol_tag}"

    python "${script_dir}/generate_scaled_poscar.py" "$sc_in" "$vol_factor" "$sc_out"
    python "${script_dir}/generate_scaled_poscar.py" "$uc_in" "$vol_factor" "$uc_out"

    # Return the filenames (without path) via global variables
    SCALED_SUPERCELL="${supercell_base}_vol${vol_tag}"
    SCALED_UNITCELL="${unitcell_base}_vol${vol_tag}"
}

# ============================================================================
# Zr Small Perfect - Equilibrium Volume: single volume
# ============================================================================
run_zr_perfect_small_eqvol_single() {
    local vol_factor=$1
    local vol_tag
    vol_tag=$(printf "%.2f" "$vol_factor")
    local vol_output_dir="${OUTPUT_DIR}/vol_${vol_tag}"
    mkdir -p "$vol_output_dir"

    echo "##############################################################"
    echo "# Zr Perfect Small - Equilibrium Volume: ${vol_factor}"
    echo "# Output: ${vol_output_dir}"
    echo "##############################################################"

    # Generate scaled POSCAR files
    generate_scaled_poscars \
        "POSCAR_Zr_perfect_supercell_54atoms" \
        "POSCAR_Zr_perfect_unitcell_2atoms" \
        "$vol_factor"

    for temp in "${ZR_TEMPERATURES[@]}"; do
        run_tdep "Zr_perfect_small" $temp \
            "$SCALED_SUPERCELL" \
            "$SCALED_UNITCELL" \
            "--eam-potential-file $ZR_EAM_POTENTIAL" \
            "$ZR_QGRID_SMALL_CELL" \
            "$vol_output_dir"
    done
}

# Zr Small Perfect - Equilibrium Volume: all volumes
run_zr_perfect_small_eqvol() {
    echo "##############################################################"
    echo "# Zr Perfect Small - Equilibrium Volume (all)"
    echo "# Volumes: ${VOLUME_FACTORS[*]}"
    echo "##############################################################"

    for vol_factor in "${VOLUME_FACTORS[@]}"; do
        run_zr_perfect_small_eqvol_single "$vol_factor"
    done
}

# ============================================================================
# Mo Perfect - Equilibrium Volume: single volume
# ============================================================================
run_mo_perfect_eqvol_single() {
    local vol_factor=$1
    local vol_tag
    vol_tag=$(printf "%.2f" "$vol_factor")
    local vol_output_dir="${OUTPUT_DIR}/vol_${vol_tag}"
    mkdir -p "$vol_output_dir"

    echo "##############################################################"
    echo "# Mo Perfect - Equilibrium Volume: ${vol_factor}"
    echo "# Output: ${vol_output_dir}"
    echo "##############################################################"

    # Generate scaled POSCAR files
    generate_scaled_poscars \
        "POSCAR_Mo_perfect_supercell" \
        "POSCAR_Mo_perfect_unitcell" \
        "$vol_factor"

    for temp in "${MO_TEMPERATURES[@]}"; do
        run_tdep "Mo_perfect" $temp \
            "$SCALED_SUPERCELL" \
            "$SCALED_UNITCELL" \
            "--kim-model $MO_KIM_MODEL" \
            "$MO_QGRID" \
            "$vol_output_dir"
    done
}

# Mo Perfect - Equilibrium Volume: all volumes
run_mo_perfect_eqvol() {
    echo "##############################################################"
    echo "# Mo Perfect - Equilibrium Volume (all)"
    echo "# Volumes: ${VOLUME_FACTORS[*]}"
    echo "##############################################################"

    for vol_factor in "${VOLUME_FACTORS[@]}"; do
        run_mo_perfect_eqvol_single "$vol_factor"
    done
}

# ============================================================================
# ZrO2 Functions (Perfect and Defected)
# ============================================================================
run_zro2_perfect_temperature() {
    local temp=$1

    echo "##############################################################"
    echo "# ZrO2 Perfect System at ${temp}K"
    echo "##############################################################"

    local temp_dir="${ZRO2_INPUT_BASE}/T_${temp}K_perfect"

    if [ ! -d "$temp_dir" ]; then
        echo "ERROR: Temperature directory not found: $temp_dir"
        return 1
    fi

    # Check if POSCAR files exist
    if [ ! -f "${temp_dir}/POSCAR_4x4x4_supercell" ] || [ ! -f "${temp_dir}/POSCAR_primitive" ]; then
        echo "ERROR: POSCAR files not found in $temp_dir"
        return 1
    fi

    # Copy POSCAR files to inputs directory temporarily
    cp "${temp_dir}/POSCAR_4x4x4_supercell" "../inputs/POSCAR_ZrO2_perfect_${temp}K_supercell"
    cp "${temp_dir}/POSCAR_primitive" "../inputs/POSCAR_ZrO2_perfect_${temp}K_unitcell"

    run_tdep_zro2 "ZrO2_perfect" $temp \
        "POSCAR_ZrO2_perfect_${temp}K_supercell" \
        "POSCAR_ZrO2_perfect_${temp}K_unitcell" \
        "" \
        "$ZRO2_QGRID" \
        "$OUTPUT_DIR"

    # Clean up temporary files
    rm -f "../inputs/POSCAR_ZrO2_perfect_${temp}K_supercell"
    rm -f "../inputs/POSCAR_ZrO2_perfect_${temp}K_unitcell"
}

run_zro2_defected_temperature() {
    local temp=$1

    echo "##############################################################"
    echo "# ZrO2 Defected System at ${temp}K (1 O vacancy)"
    echo "##############################################################"

    local temp_dir="${ZRO2_INPUT_BASE}/T_${temp}K_defected"

    if [ ! -d "$temp_dir" ]; then
        echo "ERROR: Temperature directory not found: $temp_dir"
        return 1
    fi

    # Check if POSCAR files exist
    if [ ! -f "${temp_dir}/POSCAR_4x4x4_supercell" ] || [ ! -f "${temp_dir}/POSCAR_unitcell" ]; then
        echo "ERROR: POSCAR files not found in $temp_dir"
        return 1
    fi

    # Copy POSCAR files to inputs directory temporarily
    cp "${temp_dir}/POSCAR_4x4x4_supercell" "../inputs/POSCAR_ZrO2_defected_${temp}K_supercell"
    cp "${temp_dir}/POSCAR_unitcell" "../inputs/POSCAR_ZrO2_defected_${temp}K_unitcell"

    run_tdep_zro2 "ZrO2_defected" $temp \
        "POSCAR_ZrO2_defected_${temp}K_supercell" \
        "POSCAR_ZrO2_defected_${temp}K_unitcell" \
        "" \
        "$ZRO2_QGRID" \
        "$OUTPUT_DIR"

    # Clean up temporary files
    rm -f "../inputs/POSCAR_ZrO2_defected_${temp}K_supercell"
    rm -f "../inputs/POSCAR_ZrO2_defected_${temp}K_unitcell"
}

# Combined function for both perfect and defected at a temperature
run_zro2_temperature() {
    local temp=$1
    run_zro2_perfect_temperature $temp
    run_zro2_defected_temperature $temp
}

run_zro2_2600() {
    run_zro2_temperature 2600
}

run_zro2_2800() {
    run_zro2_temperature 2800
}

run_zro2_2900() {
    run_zro2_temperature 2900
}

run_zro2_3000() {
    run_zro2_temperature 3000
}

# Perfect-only functions
run_zro2_perfect_2600() {
    run_zro2_perfect_temperature 2600
}

run_zro2_perfect_2800() {
    run_zro2_perfect_temperature 2800
}

run_zro2_perfect_2900() {
    run_zro2_perfect_temperature 2900
}

run_zro2_perfect_3000() {
    run_zro2_perfect_temperature 3000
}

run_zro2_perfect_all() {
    echo "##############################################################"
    echo "# All ZrO2 Perfect Systems"
    echo "##############################################################"

    for temp in "${ZRO2_TEMPERATURES[@]}"; do
        run_zro2_perfect_temperature $temp
    done
}

# Defected-only functions
run_zro2_defected_2600() {
    run_zro2_defected_temperature 2600
}

run_zro2_defected_2800() {
    run_zro2_defected_temperature 2800
}

run_zro2_defected_2900() {
    run_zro2_defected_temperature 2900
}

run_zro2_defected_3000() {
    run_zro2_defected_temperature 3000
}

run_zro2_defected_all() {
    echo "##############################################################"
    echo "# All ZrO2 Defected Systems"
    echo "##############################################################"

    for temp in "${ZRO2_TEMPERATURES[@]}"; do
        run_zro2_defected_temperature $temp
    done
}

# All ZrO2 (perfect and defected at all temperatures)
run_zro2_all() {
    echo "##############################################################"
    echo "# All ZrO2 Systems (Perfect and Defected)"
    echo "##############################################################"

    for temp in "${ZRO2_TEMPERATURES[@]}"; do
        run_zro2_temperature $temp
    done
}

# ============================================================================
# ZrO2 Small Functions (3x3x3 supercell - Perfect and Defected)
# ============================================================================
run_zro2_small_perfect_temperature() {
    local temp=$1

    echo "##############################################################"
    echo "# ZrO2 Small Perfect System at ${temp}K (3x3x3 - 81 atoms)"
    echo "##############################################################"

    local temp_dir="${ZRO2_SMALL_INPUT_BASE}/T_${temp}K_perfect"

    if [ ! -d "$temp_dir" ]; then
        echo "ERROR: Temperature directory not found: $temp_dir"
        return 1
    fi

    # Check if POSCAR files exist
    if [ ! -f "${temp_dir}/POSCAR_3x3x3_supercell" ] || [ ! -f "${temp_dir}/POSCAR_primitive" ]; then
        echo "ERROR: POSCAR files not found in $temp_dir"
        return 1
    fi

    # Copy POSCAR files to inputs directory temporarily
    cp "${temp_dir}/POSCAR_3x3x3_supercell" "../inputs/POSCAR_ZrO2_small_perfect_${temp}K_supercell"
    cp "${temp_dir}/POSCAR_primitive" "../inputs/POSCAR_ZrO2_small_perfect_${temp}K_unitcell"

    run_tdep_zro2 "ZrO2_small_perfect" $temp \
        "POSCAR_ZrO2_small_perfect_${temp}K_supercell" \
        "POSCAR_ZrO2_small_perfect_${temp}K_unitcell" \
        "" \
        "$ZRO2_SMALL_QGRID" \
        "$OUTPUT_DIR"

    # Clean up temporary files
    rm -f "../inputs/POSCAR_ZrO2_small_perfect_${temp}K_supercell"
    rm -f "../inputs/POSCAR_ZrO2_small_perfect_${temp}K_unitcell"
}

run_zro2_small_defected_temperature() {
    local temp=$1

    echo "##############################################################"
    echo "# ZrO2 Small Defected System at ${temp}K (3x3x3 - 80 atoms, 1 O vacancy)"
    echo "##############################################################"

    local temp_dir="${ZRO2_SMALL_INPUT_BASE}/T_${temp}K_defected"

    if [ ! -d "$temp_dir" ]; then
        echo "ERROR: Temperature directory not found: $temp_dir"
        return 1
    fi

    # Check if POSCAR files exist
    if [ ! -f "${temp_dir}/POSCAR_3x3x3_supercell" ] || [ ! -f "${temp_dir}/POSCAR_unitcell" ]; then
        echo "ERROR: POSCAR files not found in $temp_dir"
        return 1
    fi

    # Copy POSCAR files to inputs directory temporarily
    cp "${temp_dir}/POSCAR_3x3x3_supercell" "../inputs/POSCAR_ZrO2_small_defected_${temp}K_supercell"
    cp "${temp_dir}/POSCAR_unitcell" "../inputs/POSCAR_ZrO2_small_defected_${temp}K_unitcell"

    run_tdep_zro2 "ZrO2_small_defected" $temp \
        "POSCAR_ZrO2_small_defected_${temp}K_supercell" \
        "POSCAR_ZrO2_small_defected_${temp}K_unitcell" \
        "" \
        "$ZRO2_SMALL_QGRID" \
        "$OUTPUT_DIR"

    # Clean up temporary files
    rm -f "../inputs/POSCAR_ZrO2_small_defected_${temp}K_supercell"
    rm -f "../inputs/POSCAR_ZrO2_small_defected_${temp}K_unitcell"
}

# Combined function for both perfect and defected at a temperature
run_zro2_small_temperature() {
    local temp=$1
    run_zro2_small_perfect_temperature $temp
    run_zro2_small_defected_temperature $temp
}

run_zro2_small_2600() {
    run_zro2_small_temperature 2600
}

run_zro2_small_2800() {
    run_zro2_small_temperature 2800
}

run_zro2_small_2900() {
    run_zro2_small_temperature 2900
}

run_zro2_small_3000() {
    run_zro2_small_temperature 3000
}

# Perfect-only functions for small cells
run_zro2_small_perfect_2600() {
    run_zro2_small_perfect_temperature 2600
}

run_zro2_small_perfect_2800() {
    run_zro2_small_perfect_temperature 2800
}

run_zro2_small_perfect_2900() {
    run_zro2_small_perfect_temperature 2900
}

run_zro2_small_perfect_3000() {
    run_zro2_small_perfect_temperature 3000
}

run_zro2_small_perfect_all() {
    echo "##############################################################"
    echo "# All ZrO2 Small Perfect Systems (3x3x3)"
    echo "##############################################################"

    for temp in "${ZRO2_TEMPERATURES[@]}"; do
        run_zro2_small_perfect_temperature $temp
    done
}

# Defected-only functions for small cells
run_zro2_small_defected_2600() {
    run_zro2_small_defected_temperature 2600
}

run_zro2_small_defected_2800() {
    run_zro2_small_defected_temperature 2800
}

run_zro2_small_defected_2900() {
    run_zro2_small_defected_temperature 2900
}

run_zro2_small_defected_3000() {
    run_zro2_small_defected_temperature 3000
}

run_zro2_small_defected_all() {
    echo "##############################################################"
    echo "# All ZrO2 Small Defected Systems (3x3x3)"
    echo "##############################################################"

    for temp in "${ZRO2_TEMPERATURES[@]}"; do
        run_zro2_small_defected_temperature $temp
    done
}

# All ZrO2 small (perfect and defected at all temperatures)
run_zro2_small_all() {
    echo "##############################################################"
    echo "# All ZrO2 Small Systems (3x3x3 - Perfect and Defected)"
    echo "##############################################################"

    for temp in "${ZRO2_TEMPERATURES[@]}"; do
        run_zro2_small_temperature $temp
    done
}

# ============================================================================
# Main
# ============================================================================

case "${SYSTEM:-all}" in
    mo_perfect)
        run_mo_perfect
        ;;
    mo_defected)
        run_mo_defected
        ;;
    mo)
        run_mo_perfect
        run_mo_defected
        ;;
    zr_perfect)
        run_zr_perfect
        ;;
    zr_defected)
        run_zr_defected
        ;;
    zr)
        run_zr_perfect
        run_zr_defected
        ;;
    zr_perfect_small)
        run_zr_perfect_small
        ;;
    zr_defected_small)
        run_zr_defected_small
        ;;
    zr_small)
        run_zr_perfect_small
        run_zr_defected_small
        ;;
    zr_small_equilibrium_volume)
        run_zr_perfect_small_eqvol
        ;;
    zr_small_equilibrium_volume_*)
        vol="${SYSTEM#zr_small_equilibrium_volume_}"
        run_zr_perfect_small_eqvol_single "$vol"
        ;;
    mo_equilibrium_volume)
        run_mo_perfect_eqvol
        ;;
    mo_equilibrium_volume_*)
        vol="${SYSTEM#mo_equilibrium_volume_}"
        run_mo_perfect_eqvol_single "$vol"
        ;;
    zro2_2600)
        run_zro2_temperature 2600
        ;;
    zro2_2800)
        run_zro2_temperature 2800
        ;;
    zro2_2900)
        run_zro2_temperature 2900
        ;;
    zro2_3000)
        run_zro2_temperature 3000
        ;;
    zro2_perfect_2600)
        run_zro2_perfect_2600
        ;;
    zro2_perfect_2800)
        run_zro2_perfect_2800
        ;;
    zro2_perfect_2900)
        run_zro2_perfect_2900
        ;;
    zro2_perfect_3000)
        run_zro2_perfect_3000
        ;;
    zro2_perfect|zro2_perfect_all)
        run_zro2_perfect_all
        ;;
    zro2_defected_2600)
        run_zro2_defected_2600
        ;;
    zro2_defected_2800)
        run_zro2_defected_2800
        ;;
    zro2_defected_2900)
        run_zro2_defected_2900
        ;;
    zro2_defected_3000)
        run_zro2_defected_3000
        ;;
    zro2_defected|zro2_defected_all)
        run_zro2_defected_all
        ;;
    zro2|zro2_all)
        run_zro2_all
        ;;
    # ZrO2 Small (3x3x3 supercell)
    zro2_small_2600)
        run_zro2_small_temperature 2600
        ;;
    zro2_small_2800)
        run_zro2_small_temperature 2800
        ;;
    zro2_small_2900)
        run_zro2_small_temperature 2900
        ;;
    zro2_small_3000)
        run_zro2_small_temperature 3000
        ;;
    zro2_small_perfect_2600)
        run_zro2_small_perfect_2600
        ;;
    zro2_small_perfect_2800)
        run_zro2_small_perfect_2800
        ;;
    zro2_small_perfect_2900)
        run_zro2_small_perfect_2900
        ;;
    zro2_small_perfect_3000)
        run_zro2_small_perfect_3000
        ;;
    zro2_small_perfect|zro2_small_perfect_all)
        run_zro2_small_perfect_all
        ;;
    zro2_small_defected_2600)
        run_zro2_small_defected_2600
        ;;
    zro2_small_defected_2800)
        run_zro2_small_defected_2800
        ;;
    zro2_small_defected_2900)
        run_zro2_small_defected_2900
        ;;
    zro2_small_defected_3000)
        run_zro2_small_defected_3000
        ;;
    zro2_small_defected|zro2_small_defected_all)
        run_zro2_small_defected_all
        ;;
    zro2_small|zro2_small_all)
        run_zro2_small_all
        ;;
    all)
        run_mo_perfect
        run_mo_defected
        run_zr_perfect
        run_zr_defected
        ;;
    all_small)
        run_mo_perfect
        run_mo_defected
        run_zr_perfect_small
        run_zr_defected_small
        ;;
    all_with_zro2)
        run_mo_perfect
        run_mo_defected
        run_zr_perfect_small
        run_zr_defected_small
        run_zro2_all
        ;;
    all_with_zro2_small)
        run_mo_perfect
        run_mo_defected
        run_zr_perfect_small
        run_zr_defected_small
        run_zro2_small_all
        ;;
    *)
        echo "Usage: $0 [options] <system>"
        echo ""
        echo "Options:"
        echo "  --output-dir <path>    Custom output directory (default: ../outputs)"
        echo "  --cleanup-steps        Keep only last 2 TDEP step folders"
        echo "  --cleanup-calc         Delete calc_* directories after processing (saves disk space)"
        echo "  --zro2-potential <pot> ZrO2 potential choice: 'gaop' (default, LAMMPS) or 'uzro' (EAM)"
        echo ""
        echo "Systems:"
        echo "  Mo systems:"
        echo "    mo_perfect       - Mo perfect system (54 atoms)"
        echo "    mo_defected      - Mo defected system (53 atoms)"
        echo "    mo               - All Mo systems"
        echo ""
        echo "  Zr systems:"
        echo "    zr_perfect       - Zr perfect system (216 atoms, large)"
        echo "    zr_defected      - Zr defected system (215 atoms, large)"
        echo "    zr               - All Zr systems (large)"
        echo "    zr_perfect_small - Zr perfect system (54 atoms, small)"
        echo "    zr_defected_small- Zr defected system (53 atoms, small)"
        echo "    zr_small         - All Zr systems (small)"
        echo "    zr_small_equilibrium_volume - Zr perfect small at multiple volumes (${VOLUME_FACTORS[*]})"
        echo ""
        echo "  Equilibrium volume (all volumes: ${VOLUME_FACTORS[*]}):"
        echo "    mo_equilibrium_volume            - Mo perfect, all volumes"
        echo "    mo_equilibrium_volume_<vol>       - Mo perfect, single volume (e.g. mo_equilibrium_volume_0.97)"
        echo "    zr_small_equilibrium_volume       - Zr perfect small, all volumes"
        echo "    zr_small_equilibrium_volume_<vol> - Zr perfect small, single volume (e.g. zr_small_equilibrium_volume_0.97)"
        echo ""
        echo "  ZrO2 systems:"
        echo "    zro2_2600        - ZrO2 perfect & defected at 2600K"
        echo "    zro2_2800        - ZrO2 perfect & defected at 2800K"
        echo "    zro2_2900        - ZrO2 perfect & defected at 2900K"
        echo "    zro2_3000        - ZrO2 perfect & defected at 3000K"
        echo ""
        echo "    zro2_perfect_2600 - ZrO2 perfect only at 2600K"
        echo "    zro2_perfect_2800 - ZrO2 perfect only at 2800K"
        echo "    zro2_perfect_2900 - ZrO2 perfect only at 2900K"
        echo "    zro2_perfect_3000 - ZrO2 perfect only at 3000K"
        echo "    zro2_perfect     - All ZrO2 perfect (all temperatures)"
        echo ""
        echo "    zro2_defected_2600 - ZrO2 defected only at 2600K"
        echo "    zro2_defected_2800 - ZrO2 defected only at 2800K"
        echo "    zro2_defected_2900 - ZrO2 defected only at 2900K"
        echo "    zro2_defected_3000 - ZrO2 defected only at 3000K"
        echo "    zro2_defected    - All ZrO2 defected (all temperatures)"
        echo ""
        echo "    zro2, zro2_all   - All ZrO2 (perfect & defected, all temps)"
        echo ""
        echo "  ZrO2 Small systems (3x3x3 - 81/80 atoms):"
        echo "    zro2_small_2600  - ZrO2 small perfect & defected at 2600K"
        echo "    zro2_small_2800  - ZrO2 small perfect & defected at 2800K"
        echo "    zro2_small_2900  - ZrO2 small perfect & defected at 2900K"
        echo "    zro2_small_3000  - ZrO2 small perfect & defected at 3000K"
        echo ""
        echo "    zro2_small_perfect_2600 - ZrO2 small perfect only at 2600K"
        echo "    zro2_small_perfect      - All ZrO2 small perfect (all temps)"
        echo "    zro2_small_defected     - All ZrO2 small defected (all temps)"
        echo "    zro2_small, zro2_small_all - All ZrO2 small (P & D, all temps)"
        echo ""
        echo "  Combined:"
        echo "    all              - All Mo and Zr systems (large)"
        echo "    all_small        - All Mo and Zr systems (small)"
        echo "    all_with_zro2    - All small systems + ZrO2 (4x4x4)"
        echo "    all_with_zro2_small - All small systems + ZrO2 small (3x3x3)"
        echo ""
        echo "Examples:"
        echo "  $0 zr_small"
        echo "  $0 --output-dir ../outputs_custom zro2_2600"
        echo "  $0 --cleanup-steps --output-dir ../outputs_zro2 zro2"
        echo "  $0 --cleanup-steps --cleanup-calc zro2_small  # ZrO2 small with cleanup"
        echo "  $0 --zro2-potential uzro zro2_small  # Use UZrO.eam.alloy potential"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "All requested calculations completed!"
echo "Output directory: $OUTPUT_DIR"
if [ "$CLEANUP_STEPS" = "YES" ]; then
    echo "Old TDEP steps cleaned up (kept last 2 only)"
fi
if [ "$CLEANUP_CALC" = "YES" ]; then
    echo "calc_* directories cleaned up after processing"
fi
echo "============================================================"
