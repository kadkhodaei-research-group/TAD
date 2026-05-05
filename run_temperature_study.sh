#!/bin/bash

# Run temperature study for Zr system
# This script executes the Python setup and can optionally start the runs

cd "$(dirname "$0")"

echo "======================================"
echo "TEMPERATURE STUDY FOR Zr SYSTEM"
echo "======================================"
echo ""

# Run the Python script to set up the study
python3 run_temperature_study_zr.py

# Get the study directory from the output
STUDY_DIR=$(python3 -c "
import time
from pathlib import Path
timestamp = int(time.time())
study_dir = Path('$(pwd)').parent / 'outputs' / f'temperature_study_zr_{timestamp}'
# Find most recent study directory
import glob
dirs = sorted(glob.glob('../outputs/temperature_study_zr_*'))
if dirs:
    print(dirs[-1])
")

if [ -n "$STUDY_DIR" ]; then
    echo ""
    echo "Study directory created: $STUDY_DIR"
    echo ""
    echo "To start the runs, execute:"
    echo "  cd $STUDY_DIR"
    echo "  ./run_parallel.sh"
    echo ""
    echo "Or to run with limited parallelism (e.g., 5 at a time):"
    echo "  cd $STUDY_DIR"
    echo "  cat run_parallel.sh | sed 's/&$//' | xargs -I {} -P 5 bash -c '{}'"
fi