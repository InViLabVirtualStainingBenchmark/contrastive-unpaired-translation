#!/bin/bash
# setup_project.sh
# Run once manually on the login node after first SSH login.
# Creates the full project folder tree under $VSC_DATA and $VSC_SCRATCH.
#
# Usage:
#   bash setup_project.sh
#
# Do NOT submit this with sbatch.

set -euo pipefail

echo "VSC_DATA  : $VSC_DATA"
echo "VSC_SCRATCH: $VSC_SCRATCH"

BASE_DIR="$VSC_DATA/projects/cut"

echo ""
echo "Creating project folder tree at: $BASE_DIR"

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/jobs"
mkdir -p "$BASE_DIR/outputs/checkpoints"
mkdir -p "$BASE_DIR/outputs/results"

echo "Created:"
echo "  $BASE_DIR/logs          <- Slurm stdout/stderr and GPU logs"
echo "  $BASE_DIR/jobs          <- job scripts (optional copy location)"
echo "  $BASE_DIR/outputs/checkpoints  <- model checkpoints per run"
echo "  $BASE_DIR/outputs/results      <- inference output images per run"

echo ""
echo "Creating scratch dataset root at: $VSC_SCRATCH/dataset"
mkdir -p "$VSC_SCRATCH/dataset"

echo ""
echo "Done. Next step: transfer datasets via rsync, then run prepare_datasets.sh"
