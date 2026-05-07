#!/bin/bash
#SBATCH --job-name=cut_train_validate_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/train_validate_BCI.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/train_validate_BCI.%j.err

# train_validate_BCI.sh
# Runs 2 epochs of CUT training on BCI as a cluster confirmation gate.
# This job must pass before submitting the full training jobs.
#
# Submit: sbatch train_validate_BCI.sh
#
# Pass criteria:
#   1. Job exits cleanly (no Python traceback in log)
#   2. Loss values in log are not NaN
#   3. Checkpoint files exist after the job:
#        find $VSC_DATA/projects/cut/outputs/checkpoints/BCI_smoke_e2 -name "*.pth"
#   4. GPU log CSV has entries:
#        tail -5 $VSC_DATA/projects/cut/logs/gpu_train_validate_BCI.csv

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/cut_nvidia.sif"
BCI_SQSH="$VSC_SCRATCH/BCI-AB.sqsh"
BCI_MNT="$VSC_SCRATCH/sqsh_mnt/BCI"
REPO_DIR="$VSC_DATA/projects/cut/code/cut"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RUN_NAME="BCI_smoke_e2"

# =========================
# MODULES
# =========================

module purge
module load calcua/2026.1

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Container check ==="
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: container not found: $CONTAINER"
    exit 1
fi
echo "  Container found: $CONTAINER"

echo ""
echo "=== Environment ==="
apptainer exec --nv "$CONTAINER" python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== SquashFS check ==="
if [ ! -f "$BCI_SQSH" ]; then
    echo "ERROR: BCI-AB.sqsh not found: $BCI_SQSH"
    exit 1
fi
echo "  BCI-AB.sqsh found"

echo ""
echo "=== Repo check ==="
if [ ! -f "$REPO_DIR/train.py" ]; then
    echo "ERROR: train.py not found in $REPO_DIR"
    exit 1
fi
echo "  train.py found"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/cut/logs/gpu_train_validate_BCI.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

mkdir -p "$BCI_MNT"

echo ""
echo "=== Starting validation training (2 epochs, load 1024, crop 512) ==="
echo "  run name    : $RUN_NAME"
echo "  data        : $BCI_MNT (mounted from BCI-AB.sqsh)"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_SQSH:$BCI_MNT:image-src=/" \
    "$CONTAINER" \
    python "$REPO_DIR/train.py" \
        --dataroot "$BCI_MNT" \
        --name "$RUN_NAME" \
        --CUT_mode CUT \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --load_size 1024 \
        --crop_size 512 \
        --display_id 0 \
        --n_epochs 2 \
        --n_epochs_decay 0 \
        --save_epoch_freq 1 \
        --no_html \
        --gpu_ids 0

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Post-run checkpoint check ==="
find "$CHECKPOINTS_DIR/$RUN_NAME" -name "*.pth" | sort

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/cut/logs/gpu_train_validate_BCI.csv"

echo ""
echo "Validation training complete. Review the output above before submitting full runs."
