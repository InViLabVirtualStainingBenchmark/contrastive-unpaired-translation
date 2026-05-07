#!/bin/bash
#SBATCH --job-name=cut_train_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/train_BCI.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/train_BCI.%j.err

# train_BCI_e100.sh
# Full 100-epoch CUT training on BCI at load 1024 / crop 512.
# 50 epochs constant LR + 50 epochs linear LR decay = 100 total.
#
# Submit ONLY after train_validate_BCI.sh has passed.
# Submit: sbatch train_BCI_e100.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f $VSC_DATA/projects/cut/logs/train_BCI.<jobid>.out
#   tail -5 $VSC_DATA/projects/cut/logs/gpu_train_BCI.csv
#
# Checkpoints saved every 25 epochs to:
#   $VSC_DATA/projects/cut/outputs/checkpoints/BCI_e100/

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/cut_nvidia.sif"
BCI_SQSH="$VSC_SCRATCH/BCI-AB.sqsh"
BCI_MNT="$VSC_SCRATCH/sqsh_mnt/BCI"
REPO_DIR="$VSC_DATA/projects/cut/code/cut"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RUN_NAME="BCI_e100"

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
    > "$VSC_DATA/projects/cut/logs/gpu_train_BCI.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

mkdir -p "$BCI_MNT"

echo ""
echo "=== Starting full BCI training (100 epochs, load 1024, crop 512) ==="
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
        --n_epochs 50 \
        --n_epochs_decay 50 \
        --save_epoch_freq 25 \
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
tail -3 "$VSC_DATA/projects/cut/logs/gpu_train_BCI.csv"

echo ""
echo "BCI full training complete. Next step: sbatch infer_BCI_full_e100.sh"
