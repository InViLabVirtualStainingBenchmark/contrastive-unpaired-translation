#!/bin/bash
#SBATCH --job-name=cut_train_MIST_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/train_MIST_p2.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/train_MIST_p2.%j.err

# train_MIST-HER2_e100_part2.sh
# Epochs 51-100 of CUT training on MIST-HER2 at load 1024 / crop 512.
# Resumes from the latest checkpoint saved by part 1.
# Linear LR decay applied across these 50 epochs (n_epochs_decay=50).
#
# The LR schedule is equivalent to a single 100-epoch run:
#   Part 1: epochs  1-50  constant LR  (n_epochs=50, n_epochs_decay=0)
#   Part 2: epochs 51-100 linear decay (epoch_count=51, n_epochs=50, n_epochs_decay=50)
#
# DO NOT submit this manually before part 1 finishes -- use submit_MIST_e100.sh.
# If part 1 failed or was cancelled, do not submit this script.

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/cut_nvidia.sif"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/MIST-HER2"
REPO_DIR="$VSC_DATA/projects/cut/code/cut"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RUN_NAME="MIST-HER2_e100"

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
if [ ! -f "$MIST_SQSH" ]; then
    echo "ERROR: MIST-HER2 squashfs not found: $MIST_SQSH"
    exit 1
fi
echo "  MIST-HER2.sqsh found"

echo ""
echo "=== Checkpoint check (part 1 must have completed) ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -f "$CKPT_DIR/latest_net_G.pth" ]; then
    echo "ERROR: latest_net_G.pth not found in $CKPT_DIR"
    echo "Has part 1 completed successfully?"
    exit 1
fi
echo "  Latest checkpoint found:"
find "$CKPT_DIR" -name "latest_net_*.pth" | sort

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
    > "$VSC_DATA/projects/cut/logs/gpu_train_MIST_p2.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

mkdir -p "$MIST_MNT"

echo ""
echo "=== Starting MIST-HER2 training part 2 (epochs 51-100, LR decay, load 1024, crop 512) ==="
echo "  run name    : $RUN_NAME"
echo "  data        : $MIST_MNT (mounted from MIST-HER2.sqsh)"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    python "$REPO_DIR/train.py" \
        --dataroot "$MIST_MNT" \
        --name "$RUN_NAME" \
        --CUT_mode CUT \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --load_size 1024 \
        --crop_size 512 \
        --display_id 0 \
        --n_epochs 50 \
        --n_epochs_decay 50 \
        --epoch_count 51 \
        --continue_train \
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
tail -3 "$VSC_DATA/projects/cut/logs/gpu_train_MIST_p2.csv"

echo ""
echo "MIST-HER2 full training complete (epochs 1-100). Next step: sbatch infer_MIST-HER2_full_e100.sh"
