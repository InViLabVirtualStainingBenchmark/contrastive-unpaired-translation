#!/bin/bash
#SBATCH --job-name=cut_train_MIST_512_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/train_MIST_512_p2.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/train_MIST_512_p2.%j.err

# train_MIST-HER2_full_e100_part2.sh
# Epochs 51-100 of CUT training on MIST-HER2 at 512x512.
# Resumes from the latest checkpoint saved by part 1.
# Linear LR decay is applied across these 50 epochs (n_epochs_decay=50).
#
# The LR schedule is equivalent to a single 100-epoch run:
#   Part 1: epochs  1-50  constant LR  (n_epochs=50, n_epochs_decay=0)
#   Part 2: epochs 51-100 linear decay (epoch_count=51, n_epochs=50, n_epochs_decay=50)
#
# DO NOT submit this manually before part 1 finishes -- use submit_MIST_e100.sh.
# If part 1 failed or was cancelled, do not submit this script.
#
# Monitor:
#   squeue -u $USER
#   tail -f $VSC_DATA/projects/cut/logs/train_MIST_512_p2.<jobid>.out
#
# After this job completes, next step: sbatch infer_MIST-HER2_full_e100.sh

set -euo pipefail

REPO_DIR="$VSC_DATA/projects/cut/code/cut"
DATA_ROOT="$VSC_SCRATCH/cut-MIST-HER2"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RUN_NAME="MIST-HER2_512_e100"

# =========================
# MODULES
# =========================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VSC_DATA/projects/cut/venv_cut/bin/activate"

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Environment ==="
which python
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Dataset check ==="
if [ ! -d "$DATA_ROOT/trainA" ]; then
    echo "ERROR: $DATA_ROOT/trainA not found. Run prepare_datasets.sh first."
    deactivate; exit 1
fi
echo "  trainA: $(find "$DATA_ROOT"/trainA -maxdepth 1 \( -type f -o -type l \) | wc -l) images"
echo "  trainB: $(find "$DATA_ROOT"/trainB -maxdepth 1 \( -type f -o -type l \) | wc -l) images"

echo ""
echo "=== Checkpoint check (part 1 must have completed) ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -f "$CKPT_DIR/latest_net_G.pth" ]; then
    echo "ERROR: latest_net_G.pth not found in $CKPT_DIR"
    echo "Has part 1 completed successfully?"
    deactivate; exit 1
fi
echo "  latest checkpoint found:"
find "$CKPT_DIR" -name "latest_net_*.pth" | sort

echo ""
echo "=== Repo check ==="
if [ ! -f "$REPO_DIR/train.py" ]; then
    echo "ERROR: train.py not found in $REPO_DIR"
    deactivate; exit 1
fi
echo "  train.py found"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/cut/logs/gpu_train_MIST_512_p2.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST-HER2 training part 2 (epochs 51-100, LR decay) ==="
echo "  run name    : $RUN_NAME"
echo "  data        : $DATA_ROOT"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"

python train.py \
    --dataroot "$DATA_ROOT" \
    --name "$RUN_NAME" \
    --CUT_mode CUT \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --load_size 512 \
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
tail -3 "$VSC_DATA/projects/cut/logs/gpu_train_MIST_512_p2.csv"

deactivate
echo ""
echo "MIST-HER2 full training complete (epochs 1-100). Next step: sbatch infer_MIST-HER2_full_e100.sh"