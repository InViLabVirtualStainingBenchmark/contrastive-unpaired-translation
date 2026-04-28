#!/bin/bash
#SBATCH --job-name=cut_train_MIST_512_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/train_MIST_512_p1.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/train_MIST_512_p1.%j.err

# train_MIST-HER2_full_e100_part1.sh
# Epochs 1-50 of CUT training on MIST-HER2 at 512x512.
# Constant LR throughout (n_epochs_decay=0).
# MIST-HER2 has 4642 training images -- too slow to finish 100 epochs in one 24h job.
# This is part 1 of 2. Part 2 resumes from the latest checkpoint and applies LR decay.
#
# DO NOT submit part 2 manually -- use submit_MIST_e100.sh which chains them automatically.
# Or if submitting manually, wait for this job to complete before submitting part 2.
#
# Submit via wrapper (recommended):
#   bash hpc_jobs/512_e100/submit_MIST_e100.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f $VSC_DATA/projects/cut/logs/train_MIST_512_p1.<jobid>.out
#
# Checkpoints saved every 25 epochs to:
#   $VSC_DATA/projects/cut/outputs/checkpoints/MIST-HER2_512_e100/

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
    > "$VSC_DATA/projects/cut/logs/gpu_train_MIST_512_p1.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST-HER2 training part 1 (epochs 1-50, constant LR) ==="
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
    --n_epochs_decay 0 \
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
tail -3 "$VSC_DATA/projects/cut/logs/gpu_train_MIST_512_p1.csv"

deactivate
echo ""
echo "MIST-HER2 part 1 complete (epochs 1-50). Part 2 should start automatically if submitted via wrapper."