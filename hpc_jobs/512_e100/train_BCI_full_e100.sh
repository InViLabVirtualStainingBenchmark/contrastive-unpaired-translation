#!/bin/bash
#SBATCH --job-name=cut_train_BCI_512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=XX:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/train_BCI_512.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/train_BCI_512.%j.err

# train_BCI_512_e100.sh
# Full 100-epoch CUT training on the BCI dataset at 512x512 resolution.
# 50 epochs constant LR + 50 epochs linear LR decay = 100 total.
#
# TODO: Replace XX:00:00 above with the correct wall time before submitting.
#       Formula: (time_per_iter * iters_per_epoch / 60) * 100 * 1.20 = minutes needed.
#       Round up to the nearest hour.
#
# Submit ONLY after train_validate_BCI_512.sh has passed.
# Submit: sbatch train_BCI_512_e100.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f $VSC_DATA/projects/cut/logs/train_BCI_512.<jobid>.out
#   tail -5 $VSC_DATA/projects/cut/logs/gpu_train_BCI_512.csv
#
# Checkpoints saved every 25 epochs to:
#   $VSC_DATA/projects/cut/outputs/checkpoints/BCI_512_e100/

set -euo pipefail

REPO_DIR="$VSC_DATA/projects/cut/code/cut"
DATA_ROOT="$VSC_SCRATCH/cut-BCI"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RUN_NAME="BCI_512_e100"

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
    > "$VSC_DATA/projects/cut/logs/gpu_train_BCI_512.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting full BCI training (100 epochs, 512x512) ==="
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
tail -3 "$VSC_DATA/projects/cut/logs/gpu_train_BCI_512.csv"

deactivate
echo ""
echo "BCI full training complete. Next step: sbatch infer_BCI_512_e100.sh"