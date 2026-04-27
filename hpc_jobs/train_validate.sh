#!/bin/bash
#SBATCH --job-name=cut_train_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/train_validate.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/train_validate.%j.err

# train_validate.sh
# Runs 5 epochs of CUT training on BCI as a cluster confirmation gate.
# This job must pass before submitting the full training jobs.
#
# Submit: sbatch train_validate.sh
#
# Pass criteria:
#   1. Job exits cleanly (no Python traceback in log)
#   2. Loss values in log are not NaN
#   3. Checkpoint files exist after the job:
#        find $VSC_DATA/projects/cut/outputs/checkpoints/BCI_smoke_e5 -name "*.pth"
#   4. GPU log CSV has entries:
#        cat $VSC_DATA/projects/cut/logs/gpu_train_validate.csv | tail -5

set -euo pipefail

REPO_DIR="$VSC_DATA/projects/cut/code/cut"
DATA_ROOT="$VSC_SCRATCH/cut-BCI"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RUN_NAME="BCI_smoke_e5"

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
    > "$VSC_DATA/projects/cut/logs/gpu_train_validate.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting validation training (5 epochs) ==="
echo "  run name    : $RUN_NAME"
echo "  data        : $DATA_ROOT"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"

python train.py \
    --dataroot "$DATA_ROOT" \
    --name "$RUN_NAME" \
    --CUT_mode CUT \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --load_size 1024 \
    --crop_size 1024 \
    --display_id 0 \
    --n_epochs 5 \
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
tail -3 "$VSC_DATA/projects/cut/logs/gpu_train_validate.csv"

deactivate
echo ""
echo "Validation training complete. Review the output above before submitting full runs."
