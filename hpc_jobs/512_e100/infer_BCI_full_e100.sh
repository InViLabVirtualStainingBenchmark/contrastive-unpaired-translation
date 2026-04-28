#!/bin/bash
#SBATCH --job-name=cut_infer_BCI_512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/infer_BCI_512.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/infer_BCI_512.%j.err

# infer_BCI_full_e100.sh
# Runs inference on the full BCI test split using the latest checkpoint
# from the BCI 100-epoch 512x512 training run.
#
# Submit ONLY after train_BCI_full_e100.sh has completed successfully.
# Submit: sbatch infer_BCI_full_e100.sh
#
# Output images land at:
#   $VSC_DATA/projects/cut/outputs/results/BCI_512_e100/test_latest/images/fake_B/
#
# Verify after job:
#   find $VSC_DATA/projects/cut/outputs/results/BCI_512_e100 -name "*.png" | wc -l

set -euo pipefail

REPO_DIR="$VSC_DATA/projects/cut/code/cut"
DATA_ROOT="$VSC_SCRATCH/cut-BCI"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RESULTS_DIR="$VSC_DATA/projects/cut/outputs/results"
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
echo "=== Checkpoint check ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint folder not found: $CKPT_DIR"
    echo "Has train_BCI_full_e100.sh completed successfully?"
    deactivate; exit 1
fi
echo "  Checkpoints found:"
find "$CKPT_DIR" -name "*.pth" | sort

echo ""
echo "=== Test dataset check ==="
echo "  testA: $(find "$DATA_ROOT"/testA -maxdepth 1 \( -type f -o -type l \) | wc -l) images"
echo "  testB: $(find "$DATA_ROOT"/testB -maxdepth 1 \( -type f -o -type l \) | wc -l) images"

mkdir -p "$RESULTS_DIR/$RUN_NAME"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/cut/logs/gpu_infer_BCI_512.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting BCI inference ==="
echo "  run name    : $RUN_NAME"
echo "  data        : $DATA_ROOT"
echo "  results dir : $RESULTS_DIR/$RUN_NAME"

python test.py \
    --dataroot "$DATA_ROOT" \
    --name "$RUN_NAME" \
    --CUT_mode CUT \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --load_size 512 \
    --crop_size 512 \
    --num_test 9999 \
    --eval \
    --gpu_ids 0

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Output image count ==="
find "$RESULTS_DIR/$RUN_NAME" -name "*.png" | wc -l

echo ""
echo "=== Output folder structure ==="
ls "$RESULTS_DIR/$RUN_NAME/test_latest/images/" 2>/dev/null || echo "WARNING: test_latest/images/ not found"

deactivate
echo ""
echo "BCI inference complete. Next step: sbatch eval_BCI_full_e100.sh"