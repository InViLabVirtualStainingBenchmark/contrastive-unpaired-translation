#!/bin/bash
#SBATCH --job-name=cut_infer_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/infer_BCI.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/infer_BCI.%j.err

# infer_BCI_e100.sh
# Runs inference on the full BCI val split using the latest checkpoint
# from the BCI 100-epoch training run.
# load_size 1024 / crop_size 1024 -- full resolution, no cropping at inference time.
# --phase val: BCI-AB.sqsh has valA/valB (not testA/testB). Output goes to val_latest/.
#
# Submit ONLY after train_BCI_e100.sh has completed successfully.
# Submit: sbatch infer_BCI_e100.sh
#
# Output images land at:
#   $VSC_DATA/projects/cut/outputs/results/BCI_e100/val_latest/images/fake_B/
#
# Verify after job:
#   find $VSC_DATA/projects/cut/outputs/results/BCI_e100 -name "*.png" | wc -l

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/cut_nvidia.sif"
BCI_SQSH="$VSC_SCRATCH/BCI-AB.sqsh"
BCI_MNT="$VSC_SCRATCH/sqsh_mnt/BCI"
REPO_DIR="$VSC_DATA/projects/cut/code/cut"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RESULTS_DIR="$VSC_DATA/projects/cut/outputs/results"
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
echo "=== Checkpoint check ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint folder not found: $CKPT_DIR"
    echo "Has train_BCI_full_e100.sh completed successfully?"
    exit 1
fi
echo "  Checkpoints found:"
find "$CKPT_DIR" -name "*.pth" | sort

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/cut/logs/gpu_infer_BCI.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

mkdir -p "$BCI_MNT"
mkdir -p "$RESULTS_DIR/$RUN_NAME"

echo ""
echo "=== Starting BCI inference (load 1024, crop 1024) ==="
echo "  run name    : $RUN_NAME"
echo "  data        : $BCI_MNT (mounted from BCI-AB.sqsh)"
echo "  results dir : $RESULTS_DIR/$RUN_NAME"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_SQSH:$BCI_MNT:image-src=/" \
    "$CONTAINER" \
    python "$REPO_DIR/test.py" \
        --dataroot "$BCI_MNT" \
        --name "$RUN_NAME" \
        --CUT_mode CUT \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --results_dir "$RESULTS_DIR" \
        --load_size 1024 \
        --crop_size 1024 \
        --phase val \
        --num_test 9999 \
        --eval \
        --display_id 0 \
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
ls "$RESULTS_DIR/$RUN_NAME/val_latest/images/" 2>/dev/null || echo "WARNING: val_latest/images/ not found"

echo ""
echo "BCI inference complete. Next step: sbatch eval_BCI_full.sh"
