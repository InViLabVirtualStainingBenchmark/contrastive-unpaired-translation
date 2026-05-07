#!/bin/bash
#SBATCH --job-name=cut_infer_MIST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/infer_MIST.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/infer_MIST.%j.err

# infer_MIST-HER2_e100.sh
# Runs inference on the MIST-HER2 val split using the latest checkpoint
# from the MIST-HER2 100-epoch training run.
# load_size 1024 / crop_size 1024 -- full resolution, no cropping at inference time.
#
# MIST-HER2.sqsh has valA/valB (not testA/testB). --phase val tells CUT to look for
# valA/ and valB/ inside --dataroot. The results folder is still named test_latest
# by CUT -- this is expected and not an error.
#
# Submit ONLY after train_MIST-HER2_full_e100.sh (both parts) has completed.
# Submit: sbatch infer_MIST-HER2_e100.sh
#
# Output images land at:
#   $VSC_DATA/projects/cut/outputs/results/MIST-HER2_e100/test_latest/images/fake_B/
#
# Verify after job:
#   find $VSC_DATA/projects/cut/outputs/results/MIST-HER2_e100 -name "*.png" | wc -l

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/cut_nvidia.sif"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/MIST-HER2"
REPO_DIR="$VSC_DATA/projects/cut/code/cut"
CHECKPOINTS_DIR="$VSC_DATA/projects/cut/outputs/checkpoints"
RESULTS_DIR="$VSC_DATA/projects/cut/outputs/results"
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
echo "=== Checkpoint check ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint folder not found: $CKPT_DIR"
    echo "Have both parts of train_MIST-HER2_full_e100 completed successfully?"
    exit 1
fi
echo "  Checkpoints found:"
find "$CKPT_DIR" -name "*.pth" | sort

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/cut/logs/gpu_infer_MIST.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

mkdir -p "$MIST_MNT"
mkdir -p "$RESULTS_DIR/$RUN_NAME"

echo ""
echo "=== Starting MIST-HER2 inference (load 1024, crop 1024, phase val) ==="
echo "  run name    : $RUN_NAME"
echo "  data        : $MIST_MNT (mounted from MIST-HER2.sqsh)"
echo "  results dir : $RESULTS_DIR/$RUN_NAME"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    python "$REPO_DIR/test.py" \
        --dataroot "$MIST_MNT" \
        --name "$RUN_NAME" \
        --CUT_mode CUT \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --results_dir "$RESULTS_DIR" \
        --phase val \
        --load_size 1024 \
        --crop_size 1024 \
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
ls "$RESULTS_DIR/$RUN_NAME/test_latest/images/" 2>/dev/null || echo "WARNING: test_latest/images/ not found"

echo ""
echo "MIST-HER2 inference complete. Next step: sbatch eval_MIST-HER2_full.sh"
