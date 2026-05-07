#!/bin/bash
#SBATCH --job-name=cut_eval_MIST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/eval_MIST.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/eval_MIST.%j.err

# eval_MIST-HER2.sh
# Runs evaluate.py on MIST-HER2 inference outputs using the shared evaluation container.
# Computes PSNR, SSIM, MS-SSIM, LPIPS (AlexNet + VGG), MAE, FID,
# and Cellpose cell-detection metrics (precision, recall, F1) on 100 sampled pairs.
# Appends results to the shared benchmark_results.csv on $VSC_DATA.
#
# NOTE: The cluster's Apptainer auto-binds the host /data/ filesystem into every
# container, masking any /data/ subdirectories created during the container build.
# Squashfs archives are therefore mounted to paths under $VSC_SCRATCH instead.
#
# Prerequisites:
#   - infer_MIST-HER2_e100.sh must have completed successfully
#   - $VSC_SCRATCH/containers/evaluate_nvidia.sif must exist
#   - $VSC_SCRATCH/MIST-HER2.sqsh must exist
#   - LPIPS and Cellpose weights pre-downloaded on login node
#
# Submit: sbatch eval_MIST-HER2.sh

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
RUN_NAME="MIST-HER2_e100"
PRED_DIR="$VSC_DATA/projects/cut/outputs/results/$RUN_NAME/test_latest/images/fake_B"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/MIST-HER2"
GT_DIR="$MIST_MNT/valB"
OUTPUT_CSV="$VSC_DATA/benchmark_results.csv"
EVAL_SCRIPT="$VSC_DATA/evaluate/evaluate.py"

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
echo "=== Prediction folder check ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction folder not found: $PRED_DIR"
    echo "Has infer_MIST-HER2_full_e100.sh completed successfully?"
    exit 1
fi
echo "  Prediction images: $(find "$PRED_DIR" -name '*.png' | wc -l)"

echo ""
echo "=== evaluate.py check ==="
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: evaluate.py not found at $EVAL_SCRIPT"
    exit 1
fi
echo "  evaluate.py found"

# =========================
# EVALUATION
# =========================

mkdir -p "$MIST_MNT"

echo ""
echo "=== Running evaluate.py ==="
echo "  pred       : $PRED_DIR"
echo "  gt         : $GT_DIR (valB inside MIST-HER2.sqsh mounted at $MIST_MNT)"
echo "  output csv : $OUTPUT_CSV"
echo "  cellpose   : cpsam, 100 pairs sampled"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    python "$EVAL_SCRIPT" \
        --pred         "$PRED_DIR" \
        --gt           "$GT_DIR" \
        --model_name   cut \
        --dataset_name MIST-HER2 \
        --split_name   val \
        --match_by     stem \
        --output       "$OUTPUT_CSV" \
        --cellpose \
        --cellpose_model cpsam \
        --cellpose_n   100

echo ""
echo "=== Results written ==="
echo "  Last rows of $OUTPUT_CSV:"
tail -3 "$OUTPUT_CSV"

echo ""
echo "MIST-HER2 evaluation complete."
