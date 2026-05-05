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

# eval_MIST-HER2_full.sh
# Runs evaluate.py on MIST-HER2 inference outputs using the shared evaluation container.
# Computes PSNR, SSIM, MS-SSIM, LPIPS (AlexNet + VGG), MAE, FID,
# and Cellpose cell-detection metrics (precision, recall, F1) on 100 sampled pairs.
# Appends results to the shared benchmark_results.csv on $VSC_DATA.
#
# Prerequisites:
#   - infer_MIST-HER2_full.sh must have completed successfully
#   - $VSC_SCRATCH/containers/evaluate_nvidia.sif must exist
#   - $VSC_SCRATCH/MIST-HER2.sqsh must exist
#   - LPIPS and Cellpose weights pre-downloaded on login node
#
# Submit: sbatch eval_MIST-HER2_full.sh

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
RUN_NAME="MIST-HER2_512_e100"
PRED_DIR="$VSC_DATA/projects/cut/outputs/results/$RUN_NAME/test_latest/images/fake_B"
GT_DIR="/data/MIST-HER2/valB"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
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

echo "=== Environment ==="
apptainer exec --nv $CONTAINER python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Container check ==="
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: container not found: $CONTAINER"
    exit 1
fi
echo "  Container found: $CONTAINER"

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
    echo "Has infer_MIST-HER2_full.sh completed successfully?"
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

echo ""
echo "=== Running evaluate.py ==="
echo "  pred       : $PRED_DIR"
echo "  gt         : $GT_DIR (inside MIST-HER2.sqsh)"
echo "  output csv : $OUTPUT_CSV"
echo "  cellpose   : cpsam, 10 pairs sampled"

srun apptainer exec --nv \
    -B $VSC_DATA:$VSC_DATA \
    -B $MIST_SQSH:/data/MIST-HER2:image-src=/ \
    $CONTAINER \
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
        --cellpose_n   10

echo ""
echo "=== Results written ==="
echo "  Last rows of $OUTPUT_CSV:"
tail -3 "$OUTPUT_CSV"

echo ""
echo "MIST-HER2 evaluation complete."