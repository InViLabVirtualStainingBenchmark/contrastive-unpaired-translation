#!/bin/bash
#SBATCH --job-name=cut_eval_MIST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/eval_MIST.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/eval_MIST.%j.err

# eval_MIST-HER2_full.sh
# Runs evaluate.py on MIST-HER2 inference outputs.
# Computes PSNR, SSIM, MS-SSIM, LPIPS (AlexNet + VGG), MAE, FID,
# and Cellpose cell-detection metrics (precision, recall, F1) on 100 sampled pairs.
# Appends results to the shared benchmark_results.csv on $VSC_DATA.
#
# Uses the shared eval venv at $VSC_DATA/evaluate/venv_eval/.
# That venv must exist -- run install_eval.sh first if it does not.
# Cellpose cyto2 weights must be pre-downloaded (install_eval.sh handles this).
#
# Submit ONLY after infer_MIST-HER2_full.sh has completed successfully.
# Submit: sbatch eval_MIST-HER2_full.sh
#
# Results written to:
#   $VSC_DATA/benchmark_results.csv  (shared table, append)
#
# After job completes, transfer results back to local machine:
#   scp vsc21212@login.hpc.uantwerpen.be:$VSC_DATA/benchmark_results.csv \
#     ~/benchmark_results_cluster.csv

set -euo pipefail

RUN_NAME="MIST-HER2_512_e100"
PRED_DIR="$VSC_DATA/projects/cut/outputs/results/$RUN_NAME/test_latest/images/fake_B"
GT_DIR="$VSC_SCRATCH/dataset/MIST-HER2/valB"
OUTPUT_CSV="$VSC_DATA/benchmark_results.csv"
EVAL_SCRIPT="$VSC_DATA/evaluate/evaluate.py"
VENV_DIR="$VSC_DATA/evaluate/venv_eval"

# =========================
# MODULES
# =========================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Environment ==="
which python
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Cellpose check ==="
python -c "import importlib.metadata; print('cellpose:', importlib.metadata.version('cellpose'))"

echo ""
echo "=== Prediction folder check ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction folder not found: $PRED_DIR"
    echo "Has infer_MIST-HER2_full.sh completed successfully?"
    deactivate; exit 1
fi
echo "  Prediction images: $(find "$PRED_DIR" -name '*.png' | wc -l)"

echo ""
echo "=== Ground truth folder check ==="
if [ ! -d "$GT_DIR" ]; then
    echo "ERROR: Ground truth folder not found: $GT_DIR"
    deactivate; exit 1
fi
echo "  Ground truth images: $(find "$GT_DIR" \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.tif' -o -name '*.tiff' \) | wc -l)"

echo ""
echo "=== evaluate.py check ==="
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: evaluate.py not found at $EVAL_SCRIPT"
    deactivate; exit 1
fi
echo "  evaluate.py found"

# =========================
# EVALUATION
# =========================

echo ""
echo "=== Running evaluate.py ==="
echo "  pred          : $PRED_DIR"
echo "  gt            : $GT_DIR"
echo "  output csv    : $OUTPUT_CSV"
echo "  cellpose      : cyto2, 100 pairs sampled (seed=42)"

python "$EVAL_SCRIPT" \
    --pred "$PRED_DIR" \
    --gt "$GT_DIR" \
    --model_name cut \
    --dataset_name MIST-HER2 \
    --split_name val \
    --match_by stem \
    --output "$OUTPUT_CSV" \
    --cellpose \
    --cellpose_model cyto2 \
    --cellpose_n 100

echo ""
echo "=== Results written ==="
echo "  Last rows of $OUTPUT_CSV:"
tail -3 "$OUTPUT_CSV"

deactivate
echo ""
echo "MIST-HER2 evaluation complete."