#!/bin/bash
#SBATCH --job-name=install_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/evaluate/logs/install_eval.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/evaluate/logs/install_eval.%j.err

# install_eval.sh
# Creates the shared evaluation venv at $VSC_DATA/evaluate/venv_eval/.
# This venv is reused by all model eval jobs across the benchmark.
# Run once. Re-run only if you need to update an eval dependency.
#
# Also pre-downloads LPIPS backbone weights (AlexNet and VGG) while the
# login/compute node still has internet access, so eval jobs on compute
# nodes (which have no internet) can find them in the cache.
#
# Submit: sbatch install_eval.sh
# Can be submitted at the same time as the model install.sh jobs.
# Check:  cat $VSC_DATA/evaluate/logs/install_eval.<jobid>.out
# Gate:   all sanity checks must print without error before submitting
#         any eval job for any model.

set -euo pipefail

EVAL_DIR="$VSC_DATA/evaluate"
VENV_DIR="$EVAL_DIR/venv_eval"

# =========================
# MODULES
# =========================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "=== System Python ==="
which python
python -V

# =========================
# CREATE VENV
# =========================

rm -rf "$VENV_DIR"
python -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

echo ""
echo "=== Venv Python ==="
which python
python -V

python -m pip install --upgrade pip

# =========================
# EVALUATION DEPENDENCIES
# =========================

python -m pip install \
    torchmetrics \
    lpips \
    torch-fidelity \
    --no-cache-dir

# =========================
# PRE-DOWNLOAD LPIPS WEIGHTS
# Compute nodes have no internet. Downloading here caches the weights
# in ~/.cache so all future eval jobs find them without network access.
# =========================

echo ""
echo "=== Pre-downloading LPIPS backbone weights ==="
python -c "
import lpips
print('Downloading AlexNet backbone...')
lpips.LPIPS(net='alex')
print('Downloading VGG backbone...')
lpips.LPIPS(net='vgg')
print('LPIPS weights cached.')
"

# =========================
# SANITY CHECKS
# =========================

echo ""
echo "=== Sanity checks ==="
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torchmetrics; print('torchmetrics:', torchmetrics.__version__)"
python -c "import lpips; print('lpips ok')"
python -c "import torch_fidelity; print('torch-fidelity ok')"

deactivate
echo ""
echo "Eval venv install complete. All checks passed."
echo "Shared eval venv is at: $VENV_DIR"
echo "All model eval jobs should source this venv."
