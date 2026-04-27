#!/bin/bash
#SBATCH --job-name=cut_install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/cut/logs/install.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/cut/logs/install.%j.err

# install.sh
# Creates the CUT model venv and pip-installs all packages not provided by
# the PyTorch module stack.
# Evaluation dependencies live in a separate shared venv at
# $VSC_DATA/evaluate/venv_eval/ -- see install_eval.sh.
#
# Submit: sbatch install.sh
# Check:  cat $VSC_DATA/projects/cut/logs/install.<jobid>.out
# Gate:   all sanity checks must print without error before continuing.

set -euo pipefail

BASE_DIR="$VSC_DATA/projects/cut"
VENV_DIR="$BASE_DIR/venv_cut"

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
# CUT DEPENDENCIES
# Only packages not provided by the module stack.
# Eval deps are in $VSC_DATA/evaluate/venv_eval/ -- do not add them here.
# =========================

python -m pip install \
    dominate \
    visdom \
    gputil \
    --no-cache-dir

# =========================
# SANITY CHECKS
# =========================

echo ""
echo "=== Sanity checks ==="
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import dominate; print('dominate ok')"
python -c "import visdom; print('visdom ok')"

deactivate
echo ""
echo "Install job complete. All checks passed."
echo "Next: verify install_eval.sh has also completed before submitting train_validate.sh"
